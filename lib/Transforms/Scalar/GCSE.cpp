//===-- GCSE.cpp - SSA-based Global Common Subexpression Elimination ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is designed to be a very quick global transformation that
// eliminates global common subexpressions from a function.  It does this by
// using an existing value numbering implementation to identify the common
// subexpressions, eliminating them when possible.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "gcse"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ValueNumbering.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumInstRemoved, "Number of instructions removed");
STATISTIC(NumLoadRemoved, "Number of loads removed");
STATISTIC(NumCallRemoved, "Number of calls removed");
STATISTIC(NumNonInsts   , "Number of instructions removed due "
                          "to non-instruction values");
STATISTIC(NumArgsRepl   , "Number of function arguments replaced "
                          "with constant values");
namespace {
  struct VISIBILITY_HIDDEN GCSE : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    GCSE() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F);

  private:
    void ReplaceInstructionWith(Instruction *I, Value *V);

    // This transformation requires dominator and immediate dominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<ValueNumbering>();
    }
  };

  char GCSE::ID = 0;
  RegisterPass<GCSE> X("gcse", "Global Common Subexpression Elimination");
}

// createGCSEPass - The public interface to this file...
FunctionPass *llvm::createGCSEPass() { return new GCSE(); }

// GCSE::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GCSE::runOnFunction(Function &F) {
  bool Changed = false;

  // Get pointers to the analysis results that we will be using...
  DominatorTree &DT = getAnalysis<DominatorTree>();
  ValueNumbering &VN = getAnalysis<ValueNumbering>();

  std::vector<Value*> EqualValues;

  // Check for value numbers of arguments.  If the value numbering
  // implementation can prove that an incoming argument is a constant or global
  // value address, substitute it, making the argument dead.
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end(); AI != E;++AI)
    if (!AI->use_empty()) {
      VN.getEqualNumberNodes(AI, EqualValues);
      if (!EqualValues.empty()) {
        for (unsigned i = 0, e = EqualValues.size(); i != e; ++i)
          if (isa<Constant>(EqualValues[i])) {
            AI->replaceAllUsesWith(EqualValues[i]);
            ++NumArgsRepl;
            Changed = true;
            break;
          }
        EqualValues.clear();
      }
    }

  // Traverse the CFG of the function in dominator order, so that we see each
  // instruction after we see its operands.
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    BasicBlock *BB = DI->getBlock();

    // Remember which instructions we've seen in this basic block as we scan.
    std::set<Instruction*> BlockInsts;

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      Instruction *Inst = I++;

      if (Constant *C = ConstantFoldInstruction(Inst)) {
        ReplaceInstructionWith(Inst, C);
      } else if (Inst->getType() != Type::VoidTy) {
        // If this instruction computes a value, try to fold together common
        // instructions that compute it.
        //
        VN.getEqualNumberNodes(Inst, EqualValues);

        // If this instruction computes a value that is already computed
        // elsewhere, try to recycle the old value.
        if (!EqualValues.empty()) {
          if (Inst == &*BB->begin())
            I = BB->end();
          else {
            I = Inst; --I;
          }

          // First check to see if we were able to value number this instruction
          // to a non-instruction value.  If so, prefer that value over other
          // instructions which may compute the same thing.
          for (unsigned i = 0, e = EqualValues.size(); i != e; ++i)
            if (!isa<Instruction>(EqualValues[i])) {
              ++NumNonInsts;      // Keep track of # of insts repl with values

              // Change all users of Inst to use the replacement and remove it
              // from the program.
              ReplaceInstructionWith(Inst, EqualValues[i]);
              Inst = 0;
              EqualValues.clear();  // don't enter the next loop
              break;
            }

          // If there were no non-instruction values that this instruction
          // produces, find a dominating instruction that produces the same
          // value.  If we find one, use it's value instead of ours.
          for (unsigned i = 0, e = EqualValues.size(); i != e; ++i) {
            Instruction *OtherI = cast<Instruction>(EqualValues[i]);
            bool Dominates = false;
            if (OtherI->getParent() == BB)
              Dominates = BlockInsts.count(OtherI);
            else
              Dominates = DT.dominates(OtherI->getParent(), BB);

            if (Dominates) {
              // Okay, we found an instruction with the same value as this one
              // and that dominates this one.  Replace this instruction with the
              // specified one.
              ReplaceInstructionWith(Inst, OtherI);
              Inst = 0;
              break;
            }
          }

          EqualValues.clear();

          if (Inst) {
            I = Inst; ++I;             // Deleted no instructions
          } else if (I == BB->end()) { // Deleted first instruction
            I = BB->begin();
          } else {                     // Deleted inst in middle of block.
            ++I;
          }
        }

        if (Inst)
          BlockInsts.insert(Inst);
      }
    }
  }

  // When the worklist is empty, return whether or not we changed anything...
  return Changed;
}


void GCSE::ReplaceInstructionWith(Instruction *I, Value *V) {
  if (isa<LoadInst>(I))
    ++NumLoadRemoved; // Keep track of loads eliminated
  if (isa<CallInst>(I))
    ++NumCallRemoved; // Keep track of calls eliminated
  ++NumInstRemoved;   // Keep track of number of insts eliminated

  // Update value numbering
  getAnalysis<ValueNumbering>().deleteValue(I);

  I->replaceAllUsesWith(V);

  if (InvokeInst *II = dyn_cast<InvokeInst>(I)) {
    // Removing an invoke instruction requires adding a branch to the normal
    // destination and removing PHI node entries in the exception destination.
    new BranchInst(II->getNormalDest(), II);
    II->getUnwindDest()->removePredecessor(II->getParent());
  }

  // Erase the instruction from the program.
  I->getParent()->getInstList().erase(I);
}

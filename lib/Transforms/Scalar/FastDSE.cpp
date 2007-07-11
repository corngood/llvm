//===- DeadStoreElimination.cpp - Dead Store Elimination ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a trivial dead store elimination that only considers
// basic-block local redundant stores.
//
// FIXME: This should eventually be extended to be a post-dominator tree
// traversal.  Doing so would be pretty trivial.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "fdse"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumFastStores, "Number of stores deleted");
STATISTIC(NumFastOther , "Number of other instrs removed");

namespace {
  struct VISIBILITY_HIDDEN FDSE : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    FDSE() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F) {
      bool Changed = false;
      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
        Changed |= runOnBasicBlock(*I);
      return Changed;
    }

    bool runOnBasicBlock(BasicBlock &BB);
    void DeleteDeadInstructionChains(Instruction *I,
                                     SetVector<Instruction*> &DeadInsts);

    // getAnalysisUsage - We require post dominance frontiers (aka Control
    // Dependence Graph)
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addPreserved<MemoryDependenceAnalysis>();
    }
  };
  char FDSE::ID = 0;
  RegisterPass<FDSE> X("fdse", "Fast Dead Store Elimination");
}

FunctionPass *llvm::createFastDeadStoreEliminationPass() { return new FDSE(); }

bool FDSE::runOnBasicBlock(BasicBlock &BB) {
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  
  // Record the last-seen store to this pointer
  DenseMap<Value*, StoreInst*> lastStore;
  // Record instructions possibly made dead by deleting a store
  SetVector<Instruction*> possiblyDead;
  
  bool MadeChange = false;
  
  // Do a top-down walk on the BB
  for (BasicBlock::iterator BBI = BB.begin(), BBE = BB.end(); BBI != BBE; ++BBI) {
    // If we find a store...
    if (isa<StoreInst>(BBI) || isa<FreeInst>(BBI)) {
      Value* pointer = 0;
      if (StoreInst* S = dyn_cast<StoreInst>(BBI))
        pointer = S->getPointerOperand();
      else if (FreeInst* F = dyn_cast<FreeInst>(BBI))
        pointer = F->getPointerOperand();
      assert(pointer && "Not a free or a store?");
      
      StoreInst*& last = lastStore[pointer];
      
      // ... to a pointer that has been stored to before...
      if (last) {
        
        // ... and no other memory dependencies are between them....
        if (MD.getDependency(BBI) == last) {
          
          // Remove it!
          MD.removeInstruction(last);
          
          // DCE instructions only used to calculate that store
          if (Instruction* D = dyn_cast<Instruction>(last->getOperand(0)))
            possiblyDead.insert(D);
          
          last->eraseFromParent();
          NumFastStores++;
          MadeChange = true;
        }
      }
      
      // Update our most-recent-store map
      if (StoreInst* S = dyn_cast<StoreInst>(BBI))
        last = S;
      else
        last = 0;
    }
  }
  
  // Do a trivial DCE
  while (!possiblyDead.empty()) {
    Instruction *I = possiblyDead.back();
    possiblyDead.pop_back();
    DeleteDeadInstructionChains(I, possiblyDead);
  }
  
  return MadeChange;
}

void FDSE::DeleteDeadInstructionChains(Instruction *I,
                                      SetVector<Instruction*> &DeadInsts) {
  // Instruction must be dead.
  if (!I->use_empty() || !isInstructionTriviallyDead(I)) return;

  // Let the memory dependence know
  getAnalysis<MemoryDependenceAnalysis>().removeInstruction(I);

  // See if this made any operands dead.  We do it this way in case the
  // instruction uses the same operand twice.  We don't want to delete a
  // value then reference it.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    if (I->getOperand(i)->hasOneUse())
      if (Instruction* Op = dyn_cast<Instruction>(I->getOperand(i)))
        DeadInsts.insert(Op);      // Attempt to nuke it later.
    
    I->setOperand(i, 0);         // Drop from the operand list.
  }

  I->eraseFromParent();
  ++NumFastOther;
}

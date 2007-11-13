//===- SimplifyCFG.cpp - CFG Simplification Pass --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements dead code elimination and basic block merging.
// Specifically:
//
//   * Removes basic blocks with no predecessors.
//   * Merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor.
//   * Eliminates a basic block that only contains an unconditional branch.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplifycfg"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumSimpl, "Number of blocks simplified");

namespace {
  struct VISIBILITY_HIDDEN CFGSimplifyPass : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGSimplifyPass() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F);
  };
  char CFGSimplifyPass::ID = 0;
  RegisterPass<CFGSimplifyPass> X("simplifycfg", "Simplify the CFG");
}

// Public interface to the CFGSimplification pass
FunctionPass *llvm::createCFGSimplificationPass() {
  return new CFGSimplifyPass();
}

static bool MarkAliveBlocks(BasicBlock *BB,
                            SmallPtrSet<BasicBlock*, 128> &Reachable) {
  
  SmallVector<BasicBlock*, 128> Worklist;
  Worklist.push_back(BB);
  bool Changed = false;
  while (!Worklist.empty()) {
    BB = Worklist.back();
    Worklist.pop_back();
    
    if (!Reachable.insert(BB))
      continue;

    // Do a quick scan of the basic block, turning any obviously unreachable
    // instructions into LLVM unreachable insts.  The instruction combining pass
    // canonnicalizes unreachable insts into stores to null or undef.
    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E; ++BBI)
      if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
        if (isa<ConstantPointerNull>(SI->getOperand(1)) ||
            isa<UndefValue>(SI->getOperand(1))) {
          // Loop over all of the successors, removing BB's entry from any PHI
          // nodes.
          for (succ_iterator I = succ_begin(BB), SE = succ_end(BB); I != SE;++I)
            (*I)->removePredecessor(BB);

          new UnreachableInst(SI);

          // All instructions after this are dead.
          while (BBI != E) {
            if (!BBI->use_empty())
              BBI->replaceAllUsesWith(UndefValue::get(BBI->getType()));
            BB->getInstList().erase(BBI++);
          }
          break;
        }


    Changed |= ConstantFoldTerminator(BB);
    for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
      Worklist.push_back(*SI);
  }
  return Changed;
}

/// RemoveUnreachableBlocks - Remove blocks that are not reachable, even if they
/// are in a dead cycle.  Return true if a change was made, false otherwise.
static bool RemoveUnreachableBlocks(Function &F) {
  SmallPtrSet<BasicBlock*, 128> Reachable;
  bool Changed = MarkAliveBlocks(F.begin(), Reachable);
  
  // If there are unreachable blocks in the CFG...
  if (Reachable.size() == F.size())
    return Changed;
  
  assert(Reachable.size() < F.size());
  NumSimpl += F.size()-Reachable.size();
  
  // Loop over all of the basic blocks that are not reachable, dropping all of
  // their internal references...
  for (Function::iterator BB = ++F.begin(), E = F.end(); BB != E; ++BB)
    if (!Reachable.count(BB)) {
      for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI!=SE; ++SI)
        if (Reachable.count(*SI))
          (*SI)->removePredecessor(BB);
      BB->dropAllReferences();
    }
  
  for (Function::iterator I = ++F.begin(); I != F.end();)
    if (!Reachable.count(I))
      I = F.getBasicBlockList().erase(I);
    else
      ++I;
  
  return true;
}

/// IterativeSimplifyCFG - Call SimplifyCFG on all the blocks in the function,
/// iterating until no more changes are made.
static bool IterativeSimplifyCFG(Function &F) {
  bool Changed = false;
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;
    
    // Loop over all of the basic blocks (except the first one) and remove them
    // if they are unneeded...
    //
    for (Function::iterator BBIt = ++F.begin(); BBIt != F.end(); ) {
      if (SimplifyCFG(BBIt++)) {
        LocalChange = true;
        ++NumSimpl;
      }
    }
    Changed |= LocalChange;
  }
  return Changed;
}

// It is possible that we may require multiple passes over the code to fully
// simplify the CFG.
//
bool CFGSimplifyPass::runOnFunction(Function &F) {
  bool EverChanged = RemoveUnreachableBlocks(F);
  EverChanged |= IterativeSimplifyCFG(F);
  
  // If neither pass changed anything, we're done.
  if (!EverChanged) return false;

  // IterativeSimplifyCFG can (rarely) make some loops dead.  If this happens,
  // RemoveUnreachableBlocks is needed to nuke them, which means we should
  // iterate between the two optimizations.  We structure the code like this to
  // avoid reruning IterativeSimplifyCFG if the second pass of 
  // RemoveUnreachableBlocks doesn't do anything.
  if (!RemoveUnreachableBlocks(F))
    return true;
  
  do {
    EverChanged = IterativeSimplifyCFG(F);
    EverChanged |= RemoveUnreachableBlocks(F);
  } while (EverChanged);
  
  return true;
}

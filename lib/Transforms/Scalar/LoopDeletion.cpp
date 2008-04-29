//===- LoopDeletion.cpp - Dead Loop Deletion Pass ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dead Loop Elimination Pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-delete"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

STATISTIC(NumDeleted, "Number of loops deleted");

namespace {
  class VISIBILITY_HIDDEN LoopDeletion : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopDeletion() : LoopPass((intptr_t)&ID) { }
    
    // Possibly eliminate loop L if it is dead.
    bool runOnLoop(Loop* L, LPPassManager& LPM);
    
    bool SingleDominatingExit(Loop* L);
    bool IsLoopDead(Loop* L);
    bool IsLoopInvariantInst(Instruction *I, Loop* L);
    
    virtual void getAnalysisUsage(AnalysisUsage& AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<LoopInfo>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
    }
  };
  
  char LoopDeletion::ID = 0;
  RegisterPass<LoopDeletion> X ("loop-deletion", "Delete dead loops");
}

LoopPass* llvm::createLoopDeletionPass() {
  return new LoopDeletion();
}

bool LoopDeletion::SingleDominatingExit(Loop* L) {
  SmallVector<BasicBlock*, 4> exitingBlocks;
  L->getExitingBlocks(exitingBlocks);
  
  if (exitingBlocks.size() != 1)
    return 0;
  
  BasicBlock* latch = L->getLoopLatch();
  if (!latch)
    return 0;
  
  DominatorTree& DT = getAnalysis<DominatorTree>();
  if (DT.dominates(exitingBlocks[0], latch))
    return exitingBlocks[0];
  else
    return 0;
}

bool LoopDeletion::IsLoopInvariantInst(Instruction *I, Loop* L)  {
  // PHI nodes are not loop invariant if defined in  the loop.
  if (isa<PHINode>(I) && L->contains(I->getParent()))
    return false;
    
  // The instruction is loop invariant if all of its operands are loop-invariant
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (!L->isLoopInvariant(I->getOperand(i)))
      return false;

  // If we got this far, the instruction is loop invariant!
  return true;
}

bool LoopDeletion::IsLoopDead(Loop* L) {
  SmallVector<BasicBlock*, 1> exitingBlocks;
  L->getExitingBlocks(exitingBlocks);
  BasicBlock* exitingBlock = exitingBlocks[0];
    
  // Get the set of out-of-loop blocks that the exiting block branches to.
  SmallVector<BasicBlock*, 8> exitBlocks;
  L->getUniqueExitBlocks(exitBlocks);
  if (exitBlocks.size() > 1)
    return false;
  BasicBlock* exitBlock = exitBlocks[0];
  
  // Make sure that all PHI entries coming from the loop are loop invariant.
  BasicBlock::iterator BI = exitBlock->begin();
  while (PHINode* P = dyn_cast<PHINode>(BI)) {
    Value* incoming = P->getIncomingValueForBlock(exitingBlock);
    if (Instruction* I = dyn_cast<Instruction>(incoming))
      if (!IsLoopInvariantInst(I, L))
        return false;
      
    BI++;
  }
  
  // Make sure that no instructions in the block have potential side-effects.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    for (BasicBlock::iterator BI = (*LI)->begin(), BE = (*LI)->end();
         BI != BE; ++BI) {
      if (BI->mayWriteToMemory())
        return false;
    }
  }
  
  return true;
}

/// runOnLoop - Remove dead loops, by which we mean loops that do not impact the
/// observable behavior of the program other than finite running time.  Note 
/// we do ensure that this never remove a loop that might be infinite, as doing
/// so could change the halting/non-halting nature of a program.
bool LoopDeletion::runOnLoop(Loop* L, LPPassManager& LPM) {
  // Don't remove loops for which we can't solve the trip count.
  // They could be infinite, in which case we'd be changing program behavior.
  if (L->getTripCount())
    return false;
  
  // We can only remove the loop if there is a preheader that we can 
  // branch from after removing it.
  BasicBlock* preheader = L->getLoopPreheader();
  if (!preheader)
    return false;
  
  // We can't remove loops that contain subloops.  If the subloops were dead,
  // they would already have been removed in earlier executions of this pass.
  if (L->begin() != L->end())
    return false;
  
  // Loops with multiple exits or exits that don't dominate the latch
  // are too complicated to handle correctly.
  if (!SingleDominatingExit(L))
    return false;
  
  // Finally, we have to check that the loop really is dead.
  if (!IsLoopDead(L))
    return false;
  
  // Now that we know the removal is safe, change the branch from the preheader
  // to go to the single exiting block.
  SmallVector<BasicBlock*, 1> exitingBlocks;
  L->getExitingBlocks(exitingBlocks);
  BasicBlock* exitingBlock = exitingBlocks[0];
  
  SmallVector<BasicBlock*, 1> exitBlocks;
  L->getUniqueExitBlocks(exitBlocks);
  BasicBlock* exitBlock = exitBlocks[0];
  
  // Because we're deleting a large chunk of code at once, the sequence in which
  // we remove things is very important to avoid invalidation issues.  Don't
  // mess with this unless you have good reason and know what you're doing.
  
  // Move simple loop-invariant expressions out of the loop, since they
  // might be needed by the exit phis.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI)
    for (BasicBlock::iterator BI = (*LI)->begin(), BE = (*LI)->end();
         BI != BE; ) {
      Instruction* I = BI++;
      if (I->getNumUses() > 0 && IsLoopInvariantInst(I, L))
        I->moveBefore(preheader->getTerminator());
    }
  
  // Connect the preheader directly to the exit block.
  TerminatorInst* TI = preheader->getTerminator();
  if (BranchInst* BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isUnconditional())
      BI->setSuccessor(0, exitBlock);
    else if (L->contains(BI->getSuccessor(0)))
      BI->setSuccessor(0, exitBlock);
    else
      BI->setSuccessor(1, exitBlock);
  } else {
    // FIXME: Support switches
    return false;
  }
  
  // Rewrite phis in the exit block to get their inputs from
  // the preheader instead of the exiting block.
  BasicBlock::iterator BI = exitBlock->begin();
  while (PHINode* P = dyn_cast<PHINode>(BI)) {
    unsigned i = P->getBasicBlockIndex(exitingBlock);
    P->setIncomingBlock(i, preheader);
    BI++;
  }
  
  // Update lots of internal structures...
  DominatorTree& DT = getAnalysis<DominatorTree>();
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    // Move all of the block's children to be children of the preheader, which
    // allows us to remove the domtree entry for the block.
    SmallPtrSet<DomTreeNode*, 8> childNodes;
    childNodes.insert(DT[*LI]->begin(), DT[*LI]->end());
    for (SmallPtrSet<DomTreeNode*, 8>::iterator DI = childNodes.begin(),
         DE = childNodes.end(); DI != DE; ++DI)
      DT.changeImmediateDominator(*DI, DT[preheader]);
    
    DT.eraseNode(*LI);
    
    // Drop all references between the instructions and the block so
    // that we don't have reference counting problems later.
    for (BasicBlock::iterator BI = (*LI)->begin(), BE = (*LI)->end();
         BI != BE; ++BI) {
      BI->dropAllReferences();
    }
    
    (*LI)->dropAllReferences();
  }
  
  // Erase the instructions and the blocks without having to worry
  // about ordering because we already dropped the references.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    for (BasicBlock::iterator BI = (*LI)->begin(), BE = (*LI)->end();
         BI != BE; ) {
      Instruction* I = BI++;
      I->eraseFromParent();
    }
    
    (*LI)->eraseFromParent();
  }
  
  // Finally, the blocks from loopinfo.  This has to happen late because
  // otherwise our loop iterators won't work.
  LoopInfo& loopInfo = getAnalysis<LoopInfo>();
  SmallPtrSet<BasicBlock*, 8> blocks;
  blocks.insert(L->block_begin(), L->block_end());
  for (SmallPtrSet<BasicBlock*,8>::iterator I = blocks.begin(),
       E = blocks.end(); I != E; ++I)
    loopInfo.removeBlock(*I);
  
  // The last step is to inform the loop pass manager that we've
  // eliminated this loop.
  LPM.deleteLoopFromQueue(L);
  
  NumDeleted++;
  
  return true;
}

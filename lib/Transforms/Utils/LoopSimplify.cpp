//===- LoopSimplify.cpp - Loop Canonicalization Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs several transformations to transform natural loops into a
// simpler form, which makes subsequent analyses and transformations simpler and
// more effective.
//
// Loop pre-header insertion guarantees that there is a single, non-critical
// entry edge from outside of the loop to the loop header.  This simplifies a
// number of analyses and transformations, such as LICM.
//
// Loop exit-block insertion guarantees that all exit blocks from the loop
// (blocks which are outside of the loop that have predecessors inside of the
// loop) only have predecessors from inside of the loop (and are thus dominated
// by the loop header).  This simplifies transformations such as store-sinking
// that are built into LICM.
//
// This pass also guarantees that loops will have exactly one backedge.
//
// Note that the simplifycfg pass will clean up blocks which are split out but
// end up being unnecessary, so usage of this pass should not pessimize
// generated code.
//
// This pass obviously modifies the CFG, but updates loop information and
// dominator information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loopsimplify"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DepthFirstIterator.h"
using namespace llvm;

STATISTIC(NumInserted, "Number of pre-header or exit blocks inserted");
STATISTIC(NumNested  , "Number of nested loops split out");

namespace {
  struct VISIBILITY_HIDDEN LoopSimplify : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    LoopSimplify() : FunctionPass((intptr_t)&ID) {}

    // AA - If we have an alias analysis object to update, this is it, otherwise
    // this is null.
    AliasAnalysis *AA;
    LoopInfo *LI;

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // We need loop information to identify the loops...
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorTree>();

      AU.addPreserved<LoopInfo>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
      AU.addPreservedID(BreakCriticalEdgesID);  // No critical edges added.
    }
  private:
    bool ProcessLoop(Loop *L);
    BasicBlock *SplitBlockPredecessors(BasicBlock *BB, const char *Suffix,
                                       const std::vector<BasicBlock*> &Preds);
    BasicBlock *RewriteLoopExitBlock(Loop *L, BasicBlock *Exit);
    void InsertPreheaderForLoop(Loop *L);
    Loop *SeparateNestedLoop(Loop *L);
    void InsertUniqueBackedgeBlock(Loop *L);
    void PlaceSplitBlockCarefully(BasicBlock *NewBB,
                                  std::vector<BasicBlock*> &SplitPreds,
                                  Loop *L);
      
    void UpdateDomInfoForRevectoredPreds(BasicBlock *NewBB,
                                         std::vector<BasicBlock*> &PredBlocks);
  };

  char LoopSimplify::ID = 0;
  RegisterPass<LoopSimplify>
  X("loopsimplify", "Canonicalize natural loops", true);
}

// Publically exposed interface to pass...
const PassInfo *llvm::LoopSimplifyID = X.getPassInfo();
FunctionPass *llvm::createLoopSimplifyPass() { return new LoopSimplify(); }

/// runOnFunction - Run down all loops in the CFG (recursively, but we could do
/// it in any convenient order) inserting preheaders...
///
bool LoopSimplify::runOnFunction(Function &F) {
  bool Changed = false;
  LI = &getAnalysis<LoopInfo>();
  AA = getAnalysisToUpdate<AliasAnalysis>();

  // Check to see that no blocks (other than the header) in loops have
  // predecessors that are not in loops.  This is not valid for natural loops,
  // but can occur if the blocks are unreachable.  Since they are unreachable we
  // can just shamelessly destroy their terminators to make them not branch into
  // the loop!
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    // This case can only occur for unreachable blocks.  Blocks that are
    // unreachable can't be in loops, so filter those blocks out.
    if (LI->getLoopFor(BB)) continue;
    
    bool BlockUnreachable = false;
    TerminatorInst *TI = BB->getTerminator();

    // Check to see if any successors of this block are non-loop-header loops
    // that are not the header.
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
      // If this successor is not in a loop, BB is clearly ok.
      Loop *L = LI->getLoopFor(TI->getSuccessor(i));
      if (!L) continue;
      
      // If the succ is the loop header, and if L is a top-level loop, then this
      // is an entrance into a loop through the header, which is also ok.
      if (L->getHeader() == TI->getSuccessor(i) && L->getParentLoop() == 0)
        continue;
      
      // Otherwise, this is an entrance into a loop from some place invalid.
      // Either the loop structure is invalid and this is not a natural loop (in
      // which case the compiler is buggy somewhere else) or BB is unreachable.
      BlockUnreachable = true;
      break;
    }
    
    // If this block is ok, check the next one.
    if (!BlockUnreachable) continue;
    
    // Otherwise, this block is dead.  To clean up the CFG and to allow later
    // loop transformations to ignore this case, we delete the edges into the
    // loop by replacing the terminator.
    
    // Remove PHI entries from the successors.
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      TI->getSuccessor(i)->removePredecessor(BB);
   
    // Add a new unreachable instruction.
    new UnreachableInst(TI);
    
    // Delete the dead terminator.
    if (AA) AA->deleteValue(&BB->back());
    BB->getInstList().pop_back();
    Changed |= true;
  }
  
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    Changed |= ProcessLoop(*I);

  return Changed;
}

/// ProcessLoop - Walk the loop structure in depth first order, ensuring that
/// all loops have preheaders.
///
bool LoopSimplify::ProcessLoop(Loop *L) {
  bool Changed = false;
ReprocessLoop:
  
  // Canonicalize inner loops before outer loops.  Inner loop canonicalization
  // can provide work for the outer loop to canonicalize.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    Changed |= ProcessLoop(*I);
  
  assert(L->getBlocks()[0] == L->getHeader() &&
         "Header isn't first block in loop?");

  // Does the loop already have a preheader?  If so, don't insert one.
  if (L->getLoopPreheader() == 0) {
    InsertPreheaderForLoop(L);
    NumInserted++;
    Changed = true;
  }

  // Next, check to make sure that all exit nodes of the loop only have
  // predecessors that are inside of the loop.  This check guarantees that the
  // loop preheader/header will dominate the exit blocks.  If the exit block has
  // predecessors from outside of the loop, split the edge now.
  std::vector<BasicBlock*> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
    
  SetVector<BasicBlock*> ExitBlockSet(ExitBlocks.begin(), ExitBlocks.end());
  for (SetVector<BasicBlock*>::iterator I = ExitBlockSet.begin(),
         E = ExitBlockSet.end(); I != E; ++I) {
    BasicBlock *ExitBlock = *I;
    for (pred_iterator PI = pred_begin(ExitBlock), PE = pred_end(ExitBlock);
         PI != PE; ++PI)
      // Must be exactly this loop: no subloops, parent loops, or non-loop preds
      // allowed.
      if (!L->contains(*PI)) {
        RewriteLoopExitBlock(L, ExitBlock);
        NumInserted++;
        Changed = true;
        break;
      }
  }

  // If the header has more than two predecessors at this point (from the
  // preheader and from multiple backedges), we must adjust the loop.
  unsigned NumBackedges = L->getNumBackEdges();
  if (NumBackedges != 1) {
    // If this is really a nested loop, rip it out into a child loop.  Don't do
    // this for loops with a giant number of backedges, just factor them into a
    // common backedge instead.
    if (NumBackedges < 8) {
      if (Loop *NL = SeparateNestedLoop(L)) {
        ++NumNested;
        // This is a big restructuring change, reprocess the whole loop.
        ProcessLoop(NL);
        Changed = true;
        // GCC doesn't tail recursion eliminate this.
        goto ReprocessLoop;
      }
    }

    // If we either couldn't, or didn't want to, identify nesting of the loops,
    // insert a new block that all backedges target, then make it jump to the
    // loop header.
    InsertUniqueBackedgeBlock(L);
    NumInserted++;
    Changed = true;
  }

  // Scan over the PHI nodes in the loop header.  Since they now have only two
  // incoming values (the loop is canonicalized), we may have simplified the PHI
  // down to 'X = phi [X, Y]', which should be replaced with 'Y'.
  PHINode *PN;
  for (BasicBlock::iterator I = L->getHeader()->begin();
       (PN = dyn_cast<PHINode>(I++)); )
    if (Value *V = PN->hasConstantValue()) {
        PN->replaceAllUsesWith(V);
        PN->eraseFromParent();
      }

  return Changed;
}

/// SplitBlockPredecessors - Split the specified block into two blocks.  We want
/// to move the predecessors specified in the Preds list to point to the new
/// block, leaving the remaining predecessors pointing to BB.  This method
/// updates the SSA PHINode's, but no other analyses.
///
BasicBlock *LoopSimplify::SplitBlockPredecessors(BasicBlock *BB,
                                                 const char *Suffix,
                                       const std::vector<BasicBlock*> &Preds) {

  // Create new basic block, insert right before the original block...
  BasicBlock *NewBB = new BasicBlock(BB->getName()+Suffix, BB->getParent(), BB);

  // The preheader first gets an unconditional branch to the loop header...
  BranchInst *BI = new BranchInst(BB, NewBB);

  // For every PHI node in the block, insert a PHI node into NewBB where the
  // incoming values from the out of loop edges are moved to NewBB.  We have two
  // possible cases here.  If the loop is dead, we just insert dummy entries
  // into the PHI nodes for the new edge.  If the loop is not dead, we move the
  // incoming edges in BB into new PHI nodes in NewBB.
  //
  if (!Preds.empty()) {  // Is the loop not obviously dead?
    // Check to see if the values being merged into the new block need PHI
    // nodes.  If so, insert them.
    for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ) {
      PHINode *PN = cast<PHINode>(I);
      ++I;

      // Check to see if all of the values coming in are the same.  If so, we
      // don't need to create a new PHI node.
      Value *InVal = PN->getIncomingValueForBlock(Preds[0]);
      for (unsigned i = 1, e = Preds.size(); i != e; ++i)
        if (InVal != PN->getIncomingValueForBlock(Preds[i])) {
          InVal = 0;
          break;
        }

      // If the values coming into the block are not the same, we need a PHI.
      if (InVal == 0) {
        // Create the new PHI node, insert it into NewBB at the end of the block
        PHINode *NewPHI = new PHINode(PN->getType(), PN->getName()+".ph", BI);
        if (AA) AA->copyValue(PN, NewPHI);

        // Move all of the edges from blocks outside the loop to the new PHI
        for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
          Value *V = PN->removeIncomingValue(Preds[i], false);
          NewPHI->addIncoming(V, Preds[i]);
        }
        InVal = NewPHI;
      } else {
        // Remove all of the edges coming into the PHI nodes from outside of the
        // block.
        for (unsigned i = 0, e = Preds.size(); i != e; ++i)
          PN->removeIncomingValue(Preds[i], false);
      }

      // Add an incoming value to the PHI node in the loop for the preheader
      // edge.
      PN->addIncoming(InVal, NewBB);

      // Can we eliminate this phi node now?
      if (Value *V = PN->hasConstantValue(true)) {
        Instruction *I = dyn_cast<Instruction>(V);
        // If I is in NewBB, the Dominator call will fail, because NewBB isn't
        // registered in DominatorTree yet.  Handle this case explicitly.
        if (!I || (I->getParent() != NewBB &&
                   getAnalysis<DominatorTree>().dominates(I, PN))) {
          PN->replaceAllUsesWith(V);
          if (AA) AA->deleteValue(PN);
          BB->getInstList().erase(PN);
        }
      }
    }

    // Now that the PHI nodes are updated, actually move the edges from
    // Preds to point to NewBB instead of BB.
    //
    for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
      TerminatorInst *TI = Preds[i]->getTerminator();
      for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s)
        if (TI->getSuccessor(s) == BB)
          TI->setSuccessor(s, NewBB);
    }

  } else {                       // Otherwise the loop is dead...
    for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ++I) {
      PHINode *PN = cast<PHINode>(I);
      // Insert dummy values as the incoming value...
      PN->addIncoming(Constant::getNullValue(PN->getType()), NewBB);
    }
  }
  return NewBB;
}

/// InsertPreheaderForLoop - Once we discover that a loop doesn't have a
/// preheader, this method is called to insert one.  This method has two phases:
/// preheader insertion and analysis updating.
///
void LoopSimplify::InsertPreheaderForLoop(Loop *L) {
  BasicBlock *Header = L->getHeader();

  // Compute the set of predecessors of the loop that are not in the loop.
  std::vector<BasicBlock*> OutsideBlocks;
  for (pred_iterator PI = pred_begin(Header), PE = pred_end(Header);
       PI != PE; ++PI)
    if (!L->contains(*PI))           // Coming in from outside the loop?
      OutsideBlocks.push_back(*PI);  // Keep track of it...

  // Split out the loop pre-header.
  BasicBlock *NewBB =
    SplitBlockPredecessors(Header, ".preheader", OutsideBlocks);
  

  //===--------------------------------------------------------------------===//
  //  Update analysis results now that we have performed the transformation
  //

  // We know that we have loop information to update... update it now.
  if (Loop *Parent = L->getParentLoop())
    Parent->addBasicBlockToLoop(NewBB, *LI);

  UpdateDomInfoForRevectoredPreds(NewBB, OutsideBlocks);
  
  // Make sure that NewBB is put someplace intelligent, which doesn't mess up
  // code layout too horribly.
  PlaceSplitBlockCarefully(NewBB, OutsideBlocks, L);
}

/// RewriteLoopExitBlock - Ensure that the loop preheader dominates all exit
/// blocks.  This method is used to split exit blocks that have predecessors
/// outside of the loop.
BasicBlock *LoopSimplify::RewriteLoopExitBlock(Loop *L, BasicBlock *Exit) {
  std::vector<BasicBlock*> LoopBlocks;
  for (pred_iterator I = pred_begin(Exit), E = pred_end(Exit); I != E; ++I)
    if (L->contains(*I))
      LoopBlocks.push_back(*I);

  assert(!LoopBlocks.empty() && "No edges coming in from outside the loop?");
  BasicBlock *NewBB = SplitBlockPredecessors(Exit, ".loopexit", LoopBlocks);

  // Update Loop Information - we know that the new block will be in whichever
  // loop the Exit block is in.  Note that it may not be in that immediate loop,
  // if the successor is some other loop header.  In that case, we continue 
  // walking up the loop tree to find a loop that contains both the successor
  // block and the predecessor block.
  Loop *SuccLoop = LI->getLoopFor(Exit);
  while (SuccLoop && !SuccLoop->contains(L->getHeader()))
    SuccLoop = SuccLoop->getParentLoop();
  if (SuccLoop)
    SuccLoop->addBasicBlockToLoop(NewBB, *LI);

  // Update dominator information (set, immdom, domtree, and domfrontier)
  UpdateDomInfoForRevectoredPreds(NewBB, LoopBlocks);
  return NewBB;
}

/// AddBlockAndPredsToSet - Add the specified block, and all of its
/// predecessors, to the specified set, if it's not already in there.  Stop
/// predecessor traversal when we reach StopBlock.
static void AddBlockAndPredsToSet(BasicBlock *InputBB, BasicBlock *StopBlock,
                                  std::set<BasicBlock*> &Blocks) {
  std::vector<BasicBlock *> WorkList;
  WorkList.push_back(InputBB);
  do {
    BasicBlock *BB = WorkList.back(); WorkList.pop_back();
    if (Blocks.insert(BB).second && BB != StopBlock)
      // If BB is not already processed and it is not a stop block then
      // insert its predecessor in the work list
      for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
        BasicBlock *WBB = *I;
        WorkList.push_back(WBB);
      }
  } while(!WorkList.empty());
}

/// FindPHIToPartitionLoops - The first part of loop-nestification is to find a
/// PHI node that tells us how to partition the loops.
static PHINode *FindPHIToPartitionLoops(Loop *L, DominatorTree *DT,
                                        AliasAnalysis *AA) {
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I);
    ++I;
    if (Value *V = PN->hasConstantValue())
      if (!isa<Instruction>(V) || DT->dominates(cast<Instruction>(V), PN)) {
        // This is a degenerate PHI already, don't modify it!
        PN->replaceAllUsesWith(V);
        if (AA) AA->deleteValue(PN);
        PN->eraseFromParent();
        continue;
      }

    // Scan this PHI node looking for a use of the PHI node by itself.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) == PN &&
          L->contains(PN->getIncomingBlock(i)))
        // We found something tasty to remove.
        return PN;
  }
  return 0;
}

// PlaceSplitBlockCarefully - If the block isn't already, move the new block to
// right after some 'outside block' block.  This prevents the preheader from
// being placed inside the loop body, e.g. when the loop hasn't been rotated.
void LoopSimplify::PlaceSplitBlockCarefully(BasicBlock *NewBB,
                                            std::vector<BasicBlock*>&SplitPreds,
                                            Loop *L) {
  // Check to see if NewBB is already well placed.
  Function::iterator BBI = NewBB; --BBI;
  for (unsigned i = 0, e = SplitPreds.size(); i != e; ++i) {
    if (&*BBI == SplitPreds[i])
      return;
  }
  
  // If it isn't already after an outside block, move it after one.  This is
  // always good as it makes the uncond branch from the outside block into a
  // fall-through.
  
  // Figure out *which* outside block to put this after.  Prefer an outside
  // block that neighbors a BB actually in the loop.
  BasicBlock *FoundBB = 0;
  for (unsigned i = 0, e = SplitPreds.size(); i != e; ++i) {
    Function::iterator BBI = SplitPreds[i];
    if (++BBI != NewBB->getParent()->end() && 
        L->contains(BBI)) {
      FoundBB = SplitPreds[i];
      break;
    }
  }
  
  // If our heuristic for a *good* bb to place this after doesn't find
  // anything, just pick something.  It's likely better than leaving it within
  // the loop.
  if (!FoundBB)
    FoundBB = SplitPreds[0];
  NewBB->moveAfter(FoundBB);
}


/// SeparateNestedLoop - If this loop has multiple backedges, try to pull one of
/// them out into a nested loop.  This is important for code that looks like
/// this:
///
///  Loop:
///     ...
///     br cond, Loop, Next
///     ...
///     br cond2, Loop, Out
///
/// To identify this common case, we look at the PHI nodes in the header of the
/// loop.  PHI nodes with unchanging values on one backedge correspond to values
/// that change in the "outer" loop, but not in the "inner" loop.
///
/// If we are able to separate out a loop, return the new outer loop that was
/// created.
///
Loop *LoopSimplify::SeparateNestedLoop(Loop *L) {
  DominatorTree *DT = getAnalysisToUpdate<DominatorTree>();
  PHINode *PN = FindPHIToPartitionLoops(L, DT, AA);
  if (PN == 0) return 0;  // No known way to partition.

  // Pull out all predecessors that have varying values in the loop.  This
  // handles the case when a PHI node has multiple instances of itself as
  // arguments.
  std::vector<BasicBlock*> OuterLoopPreds;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingValue(i) != PN ||
        !L->contains(PN->getIncomingBlock(i)))
      OuterLoopPreds.push_back(PN->getIncomingBlock(i));

  BasicBlock *Header = L->getHeader();
  BasicBlock *NewBB = SplitBlockPredecessors(Header, ".outer", OuterLoopPreds);

  // Update dominator information (set, immdom, domtree, and domfrontier)
  UpdateDomInfoForRevectoredPreds(NewBB, OuterLoopPreds);

  // Make sure that NewBB is put someplace intelligent, which doesn't mess up
  // code layout too horribly.
  PlaceSplitBlockCarefully(NewBB, OuterLoopPreds, L);
  
  // Create the new outer loop.
  Loop *NewOuter = new Loop();

  // Change the parent loop to use the outer loop as its child now.
  if (Loop *Parent = L->getParentLoop())
    Parent->replaceChildLoopWith(L, NewOuter);
  else
    LI->changeTopLevelLoop(L, NewOuter);

  // This block is going to be our new header block: add it to this loop and all
  // parent loops.
  NewOuter->addBasicBlockToLoop(NewBB, *LI);

  // L is now a subloop of our outer loop.
  NewOuter->addChildLoop(L);

  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i)
    NewOuter->addBlockEntry(L->getBlocks()[i]);

  // Determine which blocks should stay in L and which should be moved out to
  // the Outer loop now.
  std::set<BasicBlock*> BlocksInL;
  for (pred_iterator PI = pred_begin(Header), E = pred_end(Header); PI!=E; ++PI)
    if (DT->dominates(Header, *PI))
      AddBlockAndPredsToSet(*PI, Header, BlocksInL);


  // Scan all of the loop children of L, moving them to OuterLoop if they are
  // not part of the inner loop.
  for (Loop::iterator I = L->begin(); I != L->end(); )
    if (BlocksInL.count((*I)->getHeader()))
      ++I;   // Loop remains in L
    else
      NewOuter->addChildLoop(L->removeChildLoop(I));

  // Now that we know which blocks are in L and which need to be moved to
  // OuterLoop, move any blocks that need it.
  for (unsigned i = 0; i != L->getBlocks().size(); ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    if (!BlocksInL.count(BB)) {
      // Move this block to the parent, updating the exit blocks sets
      L->removeBlockFromLoop(BB);
      if ((*LI)[BB] == L)
        LI->changeLoopFor(BB, NewOuter);
      --i;
    }
  }

  return NewOuter;
}



/// InsertUniqueBackedgeBlock - This method is called when the specified loop
/// has more than one backedge in it.  If this occurs, revector all of these
/// backedges to target a new basic block and have that block branch to the loop
/// header.  This ensures that loops have exactly one backedge.
///
void LoopSimplify::InsertUniqueBackedgeBlock(Loop *L) {
  assert(L->getNumBackEdges() > 1 && "Must have > 1 backedge!");

  // Get information about the loop
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  Function *F = Header->getParent();

  // Figure out which basic blocks contain back-edges to the loop header.
  std::vector<BasicBlock*> BackedgeBlocks;
  for (pred_iterator I = pred_begin(Header), E = pred_end(Header); I != E; ++I)
    if (*I != Preheader) BackedgeBlocks.push_back(*I);

  // Create and insert the new backedge block...
  BasicBlock *BEBlock = new BasicBlock(Header->getName()+".backedge", F);
  BranchInst *BETerminator = new BranchInst(Header, BEBlock);

  // Move the new backedge block to right after the last backedge block.
  Function::iterator InsertPos = BackedgeBlocks.back(); ++InsertPos;
  F->getBasicBlockList().splice(InsertPos, F->getBasicBlockList(), BEBlock);

  // Now that the block has been inserted into the function, create PHI nodes in
  // the backedge block which correspond to any PHI nodes in the header block.
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    PHINode *NewPN = new PHINode(PN->getType(), PN->getName()+".be",
                                 BETerminator);
    NewPN->reserveOperandSpace(BackedgeBlocks.size());
    if (AA) AA->copyValue(PN, NewPN);

    // Loop over the PHI node, moving all entries except the one for the
    // preheader over to the new PHI node.
    unsigned PreheaderIdx = ~0U;
    bool HasUniqueIncomingValue = true;
    Value *UniqueValue = 0;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *IBB = PN->getIncomingBlock(i);
      Value *IV = PN->getIncomingValue(i);
      if (IBB == Preheader) {
        PreheaderIdx = i;
      } else {
        NewPN->addIncoming(IV, IBB);
        if (HasUniqueIncomingValue) {
          if (UniqueValue == 0)
            UniqueValue = IV;
          else if (UniqueValue != IV)
            HasUniqueIncomingValue = false;
        }
      }
    }

    // Delete all of the incoming values from the old PN except the preheader's
    assert(PreheaderIdx != ~0U && "PHI has no preheader entry??");
    if (PreheaderIdx != 0) {
      PN->setIncomingValue(0, PN->getIncomingValue(PreheaderIdx));
      PN->setIncomingBlock(0, PN->getIncomingBlock(PreheaderIdx));
    }
    // Nuke all entries except the zero'th.
    for (unsigned i = 0, e = PN->getNumIncomingValues()-1; i != e; ++i)
      PN->removeIncomingValue(e-i, false);

    // Finally, add the newly constructed PHI node as the entry for the BEBlock.
    PN->addIncoming(NewPN, BEBlock);

    // As an optimization, if all incoming values in the new PhiNode (which is a
    // subset of the incoming values of the old PHI node) have the same value,
    // eliminate the PHI Node.
    if (HasUniqueIncomingValue) {
      NewPN->replaceAllUsesWith(UniqueValue);
      if (AA) AA->deleteValue(NewPN);
      BEBlock->getInstList().erase(NewPN);
    }
  }

  // Now that all of the PHI nodes have been inserted and adjusted, modify the
  // backedge blocks to just to the BEBlock instead of the header.
  for (unsigned i = 0, e = BackedgeBlocks.size(); i != e; ++i) {
    TerminatorInst *TI = BackedgeBlocks[i]->getTerminator();
    for (unsigned Op = 0, e = TI->getNumSuccessors(); Op != e; ++Op)
      if (TI->getSuccessor(Op) == Header)
        TI->setSuccessor(Op, BEBlock);
  }

  //===--- Update all analyses which we must preserve now -----------------===//

  // Update Loop Information - we know that this block is now in the current
  // loop and all parent loops.
  L->addBasicBlockToLoop(BEBlock, *LI);

  // Update dominator information (set, immdom, domtree, and domfrontier)
  UpdateDomInfoForRevectoredPreds(BEBlock, BackedgeBlocks);
}

// Returns true if BasicBlock A dominates at least one block in vector B
// Helper function for UpdateDomInfoForRevectoredPreds
static bool BlockDominatesAny(BasicBlock* A, const std::vector<BasicBlock*>& B,
                              DominatorTree &DT) {
  for (std::vector<BasicBlock*>::const_iterator BI = B.begin(), BE = B.end();
       BI != BE; ++BI) {
    if (DT.dominates(A, *BI))
      return true;
  }
  return false;
}

/// UpdateDomInfoForRevectoredPreds - This method is used to update
/// dominator trees and dominance frontiers after a new block has
/// been added to the CFG.
///
/// This only supports the case when an existing block (known as "NewBBSucc"),
/// had some of its predecessors factored into a new basic block.  This
/// transformation inserts a new basic block ("NewBB"), with a single
/// unconditional branch to NewBBSucc, and moves some predecessors of
/// "NewBBSucc" to now branch to NewBB.  These predecessors are listed in
/// PredBlocks, even though they are the same as
/// pred_begin(NewBB)/pred_end(NewBB).
///
void LoopSimplify::UpdateDomInfoForRevectoredPreds(BasicBlock *NewBB,
                                         std::vector<BasicBlock*> &PredBlocks) {
  assert(!PredBlocks.empty() && "No predblocks??");
  assert(NewBB->getTerminator()->getNumSuccessors() == 1
         && "NewBB should have a single successor!");
  BasicBlock *NewBBSucc = NewBB->getTerminator()->getSuccessor(0);
  DominatorTree &DT = getAnalysis<DominatorTree>();

  // The newly inserted basic block will dominate existing basic blocks iff the
  // PredBlocks dominate all of the non-pred blocks.  If all predblocks dominate
  // the non-pred blocks, then they all must be the same block!
  //
  bool NewBBDominatesNewBBSucc = true;
  {
    BasicBlock *OnePred = PredBlocks[0];
    unsigned i = 1, e = PredBlocks.size();
    for (i = 1; !DT.isReachableFromEntry(OnePred); ++i) {
      assert(i != e && "Didn't find reachable pred?");
      OnePred = PredBlocks[i];
    }
    
    for (; i != e; ++i)
      if (PredBlocks[i] != OnePred && DT.isReachableFromEntry(OnePred)){
        NewBBDominatesNewBBSucc = false;
        break;
      }

    if (NewBBDominatesNewBBSucc)
      for (pred_iterator PI = pred_begin(NewBBSucc), E = pred_end(NewBBSucc);
           PI != E; ++PI)
        if (*PI != NewBB && !DT.dominates(NewBBSucc, *PI)) {
          NewBBDominatesNewBBSucc = false;
          break;
        }
  }

  // The other scenario where the new block can dominate its successors are when
  // all predecessors of NewBBSucc that are not NewBB are dominated by NewBBSucc
  // already.
  if (!NewBBDominatesNewBBSucc) {
    NewBBDominatesNewBBSucc = true;
    for (pred_iterator PI = pred_begin(NewBBSucc), E = pred_end(NewBBSucc);
         PI != E; ++PI)
      if (*PI != NewBB && !DT.dominates(NewBBSucc, *PI)) {
        NewBBDominatesNewBBSucc = false;
        break;
      }
  }


  // Update DominatorTree information if it is active.

  // Find NewBB's immediate dominator and create new dominator tree node for NewBB.
  BasicBlock *NewBBIDom = 0;
  unsigned i = 0;
  for (i = 0; i < PredBlocks.size(); ++i)
    if (DT.dominates(&PredBlocks[i]->getParent()->getEntryBlock(), 
                     PredBlocks[i])) {
      NewBBIDom = PredBlocks[i];
      break;
    }
  assert(i != PredBlocks.size() && "No reachable preds?");
  for (i = i + 1; i < PredBlocks.size(); ++i) {
    if (DT.dominates(&PredBlocks[i]->getParent()->getEntryBlock(), 
                      PredBlocks[i]))
      NewBBIDom = DT.nearestCommonDominator(NewBBIDom, PredBlocks[i]);
  }
  assert(NewBBIDom && "No immediate dominator found??");
  
  // Create the new dominator tree node... and set the idom of NewBB.
  DomTreeNode *NewBBNode = DT.addNewBlock(NewBB, NewBBIDom);
  
  // If NewBB strictly dominates other blocks, then it is now the immediate
  // dominator of NewBBSucc.  Update the dominator tree as appropriate.
  if (NewBBDominatesNewBBSucc) {
    DomTreeNode *NewBBSuccNode = DT.getNode(NewBBSucc);
    DT.changeImmediateDominator(NewBBSuccNode, NewBBNode);
  }

  // Update dominance frontier information...
  if (DominanceFrontier *DF = getAnalysisToUpdate<DominanceFrontier>()) {
    // If NewBB dominates NewBBSucc, then DF(NewBB) is now going to be the
    // DF(PredBlocks[0]) without the stuff that the new block does not dominate
    // a predecessor of.
    if (NewBBDominatesNewBBSucc) {
      DominanceFrontier::iterator DFI = DF->find(PredBlocks[0]);
      if (DFI != DF->end()) {
        DominanceFrontier::DomSetType Set = DFI->second;
        // Filter out stuff in Set that we do not dominate a predecessor of.
        for (DominanceFrontier::DomSetType::iterator SetI = Set.begin(),
               E = Set.end(); SetI != E;) {
          bool DominatesPred = false;
          for (pred_iterator PI = pred_begin(*SetI), E = pred_end(*SetI);
               PI != E; ++PI)
            if (DT.dominates(NewBB, *PI))
              DominatesPred = true;
          if (!DominatesPred)
            Set.erase(SetI++);
          else
            ++SetI;
        }

        DF->addBasicBlock(NewBB, Set);
      }

    } else {
      // DF(NewBB) is {NewBBSucc} because NewBB does not strictly dominate
      // NewBBSucc, but it does dominate itself (and there is an edge (NewBB ->
      // NewBBSucc)).  NewBBSucc is the single successor of NewBB.
      DominanceFrontier::DomSetType NewDFSet;
      NewDFSet.insert(NewBBSucc);
      DF->addBasicBlock(NewBB, NewDFSet);
    }

    // Now we must loop over all of the dominance frontiers in the function,
    // replacing occurrences of NewBBSucc with NewBB in some cases.  All
    // blocks that dominate a block in PredBlocks and contained NewBBSucc in
    // their dominance frontier must be updated to contain NewBB instead.
    //
    for (Function::iterator FI = NewBB->getParent()->begin(),
         FE = NewBB->getParent()->end(); FI != FE; ++FI) {
      DominanceFrontier::iterator DFI = DF->find(FI);
      if (DFI == DF->end()) continue;  // unreachable block.
      
      // Only consider dominators of NewBBSucc
      if (!DFI->second.count(NewBBSucc)) continue;
      
      if (BlockDominatesAny(FI, PredBlocks, DT)) {
        // If NewBBSucc should not stay in our dominator frontier, remove it.
        // We remove it unless there is a predecessor of NewBBSucc that we
        // dominate, but we don't strictly dominate NewBBSucc.
        bool ShouldRemove = true;
        if ((BasicBlock*)FI == NewBBSucc
            || !DT.dominates(FI, NewBBSucc)) {
          // Okay, we know that PredDom does not strictly dominate NewBBSucc.
          // Check to see if it dominates any predecessors of NewBBSucc.
          for (pred_iterator PI = pred_begin(NewBBSucc),
               E = pred_end(NewBBSucc); PI != E; ++PI)
            if (DT.dominates(FI, *PI)) {
              ShouldRemove = false;
              break;
            }
          
          if (ShouldRemove)
            DF->removeFromFrontier(DFI, NewBBSucc);
          DF->addToFrontier(DFI, NewBB);
          
          break;
        }
      }
    }
  }
}



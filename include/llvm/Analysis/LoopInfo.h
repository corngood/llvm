//===- llvm/Analysis/LoopInfo.h - Natural Loop Calculator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopInfo class that is used to identify natural loops
// and determine the loop depth of various nodes of the CFG.  Note that natural
// loops may actually be several loops that share the same header node.
//
// This analysis calculates the nesting structure of loops in a function.  For
// each natural loop identified, this analysis identifies natural loops
// contained entirely within the loop and the basic blocks the make up the loop.
//
// It can calculate on the fly various bits of information, for example:
//
//  * whether there is a preheader for the loop
//  * the number of back edges to the header
//  * whether or not a particular block branches out of the loop
//  * the successor blocks of the loop
//  * the loop depth
//  * the trip count
//  * etc...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_INFO_H
#define LLVM_ANALYSIS_LOOP_INFO_H

#include "llvm/Pass.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {

struct ETForest;
class LoopInfo;
class PHINode;
class Instruction;

//===----------------------------------------------------------------------===//
/// Loop class - Instances of this class are used to represent loops that are
/// detected in the flow graph
///
class Loop {
  Loop *ParentLoop;
  std::vector<Loop*> SubLoops;       // Loops contained entirely within this one
  std::vector<BasicBlock*> Blocks;   // First entry is the header node

  Loop(const Loop &);                  // DO NOT IMPLEMENT
  const Loop &operator=(const Loop &); // DO NOT IMPLEMENT
public:
  /// Loop ctor - This creates an empty loop.
  Loop() : ParentLoop(0) {}
  ~Loop() {
    for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
      delete SubLoops[i];
  }

  unsigned getLoopDepth() const {
    unsigned D = 0;
    for (const Loop *CurLoop = this; CurLoop; CurLoop = CurLoop->ParentLoop)
      ++D;
    return D;
  }
  BasicBlock *getHeader() const { return Blocks.front(); }
  Loop *getParentLoop() const { return ParentLoop; }

  /// contains - Return true of the specified basic block is in this loop
  ///
  bool contains(const BasicBlock *BB) const;

  /// iterator/begin/end - Return the loops contained entirely within this loop.
  ///
  const std::vector<Loop*> &getSubLoops() const { return SubLoops; }
  typedef std::vector<Loop*>::const_iterator iterator;
  iterator begin() const { return SubLoops.begin(); }
  iterator end() const { return SubLoops.end(); }

  /// getBlocks - Get a list of the basic blocks which make up this loop.
  ///
  const std::vector<BasicBlock*> &getBlocks() const { return Blocks; }
  typedef std::vector<BasicBlock*>::const_iterator block_iterator;
  block_iterator block_begin() const { return Blocks.begin(); }
  block_iterator block_end() const { return Blocks.end(); }

  /// isLoopExit - True if terminator in the block can branch to another block
  /// that is outside of the current loop.
  ///
  bool isLoopExit(const BasicBlock *BB) const;

  /// getNumBackEdges - Calculate the number of back edges to the loop header
  ///
  unsigned getNumBackEdges() const;

  /// isLoopInvariant - Return true if the specified value is loop invariant
  ///
  bool isLoopInvariant(Value *V) const;

  //===--------------------------------------------------------------------===//
  // APIs for simple analysis of the loop.
  //
  // Note that all of these methods can fail on general loops (ie, there may not
  // be a preheader, etc).  For best success, the loop simplification and
  // induction variable canonicalization pass should be used to normalize loops
  // for easy analysis.  These methods assume canonical loops.

  /// getExitBlocks - Return all of the successor blocks of this loop.  These
  /// are the blocks _outside of the current loop_ which are branched to.
  ///
  void getExitBlocks(std::vector<BasicBlock*> &Blocks) const;

  /// getUniqueExitBlocks - Return all unique successor blocks of this loop. 
  /// These are the blocks _outside of the current loop_ which are branched to.
  /// This assumes that loop is in canonical form.
  ///
  void getUniqueExitBlocks(std::vector<BasicBlock*> &ExitBlocks) const;

  /// getLoopPreheader - If there is a preheader for this loop, return it.  A
  /// loop has a preheader if there is only one edge to the header of the loop
  /// from outside of the loop.  If this is the case, the block branching to the
  /// header of the loop is the preheader node.
  ///
  /// This method returns null if there is no preheader for the loop.
  ///
  BasicBlock *getLoopPreheader() const;

  /// getLoopLatch - If there is a latch block for this loop, return it.  A
  /// latch block is the canonical backedge for a loop.  A loop header in normal
  /// form has two edges into it: one from a preheader and one from a latch
  /// block.
  BasicBlock *getLoopLatch() const;
  
  /// getCanonicalInductionVariable - Check to see if the loop has a canonical
  /// induction variable: an integer recurrence that starts at 0 and increments
  /// by one each time through the loop.  If so, return the phi node that
  /// corresponds to it.
  ///
  PHINode *getCanonicalInductionVariable() const;

  /// getCanonicalInductionVariableIncrement - Return the LLVM value that holds
  /// the canonical induction variable value for the "next" iteration of the
  /// loop.  This always succeeds if getCanonicalInductionVariable succeeds.
  ///
  Instruction *getCanonicalInductionVariableIncrement() const;

  /// getTripCount - Return a loop-invariant LLVM value indicating the number of
  /// times the loop will be executed.  Note that this means that the backedge
  /// of the loop executes N-1 times.  If the trip-count cannot be determined,
  /// this returns null.
  ///
  Value *getTripCount() const;
  
  /// isLCSSAForm - Return true if the Loop is in LCSSA form
  bool isLCSSAForm() const;

  //===--------------------------------------------------------------------===//
  // APIs for updating loop information after changing the CFG
  //

  /// addBasicBlockToLoop - This method is used by other analyses to update loop
  /// information.  NewBB is set to be a new member of the current loop.
  /// Because of this, it is added as a member of all parent loops, and is added
  /// to the specified LoopInfo object as being in the current basic block.  It
  /// is not valid to replace the loop header with this method.
  ///
  void addBasicBlockToLoop(BasicBlock *NewBB, LoopInfo &LI);

  /// replaceChildLoopWith - This is used when splitting loops up.  It replaces
  /// the OldChild entry in our children list with NewChild, and updates the
  /// parent pointer of OldChild to be null and the NewChild to be this loop.
  /// This updates the loop depth of the new child.
  void replaceChildLoopWith(Loop *OldChild, Loop *NewChild);

  /// addChildLoop - Add the specified loop to be a child of this loop.  This
  /// updates the loop depth of the new child.
  ///
  void addChildLoop(Loop *NewChild);

  /// removeChildLoop - This removes the specified child from being a subloop of
  /// this loop.  The loop is not deleted, as it will presumably be inserted
  /// into another loop.
  Loop *removeChildLoop(iterator OldChild);

  /// addBlockEntry - This adds a basic block directly to the basic block list.
  /// This should only be used by transformations that create new loops.  Other
  /// transformations should use addBasicBlockToLoop.
  void addBlockEntry(BasicBlock *BB) {
    Blocks.push_back(BB);
  }

  /// moveToHeader - This method is used to move BB (which must be part of this
  /// loop) to be the loop header of the loop (the block that dominates all
  /// others).
  void moveToHeader(BasicBlock *BB) {
    if (Blocks[0] == BB) return;
    for (unsigned i = 0; ; ++i) {
      assert(i != Blocks.size() && "Loop does not contain BB!");
      if (Blocks[i] == BB) {
        Blocks[i] = Blocks[0];
        Blocks[0] = BB;
        return;
      }
    }
  }

  /// removeBlockFromLoop - This removes the specified basic block from the
  /// current loop, updating the Blocks as appropriate.  This does not update
  /// the mapping in the LoopInfo class.
  void removeBlockFromLoop(BasicBlock *BB);

  void print(std::ostream &O, unsigned Depth = 0) const;
  void dump() const;
private:
  friend class LoopInfo;
  Loop(BasicBlock *BB) : ParentLoop(0) {
    Blocks.push_back(BB);
  }
};



//===----------------------------------------------------------------------===//
/// LoopInfo - This class builds and contains all of the top level loop
/// structures in the specified function.
///
class LoopInfo : public FunctionPass {
  // BBMap - Mapping of basic blocks to the inner most loop they occur in
  std::map<BasicBlock*, Loop*> BBMap;
  std::vector<Loop*> TopLevelLoops;
  friend class Loop;
public:
  ~LoopInfo() { releaseMemory(); }

  /// iterator/begin/end - The interface to the top-level loops in the current
  /// function.
  ///
  typedef std::vector<Loop*>::const_iterator iterator;
  iterator begin() const { return TopLevelLoops.begin(); }
  iterator end() const { return TopLevelLoops.end(); }

  /// getLoopFor - Return the inner most loop that BB lives in.  If a basic
  /// block is in no loop (for example the entry node), null is returned.
  ///
  Loop *getLoopFor(const BasicBlock *BB) const {
    std::map<BasicBlock *, Loop*>::const_iterator I=
      BBMap.find(const_cast<BasicBlock*>(BB));
    return I != BBMap.end() ? I->second : 0;
  }

  /// operator[] - same as getLoopFor...
  ///
  const Loop *operator[](const BasicBlock *BB) const {
    return getLoopFor(BB);
  }

  /// getLoopDepth - Return the loop nesting level of the specified block...
  ///
  unsigned getLoopDepth(const BasicBlock *BB) const {
    const Loop *L = getLoopFor(BB);
    return L ? L->getLoopDepth() : 0;
  }

  // isLoopHeader - True if the block is a loop header node
  bool isLoopHeader(BasicBlock *BB) const {
    const Loop *L = getLoopFor(BB);
    return L && L->getHeader() == BB;
  }

  /// runOnFunction - Calculate the natural loop information.
  ///
  virtual bool runOnFunction(Function &F);

  virtual void releaseMemory();
  void print(std::ostream &O, const Module* = 0) const;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  /// removeLoop - This removes the specified top-level loop from this loop info
  /// object.  The loop is not deleted, as it will presumably be inserted into
  /// another loop.
  Loop *removeLoop(iterator I);

  /// changeLoopFor - Change the top-level loop that contains BB to the
  /// specified loop.  This should be used by transformations that restructure
  /// the loop hierarchy tree.
  void changeLoopFor(BasicBlock *BB, Loop *L);

  /// changeTopLevelLoop - Replace the specified loop in the top-level loops
  /// list with the indicated loop.
  void changeTopLevelLoop(Loop *OldLoop, Loop *NewLoop);

  /// addTopLevelLoop - This adds the specified loop to the collection of
  /// top-level loops.
  void addTopLevelLoop(Loop *New) {
    assert(New->getParentLoop() == 0 && "Loop already in subloop!");
    TopLevelLoops.push_back(New);
  }

  /// removeBlock - This method completely removes BB from all data structures,
  /// including all of the Loop objects it is nested in and our mapping from
  /// BasicBlocks to loops.
  void removeBlock(BasicBlock *BB);

private:
  void Calculate(ETForest &EF);
  Loop *ConsiderForLoop(BasicBlock *BB, ETForest &EF);
  void MoveSiblingLoopInto(Loop *NewChild, Loop *NewParent);
  void InsertLoopInto(Loop *L, Loop *Parent);
};


// Allow clients to walk the list of nested loops...
template <> struct GraphTraits<const Loop*> {
  typedef const Loop NodeType;
  typedef std::vector<Loop*>::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(const Loop *L) { return L; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->end();
  }
};

template <> struct GraphTraits<Loop*> {
  typedef Loop NodeType;
  typedef std::vector<Loop*>::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(Loop *L) { return L; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->end();
  }
};

} // End llvm namespace

// Make sure that any clients of this file link in LoopInfo.cpp
FORCE_DEFINING_FILE_TO_BE_LINKED(LoopInfo)

#endif

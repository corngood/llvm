//=- llvm/Analysis/PostDominators.h - Post Dominator Calculation-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes interfaces to post dominance information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_POST_DOMINATORS_H
#define LLVM_ANALYSIS_POST_DOMINATORS_H

#include "llvm/Analysis/Dominators.h"

namespace llvm {

//===-------------------------------------
/// ImmediatePostDominators Class - Concrete subclass of ImmediateDominatorsBase
/// that is used to compute a normal immediate dominator set.
///
struct ImmediatePostDominators : public ImmediateDominatorsBase {
  ImmediatePostDominators() : ImmediateDominatorsBase(false) {}
  
  virtual bool runOnFunction(Function &F);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  
private:
    struct InfoRec {
      unsigned Semi;
      unsigned Size;
      BasicBlock *Label, *Parent, *Child, *Ancestor;
      
      std::vector<BasicBlock*> Bucket;
      
      InfoRec() : Semi(0), Size(0), Label(0), Parent(0), Child(0), Ancestor(0){}
    };
  
  // Vertex - Map the DFS number to the BasicBlock*
  std::vector<BasicBlock*> Vertex;
  
  // Info - Collection of information used during the computation of idoms.
  std::map<BasicBlock*, InfoRec> Info;
  
  unsigned DFSPass(BasicBlock *V, InfoRec &VInfo, unsigned N);
  void Compress(BasicBlock *V, InfoRec &VInfo);
  BasicBlock *Eval(BasicBlock *v);
  void Link(BasicBlock *V, BasicBlock *W, InfoRec &WInfo);
};

/// PostDominatorSet Class - Concrete subclass of DominatorSetBase that is used
/// to compute the post-dominator set.  Because there can be multiple exit nodes
/// in an LLVM function, we calculate post dominators with a special null block
/// which is the virtual exit node that the real exit nodes all virtually branch
/// to.  Clients should be prepared to see an entry in the dominator sets with a
/// null BasicBlock*.
///
struct PostDominatorSet : public DominatorSetBase {
  PostDominatorSet() : DominatorSetBase(true) {}
  
  virtual bool runOnFunction(Function &F);
  
  /// getAnalysisUsage - This simply provides a dominator set
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<ImmediatePostDominators>();
    AU.setPreservesAll();
  }
  
  // stub - dummy function, just ignore it
  static void stub();
};

/// PostDominatorTree Class - Concrete subclass of DominatorTree that is used to
/// compute the a post-dominator tree.
///
struct PostDominatorTree : public DominatorTreeBase {
  PostDominatorTree() : DominatorTreeBase(true) {}

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    ImmediatePostDominators &IPD = getAnalysis<ImmediatePostDominators>();
    Roots = IPD.getRoots();
    calculate(IPD);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<ImmediatePostDominators>();
  }
private:
  void calculate(const ImmediatePostDominators &IPD);
  Node *getNodeForBlock(BasicBlock *BB);
};


/// PostETForest Class - Concrete subclass of ETForestBase that is used to
/// compute a forwards post-dominator ET-Forest.
struct PostETForest : public ETForestBase {
  PostETForest() : ETForestBase(true) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<ImmediatePostDominators>();
  }

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    ImmediatePostDominators &ID = getAnalysis<ImmediatePostDominators>();
    Roots = ID.getRoots();
    calculate(ID);
    return false;
  }

  void calculate(const ImmediatePostDominators &ID);
  ETNode *getNodeForBlock(BasicBlock *BB);
};


/// PostDominanceFrontier Class - Concrete subclass of DominanceFrontier that is
/// used to compute the a post-dominance frontier.
///
struct PostDominanceFrontier : public DominanceFrontierBase {
  PostDominanceFrontier() : DominanceFrontierBase(true) {}

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    PostDominatorTree &DT = getAnalysis<PostDominatorTree>();
    Roots = DT.getRoots();
    if (const DominatorTree::Node *Root = DT.getRootNode())
      calculate(DT, Root);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<PostDominatorTree>();
  }

  // stub - dummy function, just ignore it
  static void stub();

private:
  const DomSetType &calculate(const PostDominatorTree &DT,
                              const DominatorTree::Node *Node);
};

// Make sure that any clients of this file link in PostDominators.cpp
static IncludeFile
POST_DOMINATOR_INCLUDE_FILE((void*)&PostDominanceFrontier::stub);

} // End llvm namespace

#endif

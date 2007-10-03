//=== llvm/Analysis/DominatorInternals.h - Dominator Calculation -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATOR_INTERNALS_H
#define LLVM_ANALYSIS_DOMINATOR_INTERNALS_H

#include "llvm/Analysis/Dominators.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
//===----------------------------------------------------------------------===//
//
// DominatorTree construction - This pass constructs immediate dominator
// information for a flow-graph based on the algorithm described in this
// document:
//
//   A Fast Algorithm for Finding Dominators in a Flowgraph
//   T. Lengauer & R. Tarjan, ACM TOPLAS July 1979, pgs 121-141.
//
// This implements both the O(n*ack(n)) and the O(n*log(n)) versions of EVAL and
// LINK, but it turns out that the theoretically slower O(n*log(n))
// implementation is actually faster than the "efficient" algorithm (even for
// large CFGs) because the constant overheads are substantially smaller.  The
// lower-complexity version can be enabled with the following #define:
//
#define BALANCE_IDOM_TREE 0
//
//===----------------------------------------------------------------------===//

namespace llvm {

template<class GraphT>
unsigned DFSPass(DominatorTreeBase& DT, typename GraphT::NodeType* V,
                 unsigned N) {
  // This is more understandable as a recursive algorithm, but we can't use the
  // recursive algorithm due to stack depth issues.  Keep it here for
  // documentation purposes.
#if 0
  InfoRec &VInfo = DT.Info[DT.Roots[i]];
  VInfo.Semi = ++N;
  VInfo.Label = V;

  Vertex.push_back(V);        // Vertex[n] = V;
  //Info[V].Ancestor = 0;     // Ancestor[n] = 0
  //Info[V].Child = 0;        // Child[v] = 0
  VInfo.Size = 1;             // Size[v] = 1

  for (succ_iterator SI = succ_begin(V), E = succ_end(V); SI != E; ++SI) {
    InfoRec &SuccVInfo = DT.Info[*SI];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = V;
      N = DTDFSPass(DT, *SI, N);
    }
  }
#else
  std::vector<std::pair<typename GraphT::NodeType*,
                        typename GraphT::ChildIteratorType> > Worklist;
  Worklist.push_back(std::make_pair(V, GraphT::child_begin(V)));
  while (!Worklist.empty()) {
    typename GraphT::NodeType* BB = Worklist.back().first;
    typename GraphT::ChildIteratorType NextSucc = Worklist.back().second;

    // First time we visited this BB?
    if (NextSucc == GraphT::child_begin(BB)) {
      DominatorTree::InfoRec &BBInfo = DT.Info[BB];
      BBInfo.Semi = ++N;
      BBInfo.Label = BB;

      DT.Vertex.push_back(BB);       // Vertex[n] = V;
      //BBInfo[V].Ancestor = 0;   // Ancestor[n] = 0
      //BBInfo[V].Child = 0;      // Child[v] = 0
      BBInfo.Size = 1;            // Size[v] = 1
    }
    
    // If we are done with this block, remove it from the worklist.
    if (NextSucc == GraphT::child_end(BB)) {
      Worklist.pop_back();
      continue;
    }

    // Increment the successor number for the next time we get to it.
    ++Worklist.back().second;
    
    // Visit the successor next, if it isn't already visited.
    typename GraphT::NodeType* Succ = *NextSucc;

    DominatorTree::InfoRec &SuccVInfo = DT.Info[Succ];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = BB;
      Worklist.push_back(std::make_pair(Succ, GraphT::child_begin(Succ)));
    }
  }
#endif
    return N;
}

template<class GraphT>
void Compress(DominatorTreeBase& DT, typename GraphT::NodeType *VIn) {
  std::vector<typename GraphT::NodeType*> Work;
  SmallPtrSet<typename GraphT::NodeType*, 32> Visited;
  typename GraphT::NodeType* VInAncestor = DT.Info[VIn].Ancestor;
  DominatorTreeBase::InfoRec &VInVAInfo = DT.Info[VInAncestor];

  if (VInVAInfo.Ancestor != 0)
    Work.push_back(VIn);
  
  while (!Work.empty()) {
    typename GraphT::NodeType* V = Work.back();
    DominatorTree::InfoRec &VInfo = DT.Info[V];
    typename GraphT::NodeType* VAncestor = VInfo.Ancestor;
    DominatorTreeBase::InfoRec &VAInfo = DT.Info[VAncestor];

    // Process Ancestor first
    if (Visited.insert(VAncestor) &&
        VAInfo.Ancestor != 0) {
      Work.push_back(VAncestor);
      continue;
    } 
    Work.pop_back(); 

    // Update VInfo based on Ancestor info
    if (VAInfo.Ancestor == 0)
      continue;
    typename GraphT::NodeType* VAncestorLabel = VAInfo.Label;
    typename GraphT::NodeType* VLabel = VInfo.Label;
    if (DT.Info[VAncestorLabel].Semi < DT.Info[VLabel].Semi)
      VInfo.Label = VAncestorLabel;
    VInfo.Ancestor = VAInfo.Ancestor;
  }
}

template<class GraphT>
typename GraphT::NodeType* Eval(DominatorTreeBase& DT,
                                typename GraphT::NodeType *V) {
  DominatorTreeBase::InfoRec &VInfo = DT.Info[V];
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  if (VInfo.Ancestor == 0)
    return V;
  Compress<GraphT>(DT, V);
  return VInfo.Label;
#else
  // Lower-complexity but slower implementation
  if (VInfo.Ancestor == 0)
    return VInfo.Label;
  Compress<GraphT>(DT, V);
  GraphT::NodeType* VLabel = VInfo.Label;

  GraphT::NodeType* VAncestorLabel = DT.Info[VInfo.Ancestor].Label;
  if (DT.Info[VAncestorLabel].Semi >= DT.Info[VLabel].Semi)
    return VLabel;
  else
    return VAncestorLabel;
#endif
}

template<class GraphT>
void Link(DominatorTreeBase& DT, typename GraphT::NodeType* V,
          typename GraphT::NodeType* W, DominatorTreeBase::InfoRec &WInfo) {
#if !BALANCE_IDOM_TREE
  // Higher-complexity but faster implementation
  WInfo.Ancestor = V;
#else
  // Lower-complexity but slower implementation
  GraphT::NodeType* WLabel = WInfo.Label;
  unsigned WLabelSemi = DT.Info[WLabel].Semi;
  GraphT::NodeType* S = W;
  InfoRec *SInfo = &DT.Info[S];

  GraphT::NodeType* SChild = SInfo->Child;
  InfoRec *SChildInfo = &DT.Info[SChild];

  while (WLabelSemi < DT.Info[SChildInfo->Label].Semi) {
    GraphT::NodeType* SChildChild = SChildInfo->Child;
    if (SInfo->Size+DT.Info[SChildChild].Size >= 2*SChildInfo->Size) {
      SChildInfo->Ancestor = S;
      SInfo->Child = SChild = SChildChild;
      SChildInfo = &DT.Info[SChild];
    } else {
      SChildInfo->Size = SInfo->Size;
      S = SInfo->Ancestor = SChild;
      SInfo = SChildInfo;
      SChild = SChildChild;
      SChildInfo = &DT.Info[SChild];
    }
  }

  DominatorTreeBase::InfoRec &VInfo = DT.Info[V];
  SInfo->Label = WLabel;

  assert(V != W && "The optimization here will not work in this case!");
  unsigned WSize = WInfo.Size;
  unsigned VSize = (VInfo.Size += WSize);

  if (VSize < 2*WSize)
    std::swap(S, VInfo.Child);

  while (S) {
    SInfo = &DT.Info[S];
    SInfo->Ancestor = V;
    S = SInfo->Child;
  }
#endif
}

template<class NodeT>
void Calculate(DominatorTreeBase& DT, Function& F) {
  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  unsigned N = 0;
  for (unsigned i = 0, e = DT.Roots.size(); i != e; ++i)
    N = DFSPass<GraphTraits<NodeT> >(DT, DT.Roots[i], N);

  for (unsigned i = N; i >= 2; --i) {
    typename GraphTraits<NodeT>::NodeType* W = DT.Vertex[i];
    DominatorTree::InfoRec &WInfo = DT.Info[W];

    // Step #2: Calculate the semidominators of all vertices
    for (typename GraphTraits<Inverse<NodeT> >::ChildIteratorType CI =
         GraphTraits<Inverse<NodeT> >::child_begin(W),
         E = GraphTraits<Inverse<NodeT> >::child_end(W); CI != E; ++CI)
      if (DT.Info.count(*CI)) {  // Only if this predecessor is reachable!
        unsigned SemiU = DT.Info[Eval<GraphTraits<NodeT> >(DT, *CI)].Semi;
        if (SemiU < WInfo.Semi)
          WInfo.Semi = SemiU;
      }

    DT.Info[DT.Vertex[WInfo.Semi]].Bucket.push_back(W);

    typename GraphTraits<NodeT>::NodeType* WParent = WInfo.Parent;
    Link<GraphTraits<NodeT> >(DT, WParent, W, WInfo);

    // Step #3: Implicitly define the immediate dominator of vertices
    std::vector<typename GraphTraits<NodeT>::NodeType*> &WParentBucket =
                                                        DT.Info[WParent].Bucket;
    while (!WParentBucket.empty()) {
      typename GraphTraits<NodeT>::NodeType* V = WParentBucket.back();
      WParentBucket.pop_back();
      typename GraphTraits<NodeT>::NodeType* U =
                                               Eval<GraphTraits<NodeT> >(DT, V);
      DT.IDoms[V] = DT.Info[U].Semi < DT.Info[V].Semi ? U : WParent;
    }
  }

  // Step #4: Explicitly define the immediate dominator of each vertex
  for (unsigned i = 2; i <= N; ++i) {
    typename GraphTraits<NodeT>::NodeType* W = DT.Vertex[i];
    typename GraphTraits<NodeT>::NodeType*& WIDom = DT.IDoms[W];
    if (WIDom != DT.Vertex[DT.Info[W].Semi])
      WIDom = DT.IDoms[WIDom];
  }
  
  if (DT.Roots.empty()) return;
  
  // Add a node for the root.  This node might be the actual root, if there is
  // one exit block, or it may be the virtual exit (denoted by (BasicBlock *)0)
  // which postdominates all real exits if there are multiple exit blocks.
  typename GraphTraits<NodeT>::NodeType* Root = DT.Roots.size() == 1 ? DT.Roots[0]
                                                                     : 0;
  DT.DomTreeNodes[Root] = DT.RootNode = new DomTreeNode(Root, 0);
  
  // Loop over all of the reachable blocks in the function...
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (typename GraphTraits<NodeT>::NodeType* ImmDom = DT.getIDom(I)) {
      // Reachable block.
      DomTreeNode *BBNode = DT.DomTreeNodes[I];
      if (BBNode) continue;  // Haven't calculated this node yet?

      // Get or calculate the node for the immediate dominator
      DomTreeNode *IDomNode = DT.getNodeForBlock(ImmDom);

      // Add a new tree node for this BasicBlock, and link it as a child of
      // IDomNode
      DomTreeNode *C = new DomTreeNode(I, IDomNode);
      DT.DomTreeNodes[I] = IDomNode->addChild(C);
    }
  
  // Free temporary memory used to construct idom's
  DT.IDoms.clear();
  DT.Info.clear();
  std::vector<typename GraphTraits<NodeT>::NodeType*>().swap(DT.Vertex);
  
  // FIXME: This does not work on PostDomTrees.  It seems likely that this is
  // due to an error in the algorithm for post-dominators.  This really should
  // be investigated and fixed at some point.
  // DT.updateDFSNumbers();

  // Start out with the DFS numbers being invalid.  Let them be computed if
  // demanded.
  DT.DFSInfoValid = false;
}

}

#endif

//===- llvm/Analysis/Dominators.h - Dominator Info Calculation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the following classes:
//  1. DominatorTree: Represent the ImmediateDominator as an explicit tree
//     structure.
//  2. ETForest: Efficient data structure for dominance comparisons and 
//     nearest-common-ancestor queries.
//  3. DominanceFrontier: Calculate and hold the dominance frontier for a
//     function.
//
//  These data structures are listed in increasing order of complexity.  It
//  takes longer to calculate the dominator frontier, for example, than the
//  ImmediateDominator mapping.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATORS_H
#define LLVM_ANALYSIS_DOMINATORS_H

#include "llvm/Analysis/ET-Forest.h"
#include "llvm/Pass.h"
#include <set>

namespace llvm {

class Instruction;

template <typename GraphType> struct GraphTraits;

//===----------------------------------------------------------------------===//
/// DominatorBase - Base class that other, more interesting dominator analyses
/// inherit from.
///
class DominatorBase : public FunctionPass {
protected:
  std::vector<BasicBlock*> Roots;
  const bool IsPostDominators;

  inline DominatorBase(bool isPostDom) : Roots(), IsPostDominators(isPostDom) {}
public:
  /// getRoots -  Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  inline const std::vector<BasicBlock*> &getRoots() const { return Roots; }

  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const { return IsPostDominators; }
};

//===----------------------------------------------------------------------===//
/// DominatorTree - Calculate the immediate dominator tree for a function.
///
class DominatorTreeBase : public DominatorBase {
public:
  class Node;
protected:
  std::map<BasicBlock*, Node*> Nodes;
  void reset();
  typedef std::map<BasicBlock*, Node*> NodeMapType;

  Node *RootNode;

  struct InfoRec {
    unsigned Semi;
    unsigned Size;
    BasicBlock *Label, *Parent, *Child, *Ancestor;

    std::vector<BasicBlock*> Bucket;

    InfoRec() : Semi(0), Size(0), Label(0), Parent(0), Child(0), Ancestor(0){}
  };

  std::map<BasicBlock*, BasicBlock*> IDoms;

  // Vertex - Map the DFS number to the BasicBlock*
  std::vector<BasicBlock*> Vertex;

  // Info - Collection of information used during the computation of idoms.
  std::map<BasicBlock*, InfoRec> Info;

public:
  class Node {
    friend class DominatorTree;
    friend struct PostDominatorTree;
    friend class DominatorTreeBase;
    BasicBlock *TheBB;
    Node *IDom;
    std::vector<Node*> Children;
  public:
    typedef std::vector<Node*>::iterator iterator;
    typedef std::vector<Node*>::const_iterator const_iterator;

    iterator begin()             { return Children.begin(); }
    iterator end()               { return Children.end(); }
    const_iterator begin() const { return Children.begin(); }
    const_iterator end()   const { return Children.end(); }

    inline BasicBlock *getBlock() const { return TheBB; }
    inline Node *getIDom() const { return IDom; }
    inline const std::vector<Node*> &getChildren() const { return Children; }

    /// properlyDominates - Returns true iff this dominates N and this != N.
    /// Note that this is not a constant time operation!
    ///
    bool properlyDominates(const Node *N) const {
      const Node *IDom;
      if (this == 0 || N == 0) return false;
      while ((IDom = N->getIDom()) != 0 && IDom != this)
        N = IDom;   // Walk up the tree
      return IDom != 0;
    }

    /// dominates - Returns true iff this dominates N.  Note that this is not a
    /// constant time operation!
    ///
    inline bool dominates(const Node *N) const {
      if (N == this) return true;  // A node trivially dominates itself.
      return properlyDominates(N);
    }
    
  private:
    inline Node(BasicBlock *BB, Node *iDom) : TheBB(BB), IDom(iDom) {}
    inline Node *addChild(Node *C) { Children.push_back(C); return C; }

    void setIDom(Node *NewIDom);
  };

public:
  DominatorTreeBase(bool isPostDom) : DominatorBase(isPostDom) {}
  ~DominatorTreeBase() { reset(); }

  virtual void releaseMemory() { reset(); }

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline Node *getNode(BasicBlock *BB) const {
    NodeMapType::const_iterator i = Nodes.find(BB);
    return (i != Nodes.end()) ? i->second : 0;
  }

  inline Node *operator[](BasicBlock *BB) const {
    return getNode(BB);
  }

  /// getRootNode - This returns the entry node for the CFG of the function.  If
  /// this tree represents the post-dominance relations for a function, however,
  /// this root may be a node with the block == NULL.  This is the case when
  /// there are multiple exit nodes from a particular function.  Consumers of
  /// post-dominance information must be capable of dealing with this
  /// possibility.
  ///
  Node *getRootNode() { return RootNode; }
  const Node *getRootNode() const { return RootNode; }

  //===--------------------------------------------------------------------===//
  // API to update (Post)DominatorTree information based on modifications to
  // the CFG...

  /// createNewNode - Add a new node to the dominator tree information.  This
  /// creates a new node as a child of IDomNode, linking it into the children
  /// list of the immediate dominator.
  ///
  Node *createNewNode(BasicBlock *BB, Node *IDomNode) {
    assert(getNode(BB) == 0 && "Block already in dominator tree!");
    assert(IDomNode && "Not immediate dominator specified for block!");
    return Nodes[BB] = IDomNode->addChild(new Node(BB, IDomNode));
  }

  /// changeImmediateDominator - This method is used to update the dominator
  /// tree information when a node's immediate dominator changes.
  ///
  void changeImmediateDominator(Node *N, Node *NewIDom) {
    assert(N && NewIDom && "Cannot change null node pointers!");
    N->setIDom(NewIDom);
  }

  /// removeNode - Removes a node from the dominator tree.  Block must not
  /// dominate any other blocks.  Invalidates any node pointing to removed
  /// block.
  void removeNode(BasicBlock *BB) {
    assert(getNode(BB) && "Removing node that isn't in dominator tree.");
    Nodes.erase(BB);
  }

  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &OS, const Module* = 0) const;
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
};

//===-------------------------------------
/// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
/// compute a normal dominator tree.
///
class DominatorTree : public DominatorTreeBase {
public:
  DominatorTree() : DominatorTreeBase(false) {}
  
  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }
  
  virtual bool runOnFunction(Function &F);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
private:
  void calculate(Function& F);
  Node *getNodeForBlock(BasicBlock *BB);
  unsigned DFSPass(BasicBlock *V, InfoRec &VInfo, unsigned N);
  void Compress(BasicBlock *V, InfoRec &VInfo);
  BasicBlock *Eval(BasicBlock *v);
  void Link(BasicBlock *V, BasicBlock *W, InfoRec &WInfo);
  inline BasicBlock *getIDom(BasicBlock *BB) const {
      std::map<BasicBlock*, BasicBlock*>::const_iterator I = IDoms.find(BB);
      return I != IDoms.end() ? I->second : 0;
    }
};

//===-------------------------------------
/// DominatorTree GraphTraits specialization so the DominatorTree can be
/// iterable by generic graph iterators.
///
template <> struct GraphTraits<DominatorTree::Node*> {
  typedef DominatorTree::Node NodeType;
  typedef NodeType::iterator  ChildIteratorType;
  
  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType* N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType* N) {
    return N->end();
  }
};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DominatorTree::Node*> {
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }
};


//===-------------------------------------
/// ET-Forest Class - Class used to construct forwards and backwards 
/// ET-Forests
///
class ETForestBase : public DominatorBase {
public:
  ETForestBase(bool isPostDom) : DominatorBase(isPostDom), Nodes(), 
                                 DFSInfoValid(false), SlowQueries(0) {}
  
  virtual void releaseMemory() { reset(); }

  typedef std::map<BasicBlock*, ETNode*> ETMapType;

  void updateDFSNumbers();
    
  /// dominates - Return true if A dominates B.
  ///
  inline bool dominates(BasicBlock *A, BasicBlock *B) {
    if (A == B)
      return true;
    
    ETNode *NodeA = getNode(A);
    ETNode *NodeB = getNode(B);
    
    if (DFSInfoValid)
      return NodeB->DominatedBy(NodeA);
    else {
      // If we end up with too many slow queries, just update the
      // DFS numbers on the theory that we are going to keep querying.
      SlowQueries++;
      if (SlowQueries > 32) {
        updateDFSNumbers();
        return NodeB->DominatedBy(NodeA);
      }
      return NodeB->DominatedBySlow(NodeA);
    }
  }

  // dominates - Return true if A dominates B. This performs the
  // special checks necessary if A and B are in the same basic block.
  bool dominates(Instruction *A, Instruction *B);

  /// properlyDominates - Return true if A dominates B and A != B.
  ///
  bool properlyDominates(BasicBlock *A, BasicBlock *B) {
    return dominates(A, B) && A != B;
  }

  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  const bool isReachableFromEntry(BasicBlock* A);
  
  /// Return the nearest common dominator of A and B.
  BasicBlock *nearestCommonDominator(BasicBlock *A, BasicBlock *B) const  {
    ETNode *NodeA = getNode(A);
    ETNode *NodeB = getNode(B);
    
    ETNode *Common = NodeA->NCA(NodeB);
    if (!Common)
      return NULL;
    return Common->getData<BasicBlock>();
  }
  
  /// Return the immediate dominator of A.
  BasicBlock *getIDom(BasicBlock *A) {
    ETNode *NodeA = getNode(A);
    const ETNode *idom = NodeA->getFather();
    return idom ? idom->getData<BasicBlock>() : 0;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorTree>();
  }
  //===--------------------------------------------------------------------===//
  // API to update Forest information based on modifications
  // to the CFG...

  /// addNewBlock - Add a new block to the CFG, with the specified immediate
  /// dominator.
  ///
  void addNewBlock(BasicBlock *BB, BasicBlock *IDom);

  /// setImmediateDominator - Update the immediate dominator information to
  /// change the current immediate dominator for the specified block
  /// to another block.  This method requires that BB for NewIDom
  /// already have an ETNode, otherwise just use addNewBlock.
  ///
  void setImmediateDominator(BasicBlock *BB, BasicBlock *NewIDom);
  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &OS, const Module* = 0) const;
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
protected:
  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline ETNode *getNode(BasicBlock *BB) const {
    ETMapType::const_iterator i = Nodes.find(BB);
    return (i != Nodes.end()) ? i->second : 0;
  }

  inline ETNode *operator[](BasicBlock *BB) const {
    return getNode(BB);
  }

  void reset();
  ETMapType Nodes;
  bool DFSInfoValid;
  unsigned int SlowQueries;

};

//==-------------------------------------
/// ETForest Class - Concrete subclass of ETForestBase that is used to
/// compute a forwards ET-Forest.

class ETForest : public ETForestBase {
public:
  ETForest() : ETForestBase(false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Roots = DT.getRoots();
    calculate(DT);
    return false;
  }

  void calculate(const DominatorTree &DT);
  ETNode *getNodeForBlock(BasicBlock *BB);
};

//===----------------------------------------------------------------------===//
/// DominanceFrontierBase - Common base class for computing forward and inverse
/// dominance frontiers for a function.
///
class DominanceFrontierBase : public DominatorBase {
public:
  typedef std::set<BasicBlock*>             DomSetType;    // Dom set for a bb
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType; // Dom set map
protected:
  DomSetMapType Frontiers;
public:
  DominanceFrontierBase(bool isPostDom) : DominatorBase(isPostDom) {}

  virtual void releaseMemory() { Frontiers.clear(); }

  // Accessor interface:
  typedef DomSetMapType::iterator iterator;
  typedef DomSetMapType::const_iterator const_iterator;
  iterator       begin()       { return Frontiers.begin(); }
  const_iterator begin() const { return Frontiers.begin(); }
  iterator       end()         { return Frontiers.end(); }
  const_iterator end()   const { return Frontiers.end(); }
  iterator       find(BasicBlock *B)       { return Frontiers.find(B); }
  const_iterator find(BasicBlock *B) const { return Frontiers.find(B); }

  void addBasicBlock(BasicBlock *BB, const DomSetType &frontier) {
    assert(find(BB) == end() && "Block already in DominanceFrontier!");
    Frontiers.insert(std::make_pair(BB, frontier));
  }

  void addToFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    I->second.insert(Node);
  }

  void removeFromFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    assert(I->second.count(Node) && "Node is not in DominanceFrontier of BB");
    I->second.erase(Node);
  }

  /// print - Convert to human readable form
  ///
  virtual void print(std::ostream &OS, const Module* = 0) const;
  void print(std::ostream *OS, const Module* M = 0) const {
    if (OS) print(*OS, M);
  }
};


//===-------------------------------------
/// DominanceFrontier Class - Concrete subclass of DominanceFrontierBase that is
/// used to compute a forward dominator frontiers.
///
class DominanceFrontier : public DominanceFrontierBase {
public:
  DominanceFrontier() : DominanceFrontierBase(false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Roots = DT.getRoots();
    assert(Roots.size() == 1 && "Only one entry block for forward domfronts!");
    calculate(DT, DT[Roots[0]]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorTree>();
  }
private:
  const DomSetType &calculate(const DominatorTree &DT,
                              const DominatorTree::Node *Node);
};


} // End llvm namespace

#endif

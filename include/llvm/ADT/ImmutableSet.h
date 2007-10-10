//===--- ImmutableSet.h - Immutable (functional) set interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ImutAVLTree and ImmutableSet classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_IMSET_H
#define LLVM_ADT_IMSET_H

#include "llvm/Support/Allocator.h"
#include "llvm/ADT/FoldingSet.h"
#include <cassert>

namespace llvm {
  
//===----------------------------------------------------------------------===//    
// Immutable AVL-Tree Definition.
//===----------------------------------------------------------------------===//

template <typename ImutInfo> class ImutAVLFactory;


template <typename ImutInfo >
class ImutAVLTree : public FoldingSetNode {
  struct ComputeIsEqual;
public:
  typedef typename ImutInfo::key_type_ref   key_type_ref;
  typedef typename ImutInfo::value_type     value_type;
  typedef typename ImutInfo::value_type_ref value_type_ref;
  typedef ImutAVLFactory<ImutInfo>          Factory;
  
  friend class ImutAVLFactory<ImutInfo>;
  
  //===----------------------------------------------------===//  
  // Public Interface.
  //===----------------------------------------------------===//  
  
  ImutAVLTree* getLeft() const { return reinterpret_cast<ImutAVLTree*>(Left); }  
  
  ImutAVLTree* getRight() const { return Right; }  
  
  unsigned getHeight() const { return Height; }  
  
  const value_type& getValue() const { return Value; }
  
  ImutAVLTree* find(key_type_ref K) {
    ImutAVLTree *T = this;
    
    while (T) {
      key_type_ref CurrentKey = ImutInfo::KeyOfValue(T->getValue());
      
      if (ImutInfo::isEqual(K,CurrentKey))
        return T;
      else if (ImutInfo::isLess(K,CurrentKey))
        T = T->getLeft();
      else
        T = T->getRight();
    }
    
    return NULL;
  }
  
  unsigned size() const {
    unsigned n = 1;
    
    if (const ImutAVLTree* L = getLeft())  n += L->size();
    if (const ImutAVLTree* R = getRight()) n += R->size();
    
    return n;
  }
  
  
  bool isEqual(const ImutAVLTree& RHS) const {
    // FIXME: Todo.
    return true;    
  }
  
  bool isNotEqual(const ImutAVLTree& RHS) const { return !isEqual(RHS); }
  
  bool contains(const key_type_ref K) { return (bool) find(K); }
  
  template <typename Callback>
  void foreach(Callback& C) {
    if (ImutAVLTree* L = getLeft()) L->foreach(C);
    
    C(Value);    
    
    if (ImutAVLTree* R = getRight()) R->foreach(C);
  }
  
  unsigned verify() const {
    unsigned HL = getLeft() ? getLeft()->verify() : 0;
    unsigned HR = getRight() ? getRight()->verify() : 0;
    
    assert (getHeight() == ( HL > HR ? HL : HR ) + 1 
            && "Height calculation wrong.");
    
    assert ((HL > HR ? HL-HR : HR-HL) <= 2
            && "Balancing invariant violated.");
    
    
    assert (!getLeft()
            || ImutInfo::isLess(ImutInfo::KeyOfValue(getLeft()->getValue()),
                                ImutInfo::KeyOfValue(getValue()))
            && "Value in left child is not less that current value.");
    
    
    assert (!getRight()
            || ImutInfo::isLess(ImutInfo::KeyOfValue(getValue()),
                                ImutInfo::KeyOfValue(getRight()->getValue()))
            && "Current value is not less that value of right child.");
    
    return getHeight();
  }  
  
  //===----------------------------------------------------===//  
  // Internal Values.
  //===----------------------------------------------------===//
  
private:
  uintptr_t        Left;
  ImutAVLTree*     Right;
  unsigned         Height;
  value_type       Value;
  
  //===----------------------------------------------------===//  
  // Profiling or FoldingSet.
  //===----------------------------------------------------===//
  
  static inline
  void Profile(FoldingSetNodeID& ID, ImutAVLTree* L, ImutAVLTree* R,
               unsigned H, value_type_ref V) {    
    ID.AddPointer(L);
    ID.AddPointer(R);
    ID.AddInteger(H);
    ImutInfo::Profile(ID,V);
  }
  
public:
  
  void Profile(FoldingSetNodeID& ID) {
    Profile(ID,getSafeLeft(),getRight(),getHeight(),getValue());    
  }
  
  //===----------------------------------------------------===//    
  // Internal methods (node manipulation; used by Factory).
  //===----------------------------------------------------===//
  
private:
  
  ImutAVLTree(ImutAVLTree* l, ImutAVLTree* r, value_type_ref v, unsigned height)
  : Left(reinterpret_cast<uintptr_t>(l) | 0x1),
  Right(r), Height(height), Value(v) {}
  
  bool isMutable() const { return Left & 0x1; }
  
  ImutAVLTree* getSafeLeft() const { 
    return reinterpret_cast<ImutAVLTree*>(Left & ~0x1);
  }
  
  // Mutating operations.  A tree root can be manipulated as long as
  // its reference has not "escaped" from internal methods of a
  // factory object (see below).  When a tree pointer is externally
  // viewable by client code, the internal "mutable bit" is cleared
  // to mark the tree immutable.  Note that a tree that still has
  // its mutable bit set may have children (subtrees) that are themselves
  // immutable.
  
  void RemoveMutableFlag() {
    assert (Left & 0x1 && "Mutable flag already removed.");
    Left &= ~0x1;
  }
  
  void setLeft(ImutAVLTree* NewLeft) {
    assert (isMutable());
    Left = reinterpret_cast<uintptr_t>(NewLeft) | 0x1;
  }
  
  void setRight(ImutAVLTree* NewRight) {
    assert (isMutable());
    Right = NewRight;
  }
  
  void setHeight(unsigned h) {
    assert (isMutable());
    Height = h;
  }
};

//===----------------------------------------------------------------------===//    
// Immutable AVL-Tree Factory class.
//===----------------------------------------------------------------------===//

template <typename ImutInfo >  
class ImutAVLFactory {
  typedef ImutAVLTree<ImutInfo> TreeTy;
  typedef typename TreeTy::value_type_ref value_type_ref;
  typedef typename TreeTy::key_type_ref   key_type_ref;
  
  typedef FoldingSet<TreeTy> CacheTy;
  
  CacheTy Cache;  
  BumpPtrAllocator Allocator;    
  
  //===--------------------------------------------------===//    
  // Public interface.
  //===--------------------------------------------------===//
  
public:
  ImutAVLFactory() {}
  
  TreeTy* Add(TreeTy* T, value_type_ref V) {
    T = Add_internal(V,T);
    MarkImmutable(T);
    return T;
  }
  
  TreeTy* Remove(TreeTy* T, key_type_ref V) {
    T = Remove_internal(V,T);
    MarkImmutable(T);
    return T;
  }
  
  TreeTy* GetEmptyTree() const { return NULL; }
  
  //===--------------------------------------------------===//    
  // A bunch of quick helper functions used for reasoning
  // about the properties of trees and their children.
  // These have succinct names so that the balancing code
  // is as terse (and readable) as possible.
  //===--------------------------------------------------===//
private:
  
  bool isEmpty(TreeTy* T) const {
    return !T;
  }
  
  unsigned Height(TreeTy* T) const {
    return T ? T->getHeight() : 0;
  }
  
  TreeTy* Left(TreeTy* T) const {
    assert (T);
    return T->getSafeLeft();
  }
  
  TreeTy* Right(TreeTy* T) const {
    assert (T);
    return T->getRight();
  }
  
  value_type_ref Value(TreeTy* T) const {
    assert (T);
    return T->Value;
  }
  
  unsigned IncrementHeight(TreeTy* L, TreeTy* R) const {
    unsigned hl = Height(L);
    unsigned hr = Height(R);
    return ( hl > hr ? hl : hr ) + 1;
  }
  
  //===--------------------------------------------------===//    
  // "CreateNode" is used to generate new tree roots that link
  // to other trees.  The functon may also simply move links
  // in an existing root if that root is still marked mutable.
  // This is necessary because otherwise our balancing code
  // would leak memory as it would create nodes that are
  // then discarded later before the finished tree is
  // returned to the caller.
  //===--------------------------------------------------===//
  
  TreeTy* CreateNode(TreeTy* L, value_type_ref V, TreeTy* R) {
    FoldingSetNodeID ID;      
    unsigned height = IncrementHeight(L,R);
    
    TreeTy::Profile(ID,L,R,height,V);      
    void* InsertPos;
    
    if (TreeTy* T = Cache.FindNodeOrInsertPos(ID,InsertPos))
      return T;
    
    assert (InsertPos != NULL);
    
    // FIXME: more intelligent calculation of alignment.
    TreeTy* T = (TreeTy*) Allocator.Allocate(sizeof(*T),16);
    new (T) TreeTy(L,R,V,height);
    
    Cache.InsertNode(T,InsertPos);
    return T;      
  }
  
  TreeTy* CreateNode(TreeTy* L, TreeTy* OldTree, TreeTy* R) {      
    assert (!isEmpty(OldTree));
    
    if (OldTree->isMutable()) {
      OldTree->setLeft(L);
      OldTree->setRight(R);
      OldTree->setHeight(IncrementHeight(L,R));
      return OldTree;
    }
    else return CreateNode(L, Value(OldTree), R);
  }
  
  /// Balance - Used by Add_internal and Remove_internal to
  ///  balance a newly created tree.
  TreeTy* Balance(TreeTy* L, value_type_ref V, TreeTy* R) {
    
    unsigned hl = Height(L);
    unsigned hr = Height(R);
    
    if (hl > hr + 2) {
      assert (!isEmpty(L) &&
              "Left tree cannot be empty to have a height >= 2.");
      
      TreeTy* LL = Left(L);
      TreeTy* LR = Right(L);
      
      if (Height(LL) >= Height(LR))
        return CreateNode(LL, L, CreateNode(LR,V,R));
      
      assert (!isEmpty(LR) &&
              "LR cannot be empty because it has a height >= 1.");
      
      TreeTy* LRL = Left(LR);
      TreeTy* LRR = Right(LR);
      
      return CreateNode(CreateNode(LL,L,LRL), LR, CreateNode(LRR,V,R));                              
    }
    else if (hr > hl + 2) {
      assert (!isEmpty(R) &&
              "Right tree cannot be empty to have a height >= 2.");
      
      TreeTy* RL = Left(R);
      TreeTy* RR = Right(R);
      
      if (Height(RR) >= Height(RL))
        return CreateNode(CreateNode(L,V,RL), R, RR);
      
      assert (!isEmpty(RL) &&
              "RL cannot be empty because it has a height >= 1.");
      
      TreeTy* RLL = Left(RL);
      TreeTy* RLR = Right(RL);
      
      return CreateNode(CreateNode(L,V,RLL), RL, CreateNode(RLR,R,RR));
    }
    else
      return CreateNode(L,V,R);
  }
  
  /// Add_internal - Creates a new tree that includes the specified
  ///  data and the data from the original tree.  If the original tree
  ///  already contained the data item, the original tree is returned.
  TreeTy* Add_internal(value_type_ref V, TreeTy* T) {
    if (isEmpty(T))
      return CreateNode(T, V, T);
    
    assert (!T->isMutable());
    
    key_type_ref K = ImutInfo::KeyOfValue(V);
    key_type_ref KCurrent = ImutInfo::KeyOfValue(Value(T));
    
    if (ImutInfo::isEqual(K,KCurrent))
      return CreateNode(Left(T), V, Right(T));
    else if (ImutInfo::isLess(K,KCurrent))
      return Balance(Add_internal(V,Left(T)), Value(T), Right(T));
    else
      return Balance(Left(T), Value(T), Add_internal(V,Right(T)));
  }
  
  /// Remove_interal - Creates a new tree that includes all the data
  ///  from the original tree except the specified data.  If the
  ///  specified data did not exist in the original tree, the original
  ///  tree is returned.
  TreeTy* Remove_internal(key_type_ref K, TreeTy* T) {
    if (isEmpty(T))
      return T;
    
    assert (!T->isMutable());
    
    key_type_ref KCurrent = ImutInfo::KeyOfValue(Value(T));
    
    if (ImutInfo::isEqual(K,KCurrent))
      return CombineLeftRightTrees(Left(T),Right(T));
    else if (ImutInfo::isLess(K,KCurrent))
      return Balance(Remove_internal(K,Left(T)), Value(T), Right(T));
    else
      return Balance(Left(T), Value(T), Remove_internal(K,Right(T)));
  }
  
  TreeTy* CombineLeftRightTrees(TreeTy* L, TreeTy* R) {
    if (isEmpty(L)) return R;      
    if (isEmpty(R)) return L;
    
    TreeTy* OldNode;          
    TreeTy* NewRight = RemoveMinBinding(R,OldNode);
    return Balance(L,Value(OldNode),NewRight);
  }
  
  TreeTy* RemoveMinBinding(TreeTy* T, TreeTy*& NodeRemoved) {
    assert (!isEmpty(T));
    
    if (isEmpty(Left(T))) {
      NodeRemoved = T;
      return Right(T);
    }
    
    return Balance(RemoveMinBinding(Left(T),NodeRemoved),Value(T),Right(T));
  }    
  
  /// MarkImmutable - Clears the mutable bits of a root and all of its
  ///  descendants.
  void MarkImmutable(TreeTy* T) {
    if (!T || !T->isMutable())
      return;
    
    T->RemoveMutableFlag();
    MarkImmutable(Left(T));
    MarkImmutable(Right(T));
  }
};


//===----------------------------------------------------------------------===//    
// Trait classes for Profile information.
//===----------------------------------------------------------------------===//

/// Generic profile template.  The default behavior is to invoke the
/// profile method of an object.  Specializations for primitive integers
/// and generic handling of pointers is done below.
template <typename T>
struct ImutProfileInfo {
  typedef const T  value_type;
  typedef const T& value_type_ref;
  
  static inline void Profile(FoldingSetNodeID& ID, value_type_ref X) {
    X.Profile(ID);
  }  
};

/// Profile traits for integers.
template <typename T>
struct ImutProfileInteger {    
  typedef const T  value_type;
  typedef const T& value_type_ref;
  
  static inline void Profile(FoldingSetNodeID& ID, value_type_ref X) {
    ID.AddInteger(X);
  }  
};

#define PROFILE_INTEGER_INFO(X)\
template<> struct ImutProfileInfo<X> : ImutProfileInteger<X> {};

PROFILE_INTEGER_INFO(char)
PROFILE_INTEGER_INFO(unsigned char)
PROFILE_INTEGER_INFO(short)
PROFILE_INTEGER_INFO(unsigned short)
PROFILE_INTEGER_INFO(unsigned)
PROFILE_INTEGER_INFO(signed)
PROFILE_INTEGER_INFO(long)
PROFILE_INTEGER_INFO(unsigned long)
PROFILE_INTEGER_INFO(long long)
PROFILE_INTEGER_INFO(unsigned long long)

#undef PROFILE_INTEGER_INFO

/// Generic profile trait for pointer types.  We treat pointers as
/// references to unique objects.
template <typename T>
struct ImutProfileInfo<T*> {
  typedef const T*   value_type;
  typedef value_type value_type_ref;
  
  static inline void Profile(FoldingSetNodeID &ID, value_type_ref X) {
    ID.AddPointer(X);
  }
};

//===----------------------------------------------------------------------===//    
// Trait classes that contain element comparison operators and type
//  definitions used by ImutAVLTree, ImmutableSet, and ImmutableMap.  These
//  inherit from the profile traits (ImutProfileInfo) to include operations
//  for element profiling.
//===----------------------------------------------------------------------===//


/// ImutContainerInfo - Generic definition of comparison operations for
///   elements of immutable containers that defaults to using
///   std::equal_to<> and std::less<> to perform comparison of elements.
template <typename T>
struct ImutContainerInfo : public ImutProfileInfo<T> {
  typedef typename ImutProfileInfo<T>::value_type      value_type;
  typedef typename ImutProfileInfo<T>::value_type_ref  value_type_ref;
  typedef value_type      key_type;
  typedef value_type_ref  key_type_ref;
  
  static inline key_type_ref KeyOfValue(value_type_ref D) { return D; }
  
  static inline bool isEqual(key_type_ref LHS, key_type_ref RHS) { 
    return std::equal_to<key_type>()(LHS,RHS);
  }
  
  static inline bool isLess(key_type_ref LHS, key_type_ref RHS) {
    return std::less<key_type>()(LHS,RHS);
  }
};

/// ImutContainerInfo - Specialization for pointer values to treat pointers
///  as references to unique objects.  Pointers are thus compared by
///  their addresses.
template <typename T>
struct ImutContainerInfo<T*> : public ImutProfileInfo<T*> {
  typedef typename ImutProfileInfo<T*>::value_type      value_type;
  typedef typename ImutProfileInfo<T*>::value_type_ref  value_type_ref;
  typedef value_type      key_type;
  typedef value_type_ref  key_type_ref;
  
  static inline key_type_ref KeyOfValue(value_type_ref D) { return D; }
  
  static inline bool isEqual(key_type_ref LHS, key_type_ref RHS) {
    return LHS == RHS;
  }
  
  static inline bool isLess(key_type_ref LHS, key_type_ref RHS) {
    return LHS < RHS;
  }
};

//===----------------------------------------------------------------------===//    
// Immutable Set
//===----------------------------------------------------------------------===//

template <typename ValT, typename ValInfo = ImutContainerInfo<ValT> >
class ImmutableSet {
public:
  typedef typename ValInfo::value_type      value_type;
  typedef typename ValInfo::value_type_ref  value_type_ref;
  
private:  
  typedef ImutAVLTree<ValInfo> TreeTy;
  TreeTy* Root;
  
  ImmutableSet(TreeTy* R) : Root(R) {}
  
public:
  
  class Factory {
    typename TreeTy::Factory F;
    
  public:
    Factory() {}
    
    ImmutableSet GetEmptySet() { return ImmutableSet(F.GetEmptyTree()); }
    
    ImmutableSet Add(ImmutableSet Old, value_type_ref V) {
      return ImmutableSet(F.Add(Old.Root,V));
    }
    
    ImmutableSet Remove(ImmutableSet Old, value_type_ref V) {
      return ImmutableSet(F.Remove(Old.Root,V));
    }
    
  private:
    Factory(const Factory& RHS) {};
    void operator=(const Factory& RHS) {};    
  };
  
  friend class Factory;
  
  bool contains(const value_type_ref V) const {
    return Root ? Root->contains(V) : false;
  }
  
  bool operator==(ImmutableSet RHS) const {
    return Root && RHS.Root ? Root->isEqual(*RHS.Root) : Root == RHS.Root;
  }
  
  bool operator!=(ImmutableSet RHS) const {
    return Root && RHS.Root ? Root->isNotEqual(*RHS.Root) : Root != RHS.Root;
  }
  
  bool isEmpty() const { return !Root; }
  
  template <typename Callback>
  void foreach(Callback& C) { if (Root) Root->foreach(C); }
  
  template <typename Callback>
  void foreach() { if (Root) { Callback C; Root->foreach(C); } }
  
  //===--------------------------------------------------===//    
  // For testing.
  //===--------------------------------------------------===//  
  
  void verify() const { if (Root) Root->verify(); }
  unsigned getHeight() const { return Root ? Root->getHeight() : 0; }
};

} // end namespace llvm

#endif

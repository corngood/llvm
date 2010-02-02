//===--- ImmutableIntervalMap.h - Immutable (functional) map  ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ImmutableIntervalMap class.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/ImmutableMap.h"

namespace llvm {

class Interval {
private:
  uint64_t Start;
  uint64_t End;

public:
  Interval(uint64_t S, uint64_t E) : Start(S), End(E) {}

  uint64_t getStart() const { return Start; }
  uint64_t getEnd() const { return End; }
};

template <typename T>
struct ImutIntervalInfo {
  typedef const std::pair<Interval, T> value_type;
  typedef const value_type &value_type_ref;
  typedef const Interval key_type;
  typedef const Interval &key_type_ref;
  typedef const T data_type;
  typedef const T &data_type_ref;

  static key_type_ref KeyOfValue(value_type_ref V) {
    return V.first;
  }

  static data_type_ref DataOfValue(value_type_ref V) {
    return V.second;
  }

  static bool isEqual(key_type_ref L, key_type_ref R) {
    return L.getStart() == R.getStart() && L.getEnd() == R.getEnd();
  }

  static bool isDataEqual(data_type_ref L, data_type_ref R) {
    return ImutContainerInfo<T>::isEqual(L,R);
  }

  static bool isLess(key_type_ref L, key_type_ref R) {
    // Assume L and R does not overlap.
    if (L.getStart() < R.getStart()) {
      assert(L.getEnd() < R.getStart());
      return true;
    } else if (L.getStart() == R.getStart()) {
      assert(L.getEnd() == R.getEnd());
      return false;
    } else {
      assert(L.getStart() > R.getEnd());
      return false;
    }
  }

  static void Profile(FoldingSetNodeID &ID, value_type_ref V) {
    ID.AddInteger(V.first.getStart());
    ID.AddInteger(V.first.getEnd());
    ImutProfileInfo<T>::Profile(ID, V.second);
  }
};

template <typename ImutInfo> class ImutIntervalAVLFactory;

template <typename ImutInfo>
class ImutIntervalAVLFactory : public ImutAVLFactory<ImutInfo> {
  typedef ImutAVLTree<ImutInfo> TreeTy;
  typedef typename ImutInfo::value_type     value_type;
  typedef typename ImutInfo::value_type_ref value_type_ref;
  typedef typename ImutInfo::key_type       key_type;
  typedef typename ImutInfo::key_type_ref   key_type_ref;
  typedef typename ImutInfo::data_type      data_type;
  typedef typename ImutInfo::data_type_ref  data_type_ref;

public:
  TreeTy *Add(TreeTy* T, value_type_ref V) {
    T = Add_internal(V,T);
    MarkImmutable(T);
    return T;
  }

private:
  TreeTy *Add_internal(value_type_ref V, TreeTy *T) {
    key_type_ref K = ImutInfo::KeyOfValue(V);
    T = RemoveAllOverlaps(T, K);
    if (isEmpty(T))
      return CreateNode(NULL, V, NULL);

    assert(!T->isMutable());

    key_type_ref KCurrent = ImutInfo::KeyOfValue(Value(T));

    if (ImutInfo::isLess(K, KCurrent))
      return Balance(Add_internal(V, Left(T)), Value(T), Right(T));
    else
      return Balance(Left(T), Value(T), Add_internal(V, Right(T)));
  }

  // Remove all overlaps from T.
  TreeTy *RemoveAllOverlaps(TreeTy *T, key_type_ref K) {
    TreeTy *OldTree, *NewTree;
    NewTree = T;

    do {
      OldTree = NewTree;
      NewTree = RemoveOverlap(OldTree, K);
    } while (NewTree != OldTree);

    return NewTree;
  }

  // Remove one overlap from T.
  TreeTy *RemoveOverlap(TreeTy *T, key_type_ref K) {
    if (!T)
      return NULL;
    Interval CurrentK = ImutInfo::KeyOfValue(Value(T));

    // If current key does not overlap the inserted key.
    if (CurrentK.getStart() > K.getEnd())
      return RemoveOverlap(Left(T), K);
    else if (CurrentK.getEnd() < K.getStart())
      return RemoveOverlap(Right(T), K);

    // Current key overlaps with the inserted key.
    // Remove the current key.
    TreeTy *OldNode = T;
    T = Remove_internal(CurrentK, T);
    // Add back the unoverlapped part of the current key.
    if (CurrentK.getStart() < K.getStart()) {
      if (CurrentK.getEnd() <= K.getEnd()) {
        Interval NewK(CurrentK.getStart(), K.getStart()-1);
        return Add_internal(std::make_pair<key_type, data_type>(NewK, 
                                     ImutInfo::DataOfValue(Value(OldNode))), T);
      } else {
        Interval NewK1(CurrentK.getStart(), K.getStart()-1);
        T = Add_internal(std::make_pair<key_type, data_type>(NewK1, 
                                     ImutInfo::DataOfValue(Value(OldNode))), T);

        Interval NewK2(K.getEnd()+1, CurrentK.getEnd());
        return Add_internal(std::make_pair<key_type, data_type>(NewK2,
                                     ImutInfo::DataOfValue(Value(OldNode))), T);
      }
    } else {
      if (CurrentK.getEnd() > K.getEnd()) {
        Interval NewK(K.getEnd()+1, CurrentK.getEnd());
        return Add_internal(std::make_pair<key_type, data_type>(NewK, 
                                     ImutInfo::DataOfValue(Value(OldNode))), T);
      }
    }
  }
};

/// ImmutableIntervalMap maps an interval [start, end] to a value. The intervals
/// in the map are guaranteed to be disjoint.
template <typename ValT>
class ImmutableIntervalMap 
  : public ImmutableMap<Interval, ValT, ImutIntervalInfo<ValT> > {

  typedef typename ImutIntervalInfo<ValT>::value_type      value_type;
  typedef typename ImutIntervalInfo<ValT>::value_type_ref  value_type_ref;
  typedef typename ImutIntervalInfo<ValT>::key_type        key_type;
  typedef typename ImutIntervalInfo<ValT>::key_type_ref    key_type_ref;
  typedef typename ImutIntervalInfo<ValT>::data_type       data_type;
  typedef typename ImutIntervalInfo<ValT>::data_type_ref   data_type_ref;
  typedef ImutAVLTree<ImutIntervalInfo<ValT> > TreeTy;

public:
  explicit ImmutableIntervalMap(TreeTy *R) 
    : ImmutableMap<Interval, ValT, ImutIntervalInfo<ValT> >(R) {}

  class Factory {
    ImutIntervalAVLFactory<ImutIntervalInfo<ValT> > F;

  public:
    ImmutableIntervalMap GetEmptyMap() { 
      return ImmutableIntervalMap(F.GetEmptyTree()); 
    }

    ImmutableIntervalMap Add(ImmutableIntervalMap Old, 
                             key_type_ref K, data_type_ref D) {
      TreeTy *T = F.Add(Old.Root, std::make_pair<key_type, data_type>(K, D));
      return ImmutableIntervalMap(F.GetCanonicalTree(T));
    }

    ImmutableIntervalMap Remove(ImmutableIntervalMap Old, key_type_ref K) {
      TreeTy *T = F.Remove(Old.Root, K);
      return ImmutableIntervalMap(F.GetCanonicalTree(T));
    }
  };
};

} // end namespace llvm

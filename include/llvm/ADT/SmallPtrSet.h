//===- llvm/ADT/SmallPtrSet.h - 'Normally small' pointer set ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallPtrSet class.  See the doxygen comment for
// SmallPtrSetImpl for more details on the algorithm used.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLPTRSET_H
#define LLVM_ADT_SMALLPTRSET_H

#include <cassert>
#include <cstring>
#include "llvm/Support/DataTypes.h"

namespace llvm {

/// SmallPtrSetImpl - This is the common code shared among all the
/// SmallPtrSet<>'s, which is almost everything.  SmallPtrSet has two modes, one
/// for small and one for large sets.
///
/// Small sets use an array of pointers allocated in the SmallPtrSet object,
/// which is treated as a simple array of pointers.  When a pointer is added to
/// the set, the array is scanned to see if the element already exists, if not
/// the element is 'pushed back' onto the array.  If we run out of space in the
/// array, we grow into the 'large set' case.  SmallSet should be used when the
/// sets are often small.  In this case, no memory allocation is used, and only
/// light-weight and cache-efficient scanning is used.
///
/// Large sets use a classic exponentially-probed hash table.  Empty buckets are
/// represented with an illegal pointer value (-1) to allow null pointers to be
/// inserted.  Tombstones are represented with another illegal pointer value
/// (-2), to allow deletion.  The hash table is resized when the table is 3/4 or
/// more.  When this happens, the table is doubled in size.
///
class SmallPtrSetImpl {
protected:
  /// CurArray - This is the current set of buckets.  If it points to
  /// SmallArray, then the set is in 'small mode'.
  void **CurArray;
  /// CurArraySize - The allocated size of CurArray, always a power of two.
  /// Note that CurArray points to an array that has CurArraySize+1 elements in
  /// it, so that the end iterator actually points to valid memory.
  unsigned CurArraySize;
  
  // If small, this is # elts allocated consequtively
  unsigned NumElements;
  unsigned NumTombstones;
  void *SmallArray[1];  // Must be last ivar.

  // Helper to copy construct a SmallPtrSet.
  SmallPtrSetImpl(const SmallPtrSetImpl& that);
public:
  SmallPtrSetImpl(unsigned SmallSize) {
    assert(SmallSize && (SmallSize & (SmallSize-1)) == 0 &&
           "Initial size must be a power of two!");
    CurArray = &SmallArray[0];
    CurArraySize = SmallSize;
    // The end pointer, always valid, is set to a valid element to help the
    // iterator.
    CurArray[SmallSize] = 0;
    clear();
  }
  ~SmallPtrSetImpl() {
    if (!isSmall())
      delete[] CurArray;
  }
  
  bool empty() const { return size() == 0; }
  unsigned size() const { return NumElements; }
  
  static void *getTombstoneMarker() { return reinterpret_cast<void*>(-2); }
  static void *getEmptyMarker() {
    // Note that -1 is chosen to make clear() efficiently implementable with
    // memset and because it's not a valid pointer value.
    return reinterpret_cast<void*>(-1);
  }
  
  void clear() {
    // Fill the array with empty markers.
    memset(CurArray, -1, CurArraySize*sizeof(void*));
    NumElements = 0;
    NumTombstones = 0;
  }
  
  /// insert - This returns true if the pointer was new to the set, false if it
  /// was already in the set.
  bool insert(void *Ptr);
  
  template <typename IterT>
  void insert(IterT I, IterT E) {
    for (; I != E; ++I)
      insert((void*)*I);
  }
  
  /// erase - If the set contains the specified pointer, remove it and return
  /// true, otherwise return false.
  bool erase(void *Ptr);
  
  bool count(void *Ptr) const {
    if (isSmall()) {
      // Linear search for the item.
      for (void *const *APtr = SmallArray, *const *E = SmallArray+NumElements;
           APtr != E; ++APtr)
        if (*APtr == Ptr)
          return true;
      return false;
    }
    
    // Big set case.
    return *FindBucketFor(Ptr) == Ptr;
  }
  
private:
  bool isSmall() const { return CurArray == &SmallArray[0]; }

  unsigned Hash(void *Ptr) const {
    return ((uintptr_t)Ptr >> 4) & (CurArraySize-1);
  }
  void * const *FindBucketFor(void *Ptr) const;
  
  /// Grow - Allocate a larger backing store for the buckets and move it over.
  void Grow();
  
  void operator=(const SmallPtrSetImpl &RHS);  // DO NOT IMPLEMENT.
protected:
  void CopyFrom(const SmallPtrSetImpl &RHS);
};

/// SmallPtrSetIteratorImpl - This is the common base class shared between all
/// instances of SmallPtrSetIterator.
class SmallPtrSetIteratorImpl {
protected:
  void *const *Bucket;
public:
  SmallPtrSetIteratorImpl(void *const *BP) : Bucket(BP) {
    AdvanceIfNotValid();
  }
  
  bool operator==(const SmallPtrSetIteratorImpl &RHS) const {
    return Bucket == RHS.Bucket;
  }
  bool operator!=(const SmallPtrSetIteratorImpl &RHS) const {
    return Bucket != RHS.Bucket;
  }
  
protected:
  /// AdvanceIfNotValid - If the current bucket isn't valid, advance to a bucket
  /// that is.   This is guaranteed to stop because the end() bucket is marked
  /// valid.
  void AdvanceIfNotValid() {
    while (*Bucket == SmallPtrSetImpl::getEmptyMarker() ||
           *Bucket == SmallPtrSetImpl::getTombstoneMarker())
      ++Bucket;
  }
};

/// SmallPtrSetIterator - This implements a const_iterator for SmallPtrSet.
template<typename PtrTy>
class SmallPtrSetIterator : public SmallPtrSetIteratorImpl {
public:
  SmallPtrSetIterator(void *const *BP) : SmallPtrSetIteratorImpl(BP) {}

  // Most methods provided by baseclass.
  
  PtrTy operator*() const {
    return static_cast<PtrTy>(*Bucket);
  }
  
  inline SmallPtrSetIterator& operator++() {          // Preincrement
    ++Bucket;
    AdvanceIfNotValid();
    return *this;
  }
  
  SmallPtrSetIterator operator++(int) {        // Postincrement
    SmallPtrSetIterator tmp = *this; ++*this; return tmp;
  }
};

/// NextPowerOfTwo - This is a helper template that rounds N up to the next
/// power of two.
template<unsigned N>
struct NextPowerOfTwo;

/// NextPowerOfTwoH - If N is not a power of two, increase it.  This is a helper
/// template used to implement NextPowerOfTwo.
template<unsigned N, bool isPowerTwo>
struct NextPowerOfTwoH {
  enum { Val = N };
};
template<unsigned N>
struct NextPowerOfTwoH<N, false> {
  enum {
    // We could just use NextVal = N+1, but this converges faster.  N|(N-1) sets
    // the right-most zero bits to one all at once, e.g. 0b0011000 -> 0b0011111.
    Val = NextPowerOfTwo<(N|(N-1)) + 1>::Val
  };
};

template<unsigned N>
struct NextPowerOfTwo {
  enum { Val = NextPowerOfTwoH<N, (N&(N-1)) == 0>::Val };
};


/// SmallPtrSet - This class implements a set which is optimizer for holding
/// SmallSize or less elements.  This internally rounds up SmallSize to the next
/// power of two if it is not already a power of two.  See the comments above
/// SmallPtrSetImpl for details of the algorithm.
template<class PtrType, unsigned SmallSize>
class SmallPtrSet : public SmallPtrSetImpl {
  // Make sure that SmallSize is a power of two, round up if not.
  enum { SmallSizePowTwo = NextPowerOfTwo<SmallSize>::Val };
  void *SmallArray[SmallSizePowTwo];
public:
  SmallPtrSet() : SmallPtrSetImpl(NextPowerOfTwo<SmallSizePowTwo>::Val) {}
  SmallPtrSet(const SmallPtrSet &that) : SmallPtrSetImpl(that) {}
  
  template<typename It>
  SmallPtrSet(It I, It E)
    : SmallPtrSetImpl(NextPowerOfTwo<SmallSizePowTwo>::Val) {
    insert(I, E);
  }
  
  typedef SmallPtrSetIterator<PtrType> iterator;
  typedef SmallPtrSetIterator<PtrType> const_iterator;
  inline iterator begin() const {
    return iterator(CurArray);
  }
  inline iterator end() const {
    return iterator(CurArray+CurArraySize);
  }
  
  // Allow assignment from any smallptrset with the same element type even if it
  // doesn't have the same smallsize.
  const SmallPtrSet<PtrType, SmallSize>
  operator=(const SmallPtrSet<PtrType, SmallSize> &RHS) {
    CopyFrom(RHS);
    return *this;
  }

};

}

#endif

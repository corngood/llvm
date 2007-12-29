//===--- Allocator.h - Simple memory allocation abstraction -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MallocAllocator and BumpPtrAllocator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALLOCATOR_H
#define LLVM_SUPPORT_ALLOCATOR_H

#include "llvm/Support/AlignOf.h"
#include <cstdlib>

namespace llvm {
    
class MallocAllocator {
public:
  MallocAllocator() {}
  ~MallocAllocator() {}
  
  void Reset() {}
  void *Allocate(unsigned Size, unsigned Alignment) { return malloc(Size); }
  
  template <typename T>
  void *Allocate() { return reinterpret_cast<T*>(malloc(sizeof(T))); }
  
  void Deallocate(void *Ptr) { free(Ptr); }
  void PrintStats() const {}
};

/// BumpPtrAllocator - This allocator is useful for containers that need very
/// simple memory allocation strategies.  In particular, this just keeps
/// allocating memory, and never deletes it until the entire block is dead. This
/// makes allocation speedy, but must only be used when the trade-off is ok.
class BumpPtrAllocator {
  void *TheMemory;
public:
  BumpPtrAllocator();
  ~BumpPtrAllocator();
  
  void Reset();
  void *Allocate(unsigned Size, unsigned Alignment);

  template <typename T>
  void *Allocate() { 
    return reinterpret_cast<T*>(Allocate(sizeof(T),AlignOf<T>::Alignment));
  }

  
  void Deallocate(void *Ptr) {}
  void PrintStats() const;
};

}  // end namespace llvm

#endif

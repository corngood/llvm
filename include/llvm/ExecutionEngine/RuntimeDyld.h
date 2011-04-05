//===-- RuntimeDyld.h - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface for the runtime dynamic linker facilities of the MC-JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_H
#define LLVM_RUNTIME_DYLD_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Memory.h"

namespace llvm {

class RuntimeDyldImpl;
class MemoryBuffer;

// RuntimeDyld clients often want to handle the memory management of
// what gets placed where. For JIT clients, this is an abstraction layer
// over the JITMemoryManager, which references objects by their source
// representations in LLVM IR.
// FIXME: As the RuntimeDyld fills out, additional routines will be needed
//        for the varying types of objects to be allocated.
class RTDyldMemoryManager {
  RTDyldMemoryManager(const RTDyldMemoryManager&);  // DO NOT IMPLEMENT
  void operator=(const RTDyldMemoryManager&);       // DO NOT IMPLEMENT
public:
  RTDyldMemoryManager() {}
  virtual ~RTDyldMemoryManager() {}

  // Allocate ActualSize bytes, or more, for the named function. Return
  // a pointer to the allocated memory and update Size to reflect how much
  // memory was acutally allocated.
  virtual uint64_t startFunctionBody(const char *Name, uintptr_t &Size) = 0;

  // Mark the end of the function, including how much of the allocated
  // memory was actually used.
  virtual void endFunctionBody(const char *Name, uint64_t FunctionStart,
                               uint64_t FunctionEnd) = 0;
};

class RuntimeDyld {
  RuntimeDyld(const RuntimeDyld &);     // DO NOT IMPLEMENT
  void operator=(const RuntimeDyld &);  // DO NOT IMPLEMENT

  // RuntimeDyldImpl is the actual class. RuntimeDyld is just the public
  // interface.
  RuntimeDyldImpl *Dyld;
public:
  RuntimeDyld(RTDyldMemoryManager*);
  ~RuntimeDyld();

  bool loadObject(MemoryBuffer *InputBuffer);
  uint64_t getSymbolAddress(StringRef Name);
  void reassignSymbolAddress(StringRef Name, uint64_t Addr);
  // FIXME: Should be parameterized to get the memory block associated with
  // a particular loaded object.
  sys::MemoryBlock getMemoryBlock();
  StringRef getErrorString();
};

} // end namespace llvm

#endif

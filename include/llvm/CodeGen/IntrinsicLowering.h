//===-- IntrinsicLowering.h - Intrinsic Function Lowering -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IntrinsicLowering interface.  This interface allows
// addition of domain-specific or front-end specific intrinsics to LLVM without
// having to modify all of the code generators to support the new intrinsic.
// Later, as desired, targets can incrementally add support for particular
// intrinsic functions, as desired, to generate better code.
//
// If a code generator cannot handle or does not know about an intrinsic
// function, it will use the intrinsic lowering interface to change an intrinsic
// function name into a concrete function name which can be used to implement
// the functionality of the intrinsic.  For example, llvm.memcpy can be
// implemented as a call to the math library 'memcpy' function if the target
// doesn't have hardware support for the intrinsic, or if it has not yet been
// implemented yet.
//
// Another use for this interface is the addition of domain-specific intrinsics.
// The default implementation of this interface would then lower the intrinsics
// to noop calls, allowing the direct execution of programs with instrumentation
// or other hooks placed in them.  When a specific tool or flag is used, a
// different implementation of these interfaces may be used, which activates the
// intrinsics in some way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INTRINSICLOWERING_H
#define LLVM_CODEGEN_INTRINSICLOWERING_H

#include "llvm/Intrinsics.h"

namespace llvm {
  class CallInst;
  class Module;

  class IntrinsicLowering {
  public:
    IntrinsicLowering() {}
    virtual ~IntrinsicLowering() {}

    /// AddPrototypes - This method, if called, causes all of the prototypes
    /// that might be needed by an intrinsic lowering implementation to be
    /// inserted into the module specified.
    virtual void AddPrototypes(Module &M) = 0;

    /// LowerIntrinsicCall - This method returns the LLVM function which should
    /// be used to implement the specified intrinsic function call.  If an
    /// intrinsic function must be implemented by the code generator (such as
    /// va_start), this function should print a message and abort.
    ///
    /// Otherwise, if an intrinsic function call can be lowered, the code to
    /// implement it (often a call to a non-intrinsic function) is inserted
    /// _after_ the call instruction and the call is deleted.  The caller must
    /// be capable of handling this kind of change.
    ///
    virtual void LowerIntrinsicCall(CallInst *CI) = 0;
  };

  /// DefaultIntrinsicLower - This is the default intrinsic lowering pass which
  /// is used if no other one is specified.  Custom intrinsic lowering
  /// implementations should pass any unhandled intrinsics to this
  /// implementation to allow for future extensibility.
  struct DefaultIntrinsicLowering : public IntrinsicLowering {
    virtual void AddPrototypes(Module &M);
    virtual void LowerIntrinsicCall(CallInst *CI);
  };
}

#endif

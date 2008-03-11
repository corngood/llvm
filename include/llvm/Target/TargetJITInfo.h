//===- Target/TargetJITInfo.h - Target Information for JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an abstract interface used by the Just-In-Time code
// generator to perform target-specific activities, such as emitting stubs.  If
// a TargetMachine supports JIT code generation, it should provide one of these
// objects through the getJITInfo() method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETJITINFO_H
#define LLVM_TARGET_TARGETJITINFO_H

#include <vector>
#include <cassert>
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class Function;
  class MachineBasicBlock;
  class MachineCodeEmitter;
  class MachineRelocation;

  /// TargetJITInfo - Target specific information required by the Just-In-Time
  /// code generator.
  class TargetJITInfo {
  public:
    virtual ~TargetJITInfo() {}

    /// replaceMachineCodeForFunction - Make it so that calling the function
    /// whose machine code is at OLD turns into a call to NEW, perhaps by
    /// overwriting OLD with a branch to NEW.  This is used for self-modifying
    /// code.
    ///
    virtual void replaceMachineCodeForFunction(void *Old, void *New) = 0;

    /// emitGlobalValueLazyPtr - Use the specified MachineCodeEmitter object to
    /// emit a lazy pointer which contains the address of the specified GV.
    virtual void *emitGlobalValueLazyPtr(void *GV, MachineCodeEmitter &MCE) {
      assert(0 && "This target doesn't implement emitGlobalValueLazyPtr!");
      return 0;
    }

    /// emitFunctionStub - Use the specified MachineCodeEmitter object to emit a
    /// small native function that simply calls the function at the specified
    /// address.  Return the address of the resultant function.
    virtual void *emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
      assert(0 && "This target doesn't implement emitFunctionStub!");
      return 0;
    }

    /// getPICJumpTableEntry - Returns the value of the jumptable entry for the
    /// specific basic block.
    virtual intptr_t getPICJumpTableEntry(intptr_t BB, intptr_t JTBase) {
      assert(0 && "This target doesn't implement getPICJumpTableEntry!");
      return 0;
    }

    /// LazyResolverFn - This typedef is used to represent the function that
    /// unresolved call points should invoke.  This is a target specific
    /// function that knows how to walk the stack and find out which stub the
    /// call is coming from.
    typedef void (*LazyResolverFn)();

    /// JITCompilerFn - This typedef is used to represent the JIT function that
    /// lazily compiles the function corresponding to a stub.  The JIT keeps
    /// track of the mapping between stubs and LLVM Functions, the target
    /// provides the ability to figure out the address of a stub that is called
    /// by the LazyResolverFn.
    typedef void* (*JITCompilerFn)(void *);

    /// getLazyResolverFunction - This method is used to initialize the JIT,
    /// giving the target the function that should be used to compile a
    /// function, and giving the JIT the target function used to do the lazy
    /// resolving.
    virtual LazyResolverFn getLazyResolverFunction(JITCompilerFn) {
      assert(0 && "Not implemented for this target!");
      return 0;
    }

    /// relocate - Before the JIT can run a block of code that has been emitted,
    /// it must rewrite the code to contain the actual addresses of any
    /// referenced global symbols.
    virtual void relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
      assert(NumRelocs == 0 && "This target does not have relocations!");
    }

    /// needsGOT - Allows a target to specify that it would like the
    // JIT to manage a GOT for it.
    bool needsGOT() const { return useGOT; }

  protected:
    bool useGOT;
  };
} // End llvm namespace

#endif

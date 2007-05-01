//===- RSProfiling.cpp - Various profiling using random sampling ----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the abstract interface that a profiler must implement to
// support the random profiling transform.
//
//===----------------------------------------------------------------------===//

namespace llvm {
  //===--------------------------------------------------------------------===//
  /// RSProfilers - The basic Random Sampling Profiler Interface  Any profiler 
  /// that implements this interface can be transformed by the random sampling
  /// pass to be sample based rather than always on.
  ///
  /// The only exposed function can be queried to find out if an instruction
  /// was original or if it was inserted by the profiler.  Implementations of
  /// this interface are expected to chain to other implementations, such that
  /// multiple profilers can be support simultaniously.
  struct RSProfilers : public ModulePass {
    static const int ID; // Pass identification, replacement for typeinfo
    RSProfilers() : ModulePass((intptr_t)&ID) {}

    /// isProfiling - This method returns true if the value passed it was 
    /// inserted by the profiler.
    virtual bool isProfiling(Value* v) = 0;
  };
}

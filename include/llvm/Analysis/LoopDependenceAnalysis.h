//===- llvm/Analysis/LoopDependenceAnalysis.h --------------- -*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// LoopDependenceAnalysis is an LLVM pass that analyses dependences in memory
// accesses in loops.
//
// Please note that this is work in progress and the interface is subject to
// change.
//
// TODO: adapt as interface progresses
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_DEPENDENCE_ANALYSIS_H
#define LLVM_ANALYSIS_LOOP_DEPENDENCE_ANALYSIS_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/Allocator.h"
#include <iosfwd>

namespace llvm {

class AliasAnalysis;
class AnalysisUsage;
class ScalarEvolution;
class Value;
class raw_ostream;

class LoopDependenceAnalysis : public LoopPass {
  AliasAnalysis *AA;
  ScalarEvolution *SE;

  /// L - The loop we are currently analysing.
  Loop *L;

  /// TODO: doc
  enum DependenceResult { Independent = 0, Dependent = 1, Unknown = 2 };

  /// DependencePair - Represents a data dependence relation between to memory
  /// reference instructions.
  ///
  /// TODO: add subscripts vector
  struct DependencePair : public FastFoldingSetNode {
    Value *A;
    Value *B;
    DependenceResult Result;

    DependencePair(const FoldingSetNodeID &ID, Value *a, Value *b) :
        FastFoldingSetNode(ID), A(a), B(b), Result(Unknown) {}
  };

  /// findOrInsertDependencePair - Return true if a DependencePair for the
  /// given Values already exists, false if a new DependencePair had to be
  /// created. The third argument is set to the pair found or created.
  bool findOrInsertDependencePair(Value*, Value*, DependencePair*&);

  /// TODO: doc
  DependenceResult analysePair(DependencePair *P) const;

public:
  static char ID; // Class identification, replacement for typeinfo
  LoopDependenceAnalysis() : LoopPass(&ID) {}

  /// isDependencePair - Check wether two values can possibly give rise to a
  /// data dependence: that is the case if both are instructions accessing
  /// memory and at least one of those accesses is a write.
  bool isDependencePair(const Value*, const Value*) const;

  /// depends - Return a boolean indicating if there is a data dependence
  /// between two instructions.
  bool depends(Value*, Value*);

  bool runOnLoop(Loop*, LPPassManager&);
  virtual void releaseMemory();
  virtual void getAnalysisUsage(AnalysisUsage&) const;
  void print(raw_ostream&, const Module* = 0) const;
  virtual void print(std::ostream&, const Module* = 0) const;

private:
  FoldingSet<DependencePair> Pairs;
  BumpPtrAllocator PairAllocator;
}; // class LoopDependenceAnalysis


// createLoopDependenceAnalysisPass - This creates an instance of the
// LoopDependenceAnalysis pass.
//
LoopPass *createLoopDependenceAnalysisPass();

} // namespace llvm

#endif /* LLVM_ANALYSIS_LOOP_DEPENDENCE_ANALYSIS_H */

//===- llvm/Analysis/MemoryDependenceAnalysis.h - Memory Deps  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MemoryDependenceAnalysis analysis pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORY_DEPENDENCE_H
#define LLVM_ANALYSIS_MEMORY_DEPENDENCE_H

#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  class Function;
  class FunctionPass;
  class Instruction;
  class CallSite;

  /// MemoryDependenceAnalysis - This is an analysis that determines, for a
  /// given memory operation, what preceding memory operations it depends on.
  /// It builds on alias analysis information, and tries to provide a lazy,
  /// caching interface to a common kind of alias information query.
  class MemoryDependenceAnalysis : public FunctionPass {
  public:
    /// DepType - This enum is used to indicate what flavor of dependence this
    /// is.  If the type is Normal, there is an associated instruction pointer.
    enum DepType {
      /// Normal - This is a normal instruction dependence.  The pointer member
      /// of the DepResultTy pair holds the instruction.
      Normal = 0,

      /// None - This dependence type indicates that the query does not depend
      /// on any instructions, either because it scanned to the start of the
      /// function or it scanned to the definition of the memory
      /// (alloca/malloc).
      None,
      
      /// NonLocal - This marker indicates that the query has no dependency in
      /// the specified block.  To find out more, the client should query other
      /// predecessor blocks.
      NonLocal,
      
      /// Dirty - This is an internal marker indicating that that a cache entry
      /// is dirty.
      Dirty
    };
    typedef PointerIntPair<Instruction*, 2, DepType> DepResultTy;
  private:
    // A map from instructions to their dependency, with a boolean
    // flags for whether this mapping is confirmed or not.
    typedef DenseMap<Instruction*,
                     std::pair<DepResultTy, bool> > LocalDepMapType;
    LocalDepMapType LocalDeps;

    // A map from instructions to their non-local dependencies.
    typedef DenseMap<Instruction*,
                     DenseMap<BasicBlock*, DepResultTy> > nonLocalDepMapType;
    nonLocalDepMapType depGraphNonLocal;
    
    // A reverse mapping from dependencies to the dependees.  This is
    // used when removing instructions to keep the cache coherent.
    typedef DenseMap<DepResultTy,
                     SmallPtrSet<Instruction*, 4> > reverseDepMapType;
    reverseDepMapType reverseDep;
    
    // A reverse mapping form dependencies to the non-local dependees.
    reverseDepMapType reverseDepNonLocal;
    
  public:
    MemoryDependenceAnalysis() : FunctionPass(&ID) {}
    static char ID;

    /// Pass Implementation stuff.  This doesn't do any analysis.
    ///
    bool runOnFunction(Function &) {return false; }
    
    /// Clean up memory in between runs
    void releaseMemory() {
      LocalDeps.clear();
      depGraphNonLocal.clear();
      reverseDep.clear();
      reverseDepNonLocal.clear();
    }

    /// getAnalysisUsage - Does not modify anything.  It uses Value Numbering
    /// and Alias Analysis.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    /// getDependency - Return the instruction on which a memory operation
    /// depends, starting with start.
    DepResultTy getDependency(Instruction *query, Instruction *start = 0,
                              BasicBlock *block = 0);
    
    /// getNonLocalDependency - Fills the passed-in map with the non-local 
    /// dependencies of the queries.  The map will contain NonLocal for
    /// blocks between the query and its dependencies.
    void getNonLocalDependency(Instruction* query,
                               DenseMap<BasicBlock*, DepResultTy> &resp);
    
    /// removeInstruction - Remove an instruction from the dependence analysis,
    /// updating the dependence of instructions that previously depended on it.
    void removeInstruction(Instruction *InstToRemove);
    
    /// dropInstruction - Remove an instruction from the analysis, making 
    /// absolutely conservative assumptions when updating the cache.  This is
    /// useful, for example when an instruction is changed rather than removed.
    void dropInstruction(Instruction *InstToDrop);
    
  private:
    /// verifyRemoved - Verify that the specified instruction does not occur
    /// in our internal data structures.
    void verifyRemoved(Instruction *Inst) const;
    
    DepResultTy getCallSiteDependency(CallSite C, Instruction* start,
                                      BasicBlock* block);
    void nonLocalHelper(Instruction* query, BasicBlock* block,
                        DenseMap<BasicBlock*, DepResultTy>& resp);
  };

} // End llvm namespace

#endif

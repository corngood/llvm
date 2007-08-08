//===- llvm/Analysis/MemoryDependenceAnalysis.h - Memory Deps  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an analysis that determines, for a given memory operation,
// what preceding memory operations it depends on.  It builds on alias analysis
// information, and tries to provide a lazy, caching interface to a common kind
// of alias information query.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORY_DEPENDENCE_H
#define LLVM_ANALYSIS_MEMORY_DEPENDENCE_H

#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Function;
class FunctionPass;
class Instruction;

class MemoryDependenceAnalysis : public FunctionPass {
  private:
    // A map from instructions to their dependency, with a boolean
    // flags for whether this mapping is confirmed or not
    typedef DenseMap<Instruction*, std::pair<const Instruction*, bool> > 
            depMapType;
    depMapType depGraphLocal;

    // A reverse mapping form dependencies to the dependees.  This is
    // used when removing instructions to keep the cache coherent.
    typedef DenseMap<const Instruction*, SmallPtrSet<Instruction*, 4> >
            reverseDepMapType;
    reverseDepMapType reverseDep;
    
  public:
    // Special marker indicating that the query has no dependency
    // in the specified block.
    static const Instruction* NonLocal;
    
    // Special marker indicating that the query has no dependency at all
    static const Instruction* None;
    
    static char ID; // Class identification, replacement for typeinfo
    MemoryDependenceAnalysis() : FunctionPass((intptr_t)&ID) {}

    /// Pass Implementation stuff.  This doesn't do any analysis.
    ///
    bool runOnFunction(Function &) {return false; }
    
    /// Clean up memory in between runs
    void releaseMemory() {
      depGraphLocal.clear();
      reverseDep.clear();
    }

    /// getAnalysisUsage - Does not modify anything.  It uses Value Numbering
    /// and Alias Analysis.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    /// getDependency - Return the instruction on which a memory operation
    /// depends, starting with start.
    const Instruction* getDependency(Instruction* query, Instruction* start = 0,
                               BasicBlock* block = 0);
    
    void getNonLocalDependency(Instruction* query,
                               DenseMap<BasicBlock*, Value*>& resp);
    
    /// removeInstruction - Remove an instruction from the dependence analysis,
    /// updating the dependence of instructions that previously depended on it.
    void removeInstruction(Instruction* rem);
    
  private:
    const Instruction* getCallSiteDependency(CallSite C, Instruction* start,
                                       BasicBlock* block);
    void nonLocalHelper(Instruction* query, BasicBlock* block,
                        DenseMap<BasicBlock*, Value*>& resp);
  };

} // End llvm namespace

#endif

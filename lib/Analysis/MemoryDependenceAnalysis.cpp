//===- MemoryDependenceAnalysis.cpp - Mem Deps Implementation  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis that determines, for a given memory
// operation, what preceding memory operations it depends on.  It builds on 
// alias analysis information, and tries to provide a lazy, caching interface to 
// a common kind of alias information query.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

char MemoryDependenceAnalysis::ID = 0;
  
Instruction* MemoryDependenceAnalysis::NonLocal = (Instruction*)0;
Instruction* MemoryDependenceAnalysis::None = (Instruction*)~0;
  
// Register this pass...
RegisterPass<MemoryDependenceAnalysis> X("memdep",
                                           "Memory Dependence Analysis");

/// getAnalysisUsage - Does not modify anything.  It uses Alias Analysis.
///
void MemoryDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<TargetData>();
}

/// getDependency - Return the instruction on which a memory operation
/// depends.  The local paramter indicates if the query should only
/// evaluate dependencies within the same basic block.
Instruction* MemoryDependenceAnalysis::getDependency(Instruction* query,
                                                     bool local) {
  if (!local)
    assert(0 && "Non-local memory dependence is not yet supported.");
  
  // Start looking for dependencies with the queried inst
  BasicBlock::iterator QI = query;
  
  // Check for a cached result
  std::pair<Instruction*, bool> cachedResult = depGraphLocal[query];
  // If we have a _confirmed_ cached entry, return it
  if (cachedResult.second)
    return cachedResult.first;
  else if (cachedResult.first != NonLocal)
  // If we have an unconfirmed cached entry, we can start our search from there
    QI = cachedResult.first;
  
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  TargetData& TD = getAnalysis<TargetData>();
  
  // Get the pointer value for which dependence will be determined
  Value* dependee = 0;
  uint64_t dependeeSize = 0;
  if (StoreInst* S = dyn_cast<StoreInst>(QI)) {
    dependee = S->getPointerOperand();
    dependeeSize = TD.getTypeSize(S->getOperand(0)->getType());
  } else if (LoadInst* L = dyn_cast<LoadInst>(QI)) {
    dependee = L->getPointerOperand();
    dependeeSize = TD.getTypeSize(L->getType());
  } else if (FreeInst* F = dyn_cast<FreeInst>(QI)) {
    dependee = F->getPointerOperand();
    
    // FreeInsts erase the entire structure, not just a field
    dependeeSize = ~0UL;
  } else if (isa<AllocationInst>(query))
    return None;
  else
    // FIXME: Call/InvokeInsts need proper handling
    return None;
  
  
  BasicBlock::iterator blockBegin = query->getParent()->begin();
  
  while (QI != blockBegin) {
    --QI;
    
    // If this inst is a memory op, get the pointer it accessed
    Value* pointer = 0;
    uint64_t pointerSize = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(QI)) {
      pointer = S->getPointerOperand();
      pointerSize = TD.getTypeSize(S->getOperand(0)->getType());
    } else if (LoadInst* L = dyn_cast<LoadInst>(QI)) {
      pointer = L->getPointerOperand();
      pointerSize = TD.getTypeSize(L->getType());
    } else if (AllocationInst* AI = dyn_cast<AllocationInst>(QI)) {
      pointer = AI;
      if (ConstantInt* C = dyn_cast<ConstantInt>(AI->getArraySize()))
        pointerSize = C->getZExtValue();
      else
        pointerSize = ~0UL;
    } else if (FreeInst* F = dyn_cast<FreeInst>(QI)) {
      pointer = F->getPointerOperand();
      
      // FreeInsts erase the entire structure
      pointerSize = ~0UL;
    } else if (CallInst* C = dyn_cast<CallInst>(QI)) {
      // Call insts need special handling.  Check is they can modify our pointer
      if (AA.getModRefInfo(C, dependee, dependeeSize) != AliasAnalysis::NoModRef) {
        depGraphLocal.insert(std::make_pair(query, std::make_pair(C, true)));
        reverseDep.insert(std::make_pair(C, query));
        return C;
      } else {
        continue;
      }
    } else if (InvokeInst* I = dyn_cast<InvokeInst>(QI)) {
      // Invoke insts need special handling.  Check is they can modify our pointer
      if (AA.getModRefInfo(I, dependee, dependeeSize) != AliasAnalysis::NoModRef) {
        depGraphLocal.insert(std::make_pair(query, std::make_pair(I, true)));
        reverseDep.insert(std::make_pair(I, query));
        return I;
      } else {
        continue;
      }
    }
    
    // If we found a pointer, check if it could be the same as our pointer
    if (pointer) {
      AliasAnalysis::AliasResult R = AA.alias(pointer, pointerSize,
                                              dependee, dependeeSize);
      
      if (R != AliasAnalysis::NoAlias) {
        depGraphLocal.insert(std::make_pair(query, std::make_pair(QI, true)));
        reverseDep.insert(std::make_pair(QI, query));
        return QI;
      }
    }
  }
  
  // If we found nothing, return the non-local flag
  depGraphLocal.insert(std::make_pair(query,
                                      std::make_pair(NonLocal, true)));
  reverseDep.insert(std::make_pair(NonLocal, query));
  
  return NonLocal;
}

/// removeInstruction - Remove an instruction from the dependence analysis,
/// updating the dependence of instructions that previously depended on it.
void MemoryDependenceAnalysis::removeInstruction(Instruction* rem) {
  // Figure out the new dep for things that currently depend on rem
  Instruction* newDep = NonLocal;
  if (depGraphLocal[rem].first != NonLocal) {
    // If we have dep info for rem, set them to it
    BasicBlock::iterator RI = depGraphLocal[rem].first;
    RI++;
    newDep = RI;
  } else if (depGraphLocal[rem].first == NonLocal &&
             depGraphLocal[rem].second ) {
    // If we have a confirmed non-local flag, use it
    newDep = NonLocal;
  } else {
    // Otherwise, use the immediate successor of rem
    // NOTE: This is because, when getDependence is called, it will first check
    // the immediate predecessor of what is in the cache.
    BasicBlock::iterator RI = rem;
    RI++;
    newDep = RI;
  }

  std::multimap<Instruction*, Instruction*>::iterator I = reverseDep.find(rem);
  while (I->first == rem) {
    // Insert the new dependencies
    // Mark it as unconfirmed as long as it is not the non-local flag
    depGraphLocal[I->second] = std::make_pair(newDep, !newDep);
    reverseDep.erase(I);
    I = reverseDep.find(rem);
  }
}

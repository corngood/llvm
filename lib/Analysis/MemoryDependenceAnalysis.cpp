//===- MemoryDependenceAnalysis.cpp - Mem Deps Implementation  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis that determines, for a given memory
// operation, what preceding memory operations it depends on.  It builds on 
// alias analysis information, and tries to provide a lazy, caching interface to
// a common kind of alias information query.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "memdep"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

// Control the calculation of non-local dependencies by only examining the
// predecessors if the basic block has less than X amount (50 by default).
static cl::opt<int> 
PredLimit("nonlocaldep-threshold", cl::Hidden, cl::init(50),
          cl::desc("Control the calculation of non-local"
                   "dependencies (default = 50)"));           

STATISTIC(NumCacheNonlocal, "Number of cached non-local responses");
STATISTIC(NumUncacheNonlocal, "Number of uncached non-local responses");

char MemoryDependenceAnalysis::ID = 0;
  
// Register this pass...
static RegisterPass<MemoryDependenceAnalysis> X("memdep",
                                     "Memory Dependence Analysis", false, true);

/// verifyRemoved - Verify that the specified instruction does not occur
/// in our internal data structures.
void MemoryDependenceAnalysis::verifyRemoved(Instruction *D) const {
  for (LocalDepMapType::const_iterator I = LocalDeps.begin(),
       E = LocalDeps.end(); I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    assert(I->second.first.getPointer() != D &&
           "Inst occurs in data structures");
  }

  for (nonLocalDepMapType::const_iterator I = depGraphNonLocal.begin(),
       E = depGraphNonLocal.end(); I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    for (DenseMap<BasicBlock*, DepResultTy>::iterator II = I->second.begin(),
         EE = I->second.end(); II  != EE; ++II)
      assert(II->second.getPointer() != D && "Inst occurs in data structures");
  }

  for (reverseDepMapType::const_iterator I = reverseDep.begin(),
       E = reverseDep.end(); I != E; ++I)
    for (SmallPtrSet<Instruction*, 4>::const_iterator II = I->second.begin(),
         EE = I->second.end(); II != EE; ++II)
      assert(*II != D && "Inst occurs in data structures");

  for (reverseDepMapType::const_iterator I = reverseDepNonLocal.begin(),
       E = reverseDepNonLocal.end();
       I != E; ++I)
    for (SmallPtrSet<Instruction*, 4>::const_iterator II = I->second.begin(),
         EE = I->second.end(); II != EE; ++II)
      assert(*II != D && "Inst occurs in data structures");
}

/// getAnalysisUsage - Does not modify anything.  It uses Alias Analysis.
///
void MemoryDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<TargetData>();
}

/// getCallSiteDependency - Private helper for finding the local dependencies
/// of a call site.
MemoryDependenceAnalysis::DepResultTy
MemoryDependenceAnalysis::
getCallSiteDependency(CallSite C, Instruction *start, BasicBlock *block) {
  std::pair<DepResultTy, bool> &cachedResult = LocalDeps[C.getInstruction()];
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  TargetData& TD = getAnalysis<TargetData>();
  BasicBlock::iterator blockBegin = C.getInstruction()->getParent()->begin();
  BasicBlock::iterator QI = C.getInstruction();
  
  // If the starting point was specified, use it
  if (start) {
    QI = start;
    blockBegin = start->getParent()->begin();
  // If the starting point wasn't specified, but the block was, use it
  } else if (!start && block) {
    QI = block->end();
    blockBegin = block->begin();
  }
  
  // Walk backwards through the block, looking for dependencies
  while (QI != blockBegin) {
    --QI;
    
    // If this inst is a memory op, get the pointer it accessed
    Value* pointer = 0;
    uint64_t pointerSize = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(QI)) {
      pointer = S->getPointerOperand();
      pointerSize = TD.getTypeStoreSize(S->getOperand(0)->getType());
    } else if (AllocationInst* AI = dyn_cast<AllocationInst>(QI)) {
      pointer = AI;
      if (ConstantInt* C = dyn_cast<ConstantInt>(AI->getArraySize()))
        pointerSize = C->getZExtValue() *
                      TD.getABITypeSize(AI->getAllocatedType());
      else
        pointerSize = ~0UL;
    } else if (VAArgInst* V = dyn_cast<VAArgInst>(QI)) {
      pointer = V->getOperand(0);
      pointerSize = TD.getTypeStoreSize(V->getType());
    } else if (FreeInst* F = dyn_cast<FreeInst>(QI)) {
      pointer = F->getPointerOperand();
      
      // FreeInsts erase the entire structure
      pointerSize = ~0UL;
    } else if (CallSite::get(QI).getInstruction() != 0) {
      AliasAnalysis::ModRefBehavior result =
                   AA.getModRefBehavior(CallSite::get(QI));
      if (result != AliasAnalysis::DoesNotAccessMemory) {
        if (!start && !block) {
          cachedResult.first = DepResultTy(QI, Normal);
          cachedResult.second = true;
          reverseDep[DepResultTy(QI, Normal)].insert(C.getInstruction());
        }
        return DepResultTy(QI, Normal);
      } else {
        continue;
      }
    } else
      continue;
    
    if (AA.getModRefInfo(C, pointer, pointerSize) != AliasAnalysis::NoModRef) {
      if (!start && !block) {
        cachedResult.first = DepResultTy(QI, Normal);
        cachedResult.second = true;
        reverseDep[DepResultTy(QI, Normal)].insert(C.getInstruction());
      }
      return DepResultTy(QI, Normal);
    }
  }
  
  // No dependence found
  cachedResult.first = DepResultTy(0, NonLocal);
  cachedResult.second = true;
  reverseDep[DepResultTy(0, NonLocal)].insert(C.getInstruction());
  return DepResultTy(0, NonLocal);
}

/// nonLocalHelper - Private helper used to calculate non-local dependencies
/// by doing DFS on the predecessors of a block to find its dependencies.
void MemoryDependenceAnalysis::nonLocalHelper(Instruction* query,
                                              BasicBlock* block,
                                     DenseMap<BasicBlock*, DepResultTy> &resp) {
  // Set of blocks that we've already visited in our DFS
  SmallPtrSet<BasicBlock*, 4> visited;
  // If we're updating a dirtied cache entry, we don't need to reprocess
  // already computed entries.
  for (DenseMap<BasicBlock*, DepResultTy>::iterator I = resp.begin(), 
       E = resp.end(); I != E; ++I)
    if (I->second.getInt() != Dirty)
      visited.insert(I->first);
  
  // Current stack of the DFS
  SmallVector<BasicBlock*, 4> stack;
  for (pred_iterator PI = pred_begin(block), PE = pred_end(block);
       PI != PE; ++PI)
    stack.push_back(*PI);
  
  // Do a basic DFS
  while (!stack.empty()) {
    BasicBlock* BB = stack.back();
    
    // If we've already visited this block, no need to revist
    if (visited.count(BB)) {
      stack.pop_back();
      continue;
    }
    
    // If we find a new block with a local dependency for query,
    // then we insert the new dependency and backtrack.
    if (BB != block) {
      visited.insert(BB);
      
      DepResultTy localDep = getDependency(query, 0, BB);
      if (localDep.getInt() != NonLocal) {
        resp.insert(std::make_pair(BB, localDep));
        stack.pop_back();
        
        continue;
      }
    // If we re-encounter the starting block, we still need to search it
    // because there might be a dependency in the starting block AFTER
    // the position of the query.  This is necessary to get loops right.
    } else if (BB == block) {
      visited.insert(BB);
      
      DepResultTy localDep = getDependency(query, 0, BB);
      if (localDep != DepResultTy(query, Normal))
        resp.insert(std::make_pair(BB, localDep));
      
      stack.pop_back();
      
      continue;
    }
    
    // If we didn't find anything, recurse on the precessors of this block
    // Only do this for blocks with a small number of predecessors.
    bool predOnStack = false;
    bool inserted = false;
    if (std::distance(pred_begin(BB), pred_end(BB)) <= PredLimit) { 
      for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
           PI != PE; ++PI)
        if (!visited.count(*PI)) {
          stack.push_back(*PI);
          inserted = true;
        } else
          predOnStack = true;
    }
    
    // If we inserted a new predecessor, then we'll come back to this block
    if (inserted)
      continue;
    // If we didn't insert because we have no predecessors, then this
    // query has no dependency at all.
    else if (!inserted && !predOnStack) {
      resp.insert(std::make_pair(BB, DepResultTy(0, None)));
    // If we didn't insert because our predecessors are already on the stack,
    // then we might still have a dependency, but it will be discovered during
    // backtracking.
    } else if (!inserted && predOnStack){
      resp.insert(std::make_pair(BB, DepResultTy(0, NonLocal)));
    }
    
    stack.pop_back();
  }
}

/// getNonLocalDependency - Fills the passed-in map with the non-local 
/// dependencies of the queries.  The map will contain NonLocal for
/// blocks between the query and its dependencies.
void MemoryDependenceAnalysis::getNonLocalDependency(Instruction* query,
                                     DenseMap<BasicBlock*, DepResultTy> &resp) {
  if (depGraphNonLocal.count(query)) {
    DenseMap<BasicBlock*, DepResultTy> &cached = depGraphNonLocal[query];
    NumCacheNonlocal++;
    
    SmallVector<BasicBlock*, 4> dirtied;
    for (DenseMap<BasicBlock*, DepResultTy>::iterator I = cached.begin(),
         E = cached.end(); I != E; ++I)
      if (I->second.getInt() == Dirty)
        dirtied.push_back(I->first);
    
    for (SmallVector<BasicBlock*, 4>::iterator I = dirtied.begin(),
         E = dirtied.end(); I != E; ++I) {
      DepResultTy localDep = getDependency(query, 0, *I);
      if (localDep.getInt() != NonLocal)
        cached[*I] = localDep;
      else {
        cached.erase(*I);
        nonLocalHelper(query, *I, cached);
      }
    }
    
    resp = cached;
    
    // Update the reverse non-local dependency cache
    for (DenseMap<BasicBlock*, DepResultTy>::iterator I = resp.begin(),
         E = resp.end(); I != E; ++I)
      reverseDepNonLocal[I->second].insert(query);
    
    return;
  } else
    NumUncacheNonlocal++;
  
  // If not, go ahead and search for non-local deps.
  nonLocalHelper(query, query->getParent(), resp);
  
  // Update the non-local dependency cache
  for (DenseMap<BasicBlock*, DepResultTy>::iterator I = resp.begin(),
       E = resp.end(); I != E; ++I) {
    depGraphNonLocal[query].insert(*I);
    reverseDepNonLocal[I->second].insert(query);
  }
}

/// getDependency - Return the instruction on which a memory operation
/// depends.  The local parameter indicates if the query should only
/// evaluate dependencies within the same basic block.
MemoryDependenceAnalysis::DepResultTy
MemoryDependenceAnalysis::getDependency(Instruction *query,
                                        Instruction *start,
                                        BasicBlock *block) {
  // Start looking for dependencies with the queried inst
  BasicBlock::iterator QI = query;
  
  // Check for a cached result
  std::pair<DepResultTy, bool>& cachedResult = LocalDeps[query];
  // If we have a _confirmed_ cached entry, return it
  if (!block && !start) {
    if (cachedResult.second)
      return cachedResult.first;
    else if (cachedResult.first.getInt() == Normal &&
             cachedResult.first.getPointer())
      // If we have an unconfirmed cached entry, we can start our search from
      // it.
      QI = cachedResult.first.getPointer();
  }
  
  if (start)
    QI = start;
  else if (!start && block)
    QI = block->end();
  
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  TargetData& TD = getAnalysis<TargetData>();
  
  // Get the pointer value for which dependence will be determined
  Value* dependee = 0;
  uint64_t dependeeSize = 0;
  bool queryIsVolatile = false;
  if (StoreInst* S = dyn_cast<StoreInst>(query)) {
    dependee = S->getPointerOperand();
    dependeeSize = TD.getTypeStoreSize(S->getOperand(0)->getType());
    queryIsVolatile = S->isVolatile();
  } else if (LoadInst* L = dyn_cast<LoadInst>(query)) {
    dependee = L->getPointerOperand();
    dependeeSize = TD.getTypeStoreSize(L->getType());
    queryIsVolatile = L->isVolatile();
  } else if (VAArgInst* V = dyn_cast<VAArgInst>(query)) {
    dependee = V->getOperand(0);
    dependeeSize = TD.getTypeStoreSize(V->getType());
  } else if (FreeInst* F = dyn_cast<FreeInst>(query)) {
    dependee = F->getPointerOperand();
    
    // FreeInsts erase the entire structure, not just a field
    dependeeSize = ~0UL;
  } else if (CallSite::get(query).getInstruction() != 0)
    return getCallSiteDependency(CallSite::get(query), start, block);
  else if (isa<AllocationInst>(query))
    return DepResultTy(0, None);
  else
    return DepResultTy(0, None);
  
  BasicBlock::iterator blockBegin = block ? block->begin()
                                          : query->getParent()->begin();
  
  // Walk backwards through the basic block, looking for dependencies
  while (QI != blockBegin) {
    --QI;
    
    // If this inst is a memory op, get the pointer it accessed
    Value* pointer = 0;
    uint64_t pointerSize = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(QI)) {
      // All volatile loads/stores depend on each other
      if (queryIsVolatile && S->isVolatile()) {
        if (!start && !block) {
          cachedResult.first = DepResultTy(S, Normal);
          cachedResult.second = true;
          reverseDep[DepResultTy(S, Normal)].insert(query);
        }
        
        return DepResultTy(S, Normal);
      }
      
      pointer = S->getPointerOperand();
      pointerSize = TD.getTypeStoreSize(S->getOperand(0)->getType());
    } else if (LoadInst* L = dyn_cast<LoadInst>(QI)) {
      // All volatile loads/stores depend on each other
      if (queryIsVolatile && L->isVolatile()) {
        if (!start && !block) {
          cachedResult.first = DepResultTy(L, Normal);
          cachedResult.second = true;
          reverseDep[DepResultTy(L, Normal)].insert(query);
        }
        
        return DepResultTy(L, Normal);
      }
      
      pointer = L->getPointerOperand();
      pointerSize = TD.getTypeStoreSize(L->getType());
    } else if (AllocationInst* AI = dyn_cast<AllocationInst>(QI)) {
      pointer = AI;
      if (ConstantInt* C = dyn_cast<ConstantInt>(AI->getArraySize()))
        pointerSize = C->getZExtValue() * 
                      TD.getABITypeSize(AI->getAllocatedType());
      else
        pointerSize = ~0UL;
    } else if (VAArgInst* V = dyn_cast<VAArgInst>(QI)) {
      pointer = V->getOperand(0);
      pointerSize = TD.getTypeStoreSize(V->getType());
    } else if (FreeInst* F = dyn_cast<FreeInst>(QI)) {
      pointer = F->getPointerOperand();
      
      // FreeInsts erase the entire structure
      pointerSize = ~0UL;
    } else if (CallSite::get(QI).getInstruction() != 0) {
      // Call insts need special handling. Check if they can modify our pointer
      AliasAnalysis::ModRefResult MR = AA.getModRefInfo(CallSite::get(QI),
                                                        dependee, dependeeSize);
      
      if (MR != AliasAnalysis::NoModRef) {
        // Loads don't depend on read-only calls
        if (isa<LoadInst>(query) && MR == AliasAnalysis::Ref)
          continue;
        
        if (!start && !block) {
          cachedResult.first = DepResultTy(QI, Normal);
          cachedResult.second = true;
          reverseDep[DepResultTy(QI, Normal)].insert(query);
        }
        return DepResultTy(QI, Normal);
      } else {
        continue;
      }
    }
    
    // If we found a pointer, check if it could be the same as our pointer
    if (pointer) {
      AliasAnalysis::AliasResult R = AA.alias(pointer, pointerSize,
                                              dependee, dependeeSize);
      
      if (R != AliasAnalysis::NoAlias) {
        // May-alias loads don't depend on each other
        if (isa<LoadInst>(query) && isa<LoadInst>(QI) &&
            R == AliasAnalysis::MayAlias)
          continue;
        
        if (!start && !block) {
          cachedResult.first = DepResultTy(QI, Normal);
          cachedResult.second = true;
          reverseDep[DepResultTy(QI, Normal)].insert(query);
        }
        
        return DepResultTy(QI, Normal);
      }
    }
  }
  
  // If we found nothing, return the non-local flag
  if (!start && !block) {
    cachedResult.first = DepResultTy(0, NonLocal);
    cachedResult.second = true;
    reverseDep[DepResultTy(0, NonLocal)].insert(query);
  }
  
  return DepResultTy(0, NonLocal);
}

/// dropInstruction - Remove an instruction from the analysis, making 
/// absolutely conservative assumptions when updating the cache.  This is
/// useful, for example when an instruction is changed rather than removed.
void MemoryDependenceAnalysis::dropInstruction(Instruction* drop) {
  LocalDepMapType::iterator depGraphEntry = LocalDeps.find(drop);
  if (depGraphEntry != LocalDeps.end())
    reverseDep[depGraphEntry->second.first].erase(drop);
  
  // Drop dependency information for things that depended on this instr
  SmallPtrSet<Instruction*, 4>& set = reverseDep[DepResultTy(drop, Normal)];
  for (SmallPtrSet<Instruction*, 4>::iterator I = set.begin(), E = set.end();
       I != E; ++I)
    LocalDeps.erase(*I);
  
  LocalDeps.erase(drop);
  reverseDep.erase(DepResultTy(drop, Normal));
  
  for (DenseMap<BasicBlock*, DepResultTy>::iterator DI =
         depGraphNonLocal[drop].begin(), DE = depGraphNonLocal[drop].end();
       DI != DE; ++DI)
    if (DI->second.getInt() != None)
      reverseDepNonLocal[DI->second].erase(drop);
  
  if (reverseDepNonLocal.count(DepResultTy(drop, Normal))) {
    SmallPtrSet<Instruction*, 4>& set =
      reverseDepNonLocal[DepResultTy(drop, Normal)];
    for (SmallPtrSet<Instruction*, 4>::iterator I = set.begin(), E = set.end();
         I != E; ++I)
      for (DenseMap<BasicBlock*, DepResultTy>::iterator DI =
           depGraphNonLocal[*I].begin(), DE = depGraphNonLocal[*I].end();
           DI != DE; ++DI)
        if (DI->second == DepResultTy(drop, Normal))
          DI->second = DepResultTy(0, Dirty);
  }
  
  reverseDepNonLocal.erase(DepResultTy(drop, Normal));
  depGraphNonLocal.erase(drop);
}

/// removeInstruction - Remove an instruction from the dependence analysis,
/// updating the dependence of instructions that previously depended on it.
/// This method attempts to keep the cache coherent using the reverse map.
void MemoryDependenceAnalysis::removeInstruction(Instruction *RemInst) {
  // Walk through the Non-local dependencies, removing this one as the value
  // for any cached queries.
  for (DenseMap<BasicBlock*, DepResultTy>::iterator DI =
       depGraphNonLocal[RemInst].begin(), DE = depGraphNonLocal[RemInst].end();
       DI != DE; ++DI)
    if (DI->second.getInt() != None)
      reverseDepNonLocal[DI->second].erase(RemInst);

  // Shortly after this, we will look for things that depend on RemInst.  In
  // order to update these, we'll need a new dependency to base them on.  We
  // could completely delete any entries that depend on this, but it is better
  // to make a more accurate approximation where possible.  Compute that better
  // approximation if we can.
  DepResultTy NewDependency;
  bool NewDependencyConfirmed = false;
  
  // If we have a cached local dependence query for this instruction, remove it.
  //
  LocalDepMapType::iterator LocalDepEntry = LocalDeps.find(RemInst);
  if (LocalDepEntry != LocalDeps.end()) {
    DepResultTy LocalDep = LocalDepEntry->second.first;
    bool IsConfirmed = LocalDepEntry->second.second;
    
    // Remove this local dependency info.
    LocalDeps.erase(LocalDepEntry);
    
    // Remove us from DepInst's reverse set now that the local dep info is gone.
    reverseDep[LocalDep].erase(RemInst);

    // If we have unconfirmed info, don't trust it.
    if (IsConfirmed) {
      // If we have a confirmed non-local flag, use it.
      if (LocalDep.getInt() == NonLocal || LocalDep.getInt() == None) {
        // The only time this dependency is confirmed is if it is non-local.
        NewDependency = LocalDep;
        NewDependencyConfirmed = true;
      } else {
        // If we have dep info for RemInst, set them to it.
        Instruction *NDI = next(BasicBlock::iterator(LocalDep.getPointer()));
        if (NDI != RemInst) // Don't use RemInst for the new dependency!
          NewDependency = DepResultTy(NDI, Normal);
      }
    }
  }
  
  // If we don't already have a local dependency answer for this instruction,
  // use the immediate successor of RemInst.  We use the successor because
  // getDependence starts by checking the immediate predecessor of what is in
  // the cache.
  if (NewDependency == DepResultTy(0, Normal))
    NewDependency = DepResultTy(next(BasicBlock::iterator(RemInst)), Normal);
  
  // Loop over all of the things that depend on the instruction we're removing.
  // 
  reverseDepMapType::iterator ReverseDepIt =
    reverseDep.find(DepResultTy(RemInst, Normal));
  if (ReverseDepIt != reverseDep.end()) {
    SmallPtrSet<Instruction*, 4> &ReverseDeps = ReverseDepIt->second;
    for (SmallPtrSet<Instruction*, 4>::iterator I = ReverseDeps.begin(),
         E = ReverseDeps.end(); I != E; ++I) {
      Instruction *InstDependingOnRemInst = *I;
      
      // If we thought the instruction depended on itself (possible for
      // unconfirmed dependencies) ignore the update.
      if (InstDependingOnRemInst == RemInst) continue;
      
      // Insert the new dependencies.
      LocalDeps[InstDependingOnRemInst] =
        std::make_pair(NewDependency, NewDependencyConfirmed);
      
      // If our NewDependency is an instruction, make sure to remember that new
      // things depend on it.
      // FIXME: Just insert all deps!
      if (NewDependency.getInt() != NonLocal && NewDependency.getInt() != None)
        reverseDep[NewDependency].insert(InstDependingOnRemInst);
    }
    reverseDep.erase(DepResultTy(RemInst, Normal));
  }
  
  ReverseDepIt = reverseDepNonLocal.find(DepResultTy(RemInst, Normal));
  if (ReverseDepIt != reverseDepNonLocal.end()) {
    SmallPtrSet<Instruction*, 4>& set = ReverseDepIt->second;
    for (SmallPtrSet<Instruction*, 4>::iterator I = set.begin(), E = set.end();
         I != E; ++I)
      for (DenseMap<BasicBlock*, DepResultTy>::iterator DI =
           depGraphNonLocal[*I].begin(), DE = depGraphNonLocal[*I].end();
           DI != DE; ++DI)
        if (DI->second == DepResultTy(RemInst, Normal))
          DI->second = DepResultTy(0, Dirty);
    reverseDepNonLocal.erase(ReverseDepIt);
  }
  
  depGraphNonLocal.erase(RemInst);

  getAnalysis<AliasAnalysis>().deleteValue(RemInst);
  
  DEBUG(verifyRemoved(RemInst));
}

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

STATISTIC(NumCacheNonLocal, "Number of fully cached non-local responses");
STATISTIC(NumCacheDirtyNonLocal, "Number of dirty cached non-local responses");
STATISTIC(NumUncacheNonLocal, "Number of uncached non-local responses");
char MemoryDependenceAnalysis::ID = 0;
  
// Register this pass...
static RegisterPass<MemoryDependenceAnalysis> X("memdep",
                                     "Memory Dependence Analysis", false, true);

/// getAnalysisUsage - Does not modify anything.  It uses Alias Analysis.
///
void MemoryDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<TargetData>();
}

bool MemoryDependenceAnalysis::runOnFunction(Function &) {
  AA = &getAnalysis<AliasAnalysis>();
  TD = &getAnalysis<TargetData>();
  return false;
}


/// getCallSiteDependency - Private helper for finding the local dependencies
/// of a call site.
MemDepResult MemoryDependenceAnalysis::
getCallSiteDependency(CallSite C, BasicBlock::iterator ScanIt, BasicBlock *BB) {
  
  // Walk backwards through the block, looking for dependencies
  while (ScanIt != BB->begin()) {
    Instruction *Inst = --ScanIt;
    
    // If this inst is a memory op, get the pointer it accessed
    Value *Pointer = 0;
    uint64_t PointerSize = 0;
    if (StoreInst *S = dyn_cast<StoreInst>(Inst)) {
      Pointer = S->getPointerOperand();
      PointerSize = TD->getTypeStoreSize(S->getOperand(0)->getType());
    } else if (VAArgInst *V = dyn_cast<VAArgInst>(Inst)) {
      Pointer = V->getOperand(0);
      PointerSize = TD->getTypeStoreSize(V->getType());
    } else if (FreeInst *F = dyn_cast<FreeInst>(Inst)) {
      Pointer = F->getPointerOperand();
      
      // FreeInsts erase the entire structure
      PointerSize = ~0UL;
    } else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
      if (AA->getModRefBehavior(CallSite::get(Inst)) ==
            AliasAnalysis::DoesNotAccessMemory)
        continue;
      return MemDepResult::get(Inst);
    } else {
      // Non-memory instruction.
      continue;
    }
    
    if (AA->getModRefInfo(C, Pointer, PointerSize) != AliasAnalysis::NoModRef)
      return MemDepResult::get(Inst);
  }
  
  // No dependence found.
  return MemDepResult::getNonLocal();
}

/// getDependencyFrom - Return the instruction on which a memory operation
/// depends.
MemDepResult MemoryDependenceAnalysis::
getDependencyFrom(Instruction *QueryInst, BasicBlock::iterator ScanIt, 
                  BasicBlock *BB) {
  // Get the pointer value for which dependence will be determined
  Value *MemPtr = 0;
  uint64_t MemSize = 0;
  bool MemVolatile = false;
  
  if (StoreInst* S = dyn_cast<StoreInst>(QueryInst)) {
    MemPtr = S->getPointerOperand();
    MemSize = TD->getTypeStoreSize(S->getOperand(0)->getType());
    MemVolatile = S->isVolatile();
  } else if (LoadInst* L = dyn_cast<LoadInst>(QueryInst)) {
    MemPtr = L->getPointerOperand();
    MemSize = TD->getTypeStoreSize(L->getType());
    MemVolatile = L->isVolatile();
  } else if (VAArgInst* V = dyn_cast<VAArgInst>(QueryInst)) {
    MemPtr = V->getOperand(0);
    MemSize = TD->getTypeStoreSize(V->getType());
  } else if (FreeInst* F = dyn_cast<FreeInst>(QueryInst)) {
    MemPtr = F->getPointerOperand();
    // FreeInsts erase the entire structure, not just a field.
    MemSize = ~0UL;
  } else {
    assert((isa<CallInst>(QueryInst) || isa<InvokeInst>(QueryInst)) &&
            "Can only get dependency info for memory instructions!");
    return getCallSiteDependency(CallSite::get(QueryInst), ScanIt, BB);
  }
  
  // Walk backwards through the basic block, looking for dependencies
  while (ScanIt != BB->begin()) {
    Instruction *Inst = --ScanIt;

    // If the access is volatile and this is a volatile load/store, return a
    // dependence.
    if (MemVolatile &&
        ((isa<LoadInst>(Inst) && cast<LoadInst>(Inst)->isVolatile()) ||
         (isa<StoreInst>(Inst) && cast<StoreInst>(Inst)->isVolatile())))
      return MemDepResult::get(Inst);

    // Values depend on loads if the pointers are must aliased.  This means that
    // a load depends on another must aliased load from the same value.
    if (LoadInst *L = dyn_cast<LoadInst>(Inst)) {
      Value *Pointer = L->getPointerOperand();
      uint64_t PointerSize = TD->getTypeStoreSize(L->getType());
      
      // If we found a pointer, check if it could be the same as our pointer
      AliasAnalysis::AliasResult R =
        AA->alias(Pointer, PointerSize, MemPtr, MemSize);
      
      if (R == AliasAnalysis::NoAlias)
        continue;
      
      // May-alias loads don't depend on each other without a dependence.
      if (isa<LoadInst>(QueryInst) && R == AliasAnalysis::MayAlias)
        continue;
      return MemDepResult::get(Inst);
    }

    // If this is an allocation, and if we know that the accessed pointer is to
    // the allocation, return None.  This means that there is no dependence and
    // the access can be optimized based on that.  For example, a load could
    // turn into undef.
    if (AllocationInst *AI = dyn_cast<AllocationInst>(Inst)) {
      Value *AccessPtr = MemPtr->getUnderlyingObject();
      
      if (AccessPtr == AI ||
          AA->alias(AI, 1, AccessPtr, 1) == AliasAnalysis::MustAlias)
        return MemDepResult::getNone();
      continue;
    }
    
    // See if this instruction mod/ref's the pointer.
    AliasAnalysis::ModRefResult MRR = AA->getModRefInfo(Inst, MemPtr, MemSize);

    if (MRR == AliasAnalysis::NoModRef)
      continue;
    
    // Loads don't depend on read-only instructions.
    if (isa<LoadInst>(QueryInst) && MRR == AliasAnalysis::Ref)
      continue;
    
    // Otherwise, there is a dependence.
    return MemDepResult::get(Inst);
  }
  
  // If we found nothing, return the non-local flag.
  return MemDepResult::getNonLocal();
}

/// getDependency - Return the instruction on which a memory operation
/// depends.
MemDepResult MemoryDependenceAnalysis::getDependency(Instruction *QueryInst) {
  Instruction *ScanPos = QueryInst;
  
  // Check for a cached result
  MemDepResult &LocalCache = LocalDeps[QueryInst];
  
  // If the cached entry is non-dirty, just return it.  Note that this depends
  // on MemDepResult's default constructing to 'dirty'.
  if (!LocalCache.isDirty())
    return LocalCache;
    
  // Otherwise, if we have a dirty entry, we know we can start the scan at that
  // instruction, which may save us some work.
  if (Instruction *Inst = LocalCache.getInst()) {
    ScanPos = Inst;
   
    SmallPtrSet<Instruction*, 4> &InstMap = ReverseLocalDeps[Inst];
    InstMap.erase(QueryInst);
    if (InstMap.empty())
      ReverseLocalDeps.erase(Inst);
  }
  
  // Do the scan.
  LocalCache = getDependencyFrom(QueryInst, ScanPos, QueryInst->getParent());
  
  // Remember the result!
  if (Instruction *I = LocalCache.getInst())
    ReverseLocalDeps[I].insert(QueryInst);
  
  return LocalCache;
}

/// getNonLocalDependency - Perform a full dependency query for the
/// specified instruction, returning the set of blocks that the value is
/// potentially live across.  The returned set of results will include a
/// "NonLocal" result for all blocks where the value is live across.
///
/// This method assumes the instruction returns a "nonlocal" dependency
/// within its own block.
///
const MemoryDependenceAnalysis::NonLocalDepInfo &
MemoryDependenceAnalysis::getNonLocalDependency(Instruction *QueryInst) {
  assert(getDependency(QueryInst).isNonLocal() &&
     "getNonLocalDependency should only be used on insts with non-local deps!");
  PerInstNLInfo &CacheP = NonLocalDeps[QueryInst];
  
  NonLocalDepInfo &Cache = CacheP.first;

  /// DirtyBlocks - This is the set of blocks that need to be recomputed.  In
  /// the cached case, this can happen due to instructions being deleted etc. In
  /// the uncached case, this starts out as the set of predecessors we care
  /// about.
  SmallVector<BasicBlock*, 32> DirtyBlocks;
  
  if (!Cache.empty()) {
    // Okay, we have a cache entry.  If we know it is not dirty, just return it
    // with no computation.
    if (!CacheP.second) {
      NumCacheNonLocal++;
      return Cache;
    }
    
    // If we already have a partially computed set of results, scan them to
    // determine what is dirty, seeding our initial DirtyBlocks worklist.
    for (NonLocalDepInfo::iterator I = Cache.begin(), E = Cache.end();
       I != E; ++I)
      if (I->second.isDirty())
        DirtyBlocks.push_back(I->first);
    
    // Sort the cache so that we can do fast binary search lookups below.
    std::sort(Cache.begin(), Cache.end());
    
    ++NumCacheDirtyNonLocal;
    //cerr << "CACHED CASE: " << DirtyBlocks.size() << " dirty: "
    //     << Cache.size() << " cached: " << *QueryInst;
  } else {
    // Seed DirtyBlocks with each of the preds of QueryInst's block.
    BasicBlock *QueryBB = QueryInst->getParent();
    DirtyBlocks.append(pred_begin(QueryBB), pred_end(QueryBB));
    NumUncacheNonLocal++;
  }
  
  // Visited checked first, vector in sorted order.
  SmallPtrSet<BasicBlock*, 64> Visited;
  
  unsigned NumSortedEntries = Cache.size();
  
  // Iterate while we still have blocks to update.
  while (!DirtyBlocks.empty()) {
    BasicBlock *DirtyBB = DirtyBlocks.back();
    DirtyBlocks.pop_back();
    
    // Already processed this block?
    if (!Visited.insert(DirtyBB))
      continue;
    
    // Do a binary search to see if we already have an entry for this block in
    // the cache set.  If so, find it.
    NonLocalDepInfo::iterator Entry = 
      std::upper_bound(Cache.begin(), Cache.begin()+NumSortedEntries,
                       std::make_pair(DirtyBB, MemDepResult()));
    if (Entry != Cache.begin() && (&*Entry)[-1].first == DirtyBB)
      --Entry;
    
    MemDepResult *ExistingResult = 0;
    if (Entry != Cache.begin()+NumSortedEntries && 
        Entry->first == DirtyBB) {
      // If we already have an entry, and if it isn't already dirty, the block
      // is done.
      if (!Entry->second.isDirty())
        continue;
      
      // Otherwise, remember this slot so we can update the value.
      ExistingResult = &Entry->second;
    }
    
    // If the dirty entry has a pointer, start scanning from it so we don't have
    // to rescan the entire block.
    BasicBlock::iterator ScanPos = DirtyBB->end();
    if (ExistingResult) {
      if (Instruction *Inst = ExistingResult->getInst()) {
        ScanPos = Inst;
      
        // We're removing QueryInst's use of Inst.
        SmallPtrSet<Instruction*, 4> &InstMap = ReverseNonLocalDeps[Inst];
        InstMap.erase(QueryInst);
        if (InstMap.empty()) ReverseNonLocalDeps.erase(Inst);
      }
    }
    
    // Find out if this block has a local dependency for QueryInst.
    MemDepResult Dep = getDependencyFrom(QueryInst, ScanPos, DirtyBB);
    
    // If we had a dirty entry for the block, update it.  Otherwise, just add
    // a new entry.
    if (ExistingResult)
      *ExistingResult = Dep;
    else
      Cache.push_back(std::make_pair(DirtyBB, Dep));
    
    // If the block has a dependency (i.e. it isn't completely transparent to
    // the value), remember the association!
    if (!Dep.isNonLocal()) {
      // Keep the ReverseNonLocalDeps map up to date so we can efficiently
      // update this when we remove instructions.
      if (Instruction *Inst = Dep.getInst())
        ReverseNonLocalDeps[Inst].insert(QueryInst);
    } else {
    
      // If the block *is* completely transparent to the load, we need to check
      // the predecessors of this block.  Add them to our worklist.
      DirtyBlocks.append(pred_begin(DirtyBB), pred_end(DirtyBB));
    }
  }
  
  return Cache;
}

/// removeInstruction - Remove an instruction from the dependence analysis,
/// updating the dependence of instructions that previously depended on it.
/// This method attempts to keep the cache coherent using the reverse map.
void MemoryDependenceAnalysis::removeInstruction(Instruction *RemInst) {
  // Walk through the Non-local dependencies, removing this one as the value
  // for any cached queries.
  NonLocalDepMapType::iterator NLDI = NonLocalDeps.find(RemInst);
  if (NLDI != NonLocalDeps.end()) {
    NonLocalDepInfo &BlockMap = NLDI->second.first;
    for (NonLocalDepInfo::iterator DI = BlockMap.begin(), DE = BlockMap.end();
         DI != DE; ++DI)
      if (Instruction *Inst = DI->second.getInst())
        ReverseNonLocalDeps[Inst].erase(RemInst);
    NonLocalDeps.erase(NLDI);
  }

  // If we have a cached local dependence query for this instruction, remove it.
  //
  LocalDepMapType::iterator LocalDepEntry = LocalDeps.find(RemInst);
  if (LocalDepEntry != LocalDeps.end()) {
    // Remove us from DepInst's reverse set now that the local dep info is gone.
    if (Instruction *Inst = LocalDepEntry->second.getInst()) {
      SmallPtrSet<Instruction*, 4> &RLD = ReverseLocalDeps[Inst];
      RLD.erase(RemInst);
      if (RLD.empty())
        ReverseLocalDeps.erase(Inst);
    }

    // Remove this local dependency info.
    LocalDeps.erase(LocalDepEntry);
  }    
  
  // Loop over all of the things that depend on the instruction we're removing.
  // 
  SmallVector<std::pair<Instruction*, Instruction*>, 8> ReverseDepsToAdd;
  
  ReverseDepMapType::iterator ReverseDepIt = ReverseLocalDeps.find(RemInst);
  if (ReverseDepIt != ReverseLocalDeps.end()) {
    SmallPtrSet<Instruction*, 4> &ReverseDeps = ReverseDepIt->second;
    // RemInst can't be the terminator if it has stuff depending on it.
    assert(!ReverseDeps.empty() && !isa<TerminatorInst>(RemInst) &&
           "Nothing can locally depend on a terminator");
    
    // Anything that was locally dependent on RemInst is now going to be
    // dependent on the instruction after RemInst.  It will have the dirty flag
    // set so it will rescan.  This saves having to scan the entire block to get
    // to this point.
    Instruction *NewDepInst = next(BasicBlock::iterator(RemInst));
                        
    for (SmallPtrSet<Instruction*, 4>::iterator I = ReverseDeps.begin(),
         E = ReverseDeps.end(); I != E; ++I) {
      Instruction *InstDependingOnRemInst = *I;
      assert(InstDependingOnRemInst != RemInst &&
             "Already removed our local dep info");
                        
      LocalDeps[InstDependingOnRemInst] = MemDepResult::getDirty(NewDepInst);
      
      // Make sure to remember that new things depend on NewDepInst.
      ReverseDepsToAdd.push_back(std::make_pair(NewDepInst, 
                                                InstDependingOnRemInst));
    }
    
    ReverseLocalDeps.erase(ReverseDepIt);

    // Add new reverse deps after scanning the set, to avoid invalidating the
    // 'ReverseDeps' reference.
    while (!ReverseDepsToAdd.empty()) {
      ReverseLocalDeps[ReverseDepsToAdd.back().first]
        .insert(ReverseDepsToAdd.back().second);
      ReverseDepsToAdd.pop_back();
    }
  }
  
  ReverseDepIt = ReverseNonLocalDeps.find(RemInst);
  if (ReverseDepIt != ReverseNonLocalDeps.end()) {
    SmallPtrSet<Instruction*, 4>& set = ReverseDepIt->second;
    for (SmallPtrSet<Instruction*, 4>::iterator I = set.begin(), E = set.end();
         I != E; ++I) {
      assert(*I != RemInst && "Already removed NonLocalDep info for RemInst");
      
      PerInstNLInfo &INLD = NonLocalDeps[*I];
      // The information is now dirty!
      INLD.second = true;
      
      for (NonLocalDepInfo::iterator DI = INLD.first.begin(), 
           DE = INLD.first.end(); DI != DE; ++DI) {
        if (DI->second.getInst() != RemInst) continue;
        
        // Convert to a dirty entry for the subsequent instruction.
        Instruction *NextI = 0;
        if (!RemInst->isTerminator()) {
          NextI = next(BasicBlock::iterator(RemInst));
          ReverseDepsToAdd.push_back(std::make_pair(NextI, *I));
        }
        DI->second = MemDepResult::getDirty(NextI);
      }
    }

    ReverseNonLocalDeps.erase(ReverseDepIt);

    // Add new reverse deps after scanning the set, to avoid invalidating 'Set'
    while (!ReverseDepsToAdd.empty()) {
      ReverseNonLocalDeps[ReverseDepsToAdd.back().first]
        .insert(ReverseDepsToAdd.back().second);
      ReverseDepsToAdd.pop_back();
    }
  }
  
  assert(!NonLocalDeps.count(RemInst) && "RemInst got reinserted?");
  AA->deleteValue(RemInst);
  DEBUG(verifyRemoved(RemInst));
}

/// verifyRemoved - Verify that the specified instruction does not occur
/// in our internal data structures.
void MemoryDependenceAnalysis::verifyRemoved(Instruction *D) const {
  for (LocalDepMapType::const_iterator I = LocalDeps.begin(),
       E = LocalDeps.end(); I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    assert(I->second.getInst() != D &&
           "Inst occurs in data structures");
  }
  
  for (NonLocalDepMapType::const_iterator I = NonLocalDeps.begin(),
       E = NonLocalDeps.end(); I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    const PerInstNLInfo &INLD = I->second;
    for (NonLocalDepInfo::const_iterator II = INLD.first.begin(),
         EE = INLD.first.end(); II  != EE; ++II)
      assert(II->second.getInst() != D && "Inst occurs in data structures");
  }
  
  for (ReverseDepMapType::const_iterator I = ReverseLocalDeps.begin(),
       E = ReverseLocalDeps.end(); I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    for (SmallPtrSet<Instruction*, 4>::const_iterator II = I->second.begin(),
         EE = I->second.end(); II != EE; ++II)
      assert(*II != D && "Inst occurs in data structures");
  }
  
  for (ReverseDepMapType::const_iterator I = ReverseNonLocalDeps.begin(),
       E = ReverseNonLocalDeps.end();
       I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    for (SmallPtrSet<Instruction*, 4>::const_iterator II = I->second.begin(),
         EE = I->second.end(); II != EE; ++II)
      assert(*II != D && "Inst occurs in data structures");
  }
}

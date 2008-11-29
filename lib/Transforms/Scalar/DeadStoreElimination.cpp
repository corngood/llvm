//===- DeadStoreElimination.cpp - Fast Dead Store Elimination -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a trivial dead store elimination that only considers
// basic-block local redundant stores.
//
// FIXME: This should eventually be extended to be a post-dominator tree
// traversal.  Doing so would be pretty trivial.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dse"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumFastStores, "Number of stores deleted");
STATISTIC(NumFastOther , "Number of other instrs removed");

namespace {
  struct VISIBILITY_HIDDEN DSE : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    DSE() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F) {
      bool Changed = false;
      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
        Changed |= runOnBasicBlock(*I);
      return Changed;
    }
    
    bool runOnBasicBlock(BasicBlock &BB);
    bool handleFreeWithNonTrivialDependency(FreeInst *F, MemDepResult Dep);
    bool handleEndBlock(BasicBlock &BB);
    bool RemoveUndeadPointers(Value* pointer, uint64_t killPointerSize,
                              BasicBlock::iterator& BBI,
                              SmallPtrSet<Value*, 64>& deadPointers);
    void DeleteDeadInstruction(Instruction *I,
                               SmallPtrSet<Value*, 64> *deadPointers = 0);
    

    // getAnalysisUsage - We require post dominance frontiers (aka Control
    // Dependence Graph)
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<TargetData>();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<MemoryDependenceAnalysis>();
    }
  };
}

char DSE::ID = 0;
static RegisterPass<DSE> X("dse", "Dead Store Elimination");

FunctionPass *llvm::createDeadStoreEliminationPass() { return new DSE(); }

bool DSE::runOnBasicBlock(BasicBlock &BB) {
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  TargetData &TD = getAnalysis<TargetData>();  

  // Record the last-seen store to this pointer
  DenseMap<Value*, StoreInst*> lastStore;
  
  bool MadeChange = false;
  
  // Do a top-down walk on the BB
  for (BasicBlock::iterator BBI = BB.begin(), BBE = BB.end(); BBI != BBE; ) {
    Instruction *Inst = BBI++;
    
    // If we find a store or a free...
    if (!isa<StoreInst>(Inst) && !isa<FreeInst>(Inst))
      continue;

    Value* pointer = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(Inst)) {
      if (S->isVolatile())
        continue;
      pointer = S->getPointerOperand();
    } else {
      pointer = cast<FreeInst>(Inst)->getPointerOperand();
    }

    pointer = pointer->stripPointerCasts();
    StoreInst *&last = lastStore[pointer];
 
    // ... to a pointer that has been stored to before...
    if (last) {
      MemDepResult dep = MD.getDependency(Inst);
      bool deletedStore = false;
    
      // ... and no other memory dependencies are between them....
      while (StoreInst *DepStore = dyn_cast_or_null<StoreInst>(dep.getInst())) {
        if (DepStore != last ||
            TD.getTypeStoreSize(last->getOperand(0)->getType()) >
            TD.getTypeStoreSize(Inst->getOperand(0)->getType())) {
          dep = MD.getDependency(Inst, DepStore);
          continue;
        }
        
        // Delete the store and now-dead instructions that feed it.
        DeleteDeadInstruction(last);
        NumFastStores++;
        deletedStore = true;
        MadeChange = true;
        break;
      }
      
      // If we deleted a store, reinvestigate this instruction.
      if (deletedStore) {
        if (!isa<TerminatorInst>(BB.begin()))
          --BBI;
        continue;
      }
    }
    
    // Handle frees whose dependencies are non-trivial.
    if (FreeInst* F = dyn_cast<FreeInst>(Inst)) {
      MadeChange |= handleFreeWithNonTrivialDependency(F, MD.getDependency(F));
      
      // No known stores after the free.
      last = 0;
    } else {
      StoreInst* S = cast<StoreInst>(Inst);
      
      // If we're storing the same value back to a pointer that we just
      // loaded from, then the store can be removed;
      if (LoadInst* L = dyn_cast<LoadInst>(S->getOperand(0))) {
        // FIXME: Don't do dep query if Parents don't match and other stuff!
        MemDepResult dep = MD.getDependency(S);
        DominatorTree& DT = getAnalysis<DominatorTree>();
        
        if (!S->isVolatile() && S->getParent() == L->getParent() &&
            S->getPointerOperand() == L->getPointerOperand() &&
            (!dep.isNormal() || DT.dominates(dep.getInst(), L))) {
          
          DeleteDeadInstruction(S);
          if (!isa<TerminatorInst>(BB.begin()))
            --BBI;
          NumFastStores++;
          MadeChange = true;
        } else
          // Update our most-recent-store map.
          last = S;
      } else
        // Update our most-recent-store map.
        last = S;
    }
  }
  
  // If this block ends in a return, unwind, or unreachable, all allocas are
  // dead at its end, which means stores to them are also dead.
  if (BB.getTerminator()->getNumSuccessors() == 0)
    MadeChange |= handleEndBlock(BB);
  
  return MadeChange;
}

/// handleFreeWithNonTrivialDependency - Handle frees of entire structures whose
/// dependency is a store to a field of that structure.
bool DSE::handleFreeWithNonTrivialDependency(FreeInst* F, MemDepResult dep) {
  TargetData &TD = getAnalysis<TargetData>();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  
  StoreInst* dependency = dyn_cast_or_null<StoreInst>(dep.getInst());
  if (!dependency)
    return false;
  else if (dependency->isVolatile())
    return false;
  
  Value* depPointer = dependency->getPointerOperand();
  const Type* depType = dependency->getOperand(0)->getType();
  unsigned depPointerSize = TD.getTypeStoreSize(depType);

  // Check for aliasing
  AliasAnalysis::AliasResult A = AA.alias(F->getPointerOperand(), ~0U,
                                          depPointer, depPointerSize);

  if (A != AliasAnalysis::MustAlias)
    return false;
  
  // DCE instructions only used to calculate that store
  DeleteDeadInstruction(dependency);
  NumFastStores++;
  return true;
}

/// handleEndBlock - Remove dead stores to stack-allocated locations in the
/// function end block.  Ex:
/// %A = alloca i32
/// ...
/// store i32 1, i32* %A
/// ret void
bool DSE::handleEndBlock(BasicBlock &BB) {
  TargetData &TD = getAnalysis<TargetData>();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  
  bool MadeChange = false;
  
  // Pointers alloca'd in this function are dead in the end block
  SmallPtrSet<Value*, 64> deadPointers;
  
  // Find all of the alloca'd pointers in the entry block.
  BasicBlock *Entry = BB.getParent()->begin();
  for (BasicBlock::iterator I = Entry->begin(), E = Entry->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      deadPointers.insert(AI);
  
  // Treat byval arguments the same, stores to them are dead at the end of the
  // function.
  for (Function::arg_iterator AI = BB.getParent()->arg_begin(),
       AE = BB.getParent()->arg_end(); AI != AE; ++AI)
    if (AI->hasByValAttr())
      deadPointers.insert(AI);
  
  // Scan the basic block backwards
  for (BasicBlock::iterator BBI = BB.end(); BBI != BB.begin(); ){
    --BBI;
    
    // If we find a store whose pointer is dead.
    if (StoreInst* S = dyn_cast<StoreInst>(BBI)) {
      if (!S->isVolatile()) {
        // See through pointer-to-pointer bitcasts
        Value* pointerOperand = S->getPointerOperand()->getUnderlyingObject();

        // Alloca'd pointers or byval arguments (which are functionally like
        // alloca's) are valid candidates for removal.
        if (deadPointers.count(pointerOperand)) {
          // DCE instructions only used to calculate that store.
          BBI++;
          DeleteDeadInstruction(S, &deadPointers);
          NumFastStores++;
          MadeChange = true;
        }
      }
      
      continue;
    }
    
    // We can also remove memcpy's to local variables at the end of a function.
    if (MemCpyInst *M = dyn_cast<MemCpyInst>(BBI)) {
      Value *dest = M->getDest()->getUnderlyingObject();

      if (deadPointers.count(dest)) {
        BBI++;
        DeleteDeadInstruction(M, &deadPointers);
        NumFastOther++;
        MadeChange = true;
        continue;
      }
      
      // Because a memcpy is also a load, we can't skip it if we didn't remove
      // it.
    }
    
    Value* killPointer = 0;
    uint64_t killPointerSize = ~0UL;
    
    // If we encounter a use of the pointer, it is no longer considered dead
    if (LoadInst *L = dyn_cast<LoadInst>(BBI)) {
      // However, if this load is unused and not volatile, we can go ahead and
      // remove it, and not have to worry about it making our pointer undead!
      if (L->use_empty() && !L->isVolatile()) {
        BBI++;
        DeleteDeadInstruction(L, &deadPointers);
        NumFastOther++;
        MadeChange = true;
        continue;
      }
      
      killPointer = L->getPointerOperand();
    } else if (VAArgInst* V = dyn_cast<VAArgInst>(BBI)) {
      killPointer = V->getOperand(0);
    } else if (isa<MemCpyInst>(BBI) &&
               isa<ConstantInt>(cast<MemCpyInst>(BBI)->getLength())) {
      killPointer = cast<MemCpyInst>(BBI)->getSource();
      killPointerSize = cast<ConstantInt>(
                            cast<MemCpyInst>(BBI)->getLength())->getZExtValue();
    } else if (AllocaInst* A = dyn_cast<AllocaInst>(BBI)) {
      deadPointers.erase(A);
      
      // Dead alloca's can be DCE'd when we reach them
      if (A->use_empty()) {
        BBI++;
        DeleteDeadInstruction(A, &deadPointers);
        NumFastOther++;
        MadeChange = true;
      }
      
      continue;
    } else if (CallSite::get(BBI).getInstruction() != 0) {
      // If this call does not access memory, it can't
      // be undeadifying any of our pointers.
      CallSite CS = CallSite::get(BBI);
      if (AA.doesNotAccessMemory(CS))
        continue;
      
      unsigned modRef = 0;
      unsigned other = 0;
      
      // Remove any pointers made undead by the call from the dead set
      std::vector<Value*> dead;
      for (SmallPtrSet<Value*, 64>::iterator I = deadPointers.begin(),
           E = deadPointers.end(); I != E; ++I) {
        // HACK: if we detect that our AA is imprecise, it's not
        // worth it to scan the rest of the deadPointers set.  Just
        // assume that the AA will return ModRef for everything, and
        // go ahead and bail.
        if (modRef >= 16 && other == 0) {
          deadPointers.clear();
          return MadeChange;
        }

        // Get size information for the alloca
        unsigned pointerSize = ~0U;
        if (AllocaInst* A = dyn_cast<AllocaInst>(*I)) {
          if (ConstantInt* C = dyn_cast<ConstantInt>(A->getArraySize()))
            pointerSize = C->getZExtValue() *
                          TD.getABITypeSize(A->getAllocatedType());
        } else {
          const PointerType* PT = cast<PointerType>(
                                                 cast<Argument>(*I)->getType());
          pointerSize = TD.getABITypeSize(PT->getElementType());
        }

        // See if the call site touches it
        AliasAnalysis::ModRefResult A = AA.getModRefInfo(CS, *I, pointerSize);
        
        if (A == AliasAnalysis::ModRef)
          modRef++;
        else
          other++;
        
        if (A == AliasAnalysis::ModRef || A == AliasAnalysis::Ref)
          dead.push_back(*I);
      }

      for (std::vector<Value*>::iterator I = dead.begin(), E = dead.end();
           I != E; ++I)
        deadPointers.erase(*I);
      
      continue;
    } else if (isInstructionTriviallyDead(BBI)) {
      // For any non-memory-affecting non-terminators, DCE them as we reach them
      Instruction *Inst = BBI;
      BBI++;
      DeleteDeadInstruction(Inst, &deadPointers);
      NumFastOther++;
      MadeChange = true;
      continue;
    }
    
    if (!killPointer)
      continue;

    killPointer = killPointer->getUnderlyingObject();

    // Deal with undead pointers
    MadeChange |= RemoveUndeadPointers(killPointer, killPointerSize, BBI,
                                       deadPointers);
  }
  
  return MadeChange;
}

/// RemoveUndeadPointers - check for uses of a pointer that make it
/// undead when scanning for dead stores to alloca's.
bool DSE::RemoveUndeadPointers(Value* killPointer, uint64_t killPointerSize,
                               BasicBlock::iterator &BBI,
                               SmallPtrSet<Value*, 64>& deadPointers) {
  TargetData &TD = getAnalysis<TargetData>();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
                                  
  // If the kill pointer can be easily reduced to an alloca,
  // don't bother doing extraneous AA queries.
  if (deadPointers.count(killPointer)) {
    deadPointers.erase(killPointer);
    return false;
  }
  
  // A global can't be in the dead pointer set.
  if (isa<GlobalValue>(killPointer))
    return false;
  
  bool MadeChange = false;
  
  SmallVector<Value*, 16> undead;
    
  for (SmallPtrSet<Value*, 64>::iterator I = deadPointers.begin(),
      E = deadPointers.end(); I != E; ++I) {
    // Get size information for the alloca.
    unsigned pointerSize = ~0U;
    if (AllocaInst* A = dyn_cast<AllocaInst>(*I)) {
      if (ConstantInt* C = dyn_cast<ConstantInt>(A->getArraySize()))
        pointerSize = C->getZExtValue() *
                      TD.getABITypeSize(A->getAllocatedType());
    } else {
      const PointerType* PT = cast<PointerType>(cast<Argument>(*I)->getType());
      pointerSize = TD.getABITypeSize(PT->getElementType());
    }

    // See if this pointer could alias it
    AliasAnalysis::AliasResult A = AA.alias(*I, pointerSize,
                                            killPointer, killPointerSize);

    // If it must-alias and a store, we can delete it
    if (isa<StoreInst>(BBI) && A == AliasAnalysis::MustAlias) {
      StoreInst* S = cast<StoreInst>(BBI);

      // Remove it!
      BBI++;
      DeleteDeadInstruction(S, &deadPointers);
      NumFastStores++;
      MadeChange = true;

      continue;

      // Otherwise, it is undead
    } else if (A != AliasAnalysis::NoAlias)
      undead.push_back(*I);
  }

  for (SmallVector<Value*, 16>::iterator I = undead.begin(), E = undead.end();
       I != E; ++I)
      deadPointers.erase(*I);
  
  return MadeChange;
}

/// DeleteDeadInstruction - Delete this instruction.  Before we do, go through
/// and zero out all the operands of this instruction.  If any of them become
/// dead, delete them and the computation tree that feeds them.
///
/// If ValueSet is non-null, remove any deleted instructions from it as well.
///
void DSE::DeleteDeadInstruction(Instruction *I,
                                SmallPtrSet<Value*, 64> *ValueSet) {
  SmallVector<Instruction*, 32> NowDeadInsts;
  
  NowDeadInsts.push_back(I);
  --NumFastOther;

  // Before we touch this instruction, remove it from memdep!
  MemoryDependenceAnalysis &MDA = getAnalysis<MemoryDependenceAnalysis>();
  while (!NowDeadInsts.empty()) {
    Instruction *DeadInst = NowDeadInsts.back();
    NowDeadInsts.pop_back();
    
    ++NumFastOther;
    
    // This instruction is dead, zap it, in stages.  Start by removing it from
    // MemDep, which needs to know the operands and needs it to be in the
    // function.
    MDA.removeInstruction(DeadInst);
    
    for (unsigned op = 0, e = DeadInst->getNumOperands(); op != e; ++op) {
      Value *Op = DeadInst->getOperand(op);
      DeadInst->setOperand(op, 0);
      
      // If this operand just became dead, add it to the NowDeadInsts list.
      if (!Op->use_empty()) continue;
      
      if (Instruction *OpI = dyn_cast<Instruction>(Op))
        if (isInstructionTriviallyDead(OpI))
          NowDeadInsts.push_back(OpI);
    }
    
    DeadInst->eraseFromParent();
    
    if (ValueSet) ValueSet->erase(DeadInst);
  }
}

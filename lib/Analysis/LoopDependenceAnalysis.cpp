//===- LoopDependenceAnalysis.cpp - LDA Implementation ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the (beginning) of an implementation of a loop dependence analysis
// framework, which is used to detect dependences in memory accesses in loops.
//
// Please note that this is work in progress and the interface is subject to
// change.
//
// TODO: adapt as implementation progresses.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lda"
#include "llvm/Analysis/LoopDependenceAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Instructions.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

LoopPass *llvm::createLoopDependenceAnalysisPass() {
  return new LoopDependenceAnalysis();
}

static RegisterPass<LoopDependenceAnalysis>
R("lda", "Loop Dependence Analysis", false, true);
char LoopDependenceAnalysis::ID = 0;

//===----------------------------------------------------------------------===//
//                             Utility Functions
//===----------------------------------------------------------------------===//

static inline bool IsMemRefInstr(const Value *V) {
  const Instruction *I = dyn_cast<const Instruction>(V);
  return I && (I->mayReadFromMemory() || I->mayWriteToMemory());
}

static void GetMemRefInstrs(
    const Loop *L, SmallVectorImpl<Instruction*> &memrefs) {
  for (Loop::block_iterator b = L->block_begin(), be = L->block_end();
      b != be; ++b)
    for (BasicBlock::iterator i = (*b)->begin(), ie = (*b)->end();
        i != ie; ++i)
      if (IsMemRefInstr(i))
        memrefs.push_back(i);
}

static bool IsLoadOrStoreInst(Value *I) {
  return isa<LoadInst>(I) || isa<StoreInst>(I);
}

static Value *GetPointerOperand(Value *I) {
  if (LoadInst *i = dyn_cast<LoadInst>(I))
    return i->getPointerOperand();
  if (StoreInst *i = dyn_cast<StoreInst>(I))
    return i->getPointerOperand();
  assert(0 && "Value is no load or store instruction!");
  // Never reached.
  return 0;
}

//===----------------------------------------------------------------------===//
//                             Dependence Testing
//===----------------------------------------------------------------------===//

bool LoopDependenceAnalysis::isDependencePair(const Value *x,
                                              const Value *y) const {
  return IsMemRefInstr(x) &&
         IsMemRefInstr(y) &&
         (cast<const Instruction>(x)->mayWriteToMemory() ||
          cast<const Instruction>(y)->mayWriteToMemory());
}

bool LoopDependenceAnalysis::depends(Value *src, Value *dst) {
  assert(isDependencePair(src, dst) && "Values form no dependence pair!");
  DOUT << "== LDA test ==\n" << *src << *dst;

  // We only analyse loads and stores; for possible memory accesses by e.g.
  // free, call, or invoke instructions we conservatively assume dependence.
  if (!IsLoadOrStoreInst(src) || !IsLoadOrStoreInst(dst))
    return true;

  Value *srcPtr = GetPointerOperand(src);
  Value *dstPtr = GetPointerOperand(dst);
  const Value *srcObj = srcPtr->getUnderlyingObject();
  const Value *dstObj = dstPtr->getUnderlyingObject();
  const Type *srcTy = srcObj->getType();
  const Type *dstTy = dstObj->getType();

  // For now, we only work on (pointers to) global or stack-allocated array
  // values, as we know that their underlying memory areas will not overlap.
  // MAYBE: relax this and test for aliasing?
  if (!((isa<GlobalVariable>(srcObj) || isa<AllocaInst>(srcObj)) &&
        (isa<GlobalVariable>(dstObj) || isa<AllocaInst>(dstObj)) &&
        isa<PointerType>(srcTy) &&
        isa<PointerType>(dstTy) &&
        isa<ArrayType>(cast<PointerType>(srcTy)->getElementType()) &&
        isa<ArrayType>(cast<PointerType>(dstTy)->getElementType())))
    return true;

  // If the arrays are different, the underlying memory areas do not overlap
  // and the memory accesses are therefore independent.
  if (srcObj != dstObj)
    return false;

  // We couldn't establish a more precise result, so we have to conservatively
  // assume full dependence.
  return true;
}

//===----------------------------------------------------------------------===//
//                   LoopDependenceAnalysis Implementation
//===----------------------------------------------------------------------===//

bool LoopDependenceAnalysis::runOnLoop(Loop *L, LPPassManager &) {
  this->L = L;
  SE = &getAnalysis<ScalarEvolution>();
  return false;
}

void LoopDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<ScalarEvolution>();
}

static void PrintLoopInfo(
    raw_ostream &OS, LoopDependenceAnalysis *LDA, const Loop *L) {
  if (!L->empty()) return; // ignore non-innermost loops

  OS << "Loop at depth " << L->getLoopDepth() << ", header block: ";
  WriteAsOperand(OS, L->getHeader(), false);
  OS << "\n";

  SmallVector<Instruction*, 8> memrefs;
  GetMemRefInstrs(L, memrefs);
  OS << "  Load/store instructions: " << memrefs.size() << "\n";
  OS << "  Pairwise dependence results:\n";
  for (SmallVector<Instruction*, 8>::const_iterator x = memrefs.begin(),
      end = memrefs.end(); x != end; ++x)
    for (SmallVector<Instruction*, 8>::const_iterator y = x + 1;
        y != end; ++y)
      if (LDA->isDependencePair(*x, *y))
        OS << "\t" << (x - memrefs.begin()) << "," << (y - memrefs.begin())
           << ": " << (LDA->depends(*x, *y) ? "dependent" : "independent")
           << "\n";
}

void LoopDependenceAnalysis::print(raw_ostream &OS, const Module*) const {
  // TODO: doc why const_cast is safe
  PrintLoopInfo(OS, const_cast<LoopDependenceAnalysis*>(this), this->L);
}

void LoopDependenceAnalysis::print(std::ostream &OS, const Module *M) const {
  raw_os_ostream os(OS);
  print(os, M);
}

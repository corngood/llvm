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
// TODO: document lingo (pair, subscript, index)
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lda"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopDependenceAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Instructions.h"
#include "llvm/Operator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

STATISTIC(NumAnswered,    "Number of dependence queries answered");
STATISTIC(NumAnalysed,    "Number of distinct dependence pairs analysed");
STATISTIC(NumDependent,   "Number of pairs with dependent accesses");
STATISTIC(NumIndependent, "Number of pairs with independent accesses");
STATISTIC(NumUnknown,     "Number of pairs with unknown accesses");

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

static void GetMemRefInstrs(const Loop *L,
                            SmallVectorImpl<Instruction*> &Memrefs) {
  for (Loop::block_iterator b = L->block_begin(), be = L->block_end();
       b != be; ++b)
    for (BasicBlock::iterator i = (*b)->begin(), ie = (*b)->end();
         i != ie; ++i)
      if (IsMemRefInstr(i))
        Memrefs.push_back(i);
}

static bool IsLoadOrStoreInst(Value *I) {
  return isa<LoadInst>(I) || isa<StoreInst>(I);
}

static Value *GetPointerOperand(Value *I) {
  if (LoadInst *i = dyn_cast<LoadInst>(I))
    return i->getPointerOperand();
  if (StoreInst *i = dyn_cast<StoreInst>(I))
    return i->getPointerOperand();
  llvm_unreachable("Value is no load or store instruction!");
  // Never reached.
  return 0;
}

static AliasAnalysis::AliasResult UnderlyingObjectsAlias(AliasAnalysis *AA,
                                                         const Value *A,
                                                         const Value *B) {
  const Value *aObj = A->getUnderlyingObject();
  const Value *bObj = B->getUnderlyingObject();
  return AA->alias(aObj, AA->getTypeStoreSize(aObj->getType()),
                   bObj, AA->getTypeStoreSize(bObj->getType()));
}

static inline const SCEV *GetZeroSCEV(ScalarEvolution *SE) {
  return SE->getConstant(Type::Int32Ty, 0L);
}

//===----------------------------------------------------------------------===//
//                             Dependence Testing
//===----------------------------------------------------------------------===//

bool LoopDependenceAnalysis::isDependencePair(const Value *A,
                                              const Value *B) const {
  return IsMemRefInstr(A) &&
         IsMemRefInstr(B) &&
         (cast<const Instruction>(A)->mayWriteToMemory() ||
          cast<const Instruction>(B)->mayWriteToMemory());
}

bool LoopDependenceAnalysis::findOrInsertDependencePair(Value *A,
                                                        Value *B,
                                                        DependencePair *&P) {
  void *insertPos = 0;
  FoldingSetNodeID id;
  id.AddPointer(A);
  id.AddPointer(B);

  P = Pairs.FindNodeOrInsertPos(id, insertPos);
  if (P) return true;

  P = PairAllocator.Allocate<DependencePair>();
  new (P) DependencePair(id, A, B);
  Pairs.InsertNode(P, insertPos);
  return false;
}

bool LoopDependenceAnalysis::isLoopInvariant(const SCEV *S) const {
  for (const Loop *L = this->L; L != 0; L = L->getParentLoop())
    if (!S->isLoopInvariant(L))
      return false;
  return true;
}

bool LoopDependenceAnalysis::isAffine(const SCEV *S) const {
  const SCEVAddRecExpr *rec = dyn_cast<SCEVAddRecExpr>(S);
  return isLoopInvariant(S) || (rec && rec->isAffine());
}

LoopDependenceAnalysis::DependenceResult
LoopDependenceAnalysis::analyseSubscript(const SCEV *A,
                                         const SCEV *B,
                                         Subscript *S) const {
  DEBUG(errs() << "  Testing subscript: " << *A << ", " << *B << "\n");

  if (A == B) {
    DEBUG(errs() << "  -> [D] same SCEV\n");
    return Dependent;
  }

  if (!isAffine(A) || !isAffine(B)) {
    DEBUG(errs() << "  -> [?] not affine\n");
    return Unknown;
  }

  // TODO: Implement ZIV/SIV/MIV testers.

  DEBUG(errs() << "  -> [?] cannot analyse subscript\n");
  return Unknown;
}

LoopDependenceAnalysis::DependenceResult
LoopDependenceAnalysis::analysePair(DependencePair *P) const {
  DEBUG(errs() << "Analysing:\n" << *P->A << "\n" << *P->B << "\n");

  // We only analyse loads and stores but no possible memory accesses by e.g.
  // free, call, or invoke instructions.
  if (!IsLoadOrStoreInst(P->A) || !IsLoadOrStoreInst(P->B)) {
    DEBUG(errs() << "--> [?] no load/store\n");
    return Unknown;
  }

  Value *aPtr = GetPointerOperand(P->A);
  Value *bPtr = GetPointerOperand(P->B);

  switch (UnderlyingObjectsAlias(AA, aPtr, bPtr)) {
  case AliasAnalysis::MayAlias:
    // We can not analyse objects if we do not know about their aliasing.
    DEBUG(errs() << "---> [?] may alias\n");
    return Unknown;

  case AliasAnalysis::NoAlias:
    // If the objects noalias, they are distinct, accesses are independent.
    DEBUG(errs() << "---> [I] no alias\n");
    return Independent;

  case AliasAnalysis::MustAlias:
    break; // The underlying objects alias, test accesses for dependence.
  }

  const GEPOperator *aGEP = dyn_cast<GEPOperator>(aPtr);
  const GEPOperator *bGEP = dyn_cast<GEPOperator>(bPtr);

  if (!aGEP || !bGEP)
    return Unknown;

  // FIXME: Is filtering coupled subscripts necessary?

  // Analyse indices pairwise (FIXME: use GetGEPOperands from BasicAA), adding
  // trailing zeroes to the smaller GEP, if needed.
  GEPOperator::const_op_iterator aIdx = aGEP->idx_begin(),
                                 aEnd = aGEP->idx_end(),
                                 bIdx = bGEP->idx_begin(),
                                 bEnd = bGEP->idx_end();
  while (aIdx != aEnd && bIdx != bEnd) {
    const SCEV* aSCEV = (aIdx != aEnd) ? SE->getSCEV(*aIdx) : GetZeroSCEV(SE);
    const SCEV* bSCEV = (bIdx != bEnd) ? SE->getSCEV(*bIdx) : GetZeroSCEV(SE);
    Subscript subscript;
    DependenceResult result = analyseSubscript(aSCEV, bSCEV, &subscript);
    if (result != Dependent) {
      // We either proved independence or failed to analyse this subscript.
      // Further subscripts will not improve the situation, so abort early.
      return result;
    }
    P->Subscripts.push_back(subscript);
    if (aIdx != aEnd) ++aIdx;
    if (bIdx != bEnd) ++bIdx;
  }
  // Either there were no subscripts or all subscripts were analysed to be
  // dependent; in both cases we know the accesses are dependent.
  return Dependent;
}

bool LoopDependenceAnalysis::depends(Value *A, Value *B) {
  assert(isDependencePair(A, B) && "Values form no dependence pair!");
  ++NumAnswered;

  DependencePair *p;
  if (!findOrInsertDependencePair(A, B, p)) {
    // The pair is not cached, so analyse it.
    ++NumAnalysed;
    switch (p->Result = analysePair(p)) {
    case Dependent:   ++NumDependent;   break;
    case Independent: ++NumIndependent; break;
    case Unknown:     ++NumUnknown;     break;
    }
  }
  return p->Result != Independent;
}

//===----------------------------------------------------------------------===//
//                   LoopDependenceAnalysis Implementation
//===----------------------------------------------------------------------===//

bool LoopDependenceAnalysis::runOnLoop(Loop *L, LPPassManager &) {
  this->L = L;
  AA = &getAnalysis<AliasAnalysis>();
  SE = &getAnalysis<ScalarEvolution>();
  return false;
}

void LoopDependenceAnalysis::releaseMemory() {
  Pairs.clear();
  PairAllocator.Reset();
}

void LoopDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<ScalarEvolution>();
}

static void PrintLoopInfo(raw_ostream &OS,
                          LoopDependenceAnalysis *LDA, const Loop *L) {
  if (!L->empty()) return; // ignore non-innermost loops

  SmallVector<Instruction*, 8> memrefs;
  GetMemRefInstrs(L, memrefs);

  OS << "Loop at depth " << L->getLoopDepth() << ", header block: ";
  WriteAsOperand(OS, L->getHeader(), false);
  OS << "\n";

  OS << "  Load/store instructions: " << memrefs.size() << "\n";
  for (SmallVector<Instruction*, 8>::const_iterator x = memrefs.begin(),
       end = memrefs.end(); x != end; ++x)
    OS << "\t" << (x - memrefs.begin()) << ": " << **x << "\n";

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

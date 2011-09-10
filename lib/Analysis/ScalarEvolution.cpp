//===- ScalarEvolution.cpp - Scalar Evolution Analysis ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution analysis
// engine, which is used primarily to analyze expressions involving induction
// variables in loops.
//
// There are several aspects to this library.  First is the representation of
// scalar expressions, which are represented as subclasses of the SCEV class.
// These classes are used to represent certain types of subexpressions that we
// can handle. We only create one SCEV of a particular shape, so
// pointer-comparisons for equality are legal.
//
// One important aspect of the SCEV objects is that they are never cyclic, even
// if there is a cycle in the dataflow for an expression (ie, a PHI node).  If
// the PHI node is one of the idioms that we can represent (e.g., a polynomial
// recurrence) then we represent it directly as a recurrence node, otherwise we
// represent it as a SCEVUnknown node.
//
// In addition to being able to represent expressions of various types, we also
// have folders that are used to build the *canonical* representation for a
// particular expression.  These folders are capable of using a variety of
// rewrite rules to simplify the expressions.
//
// Once the folders are defined, we can implement the more interesting
// higher-level code, such as the code that recognizes PHI nodes of various
// types, computes the execution count of a loop, etc.
//
// TODO: We should use these routines and value representations to implement
// dependence analysis!
//
//===----------------------------------------------------------------------===//
//
// There are several good references for the techniques used in this analysis.
//
//  Chains of recurrences -- a method to expedite the evaluation
//  of closed-form functions
//  Olaf Bachmann, Paul S. Wang, Eugene V. Zima
//
//  On computational properties of chains of recurrences
//  Eugene V. Zima
//
//  Symbolic Evaluation of Chains of Recurrences for Loop Optimization
//  Robert A. van Engelen
//
//  Efficient Symbolic Analysis for Optimizing Compilers
//  Robert A. van Engelen
//
//  Using the chains of recurrences algebra for data dependence testing and
//  induction variable substitution
//  MS Thesis, Johnie Birch
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "scalar-evolution"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Operator.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumArrayLenItCounts,
          "Number of trip counts computed with array length");
STATISTIC(NumTripCountsComputed,
          "Number of loops with predictable loop counts");
STATISTIC(NumTripCountsNotComputed,
          "Number of loops without predictable loop counts");
STATISTIC(NumBruteForceTripCountsComputed,
          "Number of loops with trip counts computed by force");

static cl::opt<unsigned>
MaxBruteForceIterations("scalar-evolution-max-iterations", cl::ReallyHidden,
                        cl::desc("Maximum number of iterations SCEV will "
                                 "symbolically execute a constant "
                                 "derived loop"),
                        cl::init(100));

INITIALIZE_PASS_BEGIN(ScalarEvolution, "scalar-evolution",
                "Scalar Evolution Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(ScalarEvolution, "scalar-evolution",
                "Scalar Evolution Analysis", false, true)
char ScalarEvolution::ID = 0;

//===----------------------------------------------------------------------===//
//                           SCEV class definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Implementation of the SCEV class.
//

void SCEV::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

void SCEV::print(raw_ostream &OS) const {
  switch (getSCEVType()) {
  case scConstant:
    WriteAsOperand(OS, cast<SCEVConstant>(this)->getValue(), false);
    return;
  case scTruncate: {
    const SCEVTruncateExpr *Trunc = cast<SCEVTruncateExpr>(this);
    const SCEV *Op = Trunc->getOperand();
    OS << "(trunc " << *Op->getType() << " " << *Op << " to "
       << *Trunc->getType() << ")";
    return;
  }
  case scZeroExtend: {
    const SCEVZeroExtendExpr *ZExt = cast<SCEVZeroExtendExpr>(this);
    const SCEV *Op = ZExt->getOperand();
    OS << "(zext " << *Op->getType() << " " << *Op << " to "
       << *ZExt->getType() << ")";
    return;
  }
  case scSignExtend: {
    const SCEVSignExtendExpr *SExt = cast<SCEVSignExtendExpr>(this);
    const SCEV *Op = SExt->getOperand();
    OS << "(sext " << *Op->getType() << " " << *Op << " to "
       << *SExt->getType() << ")";
    return;
  }
  case scAddRecExpr: {
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(this);
    OS << "{" << *AR->getOperand(0);
    for (unsigned i = 1, e = AR->getNumOperands(); i != e; ++i)
      OS << ",+," << *AR->getOperand(i);
    OS << "}<";
    if (AR->getNoWrapFlags(FlagNUW))
      OS << "nuw><";
    if (AR->getNoWrapFlags(FlagNSW))
      OS << "nsw><";
    if (AR->getNoWrapFlags(FlagNW) &&
        !AR->getNoWrapFlags((NoWrapFlags)(FlagNUW | FlagNSW)))
      OS << "nw><";
    WriteAsOperand(OS, AR->getLoop()->getHeader(), /*PrintType=*/false);
    OS << ">";
    return;
  }
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr: {
    const SCEVNAryExpr *NAry = cast<SCEVNAryExpr>(this);
    const char *OpStr = 0;
    switch (NAry->getSCEVType()) {
    case scAddExpr: OpStr = " + "; break;
    case scMulExpr: OpStr = " * "; break;
    case scUMaxExpr: OpStr = " umax "; break;
    case scSMaxExpr: OpStr = " smax "; break;
    }
    OS << "(";
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      OS << **I;
      if (llvm::next(I) != E)
        OS << OpStr;
    }
    OS << ")";
    return;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(this);
    OS << "(" << *UDiv->getLHS() << " /u " << *UDiv->getRHS() << ")";
    return;
  }
  case scUnknown: {
    const SCEVUnknown *U = cast<SCEVUnknown>(this);
    Type *AllocTy;
    if (U->isSizeOf(AllocTy)) {
      OS << "sizeof(" << *AllocTy << ")";
      return;
    }
    if (U->isAlignOf(AllocTy)) {
      OS << "alignof(" << *AllocTy << ")";
      return;
    }

    Type *CTy;
    Constant *FieldNo;
    if (U->isOffsetOf(CTy, FieldNo)) {
      OS << "offsetof(" << *CTy << ", ";
      WriteAsOperand(OS, FieldNo, false);
      OS << ")";
      return;
    }

    // Otherwise just print it normally.
    WriteAsOperand(OS, U->getValue(), false);
    return;
  }
  case scCouldNotCompute:
    OS << "***COULDNOTCOMPUTE***";
    return;
  default: break;
  }
  llvm_unreachable("Unknown SCEV kind!");
}

Type *SCEV::getType() const {
  switch (getSCEVType()) {
  case scConstant:
    return cast<SCEVConstant>(this)->getType();
  case scTruncate:
  case scZeroExtend:
  case scSignExtend:
    return cast<SCEVCastExpr>(this)->getType();
  case scAddRecExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr:
    return cast<SCEVNAryExpr>(this)->getType();
  case scAddExpr:
    return cast<SCEVAddExpr>(this)->getType();
  case scUDivExpr:
    return cast<SCEVUDivExpr>(this)->getType();
  case scUnknown:
    return cast<SCEVUnknown>(this)->getType();
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
    return 0;
  default: break;
  }
  llvm_unreachable("Unknown SCEV kind!");
  return 0;
}

bool SCEV::isZero() const {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(this))
    return SC->getValue()->isZero();
  return false;
}

bool SCEV::isOne() const {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(this))
    return SC->getValue()->isOne();
  return false;
}

bool SCEV::isAllOnesValue() const {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(this))
    return SC->getValue()->isAllOnesValue();
  return false;
}

SCEVCouldNotCompute::SCEVCouldNotCompute() :
  SCEV(FoldingSetNodeIDRef(), scCouldNotCompute) {}

bool SCEVCouldNotCompute::classof(const SCEV *S) {
  return S->getSCEVType() == scCouldNotCompute;
}

const SCEV *ScalarEvolution::getConstant(ConstantInt *V) {
  FoldingSetNodeID ID;
  ID.AddInteger(scConstant);
  ID.AddPointer(V);
  void *IP = 0;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVConstant(ID.Intern(SCEVAllocator), V);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getConstant(const APInt& Val) {
  return getConstant(ConstantInt::get(getContext(), Val));
}

const SCEV *
ScalarEvolution::getConstant(Type *Ty, uint64_t V, bool isSigned) {
  IntegerType *ITy = cast<IntegerType>(getEffectiveSCEVType(Ty));
  return getConstant(ConstantInt::get(ITy, V, isSigned));
}

SCEVCastExpr::SCEVCastExpr(const FoldingSetNodeIDRef ID,
                           unsigned SCEVTy, const SCEV *op, Type *ty)
  : SCEV(ID, SCEVTy), Op(op), Ty(ty) {}

SCEVTruncateExpr::SCEVTruncateExpr(const FoldingSetNodeIDRef ID,
                                   const SCEV *op, Type *ty)
  : SCEVCastExpr(ID, scTruncate, op, ty) {
  assert((Op->getType()->isIntegerTy() || Op->getType()->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate non-integer value!");
}

SCEVZeroExtendExpr::SCEVZeroExtendExpr(const FoldingSetNodeIDRef ID,
                                       const SCEV *op, Type *ty)
  : SCEVCastExpr(ID, scZeroExtend, op, ty) {
  assert((Op->getType()->isIntegerTy() || Op->getType()->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot zero extend non-integer value!");
}

SCEVSignExtendExpr::SCEVSignExtendExpr(const FoldingSetNodeIDRef ID,
                                       const SCEV *op, Type *ty)
  : SCEVCastExpr(ID, scSignExtend, op, ty) {
  assert((Op->getType()->isIntegerTy() || Op->getType()->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot sign extend non-integer value!");
}

void SCEVUnknown::deleted() {
  // Clear this SCEVUnknown from various maps.
  SE->forgetMemoizedResults(this);

  // Remove this SCEVUnknown from the uniquing map.
  SE->UniqueSCEVs.RemoveNode(this);

  // Release the value.
  setValPtr(0);
}

void SCEVUnknown::allUsesReplacedWith(Value *New) {
  // Clear this SCEVUnknown from various maps.
  SE->forgetMemoizedResults(this);

  // Remove this SCEVUnknown from the uniquing map.
  SE->UniqueSCEVs.RemoveNode(this);

  // Update this SCEVUnknown to point to the new value. This is needed
  // because there may still be outstanding SCEVs which still point to
  // this SCEVUnknown.
  setValPtr(New);
}

bool SCEVUnknown::isSizeOf(Type *&AllocTy) const {
  if (ConstantExpr *VCE = dyn_cast<ConstantExpr>(getValue()))
    if (VCE->getOpcode() == Instruction::PtrToInt)
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(VCE->getOperand(0)))
        if (CE->getOpcode() == Instruction::GetElementPtr &&
            CE->getOperand(0)->isNullValue() &&
            CE->getNumOperands() == 2)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(CE->getOperand(1)))
            if (CI->isOne()) {
              AllocTy = cast<PointerType>(CE->getOperand(0)->getType())
                                 ->getElementType();
              return true;
            }

  return false;
}

bool SCEVUnknown::isAlignOf(Type *&AllocTy) const {
  if (ConstantExpr *VCE = dyn_cast<ConstantExpr>(getValue()))
    if (VCE->getOpcode() == Instruction::PtrToInt)
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(VCE->getOperand(0)))
        if (CE->getOpcode() == Instruction::GetElementPtr &&
            CE->getOperand(0)->isNullValue()) {
          Type *Ty =
            cast<PointerType>(CE->getOperand(0)->getType())->getElementType();
          if (StructType *STy = dyn_cast<StructType>(Ty))
            if (!STy->isPacked() &&
                CE->getNumOperands() == 3 &&
                CE->getOperand(1)->isNullValue()) {
              if (ConstantInt *CI = dyn_cast<ConstantInt>(CE->getOperand(2)))
                if (CI->isOne() &&
                    STy->getNumElements() == 2 &&
                    STy->getElementType(0)->isIntegerTy(1)) {
                  AllocTy = STy->getElementType(1);
                  return true;
                }
            }
        }

  return false;
}

bool SCEVUnknown::isOffsetOf(Type *&CTy, Constant *&FieldNo) const {
  if (ConstantExpr *VCE = dyn_cast<ConstantExpr>(getValue()))
    if (VCE->getOpcode() == Instruction::PtrToInt)
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(VCE->getOperand(0)))
        if (CE->getOpcode() == Instruction::GetElementPtr &&
            CE->getNumOperands() == 3 &&
            CE->getOperand(0)->isNullValue() &&
            CE->getOperand(1)->isNullValue()) {
          Type *Ty =
            cast<PointerType>(CE->getOperand(0)->getType())->getElementType();
          // Ignore vector types here so that ScalarEvolutionExpander doesn't
          // emit getelementptrs that index into vectors.
          if (Ty->isStructTy() || Ty->isArrayTy()) {
            CTy = Ty;
            FieldNo = CE->getOperand(2);
            return true;
          }
        }

  return false;
}

//===----------------------------------------------------------------------===//
//                               SCEV Utilities
//===----------------------------------------------------------------------===//

namespace {
  /// SCEVComplexityCompare - Return true if the complexity of the LHS is less
  /// than the complexity of the RHS.  This comparator is used to canonicalize
  /// expressions.
  class SCEVComplexityCompare {
    const LoopInfo *const LI;
  public:
    explicit SCEVComplexityCompare(const LoopInfo *li) : LI(li) {}

    // Return true or false if LHS is less than, or at least RHS, respectively.
    bool operator()(const SCEV *LHS, const SCEV *RHS) const {
      return compare(LHS, RHS) < 0;
    }

    // Return negative, zero, or positive, if LHS is less than, equal to, or
    // greater than RHS, respectively. A three-way result allows recursive
    // comparisons to be more efficient.
    int compare(const SCEV *LHS, const SCEV *RHS) const {
      // Fast-path: SCEVs are uniqued so we can do a quick equality check.
      if (LHS == RHS)
        return 0;

      // Primarily, sort the SCEVs by their getSCEVType().
      unsigned LType = LHS->getSCEVType(), RType = RHS->getSCEVType();
      if (LType != RType)
        return (int)LType - (int)RType;

      // Aside from the getSCEVType() ordering, the particular ordering
      // isn't very important except that it's beneficial to be consistent,
      // so that (a + b) and (b + a) don't end up as different expressions.
      switch (LType) {
      case scUnknown: {
        const SCEVUnknown *LU = cast<SCEVUnknown>(LHS);
        const SCEVUnknown *RU = cast<SCEVUnknown>(RHS);

        // Sort SCEVUnknown values with some loose heuristics. TODO: This is
        // not as complete as it could be.
        const Value *LV = LU->getValue(), *RV = RU->getValue();

        // Order pointer values after integer values. This helps SCEVExpander
        // form GEPs.
        bool LIsPointer = LV->getType()->isPointerTy(),
             RIsPointer = RV->getType()->isPointerTy();
        if (LIsPointer != RIsPointer)
          return (int)LIsPointer - (int)RIsPointer;

        // Compare getValueID values.
        unsigned LID = LV->getValueID(),
                 RID = RV->getValueID();
        if (LID != RID)
          return (int)LID - (int)RID;

        // Sort arguments by their position.
        if (const Argument *LA = dyn_cast<Argument>(LV)) {
          const Argument *RA = cast<Argument>(RV);
          unsigned LArgNo = LA->getArgNo(), RArgNo = RA->getArgNo();
          return (int)LArgNo - (int)RArgNo;
        }

        // For instructions, compare their loop depth, and their operand
        // count.  This is pretty loose.
        if (const Instruction *LInst = dyn_cast<Instruction>(LV)) {
          const Instruction *RInst = cast<Instruction>(RV);

          // Compare loop depths.
          const BasicBlock *LParent = LInst->getParent(),
                           *RParent = RInst->getParent();
          if (LParent != RParent) {
            unsigned LDepth = LI->getLoopDepth(LParent),
                     RDepth = LI->getLoopDepth(RParent);
            if (LDepth != RDepth)
              return (int)LDepth - (int)RDepth;
          }

          // Compare the number of operands.
          unsigned LNumOps = LInst->getNumOperands(),
                   RNumOps = RInst->getNumOperands();
          return (int)LNumOps - (int)RNumOps;
        }

        return 0;
      }

      case scConstant: {
        const SCEVConstant *LC = cast<SCEVConstant>(LHS);
        const SCEVConstant *RC = cast<SCEVConstant>(RHS);

        // Compare constant values.
        const APInt &LA = LC->getValue()->getValue();
        const APInt &RA = RC->getValue()->getValue();
        unsigned LBitWidth = LA.getBitWidth(), RBitWidth = RA.getBitWidth();
        if (LBitWidth != RBitWidth)
          return (int)LBitWidth - (int)RBitWidth;
        return LA.ult(RA) ? -1 : 1;
      }

      case scAddRecExpr: {
        const SCEVAddRecExpr *LA = cast<SCEVAddRecExpr>(LHS);
        const SCEVAddRecExpr *RA = cast<SCEVAddRecExpr>(RHS);

        // Compare addrec loop depths.
        const Loop *LLoop = LA->getLoop(), *RLoop = RA->getLoop();
        if (LLoop != RLoop) {
          unsigned LDepth = LLoop->getLoopDepth(),
                   RDepth = RLoop->getLoopDepth();
          if (LDepth != RDepth)
            return (int)LDepth - (int)RDepth;
        }

        // Addrec complexity grows with operand count.
        unsigned LNumOps = LA->getNumOperands(), RNumOps = RA->getNumOperands();
        if (LNumOps != RNumOps)
          return (int)LNumOps - (int)RNumOps;

        // Lexicographically compare.
        for (unsigned i = 0; i != LNumOps; ++i) {
          long X = compare(LA->getOperand(i), RA->getOperand(i));
          if (X != 0)
            return X;
        }

        return 0;
      }

      case scAddExpr:
      case scMulExpr:
      case scSMaxExpr:
      case scUMaxExpr: {
        const SCEVNAryExpr *LC = cast<SCEVNAryExpr>(LHS);
        const SCEVNAryExpr *RC = cast<SCEVNAryExpr>(RHS);

        // Lexicographically compare n-ary expressions.
        unsigned LNumOps = LC->getNumOperands(), RNumOps = RC->getNumOperands();
        for (unsigned i = 0; i != LNumOps; ++i) {
          if (i >= RNumOps)
            return 1;
          long X = compare(LC->getOperand(i), RC->getOperand(i));
          if (X != 0)
            return X;
        }
        return (int)LNumOps - (int)RNumOps;
      }

      case scUDivExpr: {
        const SCEVUDivExpr *LC = cast<SCEVUDivExpr>(LHS);
        const SCEVUDivExpr *RC = cast<SCEVUDivExpr>(RHS);

        // Lexicographically compare udiv expressions.
        long X = compare(LC->getLHS(), RC->getLHS());
        if (X != 0)
          return X;
        return compare(LC->getRHS(), RC->getRHS());
      }

      case scTruncate:
      case scZeroExtend:
      case scSignExtend: {
        const SCEVCastExpr *LC = cast<SCEVCastExpr>(LHS);
        const SCEVCastExpr *RC = cast<SCEVCastExpr>(RHS);

        // Compare cast expressions by operand.
        return compare(LC->getOperand(), RC->getOperand());
      }

      default:
        break;
      }

      llvm_unreachable("Unknown SCEV kind!");
      return 0;
    }
  };
}

/// GroupByComplexity - Given a list of SCEV objects, order them by their
/// complexity, and group objects of the same complexity together by value.
/// When this routine is finished, we know that any duplicates in the vector are
/// consecutive and that complexity is monotonically increasing.
///
/// Note that we go take special precautions to ensure that we get deterministic
/// results from this routine.  In other words, we don't want the results of
/// this to depend on where the addresses of various SCEV objects happened to
/// land in memory.
///
static void GroupByComplexity(SmallVectorImpl<const SCEV *> &Ops,
                              LoopInfo *LI) {
  if (Ops.size() < 2) return;  // Noop
  if (Ops.size() == 2) {
    // This is the common case, which also happens to be trivially simple.
    // Special case it.
    const SCEV *&LHS = Ops[0], *&RHS = Ops[1];
    if (SCEVComplexityCompare(LI)(RHS, LHS))
      std::swap(LHS, RHS);
    return;
  }

  // Do the rough sort by complexity.
  std::stable_sort(Ops.begin(), Ops.end(), SCEVComplexityCompare(LI));

  // Now that we are sorted by complexity, group elements of the same
  // complexity.  Note that this is, at worst, N^2, but the vector is likely to
  // be extremely short in practice.  Note that we take this approach because we
  // do not want to depend on the addresses of the objects we are grouping.
  for (unsigned i = 0, e = Ops.size(); i != e-2; ++i) {
    const SCEV *S = Ops[i];
    unsigned Complexity = S->getSCEVType();

    // If there are any objects of the same complexity and same value as this
    // one, group them.
    for (unsigned j = i+1; j != e && Ops[j]->getSCEVType() == Complexity; ++j) {
      if (Ops[j] == S) { // Found a duplicate.
        // Move it to immediately after i'th element.
        std::swap(Ops[i+1], Ops[j]);
        ++i;   // no need to rescan it.
        if (i == e-2) return;  // Done!
      }
    }
  }
}



//===----------------------------------------------------------------------===//
//                      Simple SCEV method implementations
//===----------------------------------------------------------------------===//

/// BinomialCoefficient - Compute BC(It, K).  The result has width W.
/// Assume, K > 0.
static const SCEV *BinomialCoefficient(const SCEV *It, unsigned K,
                                       ScalarEvolution &SE,
                                       Type *ResultTy) {
  // Handle the simplest case efficiently.
  if (K == 1)
    return SE.getTruncateOrZeroExtend(It, ResultTy);

  // We are using the following formula for BC(It, K):
  //
  //   BC(It, K) = (It * (It - 1) * ... * (It - K + 1)) / K!
  //
  // Suppose, W is the bitwidth of the return value.  We must be prepared for
  // overflow.  Hence, we must assure that the result of our computation is
  // equal to the accurate one modulo 2^W.  Unfortunately, division isn't
  // safe in modular arithmetic.
  //
  // However, this code doesn't use exactly that formula; the formula it uses
  // is something like the following, where T is the number of factors of 2 in
  // K! (i.e. trailing zeros in the binary representation of K!), and ^ is
  // exponentiation:
  //
  //   BC(It, K) = (It * (It - 1) * ... * (It - K + 1)) / 2^T / (K! / 2^T)
  //
  // This formula is trivially equivalent to the previous formula.  However,
  // this formula can be implemented much more efficiently.  The trick is that
  // K! / 2^T is odd, and exact division by an odd number *is* safe in modular
  // arithmetic.  To do exact division in modular arithmetic, all we have
  // to do is multiply by the inverse.  Therefore, this step can be done at
  // width W.
  //
  // The next issue is how to safely do the division by 2^T.  The way this
  // is done is by doing the multiplication step at a width of at least W + T
  // bits.  This way, the bottom W+T bits of the product are accurate. Then,
  // when we perform the division by 2^T (which is equivalent to a right shift
  // by T), the bottom W bits are accurate.  Extra bits are okay; they'll get
  // truncated out after the division by 2^T.
  //
  // In comparison to just directly using the first formula, this technique
  // is much more efficient; using the first formula requires W * K bits,
  // but this formula less than W + K bits. Also, the first formula requires
  // a division step, whereas this formula only requires multiplies and shifts.
  //
  // It doesn't matter whether the subtraction step is done in the calculation
  // width or the input iteration count's width; if the subtraction overflows,
  // the result must be zero anyway.  We prefer here to do it in the width of
  // the induction variable because it helps a lot for certain cases; CodeGen
  // isn't smart enough to ignore the overflow, which leads to much less
  // efficient code if the width of the subtraction is wider than the native
  // register width.
  //
  // (It's possible to not widen at all by pulling out factors of 2 before
  // the multiplication; for example, K=2 can be calculated as
  // It/2*(It+(It*INT_MIN/INT_MIN)+-1). However, it requires
  // extra arithmetic, so it's not an obvious win, and it gets
  // much more complicated for K > 3.)

  // Protection from insane SCEVs; this bound is conservative,
  // but it probably doesn't matter.
  if (K > 1000)
    return SE.getCouldNotCompute();

  unsigned W = SE.getTypeSizeInBits(ResultTy);

  // Calculate K! / 2^T and T; we divide out the factors of two before
  // multiplying for calculating K! / 2^T to avoid overflow.
  // Other overflow doesn't matter because we only care about the bottom
  // W bits of the result.
  APInt OddFactorial(W, 1);
  unsigned T = 1;
  for (unsigned i = 3; i <= K; ++i) {
    APInt Mult(W, i);
    unsigned TwoFactors = Mult.countTrailingZeros();
    T += TwoFactors;
    Mult = Mult.lshr(TwoFactors);
    OddFactorial *= Mult;
  }

  // We need at least W + T bits for the multiplication step
  unsigned CalculationBits = W + T;

  // Calculate 2^T, at width T+W.
  APInt DivFactor = APInt(CalculationBits, 1).shl(T);

  // Calculate the multiplicative inverse of K! / 2^T;
  // this multiplication factor will perform the exact division by
  // K! / 2^T.
  APInt Mod = APInt::getSignedMinValue(W+1);
  APInt MultiplyFactor = OddFactorial.zext(W+1);
  MultiplyFactor = MultiplyFactor.multiplicativeInverse(Mod);
  MultiplyFactor = MultiplyFactor.trunc(W);

  // Calculate the product, at width T+W
  IntegerType *CalculationTy = IntegerType::get(SE.getContext(),
                                                      CalculationBits);
  const SCEV *Dividend = SE.getTruncateOrZeroExtend(It, CalculationTy);
  for (unsigned i = 1; i != K; ++i) {
    const SCEV *S = SE.getMinusSCEV(It, SE.getConstant(It->getType(), i));
    Dividend = SE.getMulExpr(Dividend,
                             SE.getTruncateOrZeroExtend(S, CalculationTy));
  }

  // Divide by 2^T
  const SCEV *DivResult = SE.getUDivExpr(Dividend, SE.getConstant(DivFactor));

  // Truncate the result, and divide by K! / 2^T.

  return SE.getMulExpr(SE.getConstant(MultiplyFactor),
                       SE.getTruncateOrZeroExtend(DivResult, ResultTy));
}

/// evaluateAtIteration - Return the value of this chain of recurrences at
/// the specified iteration number.  We can evaluate this recurrence by
/// multiplying each element in the chain by the binomial coefficient
/// corresponding to it.  In other words, we can evaluate {A,+,B,+,C,+,D} as:
///
///   A*BC(It, 0) + B*BC(It, 1) + C*BC(It, 2) + D*BC(It, 3)
///
/// where BC(It, k) stands for binomial coefficient.
///
const SCEV *SCEVAddRecExpr::evaluateAtIteration(const SCEV *It,
                                                ScalarEvolution &SE) const {
  const SCEV *Result = getStart();
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    // The computation is correct in the face of overflow provided that the
    // multiplication is performed _after_ the evaluation of the binomial
    // coefficient.
    const SCEV *Coeff = BinomialCoefficient(It, i, SE, getType());
    if (isa<SCEVCouldNotCompute>(Coeff))
      return Coeff;

    Result = SE.getAddExpr(Result, SE.getMulExpr(getOperand(i), Coeff));
  }
  return Result;
}

//===----------------------------------------------------------------------===//
//                    SCEV Expression folder implementations
//===----------------------------------------------------------------------===//

const SCEV *ScalarEvolution::getTruncateExpr(const SCEV *Op,
                                             Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) > getTypeSizeInBits(Ty) &&
         "This is not a truncating conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  FoldingSetNodeID ID;
  ID.AddInteger(scTruncate);
  ID.AddPointer(Op);
  ID.AddPointer(Ty);
  void *IP = 0;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;

  // Fold if the operand is constant.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getConstant(
      cast<ConstantInt>(ConstantExpr::getTrunc(SC->getValue(),
                                               getEffectiveSCEVType(Ty))));

  // trunc(trunc(x)) --> trunc(x)
  if (const SCEVTruncateExpr *ST = dyn_cast<SCEVTruncateExpr>(Op))
    return getTruncateExpr(ST->getOperand(), Ty);

  // trunc(sext(x)) --> sext(x) if widening or trunc(x) if narrowing
  if (const SCEVSignExtendExpr *SS = dyn_cast<SCEVSignExtendExpr>(Op))
    return getTruncateOrSignExtend(SS->getOperand(), Ty);

  // trunc(zext(x)) --> zext(x) if widening or trunc(x) if narrowing
  if (const SCEVZeroExtendExpr *SZ = dyn_cast<SCEVZeroExtendExpr>(Op))
    return getTruncateOrZeroExtend(SZ->getOperand(), Ty);

  // trunc(x1+x2+...+xN) --> trunc(x1)+trunc(x2)+...+trunc(xN) if we can
  // eliminate all the truncates.
  if (const SCEVAddExpr *SA = dyn_cast<SCEVAddExpr>(Op)) {
    SmallVector<const SCEV *, 4> Operands;
    bool hasTrunc = false;
    for (unsigned i = 0, e = SA->getNumOperands(); i != e && !hasTrunc; ++i) {
      const SCEV *S = getTruncateExpr(SA->getOperand(i), Ty);
      hasTrunc = isa<SCEVTruncateExpr>(S);
      Operands.push_back(S);
    }
    if (!hasTrunc)
      return getAddExpr(Operands);
    UniqueSCEVs.FindNodeOrInsertPos(ID, IP);  // Mutates IP, returns NULL.
  }

  // trunc(x1*x2*...*xN) --> trunc(x1)*trunc(x2)*...*trunc(xN) if we can
  // eliminate all the truncates.
  if (const SCEVMulExpr *SM = dyn_cast<SCEVMulExpr>(Op)) {
    SmallVector<const SCEV *, 4> Operands;
    bool hasTrunc = false;
    for (unsigned i = 0, e = SM->getNumOperands(); i != e && !hasTrunc; ++i) {
      const SCEV *S = getTruncateExpr(SM->getOperand(i), Ty);
      hasTrunc = isa<SCEVTruncateExpr>(S);
      Operands.push_back(S);
    }
    if (!hasTrunc)
      return getMulExpr(Operands);
    UniqueSCEVs.FindNodeOrInsertPos(ID, IP);  // Mutates IP, returns NULL.
  }

  // If the input value is a chrec scev, truncate the chrec's operands.
  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Op)) {
    SmallVector<const SCEV *, 4> Operands;
    for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
      Operands.push_back(getTruncateExpr(AddRec->getOperand(i), Ty));
    return getAddRecExpr(Operands, AddRec->getLoop(), SCEV::FlagAnyWrap);
  }

  // As a special case, fold trunc(undef) to undef. We don't want to
  // know too much about SCEVUnknowns, but this special case is handy
  // and harmless.
  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(Op))
    if (isa<UndefValue>(U->getValue()))
      return getSCEV(UndefValue::get(Ty));

  // The cast wasn't folded; create an explicit cast node. We can reuse
  // the existing insert position since if we get here, we won't have
  // made any changes which would invalidate it.
  SCEV *S = new (SCEVAllocator) SCEVTruncateExpr(ID.Intern(SCEVAllocator),
                                                 Op, Ty);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getZeroExtendExpr(const SCEV *Op,
                                               Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) < getTypeSizeInBits(Ty) &&
         "This is not an extending conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  // Fold if the operand is constant.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getConstant(
      cast<ConstantInt>(ConstantExpr::getZExt(SC->getValue(),
                                              getEffectiveSCEVType(Ty))));

  // zext(zext(x)) --> zext(x)
  if (const SCEVZeroExtendExpr *SZ = dyn_cast<SCEVZeroExtendExpr>(Op))
    return getZeroExtendExpr(SZ->getOperand(), Ty);

  // Before doing any expensive analysis, check to see if we've already
  // computed a SCEV for this Op and Ty.
  FoldingSetNodeID ID;
  ID.AddInteger(scZeroExtend);
  ID.AddPointer(Op);
  ID.AddPointer(Ty);
  void *IP = 0;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;

  // zext(trunc(x)) --> zext(x) or x or trunc(x)
  if (const SCEVTruncateExpr *ST = dyn_cast<SCEVTruncateExpr>(Op)) {
    // It's possible the bits taken off by the truncate were all zero bits. If
    // so, we should be able to simplify this further.
    const SCEV *X = ST->getOperand();
    ConstantRange CR = getUnsignedRange(X);
    unsigned TruncBits = getTypeSizeInBits(ST->getType());
    unsigned NewBits = getTypeSizeInBits(Ty);
    if (CR.truncate(TruncBits).zeroExtend(NewBits).contains(
            CR.zextOrTrunc(NewBits)))
      return getTruncateOrZeroExtend(X, Ty);
  }

  // If the input value is a chrec scev, and we can prove that the value
  // did not overflow the old, smaller, value, we can zero extend all of the
  // operands (often constants).  This allows analysis of something like
  // this:  for (unsigned char X = 0; X < 100; ++X) { int Y = X; }
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Op))
    if (AR->isAffine()) {
      const SCEV *Start = AR->getStart();
      const SCEV *Step = AR->getStepRecurrence(*this);
      unsigned BitWidth = getTypeSizeInBits(AR->getType());
      const Loop *L = AR->getLoop();

      // If we have special knowledge that this addrec won't overflow,
      // we don't need to do any further analysis.
      if (AR->getNoWrapFlags(SCEV::FlagNUW))
        return getAddRecExpr(getZeroExtendExpr(Start, Ty),
                             getZeroExtendExpr(Step, Ty),
                             L, AR->getNoWrapFlags());

      // Check whether the backedge-taken count is SCEVCouldNotCompute.
      // Note that this serves two purposes: It filters out loops that are
      // simply not analyzable, and it covers the case where this code is
      // being called from within backedge-taken count analysis, such that
      // attempting to ask for the backedge-taken count would likely result
      // in infinite recursion. In the later case, the analysis code will
      // cope with a conservative value, and it will take care to purge
      // that value once it has finished.
      const SCEV *MaxBECount = getMaxBackedgeTakenCount(L);
      if (!isa<SCEVCouldNotCompute>(MaxBECount)) {
        // Manually compute the final value for AR, checking for
        // overflow.

        // Check whether the backedge-taken count can be losslessly casted to
        // the addrec's type. The count is always unsigned.
        const SCEV *CastedMaxBECount =
          getTruncateOrZeroExtend(MaxBECount, Start->getType());
        const SCEV *RecastedMaxBECount =
          getTruncateOrZeroExtend(CastedMaxBECount, MaxBECount->getType());
        if (MaxBECount == RecastedMaxBECount) {
          Type *WideTy = IntegerType::get(getContext(), BitWidth * 2);
          // Check whether Start+Step*MaxBECount has no unsigned overflow.
          const SCEV *ZMul = getMulExpr(CastedMaxBECount, Step);
          const SCEV *Add = getAddExpr(Start, ZMul);
          const SCEV *OperandExtendedAdd =
            getAddExpr(getZeroExtendExpr(Start, WideTy),
                       getMulExpr(getZeroExtendExpr(CastedMaxBECount, WideTy),
                                  getZeroExtendExpr(Step, WideTy)));
          if (getZeroExtendExpr(Add, WideTy) == OperandExtendedAdd) {
            // Cache knowledge of AR NUW, which is propagated to this AddRec.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNUW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(getZeroExtendExpr(Start, Ty),
                                 getZeroExtendExpr(Step, Ty),
                                 L, AR->getNoWrapFlags());
          }
          // Similar to above, only this time treat the step value as signed.
          // This covers loops that count down.
          const SCEV *SMul = getMulExpr(CastedMaxBECount, Step);
          Add = getAddExpr(Start, SMul);
          OperandExtendedAdd =
            getAddExpr(getZeroExtendExpr(Start, WideTy),
                       getMulExpr(getZeroExtendExpr(CastedMaxBECount, WideTy),
                                  getSignExtendExpr(Step, WideTy)));
          if (getZeroExtendExpr(Add, WideTy) == OperandExtendedAdd) {
            // Cache knowledge of AR NW, which is propagated to this AddRec.
            // Negative step causes unsigned wrap, but it still can't self-wrap.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(getZeroExtendExpr(Start, Ty),
                                 getSignExtendExpr(Step, Ty),
                                 L, AR->getNoWrapFlags());
          }
        }

        // If the backedge is guarded by a comparison with the pre-inc value
        // the addrec is safe. Also, if the entry is guarded by a comparison
        // with the start value and the backedge is guarded by a comparison
        // with the post-inc value, the addrec is safe.
        if (isKnownPositive(Step)) {
          const SCEV *N = getConstant(APInt::getMinValue(BitWidth) -
                                      getUnsignedRange(Step).getUnsignedMax());
          if (isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT, AR, N) ||
              (isLoopEntryGuardedByCond(L, ICmpInst::ICMP_ULT, Start, N) &&
               isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT,
                                           AR->getPostIncExpr(*this), N))) {
            // Cache knowledge of AR NUW, which is propagated to this AddRec.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNUW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(getZeroExtendExpr(Start, Ty),
                                 getZeroExtendExpr(Step, Ty),
                                 L, AR->getNoWrapFlags());
          }
        } else if (isKnownNegative(Step)) {
          const SCEV *N = getConstant(APInt::getMaxValue(BitWidth) -
                                      getSignedRange(Step).getSignedMin());
          if (isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_UGT, AR, N) ||
              (isLoopEntryGuardedByCond(L, ICmpInst::ICMP_UGT, Start, N) &&
               isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_UGT,
                                           AR->getPostIncExpr(*this), N))) {
            // Cache knowledge of AR NW, which is propagated to this AddRec.
            // Negative step causes unsigned wrap, but it still can't self-wrap.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(getZeroExtendExpr(Start, Ty),
                                 getSignExtendExpr(Step, Ty),
                                 L, AR->getNoWrapFlags());
          }
        }
      }
    }

  // The cast wasn't folded; create an explicit cast node.
  // Recompute the insert position, as it may have been invalidated.
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVZeroExtendExpr(ID.Intern(SCEVAllocator),
                                                   Op, Ty);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

// Get the limit of a recurrence such that incrementing by Step cannot cause
// signed overflow as long as the value of the recurrence within the loop does
// not exceed this limit before incrementing.
static const SCEV *getOverflowLimitForStep(const SCEV *Step,
                                           ICmpInst::Predicate *Pred,
                                           ScalarEvolution *SE) {
  unsigned BitWidth = SE->getTypeSizeInBits(Step->getType());
  if (SE->isKnownPositive(Step)) {
    *Pred = ICmpInst::ICMP_SLT;
    return SE->getConstant(APInt::getSignedMinValue(BitWidth) -
                           SE->getSignedRange(Step).getSignedMax());
  }
  if (SE->isKnownNegative(Step)) {
    *Pred = ICmpInst::ICMP_SGT;
    return SE->getConstant(APInt::getSignedMaxValue(BitWidth) -
                       SE->getSignedRange(Step).getSignedMin());
  }
  return 0;
}

// The recurrence AR has been shown to have no signed wrap. Typically, if we can
// prove NSW for AR, then we can just as easily prove NSW for its preincrement
// or postincrement sibling. This allows normalizing a sign extended AddRec as
// such: {sext(Step + Start),+,Step} => {(Step + sext(Start),+,Step} As a
// result, the expression "Step + sext(PreIncAR)" is congruent with
// "sext(PostIncAR)"
static const SCEV *getPreStartForSignExtend(const SCEVAddRecExpr *AR,
                                            Type *Ty,
                                            ScalarEvolution *SE) {
  const Loop *L = AR->getLoop();
  const SCEV *Start = AR->getStart();
  const SCEV *Step = AR->getStepRecurrence(*SE);

  // Check for a simple looking step prior to loop entry.
  const SCEVAddExpr *SA = dyn_cast<SCEVAddExpr>(Start);
  if (!SA || SA->getNumOperands() != 2 || SA->getOperand(0) != Step)
    return 0;

  // This is a postinc AR. Check for overflow on the preinc recurrence using the
  // same three conditions that getSignExtendedExpr checks.

  // 1. NSW flags on the step increment.
  const SCEV *PreStart = SA->getOperand(1);
  const SCEVAddRecExpr *PreAR = dyn_cast<SCEVAddRecExpr>(
    SE->getAddRecExpr(PreStart, Step, L, SCEV::FlagAnyWrap));

  if (PreAR && PreAR->getNoWrapFlags(SCEV::FlagNSW))
    return PreStart;

  // 2. Direct overflow check on the step operation's expression.
  unsigned BitWidth = SE->getTypeSizeInBits(AR->getType());
  Type *WideTy = IntegerType::get(SE->getContext(), BitWidth * 2);
  const SCEV *OperandExtendedStart =
    SE->getAddExpr(SE->getSignExtendExpr(PreStart, WideTy),
                   SE->getSignExtendExpr(Step, WideTy));
  if (SE->getSignExtendExpr(Start, WideTy) == OperandExtendedStart) {
    // Cache knowledge of PreAR NSW.
    if (PreAR)
      const_cast<SCEVAddRecExpr *>(PreAR)->setNoWrapFlags(SCEV::FlagNSW);
    // FIXME: this optimization needs a unit test
    DEBUG(dbgs() << "SCEV: untested prestart overflow check\n");
    return PreStart;
  }

  // 3. Loop precondition.
  ICmpInst::Predicate Pred;
  const SCEV *OverflowLimit = getOverflowLimitForStep(Step, &Pred, SE);

  if (OverflowLimit &&
      SE->isLoopEntryGuardedByCond(L, Pred, PreStart, OverflowLimit)) {
    return PreStart;
  }
  return 0;
}

// Get the normalized sign-extended expression for this AddRec's Start.
static const SCEV *getSignExtendAddRecStart(const SCEVAddRecExpr *AR,
                                            Type *Ty,
                                            ScalarEvolution *SE) {
  const SCEV *PreStart = getPreStartForSignExtend(AR, Ty, SE);
  if (!PreStart)
    return SE->getSignExtendExpr(AR->getStart(), Ty);

  return SE->getAddExpr(SE->getSignExtendExpr(AR->getStepRecurrence(*SE), Ty),
                        SE->getSignExtendExpr(PreStart, Ty));
}

const SCEV *ScalarEvolution::getSignExtendExpr(const SCEV *Op,
                                               Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) < getTypeSizeInBits(Ty) &&
         "This is not an extending conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  // Fold if the operand is constant.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    return getConstant(
      cast<ConstantInt>(ConstantExpr::getSExt(SC->getValue(),
                                              getEffectiveSCEVType(Ty))));

  // sext(sext(x)) --> sext(x)
  if (const SCEVSignExtendExpr *SS = dyn_cast<SCEVSignExtendExpr>(Op))
    return getSignExtendExpr(SS->getOperand(), Ty);

  // sext(zext(x)) --> zext(x)
  if (const SCEVZeroExtendExpr *SZ = dyn_cast<SCEVZeroExtendExpr>(Op))
    return getZeroExtendExpr(SZ->getOperand(), Ty);

  // Before doing any expensive analysis, check to see if we've already
  // computed a SCEV for this Op and Ty.
  FoldingSetNodeID ID;
  ID.AddInteger(scSignExtend);
  ID.AddPointer(Op);
  ID.AddPointer(Ty);
  void *IP = 0;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;

  // If the input value is provably positive, build a zext instead.
  if (isKnownNonNegative(Op))
    return getZeroExtendExpr(Op, Ty);

  // sext(trunc(x)) --> sext(x) or x or trunc(x)
  if (const SCEVTruncateExpr *ST = dyn_cast<SCEVTruncateExpr>(Op)) {
    // It's possible the bits taken off by the truncate were all sign bits. If
    // so, we should be able to simplify this further.
    const SCEV *X = ST->getOperand();
    ConstantRange CR = getSignedRange(X);
    unsigned TruncBits = getTypeSizeInBits(ST->getType());
    unsigned NewBits = getTypeSizeInBits(Ty);
    if (CR.truncate(TruncBits).signExtend(NewBits).contains(
            CR.sextOrTrunc(NewBits)))
      return getTruncateOrSignExtend(X, Ty);
  }

  // If the input value is a chrec scev, and we can prove that the value
  // did not overflow the old, smaller, value, we can sign extend all of the
  // operands (often constants).  This allows analysis of something like
  // this:  for (signed char X = 0; X < 100; ++X) { int Y = X; }
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Op))
    if (AR->isAffine()) {
      const SCEV *Start = AR->getStart();
      const SCEV *Step = AR->getStepRecurrence(*this);
      unsigned BitWidth = getTypeSizeInBits(AR->getType());
      const Loop *L = AR->getLoop();

      // If we have special knowledge that this addrec won't overflow,
      // we don't need to do any further analysis.
      if (AR->getNoWrapFlags(SCEV::FlagNSW))
        return getAddRecExpr(getSignExtendAddRecStart(AR, Ty, this),
                             getSignExtendExpr(Step, Ty),
                             L, SCEV::FlagNSW);

      // Check whether the backedge-taken count is SCEVCouldNotCompute.
      // Note that this serves two purposes: It filters out loops that are
      // simply not analyzable, and it covers the case where this code is
      // being called from within backedge-taken count analysis, such that
      // attempting to ask for the backedge-taken count would likely result
      // in infinite recursion. In the later case, the analysis code will
      // cope with a conservative value, and it will take care to purge
      // that value once it has finished.
      const SCEV *MaxBECount = getMaxBackedgeTakenCount(L);
      if (!isa<SCEVCouldNotCompute>(MaxBECount)) {
        // Manually compute the final value for AR, checking for
        // overflow.

        // Check whether the backedge-taken count can be losslessly casted to
        // the addrec's type. The count is always unsigned.
        const SCEV *CastedMaxBECount =
          getTruncateOrZeroExtend(MaxBECount, Start->getType());
        const SCEV *RecastedMaxBECount =
          getTruncateOrZeroExtend(CastedMaxBECount, MaxBECount->getType());
        if (MaxBECount == RecastedMaxBECount) {
          Type *WideTy = IntegerType::get(getContext(), BitWidth * 2);
          // Check whether Start+Step*MaxBECount has no signed overflow.
          const SCEV *SMul = getMulExpr(CastedMaxBECount, Step);
          const SCEV *Add = getAddExpr(Start, SMul);
          const SCEV *OperandExtendedAdd =
            getAddExpr(getSignExtendExpr(Start, WideTy),
                       getMulExpr(getZeroExtendExpr(CastedMaxBECount, WideTy),
                                  getSignExtendExpr(Step, WideTy)));
          if (getSignExtendExpr(Add, WideTy) == OperandExtendedAdd) {
            // Cache knowledge of AR NSW, which is propagated to this AddRec.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNSW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(getSignExtendAddRecStart(AR, Ty, this),
                                 getSignExtendExpr(Step, Ty),
                                 L, AR->getNoWrapFlags());
          }
          // Similar to above, only this time treat the step value as unsigned.
          // This covers loops that count up with an unsigned step.
          const SCEV *UMul = getMulExpr(CastedMaxBECount, Step);
          Add = getAddExpr(Start, UMul);
          OperandExtendedAdd =
            getAddExpr(getSignExtendExpr(Start, WideTy),
                       getMulExpr(getZeroExtendExpr(CastedMaxBECount, WideTy),
                                  getZeroExtendExpr(Step, WideTy)));
          if (getSignExtendExpr(Add, WideTy) == OperandExtendedAdd) {
            // Cache knowledge of AR NSW, which is propagated to this AddRec.
            const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNSW);
            // Return the expression with the addrec on the outside.
            return getAddRecExpr(getSignExtendAddRecStart(AR, Ty, this),
                                 getZeroExtendExpr(Step, Ty),
                                 L, AR->getNoWrapFlags());
          }
        }

        // If the backedge is guarded by a comparison with the pre-inc value
        // the addrec is safe. Also, if the entry is guarded by a comparison
        // with the start value and the backedge is guarded by a comparison
        // with the post-inc value, the addrec is safe.
        ICmpInst::Predicate Pred;
        const SCEV *OverflowLimit = getOverflowLimitForStep(Step, &Pred, this);
        if (OverflowLimit &&
            (isLoopBackedgeGuardedByCond(L, Pred, AR, OverflowLimit) ||
             (isLoopEntryGuardedByCond(L, Pred, Start, OverflowLimit) &&
              isLoopBackedgeGuardedByCond(L, Pred, AR->getPostIncExpr(*this),
                                          OverflowLimit)))) {
          // Cache knowledge of AR NSW, then propagate NSW to the wide AddRec.
          const_cast<SCEVAddRecExpr *>(AR)->setNoWrapFlags(SCEV::FlagNSW);
          return getAddRecExpr(getSignExtendAddRecStart(AR, Ty, this),
                               getSignExtendExpr(Step, Ty),
                               L, AR->getNoWrapFlags());
        }
      }
    }

  // The cast wasn't folded; create an explicit cast node.
  // Recompute the insert position, as it may have been invalidated.
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVSignExtendExpr(ID.Intern(SCEVAllocator),
                                                   Op, Ty);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

/// getAnyExtendExpr - Return a SCEV for the given operand extended with
/// unspecified bits out to the given type.
///
const SCEV *ScalarEvolution::getAnyExtendExpr(const SCEV *Op,
                                              Type *Ty) {
  assert(getTypeSizeInBits(Op->getType()) < getTypeSizeInBits(Ty) &&
         "This is not an extending conversion!");
  assert(isSCEVable(Ty) &&
         "This is not a conversion to a SCEVable type!");
  Ty = getEffectiveSCEVType(Ty);

  // Sign-extend negative constants.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Op))
    if (SC->getValue()->getValue().isNegative())
      return getSignExtendExpr(Op, Ty);

  // Peel off a truncate cast.
  if (const SCEVTruncateExpr *T = dyn_cast<SCEVTruncateExpr>(Op)) {
    const SCEV *NewOp = T->getOperand();
    if (getTypeSizeInBits(NewOp->getType()) < getTypeSizeInBits(Ty))
      return getAnyExtendExpr(NewOp, Ty);
    return getTruncateOrNoop(NewOp, Ty);
  }

  // Next try a zext cast. If the cast is folded, use it.
  const SCEV *ZExt = getZeroExtendExpr(Op, Ty);
  if (!isa<SCEVZeroExtendExpr>(ZExt))
    return ZExt;

  // Next try a sext cast. If the cast is folded, use it.
  const SCEV *SExt = getSignExtendExpr(Op, Ty);
  if (!isa<SCEVSignExtendExpr>(SExt))
    return SExt;

  // Force the cast to be folded into the operands of an addrec.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Op)) {
    SmallVector<const SCEV *, 4> Ops;
    for (SCEVAddRecExpr::op_iterator I = AR->op_begin(), E = AR->op_end();
         I != E; ++I)
      Ops.push_back(getAnyExtendExpr(*I, Ty));
    return getAddRecExpr(Ops, AR->getLoop(), SCEV::FlagNW);
  }

  // As a special case, fold anyext(undef) to undef. We don't want to
  // know too much about SCEVUnknowns, but this special case is handy
  // and harmless.
  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(Op))
    if (isa<UndefValue>(U->getValue()))
      return getSCEV(UndefValue::get(Ty));

  // If the expression is obviously signed, use the sext cast value.
  if (isa<SCEVSMaxExpr>(Op))
    return SExt;

  // Absent any other information, use the zext cast value.
  return ZExt;
}

/// CollectAddOperandsWithScales - Process the given Ops list, which is
/// a list of operands to be added under the given scale, update the given
/// map. This is a helper function for getAddRecExpr. As an example of
/// what it does, given a sequence of operands that would form an add
/// expression like this:
///
///    m + n + 13 + (A * (o + p + (B * q + m + 29))) + r + (-1 * r)
///
/// where A and B are constants, update the map with these values:
///
///    (m, 1+A*B), (n, 1), (o, A), (p, A), (q, A*B), (r, 0)
///
/// and add 13 + A*B*29 to AccumulatedConstant.
/// This will allow getAddRecExpr to produce this:
///
///    13+A*B*29 + n + (m * (1+A*B)) + ((o + p) * A) + (q * A*B)
///
/// This form often exposes folding opportunities that are hidden in
/// the original operand list.
///
/// Return true iff it appears that any interesting folding opportunities
/// may be exposed. This helps getAddRecExpr short-circuit extra work in
/// the common case where no interesting opportunities are present, and
/// is also used as a check to avoid infinite recursion.
///
static bool
CollectAddOperandsWithScales(DenseMap<const SCEV *, APInt> &M,
                             SmallVector<const SCEV *, 8> &NewOps,
                             APInt &AccumulatedConstant,
                             const SCEV *const *Ops, size_t NumOperands,
                             const APInt &Scale,
                             ScalarEvolution &SE) {
  bool Interesting = false;

  // Iterate over the add operands. They are sorted, with constants first.
  unsigned i = 0;
  while (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[i])) {
    ++i;
    // Pull a buried constant out to the outside.
    if (Scale != 1 || AccumulatedConstant != 0 || C->getValue()->isZero())
      Interesting = true;
    AccumulatedConstant += Scale * C->getValue()->getValue();
  }

  // Next comes everything else. We're especially interested in multiplies
  // here, but they're in the middle, so just visit the rest with one loop.
  for (; i != NumOperands; ++i) {
    const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Ops[i]);
    if (Mul && isa<SCEVConstant>(Mul->getOperand(0))) {
      APInt NewScale =
        Scale * cast<SCEVConstant>(Mul->getOperand(0))->getValue()->getValue();
      if (Mul->getNumOperands() == 2 && isa<SCEVAddExpr>(Mul->getOperand(1))) {
        // A multiplication of a constant with another add; recurse.
        const SCEVAddExpr *Add = cast<SCEVAddExpr>(Mul->getOperand(1));
        Interesting |=
          CollectAddOperandsWithScales(M, NewOps, AccumulatedConstant,
                                       Add->op_begin(), Add->getNumOperands(),
                                       NewScale, SE);
      } else {
        // A multiplication of a constant with some other value. Update
        // the map.
        SmallVector<const SCEV *, 4> MulOps(Mul->op_begin()+1, Mul->op_end());
        const SCEV *Key = SE.getMulExpr(MulOps);
        std::pair<DenseMap<const SCEV *, APInt>::iterator, bool> Pair =
          M.insert(std::make_pair(Key, NewScale));
        if (Pair.second) {
          NewOps.push_back(Pair.first->first);
        } else {
          Pair.first->second += NewScale;
          // The map already had an entry for this value, which may indicate
          // a folding opportunity.
          Interesting = true;
        }
      }
    } else {
      // An ordinary operand. Update the map.
      std::pair<DenseMap<const SCEV *, APInt>::iterator, bool> Pair =
        M.insert(std::make_pair(Ops[i], Scale));
      if (Pair.second) {
        NewOps.push_back(Pair.first->first);
      } else {
        Pair.first->second += Scale;
        // The map already had an entry for this value, which may indicate
        // a folding opportunity.
        Interesting = true;
      }
    }
  }

  return Interesting;
}

namespace {
  struct APIntCompare {
    bool operator()(const APInt &LHS, const APInt &RHS) const {
      return LHS.ult(RHS);
    }
  };
}

/// getAddExpr - Get a canonical add expression, or something simpler if
/// possible.
const SCEV *ScalarEvolution::getAddExpr(SmallVectorImpl<const SCEV *> &Ops,
                                        SCEV::NoWrapFlags Flags) {
  assert(!(Flags & ~(SCEV::FlagNUW | SCEV::FlagNSW)) &&
         "only nuw or nsw allowed");
  assert(!Ops.empty() && "Cannot get empty add!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVAddExpr operand types don't match!");
#endif

  // If FlagNSW is true and all the operands are non-negative, infer FlagNUW.
  // And vice-versa.
  int SignOrUnsignMask = SCEV::FlagNUW | SCEV::FlagNSW;
  SCEV::NoWrapFlags SignOrUnsignWrap = maskFlags(Flags, SignOrUnsignMask);
  if (SignOrUnsignWrap && (SignOrUnsignWrap != SignOrUnsignMask)) {
    bool All = true;
    for (SmallVectorImpl<const SCEV *>::const_iterator I = Ops.begin(),
         E = Ops.end(); I != E; ++I)
      if (!isKnownNonNegative(*I)) {
        All = false;
        break;
      }
    if (All) Flags = setFlags(Flags, (SCEV::NoWrapFlags)SignOrUnsignMask);
  }

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      Ops[0] = getConstant(LHSC->getValue()->getValue() +
                           RHSC->getValue()->getValue());
      if (Ops.size() == 2) return Ops[0];
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant zero being added, strip it off.
    if (LHSC->getValue()->isZero()) {
      Ops.erase(Ops.begin());
      --Idx;
    }

    if (Ops.size() == 1) return Ops[0];
  }

  // Okay, check to see if the same value occurs in the operand list more than
  // once.  If so, merge them together into an multiply expression.  Since we
  // sorted the list, these values are required to be adjacent.
  Type *Ty = Ops[0]->getType();
  bool FoundMatch = false;
  for (unsigned i = 0, e = Ops.size(); i != e-1; ++i)
    if (Ops[i] == Ops[i+1]) {      //  X + Y + Y  -->  X + Y*2
      // Scan ahead to count how many equal operands there are.
      unsigned Count = 2;
      while (i+Count != e && Ops[i+Count] == Ops[i])
        ++Count;
      // Merge the values into a multiply.
      const SCEV *Scale = getConstant(Ty, Count);
      const SCEV *Mul = getMulExpr(Scale, Ops[i]);
      if (Ops.size() == Count)
        return Mul;
      Ops[i] = Mul;
      Ops.erase(Ops.begin()+i+1, Ops.begin()+i+Count);
      --i; e -= Count - 1;
      FoundMatch = true;
    }
  if (FoundMatch)
    return getAddExpr(Ops, Flags);

  // Check for truncates. If all the operands are truncated from the same
  // type, see if factoring out the truncate would permit the result to be
  // folded. eg., trunc(x) + m*trunc(n) --> trunc(x + trunc(m)*n)
  // if the contents of the resulting outer trunc fold to something simple.
  for (; Idx < Ops.size() && isa<SCEVTruncateExpr>(Ops[Idx]); ++Idx) {
    const SCEVTruncateExpr *Trunc = cast<SCEVTruncateExpr>(Ops[Idx]);
    Type *DstType = Trunc->getType();
    Type *SrcType = Trunc->getOperand()->getType();
    SmallVector<const SCEV *, 8> LargeOps;
    bool Ok = true;
    // Check all the operands to see if they can be represented in the
    // source type of the truncate.
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      if (const SCEVTruncateExpr *T = dyn_cast<SCEVTruncateExpr>(Ops[i])) {
        if (T->getOperand()->getType() != SrcType) {
          Ok = false;
          break;
        }
        LargeOps.push_back(T->getOperand());
      } else if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Ops[i])) {
        LargeOps.push_back(getAnyExtendExpr(C, SrcType));
      } else if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(Ops[i])) {
        SmallVector<const SCEV *, 8> LargeMulOps;
        for (unsigned j = 0, f = M->getNumOperands(); j != f && Ok; ++j) {
          if (const SCEVTruncateExpr *T =
                dyn_cast<SCEVTruncateExpr>(M->getOperand(j))) {
            if (T->getOperand()->getType() != SrcType) {
              Ok = false;
              break;
            }
            LargeMulOps.push_back(T->getOperand());
          } else if (const SCEVConstant *C =
                       dyn_cast<SCEVConstant>(M->getOperand(j))) {
            LargeMulOps.push_back(getAnyExtendExpr(C, SrcType));
          } else {
            Ok = false;
            break;
          }
        }
        if (Ok)
          LargeOps.push_back(getMulExpr(LargeMulOps));
      } else {
        Ok = false;
        break;
      }
    }
    if (Ok) {
      // Evaluate the expression in the larger type.
      const SCEV *Fold = getAddExpr(LargeOps, Flags);
      // If it folds to something simple, use it. Otherwise, don't.
      if (isa<SCEVConstant>(Fold) || isa<SCEVUnknown>(Fold))
        return getTruncateExpr(Fold, DstType);
    }
  }

  // Skip past any other cast SCEVs.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddExpr)
    ++Idx;

  // If there are add operands they would be next.
  if (Idx < Ops.size()) {
    bool DeletedAdd = false;
    while (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[Idx])) {
      // If we have an add, expand the add operands onto the end of the operands
      // list.
      Ops.erase(Ops.begin()+Idx);
      Ops.append(Add->op_begin(), Add->op_end());
      DeletedAdd = true;
    }

    // If we deleted at least one add, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just acquired.
    if (DeletedAdd)
      return getAddExpr(Ops);
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  // Check to see if there are any folding opportunities present with
  // operands multiplied by constant values.
  if (Idx < Ops.size() && isa<SCEVMulExpr>(Ops[Idx])) {
    uint64_t BitWidth = getTypeSizeInBits(Ty);
    DenseMap<const SCEV *, APInt> M;
    SmallVector<const SCEV *, 8> NewOps;
    APInt AccumulatedConstant(BitWidth, 0);
    if (CollectAddOperandsWithScales(M, NewOps, AccumulatedConstant,
                                     Ops.data(), Ops.size(),
                                     APInt(BitWidth, 1), *this)) {
      // Some interesting folding opportunity is present, so its worthwhile to
      // re-generate the operands list. Group the operands by constant scale,
      // to avoid multiplying by the same constant scale multiple times.
      std::map<APInt, SmallVector<const SCEV *, 4>, APIntCompare> MulOpLists;
      for (SmallVector<const SCEV *, 8>::const_iterator I = NewOps.begin(),
           E = NewOps.end(); I != E; ++I)
        MulOpLists[M.find(*I)->second].push_back(*I);
      // Re-generate the operands list.
      Ops.clear();
      if (AccumulatedConstant != 0)
        Ops.push_back(getConstant(AccumulatedConstant));
      for (std::map<APInt, SmallVector<const SCEV *, 4>, APIntCompare>::iterator
           I = MulOpLists.begin(), E = MulOpLists.end(); I != E; ++I)
        if (I->first != 0)
          Ops.push_back(getMulExpr(getConstant(I->first),
                                   getAddExpr(I->second)));
      if (Ops.empty())
        return getConstant(Ty, 0);
      if (Ops.size() == 1)
        return Ops[0];
      return getAddExpr(Ops);
    }
  }

  // If we are adding something to a multiply expression, make sure the
  // something is not already an operand of the multiply.  If so, merge it into
  // the multiply.
  for (; Idx < Ops.size() && isa<SCEVMulExpr>(Ops[Idx]); ++Idx) {
    const SCEVMulExpr *Mul = cast<SCEVMulExpr>(Ops[Idx]);
    for (unsigned MulOp = 0, e = Mul->getNumOperands(); MulOp != e; ++MulOp) {
      const SCEV *MulOpSCEV = Mul->getOperand(MulOp);
      if (isa<SCEVConstant>(MulOpSCEV))
        continue;
      for (unsigned AddOp = 0, e = Ops.size(); AddOp != e; ++AddOp)
        if (MulOpSCEV == Ops[AddOp]) {
          // Fold W + X + (X * Y * Z)  -->  W + (X * ((Y*Z)+1))
          const SCEV *InnerMul = Mul->getOperand(MulOp == 0);
          if (Mul->getNumOperands() != 2) {
            // If the multiply has more than two operands, we must get the
            // Y*Z term.
            SmallVector<const SCEV *, 4> MulOps(Mul->op_begin(),
                                                Mul->op_begin()+MulOp);
            MulOps.append(Mul->op_begin()+MulOp+1, Mul->op_end());
            InnerMul = getMulExpr(MulOps);
          }
          const SCEV *One = getConstant(Ty, 1);
          const SCEV *AddOne = getAddExpr(One, InnerMul);
          const SCEV *OuterMul = getMulExpr(AddOne, MulOpSCEV);
          if (Ops.size() == 2) return OuterMul;
          if (AddOp < Idx) {
            Ops.erase(Ops.begin()+AddOp);
            Ops.erase(Ops.begin()+Idx-1);
          } else {
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+AddOp-1);
          }
          Ops.push_back(OuterMul);
          return getAddExpr(Ops);
        }

      // Check this multiply against other multiplies being added together.
      for (unsigned OtherMulIdx = Idx+1;
           OtherMulIdx < Ops.size() && isa<SCEVMulExpr>(Ops[OtherMulIdx]);
           ++OtherMulIdx) {
        const SCEVMulExpr *OtherMul = cast<SCEVMulExpr>(Ops[OtherMulIdx]);
        // If MulOp occurs in OtherMul, we can fold the two multiplies
        // together.
        for (unsigned OMulOp = 0, e = OtherMul->getNumOperands();
             OMulOp != e; ++OMulOp)
          if (OtherMul->getOperand(OMulOp) == MulOpSCEV) {
            // Fold X + (A*B*C) + (A*D*E) --> X + (A*(B*C+D*E))
            const SCEV *InnerMul1 = Mul->getOperand(MulOp == 0);
            if (Mul->getNumOperands() != 2) {
              SmallVector<const SCEV *, 4> MulOps(Mul->op_begin(),
                                                  Mul->op_begin()+MulOp);
              MulOps.append(Mul->op_begin()+MulOp+1, Mul->op_end());
              InnerMul1 = getMulExpr(MulOps);
            }
            const SCEV *InnerMul2 = OtherMul->getOperand(OMulOp == 0);
            if (OtherMul->getNumOperands() != 2) {
              SmallVector<const SCEV *, 4> MulOps(OtherMul->op_begin(),
                                                  OtherMul->op_begin()+OMulOp);
              MulOps.append(OtherMul->op_begin()+OMulOp+1, OtherMul->op_end());
              InnerMul2 = getMulExpr(MulOps);
            }
            const SCEV *InnerMulSum = getAddExpr(InnerMul1,InnerMul2);
            const SCEV *OuterMul = getMulExpr(MulOpSCEV, InnerMulSum);
            if (Ops.size() == 2) return OuterMul;
            Ops.erase(Ops.begin()+Idx);
            Ops.erase(Ops.begin()+OtherMulIdx-1);
            Ops.push_back(OuterMul);
            return getAddExpr(Ops);
          }
      }
    }
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this add and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    SmallVector<const SCEV *, 8> LIOps;
    const SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    const Loop *AddRecLoop = AddRec->getLoop();
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (isLoopInvariant(Ops[i], AddRecLoop)) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI + LI + {Start,+,Step}  -->  NLI + {LI+Start,+,Step}
      LIOps.push_back(AddRec->getStart());

      SmallVector<const SCEV *, 4> AddRecOps(AddRec->op_begin(),
                                             AddRec->op_end());
      AddRecOps[0] = getAddExpr(LIOps);

      // Build the new addrec. Propagate the NUW and NSW flags if both the
      // outer add and the inner addrec are guaranteed to have no overflow.
      // Always propagate NW.
      Flags = AddRec->getNoWrapFlags(setFlags(Flags, SCEV::FlagNW));
      const SCEV *NewRec = getAddRecExpr(AddRecOps, AddRecLoop, Flags);

      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, add the folded AddRec by the non-invariant parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return getAddExpr(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // added together.  If so, we can fold them.
    for (unsigned OtherIdx = Idx+1;
         OtherIdx < Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);
         ++OtherIdx)
      if (AddRecLoop == cast<SCEVAddRecExpr>(Ops[OtherIdx])->getLoop()) {
        // Other + {A,+,B}<L> + {C,+,D}<L>  -->  Other + {A+C,+,B+D}<L>
        SmallVector<const SCEV *, 4> AddRecOps(AddRec->op_begin(),
                                               AddRec->op_end());
        for (; OtherIdx != Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);
             ++OtherIdx)
          if (const SCEVAddRecExpr *OtherAddRec =
                dyn_cast<SCEVAddRecExpr>(Ops[OtherIdx]))
            if (OtherAddRec->getLoop() == AddRecLoop) {
              for (unsigned i = 0, e = OtherAddRec->getNumOperands();
                   i != e; ++i) {
                if (i >= AddRecOps.size()) {
                  AddRecOps.append(OtherAddRec->op_begin()+i,
                                   OtherAddRec->op_end());
                  break;
                }
                AddRecOps[i] = getAddExpr(AddRecOps[i],
                                          OtherAddRec->getOperand(i));
              }
              Ops.erase(Ops.begin() + OtherIdx); --OtherIdx;
            }
        // Step size has changed, so we cannot guarantee no self-wraparound.
        Ops[Idx] = getAddRecExpr(AddRecOps, AddRecLoop, SCEV::FlagAnyWrap);
        return getAddExpr(Ops);
      }

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an add expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scAddExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = 0;
  SCEVAddExpr *S =
    static_cast<SCEVAddExpr *>(UniqueSCEVs.FindNodeOrInsertPos(ID, IP));
  if (!S) {
    const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
    std::uninitialized_copy(Ops.begin(), Ops.end(), O);
    S = new (SCEVAllocator) SCEVAddExpr(ID.Intern(SCEVAllocator),
                                        O, Ops.size());
    UniqueSCEVs.InsertNode(S, IP);
  }
  S->setNoWrapFlags(Flags);
  return S;
}

/// getMulExpr - Get a canonical multiply expression, or something simpler if
/// possible.
const SCEV *ScalarEvolution::getMulExpr(SmallVectorImpl<const SCEV *> &Ops,
                                        SCEV::NoWrapFlags Flags) {
  assert(Flags == maskFlags(Flags, SCEV::FlagNUW | SCEV::FlagNSW) &&
         "only nuw or nsw allowed");
  assert(!Ops.empty() && "Cannot get empty mul!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVMulExpr operand types don't match!");
#endif

  // If FlagNSW is true and all the operands are non-negative, infer FlagNUW.
  // And vice-versa.
  int SignOrUnsignMask = SCEV::FlagNUW | SCEV::FlagNSW;
  SCEV::NoWrapFlags SignOrUnsignWrap = maskFlags(Flags, SignOrUnsignMask);
  if (SignOrUnsignWrap && (SignOrUnsignWrap != SignOrUnsignMask)) {
    bool All = true;
    for (SmallVectorImpl<const SCEV *>::const_iterator I = Ops.begin(),
         E = Ops.end(); I != E; ++I)
      if (!isKnownNonNegative(*I)) {
        All = false;
        break;
      }
    if (All) Flags = setFlags(Flags, (SCEV::NoWrapFlags)SignOrUnsignMask);
  }

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {

    // C1*(C2+V) -> C1*C2 + C1*V
    if (Ops.size() == 2)
      if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[1]))
        if (Add->getNumOperands() == 2 &&
            isa<SCEVConstant>(Add->getOperand(0)))
          return getAddExpr(getMulExpr(LHSC, Add->getOperand(0)),
                            getMulExpr(LHSC, Add->getOperand(1)));

    ++Idx;
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(getContext(),
                                           LHSC->getValue()->getValue() *
                                           RHSC->getValue()->getValue());
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant one being multiplied, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->equalsInt(1)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isZero()) {
      // If we have a multiply of zero, it will always be zero.
      return Ops[0];
    } else if (Ops[0]->isAllOnesValue()) {
      // If we have a mul by -1 of an add, try distributing the -1 among the
      // add operands.
      if (Ops.size() == 2) {
        if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(Ops[1])) {
          SmallVector<const SCEV *, 4> NewOps;
          bool AnyFolded = false;
          for (SCEVAddRecExpr::op_iterator I = Add->op_begin(),
                 E = Add->op_end(); I != E; ++I) {
            const SCEV *Mul = getMulExpr(Ops[0], *I);
            if (!isa<SCEVMulExpr>(Mul)) AnyFolded = true;
            NewOps.push_back(Mul);
          }
          if (AnyFolded)
            return getAddExpr(NewOps);
        }
        else if (const SCEVAddRecExpr *
                 AddRec = dyn_cast<SCEVAddRecExpr>(Ops[1])) {
          // Negation preserves a recurrence's no self-wrap property.
          SmallVector<const SCEV *, 4> Operands;
          for (SCEVAddRecExpr::op_iterator I = AddRec->op_begin(),
                 E = AddRec->op_end(); I != E; ++I) {
            Operands.push_back(getMulExpr(Ops[0], *I));
          }
          return getAddRecExpr(Operands, AddRec->getLoop(),
                               AddRec->getNoWrapFlags(SCEV::FlagNW));
        }
      }
    }

    if (Ops.size() == 1)
      return Ops[0];
  }

  // Skip over the add expression until we get to a multiply.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scMulExpr)
    ++Idx;

  // If there are mul operands inline them all into this expression.
  if (Idx < Ops.size()) {
    bool DeletedMul = false;
    while (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Ops[Idx])) {
      // If we have an mul, expand the mul operands onto the end of the operands
      // list.
      Ops.erase(Ops.begin()+Idx);
      Ops.append(Mul->op_begin(), Mul->op_end());
      DeletedMul = true;
    }

    // If we deleted at least one mul, we added operands to the end of the list,
    // and they are not necessarily sorted.  Recurse to resort and resimplify
    // any operands we just acquired.
    if (DeletedMul)
      return getMulExpr(Ops);
  }

  // If there are any add recurrences in the operands list, see if any other
  // added values are loop invariant.  If so, we can fold them into the
  // recurrence.
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scAddRecExpr)
    ++Idx;

  // Scan over all recurrences, trying to fold loop invariants into them.
  for (; Idx < Ops.size() && isa<SCEVAddRecExpr>(Ops[Idx]); ++Idx) {
    // Scan all of the other operands to this mul and add them to the vector if
    // they are loop invariant w.r.t. the recurrence.
    SmallVector<const SCEV *, 8> LIOps;
    const SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ops[Idx]);
    const Loop *AddRecLoop = AddRec->getLoop();
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      if (isLoopInvariant(Ops[i], AddRecLoop)) {
        LIOps.push_back(Ops[i]);
        Ops.erase(Ops.begin()+i);
        --i; --e;
      }

    // If we found some loop invariants, fold them into the recurrence.
    if (!LIOps.empty()) {
      //  NLI * LI * {Start,+,Step}  -->  NLI * {LI*Start,+,LI*Step}
      SmallVector<const SCEV *, 4> NewOps;
      NewOps.reserve(AddRec->getNumOperands());
      const SCEV *Scale = getMulExpr(LIOps);
      for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i)
        NewOps.push_back(getMulExpr(Scale, AddRec->getOperand(i)));

      // Build the new addrec. Propagate the NUW and NSW flags if both the
      // outer mul and the inner addrec are guaranteed to have no overflow.
      //
      // No self-wrap cannot be guaranteed after changing the step size, but
      // will be inferred if either NUW or NSW is true.
      Flags = AddRec->getNoWrapFlags(clearFlags(Flags, SCEV::FlagNW));
      const SCEV *NewRec = getAddRecExpr(NewOps, AddRecLoop, Flags);

      // If all of the other operands were loop invariant, we are done.
      if (Ops.size() == 1) return NewRec;

      // Otherwise, multiply the folded AddRec by the non-invariant parts.
      for (unsigned i = 0;; ++i)
        if (Ops[i] == AddRec) {
          Ops[i] = NewRec;
          break;
        }
      return getMulExpr(Ops);
    }

    // Okay, if there weren't any loop invariants to be folded, check to see if
    // there are multiple AddRec's with the same loop induction variable being
    // multiplied together.  If so, we can fold them.
    for (unsigned OtherIdx = Idx+1;
         OtherIdx < Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);
         ++OtherIdx) {
      bool Retry = false;
      if (AddRecLoop == cast<SCEVAddRecExpr>(Ops[OtherIdx])->getLoop()) {
        // {A,+,B}<L> * {C,+,D}<L>  -->  {A*C,+,A*D + B*C + B*D,+,2*B*D}<L>
        //
        // {A,+,B} * {C,+,D} = A+It*B * C+It*D = A*C + (A*D + B*C)*It + B*D*It^2
        // Given an equation of the form x + y*It + z*It^2 (above), we want to
        // express it in terms of {X,+,Y,+,Z}.
        // {X,+,Y,+,Z} = X + Y*It + Z*(It^2 - It)/2.
        // Rearranging, X = x, Y = y+z, Z = 2z.
        //
        // x = A*C, y = (A*D + B*C), z = B*D.
        // Therefore X = A*C, Y = A*D + B*C + B*D and Z = 2*B*D.
        for (; OtherIdx != Ops.size() && isa<SCEVAddRecExpr>(Ops[OtherIdx]);
             ++OtherIdx)
          if (const SCEVAddRecExpr *OtherAddRec =
                dyn_cast<SCEVAddRecExpr>(Ops[OtherIdx]))
            if (OtherAddRec->getLoop() == AddRecLoop) {
              const SCEV *A = AddRec->getStart();
              const SCEV *B = AddRec->getStepRecurrence(*this);
              const SCEV *C = OtherAddRec->getStart();
              const SCEV *D = OtherAddRec->getStepRecurrence(*this);
              const SCEV *NewStart = getMulExpr(A, C);
              const SCEV *BD = getMulExpr(B, D);
              const SCEV *NewStep = getAddExpr(getMulExpr(A, D),
                                               getMulExpr(B, C), BD);
              const SCEV *NewSecondOrderStep =
                  getMulExpr(BD, getConstant(BD->getType(), 2));

              // This can happen when AddRec or OtherAddRec have >3 operands.
              // TODO: support these add-recs.
              if (isLoopInvariant(NewStart, AddRecLoop) &&
                  isLoopInvariant(NewStep, AddRecLoop) &&
                  isLoopInvariant(NewSecondOrderStep, AddRecLoop)) {
                SmallVector<const SCEV *, 3> AddRecOps;
                AddRecOps.push_back(NewStart);
                AddRecOps.push_back(NewStep);
                AddRecOps.push_back(NewSecondOrderStep);
                const SCEV *NewAddRec = getAddRecExpr(AddRecOps,
                                                      AddRec->getLoop(),
                                                      SCEV::FlagAnyWrap);
                if (Ops.size() == 2) return NewAddRec;
                Ops[Idx] = AddRec = cast<SCEVAddRecExpr>(NewAddRec);
                Ops.erase(Ops.begin() + OtherIdx); --OtherIdx;
                Retry = true;
              }
            }
        if (Retry)
          return getMulExpr(Ops);
      }
    }

    // Otherwise couldn't fold anything into this recurrence.  Move onto the
    // next one.
  }

  // Okay, it looks like we really DO need an mul expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scMulExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = 0;
  SCEVMulExpr *S =
    static_cast<SCEVMulExpr *>(UniqueSCEVs.FindNodeOrInsertPos(ID, IP));
  if (!S) {
    const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
    std::uninitialized_copy(Ops.begin(), Ops.end(), O);
    S = new (SCEVAllocator) SCEVMulExpr(ID.Intern(SCEVAllocator),
                                        O, Ops.size());
    UniqueSCEVs.InsertNode(S, IP);
  }
  S->setNoWrapFlags(Flags);
  return S;
}

/// getUDivExpr - Get a canonical unsigned division expression, or something
/// simpler if possible.
const SCEV *ScalarEvolution::getUDivExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  assert(getEffectiveSCEVType(LHS->getType()) ==
         getEffectiveSCEVType(RHS->getType()) &&
         "SCEVUDivExpr operand types don't match!");

  if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS)) {
    if (RHSC->getValue()->equalsInt(1))
      return LHS;                               // X udiv 1 --> x
    // If the denominator is zero, the result of the udiv is undefined. Don't
    // try to analyze it, because the resolution chosen here may differ from
    // the resolution chosen in other parts of the compiler.
    if (!RHSC->getValue()->isZero()) {
      // Determine if the division can be folded into the operands of
      // its operands.
      // TODO: Generalize this to non-constants by using known-bits information.
      Type *Ty = LHS->getType();
      unsigned LZ = RHSC->getValue()->getValue().countLeadingZeros();
      unsigned MaxShiftAmt = getTypeSizeInBits(Ty) - LZ - 1;
      // For non-power-of-two values, effectively round the value up to the
      // nearest power of two.
      if (!RHSC->getValue()->getValue().isPowerOf2())
        ++MaxShiftAmt;
      IntegerType *ExtTy =
        IntegerType::get(getContext(), getTypeSizeInBits(Ty) + MaxShiftAmt);
      if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(LHS))
        if (const SCEVConstant *Step =
            dyn_cast<SCEVConstant>(AR->getStepRecurrence(*this))) {
          // {X,+,N}/C --> {X/C,+,N/C} if safe and N/C can be folded.
          const APInt &StepInt = Step->getValue()->getValue();
          const APInt &DivInt = RHSC->getValue()->getValue();
          if (!StepInt.urem(DivInt) &&
              getZeroExtendExpr(AR, ExtTy) ==
              getAddRecExpr(getZeroExtendExpr(AR->getStart(), ExtTy),
                            getZeroExtendExpr(Step, ExtTy),
                            AR->getLoop(), SCEV::FlagAnyWrap)) {
            SmallVector<const SCEV *, 4> Operands;
            for (unsigned i = 0, e = AR->getNumOperands(); i != e; ++i)
              Operands.push_back(getUDivExpr(AR->getOperand(i), RHS));
            return getAddRecExpr(Operands, AR->getLoop(),
                                 SCEV::FlagNW);
          }
          /// Get a canonical UDivExpr for a recurrence.
          /// {X,+,N}/C => {Y,+,N}/C where Y=X-(X%N). Safe when C%N=0.
          // We can currently only fold X%N if X is constant.
          const SCEVConstant *StartC = dyn_cast<SCEVConstant>(AR->getStart());
          if (StartC && !DivInt.urem(StepInt) &&
              getZeroExtendExpr(AR, ExtTy) ==
              getAddRecExpr(getZeroExtendExpr(AR->getStart(), ExtTy),
                            getZeroExtendExpr(Step, ExtTy),
                            AR->getLoop(), SCEV::FlagAnyWrap)) {
            const APInt &StartInt = StartC->getValue()->getValue();
            const APInt &StartRem = StartInt.urem(StepInt);
            if (StartRem != 0)
              LHS = getAddRecExpr(getConstant(StartInt - StartRem), Step,
                                  AR->getLoop(), SCEV::FlagNW);
          }
        }
      // (A*B)/C --> A*(B/C) if safe and B/C can be folded.
      if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(LHS)) {
        SmallVector<const SCEV *, 4> Operands;
        for (unsigned i = 0, e = M->getNumOperands(); i != e; ++i)
          Operands.push_back(getZeroExtendExpr(M->getOperand(i), ExtTy));
        if (getZeroExtendExpr(M, ExtTy) == getMulExpr(Operands))
          // Find an operand that's safely divisible.
          for (unsigned i = 0, e = M->getNumOperands(); i != e; ++i) {
            const SCEV *Op = M->getOperand(i);
            const SCEV *Div = getUDivExpr(Op, RHSC);
            if (!isa<SCEVUDivExpr>(Div) && getMulExpr(Div, RHSC) == Op) {
              Operands = SmallVector<const SCEV *, 4>(M->op_begin(),
                                                      M->op_end());
              Operands[i] = Div;
              return getMulExpr(Operands);
            }
          }
      }
      // (A+B)/C --> (A/C + B/C) if safe and A/C and B/C can be folded.
      if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(LHS)) {
        SmallVector<const SCEV *, 4> Operands;
        for (unsigned i = 0, e = A->getNumOperands(); i != e; ++i)
          Operands.push_back(getZeroExtendExpr(A->getOperand(i), ExtTy));
        if (getZeroExtendExpr(A, ExtTy) == getAddExpr(Operands)) {
          Operands.clear();
          for (unsigned i = 0, e = A->getNumOperands(); i != e; ++i) {
            const SCEV *Op = getUDivExpr(A->getOperand(i), RHS);
            if (isa<SCEVUDivExpr>(Op) ||
                getMulExpr(Op, RHS) != A->getOperand(i))
              break;
            Operands.push_back(Op);
          }
          if (Operands.size() == A->getNumOperands())
            return getAddExpr(Operands);
        }
      }

      // Fold if both operands are constant.
      if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS)) {
        Constant *LHSCV = LHSC->getValue();
        Constant *RHSCV = RHSC->getValue();
        return getConstant(cast<ConstantInt>(ConstantExpr::getUDiv(LHSCV,
                                                                   RHSCV)));
      }
    }
  }

  FoldingSetNodeID ID;
  ID.AddInteger(scUDivExpr);
  ID.AddPointer(LHS);
  ID.AddPointer(RHS);
  void *IP = 0;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  SCEV *S = new (SCEVAllocator) SCEVUDivExpr(ID.Intern(SCEVAllocator),
                                             LHS, RHS);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}


/// getAddRecExpr - Get an add recurrence expression for the specified loop.
/// Simplify the expression as much as possible.
const SCEV *ScalarEvolution::getAddRecExpr(const SCEV *Start, const SCEV *Step,
                                           const Loop *L,
                                           SCEV::NoWrapFlags Flags) {
  SmallVector<const SCEV *, 4> Operands;
  Operands.push_back(Start);
  if (const SCEVAddRecExpr *StepChrec = dyn_cast<SCEVAddRecExpr>(Step))
    if (StepChrec->getLoop() == L) {
      Operands.append(StepChrec->op_begin(), StepChrec->op_end());
      return getAddRecExpr(Operands, L, maskFlags(Flags, SCEV::FlagNW));
    }

  Operands.push_back(Step);
  return getAddRecExpr(Operands, L, Flags);
}

/// getAddRecExpr - Get an add recurrence expression for the specified loop.
/// Simplify the expression as much as possible.
const SCEV *
ScalarEvolution::getAddRecExpr(SmallVectorImpl<const SCEV *> &Operands,
                               const Loop *L, SCEV::NoWrapFlags Flags) {
  if (Operands.size() == 1) return Operands[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Operands[0]->getType());
  for (unsigned i = 1, e = Operands.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Operands[i]->getType()) == ETy &&
           "SCEVAddRecExpr operand types don't match!");
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    assert(isLoopInvariant(Operands[i], L) &&
           "SCEVAddRecExpr operand is not loop-invariant!");
#endif

  if (Operands.back()->isZero()) {
    Operands.pop_back();
    return getAddRecExpr(Operands, L, SCEV::FlagAnyWrap); // {X,+,0}  -->  X
  }

  // It's tempting to want to call getMaxBackedgeTakenCount count here and
  // use that information to infer NUW and NSW flags. However, computing a
  // BE count requires calling getAddRecExpr, so we may not yet have a
  // meaningful BE count at this point (and if we don't, we'd be stuck
  // with a SCEVCouldNotCompute as the cached BE count).

  // If FlagNSW is true and all the operands are non-negative, infer FlagNUW.
  // And vice-versa.
  int SignOrUnsignMask = SCEV::FlagNUW | SCEV::FlagNSW;
  SCEV::NoWrapFlags SignOrUnsignWrap = maskFlags(Flags, SignOrUnsignMask);
  if (SignOrUnsignWrap && (SignOrUnsignWrap != SignOrUnsignMask)) {
    bool All = true;
    for (SmallVectorImpl<const SCEV *>::const_iterator I = Operands.begin(),
         E = Operands.end(); I != E; ++I)
      if (!isKnownNonNegative(*I)) {
        All = false;
        break;
      }
    if (All) Flags = setFlags(Flags, (SCEV::NoWrapFlags)SignOrUnsignMask);
  }

  // Canonicalize nested AddRecs in by nesting them in order of loop depth.
  if (const SCEVAddRecExpr *NestedAR = dyn_cast<SCEVAddRecExpr>(Operands[0])) {
    const Loop *NestedLoop = NestedAR->getLoop();
    if (L->contains(NestedLoop) ?
        (L->getLoopDepth() < NestedLoop->getLoopDepth()) :
        (!NestedLoop->contains(L) &&
         DT->dominates(L->getHeader(), NestedLoop->getHeader()))) {
      SmallVector<const SCEV *, 4> NestedOperands(NestedAR->op_begin(),
                                                  NestedAR->op_end());
      Operands[0] = NestedAR->getStart();
      // AddRecs require their operands be loop-invariant with respect to their
      // loops. Don't perform this transformation if it would break this
      // requirement.
      bool AllInvariant = true;
      for (unsigned i = 0, e = Operands.size(); i != e; ++i)
        if (!isLoopInvariant(Operands[i], L)) {
          AllInvariant = false;
          break;
        }
      if (AllInvariant) {
        // Create a recurrence for the outer loop with the same step size.
        //
        // The outer recurrence keeps its NW flag but only keeps NUW/NSW if the
        // inner recurrence has the same property.
        SCEV::NoWrapFlags OuterFlags =
          maskFlags(Flags, SCEV::FlagNW | NestedAR->getNoWrapFlags());

        NestedOperands[0] = getAddRecExpr(Operands, L, OuterFlags);
        AllInvariant = true;
        for (unsigned i = 0, e = NestedOperands.size(); i != e; ++i)
          if (!isLoopInvariant(NestedOperands[i], NestedLoop)) {
            AllInvariant = false;
            break;
          }
        if (AllInvariant) {
          // Ok, both add recurrences are valid after the transformation.
          //
          // The inner recurrence keeps its NW flag but only keeps NUW/NSW if
          // the outer recurrence has the same property.
          SCEV::NoWrapFlags InnerFlags =
            maskFlags(NestedAR->getNoWrapFlags(), SCEV::FlagNW | Flags);
          return getAddRecExpr(NestedOperands, NestedLoop, InnerFlags);
        }
      }
      // Reset Operands to its original state.
      Operands[0] = NestedAR;
    }
  }

  // Okay, it looks like we really DO need an addrec expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scAddRecExpr);
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    ID.AddPointer(Operands[i]);
  ID.AddPointer(L);
  void *IP = 0;
  SCEVAddRecExpr *S =
    static_cast<SCEVAddRecExpr *>(UniqueSCEVs.FindNodeOrInsertPos(ID, IP));
  if (!S) {
    const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Operands.size());
    std::uninitialized_copy(Operands.begin(), Operands.end(), O);
    S = new (SCEVAllocator) SCEVAddRecExpr(ID.Intern(SCEVAllocator),
                                           O, Operands.size(), L);
    UniqueSCEVs.InsertNode(S, IP);
  }
  S->setNoWrapFlags(Flags);
  return S;
}

const SCEV *ScalarEvolution::getSMaxExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  SmallVector<const SCEV *, 2> Ops;
  Ops.push_back(LHS);
  Ops.push_back(RHS);
  return getSMaxExpr(Ops);
}

const SCEV *
ScalarEvolution::getSMaxExpr(SmallVectorImpl<const SCEV *> &Ops) {
  assert(!Ops.empty() && "Cannot get empty smax!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVSMaxExpr operand types don't match!");
#endif

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(getContext(),
                              APIntOps::smax(LHSC->getValue()->getValue(),
                                             RHSC->getValue()->getValue()));
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant minimum-int, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isMinValue(true)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isMaxValue(true)) {
      // If we have an smax with a constant maximum-int, it will always be
      // maximum-int.
      return Ops[0];
    }

    if (Ops.size() == 1) return Ops[0];
  }

  // Find the first SMax
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scSMaxExpr)
    ++Idx;

  // Check to see if one of the operands is an SMax. If so, expand its operands
  // onto our operand list, and recurse to simplify.
  if (Idx < Ops.size()) {
    bool DeletedSMax = false;
    while (const SCEVSMaxExpr *SMax = dyn_cast<SCEVSMaxExpr>(Ops[Idx])) {
      Ops.erase(Ops.begin()+Idx);
      Ops.append(SMax->op_begin(), SMax->op_end());
      DeletedSMax = true;
    }

    if (DeletedSMax)
      return getSMaxExpr(Ops);
  }

  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, delete one.  Since we sorted the list, these values are required to
  // be adjacent.
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    //  X smax Y smax Y  -->  X smax Y
    //  X smax Y         -->  X, if X is always greater than Y
    if (Ops[i] == Ops[i+1] ||
        isKnownPredicate(ICmpInst::ICMP_SGE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i+1, Ops.begin()+i+2);
      --i; --e;
    } else if (isKnownPredicate(ICmpInst::ICMP_SLE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i, Ops.begin()+i+1);
      --i; --e;
    }

  if (Ops.size() == 1) return Ops[0];

  assert(!Ops.empty() && "Reduced smax down to nothing!");

  // Okay, it looks like we really DO need an smax expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scSMaxExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = 0;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
  std::uninitialized_copy(Ops.begin(), Ops.end(), O);
  SCEV *S = new (SCEVAllocator) SCEVSMaxExpr(ID.Intern(SCEVAllocator),
                                             O, Ops.size());
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getUMaxExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  SmallVector<const SCEV *, 2> Ops;
  Ops.push_back(LHS);
  Ops.push_back(RHS);
  return getUMaxExpr(Ops);
}

const SCEV *
ScalarEvolution::getUMaxExpr(SmallVectorImpl<const SCEV *> &Ops) {
  assert(!Ops.empty() && "Cannot get empty umax!");
  if (Ops.size() == 1) return Ops[0];
#ifndef NDEBUG
  Type *ETy = getEffectiveSCEVType(Ops[0]->getType());
  for (unsigned i = 1, e = Ops.size(); i != e; ++i)
    assert(getEffectiveSCEVType(Ops[i]->getType()) == ETy &&
           "SCEVUMaxExpr operand types don't match!");
#endif

  // Sort by complexity, this groups all similar expression types together.
  GroupByComplexity(Ops, LI);

  // If there are any constants, fold them together.
  unsigned Idx = 0;
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(Ops[0])) {
    ++Idx;
    assert(Idx < Ops.size());
    while (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(Ops[Idx])) {
      // We found two constants, fold them together!
      ConstantInt *Fold = ConstantInt::get(getContext(),
                              APIntOps::umax(LHSC->getValue()->getValue(),
                                             RHSC->getValue()->getValue()));
      Ops[0] = getConstant(Fold);
      Ops.erase(Ops.begin()+1);  // Erase the folded element
      if (Ops.size() == 1) return Ops[0];
      LHSC = cast<SCEVConstant>(Ops[0]);
    }

    // If we are left with a constant minimum-int, strip it off.
    if (cast<SCEVConstant>(Ops[0])->getValue()->isMinValue(false)) {
      Ops.erase(Ops.begin());
      --Idx;
    } else if (cast<SCEVConstant>(Ops[0])->getValue()->isMaxValue(false)) {
      // If we have an umax with a constant maximum-int, it will always be
      // maximum-int.
      return Ops[0];
    }

    if (Ops.size() == 1) return Ops[0];
  }

  // Find the first UMax
  while (Idx < Ops.size() && Ops[Idx]->getSCEVType() < scUMaxExpr)
    ++Idx;

  // Check to see if one of the operands is a UMax. If so, expand its operands
  // onto our operand list, and recurse to simplify.
  if (Idx < Ops.size()) {
    bool DeletedUMax = false;
    while (const SCEVUMaxExpr *UMax = dyn_cast<SCEVUMaxExpr>(Ops[Idx])) {
      Ops.erase(Ops.begin()+Idx);
      Ops.append(UMax->op_begin(), UMax->op_end());
      DeletedUMax = true;
    }

    if (DeletedUMax)
      return getUMaxExpr(Ops);
  }

  // Okay, check to see if the same value occurs in the operand list twice.  If
  // so, delete one.  Since we sorted the list, these values are required to
  // be adjacent.
  for (unsigned i = 0, e = Ops.size()-1; i != e; ++i)
    //  X umax Y umax Y  -->  X umax Y
    //  X umax Y         -->  X, if X is always greater than Y
    if (Ops[i] == Ops[i+1] ||
        isKnownPredicate(ICmpInst::ICMP_UGE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i+1, Ops.begin()+i+2);
      --i; --e;
    } else if (isKnownPredicate(ICmpInst::ICMP_ULE, Ops[i], Ops[i+1])) {
      Ops.erase(Ops.begin()+i, Ops.begin()+i+1);
      --i; --e;
    }

  if (Ops.size() == 1) return Ops[0];

  assert(!Ops.empty() && "Reduced umax down to nothing!");

  // Okay, it looks like we really DO need a umax expr.  Check to see if we
  // already have one, otherwise create a new one.
  FoldingSetNodeID ID;
  ID.AddInteger(scUMaxExpr);
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    ID.AddPointer(Ops[i]);
  void *IP = 0;
  if (const SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) return S;
  const SCEV **O = SCEVAllocator.Allocate<const SCEV *>(Ops.size());
  std::uninitialized_copy(Ops.begin(), Ops.end(), O);
  SCEV *S = new (SCEVAllocator) SCEVUMaxExpr(ID.Intern(SCEVAllocator),
                                             O, Ops.size());
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

const SCEV *ScalarEvolution::getSMinExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  // ~smax(~x, ~y) == smin(x, y).
  return getNotSCEV(getSMaxExpr(getNotSCEV(LHS), getNotSCEV(RHS)));
}

const SCEV *ScalarEvolution::getUMinExpr(const SCEV *LHS,
                                         const SCEV *RHS) {
  // ~umax(~x, ~y) == umin(x, y)
  return getNotSCEV(getUMaxExpr(getNotSCEV(LHS), getNotSCEV(RHS)));
}

const SCEV *ScalarEvolution::getSizeOfExpr(Type *AllocTy) {
  // If we have TargetData, we can bypass creating a target-independent
  // constant expression and then folding it back into a ConstantInt.
  // This is just a compile-time optimization.
  if (TD)
    return getConstant(TD->getIntPtrType(getContext()),
                       TD->getTypeAllocSize(AllocTy));

  Constant *C = ConstantExpr::getSizeOf(AllocTy);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (Constant *Folded = ConstantFoldConstantExpression(CE, TD))
      C = Folded;
  Type *Ty = getEffectiveSCEVType(PointerType::getUnqual(AllocTy));
  return getTruncateOrZeroExtend(getSCEV(C), Ty);
}

const SCEV *ScalarEvolution::getAlignOfExpr(Type *AllocTy) {
  Constant *C = ConstantExpr::getAlignOf(AllocTy);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (Constant *Folded = ConstantFoldConstantExpression(CE, TD))
      C = Folded;
  Type *Ty = getEffectiveSCEVType(PointerType::getUnqual(AllocTy));
  return getTruncateOrZeroExtend(getSCEV(C), Ty);
}

const SCEV *ScalarEvolution::getOffsetOfExpr(StructType *STy,
                                             unsigned FieldNo) {
  // If we have TargetData, we can bypass creating a target-independent
  // constant expression and then folding it back into a ConstantInt.
  // This is just a compile-time optimization.
  if (TD)
    return getConstant(TD->getIntPtrType(getContext()),
                       TD->getStructLayout(STy)->getElementOffset(FieldNo));

  Constant *C = ConstantExpr::getOffsetOf(STy, FieldNo);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (Constant *Folded = ConstantFoldConstantExpression(CE, TD))
      C = Folded;
  Type *Ty = getEffectiveSCEVType(PointerType::getUnqual(STy));
  return getTruncateOrZeroExtend(getSCEV(C), Ty);
}

const SCEV *ScalarEvolution::getOffsetOfExpr(Type *CTy,
                                             Constant *FieldNo) {
  Constant *C = ConstantExpr::getOffsetOf(CTy, FieldNo);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (Constant *Folded = ConstantFoldConstantExpression(CE, TD))
      C = Folded;
  Type *Ty = getEffectiveSCEVType(PointerType::getUnqual(CTy));
  return getTruncateOrZeroExtend(getSCEV(C), Ty);
}

const SCEV *ScalarEvolution::getUnknown(Value *V) {
  // Don't attempt to do anything other than create a SCEVUnknown object
  // here.  createSCEV only calls getUnknown after checking for all other
  // interesting possibilities, and any other code that calls getUnknown
  // is doing so in order to hide a value from SCEV canonicalization.

  FoldingSetNodeID ID;
  ID.AddInteger(scUnknown);
  ID.AddPointer(V);
  void *IP = 0;
  if (SCEV *S = UniqueSCEVs.FindNodeOrInsertPos(ID, IP)) {
    assert(cast<SCEVUnknown>(S)->getValue() == V &&
           "Stale SCEVUnknown in uniquing map!");
    return S;
  }
  SCEV *S = new (SCEVAllocator) SCEVUnknown(ID.Intern(SCEVAllocator), V, this,
                                            FirstUnknown);
  FirstUnknown = cast<SCEVUnknown>(S);
  UniqueSCEVs.InsertNode(S, IP);
  return S;
}

//===----------------------------------------------------------------------===//
//            Basic SCEV Analysis and PHI Idiom Recognition Code
//

/// isSCEVable - Test if values of the given type are analyzable within
/// the SCEV framework. This primarily includes integer types, and it
/// can optionally include pointer types if the ScalarEvolution class
/// has access to target-specific information.
bool ScalarEvolution::isSCEVable(Type *Ty) const {
  // Integers and pointers are always SCEVable.
  return Ty->isIntegerTy() || Ty->isPointerTy();
}

/// getTypeSizeInBits - Return the size in bits of the specified type,
/// for which isSCEVable must return true.
uint64_t ScalarEvolution::getTypeSizeInBits(Type *Ty) const {
  assert(isSCEVable(Ty) && "Type is not SCEVable!");

  // If we have a TargetData, use it!
  if (TD)
    return TD->getTypeSizeInBits(Ty);

  // Integer types have fixed sizes.
  if (Ty->isIntegerTy())
    return Ty->getPrimitiveSizeInBits();

  // The only other support type is pointer. Without TargetData, conservatively
  // assume pointers are 64-bit.
  assert(Ty->isPointerTy() && "isSCEVable permitted a non-SCEVable type!");
  return 64;
}

/// getEffectiveSCEVType - Return a type with the same bitwidth as
/// the given type and which represents how SCEV will treat the given
/// type, for which isSCEVable must return true. For pointer types,
/// this is the pointer-sized integer type.
Type *ScalarEvolution::getEffectiveSCEVType(Type *Ty) const {
  assert(isSCEVable(Ty) && "Type is not SCEVable!");

  if (Ty->isIntegerTy())
    return Ty;

  // The only other support type is pointer.
  assert(Ty->isPointerTy() && "Unexpected non-pointer non-integer type!");
  if (TD) return TD->getIntPtrType(getContext());

  // Without TargetData, conservatively assume pointers are 64-bit.
  return Type::getInt64Ty(getContext());
}

const SCEV *ScalarEvolution::getCouldNotCompute() {
  return &CouldNotCompute;
}

/// getSCEV - Return an existing SCEV if it exists, otherwise analyze the
/// expression and create a new one.
const SCEV *ScalarEvolution::getSCEV(Value *V) {
  assert(isSCEVable(V->getType()) && "Value is not SCEVable!");

  ValueExprMapType::const_iterator I = ValueExprMap.find(V);
  if (I != ValueExprMap.end()) return I->second;
  const SCEV *S = createSCEV(V);

  // The process of creating a SCEV for V may have caused other SCEVs
  // to have been created, so it's necessary to insert the new entry
  // from scratch, rather than trying to remember the insert position
  // above.
  ValueExprMap.insert(std::make_pair(SCEVCallbackVH(V, this), S));
  return S;
}

/// getNegativeSCEV - Return a SCEV corresponding to -V = -1*V
///
const SCEV *ScalarEvolution::getNegativeSCEV(const SCEV *V) {
  if (const SCEVConstant *VC = dyn_cast<SCEVConstant>(V))
    return getConstant(
               cast<ConstantInt>(ConstantExpr::getNeg(VC->getValue())));

  Type *Ty = V->getType();
  Ty = getEffectiveSCEVType(Ty);
  return getMulExpr(V,
                  getConstant(cast<ConstantInt>(Constant::getAllOnesValue(Ty))));
}

/// getNotSCEV - Return a SCEV corresponding to ~V = -1-V
const SCEV *ScalarEvolution::getNotSCEV(const SCEV *V) {
  if (const SCEVConstant *VC = dyn_cast<SCEVConstant>(V))
    return getConstant(
                cast<ConstantInt>(ConstantExpr::getNot(VC->getValue())));

  Type *Ty = V->getType();
  Ty = getEffectiveSCEVType(Ty);
  const SCEV *AllOnes =
                   getConstant(cast<ConstantInt>(Constant::getAllOnesValue(Ty)));
  return getMinusSCEV(AllOnes, V);
}

/// getMinusSCEV - Return LHS-RHS.  Minus is represented in SCEV as A+B*-1.
const SCEV *ScalarEvolution::getMinusSCEV(const SCEV *LHS, const SCEV *RHS,
                                          SCEV::NoWrapFlags Flags) {
  assert(!maskFlags(Flags, SCEV::FlagNUW) && "subtraction does not have NUW");

  // Fast path: X - X --> 0.
  if (LHS == RHS)
    return getConstant(LHS->getType(), 0);

  // X - Y --> X + -Y
  return getAddExpr(LHS, getNegativeSCEV(RHS), Flags);
}

/// getTruncateOrZeroExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is zero
/// extended.
const SCEV *
ScalarEvolution::getTruncateOrZeroExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate or zero extend with non-integer arguments!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  if (getTypeSizeInBits(SrcTy) > getTypeSizeInBits(Ty))
    return getTruncateExpr(V, Ty);
  return getZeroExtendExpr(V, Ty);
}

/// getTruncateOrSignExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is sign
/// extended.
const SCEV *
ScalarEvolution::getTruncateOrSignExtend(const SCEV *V,
                                         Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate or zero extend with non-integer arguments!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  if (getTypeSizeInBits(SrcTy) > getTypeSizeInBits(Ty))
    return getTruncateExpr(V, Ty);
  return getSignExtendExpr(V, Ty);
}

/// getNoopOrZeroExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is zero
/// extended.  The conversion must not be narrowing.
const SCEV *
ScalarEvolution::getNoopOrZeroExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot noop or zero extend with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) <= getTypeSizeInBits(Ty) &&
         "getNoopOrZeroExtend cannot truncate!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getZeroExtendExpr(V, Ty);
}

/// getNoopOrSignExtend - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  If the type must be extended, it is sign
/// extended.  The conversion must not be narrowing.
const SCEV *
ScalarEvolution::getNoopOrSignExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot noop or sign extend with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) <= getTypeSizeInBits(Ty) &&
         "getNoopOrSignExtend cannot truncate!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getSignExtendExpr(V, Ty);
}

/// getNoopOrAnyExtend - Return a SCEV corresponding to a conversion of
/// the input value to the specified type. If the type must be extended,
/// it is extended with unspecified bits. The conversion must not be
/// narrowing.
const SCEV *
ScalarEvolution::getNoopOrAnyExtend(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot noop or any extend with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) <= getTypeSizeInBits(Ty) &&
         "getNoopOrAnyExtend cannot truncate!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getAnyExtendExpr(V, Ty);
}

/// getTruncateOrNoop - Return a SCEV corresponding to a conversion of the
/// input value to the specified type.  The conversion must not be widening.
const SCEV *
ScalarEvolution::getTruncateOrNoop(const SCEV *V, Type *Ty) {
  Type *SrcTy = V->getType();
  assert((SrcTy->isIntegerTy() || SrcTy->isPointerTy()) &&
         (Ty->isIntegerTy() || Ty->isPointerTy()) &&
         "Cannot truncate or noop with non-integer arguments!");
  assert(getTypeSizeInBits(SrcTy) >= getTypeSizeInBits(Ty) &&
         "getTruncateOrNoop cannot extend!");
  if (getTypeSizeInBits(SrcTy) == getTypeSizeInBits(Ty))
    return V;  // No conversion
  return getTruncateExpr(V, Ty);
}

/// getUMaxFromMismatchedTypes - Promote the operands to the wider of
/// the types using zero-extension, and then perform a umax operation
/// with them.
const SCEV *ScalarEvolution::getUMaxFromMismatchedTypes(const SCEV *LHS,
                                                        const SCEV *RHS) {
  const SCEV *PromotedLHS = LHS;
  const SCEV *PromotedRHS = RHS;

  if (getTypeSizeInBits(LHS->getType()) > getTypeSizeInBits(RHS->getType()))
    PromotedRHS = getZeroExtendExpr(RHS, LHS->getType());
  else
    PromotedLHS = getNoopOrZeroExtend(LHS, RHS->getType());

  return getUMaxExpr(PromotedLHS, PromotedRHS);
}

/// getUMinFromMismatchedTypes - Promote the operands to the wider of
/// the types using zero-extension, and then perform a umin operation
/// with them.
const SCEV *ScalarEvolution::getUMinFromMismatchedTypes(const SCEV *LHS,
                                                        const SCEV *RHS) {
  const SCEV *PromotedLHS = LHS;
  const SCEV *PromotedRHS = RHS;

  if (getTypeSizeInBits(LHS->getType()) > getTypeSizeInBits(RHS->getType()))
    PromotedRHS = getZeroExtendExpr(RHS, LHS->getType());
  else
    PromotedLHS = getNoopOrZeroExtend(LHS, RHS->getType());

  return getUMinExpr(PromotedLHS, PromotedRHS);
}

/// getPointerBase - Transitively follow the chain of pointer-type operands
/// until reaching a SCEV that does not have a single pointer operand. This
/// returns a SCEVUnknown pointer for well-formed pointer-type expressions,
/// but corner cases do exist.
const SCEV *ScalarEvolution::getPointerBase(const SCEV *V) {
  // A pointer operand may evaluate to a nonpointer expression, such as null.
  if (!V->getType()->isPointerTy())
    return V;

  if (const SCEVCastExpr *Cast = dyn_cast<SCEVCastExpr>(V)) {
    return getPointerBase(Cast->getOperand());
  }
  else if (const SCEVNAryExpr *NAry = dyn_cast<SCEVNAryExpr>(V)) {
    const SCEV *PtrOp = 0;
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      if ((*I)->getType()->isPointerTy()) {
        // Cannot find the base of an expression with multiple pointer operands.
        if (PtrOp)
          return V;
        PtrOp = *I;
      }
    }
    if (!PtrOp)
      return V;
    return getPointerBase(PtrOp);
  }
  return V;
}

/// PushDefUseChildren - Push users of the given Instruction
/// onto the given Worklist.
static void
PushDefUseChildren(Instruction *I,
                   SmallVectorImpl<Instruction *> &Worklist) {
  // Push the def-use children onto the Worklist stack.
  for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
       UI != UE; ++UI)
    Worklist.push_back(cast<Instruction>(*UI));
}

/// ForgetSymbolicValue - This looks up computed SCEV values for all
/// instructions that depend on the given instruction and removes them from
/// the ValueExprMapType map if they reference SymName. This is used during PHI
/// resolution.
void
ScalarEvolution::ForgetSymbolicName(Instruction *PN, const SCEV *SymName) {
  SmallVector<Instruction *, 16> Worklist;
  PushDefUseChildren(PN, Worklist);

  SmallPtrSet<Instruction *, 8> Visited;
  Visited.insert(PN);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!Visited.insert(I)) continue;

    ValueExprMapType::iterator It =
      ValueExprMap.find(static_cast<Value *>(I));
    if (It != ValueExprMap.end()) {
      const SCEV *Old = It->second;

      // Short-circuit the def-use traversal if the symbolic name
      // ceases to appear in expressions.
      if (Old != SymName && !hasOperand(Old, SymName))
        continue;

      // SCEVUnknown for a PHI either means that it has an unrecognized
      // structure, it's a PHI that's in the progress of being computed
      // by createNodeForPHI, or it's a single-value PHI. In the first case,
      // additional loop trip count information isn't going to change anything.
      // In the second case, createNodeForPHI will perform the necessary
      // updates on its own when it gets to that point. In the third, we do
      // want to forget the SCEVUnknown.
      if (!isa<PHINode>(I) ||
          !isa<SCEVUnknown>(Old) ||
          (I != PN && Old == SymName)) {
        forgetMemoizedResults(Old);
        ValueExprMap.erase(It);
      }
    }

    PushDefUseChildren(I, Worklist);
  }
}

/// createNodeForPHI - PHI nodes have two cases.  Either the PHI node exists in
/// a loop header, making it a potential recurrence, or it doesn't.
///
const SCEV *ScalarEvolution::createNodeForPHI(PHINode *PN) {
  if (const Loop *L = LI->getLoopFor(PN->getParent()))
    if (L->getHeader() == PN->getParent()) {
      // The loop may have multiple entrances or multiple exits; we can analyze
      // this phi as an addrec if it has a unique entry value and a unique
      // backedge value.
      Value *BEValueV = 0, *StartValueV = 0;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        Value *V = PN->getIncomingValue(i);
        if (L->contains(PN->getIncomingBlock(i))) {
          if (!BEValueV) {
            BEValueV = V;
          } else if (BEValueV != V) {
            BEValueV = 0;
            break;
          }
        } else if (!StartValueV) {
          StartValueV = V;
        } else if (StartValueV != V) {
          StartValueV = 0;
          break;
        }
      }
      if (BEValueV && StartValueV) {
        // While we are analyzing this PHI node, handle its value symbolically.
        const SCEV *SymbolicName = getUnknown(PN);
        assert(ValueExprMap.find(PN) == ValueExprMap.end() &&
               "PHI node already processed?");
        ValueExprMap.insert(std::make_pair(SCEVCallbackVH(PN, this), SymbolicName));

        // Using this symbolic name for the PHI, analyze the value coming around
        // the back-edge.
        const SCEV *BEValue = getSCEV(BEValueV);

        // NOTE: If BEValue is loop invariant, we know that the PHI node just
        // has a special value for the first iteration of the loop.

        // If the value coming around the backedge is an add with the symbolic
        // value we just inserted, then we found a simple induction variable!
        if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(BEValue)) {
          // If there is a single occurrence of the symbolic value, replace it
          // with a recurrence.
          unsigned FoundIndex = Add->getNumOperands();
          for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
            if (Add->getOperand(i) == SymbolicName)
              if (FoundIndex == e) {
                FoundIndex = i;
                break;
              }

          if (FoundIndex != Add->getNumOperands()) {
            // Create an add with everything but the specified operand.
            SmallVector<const SCEV *, 8> Ops;
            for (unsigned i = 0, e = Add->getNumOperands(); i != e; ++i)
              if (i != FoundIndex)
                Ops.push_back(Add->getOperand(i));
            const SCEV *Accum = getAddExpr(Ops);

            // This is not a valid addrec if the step amount is varying each
            // loop iteration, but is not itself an addrec in this loop.
            if (isLoopInvariant(Accum, L) ||
                (isa<SCEVAddRecExpr>(Accum) &&
                 cast<SCEVAddRecExpr>(Accum)->getLoop() == L)) {
              SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap;

              // If the increment doesn't overflow, then neither the addrec nor
              // the post-increment will overflow.
              if (const AddOperator *OBO = dyn_cast<AddOperator>(BEValueV)) {
                if (OBO->hasNoUnsignedWrap())
                  Flags = setFlags(Flags, SCEV::FlagNUW);
                if (OBO->hasNoSignedWrap())
                  Flags = setFlags(Flags, SCEV::FlagNSW);
              } else if (const GEPOperator *GEP =
                         dyn_cast<GEPOperator>(BEValueV)) {
                // If the increment is an inbounds GEP, then we know the address
                // space cannot be wrapped around. We cannot make any guarantee
                // about signed or unsigned overflow because pointers are
                // unsigned but we may have a negative index from the base
                // pointer.
                if (GEP->isInBounds())
                  Flags = setFlags(Flags, SCEV::FlagNW);
              }

              const SCEV *StartVal = getSCEV(StartValueV);
              const SCEV *PHISCEV = getAddRecExpr(StartVal, Accum, L, Flags);

              // Since the no-wrap flags are on the increment, they apply to the
              // post-incremented value as well.
              if (isLoopInvariant(Accum, L))
                (void)getAddRecExpr(getAddExpr(StartVal, Accum),
                                    Accum, L, Flags);

              // Okay, for the entire analysis of this edge we assumed the PHI
              // to be symbolic.  We now need to go back and purge all of the
              // entries for the scalars that use the symbolic expression.
              ForgetSymbolicName(PN, SymbolicName);
              ValueExprMap[SCEVCallbackVH(PN, this)] = PHISCEV;
              return PHISCEV;
            }
          }
        } else if (const SCEVAddRecExpr *AddRec =
                     dyn_cast<SCEVAddRecExpr>(BEValue)) {
          // Otherwise, this could be a loop like this:
          //     i = 0;  for (j = 1; ..; ++j) { ....  i = j; }
          // In this case, j = {1,+,1}  and BEValue is j.
          // Because the other in-value of i (0) fits the evolution of BEValue
          // i really is an addrec evolution.
          if (AddRec->getLoop() == L && AddRec->isAffine()) {
            const SCEV *StartVal = getSCEV(StartValueV);

            // If StartVal = j.start - j.stride, we can use StartVal as the
            // initial step of the addrec evolution.
            if (StartVal == getMinusSCEV(AddRec->getOperand(0),
                                         AddRec->getOperand(1))) {
              // FIXME: For constant StartVal, we should be able to infer
              // no-wrap flags.
              const SCEV *PHISCEV =
                getAddRecExpr(StartVal, AddRec->getOperand(1), L,
                              SCEV::FlagAnyWrap);

              // Okay, for the entire analysis of this edge we assumed the PHI
              // to be symbolic.  We now need to go back and purge all of the
              // entries for the scalars that use the symbolic expression.
              ForgetSymbolicName(PN, SymbolicName);
              ValueExprMap[SCEVCallbackVH(PN, this)] = PHISCEV;
              return PHISCEV;
            }
          }
        }
      }
    }

  // If the PHI has a single incoming value, follow that value, unless the
  // PHI's incoming blocks are in a different loop, in which case doing so
  // risks breaking LCSSA form. Instcombine would normally zap these, but
  // it doesn't have DominatorTree information, so it may miss cases.
  if (Value *V = SimplifyInstruction(PN, TD, DT))
    if (LI->replacementPreservesLCSSAForm(PN, V))
      return getSCEV(V);

  // If it's not a loop phi, we can't handle it yet.
  return getUnknown(PN);
}

/// createNodeForGEP - Expand GEP instructions into add and multiply
/// operations. This allows them to be analyzed by regular SCEV code.
///
const SCEV *ScalarEvolution::createNodeForGEP(GEPOperator *GEP) {

  // Don't blindly transfer the inbounds flag from the GEP instruction to the
  // Add expression, because the Instruction may be guarded by control flow
  // and the no-overflow bits may not be valid for the expression in any
  // context.
  bool isInBounds = GEP->isInBounds();

  Type *IntPtrTy = getEffectiveSCEVType(GEP->getType());
  Value *Base = GEP->getOperand(0);
  // Don't attempt to analyze GEPs over unsized objects.
  if (!cast<PointerType>(Base->getType())->getElementType()->isSized())
    return getUnknown(GEP);
  const SCEV *TotalOffset = getConstant(IntPtrTy, 0);
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (GetElementPtrInst::op_iterator I = llvm::next(GEP->op_begin()),
                                      E = GEP->op_end();
       I != E; ++I) {
    Value *Index = *I;
    // Compute the (potentially symbolic) offset in bytes for this index.
    if (StructType *STy = dyn_cast<StructType>(*GTI++)) {
      // For a struct, add the member offset.
      unsigned FieldNo = cast<ConstantInt>(Index)->getZExtValue();
      const SCEV *FieldOffset = getOffsetOfExpr(STy, FieldNo);

      // Add the field offset to the running total offset.
      TotalOffset = getAddExpr(TotalOffset, FieldOffset);
    } else {
      // For an array, add the element offset, explicitly scaled.
      const SCEV *ElementSize = getSizeOfExpr(*GTI);
      const SCEV *IndexS = getSCEV(Index);
      // Getelementptr indices are signed.
      IndexS = getTruncateOrSignExtend(IndexS, IntPtrTy);

      // Multiply the index by the element size to compute the element offset.
      const SCEV *LocalOffset = getMulExpr(IndexS, ElementSize,
                                           isInBounds ? SCEV::FlagNSW :
                                           SCEV::FlagAnyWrap);

      // Add the element offset to the running total offset.
      TotalOffset = getAddExpr(TotalOffset, LocalOffset);
    }
  }

  // Get the SCEV for the GEP base.
  const SCEV *BaseS = getSCEV(Base);

  // Add the total offset from all the GEP indices to the base.
  return getAddExpr(BaseS, TotalOffset,
                    isInBounds ? SCEV::FlagNSW : SCEV::FlagAnyWrap);
}

/// GetMinTrailingZeros - Determine the minimum number of zero bits that S is
/// guaranteed to end in (at every loop iteration).  It is, at the same time,
/// the minimum number of times S is divisible by 2.  For example, given {4,+,8}
/// it returns 2.  If S is guaranteed to be 0, it returns the bitwidth of S.
uint32_t
ScalarEvolution::GetMinTrailingZeros(const SCEV *S) {
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S))
    return C->getValue()->getValue().countTrailingZeros();

  if (const SCEVTruncateExpr *T = dyn_cast<SCEVTruncateExpr>(S))
    return std::min(GetMinTrailingZeros(T->getOperand()),
                    (uint32_t)getTypeSizeInBits(T->getType()));

  if (const SCEVZeroExtendExpr *E = dyn_cast<SCEVZeroExtendExpr>(S)) {
    uint32_t OpRes = GetMinTrailingZeros(E->getOperand());
    return OpRes == getTypeSizeInBits(E->getOperand()->getType()) ?
             getTypeSizeInBits(E->getType()) : OpRes;
  }

  if (const SCEVSignExtendExpr *E = dyn_cast<SCEVSignExtendExpr>(S)) {
    uint32_t OpRes = GetMinTrailingZeros(E->getOperand());
    return OpRes == getTypeSizeInBits(E->getOperand()->getType()) ?
             getTypeSizeInBits(E->getType()) : OpRes;
  }

  if (const SCEVAddExpr *A = dyn_cast<SCEVAddExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(A->getOperand(0));
    for (unsigned i = 1, e = A->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(A->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(S)) {
    // The result is the sum of all operands results.
    uint32_t SumOpRes = GetMinTrailingZeros(M->getOperand(0));
    uint32_t BitWidth = getTypeSizeInBits(M->getType());
    for (unsigned i = 1, e = M->getNumOperands();
         SumOpRes != BitWidth && i != e; ++i)
      SumOpRes = std::min(SumOpRes + GetMinTrailingZeros(M->getOperand(i)),
                          BitWidth);
    return SumOpRes;
  }

  if (const SCEVAddRecExpr *A = dyn_cast<SCEVAddRecExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(A->getOperand(0));
    for (unsigned i = 1, e = A->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(A->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVSMaxExpr *M = dyn_cast<SCEVSMaxExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(M->getOperand(0));
    for (unsigned i = 1, e = M->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(M->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVUMaxExpr *M = dyn_cast<SCEVUMaxExpr>(S)) {
    // The result is the min of all operands results.
    uint32_t MinOpRes = GetMinTrailingZeros(M->getOperand(0));
    for (unsigned i = 1, e = M->getNumOperands(); MinOpRes && i != e; ++i)
      MinOpRes = std::min(MinOpRes, GetMinTrailingZeros(M->getOperand(i)));
    return MinOpRes;
  }

  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    // For a SCEVUnknown, ask ValueTracking.
    unsigned BitWidth = getTypeSizeInBits(U->getType());
    APInt Mask = APInt::getAllOnesValue(BitWidth);
    APInt Zeros(BitWidth, 0), Ones(BitWidth, 0);
    ComputeMaskedBits(U->getValue(), Mask, Zeros, Ones);
    return Zeros.countTrailingOnes();
  }

  // SCEVUDivExpr
  return 0;
}

/// getUnsignedRange - Determine the unsigned range for a particular SCEV.
///
ConstantRange
ScalarEvolution::getUnsignedRange(const SCEV *S) {
  // See if we've computed this range already.
  DenseMap<const SCEV *, ConstantRange>::iterator I = UnsignedRanges.find(S);
  if (I != UnsignedRanges.end())
    return I->second;

  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S))
    return setUnsignedRange(C, ConstantRange(C->getValue()->getValue()));

  unsigned BitWidth = getTypeSizeInBits(S->getType());
  ConstantRange ConservativeResult(BitWidth, /*isFullSet=*/true);

  // If the value has known zeros, the maximum unsigned value will have those
  // known zeros as well.
  uint32_t TZ = GetMinTrailingZeros(S);
  if (TZ != 0)
    ConservativeResult =
      ConstantRange(APInt::getMinValue(BitWidth),
                    APInt::getMaxValue(BitWidth).lshr(TZ).shl(TZ) + 1);

  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    ConstantRange X = getUnsignedRange(Add->getOperand(0));
    for (unsigned i = 1, e = Add->getNumOperands(); i != e; ++i)
      X = X.add(getUnsignedRange(Add->getOperand(i)));
    return setUnsignedRange(Add, ConservativeResult.intersectWith(X));
  }

  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(S)) {
    ConstantRange X = getUnsignedRange(Mul->getOperand(0));
    for (unsigned i = 1, e = Mul->getNumOperands(); i != e; ++i)
      X = X.multiply(getUnsignedRange(Mul->getOperand(i)));
    return setUnsignedRange(Mul, ConservativeResult.intersectWith(X));
  }

  if (const SCEVSMaxExpr *SMax = dyn_cast<SCEVSMaxExpr>(S)) {
    ConstantRange X = getUnsignedRange(SMax->getOperand(0));
    for (unsigned i = 1, e = SMax->getNumOperands(); i != e; ++i)
      X = X.smax(getUnsignedRange(SMax->getOperand(i)));
    return setUnsignedRange(SMax, ConservativeResult.intersectWith(X));
  }

  if (const SCEVUMaxExpr *UMax = dyn_cast<SCEVUMaxExpr>(S)) {
    ConstantRange X = getUnsignedRange(UMax->getOperand(0));
    for (unsigned i = 1, e = UMax->getNumOperands(); i != e; ++i)
      X = X.umax(getUnsignedRange(UMax->getOperand(i)));
    return setUnsignedRange(UMax, ConservativeResult.intersectWith(X));
  }

  if (const SCEVUDivExpr *UDiv = dyn_cast<SCEVUDivExpr>(S)) {
    ConstantRange X = getUnsignedRange(UDiv->getLHS());
    ConstantRange Y = getUnsignedRange(UDiv->getRHS());
    return setUnsignedRange(UDiv, ConservativeResult.intersectWith(X.udiv(Y)));
  }

  if (const SCEVZeroExtendExpr *ZExt = dyn_cast<SCEVZeroExtendExpr>(S)) {
    ConstantRange X = getUnsignedRange(ZExt->getOperand());
    return setUnsignedRange(ZExt,
      ConservativeResult.intersectWith(X.zeroExtend(BitWidth)));
  }

  if (const SCEVSignExtendExpr *SExt = dyn_cast<SCEVSignExtendExpr>(S)) {
    ConstantRange X = getUnsignedRange(SExt->getOperand());
    return setUnsignedRange(SExt,
      ConservativeResult.intersectWith(X.signExtend(BitWidth)));
  }

  if (const SCEVTruncateExpr *Trunc = dyn_cast<SCEVTruncateExpr>(S)) {
    ConstantRange X = getUnsignedRange(Trunc->getOperand());
    return setUnsignedRange(Trunc,
      ConservativeResult.intersectWith(X.truncate(BitWidth)));
  }

  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(S)) {
    // If there's no unsigned wrap, the value will never be less than its
    // initial value.
    if (AddRec->getNoWrapFlags(SCEV::FlagNUW))
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(AddRec->getStart()))
        if (!C->getValue()->isZero())
          ConservativeResult =
            ConservativeResult.intersectWith(
              ConstantRange(C->getValue()->getValue(), APInt(BitWidth, 0)));

    // TODO: non-affine addrec
    if (AddRec->isAffine()) {
      Type *Ty = AddRec->getType();
      const SCEV *MaxBECount = getMaxBackedgeTakenCount(AddRec->getLoop());
      if (!isa<SCEVCouldNotCompute>(MaxBECount) &&
          getTypeSizeInBits(MaxBECount->getType()) <= BitWidth) {
        MaxBECount = getNoopOrZeroExtend(MaxBECount, Ty);

        const SCEV *Start = AddRec->getStart();
        const SCEV *Step = AddRec->getStepRecurrence(*this);

        ConstantRange StartRange = getUnsignedRange(Start);
        ConstantRange StepRange = getSignedRange(Step);
        ConstantRange MaxBECountRange = getUnsignedRange(MaxBECount);
        ConstantRange EndRange =
          StartRange.add(MaxBECountRange.multiply(StepRange));

        // Check for overflow. This must be done with ConstantRange arithmetic
        // because we could be called from within the ScalarEvolution overflow
        // checking code.
        ConstantRange ExtStartRange = StartRange.zextOrTrunc(BitWidth*2+1);
        ConstantRange ExtStepRange = StepRange.sextOrTrunc(BitWidth*2+1);
        ConstantRange ExtMaxBECountRange =
          MaxBECountRange.zextOrTrunc(BitWidth*2+1);
        ConstantRange ExtEndRange = EndRange.zextOrTrunc(BitWidth*2+1);
        if (ExtStartRange.add(ExtMaxBECountRange.multiply(ExtStepRange)) !=
            ExtEndRange)
          return setUnsignedRange(AddRec, ConservativeResult);

        APInt Min = APIntOps::umin(StartRange.getUnsignedMin(),
                                   EndRange.getUnsignedMin());
        APInt Max = APIntOps::umax(StartRange.getUnsignedMax(),
                                   EndRange.getUnsignedMax());
        if (Min.isMinValue() && Max.isMaxValue())
          return setUnsignedRange(AddRec, ConservativeResult);
        return setUnsignedRange(AddRec,
          ConservativeResult.intersectWith(ConstantRange(Min, Max+1)));
      }
    }

    return setUnsignedRange(AddRec, ConservativeResult);
  }

  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    // For a SCEVUnknown, ask ValueTracking.
    APInt Mask = APInt::getAllOnesValue(BitWidth);
    APInt Zeros(BitWidth, 0), Ones(BitWidth, 0);
    ComputeMaskedBits(U->getValue(), Mask, Zeros, Ones, TD);
    if (Ones == ~Zeros + 1)
      return setUnsignedRange(U, ConservativeResult);
    return setUnsignedRange(U,
      ConservativeResult.intersectWith(ConstantRange(Ones, ~Zeros + 1)));
  }

  return setUnsignedRange(S, ConservativeResult);
}

/// getSignedRange - Determine the signed range for a particular SCEV.
///
ConstantRange
ScalarEvolution::getSignedRange(const SCEV *S) {
  // See if we've computed this range already.
  DenseMap<const SCEV *, ConstantRange>::iterator I = SignedRanges.find(S);
  if (I != SignedRanges.end())
    return I->second;

  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S))
    return setSignedRange(C, ConstantRange(C->getValue()->getValue()));

  unsigned BitWidth = getTypeSizeInBits(S->getType());
  ConstantRange ConservativeResult(BitWidth, /*isFullSet=*/true);

  // If the value has known zeros, the maximum signed value will have those
  // known zeros as well.
  uint32_t TZ = GetMinTrailingZeros(S);
  if (TZ != 0)
    ConservativeResult =
      ConstantRange(APInt::getSignedMinValue(BitWidth),
                    APInt::getSignedMaxValue(BitWidth).ashr(TZ).shl(TZ) + 1);

  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    ConstantRange X = getSignedRange(Add->getOperand(0));
    for (unsigned i = 1, e = Add->getNumOperands(); i != e; ++i)
      X = X.add(getSignedRange(Add->getOperand(i)));
    return setSignedRange(Add, ConservativeResult.intersectWith(X));
  }

  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(S)) {
    ConstantRange X = getSignedRange(Mul->getOperand(0));
    for (unsigned i = 1, e = Mul->getNumOperands(); i != e; ++i)
      X = X.multiply(getSignedRange(Mul->getOperand(i)));
    return setSignedRange(Mul, ConservativeResult.intersectWith(X));
  }

  if (const SCEVSMaxExpr *SMax = dyn_cast<SCEVSMaxExpr>(S)) {
    ConstantRange X = getSignedRange(SMax->getOperand(0));
    for (unsigned i = 1, e = SMax->getNumOperands(); i != e; ++i)
      X = X.smax(getSignedRange(SMax->getOperand(i)));
    return setSignedRange(SMax, ConservativeResult.intersectWith(X));
  }

  if (const SCEVUMaxExpr *UMax = dyn_cast<SCEVUMaxExpr>(S)) {
    ConstantRange X = getSignedRange(UMax->getOperand(0));
    for (unsigned i = 1, e = UMax->getNumOperands(); i != e; ++i)
      X = X.umax(getSignedRange(UMax->getOperand(i)));
    return setSignedRange(UMax, ConservativeResult.intersectWith(X));
  }

  if (const SCEVUDivExpr *UDiv = dyn_cast<SCEVUDivExpr>(S)) {
    ConstantRange X = getSignedRange(UDiv->getLHS());
    ConstantRange Y = getSignedRange(UDiv->getRHS());
    return setSignedRange(UDiv, ConservativeResult.intersectWith(X.udiv(Y)));
  }

  if (const SCEVZeroExtendExpr *ZExt = dyn_cast<SCEVZeroExtendExpr>(S)) {
    ConstantRange X = getSignedRange(ZExt->getOperand());
    return setSignedRange(ZExt,
      ConservativeResult.intersectWith(X.zeroExtend(BitWidth)));
  }

  if (const SCEVSignExtendExpr *SExt = dyn_cast<SCEVSignExtendExpr>(S)) {
    ConstantRange X = getSignedRange(SExt->getOperand());
    return setSignedRange(SExt,
      ConservativeResult.intersectWith(X.signExtend(BitWidth)));
  }

  if (const SCEVTruncateExpr *Trunc = dyn_cast<SCEVTruncateExpr>(S)) {
    ConstantRange X = getSignedRange(Trunc->getOperand());
    return setSignedRange(Trunc,
      ConservativeResult.intersectWith(X.truncate(BitWidth)));
  }

  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(S)) {
    // If there's no signed wrap, and all the operands have the same sign or
    // zero, the value won't ever change sign.
    if (AddRec->getNoWrapFlags(SCEV::FlagNSW)) {
      bool AllNonNeg = true;
      bool AllNonPos = true;
      for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i) {
        if (!isKnownNonNegative(AddRec->getOperand(i))) AllNonNeg = false;
        if (!isKnownNonPositive(AddRec->getOperand(i))) AllNonPos = false;
      }
      if (AllNonNeg)
        ConservativeResult = ConservativeResult.intersectWith(
          ConstantRange(APInt(BitWidth, 0),
                        APInt::getSignedMinValue(BitWidth)));
      else if (AllNonPos)
        ConservativeResult = ConservativeResult.intersectWith(
          ConstantRange(APInt::getSignedMinValue(BitWidth),
                        APInt(BitWidth, 1)));
    }

    // TODO: non-affine addrec
    if (AddRec->isAffine()) {
      Type *Ty = AddRec->getType();
      const SCEV *MaxBECount = getMaxBackedgeTakenCount(AddRec->getLoop());
      if (!isa<SCEVCouldNotCompute>(MaxBECount) &&
          getTypeSizeInBits(MaxBECount->getType()) <= BitWidth) {
        MaxBECount = getNoopOrZeroExtend(MaxBECount, Ty);

        const SCEV *Start = AddRec->getStart();
        const SCEV *Step = AddRec->getStepRecurrence(*this);

        ConstantRange StartRange = getSignedRange(Start);
        ConstantRange StepRange = getSignedRange(Step);
        ConstantRange MaxBECountRange = getUnsignedRange(MaxBECount);
        ConstantRange EndRange =
          StartRange.add(MaxBECountRange.multiply(StepRange));

        // Check for overflow. This must be done with ConstantRange arithmetic
        // because we could be called from within the ScalarEvolution overflow
        // checking code.
        ConstantRange ExtStartRange = StartRange.sextOrTrunc(BitWidth*2+1);
        ConstantRange ExtStepRange = StepRange.sextOrTrunc(BitWidth*2+1);
        ConstantRange ExtMaxBECountRange =
          MaxBECountRange.zextOrTrunc(BitWidth*2+1);
        ConstantRange ExtEndRange = EndRange.sextOrTrunc(BitWidth*2+1);
        if (ExtStartRange.add(ExtMaxBECountRange.multiply(ExtStepRange)) !=
            ExtEndRange)
          return setSignedRange(AddRec, ConservativeResult);

        APInt Min = APIntOps::smin(StartRange.getSignedMin(),
                                   EndRange.getSignedMin());
        APInt Max = APIntOps::smax(StartRange.getSignedMax(),
                                   EndRange.getSignedMax());
        if (Min.isMinSignedValue() && Max.isMaxSignedValue())
          return setSignedRange(AddRec, ConservativeResult);
        return setSignedRange(AddRec,
          ConservativeResult.intersectWith(ConstantRange(Min, Max+1)));
      }
    }

    return setSignedRange(AddRec, ConservativeResult);
  }

  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    // For a SCEVUnknown, ask ValueTracking.
    if (!U->getValue()->getType()->isIntegerTy() && !TD)
      return setSignedRange(U, ConservativeResult);
    unsigned NS = ComputeNumSignBits(U->getValue(), TD);
    if (NS == 1)
      return setSignedRange(U, ConservativeResult);
    return setSignedRange(U, ConservativeResult.intersectWith(
      ConstantRange(APInt::getSignedMinValue(BitWidth).ashr(NS - 1),
                    APInt::getSignedMaxValue(BitWidth).ashr(NS - 1)+1)));
  }

  return setSignedRange(S, ConservativeResult);
}

/// createSCEV - We know that there is no SCEV for the specified value.
/// Analyze the expression.
///
const SCEV *ScalarEvolution::createSCEV(Value *V) {
  if (!isSCEVable(V->getType()))
    return getUnknown(V);

  unsigned Opcode = Instruction::UserOp1;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    Opcode = I->getOpcode();

    // Don't attempt to analyze instructions in blocks that aren't
    // reachable. Such instructions don't matter, and they aren't required
    // to obey basic rules for definitions dominating uses which this
    // analysis depends on.
    if (!DT->isReachableFromEntry(I->getParent()))
      return getUnknown(V);
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    Opcode = CE->getOpcode();
  else if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
    return getConstant(CI);
  else if (isa<ConstantPointerNull>(V))
    return getConstant(V->getType(), 0);
  else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return GA->mayBeOverridden() ? getUnknown(V) : getSCEV(GA->getAliasee());
  else
    return getUnknown(V);

  Operator *U = cast<Operator>(V);
  switch (Opcode) {
  case Instruction::Add: {
    // The simple thing to do would be to just call getSCEV on both operands
    // and call getAddExpr with the result. However if we're looking at a
    // bunch of things all added together, this can be quite inefficient,
    // because it leads to N-1 getAddExpr calls for N ultimate operands.
    // Instead, gather up all the operands and make a single getAddExpr call.
    // LLVM IR canonical form means we need only traverse the left operands.
    SmallVector<const SCEV *, 4> AddOps;
    AddOps.push_back(getSCEV(U->getOperand(1)));
    for (Value *Op = U->getOperand(0); ; Op = U->getOperand(0)) {
      unsigned Opcode = Op->getValueID() - Value::InstructionVal;
      if (Opcode != Instruction::Add && Opcode != Instruction::Sub)
        break;
      U = cast<Operator>(Op);
      const SCEV *Op1 = getSCEV(U->getOperand(1));
      if (Opcode == Instruction::Sub)
        AddOps.push_back(getNegativeSCEV(Op1));
      else
        AddOps.push_back(Op1);
    }
    AddOps.push_back(getSCEV(U->getOperand(0)));
    SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap;
    OverflowingBinaryOperator *OBO = cast<OverflowingBinaryOperator>(V);
    if (OBO->hasNoSignedWrap())
      setFlags(Flags, SCEV::FlagNSW);
    if (OBO->hasNoUnsignedWrap())
      setFlags(Flags, SCEV::FlagNUW);
    return getAddExpr(AddOps, Flags);
  }
  case Instruction::Mul: {
    // See the Add code above.
    SmallVector<const SCEV *, 4> MulOps;
    MulOps.push_back(getSCEV(U->getOperand(1)));
    for (Value *Op = U->getOperand(0);
         Op->getValueID() == Instruction::Mul + Value::InstructionVal;
         Op = U->getOperand(0)) {
      U = cast<Operator>(Op);
      MulOps.push_back(getSCEV(U->getOperand(1)));
    }
    MulOps.push_back(getSCEV(U->getOperand(0)));
    return getMulExpr(MulOps);
  }
  case Instruction::UDiv:
    return getUDivExpr(getSCEV(U->getOperand(0)),
                       getSCEV(U->getOperand(1)));
  case Instruction::Sub:
    return getMinusSCEV(getSCEV(U->getOperand(0)),
                        getSCEV(U->getOperand(1)));
  case Instruction::And:
    // For an expression like x&255 that merely masks off the high bits,
    // use zext(trunc(x)) as the SCEV expression.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      if (CI->isNullValue())
        return getSCEV(U->getOperand(1));
      if (CI->isAllOnesValue())
        return getSCEV(U->getOperand(0));
      const APInt &A = CI->getValue();

      // Instcombine's ShrinkDemandedConstant may strip bits out of
      // constants, obscuring what would otherwise be a low-bits mask.
      // Use ComputeMaskedBits to compute what ShrinkDemandedConstant
      // knew about to reconstruct a low-bits mask value.
      unsigned LZ = A.countLeadingZeros();
      unsigned BitWidth = A.getBitWidth();
      APInt AllOnes = APInt::getAllOnesValue(BitWidth);
      APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
      ComputeMaskedBits(U->getOperand(0), AllOnes, KnownZero, KnownOne, TD);

      APInt EffectiveMask = APInt::getLowBitsSet(BitWidth, BitWidth - LZ);

      if (LZ != 0 && !((~A & ~KnownZero) & EffectiveMask))
        return
          getZeroExtendExpr(getTruncateExpr(getSCEV(U->getOperand(0)),
                                IntegerType::get(getContext(), BitWidth - LZ)),
                            U->getType());
    }
    break;

  case Instruction::Or:
    // If the RHS of the Or is a constant, we may have something like:
    // X*4+1 which got turned into X*4|1.  Handle this as an Add so loop
    // optimizations will transparently handle this case.
    //
    // In order for this transformation to be safe, the LHS must be of the
    // form X*(2^n) and the Or constant must be less than 2^n.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      const SCEV *LHS = getSCEV(U->getOperand(0));
      const APInt &CIVal = CI->getValue();
      if (GetMinTrailingZeros(LHS) >=
          (CIVal.getBitWidth() - CIVal.countLeadingZeros())) {
        // Build a plain add SCEV.
        const SCEV *S = getAddExpr(LHS, getSCEV(CI));
        // If the LHS of the add was an addrec and it has no-wrap flags,
        // transfer the no-wrap flags, since an or won't introduce a wrap.
        if (const SCEVAddRecExpr *NewAR = dyn_cast<SCEVAddRecExpr>(S)) {
          const SCEVAddRecExpr *OldAR = cast<SCEVAddRecExpr>(LHS);
          const_cast<SCEVAddRecExpr *>(NewAR)->setNoWrapFlags(
            OldAR->getNoWrapFlags());
        }
        return S;
      }
    }
    break;
  case Instruction::Xor:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1))) {
      // If the RHS of the xor is a signbit, then this is just an add.
      // Instcombine turns add of signbit into xor as a strength reduction step.
      if (CI->getValue().isSignBit())
        return getAddExpr(getSCEV(U->getOperand(0)),
                          getSCEV(U->getOperand(1)));

      // If the RHS of xor is -1, then this is a not operation.
      if (CI->isAllOnesValue())
        return getNotSCEV(getSCEV(U->getOperand(0)));

      // Model xor(and(x, C), C) as and(~x, C), if C is a low-bits mask.
      // This is a variant of the check for xor with -1, and it handles
      // the case where instcombine has trimmed non-demanded bits out
      // of an xor with -1.
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(U->getOperand(0)))
        if (ConstantInt *LCI = dyn_cast<ConstantInt>(BO->getOperand(1)))
          if (BO->getOpcode() == Instruction::And &&
              LCI->getValue() == CI->getValue())
            if (const SCEVZeroExtendExpr *Z =
                  dyn_cast<SCEVZeroExtendExpr>(getSCEV(U->getOperand(0)))) {
              Type *UTy = U->getType();
              const SCEV *Z0 = Z->getOperand();
              Type *Z0Ty = Z0->getType();
              unsigned Z0TySize = getTypeSizeInBits(Z0Ty);

              // If C is a low-bits mask, the zero extend is serving to
              // mask off the high bits. Complement the operand and
              // re-apply the zext.
              if (APIntOps::isMask(Z0TySize, CI->getValue()))
                return getZeroExtendExpr(getNotSCEV(Z0), UTy);

              // If C is a single bit, it may be in the sign-bit position
              // before the zero-extend. In this case, represent the xor
              // using an add, which is equivalent, and re-apply the zext.
              APInt Trunc = CI->getValue().trunc(Z0TySize);
              if (Trunc.zext(getTypeSizeInBits(UTy)) == CI->getValue() &&
                  Trunc.isSignBit())
                return getZeroExtendExpr(getAddExpr(Z0, getConstant(Trunc)),
                                         UTy);
            }
    }
    break;

  case Instruction::Shl:
    // Turn shift left of a constant amount into a multiply.
    if (ConstantInt *SA = dyn_cast<ConstantInt>(U->getOperand(1))) {
      uint32_t BitWidth = cast<IntegerType>(U->getType())->getBitWidth();

      // If the shift count is not less than the bitwidth, the result of
      // the shift is undefined. Don't try to analyze it, because the
      // resolution chosen here may differ from the resolution chosen in
      // other parts of the compiler.
      if (SA->getValue().uge(BitWidth))
        break;

      Constant *X = ConstantInt::get(getContext(),
        APInt(BitWidth, 1).shl(SA->getZExtValue()));
      return getMulExpr(getSCEV(U->getOperand(0)), getSCEV(X));
    }
    break;

  case Instruction::LShr:
    // Turn logical shift right of a constant into a unsigned divide.
    if (ConstantInt *SA = dyn_cast<ConstantInt>(U->getOperand(1))) {
      uint32_t BitWidth = cast<IntegerType>(U->getType())->getBitWidth();

      // If the shift count is not less than the bitwidth, the result of
      // the shift is undefined. Don't try to analyze it, because the
      // resolution chosen here may differ from the resolution chosen in
      // other parts of the compiler.
      if (SA->getValue().uge(BitWidth))
        break;

      Constant *X = ConstantInt::get(getContext(),
        APInt(BitWidth, 1).shl(SA->getZExtValue()));
      return getUDivExpr(getSCEV(U->getOperand(0)), getSCEV(X));
    }
    break;

  case Instruction::AShr:
    // For a two-shift sext-inreg, use sext(trunc(x)) as the SCEV expression.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(U->getOperand(1)))
      if (Operator *L = dyn_cast<Operator>(U->getOperand(0)))
        if (L->getOpcode() == Instruction::Shl &&
            L->getOperand(1) == U->getOperand(1)) {
          uint64_t BitWidth = getTypeSizeInBits(U->getType());

          // If the shift count is not less than the bitwidth, the result of
          // the shift is undefined. Don't try to analyze it, because the
          // resolution chosen here may differ from the resolution chosen in
          // other parts of the compiler.
          if (CI->getValue().uge(BitWidth))
            break;

          uint64_t Amt = BitWidth - CI->getZExtValue();
          if (Amt == BitWidth)
            return getSCEV(L->getOperand(0));       // shift by zero --> noop
          return
            getSignExtendExpr(getTruncateExpr(getSCEV(L->getOperand(0)),
                                              IntegerType::get(getContext(),
                                                               Amt)),
                              U->getType());
        }
    break;

  case Instruction::Trunc:
    return getTruncateExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::ZExt:
    return getZeroExtendExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::SExt:
    return getSignExtendExpr(getSCEV(U->getOperand(0)), U->getType());

  case Instruction::BitCast:
    // BitCasts are no-op casts so we just eliminate the cast.
    if (isSCEVable(U->getType()) && isSCEVable(U->getOperand(0)->getType()))
      return getSCEV(U->getOperand(0));
    break;

  // It's tempting to handle inttoptr and ptrtoint as no-ops, however this can
  // lead to pointer expressions which cannot safely be expanded to GEPs,
  // because ScalarEvolution doesn't respect the GEP aliasing rules when
  // simplifying integer expressions.

  case Instruction::GetElementPtr:
    return createNodeForGEP(cast<GEPOperator>(U));

  case Instruction::PHI:
    return createNodeForPHI(cast<PHINode>(U));

  case Instruction::Select:
    // This could be a smax or umax that was lowered earlier.
    // Try to recover it.
    if (ICmpInst *ICI = dyn_cast<ICmpInst>(U->getOperand(0))) {
      Value *LHS = ICI->getOperand(0);
      Value *RHS = ICI->getOperand(1);
      switch (ICI->getPredicate()) {
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_SLE:
        std::swap(LHS, RHS);
        // fall through
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_SGE:
        // a >s b ? a+x : b+x  ->  smax(a, b)+x
        // a >s b ? b+x : a+x  ->  smin(a, b)+x
        if (LHS->getType() == U->getType()) {
          const SCEV *LS = getSCEV(LHS);
          const SCEV *RS = getSCEV(RHS);
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, LS);
          const SCEV *RDiff = getMinusSCEV(RA, RS);
          if (LDiff == RDiff)
            return getAddExpr(getSMaxExpr(LS, RS), LDiff);
          LDiff = getMinusSCEV(LA, RS);
          RDiff = getMinusSCEV(RA, LS);
          if (LDiff == RDiff)
            return getAddExpr(getSMinExpr(LS, RS), LDiff);
        }
        break;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_ULE:
        std::swap(LHS, RHS);
        // fall through
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_UGE:
        // a >u b ? a+x : b+x  ->  umax(a, b)+x
        // a >u b ? b+x : a+x  ->  umin(a, b)+x
        if (LHS->getType() == U->getType()) {
          const SCEV *LS = getSCEV(LHS);
          const SCEV *RS = getSCEV(RHS);
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, LS);
          const SCEV *RDiff = getMinusSCEV(RA, RS);
          if (LDiff == RDiff)
            return getAddExpr(getUMaxExpr(LS, RS), LDiff);
          LDiff = getMinusSCEV(LA, RS);
          RDiff = getMinusSCEV(RA, LS);
          if (LDiff == RDiff)
            return getAddExpr(getUMinExpr(LS, RS), LDiff);
        }
        break;
      case ICmpInst::ICMP_NE:
        // n != 0 ? n+x : 1+x  ->  umax(n, 1)+x
        if (LHS->getType() == U->getType() &&
            isa<ConstantInt>(RHS) &&
            cast<ConstantInt>(RHS)->isZero()) {
          const SCEV *One = getConstant(LHS->getType(), 1);
          const SCEV *LS = getSCEV(LHS);
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, LS);
          const SCEV *RDiff = getMinusSCEV(RA, One);
          if (LDiff == RDiff)
            return getAddExpr(getUMaxExpr(One, LS), LDiff);
        }
        break;
      case ICmpInst::ICMP_EQ:
        // n == 0 ? 1+x : n+x  ->  umax(n, 1)+x
        if (LHS->getType() == U->getType() &&
            isa<ConstantInt>(RHS) &&
            cast<ConstantInt>(RHS)->isZero()) {
          const SCEV *One = getConstant(LHS->getType(), 1);
          const SCEV *LS = getSCEV(LHS);
          const SCEV *LA = getSCEV(U->getOperand(1));
          const SCEV *RA = getSCEV(U->getOperand(2));
          const SCEV *LDiff = getMinusSCEV(LA, One);
          const SCEV *RDiff = getMinusSCEV(RA, LS);
          if (LDiff == RDiff)
            return getAddExpr(getUMaxExpr(One, LS), LDiff);
        }
        break;
      default:
        break;
      }
    }

  default: // We cannot analyze this expression.
    break;
  }

  return getUnknown(V);
}



//===----------------------------------------------------------------------===//
//                   Iteration Count Computation Code
//

/// getSmallConstantTripCount - Returns the maximum trip count of this loop as a
/// normal unsigned value, if possible. Returns 0 if the trip count is unknown
/// or not constant. Will also return 0 if the maximum trip count is very large
/// (>= 2^32)
unsigned ScalarEvolution::getSmallConstantTripCount(Loop *L,
                                                    BasicBlock *ExitBlock) {
  const SCEVConstant *ExitCount =
    dyn_cast<SCEVConstant>(getExitCount(L, ExitBlock));
  if (!ExitCount)
    return 0;

  ConstantInt *ExitConst = ExitCount->getValue();

  // Guard against huge trip counts.
  if (ExitConst->getValue().getActiveBits() > 32)
    return 0;

  // In case of integer overflow, this returns 0, which is correct.
  return ((unsigned)ExitConst->getZExtValue()) + 1;
}

/// getSmallConstantTripMultiple - Returns the largest constant divisor of the
/// trip count of this loop as a normal unsigned value, if possible. This
/// means that the actual trip count is always a multiple of the returned
/// value (don't forget the trip count could very well be zero as well!).
///
/// Returns 1 if the trip count is unknown or not guaranteed to be the
/// multiple of a constant (which is also the case if the trip count is simply
/// constant, use getSmallConstantTripCount for that case), Will also return 1
/// if the trip count is very large (>= 2^32).
unsigned ScalarEvolution::getSmallConstantTripMultiple(Loop *L,
                                                       BasicBlock *ExitBlock) {
  const SCEV *ExitCount = getExitCount(L, ExitBlock);
  if (ExitCount == getCouldNotCompute())
    return 1;

  // Get the trip count from the BE count by adding 1.
  const SCEV *TCMul = getAddExpr(ExitCount,
                                 getConstant(ExitCount->getType(), 1));
  // FIXME: SCEV distributes multiplication as V1*C1 + V2*C1. We could attempt
  // to factor simple cases.
  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(TCMul))
    TCMul = Mul->getOperand(0);

  const SCEVConstant *MulC = dyn_cast<SCEVConstant>(TCMul);
  if (!MulC)
    return 1;

  ConstantInt *Result = MulC->getValue();

  // Guard against huge trip counts.
  if (!Result || Result->getValue().getActiveBits() > 32)
    return 1;

  return (unsigned)Result->getZExtValue();
}

// getExitCount - Get the expression for the number of loop iterations for which
// this loop is guaranteed not to exit via ExitintBlock. Otherwise return
// SCEVCouldNotCompute.
const SCEV *ScalarEvolution::getExitCount(Loop *L, BasicBlock *ExitingBlock) {
  return getBackedgeTakenInfo(L).getExact(ExitingBlock, this);
}

/// getBackedgeTakenCount - If the specified loop has a predictable
/// backedge-taken count, return it, otherwise return a SCEVCouldNotCompute
/// object. The backedge-taken count is the number of times the loop header
/// will be branched to from within the loop. This is one less than the
/// trip count of the loop, since it doesn't count the first iteration,
/// when the header is branched to from outside the loop.
///
/// Note that it is not valid to call this method on a loop without a
/// loop-invariant backedge-taken count (see
/// hasLoopInvariantBackedgeTakenCount).
///
const SCEV *ScalarEvolution::getBackedgeTakenCount(const Loop *L) {
  return getBackedgeTakenInfo(L).getExact(this);
}

/// getMaxBackedgeTakenCount - Similar to getBackedgeTakenCount, except
/// return the least SCEV value that is known never to be less than the
/// actual backedge taken count.
const SCEV *ScalarEvolution::getMaxBackedgeTakenCount(const Loop *L) {
  return getBackedgeTakenInfo(L).getMax(this);
}

/// PushLoopPHIs - Push PHI nodes in the header of the given loop
/// onto the given Worklist.
static void
PushLoopPHIs(const Loop *L, SmallVectorImpl<Instruction *> &Worklist) {
  BasicBlock *Header = L->getHeader();

  // Push all Loop-header PHIs onto the Worklist stack.
  for (BasicBlock::iterator I = Header->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    Worklist.push_back(PN);
}

const ScalarEvolution::BackedgeTakenInfo &
ScalarEvolution::getBackedgeTakenInfo(const Loop *L) {
  // Initially insert an invalid entry for this loop. If the insertion
  // succeeds, proceed to actually compute a backedge-taken count and
  // update the value. The temporary CouldNotCompute value tells SCEV
  // code elsewhere that it shouldn't attempt to request a new
  // backedge-taken count, which could result in infinite recursion.
  std::pair<DenseMap<const Loop *, BackedgeTakenInfo>::iterator, bool> Pair =
    BackedgeTakenCounts.insert(std::make_pair(L, BackedgeTakenInfo()));
  if (!Pair.second)
    return Pair.first->second;

  // ComputeBackedgeTakenCount may allocate memory for its result. Inserting it
  // into the BackedgeTakenCounts map transfers ownership. Otherwise, the result
  // must be cleared in this scope.
  BackedgeTakenInfo Result = ComputeBackedgeTakenCount(L);

  if (Result.getExact(this) != getCouldNotCompute()) {
    assert(isLoopInvariant(Result.getExact(this), L) &&
           isLoopInvariant(Result.getMax(this), L) &&
           "Computed backedge-taken count isn't loop invariant for loop!");
    ++NumTripCountsComputed;
  }
  else if (Result.getMax(this) == getCouldNotCompute() &&
           isa<PHINode>(L->getHeader()->begin())) {
    // Only count loops that have phi nodes as not being computable.
    ++NumTripCountsNotComputed;
  }

  // Now that we know more about the trip count for this loop, forget any
  // existing SCEV values for PHI nodes in this loop since they are only
  // conservative estimates made without the benefit of trip count
  // information. This is similar to the code in forgetLoop, except that
  // it handles SCEVUnknown PHI nodes specially.
  if (Result.hasAnyInfo()) {
    SmallVector<Instruction *, 16> Worklist;
    PushLoopPHIs(L, Worklist);

    SmallPtrSet<Instruction *, 8> Visited;
    while (!Worklist.empty()) {
      Instruction *I = Worklist.pop_back_val();
      if (!Visited.insert(I)) continue;

      ValueExprMapType::iterator It =
        ValueExprMap.find(static_cast<Value *>(I));
      if (It != ValueExprMap.end()) {
        const SCEV *Old = It->second;

        // SCEVUnknown for a PHI either means that it has an unrecognized
        // structure, or it's a PHI that's in the progress of being computed
        // by createNodeForPHI.  In the former case, additional loop trip
        // count information isn't going to change anything. In the later
        // case, createNodeForPHI will perform the necessary updates on its
        // own when it gets to that point.
        if (!isa<PHINode>(I) || !isa<SCEVUnknown>(Old)) {
          forgetMemoizedResults(Old);
          ValueExprMap.erase(It);
        }
        if (PHINode *PN = dyn_cast<PHINode>(I))
          ConstantEvolutionLoopExitValue.erase(PN);
      }

      PushDefUseChildren(I, Worklist);
    }
  }

  // Re-lookup the insert position, since the call to
  // ComputeBackedgeTakenCount above could result in a
  // recusive call to getBackedgeTakenInfo (on a different
  // loop), which would invalidate the iterator computed
  // earlier.
  return BackedgeTakenCounts.find(L)->second = Result;
}

/// forgetLoop - This method should be called by the client when it has
/// changed a loop in a way that may effect ScalarEvolution's ability to
/// compute a trip count, or if the loop is deleted.
void ScalarEvolution::forgetLoop(const Loop *L) {
  // Drop any stored trip count value.
  DenseMap<const Loop*, BackedgeTakenInfo>::iterator BTCPos =
    BackedgeTakenCounts.find(L);
  if (BTCPos != BackedgeTakenCounts.end()) {
    BTCPos->second.clear();
    BackedgeTakenCounts.erase(BTCPos);
  }

  // Drop information about expressions based on loop-header PHIs.
  SmallVector<Instruction *, 16> Worklist;
  PushLoopPHIs(L, Worklist);

  SmallPtrSet<Instruction *, 8> Visited;
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!Visited.insert(I)) continue;

    ValueExprMapType::iterator It = ValueExprMap.find(static_cast<Value *>(I));
    if (It != ValueExprMap.end()) {
      forgetMemoizedResults(It->second);
      ValueExprMap.erase(It);
      if (PHINode *PN = dyn_cast<PHINode>(I))
        ConstantEvolutionLoopExitValue.erase(PN);
    }

    PushDefUseChildren(I, Worklist);
  }

  // Forget all contained loops too, to avoid dangling entries in the
  // ValuesAtScopes map.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    forgetLoop(*I);
}

/// forgetValue - This method should be called by the client when it has
/// changed a value in a way that may effect its value, or which may
/// disconnect it from a def-use chain linking it to a loop.
void ScalarEvolution::forgetValue(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return;

  // Drop information about expressions based on loop-header PHIs.
  SmallVector<Instruction *, 16> Worklist;
  Worklist.push_back(I);

  SmallPtrSet<Instruction *, 8> Visited;
  while (!Worklist.empty()) {
    I = Worklist.pop_back_val();
    if (!Visited.insert(I)) continue;

    ValueExprMapType::iterator It = ValueExprMap.find(static_cast<Value *>(I));
    if (It != ValueExprMap.end()) {
      forgetMemoizedResults(It->second);
      ValueExprMap.erase(It);
      if (PHINode *PN = dyn_cast<PHINode>(I))
        ConstantEvolutionLoopExitValue.erase(PN);
    }

    PushDefUseChildren(I, Worklist);
  }
}

/// getExact - Get the exact loop backedge taken count considering all loop
/// exits. If all exits are computable, this is the minimum computed count.
const SCEV *
ScalarEvolution::BackedgeTakenInfo::getExact(ScalarEvolution *SE) const {
  // If any exits were not computable, the loop is not computable.
  if (!ExitNotTaken.isCompleteList()) return SE->getCouldNotCompute();

  // We need at least one computable exit.
  if (!ExitNotTaken.ExitingBlock) return SE->getCouldNotCompute();
  assert(ExitNotTaken.ExactNotTaken && "uninitialized not-taken info");

  const SCEV *BECount = 0;
  for (const ExitNotTakenInfo *ENT = &ExitNotTaken;
       ENT != 0; ENT = ENT->getNextExit()) {

    assert(ENT->ExactNotTaken != SE->getCouldNotCompute() && "bad exit SCEV");

    if (!BECount)
      BECount = ENT->ExactNotTaken;
    else
      BECount = SE->getUMinFromMismatchedTypes(BECount, ENT->ExactNotTaken);
  }
  assert(BECount && "Invalid not taken count for loop exit");
  return BECount;
}

/// getExact - Get the exact not taken count for this loop exit.
const SCEV *
ScalarEvolution::BackedgeTakenInfo::getExact(BasicBlock *ExitingBlock,
                                             ScalarEvolution *SE) const {
  for (const ExitNotTakenInfo *ENT = &ExitNotTaken;
       ENT != 0; ENT = ENT->getNextExit()) {

    if (ENT->ExitingBlock == ExitingBlock)
      return ENT->ExactNotTaken;
  }
  return SE->getCouldNotCompute();
}

/// getMax - Get the max backedge taken count for the loop.
const SCEV *
ScalarEvolution::BackedgeTakenInfo::getMax(ScalarEvolution *SE) const {
  return Max ? Max : SE->getCouldNotCompute();
}

/// Allocate memory for BackedgeTakenInfo and copy the not-taken count of each
/// computable exit into a persistent ExitNotTakenInfo array.
ScalarEvolution::BackedgeTakenInfo::BackedgeTakenInfo(
  SmallVectorImpl< std::pair<BasicBlock *, const SCEV *> > &ExitCounts,
  bool Complete, const SCEV *MaxCount) : Max(MaxCount) {

  if (!Complete)
    ExitNotTaken.setIncomplete();

  unsigned NumExits = ExitCounts.size();
  if (NumExits == 0) return;

  ExitNotTaken.ExitingBlock = ExitCounts[0].first;
  ExitNotTaken.ExactNotTaken = ExitCounts[0].second;
  if (NumExits == 1) return;

  // Handle the rare case of multiple computable exits.
  ExitNotTakenInfo *ENT = new ExitNotTakenInfo[NumExits-1];

  ExitNotTakenInfo *PrevENT = &ExitNotTaken;
  for (unsigned i = 1; i < NumExits; ++i, PrevENT = ENT, ++ENT) {
    PrevENT->setNextExit(ENT);
    ENT->ExitingBlock = ExitCounts[i].first;
    ENT->ExactNotTaken = ExitCounts[i].second;
  }
}

/// clear - Invalidate this result and free the ExitNotTakenInfo array.
void ScalarEvolution::BackedgeTakenInfo::clear() {
  ExitNotTaken.ExitingBlock = 0;
  ExitNotTaken.ExactNotTaken = 0;
  delete[] ExitNotTaken.getNextExit();
}

/// ComputeBackedgeTakenCount - Compute the number of times the backedge
/// of the specified loop will execute.
ScalarEvolution::BackedgeTakenInfo
ScalarEvolution::ComputeBackedgeTakenCount(const Loop *L) {
  SmallVector<BasicBlock *, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  // Examine all exits and pick the most conservative values.
  const SCEV *MaxBECount = getCouldNotCompute();
  bool CouldComputeBECount = true;
  SmallVector<std::pair<BasicBlock *, const SCEV *>, 4> ExitCounts;
  for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
    ExitLimit EL = ComputeExitLimit(L, ExitingBlocks[i]);
    if (EL.Exact == getCouldNotCompute())
      // We couldn't compute an exact value for this exit, so
      // we won't be able to compute an exact value for the loop.
      CouldComputeBECount = false;
    else
      ExitCounts.push_back(std::make_pair(ExitingBlocks[i], EL.Exact));

    if (MaxBECount == getCouldNotCompute())
      MaxBECount = EL.Max;
    else if (EL.Max != getCouldNotCompute())
      MaxBECount = getUMinFromMismatchedTypes(MaxBECount, EL.Max);
  }

  return BackedgeTakenInfo(ExitCounts, CouldComputeBECount, MaxBECount);
}

/// ComputeExitLimit - Compute the number of times the backedge of the specified
/// loop will execute if it exits via the specified block.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeExitLimit(const Loop *L, BasicBlock *ExitingBlock) {

  // Okay, we've chosen an exiting block.  See what condition causes us to
  // exit at this block.
  //
  // FIXME: we should be able to handle switch instructions (with a single exit)
  BranchInst *ExitBr = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  if (ExitBr == 0) return getCouldNotCompute();
  assert(ExitBr->isConditional() && "If unconditional, it can't be in loop!");

  // At this point, we know we have a conditional branch that determines whether
  // the loop is exited.  However, we don't know if the branch is executed each
  // time through the loop.  If not, then the execution count of the branch will
  // not be equal to the trip count of the loop.
  //
  // Currently we check for this by checking to see if the Exit branch goes to
  // the loop header.  If so, we know it will always execute the same number of
  // times as the loop.  We also handle the case where the exit block *is* the
  // loop header.  This is common for un-rotated loops.
  //
  // If both of those tests fail, walk up the unique predecessor chain to the
  // header, stopping if there is an edge that doesn't exit the loop. If the
  // header is reached, the execution count of the branch will be equal to the
  // trip count of the loop.
  //
  //  More extensive analysis could be done to handle more cases here.
  //
  if (ExitBr->getSuccessor(0) != L->getHeader() &&
      ExitBr->getSuccessor(1) != L->getHeader() &&
      ExitBr->getParent() != L->getHeader()) {
    // The simple checks failed, try climbing the unique predecessor chain
    // up to the header.
    bool Ok = false;
    for (BasicBlock *BB = ExitBr->getParent(); BB; ) {
      BasicBlock *Pred = BB->getUniquePredecessor();
      if (!Pred)
        return getCouldNotCompute();
      TerminatorInst *PredTerm = Pred->getTerminator();
      for (unsigned i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i) {
        BasicBlock *PredSucc = PredTerm->getSuccessor(i);
        if (PredSucc == BB)
          continue;
        // If the predecessor has a successor that isn't BB and isn't
        // outside the loop, assume the worst.
        if (L->contains(PredSucc))
          return getCouldNotCompute();
      }
      if (Pred == L->getHeader()) {
        Ok = true;
        break;
      }
      BB = Pred;
    }
    if (!Ok)
      return getCouldNotCompute();
  }

  // Proceed to the next level to examine the exit condition expression.
  return ComputeExitLimitFromCond(L, ExitBr->getCondition(),
                                  ExitBr->getSuccessor(0),
                                  ExitBr->getSuccessor(1));
}

/// ComputeExitLimitFromCond - Compute the number of times the
/// backedge of the specified loop will execute if its exit condition
/// were a conditional branch of ExitCond, TBB, and FBB.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeExitLimitFromCond(const Loop *L,
                                          Value *ExitCond,
                                          BasicBlock *TBB,
                                          BasicBlock *FBB) {
  // Check if the controlling expression for this loop is an And or Or.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(ExitCond)) {
    if (BO->getOpcode() == Instruction::And) {
      // Recurse on the operands of the and.
      ExitLimit EL0 = ComputeExitLimitFromCond(L, BO->getOperand(0), TBB, FBB);
      ExitLimit EL1 = ComputeExitLimitFromCond(L, BO->getOperand(1), TBB, FBB);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (L->contains(TBB)) {
        // Both conditions must be true for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.Exact == getCouldNotCompute() ||
            EL1.Exact == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount = getUMinFromMismatchedTypes(EL0.Exact, EL1.Exact);
        if (EL0.Max == getCouldNotCompute())
          MaxBECount = EL1.Max;
        else if (EL1.Max == getCouldNotCompute())
          MaxBECount = EL0.Max;
        else
          MaxBECount = getUMinFromMismatchedTypes(EL0.Max, EL1.Max);
      } else {
        // Both conditions must be true at the same time for the loop to exit.
        // For now, be conservative.
        assert(L->contains(FBB) && "Loop block has no successor in loop!");
        if (EL0.Max == EL1.Max)
          MaxBECount = EL0.Max;
        if (EL0.Exact == EL1.Exact)
          BECount = EL0.Exact;
      }

      return ExitLimit(BECount, MaxBECount);
    }
    if (BO->getOpcode() == Instruction::Or) {
      // Recurse on the operands of the or.
      ExitLimit EL0 = ComputeExitLimitFromCond(L, BO->getOperand(0), TBB, FBB);
      ExitLimit EL1 = ComputeExitLimitFromCond(L, BO->getOperand(1), TBB, FBB);
      const SCEV *BECount = getCouldNotCompute();
      const SCEV *MaxBECount = getCouldNotCompute();
      if (L->contains(FBB)) {
        // Both conditions must be false for the loop to continue executing.
        // Choose the less conservative count.
        if (EL0.Exact == getCouldNotCompute() ||
            EL1.Exact == getCouldNotCompute())
          BECount = getCouldNotCompute();
        else
          BECount = getUMinFromMismatchedTypes(EL0.Exact, EL1.Exact);
        if (EL0.Max == getCouldNotCompute())
          MaxBECount = EL1.Max;
        else if (EL1.Max == getCouldNotCompute())
          MaxBECount = EL0.Max;
        else
          MaxBECount = getUMinFromMismatchedTypes(EL0.Max, EL1.Max);
      } else {
        // Both conditions must be false at the same time for the loop to exit.
        // For now, be conservative.
        assert(L->contains(TBB) && "Loop block has no successor in loop!");
        if (EL0.Max == EL1.Max)
          MaxBECount = EL0.Max;
        if (EL0.Exact == EL1.Exact)
          BECount = EL0.Exact;
      }

      return ExitLimit(BECount, MaxBECount);
    }
  }

  // With an icmp, it may be feasible to compute an exact backedge-taken count.
  // Proceed to the next level to examine the icmp.
  if (ICmpInst *ExitCondICmp = dyn_cast<ICmpInst>(ExitCond))
    return ComputeExitLimitFromICmp(L, ExitCondICmp, TBB, FBB);

  // Check for a constant condition. These are normally stripped out by
  // SimplifyCFG, but ScalarEvolution may be used by a pass which wishes to
  // preserve the CFG and is temporarily leaving constant conditions
  // in place.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(ExitCond)) {
    if (L->contains(FBB) == !CI->getZExtValue())
      // The backedge is always taken.
      return getCouldNotCompute();
    else
      // The backedge is never taken.
      return getConstant(CI->getType(), 0);
  }

  // If it's not an integer or pointer comparison then compute it the hard way.
  return ComputeExitCountExhaustively(L, ExitCond, !L->contains(TBB));
}

/// ComputeExitLimitFromICmp - Compute the number of times the
/// backedge of the specified loop will execute if its exit condition
/// were a conditional branch of the ICmpInst ExitCond, TBB, and FBB.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeExitLimitFromICmp(const Loop *L,
                                          ICmpInst *ExitCond,
                                          BasicBlock *TBB,
                                          BasicBlock *FBB) {

  // If the condition was exit on true, convert the condition to exit on false
  ICmpInst::Predicate Cond;
  if (!L->contains(FBB))
    Cond = ExitCond->getPredicate();
  else
    Cond = ExitCond->getInversePredicate();

  // Handle common loops like: for (X = "string"; *X; ++X)
  if (LoadInst *LI = dyn_cast<LoadInst>(ExitCond->getOperand(0)))
    if (Constant *RHS = dyn_cast<Constant>(ExitCond->getOperand(1))) {
      ExitLimit ItCnt =
        ComputeLoadConstantCompareExitLimit(LI, RHS, L, Cond);
      if (ItCnt.hasAnyInfo())
        return ItCnt;
    }

  const SCEV *LHS = getSCEV(ExitCond->getOperand(0));
  const SCEV *RHS = getSCEV(ExitCond->getOperand(1));

  // Try to evaluate any dependencies out of the loop.
  LHS = getSCEVAtScope(LHS, L);
  RHS = getSCEVAtScope(RHS, L);

  // At this point, we would like to compute how many iterations of the
  // loop the predicate will return true for these inputs.
  if (isLoopInvariant(LHS, L) && !isLoopInvariant(RHS, L)) {
    // If there is a loop-invariant, force it into the RHS.
    std::swap(LHS, RHS);
    Cond = ICmpInst::getSwappedPredicate(Cond);
  }

  // Simplify the operands before analyzing them.
  (void)SimplifyICmpOperands(Cond, LHS, RHS);

  // If we have a comparison of a chrec against a constant, try to use value
  // ranges to answer this query.
  if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS))
    if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS))
      if (AddRec->getLoop() == L) {
        // Form the constant range.
        ConstantRange CompRange(
            ICmpInst::makeConstantRange(Cond, RHSC->getValue()->getValue()));

        const SCEV *Ret = AddRec->getNumIterationsInRange(CompRange, *this);
        if (!isa<SCEVCouldNotCompute>(Ret)) return Ret;
      }

  switch (Cond) {
  case ICmpInst::ICMP_NE: {                     // while (X != Y)
    // Convert to: while (X-Y != 0)
    ExitLimit EL = HowFarToZero(getMinusSCEV(LHS, RHS), L);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_EQ: {                     // while (X == Y)
    // Convert to: while (X-Y == 0)
    ExitLimit EL = HowFarToNonZero(getMinusSCEV(LHS, RHS), L);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SLT: {
    ExitLimit EL = HowManyLessThans(LHS, RHS, L, true);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_SGT: {
    ExitLimit EL = HowManyLessThans(getNotSCEV(LHS),
                                             getNotSCEV(RHS), L, true);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_ULT: {
    ExitLimit EL = HowManyLessThans(LHS, RHS, L, false);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  case ICmpInst::ICMP_UGT: {
    ExitLimit EL = HowManyLessThans(getNotSCEV(LHS),
                                             getNotSCEV(RHS), L, false);
    if (EL.hasAnyInfo()) return EL;
    break;
  }
  default:
#if 0
    dbgs() << "ComputeBackedgeTakenCount ";
    if (ExitCond->getOperand(0)->getType()->isUnsigned())
      dbgs() << "[unsigned] ";
    dbgs() << *LHS << "   "
         << Instruction::getOpcodeName(Instruction::ICmp)
         << "   " << *RHS << "\n";
#endif
    break;
  }
  return ComputeExitCountExhaustively(L, ExitCond, !L->contains(TBB));
}

static ConstantInt *
EvaluateConstantChrecAtConstant(const SCEVAddRecExpr *AddRec, ConstantInt *C,
                                ScalarEvolution &SE) {
  const SCEV *InVal = SE.getConstant(C);
  const SCEV *Val = AddRec->evaluateAtIteration(InVal, SE);
  assert(isa<SCEVConstant>(Val) &&
         "Evaluation of SCEV at constant didn't fold correctly?");
  return cast<SCEVConstant>(Val)->getValue();
}

/// GetAddressedElementFromGlobal - Given a global variable with an initializer
/// and a GEP expression (missing the pointer index) indexing into it, return
/// the addressed element of the initializer or null if the index expression is
/// invalid.
static Constant *
GetAddressedElementFromGlobal(GlobalVariable *GV,
                              const std::vector<ConstantInt*> &Indices) {
  Constant *Init = GV->getInitializer();
  for (unsigned i = 0, e = Indices.size(); i != e; ++i) {
    uint64_t Idx = Indices[i]->getZExtValue();
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(Init)) {
      assert(Idx < CS->getNumOperands() && "Bad struct index!");
      Init = cast<Constant>(CS->getOperand(Idx));
    } else if (ConstantArray *CA = dyn_cast<ConstantArray>(Init)) {
      if (Idx >= CA->getNumOperands()) return 0;  // Bogus program
      Init = cast<Constant>(CA->getOperand(Idx));
    } else if (isa<ConstantAggregateZero>(Init)) {
      if (StructType *STy = dyn_cast<StructType>(Init->getType())) {
        assert(Idx < STy->getNumElements() && "Bad struct index!");
        Init = Constant::getNullValue(STy->getElementType(Idx));
      } else if (ArrayType *ATy = dyn_cast<ArrayType>(Init->getType())) {
        if (Idx >= ATy->getNumElements()) return 0;  // Bogus program
        Init = Constant::getNullValue(ATy->getElementType());
      } else {
        llvm_unreachable("Unknown constant aggregate type!");
      }
      return 0;
    } else {
      return 0; // Unknown initializer type
    }
  }
  return Init;
}

/// ComputeLoadConstantCompareExitLimit - Given an exit condition of
/// 'icmp op load X, cst', try to see if we can compute the backedge
/// execution count.
ScalarEvolution::ExitLimit
ScalarEvolution::ComputeLoadConstantCompareExitLimit(
  LoadInst *LI,
  Constant *RHS,
  const Loop *L,
  ICmpInst::Predicate predicate) {

  if (LI->isVolatile()) return getCouldNotCompute();

  // Check to see if the loaded pointer is a getelementptr of a global.
  // TODO: Use SCEV instead of manually grubbing with GEPs.
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(LI->getOperand(0));
  if (!GEP) return getCouldNotCompute();

  // Make sure that it is really a constant global we are gepping, with an
  // initializer, and make sure the first IDX is really 0.
  GlobalVariable *GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer() ||
      GEP->getNumOperands() < 3 || !isa<Constant>(GEP->getOperand(1)) ||
      !cast<Constant>(GEP->getOperand(1))->isNullValue())
    return getCouldNotCompute();

  // Okay, we allow one non-constant index into the GEP instruction.
  Value *VarIdx = 0;
  std::vector<ConstantInt*> Indexes;
  unsigned VarIdxNum = 0;
  for (unsigned i = 2, e = GEP->getNumOperands(); i != e; ++i)
    if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i))) {
      Indexes.push_back(CI);
    } else if (!isa<ConstantInt>(GEP->getOperand(i))) {
      if (VarIdx) return getCouldNotCompute();  // Multiple non-constant idx's.
      VarIdx = GEP->getOperand(i);
      VarIdxNum = i-2;
      Indexes.push_back(0);
    }

  // Okay, we know we have a (load (gep GV, 0, X)) comparison with a constant.
  // Check to see if X is a loop variant variable value now.
  const SCEV *Idx = getSCEV(VarIdx);
  Idx = getSCEVAtScope(Idx, L);

  // We can only recognize very limited forms of loop index expressions, in
  // particular, only affine AddRec's like {C1,+,C2}.
  const SCEVAddRecExpr *IdxExpr = dyn_cast<SCEVAddRecExpr>(Idx);
  if (!IdxExpr || !IdxExpr->isAffine() || isLoopInvariant(IdxExpr, L) ||
      !isa<SCEVConstant>(IdxExpr->getOperand(0)) ||
      !isa<SCEVConstant>(IdxExpr->getOperand(1)))
    return getCouldNotCompute();

  unsigned MaxSteps = MaxBruteForceIterations;
  for (unsigned IterationNum = 0; IterationNum != MaxSteps; ++IterationNum) {
    ConstantInt *ItCst = ConstantInt::get(
                           cast<IntegerType>(IdxExpr->getType()), IterationNum);
    ConstantInt *Val = EvaluateConstantChrecAtConstant(IdxExpr, ItCst, *this);

    // Form the GEP offset.
    Indexes[VarIdxNum] = Val;

    Constant *Result = GetAddressedElementFromGlobal(GV, Indexes);
    if (Result == 0) break;  // Cannot compute!

    // Evaluate the condition for this iteration.
    Result = ConstantExpr::getICmp(predicate, Result, RHS);
    if (!isa<ConstantInt>(Result)) break;  // Couldn't decide for sure
    if (cast<ConstantInt>(Result)->getValue().isMinValue()) {
#if 0
      dbgs() << "\n***\n*** Computed loop count " << *ItCst
             << "\n*** From global " << *GV << "*** BB: " << *L->getHeader()
             << "***\n";
#endif
      ++NumArrayLenItCounts;
      return getConstant(ItCst);   // Found terminating iteration!
    }
  }
  return getCouldNotCompute();
}


/// CanConstantFold - Return true if we can constant fold an instruction of the
/// specified type, assuming that all operands were constants.
static bool CanConstantFold(const Instruction *I) {
  if (isa<BinaryOperator>(I) || isa<CmpInst>(I) ||
      isa<SelectInst>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I))
    return true;

  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (const Function *F = CI->getCalledFunction())
      return canConstantFoldCallTo(F);
  return false;
}

/// getConstantEvolvingPHI - Given an LLVM value and a loop, return a PHI node
/// in the loop that V is derived from.  We allow arbitrary operations along the
/// way, but the operands of an operation must either be constants or a value
/// derived from a constant PHI.  If this expression does not fit with these
/// constraints, return null.
static PHINode *getConstantEvolvingPHI(Value *V, const Loop *L) {
  // If this is not an instruction, or if this is an instruction outside of the
  // loop, it can't be derived from a loop PHI.
  Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0 || !L->contains(I)) return 0;

  if (PHINode *PN = dyn_cast<PHINode>(I)) {
    if (L->getHeader() == I->getParent())
      return PN;
    else
      // We don't currently keep track of the control flow needed to evaluate
      // PHIs, so we cannot handle PHIs inside of loops.
      return 0;
  }

  // If we won't be able to constant fold this expression even if the operands
  // are constants, return early.
  if (!CanConstantFold(I)) return 0;

  // Otherwise, we can evaluate this instruction if all of its operands are
  // constant or derived from a PHI node themselves.
  PHINode *PHI = 0;
  for (unsigned Op = 0, e = I->getNumOperands(); Op != e; ++Op)
    if (!isa<Constant>(I->getOperand(Op))) {
      PHINode *P = getConstantEvolvingPHI(I->getOperand(Op), L);
      if (P == 0) return 0;  // Not evolving from PHI
      if (PHI == 0)
        PHI = P;
      else if (PHI != P)
        return 0;  // Evolving from multiple different PHIs.
    }

  // This is a expression evolving from a constant PHI!
  return PHI;
}

/// EvaluateExpression - Given an expression that passes the
/// getConstantEvolvingPHI predicate, evaluate its value assuming the PHI node
/// in the loop has the value PHIVal.  If we can't fold this expression for some
/// reason, return null.
static Constant *EvaluateExpression(Value *V, Constant *PHIVal,
                                    const TargetData *TD) {
  if (isa<PHINode>(V)) return PHIVal;
  if (Constant *C = dyn_cast<Constant>(V)) return C;
  Instruction *I = cast<Instruction>(V);

  std::vector<Constant*> Operands(I->getNumOperands());

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Operands[i] = EvaluateExpression(I->getOperand(i), PHIVal, TD);
    if (Operands[i] == 0) return 0;
  }

  if (const CmpInst *CI = dyn_cast<CmpInst>(I))
    return ConstantFoldCompareInstOperands(CI->getPredicate(), Operands[0],
                                           Operands[1], TD);
  return ConstantFoldInstOperands(I->getOpcode(), I->getType(), Operands, TD);
}

/// getConstantEvolutionLoopExitValue - If we know that the specified Phi is
/// in the header of its containing loop, we know the loop executes a
/// constant number of times, and the PHI node is just a recurrence
/// involving constants, fold it.
Constant *
ScalarEvolution::getConstantEvolutionLoopExitValue(PHINode *PN,
                                                   const APInt &BEs,
                                                   const Loop *L) {
  DenseMap<PHINode*, Constant*>::const_iterator I =
    ConstantEvolutionLoopExitValue.find(PN);
  if (I != ConstantEvolutionLoopExitValue.end())
    return I->second;

  if (BEs.ugt(MaxBruteForceIterations))
    return ConstantEvolutionLoopExitValue[PN] = 0;  // Not going to evaluate it.

  Constant *&RetVal = ConstantEvolutionLoopExitValue[PN];

  // Since the loop is canonicalized, the PHI node must have two entries.  One
  // entry must be a constant (coming in from outside of the loop), and the
  // second must be derived from the same PHI.
  bool SecondIsBackedge = L->contains(PN->getIncomingBlock(1));
  Constant *StartCST =
    dyn_cast<Constant>(PN->getIncomingValue(!SecondIsBackedge));
  if (StartCST == 0)
    return RetVal = 0;  // Must be a constant.

  Value *BEValue = PN->getIncomingValue(SecondIsBackedge);
  if (getConstantEvolvingPHI(BEValue, L) != PN &&
      !isa<Constant>(BEValue))
    return RetVal = 0;  // Not derived from same PHI.

  // Execute the loop symbolically to determine the exit value.
  if (BEs.getActiveBits() >= 32)
    return RetVal = 0; // More than 2^32-1 iterations?? Not doing it!

  unsigned NumIterations = BEs.getZExtValue(); // must be in range
  unsigned IterationNum = 0;
  for (Constant *PHIVal = StartCST; ; ++IterationNum) {
    if (IterationNum == NumIterations)
      return RetVal = PHIVal;  // Got exit value!

    // Compute the value of the PHI node for the next iteration.
    Constant *NextPHI = EvaluateExpression(BEValue, PHIVal, TD);
    if (NextPHI == PHIVal)
      return RetVal = NextPHI;  // Stopped evolving!
    if (NextPHI == 0)
      return 0;        // Couldn't evaluate!
    PHIVal = NextPHI;
  }
}

/// ComputeExitCountExhaustively - If the loop is known to execute a
/// constant number of times (the condition evolves only from constants),
/// try to evaluate a few iterations of the loop until we get the exit
/// condition gets a value of ExitWhen (true or false).  If we cannot
/// evaluate the trip count of the loop, return getCouldNotCompute().
const SCEV * ScalarEvolution::ComputeExitCountExhaustively(const Loop *L,
                                                           Value *Cond,
                                                           bool ExitWhen) {
  PHINode *PN = getConstantEvolvingPHI(Cond, L);
  if (PN == 0) return getCouldNotCompute();

  // If the loop is canonicalized, the PHI will have exactly two entries.
  // That's the only form we support here.
  if (PN->getNumIncomingValues() != 2) return getCouldNotCompute();

  // One entry must be a constant (coming in from outside of the loop), and the
  // second must be derived from the same PHI.
  bool SecondIsBackedge = L->contains(PN->getIncomingBlock(1));
  Constant *StartCST =
    dyn_cast<Constant>(PN->getIncomingValue(!SecondIsBackedge));
  if (StartCST == 0) return getCouldNotCompute();  // Must be a constant.

  Value *BEValue = PN->getIncomingValue(SecondIsBackedge);
  if (getConstantEvolvingPHI(BEValue, L) != PN &&
      !isa<Constant>(BEValue))
    return getCouldNotCompute();  // Not derived from same PHI.

  // Okay, we find a PHI node that defines the trip count of this loop.  Execute
  // the loop symbolically to determine when the condition gets a value of
  // "ExitWhen".
  unsigned IterationNum = 0;
  unsigned MaxIterations = MaxBruteForceIterations;   // Limit analysis.
  for (Constant *PHIVal = StartCST;
       IterationNum != MaxIterations; ++IterationNum) {
    ConstantInt *CondVal =
      dyn_cast_or_null<ConstantInt>(EvaluateExpression(Cond, PHIVal, TD));

    // Couldn't symbolically evaluate.
    if (!CondVal) return getCouldNotCompute();

    if (CondVal->getValue() == uint64_t(ExitWhen)) {
      ++NumBruteForceTripCountsComputed;
      return getConstant(Type::getInt32Ty(getContext()), IterationNum);
    }

    // Compute the value of the PHI node for the next iteration.
    Constant *NextPHI = EvaluateExpression(BEValue, PHIVal, TD);
    if (NextPHI == 0 || NextPHI == PHIVal)
      return getCouldNotCompute();// Couldn't evaluate or not making progress...
    PHIVal = NextPHI;
  }

  // Too many iterations were needed to evaluate.
  return getCouldNotCompute();
}

/// getSCEVAtScope - Return a SCEV expression for the specified value
/// at the specified scope in the program.  The L value specifies a loop
/// nest to evaluate the expression at, where null is the top-level or a
/// specified loop is immediately inside of the loop.
///
/// This method can be used to compute the exit value for a variable defined
/// in a loop by querying what the value will hold in the parent loop.
///
/// In the case that a relevant loop exit value cannot be computed, the
/// original value V is returned.
const SCEV *ScalarEvolution::getSCEVAtScope(const SCEV *V, const Loop *L) {
  // Check to see if we've folded this expression at this loop before.
  std::map<const Loop *, const SCEV *> &Values = ValuesAtScopes[V];
  std::pair<std::map<const Loop *, const SCEV *>::iterator, bool> Pair =
    Values.insert(std::make_pair(L, static_cast<const SCEV *>(0)));
  if (!Pair.second)
    return Pair.first->second ? Pair.first->second : V;

  // Otherwise compute it.
  const SCEV *C = computeSCEVAtScope(V, L);
  ValuesAtScopes[V][L] = C;
  return C;
}

const SCEV *ScalarEvolution::computeSCEVAtScope(const SCEV *V, const Loop *L) {
  if (isa<SCEVConstant>(V)) return V;

  // If this instruction is evolved from a constant-evolving PHI, compute the
  // exit value from the loop without using SCEVs.
  if (const SCEVUnknown *SU = dyn_cast<SCEVUnknown>(V)) {
    if (Instruction *I = dyn_cast<Instruction>(SU->getValue())) {
      const Loop *LI = (*this->LI)[I->getParent()];
      if (LI && LI->getParentLoop() == L)  // Looking for loop exit value.
        if (PHINode *PN = dyn_cast<PHINode>(I))
          if (PN->getParent() == LI->getHeader()) {
            // Okay, there is no closed form solution for the PHI node.  Check
            // to see if the loop that contains it has a known backedge-taken
            // count.  If so, we may be able to force computation of the exit
            // value.
            const SCEV *BackedgeTakenCount = getBackedgeTakenCount(LI);
            if (const SCEVConstant *BTCC =
                  dyn_cast<SCEVConstant>(BackedgeTakenCount)) {
              // Okay, we know how many times the containing loop executes.  If
              // this is a constant evolving PHI node, get the final value at
              // the specified iteration number.
              Constant *RV = getConstantEvolutionLoopExitValue(PN,
                                                   BTCC->getValue()->getValue(),
                                                               LI);
              if (RV) return getSCEV(RV);
            }
          }

      // Okay, this is an expression that we cannot symbolically evaluate
      // into a SCEV.  Check to see if it's possible to symbolically evaluate
      // the arguments into constants, and if so, try to constant propagate the
      // result.  This is particularly useful for computing loop exit values.
      if (CanConstantFold(I)) {
        SmallVector<Constant *, 4> Operands;
        bool MadeImprovement = false;
        for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
          Value *Op = I->getOperand(i);
          if (Constant *C = dyn_cast<Constant>(Op)) {
            Operands.push_back(C);
            continue;
          }

          // If any of the operands is non-constant and if they are
          // non-integer and non-pointer, don't even try to analyze them
          // with scev techniques.
          if (!isSCEVable(Op->getType()))
            return V;

          const SCEV *OrigV = getSCEV(Op);
          const SCEV *OpV = getSCEVAtScope(OrigV, L);
          MadeImprovement |= OrigV != OpV;

          Constant *C = 0;
          if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(OpV))
            C = SC->getValue();
          if (const SCEVUnknown *SU = dyn_cast<SCEVUnknown>(OpV))
            C = dyn_cast<Constant>(SU->getValue());
          if (!C) return V;
          if (C->getType() != Op->getType())
            C = ConstantExpr::getCast(CastInst::getCastOpcode(C, false,
                                                              Op->getType(),
                                                              false),
                                      C, Op->getType());
          Operands.push_back(C);
        }

        // Check to see if getSCEVAtScope actually made an improvement.
        if (MadeImprovement) {
          Constant *C = 0;
          if (const CmpInst *CI = dyn_cast<CmpInst>(I))
            C = ConstantFoldCompareInstOperands(CI->getPredicate(),
                                                Operands[0], Operands[1], TD);
          else
            C = ConstantFoldInstOperands(I->getOpcode(), I->getType(),
                                         Operands, TD);
          if (!C) return V;
          return getSCEV(C);
        }
      }
    }

    // This is some other type of SCEVUnknown, just return it.
    return V;
  }

  if (const SCEVCommutativeExpr *Comm = dyn_cast<SCEVCommutativeExpr>(V)) {
    // Avoid performing the look-up in the common case where the specified
    // expression has no loop-variant portions.
    for (unsigned i = 0, e = Comm->getNumOperands(); i != e; ++i) {
      const SCEV *OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
      if (OpAtScope != Comm->getOperand(i)) {
        // Okay, at least one of these operands is loop variant but might be
        // foldable.  Build a new instance of the folded commutative expression.
        SmallVector<const SCEV *, 8> NewOps(Comm->op_begin(),
                                            Comm->op_begin()+i);
        NewOps.push_back(OpAtScope);

        for (++i; i != e; ++i) {
          OpAtScope = getSCEVAtScope(Comm->getOperand(i), L);
          NewOps.push_back(OpAtScope);
        }
        if (isa<SCEVAddExpr>(Comm))
          return getAddExpr(NewOps);
        if (isa<SCEVMulExpr>(Comm))
          return getMulExpr(NewOps);
        if (isa<SCEVSMaxExpr>(Comm))
          return getSMaxExpr(NewOps);
        if (isa<SCEVUMaxExpr>(Comm))
          return getUMaxExpr(NewOps);
        llvm_unreachable("Unknown commutative SCEV type!");
      }
    }
    // If we got here, all operands are loop invariant.
    return Comm;
  }

  if (const SCEVUDivExpr *Div = dyn_cast<SCEVUDivExpr>(V)) {
    const SCEV *LHS = getSCEVAtScope(Div->getLHS(), L);
    const SCEV *RHS = getSCEVAtScope(Div->getRHS(), L);
    if (LHS == Div->getLHS() && RHS == Div->getRHS())
      return Div;   // must be loop invariant
    return getUDivExpr(LHS, RHS);
  }

  // If this is a loop recurrence for a loop that does not contain L, then we
  // are dealing with the final value computed by the loop.
  if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V)) {
    // First, attempt to evaluate each operand.
    // Avoid performing the look-up in the common case where the specified
    // expression has no loop-variant portions.
    for (unsigned i = 0, e = AddRec->getNumOperands(); i != e; ++i) {
      const SCEV *OpAtScope = getSCEVAtScope(AddRec->getOperand(i), L);
      if (OpAtScope == AddRec->getOperand(i))
        continue;

      // Okay, at least one of these operands is loop variant but might be
      // foldable.  Build a new instance of the folded commutative expression.
      SmallVector<const SCEV *, 8> NewOps(AddRec->op_begin(),
                                          AddRec->op_begin()+i);
      NewOps.push_back(OpAtScope);
      for (++i; i != e; ++i)
        NewOps.push_back(getSCEVAtScope(AddRec->getOperand(i), L));

      const SCEV *FoldedRec =
        getAddRecExpr(NewOps, AddRec->getLoop(),
                      AddRec->getNoWrapFlags(SCEV::FlagNW));
      AddRec = dyn_cast<SCEVAddRecExpr>(FoldedRec);
      // The addrec may be folded to a nonrecurrence, for example, if the
      // induction variable is multiplied by zero after constant folding. Go
      // ahead and return the folded value.
      if (!AddRec)
        return FoldedRec;
      break;
    }

    // If the scope is outside the addrec's loop, evaluate it by using the
    // loop exit value of the addrec.
    if (!AddRec->getLoop()->contains(L)) {
      // To evaluate this recurrence, we need to know how many times the AddRec
      // loop iterates.  Compute this now.
      const SCEV *BackedgeTakenCount = getBackedgeTakenCount(AddRec->getLoop());
      if (BackedgeTakenCount == getCouldNotCompute()) return AddRec;

      // Then, evaluate the AddRec.
      return AddRec->evaluateAtIteration(BackedgeTakenCount, *this);
    }

    return AddRec;
  }

  if (const SCEVZeroExtendExpr *Cast = dyn_cast<SCEVZeroExtendExpr>(V)) {
    const SCEV *Op = getSCEVAtScope(Cast->getOperand(), L);
    if (Op == Cast->getOperand())
      return Cast;  // must be loop invariant
    return getZeroExtendExpr(Op, Cast->getType());
  }

  if (const SCEVSignExtendExpr *Cast = dyn_cast<SCEVSignExtendExpr>(V)) {
    const SCEV *Op = getSCEVAtScope(Cast->getOperand(), L);
    if (Op == Cast->getOperand())
      return Cast;  // must be loop invariant
    return getSignExtendExpr(Op, Cast->getType());
  }

  if (const SCEVTruncateExpr *Cast = dyn_cast<SCEVTruncateExpr>(V)) {
    const SCEV *Op = getSCEVAtScope(Cast->getOperand(), L);
    if (Op == Cast->getOperand())
      return Cast;  // must be loop invariant
    return getTruncateExpr(Op, Cast->getType());
  }

  llvm_unreachable("Unknown SCEV type!");
  return 0;
}

/// getSCEVAtScope - This is a convenience function which does
/// getSCEVAtScope(getSCEV(V), L).
const SCEV *ScalarEvolution::getSCEVAtScope(Value *V, const Loop *L) {
  return getSCEVAtScope(getSCEV(V), L);
}

/// SolveLinEquationWithOverflow - Finds the minimum unsigned root of the
/// following equation:
///
///     A * X = B (mod N)
///
/// where N = 2^BW and BW is the common bit width of A and B. The signedness of
/// A and B isn't important.
///
/// If the equation does not have a solution, SCEVCouldNotCompute is returned.
static const SCEV *SolveLinEquationWithOverflow(const APInt &A, const APInt &B,
                                               ScalarEvolution &SE) {
  uint32_t BW = A.getBitWidth();
  assert(BW == B.getBitWidth() && "Bit widths must be the same.");
  assert(A != 0 && "A must be non-zero.");

  // 1. D = gcd(A, N)
  //
  // The gcd of A and N may have only one prime factor: 2. The number of
  // trailing zeros in A is its multiplicity
  uint32_t Mult2 = A.countTrailingZeros();
  // D = 2^Mult2

  // 2. Check if B is divisible by D.
  //
  // B is divisible by D if and only if the multiplicity of prime factor 2 for B
  // is not less than multiplicity of this prime factor for D.
  if (B.countTrailingZeros() < Mult2)
    return SE.getCouldNotCompute();

  // 3. Compute I: the multiplicative inverse of (A / D) in arithmetic
  // modulo (N / D).
  //
  // (N / D) may need BW+1 bits in its representation.  Hence, we'll use this
  // bit width during computations.
  APInt AD = A.lshr(Mult2).zext(BW + 1);  // AD = A / D
  APInt Mod(BW + 1, 0);
  Mod.setBit(BW - Mult2);  // Mod = N / D
  APInt I = AD.multiplicativeInverse(Mod);

  // 4. Compute the minimum unsigned root of the equation:
  // I * (B / D) mod (N / D)
  APInt Result = (I * B.lshr(Mult2).zext(BW + 1)).urem(Mod);

  // The result is guaranteed to be less than 2^BW so we may truncate it to BW
  // bits.
  return SE.getConstant(Result.trunc(BW));
}

/// SolveQuadraticEquation - Find the roots of the quadratic equation for the
/// given quadratic chrec {L,+,M,+,N}.  This returns either the two roots (which
/// might be the same) or two SCEVCouldNotCompute objects.
///
static std::pair<const SCEV *,const SCEV *>
SolveQuadraticEquation(const SCEVAddRecExpr *AddRec, ScalarEvolution &SE) {
  assert(AddRec->getNumOperands() == 3 && "This is not a quadratic chrec!");
  const SCEVConstant *LC = dyn_cast<SCEVConstant>(AddRec->getOperand(0));
  const SCEVConstant *MC = dyn_cast<SCEVConstant>(AddRec->getOperand(1));
  const SCEVConstant *NC = dyn_cast<SCEVConstant>(AddRec->getOperand(2));

  // We currently can only solve this if the coefficients are constants.
  if (!LC || !MC || !NC) {
    const SCEV *CNC = SE.getCouldNotCompute();
    return std::make_pair(CNC, CNC);
  }

  uint32_t BitWidth = LC->getValue()->getValue().getBitWidth();
  const APInt &L = LC->getValue()->getValue();
  const APInt &M = MC->getValue()->getValue();
  const APInt &N = NC->getValue()->getValue();
  APInt Two(BitWidth, 2);
  APInt Four(BitWidth, 4);

  {
    using namespace APIntOps;
    const APInt& C = L;
    // Convert from chrec coefficients to polynomial coefficients AX^2+BX+C
    // The B coefficient is M-N/2
    APInt B(M);
    B -= sdiv(N,Two);

    // The A coefficient is N/2
    APInt A(N.sdiv(Two));

    // Compute the B^2-4ac term.
    APInt SqrtTerm(B);
    SqrtTerm *= B;
    SqrtTerm -= Four * (A * C);

    // Compute sqrt(B^2-4ac). This is guaranteed to be the nearest
    // integer value or else APInt::sqrt() will assert.
    APInt SqrtVal(SqrtTerm.sqrt());

    // Compute the two solutions for the quadratic formula.
    // The divisions must be performed as signed divisions.
    APInt NegB(-B);
    APInt TwoA( A << 1 );
    if (TwoA.isMinValue()) {
      const SCEV *CNC = SE.getCouldNotCompute();
      return std::make_pair(CNC, CNC);
    }

    LLVMContext &Context = SE.getContext();

    ConstantInt *Solution1 =
      ConstantInt::get(Context, (NegB + SqrtVal).sdiv(TwoA));
    ConstantInt *Solution2 =
      ConstantInt::get(Context, (NegB - SqrtVal).sdiv(TwoA));

    return std::make_pair(SE.getConstant(Solution1),
                          SE.getConstant(Solution2));
    } // end APIntOps namespace
}

/// HowFarToZero - Return the number of times a backedge comparing the specified
/// value to zero will execute.  If not computable, return CouldNotCompute.
///
/// This is only used for loops with a "x != y" exit test. The exit condition is
/// now expressed as a single expression, V = x-y. So the exit test is
/// effectively V != 0.  We know and take advantage of the fact that this
/// expression only being used in a comparison by zero context.
ScalarEvolution::ExitLimit
ScalarEvolution::HowFarToZero(const SCEV *V, const Loop *L) {
  // If the value is a constant
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    // If the value is already zero, the branch will execute zero times.
    if (C->getValue()->isZero()) return C;
    return getCouldNotCompute();  // Otherwise it will loop infinitely.
  }

  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(V);
  if (!AddRec || AddRec->getLoop() != L)
    return getCouldNotCompute();

  // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of
  // the quadratic equation to solve it.
  if (AddRec->isQuadratic() && AddRec->getType()->isIntegerTy()) {
    std::pair<const SCEV *,const SCEV *> Roots =
      SolveQuadraticEquation(AddRec, *this);
    const SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    const SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1 && R2) {
#if 0
      dbgs() << "HFTZ: " << *V << " - sol#1: " << *R1
             << "  sol#2: " << *R2 << "\n";
#endif
      // Pick the smallest positive root value.
      if (ConstantInt *CB =
          dyn_cast<ConstantInt>(ConstantExpr::getICmp(CmpInst::ICMP_ULT,
                                                      R1->getValue(),
                                                      R2->getValue()))) {
        if (CB->getZExtValue() == false)
          std::swap(R1, R2);   // R1 is the minimum root now.

        // We can only use this value if the chrec ends up with an exact zero
        // value at this index.  When solving for "X*X != 5", for example, we
        // should not accept a root of 2.
        const SCEV *Val = AddRec->evaluateAtIteration(R1, *this);
        if (Val->isZero())
          return R1;  // We found a quadratic root!
      }
    }
    return getCouldNotCompute();
  }

  // Otherwise we can only handle this if it is affine.
  if (!AddRec->isAffine())
    return getCouldNotCompute();

  // If this is an affine expression, the execution count of this branch is
  // the minimum unsigned root of the following equation:
  //
  //     Start + Step*N = 0 (mod 2^BW)
  //
  // equivalent to:
  //
  //             Step*N = -Start (mod 2^BW)
  //
  // where BW is the common bit width of Start and Step.

  // Get the initial value for the loop.
  const SCEV *Start = getSCEVAtScope(AddRec->getStart(), L->getParentLoop());
  const SCEV *Step = getSCEVAtScope(AddRec->getOperand(1), L->getParentLoop());

  // For now we handle only constant steps.
  //
  // TODO: Handle a nonconstant Step given AddRec<NUW>. If the
  // AddRec is NUW, then (in an unsigned sense) it cannot be counting up to wrap
  // to 0, it must be counting down to equal 0. Consequently, N = Start / -Step.
  // We have not yet seen any such cases.
  const SCEVConstant *StepC = dyn_cast<SCEVConstant>(Step);
  if (StepC == 0)
    return getCouldNotCompute();

  // For positive steps (counting up until unsigned overflow):
  //   N = -Start/Step (as unsigned)
  // For negative steps (counting down to zero):
  //   N = Start/-Step
  // First compute the unsigned distance from zero in the direction of Step.
  bool CountDown = StepC->getValue()->getValue().isNegative();
  const SCEV *Distance = CountDown ? Start : getNegativeSCEV(Start);

  // Handle unitary steps, which cannot wraparound.
  // 1*N = -Start; -1*N = Start (mod 2^BW), so:
  //   N = Distance (as unsigned)
  if (StepC->getValue()->equalsInt(1) || StepC->getValue()->isAllOnesValue())
    return Distance;

  // If the recurrence is known not to wraparound, unsigned divide computes the
  // back edge count. We know that the value will either become zero (and thus
  // the loop terminates), that the loop will terminate through some other exit
  // condition first, or that the loop has undefined behavior.  This means
  // we can't "miss" the exit value, even with nonunit stride.
  //
  // FIXME: Prove that loops always exhibits *acceptable* undefined
  // behavior. Loops must exhibit defined behavior until a wrapped value is
  // actually used. So the trip count computed by udiv could be smaller than the
  // number of well-defined iterations.
  if (AddRec->getNoWrapFlags(SCEV::FlagNW))
    // FIXME: We really want an "isexact" bit for udiv.
    return getUDivExpr(Distance, CountDown ? getNegativeSCEV(Step) : Step);

  // Then, try to solve the above equation provided that Start is constant.
  if (const SCEVConstant *StartC = dyn_cast<SCEVConstant>(Start))
    return SolveLinEquationWithOverflow(StepC->getValue()->getValue(),
                                        -StartC->getValue()->getValue(),
                                        *this);
  return getCouldNotCompute();
}

/// HowFarToNonZero - Return the number of times a backedge checking the
/// specified value for nonzero will execute.  If not computable, return
/// CouldNotCompute
ScalarEvolution::ExitLimit
ScalarEvolution::HowFarToNonZero(const SCEV *V, const Loop *L) {
  // Loops that look like: while (X == 0) are very strange indeed.  We don't
  // handle them yet except for the trivial case.  This could be expanded in the
  // future as needed.

  // If the value is a constant, check to see if it is known to be non-zero
  // already.  If so, the backedge will execute zero times.
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(V)) {
    if (!C->getValue()->isNullValue())
      return getConstant(C->getType(), 0);
    return getCouldNotCompute();  // Otherwise it will loop infinitely.
  }

  // We could implement others, but I really doubt anyone writes loops like
  // this, and if they did, they would already be constant folded.
  return getCouldNotCompute();
}

/// getPredecessorWithUniqueSuccessorForBB - Return a predecessor of BB
/// (which may not be an immediate predecessor) which has exactly one
/// successor from which BB is reachable, or null if no such block is
/// found.
///
std::pair<BasicBlock *, BasicBlock *>
ScalarEvolution::getPredecessorWithUniqueSuccessorForBB(BasicBlock *BB) {
  // If the block has a unique predecessor, then there is no path from the
  // predecessor to the block that does not go through the direct edge
  // from the predecessor to the block.
  if (BasicBlock *Pred = BB->getSinglePredecessor())
    return std::make_pair(Pred, BB);

  // A loop's header is defined to be a block that dominates the loop.
  // If the header has a unique predecessor outside the loop, it must be
  // a block that has exactly one successor that can reach the loop.
  if (Loop *L = LI->getLoopFor(BB))
    return std::make_pair(L->getLoopPredecessor(), L->getHeader());

  return std::pair<BasicBlock *, BasicBlock *>();
}

/// HasSameValue - SCEV structural equivalence is usually sufficient for
/// testing whether two expressions are equal, however for the purposes of
/// looking for a condition guarding a loop, it can be useful to be a little
/// more general, since a front-end may have replicated the controlling
/// expression.
///
static bool HasSameValue(const SCEV *A, const SCEV *B) {
  // Quick check to see if they are the same SCEV.
  if (A == B) return true;

  // Otherwise, if they're both SCEVUnknown, it's possible that they hold
  // two different instructions with the same value. Check for this case.
  if (const SCEVUnknown *AU = dyn_cast<SCEVUnknown>(A))
    if (const SCEVUnknown *BU = dyn_cast<SCEVUnknown>(B))
      if (const Instruction *AI = dyn_cast<Instruction>(AU->getValue()))
        if (const Instruction *BI = dyn_cast<Instruction>(BU->getValue()))
          if (AI->isIdenticalTo(BI) && !AI->mayReadFromMemory())
            return true;

  // Otherwise assume they may have a different value.
  return false;
}

/// SimplifyICmpOperands - Simplify LHS and RHS in a comparison with
/// predicate Pred. Return true iff any changes were made.
///
bool ScalarEvolution::SimplifyICmpOperands(ICmpInst::Predicate &Pred,
                                           const SCEV *&LHS, const SCEV *&RHS) {
  bool Changed = false;

  // Canonicalize a constant to the right side.
  if (const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS)) {
    // Check for both operands constant.
    if (const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS)) {
      if (ConstantExpr::getICmp(Pred,
                                LHSC->getValue(),
                                RHSC->getValue())->isNullValue())
        goto trivially_false;
      else
        goto trivially_true;
    }
    // Otherwise swap the operands to put the constant on the right.
    std::swap(LHS, RHS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
    Changed = true;
  }

  // If we're comparing an addrec with a value which is loop-invariant in the
  // addrec's loop, put the addrec on the left. Also make a dominance check,
  // as both operands could be addrecs loop-invariant in each other's loop.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(RHS)) {
    const Loop *L = AR->getLoop();
    if (isLoopInvariant(LHS, L) && properlyDominates(LHS, L->getHeader())) {
      std::swap(LHS, RHS);
      Pred = ICmpInst::getSwappedPredicate(Pred);
      Changed = true;
    }
  }

  // If there's a constant operand, canonicalize comparisons with boundary
  // cases, and canonicalize *-or-equal comparisons to regular comparisons.
  if (const SCEVConstant *RC = dyn_cast<SCEVConstant>(RHS)) {
    const APInt &RA = RC->getValue()->getValue();
    switch (Pred) {
    default: llvm_unreachable("Unexpected ICmpInst::Predicate value!");
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_NE:
      break;
    case ICmpInst::ICMP_UGE:
      if ((RA - 1).isMinValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA - 1);
        Changed = true;
        break;
      }
      if (RA.isMaxValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMinValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_UGT;
      RHS = getConstant(RA - 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_ULE:
      if ((RA + 1).isMaxValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMinValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMaxValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_ULT;
      RHS = getConstant(RA + 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_SGE:
      if ((RA - 1).isMinSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA - 1);
        Changed = true;
        break;
      }
      if (RA.isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMinSignedValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_SGT;
      RHS = getConstant(RA - 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_SLE:
      if ((RA + 1).isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMinSignedValue()) {
        Pred = ICmpInst::ICMP_EQ;
        Changed = true;
        break;
      }
      if (RA.isMaxSignedValue()) goto trivially_true;

      Pred = ICmpInst::ICMP_SLT;
      RHS = getConstant(RA + 1);
      Changed = true;
      break;
    case ICmpInst::ICMP_UGT:
      if (RA.isMinValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA + 1).isMaxValue()) {
        Pred = ICmpInst::ICMP_EQ;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMaxValue()) goto trivially_false;
      break;
    case ICmpInst::ICMP_ULT:
      if (RA.isMaxValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA - 1).isMinValue()) {
        Pred = ICmpInst::ICMP_EQ;
        RHS = getConstant(RA - 1);
        Changed = true;
        break;
      }
      if (RA.isMinValue()) goto trivially_false;
      break;
    case ICmpInst::ICMP_SGT:
      if (RA.isMinSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA + 1).isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_EQ;
        RHS = getConstant(RA + 1);
        Changed = true;
        break;
      }
      if (RA.isMaxSignedValue()) goto trivially_false;
      break;
    case ICmpInst::ICMP_SLT:
      if (RA.isMaxSignedValue()) {
        Pred = ICmpInst::ICMP_NE;
        Changed = true;
        break;
      }
      if ((RA - 1).isMinSignedValue()) {
       Pred = ICmpInst::ICMP_EQ;
       RHS = getConstant(RA - 1);
        Changed = true;
       break;
      }
      if (RA.isMinSignedValue()) goto trivially_false;
      break;
    }
  }

  // Check for obvious equality.
  if (HasSameValue(LHS, RHS)) {
    if (ICmpInst::isTrueWhenEqual(Pred))
      goto trivially_true;
    if (ICmpInst::isFalseWhenEqual(Pred))
      goto trivially_false;
  }

  // If possible, canonicalize GE/LE comparisons to GT/LT comparisons, by
  // adding or subtracting 1 from one of the operands.
  switch (Pred) {
  case ICmpInst::ICMP_SLE:
    if (!getSignedRange(RHS).getSignedMax().isMaxSignedValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), 1, true), RHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SLT;
      Changed = true;
    } else if (!getSignedRange(LHS).getSignedMin().isMinSignedValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), LHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SLT;
      Changed = true;
    }
    break;
  case ICmpInst::ICMP_SGE:
    if (!getSignedRange(RHS).getSignedMin().isMinSignedValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), RHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SGT;
      Changed = true;
    } else if (!getSignedRange(LHS).getSignedMax().isMaxSignedValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), 1, true), LHS,
                       SCEV::FlagNSW);
      Pred = ICmpInst::ICMP_SGT;
      Changed = true;
    }
    break;
  case ICmpInst::ICMP_ULE:
    if (!getUnsignedRange(RHS).getUnsignedMax().isMaxValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), 1, true), RHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_ULT;
      Changed = true;
    } else if (!getUnsignedRange(LHS).getUnsignedMin().isMinValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), LHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_ULT;
      Changed = true;
    }
    break;
  case ICmpInst::ICMP_UGE:
    if (!getUnsignedRange(RHS).getUnsignedMin().isMinValue()) {
      RHS = getAddExpr(getConstant(RHS->getType(), (uint64_t)-1, true), RHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_UGT;
      Changed = true;
    } else if (!getUnsignedRange(LHS).getUnsignedMax().isMaxValue()) {
      LHS = getAddExpr(getConstant(RHS->getType(), 1, true), LHS,
                       SCEV::FlagNUW);
      Pred = ICmpInst::ICMP_UGT;
      Changed = true;
    }
    break;
  default:
    break;
  }

  // TODO: More simplifications are possible here.

  return Changed;

trivially_true:
  // Return 0 == 0.
  LHS = RHS = getConstant(ConstantInt::getFalse(getContext()));
  Pred = ICmpInst::ICMP_EQ;
  return true;

trivially_false:
  // Return 0 != 0.
  LHS = RHS = getConstant(ConstantInt::getFalse(getContext()));
  Pred = ICmpInst::ICMP_NE;
  return true;
}

bool ScalarEvolution::isKnownNegative(const SCEV *S) {
  return getSignedRange(S).getSignedMax().isNegative();
}

bool ScalarEvolution::isKnownPositive(const SCEV *S) {
  return getSignedRange(S).getSignedMin().isStrictlyPositive();
}

bool ScalarEvolution::isKnownNonNegative(const SCEV *S) {
  return !getSignedRange(S).getSignedMin().isNegative();
}

bool ScalarEvolution::isKnownNonPositive(const SCEV *S) {
  return !getSignedRange(S).getSignedMax().isStrictlyPositive();
}

bool ScalarEvolution::isKnownNonZero(const SCEV *S) {
  return isKnownNegative(S) || isKnownPositive(S);
}

bool ScalarEvolution::isKnownPredicate(ICmpInst::Predicate Pred,
                                       const SCEV *LHS, const SCEV *RHS) {
  // Canonicalize the inputs first.
  (void)SimplifyICmpOperands(Pred, LHS, RHS);

  // If LHS or RHS is an addrec, check to see if the condition is true in
  // every iteration of the loop.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(LHS))
    if (isLoopEntryGuardedByCond(
          AR->getLoop(), Pred, AR->getStart(), RHS) &&
        isLoopBackedgeGuardedByCond(
          AR->getLoop(), Pred, AR->getPostIncExpr(*this), RHS))
      return true;
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(RHS))
    if (isLoopEntryGuardedByCond(
          AR->getLoop(), Pred, LHS, AR->getStart()) &&
        isLoopBackedgeGuardedByCond(
          AR->getLoop(), Pred, LHS, AR->getPostIncExpr(*this)))
      return true;

  // Otherwise see what can be done with known constant ranges.
  return isKnownPredicateWithRanges(Pred, LHS, RHS);
}

bool
ScalarEvolution::isKnownPredicateWithRanges(ICmpInst::Predicate Pred,
                                            const SCEV *LHS, const SCEV *RHS) {
  if (HasSameValue(LHS, RHS))
    return ICmpInst::isTrueWhenEqual(Pred);

  // This code is split out from isKnownPredicate because it is called from
  // within isLoopEntryGuardedByCond.
  switch (Pred) {
  default:
    llvm_unreachable("Unexpected ICmpInst::Predicate value!");
    break;
  case ICmpInst::ICMP_SGT:
    Pred = ICmpInst::ICMP_SLT;
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_SLT: {
    ConstantRange LHSRange = getSignedRange(LHS);
    ConstantRange RHSRange = getSignedRange(RHS);
    if (LHSRange.getSignedMax().slt(RHSRange.getSignedMin()))
      return true;
    if (LHSRange.getSignedMin().sge(RHSRange.getSignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_SGE:
    Pred = ICmpInst::ICMP_SLE;
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_SLE: {
    ConstantRange LHSRange = getSignedRange(LHS);
    ConstantRange RHSRange = getSignedRange(RHS);
    if (LHSRange.getSignedMax().sle(RHSRange.getSignedMin()))
      return true;
    if (LHSRange.getSignedMin().sgt(RHSRange.getSignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_UGT:
    Pred = ICmpInst::ICMP_ULT;
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_ULT: {
    ConstantRange LHSRange = getUnsignedRange(LHS);
    ConstantRange RHSRange = getUnsignedRange(RHS);
    if (LHSRange.getUnsignedMax().ult(RHSRange.getUnsignedMin()))
      return true;
    if (LHSRange.getUnsignedMin().uge(RHSRange.getUnsignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_UGE:
    Pred = ICmpInst::ICMP_ULE;
    std::swap(LHS, RHS);
  case ICmpInst::ICMP_ULE: {
    ConstantRange LHSRange = getUnsignedRange(LHS);
    ConstantRange RHSRange = getUnsignedRange(RHS);
    if (LHSRange.getUnsignedMax().ule(RHSRange.getUnsignedMin()))
      return true;
    if (LHSRange.getUnsignedMin().ugt(RHSRange.getUnsignedMax()))
      return false;
    break;
  }
  case ICmpInst::ICMP_NE: {
    if (getUnsignedRange(LHS).intersectWith(getUnsignedRange(RHS)).isEmptySet())
      return true;
    if (getSignedRange(LHS).intersectWith(getSignedRange(RHS)).isEmptySet())
      return true;

    const SCEV *Diff = getMinusSCEV(LHS, RHS);
    if (isKnownNonZero(Diff))
      return true;
    break;
  }
  case ICmpInst::ICMP_EQ:
    // The check at the top of the function catches the case where
    // the values are known to be equal.
    break;
  }
  return false;
}

/// isLoopBackedgeGuardedByCond - Test whether the backedge of the loop is
/// protected by a conditional between LHS and RHS.  This is used to
/// to eliminate casts.
bool
ScalarEvolution::isLoopBackedgeGuardedByCond(const Loop *L,
                                             ICmpInst::Predicate Pred,
                                             const SCEV *LHS, const SCEV *RHS) {
  // Interpret a null as meaning no loop, where there is obviously no guard
  // (interprocedural conditions notwithstanding).
  if (!L) return true;

  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return false;

  BranchInst *LoopContinuePredicate =
    dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LoopContinuePredicate ||
      LoopContinuePredicate->isUnconditional())
    return false;

  return isImpliedCond(Pred, LHS, RHS,
                       LoopContinuePredicate->getCondition(),
                       LoopContinuePredicate->getSuccessor(0) != L->getHeader());
}

/// isLoopEntryGuardedByCond - Test whether entry to the loop is protected
/// by a conditional between LHS and RHS.  This is used to help avoid max
/// expressions in loop trip counts, and to eliminate casts.
bool
ScalarEvolution::isLoopEntryGuardedByCond(const Loop *L,
                                          ICmpInst::Predicate Pred,
                                          const SCEV *LHS, const SCEV *RHS) {
  // Interpret a null as meaning no loop, where there is obviously no guard
  // (interprocedural conditions notwithstanding).
  if (!L) return false;

  // Starting at the loop predecessor, climb up the predecessor chain, as long
  // as there are predecessors that can be found that have unique successors
  // leading to the original header.
  for (std::pair<BasicBlock *, BasicBlock *>
         Pair(L->getLoopPredecessor(), L->getHeader());
       Pair.first;
       Pair = getPredecessorWithUniqueSuccessorForBB(Pair.first)) {

    BranchInst *LoopEntryPredicate =
      dyn_cast<BranchInst>(Pair.first->getTerminator());
    if (!LoopEntryPredicate ||
        LoopEntryPredicate->isUnconditional())
      continue;

    if (isImpliedCond(Pred, LHS, RHS,
                      LoopEntryPredicate->getCondition(),
                      LoopEntryPredicate->getSuccessor(0) != Pair.second))
      return true;
  }

  return false;
}

/// isImpliedCond - Test whether the condition described by Pred, LHS,
/// and RHS is true whenever the given Cond value evaluates to true.
bool ScalarEvolution::isImpliedCond(ICmpInst::Predicate Pred,
                                    const SCEV *LHS, const SCEV *RHS,
                                    Value *FoundCondValue,
                                    bool Inverse) {
  // Recursively handle And and Or conditions.
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FoundCondValue)) {
    if (BO->getOpcode() == Instruction::And) {
      if (!Inverse)
        return isImpliedCond(Pred, LHS, RHS, BO->getOperand(0), Inverse) ||
               isImpliedCond(Pred, LHS, RHS, BO->getOperand(1), Inverse);
    } else if (BO->getOpcode() == Instruction::Or) {
      if (Inverse)
        return isImpliedCond(Pred, LHS, RHS, BO->getOperand(0), Inverse) ||
               isImpliedCond(Pred, LHS, RHS, BO->getOperand(1), Inverse);
    }
  }

  ICmpInst *ICI = dyn_cast<ICmpInst>(FoundCondValue);
  if (!ICI) return false;

  // Bail if the ICmp's operands' types are wider than the needed type
  // before attempting to call getSCEV on them. This avoids infinite
  // recursion, since the analysis of widening casts can require loop
  // exit condition information for overflow checking, which would
  // lead back here.
  if (getTypeSizeInBits(LHS->getType()) <
      getTypeSizeInBits(ICI->getOperand(0)->getType()))
    return false;

  // Now that we found a conditional branch that dominates the loop, check to
  // see if it is the comparison we are looking for.
  ICmpInst::Predicate FoundPred;
  if (Inverse)
    FoundPred = ICI->getInversePredicate();
  else
    FoundPred = ICI->getPredicate();

  const SCEV *FoundLHS = getSCEV(ICI->getOperand(0));
  const SCEV *FoundRHS = getSCEV(ICI->getOperand(1));

  // Balance the types. The case where FoundLHS' type is wider than
  // LHS' type is checked for above.
  if (getTypeSizeInBits(LHS->getType()) >
      getTypeSizeInBits(FoundLHS->getType())) {
    if (CmpInst::isSigned(Pred)) {
      FoundLHS = getSignExtendExpr(FoundLHS, LHS->getType());
      FoundRHS = getSignExtendExpr(FoundRHS, LHS->getType());
    } else {
      FoundLHS = getZeroExtendExpr(FoundLHS, LHS->getType());
      FoundRHS = getZeroExtendExpr(FoundRHS, LHS->getType());
    }
  }

  // Canonicalize the query to match the way instcombine will have
  // canonicalized the comparison.
  if (SimplifyICmpOperands(Pred, LHS, RHS))
    if (LHS == RHS)
      return CmpInst::isTrueWhenEqual(Pred);
  if (SimplifyICmpOperands(FoundPred, FoundLHS, FoundRHS))
    if (FoundLHS == FoundRHS)
      return CmpInst::isFalseWhenEqual(Pred);

  // Check to see if we can make the LHS or RHS match.
  if (LHS == FoundRHS || RHS == FoundLHS) {
    if (isa<SCEVConstant>(RHS)) {
      std::swap(FoundLHS, FoundRHS);
      FoundPred = ICmpInst::getSwappedPredicate(FoundPred);
    } else {
      std::swap(LHS, RHS);
      Pred = ICmpInst::getSwappedPredicate(Pred);
    }
  }

  // Check whether the found predicate is the same as the desired predicate.
  if (FoundPred == Pred)
    return isImpliedCondOperands(Pred, LHS, RHS, FoundLHS, FoundRHS);

  // Check whether swapping the found predicate makes it the same as the
  // desired predicate.
  if (ICmpInst::getSwappedPredicate(FoundPred) == Pred) {
    if (isa<SCEVConstant>(RHS))
      return isImpliedCondOperands(Pred, LHS, RHS, FoundRHS, FoundLHS);
    else
      return isImpliedCondOperands(ICmpInst::getSwappedPredicate(Pred),
                                   RHS, LHS, FoundLHS, FoundRHS);
  }

  // Check whether the actual condition is beyond sufficient.
  if (FoundPred == ICmpInst::ICMP_EQ)
    if (ICmpInst::isTrueWhenEqual(Pred))
      if (isImpliedCondOperands(Pred, LHS, RHS, FoundLHS, FoundRHS))
        return true;
  if (Pred == ICmpInst::ICMP_NE)
    if (!ICmpInst::isTrueWhenEqual(FoundPred))
      if (isImpliedCondOperands(FoundPred, LHS, RHS, FoundLHS, FoundRHS))
        return true;

  // Otherwise assume the worst.
  return false;
}

/// isImpliedCondOperands - Test whether the condition described by Pred,
/// LHS, and RHS is true whenever the condition described by Pred, FoundLHS,
/// and FoundRHS is true.
bool ScalarEvolution::isImpliedCondOperands(ICmpInst::Predicate Pred,
                                            const SCEV *LHS, const SCEV *RHS,
                                            const SCEV *FoundLHS,
                                            const SCEV *FoundRHS) {
  return isImpliedCondOperandsHelper(Pred, LHS, RHS,
                                     FoundLHS, FoundRHS) ||
         // ~x < ~y --> x > y
         isImpliedCondOperandsHelper(Pred, LHS, RHS,
                                     getNotSCEV(FoundRHS),
                                     getNotSCEV(FoundLHS));
}

/// isImpliedCondOperandsHelper - Test whether the condition described by
/// Pred, LHS, and RHS is true whenever the condition described by Pred,
/// FoundLHS, and FoundRHS is true.
bool
ScalarEvolution::isImpliedCondOperandsHelper(ICmpInst::Predicate Pred,
                                             const SCEV *LHS, const SCEV *RHS,
                                             const SCEV *FoundLHS,
                                             const SCEV *FoundRHS) {
  switch (Pred) {
  default: llvm_unreachable("Unexpected ICmpInst::Predicate value!");
  case ICmpInst::ICMP_EQ:
  case ICmpInst::ICMP_NE:
    if (HasSameValue(LHS, FoundLHS) && HasSameValue(RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_SLE:
    if (isKnownPredicateWithRanges(ICmpInst::ICMP_SLE, LHS, FoundLHS) &&
        isKnownPredicateWithRanges(ICmpInst::ICMP_SGE, RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_SGE:
    if (isKnownPredicateWithRanges(ICmpInst::ICMP_SGE, LHS, FoundLHS) &&
        isKnownPredicateWithRanges(ICmpInst::ICMP_SLE, RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_ULE:
    if (isKnownPredicateWithRanges(ICmpInst::ICMP_ULE, LHS, FoundLHS) &&
        isKnownPredicateWithRanges(ICmpInst::ICMP_UGE, RHS, FoundRHS))
      return true;
    break;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_UGE:
    if (isKnownPredicateWithRanges(ICmpInst::ICMP_UGE, LHS, FoundLHS) &&
        isKnownPredicateWithRanges(ICmpInst::ICMP_ULE, RHS, FoundRHS))
      return true;
    break;
  }

  return false;
}

/// getBECount - Subtract the end and start values and divide by the step,
/// rounding up, to get the number of times the backedge is executed. Return
/// CouldNotCompute if an intermediate computation overflows.
const SCEV *ScalarEvolution::getBECount(const SCEV *Start,
                                        const SCEV *End,
                                        const SCEV *Step,
                                        bool NoWrap) {
  assert(!isKnownNegative(Step) &&
         "This code doesn't handle negative strides yet!");

  Type *Ty = Start->getType();

  // When Start == End, we have an exact BECount == 0. Short-circuit this case
  // here because SCEV may not be able to determine that the unsigned division
  // after rounding is zero.
  if (Start == End)
    return getConstant(Ty, 0);

  const SCEV *NegOne = getConstant(Ty, (uint64_t)-1);
  const SCEV *Diff = getMinusSCEV(End, Start);
  const SCEV *RoundUp = getAddExpr(Step, NegOne);

  // Add an adjustment to the difference between End and Start so that
  // the division will effectively round up.
  const SCEV *Add = getAddExpr(Diff, RoundUp);

  if (!NoWrap) {
    // Check Add for unsigned overflow.
    // TODO: More sophisticated things could be done here.
    Type *WideTy = IntegerType::get(getContext(),
                                          getTypeSizeInBits(Ty) + 1);
    const SCEV *EDiff = getZeroExtendExpr(Diff, WideTy);
    const SCEV *ERoundUp = getZeroExtendExpr(RoundUp, WideTy);
    const SCEV *OperandExtendedAdd = getAddExpr(EDiff, ERoundUp);
    if (getZeroExtendExpr(Add, WideTy) != OperandExtendedAdd)
      return getCouldNotCompute();
  }

  return getUDivExpr(Add, Step);
}

/// HowManyLessThans - Return the number of times a backedge containing the
/// specified less-than comparison will execute.  If not computable, return
/// CouldNotCompute.
ScalarEvolution::ExitLimit
ScalarEvolution::HowManyLessThans(const SCEV *LHS, const SCEV *RHS,
                                  const Loop *L, bool isSigned) {
  // Only handle:  "ADDREC < LoopInvariant".
  if (!isLoopInvariant(RHS, L)) return getCouldNotCompute();

  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(LHS);
  if (!AddRec || AddRec->getLoop() != L)
    return getCouldNotCompute();

  // Check to see if we have a flag which makes analysis easy.
  bool NoWrap = isSigned ? AddRec->getNoWrapFlags(SCEV::FlagNSW) :
                           AddRec->getNoWrapFlags(SCEV::FlagNUW);

  if (AddRec->isAffine()) {
    unsigned BitWidth = getTypeSizeInBits(AddRec->getType());
    const SCEV *Step = AddRec->getStepRecurrence(*this);

    if (Step->isZero())
      return getCouldNotCompute();
    if (Step->isOne()) {
      // With unit stride, the iteration never steps past the limit value.
    } else if (isKnownPositive(Step)) {
      // Test whether a positive iteration can step past the limit
      // value and past the maximum value for its type in a single step.
      // Note that it's not sufficient to check NoWrap here, because even
      // though the value after a wrap is undefined, it's not undefined
      // behavior, so if wrap does occur, the loop could either terminate or
      // loop infinitely, but in either case, the loop is guaranteed to
      // iterate at least until the iteration where the wrapping occurs.
      const SCEV *One = getConstant(Step->getType(), 1);
      if (isSigned) {
        APInt Max = APInt::getSignedMaxValue(BitWidth);
        if ((Max - getSignedRange(getMinusSCEV(Step, One)).getSignedMax())
              .slt(getSignedRange(RHS).getSignedMax()))
          return getCouldNotCompute();
      } else {
        APInt Max = APInt::getMaxValue(BitWidth);
        if ((Max - getUnsignedRange(getMinusSCEV(Step, One)).getUnsignedMax())
              .ult(getUnsignedRange(RHS).getUnsignedMax()))
          return getCouldNotCompute();
      }
    } else
      // TODO: Handle negative strides here and below.
      return getCouldNotCompute();

    // We know the LHS is of the form {n,+,s} and the RHS is some loop-invariant
    // m.  So, we count the number of iterations in which {n,+,s} < m is true.
    // Note that we cannot simply return max(m-n,0)/s because it's not safe to
    // treat m-n as signed nor unsigned due to overflow possibility.

    // First, we get the value of the LHS in the first iteration: n
    const SCEV *Start = AddRec->getOperand(0);

    // Determine the minimum constant start value.
    const SCEV *MinStart = getConstant(isSigned ?
      getSignedRange(Start).getSignedMin() :
      getUnsignedRange(Start).getUnsignedMin());

    // If we know that the condition is true in order to enter the loop,
    // then we know that it will run exactly (m-n)/s times. Otherwise, we
    // only know that it will execute (max(m,n)-n)/s times. In both cases,
    // the division must round up.
    const SCEV *End = RHS;
    if (!isLoopEntryGuardedByCond(L,
                                  isSigned ? ICmpInst::ICMP_SLT :
                                             ICmpInst::ICMP_ULT,
                                  getMinusSCEV(Start, Step), RHS))
      End = isSigned ? getSMaxExpr(RHS, Start)
                     : getUMaxExpr(RHS, Start);

    // Determine the maximum constant end value.
    const SCEV *MaxEnd = getConstant(isSigned ?
      getSignedRange(End).getSignedMax() :
      getUnsignedRange(End).getUnsignedMax());

    // If MaxEnd is within a step of the maximum integer value in its type,
    // adjust it down to the minimum value which would produce the same effect.
    // This allows the subsequent ceiling division of (N+(step-1))/step to
    // compute the correct value.
    const SCEV *StepMinusOne = getMinusSCEV(Step,
                                            getConstant(Step->getType(), 1));
    MaxEnd = isSigned ?
      getSMinExpr(MaxEnd,
                  getMinusSCEV(getConstant(APInt::getSignedMaxValue(BitWidth)),
                               StepMinusOne)) :
      getUMinExpr(MaxEnd,
                  getMinusSCEV(getConstant(APInt::getMaxValue(BitWidth)),
                               StepMinusOne));

    // Finally, we subtract these two values and divide, rounding up, to get
    // the number of times the backedge is executed.
    const SCEV *BECount = getBECount(Start, End, Step, NoWrap);

    // The maximum backedge count is similar, except using the minimum start
    // value and the maximum end value.
    // If we already have an exact constant BECount, use it instead.
    const SCEV *MaxBECount = isa<SCEVConstant>(BECount) ? BECount
      : getBECount(MinStart, MaxEnd, Step, NoWrap);

    // If the stride is nonconstant, and NoWrap == true, then
    // getBECount(MinStart, MaxEnd) may not compute. This would result in an
    // exact BECount and invalid MaxBECount, which should be avoided to catch
    // more optimization opportunities.
    if (isa<SCEVCouldNotCompute>(MaxBECount))
      MaxBECount = BECount;

    return ExitLimit(BECount, MaxBECount);
  }

  return getCouldNotCompute();
}

/// getNumIterationsInRange - Return the number of iterations of this loop that
/// produce values in the specified constant range.  Another way of looking at
/// this is that it returns the first iteration number where the value is not in
/// the condition, thus computing the exit count. If the iteration count can't
/// be computed, an instance of SCEVCouldNotCompute is returned.
const SCEV *SCEVAddRecExpr::getNumIterationsInRange(ConstantRange Range,
                                                    ScalarEvolution &SE) const {
  if (Range.isFullSet())  // Infinite loop.
    return SE.getCouldNotCompute();

  // If the start is a non-zero constant, shift the range to simplify things.
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(getStart()))
    if (!SC->getValue()->isZero()) {
      SmallVector<const SCEV *, 4> Operands(op_begin(), op_end());
      Operands[0] = SE.getConstant(SC->getType(), 0);
      const SCEV *Shifted = SE.getAddRecExpr(Operands, getLoop(),
                                             getNoWrapFlags(FlagNW));
      if (const SCEVAddRecExpr *ShiftedAddRec =
            dyn_cast<SCEVAddRecExpr>(Shifted))
        return ShiftedAddRec->getNumIterationsInRange(
                           Range.subtract(SC->getValue()->getValue()), SE);
      // This is strange and shouldn't happen.
      return SE.getCouldNotCompute();
    }

  // The only time we can solve this is when we have all constant indices.
  // Otherwise, we cannot determine the overflow conditions.
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (!isa<SCEVConstant>(getOperand(i)))
      return SE.getCouldNotCompute();


  // Okay at this point we know that all elements of the chrec are constants and
  // that the start element is zero.

  // First check to see if the range contains zero.  If not, the first
  // iteration exits.
  unsigned BitWidth = SE.getTypeSizeInBits(getType());
  if (!Range.contains(APInt(BitWidth, 0)))
    return SE.getConstant(getType(), 0);

  if (isAffine()) {
    // If this is an affine expression then we have this situation:
    //   Solve {0,+,A} in Range  ===  Ax in Range

    // We know that zero is in the range.  If A is positive then we know that
    // the upper value of the range must be the first possible exit value.
    // If A is negative then the lower of the range is the last possible loop
    // value.  Also note that we already checked for a full range.
    APInt One(BitWidth,1);
    APInt A     = cast<SCEVConstant>(getOperand(1))->getValue()->getValue();
    APInt End = A.sge(One) ? (Range.getUpper() - One) : Range.getLower();

    // The exit value should be (End+A)/A.
    APInt ExitVal = (End + A).udiv(A);
    ConstantInt *ExitValue = ConstantInt::get(SE.getContext(), ExitVal);

    // Evaluate at the exit value.  If we really did fall out of the valid
    // range, then we computed our trip count, otherwise wrap around or other
    // things must have happened.
    ConstantInt *Val = EvaluateConstantChrecAtConstant(this, ExitValue, SE);
    if (Range.contains(Val->getValue()))
      return SE.getCouldNotCompute();  // Something strange happened

    // Ensure that the previous value is in the range.  This is a sanity check.
    assert(Range.contains(
           EvaluateConstantChrecAtConstant(this,
           ConstantInt::get(SE.getContext(), ExitVal - One), SE)->getValue()) &&
           "Linear scev computation is off in a bad way!");
    return SE.getConstant(ExitValue);
  } else if (isQuadratic()) {
    // If this is a quadratic (3-term) AddRec {L,+,M,+,N}, find the roots of the
    // quadratic equation to solve it.  To do this, we must frame our problem in
    // terms of figuring out when zero is crossed, instead of when
    // Range.getUpper() is crossed.
    SmallVector<const SCEV *, 4> NewOps(op_begin(), op_end());
    NewOps[0] = SE.getNegativeSCEV(SE.getConstant(Range.getUpper()));
    const SCEV *NewAddRec = SE.getAddRecExpr(NewOps, getLoop(),
                                             // getNoWrapFlags(FlagNW)
                                             FlagAnyWrap);

    // Next, solve the constructed addrec
    std::pair<const SCEV *,const SCEV *> Roots =
      SolveQuadraticEquation(cast<SCEVAddRecExpr>(NewAddRec), SE);
    const SCEVConstant *R1 = dyn_cast<SCEVConstant>(Roots.first);
    const SCEVConstant *R2 = dyn_cast<SCEVConstant>(Roots.second);
    if (R1) {
      // Pick the smallest positive root value.
      if (ConstantInt *CB =
          dyn_cast<ConstantInt>(ConstantExpr::getICmp(ICmpInst::ICMP_ULT,
                         R1->getValue(), R2->getValue()))) {
        if (CB->getZExtValue() == false)
          std::swap(R1, R2);   // R1 is the minimum root now.

        // Make sure the root is not off by one.  The returned iteration should
        // not be in the range, but the previous one should be.  When solving
        // for "X*X < 5", for example, we should not return a root of 2.
        ConstantInt *R1Val = EvaluateConstantChrecAtConstant(this,
                                                             R1->getValue(),
                                                             SE);
        if (Range.contains(R1Val->getValue())) {
          // The next iteration must be out of the range...
          ConstantInt *NextVal =
                ConstantInt::get(SE.getContext(), R1->getValue()->getValue()+1);

          R1Val = EvaluateConstantChrecAtConstant(this, NextVal, SE);
          if (!Range.contains(R1Val->getValue()))
            return SE.getConstant(NextVal);
          return SE.getCouldNotCompute();  // Something strange happened
        }

        // If R1 was not in the range, then it is a good return value.  Make
        // sure that R1-1 WAS in the range though, just in case.
        ConstantInt *NextVal =
               ConstantInt::get(SE.getContext(), R1->getValue()->getValue()-1);
        R1Val = EvaluateConstantChrecAtConstant(this, NextVal, SE);
        if (Range.contains(R1Val->getValue()))
          return R1;
        return SE.getCouldNotCompute();  // Something strange happened
      }
    }
  }

  return SE.getCouldNotCompute();
}



//===----------------------------------------------------------------------===//
//                   SCEVCallbackVH Class Implementation
//===----------------------------------------------------------------------===//

void ScalarEvolution::SCEVCallbackVH::deleted() {
  assert(SE && "SCEVCallbackVH called with a null ScalarEvolution!");
  if (PHINode *PN = dyn_cast<PHINode>(getValPtr()))
    SE->ConstantEvolutionLoopExitValue.erase(PN);
  SE->ValueExprMap.erase(getValPtr());
  // this now dangles!
}

void ScalarEvolution::SCEVCallbackVH::allUsesReplacedWith(Value *V) {
  assert(SE && "SCEVCallbackVH called with a null ScalarEvolution!");

  // Forget all the expressions associated with users of the old value,
  // so that future queries will recompute the expressions using the new
  // value.
  Value *Old = getValPtr();
  SmallVector<User *, 16> Worklist;
  SmallPtrSet<User *, 8> Visited;
  for (Value::use_iterator UI = Old->use_begin(), UE = Old->use_end();
       UI != UE; ++UI)
    Worklist.push_back(*UI);
  while (!Worklist.empty()) {
    User *U = Worklist.pop_back_val();
    // Deleting the Old value will cause this to dangle. Postpone
    // that until everything else is done.
    if (U == Old)
      continue;
    if (!Visited.insert(U))
      continue;
    if (PHINode *PN = dyn_cast<PHINode>(U))
      SE->ConstantEvolutionLoopExitValue.erase(PN);
    SE->ValueExprMap.erase(U);
    for (Value::use_iterator UI = U->use_begin(), UE = U->use_end();
         UI != UE; ++UI)
      Worklist.push_back(*UI);
  }
  // Delete the Old value.
  if (PHINode *PN = dyn_cast<PHINode>(Old))
    SE->ConstantEvolutionLoopExitValue.erase(PN);
  SE->ValueExprMap.erase(Old);
  // this now dangles!
}

ScalarEvolution::SCEVCallbackVH::SCEVCallbackVH(Value *V, ScalarEvolution *se)
  : CallbackVH(V), SE(se) {}

//===----------------------------------------------------------------------===//
//                   ScalarEvolution Class Implementation
//===----------------------------------------------------------------------===//

ScalarEvolution::ScalarEvolution()
  : FunctionPass(ID), FirstUnknown(0) {
  initializeScalarEvolutionPass(*PassRegistry::getPassRegistry());
}

bool ScalarEvolution::runOnFunction(Function &F) {
  this->F = &F;
  LI = &getAnalysis<LoopInfo>();
  TD = getAnalysisIfAvailable<TargetData>();
  DT = &getAnalysis<DominatorTree>();
  return false;
}

void ScalarEvolution::releaseMemory() {
  // Iterate through all the SCEVUnknown instances and call their
  // destructors, so that they release their references to their values.
  for (SCEVUnknown *U = FirstUnknown; U; U = U->Next)
    U->~SCEVUnknown();
  FirstUnknown = 0;

  ValueExprMap.clear();

  // Free any extra memory created for ExitNotTakenInfo in the unlikely event
  // that a loop had multiple computable exits.
  for (DenseMap<const Loop*, BackedgeTakenInfo>::iterator I =
         BackedgeTakenCounts.begin(), E = BackedgeTakenCounts.end();
       I != E; ++I) {
    I->second.clear();
  }

  BackedgeTakenCounts.clear();
  ConstantEvolutionLoopExitValue.clear();
  ValuesAtScopes.clear();
  LoopDispositions.clear();
  BlockDispositions.clear();
  UnsignedRanges.clear();
  SignedRanges.clear();
  UniqueSCEVs.clear();
  SCEVAllocator.Reset();
}

void ScalarEvolution::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<LoopInfo>();
  AU.addRequiredTransitive<DominatorTree>();
}

bool ScalarEvolution::hasLoopInvariantBackedgeTakenCount(const Loop *L) {
  return !isa<SCEVCouldNotCompute>(getBackedgeTakenCount(L));
}

static void PrintLoopInfo(raw_ostream &OS, ScalarEvolution *SE,
                          const Loop *L) {
  // Print all inner loops first
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    PrintLoopInfo(OS, SE, *I);

  OS << "Loop ";
  WriteAsOperand(OS, L->getHeader(), /*PrintType=*/false);
  OS << ": ";

  SmallVector<BasicBlock *, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() != 1)
    OS << "<multiple exits> ";

  if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
    OS << "backedge-taken count is " << *SE->getBackedgeTakenCount(L);
  } else {
    OS << "Unpredictable backedge-taken count. ";
  }

  OS << "\n"
        "Loop ";
  WriteAsOperand(OS, L->getHeader(), /*PrintType=*/false);
  OS << ": ";

  if (!isa<SCEVCouldNotCompute>(SE->getMaxBackedgeTakenCount(L))) {
    OS << "max backedge-taken count is " << *SE->getMaxBackedgeTakenCount(L);
  } else {
    OS << "Unpredictable max backedge-taken count. ";
  }

  OS << "\n";
}

void ScalarEvolution::print(raw_ostream &OS, const Module *) const {
  // ScalarEvolution's implementation of the print method is to print
  // out SCEV values of all instructions that are interesting. Doing
  // this potentially causes it to create new SCEV objects though,
  // which technically conflicts with the const qualifier. This isn't
  // observable from outside the class though, so casting away the
  // const isn't dangerous.
  ScalarEvolution &SE = *const_cast<ScalarEvolution *>(this);

  OS << "Classifying expressions for: ";
  WriteAsOperand(OS, F, /*PrintType=*/false);
  OS << "\n";
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (isSCEVable(I->getType()) && !isa<CmpInst>(*I)) {
      OS << *I << '\n';
      OS << "  -->  ";
      const SCEV *SV = SE.getSCEV(&*I);
      SV->print(OS);

      const Loop *L = LI->getLoopFor((*I).getParent());

      const SCEV *AtUse = SE.getSCEVAtScope(SV, L);
      if (AtUse != SV) {
        OS << "  -->  ";
        AtUse->print(OS);
      }

      if (L) {
        OS << "\t\t" "Exits: ";
        const SCEV *ExitValue = SE.getSCEVAtScope(SV, L->getParentLoop());
        if (!SE.isLoopInvariant(ExitValue, L)) {
          OS << "<<Unknown>>";
        } else {
          OS << *ExitValue;
        }
      }

      OS << "\n";
    }

  OS << "Determining loop execution counts for: ";
  WriteAsOperand(OS, F, /*PrintType=*/false);
  OS << "\n";
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    PrintLoopInfo(OS, &SE, *I);
}

ScalarEvolution::LoopDisposition
ScalarEvolution::getLoopDisposition(const SCEV *S, const Loop *L) {
  std::map<const Loop *, LoopDisposition> &Values = LoopDispositions[S];
  std::pair<std::map<const Loop *, LoopDisposition>::iterator, bool> Pair =
    Values.insert(std::make_pair(L, LoopVariant));
  if (!Pair.second)
    return Pair.first->second;

  LoopDisposition D = computeLoopDisposition(S, L);
  return LoopDispositions[S][L] = D;
}

ScalarEvolution::LoopDisposition
ScalarEvolution::computeLoopDisposition(const SCEV *S, const Loop *L) {
  switch (S->getSCEVType()) {
  case scConstant:
    return LoopInvariant;
  case scTruncate:
  case scZeroExtend:
  case scSignExtend:
    return getLoopDisposition(cast<SCEVCastExpr>(S)->getOperand(), L);
  case scAddRecExpr: {
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(S);

    // If L is the addrec's loop, it's computable.
    if (AR->getLoop() == L)
      return LoopComputable;

    // Add recurrences are never invariant in the function-body (null loop).
    if (!L)
      return LoopVariant;

    // This recurrence is variant w.r.t. L if L contains AR's loop.
    if (L->contains(AR->getLoop()))
      return LoopVariant;

    // This recurrence is invariant w.r.t. L if AR's loop contains L.
    if (AR->getLoop()->contains(L))
      return LoopInvariant;

    // This recurrence is variant w.r.t. L if any of its operands
    // are variant.
    for (SCEVAddRecExpr::op_iterator I = AR->op_begin(), E = AR->op_end();
         I != E; ++I)
      if (!isLoopInvariant(*I, L))
        return LoopVariant;

    // Otherwise it's loop-invariant.
    return LoopInvariant;
  }
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr: {
    const SCEVNAryExpr *NAry = cast<SCEVNAryExpr>(S);
    bool HasVarying = false;
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      LoopDisposition D = getLoopDisposition(*I, L);
      if (D == LoopVariant)
        return LoopVariant;
      if (D == LoopComputable)
        HasVarying = true;
    }
    return HasVarying ? LoopComputable : LoopInvariant;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(S);
    LoopDisposition LD = getLoopDisposition(UDiv->getLHS(), L);
    if (LD == LoopVariant)
      return LoopVariant;
    LoopDisposition RD = getLoopDisposition(UDiv->getRHS(), L);
    if (RD == LoopVariant)
      return LoopVariant;
    return (LD == LoopInvariant && RD == LoopInvariant) ?
           LoopInvariant : LoopComputable;
  }
  case scUnknown:
    // All non-instruction values are loop invariant.  All instructions are loop
    // invariant if they are not contained in the specified loop.
    // Instructions are never considered invariant in the function body
    // (null loop) because they are defined within the "loop".
    if (Instruction *I = dyn_cast<Instruction>(cast<SCEVUnknown>(S)->getValue()))
      return (L && !L->contains(I)) ? LoopInvariant : LoopVariant;
    return LoopInvariant;
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
    return LoopVariant;
  default: break;
  }
  llvm_unreachable("Unknown SCEV kind!");
  return LoopVariant;
}

bool ScalarEvolution::isLoopInvariant(const SCEV *S, const Loop *L) {
  return getLoopDisposition(S, L) == LoopInvariant;
}

bool ScalarEvolution::hasComputableLoopEvolution(const SCEV *S, const Loop *L) {
  return getLoopDisposition(S, L) == LoopComputable;
}

ScalarEvolution::BlockDisposition
ScalarEvolution::getBlockDisposition(const SCEV *S, const BasicBlock *BB) {
  std::map<const BasicBlock *, BlockDisposition> &Values = BlockDispositions[S];
  std::pair<std::map<const BasicBlock *, BlockDisposition>::iterator, bool>
    Pair = Values.insert(std::make_pair(BB, DoesNotDominateBlock));
  if (!Pair.second)
    return Pair.first->second;

  BlockDisposition D = computeBlockDisposition(S, BB);
  return BlockDispositions[S][BB] = D;
}

ScalarEvolution::BlockDisposition
ScalarEvolution::computeBlockDisposition(const SCEV *S, const BasicBlock *BB) {
  switch (S->getSCEVType()) {
  case scConstant:
    return ProperlyDominatesBlock;
  case scTruncate:
  case scZeroExtend:
  case scSignExtend:
    return getBlockDisposition(cast<SCEVCastExpr>(S)->getOperand(), BB);
  case scAddRecExpr: {
    // This uses a "dominates" query instead of "properly dominates" query
    // to test for proper dominance too, because the instruction which
    // produces the addrec's value is a PHI, and a PHI effectively properly
    // dominates its entire containing block.
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(S);
    if (!DT->dominates(AR->getLoop()->getHeader(), BB))
      return DoesNotDominateBlock;
  }
  // FALL THROUGH into SCEVNAryExpr handling.
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr: {
    const SCEVNAryExpr *NAry = cast<SCEVNAryExpr>(S);
    bool Proper = true;
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      BlockDisposition D = getBlockDisposition(*I, BB);
      if (D == DoesNotDominateBlock)
        return DoesNotDominateBlock;
      if (D == DominatesBlock)
        Proper = false;
    }
    return Proper ? ProperlyDominatesBlock : DominatesBlock;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(S);
    const SCEV *LHS = UDiv->getLHS(), *RHS = UDiv->getRHS();
    BlockDisposition LD = getBlockDisposition(LHS, BB);
    if (LD == DoesNotDominateBlock)
      return DoesNotDominateBlock;
    BlockDisposition RD = getBlockDisposition(RHS, BB);
    if (RD == DoesNotDominateBlock)
      return DoesNotDominateBlock;
    return (LD == ProperlyDominatesBlock && RD == ProperlyDominatesBlock) ?
      ProperlyDominatesBlock : DominatesBlock;
  }
  case scUnknown:
    if (Instruction *I =
          dyn_cast<Instruction>(cast<SCEVUnknown>(S)->getValue())) {
      if (I->getParent() == BB)
        return DominatesBlock;
      if (DT->properlyDominates(I->getParent(), BB))
        return ProperlyDominatesBlock;
      return DoesNotDominateBlock;
    }
    return ProperlyDominatesBlock;
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
    return DoesNotDominateBlock;
  default: break;
  }
  llvm_unreachable("Unknown SCEV kind!");
  return DoesNotDominateBlock;
}

bool ScalarEvolution::dominates(const SCEV *S, const BasicBlock *BB) {
  return getBlockDisposition(S, BB) >= DominatesBlock;
}

bool ScalarEvolution::properlyDominates(const SCEV *S, const BasicBlock *BB) {
  return getBlockDisposition(S, BB) == ProperlyDominatesBlock;
}

bool ScalarEvolution::hasOperand(const SCEV *S, const SCEV *Op) const {
  switch (S->getSCEVType()) {
  case scConstant:
    return false;
  case scTruncate:
  case scZeroExtend:
  case scSignExtend: {
    const SCEVCastExpr *Cast = cast<SCEVCastExpr>(S);
    const SCEV *CastOp = Cast->getOperand();
    return Op == CastOp || hasOperand(CastOp, Op);
  }
  case scAddRecExpr:
  case scAddExpr:
  case scMulExpr:
  case scUMaxExpr:
  case scSMaxExpr: {
    const SCEVNAryExpr *NAry = cast<SCEVNAryExpr>(S);
    for (SCEVNAryExpr::op_iterator I = NAry->op_begin(), E = NAry->op_end();
         I != E; ++I) {
      const SCEV *NAryOp = *I;
      if (NAryOp == Op || hasOperand(NAryOp, Op))
        return true;
    }
    return false;
  }
  case scUDivExpr: {
    const SCEVUDivExpr *UDiv = cast<SCEVUDivExpr>(S);
    const SCEV *LHS = UDiv->getLHS(), *RHS = UDiv->getRHS();
    return LHS == Op || hasOperand(LHS, Op) ||
           RHS == Op || hasOperand(RHS, Op);
  }
  case scUnknown:
    return false;
  case scCouldNotCompute:
    llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
    return false;
  default: break;
  }
  llvm_unreachable("Unknown SCEV kind!");
  return false;
}

void ScalarEvolution::forgetMemoizedResults(const SCEV *S) {
  ValuesAtScopes.erase(S);
  LoopDispositions.erase(S);
  BlockDispositions.erase(S);
  UnsignedRanges.erase(S);
  SignedRanges.erase(S);
}

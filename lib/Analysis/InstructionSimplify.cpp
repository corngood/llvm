//===- InstructionSimplify.cpp - Fold instruction operands ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements routines for folding instructions into simpler forms
// that do not require creating new instructions.  For example, this does
// constant folding, and can handle identities like (X&0)->0.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Instructions.h"
using namespace llvm;


/// SimplifyBinOp - Given operands for a BinaryOperator, see if we can
/// fold the result.  If not, this returns null.
Value *llvm::SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS, 
                           const TargetData *TD) {
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS)) {
      Constant *COps[] = {CLHS, CRHS};
      return ConstantFoldInstOperands(Opcode, LHS->getType(), COps, 2, TD);
    }     
  return 0;
}

static const Type *GetCompareTy(Value *Op) {
  return CmpInst::makeCmpResultType(Op->getType());
}


/// SimplifyICmpInst - Given operands for an ICmpInst, see if we can
/// fold the result.  If not, this returns null.
Value *llvm::SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const TargetData *TD) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isIntPredicate(Pred) && "Not an integer compare!");
  
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, TD);
  
  // ITy - This is the return type of the compare we're considering.
  const Type *ITy = GetCompareTy(LHS);
  
  // icmp X, X -> true/false
  if (LHS == RHS)
    return ConstantInt::get(ITy, CmpInst::isTrueWhenEqual(Pred));

  // If we have a constant, make sure it is on the RHS.
  if (isa<Constant>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }

  if (isa<UndefValue>(RHS))                  // X icmp undef -> undef
    return UndefValue::get(ITy);
  
  // icmp <global/alloca*/null>, <global/alloca*/null> - Global/Stack value
  // addresses never equal each other!  We already know that Op0 != Op1.
  if ((isa<GlobalValue>(LHS) || isa<AllocaInst>(LHS) || 
       isa<ConstantPointerNull>(LHS)) &&
      (isa<GlobalValue>(RHS) || isa<AllocaInst>(RHS) || 
       isa<ConstantPointerNull>(RHS)))
    return ConstantInt::get(ITy, CmpInst::isFalseWhenEqual(Pred));
  
  // See if we are doing a comparison with a constant.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
    // If we have an icmp le or icmp ge instruction, turn it into the
    // appropriate icmp lt or icmp gt instruction.  This allows us to rely on
    // them being folded in the code below.
    switch (Pred) {
    default: break;
    case ICmpInst::ICMP_ULE:
      if (CI->isMaxValue(false))                 // A <=u MAX -> TRUE
        return ConstantInt::getTrue(CI->getContext());
      break;
    case ICmpInst::ICMP_SLE:
      if (CI->isMaxValue(true))                  // A <=s MAX -> TRUE
        return ConstantInt::getTrue(CI->getContext());
      break;
    case ICmpInst::ICMP_UGE:
      if (CI->isMinValue(false))                 // A >=u MIN -> TRUE
        return ConstantInt::getTrue(CI->getContext());
      break;
    case ICmpInst::ICMP_SGE:
      if (CI->isMinValue(true))                  // A >=s MIN -> TRUE
        return ConstantInt::getTrue(CI->getContext());
      break;
    }
    
    
  }
  
  
  return 0;
}

/// SimplifyFCmpInst - Given operands for an FCmpInst, see if we can
/// fold the result.  If not, this returns null.
Value *llvm::SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const TargetData *TD) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isFPPredicate(Pred) && "Not an FP compare!");

  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, TD);
  
  // Fold trivial predicates.
  if (Pred == FCmpInst::FCMP_FALSE)
    return ConstantInt::get(GetCompareTy(LHS), 0);
  if (Pred == FCmpInst::FCMP_TRUE)
    return ConstantInt::get(GetCompareTy(LHS), 1);

  // If we have a constant, make sure it is on the RHS.
  if (isa<Constant>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  
  if (isa<UndefValue>(RHS))                  // fcmp pred X, undef -> undef
    return UndefValue::get(GetCompareTy(LHS));

  // fcmp x,x -> true/false.  Not all compares are foldable.
  if (LHS == RHS) {
    if (CmpInst::isTrueWhenEqual(Pred))
      return ConstantInt::get(GetCompareTy(LHS), 1);
    if (CmpInst::isFalseWhenEqual(Pred))
      return ConstantInt::get(GetCompareTy(LHS), 0);
  }
  
  // Handle fcmp with constant RHS
  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // If the constant is a nan, see if we can fold the comparison based on it.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->getValueAPF().isNaN()) {
        if (FCmpInst::isOrdered(Pred))   // True "if ordered and foo"
          return ConstantInt::getFalse(CFP->getContext());
        assert(FCmpInst::isUnordered(Pred) &&
               "Comparison must be either ordered or unordered!");
        // True if unordered.
        return ConstantInt::getTrue(CFP->getContext());
      }
    }
  }
  
  return 0;
}



/// SimplifyCmpInst - Given operands for a CmpInst, see if we can
/// fold the result.
Value *llvm::SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                             const TargetData *TD) {
  if (CmpInst::isIntPredicate((CmpInst::Predicate)Predicate))
    return SimplifyICmpInst(Predicate, LHS, RHS, TD);
  return SimplifyFCmpInst(Predicate, LHS, RHS, TD);
}


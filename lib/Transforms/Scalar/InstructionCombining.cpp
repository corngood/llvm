//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG This pass is where algebraic
// simplification happens.
//
// This pass combines things like:
//    %Y = add int %X, 1
//    %Z = add int %Y, 1
// into:
//    %Z = add int %X, 2
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. SetCC instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All SetCC instructions on boolean values are replaced with logical ops
//    5. add X, X is represented as (X*2) => (X << 1)
//    6. Multiplies with a power-of-two constant argument are transformed into
//       shifts.
//   ... etc.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
  Statistic<> NumCombined ("instcombine", "Number of insts combined");
  Statistic<> NumConstProp("instcombine", "Number of constant folds");
  Statistic<> NumDeadInst ("instcombine", "Number of dead inst eliminated");
  Statistic<> NumSunkInst ("instcombine", "Number of instructions sunk");

  class InstCombiner : public FunctionPass,
                       public InstVisitor<InstCombiner, Instruction*> {
    // Worklist of all of the instructions that need to be simplified.
    std::vector<Instruction*> WorkList;
    TargetData *TD;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(Instruction &I) {
      for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
           UI != UE; ++UI)
        WorkList.push_back(cast<Instruction>(*UI));
    }

    /// AddUsesToWorkList - When an instruction is simplified, add operands to
    /// the work lists because they might get more simplified now.
    ///
    void AddUsesToWorkList(Instruction &I) {
      for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(I.getOperand(i)))
          WorkList.push_back(Op);
    }

    // removeFromWorkList - remove all instances of I from the worklist.
    void removeFromWorkList(Instruction *I);
  public:
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();
    }

    TargetData &getTargetData() const { return *TD; }

    // Visitation implementation - Implement instruction combining for different
    // instruction types.  The semantics are as follows:
    // Return Value:
    //    null        - No change was made
    //     I          - Change was made, I is still valid, I may be dead though
    //   otherwise    - Change was made, replace I with returned instruction
    //
    Instruction *visitAdd(BinaryOperator &I);
    Instruction *visitSub(BinaryOperator &I);
    Instruction *visitMul(BinaryOperator &I);
    Instruction *visitDiv(BinaryOperator &I);
    Instruction *visitRem(BinaryOperator &I);
    Instruction *visitAnd(BinaryOperator &I);
    Instruction *visitOr (BinaryOperator &I);
    Instruction *visitXor(BinaryOperator &I);
    Instruction *visitSetCondInst(SetCondInst &I);
    Instruction *visitSetCondInstWithCastAndCast(SetCondInst &SCI);

    Instruction *FoldGEPSetCC(User *GEPLHS, Value *RHS,
                              Instruction::BinaryOps Cond, Instruction &I);
    Instruction *visitShiftInst(ShiftInst &I);
    Instruction *FoldShiftByConstant(Value *Op0, ConstantUInt *Op1,
                                     ShiftInst &I);
    Instruction *visitCastInst(CastInst &CI);
    Instruction *FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                Instruction *FI);
    Instruction *visitSelectInst(SelectInst &CI);
    Instruction *visitCallInst(CallInst &CI);
    Instruction *visitInvokeInst(InvokeInst &II);
    Instruction *visitPHINode(PHINode &PN);
    Instruction *visitGetElementPtrInst(GetElementPtrInst &GEP);
    Instruction *visitAllocationInst(AllocationInst &AI);
    Instruction *visitFreeInst(FreeInst &FI);
    Instruction *visitLoadInst(LoadInst &LI);
    Instruction *visitStoreInst(StoreInst &SI);
    Instruction *visitBranchInst(BranchInst &BI);
    Instruction *visitSwitchInst(SwitchInst &SI);

    // visitInstruction - Specify what to return for unhandled instructions...
    Instruction *visitInstruction(Instruction &I) { return 0; }

  private:
    Instruction *visitCallSite(CallSite CS);
    bool transformConstExprCastCall(CallSite CS);

  public:
    // InsertNewInstBefore - insert an instruction New before instruction Old
    // in the program.  Add the new instruction to the worklist.
    //
    Instruction *InsertNewInstBefore(Instruction *New, Instruction &Old) {
      assert(New && New->getParent() == 0 &&
             "New instruction already inserted into a basic block!");
      BasicBlock *BB = Old.getParent();
      BB->getInstList().insert(&Old, New);  // Insert inst
      WorkList.push_back(New);              // Add to worklist
      return New;
    }

    /// InsertCastBefore - Insert a cast of V to TY before the instruction POS.
    /// This also adds the cast to the worklist.  Finally, this returns the
    /// cast.
    Value *InsertCastBefore(Value *V, const Type *Ty, Instruction &Pos) {
      if (V->getType() == Ty) return V;

      Instruction *C = new CastInst(V, Ty, V->getName(), &Pos);
      WorkList.push_back(C);
      return C;
    }

    // ReplaceInstUsesWith - This method is to be used when an instruction is
    // found to be dead, replacable with another preexisting expression.  Here
    // we add all uses of I to the worklist, replace all uses of I with the new
    // value, then return I, so that the inst combiner will know that I was
    // modified.
    //
    Instruction *ReplaceInstUsesWith(Instruction &I, Value *V) {
      AddUsersToWorkList(I);         // Add all modified instrs to worklist
      if (&I != V) {
        I.replaceAllUsesWith(V);
        return &I;
      } else {
        // If we are replacing the instruction with itself, this must be in a
        // segment of unreachable code, so just clobber the instruction.
        I.replaceAllUsesWith(UndefValue::get(I.getType()));
        return &I;
      }
    }

    // EraseInstFromFunction - When dealing with an instruction that has side
    // effects or produces a void value, we can't rely on DCE to delete the
    // instruction.  Instead, visit methods should return the value returned by
    // this function.
    Instruction *EraseInstFromFunction(Instruction &I) {
      assert(I.use_empty() && "Cannot erase instruction that is used!");
      AddUsesToWorkList(I);
      removeFromWorkList(&I);
      I.eraseFromParent();
      return 0;  // Don't do anything with FI
    }


  private:
    /// InsertOperandCastBefore - This inserts a cast of V to DestTy before the
    /// InsertBefore instruction.  This is specialized a bit to avoid inserting
    /// casts that are known to not do anything...
    ///
    Value *InsertOperandCastBefore(Value *V, const Type *DestTy,
                                   Instruction *InsertBefore);

    // SimplifyCommutative - This performs a few simplifications for commutative
    // operators.
    bool SimplifyCommutative(BinaryOperator &I);


    // FoldOpIntoPhi - Given a binary operator or cast instruction which has a
    // PHI node as operand #0, see if we can fold the instruction into the PHI
    // (which is only possible if all operands to the PHI are constants).
    Instruction *FoldOpIntoPhi(Instruction &I);

    // FoldPHIArgOpIntoPHI - If all operands to a PHI node are the same "unary"
    // operator and they all are only used by the PHI, PHI together their
    // inputs, and do the operation once, to the result of the PHI.
    Instruction *FoldPHIArgOpIntoPHI(PHINode &PN);

    Instruction *OptAndOp(Instruction *Op, ConstantIntegral *OpRHS,
                          ConstantIntegral *AndRHS, BinaryOperator &TheAnd);
    
    Value *FoldLogicalPlusAnd(Value *LHS, Value *RHS, ConstantIntegral *Mask,
                              bool isSub, Instruction &I);
    Instruction *InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                 bool Inside, Instruction &IB);
    Instruction *PromoteCastOfAllocation(CastInst &CI, AllocationInst &AI);
  };

  RegisterOpt<InstCombiner> X("instcombine", "Combine redundant instructions");
}

// getComplexity:  Assign a complexity or rank value to LLVM Values...
//   0 -> undef, 1 -> Const, 2 -> Other, 3 -> Arg, 3 -> Unary, 4 -> OtherInst
static unsigned getComplexity(Value *V) {
  if (isa<Instruction>(V)) {
    if (BinaryOperator::isNeg(V) || BinaryOperator::isNot(V))
      return 3;
    return 4;
  }
  if (isa<Argument>(V)) return 3;
  return isa<Constant>(V) ? (isa<UndefValue>(V) ? 0 : 1) : 2;
}

// isOnlyUse - Return true if this instruction will be deleted if we stop using
// it.
static bool isOnlyUse(Value *V) {
  return V->hasOneUse() || isa<Constant>(V);
}

// getPromotedType - Return the specified type promoted as it would be to pass
// though a va_arg area...
static const Type *getPromotedType(const Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::SByteTyID:
  case Type::ShortTyID:  return Type::IntTy;
  case Type::UByteTyID:
  case Type::UShortTyID: return Type::UIntTy;
  case Type::FloatTyID:  return Type::DoubleTy;
  default:               return Ty;
  }
}

/// isCast - If the specified operand is a CastInst or a constant expr cast,
/// return the operand value, otherwise return null.
static Value *isCast(Value *V) {
  if (CastInst *I = dyn_cast<CastInst>(V))
    return I->getOperand(0);
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::Cast)
      return CE->getOperand(0);
  return 0;
}

// SimplifyCommutative - This performs a few simplifications for commutative
// operators:
//
//  1. Order operands such that they are listed from right (least complex) to
//     left (most complex).  This puts constants before unary operators before
//     binary operators.
//
//  2. Transform: (op (op V, C1), C2) ==> (op V, (op C1, C2))
//  3. Transform: (op (op V1, C1), (op V2, C2)) ==> (op (op V1, V2), (op C1,C2))
//
bool InstCombiner::SimplifyCommutative(BinaryOperator &I) {
  bool Changed = false;
  if (getComplexity(I.getOperand(0)) < getComplexity(I.getOperand(1)))
    Changed = !I.swapOperands();

  if (!I.isAssociative()) return Changed;
  Instruction::BinaryOps Opcode = I.getOpcode();
  if (BinaryOperator *Op = dyn_cast<BinaryOperator>(I.getOperand(0)))
    if (Op->getOpcode() == Opcode && isa<Constant>(Op->getOperand(1))) {
      if (isa<Constant>(I.getOperand(1))) {
        Constant *Folded = ConstantExpr::get(I.getOpcode(),
                                             cast<Constant>(I.getOperand(1)),
                                             cast<Constant>(Op->getOperand(1)));
        I.setOperand(0, Op->getOperand(0));
        I.setOperand(1, Folded);
        return true;
      } else if (BinaryOperator *Op1=dyn_cast<BinaryOperator>(I.getOperand(1)))
        if (Op1->getOpcode() == Opcode && isa<Constant>(Op1->getOperand(1)) &&
            isOnlyUse(Op) && isOnlyUse(Op1)) {
          Constant *C1 = cast<Constant>(Op->getOperand(1));
          Constant *C2 = cast<Constant>(Op1->getOperand(1));

          // Fold (op (op V1, C1), (op V2, C2)) ==> (op (op V1, V2), (op C1,C2))
          Constant *Folded = ConstantExpr::get(I.getOpcode(), C1, C2);
          Instruction *New = BinaryOperator::create(Opcode, Op->getOperand(0),
                                                    Op1->getOperand(0),
                                                    Op1->getName(), &I);
          WorkList.push_back(New);
          I.setOperand(0, New);
          I.setOperand(1, Folded);
          return true;
        }
    }
  return Changed;
}

// dyn_castNegVal - Given a 'sub' instruction, return the RHS of the instruction
// if the LHS is a constant zero (which is the 'negate' form).
//
static inline Value *dyn_castNegVal(Value *V) {
  if (BinaryOperator::isNeg(V))
    return BinaryOperator::getNegArgument(V);

  // Constants can be considered to be negated values if they can be folded.
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantExpr::getNeg(C);
  return 0;
}

static inline Value *dyn_castNotVal(Value *V) {
  if (BinaryOperator::isNot(V))
    return BinaryOperator::getNotArgument(V);

  // Constants can be considered to be not'ed values...
  if (ConstantIntegral *C = dyn_cast<ConstantIntegral>(V))
    return ConstantExpr::getNot(C);
  return 0;
}

// dyn_castFoldableMul - If this value is a multiply that can be folded into
// other computations (because it has a constant operand), return the
// non-constant operand of the multiply, and set CST to point to the multiplier.
// Otherwise, return null.
//
static inline Value *dyn_castFoldableMul(Value *V, ConstantInt *&CST) {
  if (V->hasOneUse() && V->getType()->isInteger())
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (I->getOpcode() == Instruction::Mul)
        if ((CST = dyn_cast<ConstantInt>(I->getOperand(1))))
          return I->getOperand(0);
      if (I->getOpcode() == Instruction::Shl)
        if ((CST = dyn_cast<ConstantInt>(I->getOperand(1)))) {
          // The multiplier is really 1 << CST.
          Constant *One = ConstantInt::get(V->getType(), 1);
          CST = cast<ConstantInt>(ConstantExpr::getShl(One, CST));
          return I->getOperand(0);
        }
    }
  return 0;
}

/// dyn_castGetElementPtr - If this is a getelementptr instruction or constant
/// expression, return it.
static User *dyn_castGetElementPtr(Value *V) {
  if (isa<GetElementPtrInst>(V)) return cast<User>(V);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    if (CE->getOpcode() == Instruction::GetElementPtr)
      return cast<User>(V);
  return false;
}

// AddOne, SubOne - Add or subtract a constant one from an integer constant...
static ConstantInt *AddOne(ConstantInt *C) {
  return cast<ConstantInt>(ConstantExpr::getAdd(C,
                                         ConstantInt::get(C->getType(), 1)));
}
static ConstantInt *SubOne(ConstantInt *C) {
  return cast<ConstantInt>(ConstantExpr::getSub(C,
                                         ConstantInt::get(C->getType(), 1)));
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  V and Mask are known to
/// be the same type.
static bool MaskedValueIsZero(Value *V, ConstantIntegral *Mask, 
                              unsigned Depth = 0) {
  // Note, we cannot consider 'undef' to be "IsZero" here.  The problem is that
  // we cannot optimize based on the assumption that it is zero without changing
  // to to an explicit zero.  If we don't change it to zero, other code could
  // optimized based on the contradictory assumption that it is non-zero.
  // Because instcombine aggressively folds operations with undef args anyway,
  // this won't lose us code quality.
  if (Mask->isNullValue())
    return true;
  if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V))
    return ConstantExpr::getAnd(CI, Mask)->isNullValue();

  if (Depth == 6) return false;  // Limit search depth.
  
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    switch (I->getOpcode()) {
    case Instruction::And:
      // (X & C1) & C2 == 0   iff   C1 & C2 == 0.
      if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(I->getOperand(1))) {
        ConstantIntegral *C1C2 = 
          cast<ConstantIntegral>(ConstantExpr::getAnd(CI, Mask));
        if (MaskedValueIsZero(I->getOperand(0), C1C2, Depth+1))
          return true;
      }
      // If either the LHS or the RHS are MaskedValueIsZero, the result is zero.
      return MaskedValueIsZero(I->getOperand(1), Mask, Depth+1) ||
             MaskedValueIsZero(I->getOperand(0), Mask, Depth+1);
    case Instruction::Or:
    case Instruction::Xor:
      // If the LHS and the RHS are MaskedValueIsZero, the result is also zero.
      return MaskedValueIsZero(I->getOperand(1), Mask, Depth+1) &&
             MaskedValueIsZero(I->getOperand(0), Mask, Depth+1);
    case Instruction::Select:
      // If the T and F values are MaskedValueIsZero, the result is also zero.
      return MaskedValueIsZero(I->getOperand(2), Mask, Depth+1) &&
             MaskedValueIsZero(I->getOperand(1), Mask, Depth+1);
    case Instruction::Cast: {
      const Type *SrcTy = I->getOperand(0)->getType();
      if (SrcTy == Type::BoolTy)
        return (Mask->getRawValue() & 1) == 0;
      
      if (SrcTy->isInteger()) {
        // (cast <ty> X to int) & C2 == 0  iff <ty> could not have contained C2.
        if (SrcTy->isUnsigned() &&                      // Only handle zero ext.
            ConstantExpr::getCast(Mask, SrcTy)->isNullValue())
          return true;
        
        // If this is a noop cast, recurse.
        if ((SrcTy->isSigned() && SrcTy->getUnsignedVersion() == I->getType())||
            SrcTy->getSignedVersion() == I->getType()) {
          Constant *NewMask =
          ConstantExpr::getCast(Mask, I->getOperand(0)->getType());
          return MaskedValueIsZero(I->getOperand(0),
                                   cast<ConstantIntegral>(NewMask), Depth+1);
        }
      }
      break;
    }
    case Instruction::Shl:
      // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
      if (ConstantUInt *SA = dyn_cast<ConstantUInt>(I->getOperand(1)))
        return MaskedValueIsZero(I->getOperand(0),
                    cast<ConstantIntegral>(ConstantExpr::getUShr(Mask, SA)), 
                                 Depth+1);
      break;
    case Instruction::Shr:
      // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
      if (ConstantUInt *SA = dyn_cast<ConstantUInt>(I->getOperand(1)))
        if (I->getType()->isUnsigned()) {
          Constant *C1 = ConstantIntegral::getAllOnesValue(I->getType());
          C1 = ConstantExpr::getShr(C1, SA);
          C1 = ConstantExpr::getAnd(C1, Mask);
          if (C1->isNullValue())
            return true;
        }
      break;
    }
  }
  
  return false;
}

// isTrueWhenEqual - Return true if the specified setcondinst instruction is
// true when both operands are equal...
//
static bool isTrueWhenEqual(Instruction &I) {
  return I.getOpcode() == Instruction::SetEQ ||
         I.getOpcode() == Instruction::SetGE ||
         I.getOpcode() == Instruction::SetLE;
}

/// AssociativeOpt - Perform an optimization on an associative operator.  This
/// function is designed to check a chain of associative operators for a
/// potential to apply a certain optimization.  Since the optimization may be
/// applicable if the expression was reassociated, this checks the chain, then
/// reassociates the expression as necessary to expose the optimization
/// opportunity.  This makes use of a special Functor, which must define
/// 'shouldApply' and 'apply' methods.
///
template<typename Functor>
Instruction *AssociativeOpt(BinaryOperator &Root, const Functor &F) {
  unsigned Opcode = Root.getOpcode();
  Value *LHS = Root.getOperand(0);

  // Quick check, see if the immediate LHS matches...
  if (F.shouldApply(LHS))
    return F.apply(Root);

  // Otherwise, if the LHS is not of the same opcode as the root, return.
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  while (LHSI && LHSI->getOpcode() == Opcode && LHSI->hasOneUse()) {
    // Should we apply this transform to the RHS?
    bool ShouldApply = F.shouldApply(LHSI->getOperand(1));

    // If not to the RHS, check to see if we should apply to the LHS...
    if (!ShouldApply && F.shouldApply(LHSI->getOperand(0))) {
      cast<BinaryOperator>(LHSI)->swapOperands();   // Make the LHS the RHS
      ShouldApply = true;
    }

    // If the functor wants to apply the optimization to the RHS of LHSI,
    // reassociate the expression from ((? op A) op B) to (? op (A op B))
    if (ShouldApply) {
      BasicBlock *BB = Root.getParent();

      // Now all of the instructions are in the current basic block, go ahead
      // and perform the reassociation.
      Instruction *TmpLHSI = cast<Instruction>(Root.getOperand(0));

      // First move the selected RHS to the LHS of the root...
      Root.setOperand(0, LHSI->getOperand(1));

      // Make what used to be the LHS of the root be the user of the root...
      Value *ExtraOperand = TmpLHSI->getOperand(1);
      if (&Root == TmpLHSI) {
        Root.replaceAllUsesWith(Constant::getNullValue(TmpLHSI->getType()));
        return 0;
      }
      Root.replaceAllUsesWith(TmpLHSI);          // Users now use TmpLHSI
      TmpLHSI->setOperand(1, &Root);             // TmpLHSI now uses the root
      TmpLHSI->getParent()->getInstList().remove(TmpLHSI);
      BasicBlock::iterator ARI = &Root; ++ARI;
      BB->getInstList().insert(ARI, TmpLHSI);    // Move TmpLHSI to after Root
      ARI = Root;

      // Now propagate the ExtraOperand down the chain of instructions until we
      // get to LHSI.
      while (TmpLHSI != LHSI) {
        Instruction *NextLHSI = cast<Instruction>(TmpLHSI->getOperand(0));
        // Move the instruction to immediately before the chain we are
        // constructing to avoid breaking dominance properties.
        NextLHSI->getParent()->getInstList().remove(NextLHSI);
        BB->getInstList().insert(ARI, NextLHSI);
        ARI = NextLHSI;

        Value *NextOp = NextLHSI->getOperand(1);
        NextLHSI->setOperand(1, ExtraOperand);
        TmpLHSI = NextLHSI;
        ExtraOperand = NextOp;
      }

      // Now that the instructions are reassociated, have the functor perform
      // the transformation...
      return F.apply(Root);
    }

    LHSI = dyn_cast<Instruction>(LHSI->getOperand(0));
  }
  return 0;
}


// AddRHS - Implements: X + X --> X << 1
struct AddRHS {
  Value *RHS;
  AddRHS(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Add) const {
    return new ShiftInst(Instruction::Shl, Add.getOperand(0),
                         ConstantInt::get(Type::UByteTy, 1));
  }
};

// AddMaskingAnd - Implements (A & C1)+(B & C2) --> (A & C1)|(B & C2)
//                 iff C1&C2 == 0
struct AddMaskingAnd {
  Constant *C2;
  AddMaskingAnd(Constant *c) : C2(c) {}
  bool shouldApply(Value *LHS) const {
    ConstantInt *C1;
    return match(LHS, m_And(m_Value(), m_ConstantInt(C1))) &&
           ConstantExpr::getAnd(C1, C2)->isNullValue();
  }
  Instruction *apply(BinaryOperator &Add) const {
    return BinaryOperator::createOr(Add.getOperand(0), Add.getOperand(1));
  }
};

static Value *FoldOperationIntoSelectOperand(Instruction &I, Value *SO,
                                             InstCombiner *IC) {
  if (isa<CastInst>(I)) {
    if (Constant *SOC = dyn_cast<Constant>(SO))
      return ConstantExpr::getCast(SOC, I.getType());

    return IC->InsertNewInstBefore(new CastInst(SO, I.getType(),
                                                SO->getName() + ".cast"), I);
  }

  // Figure out if the constant is the left or the right argument.
  bool ConstIsRHS = isa<Constant>(I.getOperand(1));
  Constant *ConstOperand = cast<Constant>(I.getOperand(ConstIsRHS));

  if (Constant *SOC = dyn_cast<Constant>(SO)) {
    if (ConstIsRHS)
      return ConstantExpr::get(I.getOpcode(), SOC, ConstOperand);
    return ConstantExpr::get(I.getOpcode(), ConstOperand, SOC);
  }

  Value *Op0 = SO, *Op1 = ConstOperand;
  if (!ConstIsRHS)
    std::swap(Op0, Op1);
  Instruction *New;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I))
    New = BinaryOperator::create(BO->getOpcode(), Op0, Op1,SO->getName()+".op");
  else if (ShiftInst *SI = dyn_cast<ShiftInst>(&I))
    New = new ShiftInst(SI->getOpcode(), Op0, Op1, SO->getName()+".sh");
  else {
    assert(0 && "Unknown binary instruction type!");
    abort();
  }
  return IC->InsertNewInstBefore(New, I);
}

// FoldOpIntoSelect - Given an instruction with a select as one operand and a
// constant as the other operand, try to fold the binary operator into the
// select arguments.  This also works for Cast instructions, which obviously do
// not have a second operand.
static Instruction *FoldOpIntoSelect(Instruction &Op, SelectInst *SI,
                                     InstCombiner *IC) {
  // Don't modify shared select instructions
  if (!SI->hasOneUse()) return 0;
  Value *TV = SI->getOperand(1);
  Value *FV = SI->getOperand(2);

  if (isa<Constant>(TV) || isa<Constant>(FV)) {
    // Bool selects with constant operands can be folded to logical ops.
    if (SI->getType() == Type::BoolTy) return 0;

    Value *SelectTrueVal = FoldOperationIntoSelectOperand(Op, TV, IC);
    Value *SelectFalseVal = FoldOperationIntoSelectOperand(Op, FV, IC);

    return new SelectInst(SI->getCondition(), SelectTrueVal,
                          SelectFalseVal);
  }
  return 0;
}


/// FoldOpIntoPhi - Given a binary operator or cast instruction which has a PHI
/// node as operand #0, see if we can fold the instruction into the PHI (which
/// is only possible if all operands to the PHI are constants).
Instruction *InstCombiner::FoldOpIntoPhi(Instruction &I) {
  PHINode *PN = cast<PHINode>(I.getOperand(0));
  unsigned NumPHIValues = PN->getNumIncomingValues();
  if (!PN->hasOneUse() || NumPHIValues == 0 ||
      !isa<Constant>(PN->getIncomingValue(0))) return 0;

  // Check to see if all of the operands of the PHI are constants.  If not, we
  // cannot do the transformation.
  for (unsigned i = 1; i != NumPHIValues; ++i)
    if (!isa<Constant>(PN->getIncomingValue(i)))
      return 0;

  // Okay, we can do the transformation: create the new PHI node.
  PHINode *NewPN = new PHINode(I.getType(), I.getName());
  I.setName("");
  NewPN->reserveOperandSpace(PN->getNumOperands()/2);
  InsertNewInstBefore(NewPN, *PN);

  // Next, add all of the operands to the PHI.
  if (I.getNumOperands() == 2) {
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Constant *InV = cast<Constant>(PN->getIncomingValue(i));
      NewPN->addIncoming(ConstantExpr::get(I.getOpcode(), InV, C),
                         PN->getIncomingBlock(i));
    }
  } else {
    assert(isa<CastInst>(I) && "Unary op should be a cast!");
    const Type *RetTy = I.getType();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Constant *InV = cast<Constant>(PN->getIncomingValue(i));
      NewPN->addIncoming(ConstantExpr::getCast(InV, RetTy),
                         PN->getIncomingBlock(i));
    }
  }
  return ReplaceInstUsesWith(I, NewPN);
}

Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // X + undef -> undef
    if (isa<UndefValue>(RHS))
      return ReplaceInstUsesWith(I, RHS);

    // X + 0 --> X
    if (!I.getType()->isFloatingPoint()) { // NOTE: -0 + +0 = +0.
      if (RHSC->isNullValue())
        return ReplaceInstUsesWith(I, LHS);
    } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->isExactlyValue(-0.0))
        return ReplaceInstUsesWith(I, LHS);
    }

    // X + (signbit) --> X ^ signbit
    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHSC)) {
      unsigned NumBits = CI->getType()->getPrimitiveSizeInBits();
      uint64_t Val = CI->getRawValue() & (~0ULL >> (64- NumBits));
      if (Val == (1ULL << (NumBits-1)))
        return BinaryOperator::createXor(LHS, RHS);
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
    
    ConstantInt *XorRHS = 0;
    Value *XorLHS = 0;
    if (match(LHS, m_Xor(m_Value(XorLHS), m_ConstantInt(XorRHS)))) {
      unsigned TySizeBits = I.getType()->getPrimitiveSizeInBits();
      int64_t  RHSSExt = cast<ConstantInt>(RHSC)->getSExtValue();
      uint64_t RHSZExt = cast<ConstantInt>(RHSC)->getZExtValue();
      
      uint64_t C0080Val = 1ULL << 31;
      int64_t CFF80Val = -C0080Val;
      unsigned Size = 32;
      do {
        if (TySizeBits > Size) {
          bool Found = false;
          // If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
          // If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
          if (RHSSExt == CFF80Val) {
            if (XorRHS->getZExtValue() == C0080Val)
              Found = true;
          } else if (RHSZExt == C0080Val) {
            if (XorRHS->getSExtValue() == CFF80Val)
              Found = true;
          }
          if (Found) {
            // This is a sign extend if the top bits are known zero.
            Constant *Mask = ConstantInt::getAllOnesValue(XorLHS->getType());
            Mask = ConstantExpr::getShl(Mask, 
                           ConstantInt::get(Type::UByteTy, 64-TySizeBits-Size));
            if (!MaskedValueIsZero(XorLHS, cast<ConstantInt>(Mask)))
              Size = 0;  // Not a sign ext, but can't be any others either.
            goto FoundSExt;
          }
        }
        Size >>= 1;
        C0080Val >>= Size;
        CFF80Val >>= Size;
      } while (Size >= 8);
      
FoundSExt:
      const Type *MiddleType = 0;
      switch (Size) {
      default: break;
      case 32: MiddleType = Type::IntTy; break;
      case 16: MiddleType = Type::ShortTy; break;
      case 8:  MiddleType = Type::SByteTy; break;
      }
      if (MiddleType) {
        Instruction *NewTrunc = new CastInst(XorLHS, MiddleType, "sext");
        InsertNewInstBefore(NewTrunc, I);
        return new CastInst(NewTrunc, I.getType());
      }
    }
  }

  // X + X --> X << 1
  if (I.getType()->isInteger()) {
    if (Instruction *Result = AssociativeOpt(I, AddRHS(RHS))) return Result;

    if (Instruction *RHSI = dyn_cast<Instruction>(RHS)) {
      if (RHSI->getOpcode() == Instruction::Sub)
        if (LHS == RHSI->getOperand(1))                   // A + (B - A) --> B
          return ReplaceInstUsesWith(I, RHSI->getOperand(0));
    }
    if (Instruction *LHSI = dyn_cast<Instruction>(LHS)) {
      if (LHSI->getOpcode() == Instruction::Sub)
        if (RHS == LHSI->getOperand(1))                   // (B - A) + A --> B
          return ReplaceInstUsesWith(I, LHSI->getOperand(0));
    }
  }

  // -A + B  -->  B - A
  if (Value *V = dyn_castNegVal(LHS))
    return BinaryOperator::createSub(RHS, V);

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::createSub(LHS, V);


  ConstantInt *C2;
  if (Value *X = dyn_castFoldableMul(LHS, C2)) {
    if (X == RHS)   // X*C + X --> X * (C+1)
      return BinaryOperator::createMul(RHS, AddOne(C2));

    // X*C1 + X*C2 --> X * (C1+C2)
    ConstantInt *C1;
    if (X == dyn_castFoldableMul(RHS, C1))
      return BinaryOperator::createMul(X, ConstantExpr::getAdd(C1, C2));
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS, C2) == LHS)
    return BinaryOperator::createMul(LHS, AddOne(C2));


  // (A & C1)+(B & C2) --> (A & C1)|(B & C2) iff C1&C2 == 0
  if (match(RHS, m_And(m_Value(), m_ConstantInt(C2))))
    if (Instruction *R = AssociativeOpt(I, AddMaskingAnd(C2))) return R;

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    Value *X = 0;
    if (match(LHS, m_Not(m_Value(X)))) {   // ~X + C --> (C-1) - X
      Constant *C= ConstantExpr::getSub(CRHS, ConstantInt::get(I.getType(), 1));
      return BinaryOperator::createSub(C, X);
    }

    // (X & FF00) + xx00  -> (X+xx00) & FF00
    if (LHS->hasOneUse() && match(LHS, m_And(m_Value(X), m_ConstantInt(C2)))) {
      Constant *Anded = ConstantExpr::getAnd(CRHS, C2);
      if (Anded == CRHS) {
        // See if all bits from the first bit set in the Add RHS up are included
        // in the mask.  First, get the rightmost bit.
        uint64_t AddRHSV = CRHS->getRawValue();

        // Form a mask of all bits from the lowest bit added through the top.
        uint64_t AddRHSHighBits = ~((AddRHSV & -AddRHSV)-1);
        AddRHSHighBits &= ~0ULL >> (64-C2->getType()->getPrimitiveSizeInBits());

        // See if the and mask includes all of these bits.
        uint64_t AddRHSHighBitsAnd = AddRHSHighBits & C2->getRawValue();

        if (AddRHSHighBits == AddRHSHighBitsAnd) {
          // Okay, the xform is safe.  Insert the new add pronto.
          Value *NewAdd = InsertNewInstBefore(BinaryOperator::createAdd(X, CRHS,
                                                            LHS->getName()), I);
          return BinaryOperator::createAnd(NewAdd, C2);
        }
      }
    }

    // Try to fold constant add into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(LHS))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
  }

  return Changed ? &I : 0;
}

// isSignBit - Return true if the value represented by the constant only has the
// highest order bit set.
static bool isSignBit(ConstantInt *CI) {
  unsigned NumBits = CI->getType()->getPrimitiveSizeInBits();
  return (CI->getRawValue() & (~0ULL >> (64-NumBits))) == (1ULL << (NumBits-1));
}

/// RemoveNoopCast - Strip off nonconverting casts from the value.
///
static Value *RemoveNoopCast(Value *V) {
  if (CastInst *CI = dyn_cast<CastInst>(V)) {
    const Type *CTy = CI->getType();
    const Type *OpTy = CI->getOperand(0)->getType();
    if (CTy->isInteger() && OpTy->isInteger()) {
      if (CTy->getPrimitiveSizeInBits() == OpTy->getPrimitiveSizeInBits())
        return RemoveNoopCast(CI->getOperand(0));
    } else if (isa<PointerType>(CTy) && isa<PointerType>(OpTy))
      return RemoveNoopCast(CI->getOperand(0));
  }
  return V;
}

Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Op0 == Op1)         // sub X, X  -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castNegVal(Op1))
    return BinaryOperator::createAdd(Op0, V);

  if (isa<UndefValue>(Op0))
    return ReplaceInstUsesWith(I, Op0);    // undef - X -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);    // X - undef -> undef

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // Replace (-1 - A) with (~A)...
    if (C->isAllOnesValue())
      return BinaryOperator::createNot(Op1);

    // C - ~X == X + (1+C)
    Value *X = 0;
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::createAdd(X,
                    ConstantExpr::getAdd(C, ConstantInt::get(I.getType(), 1)));
    // -((uint)X >> 31) -> ((int)X >> 31)
    // -((int)X >> 31) -> ((uint)X >> 31)
    if (C->isNullValue()) {
      Value *NoopCastedRHS = RemoveNoopCast(Op1);
      if (ShiftInst *SI = dyn_cast<ShiftInst>(NoopCastedRHS))
        if (SI->getOpcode() == Instruction::Shr)
          if (ConstantUInt *CU = dyn_cast<ConstantUInt>(SI->getOperand(1))) {
            const Type *NewTy;
            if (SI->getType()->isSigned())
              NewTy = SI->getType()->getUnsignedVersion();
            else
              NewTy = SI->getType()->getSignedVersion();
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getValue() == SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert a cast of the incoming
              // value, then the new shift, then the new cast.
              Instruction *FirstCast = new CastInst(SI->getOperand(0), NewTy,
                                                 SI->getOperand(0)->getName());
              Value *InV = InsertNewInstBefore(FirstCast, I);
              Instruction *NewShift = new ShiftInst(Instruction::Shr, FirstCast,
                                                    CU, SI->getName());
              if (NewShift->getType() == I.getType())
                return NewShift;
              else {
                InV = InsertNewInstBefore(NewShift, I);
                return new CastInst(NewShift, I.getType());
              }
            }
          }
    }

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1)) {
    if (Op1I->getOpcode() == Instruction::Add &&
        !Op0->getType()->isFloatingPoint()) {
      if (Op1I->getOperand(0) == Op0)              // X-(X+Y) == -Y
        return BinaryOperator::createNeg(Op1I->getOperand(1), I.getName());
      else if (Op1I->getOperand(1) == Op0)         // X-(Y+X) == -Y
        return BinaryOperator::createNeg(Op1I->getOperand(0), I.getName());
      else if (ConstantInt *CI1 = dyn_cast<ConstantInt>(I.getOperand(0))) {
        if (ConstantInt *CI2 = dyn_cast<ConstantInt>(Op1I->getOperand(1)))
          // C1-(X+C2) --> (C1-C2)-X
          return BinaryOperator::createSub(ConstantExpr::getSub(CI1, CI2),
                                           Op1I->getOperand(0));
      }
    }

    if (Op1I->hasOneUse()) {
      // Replace (x - (y - z)) with (x + (z - y)) if the (y - z) subexpression
      // is not used by anyone else...
      //
      if (Op1I->getOpcode() == Instruction::Sub &&
          !Op1I->getType()->isFloatingPoint()) {
        // Swap the two operands of the subexpr...
        Value *IIOp0 = Op1I->getOperand(0), *IIOp1 = Op1I->getOperand(1);
        Op1I->setOperand(0, IIOp1);
        Op1I->setOperand(1, IIOp0);

        // Create the new top level add instruction...
        return BinaryOperator::createAdd(Op0, Op1);
      }

      // Replace (A - (A & B)) with (A & ~B) if this is the only use of (A&B)...
      //
      if (Op1I->getOpcode() == Instruction::And &&
          (Op1I->getOperand(0) == Op0 || Op1I->getOperand(1) == Op0)) {
        Value *OtherOp = Op1I->getOperand(Op1I->getOperand(0) == Op0);

        Value *NewNot =
          InsertNewInstBefore(BinaryOperator::createNot(OtherOp, "B.not"), I);
        return BinaryOperator::createAnd(Op0, NewNot);
      }

      // -(X sdiv C)  -> (X sdiv -C)
      if (Op1I->getOpcode() == Instruction::Div)
        if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(Op0))
          if (CSI->isNullValue())
            if (Constant *DivRHS = dyn_cast<Constant>(Op1I->getOperand(1)))
              return BinaryOperator::createDiv(Op1I->getOperand(0),
                                               ConstantExpr::getNeg(DivRHS));

      // X - X*C --> X * (1-C)
      ConstantInt *C2 = 0;
      if (dyn_castFoldableMul(Op1I, C2) == Op0) {
        Constant *CP1 =
          ConstantExpr::getSub(ConstantInt::get(I.getType(), 1), C2);
        return BinaryOperator::createMul(Op0, CP1);
      }
    }
  }

  if (!Op0->getType()->isFloatingPoint())
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0))
      if (Op0I->getOpcode() == Instruction::Add) {
        if (Op0I->getOperand(0) == Op1)             // (Y+X)-Y == X
          return ReplaceInstUsesWith(I, Op0I->getOperand(1));
        else if (Op0I->getOperand(1) == Op1)        // (X+Y)-Y == X
          return ReplaceInstUsesWith(I, Op0I->getOperand(0));
      } else if (Op0I->getOpcode() == Instruction::Sub) {
        if (Op0I->getOperand(0) == Op1)             // (X-Y)-X == -Y
          return BinaryOperator::createNeg(Op0I->getOperand(1), I.getName());
      }

  ConstantInt *C1;
  if (Value *X = dyn_castFoldableMul(Op0, C1)) {
    if (X == Op1) { // X*C - X --> X * (C-1)
      Constant *CP1 = ConstantExpr::getSub(C1, ConstantInt::get(I.getType(),1));
      return BinaryOperator::createMul(Op1, CP1);
    }

    ConstantInt *C2;   // X*C1 - X*C2 -> X * (C1-C2)
    if (X == dyn_castFoldableMul(Op1, C2))
      return BinaryOperator::createMul(Op1, ConstantExpr::getSub(C1, C2));
  }
  return 0;
}

/// isSignBitCheck - Given an exploded setcc instruction, return true if it is
/// really just returns true if the most significant (sign) bit is set.
static bool isSignBitCheck(unsigned Opcode, Value *LHS, ConstantInt *RHS) {
  if (RHS->getType()->isSigned()) {
    // True if source is LHS < 0 or LHS <= -1
    return Opcode == Instruction::SetLT && RHS->isNullValue() ||
           Opcode == Instruction::SetLE && RHS->isAllOnesValue();
  } else {
    ConstantUInt *RHSC = cast<ConstantUInt>(RHS);
    // True if source is LHS > 127 or LHS >= 128, where the constants depend on
    // the size of the integer type.
    if (Opcode == Instruction::SetGE)
      return RHSC->getValue() ==
        1ULL << (RHS->getType()->getPrimitiveSizeInBits()-1);
    if (Opcode == Instruction::SetGT)
      return RHSC->getValue() ==
        (1ULL << (RHS->getType()->getPrimitiveSizeInBits()-1))-1;
  }
  return false;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0);

  if (isa<UndefValue>(I.getOperand(1)))              // undef * X -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // Simplify mul instructions with a constant RHS...
  if (Constant *Op1 = dyn_cast<Constant>(I.getOperand(1))) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {

      // ((X << C1)*C2) == (X * (C2 << C1))
      if (ShiftInst *SI = dyn_cast<ShiftInst>(Op0))
        if (SI->getOpcode() == Instruction::Shl)
          if (Constant *ShOp = dyn_cast<Constant>(SI->getOperand(1)))
            return BinaryOperator::createMul(SI->getOperand(0),
                                             ConstantExpr::getShl(CI, ShOp));

      if (CI->isNullValue())
        return ReplaceInstUsesWith(I, Op1);  // X * 0  == 0
      if (CI->equalsInt(1))                  // X * 1  == X
        return ReplaceInstUsesWith(I, Op0);
      if (CI->isAllOnesValue())              // X * -1 == 0 - X
        return BinaryOperator::createNeg(Op0, I.getName());

      int64_t Val = (int64_t)cast<ConstantInt>(CI)->getRawValue();
      if (isPowerOf2_64(Val)) {          // Replace X*(2^C) with X << C
        uint64_t C = Log2_64(Val);
        return new ShiftInst(Instruction::Shl, Op0,
                             ConstantUInt::get(Type::UByteTy, C));
      }
    } else if (ConstantFP *Op1F = dyn_cast<ConstantFP>(Op1)) {
      if (Op1F->isNullValue())
        return ReplaceInstUsesWith(I, Op1);

      // "In IEEE floating point, x*1 is not equivalent to x for nans.  However,
      // ANSI says we can drop signals, so we can do this anyway." (from GCC)
      if (Op1F->getValue() == 1.0)
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul double %X, 1.0'
    }

    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *Op0v = dyn_castNegVal(Op0))     // -X * -Y = X*Y
    if (Value *Op1v = dyn_castNegVal(I.getOperand(1)))
      return BinaryOperator::createMul(Op0v, Op1v);

  // If one of the operands of the multiply is a cast from a boolean value, then
  // we know the bool is either zero or one, so this is a 'masking' multiply.
  // See if we can simplify things based on how the boolean was originally
  // formed.
  CastInst *BoolCast = 0;
  if (CastInst *CI = dyn_cast<CastInst>(I.getOperand(0)))
    if (CI->getOperand(0)->getType() == Type::BoolTy)
      BoolCast = CI;
  if (!BoolCast)
    if (CastInst *CI = dyn_cast<CastInst>(I.getOperand(1)))
      if (CI->getOperand(0)->getType() == Type::BoolTy)
        BoolCast = CI;
  if (BoolCast) {
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(BoolCast->getOperand(0))) {
      Value *SCIOp0 = SCI->getOperand(0), *SCIOp1 = SCI->getOperand(1);
      const Type *SCOpTy = SCIOp0->getType();

      // If the setcc is true iff the sign bit of X is set, then convert this
      // multiply into a shift/and combination.
      if (isa<ConstantInt>(SCIOp1) &&
          isSignBitCheck(SCI->getOpcode(), SCIOp0, cast<ConstantInt>(SCIOp1))) {
        // Shift the X value right to turn it into "all signbits".
        Constant *Amt = ConstantUInt::get(Type::UByteTy,
                                          SCOpTy->getPrimitiveSizeInBits()-1);
        if (SCIOp0->getType()->isUnsigned()) {
          const Type *NewTy = SCIOp0->getType()->getSignedVersion();
          SCIOp0 = InsertNewInstBefore(new CastInst(SCIOp0, NewTy,
                                                    SCIOp0->getName()), I);
        }

        Value *V =
          InsertNewInstBefore(new ShiftInst(Instruction::Shr, SCIOp0, Amt,
                                            BoolCast->getOperand(0)->getName()+
                                            ".mask"), I);

        // If the multiply type is not the same as the source type, sign extend
        // or truncate to the multiply type.
        if (I.getType() != V->getType())
          V = InsertNewInstBefore(new CastInst(V, I.getType(), V->getName()),I);

        Value *OtherOp = Op0 == BoolCast ? I.getOperand(1) : Op0;
        return BinaryOperator::createAnd(V, OtherOp);
      }
    }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op0))              // undef / X -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X / undef -> undef

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // div X, 1 == X
    if (RHS->equalsInt(1))
      return ReplaceInstUsesWith(I, Op0);

    // div X, -1 == -X
    if (RHS->isAllOnesValue())
      return BinaryOperator::createNeg(Op0);

    if (Instruction *LHS = dyn_cast<Instruction>(Op0))
      if (LHS->getOpcode() == Instruction::Div)
        if (ConstantInt *LHSRHS = dyn_cast<ConstantInt>(LHS->getOperand(1))) {
          // (X / C1) / C2  -> X / (C1*C2)
          return BinaryOperator::createDiv(LHS->getOperand(0),
                                           ConstantExpr::getMul(RHS, LHSRHS));
        }

    // Check to see if this is an unsigned division with an exact power of 2,
    // if so, convert to a right shift.
    if (ConstantUInt *C = dyn_cast<ConstantUInt>(RHS))
      if (uint64_t Val = C->getValue())    // Don't break X / 0
        if (isPowerOf2_64(Val)) {
          uint64_t C = Log2_64(Val);
          return new ShiftInst(Instruction::Shr, Op0,
                               ConstantUInt::get(Type::UByteTy, C));
        }

    // -X/C -> X/-C
    if (RHS->getType()->isSigned())
      if (Value *LHSNeg = dyn_castNegVal(Op0))
        return BinaryOperator::createDiv(LHSNeg, ConstantExpr::getNeg(RHS));

    if (!RHS->isNullValue()) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
        if (Instruction *R = FoldOpIntoSelect(I, SI, this))
          return R;
      if (isa<PHINode>(Op0))
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
    }
  }

  // If this is 'udiv X, (Cond ? C1, C2)' where C1&C2 are powers of two,
  // transform this into: '(Cond ? (udiv X, C1) : (udiv X, C2))'.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
    if (ConstantUInt *STO = dyn_cast<ConstantUInt>(SI->getOperand(1)))
      if (ConstantUInt *SFO = dyn_cast<ConstantUInt>(SI->getOperand(2))) {
        if (STO->getValue() == 0) { // Couldn't be this argument.
          I.setOperand(1, SFO);
          return &I;
        } else if (SFO->getValue() == 0) {
          I.setOperand(1, STO);
          return &I;
        }

        uint64_t TVA = STO->getValue(), FVA = SFO->getValue();
        if (isPowerOf2_64(TVA) && isPowerOf2_64(FVA)) {
          unsigned TSA = Log2_64(TVA), FSA = Log2_64(FVA);
          Constant *TC = ConstantUInt::get(Type::UByteTy, TSA);
          Instruction *TSI = new ShiftInst(Instruction::Shr, Op0,
                                           TC, SI->getName()+".t");
          TSI = InsertNewInstBefore(TSI, I);

          Constant *FC = ConstantUInt::get(Type::UByteTy, FSA);
          Instruction *FSI = new ShiftInst(Instruction::Shr, Op0,
                                           FC, SI->getName()+".f");
          FSI = InsertNewInstBefore(FSI, I);
          return new SelectInst(SI->getOperand(0), TSI, FSI);
        }
      }

  // 0 / X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(Op0))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  if (I.getType()->isSigned()) {
    // If the top bits of both operands are zero (i.e. we can prove they are
    // unsigned inputs), turn this into a udiv.
    ConstantIntegral *MaskV = ConstantSInt::getMinValue(I.getType());
    if (MaskedValueIsZero(Op1, MaskV) && MaskedValueIsZero(Op0, MaskV)) {
      const Type *NTy = Op0->getType()->getUnsignedVersion();
      Instruction *LHS = new CastInst(Op0, NTy, Op0->getName());
      InsertNewInstBefore(LHS, I);
      Value *RHS;
      if (Constant *R = dyn_cast<Constant>(Op1))
        RHS = ConstantExpr::getCast(R, NTy);
      else
        RHS = InsertNewInstBefore(new CastInst(Op1, NTy, Op1->getName()), I);
      Instruction *Div = BinaryOperator::createDiv(LHS, RHS, I.getName());
      InsertNewInstBefore(Div, I);
      return new CastInst(Div, I.getType());
    }      
  }
  
  return 0;
}


Instruction *InstCombiner::visitRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  if (I.getType()->isSigned()) {
    if (Value *RHSNeg = dyn_castNegVal(Op1))
      if (!isa<ConstantSInt>(RHSNeg) ||
          cast<ConstantSInt>(RHSNeg)->getValue() > 0) {
        // X % -Y -> X % Y
        AddUsesToWorkList(I);
        I.setOperand(1, RHSNeg);
        return &I;
      }
   
    // If the top bits of both operands are zero (i.e. we can prove they are
    // unsigned inputs), turn this into a urem.
    ConstantIntegral *MaskV = ConstantSInt::getMinValue(I.getType());
    if (MaskedValueIsZero(Op1, MaskV) && MaskedValueIsZero(Op0, MaskV)) {
      const Type *NTy = Op0->getType()->getUnsignedVersion();
      Instruction *LHS = new CastInst(Op0, NTy, Op0->getName());
      InsertNewInstBefore(LHS, I);
      Value *RHS;
      if (Constant *R = dyn_cast<Constant>(Op1))
        RHS = ConstantExpr::getCast(R, NTy);
      else
        RHS = InsertNewInstBefore(new CastInst(Op1, NTy, Op1->getName()), I);
      Instruction *Rem = BinaryOperator::createRem(LHS, RHS, I.getName());
      InsertNewInstBefore(Rem, I);
      return new CastInst(Rem, I.getType());
    }
  }

  if (isa<UndefValue>(Op0))              // undef % X -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X % undef -> undef

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    if (RHS->equalsInt(1))  // X % 1 == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

    // Check to see if this is an unsigned remainder with an exact power of 2,
    // if so, convert to a bitwise and.
    if (ConstantUInt *C = dyn_cast<ConstantUInt>(RHS))
      if (uint64_t Val = C->getValue())    // Don't break X % 0 (divide by zero)
        if (!(Val & (Val-1)))              // Power of 2
          return BinaryOperator::createAnd(Op0,
                                         ConstantUInt::get(I.getType(), Val-1));

    if (!RHS->isNullValue()) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
        if (Instruction *R = FoldOpIntoSelect(I, SI, this))
          return R;
      if (isa<PHINode>(Op0))
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
    }
  }

  // If this is 'urem X, (Cond ? C1, C2)' where C1&C2 are powers of two,
  // transform this into: '(Cond ? (urem X, C1) : (urem X, C2))'.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
    if (ConstantUInt *STO = dyn_cast<ConstantUInt>(SI->getOperand(1)))
      if (ConstantUInt *SFO = dyn_cast<ConstantUInt>(SI->getOperand(2))) {
        if (STO->getValue() == 0) { // Couldn't be this argument.
          I.setOperand(1, SFO);
          return &I;
        } else if (SFO->getValue() == 0) {
          I.setOperand(1, STO);
          return &I;
        }

        if (!(STO->getValue() & (STO->getValue()-1)) &&
            !(SFO->getValue() & (SFO->getValue()-1))) {
          Value *TrueAnd = InsertNewInstBefore(BinaryOperator::createAnd(Op0,
                                         SubOne(STO), SI->getName()+".t"), I);
          Value *FalseAnd = InsertNewInstBefore(BinaryOperator::createAnd(Op0,
                                         SubOne(SFO), SI->getName()+".f"), I);
          return new SelectInst(SI->getOperand(0), TrueAnd, FalseAnd);
        }
      }

  // 0 % X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(Op0))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  return 0;
}

// isMaxValueMinusOne - return true if this is Max-1
static bool isMaxValueMinusOne(const ConstantInt *C) {
  if (const ConstantUInt *CU = dyn_cast<ConstantUInt>(C)) {
    // Calculate -1 casted to the right type...
    unsigned TypeBits = C->getType()->getPrimitiveSizeInBits();
    uint64_t Val = ~0ULL;                // All ones
    Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
    return CU->getValue() == Val-1;
  }

  const ConstantSInt *CS = cast<ConstantSInt>(C);

  // Calculate 0111111111..11111
  unsigned TypeBits = C->getType()->getPrimitiveSizeInBits();
  int64_t Val = INT64_MAX;             // All ones
  Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
  return CS->getValue() == Val-1;
}

// isMinValuePlusOne - return true if this is Min+1
static bool isMinValuePlusOne(const ConstantInt *C) {
  if (const ConstantUInt *CU = dyn_cast<ConstantUInt>(C))
    return CU->getValue() == 1;

  const ConstantSInt *CS = cast<ConstantSInt>(C);

  // Calculate 1111111111000000000000
  unsigned TypeBits = C->getType()->getPrimitiveSizeInBits();
  int64_t Val = -1;                    // All ones
  Val <<= TypeBits-1;                  // Shift over to the right spot
  return CS->getValue() == Val+1;
}

// isOneBitSet - Return true if there is exactly one bit set in the specified
// constant.
static bool isOneBitSet(const ConstantInt *CI) {
  uint64_t V = CI->getRawValue();
  return V && (V & (V-1)) == 0;
}

#if 0   // Currently unused
// isLowOnes - Return true if the constant is of the form 0+1+.
static bool isLowOnes(const ConstantInt *CI) {
  uint64_t V = CI->getRawValue();

  // There won't be bits set in parts that the type doesn't contain.
  V &= ConstantInt::getAllOnesValue(CI->getType())->getRawValue();

  uint64_t U = V+1;  // If it is low ones, this should be a power of two.
  return U && V && (U & V) == 0;
}
#endif

// isHighOnes - Return true if the constant is of the form 1+0+.
// This is the same as lowones(~X).
static bool isHighOnes(const ConstantInt *CI) {
  uint64_t V = ~CI->getRawValue();
  if (~V == 0) return false;  // 0's does not match "1+"

  // There won't be bits set in parts that the type doesn't contain.
  V &= ConstantInt::getAllOnesValue(CI->getType())->getRawValue();

  uint64_t U = V+1;  // If it is low ones, this should be a power of two.
  return U && V && (U & V) == 0;
}


/// getSetCondCode - Encode a setcc opcode into a three bit mask.  These bits
/// are carefully arranged to allow folding of expressions such as:
///
///      (A < B) | (A > B) --> (A != B)
///
/// Bit value '4' represents that the comparison is true if A > B, bit value '2'
/// represents that the comparison is true if A == B, and bit value '1' is true
/// if A < B.
///
static unsigned getSetCondCode(const SetCondInst *SCI) {
  switch (SCI->getOpcode()) {
    // False -> 0
  case Instruction::SetGT: return 1;
  case Instruction::SetEQ: return 2;
  case Instruction::SetGE: return 3;
  case Instruction::SetLT: return 4;
  case Instruction::SetNE: return 5;
  case Instruction::SetLE: return 6;
    // True -> 7
  default:
    assert(0 && "Invalid SetCC opcode!");
    return 0;
  }
}

/// getSetCCValue - This is the complement of getSetCondCode, which turns an
/// opcode and two operands into either a constant true or false, or a brand new
/// SetCC instruction.
static Value *getSetCCValue(unsigned Opcode, Value *LHS, Value *RHS) {
  switch (Opcode) {
  case 0: return ConstantBool::False;
  case 1: return new SetCondInst(Instruction::SetGT, LHS, RHS);
  case 2: return new SetCondInst(Instruction::SetEQ, LHS, RHS);
  case 3: return new SetCondInst(Instruction::SetGE, LHS, RHS);
  case 4: return new SetCondInst(Instruction::SetLT, LHS, RHS);
  case 5: return new SetCondInst(Instruction::SetNE, LHS, RHS);
  case 6: return new SetCondInst(Instruction::SetLE, LHS, RHS);
  case 7: return ConstantBool::True;
  default: assert(0 && "Illegal SetCCCode!"); return 0;
  }
}

// FoldSetCCLogical - Implements (setcc1 A, B) & (setcc2 A, B) --> (setcc3 A, B)
struct FoldSetCCLogical {
  InstCombiner &IC;
  Value *LHS, *RHS;
  FoldSetCCLogical(InstCombiner &ic, SetCondInst *SCI)
    : IC(ic), LHS(SCI->getOperand(0)), RHS(SCI->getOperand(1)) {}
  bool shouldApply(Value *V) const {
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(V))
      return (SCI->getOperand(0) == LHS && SCI->getOperand(1) == RHS ||
              SCI->getOperand(0) == RHS && SCI->getOperand(1) == LHS);
    return false;
  }
  Instruction *apply(BinaryOperator &Log) const {
    SetCondInst *SCI = cast<SetCondInst>(Log.getOperand(0));
    if (SCI->getOperand(0) != LHS) {
      assert(SCI->getOperand(1) == LHS);
      SCI->swapOperands();  // Swap the LHS and RHS of the SetCC
    }

    unsigned LHSCode = getSetCondCode(SCI);
    unsigned RHSCode = getSetCondCode(cast<SetCondInst>(Log.getOperand(1)));
    unsigned Code;
    switch (Log.getOpcode()) {
    case Instruction::And: Code = LHSCode & RHSCode; break;
    case Instruction::Or:  Code = LHSCode | RHSCode; break;
    case Instruction::Xor: Code = LHSCode ^ RHSCode; break;
    default: assert(0 && "Illegal logical opcode!"); return 0;
    }

    Value *RV = getSetCCValue(Code, LHS, RHS);
    if (Instruction *I = dyn_cast<Instruction>(RV))
      return I;
    // Otherwise, it's a constant boolean value...
    return IC.ReplaceInstUsesWith(Log, RV);
  }
};

// OptAndOp - This handles expressions of the form ((val OP C1) & C2).  Where
// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.  Op is
// guaranteed to be either a shift instruction or a binary operator.
Instruction *InstCombiner::OptAndOp(Instruction *Op,
                                    ConstantIntegral *OpRHS,
                                    ConstantIntegral *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = 0;
  if (!isa<ShiftInst>(Op))
    Together = ConstantExpr::getAnd(AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  case Instruction::Xor:
    if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      std::string OpName = Op->getName(); Op->setName("");
      Instruction *And = BinaryOperator::createAnd(X, AndRHS, OpName);
      InsertNewInstBefore(And, TheAnd);
      return BinaryOperator::createXor(And, Together);
    }
    break;
  case Instruction::Or:
    if (Together == AndRHS) // (X | C) & C --> C
      return ReplaceInstUsesWith(TheAnd, AndRHS);

    if (Op->hasOneUse() && Together != OpRHS) {
      // (X | C1) & C2 --> (X | (C1&C2)) & C2
      std::string Op0Name = Op->getName(); Op->setName("");
      Instruction *Or = BinaryOperator::createOr(X, Together, Op0Name);
      InsertNewInstBefore(Or, TheAnd);
      return BinaryOperator::createAnd(Or, AndRHS);
    }
    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      uint64_t AndRHSV = cast<ConstantInt>(AndRHS)->getRawValue();

      // Clear bits that are not part of the constant.
      AndRHSV &= ~0ULL >> (64-AndRHS->getType()->getPrimitiveSizeInBits());

      // If there is only one bit set...
      if (isOneBitSet(cast<ConstantInt>(AndRHS))) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        uint64_t AddRHS = cast<ConstantInt>(OpRHS)->getRawValue();

        // Check to see if any bits below the one bit set in AndRHSV are set.
        if ((AddRHS & (AndRHSV-1)) == 0) {
          // If not, the only thing that can effect the output of the AND is
          // the bit specified by AndRHSV.  If that bit is set, the effect of
          // the XOR is to toggle the bit.  If it is clear, then the ADD has
          // no effect.
          if ((AddRHS & AndRHSV) == 0) { // Bit is not set, noop
            TheAnd.setOperand(0, X);
            return &TheAnd;
          } else {
            std::string Name = Op->getName(); Op->setName("");
            // Pull the XOR out of the AND.
            Instruction *NewAnd = BinaryOperator::createAnd(X, AndRHS, Name);
            InsertNewInstBefore(NewAnd, TheAnd);
            return BinaryOperator::createXor(NewAnd, AndRHS);
          }
        }
      }
    }
    break;

  case Instruction::Shl: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!
    //
    Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
    Constant *ShlMask = ConstantExpr::getShl(AllOne, OpRHS);
    Constant *CI = ConstantExpr::getAnd(AndRHS, ShlMask);

    if (CI == ShlMask) {   // Masking out bits that the shift already masks
      return ReplaceInstUsesWith(TheAnd, Op);   // No need for the and.
    } else if (CI != AndRHS) {                  // Reducing bits set in and.
      TheAnd.setOperand(1, CI);
      return &TheAnd;
    }
    break;
  }
  case Instruction::Shr:
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!  This only applies to
    // unsigned shifts, because a signed shr may bring in set bits!
    //
    if (AndRHS->getType()->isUnsigned()) {
      Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
      Constant *ShrMask = ConstantExpr::getShr(AllOne, OpRHS);
      Constant *CI = ConstantExpr::getAnd(AndRHS, ShrMask);

      if (CI == ShrMask) {   // Masking out bits that the shift already masks.
        return ReplaceInstUsesWith(TheAnd, Op);
      } else if (CI != AndRHS) {
        TheAnd.setOperand(1, CI);  // Reduce bits set in and cst.
        return &TheAnd;
      }
    } else {   // Signed shr.
      // See if this is shifting in some sign extension, then masking it out
      // with an and.
      if (Op->hasOneUse()) {
        Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
        Constant *ShrMask = ConstantExpr::getUShr(AllOne, OpRHS);
        Constant *CI = ConstantExpr::getAnd(AndRHS, ShrMask);
        if (CI == AndRHS) {          // Masking out bits shifted in.
          // Make the argument unsigned.
          Value *ShVal = Op->getOperand(0);
          ShVal = InsertCastBefore(ShVal,
                                   ShVal->getType()->getUnsignedVersion(),
                                   TheAnd);
          ShVal = InsertNewInstBefore(new ShiftInst(Instruction::Shr, ShVal,
                                                    OpRHS, Op->getName()),
                                      TheAnd);
          Value *AndRHS2 = ConstantExpr::getCast(AndRHS, ShVal->getType());
          ShVal = InsertNewInstBefore(BinaryOperator::createAnd(ShVal, AndRHS2,
                                                             TheAnd.getName()),
                                      TheAnd);
          return new CastInst(ShVal, Op->getType());
        }
      }
    }
    break;
  }
  return 0;
}


/// InsertRangeTest - Emit a computation of: (V >= Lo && V < Hi) if Inside is
/// true, otherwise (V < Lo || V >= Hi).  In pratice, we emit the more efficient
/// (V-Lo) <u Hi-Lo.  This method expects that Lo <= Hi.  IB is the location to
/// insert new instructions.
Instruction *InstCombiner::InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                           bool Inside, Instruction &IB) {
  assert(cast<ConstantBool>(ConstantExpr::getSetLE(Lo, Hi))->getValue() &&
         "Lo is not <= Hi in range emission code!");
  if (Inside) {
    if (Lo == Hi)  // Trivially false.
      return new SetCondInst(Instruction::SetNE, V, V);
    if (cast<ConstantIntegral>(Lo)->isMinValue())
      return new SetCondInst(Instruction::SetLT, V, Hi);

    Constant *AddCST = ConstantExpr::getNeg(Lo);
    Instruction *Add = BinaryOperator::createAdd(V, AddCST,V->getName()+".off");
    InsertNewInstBefore(Add, IB);
    // Convert to unsigned for the comparison.
    const Type *UnsType = Add->getType()->getUnsignedVersion();
    Value *OffsetVal = InsertCastBefore(Add, UnsType, IB);
    AddCST = ConstantExpr::getAdd(AddCST, Hi);
    AddCST = ConstantExpr::getCast(AddCST, UnsType);
    return new SetCondInst(Instruction::SetLT, OffsetVal, AddCST);
  }

  if (Lo == Hi)  // Trivially true.
    return new SetCondInst(Instruction::SetEQ, V, V);

  Hi = SubOne(cast<ConstantInt>(Hi));
  if (cast<ConstantIntegral>(Lo)->isMinValue()) // V < 0 || V >= Hi ->'V > Hi-1'
    return new SetCondInst(Instruction::SetGT, V, Hi);

  // Emit X-Lo > Hi-Lo-1
  Constant *AddCST = ConstantExpr::getNeg(Lo);
  Instruction *Add = BinaryOperator::createAdd(V, AddCST, V->getName()+".off");
  InsertNewInstBefore(Add, IB);
  // Convert to unsigned for the comparison.
  const Type *UnsType = Add->getType()->getUnsignedVersion();
  Value *OffsetVal = InsertCastBefore(Add, UnsType, IB);
  AddCST = ConstantExpr::getAdd(AddCST, Hi);
  AddCST = ConstantExpr::getCast(AddCST, UnsType);
  return new SetCondInst(Instruction::SetGT, OffsetVal, AddCST);
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(ConstantIntegral *Val, unsigned &MB, unsigned &ME) {
  uint64_t V = Val->getRawValue();
  if (!isShiftedMask_64(V)) return false;

  // look for the first zero bit after the run of ones
  MB = 64-CountLeadingZeros_64((V - 1) ^ V);
  // look for the first non-zero bit
  ME = 64-CountLeadingZeros_64(V);
  return true;
}



/// FoldLogicalPlusAnd - This is part of an expression (LHS +/- RHS) & Mask,
/// where isSub determines whether the operator is a sub.  If we can fold one of
/// the following xforms:
/// 
/// ((A & N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == Mask
/// ((A | N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
/// ((A ^ N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
///
/// return (A +/- B).
///
Value *InstCombiner::FoldLogicalPlusAnd(Value *LHS, Value *RHS,
                                        ConstantIntegral *Mask, bool isSub,
                                        Instruction &I) {
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  if (!LHSI || LHSI->getNumOperands() != 2 ||
      !isa<ConstantInt>(LHSI->getOperand(1))) return 0;

  ConstantInt *N = cast<ConstantInt>(LHSI->getOperand(1));

  switch (LHSI->getOpcode()) {
  default: return 0;
  case Instruction::And:
    if (ConstantExpr::getAnd(N, Mask) == Mask) {
      // If the AndRHS is a power of two minus one (0+1+), this is simple.
      if ((Mask->getRawValue() & Mask->getRawValue()+1) == 0)
        break;

      // Otherwise, if Mask is 0+1+0+, and if B is known to have the low 0+
      // part, we don't need any explicit masks to take them out of A.  If that
      // is all N is, ignore it.
      unsigned MB, ME;
      if (isRunOfOnes(Mask, MB, ME)) {  // begin/end bit of run, inclusive
        Constant *Mask = ConstantInt::getAllOnesValue(RHS->getType());
        Mask = ConstantExpr::getUShr(Mask,
                                     ConstantInt::get(Type::UByteTy,
                                                      (64-MB+1)));
        if (MaskedValueIsZero(RHS, cast<ConstantIntegral>(Mask)))
          break;
      }
    }
    return 0;
  case Instruction::Or:
  case Instruction::Xor:
    // If the AndRHS is a power of two minus one (0+1+), and N&Mask == 0
    if ((Mask->getRawValue() & Mask->getRawValue()+1) == 0 &&
        ConstantExpr::getAnd(N, Mask)->isNullValue())
      break;
    return 0;
  }
  
  Instruction *New;
  if (isSub)
    New = BinaryOperator::createSub(LHSI->getOperand(0), RHS, "fold");
  else
    New = BinaryOperator::createAdd(LHSI->getOperand(0), RHS, "fold");
  return InsertNewInstBefore(New, I);
}

Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))                         // X & undef -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // and X, X = X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, Op1);

  if (ConstantIntegral *AndRHS = dyn_cast<ConstantIntegral>(Op1)) {
    // and X, -1 == X
    if (AndRHS->isAllOnesValue())
      return ReplaceInstUsesWith(I, Op0);
    
    // and (and X, c1), c2 -> and (x, c1&c2).  Handle this case here, before
    // calling MaskedValueIsZero, to avoid inefficient cases where we traipse
    // through many levels of ands.
    {
      Value *X = 0; ConstantInt *C1 = 0;
      if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))))
        return BinaryOperator::createAnd(X, ConstantExpr::getAnd(C1, AndRHS));
    }

    if (MaskedValueIsZero(Op0, AndRHS))        // LHS & RHS == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

    // If the mask is not masking out any bits, there is no reason to do the
    // and in the first place.
    ConstantIntegral *NotAndRHS =
      cast<ConstantIntegral>(ConstantExpr::getNot(AndRHS));
    if (MaskedValueIsZero(Op0, NotAndRHS))
      return ReplaceInstUsesWith(I, Op0);

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (isa<BinaryOperator>(Op0) || isa<ShiftInst>(Op0)) {
      Instruction *Op0I = cast<Instruction>(Op0);
      Value *Op0LHS = Op0I->getOperand(0);
      Value *Op0RHS = Op0I->getOperand(1);
      switch (Op0I->getOpcode()) {
      case Instruction::Xor:
      case Instruction::Or:
        // (X ^ V) & C2 --> (X & C2) iff (V & C2) == 0
        // (X | V) & C2 --> (X & C2) iff (V & C2) == 0
        if (MaskedValueIsZero(Op0LHS, AndRHS))
          return BinaryOperator::createAnd(Op0RHS, AndRHS);
        if (MaskedValueIsZero(Op0RHS, AndRHS))
          return BinaryOperator::createAnd(Op0LHS, AndRHS);

        // If the mask is only needed on one incoming arm, push it up.
        if (Op0I->hasOneUse()) {
          if (MaskedValueIsZero(Op0LHS, NotAndRHS)) {
            // Not masking anything out for the LHS, move to RHS.
            Instruction *NewRHS = BinaryOperator::createAnd(Op0RHS, AndRHS,
                                                   Op0RHS->getName()+".masked");
            InsertNewInstBefore(NewRHS, I);
            return BinaryOperator::create(
                       cast<BinaryOperator>(Op0I)->getOpcode(), Op0LHS, NewRHS);
          }
          if (!isa<Constant>(NotAndRHS) &&
              MaskedValueIsZero(Op0RHS, NotAndRHS)) {
            // Not masking anything out for the RHS, move to LHS.
            Instruction *NewLHS = BinaryOperator::createAnd(Op0LHS, AndRHS,
                                                   Op0LHS->getName()+".masked");
            InsertNewInstBefore(NewLHS, I);
            return BinaryOperator::create(
                       cast<BinaryOperator>(Op0I)->getOpcode(), NewLHS, Op0RHS);
          }
        }

        break;
      case Instruction::And:
        // (X & V) & C2 --> 0 iff (V & C2) == 0
        if (MaskedValueIsZero(Op0LHS, AndRHS) ||
            MaskedValueIsZero(Op0RHS, AndRHS))
          return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
        break;
      case Instruction::Add:
        // ((A & N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, false, I))
          return BinaryOperator::createAnd(V, AndRHS);
        if (Value *V = FoldLogicalPlusAnd(Op0RHS, Op0LHS, AndRHS, false, I))
          return BinaryOperator::createAnd(V, AndRHS);  // Add commutes
        break;

      case Instruction::Sub:
        // ((A & N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, true, I))
          return BinaryOperator::createAnd(V, AndRHS);
        break;
      }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, AndRHS, I))
          return Res;
    } else if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
      const Type *SrcTy = CI->getOperand(0)->getType();

      // If this is an integer truncation or change from signed-to-unsigned, and
      // if the source is an and/or with immediate, transform it.  This
      // frequently occurs for bitfield accesses.
      if (Instruction *CastOp = dyn_cast<Instruction>(CI->getOperand(0))) {
        if (SrcTy->getPrimitiveSizeInBits() >= 
              I.getType()->getPrimitiveSizeInBits() &&
            CastOp->getNumOperands() == 2)
          if (ConstantInt *AndCI =dyn_cast<ConstantInt>(CastOp->getOperand(1)))
            if (CastOp->getOpcode() == Instruction::And) {
              // Change: and (cast (and X, C1) to T), C2
              // into  : and (cast X to T), trunc(C1)&C2
              // This will folds the two ands together, which may allow other
              // simplifications.
              Instruction *NewCast =
                new CastInst(CastOp->getOperand(0), I.getType(),
                             CastOp->getName()+".shrunk");
              NewCast = InsertNewInstBefore(NewCast, I);
              
              Constant *C3=ConstantExpr::getCast(AndCI, I.getType());//trunc(C1)
              C3 = ConstantExpr::getAnd(C3, AndRHS);            // trunc(C1)&C2
              return BinaryOperator::createAnd(NewCast, C3);
            } else if (CastOp->getOpcode() == Instruction::Or) {
              // Change: and (cast (or X, C1) to T), C2
              // into  : trunc(C1)&C2 iff trunc(C1)&C2 == C2
              Constant *C3=ConstantExpr::getCast(AndCI, I.getType());//trunc(C1)
              if (ConstantExpr::getAnd(C3, AndRHS) == AndRHS)   // trunc(C1)&C2
                return ReplaceInstUsesWith(I, AndRHS);
            }
      }


      // If this is an integer sign or zero extension instruction.
      if (SrcTy->isIntegral() &&
          SrcTy->getPrimitiveSizeInBits() <
          CI->getType()->getPrimitiveSizeInBits()) {

        if (SrcTy->isUnsigned()) {
          // See if this and is clearing out bits that are known to be zero
          // anyway (due to the zero extension).
          Constant *Mask = ConstantIntegral::getAllOnesValue(SrcTy);
          Mask = ConstantExpr::getZeroExtend(Mask, CI->getType());
          Constant *Result = ConstantExpr::getAnd(Mask, AndRHS);
          if (Result == Mask)  // The "and" isn't doing anything, remove it.
            return ReplaceInstUsesWith(I, CI);
          if (Result != AndRHS) { // Reduce the and RHS constant.
            I.setOperand(1, Result);
            return &I;
          }

        } else {
          if (CI->hasOneUse() && SrcTy->isInteger()) {
            // We can only do this if all of the sign bits brought in are masked
            // out.  Compute this by first getting 0000011111, then inverting
            // it.
            Constant *Mask = ConstantIntegral::getAllOnesValue(SrcTy);
            Mask = ConstantExpr::getZeroExtend(Mask, CI->getType());
            Mask = ConstantExpr::getNot(Mask);    // 1's in the new bits.
            if (ConstantExpr::getAnd(Mask, AndRHS)->isNullValue()) {
              // If the and is clearing all of the sign bits, change this to a
              // zero extension cast.  To do this, cast the cast input to
              // unsigned, then to the requested size.
              Value *CastOp = CI->getOperand(0);
              Instruction *NC =
                new CastInst(CastOp, CastOp->getType()->getUnsignedVersion(),
                             CI->getName()+".uns");
              NC = InsertNewInstBefore(NC, I);
              // Finally, insert a replacement for CI.
              NC = new CastInst(NC, CI->getType(), CI->getName());
              CI->setName("");
              NC = InsertNewInstBefore(NC, I);
              WorkList.push_back(CI);  // Delete CI later.
              I.setOperand(0, NC);
              return &I;               // The AND operand was modified.
            }
          }
        }
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  Value *Op0NotVal = dyn_castNotVal(Op0);
  Value *Op1NotVal = dyn_castNotVal(Op1);

  if (Op0NotVal == Op1 || Op1NotVal == Op0)  // A & ~A  == ~A & A == 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // (~A & ~B) == (~(A | B)) - De Morgan's Law
  if (Op0NotVal && Op1NotVal && isOnlyUse(Op0) && isOnlyUse(Op1)) {
    Instruction *Or = BinaryOperator::createOr(Op0NotVal, Op1NotVal,
                                               I.getName()+".demorgan");
    InsertNewInstBefore(Or, I);
    return BinaryOperator::createNot(Or);
  }

  if (SetCondInst *RHS = dyn_cast<SetCondInst>(Op1)) {
    // (setcc1 A, B) & (setcc2 A, B) --> (setcc3 A, B)
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

    Value *LHSVal, *RHSVal;
    ConstantInt *LHSCst, *RHSCst;
    Instruction::BinaryOps LHSCC, RHSCC;
    if (match(Op0, m_SetCond(LHSCC, m_Value(LHSVal), m_ConstantInt(LHSCst))))
      if (match(RHS, m_SetCond(RHSCC, m_Value(RHSVal), m_ConstantInt(RHSCst))))
        if (LHSVal == RHSVal &&    // Found (X setcc C1) & (X setcc C2)
            // Set[GL]E X, CST is folded to Set[GL]T elsewhere.
            LHSCC != Instruction::SetGE && LHSCC != Instruction::SetLE &&
            RHSCC != Instruction::SetGE && RHSCC != Instruction::SetLE) {
          // Ensure that the larger constant is on the RHS.
          Constant *Cmp = ConstantExpr::getSetGT(LHSCst, RHSCst);
          SetCondInst *LHS = cast<SetCondInst>(Op0);
          if (cast<ConstantBool>(Cmp)->getValue()) {
            std::swap(LHS, RHS);
            std::swap(LHSCst, RHSCst);
            std::swap(LHSCC, RHSCC);
          }

          // At this point, we know we have have two setcc instructions
          // comparing a value against two constants and and'ing the result
          // together.  Because of the above check, we know that we only have
          // SetEQ, SetNE, SetLT, and SetGT here.  We also know (from the
          // FoldSetCCLogical check above), that the two constants are not
          // equal.
          assert(LHSCst != RHSCst && "Compares not folded above?");

          switch (LHSCC) {
          default: assert(0 && "Unknown integer condition code!");
          case Instruction::SetEQ:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X == 13 & X == 15) -> false
            case Instruction::SetGT:  // (X == 13 & X > 15)  -> false
              return ReplaceInstUsesWith(I, ConstantBool::False);
            case Instruction::SetNE:  // (X == 13 & X != 15) -> X == 13
            case Instruction::SetLT:  // (X == 13 & X < 15)  -> X == 13
              return ReplaceInstUsesWith(I, LHS);
            }
          case Instruction::SetNE:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetLT:
              if (LHSCst == SubOne(RHSCst)) // (X != 13 & X < 14) -> X < 13
                return new SetCondInst(Instruction::SetLT, LHSVal, LHSCst);
              break;                        // (X != 13 & X < 15) -> no change
            case Instruction::SetEQ:        // (X != 13 & X == 15) -> X == 15
            case Instruction::SetGT:        // (X != 13 & X > 15)  -> X > 15
              return ReplaceInstUsesWith(I, RHS);
            case Instruction::SetNE:
              if (LHSCst == SubOne(RHSCst)) {// (X != 13 & X != 14) -> X-13 >u 1
                Constant *AddCST = ConstantExpr::getNeg(LHSCst);
                Instruction *Add = BinaryOperator::createAdd(LHSVal, AddCST,
                                                      LHSVal->getName()+".off");
                InsertNewInstBefore(Add, I);
                const Type *UnsType = Add->getType()->getUnsignedVersion();
                Value *OffsetVal = InsertCastBefore(Add, UnsType, I);
                AddCST = ConstantExpr::getSub(RHSCst, LHSCst);
                AddCST = ConstantExpr::getCast(AddCST, UnsType);
                return new SetCondInst(Instruction::SetGT, OffsetVal, AddCST);
              }
              break;                        // (X != 13 & X != 15) -> no change
            }
            break;
          case Instruction::SetLT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X < 13 & X == 15) -> false
            case Instruction::SetGT:  // (X < 13 & X > 15)  -> false
              return ReplaceInstUsesWith(I, ConstantBool::False);
            case Instruction::SetNE:  // (X < 13 & X != 15) -> X < 13
            case Instruction::SetLT:  // (X < 13 & X < 15) -> X < 13
              return ReplaceInstUsesWith(I, LHS);
            }
          case Instruction::SetGT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X > 13 & X == 15) -> X > 13
              return ReplaceInstUsesWith(I, LHS);
            case Instruction::SetGT:  // (X > 13 & X > 15)  -> X > 15
              return ReplaceInstUsesWith(I, RHS);
            case Instruction::SetNE:
              if (RHSCst == AddOne(LHSCst)) // (X > 13 & X != 14) -> X > 14
                return new SetCondInst(Instruction::SetGT, LHSVal, RHSCst);
              break;                        // (X > 13 & X != 15) -> no change
            case Instruction::SetLT:   // (X > 13 & X < 15) -> (X-14) <u 1
              return InsertRangeTest(LHSVal, AddOne(LHSCst), RHSCst, true, I);
            }
          }
        }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I,                         // X | undef -> -1
                               ConstantIntegral::getAllOnesValue(I.getType()));

  // or X, X = X   or X, 0 == X
  if (Op0 == Op1 || Op1 == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, Op0);

  // or X, -1 == -1
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    // If X is known to only contain bits that already exist in RHS, just
    // replace this instruction with RHS directly.
    if (MaskedValueIsZero(Op0,
                          cast<ConstantIntegral>(ConstantExpr::getNot(RHS))))
      return ReplaceInstUsesWith(I, RHS);

    ConstantInt *C1 = 0; Value *X = 0;
    // (X & C1) | C2 --> (X | C2) & (C1|C2)
    if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))) && isOnlyUse(Op0)) {
      Instruction *Or = BinaryOperator::createOr(X, RHS, Op0->getName());
      Op0->setName("");
      InsertNewInstBefore(Or, I);
      return BinaryOperator::createAnd(Or, ConstantExpr::getOr(RHS, C1));
    }

    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) && isOnlyUse(Op0)) {
      std::string Op0Name = Op0->getName(); Op0->setName("");
      Instruction *Or = BinaryOperator::createOr(X, RHS, Op0Name);
      InsertNewInstBefore(Or, I);
      return BinaryOperator::createXor(Or,
                 ConstantExpr::getAnd(C1, ConstantExpr::getNot(RHS)));
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  Value *A = 0, *B = 0;
  ConstantInt *C1 = 0, *C2 = 0;

  if (match(Op0, m_And(m_Value(A), m_Value(B))))
    if (A == Op1 || B == Op1)    // (A & ?) | A  --> A
      return ReplaceInstUsesWith(I, Op1);
  if (match(Op1, m_And(m_Value(A), m_Value(B))))
    if (A == Op0 || B == Op0)    // A | (A & ?)  --> A
      return ReplaceInstUsesWith(I, Op0);

  // (X^C)|Y -> (X|Y)^C iff Y&C == 0
  if (Op0->hasOneUse() && match(Op0, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op1, C1)) {
    Instruction *NOr = BinaryOperator::createOr(A, Op1, Op0->getName());
    Op0->setName("");
    return BinaryOperator::createXor(InsertNewInstBefore(NOr, I), C1);
  }

  // Y|(X^C) -> (X|Y)^C iff Y&C == 0
  if (Op1->hasOneUse() && match(Op1, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op0, C1)) {
    Instruction *NOr = BinaryOperator::createOr(A, Op0, Op1->getName());
    Op0->setName("");
    return BinaryOperator::createXor(InsertNewInstBefore(NOr, I), C1);
  }

  // (A & C1)|(B & C2)
  if (match(Op0, m_And(m_Value(A), m_ConstantInt(C1))) &&
      match(Op1, m_And(m_Value(B), m_ConstantInt(C2)))) {

    if (A == B)  // (A & C1)|(A & C2) == A & (C1|C2)
      return BinaryOperator::createAnd(A, ConstantExpr::getOr(C1, C2));


    // If we have: ((V + N) & C1) | (V & C2)
    // .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
    // replace with V+N.
    if (C1 == ConstantExpr::getNot(C2)) {
      Value *V1 = 0, *V2 = 0;
      if ((C2->getRawValue() & (C2->getRawValue()+1)) == 0 && // C2 == 0+1+
          match(A, m_Add(m_Value(V1), m_Value(V2)))) {
        // Add commutes, try both ways.
        if (V1 == B && MaskedValueIsZero(V2, C2))
          return ReplaceInstUsesWith(I, A);
        if (V2 == B && MaskedValueIsZero(V1, C2))
          return ReplaceInstUsesWith(I, A);
      }
      // Or commutes, try both ways.
      if ((C1->getRawValue() & (C1->getRawValue()+1)) == 0 &&
          match(B, m_Add(m_Value(V1), m_Value(V2)))) {
        // Add commutes, try both ways.
        if (V1 == A && MaskedValueIsZero(V2, C1))
          return ReplaceInstUsesWith(I, B);
        if (V2 == A && MaskedValueIsZero(V1, C1))
          return ReplaceInstUsesWith(I, B);
      }
    }
  }

  if (match(Op0, m_Not(m_Value(A)))) {   // ~A | Op1
    if (A == Op1)   // ~A | A == -1
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));
  } else {
    A = 0;
  }
  // Note, A is still live here!
  if (match(Op1, m_Not(m_Value(B)))) {   // Op0 | ~B
    if (Op0 == B)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

    // (~A | ~B) == (~(A & B)) - De Morgan's Law
    if (A && isOnlyUse(Op0) && isOnlyUse(Op1)) {
      Value *And = InsertNewInstBefore(BinaryOperator::createAnd(A, B,
                                              I.getName()+".demorgan"), I);
      return BinaryOperator::createNot(And);
    }
  }

  // (setcc1 A, B) | (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1))) {
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

    Value *LHSVal, *RHSVal;
    ConstantInt *LHSCst, *RHSCst;
    Instruction::BinaryOps LHSCC, RHSCC;
    if (match(Op0, m_SetCond(LHSCC, m_Value(LHSVal), m_ConstantInt(LHSCst))))
      if (match(RHS, m_SetCond(RHSCC, m_Value(RHSVal), m_ConstantInt(RHSCst))))
        if (LHSVal == RHSVal &&    // Found (X setcc C1) | (X setcc C2)
            // Set[GL]E X, CST is folded to Set[GL]T elsewhere.
            LHSCC != Instruction::SetGE && LHSCC != Instruction::SetLE &&
            RHSCC != Instruction::SetGE && RHSCC != Instruction::SetLE) {
          // Ensure that the larger constant is on the RHS.
          Constant *Cmp = ConstantExpr::getSetGT(LHSCst, RHSCst);
          SetCondInst *LHS = cast<SetCondInst>(Op0);
          if (cast<ConstantBool>(Cmp)->getValue()) {
            std::swap(LHS, RHS);
            std::swap(LHSCst, RHSCst);
            std::swap(LHSCC, RHSCC);
          }

          // At this point, we know we have have two setcc instructions
          // comparing a value against two constants and or'ing the result
          // together.  Because of the above check, we know that we only have
          // SetEQ, SetNE, SetLT, and SetGT here.  We also know (from the
          // FoldSetCCLogical check above), that the two constants are not
          // equal.
          assert(LHSCst != RHSCst && "Compares not folded above?");

          switch (LHSCC) {
          default: assert(0 && "Unknown integer condition code!");
          case Instruction::SetEQ:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:
              if (LHSCst == SubOne(RHSCst)) {// (X == 13 | X == 14) -> X-13 <u 2
                Constant *AddCST = ConstantExpr::getNeg(LHSCst);
                Instruction *Add = BinaryOperator::createAdd(LHSVal, AddCST,
                                                      LHSVal->getName()+".off");
                InsertNewInstBefore(Add, I);
                const Type *UnsType = Add->getType()->getUnsignedVersion();
                Value *OffsetVal = InsertCastBefore(Add, UnsType, I);
                AddCST = ConstantExpr::getSub(AddOne(RHSCst), LHSCst);
                AddCST = ConstantExpr::getCast(AddCST, UnsType);
                return new SetCondInst(Instruction::SetLT, OffsetVal, AddCST);
              }
              break;                  // (X == 13 | X == 15) -> no change

            case Instruction::SetGT:  // (X == 13 | X > 14) -> no change
              break;
            case Instruction::SetNE:  // (X == 13 | X != 15) -> X != 15
            case Instruction::SetLT:  // (X == 13 | X < 15)  -> X < 15
              return ReplaceInstUsesWith(I, RHS);
            }
            break;
          case Instruction::SetNE:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:        // (X != 13 | X == 15) -> X != 13
            case Instruction::SetGT:        // (X != 13 | X > 15)  -> X != 13
              return ReplaceInstUsesWith(I, LHS);
            case Instruction::SetNE:        // (X != 13 | X != 15) -> true
            case Instruction::SetLT:        // (X != 13 | X < 15)  -> true
              return ReplaceInstUsesWith(I, ConstantBool::True);
            }
            break;
          case Instruction::SetLT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X < 13 | X == 14) -> no change
              break;
            case Instruction::SetGT:  // (X < 13 | X > 15)  -> (X-13) > 2
              return InsertRangeTest(LHSVal, LHSCst, AddOne(RHSCst), false, I);
            case Instruction::SetNE:  // (X < 13 | X != 15) -> X != 15
            case Instruction::SetLT:  // (X < 13 | X < 15) -> X < 15
              return ReplaceInstUsesWith(I, RHS);
            }
            break;
          case Instruction::SetGT:
            switch (RHSCC) {
            default: assert(0 && "Unknown integer condition code!");
            case Instruction::SetEQ:  // (X > 13 | X == 15) -> X > 13
            case Instruction::SetGT:  // (X > 13 | X > 15)  -> X > 13
              return ReplaceInstUsesWith(I, LHS);
            case Instruction::SetNE:  // (X > 13 | X != 15)  -> true
            case Instruction::SetLT:  // (X > 13 | X < 15) -> true
              return ReplaceInstUsesWith(I, ConstantBool::True);
            }
          }
        }
  }

  return Changed ? &I : 0;
}

// XorSelf - Implements: X ^ X --> 0
struct XorSelf {
  Value *RHS;
  XorSelf(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Xor) const {
    return &Xor;
  }
};


Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X ^ undef -> undef

  // xor X, X = 0, even if X is nested in a sequence of Xor's.
  if (Instruction *Result = AssociativeOpt(I, XorSelf(Op1))) {
    assert(Result == &I && "AssociativeOpt didn't work?");
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    // xor X, 0 == X
    if (RHS->isNullValue())
      return ReplaceInstUsesWith(I, Op0);

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // xor (setcc A, B), true = not (setcc A, B) = setncc A, B
      if (SetCondInst *SCI = dyn_cast<SetCondInst>(Op0I))
        if (RHS == ConstantBool::True && SCI->hasOneUse())
          return new SetCondInst(SCI->getInverseCondition(),
                                 SCI->getOperand(0), SCI->getOperand(1));

      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHS->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::getNeg(Op0I0C);
          Constant *ConstantRHS = ConstantExpr::getSub(NegOp0I0C,
                                              ConstantInt::get(I.getType(), 1));
          return BinaryOperator::createAdd(Op0I->getOperand(1), ConstantRHS);
        }

      // ~(~X & Y) --> (X | ~Y)
      if (Op0I->getOpcode() == Instruction::And && RHS->isAllOnesValue()) {
        if (dyn_castNotVal(Op0I->getOperand(1))) Op0I->swapOperands();
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0))) {
          Instruction *NotY =
            BinaryOperator::createNot(Op0I->getOperand(1),
                                      Op0I->getOperand(1)->getName()+".not");
          InsertNewInstBefore(NotY, I);
          return BinaryOperator::createOr(Op0NotVal, NotY);
        }
      }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        switch (Op0I->getOpcode()) {
        case Instruction::Add:
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::createSub(
                           ConstantExpr::getSub(NegOp0CI,
                                             ConstantInt::get(I.getType(), 1)),
                                          Op0I->getOperand(0));
          }
          break;
        case Instruction::And:
          // (X & C1) ^ C2 --> (X & C1) | C2 iff (C1&C2) == 0
          if (ConstantExpr::getAnd(RHS, Op0CI)->isNullValue())
            return BinaryOperator::createOr(Op0, RHS);
          break;
        case Instruction::Or:
          // (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
          if (ConstantExpr::getAnd(RHS, Op0CI) == RHS)
            return BinaryOperator::createAnd(Op0, ConstantExpr::getNot(RHS));
          break;
        default: break;
        }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *X = dyn_castNotVal(Op0))   // ~A ^ A == -1
    if (X == Op1)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotVal(Op1))   // A ^ ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Instruction *Op1I = dyn_cast<Instruction>(Op1))
    if (Op1I->getOpcode() == Instruction::Or) {
      if (Op1I->getOperand(0) == Op0) {              // B^(B|A) == (A|B)^B
        cast<BinaryOperator>(Op1I)->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (Op1I->getOperand(1) == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();
        std::swap(Op0, Op1);
      }
    } else if (Op1I->getOpcode() == Instruction::Xor) {
      if (Op0 == Op1I->getOperand(0))                        // A^(A^B) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(1));
      else if (Op0 == Op1I->getOperand(1))                   // A^(B^A) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(0));
    }

  if (Instruction *Op0I = dyn_cast<Instruction>(Op0))
    if (Op0I->getOpcode() == Instruction::Or && Op0I->hasOneUse()) {
      if (Op0I->getOperand(0) == Op1)                // (B|A)^B == (A|B)^B
        cast<BinaryOperator>(Op0I)->swapOperands();
      if (Op0I->getOperand(1) == Op1) {              // (A|B)^B == A & ~B
        Value *NotB = InsertNewInstBefore(BinaryOperator::createNot(Op1,
                                                     Op1->getName()+".not"), I);
        return BinaryOperator::createAnd(Op0I->getOperand(0), NotB);
      }
    } else if (Op0I->getOpcode() == Instruction::Xor) {
      if (Op1 == Op0I->getOperand(0))                        // (A^B)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(1));
      else if (Op1 == Op0I->getOperand(1))                   // (B^A)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(0));
    }

  // (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
  ConstantInt *C1 = 0, *C2 = 0;
  if (match(Op0, m_And(m_Value(), m_ConstantInt(C1))) &&
      match(Op1, m_And(m_Value(), m_ConstantInt(C2))) &&
      ConstantExpr::getAnd(C1, C2)->isNullValue())
    return BinaryOperator::createOr(Op0, Op1);

  // (setcc1 A, B) ^ (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  return Changed ? &I : 0;
}

/// MulWithOverflow - Compute Result = In1*In2, returning true if the result
/// overflowed for this type.
static bool MulWithOverflow(ConstantInt *&Result, ConstantInt *In1,
                            ConstantInt *In2) {
  Result = cast<ConstantInt>(ConstantExpr::getMul(In1, In2));
  return !In2->isNullValue() && ConstantExpr::getDiv(Result, In2) != In1;
}

static bool isPositive(ConstantInt *C) {
  return cast<ConstantSInt>(C)->getValue() >= 0;
}

/// AddWithOverflow - Compute Result = In1+In2, returning true if the result
/// overflowed for this type.
static bool AddWithOverflow(ConstantInt *&Result, ConstantInt *In1,
                            ConstantInt *In2) {
  Result = cast<ConstantInt>(ConstantExpr::getAdd(In1, In2));

  if (In1->getType()->isUnsigned())
    return cast<ConstantUInt>(Result)->getValue() <
           cast<ConstantUInt>(In1)->getValue();
  if (isPositive(In1) != isPositive(In2))
    return false;
  if (isPositive(In1))
    return cast<ConstantSInt>(Result)->getValue() <
           cast<ConstantSInt>(In1)->getValue();
  return cast<ConstantSInt>(Result)->getValue() >
         cast<ConstantSInt>(In1)->getValue();
}

/// EmitGEPOffset - Given a getelementptr instruction/constantexpr, emit the
/// code necessary to compute the offset from the base pointer (without adding
/// in the base pointer).  Return the result as a signed integer of intptr size.
static Value *EmitGEPOffset(User *GEP, Instruction &I, InstCombiner &IC) {
  TargetData &TD = IC.getTargetData();
  gep_type_iterator GTI = gep_type_begin(GEP);
  const Type *UIntPtrTy = TD.getIntPtrType();
  const Type *SIntPtrTy = UIntPtrTy->getSignedVersion();
  Value *Result = Constant::getNullValue(SIntPtrTy);

  // Build a mask for high order bits.
  uint64_t PtrSizeMask = ~0ULL;
  PtrSizeMask >>= 64-(TD.getPointerSize()*8);

  for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i, ++GTI) {
    Value *Op = GEP->getOperand(i);
    uint64_t Size = TD.getTypeSize(GTI.getIndexedType()) & PtrSizeMask;
    Constant *Scale = ConstantExpr::getCast(ConstantUInt::get(UIntPtrTy, Size),
                                            SIntPtrTy);
    if (Constant *OpC = dyn_cast<Constant>(Op)) {
      if (!OpC->isNullValue()) {
        OpC = ConstantExpr::getCast(OpC, SIntPtrTy);
        Scale = ConstantExpr::getMul(OpC, Scale);
        if (Constant *RC = dyn_cast<Constant>(Result))
          Result = ConstantExpr::getAdd(RC, Scale);
        else {
          // Emit an add instruction.
          Result = IC.InsertNewInstBefore(
             BinaryOperator::createAdd(Result, Scale,
                                       GEP->getName()+".offs"), I);
        }
      }
    } else {
      // Convert to correct type.
      Op = IC.InsertNewInstBefore(new CastInst(Op, SIntPtrTy,
                                               Op->getName()+".c"), I);
      if (Size != 1)
        // We'll let instcombine(mul) convert this to a shl if possible.
        Op = IC.InsertNewInstBefore(BinaryOperator::createMul(Op, Scale,
                                                    GEP->getName()+".idx"), I);

      // Emit an add instruction.
      Result = IC.InsertNewInstBefore(BinaryOperator::createAdd(Op, Result,
                                                    GEP->getName()+".offs"), I);
    }
  }
  return Result;
}

/// FoldGEPSetCC - Fold comparisons between a GEP instruction and something
/// else.  At this point we know that the GEP is on the LHS of the comparison.
Instruction *InstCombiner::FoldGEPSetCC(User *GEPLHS, Value *RHS,
                                        Instruction::BinaryOps Cond,
                                        Instruction &I) {
  assert(dyn_castGetElementPtr(GEPLHS) && "LHS is not a getelementptr!");

  if (CastInst *CI = dyn_cast<CastInst>(RHS))
    if (isa<PointerType>(CI->getOperand(0)->getType()))
      RHS = CI->getOperand(0);

  Value *PtrBase = GEPLHS->getOperand(0);
  if (PtrBase == RHS) {
    // As an optimization, we don't actually have to compute the actual value of
    // OFFSET if this is a seteq or setne comparison, just return whether each
    // index is zero or not.
    if (Cond == Instruction::SetEQ || Cond == Instruction::SetNE) {
      Instruction *InVal = 0;
      gep_type_iterator GTI = gep_type_begin(GEPLHS);
      for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i, ++GTI) {
        bool EmitIt = true;
        if (Constant *C = dyn_cast<Constant>(GEPLHS->getOperand(i))) {
          if (isa<UndefValue>(C))  // undef index -> undef.
            return ReplaceInstUsesWith(I, UndefValue::get(I.getType()));
          if (C->isNullValue())
            EmitIt = false;
          else if (TD->getTypeSize(GTI.getIndexedType()) == 0) {
            EmitIt = false;  // This is indexing into a zero sized array?
          } else if (isa<ConstantInt>(C))
            return ReplaceInstUsesWith(I, // No comparison is needed here.
                                 ConstantBool::get(Cond == Instruction::SetNE));
        }

        if (EmitIt) {
          Instruction *Comp =
            new SetCondInst(Cond, GEPLHS->getOperand(i),
                    Constant::getNullValue(GEPLHS->getOperand(i)->getType()));
          if (InVal == 0)
            InVal = Comp;
          else {
            InVal = InsertNewInstBefore(InVal, I);
            InsertNewInstBefore(Comp, I);
            if (Cond == Instruction::SetNE)   // True if any are unequal
              InVal = BinaryOperator::createOr(InVal, Comp);
            else                              // True if all are equal
              InVal = BinaryOperator::createAnd(InVal, Comp);
          }
        }
      }

      if (InVal)
        return InVal;
      else
        ReplaceInstUsesWith(I, // No comparison is needed here, all indexes = 0
                            ConstantBool::get(Cond == Instruction::SetEQ));
    }

    // Only lower this if the setcc is the only user of the GEP or if we expect
    // the result to fold to a constant!
    if (isa<ConstantExpr>(GEPLHS) || GEPLHS->hasOneUse()) {
      // ((gep Ptr, OFFSET) cmp Ptr)   ---> (OFFSET cmp 0).
      Value *Offset = EmitGEPOffset(GEPLHS, I, *this);
      return new SetCondInst(Cond, Offset,
                             Constant::getNullValue(Offset->getType()));
    }
  } else if (User *GEPRHS = dyn_castGetElementPtr(RHS)) {
    // If the base pointers are different, but the indices are the same, just
    // compare the base pointer.
    if (PtrBase != GEPRHS->getOperand(0)) {
      bool IndicesTheSame = GEPLHS->getNumOperands()==GEPRHS->getNumOperands();
      IndicesTheSame &= GEPLHS->getOperand(0)->getType() ==
                        GEPRHS->getOperand(0)->getType();
      if (IndicesTheSame)
        for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i)
          if (GEPLHS->getOperand(i) != GEPRHS->getOperand(i)) {
            IndicesTheSame = false;
            break;
          }

      // If all indices are the same, just compare the base pointers.
      if (IndicesTheSame)
        return new SetCondInst(Cond, GEPLHS->getOperand(0),
                               GEPRHS->getOperand(0));

      // Otherwise, the base pointers are different and the indices are
      // different, bail out.
      return 0;
    }

    // If one of the GEPs has all zero indices, recurse.
    bool AllZeros = true;
    for (unsigned i = 1, e = GEPLHS->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPLHS->getOperand(i)) ||
          !cast<Constant>(GEPLHS->getOperand(i))->isNullValue()) {
        AllZeros = false;
        break;
      }
    if (AllZeros)
      return FoldGEPSetCC(GEPRHS, GEPLHS->getOperand(0),
                          SetCondInst::getSwappedCondition(Cond), I);

    // If the other GEP has all zero indices, recurse.
    AllZeros = true;
    for (unsigned i = 1, e = GEPRHS->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEPRHS->getOperand(i)) ||
          !cast<Constant>(GEPRHS->getOperand(i))->isNullValue()) {
        AllZeros = false;
        break;
      }
    if (AllZeros)
      return FoldGEPSetCC(GEPLHS, GEPRHS->getOperand(0), Cond, I);

    if (GEPLHS->getNumOperands() == GEPRHS->getNumOperands()) {
      // If the GEPs only differ by one index, compare it.
      unsigned NumDifferences = 0;  // Keep track of # differences.
      unsigned DiffOperand = 0;     // The operand that differs.
      for (unsigned i = 1, e = GEPRHS->getNumOperands(); i != e; ++i)
        if (GEPLHS->getOperand(i) != GEPRHS->getOperand(i)) {
          if (GEPLHS->getOperand(i)->getType()->getPrimitiveSizeInBits() !=
                   GEPRHS->getOperand(i)->getType()->getPrimitiveSizeInBits()) {
            // Irreconcilable differences.
            NumDifferences = 2;
            break;
          } else {
            if (NumDifferences++) break;
            DiffOperand = i;
          }
        }

      if (NumDifferences == 0)   // SAME GEP?
        return ReplaceInstUsesWith(I, // No comparison is needed here.
                                 ConstantBool::get(Cond == Instruction::SetEQ));
      else if (NumDifferences == 1) {
        Value *LHSV = GEPLHS->getOperand(DiffOperand);
        Value *RHSV = GEPRHS->getOperand(DiffOperand);

        // Convert the operands to signed values to make sure to perform a
        // signed comparison.
        const Type *NewTy = LHSV->getType()->getSignedVersion();
        if (LHSV->getType() != NewTy)
          LHSV = InsertNewInstBefore(new CastInst(LHSV, NewTy,
                                                  LHSV->getName()), I);
        if (RHSV->getType() != NewTy)
          RHSV = InsertNewInstBefore(new CastInst(RHSV, NewTy,
                                                  RHSV->getName()), I);
        return new SetCondInst(Cond, LHSV, RHSV);
      }
    }

    // Only lower this if the setcc is the only user of the GEP or if we expect
    // the result to fold to a constant!
    if ((isa<ConstantExpr>(GEPLHS) || GEPLHS->hasOneUse()) &&
        (isa<ConstantExpr>(GEPRHS) || GEPRHS->hasOneUse())) {
      // ((gep Ptr, OFFSET1) cmp (gep Ptr, OFFSET2)  --->  (OFFSET1 cmp OFFSET2)
      Value *L = EmitGEPOffset(GEPLHS, I, *this);
      Value *R = EmitGEPOffset(GEPRHS, I, *this);
      return new SetCondInst(Cond, L, R);
    }
  }
  return 0;
}


Instruction *InstCombiner::visitSetCondInst(SetCondInst &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  const Type *Ty = Op0->getType();

  // setcc X, X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, ConstantBool::get(isTrueWhenEqual(I)));

  if (isa<UndefValue>(Op1))                  // X setcc undef -> undef
    return ReplaceInstUsesWith(I, UndefValue::get(Type::BoolTy));

  // setcc <global/alloca*/null>, <global/alloca*/null> - Global/Stack value
  // addresses never equal each other!  We already know that Op0 != Op1.
  if ((isa<GlobalValue>(Op0) || isa<AllocaInst>(Op0) ||
       isa<ConstantPointerNull>(Op0)) &&
      (isa<GlobalValue>(Op1) || isa<AllocaInst>(Op1) ||
       isa<ConstantPointerNull>(Op1)))
    return ReplaceInstUsesWith(I, ConstantBool::get(!isTrueWhenEqual(I)));

  // setcc's with boolean values can always be turned into bitwise operations
  if (Ty == Type::BoolTy) {
    switch (I.getOpcode()) {
    default: assert(0 && "Invalid setcc instruction!");
    case Instruction::SetEQ: {     //  seteq bool %A, %B -> ~(A^B)
      Instruction *Xor = BinaryOperator::createXor(Op0, Op1, I.getName()+"tmp");
      InsertNewInstBefore(Xor, I);
      return BinaryOperator::createNot(Xor);
    }
    case Instruction::SetNE:
      return BinaryOperator::createXor(Op0, Op1);

    case Instruction::SetGT:
      std::swap(Op0, Op1);                   // Change setgt -> setlt
      // FALL THROUGH
    case Instruction::SetLT: {               // setlt bool A, B -> ~X & Y
      Instruction *Not = BinaryOperator::createNot(Op0, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::createAnd(Not, Op1);
    }
    case Instruction::SetGE:
      std::swap(Op0, Op1);                   // Change setge -> setle
      // FALL THROUGH
    case Instruction::SetLE: {     //  setle bool %A, %B -> ~A | B
      Instruction *Not = BinaryOperator::createNot(Op0, I.getName()+"tmp");
      InsertNewInstBefore(Not, I);
      return BinaryOperator::createOr(Not, Op1);
    }
    }
  }

  // See if we are doing a comparison between a constant and an instruction that
  // can be folded into the comparison.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // Check to see if we are comparing against the minimum or maximum value...
    if (CI->isMinValue()) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MIN -> A == MIN
        return BinaryOperator::createSetEQ(Op0, Op1);
      if (I.getOpcode() == Instruction::SetGT)       // A > MIN -> A != MIN
        return BinaryOperator::createSetNE(Op0, Op1);

    } else if (CI->isMaxValue()) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MAX -> A == MAX
        return BinaryOperator::createSetEQ(Op0, Op1);
      if (I.getOpcode() == Instruction::SetLT)       // A < MAX -> A != MAX
        return BinaryOperator::createSetNE(Op0, Op1);

      // Comparing against a value really close to min or max?
    } else if (isMinValuePlusOne(CI)) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN+1 -> A == MIN
        return BinaryOperator::createSetEQ(Op0, SubOne(CI));
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN-1 -> A != MIN
        return BinaryOperator::createSetNE(Op0, SubOne(CI));

    } else if (isMaxValueMinusOne(CI)) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX-1 -> A == MAX
        return BinaryOperator::createSetEQ(Op0, AddOne(CI));
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX-1 -> A != MAX
        return BinaryOperator::createSetNE(Op0, AddOne(CI));
    }

    // If we still have a setle or setge instruction, turn it into the
    // appropriate setlt or setgt instruction.  Since the border cases have
    // already been handled above, this requires little checking.
    //
    if (I.getOpcode() == Instruction::SetLE)
      return BinaryOperator::createSetLT(Op0, AddOne(CI));
    if (I.getOpcode() == Instruction::SetGE)
      return BinaryOperator::createSetGT(Op0, SubOne(CI));

    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::And:
        if (LHSI->hasOneUse() && isa<ConstantInt>(LHSI->getOperand(1)) &&
            LHSI->getOperand(0)->hasOneUse()) {
          // If this is: (X >> C1) & C2 != C3 (where any shift and any compare
          // could exist), turn it into (X & (C2 << C1)) != (C3 << C1).  This
          // happens a LOT in code produced by the C front-end, for bitfield
          // access.
          ShiftInst *Shift = dyn_cast<ShiftInst>(LHSI->getOperand(0));
          ConstantUInt *ShAmt;
          ShAmt = Shift ? dyn_cast<ConstantUInt>(Shift->getOperand(1)) : 0;
          ConstantInt *AndCST = cast<ConstantInt>(LHSI->getOperand(1));
          const Type *Ty = LHSI->getType();

          // We can fold this as long as we can't shift unknown bits
          // into the mask.  This can only happen with signed shift
          // rights, as they sign-extend.
          if (ShAmt) {
            bool CanFold = Shift->getOpcode() != Instruction::Shr ||
                           Shift->getType()->isUnsigned();
            if (!CanFold) {
              // To test for the bad case of the signed shr, see if any
              // of the bits shifted in could be tested after the mask.
              int ShAmtVal = Ty->getPrimitiveSizeInBits()-ShAmt->getValue();
              if (ShAmtVal < 0) ShAmtVal = 0; // Out of range shift.

              Constant *OShAmt = ConstantUInt::get(Type::UByteTy, ShAmtVal);
              Constant *ShVal =
                ConstantExpr::getShl(ConstantInt::getAllOnesValue(Ty), OShAmt);
              if (ConstantExpr::getAnd(ShVal, AndCST)->isNullValue())
                CanFold = true;
            }

            if (CanFold) {
              Constant *NewCst;
              if (Shift->getOpcode() == Instruction::Shl)
                NewCst = ConstantExpr::getUShr(CI, ShAmt);
              else
                NewCst = ConstantExpr::getShl(CI, ShAmt);

              // Check to see if we are shifting out any of the bits being
              // compared.
              if (ConstantExpr::get(Shift->getOpcode(), NewCst, ShAmt) != CI){
                // If we shifted bits out, the fold is not going to work out.
                // As a special case, check to see if this means that the
                // result is always true or false now.
                if (I.getOpcode() == Instruction::SetEQ)
                  return ReplaceInstUsesWith(I, ConstantBool::False);
                if (I.getOpcode() == Instruction::SetNE)
                  return ReplaceInstUsesWith(I, ConstantBool::True);
              } else {
                I.setOperand(1, NewCst);
                Constant *NewAndCST;
                if (Shift->getOpcode() == Instruction::Shl)
                  NewAndCST = ConstantExpr::getUShr(AndCST, ShAmt);
                else
                  NewAndCST = ConstantExpr::getShl(AndCST, ShAmt);
                LHSI->setOperand(1, NewAndCST);
                LHSI->setOperand(0, Shift->getOperand(0));
                WorkList.push_back(Shift); // Shift is dead.
                AddUsesToWorkList(I);
                return &I;
              }
            }
          }
        }
        break;

      case Instruction::Shl:         // (setcc (shl X, ShAmt), CI)
        if (ConstantUInt *ShAmt = dyn_cast<ConstantUInt>(LHSI->getOperand(1))) {
          switch (I.getOpcode()) {
          default: break;
          case Instruction::SetEQ:
          case Instruction::SetNE: {
            unsigned TypeBits = CI->getType()->getPrimitiveSizeInBits();

            // Check that the shift amount is in range.  If not, don't perform
            // undefined shifts.  When the shift is visited it will be
            // simplified.
            if (ShAmt->getValue() >= TypeBits)
              break;

            // If we are comparing against bits always shifted out, the
            // comparison cannot succeed.
            Constant *Comp =
              ConstantExpr::getShl(ConstantExpr::getShr(CI, ShAmt), ShAmt);
            if (Comp != CI) {// Comparing against a bit that we know is zero.
              bool IsSetNE = I.getOpcode() == Instruction::SetNE;
              Constant *Cst = ConstantBool::get(IsSetNE);
              return ReplaceInstUsesWith(I, Cst);
            }

            if (LHSI->hasOneUse()) {
              // Otherwise strength reduce the shift into an and.
              unsigned ShAmtVal = (unsigned)ShAmt->getValue();
              uint64_t Val = (1ULL << (TypeBits-ShAmtVal))-1;

              Constant *Mask;
              if (CI->getType()->isUnsigned()) {
                Mask = ConstantUInt::get(CI->getType(), Val);
              } else if (ShAmtVal != 0) {
                Mask = ConstantSInt::get(CI->getType(), Val);
              } else {
                Mask = ConstantInt::getAllOnesValue(CI->getType());
              }

              Instruction *AndI =
                BinaryOperator::createAnd(LHSI->getOperand(0),
                                          Mask, LHSI->getName()+".mask");
              Value *And = InsertNewInstBefore(AndI, I);
              return new SetCondInst(I.getOpcode(), And,
                                     ConstantExpr::getUShr(CI, ShAmt));
            }
          }
          }
        }
        break;

      case Instruction::Shr:         // (setcc (shr X, ShAmt), CI)
        if (ConstantUInt *ShAmt = dyn_cast<ConstantUInt>(LHSI->getOperand(1))) {
          switch (I.getOpcode()) {
          default: break;
          case Instruction::SetEQ:
          case Instruction::SetNE: {

            // Check that the shift amount is in range.  If not, don't perform
            // undefined shifts.  When the shift is visited it will be
            // simplified.
            unsigned TypeBits = CI->getType()->getPrimitiveSizeInBits();
            if (ShAmt->getValue() >= TypeBits)
              break;

            // If we are comparing against bits always shifted out, the
            // comparison cannot succeed.
            Constant *Comp =
              ConstantExpr::getShr(ConstantExpr::getShl(CI, ShAmt), ShAmt);

            if (Comp != CI) {// Comparing against a bit that we know is zero.
              bool IsSetNE = I.getOpcode() == Instruction::SetNE;
              Constant *Cst = ConstantBool::get(IsSetNE);
              return ReplaceInstUsesWith(I, Cst);
            }

            if (LHSI->hasOneUse() || CI->isNullValue()) {
              unsigned ShAmtVal = (unsigned)ShAmt->getValue();

              // Otherwise strength reduce the shift into an and.
              uint64_t Val = ~0ULL;          // All ones.
              Val <<= ShAmtVal;              // Shift over to the right spot.

              Constant *Mask;
              if (CI->getType()->isUnsigned()) {
                Val &= ~0ULL >> (64-TypeBits);
                Mask = ConstantUInt::get(CI->getType(), Val);
              } else {
                Mask = ConstantSInt::get(CI->getType(), Val);
              }

              Instruction *AndI =
                BinaryOperator::createAnd(LHSI->getOperand(0),
                                          Mask, LHSI->getName()+".mask");
              Value *And = InsertNewInstBefore(AndI, I);
              return new SetCondInst(I.getOpcode(), And,
                                     ConstantExpr::getShl(CI, ShAmt));
            }
            break;
          }
          }
        }
        break;

      case Instruction::Div:
        // Fold: (div X, C1) op C2 -> range check
        if (ConstantInt *DivRHS = dyn_cast<ConstantInt>(LHSI->getOperand(1))) {
          // Fold this div into the comparison, producing a range check.
          // Determine, based on the divide type, what the range is being
          // checked.  If there is an overflow on the low or high side, remember
          // it, otherwise compute the range [low, hi) bounding the new value.
          bool LoOverflow = false, HiOverflow = 0;
          ConstantInt *LoBound = 0, *HiBound = 0;

          ConstantInt *Prod;
          bool ProdOV = MulWithOverflow(Prod, CI, DivRHS);

          Instruction::BinaryOps Opcode = I.getOpcode();

          if (DivRHS->isNullValue()) {  // Don't hack on divide by zeros.
          } else if (LHSI->getType()->isUnsigned()) {  // udiv
            LoBound = Prod;
            LoOverflow = ProdOV;
            HiOverflow = ProdOV || AddWithOverflow(HiBound, LoBound, DivRHS);
          } else if (isPositive(DivRHS)) {             // Divisor is > 0.
            if (CI->isNullValue()) {       // (X / pos) op 0
              // Can't overflow.
              LoBound = cast<ConstantInt>(ConstantExpr::getNeg(SubOne(DivRHS)));
              HiBound = DivRHS;
            } else if (isPositive(CI)) {   // (X / pos) op pos
              LoBound = Prod;
              LoOverflow = ProdOV;
              HiOverflow = ProdOV || AddWithOverflow(HiBound, Prod, DivRHS);
            } else {                       // (X / pos) op neg
              Constant *DivRHSH = ConstantExpr::getNeg(SubOne(DivRHS));
              LoOverflow = AddWithOverflow(LoBound, Prod,
                                           cast<ConstantInt>(DivRHSH));
              HiBound = Prod;
              HiOverflow = ProdOV;
            }
          } else {                                     // Divisor is < 0.
            if (CI->isNullValue()) {       // (X / neg) op 0
              LoBound = AddOne(DivRHS);
              HiBound = cast<ConstantInt>(ConstantExpr::getNeg(DivRHS));
              if (HiBound == DivRHS)
                LoBound = 0;  // - INTMIN = INTMIN
            } else if (isPositive(CI)) {   // (X / neg) op pos
              HiOverflow = LoOverflow = ProdOV;
              if (!LoOverflow)
                LoOverflow = AddWithOverflow(LoBound, Prod, AddOne(DivRHS));
              HiBound = AddOne(Prod);
            } else {                       // (X / neg) op neg
              LoBound = Prod;
              LoOverflow = HiOverflow = ProdOV;
              HiBound = cast<ConstantInt>(ConstantExpr::getSub(Prod, DivRHS));
            }

            // Dividing by a negate swaps the condition.
            Opcode = SetCondInst::getSwappedCondition(Opcode);
          }

          if (LoBound) {
            Value *X = LHSI->getOperand(0);
            switch (Opcode) {
            default: assert(0 && "Unhandled setcc opcode!");
            case Instruction::SetEQ:
              if (LoOverflow && HiOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::False);
              else if (HiOverflow)
                return new SetCondInst(Instruction::SetGE, X, LoBound);
              else if (LoOverflow)
                return new SetCondInst(Instruction::SetLT, X, HiBound);
              else
                return InsertRangeTest(X, LoBound, HiBound, true, I);
            case Instruction::SetNE:
              if (LoOverflow && HiOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::True);
              else if (HiOverflow)
                return new SetCondInst(Instruction::SetLT, X, LoBound);
              else if (LoOverflow)
                return new SetCondInst(Instruction::SetGE, X, HiBound);
              else
                return InsertRangeTest(X, LoBound, HiBound, false, I);
            case Instruction::SetLT:
              if (LoOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::False);
              return new SetCondInst(Instruction::SetLT, X, LoBound);
            case Instruction::SetGT:
              if (HiOverflow)
                return ReplaceInstUsesWith(I, ConstantBool::False);
              return new SetCondInst(Instruction::SetGE, X, HiBound);
            }
          }
        }
        break;
      }

    // Simplify seteq and setne instructions...
    if (I.getOpcode() == Instruction::SetEQ ||
        I.getOpcode() == Instruction::SetNE) {
      bool isSetNE = I.getOpcode() == Instruction::SetNE;

      // If the first operand is (and|or|xor) with a constant, and the second
      // operand is a constant, simplify a bit.
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0)) {
        switch (BO->getOpcode()) {
        case Instruction::Rem:
          // If we have a signed (X % (2^c)) == 0, turn it into an unsigned one.
          if (CI->isNullValue() && isa<ConstantSInt>(BO->getOperand(1)) &&
              BO->hasOneUse() &&
              cast<ConstantSInt>(BO->getOperand(1))->getValue() > 1) {
            int64_t V = cast<ConstantSInt>(BO->getOperand(1))->getValue();
            if (isPowerOf2_64(V)) {
              unsigned L2 = Log2_64(V);
              const Type *UTy = BO->getType()->getUnsignedVersion();
              Value *NewX = InsertNewInstBefore(new CastInst(BO->getOperand(0),
                                                             UTy, "tmp"), I);
              Constant *RHSCst = ConstantUInt::get(UTy, 1ULL << L2);
              Value *NewRem =InsertNewInstBefore(BinaryOperator::createRem(NewX,
                                                    RHSCst, BO->getName()), I);
              return BinaryOperator::create(I.getOpcode(), NewRem,
                                            Constant::getNullValue(UTy));
            }
          }
          break;

        case Instruction::Add:
          // Replace ((add A, B) != C) with (A != C-B) if B & C are constants.
          if (ConstantInt *BOp1C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
            if (BO->hasOneUse())
              return new SetCondInst(I.getOpcode(), BO->getOperand(0),
                                     ConstantExpr::getSub(CI, BOp1C));
          } else if (CI->isNullValue()) {
            // Replace ((add A, B) != 0) with (A != -B) if A or B is
            // efficiently invertible, or if the add has just this one use.
            Value *BOp0 = BO->getOperand(0), *BOp1 = BO->getOperand(1);

            if (Value *NegVal = dyn_castNegVal(BOp1))
              return new SetCondInst(I.getOpcode(), BOp0, NegVal);
            else if (Value *NegVal = dyn_castNegVal(BOp0))
              return new SetCondInst(I.getOpcode(), NegVal, BOp1);
            else if (BO->hasOneUse()) {
              Instruction *Neg = BinaryOperator::createNeg(BOp1, BO->getName());
              BO->setName("");
              InsertNewInstBefore(Neg, I);
              return new SetCondInst(I.getOpcode(), BOp0, Neg);
            }
          }
          break;
        case Instruction::Xor:
          // For the xor case, we can xor two constants together, eliminating
          // the explicit xor.
          if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1)))
            return BinaryOperator::create(I.getOpcode(), BO->getOperand(0),
                                  ConstantExpr::getXor(CI, BOC));

          // FALLTHROUGH
        case Instruction::Sub:
          // Replace (([sub|xor] A, B) != 0) with (A != B)
          if (CI->isNullValue())
            return new SetCondInst(I.getOpcode(), BO->getOperand(0),
                                   BO->getOperand(1));
          break;

        case Instruction::Or:
          // If bits are being or'd in that are not present in the constant we
          // are comparing against, then the comparison could never succeed!
          if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1))) {
            Constant *NotCI = ConstantExpr::getNot(CI);
            if (!ConstantExpr::getAnd(BOC, NotCI)->isNullValue())
              return ReplaceInstUsesWith(I, ConstantBool::get(isSetNE));
          }
          break;

        case Instruction::And:
          if (ConstantInt *BOC = dyn_cast<ConstantInt>(BO->getOperand(1))) {
            // If bits are being compared against that are and'd out, then the
            // comparison can never succeed!
            if (!ConstantExpr::getAnd(CI,
                                      ConstantExpr::getNot(BOC))->isNullValue())
              return ReplaceInstUsesWith(I, ConstantBool::get(isSetNE));

            // If we have ((X & C) == C), turn it into ((X & C) != 0).
            if (CI == BOC && isOneBitSet(CI))
              return new SetCondInst(isSetNE ? Instruction::SetEQ :
                                     Instruction::SetNE, Op0,
                                     Constant::getNullValue(CI->getType()));

            // Replace (and X, (1 << size(X)-1) != 0) with x < 0, converting X
            // to be a signed value as appropriate.
            if (isSignBit(BOC)) {
              Value *X = BO->getOperand(0);
              // If 'X' is not signed, insert a cast now...
              if (!BOC->getType()->isSigned()) {
                const Type *DestTy = BOC->getType()->getSignedVersion();
                X = InsertCastBefore(X, DestTy, I);
              }
              return new SetCondInst(isSetNE ? Instruction::SetLT :
                                         Instruction::SetGE, X,
                                     Constant::getNullValue(X->getType()));
            }

            // ((X & ~7) == 0) --> X < 8
            if (CI->isNullValue() && isHighOnes(BOC)) {
              Value *X = BO->getOperand(0);
              Constant *NegX = ConstantExpr::getNeg(BOC);

              // If 'X' is signed, insert a cast now.
              if (NegX->getType()->isSigned()) {
                const Type *DestTy = NegX->getType()->getUnsignedVersion();
                X = InsertCastBefore(X, DestTy, I);
                NegX = ConstantExpr::getCast(NegX, DestTy);
              }

              return new SetCondInst(isSetNE ? Instruction::SetGE :
                                     Instruction::SetLT, X, NegX);
            }

          }
        default: break;
        }
      }
    } else {  // Not a SetEQ/SetNE
      // If the LHS is a cast from an integral value of the same size,
      if (CastInst *Cast = dyn_cast<CastInst>(Op0)) {
        Value *CastOp = Cast->getOperand(0);
        const Type *SrcTy = CastOp->getType();
        unsigned SrcTySize = SrcTy->getPrimitiveSizeInBits();
        if (SrcTy != Cast->getType() && SrcTy->isInteger() &&
            SrcTySize == Cast->getType()->getPrimitiveSizeInBits()) {
          assert((SrcTy->isSigned() ^ Cast->getType()->isSigned()) &&
                 "Source and destination signednesses should differ!");
          if (Cast->getType()->isSigned()) {
            // If this is a signed comparison, check for comparisons in the
            // vicinity of zero.
            if (I.getOpcode() == Instruction::SetLT && CI->isNullValue())
              // X < 0  => x > 127
              return BinaryOperator::createSetGT(CastOp,
                         ConstantUInt::get(SrcTy, (1ULL << (SrcTySize-1))-1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     cast<ConstantSInt>(CI)->getValue() == -1)
              // X > -1  => x < 128
              return BinaryOperator::createSetLT(CastOp,
                         ConstantUInt::get(SrcTy, 1ULL << (SrcTySize-1)));
          } else {
            ConstantUInt *CUI = cast<ConstantUInt>(CI);
            if (I.getOpcode() == Instruction::SetLT &&
                CUI->getValue() == 1ULL << (SrcTySize-1))
              // X < 128 => X > -1
              return BinaryOperator::createSetGT(CastOp,
                                                 ConstantSInt::get(SrcTy, -1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     CUI->getValue() == (1ULL << (SrcTySize-1))-1)
              // X > 127 => X < 0
              return BinaryOperator::createSetLT(CastOp,
                                                 Constant::getNullValue(SrcTy));
          }
        }
      }
    }
  }

  // Handle setcc with constant RHS's that can be integer, FP or pointer.
  if (Constant *RHSC = dyn_cast<Constant>(Op1)) {
    if (Instruction *LHSI = dyn_cast<Instruction>(Op0))
      switch (LHSI->getOpcode()) {
      case Instruction::GetElementPtr:
        if (RHSC->isNullValue()) {
          // Transform setcc GEP P, int 0, int 0, int 0, null -> setcc P, null
          bool isAllZeros = true;
          for (unsigned i = 1, e = LHSI->getNumOperands(); i != e; ++i)
            if (!isa<Constant>(LHSI->getOperand(i)) ||
                !cast<Constant>(LHSI->getOperand(i))->isNullValue()) {
              isAllZeros = false;
              break;
            }
          if (isAllZeros)
            return new SetCondInst(I.getOpcode(), LHSI->getOperand(0),
                    Constant::getNullValue(LHSI->getOperand(0)->getType()));
        }
        break;

      case Instruction::PHI:
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
        break;
      case Instruction::Select:
        // If either operand of the select is a constant, we can fold the
        // comparison into the select arms, which will cause one to be
        // constant folded and the select turned into a bitwise or.
        Value *Op1 = 0, *Op2 = 0;
        if (LHSI->hasOneUse()) {
          if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(1))) {
            // Fold the known value into the constant operand.
            Op1 = ConstantExpr::get(I.getOpcode(), C, RHSC);
            // Insert a new SetCC of the other select operand.
            Op2 = InsertNewInstBefore(new SetCondInst(I.getOpcode(),
                                                      LHSI->getOperand(2), RHSC,
                                                      I.getName()), I);
          } else if (Constant *C = dyn_cast<Constant>(LHSI->getOperand(2))) {
            // Fold the known value into the constant operand.
            Op2 = ConstantExpr::get(I.getOpcode(), C, RHSC);
            // Insert a new SetCC of the other select operand.
            Op1 = InsertNewInstBefore(new SetCondInst(I.getOpcode(),
                                                      LHSI->getOperand(1), RHSC,
                                                      I.getName()), I);
          }
        }

        if (Op1)
          return new SelectInst(LHSI->getOperand(0), Op1, Op2);
        break;
      }
  }

  // If we can optimize a 'setcc GEP, P' or 'setcc P, GEP', do so now.
  if (User *GEP = dyn_castGetElementPtr(Op0))
    if (Instruction *NI = FoldGEPSetCC(GEP, Op1, I.getOpcode(), I))
      return NI;
  if (User *GEP = dyn_castGetElementPtr(Op1))
    if (Instruction *NI = FoldGEPSetCC(GEP, Op0,
                           SetCondInst::getSwappedCondition(I.getOpcode()), I))
      return NI;

  // Test to see if the operands of the setcc are casted versions of other
  // values.  If the cast can be stripped off both arguments, we do so now.
  if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
    Value *CastOp0 = CI->getOperand(0);
    if (CastOp0->getType()->isLosslesslyConvertibleTo(CI->getType()) &&
        (isa<Constant>(Op1) || isa<CastInst>(Op1)) &&
        (I.getOpcode() == Instruction::SetEQ ||
         I.getOpcode() == Instruction::SetNE)) {
      // We keep moving the cast from the left operand over to the right
      // operand, where it can often be eliminated completely.
      Op0 = CastOp0;

      // If operand #1 is a cast instruction, see if we can eliminate it as
      // well.
      if (CastInst *CI2 = dyn_cast<CastInst>(Op1))
        if (CI2->getOperand(0)->getType()->isLosslesslyConvertibleTo(
                                                               Op0->getType()))
          Op1 = CI2->getOperand(0);

      // If Op1 is a constant, we can fold the cast into the constant.
      if (Op1->getType() != Op0->getType())
        if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
          Op1 = ConstantExpr::getCast(Op1C, Op0->getType());
        } else {
          // Otherwise, cast the RHS right before the setcc
          Op1 = new CastInst(Op1, Op0->getType(), Op1->getName());
          InsertNewInstBefore(cast<Instruction>(Op1), I);
        }
      return BinaryOperator::create(I.getOpcode(), Op0, Op1);
    }

    // Handle the special case of: setcc (cast bool to X), <cst>
    // This comes up when you have code like
    //   int X = A < B;
    //   if (X) ...
    // For generality, we handle any zero-extension of any operand comparison
    // with a constant or another cast from the same type.
    if (isa<ConstantInt>(Op1) || isa<CastInst>(Op1))
      if (Instruction *R = visitSetCondInstWithCastAndCast(I))
        return R;
  }
  return Changed ? &I : 0;
}

// visitSetCondInstWithCastAndCast - Handle setcond (cast x to y), (cast/cst).
// We only handle extending casts so far.
//
Instruction *InstCombiner::visitSetCondInstWithCastAndCast(SetCondInst &SCI) {
  Value *LHSCIOp = cast<CastInst>(SCI.getOperand(0))->getOperand(0);
  const Type *SrcTy = LHSCIOp->getType();
  const Type *DestTy = SCI.getOperand(0)->getType();
  Value *RHSCIOp;

  if (!DestTy->isIntegral() || !SrcTy->isIntegral())
    return 0;

  unsigned SrcBits  = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBits = DestTy->getPrimitiveSizeInBits();
  if (SrcBits >= DestBits) return 0;  // Only handle extending cast.

  // Is this a sign or zero extension?
  bool isSignSrc  = SrcTy->isSigned();
  bool isSignDest = DestTy->isSigned();

  if (CastInst *CI = dyn_cast<CastInst>(SCI.getOperand(1))) {
    // Not an extension from the same type?
    RHSCIOp = CI->getOperand(0);
    if (RHSCIOp->getType() != LHSCIOp->getType()) return 0;
  } else if (ConstantInt *CI = dyn_cast<ConstantInt>(SCI.getOperand(1))) {
    // Compute the constant that would happen if we truncated to SrcTy then
    // reextended to DestTy.
    Constant *Res = ConstantExpr::getCast(CI, SrcTy);

    if (ConstantExpr::getCast(Res, DestTy) == CI) {
      RHSCIOp = Res;
    } else {
      // If the value cannot be represented in the shorter type, we cannot emit
      // a simple comparison.
      if (SCI.getOpcode() == Instruction::SetEQ)
        return ReplaceInstUsesWith(SCI, ConstantBool::False);
      if (SCI.getOpcode() == Instruction::SetNE)
        return ReplaceInstUsesWith(SCI, ConstantBool::True);

      // Evaluate the comparison for LT.
      Value *Result;
      if (DestTy->isSigned()) {
        // We're performing a signed comparison.
        if (isSignSrc) {
          // Signed extend and signed comparison.
          if (cast<ConstantSInt>(CI)->getValue() < 0) // X < (small) --> false
            Result = ConstantBool::False;
          else
            Result = ConstantBool::True;              // X < (large) --> true
        } else {
          // Unsigned extend and signed comparison.
          if (cast<ConstantSInt>(CI)->getValue() < 0)
            Result = ConstantBool::False;
          else
            Result = ConstantBool::True;
        }
      } else {
        // We're performing an unsigned comparison.
        if (!isSignSrc) {
          // Unsigned extend & compare -> always true.
          Result = ConstantBool::True;
        } else {
          // We're performing an unsigned comp with a sign extended value.
          // This is true if the input is >= 0. [aka >s -1]
          Constant *NegOne = ConstantIntegral::getAllOnesValue(SrcTy);
          Result = InsertNewInstBefore(BinaryOperator::createSetGT(LHSCIOp,
                                                  NegOne, SCI.getName()), SCI);
        }
      }

      // Finally, return the value computed.
      if (SCI.getOpcode() == Instruction::SetLT) {
        return ReplaceInstUsesWith(SCI, Result);
      } else {
        assert(SCI.getOpcode()==Instruction::SetGT &&"SetCC should be folded!");
        if (Constant *CI = dyn_cast<Constant>(Result))
          return ReplaceInstUsesWith(SCI, ConstantExpr::getNot(CI));
        else
          return BinaryOperator::createNot(Result);
      }
    }
  } else {
    return 0;
  }

  // Okay, just insert a compare of the reduced operands now!
  return BinaryOperator::create(SCI.getOpcode(), LHSCIOp, RHSCIOp);
}

Instruction *InstCombiner::visitShiftInst(ShiftInst &I) {
  assert(I.getOperand(1)->getType() == Type::UByteTy);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  bool isLeftShift = I.getOpcode() == Instruction::Shl;

  // shl X, 0 == X and shr X, 0 == X
  // shl 0, X == 0 and shr 0, X == 0
  if (Op1 == Constant::getNullValue(Type::UByteTy) ||
      Op0 == Constant::getNullValue(Op0->getType()))
    return ReplaceInstUsesWith(I, Op0);

  if (isa<UndefValue>(Op0)) {            // undef >>s X -> undef
    if (!isLeftShift && I.getType()->isSigned())
      return ReplaceInstUsesWith(I, Op0);
    else                         // undef << X -> 0   AND  undef >>u X -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1)) {
    if (isLeftShift || I.getType()->isUnsigned())// X << undef, X >>u undef -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
    else
      return ReplaceInstUsesWith(I, Op0);          // X >>s undef -> X
  }

  // shr int -1, X = -1   (for any arithmetic shift rights of ~0)
  if (!isLeftShift)
    if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(Op0))
      if (CSI->isAllOnesValue())
        return ReplaceInstUsesWith(I, CSI);

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI, this))
        return R;

  // See if we can turn a signed shr into an unsigned shr.
  if (!isLeftShift && I.getType()->isSigned()) {
    if (MaskedValueIsZero(Op0, ConstantInt::getMinValue(I.getType()))) {
      Value *V = InsertCastBefore(Op0, I.getType()->getUnsignedVersion(), I);
      V = InsertNewInstBefore(new ShiftInst(Instruction::Shr, V, Op1,
                                            I.getName()), I);
      return new CastInst(V, I.getType());
    }
  }

  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(Op1))
    if (Instruction *Res = FoldShiftByConstant(Op0, CUI, I))
      return Res;
  return 0;
}

Instruction *InstCombiner::FoldShiftByConstant(Value *Op0, ConstantUInt *Op1,
                                               ShiftInst &I) {
  bool isLeftShift = I.getOpcode() == Instruction::Shl;
  bool isSignedShift = Op0->getType()->isSigned();
  bool isUnsignedShift = !isSignedShift;

  // shl uint X, 32 = 0 and shr ubyte Y, 9 = 0, ... just don't eliminate shr
  // of a signed value.
  //
  unsigned TypeBits = Op0->getType()->getPrimitiveSizeInBits();
  if (Op1->getValue() >= TypeBits) {
    if (isUnsignedShift || isLeftShift)
      return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
    else {
      I.setOperand(1, ConstantUInt::get(Type::UByteTy, TypeBits-1));
      return &I;
    }
  }
  
  // ((X*C1) << C2) == (X * (C1 << C2))
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
    if (BO->getOpcode() == Instruction::Mul && isLeftShift)
      if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
        return BinaryOperator::createMul(BO->getOperand(0),
                                         ConstantExpr::getShl(BOOp, Op1));
  
  // Try to fold constant and into select arguments.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = FoldOpIntoSelect(I, SI, this))
      return R;
  if (isa<PHINode>(Op0))
    if (Instruction *NV = FoldOpIntoPhi(I))
      return NV;
  
  if (Op0->hasOneUse()) {
    // If this is a SHL of a sign-extending cast, see if we can turn the input
    // into a zero extending cast (a simple strength reduction).
    if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
      const Type *SrcTy = CI->getOperand(0)->getType();
      if (isLeftShift && SrcTy->isInteger() && SrcTy->isSigned() &&
          SrcTy->getPrimitiveSizeInBits() <
          CI->getType()->getPrimitiveSizeInBits()) {
        // We can change it to a zero extension if we are shifting out all of
        // the sign extended bits.  To check this, form a mask of all of the
        // sign extend bits, then shift them left and see if we have anything
        // left.
        Constant *Mask = ConstantIntegral::getAllOnesValue(SrcTy); //     1111
        Mask = ConstantExpr::getZeroExtend(Mask, CI->getType());   // 00001111
        Mask = ConstantExpr::getNot(Mask);   // 1's in the sign bits: 11110000
        if (ConstantExpr::getShl(Mask, Op1)->isNullValue()) {
          // If the shift is nuking all of the sign bits, change this to a
          // zero extension cast.  To do this, cast the cast input to
          // unsigned, then to the requested size.
          Value *CastOp = CI->getOperand(0);
          Instruction *NC =
            new CastInst(CastOp, CastOp->getType()->getUnsignedVersion(),
                         CI->getName()+".uns");
          NC = InsertNewInstBefore(NC, I);
          // Finally, insert a replacement for CI.
          NC = new CastInst(NC, CI->getType(), CI->getName());
          CI->setName("");
          NC = InsertNewInstBefore(NC, I);
          WorkList.push_back(CI);  // Delete CI later.
          I.setOperand(0, NC);
          return &I;               // The SHL operand was modified.
        }
      }
    }
    
    if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0)) {
      // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
      Value *V1, *V2;
      ConstantInt *CC;
      switch (Op0BO->getOpcode()) {
        default: break;
        case Instruction::Add:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
          // These operators commute.
          // Turn (Y + (X >> C)) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
              match(Op0BO->getOperand(1),
                    m_Shr(m_Value(V1), m_ConstantInt(CC))) && CC == Op1) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(0), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *X = BinaryOperator::create(Op0BO->getOpcode(), YS,
                                                    V1,
                                                    Op0BO->getOperand(1)->getName());
            InsertNewInstBefore(X, I);  // (X + (Y << C))
            Constant *C2 = ConstantInt::getAllOnesValue(X->getType());
            C2 = ConstantExpr::getShl(C2, Op1);
            return BinaryOperator::createAnd(X, C2);
          }
          
          // Turn (Y + ((X >> C) & CC)) << C  ->  ((X & (CC << C)) + (Y << C))
          if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
              match(Op0BO->getOperand(1),
                    m_And(m_Shr(m_Value(V1), m_Value(V2)),
                          m_ConstantInt(CC))) && V2 == Op1 &&
              cast<BinaryOperator>(Op0BO->getOperand(1))->getOperand(0)->hasOneUse()) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(0), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *XM =
              BinaryOperator::createAnd(V1, ConstantExpr::getShl(CC, Op1),
                                        V1->getName()+".mask");
            InsertNewInstBefore(XM, I); // X & (CC << C)
            
            return BinaryOperator::create(Op0BO->getOpcode(), YS, XM);
          }
          
          // FALL THROUGH.
        case Instruction::Sub:
          // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0),
                    m_Shr(m_Value(V1), m_ConstantInt(CC))) && CC == Op1) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(1), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *X = BinaryOperator::create(Op0BO->getOpcode(), YS,
                                                    V1,
                                                    Op0BO->getOperand(0)->getName());
            InsertNewInstBefore(X, I);  // (X + (Y << C))
            Constant *C2 = ConstantInt::getAllOnesValue(X->getType());
            C2 = ConstantExpr::getShl(C2, Op1);
            return BinaryOperator::createAnd(X, C2);
          }
          
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0),
                    m_And(m_Shr(m_Value(V1), m_Value(V2)),
                          m_ConstantInt(CC))) && V2 == Op1 &&
              cast<BinaryOperator>(Op0BO->getOperand(0))->getOperand(0)->hasOneUse()) {
            Instruction *YS = new ShiftInst(Instruction::Shl, 
                                            Op0BO->getOperand(1), Op1,
                                            Op0BO->getName());
            InsertNewInstBefore(YS, I); // (Y << C)
            Instruction *XM =
              BinaryOperator::createAnd(V1, ConstantExpr::getShl(CC, Op1),
                                        V1->getName()+".mask");
            InsertNewInstBefore(XM, I); // X & (CC << C)
            
            return BinaryOperator::create(Op0BO->getOpcode(), YS, XM);
          }
          
          break;
      }
      
      
      // If the operand is an bitwise operator with a constant RHS, and the
      // shift is the only use, we can pull it out of the shift.
      if (ConstantInt *Op0C = dyn_cast<ConstantInt>(Op0BO->getOperand(1))) {
        bool isValid = true;     // Valid only for And, Or, Xor
        bool highBitSet = false; // Transform if high bit of constant set?
        
        switch (Op0BO->getOpcode()) {
          default: isValid = false; break;   // Do not perform transform!
          case Instruction::Add:
            isValid = isLeftShift;
            break;
          case Instruction::Or:
          case Instruction::Xor:
            highBitSet = false;
            break;
          case Instruction::And:
            highBitSet = true;
            break;
        }
        
        // If this is a signed shift right, and the high bit is modified
        // by the logical operation, do not perform the transformation.
        // The highBitSet boolean indicates the value of the high bit of
        // the constant which would cause it to be modified for this
        // operation.
        //
        if (isValid && !isLeftShift && isSignedShift) {
          uint64_t Val = Op0C->getRawValue();
          isValid = ((Val & (1 << (TypeBits-1))) != 0) == highBitSet;
        }
        
        if (isValid) {
          Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, Op1);
          
          Instruction *NewShift =
            new ShiftInst(I.getOpcode(), Op0BO->getOperand(0), Op1,
                          Op0BO->getName());
          Op0BO->setName("");
          InsertNewInstBefore(NewShift, I);
          
          return BinaryOperator::create(Op0BO->getOpcode(), NewShift,
                                        NewRHS);
        }
      }
    }
  }
  
  // Find out if this is a shift of a shift by a constant.
  ShiftInst *ShiftOp = 0;
  if (ShiftInst *Op0SI = dyn_cast<ShiftInst>(Op0))
    ShiftOp = Op0SI;
  else if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
    // If this is a noop-integer case of a shift instruction, use the shift.
    if (CI->getOperand(0)->getType()->isInteger() &&
        CI->getOperand(0)->getType()->getPrimitiveSizeInBits() ==
        CI->getType()->getPrimitiveSizeInBits() &&
        isa<ShiftInst>(CI->getOperand(0))) {
      ShiftOp = cast<ShiftInst>(CI->getOperand(0));
    }
  }
  
  if (ShiftOp && isa<ConstantUInt>(ShiftOp->getOperand(1))) {
    // Find the operands and properties of the input shift.  Note that the
    // signedness of the input shift may differ from the current shift if there
    // is a noop cast between the two.
    bool isShiftOfLeftShift = ShiftOp->getOpcode() == Instruction::Shl;
    bool isShiftOfSignedShift = ShiftOp->getType()->isSigned();
    bool isShiftOfUnsignedShift = !isShiftOfSignedShift;
    
    ConstantUInt *ShiftAmt1C = cast<ConstantUInt>(ShiftOp->getOperand(1));

    unsigned ShiftAmt1 = (unsigned)ShiftAmt1C->getValue();
    unsigned ShiftAmt2 = (unsigned)Op1->getValue();
    
    // Check for (A << c1) << c2   and   (A >> c1) >> c2.
    if (isLeftShift == isShiftOfLeftShift) {
      // Do not fold these shifts if the first one is signed and the second one
      // is unsigned and this is a right shift.  Further, don't do any folding
      // on them.
      if (isShiftOfSignedShift && isUnsignedShift && !isLeftShift)
        return 0;
      
      unsigned Amt = ShiftAmt1+ShiftAmt2;   // Fold into one big shift.
      if (Amt > Op0->getType()->getPrimitiveSizeInBits())
        Amt = Op0->getType()->getPrimitiveSizeInBits();
      
      Value *Op = ShiftOp->getOperand(0);
      if (isShiftOfSignedShift != isSignedShift)
        Op = InsertNewInstBefore(new CastInst(Op, I.getType(), "tmp"), I);
      return new ShiftInst(I.getOpcode(), Op,
                           ConstantUInt::get(Type::UByteTy, Amt));
    }
    
    // Check for (A << c1) >> c2 or (A >> c1) << c2.  If we are dealing with
    // signed types, we can only support the (A >> c1) << c2 configuration,
    // because it can not turn an arbitrary bit of A into a sign bit.
    if (isUnsignedShift || isLeftShift) {
      // Calculate bitmask for what gets shifted off the edge.
      Constant *C = ConstantIntegral::getAllOnesValue(I.getType());
      if (isLeftShift)
        C = ConstantExpr::getShl(C, ShiftAmt1C);
      else
        C = ConstantExpr::getUShr(C, ShiftAmt1C);
      
      Value *Op = ShiftOp->getOperand(0);
      if (isShiftOfSignedShift != isSignedShift)
        Op = InsertNewInstBefore(new CastInst(Op, I.getType(),Op->getName()),I);
      
      Instruction *Mask =
        BinaryOperator::createAnd(Op, C, Op->getName()+".mask");
      InsertNewInstBefore(Mask, I);
      
      // Figure out what flavor of shift we should use...
      if (ShiftAmt1 == ShiftAmt2) {
        return ReplaceInstUsesWith(I, Mask);  // (A << c) >> c  === A & c2
      } else if (ShiftAmt1 < ShiftAmt2) {
        return new ShiftInst(I.getOpcode(), Mask,
                         ConstantUInt::get(Type::UByteTy, ShiftAmt2-ShiftAmt1));
      } else if (isShiftOfUnsignedShift || isShiftOfLeftShift) {
        if (isShiftOfUnsignedShift && !isShiftOfLeftShift && isSignedShift) {
          // Make sure to emit an unsigned shift right, not a signed one.
          Mask = InsertNewInstBefore(new CastInst(Mask, 
                                        Mask->getType()->getUnsignedVersion(),
                                                  Op->getName()), I);
          Mask = new ShiftInst(Instruction::Shr, Mask,
                         ConstantUInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
          InsertNewInstBefore(Mask, I);
          return new CastInst(Mask, I.getType());
        } else {
          return new ShiftInst(ShiftOp->getOpcode(), Mask,
                    ConstantUInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
        }
      } else {
        // (X >>s C1) << C2  where C1 > C2  === (X >>s (C1-C2)) & mask
        Op = InsertNewInstBefore(new CastInst(Mask,
                                              I.getType()->getSignedVersion(),
                                              Mask->getName()), I);
        Instruction *Shift =
          new ShiftInst(ShiftOp->getOpcode(), Op,
                        ConstantUInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
        InsertNewInstBefore(Shift, I);
        
        C = ConstantIntegral::getAllOnesValue(Shift->getType());
        C = ConstantExpr::getShl(C, Op1);
        Mask = BinaryOperator::createAnd(Shift, C, Op->getName()+".mask");
        InsertNewInstBefore(Mask, I);
        return new CastInst(Mask, I.getType());
      }
    } else {
      // We can handle signed (X << C1) >>s C2 if it's a sign extend.  In
      // this case, C1 == C2 and C1 is 8, 16, or 32.
      if (ShiftAmt1 == ShiftAmt2) {
        const Type *SExtType = 0;
        switch (ShiftAmt1) {
        case 8 : SExtType = Type::SByteTy; break;
        case 16: SExtType = Type::ShortTy; break;
        case 32: SExtType = Type::IntTy; break;
        }
        
        if (SExtType) {
          Instruction *NewTrunc = new CastInst(ShiftOp->getOperand(0),
                                               SExtType, "sext");
          InsertNewInstBefore(NewTrunc, I);
          return new CastInst(NewTrunc, I.getType());
        }
      }
    }
  }
  return 0;
}

enum CastType {
  Noop     = 0,
  Truncate = 1,
  Signext  = 2,
  Zeroext  = 3
};

/// getCastType - In the future, we will split the cast instruction into these
/// various types.  Until then, we have to do the analysis here.
static CastType getCastType(const Type *Src, const Type *Dest) {
  assert(Src->isIntegral() && Dest->isIntegral() &&
         "Only works on integral types!");
  unsigned SrcSize = Src->getPrimitiveSizeInBits();
  unsigned DestSize = Dest->getPrimitiveSizeInBits();

  if (SrcSize == DestSize) return Noop;
  if (SrcSize > DestSize)  return Truncate;
  if (Src->isSigned()) return Signext;
  return Zeroext;
}


// isEliminableCastOfCast - Return true if it is valid to eliminate the CI
// instruction.
//
static inline bool isEliminableCastOfCast(const Type *SrcTy, const Type *MidTy,
                                          const Type *DstTy, TargetData *TD) {

  // It is legal to eliminate the instruction if casting A->B->A if the sizes
  // are identical and the bits don't get reinterpreted (for example
  // int->float->int would not be allowed).
  if (SrcTy == DstTy && SrcTy->isLosslesslyConvertibleTo(MidTy))
    return true;

  // If we are casting between pointer and integer types, treat pointers as
  // integers of the appropriate size for the code below.
  if (isa<PointerType>(SrcTy)) SrcTy = TD->getIntPtrType();
  if (isa<PointerType>(MidTy)) MidTy = TD->getIntPtrType();
  if (isa<PointerType>(DstTy)) DstTy = TD->getIntPtrType();

  // Allow free casting and conversion of sizes as long as the sign doesn't
  // change...
  if (SrcTy->isIntegral() && MidTy->isIntegral() && DstTy->isIntegral()) {
    CastType FirstCast = getCastType(SrcTy, MidTy);
    CastType SecondCast = getCastType(MidTy, DstTy);

    // Capture the effect of these two casts.  If the result is a legal cast,
    // the CastType is stored here, otherwise a special code is used.
    static const unsigned CastResult[] = {
      // First cast is noop
      0, 1, 2, 3,
      // First cast is a truncate
      1, 1, 4, 4,         // trunc->extend is not safe to eliminate
      // First cast is a sign ext
      2, 5, 2, 4,         // signext->zeroext never ok
      // First cast is a zero ext
      3, 5, 3, 3,
    };

    unsigned Result = CastResult[FirstCast*4+SecondCast];
    switch (Result) {
    default: assert(0 && "Illegal table value!");
    case 0:
    case 1:
    case 2:
    case 3:
      // FIXME: in the future, when LLVM has explicit sign/zeroextends and
      // truncates, we could eliminate more casts.
      return (unsigned)getCastType(SrcTy, DstTy) == Result;
    case 4:
      return false;  // Not possible to eliminate this here.
    case 5:
      // Sign or zero extend followed by truncate is always ok if the result
      // is a truncate or noop.
      CastType ResultCast = getCastType(SrcTy, DstTy);
      if (ResultCast == Noop || ResultCast == Truncate)
        return true;
      // Otherwise we are still growing the value, we are only safe if the
      // result will match the sign/zeroextendness of the result.
      return ResultCast == FirstCast;
    }
  }
  return false;
}

static bool ValueRequiresCast(const Value *V, const Type *Ty, TargetData *TD) {
  if (V->getType() == Ty || isa<Constant>(V)) return false;
  if (const CastInst *CI = dyn_cast<CastInst>(V))
    if (isEliminableCastOfCast(CI->getOperand(0)->getType(), CI->getType(), Ty,
                               TD))
      return false;
  return true;
}

/// InsertOperandCastBefore - This inserts a cast of V to DestTy before the
/// InsertBefore instruction.  This is specialized a bit to avoid inserting
/// casts that are known to not do anything...
///
Value *InstCombiner::InsertOperandCastBefore(Value *V, const Type *DestTy,
                                             Instruction *InsertBefore) {
  if (V->getType() == DestTy) return V;
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(C, DestTy);

  CastInst *CI = new CastInst(V, DestTy, V->getName());
  InsertNewInstBefore(CI, *InsertBefore);
  return CI;
}

/// DecomposeSimpleLinearExpr - Analyze 'Val', seeing if it is a simple linear
/// expression.  If so, decompose it, returning some value X, such that Val is
/// X*Scale+Offset.
///
static Value *DecomposeSimpleLinearExpr(Value *Val, unsigned &Scale,
                                        unsigned &Offset) {
  assert(Val->getType() == Type::UIntTy && "Unexpected allocation size type!");
  if (ConstantUInt *CI = dyn_cast<ConstantUInt>(Val)) {
    Offset = CI->getValue();
    Scale  = 1;
    return ConstantUInt::get(Type::UIntTy, 0);
  } else if (Instruction *I = dyn_cast<Instruction>(Val)) {
    if (I->getNumOperands() == 2) {
      if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(I->getOperand(1))) {
        if (I->getOpcode() == Instruction::Shl) {
          // This is a value scaled by '1 << the shift amt'.
          Scale = 1U << CUI->getValue();
          Offset = 0;
          return I->getOperand(0);
        } else if (I->getOpcode() == Instruction::Mul) {
          // This value is scaled by 'CUI'.
          Scale = CUI->getValue();
          Offset = 0;
          return I->getOperand(0);
        } else if (I->getOpcode() == Instruction::Add) {
          // We have X+C.  Check to see if we really have (X*C2)+C1, where C1 is
          // divisible by C2.
          unsigned SubScale;
          Value *SubVal = DecomposeSimpleLinearExpr(I->getOperand(0), SubScale,
                                                    Offset);
          Offset += CUI->getValue();
          if (SubScale > 1 && (Offset % SubScale == 0)) {
            Scale = SubScale;
            return SubVal;
          }
        }
      }
    }
  }

  // Otherwise, we can't look past this.
  Scale = 1;
  Offset = 0;
  return Val;
}


/// PromoteCastOfAllocation - If we find a cast of an allocation instruction,
/// try to eliminate the cast by moving the type information into the alloc.
Instruction *InstCombiner::PromoteCastOfAllocation(CastInst &CI,
                                                   AllocationInst &AI) {
  const PointerType *PTy = dyn_cast<PointerType>(CI.getType());
  if (!PTy) return 0;   // Not casting the allocation to a pointer type.
  
  // Remove any uses of AI that are dead.
  assert(!CI.use_empty() && "Dead instructions should be removed earlier!");
  std::vector<Instruction*> DeadUsers;
  for (Value::use_iterator UI = AI.use_begin(), E = AI.use_end(); UI != E; ) {
    Instruction *User = cast<Instruction>(*UI++);
    if (isInstructionTriviallyDead(User)) {
      while (UI != E && *UI == User)
        ++UI; // If this instruction uses AI more than once, don't break UI.
      
      // Add operands to the worklist.
      AddUsesToWorkList(*User);
      ++NumDeadInst;
      DEBUG(std::cerr << "IC: DCE: " << *User);
      
      User->eraseFromParent();
      removeFromWorkList(User);
    }
  }
  
  // Get the type really allocated and the type casted to.
  const Type *AllocElTy = AI.getAllocatedType();
  const Type *CastElTy = PTy->getElementType();
  if (!AllocElTy->isSized() || !CastElTy->isSized()) return 0;

  unsigned AllocElTyAlign = TD->getTypeSize(AllocElTy);
  unsigned CastElTyAlign = TD->getTypeSize(CastElTy);
  if (CastElTyAlign < AllocElTyAlign) return 0;

  // If the allocation has multiple uses, only promote it if we are strictly
  // increasing the alignment of the resultant allocation.  If we keep it the
  // same, we open the door to infinite loops of various kinds.
  if (!AI.hasOneUse() && CastElTyAlign == AllocElTyAlign) return 0;

  uint64_t AllocElTySize = TD->getTypeSize(AllocElTy);
  uint64_t CastElTySize = TD->getTypeSize(CastElTy);
  if (CastElTySize == 0 || AllocElTySize == 0) return 0;

  // See if we can satisfy the modulus by pulling a scale out of the array
  // size argument.
  unsigned ArraySizeScale, ArrayOffset;
  Value *NumElements = // See if the array size is a decomposable linear expr.
    DecomposeSimpleLinearExpr(AI.getOperand(0), ArraySizeScale, ArrayOffset);
 
  // If we can now satisfy the modulus, by using a non-1 scale, we really can
  // do the xform.
  if ((AllocElTySize*ArraySizeScale) % CastElTySize != 0 ||
      (AllocElTySize*ArrayOffset   ) % CastElTySize != 0) return 0;

  unsigned Scale = (AllocElTySize*ArraySizeScale)/CastElTySize;
  Value *Amt = 0;
  if (Scale == 1) {
    Amt = NumElements;
  } else {
    Amt = ConstantUInt::get(Type::UIntTy, Scale);
    if (ConstantUInt *CI = dyn_cast<ConstantUInt>(NumElements))
      Amt = ConstantExpr::getMul(CI, cast<ConstantUInt>(Amt));
    else if (Scale != 1) {
      Instruction *Tmp = BinaryOperator::createMul(Amt, NumElements, "tmp");
      Amt = InsertNewInstBefore(Tmp, AI);
    }
  }
  
  if (unsigned Offset = (AllocElTySize*ArrayOffset)/CastElTySize) {
    Value *Off = ConstantUInt::get(Type::UIntTy, Offset);
    Instruction *Tmp = BinaryOperator::createAdd(Amt, Off, "tmp");
    Amt = InsertNewInstBefore(Tmp, AI);
  }
  
  std::string Name = AI.getName(); AI.setName("");
  AllocationInst *New;
  if (isa<MallocInst>(AI))
    New = new MallocInst(CastElTy, Amt, AI.getAlignment(), Name);
  else
    New = new AllocaInst(CastElTy, Amt, AI.getAlignment(), Name);
  InsertNewInstBefore(New, AI);
  
  // If the allocation has multiple uses, insert a cast and change all things
  // that used it to use the new cast.  This will also hack on CI, but it will
  // die soon.
  if (!AI.hasOneUse()) {
    AddUsesToWorkList(AI);
    CastInst *NewCast = new CastInst(New, AI.getType(), "tmpcast");
    InsertNewInstBefore(NewCast, AI);
    AI.replaceAllUsesWith(NewCast);
  }
  return ReplaceInstUsesWith(CI, New);
}


// CastInst simplification
//
Instruction *InstCombiner::visitCastInst(CastInst &CI) {
  Value *Src = CI.getOperand(0);

  // If the user is casting a value to the same type, eliminate this cast
  // instruction...
  if (CI.getType() == Src->getType())
    return ReplaceInstUsesWith(CI, Src);

  if (isa<UndefValue>(Src))   // cast undef -> undef
    return ReplaceInstUsesWith(CI, UndefValue::get(CI.getType()));

  // If casting the result of another cast instruction, try to eliminate this
  // one!
  //
  if (CastInst *CSrc = dyn_cast<CastInst>(Src)) {   // A->B->C cast
    Value *A = CSrc->getOperand(0);
    if (isEliminableCastOfCast(A->getType(), CSrc->getType(),
                               CI.getType(), TD)) {
      // This instruction now refers directly to the cast's src operand.  This
      // has a good chance of making CSrc dead.
      CI.setOperand(0, CSrc->getOperand(0));
      return &CI;
    }

    // If this is an A->B->A cast, and we are dealing with integral types, try
    // to convert this into a logical 'and' instruction.
    //
    if (A->getType()->isInteger() &&
        CI.getType()->isInteger() && CSrc->getType()->isInteger() &&
        CSrc->getType()->isUnsigned() &&   // B->A cast must zero extend
        CSrc->getType()->getPrimitiveSizeInBits() <
                    CI.getType()->getPrimitiveSizeInBits()&&
        A->getType()->getPrimitiveSizeInBits() ==
              CI.getType()->getPrimitiveSizeInBits()) {
      assert(CSrc->getType() != Type::ULongTy &&
             "Cannot have type bigger than ulong!");
      uint64_t AndValue = ~0ULL>>(64-CSrc->getType()->getPrimitiveSizeInBits());
      Constant *AndOp = ConstantUInt::get(A->getType()->getUnsignedVersion(),
                                          AndValue);
      AndOp = ConstantExpr::getCast(AndOp, A->getType());
      Instruction *And = BinaryOperator::createAnd(CSrc->getOperand(0), AndOp);
      if (And->getType() != CI.getType()) {
        And->setName(CSrc->getName()+".mask");
        InsertNewInstBefore(And, CI);
        And = new CastInst(And, CI.getType());
      }
      return And;
    }
  }

  // If this is a cast to bool, turn it into the appropriate setne instruction.
  if (CI.getType() == Type::BoolTy)
    return BinaryOperator::createSetNE(CI.getOperand(0),
                       Constant::getNullValue(CI.getOperand(0)->getType()));

  // If casting the result of a getelementptr instruction with no offset, turn
  // this into a cast of the original pointer!
  //
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Src)) {
    bool AllZeroOperands = true;
    for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEP->getOperand(i)) ||
          !cast<Constant>(GEP->getOperand(i))->isNullValue()) {
        AllZeroOperands = false;
        break;
      }
    if (AllZeroOperands) {
      CI.setOperand(0, GEP->getOperand(0));
      return &CI;
    }
  }

  // If we are casting a malloc or alloca to a pointer to a type of the same
  // size, rewrite the allocation instruction to allocate the "right" type.
  //
  if (AllocationInst *AI = dyn_cast<AllocationInst>(Src))
    if (Instruction *V = PromoteCastOfAllocation(CI, *AI))
      return V;

  if (SelectInst *SI = dyn_cast<SelectInst>(Src))
    if (Instruction *NV = FoldOpIntoSelect(CI, SI, this))
      return NV;
  if (isa<PHINode>(Src))
    if (Instruction *NV = FoldOpIntoPhi(CI))
      return NV;

  // If the source value is an instruction with only this use, we can attempt to
  // propagate the cast into the instruction.  Also, only handle integral types
  // for now.
  if (Instruction *SrcI = dyn_cast<Instruction>(Src))
    if (SrcI->hasOneUse() && Src->getType()->isIntegral() &&
        CI.getType()->isInteger()) {  // Don't mess with casts to bool here
      const Type *DestTy = CI.getType();
      unsigned SrcBitSize = Src->getType()->getPrimitiveSizeInBits();
      unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

      Value *Op0 = SrcI->getNumOperands() > 0 ? SrcI->getOperand(0) : 0;
      Value *Op1 = SrcI->getNumOperands() > 1 ? SrcI->getOperand(1) : 0;

      switch (SrcI->getOpcode()) {
      case Instruction::Add:
      case Instruction::Mul:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
        // If we are discarding information, or just changing the sign, rewrite.
        if (DestBitSize <= SrcBitSize && DestBitSize != 1) {
          // Don't insert two casts if they cannot be eliminated.  We allow two
          // casts to be inserted if the sizes are the same.  This could only be
          // converting signedness, which is a noop.
          if (DestBitSize == SrcBitSize || !ValueRequiresCast(Op1, DestTy,TD) ||
              !ValueRequiresCast(Op0, DestTy, TD)) {
            Value *Op0c = InsertOperandCastBefore(Op0, DestTy, SrcI);
            Value *Op1c = InsertOperandCastBefore(Op1, DestTy, SrcI);
            return BinaryOperator::create(cast<BinaryOperator>(SrcI)
                             ->getOpcode(), Op0c, Op1c);
          }
        }

        // cast (xor bool X, true) to int  --> xor (cast bool X to int), 1
        if (SrcBitSize == 1 && SrcI->getOpcode() == Instruction::Xor &&
            Op1 == ConstantBool::True &&
            (!Op0->hasOneUse() || !isa<SetCondInst>(Op0))) {
          Value *New = InsertOperandCastBefore(Op0, DestTy, &CI);
          return BinaryOperator::createXor(New,
                                           ConstantInt::get(CI.getType(), 1));
        }
        break;
      case Instruction::Shl:
        // Allow changing the sign of the source operand.  Do not allow changing
        // the size of the shift, UNLESS the shift amount is a constant.  We
        // mush not change variable sized shifts to a smaller size, because it
        // is undefined to shift more bits out than exist in the value.
        if (DestBitSize == SrcBitSize ||
            (DestBitSize < SrcBitSize && isa<Constant>(Op1))) {
          Value *Op0c = InsertOperandCastBefore(Op0, DestTy, SrcI);
          return new ShiftInst(Instruction::Shl, Op0c, Op1);
        }
        break;
      case Instruction::Shr:
        // If this is a signed shr, and if all bits shifted in are about to be
        // truncated off, turn it into an unsigned shr to allow greater
        // simplifications.
        if (DestBitSize < SrcBitSize && Src->getType()->isSigned() &&
            isa<ConstantInt>(Op1)) {
          unsigned ShiftAmt = cast<ConstantUInt>(Op1)->getValue();
          if (SrcBitSize > ShiftAmt && SrcBitSize-ShiftAmt >= DestBitSize) {
            // Convert to unsigned.
            Value *N1 = InsertOperandCastBefore(Op0,
                                     Op0->getType()->getUnsignedVersion(), &CI);
            // Insert the new shift, which is now unsigned.
            N1 = InsertNewInstBefore(new ShiftInst(Instruction::Shr, N1,
                                                   Op1, Src->getName()), CI);
            return new CastInst(N1, CI.getType());
          }
        }
        break;

      case Instruction::SetNE:
        if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
          if (Op1C->getRawValue() == 0) {
            // If the input only has the low bit set, simplify directly.
            Constant *Not1 =
              ConstantExpr::getNot(ConstantInt::get(Op0->getType(), 1));
            // cast (X != 0) to int  --> X if X&~1 == 0
            if (MaskedValueIsZero(Op0, cast<ConstantIntegral>(Not1))) {
              if (CI.getType() == Op0->getType())
                return ReplaceInstUsesWith(CI, Op0);
              else
                return new CastInst(Op0, CI.getType());
            }

            // If the input is an and with a single bit, shift then simplify.
            ConstantInt *AndRHS;
            if (match(Op0, m_And(m_Value(), m_ConstantInt(AndRHS))))
              if (AndRHS->getRawValue() &&
                  (AndRHS->getRawValue() & (AndRHS->getRawValue()-1)) == 0) {
                unsigned ShiftAmt = Log2_64(AndRHS->getRawValue());
                // Perform an unsigned shr by shiftamt.  Convert input to
                // unsigned if it is signed.
                Value *In = Op0;
                if (In->getType()->isSigned())
                  In = InsertNewInstBefore(new CastInst(In,
                        In->getType()->getUnsignedVersion(), In->getName()),CI);
                // Insert the shift to put the result in the low bit.
                In = InsertNewInstBefore(new ShiftInst(Instruction::Shr, In,
                                      ConstantInt::get(Type::UByteTy, ShiftAmt),
                                                   In->getName()+".lobit"), CI);
                if (CI.getType() == In->getType())
                  return ReplaceInstUsesWith(CI, In);
                else
                  return new CastInst(In, CI.getType());
              }
          }
        }
        break;
      case Instruction::SetEQ:
        // We if we are just checking for a seteq of a single bit and casting it
        // to an integer.  If so, shift the bit to the appropriate place then
        // cast to integer to avoid the comparison.
        if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
          // Is Op1C a power of two or zero?
          if ((Op1C->getRawValue() & Op1C->getRawValue()-1) == 0) {
            // cast (X == 1) to int -> X iff X has only the low bit set.
            if (Op1C->getRawValue() == 1) {
              Constant *Not1 =
                ConstantExpr::getNot(ConstantInt::get(Op0->getType(), 1));
              if (MaskedValueIsZero(Op0, cast<ConstantIntegral>(Not1))) {
                if (CI.getType() == Op0->getType())
                  return ReplaceInstUsesWith(CI, Op0);
                else
                  return new CastInst(Op0, CI.getType());
              }
            }
          }
        }
        break;
      }
    }
      
  return 0;
}

/// GetSelectFoldableOperands - We want to turn code that looks like this:
///   %C = or %A, %B
///   %D = select %cond, %C, %A
/// into:
///   %C = select %cond, %B, 0
///   %D = or %A, %C
///
/// Assuming that the specified instruction is an operand to the select, return
/// a bitmask indicating which operands of this instruction are foldable if they
/// equal the other incoming value of the select.
///
static unsigned GetSelectFoldableOperands(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return 3;              // Can fold through either operand.
  case Instruction::Sub:   // Can only fold on the amount subtracted.
  case Instruction::Shl:   // Can only fold on the shift amount.
  case Instruction::Shr:
    return 1;
  default:
    return 0;              // Cannot fold
  }
}

/// GetSelectFoldableConstant - For the same transformation as the previous
/// function, return the identity constant that goes into the select.
static Constant *GetSelectFoldableConstant(Instruction *I) {
  switch (I->getOpcode()) {
  default: assert(0 && "This cannot happen!"); abort();
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Or:
  case Instruction::Xor:
    return Constant::getNullValue(I->getType());
  case Instruction::Shl:
  case Instruction::Shr:
    return Constant::getNullValue(Type::UByteTy);
  case Instruction::And:
    return ConstantInt::getAllOnesValue(I->getType());
  case Instruction::Mul:
    return ConstantInt::get(I->getType(), 1);
  }
}

/// FoldSelectOpOp - Here we have (select c, TI, FI), and we know that TI and FI
/// have the same opcode and only one use each.  Try to simplify this.
Instruction *InstCombiner::FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                          Instruction *FI) {
  if (TI->getNumOperands() == 1) {
    // If this is a non-volatile load or a cast from the same type,
    // merge.
    if (TI->getOpcode() == Instruction::Cast) {
      if (TI->getOperand(0)->getType() != FI->getOperand(0)->getType())
        return 0;
    } else {
      return 0;  // unknown unary op.
    }

    // Fold this by inserting a select from the input values.
    SelectInst *NewSI = new SelectInst(SI.getCondition(), TI->getOperand(0),
                                       FI->getOperand(0), SI.getName()+".v");
    InsertNewInstBefore(NewSI, SI);
    return new CastInst(NewSI, TI->getType());
  }

  // Only handle binary operators here.
  if (!isa<ShiftInst>(TI) && !isa<BinaryOperator>(TI))
    return 0;

  // Figure out if the operations have any operands in common.
  Value *MatchOp, *OtherOpT, *OtherOpF;
  bool MatchIsOpZero;
  if (TI->getOperand(0) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = false;
  } else if (!TI->isCommutative()) {
    return 0;
  } else if (TI->getOperand(0) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else {
    return 0;
  }

  // If we reach here, they do have operations in common.
  SelectInst *NewSI = new SelectInst(SI.getCondition(), OtherOpT,
                                     OtherOpF, SI.getName()+".v");
  InsertNewInstBefore(NewSI, SI);

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TI)) {
    if (MatchIsOpZero)
      return BinaryOperator::create(BO->getOpcode(), MatchOp, NewSI);
    else
      return BinaryOperator::create(BO->getOpcode(), NewSI, MatchOp);
  } else {
    if (MatchIsOpZero)
      return new ShiftInst(cast<ShiftInst>(TI)->getOpcode(), MatchOp, NewSI);
    else
      return new ShiftInst(cast<ShiftInst>(TI)->getOpcode(), NewSI, MatchOp);
  }
}

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // select true, X, Y  -> X
  // select false, X, Y -> Y
  if (ConstantBool *C = dyn_cast<ConstantBool>(CondVal))
    if (C == ConstantBool::True)
      return ReplaceInstUsesWith(SI, TrueVal);
    else {
      assert(C == ConstantBool::False);
      return ReplaceInstUsesWith(SI, FalseVal);
    }

  // select C, X, X -> X
  if (TrueVal == FalseVal)
    return ReplaceInstUsesWith(SI, TrueVal);

  if (isa<UndefValue>(TrueVal))   // select C, undef, X -> X
    return ReplaceInstUsesWith(SI, FalseVal);
  if (isa<UndefValue>(FalseVal))   // select C, X, undef -> X
    return ReplaceInstUsesWith(SI, TrueVal);
  if (isa<UndefValue>(CondVal)) {  // select undef, X, Y -> X or Y
    if (isa<Constant>(TrueVal))
      return ReplaceInstUsesWith(SI, TrueVal);
    else
      return ReplaceInstUsesWith(SI, FalseVal);
  }

  if (SI.getType() == Type::BoolTy)
    if (ConstantBool *C = dyn_cast<ConstantBool>(TrueVal)) {
      if (C == ConstantBool::True) {
        // Change: A = select B, true, C --> A = or B, C
        return BinaryOperator::createOr(CondVal, FalseVal);
      } else {
        // Change: A = select B, false, C --> A = and !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::createNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::createAnd(NotCond, FalseVal);
      }
    } else if (ConstantBool *C = dyn_cast<ConstantBool>(FalseVal)) {
      if (C == ConstantBool::False) {
        // Change: A = select B, C, false --> A = and B, C
        return BinaryOperator::createAnd(CondVal, TrueVal);
      } else {
        // Change: A = select B, C, true --> A = or !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::createNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::createOr(NotCond, TrueVal);
      }
    }

  // Selecting between two integer constants?
  if (ConstantInt *TrueValC = dyn_cast<ConstantInt>(TrueVal))
    if (ConstantInt *FalseValC = dyn_cast<ConstantInt>(FalseVal)) {
      // select C, 1, 0 -> cast C to int
      if (FalseValC->isNullValue() && TrueValC->getRawValue() == 1) {
        return new CastInst(CondVal, SI.getType());
      } else if (TrueValC->isNullValue() && FalseValC->getRawValue() == 1) {
        // select C, 0, 1 -> cast !C to int
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::createNot(CondVal,
                                               "not."+CondVal->getName()), SI);
        return new CastInst(NotCond, SI.getType());
      }

      // If one of the constants is zero (we know they can't both be) and we
      // have a setcc instruction with zero, and we have an 'and' with the
      // non-constant value, eliminate this whole mess.  This corresponds to
      // cases like this: ((X & 27) ? 27 : 0)
      if (TrueValC->isNullValue() || FalseValC->isNullValue())
        if (Instruction *IC = dyn_cast<Instruction>(SI.getCondition()))
          if ((IC->getOpcode() == Instruction::SetEQ ||
               IC->getOpcode() == Instruction::SetNE) &&
              isa<ConstantInt>(IC->getOperand(1)) &&
              cast<Constant>(IC->getOperand(1))->isNullValue())
            if (Instruction *ICA = dyn_cast<Instruction>(IC->getOperand(0)))
              if (ICA->getOpcode() == Instruction::And &&
                  isa<ConstantInt>(ICA->getOperand(1)) &&
                  (ICA->getOperand(1) == TrueValC ||
                   ICA->getOperand(1) == FalseValC) &&
                  isOneBitSet(cast<ConstantInt>(ICA->getOperand(1)))) {
                // Okay, now we know that everything is set up, we just don't
                // know whether we have a setne or seteq and whether the true or
                // false val is the zero.
                bool ShouldNotVal = !TrueValC->isNullValue();
                ShouldNotVal ^= IC->getOpcode() == Instruction::SetNE;
                Value *V = ICA;
                if (ShouldNotVal)
                  V = InsertNewInstBefore(BinaryOperator::create(
                                  Instruction::Xor, V, ICA->getOperand(1)), SI);
                return ReplaceInstUsesWith(SI, V);
              }
    }

  // See if we are selecting two values based on a comparison of the two values.
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(CondVal)) {
    if (SCI->getOperand(0) == TrueVal && SCI->getOperand(1) == FalseVal) {
      // Transform (X == Y) ? X : Y  -> Y
      if (SCI->getOpcode() == Instruction::SetEQ)
        return ReplaceInstUsesWith(SI, FalseVal);
      // Transform (X != Y) ? X : Y  -> X
      if (SCI->getOpcode() == Instruction::SetNE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX/ABS/etc.

    } else if (SCI->getOperand(0) == FalseVal && SCI->getOperand(1) == TrueVal){
      // Transform (X == Y) ? Y : X  -> X
      if (SCI->getOpcode() == Instruction::SetEQ)
        return ReplaceInstUsesWith(SI, FalseVal);
      // Transform (X != Y) ? Y : X  -> Y
      if (SCI->getOpcode() == Instruction::SetNE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX/ABS/etc.
    }
  }

  if (Instruction *TI = dyn_cast<Instruction>(TrueVal))
    if (Instruction *FI = dyn_cast<Instruction>(FalseVal))
      if (TI->hasOneUse() && FI->hasOneUse()) {
        bool isInverse = false;
        Instruction *AddOp = 0, *SubOp = 0;

        // Turn (select C, (op X, Y), (op X, Z)) -> (op X, (select C, Y, Z))
        if (TI->getOpcode() == FI->getOpcode())
          if (Instruction *IV = FoldSelectOpOp(SI, TI, FI))
            return IV;

        // Turn select C, (X+Y), (X-Y) --> (X+(select C, Y, (-Y))).  This is
        // even legal for FP.
        if (TI->getOpcode() == Instruction::Sub &&
            FI->getOpcode() == Instruction::Add) {
          AddOp = FI; SubOp = TI;
        } else if (FI->getOpcode() == Instruction::Sub &&
                   TI->getOpcode() == Instruction::Add) {
          AddOp = TI; SubOp = FI;
        }

        if (AddOp) {
          Value *OtherAddOp = 0;
          if (SubOp->getOperand(0) == AddOp->getOperand(0)) {
            OtherAddOp = AddOp->getOperand(1);
          } else if (SubOp->getOperand(0) == AddOp->getOperand(1)) {
            OtherAddOp = AddOp->getOperand(0);
          }

          if (OtherAddOp) {
            // So at this point we know we have:
            //        select C, (add X, Y), (sub X, ?)
            // We can do the transform profitably if either 'Y' = '?' or '?' is
            // a constant.
            if (SubOp->getOperand(1) == AddOp ||
                isa<Constant>(SubOp->getOperand(1))) {
              Value *NegVal;
              if (Constant *C = dyn_cast<Constant>(SubOp->getOperand(1))) {
                NegVal = ConstantExpr::getNeg(C);
              } else {
                NegVal = InsertNewInstBefore(
                           BinaryOperator::createNeg(SubOp->getOperand(1)), SI);
              }

              Value *NewTrueOp = OtherAddOp;
              Value *NewFalseOp = NegVal;
              if (AddOp != TI)
                std::swap(NewTrueOp, NewFalseOp);
              Instruction *NewSel =
                new SelectInst(CondVal, NewTrueOp,NewFalseOp,SI.getName()+".p");

              NewSel = InsertNewInstBefore(NewSel, SI);
              return BinaryOperator::createAdd(SubOp->getOperand(0), NewSel);
            }
          }
        }
      }

  // See if we can fold the select into one of our operands.
  if (SI.getType()->isInteger()) {
    // See the comment above GetSelectFoldableOperands for a description of the
    // transformation we are doing here.
    if (Instruction *TVI = dyn_cast<Instruction>(TrueVal))
      if (TVI->hasOneUse() && TVI->getNumOperands() == 2 &&
          !isa<Constant>(FalseVal))
        if (unsigned SFO = GetSelectFoldableOperands(TVI)) {
          unsigned OpToFold = 0;
          if ((SFO & 1) && FalseVal == TVI->getOperand(0)) {
            OpToFold = 1;
          } else  if ((SFO & 2) && FalseVal == TVI->getOperand(1)) {
            OpToFold = 2;
          }

          if (OpToFold) {
            Constant *C = GetSelectFoldableConstant(TVI);
            std::string Name = TVI->getName(); TVI->setName("");
            Instruction *NewSel =
              new SelectInst(SI.getCondition(), TVI->getOperand(2-OpToFold), C,
                             Name);
            InsertNewInstBefore(NewSel, SI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TVI))
              return BinaryOperator::create(BO->getOpcode(), FalseVal, NewSel);
            else if (ShiftInst *SI = dyn_cast<ShiftInst>(TVI))
              return new ShiftInst(SI->getOpcode(), FalseVal, NewSel);
            else {
              assert(0 && "Unknown instruction!!");
            }
          }
        }

    if (Instruction *FVI = dyn_cast<Instruction>(FalseVal))
      if (FVI->hasOneUse() && FVI->getNumOperands() == 2 &&
          !isa<Constant>(TrueVal))
        if (unsigned SFO = GetSelectFoldableOperands(FVI)) {
          unsigned OpToFold = 0;
          if ((SFO & 1) && TrueVal == FVI->getOperand(0)) {
            OpToFold = 1;
          } else  if ((SFO & 2) && TrueVal == FVI->getOperand(1)) {
            OpToFold = 2;
          }

          if (OpToFold) {
            Constant *C = GetSelectFoldableConstant(FVI);
            std::string Name = FVI->getName(); FVI->setName("");
            Instruction *NewSel =
              new SelectInst(SI.getCondition(), C, FVI->getOperand(2-OpToFold),
                             Name);
            InsertNewInstBefore(NewSel, SI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FVI))
              return BinaryOperator::create(BO->getOpcode(), TrueVal, NewSel);
            else if (ShiftInst *SI = dyn_cast<ShiftInst>(FVI))
              return new ShiftInst(SI->getOpcode(), TrueVal, NewSel);
            else {
              assert(0 && "Unknown instruction!!");
            }
          }
        }
  }

  if (BinaryOperator::isNot(CondVal)) {
    SI.setOperand(0, BinaryOperator::getNotArgument(CondVal));
    SI.setOperand(1, FalseVal);
    SI.setOperand(2, TrueVal);
    return &SI;
  }

  return 0;
}


/// visitCallInst - CallInst simplification.  This mostly only handles folding 
/// of intrinsic instructions.  For normal calls, it allows visitCallSite to do
/// the heavy lifting.
///
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CI);
  if (!II) return visitCallSite(&CI);
  
  // Intrinsics cannot occur in an invoke, so handle them here instead of in
  // visitCallSite.
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(II)) {
    bool Changed = false;

    // memmove/cpy/set of zero bytes is a noop.
    if (Constant *NumBytes = dyn_cast<Constant>(MI->getLength())) {
      if (NumBytes->isNullValue()) return EraseInstFromFunction(CI);

      // FIXME: Increase alignment here.

      if (ConstantInt *CI = dyn_cast<ConstantInt>(NumBytes))
        if (CI->getRawValue() == 1) {
          // Replace the instruction with just byte operations.  We would
          // transform other cases to loads/stores, but we don't know if
          // alignment is sufficient.
        }
    }

    // If we have a memmove and the source operation is a constant global,
    // then the source and dest pointers can't alias, so we can change this
    // into a call to memcpy.
    if (MemMoveInst *MMI = dyn_cast<MemMoveInst>(II))
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getParent()->getParent()->getParent();
          Function *MemCpy = M->getOrInsertFunction("llvm.memcpy",
                                     CI.getCalledFunction()->getFunctionType());
          CI.setOperand(0, MemCpy);
          Changed = true;
        }

    if (Changed) return II;
  } else if (DbgStopPointInst *SPI = dyn_cast<DbgStopPointInst>(II)) {
    // If this stoppoint is at the same source location as the previous
    // stoppoint in the chain, it is not needed.
    if (DbgStopPointInst *PrevSPI =
        dyn_cast<DbgStopPointInst>(SPI->getChain()))
      if (SPI->getLineNo() == PrevSPI->getLineNo() &&
          SPI->getColNo() == PrevSPI->getColNo()) {
        SPI->replaceAllUsesWith(PrevSPI);
        return EraseInstFromFunction(CI);
      }
  } else {
    switch (II->getIntrinsicID()) {
    default: break;
    case Intrinsic::stackrestore: {
      // If the save is right next to the restore, remove the restore.  This can
      // happen when variable allocas are DCE'd.
      if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getOperand(1))) {
        if (SS->getIntrinsicID() == Intrinsic::stacksave) {
          BasicBlock::iterator BI = SS;
          if (&*++BI == II)
            return EraseInstFromFunction(CI);
        }
      }
      
      // If the stack restore is in a return/unwind block and if there are no
      // allocas or calls between the restore and the return, nuke the restore.
      TerminatorInst *TI = II->getParent()->getTerminator();
      if (isa<ReturnInst>(TI) || isa<UnwindInst>(TI)) {
        BasicBlock::iterator BI = II;
        bool CannotRemove = false;
        for (++BI; &*BI != TI; ++BI) {
          if (isa<AllocaInst>(BI) ||
              (isa<CallInst>(BI) && !isa<IntrinsicInst>(BI))) {
            CannotRemove = true;
            break;
          }
        }
        if (!CannotRemove)
          return EraseInstFromFunction(CI);
      }
      break;
    }
    }
  }

  return visitCallSite(II);
}

// InvokeInst simplification
//
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallSite(&II);
}

// visitCallSite - Improvements for call and invoke instructions.
//
Instruction *InstCombiner::visitCallSite(CallSite CS) {
  bool Changed = false;

  // If the callee is a constexpr cast of a function, attempt to move the cast
  // to the arguments of the call/invoke.
  if (transformConstExprCastCall(CS)) return 0;

  Value *Callee = CS.getCalledValue();

  if (Function *CalleeF = dyn_cast<Function>(Callee))
    if (CalleeF->getCallingConv() != CS.getCallingConv()) {
      Instruction *OldCall = CS.getInstruction();
      // If the call and callee calling conventions don't match, this call must
      // be unreachable, as the call is undefined.
      new StoreInst(ConstantBool::True,
                    UndefValue::get(PointerType::get(Type::BoolTy)), OldCall);
      if (!OldCall->use_empty())
        OldCall->replaceAllUsesWith(UndefValue::get(OldCall->getType()));
      if (isa<CallInst>(OldCall))   // Not worth removing an invoke here.
        return EraseInstFromFunction(*OldCall);
      return 0;
    }

  if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
    // This instruction is not reachable, just remove it.  We insert a store to
    // undef so that we know that this code is not reachable, despite the fact
    // that we can't modify the CFG here.
    new StoreInst(ConstantBool::True,
                  UndefValue::get(PointerType::get(Type::BoolTy)),
                  CS.getInstruction());

    if (!CS.getInstruction()->use_empty())
      CS.getInstruction()->
        replaceAllUsesWith(UndefValue::get(CS.getInstruction()->getType()));

    if (InvokeInst *II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      // Don't break the CFG, insert a dummy cond branch.
      new BranchInst(II->getNormalDest(), II->getUnwindDest(),
                     ConstantBool::True, II);
    }
    return EraseInstFromFunction(*CS.getInstruction());
  }

  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (CallSite::arg_iterator I = CS.arg_begin()+FTy->getNumParams(),
           E = CS.arg_end(); I != E; ++I)
      if (CastInst *CI = dyn_cast<CastInst>(*I)) {
        // If this cast does not effect the value passed through the varargs
        // area, we can eliminate the use of the cast.
        Value *Op = CI->getOperand(0);
        if (CI->getType()->isLosslesslyConvertibleTo(Op->getType())) {
          *I = Op;
          Changed = true;
        }
      }
  }

  return Changed ? CS.getInstruction() : 0;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  if (!isa<ConstantExpr>(CS.getCalledValue())) return false;
  ConstantExpr *CE = cast<ConstantExpr>(CS.getCalledValue());
  if (CE->getOpcode() != Instruction::Cast || !isa<Function>(CE->getOperand(0)))
    return false;
  Function *Callee = cast<Function>(CE->getOperand(0));
  Instruction *Caller = CS.getInstruction();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  const FunctionType *FT = Callee->getFunctionType();
  const Type *OldRetTy = Caller->getType();

  // Check to see if we are changing the return type...
  if (OldRetTy != FT->getReturnType()) {
    if (Callee->isExternal() &&
        !OldRetTy->isLosslesslyConvertibleTo(FT->getReturnType()) &&
        !Caller->use_empty())
      return false;   // Cannot transform this return value...

    // If the callsite is an invoke instruction, and the return value is used by
    // a PHI node in a successor, we cannot change the return type of the call
    // because there is no place to put the cast instruction (without breaking
    // the critical edge).  Bail out in this case.
    if (!Caller->use_empty())
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller))
        for (Value::use_iterator UI = II->use_begin(), E = II->use_end();
             UI != E; ++UI)
          if (PHINode *PN = dyn_cast<PHINode>(*UI))
            if (PN->getParent() == II->getNormalDest() ||
                PN->getParent() == II->getUnwindDest())
              return false;
  }

  unsigned NumActualArgs = unsigned(CS.arg_end()-CS.arg_begin());
  unsigned NumCommonArgs = std::min(FT->getNumParams(), NumActualArgs);

  CallSite::arg_iterator AI = CS.arg_begin();
  for (unsigned i = 0, e = NumCommonArgs; i != e; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    bool isConvertible = (*AI)->getType()->isLosslesslyConvertibleTo(ParamTy);
    if (Callee->isExternal() && !isConvertible) return false;
  }

  if (FT->getNumParams() < NumActualArgs && !FT->isVarArg() &&
      Callee->isExternal())
    return false;   // Do not delete arguments unless we have a function body...

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary...
  std::vector<Value*> Args;
  Args.reserve(NumActualArgs);

  AI = CS.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    if ((*AI)->getType() == ParamTy) {
      Args.push_back(*AI);
    } else {
      Args.push_back(InsertNewInstBefore(new CastInst(*AI, ParamTy, "tmp"),
                                         *Caller));
    }
  }

  // If the function takes more arguments than the call was taking, add them
  // now...
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i)
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));

  // If we are removing arguments to the function, emit an obnoxious warning...
  if (FT->getNumParams() < NumActualArgs)
    if (!FT->isVarArg()) {
      std::cerr << "WARNING: While resolving call to function '"
                << Callee->getName() << "' arguments were dropped!\n";
    } else {
      // Add all of the arguments in their promoted form to the arg list...
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        const Type *PTy = getPromotedType((*AI)->getType());
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction *Cast = new CastInst(*AI, PTy, "tmp");
          InsertNewInstBefore(Cast, *Caller);
          Args.push_back(Cast);
        } else {
          Args.push_back(*AI);
        }
      }
    }

  if (FT->getReturnType() == Type::VoidTy)
    Caller->setName("");   // Void type should not have a name...

  Instruction *NC;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NC = new InvokeInst(Callee, II->getNormalDest(), II->getUnwindDest(),
                        Args, Caller->getName(), Caller);
    cast<InvokeInst>(II)->setCallingConv(II->getCallingConv());
  } else {
    NC = new CallInst(Callee, Args, Caller->getName(), Caller);
    if (cast<CallInst>(Caller)->isTailCall())
      cast<CallInst>(NC)->setTailCall();
   cast<CallInst>(NC)->setCallingConv(cast<CallInst>(Caller)->getCallingConv());
  }

  // Insert a cast of the return type as necessary...
  Value *NV = NC;
  if (Caller->getType() != NV->getType() && !Caller->use_empty()) {
    if (NV->getType() != Type::VoidTy) {
      NV = NC = new CastInst(NC, Caller->getType(), "tmp");

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->begin();
        while (isa<PHINode>(I)) ++I;
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call instr
        InsertNewInstBefore(NC, *Caller);
      }
      AddUsersToWorkList(*Caller);
    } else {
      NV = UndefValue::get(Caller->getType());
    }
  }

  if (Caller->getType() != Type::VoidTy && !Caller->use_empty())
    Caller->replaceAllUsesWith(NV);
  Caller->getParent()->getInstList().erase(Caller);
  removeFromWorkList(Caller);
  return true;
}


// FoldPHIArgOpIntoPHI - If all operands to a PHI node are the same "unary"
// operator and they all are only used by the PHI, PHI together their
// inputs, and do the operation once, to the result of the PHI.
Instruction *InstCombiner::FoldPHIArgOpIntoPHI(PHINode &PN) {
  Instruction *FirstInst = cast<Instruction>(PN.getIncomingValue(0));

  // Scan the instruction, looking for input operations that can be folded away.
  // If all input operands to the phi are the same instruction (e.g. a cast from
  // the same type or "+42") we can pull the operation through the PHI, reducing
  // code size and simplifying code.
  Constant *ConstantOp = 0;
  const Type *CastSrcTy = 0;
  if (isa<CastInst>(FirstInst)) {
    CastSrcTy = FirstInst->getOperand(0)->getType();
  } else if (isa<BinaryOperator>(FirstInst) || isa<ShiftInst>(FirstInst)) {
    // Can fold binop or shift if the RHS is a constant.
    ConstantOp = dyn_cast<Constant>(FirstInst->getOperand(1));
    if (ConstantOp == 0) return 0;
  } else {
    return 0;  // Cannot fold this operation.
  }

  // Check to see if all arguments are the same operation.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    if (!isa<Instruction>(PN.getIncomingValue(i))) return 0;
    Instruction *I = cast<Instruction>(PN.getIncomingValue(i));
    if (!I->hasOneUse() || I->getOpcode() != FirstInst->getOpcode())
      return 0;
    if (CastSrcTy) {
      if (I->getOperand(0)->getType() != CastSrcTy)
        return 0;  // Cast operation must match.
    } else if (I->getOperand(1) != ConstantOp) {
      return 0;
    }
  }

  // Okay, they are all the same operation.  Create a new PHI node of the
  // correct type, and PHI together all of the LHS's of the instructions.
  PHINode *NewPN = new PHINode(FirstInst->getOperand(0)->getType(),
                               PN.getName()+".in");
  NewPN->reserveOperandSpace(PN.getNumOperands()/2);

  Value *InVal = FirstInst->getOperand(0);
  NewPN->addIncoming(InVal, PN.getIncomingBlock(0));

  // Add all operands to the new PHI.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    Value *NewInVal = cast<Instruction>(PN.getIncomingValue(i))->getOperand(0);
    if (NewInVal != InVal)
      InVal = 0;
    NewPN->addIncoming(NewInVal, PN.getIncomingBlock(i));
  }

  Value *PhiVal;
  if (InVal) {
    // The new PHI unions all of the same values together.  This is really
    // common, so we handle it intelligently here for compile-time speed.
    PhiVal = InVal;
    delete NewPN;
  } else {
    InsertNewInstBefore(NewPN, PN);
    PhiVal = NewPN;
  }

  // Insert and return the new operation.
  if (isa<CastInst>(FirstInst))
    return new CastInst(PhiVal, PN.getType());
  else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(FirstInst))
    return BinaryOperator::create(BinOp->getOpcode(), PhiVal, ConstantOp);
  else
    return new ShiftInst(cast<ShiftInst>(FirstInst)->getOpcode(),
                         PhiVal, ConstantOp);
}

/// DeadPHICycle - Return true if this PHI node is only used by a PHI node cycle
/// that is dead.
static bool DeadPHICycle(PHINode *PN, std::set<PHINode*> &PotentiallyDeadPHIs) {
  if (PN->use_empty()) return true;
  if (!PN->hasOneUse()) return false;

  // Remember this node, and if we find the cycle, return.
  if (!PotentiallyDeadPHIs.insert(PN).second)
    return true;

  if (PHINode *PU = dyn_cast<PHINode>(PN->use_back()))
    return DeadPHICycle(PU, PotentiallyDeadPHIs);

  return false;
}

// PHINode simplification
//
Instruction *InstCombiner::visitPHINode(PHINode &PN) {
  if (Value *V = PN.hasConstantValue())
    return ReplaceInstUsesWith(PN, V);

  // If the only user of this instruction is a cast instruction, and all of the
  // incoming values are constants, change this PHI to merge together the casted
  // constants.
  if (PN.hasOneUse())
    if (CastInst *CI = dyn_cast<CastInst>(PN.use_back()))
      if (CI->getType() != PN.getType()) {  // noop casts will be folded
        bool AllConstant = true;
        for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
          if (!isa<Constant>(PN.getIncomingValue(i))) {
            AllConstant = false;
            break;
          }
        if (AllConstant) {
          // Make a new PHI with all casted values.
          PHINode *New = new PHINode(CI->getType(), PN.getName(), &PN);
          for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
            Constant *OldArg = cast<Constant>(PN.getIncomingValue(i));
            New->addIncoming(ConstantExpr::getCast(OldArg, New->getType()),
                             PN.getIncomingBlock(i));
          }

          // Update the cast instruction.
          CI->setOperand(0, New);
          WorkList.push_back(CI);    // revisit the cast instruction to fold.
          WorkList.push_back(New);   // Make sure to revisit the new Phi
          return &PN;                // PN is now dead!
        }
      }

  // If all PHI operands are the same operation, pull them through the PHI,
  // reducing code size.
  if (isa<Instruction>(PN.getIncomingValue(0)) &&
      PN.getIncomingValue(0)->hasOneUse())
    if (Instruction *Result = FoldPHIArgOpIntoPHI(PN))
      return Result;

  // If this is a trivial cycle in the PHI node graph, remove it.  Basically, if
  // this PHI only has a single use (a PHI), and if that PHI only has one use (a
  // PHI)... break the cycle.
  if (PN.hasOneUse())
    if (PHINode *PU = dyn_cast<PHINode>(PN.use_back())) {
      std::set<PHINode*> PotentiallyDeadPHIs;
      PotentiallyDeadPHIs.insert(&PN);
      if (DeadPHICycle(PU, PotentiallyDeadPHIs))
        return ReplaceInstUsesWith(PN, UndefValue::get(PN.getType()));
    }

  return 0;
}

static Value *InsertSignExtendToPtrTy(Value *V, const Type *DTy,
                                      Instruction *InsertPoint,
                                      InstCombiner *IC) {
  unsigned PS = IC->getTargetData().getPointerSize();
  const Type *VTy = V->getType();
  if (!VTy->isSigned() && VTy->getPrimitiveSize() < PS)
    // We must insert a cast to ensure we sign-extend.
    V = IC->InsertNewInstBefore(new CastInst(V, VTy->getSignedVersion(),
                                             V->getName()), *InsertPoint);
  return IC->InsertNewInstBefore(new CastInst(V, DTy, V->getName()),
                                 *InsertPoint);
}


Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getOperand(0);
  // Is it 'getelementptr %P, long 0'  or 'getelementptr %P'
  // If so, eliminate the noop.
  if (GEP.getNumOperands() == 1)
    return ReplaceInstUsesWith(GEP, PtrOp);

  if (isa<UndefValue>(GEP.getOperand(0)))
    return ReplaceInstUsesWith(GEP, UndefValue::get(GEP.getType()));

  bool HasZeroPointerIndex = false;
  if (Constant *C = dyn_cast<Constant>(GEP.getOperand(1)))
    HasZeroPointerIndex = C->isNullValue();

  if (GEP.getNumOperands() == 2 && HasZeroPointerIndex)
    return ReplaceInstUsesWith(GEP, PtrOp);

  // Eliminate unneeded casts for indices.
  bool MadeChange = false;
  gep_type_iterator GTI = gep_type_begin(GEP);
  for (unsigned i = 1, e = GEP.getNumOperands(); i != e; ++i, ++GTI)
    if (isa<SequentialType>(*GTI)) {
      if (CastInst *CI = dyn_cast<CastInst>(GEP.getOperand(i))) {
        Value *Src = CI->getOperand(0);
        const Type *SrcTy = Src->getType();
        const Type *DestTy = CI->getType();
        if (Src->getType()->isInteger()) {
          if (SrcTy->getPrimitiveSizeInBits() ==
                       DestTy->getPrimitiveSizeInBits()) {
            // We can always eliminate a cast from ulong or long to the other.
            // We can always eliminate a cast from uint to int or the other on
            // 32-bit pointer platforms.
            if (DestTy->getPrimitiveSizeInBits() >= TD->getPointerSizeInBits()){
              MadeChange = true;
              GEP.setOperand(i, Src);
            }
          } else if (SrcTy->getPrimitiveSize() < DestTy->getPrimitiveSize() &&
                     SrcTy->getPrimitiveSize() == 4) {
            // We can always eliminate a cast from int to [u]long.  We can
            // eliminate a cast from uint to [u]long iff the target is a 32-bit
            // pointer target.
            if (SrcTy->isSigned() ||
                SrcTy->getPrimitiveSizeInBits() >= TD->getPointerSizeInBits()) {
              MadeChange = true;
              GEP.setOperand(i, Src);
            }
          }
        }
      }
      // If we are using a wider index than needed for this platform, shrink it
      // to what we need.  If the incoming value needs a cast instruction,
      // insert it.  This explicit cast can make subsequent optimizations more
      // obvious.
      Value *Op = GEP.getOperand(i);
      if (Op->getType()->getPrimitiveSize() > TD->getPointerSize())
        if (Constant *C = dyn_cast<Constant>(Op)) {
          GEP.setOperand(i, ConstantExpr::getCast(C,
                                     TD->getIntPtrType()->getSignedVersion()));
          MadeChange = true;
        } else {
          Op = InsertNewInstBefore(new CastInst(Op, TD->getIntPtrType(),
                                                Op->getName()), GEP);
          GEP.setOperand(i, Op);
          MadeChange = true;
        }

      // If this is a constant idx, make sure to canonicalize it to be a signed
      // operand, otherwise CSE and other optimizations are pessimized.
      if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(Op)) {
        GEP.setOperand(i, ConstantExpr::getCast(CUI,
                                          CUI->getType()->getSignedVersion()));
        MadeChange = true;
      }
    }
  if (MadeChange) return &GEP;

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  std::vector<Value*> SrcGEPOperands;
  if (User *Src = dyn_castGetElementPtr(PtrOp))
    SrcGEPOperands.assign(Src->op_begin(), Src->op_end());

  if (!SrcGEPOperands.empty()) {
    // Note that if our source is a gep chain itself that we wait for that
    // chain to be resolved before we perform this transformation.  This
    // avoids us creating a TON of code in some cases.
    //
    if (isa<GetElementPtrInst>(SrcGEPOperands[0]) &&
        cast<Instruction>(SrcGEPOperands[0])->getNumOperands() == 2)
      return 0;   // Wait until our source is folded to completion.

    std::vector<Value *> Indices;

    // Find out whether the last index in the source GEP is a sequential idx.
    bool EndsWithSequential = false;
    for (gep_type_iterator I = gep_type_begin(*cast<User>(PtrOp)),
           E = gep_type_end(*cast<User>(PtrOp)); I != E; ++I)
      EndsWithSequential = !isa<StructType>(*I);

    // Can we combine the two pointer arithmetics offsets?
    if (EndsWithSequential) {
      // Replace: gep (gep %P, long B), long A, ...
      // With:    T = long A+B; gep %P, T, ...
      //
      Value *Sum, *SO1 = SrcGEPOperands.back(), *GO1 = GEP.getOperand(1);
      if (SO1 == Constant::getNullValue(SO1->getType())) {
        Sum = GO1;
      } else if (GO1 == Constant::getNullValue(GO1->getType())) {
        Sum = SO1;
      } else {
        // If they aren't the same type, convert both to an integer of the
        // target's pointer size.
        if (SO1->getType() != GO1->getType()) {
          if (Constant *SO1C = dyn_cast<Constant>(SO1)) {
            SO1 = ConstantExpr::getCast(SO1C, GO1->getType());
          } else if (Constant *GO1C = dyn_cast<Constant>(GO1)) {
            GO1 = ConstantExpr::getCast(GO1C, SO1->getType());
          } else {
            unsigned PS = TD->getPointerSize();
            if (SO1->getType()->getPrimitiveSize() == PS) {
              // Convert GO1 to SO1's type.
              GO1 = InsertSignExtendToPtrTy(GO1, SO1->getType(), &GEP, this);

            } else if (GO1->getType()->getPrimitiveSize() == PS) {
              // Convert SO1 to GO1's type.
              SO1 = InsertSignExtendToPtrTy(SO1, GO1->getType(), &GEP, this);
            } else {
              const Type *PT = TD->getIntPtrType();
              SO1 = InsertSignExtendToPtrTy(SO1, PT, &GEP, this);
              GO1 = InsertSignExtendToPtrTy(GO1, PT, &GEP, this);
            }
          }
        }
        if (isa<Constant>(SO1) && isa<Constant>(GO1))
          Sum = ConstantExpr::getAdd(cast<Constant>(SO1), cast<Constant>(GO1));
        else {
          Sum = BinaryOperator::createAdd(SO1, GO1, PtrOp->getName()+".sum");
          InsertNewInstBefore(cast<Instruction>(Sum), GEP);
        }
      }

      // Recycle the GEP we already have if possible.
      if (SrcGEPOperands.size() == 2) {
        GEP.setOperand(0, SrcGEPOperands[0]);
        GEP.setOperand(1, Sum);
        return &GEP;
      } else {
        Indices.insert(Indices.end(), SrcGEPOperands.begin()+1,
                       SrcGEPOperands.end()-1);
        Indices.push_back(Sum);
        Indices.insert(Indices.end(), GEP.op_begin()+2, GEP.op_end());
      }
    } else if (isa<Constant>(*GEP.idx_begin()) &&
               cast<Constant>(*GEP.idx_begin())->isNullValue() &&
               SrcGEPOperands.size() != 1) {
      // Otherwise we can do the fold if the first index of the GEP is a zero
      Indices.insert(Indices.end(), SrcGEPOperands.begin()+1,
                     SrcGEPOperands.end());
      Indices.insert(Indices.end(), GEP.idx_begin()+1, GEP.idx_end());
    }

    if (!Indices.empty())
      return new GetElementPtrInst(SrcGEPOperands[0], Indices, GEP.getName());

  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(PtrOp)) {
    // GEP of global variable.  If all of the indices for this GEP are
    // constants, we can promote this to a constexpr instead of an instruction.

    // Scan for nonconstants...
    std::vector<Constant*> Indices;
    User::op_iterator I = GEP.idx_begin(), E = GEP.idx_end();
    for (; I != E && isa<Constant>(*I); ++I)
      Indices.push_back(cast<Constant>(*I));

    if (I == E) {  // If they are all constants...
      Constant *CE = ConstantExpr::getGetElementPtr(GV, Indices);

      // Replace all uses of the GEP with the new constexpr...
      return ReplaceInstUsesWith(GEP, CE);
    }
  } else if (Value *X = isCast(PtrOp)) {  // Is the operand a cast?
    if (!isa<PointerType>(X->getType())) {
      // Not interesting.  Source pointer must be a cast from pointer.
    } else if (HasZeroPointerIndex) {
      // transform: GEP (cast [10 x ubyte]* X to [0 x ubyte]*), long 0, ...
      // into     : GEP [10 x ubyte]* X, long 0, ...
      //
      // This occurs when the program declares an array extern like "int X[];"
      //
      const PointerType *CPTy = cast<PointerType>(PtrOp->getType());
      const PointerType *XTy = cast<PointerType>(X->getType());
      if (const ArrayType *XATy =
          dyn_cast<ArrayType>(XTy->getElementType()))
        if (const ArrayType *CATy =
            dyn_cast<ArrayType>(CPTy->getElementType()))
          if (CATy->getElementType() == XATy->getElementType()) {
            // At this point, we know that the cast source type is a pointer
            // to an array of the same type as the destination pointer
            // array.  Because the array type is never stepped over (there
            // is a leading zero) we can fold the cast into this GEP.
            GEP.setOperand(0, X);
            return &GEP;
          }
    } else if (GEP.getNumOperands() == 2) {
      // Transform things like:
      // %t = getelementptr ubyte* cast ([2 x int]* %str to uint*), uint %V
      // into:  %t1 = getelementptr [2 x int*]* %str, int 0, uint %V; cast
      const Type *SrcElTy = cast<PointerType>(X->getType())->getElementType();
      const Type *ResElTy=cast<PointerType>(PtrOp->getType())->getElementType();
      if (isa<ArrayType>(SrcElTy) &&
          TD->getTypeSize(cast<ArrayType>(SrcElTy)->getElementType()) ==
          TD->getTypeSize(ResElTy)) {
        Value *V = InsertNewInstBefore(
               new GetElementPtrInst(X, Constant::getNullValue(Type::IntTy),
                                     GEP.getOperand(1), GEP.getName()), GEP);
        return new CastInst(V, GEP.getType());
      }
      
      // Transform things like:
      // getelementptr sbyte* cast ([100 x double]* X to sbyte*), int %tmp
      //   (where tmp = 8*tmp2) into:
      // getelementptr [100 x double]* %arr, int 0, int %tmp.2
      
      if (isa<ArrayType>(SrcElTy) &&
          (ResElTy == Type::SByteTy || ResElTy == Type::UByteTy)) {
        uint64_t ArrayEltSize =
            TD->getTypeSize(cast<ArrayType>(SrcElTy)->getElementType());
        
        // Check to see if "tmp" is a scale by a multiple of ArrayEltSize.  We
        // allow either a mul, shift, or constant here.
        Value *NewIdx = 0;
        ConstantInt *Scale = 0;
        if (ArrayEltSize == 1) {
          NewIdx = GEP.getOperand(1);
          Scale = ConstantInt::get(NewIdx->getType(), 1);
        } else if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP.getOperand(1))) {
          NewIdx = ConstantInt::get(CI->getType(), 1);
          Scale = CI;
        } else if (Instruction *Inst =dyn_cast<Instruction>(GEP.getOperand(1))){
          if (Inst->getOpcode() == Instruction::Shl &&
              isa<ConstantInt>(Inst->getOperand(1))) {
            unsigned ShAmt =cast<ConstantUInt>(Inst->getOperand(1))->getValue();
            if (Inst->getType()->isSigned())
              Scale = ConstantSInt::get(Inst->getType(), 1ULL << ShAmt);
            else
              Scale = ConstantUInt::get(Inst->getType(), 1ULL << ShAmt);
            NewIdx = Inst->getOperand(0);
          } else if (Inst->getOpcode() == Instruction::Mul &&
                     isa<ConstantInt>(Inst->getOperand(1))) {
            Scale = cast<ConstantInt>(Inst->getOperand(1));
            NewIdx = Inst->getOperand(0);
          }
        }

        // If the index will be to exactly the right offset with the scale taken
        // out, perform the transformation.
        if (Scale && Scale->getRawValue() % ArrayEltSize == 0) {
          if (ConstantSInt *C = dyn_cast<ConstantSInt>(Scale))
            Scale = ConstantSInt::get(C->getType(),
                                      (int64_t)C->getRawValue() / 
                                      (int64_t)ArrayEltSize);
          else
            Scale = ConstantUInt::get(Scale->getType(),
                                      Scale->getRawValue() / ArrayEltSize);
          if (Scale->getRawValue() != 1) {
            Constant *C = ConstantExpr::getCast(Scale, NewIdx->getType());
            Instruction *Sc = BinaryOperator::createMul(NewIdx, C, "idxscale");
            NewIdx = InsertNewInstBefore(Sc, GEP);
          }

          // Insert the new GEP instruction.
          Instruction *Idx =
            new GetElementPtrInst(X, Constant::getNullValue(Type::IntTy),
                                  NewIdx, GEP.getName());
          Idx = InsertNewInstBefore(Idx, GEP);
          return new CastInst(Idx, GEP.getType());
        }
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitAllocationInst(AllocationInst &AI) {
  // Convert: malloc Ty, C - where C is a constant != 1 into: malloc [C x Ty], 1
  if (AI.isArrayAllocation())    // Check C != 1
    if (const ConstantUInt *C = dyn_cast<ConstantUInt>(AI.getArraySize())) {
      const Type *NewTy = ArrayType::get(AI.getAllocatedType(), C->getValue());
      AllocationInst *New = 0;

      // Create and insert the replacement instruction...
      if (isa<MallocInst>(AI))
        New = new MallocInst(NewTy, 0, AI.getAlignment(), AI.getName());
      else {
        assert(isa<AllocaInst>(AI) && "Unknown type of allocation inst!");
        New = new AllocaInst(NewTy, 0, AI.getAlignment(), AI.getName());
      }

      InsertNewInstBefore(New, AI);

      // Scan to the end of the allocation instructions, to skip over a block of
      // allocas if possible...
      //
      BasicBlock::iterator It = New;
      while (isa<AllocationInst>(*It)) ++It;

      // Now that I is pointing to the first non-allocation-inst in the block,
      // insert our getelementptr instruction...
      //
      Value *NullIdx = Constant::getNullValue(Type::IntTy);
      Value *V = new GetElementPtrInst(New, NullIdx, NullIdx,
                                       New->getName()+".sub", It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, V);
    } else if (isa<UndefValue>(AI.getArraySize())) {
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));
    }

  // If alloca'ing a zero byte object, replace the alloca with a null pointer.
  // Note that we only do this for alloca's, because malloc should allocate and
  // return a unique pointer, even for a zero byte allocation.
  if (isa<AllocaInst>(AI) && AI.getAllocatedType()->isSized() &&
      TD->getTypeSize(AI.getAllocatedType()) == 0)
    return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));

  return 0;
}

Instruction *InstCombiner::visitFreeInst(FreeInst &FI) {
  Value *Op = FI.getOperand(0);

  // Change free <ty>* (cast <ty2>* X to <ty>*) into free <ty2>* X
  if (CastInst *CI = dyn_cast<CastInst>(Op))
    if (isa<PointerType>(CI->getOperand(0)->getType())) {
      FI.setOperand(0, CI->getOperand(0));
      return &FI;
    }

  // free undef -> unreachable.
  if (isa<UndefValue>(Op)) {
    // Insert a new store to null because we cannot modify the CFG here.
    new StoreInst(ConstantBool::True,
                  UndefValue::get(PointerType::get(Type::BoolTy)), &FI);
    return EraseInstFromFunction(FI);
  }

  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op))
    return EraseInstFromFunction(FI);

  return 0;
}


/// InstCombineLoadCast - Fold 'load (cast P)' -> cast (load P)' when possible.
static Instruction *InstCombineLoadCast(InstCombiner &IC, LoadInst &LI) {
  User *CI = cast<User>(LI.getOperand(0));
  Value *CastOp = CI->getOperand(0);

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  if (const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {
    const Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isInteger() || isa<PointerType>(DestPTy)) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (const ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            std::vector<Value*> Idxs(2, Constant::getNullValue(Type::IntTy));
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if ((SrcPTy->isInteger() || isa<PointerType>(SrcPTy)) &&
          // Do not allow turning this into a load of an integer, which is then
          // casted to a pointer, this pessimizes pointer analysis a lot.
          (isa<PointerType>(SrcPTy) == isa<PointerType>(LI.getType())) &&
          IC.getTargetData().getTypeSize(SrcPTy) ==
               IC.getTargetData().getTypeSize(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the load, cast
        // the result of the loaded value.
        Value *NewLoad = IC.InsertNewInstBefore(new LoadInst(CastOp,
                                                             CI->getName(),
                                                         LI.isVolatile()),LI);
        // Now cast the result of the load.
        return new CastInst(NewLoad, LI.getType());
      }
    }
  }
  return 0;
}

/// isSafeToLoadUnconditionally - Return true if we know that executing a load
/// from this value cannot trap.  If it is not obviously safe to load from the
/// specified pointer, we do a quick local scan of the basic block containing
/// ScanFrom, to determine if the address is already accessed.
static bool isSafeToLoadUnconditionally(Value *V, Instruction *ScanFrom) {
  // If it is an alloca or global variable, it is always safe to load from.
  if (isa<AllocaInst>(V) || isa<GlobalVariable>(V)) return true;

  // Otherwise, be a little bit agressive by scanning the local block where we
  // want to check to see if the pointer is already being loaded or stored
  // from/to.  If so, the previous load or store would have already trapped,
  // so there is no harm doing an extra load (also, CSE will later eliminate
  // the load entirely).
  BasicBlock::iterator BBI = ScanFrom, E = ScanFrom->getParent()->begin();

  while (BBI != E) {
    --BBI;

    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI->getOperand(0) == V) return true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
      if (SI->getOperand(1) == V) return true;

  }
  return false;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);

  // load (cast X) --> cast (load X) iff safe
  if (CastInst *CI = dyn_cast<CastInst>(Op))
    if (Instruction *Res = InstCombineLoadCast(*this, LI))
      return Res;

  // None of the following transforms are legal for volatile loads.
  if (LI.isVolatile()) return 0;
  
  if (&LI.getParent()->front() != &LI) {
    BasicBlock::iterator BBI = &LI; --BBI;
    // If the instruction immediately before this is a store to the same
    // address, do a simple form of store->load forwarding.
    if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
      if (SI->getOperand(1) == LI.getOperand(0))
        return ReplaceInstUsesWith(LI, SI->getOperand(0));
    if (LoadInst *LIB = dyn_cast<LoadInst>(BBI))
      if (LIB->getOperand(0) == LI.getOperand(0))
        return ReplaceInstUsesWith(LI, LIB);
  }

  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op))
    if (isa<ConstantPointerNull>(GEPI->getOperand(0)) ||
        isa<UndefValue>(GEPI->getOperand(0))) {
      // Insert a new store to null instruction before the load to indicate
      // that this code is not reachable.  We do this instead of inserting
      // an unreachable instruction directly because we cannot modify the
      // CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }

  if (Constant *C = dyn_cast<Constant>(Op)) {
    // load null/undef -> undef
    if ((C->isNullValue() || isa<UndefValue>(C))) {
      // Insert a new store to null instruction before the load to indicate that
      // this code is not reachable.  We do this instead of inserting an
      // unreachable instruction directly because we cannot modify the CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }

    // Instcombine load (constant global) into the value loaded.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op))
      if (GV->isConstant() && !GV->isExternal())
        return ReplaceInstUsesWith(LI, GV->getInitializer());

    // Instcombine load (constantexpr_GEP global, 0, ...) into the value loaded.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op))
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
          if (GV->isConstant() && !GV->isExternal())
            if (Constant *V = 
               ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE))
              return ReplaceInstUsesWith(LI, V);
        if (CE->getOperand(0)->isNullValue()) {
          // Insert a new store to null instruction before the load to indicate
          // that this code is not reachable.  We do this instead of inserting
          // an unreachable instruction directly because we cannot modify the
          // CFG.
          new StoreInst(UndefValue::get(LI.getType()),
                        Constant::getNullValue(Op->getType()), &LI);
          return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
        }

      } else if (CE->getOpcode() == Instruction::Cast) {
        if (Instruction *Res = InstCombineLoadCast(*this, LI))
          return Res;
      }
  }

  if (Op->hasOneUse()) {
    // Change select and PHI nodes to select values instead of addresses: this
    // helps alias analysis out a lot, allows many others simplifications, and
    // exposes redundancy in the code.
    //
    // Note that we cannot do the transformation unless we know that the
    // introduced loads cannot trap!  Something like this is valid as long as
    // the condition is always false: load (select bool %C, int* null, int* %G),
    // but it would not be valid if we transformed it to load from null
    // unconditionally.
    //
    if (SelectInst *SI = dyn_cast<SelectInst>(Op)) {
      // load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2).
      if (isSafeToLoadUnconditionally(SI->getOperand(1), SI) &&
          isSafeToLoadUnconditionally(SI->getOperand(2), SI)) {
        Value *V1 = InsertNewInstBefore(new LoadInst(SI->getOperand(1),
                                     SI->getOperand(1)->getName()+".val"), LI);
        Value *V2 = InsertNewInstBefore(new LoadInst(SI->getOperand(2),
                                     SI->getOperand(2)->getName()+".val"), LI);
        return new SelectInst(SI->getCondition(), V1, V2);
      }

      // load (select (cond, null, P)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(1)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(2));
          return &LI;
        }

      // load (select (cond, P, null)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(2)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(1));
          return &LI;
        }

    } else if (PHINode *PN = dyn_cast<PHINode>(Op)) {
      // load (phi (&V1, &V2, &V3))  --> phi(load &V1, load &V2, load &V3)
      bool Safe = PN->getParent() == LI.getParent();

      // Scan all of the instructions between the PHI and the load to make
      // sure there are no instructions that might possibly alter the value
      // loaded from the PHI.
      if (Safe) {
        BasicBlock::iterator I = &LI;
        for (--I; !isa<PHINode>(I); --I)
          if (isa<StoreInst>(I) || isa<CallInst>(I)) {
            Safe = false;
            break;
          }
      }

      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e && Safe; ++i)
        if (!isSafeToLoadUnconditionally(PN->getIncomingValue(i),
                                    PN->getIncomingBlock(i)->getTerminator()))
          Safe = false;

      if (Safe) {
        // Create the PHI.
        PHINode *NewPN = new PHINode(LI.getType(), PN->getName());
        InsertNewInstBefore(NewPN, *PN);
        std::map<BasicBlock*,Value*> LoadMap;  // Don't insert duplicate loads

        for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
          BasicBlock *BB = PN->getIncomingBlock(i);
          Value *&TheLoad = LoadMap[BB];
          if (TheLoad == 0) {
            Value *InVal = PN->getIncomingValue(i);
            TheLoad = InsertNewInstBefore(new LoadInst(InVal,
                                                       InVal->getName()+".val"),
                                          *BB->getTerminator());
          }
          NewPN->addIncoming(TheLoad, BB);
        }
        return ReplaceInstUsesWith(LI, NewPN);
      }
    }
  }
  return 0;
}

/// InstCombineStoreToCast - Fold 'store V, (cast P)' -> store (cast V), P'
/// when possible.
static Instruction *InstCombineStoreToCast(InstCombiner &IC, StoreInst &SI) {
  User *CI = cast<User>(SI.getOperand(1));
  Value *CastOp = CI->getOperand(0);

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  if (const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {
    const Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isInteger() || isa<PointerType>(DestPTy)) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (const ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            std::vector<Value*> Idxs(2, Constant::getNullValue(Type::IntTy));
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if ((SrcPTy->isInteger() || isa<PointerType>(SrcPTy)) &&
          IC.getTargetData().getTypeSize(SrcPTy) ==
               IC.getTargetData().getTypeSize(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the store, cast
        // the value to be stored.
        Value *NewCast;
        if (Constant *C = dyn_cast<Constant>(SI.getOperand(0)))
          NewCast = ConstantExpr::getCast(C, SrcPTy);
        else
          NewCast = IC.InsertNewInstBefore(new CastInst(SI.getOperand(0),
                                                        SrcPTy,
                                         SI.getOperand(0)->getName()+".c"), SI);

        return new StoreInst(NewCast, CastOp);
      }
    }
  }
  return 0;
}

Instruction *InstCombiner::visitStoreInst(StoreInst &SI) {
  Value *Val = SI.getOperand(0);
  Value *Ptr = SI.getOperand(1);

  if (isa<UndefValue>(Ptr)) {     // store X, undef -> noop (even if volatile)
    removeFromWorkList(&SI);
    SI.eraseFromParent();
    ++NumCombined;
    return 0;
  }

  if (SI.isVolatile()) return 0;  // Don't hack volatile loads.

  // store X, null    -> turns into 'unreachable' in SimplifyCFG
  if (isa<ConstantPointerNull>(Ptr)) {
    if (!isa<UndefValue>(Val)) {
      SI.setOperand(0, UndefValue::get(Val->getType()));
      if (Instruction *U = dyn_cast<Instruction>(Val))
        WorkList.push_back(U);  // Dropped a use.
      ++NumCombined;
    }
    return 0;  // Do not modify these!
  }

  // store undef, Ptr -> noop
  if (isa<UndefValue>(Val)) {
    removeFromWorkList(&SI);
    SI.eraseFromParent();
    ++NumCombined;
    return 0;
  }

  // If the pointer destination is a cast, see if we can fold the cast into the
  // source instead.
  if (CastInst *CI = dyn_cast<CastInst>(Ptr))
    if (Instruction *Res = InstCombineStoreToCast(*this, SI))
      return Res;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
    if (CE->getOpcode() == Instruction::Cast)
      if (Instruction *Res = InstCombineStoreToCast(*this, SI))
        return Res;

  
  // If this store is the last instruction in the basic block, and if the block
  // ends with an unconditional branch, try to move it to the successor block.
  BasicBlock::iterator BBI = &SI; ++BBI;
  if (BranchInst *BI = dyn_cast<BranchInst>(BBI))
    if (BI->isUnconditional()) {
      // Check to see if the successor block has exactly two incoming edges.  If
      // so, see if the other predecessor contains a store to the same location.
      // if so, insert a PHI node (if needed) and move the stores down.
      BasicBlock *Dest = BI->getSuccessor(0);

      pred_iterator PI = pred_begin(Dest);
      BasicBlock *Other = 0;
      if (*PI != BI->getParent())
        Other = *PI;
      ++PI;
      if (PI != pred_end(Dest)) {
        if (*PI != BI->getParent())
          if (Other)
            Other = 0;
          else
            Other = *PI;
        if (++PI != pred_end(Dest))
          Other = 0;
      }
      if (Other) {  // If only one other pred...
        BBI = Other->getTerminator();
        // Make sure this other block ends in an unconditional branch and that
        // there is an instruction before the branch.
        if (isa<BranchInst>(BBI) && cast<BranchInst>(BBI)->isUnconditional() &&
            BBI != Other->begin()) {
          --BBI;
          StoreInst *OtherStore = dyn_cast<StoreInst>(BBI);
          
          // If this instruction is a store to the same location.
          if (OtherStore && OtherStore->getOperand(1) == SI.getOperand(1)) {
            // Okay, we know we can perform this transformation.  Insert a PHI
            // node now if we need it.
            Value *MergedVal = OtherStore->getOperand(0);
            if (MergedVal != SI.getOperand(0)) {
              PHINode *PN = new PHINode(MergedVal->getType(), "storemerge");
              PN->reserveOperandSpace(2);
              PN->addIncoming(SI.getOperand(0), SI.getParent());
              PN->addIncoming(OtherStore->getOperand(0), Other);
              MergedVal = InsertNewInstBefore(PN, Dest->front());
            }
            
            // Advance to a place where it is safe to insert the new store and
            // insert it.
            BBI = Dest->begin();
            while (isa<PHINode>(BBI)) ++BBI;
            InsertNewInstBefore(new StoreInst(MergedVal, SI.getOperand(1),
                                              OtherStore->isVolatile()), *BBI);

            // Nuke the old stores.
            removeFromWorkList(&SI);
            removeFromWorkList(OtherStore);
            SI.eraseFromParent();
            OtherStore->eraseFromParent();
            ++NumCombined;
            return 0;
          }
        }
      }
    }
  
  return 0;
}


Instruction *InstCombiner::visitBranchInst(BranchInst &BI) {
  // Change br (not X), label True, label False to: br X, label False, True
  Value *X = 0;
  BasicBlock *TrueDest;
  BasicBlock *FalseDest;
  if (match(&BI, m_Br(m_Not(m_Value(X)), TrueDest, FalseDest)) &&
      !isa<Constant>(X)) {
    // Swap Destinations and condition...
    BI.setCondition(X);
    BI.setSuccessor(0, FalseDest);
    BI.setSuccessor(1, TrueDest);
    return &BI;
  }

  // Cannonicalize setne -> seteq
  Instruction::BinaryOps Op; Value *Y;
  if (match(&BI, m_Br(m_SetCond(Op, m_Value(X), m_Value(Y)),
                      TrueDest, FalseDest)))
    if ((Op == Instruction::SetNE || Op == Instruction::SetLE ||
         Op == Instruction::SetGE) && BI.getCondition()->hasOneUse()) {
      SetCondInst *I = cast<SetCondInst>(BI.getCondition());
      std::string Name = I->getName(); I->setName("");
      Instruction::BinaryOps NewOpcode = SetCondInst::getInverseCondition(Op);
      Value *NewSCC =  BinaryOperator::create(NewOpcode, X, Y, Name, I);
      // Swap Destinations and condition...
      BI.setCondition(NewSCC);
      BI.setSuccessor(0, FalseDest);
      BI.setSuccessor(1, TrueDest);
      removeFromWorkList(I);
      I->getParent()->getInstList().erase(I);
      WorkList.push_back(cast<Instruction>(NewSCC));
      return &BI;
    }

  return 0;
}

Instruction *InstCombiner::visitSwitchInst(SwitchInst &SI) {
  Value *Cond = SI.getCondition();
  if (Instruction *I = dyn_cast<Instruction>(Cond)) {
    if (I->getOpcode() == Instruction::Add)
      if (ConstantInt *AddRHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
        // change 'switch (X+4) case 1:' into 'switch (X) case -3'
        for (unsigned i = 2, e = SI.getNumOperands(); i != e; i += 2)
          SI.setOperand(i,ConstantExpr::getSub(cast<Constant>(SI.getOperand(i)),
                                                AddRHS));
        SI.setOperand(0, I->getOperand(0));
        WorkList.push_back(I);
        return &SI;
      }
  }
  return 0;
}

void InstCombiner::removeFromWorkList(Instruction *I) {
  WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), I),
                 WorkList.end());
}


/// TryToSinkInstruction - Try to move the specified instruction from its
/// current block into the beginning of DestBlock, which can only happen if it's
/// safe to move the instruction past all of the instructions between it and the
/// end of its block.
static bool TryToSinkInstruction(Instruction *I, BasicBlock *DestBlock) {
  assert(I->hasOneUse() && "Invariants didn't hold!");

  // Cannot move control-flow-involving, volatile loads, vaarg, etc.
  if (isa<PHINode>(I) || I->mayWriteToMemory()) return false;

  // Do not sink alloca instructions out of the entry block.
  if (isa<AllocaInst>(I) && I->getParent() == &DestBlock->getParent()->front())
    return false;

  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    for (BasicBlock::iterator Scan = LI, E = LI->getParent()->end();
         Scan != E; ++Scan)
      if (Scan->mayWriteToMemory())
        return false;
  }

  BasicBlock::iterator InsertPos = DestBlock->begin();
  while (isa<PHINode>(InsertPos)) ++InsertPos;

  I->moveBefore(InsertPos);
  ++NumSunkInst;
  return true;
}

bool InstCombiner::runOnFunction(Function &F) {
  bool Changed = false;
  TD = &getAnalysis<TargetData>();

  {
    // Populate the worklist with the reachable instructions.
    std::set<BasicBlock*> Visited;
    for (df_ext_iterator<BasicBlock*> BB = df_ext_begin(&F.front(), Visited),
           E = df_ext_end(&F.front(), Visited); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        WorkList.push_back(I);

    // Do a quick scan over the function.  If we find any blocks that are
    // unreachable, remove any instructions inside of them.  This prevents
    // the instcombine code from having to deal with some bad special cases.
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      if (!Visited.count(BB)) {
        Instruction *Term = BB->getTerminator();
        while (Term != BB->begin()) {   // Remove instrs bottom-up
          BasicBlock::iterator I = Term; --I;

          DEBUG(std::cerr << "IC: DCE: " << *I);
          ++NumDeadInst;

          if (!I->use_empty())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          I->eraseFromParent();
        }
      }
  }

  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();  // Get an instruction from the worklist
    WorkList.pop_back();

    // Check to see if we can DCE or ConstantPropagate the instruction...
    // Check to see if we can DIE the instruction...
    if (isInstructionTriviallyDead(I)) {
      // Add operands to the worklist...
      if (I->getNumOperands() < 4)
        AddUsesToWorkList(*I);
      ++NumDeadInst;

      DEBUG(std::cerr << "IC: DCE: " << *I);

      I->eraseFromParent();
      removeFromWorkList(I);
      continue;
    }

    // Instruction isn't dead, see if we can constant propagate it...
    if (Constant *C = ConstantFoldInstruction(I)) {
      Value* Ptr = I->getOperand(0);
      if (isa<GetElementPtrInst>(I) &&
          cast<Constant>(Ptr)->isNullValue() &&
          !isa<ConstantPointerNull>(C) &&
          cast<PointerType>(Ptr->getType())->getElementType()->isSized()) {
        // If this is a constant expr gep that is effectively computing an
        // "offsetof", fold it into 'cast int X to T*' instead of 'gep 0, 0, 12'
        bool isFoldableGEP = true;
        for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i)
          if (!isa<ConstantInt>(I->getOperand(i)))
            isFoldableGEP = false;
        if (isFoldableGEP) {
          uint64_t Offset = TD->getIndexedOffset(Ptr->getType(),
                             std::vector<Value*>(I->op_begin()+1, I->op_end()));
          C = ConstantUInt::get(Type::ULongTy, Offset);
          C = ConstantExpr::getCast(C, TD->getIntPtrType());
          C = ConstantExpr::getCast(C, I->getType());
        }
      }

      DEBUG(std::cerr << "IC: ConstFold to: " << *C << " from: " << *I);

      // Add operands to the worklist...
      AddUsesToWorkList(*I);
      ReplaceInstUsesWith(*I, C);

      ++NumConstProp;
      I->getParent()->getInstList().erase(I);
      removeFromWorkList(I);
      continue;
    }

    // See if we can trivially sink this instruction to a successor basic block.
    if (I->hasOneUse()) {
      BasicBlock *BB = I->getParent();
      BasicBlock *UserParent = cast<Instruction>(I->use_back())->getParent();
      if (UserParent != BB) {
        bool UserIsSuccessor = false;
        // See if the user is one of our successors.
        for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
          if (*SI == UserParent) {
            UserIsSuccessor = true;
            break;
          }

        // If the user is one of our immediate successors, and if that successor
        // only has us as a predecessors (we'd have to split the critical edge
        // otherwise), we can keep going.
        if (UserIsSuccessor && !isa<PHINode>(I->use_back()) &&
            next(pred_begin(UserParent)) == pred_end(UserParent))
          // Okay, the CFG is simple enough, try to sink this instruction.
          Changed |= TryToSinkInstruction(I, UserParent);
      }
    }

    // Now that we have an instruction, try combining it to simplify it...
    if (Instruction *Result = visit(*I)) {
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        DEBUG(std::cerr << "IC: Old = " << *I
                        << "    New = " << *Result);

        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Push the new instruction and any users onto the worklist.
        WorkList.push_back(Result);
        AddUsersToWorkList(*Result);

        // Move the name to the new instruction first...
        std::string OldName = I->getName(); I->setName("");
        Result->setName(OldName);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        BasicBlock::iterator InsertPos = I;

        if (!isa<PHINode>(Result))        // If combining a PHI, don't insert
          while (isa<PHINode>(InsertPos)) // middle of a block of PHIs.
            ++InsertPos;

        InstParent->getInstList().insert(InsertPos, Result);

        // Make sure that we reprocess all operands now that we reduced their
        // use counts.
        for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
          if (Instruction *OpI = dyn_cast<Instruction>(I->getOperand(i)))
            WorkList.push_back(OpI);

        // Instructions can end up on the worklist more than once.  Make sure
        // we do not process an instruction that has been deleted.
        removeFromWorkList(I);

        // Erase the old instruction.
        InstParent->getInstList().erase(I);
      } else {
        DEBUG(std::cerr << "IC: MOD = " << *I);

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I)) {
          // Make sure we process all operands now that we are reducing their
          // use counts.
          for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
            if (Instruction *OpI = dyn_cast<Instruction>(I->getOperand(i)))
              WorkList.push_back(OpI);

          // Instructions may end up in the worklist more than once.  Erase all
          // occurrances of this instruction.
          removeFromWorkList(I);
          I->eraseFromParent();
        } else {
          WorkList.push_back(Result);
          AddUsersToWorkList(*Result);
        }
      }
      Changed = true;
    }
  }

  return Changed;
}

FunctionPass *llvm::createInstructionCombiningPass() {
  return new InstCombiner();
}


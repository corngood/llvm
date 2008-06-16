//===- ValueTracking.cpp - Walk computations to compute properties --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help analyze properties that chains of
// computations have.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include <cstring>
using namespace llvm;

/// getOpcode - If this is an Instruction or a ConstantExpr, return the
/// opcode value. Otherwise return UserOp1.
static unsigned getOpcode(const Value *V) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getOpcode();
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
    return CE->getOpcode();
  // Use UserOp1 to mean there's no opcode.
  return Instruction::UserOp1;
}


/// ComputeMaskedBits - Determine which of the bits specified in Mask are
/// known to be either zero or one and return them in the KnownZero/KnownOne
/// bit sets.  This code only analyzes bits in Mask, in order to short-circuit
/// processing.
/// NOTE: we cannot consider 'undef' to be "IsZero" here.  The problem is that
/// we cannot optimize based on the assumption that it is zero without changing
/// it to be an explicit zero.  If we don't change it to zero, other code could
/// optimized based on the contradictory assumption that it is non-zero.
/// Because instcombine aggressively folds operations with undef args anyway,
/// this won't lose us code quality.
void llvm::ComputeMaskedBits(Value *V, const APInt &Mask,
                             APInt &KnownZero, APInt &KnownOne,
                             TargetData *TD, unsigned Depth) {
  assert(V && "No Value?");
  assert(Depth <= 6 && "Limit Search Depth");
  uint32_t BitWidth = Mask.getBitWidth();
  assert((V->getType()->isInteger() || isa<PointerType>(V->getType())) &&
         "Not integer or pointer type!");
  assert((!TD || TD->getTypeSizeInBits(V->getType()) == BitWidth) &&
         (!isa<IntegerType>(V->getType()) ||
          V->getType()->getPrimitiveSizeInBits() == BitWidth) &&
         KnownZero.getBitWidth() == BitWidth && 
         KnownOne.getBitWidth() == BitWidth &&
         "V, Mask, KnownOne and KnownZero should have same BitWidth");

  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    // We know all of the bits for a constant!
    KnownOne = CI->getValue() & Mask;
    KnownZero = ~KnownOne & Mask;
    return;
  }
  // Null is all-zeros.
  if (isa<ConstantPointerNull>(V)) {
    KnownOne.clear();
    KnownZero = Mask;
    return;
  }
  // The address of an aligned GlobalValue has trailing zeros.
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    unsigned Align = GV->getAlignment();
    if (Align == 0 && TD && GV->getType()->getElementType()->isSized()) 
      Align = TD->getPrefTypeAlignment(GV->getType()->getElementType());
    if (Align > 0)
      KnownZero = Mask & APInt::getLowBitsSet(BitWidth,
                                              CountTrailingZeros_32(Align));
    else
      KnownZero.clear();
    KnownOne.clear();
    return;
  }

  KnownZero.clear(); KnownOne.clear();   // Start out not knowing anything.

  if (Depth == 6 || Mask == 0)
    return;  // Limit search depth.

  User *I = dyn_cast<User>(V);
  if (!I) return;

  APInt KnownZero2(KnownZero), KnownOne2(KnownOne);
  switch (getOpcode(I)) {
  default: break;
  case Instruction::And: {
    // If either the LHS or the RHS are Zero, the result is zero.
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero, KnownOne, TD, Depth+1);
    APInt Mask2(Mask & ~KnownZero);
    ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero2, KnownOne2, TD,
                      Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    return;
  }
  case Instruction::Or: {
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero, KnownOne, TD, Depth+1);
    APInt Mask2(Mask & ~KnownOne);
    ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero2, KnownOne2, TD,
                      Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    return;
  }
  case Instruction::Xor: {
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero, KnownOne, TD, Depth+1);
    ComputeMaskedBits(I->getOperand(0), Mask, KnownZero2, KnownOne2, TD,
                      Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    APInt KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOne = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    KnownZero = KnownZeroOut;
    return;
  }
  case Instruction::Mul: {
    APInt Mask2 = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(I->getOperand(1), Mask2, KnownZero, KnownOne, TD,Depth+1);
    ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero2, KnownOne2, TD,
                      Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If low bits are zero in either operand, output low known-0 bits.
    // Also compute a conserative estimate for high known-0 bits.
    // More trickiness is possible, but this is sufficient for the
    // interesting case of alignment computation.
    KnownOne.clear();
    unsigned TrailZ = KnownZero.countTrailingOnes() +
                      KnownZero2.countTrailingOnes();
    unsigned LeadZ =  std::max(KnownZero.countLeadingOnes() +
                               KnownZero2.countLeadingOnes(),
                               BitWidth) - BitWidth;

    TrailZ = std::min(TrailZ, BitWidth);
    LeadZ = std::min(LeadZ, BitWidth);
    KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ) |
                APInt::getHighBitsSet(BitWidth, LeadZ);
    KnownZero &= Mask;
    return;
  }
  case Instruction::UDiv: {
    // For the purposes of computing leading zeros we can conservatively
    // treat a udiv as a logical right shift by the power of 2 known to
    // be less than the denominator.
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(I->getOperand(0),
                      AllOnes, KnownZero2, KnownOne2, TD, Depth+1);
    unsigned LeadZ = KnownZero2.countLeadingOnes();

    KnownOne2.clear();
    KnownZero2.clear();
    ComputeMaskedBits(I->getOperand(1),
                      AllOnes, KnownZero2, KnownOne2, TD, Depth+1);
    unsigned RHSUnknownLeadingOnes = KnownOne2.countLeadingZeros();
    if (RHSUnknownLeadingOnes != BitWidth)
      LeadZ = std::min(BitWidth,
                       LeadZ + BitWidth - RHSUnknownLeadingOnes - 1);

    KnownZero = APInt::getHighBitsSet(BitWidth, LeadZ) & Mask;
    return;
  }
  case Instruction::Select:
    ComputeMaskedBits(I->getOperand(2), Mask, KnownZero, KnownOne, TD, Depth+1);
    ComputeMaskedBits(I->getOperand(1), Mask, KnownZero2, KnownOne2, TD,
                      Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    return;
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
    return; // Can't work with floating point.
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
    // We can't handle these if we don't know the pointer size.
    if (!TD) return;
    // FALL THROUGH and handle them the same as zext/trunc.
  case Instruction::ZExt:
  case Instruction::Trunc: {
    // Note that we handle pointer operands here because of inttoptr/ptrtoint
    // which fall through here.
    const Type *SrcTy = I->getOperand(0)->getType();
    uint32_t SrcBitWidth = TD ?
      TD->getTypeSizeInBits(SrcTy) :
      SrcTy->getPrimitiveSizeInBits();
    APInt MaskIn(Mask);
    MaskIn.zextOrTrunc(SrcBitWidth);
    KnownZero.zextOrTrunc(SrcBitWidth);
    KnownOne.zextOrTrunc(SrcBitWidth);
    ComputeMaskedBits(I->getOperand(0), MaskIn, KnownZero, KnownOne, TD,
                      Depth+1);
    KnownZero.zextOrTrunc(BitWidth);
    KnownOne.zextOrTrunc(BitWidth);
    // Any top bits are known to be zero.
    if (BitWidth > SrcBitWidth)
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    return;
  }
  case Instruction::BitCast: {
    const Type *SrcTy = I->getOperand(0)->getType();
    if (SrcTy->isInteger() || isa<PointerType>(SrcTy)) {
      ComputeMaskedBits(I->getOperand(0), Mask, KnownZero, KnownOne, TD,
                        Depth+1);
      return;
    }
    break;
  }
  case Instruction::SExt: {
    // Compute the bits in the result that are not present in the input.
    const IntegerType *SrcTy = cast<IntegerType>(I->getOperand(0)->getType());
    uint32_t SrcBitWidth = SrcTy->getBitWidth();
      
    APInt MaskIn(Mask); 
    MaskIn.trunc(SrcBitWidth);
    KnownZero.trunc(SrcBitWidth);
    KnownOne.trunc(SrcBitWidth);
    ComputeMaskedBits(I->getOperand(0), MaskIn, KnownZero, KnownOne, TD,
                      Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero.zext(BitWidth);
    KnownOne.zext(BitWidth);

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    if (KnownZero[SrcBitWidth-1])             // Input sign bit known zero
      KnownZero |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    else if (KnownOne[SrcBitWidth-1])           // Input sign bit known set
      KnownOne |= APInt::getHighBitsSet(BitWidth, BitWidth - SrcBitWidth);
    return;
  }
  case Instruction::Shl:
    // (shl X, C1) & C2 == 0   iff   (X & C2 >>u C1) == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth);
      APInt Mask2(Mask.lshr(ShiftAmt));
      ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero, KnownOne, TD,
                        Depth+1);
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero <<= ShiftAmt;
      KnownOne  <<= ShiftAmt;
      KnownZero |= APInt::getLowBitsSet(BitWidth, ShiftAmt); // low bits known 0
      return;
    }
    break;
  case Instruction::LShr:
    // (ushr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // Compute the new bits that are at the top now.
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth);
      
      // Unsigned shift right.
      APInt Mask2(Mask.shl(ShiftAmt));
      ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero,KnownOne, TD,
                        Depth+1);
      assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?"); 
      KnownZero = APIntOps::lshr(KnownZero, ShiftAmt);
      KnownOne  = APIntOps::lshr(KnownOne, ShiftAmt);
      // high bits known zero.
      KnownZero |= APInt::getHighBitsSet(BitWidth, ShiftAmt);
      return;
    }
    break;
  case Instruction::AShr:
    // (ashr X, C1) & C2 == 0   iff  (-1 >> C1) & C2 == 0
    if (ConstantInt *SA = dyn_cast<ConstantInt>(I->getOperand(1))) {
      // Compute the new bits that are at the top now.
      uint64_t ShiftAmt = SA->getLimitedValue(BitWidth);
      
      // Signed shift right.
      APInt Mask2(Mask.shl(ShiftAmt));
      ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero, KnownOne, TD,
                        Depth+1);
      assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?"); 
      KnownZero = APIntOps::lshr(KnownZero, ShiftAmt);
      KnownOne  = APIntOps::lshr(KnownOne, ShiftAmt);
        
      APInt HighBits(APInt::getHighBitsSet(BitWidth, ShiftAmt));
      if (KnownZero[BitWidth-ShiftAmt-1])    // New bits are known zero.
        KnownZero |= HighBits;
      else if (KnownOne[BitWidth-ShiftAmt-1])  // New bits are known one.
        KnownOne |= HighBits;
      return;
    }
    break;
  case Instruction::Sub: {
    if (ConstantInt *CLHS = dyn_cast<ConstantInt>(I->getOperand(0))) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      if (!CLHS->getValue().isNegative()) {
        unsigned NLZ = (CLHS->getValue()+1).countLeadingZeros();
        // NLZ can't be BitWidth with no sign bit
        APInt MaskV = APInt::getHighBitsSet(BitWidth, NLZ+1);
        ComputeMaskedBits(I->getOperand(1), MaskV, KnownZero2, KnownOne2,
                          TD, Depth+1);
    
        // If all of the MaskV bits are known to be zero, then we know the
        // output top bits are zero, because we now know that the output is
        // from [0-C].
        if ((KnownZero2 & MaskV) == MaskV) {
          unsigned NLZ2 = CLHS->getValue().countLeadingZeros();
          // Top bits known zero.
          KnownZero = APInt::getHighBitsSet(BitWidth, NLZ2) & Mask;
        }
      }        
    }
  }
  // fall through
  case Instruction::Add: {
    // Output known-0 bits are known if clear or set in both the low clear bits
    // common to both LHS & RHS.  For example, 8+(X<<3) is known to have the
    // low 3 bits clear.
    APInt Mask2 = APInt::getLowBitsSet(BitWidth, Mask.countTrailingOnes());
    ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero2, KnownOne2, TD,
                      Depth+1);
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    unsigned KnownZeroOut = KnownZero2.countTrailingOnes();

    ComputeMaskedBits(I->getOperand(1), Mask2, KnownZero2, KnownOne2, TD, 
                      Depth+1);
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    KnownZeroOut = std::min(KnownZeroOut, 
                            KnownZero2.countTrailingOnes());

    KnownZero |= APInt::getLowBitsSet(BitWidth, KnownZeroOut);
    return;
  }
  case Instruction::SRem:
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      APInt RA = Rem->getValue();
      if (RA.isPowerOf2() || (-RA).isPowerOf2()) {
        APInt LowBits = RA.isStrictlyPositive() ? (RA - 1) : ~RA;
        APInt Mask2 = LowBits | APInt::getSignBit(BitWidth);
        ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero2, KnownOne2, TD, 
                          Depth+1);

        // The sign of a remainder is equal to the sign of the first
        // operand (zero being positive).
        if (KnownZero2[BitWidth-1] || ((KnownZero2 & LowBits) == LowBits))
          KnownZero2 |= ~LowBits;
        else if (KnownOne2[BitWidth-1])
          KnownOne2 |= ~LowBits;

        KnownZero |= KnownZero2 & Mask;
        KnownOne |= KnownOne2 & Mask;

        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?"); 
      }
    }
    break;
  case Instruction::URem: {
    if (ConstantInt *Rem = dyn_cast<ConstantInt>(I->getOperand(1))) {
      APInt RA = Rem->getValue();
      if (RA.isPowerOf2()) {
        APInt LowBits = (RA - 1);
        APInt Mask2 = LowBits & Mask;
        KnownZero |= ~LowBits & Mask;
        ComputeMaskedBits(I->getOperand(0), Mask2, KnownZero, KnownOne, TD,
                          Depth+1);
        assert((KnownZero & KnownOne) == 0&&"Bits known to be one AND zero?");
        break;
      }
    }

    // Since the result is less than or equal to either operand, any leading
    // zero bits in either operand must also exist in the result.
    APInt AllOnes = APInt::getAllOnesValue(BitWidth);
    ComputeMaskedBits(I->getOperand(0), AllOnes, KnownZero, KnownOne,
                      TD, Depth+1);
    ComputeMaskedBits(I->getOperand(1), AllOnes, KnownZero2, KnownOne2,
                      TD, Depth+1);

    uint32_t Leaders = std::max(KnownZero.countLeadingOnes(),
                                KnownZero2.countLeadingOnes());
    KnownOne.clear();
    KnownZero = APInt::getHighBitsSet(BitWidth, Leaders) & Mask;
    break;
  }

  case Instruction::Alloca:
  case Instruction::Malloc: {
    AllocationInst *AI = cast<AllocationInst>(V);
    unsigned Align = AI->getAlignment();
    if (Align == 0 && TD) {
      if (isa<AllocaInst>(AI))
        Align = TD->getPrefTypeAlignment(AI->getType()->getElementType());
      else if (isa<MallocInst>(AI)) {
        // Malloc returns maximally aligned memory.
        Align = TD->getABITypeAlignment(AI->getType()->getElementType());
        Align =
          std::max(Align,
                   (unsigned)TD->getABITypeAlignment(Type::DoubleTy));
        Align =
          std::max(Align,
                   (unsigned)TD->getABITypeAlignment(Type::Int64Ty));
      }
    }
    
    if (Align > 0)
      KnownZero = Mask & APInt::getLowBitsSet(BitWidth,
                                              CountTrailingZeros_32(Align));
    break;
  }
  case Instruction::GetElementPtr: {
    // Analyze all of the subscripts of this getelementptr instruction
    // to determine if we can prove known low zero bits.
    APInt LocalMask = APInt::getAllOnesValue(BitWidth);
    APInt LocalKnownZero(BitWidth, 0), LocalKnownOne(BitWidth, 0);
    ComputeMaskedBits(I->getOperand(0), LocalMask,
                      LocalKnownZero, LocalKnownOne, TD, Depth+1);
    unsigned TrailZ = LocalKnownZero.countTrailingOnes();

    gep_type_iterator GTI = gep_type_begin(I);
    for (unsigned i = 1, e = I->getNumOperands(); i != e; ++i, ++GTI) {
      Value *Index = I->getOperand(i);
      if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
        // Handle struct member offset arithmetic.
        if (!TD) return;
        const StructLayout *SL = TD->getStructLayout(STy);
        unsigned Idx = cast<ConstantInt>(Index)->getZExtValue();
        uint64_t Offset = SL->getElementOffset(Idx);
        TrailZ = std::min(TrailZ,
                          CountTrailingZeros_64(Offset));
      } else {
        // Handle array index arithmetic.
        const Type *IndexedTy = GTI.getIndexedType();
        if (!IndexedTy->isSized()) return;
        unsigned GEPOpiBits = Index->getType()->getPrimitiveSizeInBits();
        uint64_t TypeSize = TD ? TD->getABITypeSize(IndexedTy) : 1;
        LocalMask = APInt::getAllOnesValue(GEPOpiBits);
        LocalKnownZero = LocalKnownOne = APInt(GEPOpiBits, 0);
        ComputeMaskedBits(Index, LocalMask,
                          LocalKnownZero, LocalKnownOne, TD, Depth+1);
        TrailZ = std::min(TrailZ,
                          CountTrailingZeros_64(TypeSize) +
                            LocalKnownZero.countTrailingOnes());
      }
    }
    
    KnownZero = APInt::getLowBitsSet(BitWidth, TrailZ) & Mask;
    break;
  }
  case Instruction::PHI: {
    PHINode *P = cast<PHINode>(I);
    // Handle the case of a simple two-predecessor recurrence PHI.
    // There's a lot more that could theoretically be done here, but
    // this is sufficient to catch some interesting cases.
    if (P->getNumIncomingValues() == 2) {
      for (unsigned i = 0; i != 2; ++i) {
        Value *L = P->getIncomingValue(i);
        Value *R = P->getIncomingValue(!i);
        User *LU = dyn_cast<User>(L);
        if (!LU)
          continue;
        unsigned Opcode = getOpcode(LU);
        // Check for operations that have the property that if
        // both their operands have low zero bits, the result
        // will have low zero bits.
        if (Opcode == Instruction::Add ||
            Opcode == Instruction::Sub ||
            Opcode == Instruction::And ||
            Opcode == Instruction::Or ||
            Opcode == Instruction::Mul) {
          Value *LL = LU->getOperand(0);
          Value *LR = LU->getOperand(1);
          // Find a recurrence.
          if (LL == I)
            L = LR;
          else if (LR == I)
            L = LL;
          else
            break;
          // Ok, we have a PHI of the form L op= R. Check for low
          // zero bits.
          APInt Mask2 = APInt::getAllOnesValue(BitWidth);
          ComputeMaskedBits(R, Mask2, KnownZero2, KnownOne2, TD, Depth+1);
          Mask2 = APInt::getLowBitsSet(BitWidth,
                                       KnownZero2.countTrailingOnes());
          KnownOne2.clear();
          KnownZero2.clear();
          ComputeMaskedBits(L, Mask2, KnownZero2, KnownOne2, TD, Depth+1);
          KnownZero = Mask &
                      APInt::getLowBitsSet(BitWidth,
                                           KnownZero2.countTrailingOnes());
          break;
        }
      }
    }
    break;
  }
  case Instruction::Call:
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::ctpop:
      case Intrinsic::ctlz:
      case Intrinsic::cttz: {
        unsigned LowBits = Log2_32(BitWidth)+1;
        KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - LowBits);
        break;
      }
      }
    }
    break;
  }
}

/// MaskedValueIsZero - Return true if 'V & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Mask is known to be zero
/// for bits that V cannot have.
bool llvm::MaskedValueIsZero(Value *V, const APInt &Mask,
                             TargetData *TD, unsigned Depth) {
  APInt KnownZero(Mask.getBitWidth(), 0), KnownOne(Mask.getBitWidth(), 0);
  ComputeMaskedBits(V, Mask, KnownZero, KnownOne, TD, Depth);
  assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
  return (KnownZero & Mask) == Mask;
}



/// ComputeNumSignBits - Return the number of times the sign bit of the
/// register is replicated into the other bits.  We know that at least 1 bit
/// is always equal to the sign bit (itself), but other cases can give us
/// information.  For example, immediately after an "ashr X, 2", we know that
/// the top 3 bits are all equal to each other, so we return 3.
///
/// 'Op' must have a scalar integer type.
///
unsigned llvm::ComputeNumSignBits(Value *V, TargetData *TD, unsigned Depth) {
  const IntegerType *Ty = cast<IntegerType>(V->getType());
  unsigned TyBits = Ty->getBitWidth();
  unsigned Tmp, Tmp2;
  unsigned FirstAnswer = 1;

  // Note that ConstantInt is handled by the general ComputeMaskedBits case
  // below.

  if (Depth == 6)
    return 1;  // Limit search depth.
  
  User *U = dyn_cast<User>(V);
  switch (getOpcode(V)) {
  default: break;
  case Instruction::SExt:
    Tmp = TyBits-cast<IntegerType>(U->getOperand(0)->getType())->getBitWidth();
    return ComputeNumSignBits(U->getOperand(0), TD, Depth+1) + Tmp;
    
  case Instruction::AShr:
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    // ashr X, C   -> adds C sign bits.
    if (ConstantInt *C = dyn_cast<ConstantInt>(U->getOperand(1))) {
      Tmp += C->getZExtValue();
      if (Tmp > TyBits) Tmp = TyBits;
    }
    return Tmp;
  case Instruction::Shl:
    if (ConstantInt *C = dyn_cast<ConstantInt>(U->getOperand(1))) {
      // shl destroys sign bits.
      Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
      if (C->getZExtValue() >= TyBits ||      // Bad shift.
          C->getZExtValue() >= Tmp) break;    // Shifted all sign bits out.
      return Tmp - C->getZExtValue();
    }
    break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:    // NOT is handled here.
    // Logical binary ops preserve the number of sign bits at the worst.
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    if (Tmp != 1) {
      Tmp2 = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
      FirstAnswer = std::min(Tmp, Tmp2);
      // We computed what we know about the sign bits as our first
      // answer. Now proceed to the generic code that uses
      // ComputeMaskedBits, and pick whichever answer is better.
    }
    break;

  case Instruction::Select:
    Tmp = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
    if (Tmp == 1) return 1;  // Early out.
    Tmp2 = ComputeNumSignBits(U->getOperand(2), TD, Depth+1);
    return std::min(Tmp, Tmp2);
    
  case Instruction::Add:
    // Add can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    if (Tmp == 1) return 1;  // Early out.
      
    // Special case decrementing a value (ADD X, -1):
    if (ConstantInt *CRHS = dyn_cast<ConstantInt>(U->getOperand(0)))
      if (CRHS->isAllOnesValue()) {
        APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
        APInt Mask = APInt::getAllOnesValue(TyBits);
        ComputeMaskedBits(U->getOperand(0), Mask, KnownZero, KnownOne, TD,
                          Depth+1);
        
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(TyBits, 1)) == Mask)
          return TyBits;
        
        // If we are subtracting one from a positive number, there is no carry
        // out of the result.
        if (KnownZero.isNegative())
          return Tmp;
      }
      
    Tmp2 = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
    if (Tmp2 == 1) return 1;
      return std::min(Tmp, Tmp2)-1;
    break;
    
  case Instruction::Sub:
    Tmp2 = ComputeNumSignBits(U->getOperand(1), TD, Depth+1);
    if (Tmp2 == 1) return 1;
      
    // Handle NEG.
    if (ConstantInt *CLHS = dyn_cast<ConstantInt>(U->getOperand(0)))
      if (CLHS->isNullValue()) {
        APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
        APInt Mask = APInt::getAllOnesValue(TyBits);
        ComputeMaskedBits(U->getOperand(1), Mask, KnownZero, KnownOne, 
                          TD, Depth+1);
        // If the input is known to be 0 or 1, the output is 0/-1, which is all
        // sign bits set.
        if ((KnownZero | APInt(TyBits, 1)) == Mask)
          return TyBits;
        
        // If the input is known to be positive (the sign bit is known clear),
        // the output of the NEG has the same number of sign bits as the input.
        if (KnownZero.isNegative())
          return Tmp2;
        
        // Otherwise, we treat this like a SUB.
      }
    
    // Sub can have at most one carry bit.  Thus we know that the output
    // is, at worst, one more bit than the inputs.
    Tmp = ComputeNumSignBits(U->getOperand(0), TD, Depth+1);
    if (Tmp == 1) return 1;  // Early out.
      return std::min(Tmp, Tmp2)-1;
    break;
  case Instruction::Trunc:
    // FIXME: it's tricky to do anything useful for this, but it is an important
    // case for targets like X86.
    break;
  }
  
  // Finally, if we can prove that the top bits of the result are 0's or 1's,
  // use this information.
  APInt KnownZero(TyBits, 0), KnownOne(TyBits, 0);
  APInt Mask = APInt::getAllOnesValue(TyBits);
  ComputeMaskedBits(V, Mask, KnownZero, KnownOne, TD, Depth);
  
  if (KnownZero.isNegative()) {        // sign bit is 0
    Mask = KnownZero;
  } else if (KnownOne.isNegative()) {  // sign bit is 1;
    Mask = KnownOne;
  } else {
    // Nothing known.
    return FirstAnswer;
  }
  
  // Okay, we know that the sign bit in Mask is set.  Use CLZ to determine
  // the number of identical bits in the top of the input value.
  Mask = ~Mask;
  Mask <<= Mask.getBitWidth()-TyBits;
  // Return # leading zeros.  We use 'min' here in case Val was zero before
  // shifting.  We don't want to return '64' as for an i32 "0".
  return std::max(FirstAnswer, std::min(TyBits, Mask.countLeadingZeros()));
}

/// CannotBeNegativeZero - Return true if we can prove that the specified FP 
/// value is never equal to -0.0.
///
/// NOTE: this function will need to be revisited when we support non-default
/// rounding modes!
///
bool llvm::CannotBeNegativeZero(const Value *V, unsigned Depth) {
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(V))
    return !CFP->getValueAPF().isNegZero();
  
  if (Depth == 6)
    return 1;  // Limit search depth.

  const Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return false;
  
  // (add x, 0.0) is guaranteed to return +0.0, not -0.0.
  if (I->getOpcode() == Instruction::Add &&
      isa<ConstantFP>(I->getOperand(1)) && 
      cast<ConstantFP>(I->getOperand(1))->isNullValue())
    return true;
    
  // sitofp and uitofp turn into +0.0 for zero.
  if (isa<SIToFPInst>(I) || isa<UIToFPInst>(I))
    return true;
  
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    // sqrt(-0.0) = -0.0, no other negative results are possible.
    if (II->getIntrinsicID() == Intrinsic::sqrt)
      return CannotBeNegativeZero(II->getOperand(1), Depth+1);
  
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    if (const Function *F = CI->getCalledFunction()) {
      if (F->isDeclaration()) {
        switch (F->getNameLen()) {
        case 3:  // abs(x) != -0.0
          if (!strcmp(F->getNameStart(), "abs")) return true;
          break;
        case 4:  // abs[lf](x) != -0.0
          if (!strcmp(F->getNameStart(), "absf")) return true;
          if (!strcmp(F->getNameStart(), "absl")) return true;
          break;
        }
      }
    }
  
  return false;
}

// This is the recursive version of BuildSubAggregate. It takes a few different
// arguments. Idxs is the index within the nested struct From that we are
// looking at now (which is of type IndexedType). IdxSkip is the number of
// indices from Idxs that should be left out when inserting into the resulting
// struct. To is the result struct built so far, new insertvalue instructions
// build on that.
Value *BuildSubAggregate(Value *From, Value* To, const Type *IndexedType,
                                 SmallVector<unsigned, 10> &Idxs,
                                 unsigned IdxSkip,
                                 Instruction *InsertBefore) {
  const llvm::StructType *STy = llvm::dyn_cast<llvm::StructType>(IndexedType);
  if (STy) {
    // Save the original To argument so we can modify it
    Value *OrigTo = To;
    // General case, the type indexed by Idxs is a struct
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      // Process each struct element recursively
      Idxs.push_back(i);
      Value *PrevTo = To;
      To = BuildSubAggregate(From, To, STy->getElementType(i), Idxs, IdxSkip,
                             InsertBefore);
      Idxs.pop_back();
      if (!To) {
        // Couldn't find any inserted value for this index? Cleanup
        while (PrevTo != OrigTo) {
          InsertValueInst* Del = cast<InsertValueInst>(PrevTo);
          PrevTo = Del->getAggregateOperand();
          Del->eraseFromParent();
        }
        // Stop processing elements
        break;
      }
    }
    // If we succesfully found a value for each of our subaggregates 
    if (To)
      return To;
  }
  // Base case, the type indexed by SourceIdxs is not a struct, or not all of
  // the struct's elements had a value that was inserted directly. In the latter
  // case, perhaps we can't determine each of the subelements individually, but
  // we might be able to find the complete struct somewhere.
  
  // Find the value that is at that particular spot
  Value *V = FindInsertedValue(From, Idxs.begin(), Idxs.end());

  if (!V)
    return NULL;

  // Insert the value in the new (sub) aggregrate
  return llvm::InsertValueInst::Create(To, V, Idxs.begin() + IdxSkip,
                                       Idxs.end(), "tmp", InsertBefore);
}

// This helper takes a nested struct and extracts a part of it (which is again a
// struct) into a new value. For example, given the struct:
// { a, { b, { c, d }, e } }
// and the indices "1, 1" this returns
// { c, d }.
//
// It does this by inserting an insertvalue for each element in the resulting
// struct, as opposed to just inserting a single struct. This will only work if
// each of the elements of the substruct are known (ie, inserted into From by an
// insertvalue instruction somewhere).
//
// All inserted insertvalue instructions are inserted before InsertBefore
Value *BuildSubAggregate(Value *From, const unsigned *idx_begin,
                         const unsigned *idx_end, Instruction *InsertBefore) {
  assert(InsertBefore && "Must have someplace to insert!");
  const Type *IndexedType = ExtractValueInst::getIndexedType(From->getType(),
                                                             idx_begin,
                                                             idx_end);
  Value *To = UndefValue::get(IndexedType);
  SmallVector<unsigned, 10> Idxs(idx_begin, idx_end);
  unsigned IdxSkip = Idxs.size();

  return BuildSubAggregate(From, To, IndexedType, Idxs, IdxSkip, InsertBefore);
}

/// FindInsertedValue - Given an aggregrate and an sequence of indices, see if
/// the scalar value indexed is already around as a register, for example if it
/// were inserted directly into the aggregrate.
///
/// If InsertBefore is not null, this function will duplicate (modified)
/// insertvalues when a part of a nested struct is extracted.
Value *llvm::FindInsertedValue(Value *V, const unsigned *idx_begin,
                         const unsigned *idx_end, Instruction *InsertBefore) {
  // Nothing to index? Just return V then (this is useful at the end of our
  // recursion)
  if (idx_begin == idx_end)
    return V;
  // We have indices, so V should have an indexable type
  assert((isa<StructType>(V->getType()) || isa<ArrayType>(V->getType()))
         && "Not looking at a struct or array?");
  assert(ExtractValueInst::getIndexedType(V->getType(), idx_begin, idx_end)
         && "Invalid indices for type?");
  const CompositeType *PTy = cast<CompositeType>(V->getType());
  
  if (isa<UndefValue>(V))
    return UndefValue::get(ExtractValueInst::getIndexedType(PTy,
                                                              idx_begin,
                                                              idx_end));
  else if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(ExtractValueInst::getIndexedType(PTy, 
                                                                     idx_begin,
                                                                     idx_end));
  else if (Constant *C = dyn_cast<Constant>(V)) {
    if (isa<ConstantArray>(C) || isa<ConstantStruct>(C))
      // Recursively process this constant
      return FindInsertedValue(C->getOperand(*idx_begin), ++idx_begin, idx_end,
                               InsertBefore);
  } else if (InsertValueInst *I = dyn_cast<InsertValueInst>(V)) {
    // Loop the indices for the insertvalue instruction in parallel with the
    // requested indices
    const unsigned *req_idx = idx_begin;
    for (const unsigned *i = I->idx_begin(), *e = I->idx_end();
         i != e; ++i, ++req_idx) {
      if (req_idx == idx_end)
        if (InsertBefore)
          // The requested index identifies a part of a nested aggregate. Handle
          // this specially. For example,
          // %A = insertvalue { i32, {i32, i32 } } undef, i32 10, 1, 0
          // %B = insertvalue { i32, {i32, i32 } } %A, i32 11, 1, 1
          // %C = extractvalue {i32, { i32, i32 } } %B, 1
          // This can be changed into
          // %A = insertvalue {i32, i32 } undef, i32 10, 0
          // %C = insertvalue {i32, i32 } %A, i32 11, 1
          // which allows the unused 0,0 element from the nested struct to be
          // removed.
          return BuildSubAggregate(V, idx_begin, req_idx, InsertBefore);
        else
          // We can't handle this without inserting insertvalues
          return 0;
      
      // This insert value inserts something else than what we are looking for.
      // See if the (aggregrate) value inserted into has the value we are
      // looking for, then.
      if (*req_idx != *i)
        return FindInsertedValue(I->getAggregateOperand(), idx_begin, idx_end,
                                 InsertBefore);
    }
    // If we end up here, the indices of the insertvalue match with those
    // requested (though possibly only partially). Now we recursively look at
    // the inserted value, passing any remaining indices.
    return FindInsertedValue(I->getInsertedValueOperand(), req_idx, idx_end,
                             InsertBefore);
  } else if (ExtractValueInst *I = dyn_cast<ExtractValueInst>(V)) {
    // If we're extracting a value from an aggregrate that was extracted from
    // something else, we can extract from that something else directly instead.
    // However, we will need to chain I's indices with the requested indices.
   
    // Calculate the number of indices required 
    unsigned size = I->getNumIndices() + (idx_end - idx_begin);
    // Allocate some space to put the new indices in
    unsigned *new_begin = new unsigned[size];
    // Auto cleanup this array
    std::auto_ptr<unsigned> newptr(new_begin);
    // Start inserting at the beginning
    unsigned *new_end = new_begin;
    // Add indices from the extract value instruction
    for (const unsigned *i = I->idx_begin(), *e = I->idx_end();
         i != e; ++i, ++new_end)
      *new_end = *i;
    
    // Add requested indices
    for (const unsigned *i = idx_begin, *e = idx_end; i != e; ++i, ++new_end)
      *new_end = *i;

    assert((unsigned)(new_end - new_begin) == size 
           && "Number of indices added not correct?");
    
    return FindInsertedValue(I->getAggregateOperand(), new_begin, new_end,
                             InsertBefore);
  }
  // Otherwise, we don't know (such as, extracting from a function return value
  // or load instruction)
  return 0;
}

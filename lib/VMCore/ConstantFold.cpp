//===- ConstantFolding.cpp - LLVM constant folder -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements folding of constants for LLVM.  This implements the
// (internal) ConstantFolding.h interface, which is used by the
// ConstantExpr::get* methods to automatically fold constants when possible.
//
// The current constant folding implementation is implemented in two pieces: the
// template-based folder for simple primitive constants like ConstantInt, and
// the special case hackery that we use to symbolically evaluate expressions
// that use ConstantExprs.
//
//===----------------------------------------------------------------------===//

#include "ConstantFolding.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                ConstantFold*Instruction Implementations
//===----------------------------------------------------------------------===//

/// CastConstantPacked - Convert the specified ConstantPacked node to the
/// specified packed type.  At this point, we know that the elements of the
/// input packed constant are all simple integer or FP values.
static Constant *CastConstantPacked(ConstantPacked *CP,
                                    const PackedType *DstTy) {
  unsigned SrcNumElts = CP->getType()->getNumElements();
  unsigned DstNumElts = DstTy->getNumElements();
  const Type *SrcEltTy = CP->getType()->getElementType();
  const Type *DstEltTy = DstTy->getElementType();
  
  // If both vectors have the same number of elements (thus, the elements
  // are the same size), perform the conversion now.
  if (SrcNumElts == DstNumElts) {
    std::vector<Constant*> Result;
    
    // If the src and dest elements are both integers, or both floats, we can 
    // just BitCast each element because the elements are the same size.
    if ((SrcEltTy->isIntegral() && DstEltTy->isIntegral()) ||
        (SrcEltTy->isFloatingPoint() && DstEltTy->isFloatingPoint())) {
      for (unsigned i = 0; i != SrcNumElts; ++i)
        Result.push_back(
          ConstantExpr::getBitCast(CP->getOperand(i), DstEltTy));
      return ConstantPacked::get(Result);
    }
    
    // If this is an int-to-fp cast ..
    if (SrcEltTy->isIntegral()) {
      // Ensure that it is int-to-fp cast
      assert(DstEltTy->isFloatingPoint());
      if (DstEltTy->getTypeID() == Type::DoubleTyID) {
        for (unsigned i = 0; i != SrcNumElts; ++i) {
          double V =
            BitsToDouble(cast<ConstantInt>(CP->getOperand(i))->getZExtValue());
          Result.push_back(ConstantFP::get(Type::DoubleTy, V));
        }
        return ConstantPacked::get(Result);
      }
      assert(DstEltTy == Type::FloatTy && "Unknown fp type!");
      for (unsigned i = 0; i != SrcNumElts; ++i) {
        float V =
        BitsToFloat(cast<ConstantInt>(CP->getOperand(i))->getZExtValue());
        Result.push_back(ConstantFP::get(Type::FloatTy, V));
      }
      return ConstantPacked::get(Result);
    }
    
    // Otherwise, this is an fp-to-int cast.
    assert(SrcEltTy->isFloatingPoint() && DstEltTy->isIntegral());
    
    if (SrcEltTy->getTypeID() == Type::DoubleTyID) {
      for (unsigned i = 0; i != SrcNumElts; ++i) {
        uint64_t V =
          DoubleToBits(cast<ConstantFP>(CP->getOperand(i))->getValue());
        Constant *C = ConstantInt::get(Type::Int64Ty, V);
        Result.push_back(ConstantExpr::getBitCast(C, DstEltTy ));
      }
      return ConstantPacked::get(Result);
    }

    assert(SrcEltTy->getTypeID() == Type::FloatTyID);
    for (unsigned i = 0; i != SrcNumElts; ++i) {
      uint32_t V = FloatToBits(cast<ConstantFP>(CP->getOperand(i))->getValue());
      Constant *C = ConstantInt::get(Type::Int32Ty, V);
      Result.push_back(ConstantExpr::getBitCast(C, DstEltTy));
    }
    return ConstantPacked::get(Result);
  }
  
  // Otherwise, this is a cast that changes element count and size.  Handle
  // casts which shrink the elements here.
  
  // FIXME: We need to know endianness to do this!
  
  return 0;
}

/// This function determines which opcode to use to fold two constant cast 
/// expressions together. It uses CastInst::isEliminableCastPair to determine
/// the opcode. Consequently its just a wrapper around that function.
/// @Determine if it is valid to fold a cast of a cast
static unsigned
foldConstantCastPair(
  unsigned opc,          ///< opcode of the second cast constant expression
  const ConstantExpr*Op, ///< the first cast constant expression
  const Type *DstTy      ///< desintation type of the first cast
) {
  assert(Op && Op->isCast() && "Can't fold cast of cast without a cast!");
  assert(DstTy && DstTy->isFirstClassType() && "Invalid cast destination type");
  assert(CastInst::isCast(opc) && "Invalid cast opcode");
  
  // The the types and opcodes for the two Cast constant expressions
  const Type *SrcTy = Op->getOperand(0)->getType();
  const Type *MidTy = Op->getType();
  Instruction::CastOps firstOp = Instruction::CastOps(Op->getOpcode());
  Instruction::CastOps secondOp = Instruction::CastOps(opc);

  // Let CastInst::isEliminableCastPair do the heavy lifting.
  return CastInst::isEliminableCastPair(firstOp, secondOp, SrcTy, MidTy, DstTy,
                                        Type::Int64Ty);
}

Constant *llvm::ConstantFoldCastInstruction(unsigned opc, const Constant *V,
                                            const Type *DestTy) {
  const Type *SrcTy = V->getType();

  if (isa<UndefValue>(V))
    return UndefValue::get(DestTy);

  // If the cast operand is a constant expression, there's a few things we can
  // do to try to simplify it.
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->isCast()) {
      // Try hard to fold cast of cast because they are often eliminable.
      if (unsigned newOpc = foldConstantCastPair(opc, CE, DestTy))
        return ConstantExpr::getCast(newOpc, CE->getOperand(0), DestTy);
    } else if (CE->getOpcode() == Instruction::GetElementPtr) {
      // If all of the indexes in the GEP are null values, there is no pointer
      // adjustment going on.  We might as well cast the source pointer.
      bool isAllNull = true;
      for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
        if (!CE->getOperand(i)->isNullValue()) {
          isAllNull = false;
          break;
        }
      if (isAllNull)
        // This is casting one pointer type to another, always BitCast
        return ConstantExpr::getPointerCast(CE->getOperand(0), DestTy);
    }
  }

  // We actually have to do a cast now. Perform the cast according to the
  // opcode specified.
  switch (opc) {
  case Instruction::FPTrunc:
  case Instruction::FPExt:
    if (const ConstantFP *FPC = dyn_cast<ConstantFP>(V))
      return ConstantFP::get(DestTy, FPC->getValue());
    return 0; // Can't fold.
  case Instruction::FPToUI: 
    if (const ConstantFP *FPC = dyn_cast<ConstantFP>(V))
      return ConstantIntegral::get(DestTy,(uint64_t) FPC->getValue());
    return 0; // Can't fold.
  case Instruction::FPToSI:
    if (const ConstantFP *FPC = dyn_cast<ConstantFP>(V))
      return ConstantIntegral::get(DestTy,(int64_t) FPC->getValue());
    return 0; // Can't fold.
  case Instruction::IntToPtr:   //always treated as unsigned
    if (V->isNullValue())       // Is it an integral null value?
      return ConstantPointerNull::get(cast<PointerType>(DestTy));
    return 0;                   // Other pointer types cannot be casted
  case Instruction::PtrToInt:   // always treated as unsigned
    if (V->isNullValue())       // is it a null pointer value?
      return ConstantIntegral::get(DestTy, 0);
    return 0;                   // Other pointer types cannot be casted
  case Instruction::UIToFP:
    if (const ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V))
      return ConstantFP::get(DestTy, double(CI->getZExtValue()));
    return 0;
  case Instruction::SIToFP:
    if (const ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V))
      return ConstantFP::get(DestTy, double(CI->getSExtValue()));
    return 0;
  case Instruction::ZExt:
    if (const ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V))
      return ConstantInt::get(DestTy, CI->getZExtValue());
    return 0;
  case Instruction::SExt:
    if (const ConstantIntegral *CI = dyn_cast<ConstantIntegral>(V))
      return ConstantInt::get(DestTy, CI->getSExtValue());
    return 0;
  case Instruction::Trunc:
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) // Can't trunc a bool
      return ConstantIntegral::get(DestTy, CI->getZExtValue());
    return 0;
  case Instruction::BitCast:
    if (SrcTy == DestTy) 
      return (Constant*)V; // no-op cast
    
    // Check to see if we are casting a pointer to an aggregate to a pointer to
    // the first element.  If so, return the appropriate GEP instruction.
    if (const PointerType *PTy = dyn_cast<PointerType>(V->getType()))
      if (const PointerType *DPTy = dyn_cast<PointerType>(DestTy)) {
        std::vector<Value*> IdxList;
        IdxList.push_back(Constant::getNullValue(Type::Int32Ty));
        const Type *ElTy = PTy->getElementType();
        while (ElTy != DPTy->getElementType()) {
          if (const StructType *STy = dyn_cast<StructType>(ElTy)) {
            if (STy->getNumElements() == 0) break;
            ElTy = STy->getElementType(0);
            IdxList.push_back(Constant::getNullValue(Type::Int32Ty));
          } else if (const SequentialType *STy = 
                     dyn_cast<SequentialType>(ElTy)) {
            if (isa<PointerType>(ElTy)) break;  // Can't index into pointers!
            ElTy = STy->getElementType();
            IdxList.push_back(IdxList[0]);
          } else {
            break;
          }
        }

        if (ElTy == DPTy->getElementType())
          return ConstantExpr::getGetElementPtr(
              const_cast<Constant*>(V),IdxList);
      }
        
    // Handle casts from one packed constant to another.  We know that the src 
    // and dest type have the same size (otherwise its an illegal cast).
    if (const PackedType *DestPTy = dyn_cast<PackedType>(DestTy)) {
      if (const PackedType *SrcTy = dyn_cast<PackedType>(V->getType())) {
        assert(DestPTy->getBitWidth() == SrcTy->getBitWidth() &&
               "Not cast between same sized vectors!");
        // First, check for null and undef
        if (isa<ConstantAggregateZero>(V))
          return Constant::getNullValue(DestTy);
        if (isa<UndefValue>(V))
          return UndefValue::get(DestTy);

        if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(V)) {
          // This is a cast from a ConstantPacked of one type to a 
          // ConstantPacked of another type.  Check to see if all elements of 
          // the input are simple.
          bool AllSimpleConstants = true;
          for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i) {
            if (!isa<ConstantInt>(CP->getOperand(i)) &&
                !isa<ConstantFP>(CP->getOperand(i))) {
              AllSimpleConstants = false;
              break;
            }
          }
              
          // If all of the elements are simple constants, we can fold this.
          if (AllSimpleConstants)
            return CastConstantPacked(const_cast<ConstantPacked*>(CP), DestPTy);
        }
      }
    }

    // Finally, implement bitcast folding now.   The code below doesn't handle
    // bitcast right.
    if (isa<ConstantPointerNull>(V))  // ptr->ptr cast.
      return ConstantPointerNull::get(cast<PointerType>(DestTy));

    // Handle integral constant input.
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      // Integral -> Integral, must be changing sign.
      if (DestTy->isIntegral())
        return ConstantInt::get(DestTy, CI->getZExtValue());

      if (DestTy->isFloatingPoint()) {
        if (DestTy == Type::FloatTy)
          return ConstantFP::get(DestTy, BitsToFloat(CI->getZExtValue()));
        assert(DestTy == Type::DoubleTy && "Unknown FP type!");
        return ConstantFP::get(DestTy, BitsToDouble(CI->getZExtValue()));
      }
      // Otherwise, can't fold this (packed?)
      return 0;
    }
      
    // Handle ConstantFP input.
    if (const ConstantFP *FP = dyn_cast<ConstantFP>(V)) {
      // FP -> Integral.
      if (DestTy->isIntegral()) {
        if (DestTy == Type::Int32Ty)
          return ConstantInt::get(DestTy, FloatToBits(FP->getValue()));
        assert(DestTy == Type::Int64Ty && 
               "Incorrect integer  type for bitcast!");
        return ConstantInt::get(DestTy, DoubleToBits(FP->getValue()));
      }
    }
    return 0;
  default:
    assert(!"Invalid CE CastInst opcode");
    break;
  }

  assert(0 && "Failed to cast constant expression");
  return 0;
}

Constant *llvm::ConstantFoldSelectInstruction(const Constant *Cond,
                                              const Constant *V1,
                                              const Constant *V2) {
  if (const ConstantBool *CB = dyn_cast<ConstantBool>(Cond))
    return const_cast<Constant*>(CB->getValue() ? V1 : V2);

  if (isa<UndefValue>(V1)) return const_cast<Constant*>(V2);
  if (isa<UndefValue>(V2)) return const_cast<Constant*>(V1);
  if (isa<UndefValue>(Cond)) return const_cast<Constant*>(V1);
  if (V1 == V2) return const_cast<Constant*>(V1);
  return 0;
}

Constant *llvm::ConstantFoldExtractElementInstruction(const Constant *Val,
                                                      const Constant *Idx) {
  if (isa<UndefValue>(Val))  // ee(undef, x) -> undef
    return UndefValue::get(cast<PackedType>(Val->getType())->getElementType());
  if (Val->isNullValue())  // ee(zero, x) -> zero
    return Constant::getNullValue(
                          cast<PackedType>(Val->getType())->getElementType());
  
  if (const ConstantPacked *CVal = dyn_cast<ConstantPacked>(Val)) {
    if (const ConstantInt *CIdx = dyn_cast<ConstantInt>(Idx)) {
      return const_cast<Constant*>(CVal->getOperand(CIdx->getZExtValue()));
    } else if (isa<UndefValue>(Idx)) {
      // ee({w,x,y,z}, undef) -> w (an arbitrary value).
      return const_cast<Constant*>(CVal->getOperand(0));
    }
  }
  return 0;
}

Constant *llvm::ConstantFoldInsertElementInstruction(const Constant *Val,
                                                     const Constant *Elt,
                                                     const Constant *Idx) {
  const ConstantInt *CIdx = dyn_cast<ConstantInt>(Idx);
  if (!CIdx) return 0;
  uint64_t idxVal = CIdx->getZExtValue();
  if (isa<UndefValue>(Val)) { 
    // Insertion of scalar constant into packed undef
    // Optimize away insertion of undef
    if (isa<UndefValue>(Elt))
      return const_cast<Constant*>(Val);
    // Otherwise break the aggregate undef into multiple undefs and do
    // the insertion
    unsigned numOps = 
      cast<PackedType>(Val->getType())->getNumElements();
    std::vector<Constant*> Ops; 
    Ops.reserve(numOps);
    for (unsigned i = 0; i < numOps; ++i) {
      const Constant *Op =
        (i == idxVal) ? Elt : UndefValue::get(Elt->getType());
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantPacked::get(Ops);
  }
  if (isa<ConstantAggregateZero>(Val)) {
    // Insertion of scalar constant into packed aggregate zero
    // Optimize away insertion of zero
    if (Elt->isNullValue())
      return const_cast<Constant*>(Val);
    // Otherwise break the aggregate zero into multiple zeros and do
    // the insertion
    unsigned numOps = 
      cast<PackedType>(Val->getType())->getNumElements();
    std::vector<Constant*> Ops; 
    Ops.reserve(numOps);
    for (unsigned i = 0; i < numOps; ++i) {
      const Constant *Op =
        (i == idxVal) ? Elt : Constant::getNullValue(Elt->getType());
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantPacked::get(Ops);
  }
  if (const ConstantPacked *CVal = dyn_cast<ConstantPacked>(Val)) {
    // Insertion of scalar constant into packed constant
    std::vector<Constant*> Ops; 
    Ops.reserve(CVal->getNumOperands());
    for (unsigned i = 0; i < CVal->getNumOperands(); ++i) {
      const Constant *Op =
        (i == idxVal) ? Elt : cast<Constant>(CVal->getOperand(i));
      Ops.push_back(const_cast<Constant*>(Op));
    }
    return ConstantPacked::get(Ops);
  }
  return 0;
}

Constant *llvm::ConstantFoldShuffleVectorInstruction(const Constant *V1,
                                                     const Constant *V2,
                                                     const Constant *Mask) {
  // TODO:
  return 0;
}

/// EvalVectorOp - Given two packed constants and a function pointer, apply the
/// function pointer to each element pair, producing a new ConstantPacked
/// constant.
static Constant *EvalVectorOp(const ConstantPacked *V1, 
                              const ConstantPacked *V2,
                              Constant *(*FP)(Constant*, Constant*)) {
  std::vector<Constant*> Res;
  for (unsigned i = 0, e = V1->getNumOperands(); i != e; ++i)
    Res.push_back(FP(const_cast<Constant*>(V1->getOperand(i)),
                     const_cast<Constant*>(V2->getOperand(i))));
  return ConstantPacked::get(Res);
}

Constant *llvm::ConstantFoldBinaryInstruction(unsigned Opcode,
                                              const Constant *C1,
                                              const Constant *C2) {
  // Handle UndefValue up front
  if (isa<UndefValue>(C1) || isa<UndefValue>(C2)) {
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Xor:
      return UndefValue::get(C1->getType());
    case Instruction::Mul:
    case Instruction::And:
      return Constant::getNullValue(C1->getType());
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
      if (!isa<UndefValue>(C2))                    // undef / X -> 0
        return Constant::getNullValue(C1->getType());
      return const_cast<Constant*>(C2);            // X / undef -> undef
    case Instruction::Or:                          // X | undef -> -1
      return ConstantInt::getAllOnesValue(C1->getType());
    case Instruction::LShr:
      if (isa<UndefValue>(C2) && isa<UndefValue>(C1))
        return const_cast<Constant*>(C1);           // undef lshr undef -> undef
      return Constant::getNullValue(C1->getType()); // X lshr undef -> 0
                                                    // undef lshr X -> 0
    case Instruction::AShr:
      if (!isa<UndefValue>(C2))
        return const_cast<Constant*>(C1);           // undef ashr X --> undef
      else if (isa<UndefValue>(C1)) 
        return const_cast<Constant*>(C1);           // undef ashr undef -> undef
      else
        return const_cast<Constant*>(C1);           // X ashr undef --> X
    case Instruction::Shl:
      // undef << X -> 0   or   X << undef -> 0
      return Constant::getNullValue(C1->getType());
    }
  }

  if (const ConstantExpr *CE1 = dyn_cast<ConstantExpr>(C1)) {
    if (isa<ConstantExpr>(C2)) {
      // There are many possible foldings we could do here.  We should probably
      // at least fold add of a pointer with an integer into the appropriate
      // getelementptr.  This will improve alias analysis a bit.
    } else {
      // Just implement a couple of simple identities.
      switch (Opcode) {
      case Instruction::Add:
        if (C2->isNullValue()) return const_cast<Constant*>(C1);  // X + 0 == X
        break;
      case Instruction::Sub:
        if (C2->isNullValue()) return const_cast<Constant*>(C1);  // X - 0 == X
        break;
      case Instruction::Mul:
        if (C2->isNullValue()) return const_cast<Constant*>(C2);  // X * 0 == 0
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(C2))
          if (CI->getZExtValue() == 1)
            return const_cast<Constant*>(C1);                     // X * 1 == X
        break;
      case Instruction::UDiv:
      case Instruction::SDiv:
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(C2))
          if (CI->getZExtValue() == 1)
            return const_cast<Constant*>(C1);                     // X / 1 == X
        break;
      case Instruction::URem:
      case Instruction::SRem:
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(C2))
          if (CI->getZExtValue() == 1)
            return Constant::getNullValue(CI->getType());         // X % 1 == 0
        break;
      case Instruction::And:
        if (cast<ConstantIntegral>(C2)->isAllOnesValue())
          return const_cast<Constant*>(C1);                       // X & -1 == X
        if (C2->isNullValue()) return const_cast<Constant*>(C2);  // X & 0 == 0
        if (CE1->isCast() && isa<GlobalValue>(CE1->getOperand(0))) {
          GlobalValue *CPR = cast<GlobalValue>(CE1->getOperand(0));

          // Functions are at least 4-byte aligned.  If and'ing the address of a
          // function with a constant < 4, fold it to zero.
          if (const ConstantInt *CI = dyn_cast<ConstantInt>(C2))
            if (CI->getZExtValue() < 4 && isa<Function>(CPR))
              return Constant::getNullValue(CI->getType());
        }
        break;
      case Instruction::Or:
        if (C2->isNullValue()) return const_cast<Constant*>(C1);  // X | 0 == X
        if (cast<ConstantIntegral>(C2)->isAllOnesValue())
          return const_cast<Constant*>(C2);  // X | -1 == -1
        break;
      case Instruction::Xor:
        if (C2->isNullValue()) return const_cast<Constant*>(C1);  // X ^ 0 == X
        break;
      }
    }
  } else if (isa<ConstantExpr>(C2)) {
    // If C2 is a constant expr and C1 isn't, flop them around and fold the
    // other way if possible.
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Mul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      // No change of opcode required.
      return ConstantFoldBinaryInstruction(Opcode, C2, C1);

    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Sub:
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    default:  // These instructions cannot be flopped around.
      return 0;
    }
  }

  // At this point we know neither constant is an UndefValue nor a ConstantExpr
  // so look at directly computing the 
  if (const ConstantBool *CB1 = dyn_cast<ConstantBool>(C1)) {
    if (const ConstantBool *CB2 = dyn_cast<ConstantBool>(C2)) {
      switch (Opcode) {
        default:
          break;
        case Instruction::And:
          return ConstantBool::get(CB1->getValue() & CB2->getValue());
        case Instruction::Or:
          return ConstantBool::get(CB1->getValue() | CB2->getValue());
        case Instruction::Xor:
          return ConstantBool::get(CB1->getValue() ^ CB2->getValue());
      }
    }
  } else if (const ConstantInt *CI1 = dyn_cast<ConstantInt>(C1)) {
    if (const ConstantInt *CI2 = dyn_cast<ConstantInt>(C2)) {
      uint64_t C1Val = CI1->getZExtValue();
      uint64_t C2Val = CI2->getZExtValue();
      switch (Opcode) {
      default:
        break;
      case Instruction::Add:     
        return ConstantInt::get(C1->getType(), C1Val + C2Val);
      case Instruction::Sub:     
        return ConstantInt::get(C1->getType(), C1Val - C2Val);
      case Instruction::Mul:     
        return ConstantInt::get(C1->getType(), C1Val * C2Val);
      case Instruction::UDiv:
        if (CI2->isNullValue())                  // X / 0 -> can't fold
          return 0;
        return ConstantInt::get(C1->getType(), C1Val / C2Val);
      case Instruction::SDiv:
        if (CI2->isNullValue()) return 0;        // X / 0 -> can't fold
        if (CI2->isAllOnesValue() &&
            (((CI1->getType()->getPrimitiveSizeInBits() == 64) && 
              (CI1->getSExtValue() == INT64_MIN)) ||
             (CI1->getSExtValue() == -CI1->getSExtValue())))
          return 0;                              // MIN_INT / -1 -> overflow
        return ConstantInt::get(C1->getType(), 
                                CI1->getSExtValue() / CI2->getSExtValue());
      case Instruction::URem:    
        if (C2->isNullValue()) return 0;         // X / 0 -> can't fold
        return ConstantInt::get(C1->getType(), C1Val % C2Val);
      case Instruction::SRem:    
        if (CI2->isNullValue()) return 0;        // X % 0 -> can't fold
        if (CI2->isAllOnesValue() &&              
            (((CI1->getType()->getPrimitiveSizeInBits() == 64) && 
              (CI1->getSExtValue() == INT64_MIN)) ||
             (CI1->getSExtValue() == -CI1->getSExtValue())))
          return 0;                              // MIN_INT % -1 -> overflow
        return ConstantInt::get(C1->getType(), 
                                CI1->getSExtValue() % CI2->getSExtValue());
      case Instruction::And:
        return ConstantInt::get(C1->getType(), C1Val & C2Val);
      case Instruction::Or:
        return ConstantInt::get(C1->getType(), C1Val | C2Val);
      case Instruction::Xor:
        return ConstantInt::get(C1->getType(), C1Val ^ C2Val);
      case Instruction::Shl:
        return ConstantInt::get(C1->getType(), C1Val << C2Val);
      case Instruction::LShr:
        return ConstantInt::get(C1->getType(), C1Val >> C2Val);
      case Instruction::AShr:
        return ConstantInt::get(C1->getType(), 
                                CI1->getSExtValue() >> C2Val);
      }
    }
  } else if (const ConstantFP *CFP1 = dyn_cast<ConstantFP>(C1)) {
    if (const ConstantFP *CFP2 = dyn_cast<ConstantFP>(C2)) {
      double C1Val = CFP1->getValue();
      double C2Val = CFP2->getValue();
      switch (Opcode) {
      default:                   
        break;
      case Instruction::Add: 
        return ConstantFP::get(CFP1->getType(), C1Val + C2Val);
      case Instruction::Sub:     
        return ConstantFP::get(CFP1->getType(), C1Val - C2Val);
      case Instruction::Mul:     
        return ConstantFP::get(CFP1->getType(), C1Val * C2Val);
      case Instruction::FDiv:
        if (CFP2->isExactlyValue(0.0)) 
          return ConstantFP::get(CFP1->getType(),
                                 std::numeric_limits<double>::infinity());
        if (CFP2->isExactlyValue(-0.0))
          return ConstantFP::get(CFP1->getType(),
                                 -std::numeric_limits<double>::infinity());
        return ConstantFP::get(CFP1->getType(), C1Val / C2Val);
      case Instruction::FRem:
        if (CFP2->isNullValue()) 
          return 0;
        return ConstantFP::get(CFP1->getType(), std::fmod(C1Val, C2Val));
      }
    }
  } else if (const ConstantPacked *CP1 = dyn_cast<ConstantPacked>(C1)) {
    if (const ConstantPacked *CP2 = dyn_cast<ConstantPacked>(C2)) {
      switch (Opcode) {
        default:
          break;
        case Instruction::Add: 
          return EvalVectorOp(CP1, CP2, ConstantExpr::getAdd);
        case Instruction::Sub: 
          return EvalVectorOp(CP1, CP2, ConstantExpr::getSub);
        case Instruction::Mul: 
          return EvalVectorOp(CP1, CP2, ConstantExpr::getMul);
        case Instruction::UDiv:
          return EvalVectorOp(CP1, CP2, ConstantExpr::getUDiv);
        case Instruction::SDiv:
          return EvalVectorOp(CP1, CP2, ConstantExpr::getSDiv);
        case Instruction::FDiv:
          return EvalVectorOp(CP1, CP2, ConstantExpr::getFDiv);
        case Instruction::URem:
          return EvalVectorOp(CP1, CP2, ConstantExpr::getURem);
        case Instruction::SRem:
          return EvalVectorOp(CP1, CP2, ConstantExpr::getSRem);
        case Instruction::FRem:
          return EvalVectorOp(CP1, CP2, ConstantExpr::getFRem);
        case Instruction::And: 
          return EvalVectorOp(CP1, CP2, ConstantExpr::getAnd);
        case Instruction::Or:  
          return EvalVectorOp(CP1, CP2, ConstantExpr::getOr);
        case Instruction::Xor: 
          return EvalVectorOp(CP1, CP2, ConstantExpr::getXor);
      }
    }
  }

  // We don't know how to fold this
  return 0;
}

/// isZeroSizedType - This type is zero sized if its an array or structure of
/// zero sized types.  The only leaf zero sized type is an empty structure.
static bool isMaybeZeroSizedType(const Type *Ty) {
  if (isa<OpaqueType>(Ty)) return true;  // Can't say.
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {

    // If all of elements have zero size, this does too.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
      if (!isMaybeZeroSizedType(STy->getElementType(i))) return false;
    return true;

  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    return isMaybeZeroSizedType(ATy->getElementType());
  }
  return false;
}

/// IdxCompare - Compare the two constants as though they were getelementptr
/// indices.  This allows coersion of the types to be the same thing.
///
/// If the two constants are the "same" (after coersion), return 0.  If the
/// first is less than the second, return -1, if the second is less than the
/// first, return 1.  If the constants are not integral, return -2.
///
static int IdxCompare(Constant *C1, Constant *C2, const Type *ElTy) {
  if (C1 == C2) return 0;

  // Ok, we found a different index.  Are either of the operands ConstantExprs?
  // If so, we can't do anything with them.
  if (!isa<ConstantInt>(C1) || !isa<ConstantInt>(C2))
    return -2; // don't know!

  // Ok, we have two differing integer indices.  Sign extend them to be the same
  // type.  Long is always big enough, so we use it.
  if (C1->getType() != Type::Int64Ty)
    C1 = ConstantExpr::getSExt(C1, Type::Int64Ty);

  if (C2->getType() != Type::Int64Ty)
    C1 = ConstantExpr::getSExt(C2, Type::Int64Ty);

  if (C1 == C2) return 0;  // They are equal

  // If the type being indexed over is really just a zero sized type, there is
  // no pointer difference being made here.
  if (isMaybeZeroSizedType(ElTy))
    return -2; // dunno.

  // If they are really different, now that they are the same type, then we
  // found a difference!
  if (cast<ConstantInt>(C1)->getSExtValue() < 
      cast<ConstantInt>(C2)->getSExtValue())
    return -1;
  else
    return 1;
}

/// evaluatFCmpeRelation - This function determines if there is anything we can
/// decide about the two constants provided.  This doesn't need to handle simple
/// things like ConstantFP comparisons, but should instead handle ConstantExprs.
/// If we can determine that the two constants have a particular relation to 
/// each other, we should return the corresponding FCmpInst predicate, 
/// otherwise return FCmpInst::BAD_FCMP_PREDICATE. This is used below in
/// ConstantFoldCompareInstruction.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider ConstantFP
/// to be the simplest, and ConstantExprs to be the most complex.
static FCmpInst::Predicate evaluateFCmpRelation(const Constant *V1, 
                                                const Constant *V2) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare values of different types!");
  // Handle degenerate case quickly
  if (V1 == V2) return FCmpInst::FCMP_OEQ;

  if (!isa<ConstantExpr>(V1)) {
    if (!isa<ConstantExpr>(V2)) {
      // We distilled thisUse the standard constant folder for a few cases
      ConstantBool *R = 0;
      Constant *C1 = const_cast<Constant*>(V1);
      Constant *C2 = const_cast<Constant*>(V2);
      R = dyn_cast<ConstantBool>(
                             ConstantExpr::getFCmp(FCmpInst::FCMP_OEQ, C1, C2));
      if (R && R->getValue()) 
        return FCmpInst::FCMP_OEQ;
      R = dyn_cast<ConstantBool>(
                             ConstantExpr::getFCmp(FCmpInst::FCMP_OLT, C1, C2));
      if (R && R->getValue()) 
        return FCmpInst::FCMP_OLT;
      R = dyn_cast<ConstantBool>(
                             ConstantExpr::getFCmp(FCmpInst::FCMP_OGT, C1, C2));
      if (R && R->getValue()) 
        return FCmpInst::FCMP_OGT;

      // Nothing more we can do
      return FCmpInst::BAD_FCMP_PREDICATE;
    }
    
    // If the first operand is simple and second is ConstantExpr, swap operands.
    FCmpInst::Predicate SwappedRelation = evaluateFCmpRelation(V2, V1);
    if (SwappedRelation != FCmpInst::BAD_FCMP_PREDICATE)
      return FCmpInst::getSwappedPredicate(SwappedRelation);
  } else {
    // Ok, the LHS is known to be a constantexpr.  The RHS can be any of a
    // constantexpr or a simple constant.
    const ConstantExpr *CE1 = cast<ConstantExpr>(V1);
    switch (CE1->getOpcode()) {
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
      // We might be able to do something with these but we don't right now.
      break;
    default:
      break;
    }
  }
  // There are MANY other foldings that we could perform here.  They will
  // probably be added on demand, as they seem needed.
  return FCmpInst::BAD_FCMP_PREDICATE;
}

/// evaluateICmpRelation - This function determines if there is anything we can
/// decide about the two constants provided.  This doesn't need to handle simple
/// things like integer comparisons, but should instead handle ConstantExprs
/// and GlobalValues.  If we can determine that the two constants have a
/// particular relation to each other, we should return the corresponding ICmp
/// predicate, otherwise return ICmpInst::BAD_ICMP_PREDICATE.
///
/// To simplify this code we canonicalize the relation so that the first
/// operand is always the most "complex" of the two.  We consider simple
/// constants (like ConstantInt) to be the simplest, followed by
/// GlobalValues, followed by ConstantExpr's (the most complex).
///
static ICmpInst::Predicate evaluateICmpRelation(const Constant *V1, 
                                                const Constant *V2,
                                                bool isSigned) {
  assert(V1->getType() == V2->getType() &&
         "Cannot compare different types of values!");
  if (V1 == V2) return ICmpInst::ICMP_EQ;

  if (!isa<ConstantExpr>(V1) && !isa<GlobalValue>(V1)) {
    if (!isa<GlobalValue>(V2) && !isa<ConstantExpr>(V2)) {
      // We distilled this down to a simple case, use the standard constant
      // folder.
      ConstantBool *R = 0;
      Constant *C1 = const_cast<Constant*>(V1);
      Constant *C2 = const_cast<Constant*>(V2);
      ICmpInst::Predicate pred = ICmpInst::ICMP_EQ;
      R = dyn_cast<ConstantBool>(ConstantExpr::getICmp(pred, C1, C2));
      if (R && R->getValue()) 
        return pred;
      pred = isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
      R = dyn_cast<ConstantBool>(ConstantExpr::getICmp(pred, C1, C2));
      if (R && R->getValue())
        return pred;
      pred = isSigned ?  ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
      R = dyn_cast<ConstantBool>(ConstantExpr::getICmp(pred, C1, C2));
      if (R && R->getValue())
        return pred;
      
      // If we couldn't figure it out, bail.
      return ICmpInst::BAD_ICMP_PREDICATE;
    }
    
    // If the first operand is simple, swap operands.
    ICmpInst::Predicate SwappedRelation = 
      evaluateICmpRelation(V2, V1, isSigned);
    if (SwappedRelation != ICmpInst::BAD_ICMP_PREDICATE)
      return ICmpInst::getSwappedPredicate(SwappedRelation);

  } else if (const GlobalValue *CPR1 = dyn_cast<GlobalValue>(V1)) {
    if (isa<ConstantExpr>(V2)) {  // Swap as necessary.
      ICmpInst::Predicate SwappedRelation = 
        evaluateICmpRelation(V2, V1, isSigned);
      if (SwappedRelation != ICmpInst::BAD_ICMP_PREDICATE)
        return ICmpInst::getSwappedPredicate(SwappedRelation);
      else
        return ICmpInst::BAD_ICMP_PREDICATE;
    }

    // Now we know that the RHS is a GlobalValue or simple constant,
    // which (since the types must match) means that it's a ConstantPointerNull.
    if (const GlobalValue *CPR2 = dyn_cast<GlobalValue>(V2)) {
      if (!CPR1->hasExternalWeakLinkage() || !CPR2->hasExternalWeakLinkage())
        return ICmpInst::ICMP_NE;
    } else {
      // GlobalVals can never be null.
      assert(isa<ConstantPointerNull>(V2) && "Canonicalization guarantee!");
      if (!CPR1->hasExternalWeakLinkage())
        return ICmpInst::ICMP_NE;
    }
  } else {
    // Ok, the LHS is known to be a constantexpr.  The RHS can be any of a
    // constantexpr, a CPR, or a simple constant.
    const ConstantExpr *CE1 = cast<ConstantExpr>(V1);
    const Constant *CE1Op0 = CE1->getOperand(0);

    switch (CE1->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      break; // We can't evaluate floating point casts or truncations.

    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::PtrToInt:
      // If the cast is not actually changing bits, and the second operand is a
      // null pointer, do the comparison with the pre-casted value.
      if (V2->isNullValue() &&
          (isa<PointerType>(CE1->getType()) || CE1->getType()->isIntegral())) {
        bool sgnd = CE1->getOpcode() == Instruction::ZExt ? false :
          (CE1->getOpcode() == Instruction::SExt ? true :
           (CE1->getOpcode() == Instruction::PtrToInt ? false : isSigned));
        return evaluateICmpRelation(
            CE1Op0, Constant::getNullValue(CE1Op0->getType()), sgnd);
      }

      // If the dest type is a pointer type, and the RHS is a constantexpr cast
      // from the same type as the src of the LHS, evaluate the inputs.  This is
      // important for things like "icmp eq (cast 4 to int*), (cast 5 to int*)",
      // which happens a lot in compilers with tagged integers.
      if (const ConstantExpr *CE2 = dyn_cast<ConstantExpr>(V2))
        if (CE2->isCast() && isa<PointerType>(CE1->getType()) &&
            CE1->getOperand(0)->getType() == CE2->getOperand(0)->getType() &&
            CE1->getOperand(0)->getType()->isIntegral()) {
          bool sgnd = CE1->getOpcode() == Instruction::ZExt ? false :
            (CE1->getOpcode() == Instruction::SExt ? true :
             (CE1->getOpcode() == Instruction::PtrToInt ? false : isSigned));
          return evaluateICmpRelation(CE1->getOperand(0), CE2->getOperand(0),
              sgnd);
        }
      break;

    case Instruction::GetElementPtr:
      // Ok, since this is a getelementptr, we know that the constant has a
      // pointer type.  Check the various cases.
      if (isa<ConstantPointerNull>(V2)) {
        // If we are comparing a GEP to a null pointer, check to see if the base
        // of the GEP equals the null pointer.
        if (const GlobalValue *GV = dyn_cast<GlobalValue>(CE1Op0)) {
          if (GV->hasExternalWeakLinkage())
            // Weak linkage GVals could be zero or not. We're comparing that
            // to null pointer so its greater-or-equal
            return isSigned ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_UGE;
          else 
            // If its not weak linkage, the GVal must have a non-zero address
            // so the result is greater-than
            return isSigned ? ICmpInst::ICMP_SGT :  ICmpInst::ICMP_UGT;
        } else if (isa<ConstantPointerNull>(CE1Op0)) {
          // If we are indexing from a null pointer, check to see if we have any
          // non-zero indices.
          for (unsigned i = 1, e = CE1->getNumOperands(); i != e; ++i)
            if (!CE1->getOperand(i)->isNullValue())
              // Offsetting from null, must not be equal.
              return isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
          // Only zero indexes from null, must still be zero.
          return ICmpInst::ICMP_EQ;
        }
        // Otherwise, we can't really say if the first operand is null or not.
      } else if (const GlobalValue *CPR2 = dyn_cast<GlobalValue>(V2)) {
        if (isa<ConstantPointerNull>(CE1Op0)) {
          if (CPR2->hasExternalWeakLinkage())
            // Weak linkage GVals could be zero or not. We're comparing it to
            // a null pointer, so its less-or-equal
            return isSigned ? ICmpInst::ICMP_SLE : ICmpInst::ICMP_ULE;
          else
            // If its not weak linkage, the GVal must have a non-zero address
            // so the result is less-than
            return isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
        } else if (const GlobalValue *CPR1 = dyn_cast<GlobalValue>(CE1Op0)) {
          if (CPR1 == CPR2) {
            // If this is a getelementptr of the same global, then it must be
            // different.  Because the types must match, the getelementptr could
            // only have at most one index, and because we fold getelementptr's
            // with a single zero index, it must be nonzero.
            assert(CE1->getNumOperands() == 2 &&
                   !CE1->getOperand(1)->isNullValue() &&
                   "Suprising getelementptr!");
            return isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
          } else {
            // If they are different globals, we don't know what the value is,
            // but they can't be equal.
            return ICmpInst::ICMP_NE;
          }
        }
      } else {
        const ConstantExpr *CE2 = cast<ConstantExpr>(V2);
        const Constant *CE2Op0 = CE2->getOperand(0);

        // There are MANY other foldings that we could perform here.  They will
        // probably be added on demand, as they seem needed.
        switch (CE2->getOpcode()) {
        default: break;
        case Instruction::GetElementPtr:
          // By far the most common case to handle is when the base pointers are
          // obviously to the same or different globals.
          if (isa<GlobalValue>(CE1Op0) && isa<GlobalValue>(CE2Op0)) {
            if (CE1Op0 != CE2Op0) // Don't know relative ordering, but not equal
              return ICmpInst::ICMP_NE;
            // Ok, we know that both getelementptr instructions are based on the
            // same global.  From this, we can precisely determine the relative
            // ordering of the resultant pointers.
            unsigned i = 1;

            // Compare all of the operands the GEP's have in common.
            gep_type_iterator GTI = gep_type_begin(CE1);
            for (;i != CE1->getNumOperands() && i != CE2->getNumOperands();
                 ++i, ++GTI)
              switch (IdxCompare(CE1->getOperand(i), CE2->getOperand(i),
                                 GTI.getIndexedType())) {
              case -1: return isSigned ? ICmpInst::ICMP_SLT:ICmpInst::ICMP_ULT;
              case 1:  return isSigned ? ICmpInst::ICMP_SGT:ICmpInst::ICMP_UGT;
              case -2: return ICmpInst::BAD_ICMP_PREDICATE;
              }

            // Ok, we ran out of things they have in common.  If any leftovers
            // are non-zero then we have a difference, otherwise we are equal.
            for (; i < CE1->getNumOperands(); ++i)
              if (!CE1->getOperand(i)->isNullValue())
                if (isa<ConstantIntegral>(CE1->getOperand(i)))
                  return isSigned ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
                else
                  return ICmpInst::BAD_ICMP_PREDICATE; // Might be equal.

            for (; i < CE2->getNumOperands(); ++i)
              if (!CE2->getOperand(i)->isNullValue())
                if (isa<ConstantIntegral>(CE2->getOperand(i)))
                  return isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
                else
                  return ICmpInst::BAD_ICMP_PREDICATE; // Might be equal.
            return ICmpInst::ICMP_EQ;
          }
        }
      }
    default:
      break;
    }
  }

  return ICmpInst::BAD_ICMP_PREDICATE;
}

Constant *llvm::ConstantFoldCompareInstruction(unsigned short pred, 
                                               const Constant *C1, 
                                               const Constant *C2) {

  // Handle some degenerate cases first
  if (isa<UndefValue>(C1) || isa<UndefValue>(C2))
    return UndefValue::get(Type::BoolTy);

  // icmp eq/ne(null,GV) -> false/true
  if (C1->isNullValue()) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C2))
      if (!GV->hasExternalWeakLinkage()) // External weak GV can be null
        if (pred == ICmpInst::ICMP_EQ)
          return ConstantBool::getFalse();
        else if (pred == ICmpInst::ICMP_NE)
          return ConstantBool::getTrue();
  // icmp eq/ne(GV,null) -> false/true
  } else if (C2->isNullValue()) {
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C1))
      if (!GV->hasExternalWeakLinkage()) // External weak GV can be null
        if (pred == ICmpInst::ICMP_EQ)
          return ConstantBool::getFalse();
        else if (pred == ICmpInst::ICMP_NE)
          return ConstantBool::getTrue();
  }

  if (isa<ConstantBool>(C1) && isa<ConstantBool>(C2)) {
    bool C1Val = cast<ConstantBool>(C1)->getValue();
    bool C2Val = cast<ConstantBool>(C2)->getValue();
    switch (pred) {
    default: assert(0 && "Invalid ICmp Predicate"); return 0;
    case ICmpInst::ICMP_EQ: return ConstantBool::get(C1Val == C2Val);
    case ICmpInst::ICMP_NE: return ConstantBool::get(C1Val != C2Val);
    case ICmpInst::ICMP_ULT:return ConstantBool::get(C1Val <  C2Val);
    case ICmpInst::ICMP_UGT:return ConstantBool::get(C1Val >  C2Val);
    case ICmpInst::ICMP_ULE:return ConstantBool::get(C1Val <= C2Val);
    case ICmpInst::ICMP_UGE:return ConstantBool::get(C1Val >= C2Val);
    case ICmpInst::ICMP_SLT:return ConstantBool::get(C1Val <  C2Val);
    case ICmpInst::ICMP_SGT:return ConstantBool::get(C1Val >  C2Val);
    case ICmpInst::ICMP_SLE:return ConstantBool::get(C1Val <= C2Val);
    case ICmpInst::ICMP_SGE:return ConstantBool::get(C1Val >= C2Val);
    }
  } else if (isa<ConstantInt>(C1) && isa<ConstantInt>(C2)) {
    if (ICmpInst::isSignedPredicate(ICmpInst::Predicate(pred))) {
      int64_t V1 = cast<ConstantInt>(C1)->getSExtValue();
      int64_t V2 = cast<ConstantInt>(C2)->getSExtValue();
      switch (pred) {
      default: assert(0 && "Invalid ICmp Predicate"); return 0;
      case ICmpInst::ICMP_SLT:return ConstantBool::get(V1 <  V2);
      case ICmpInst::ICMP_SGT:return ConstantBool::get(V1 >  V2);
      case ICmpInst::ICMP_SLE:return ConstantBool::get(V1 <= V2);
      case ICmpInst::ICMP_SGE:return ConstantBool::get(V1 >= V2);
      }
    } else {
      uint64_t V1 = cast<ConstantInt>(C1)->getZExtValue();
      uint64_t V2 = cast<ConstantInt>(C2)->getZExtValue();
      switch (pred) {
      default: assert(0 && "Invalid ICmp Predicate"); return 0;
      case ICmpInst::ICMP_EQ: return ConstantBool::get(V1 == V2);
      case ICmpInst::ICMP_NE: return ConstantBool::get(V1 != V2);
      case ICmpInst::ICMP_ULT:return ConstantBool::get(V1 <  V2);
      case ICmpInst::ICMP_UGT:return ConstantBool::get(V1 >  V2);
      case ICmpInst::ICMP_ULE:return ConstantBool::get(V1 <= V2);
      case ICmpInst::ICMP_UGE:return ConstantBool::get(V1 >= V2);
      }
    }
  } else if (isa<ConstantFP>(C1) && isa<ConstantFP>(C2)) {
    double C1Val = cast<ConstantFP>(C1)->getValue();
    double C2Val = cast<ConstantFP>(C2)->getValue();
    switch (pred) {
    default: assert(0 && "Invalid FCmp Predicate"); return 0;
    case FCmpInst::FCMP_FALSE: return ConstantBool::getFalse();
    case FCmpInst::FCMP_TRUE:  return ConstantBool::getTrue();
    case FCmpInst::FCMP_UNO:
    case FCmpInst::FCMP_ORD:   break; // Can't fold these
    case FCmpInst::FCMP_UEQ:
    case FCmpInst::FCMP_OEQ:   return ConstantBool::get(C1Val == C2Val);
    case FCmpInst::FCMP_ONE:
    case FCmpInst::FCMP_UNE:   return ConstantBool::get(C1Val != C2Val);
    case FCmpInst::FCMP_OLT: 
    case FCmpInst::FCMP_ULT:   return ConstantBool::get(C1Val < C2Val);
    case FCmpInst::FCMP_UGT:
    case FCmpInst::FCMP_OGT:   return ConstantBool::get(C1Val > C2Val);
    case FCmpInst::FCMP_OLE:
    case FCmpInst::FCMP_ULE:   return ConstantBool::get(C1Val <= C2Val);
    case FCmpInst::FCMP_UGE:
    case FCmpInst::FCMP_OGE:   return ConstantBool::get(C1Val >= C2Val);
    }
  } else if (const ConstantPacked *CP1 = dyn_cast<ConstantPacked>(C1)) {
    if (const ConstantPacked *CP2 = dyn_cast<ConstantPacked>(C2)) {
      if (pred == FCmpInst::FCMP_OEQ || pred == FCmpInst::FCMP_UEQ) {
        for (unsigned i = 0, e = CP1->getNumOperands(); i != e; ++i) {
          Constant *C= ConstantExpr::getFCmp(FCmpInst::FCMP_OEQ,
              const_cast<Constant*>(CP1->getOperand(i)),
              const_cast<Constant*>(CP2->getOperand(i)));
          if (ConstantBool *CB = dyn_cast<ConstantBool>(C))
            return CB;
        }
        // Otherwise, could not decide from any element pairs.
        return 0;
      } else if (pred == ICmpInst::ICMP_EQ) {
        for (unsigned i = 0, e = CP1->getNumOperands(); i != e; ++i) {
          Constant *C = ConstantExpr::getICmp(ICmpInst::ICMP_EQ,
              const_cast<Constant*>(CP1->getOperand(i)),
              const_cast<Constant*>(CP2->getOperand(i)));
          if (ConstantBool *CB = dyn_cast<ConstantBool>(C))
            return CB;
        }
        // Otherwise, could not decide from any element pairs.
        return 0;
      }
    }
  }

  if (C1->getType()->isFloatingPoint()) {
    switch (evaluateFCmpRelation(C1, C2)) {
    default: assert(0 && "Unknown relation!");
    case FCmpInst::FCMP_UNO:
    case FCmpInst::FCMP_ORD:
    case FCmpInst::FCMP_UEQ:
    case FCmpInst::FCMP_UNE:
    case FCmpInst::FCMP_ULT:
    case FCmpInst::FCMP_UGT:
    case FCmpInst::FCMP_ULE:
    case FCmpInst::FCMP_UGE:
    case FCmpInst::FCMP_TRUE:
    case FCmpInst::FCMP_FALSE:
    case FCmpInst::BAD_FCMP_PREDICATE:
      break; // Couldn't determine anything about these constants.
    case FCmpInst::FCMP_OEQ: // We know that C1 == C2
      return ConstantBool::get(
          pred == FCmpInst::FCMP_UEQ || pred == FCmpInst::FCMP_OEQ ||
          pred == FCmpInst::FCMP_ULE || pred == FCmpInst::FCMP_OLE ||
          pred == FCmpInst::FCMP_UGE || pred == FCmpInst::FCMP_OGE);
    case FCmpInst::FCMP_OLT: // We know that C1 < C2
      return ConstantBool::get(
          pred == FCmpInst::FCMP_UNE || pred == FCmpInst::FCMP_ONE ||
          pred == FCmpInst::FCMP_ULT || pred == FCmpInst::FCMP_OLT ||
          pred == FCmpInst::FCMP_ULE || pred == FCmpInst::FCMP_OLE);
    case FCmpInst::FCMP_OGT: // We know that C1 > C2
      return ConstantBool::get(
          pred == FCmpInst::FCMP_UNE || pred == FCmpInst::FCMP_ONE ||
          pred == FCmpInst::FCMP_UGT || pred == FCmpInst::FCMP_OGT ||
          pred == FCmpInst::FCMP_UGE || pred == FCmpInst::FCMP_OGE);
    case FCmpInst::FCMP_OLE: // We know that C1 <= C2
      // We can only partially decide this relation.
      if (pred == FCmpInst::FCMP_UGT || pred == FCmpInst::FCMP_OGT) 
        return ConstantBool::getFalse();
      if (pred == FCmpInst::FCMP_ULT || pred == FCmpInst::FCMP_OLT) 
        return ConstantBool::getTrue();
      break;
    case FCmpInst::FCMP_OGE: // We known that C1 >= C2
      // We can only partially decide this relation.
      if (pred == FCmpInst::FCMP_ULT || pred == FCmpInst::FCMP_OLT) 
        return ConstantBool::getFalse();
      if (pred == FCmpInst::FCMP_UGT || pred == FCmpInst::FCMP_OGT) 
        return ConstantBool::getTrue();
      break;
    case ICmpInst::ICMP_NE: // We know that C1 != C2
      // We can only partially decide this relation.
      if (pred == FCmpInst::FCMP_OEQ || pred == FCmpInst::FCMP_UEQ) 
        return ConstantBool::getFalse();
      if (pred == FCmpInst::FCMP_ONE || pred == FCmpInst::FCMP_UNE) 
        return ConstantBool::getTrue();
      break;
    }
  } else {
    // Evaluate the relation between the two constants, per the predicate.
    switch (evaluateICmpRelation(C1, C2, CmpInst::isSigned(pred))) {
    default: assert(0 && "Unknown relational!");
    case ICmpInst::BAD_ICMP_PREDICATE:
      break;  // Couldn't determine anything about these constants.
    case ICmpInst::ICMP_EQ:   // We know the constants are equal!
      // If we know the constants are equal, we can decide the result of this
      // computation precisely.
      return ConstantBool::get(pred == ICmpInst::ICMP_EQ  ||
                               pred == ICmpInst::ICMP_ULE ||
                               pred == ICmpInst::ICMP_SLE ||
                               pred == ICmpInst::ICMP_UGE ||
                               pred == ICmpInst::ICMP_SGE);
    case ICmpInst::ICMP_ULT:
      // If we know that C1 < C2, we can decide the result of this computation
      // precisely.
      return ConstantBool::get(pred == ICmpInst::ICMP_ULT ||
                               pred == ICmpInst::ICMP_NE  ||
                               pred == ICmpInst::ICMP_ULE);
    case ICmpInst::ICMP_SLT:
      // If we know that C1 < C2, we can decide the result of this computation
      // precisely.
      return ConstantBool::get(pred == ICmpInst::ICMP_SLT ||
                               pred == ICmpInst::ICMP_NE  ||
                               pred == ICmpInst::ICMP_SLE);
    case ICmpInst::ICMP_UGT:
      // If we know that C1 > C2, we can decide the result of this computation
      // precisely.
      return ConstantBool::get(pred == ICmpInst::ICMP_UGT ||
                               pred == ICmpInst::ICMP_NE  ||
                               pred == ICmpInst::ICMP_UGE);
    case ICmpInst::ICMP_SGT:
      // If we know that C1 > C2, we can decide the result of this computation
      // precisely.
      return ConstantBool::get(pred == ICmpInst::ICMP_SGT ||
                               pred == ICmpInst::ICMP_NE  ||
                               pred == ICmpInst::ICMP_SGE);
    case ICmpInst::ICMP_ULE:
      // If we know that C1 <= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_UGT) return ConstantBool::getFalse();
      if (pred == ICmpInst::ICMP_ULT) return ConstantBool::getTrue();
      break;
    case ICmpInst::ICMP_SLE:
      // If we know that C1 <= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_SGT) return ConstantBool::getFalse();
      if (pred == ICmpInst::ICMP_SLT) return ConstantBool::getTrue();
      break;

    case ICmpInst::ICMP_UGE:
      // If we know that C1 >= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_ULT) return ConstantBool::getFalse();
      if (pred == ICmpInst::ICMP_UGT) return ConstantBool::getTrue();
      break;
    case ICmpInst::ICMP_SGE:
      // If we know that C1 >= C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_SLT) return ConstantBool::getFalse();
      if (pred == ICmpInst::ICMP_SGT) return ConstantBool::getTrue();
      break;

    case ICmpInst::ICMP_NE:
      // If we know that C1 != C2, we can only partially decide this relation.
      if (pred == ICmpInst::ICMP_EQ) return ConstantBool::getFalse();
      if (pred == ICmpInst::ICMP_NE) return ConstantBool::getTrue();
      break;
    }

    if (!isa<ConstantExpr>(C1) && isa<ConstantExpr>(C2)) {
      // If C2 is a constant expr and C1 isn't, flop them around and fold the
      // other way if possible.
      switch (pred) {
      case ICmpInst::ICMP_EQ:
      case ICmpInst::ICMP_NE:
        // No change of predicate required.
        return ConstantFoldCompareInstruction(pred, C2, C1);

      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_ULE:
      case ICmpInst::ICMP_SLE:
      case ICmpInst::ICMP_UGE:
      case ICmpInst::ICMP_SGE:
        // Change the predicate as necessary to swap the operands.
        pred = ICmpInst::getSwappedPredicate((ICmpInst::Predicate)pred);
        return ConstantFoldCompareInstruction(pred, C2, C1);

      default:  // These predicates cannot be flopped around.
        break;
      }
    }
  }
  return 0;
}

Constant *llvm::ConstantFoldGetElementPtr(const Constant *C,
                                          const std::vector<Value*> &IdxList) {
  if (IdxList.size() == 0 ||
      (IdxList.size() == 1 && cast<Constant>(IdxList[0])->isNullValue()))
    return const_cast<Constant*>(C);

  if (isa<UndefValue>(C)) {
    const Type *Ty = GetElementPtrInst::getIndexedType(C->getType(), IdxList,
                                                       true);
    assert(Ty != 0 && "Invalid indices for GEP!");
    return UndefValue::get(PointerType::get(Ty));
  }

  Constant *Idx0 = cast<Constant>(IdxList[0]);
  if (C->isNullValue()) {
    bool isNull = true;
    for (unsigned i = 0, e = IdxList.size(); i != e; ++i)
      if (!cast<Constant>(IdxList[i])->isNullValue()) {
        isNull = false;
        break;
      }
    if (isNull) {
      const Type *Ty = GetElementPtrInst::getIndexedType(C->getType(), IdxList,
                                                         true);
      assert(Ty != 0 && "Invalid indices for GEP!");
      return ConstantPointerNull::get(PointerType::get(Ty));
    }

    if (IdxList.size() == 1) {
      const Type *ElTy = cast<PointerType>(C->getType())->getElementType();
      if (uint32_t ElSize = ElTy->getPrimitiveSize()) {
        // gep null, C is equal to C*sizeof(nullty).  If nullty is a known llvm
        // type, we can statically fold this.
        Constant *R = ConstantInt::get(Type::Int32Ty, ElSize);
        // We know R is unsigned, Idx0 is signed because it must be an index
        // through a sequential type (gep pointer operand) which is always
        // signed.
        R = ConstantExpr::getSExtOrBitCast(R, Idx0->getType());
        R = ConstantExpr::getMul(R, Idx0); // signed multiply
        // R is a signed integer, C is the GEP pointer so -> IntToPtr
        return ConstantExpr::getIntToPtr(R, C->getType());
      }
    }
  }

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(const_cast<Constant*>(C))) {
    // Combine Indices - If the source pointer to this getelementptr instruction
    // is a getelementptr instruction, combine the indices of the two
    // getelementptr instructions into a single instruction.
    //
    if (CE->getOpcode() == Instruction::GetElementPtr) {
      const Type *LastTy = 0;
      for (gep_type_iterator I = gep_type_begin(CE), E = gep_type_end(CE);
           I != E; ++I)
        LastTy = *I;

      if ((LastTy && isa<ArrayType>(LastTy)) || Idx0->isNullValue()) {
        std::vector<Value*> NewIndices;
        NewIndices.reserve(IdxList.size() + CE->getNumOperands());
        for (unsigned i = 1, e = CE->getNumOperands()-1; i != e; ++i)
          NewIndices.push_back(CE->getOperand(i));

        // Add the last index of the source with the first index of the new GEP.
        // Make sure to handle the case when they are actually different types.
        Constant *Combined = CE->getOperand(CE->getNumOperands()-1);
        // Otherwise it must be an array.
        if (!Idx0->isNullValue()) {
          const Type *IdxTy = Combined->getType();
          if (IdxTy != Idx0->getType()) {
            Constant *C1 = ConstantExpr::getSExtOrBitCast(Idx0, Type::Int64Ty);
            Constant *C2 = ConstantExpr::getSExtOrBitCast(Combined, 
                                                          Type::Int64Ty);
            Combined = ConstantExpr::get(Instruction::Add, C1, C2);
          } else {
            Combined =
              ConstantExpr::get(Instruction::Add, Idx0, Combined);
          }
        }

        NewIndices.push_back(Combined);
        NewIndices.insert(NewIndices.end(), IdxList.begin()+1, IdxList.end());
        return ConstantExpr::getGetElementPtr(CE->getOperand(0), NewIndices);
      }
    }

    // Implement folding of:
    //    int* getelementptr ([2 x int]* cast ([3 x int]* %X to [2 x int]*),
    //                        long 0, long 0)
    // To: int* getelementptr ([3 x int]* %X, long 0, long 0)
    //
    if (CE->isCast() && IdxList.size() > 1 && Idx0->isNullValue())
      if (const PointerType *SPT =
          dyn_cast<PointerType>(CE->getOperand(0)->getType()))
        if (const ArrayType *SAT = dyn_cast<ArrayType>(SPT->getElementType()))
          if (const ArrayType *CAT =
        dyn_cast<ArrayType>(cast<PointerType>(C->getType())->getElementType()))
            if (CAT->getElementType() == SAT->getElementType())
              return ConstantExpr::getGetElementPtr(
                      (Constant*)CE->getOperand(0), IdxList);
  }
  return 0;
}


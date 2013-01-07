//===- BasicTargetTransformInfo.cpp - Basic target-independent TTI impl ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the implementation of a basic TargetTransformInfo pass
/// predicated on the target abstractions present in the target independent
/// code generator. It uses these (primarily TargetLowering) to model as much
/// of the TTI query interface as possible. It is included by most targets so
/// that they can specialize only a small subset of the query space.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "basictti"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/TargetTransformInfo.h"
#include <utility>

using namespace llvm;

namespace {

class BasicTTI : public ImmutablePass, public TargetTransformInfo {
  const TargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) const;

public:
  BasicTTI() : ImmutablePass(ID), TLI(0) {
    llvm_unreachable("This pass cannot be directly constructed");
  }

  BasicTTI(const TargetLowering *TLI) : ImmutablePass(ID), TLI(TLI) {
    initializeBasicTTIPass(*PassRegistry::getPassRegistry());
  }

  virtual void initializePass() {
    pushTTIStack(this);
  }

  virtual void finalizePass() {
    popTTIStack();
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    TargetTransformInfo::getAnalysisUsage(AU);
  }

  /// Pass identification.
  static char ID;

  /// Provide necessary pointer adjustments for the two base classes.
  virtual void *getAdjustedAnalysisPointer(const void *ID) {
    if (ID == &TargetTransformInfo::ID)
      return (TargetTransformInfo*)this;
    return this;
  }

  /// \name Scalar TTI Implementations
  /// @{

  virtual bool isLegalAddImmediate(int64_t imm) const;
  virtual bool isLegalICmpImmediate(int64_t imm) const;
  virtual bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale) const;
  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const;
  virtual bool isTypeLegal(Type *Ty) const;
  virtual unsigned getJumpBufAlignment() const;
  virtual unsigned getJumpBufSize() const;
  virtual bool shouldBuildLookupTables() const;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  virtual unsigned getNumberOfRegisters(bool Vector) const;
  virtual unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty) const;
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp,
                                  int Index, Type *SubTp) const;
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const;
  virtual unsigned getCFInstrCost(unsigned Opcode) const;
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) const;
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) const;
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const;
  virtual unsigned getIntrinsicInstrCost(Intrinsic::ID, Type *RetTy,
                                         ArrayRef<Type*> Tys) const;
  virtual unsigned getNumberOfParts(Type *Tp) const;

  /// @}
};

}

INITIALIZE_AG_PASS(BasicTTI, TargetTransformInfo, "basictti",
                   "Target independent code generator's TTI", true, true, false)
char BasicTTI::ID = 0;

ImmutablePass *
llvm::createBasicTargetTransformInfoPass(const TargetLowering *TLI) {
  return new BasicTTI(TLI);
}


bool BasicTTI::isLegalAddImmediate(int64_t imm) const {
  return TLI->isLegalAddImmediate(imm);
}

bool BasicTTI::isLegalICmpImmediate(int64_t imm) const {
  return TLI->isLegalICmpImmediate(imm);
}

bool BasicTTI::isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale) const {
  AddrMode AM;
  AM.BaseGV = BaseGV;
  AM.BaseOffs = BaseOffset;
  AM.HasBaseReg = HasBaseReg;
  AM.Scale = Scale;
  return TLI->isLegalAddressingMode(AM, Ty);
}

bool BasicTTI::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return TLI->isTruncateFree(Ty1, Ty2);
}

bool BasicTTI::isTypeLegal(Type *Ty) const {
  EVT T = TLI->getValueType(Ty);
  return TLI->isTypeLegal(T);
}

unsigned BasicTTI::getJumpBufAlignment() const {
  return TLI->getJumpBufAlignment();
}

unsigned BasicTTI::getJumpBufSize() const {
  return TLI->getJumpBufSize();
}

bool BasicTTI::shouldBuildLookupTables() const {
  return TLI->supportJumpTables() &&
      (TLI->isOperationLegalOrCustom(ISD::BR_JT, MVT::Other) ||
       TLI->isOperationLegalOrCustom(ISD::BRIND, MVT::Other));
}

//===----------------------------------------------------------------------===//
//
// Calls used by the vectorizers.
//
//===----------------------------------------------------------------------===//

unsigned BasicTTI::getScalarizationOverhead(Type *Ty, bool Insert,
                                            bool Extract) const {
  assert (Ty->isVectorTy() && "Can only scalarize vectors");
  unsigned Cost = 0;

  for (int i = 0, e = Ty->getVectorNumElements(); i < e; ++i) {
    if (Insert)
      Cost += TopTTI->getVectorInstrCost(Instruction::InsertElement, Ty, i);
    if (Extract)
      Cost += TopTTI->getVectorInstrCost(Instruction::ExtractElement, Ty, i);
  }

  return Cost;
}

unsigned BasicTTI::getNumberOfRegisters(bool Vector) const {
  return 1;
}

unsigned BasicTTI::getArithmeticInstrCost(unsigned Opcode, Type *Ty) const {
  // Check if any of the operands are vector operands.
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Ty);

  if (TLI->isOperationLegalOrPromote(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1.
    // If the type is split to multiple registers, assume that thre is some
    // overhead to this.
    // TODO: Once we have extract/insert subvector cost we need to use them.
    if (LT.first > 1)
      return LT.first * 2;
    return LT.first * 1;
  }

  if (!TLI->isOperationExpand(ISD, LT.second)) {
    // If the operation is custom lowered then assume
    // thare the code is twice as expensive.
    return LT.first * 2;
  }

  // Else, assume that we need to scalarize this op.
  if (Ty->isVectorTy()) {
    unsigned Num = Ty->getVectorNumElements();
    unsigned Cost = TopTTI->getArithmeticInstrCost(Opcode, Ty->getScalarType());
    // return the cost of multiple scalar invocation plus the cost of inserting
    // and extracting the values.
    return getScalarizationOverhead(Ty, true, true) + Num * Cost;
  }

  // We don't know anything about this scalar instruction.
  return 1;
}

unsigned BasicTTI::getShuffleCost(ShuffleKind Kind, Type *Tp, int Index,
                                  Type *SubTp) const {
  return 1;
}

unsigned BasicTTI::getCastInstrCost(unsigned Opcode, Type *Dst,
                                    Type *Src) const {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  std::pair<unsigned, MVT> SrcLT = TLI->getTypeLegalizationCost(Src);
  std::pair<unsigned, MVT> DstLT = TLI->getTypeLegalizationCost(Dst);

  // Handle scalar conversions.
  if (!Src->isVectorTy() && !Dst->isVectorTy()) {

    // Scalar bitcasts are usually free.
    if (Opcode == Instruction::BitCast)
      return 0;

    if (Opcode == Instruction::Trunc &&
        TLI->isTruncateFree(SrcLT.second, DstLT.second))
      return 0;

    if (Opcode == Instruction::ZExt &&
        TLI->isZExtFree(SrcLT.second, DstLT.second))
      return 0;

    // Just check the op cost. If the operation is legal then assume it costs 1.
    if (!TLI->isOperationExpand(ISD, DstLT.second))
      return  1;

    // Assume that illegal scalar instruction are expensive.
    return 4;
  }

  // Check vector-to-vector casts.
  if (Dst->isVectorTy() && Src->isVectorTy()) {

    // If the cast is between same-sized registers, then the check is simple.
    if (SrcLT.first == DstLT.first &&
        SrcLT.second.getSizeInBits() == DstLT.second.getSizeInBits()) {

      // Bitcast between types that are legalized to the same type are free.
      if (Opcode == Instruction::BitCast || Opcode == Instruction::Trunc)
        return 0;

      // Assume that Zext is done using AND.
      if (Opcode == Instruction::ZExt)
        return 1;

      // Assume that sext is done using SHL and SRA.
      if (Opcode == Instruction::SExt)
        return 2;

      // Just check the op cost. If the operation is legal then assume it costs
      // 1 and multiply by the type-legalization overhead.
      if (!TLI->isOperationExpand(ISD, DstLT.second))
        return SrcLT.first * 1;
    }

    // If we are converting vectors and the operation is illegal, or
    // if the vectors are legalized to different types, estimate the
    // scalarization costs.
    unsigned Num = Dst->getVectorNumElements();
    unsigned Cost = TopTTI->getCastInstrCost(Opcode, Dst->getScalarType(),
                                             Src->getScalarType());

    // Return the cost of multiple scalar invocation plus the cost of
    // inserting and extracting the values.
    return getScalarizationOverhead(Dst, true, true) + Num * Cost;
  }

  // We already handled vector-to-vector and scalar-to-scalar conversions. This
  // is where we handle bitcast between vectors and scalars. We need to assume
  //  that the conversion is scalarized in one way or another.
  if (Opcode == Instruction::BitCast)
    // Illegal bitcasts are done by storing and loading from a stack slot.
    return (Src->isVectorTy()? getScalarizationOverhead(Src, false, true):0) +
           (Dst->isVectorTy()? getScalarizationOverhead(Dst, true, false):0);

  llvm_unreachable("Unhandled cast");
 }

unsigned BasicTTI::getCFInstrCost(unsigned Opcode) const {
  // Branches are assumed to be predicted.
  return 0;
}

unsigned BasicTTI::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) const {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  // Selects on vectors are actually vector selects.
  if (ISD == ISD::SELECT) {
    assert(CondTy && "CondTy must exist");
    if (CondTy->isVectorTy())
      ISD = ISD::VSELECT;
  }

  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(ValTy);

  if (!TLI->isOperationExpand(ISD, LT.second)) {
    // The operation is legal. Assume it costs 1. Multiply
    // by the type-legalization overhead.
    return LT.first * 1;
  }

  // Otherwise, assume that the cast is scalarized.
  if (ValTy->isVectorTy()) {
    unsigned Num = ValTy->getVectorNumElements();
    if (CondTy)
      CondTy = CondTy->getScalarType();
    unsigned Cost = TopTTI->getCmpSelInstrCost(Opcode, ValTy->getScalarType(),
                                               CondTy);

    // Return the cost of multiple scalar invocation plus the cost of inserting
    // and extracting the values.
    return getScalarizationOverhead(ValTy, true, false) + Num * Cost;
  }

  // Unknown scalar opcode.
  return 1;
}

unsigned BasicTTI::getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) const {
  return 1;
}

unsigned BasicTTI::getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) const {
  assert(!Src->isVoidTy() && "Invalid type");
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Src);

  // Assume that all loads of legal types cost 1.
  return LT.first;
}

unsigned BasicTTI::getIntrinsicInstrCost(Intrinsic::ID, Type *RetTy,
                                         ArrayRef<Type *> Tys) const {
  // assume that we need to scalarize this intrinsic.
  unsigned ScalarizationCost = 0;
  unsigned ScalarCalls = 1;
  if (RetTy->isVectorTy()) {
    ScalarizationCost = getScalarizationOverhead(RetTy, true, false);
    ScalarCalls = std::max(ScalarCalls, RetTy->getVectorNumElements());
  }
  for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
    if (Tys[i]->isVectorTy()) {
      ScalarizationCost += getScalarizationOverhead(Tys[i], false, true);
      ScalarCalls = std::max(ScalarCalls, RetTy->getVectorNumElements());
    }
  }
  return ScalarCalls + ScalarizationCost;
}

unsigned BasicTTI::getNumberOfParts(Type *Tp) const {
  std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(Tp);
  return LT.first;
}

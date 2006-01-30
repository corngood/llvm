//===-- TargetLowering.cpp - Implement the TargetLowering class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

TargetLowering::TargetLowering(TargetMachine &tm)
  : TM(tm), TD(TM.getTargetData()) {
  assert(ISD::BUILTIN_OP_END <= 128 &&
         "Fixed size array in TargetLowering is not large enough!");
  // All operations default to being supported.
  memset(OpActions, 0, sizeof(OpActions));

  IsLittleEndian = TD.isLittleEndian();
  ShiftAmountTy = SetCCResultTy = PointerTy = getValueType(TD.getIntPtrType());
  ShiftAmtHandling = Undefined;
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
  maxStoresPerMemSet = maxStoresPerMemCpy = maxStoresPerMemMove = 8;
  allowUnalignedMemoryAccesses = false;
  UseUnderscoreSetJmpLongJmp = false;
  IntDivIsCheap = false;
  Pow2DivIsCheap = false;
  StackPointerRegisterToSaveRestore = 0;
  SchedPreferenceInfo = SchedulingForLatency;
}

TargetLowering::~TargetLowering() {}

/// setValueTypeAction - Set the action for a particular value type.  This
/// assumes an action has not already been set for this value type.
static void SetValueTypeAction(MVT::ValueType VT,
                               TargetLowering::LegalizeAction Action,
                               TargetLowering &TLI,
                               MVT::ValueType *TransformToType,
                        TargetLowering::ValueTypeActionImpl &ValueTypeActions) {
  ValueTypeActions.setTypeAction(VT, Action);
  if (Action == TargetLowering::Promote) {
    MVT::ValueType PromoteTo;
    if (VT == MVT::f32)
      PromoteTo = MVT::f64;
    else {
      unsigned LargerReg = VT+1;
      while (!TLI.isTypeLegal((MVT::ValueType)LargerReg)) {
        ++LargerReg;
        assert(MVT::isInteger((MVT::ValueType)LargerReg) &&
               "Nothing to promote to??");
      }
      PromoteTo = (MVT::ValueType)LargerReg;
    }

    assert(MVT::isInteger(VT) == MVT::isInteger(PromoteTo) &&
           MVT::isFloatingPoint(VT) == MVT::isFloatingPoint(PromoteTo) &&
           "Can only promote from int->int or fp->fp!");
    assert(VT < PromoteTo && "Must promote to a larger type!");
    TransformToType[VT] = PromoteTo;
  } else if (Action == TargetLowering::Expand) {
    assert((VT == MVT::Vector || MVT::isInteger(VT)) && VT > MVT::i8 &&
           "Cannot expand this type: target must support SOME integer reg!");
    // Expand to the next smaller integer type!
    TransformToType[VT] = (MVT::ValueType)(VT-1);
  }
}


/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLowering::computeRegisterProperties() {
  assert(MVT::LAST_VALUETYPE <= 32 &&
         "Too many value types for ValueTypeActions to hold!");

  // Everything defaults to one.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i)
    NumElementsForVT[i] = 1;

  // Find the largest integer register class.
  unsigned LargestIntReg = MVT::i128;
  for (; RegClassForVT[LargestIntReg] == 0; --LargestIntReg)
    assert(LargestIntReg != MVT::i1 && "No integer registers defined!");

  // Every integer value type larger than this largest register takes twice as
  // many registers to represent as the previous ValueType.
  unsigned ExpandedReg = LargestIntReg; ++LargestIntReg;
  for (++ExpandedReg; MVT::isInteger((MVT::ValueType)ExpandedReg);++ExpandedReg)
    NumElementsForVT[ExpandedReg] = 2*NumElementsForVT[ExpandedReg-1];

  // Inspect all of the ValueType's possible, deciding how to process them.
  for (unsigned IntReg = MVT::i1; IntReg <= MVT::i128; ++IntReg)
    // If we are expanding this type, expand it!
    if (getNumElements((MVT::ValueType)IntReg) != 1)
      SetValueTypeAction((MVT::ValueType)IntReg, Expand, *this, TransformToType,
                         ValueTypeActions);
    else if (!isTypeLegal((MVT::ValueType)IntReg))
      // Otherwise, if we don't have native support, we must promote to a
      // larger type.
      SetValueTypeAction((MVT::ValueType)IntReg, Promote, *this,
                         TransformToType, ValueTypeActions);
    else
      TransformToType[(MVT::ValueType)IntReg] = (MVT::ValueType)IntReg;

  // If the target does not have native support for F32, promote it to F64.
  if (!isTypeLegal(MVT::f32))
    SetValueTypeAction(MVT::f32, Promote, *this,
                       TransformToType, ValueTypeActions);
  else
    TransformToType[MVT::f32] = MVT::f32;
  
  // Set MVT::Vector to always be Expanded
  SetValueTypeAction(MVT::Vector, Expand, *this, TransformToType, 
                     ValueTypeActions);

  assert(isTypeLegal(MVT::f64) && "Target does not support FP?");
  TransformToType[MVT::f64] = MVT::f64;
}

const char *TargetLowering::getTargetNodeName(unsigned Opcode) const {
  return NULL;
}

bool TargetLowering::isMaskedValueZeroForTargetNode(const SDOperand &Op,
                                                    uint64_t Mask,
                                                     MVIZFnPtr MVIZ) const {
  return false;
}

std::vector<unsigned> TargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint) const {
  // Scan to see if this constraint is a register name.
  const MRegisterInfo *RI = TM.getRegisterInfo();
  for (unsigned i = 1, e = RI->getNumRegs(); i != e; ++i) {
    if (const char *Name = RI->get(i).Name)
      if (StringsEqualNoCase(Constraint, Name))
        return std::vector<unsigned>(1, i);
  }

  // Not a physreg, must not be a register reference or something.
  return std::vector<unsigned>();
}


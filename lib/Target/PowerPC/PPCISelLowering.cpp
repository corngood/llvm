//===-- PPCISelLowering.cpp - PPC DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPCISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "PPCISelLowering.h"
#include "PPCTargetMachine.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

PPCTargetLowering::PPCTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
    
  // Fold away setcc operations if possible.
  setSetCCIsExpensive();
  setPow2DivIsCheap();
  
  // Use _setjmp/_longjmp instead of setjmp/longjmp.
  setUseUnderscoreSetJmpLongJmp(true);
    
  // Set up the register classes.
  addRegisterClass(MVT::i32, PPC::GPRCRegisterClass);
  addRegisterClass(MVT::f32, PPC::F4RCRegisterClass);
  addRegisterClass(MVT::f64, PPC::F8RCRegisterClass);
  
  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);

  // PowerPC has no intrinsics for these particular operations
  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);
  
  // PowerPC has an i16 but no i8 (or i1) SEXTLOAD
  setOperationAction(ISD::SEXTLOAD, MVT::i1, Expand);
  setOperationAction(ISD::SEXTLOAD, MVT::i8, Expand);
  
  // PowerPC has no SREM/UREM instructions
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  
  // We don't support sin/cos/sqrt/fmod
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FREM , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FREM , MVT::f32, Expand);
  
  // If we're enabling GP optimizations, use hardware square root
  if (!TM.getSubtarget<PPCSubtarget>().hasFSQRT()) {
    setOperationAction(ISD::FSQRT, MVT::f64, Expand);
    setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  }
  
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
  
  // PowerPC does not have BSWAP, CTPOP or CTTZ
  setOperationAction(ISD::BSWAP, MVT::i32  , Expand);
  setOperationAction(ISD::CTPOP, MVT::i32  , Expand);
  setOperationAction(ISD::CTTZ , MVT::i32  , Expand);
  
  // PowerPC does not have ROTR
  setOperationAction(ISD::ROTR, MVT::i32   , Expand);
  
  // PowerPC does not have Select
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);
  
  // PowerPC wants to turn select_cc of FP into fsel when possible.
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  // PowerPC wants to optimize integer setcc a bit
  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  
  // PowerPC does not have BRCOND which requires SetCC
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  
  // PowerPC turns FP_TO_SINT into FCTIWZ and some load/stores.
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);

  // PowerPC does not have [U|S]INT_TO_FP
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);

  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Expand);
  setOperationAction(ISD::BIT_CONVERT, MVT::i32, Expand);

  // PowerPC does not have truncstore for i1.
  setOperationAction(ISD::TRUNCSTORE, MVT::i1, Promote);

  // Support label based line numbers.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  // FIXME - use subtarget debug flags
  if (!TM.getSubtarget<PPCSubtarget>().isDarwin())
    setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);
  
  // We want to legalize GlobalAddress and ConstantPool nodes into the 
  // appropriate instructions to materialize the address.
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32, Custom);

  // RET must be custom lowered, to meet ABI requirements
  setOperationAction(ISD::RET               , MVT::Other, Custom);
  
  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  
  // Use the default implementation.
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);
  
  // We want to custom lower some of our intrinsics.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  
  if (TM.getSubtarget<PPCSubtarget>().is64Bit()) {
    // They also have instructions for converting between i64 and fp.
    setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
    setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
    
    // FIXME: disable this lowered code.  This generates 64-bit register values,
    // and we don't model the fact that the top part is clobbered by calls.  We
    // need to flag these together so that the value isn't live across a call.
    //setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
    
    // To take advantage of the above i64 FP_TO_SINT, promote i32 FP_TO_UINT
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Promote);
  } else {
    // PowerPC does not have FP_TO_UINT on 32-bit implementations.
    setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  }

  if (TM.getSubtarget<PPCSubtarget>().has64BitRegs()) {
    // 64 bit PowerPC implementations can support i64 types directly
    addRegisterClass(MVT::i64, PPC::G8RCRegisterClass);
    // BUILD_PAIR can't be handled natively, and should be expanded to shl/or
    setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);
  } else {
    // 32 bit PowerPC wants to expand i64 shifts itself.
    setOperationAction(ISD::SHL, MVT::i64, Custom);
    setOperationAction(ISD::SRL, MVT::i64, Custom);
    setOperationAction(ISD::SRA, MVT::i64, Custom);
  }

  if (TM.getSubtarget<PPCSubtarget>().hasAltivec()) {
    // First set operation action for all vector types to expand. Then we
    // will selectively turn on ones that can be effectively codegen'd.
    for (unsigned VT = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
         VT != (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++VT) {
      // add/sub/and/or/xor are legal for all supported vector VT's.
      setOperationAction(ISD::ADD , (MVT::ValueType)VT, Legal);
      setOperationAction(ISD::SUB , (MVT::ValueType)VT, Legal);
      setOperationAction(ISD::AND , (MVT::ValueType)VT, Legal);
      setOperationAction(ISD::OR  , (MVT::ValueType)VT, Legal);
      setOperationAction(ISD::XOR , (MVT::ValueType)VT, Legal);
      
      // We can custom expand all VECTOR_SHUFFLEs to VPERM.
      setOperationAction(ISD::VECTOR_SHUFFLE, (MVT::ValueType)VT, Custom);
      
      setOperationAction(ISD::MUL , (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::SDIV, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::SREM, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::UDIV, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::UREM, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::EXTRACT_VECTOR_ELT, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::INSERT_VECTOR_ELT, (MVT::ValueType)VT, Expand);
      setOperationAction(ISD::BUILD_VECTOR, (MVT::ValueType)VT, Expand);
    }

    addRegisterClass(MVT::v4f32, PPC::VRRCRegisterClass);
    addRegisterClass(MVT::v4i32, PPC::VRRCRegisterClass);
    addRegisterClass(MVT::v8i16, PPC::VRRCRegisterClass);
    addRegisterClass(MVT::v16i8, PPC::VRRCRegisterClass);
    
    setOperationAction(ISD::MUL, MVT::v4f32, Legal);

    setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4f32, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4i32, Custom);
    
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4i32, Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4f32, Custom);
  }
  
  setSetCCResultContents(ZeroOrOneSetCCResult);
  setStackPointerRegisterToSaveRestore(PPC::R1);
  
  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::SINT_TO_FP);
  setTargetDAGCombine(ISD::STORE);
  
  computeRegisterProperties();
}

const char *PPCTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case PPCISD::FSEL:          return "PPCISD::FSEL";
  case PPCISD::FCFID:         return "PPCISD::FCFID";
  case PPCISD::FCTIDZ:        return "PPCISD::FCTIDZ";
  case PPCISD::FCTIWZ:        return "PPCISD::FCTIWZ";
  case PPCISD::STFIWX:        return "PPCISD::STFIWX";
  case PPCISD::VMADDFP:       return "PPCISD::VMADDFP";
  case PPCISD::VNMSUBFP:      return "PPCISD::VNMSUBFP";
  case PPCISD::VPERM:         return "PPCISD::VPERM";
  case PPCISD::Hi:            return "PPCISD::Hi";
  case PPCISD::Lo:            return "PPCISD::Lo";
  case PPCISD::GlobalBaseReg: return "PPCISD::GlobalBaseReg";
  case PPCISD::SRL:           return "PPCISD::SRL";
  case PPCISD::SRA:           return "PPCISD::SRA";
  case PPCISD::SHL:           return "PPCISD::SHL";
  case PPCISD::EXTSW_32:      return "PPCISD::EXTSW_32";
  case PPCISD::STD_32:        return "PPCISD::STD_32";
  case PPCISD::CALL:          return "PPCISD::CALL";
  case PPCISD::RET_FLAG:      return "PPCISD::RET_FLAG";
  case PPCISD::MFCR:          return "PPCISD::MFCR";
  case PPCISD::VCMP:          return "PPCISD::VCMP";
  case PPCISD::VCMPo:         return "PPCISD::VCMPo";
  }
}

/// isFloatingPointZero - Return true if this is 0.0 or -0.0.
static bool isFloatingPointZero(SDOperand Op) {
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op))
    return CFP->isExactlyValue(-0.0) || CFP->isExactlyValue(0.0);
  else if (Op.getOpcode() == ISD::EXTLOAD || Op.getOpcode() == ISD::LOAD) {
    // Maybe this has already been legalized into the constant pool?
    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op.getOperand(1)))
      if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->get()))
        return CFP->isExactlyValue(-0.0) || CFP->isExactlyValue(0.0);
  }
  return false;
}


/// isSplatShuffleMask - Return true if the specified VECTOR_SHUFFLE operand
/// specifies a splat of a single element that is suitable for input to
/// VSPLTB/VSPLTH/VSPLTW.
bool PPC::isSplatShuffleMask(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR);
  
  // We can only splat 8-bit, 16-bit, and 32-bit quantities.
  if (N->getNumOperands() != 4 && N->getNumOperands() != 8 &&
      N->getNumOperands() != 16)
    return false;
  
  // This is a splat operation if each element of the permute is the same, and
  // if the value doesn't reference the second vector.
  SDOperand Elt = N->getOperand(0);
  assert(isa<ConstantSDNode>(Elt) && "Invalid VECTOR_SHUFFLE mask!");
  for (unsigned i = 1, e = N->getNumOperands(); i != e; ++i) {
    assert(isa<ConstantSDNode>(N->getOperand(i)) &&
           "Invalid VECTOR_SHUFFLE mask!");
    if (N->getOperand(i) != Elt) return false;
  }

  // Make sure it is a splat of the first vector operand.
  return cast<ConstantSDNode>(Elt)->getValue() < N->getNumOperands();
}

/// getVSPLTImmediate - Return the appropriate VSPLT* immediate to splat the
/// specified isSplatShuffleMask VECTOR_SHUFFLE mask.
unsigned PPC::getVSPLTImmediate(SDNode *N) {
  assert(isSplatShuffleMask(N));
  return cast<ConstantSDNode>(N->getOperand(0))->getValue();
}

/// isVecSplatImm - Return true if this is a build_vector of constants which
/// can be formed by using a vspltis[bhw] instruction.  The ByteSize field
/// indicates the number of bytes of each element [124] -> [bhw].
bool PPC::isVecSplatImm(SDNode *N, unsigned ByteSize, char *Val) {
  SDOperand OpVal(0, 0);
  // Check to see if this buildvec has a single non-undef value in its elements.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    if (OpVal.Val == 0)
      OpVal = N->getOperand(i);
    else if (OpVal != N->getOperand(i))
      return false;
  }
  
  if (OpVal.Val == 0) return false;  // All UNDEF: use implicit def.
  
  unsigned ValSizeInBytes = 0;
  uint64_t Value = 0;
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal)) {
    Value = CN->getValue();
    ValSizeInBytes = MVT::getSizeInBits(CN->getValueType(0))/8;
  } else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(OpVal)) {
    assert(CN->getValueType(0) == MVT::f32 && "Only one legal FP vector type!");
    Value = FloatToBits(CN->getValue());
    ValSizeInBytes = 4;
  }

  // If the splat value is larger than the element value, then we can never do
  // this splat.  The only case that we could fit the replicated bits into our
  // immediate field for would be zero, and we prefer to use vxor for it.
  if (ValSizeInBytes < ByteSize) return false;
  
  // If the element value is larger than the splat value, cut it in half and
  // check to see if the two halves are equal.  Continue doing this until we
  // get to ByteSize.  This allows us to handle 0x01010101 as 0x01.
  while (ValSizeInBytes > ByteSize) {
    ValSizeInBytes >>= 1;
    
    // If the top half equals the bottom half, we're still ok.
    if (((Value >> (ValSizeInBytes*8)) & ((8 << ValSizeInBytes)-1)) !=
         (Value                        & ((8 << ValSizeInBytes)-1)))
      return false;
  }

  // Properly sign extend the value.
  int ShAmt = (4-ByteSize)*8;
  int MaskVal = ((int)Value << ShAmt) >> ShAmt;
  
  // If this is zero, don't match, zero matches ISD::isBuildVectorAllZeros.
  if (MaskVal == 0) return false;

  if (Val) *Val = MaskVal;

  // Finally, if this value fits in a 5 bit sext field, return true.
  return ((MaskVal << (32-5)) >> (32-5)) == MaskVal; 
}


/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand PPCTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Wasn't expecting to be able to lower this!"); 
  case ISD::FP_TO_SINT: {
    assert(MVT::isFloatingPoint(Op.getOperand(0).getValueType()));
    SDOperand Src = Op.getOperand(0);
    if (Src.getValueType() == MVT::f32)
      Src = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Src);
    
    SDOperand Tmp;
    switch (Op.getValueType()) {
    default: assert(0 && "Unhandled FP_TO_SINT type in custom expander!");
    case MVT::i32:
      Tmp = DAG.getNode(PPCISD::FCTIWZ, MVT::f64, Src);
      break;
    case MVT::i64:
      Tmp = DAG.getNode(PPCISD::FCTIDZ, MVT::f64, Src);
      break;
    }
   
    // Convert the FP value to an int value through memory.
    SDOperand Bits = DAG.getNode(ISD::BIT_CONVERT, MVT::i64, Tmp);
    if (Op.getValueType() == MVT::i32)
      Bits = DAG.getNode(ISD::TRUNCATE, MVT::i32, Bits);
    return Bits;
  }
  case ISD::SINT_TO_FP:
    if (Op.getOperand(0).getValueType() == MVT::i64) {
      SDOperand Bits = DAG.getNode(ISD::BIT_CONVERT, MVT::f64, Op.getOperand(0));
      SDOperand FP = DAG.getNode(PPCISD::FCFID, MVT::f64, Bits);
      if (Op.getValueType() == MVT::f32)
        FP = DAG.getNode(ISD::FP_ROUND, MVT::f32, FP);
      return FP;
    } else {
      assert(Op.getOperand(0).getValueType() == MVT::i32 &&
             "Unhandled SINT_TO_FP type in custom expander!");
      // Since we only generate this in 64-bit mode, we can take advantage of
      // 64-bit registers.  In particular, sign extend the input value into the
      // 64-bit register with extsw, store the WHOLE 64-bit value into the stack
      // then lfd it and fcfid it.
      MachineFrameInfo *FrameInfo = DAG.getMachineFunction().getFrameInfo();
      int FrameIdx = FrameInfo->CreateStackObject(8, 8);
      SDOperand FIdx = DAG.getFrameIndex(FrameIdx, MVT::i32);
      
      SDOperand Ext64 = DAG.getNode(PPCISD::EXTSW_32, MVT::i32,
                                    Op.getOperand(0));

      // STD the extended value into the stack slot.
      SDOperand Store = DAG.getNode(PPCISD::STD_32, MVT::Other,
                                    DAG.getEntryNode(), Ext64, FIdx,
                                    DAG.getSrcValue(NULL));
      // Load the value as a double.
      SDOperand Ld = DAG.getLoad(MVT::f64, Store, FIdx, DAG.getSrcValue(NULL));
      
      // FCFID it and return it.
      SDOperand FP = DAG.getNode(PPCISD::FCFID, MVT::f64, Ld);
      if (Op.getValueType() == MVT::f32)
        FP = DAG.getNode(ISD::FP_ROUND, MVT::f32, FP);
      return FP;
    }
    break;

  case ISD::SELECT_CC: {
    // Turn FP only select_cc's into fsel instructions.
    if (!MVT::isFloatingPoint(Op.getOperand(0).getValueType()) ||
        !MVT::isFloatingPoint(Op.getOperand(2).getValueType()))
      break;
    
    ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
    
    // Cannot handle SETEQ/SETNE.
    if (CC == ISD::SETEQ || CC == ISD::SETNE) break;
    
    MVT::ValueType ResVT = Op.getValueType();
    MVT::ValueType CmpVT = Op.getOperand(0).getValueType();
    SDOperand LHS = Op.getOperand(0), RHS = Op.getOperand(1);
    SDOperand TV  = Op.getOperand(2), FV  = Op.getOperand(3);

    // If the RHS of the comparison is a 0.0, we don't need to do the
    // subtraction at all.
    if (isFloatingPointZero(RHS))
      switch (CC) {
      default: break;       // SETUO etc aren't handled by fsel.
      case ISD::SETULT:
      case ISD::SETLT:
        std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
      case ISD::SETUGE:
      case ISD::SETGE:
        if (LHS.getValueType() == MVT::f32)   // Comparison is always 64-bits
          LHS = DAG.getNode(ISD::FP_EXTEND, MVT::f64, LHS);
        return DAG.getNode(PPCISD::FSEL, ResVT, LHS, TV, FV);
      case ISD::SETUGT:
      case ISD::SETGT:
        std::swap(TV, FV);  // fsel is natively setge, swap operands for setlt
      case ISD::SETULE:
      case ISD::SETLE:
        if (LHS.getValueType() == MVT::f32)   // Comparison is always 64-bits
          LHS = DAG.getNode(ISD::FP_EXTEND, MVT::f64, LHS);
        return DAG.getNode(PPCISD::FSEL, ResVT,
                           DAG.getNode(ISD::FNEG, MVT::f64, LHS), TV, FV);
      }
    
    SDOperand Cmp;
    switch (CC) {
    default: break;       // SETUO etc aren't handled by fsel.
    case ISD::SETULT:
    case ISD::SETLT:
      Cmp = DAG.getNode(ISD::FSUB, CmpVT, LHS, RHS);
      if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
        Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, FV, TV);
    case ISD::SETUGE:
    case ISD::SETGE:
      Cmp = DAG.getNode(ISD::FSUB, CmpVT, LHS, RHS);
      if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
        Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, TV, FV);
    case ISD::SETUGT:
    case ISD::SETGT:
      Cmp = DAG.getNode(ISD::FSUB, CmpVT, RHS, LHS);
      if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
        Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, FV, TV);
    case ISD::SETULE:
    case ISD::SETLE:
      Cmp = DAG.getNode(ISD::FSUB, CmpVT, RHS, LHS);
      if (Cmp.getValueType() == MVT::f32)   // Comparison is always 64-bits
        Cmp = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Cmp);
      return DAG.getNode(PPCISD::FSEL, ResVT, Cmp, TV, FV);
    }
    break;
  }
  case ISD::SHL: {
    assert(Op.getValueType() == MVT::i64 &&
           Op.getOperand(1).getValueType() == MVT::i32 && "Unexpected SHL!");
    // The generic code does a fine job expanding shift by a constant.
    if (isa<ConstantSDNode>(Op.getOperand(1))) break;
    
    // Otherwise, expand into a bunch of logical ops.  Note that these ops
    // depend on the PPC behavior for oversized shift amounts.
    SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                               DAG.getConstant(0, MVT::i32));
    SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                               DAG.getConstant(1, MVT::i32));
    SDOperand Amt = Op.getOperand(1);
    
    SDOperand Tmp1 = DAG.getNode(ISD::SUB, MVT::i32,
                                 DAG.getConstant(32, MVT::i32), Amt);
    SDOperand Tmp2 = DAG.getNode(PPCISD::SHL, MVT::i32, Hi, Amt);
    SDOperand Tmp3 = DAG.getNode(PPCISD::SRL, MVT::i32, Lo, Tmp1);
    SDOperand Tmp4 = DAG.getNode(ISD::OR , MVT::i32, Tmp2, Tmp3);
    SDOperand Tmp5 = DAG.getNode(ISD::ADD, MVT::i32, Amt,
                                 DAG.getConstant(-32U, MVT::i32));
    SDOperand Tmp6 = DAG.getNode(PPCISD::SHL, MVT::i32, Lo, Tmp5);
    SDOperand OutHi = DAG.getNode(ISD::OR, MVT::i32, Tmp4, Tmp6);
    SDOperand OutLo = DAG.getNode(PPCISD::SHL, MVT::i32, Lo, Amt);
    return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, OutLo, OutHi);
  }
  case ISD::SRL: {
    assert(Op.getValueType() == MVT::i64 &&
           Op.getOperand(1).getValueType() == MVT::i32 && "Unexpected SHL!");
    // The generic code does a fine job expanding shift by a constant.
    if (isa<ConstantSDNode>(Op.getOperand(1))) break;
    
    // Otherwise, expand into a bunch of logical ops.  Note that these ops
    // depend on the PPC behavior for oversized shift amounts.
    SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                               DAG.getConstant(0, MVT::i32));
    SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                               DAG.getConstant(1, MVT::i32));
    SDOperand Amt = Op.getOperand(1);
    
    SDOperand Tmp1 = DAG.getNode(ISD::SUB, MVT::i32,
                                 DAG.getConstant(32, MVT::i32), Amt);
    SDOperand Tmp2 = DAG.getNode(PPCISD::SRL, MVT::i32, Lo, Amt);
    SDOperand Tmp3 = DAG.getNode(PPCISD::SHL, MVT::i32, Hi, Tmp1);
    SDOperand Tmp4 = DAG.getNode(ISD::OR , MVT::i32, Tmp2, Tmp3);
    SDOperand Tmp5 = DAG.getNode(ISD::ADD, MVT::i32, Amt,
                                 DAG.getConstant(-32U, MVT::i32));
    SDOperand Tmp6 = DAG.getNode(PPCISD::SRL, MVT::i32, Hi, Tmp5);
    SDOperand OutLo = DAG.getNode(ISD::OR, MVT::i32, Tmp4, Tmp6);
    SDOperand OutHi = DAG.getNode(PPCISD::SRL, MVT::i32, Hi, Amt);
    return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, OutLo, OutHi);
  }    
  case ISD::SRA: {
    assert(Op.getValueType() == MVT::i64 &&
           Op.getOperand(1).getValueType() == MVT::i32 && "Unexpected SRA!");
    // The generic code does a fine job expanding shift by a constant.
    if (isa<ConstantSDNode>(Op.getOperand(1))) break;
      
    // Otherwise, expand into a bunch of logical ops, followed by a select_cc.
    SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                               DAG.getConstant(0, MVT::i32));
    SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                               DAG.getConstant(1, MVT::i32));
    SDOperand Amt = Op.getOperand(1);
    
    SDOperand Tmp1 = DAG.getNode(ISD::SUB, MVT::i32,
                                 DAG.getConstant(32, MVT::i32), Amt);
    SDOperand Tmp2 = DAG.getNode(PPCISD::SRL, MVT::i32, Lo, Amt);
    SDOperand Tmp3 = DAG.getNode(PPCISD::SHL, MVT::i32, Hi, Tmp1);
    SDOperand Tmp4 = DAG.getNode(ISD::OR , MVT::i32, Tmp2, Tmp3);
    SDOperand Tmp5 = DAG.getNode(ISD::ADD, MVT::i32, Amt,
                                 DAG.getConstant(-32U, MVT::i32));
    SDOperand Tmp6 = DAG.getNode(PPCISD::SRA, MVT::i32, Hi, Tmp5);
    SDOperand OutHi = DAG.getNode(PPCISD::SRA, MVT::i32, Hi, Amt);
    SDOperand OutLo = DAG.getSelectCC(Tmp5, DAG.getConstant(0, MVT::i32),
                                      Tmp4, Tmp6, ISD::SETLE);
    return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, OutLo, OutHi);
  }
  case ISD::ConstantPool: {
    ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
    Constant *C = CP->get();
    SDOperand CPI = DAG.getTargetConstantPool(C, MVT::i32, CP->getAlignment());
    SDOperand Zero = DAG.getConstant(0, MVT::i32);
    
    if (getTargetMachine().getRelocationModel() == Reloc::Static) {
      // Generate non-pic code that has direct accesses to the constant pool.
      // The address of the global is just (hi(&g)+lo(&g)).
      SDOperand Hi = DAG.getNode(PPCISD::Hi, MVT::i32, CPI, Zero);
      SDOperand Lo = DAG.getNode(PPCISD::Lo, MVT::i32, CPI, Zero);
      return DAG.getNode(ISD::ADD, MVT::i32, Hi, Lo);
    }
    
    // Only lower ConstantPool on Darwin.
    if (!getTargetMachine().getSubtarget<PPCSubtarget>().isDarwin()) break;
    SDOperand Hi = DAG.getNode(PPCISD::Hi, MVT::i32, CPI, Zero);
    if (getTargetMachine().getRelocationModel() == Reloc::PIC) {
      // With PIC, the first instruction is actually "GR+hi(&G)".
      Hi = DAG.getNode(ISD::ADD, MVT::i32,
                       DAG.getNode(PPCISD::GlobalBaseReg, MVT::i32), Hi);
    }

    SDOperand Lo = DAG.getNode(PPCISD::Lo, MVT::i32, CPI, Zero);
    Lo = DAG.getNode(ISD::ADD, MVT::i32, Hi, Lo);
    return Lo;
  }
  case ISD::GlobalAddress: {
    GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
    GlobalValue *GV = GSDN->getGlobal();
    SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i32, GSDN->getOffset());
    SDOperand Zero = DAG.getConstant(0, MVT::i32);

    if (getTargetMachine().getRelocationModel() == Reloc::Static) {
      // Generate non-pic code that has direct accesses to globals.
      // The address of the global is just (hi(&g)+lo(&g)).
      SDOperand Hi = DAG.getNode(PPCISD::Hi, MVT::i32, GA, Zero);
      SDOperand Lo = DAG.getNode(PPCISD::Lo, MVT::i32, GA, Zero);
      return DAG.getNode(ISD::ADD, MVT::i32, Hi, Lo);
    }
    
    // Only lower GlobalAddress on Darwin.
    if (!getTargetMachine().getSubtarget<PPCSubtarget>().isDarwin()) break;
    
    SDOperand Hi = DAG.getNode(PPCISD::Hi, MVT::i32, GA, Zero);
    if (getTargetMachine().getRelocationModel() == Reloc::PIC) {
      // With PIC, the first instruction is actually "GR+hi(&G)".
      Hi = DAG.getNode(ISD::ADD, MVT::i32,
                       DAG.getNode(PPCISD::GlobalBaseReg, MVT::i32), Hi);
    }
    
    SDOperand Lo = DAG.getNode(PPCISD::Lo, MVT::i32, GA, Zero);
    Lo = DAG.getNode(ISD::ADD, MVT::i32, Hi, Lo);
                                   
    if (!GV->hasWeakLinkage() && !GV->hasLinkOnceLinkage() &&
        (!GV->isExternal() || GV->hasNotBeenReadFromBytecode()))
      return Lo;

    // If the global is weak or external, we have to go through the lazy
    // resolution stub.
    return DAG.getLoad(MVT::i32, DAG.getEntryNode(), Lo, DAG.getSrcValue(0));
  }
  case ISD::SETCC: {
    ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
    
    // If we're comparing for equality to zero, expose the fact that this is
    // implented as a ctlz/srl pair on ppc, so that the dag combiner can
    // fold the new nodes.
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      if (C->isNullValue() && CC == ISD::SETEQ) {
        MVT::ValueType VT = Op.getOperand(0).getValueType();
        SDOperand Zext = Op.getOperand(0);
        if (VT < MVT::i32) {
          VT = MVT::i32;
          Zext = DAG.getNode(ISD::ZERO_EXTEND, VT, Op.getOperand(0));
        } 
        unsigned Log2b = Log2_32(MVT::getSizeInBits(VT));
        SDOperand Clz = DAG.getNode(ISD::CTLZ, VT, Zext);
        SDOperand Scc = DAG.getNode(ISD::SRL, VT, Clz,
                                    DAG.getConstant(Log2b, getShiftAmountTy()));
        return DAG.getNode(ISD::TRUNCATE, getSetCCResultTy(), Scc);
      }
      // Leave comparisons against 0 and -1 alone for now, since they're usually 
      // optimized.  FIXME: revisit this when we can custom lower all setcc
      // optimizations.
      if (C->isAllOnesValue() || C->isNullValue())
        break;
    }
        
    // If we have an integer seteq/setne, turn it into a compare against zero
    // by subtracting the rhs from the lhs, which is faster than setting a
    // condition register, reading it back out, and masking the correct bit.
    MVT::ValueType LHSVT = Op.getOperand(0).getValueType();
    if (MVT::isInteger(LHSVT) && (CC == ISD::SETEQ || CC == ISD::SETNE)) {
      MVT::ValueType VT = Op.getValueType();
      SDOperand Sub = DAG.getNode(ISD::SUB, LHSVT, Op.getOperand(0), 
                                  Op.getOperand(1));
      return DAG.getSetCC(VT, Sub, DAG.getConstant(0, LHSVT), CC);
    }
    break;
  }
  case ISD::VASTART: {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    // FIXME: Replace MVT::i32 with PointerTy
    SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
    return DAG.getNode(ISD::STORE, MVT::Other, Op.getOperand(0), FR, 
                       Op.getOperand(1), Op.getOperand(2));
  }
  case ISD::RET: {
    SDOperand Copy;
    
    switch(Op.getNumOperands()) {
    default:
      assert(0 && "Do not know how to return this many arguments!");
      abort();
    case 1: 
      return SDOperand(); // ret void is legal
    case 2: {
      MVT::ValueType ArgVT = Op.getOperand(1).getValueType();
      unsigned ArgReg = MVT::isInteger(ArgVT) ? PPC::R3 : PPC::F1;
      Copy = DAG.getCopyToReg(Op.getOperand(0), ArgReg, Op.getOperand(1),
                              SDOperand());
      break;
    }
    case 3:
      Copy = DAG.getCopyToReg(Op.getOperand(0), PPC::R3, Op.getOperand(2), 
                              SDOperand());
      Copy = DAG.getCopyToReg(Copy, PPC::R4, Op.getOperand(1),Copy.getValue(1));
      break;
    }
    return DAG.getNode(PPCISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
  }
  case ISD::SCALAR_TO_VECTOR: {
    // Create a stack slot that is 16-byte aligned.
    MachineFrameInfo *FrameInfo = DAG.getMachineFunction().getFrameInfo();
    int FrameIdx = FrameInfo->CreateStackObject(16, 16);
    SDOperand FIdx = DAG.getFrameIndex(FrameIdx, MVT::i32);
    
    // Store the input value into Value#0 of the stack slot.
    SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                                  Op.getOperand(0), FIdx,DAG.getSrcValue(NULL));
    // Load it out.
    return DAG.getLoad(Op.getValueType(), Store, FIdx, DAG.getSrcValue(NULL));
  }
  case ISD::BUILD_VECTOR:
    // If this is a case we can't handle, return null and let the default
    // expansion code take care of it.  If we CAN select this case, return Op.
    
    // See if this is all zeros.
    // FIXME: We should handle splat(-0.0), and other cases here.
    if (ISD::isBuildVectorAllZeros(Op.Val))
      return Op;
    
    if (PPC::isVecSplatImm(Op.Val, 1) ||    // vspltisb
        PPC::isVecSplatImm(Op.Val, 2) ||    // vspltish
        PPC::isVecSplatImm(Op.Val, 4))      // vspltisw
      return Op;
      
    return SDOperand();
    
  case ISD::VECTOR_SHUFFLE: {
    SDOperand V1 = Op.getOperand(0);
    SDOperand V2 = Op.getOperand(1);
    SDOperand PermMask = Op.getOperand(2);
    
    // Cases that are handled by instructions that take permute immediates
    // (such as vsplt*) should be left as VECTOR_SHUFFLE nodes so they can be
    // selected by the instruction selector.
    if (PPC::isSplatShuffleMask(PermMask.Val) && V2.getOpcode() == ISD::UNDEF)
      break;
    
    // TODO: Handle more cases, and also handle cases that are cheaper to do as
    // multiple such instructions than as a constant pool load/vperm pair.
    
    // Lower this to a VPERM(V1, V2, V3) expression, where V3 is a constant
    // vector that will get spilled to the constant pool.
    if (V2.getOpcode() == ISD::UNDEF) V2 = V1;
    
    // The SHUFFLE_VECTOR mask is almost exactly what we want for vperm, except
    // that it is in input element units, not in bytes.  Convert now.
    MVT::ValueType EltVT = MVT::getVectorBaseType(V1.getValueType());
    unsigned BytesPerElement = MVT::getSizeInBits(EltVT)/8;
    
    std::vector<SDOperand> ResultMask;
    for (unsigned i = 0, e = PermMask.getNumOperands(); i != e; ++i) {
      unsigned SrcElt =cast<ConstantSDNode>(PermMask.getOperand(i))->getValue();
      
      for (unsigned j = 0; j != BytesPerElement; ++j)
        ResultMask.push_back(DAG.getConstant(SrcElt*BytesPerElement+j,
                                             MVT::i8));
    }
    
    SDOperand VPermMask =DAG.getNode(ISD::BUILD_VECTOR, MVT::v16i8, ResultMask);
    return DAG.getNode(PPCISD::VPERM, V1.getValueType(), V1, V2, VPermMask);
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo=cast<ConstantSDNode>(Op.getOperand(0))->getValue();
    
    // If this is a lowered altivec predicate compare, CompareOpc is set to the
    // opcode number of the comparison.
    int CompareOpc = -1;
    bool isDot = false;
    switch (IntNo) {
    default: return SDOperand();    // Don't custom lower most intrinsics.
    // Comparison predicates.
    case Intrinsic::ppc_altivec_vcmpbfp_p:  CompareOpc = 966; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpeqfp_p: CompareOpc = 198; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpequb_p: CompareOpc =   6; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpequh_p: CompareOpc =  70; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpequw_p: CompareOpc = 134; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgefp_p: CompareOpc = 454; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgtfp_p: CompareOpc = 710; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgtsb_p: CompareOpc = 774; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgtsh_p: CompareOpc = 838; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgtsw_p: CompareOpc = 902; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgtub_p: CompareOpc = 518; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgtuh_p: CompareOpc = 582; isDot = 1; break;
    case Intrinsic::ppc_altivec_vcmpgtuw_p: CompareOpc = 646; isDot = 1; break;

    // Normal Comparisons.
    case Intrinsic::ppc_altivec_vcmpbfp:    CompareOpc = 966; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpeqfp:   CompareOpc = 198; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpequb:   CompareOpc =   6; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpequh:   CompareOpc =  70; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpequw:   CompareOpc = 134; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgefp:   CompareOpc = 454; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgtfp:   CompareOpc = 710; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgtsb:   CompareOpc = 774; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgtsh:   CompareOpc = 838; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgtsw:   CompareOpc = 902; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgtub:   CompareOpc = 518; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgtuh:   CompareOpc = 582; isDot = 0; break;
    case Intrinsic::ppc_altivec_vcmpgtuw:   CompareOpc = 646; isDot = 0; break;
    }
    
    assert(CompareOpc>0 && "We only lower altivec predicate compares so far!");

    // If this is a non-dot comparison, make the VCMP node.
    if (!isDot)
      return DAG.getNode(PPCISD::VCMP, Op.getOperand(2).getValueType(),
                         Op.getOperand(1), Op.getOperand(2),
                         DAG.getConstant(CompareOpc, MVT::i32));
    
    // Create the PPCISD altivec 'dot' comparison node.
    std::vector<SDOperand> Ops;
    std::vector<MVT::ValueType> VTs;
    Ops.push_back(Op.getOperand(2));  // LHS
    Ops.push_back(Op.getOperand(3));  // RHS
    Ops.push_back(DAG.getConstant(CompareOpc, MVT::i32));
    VTs.push_back(Op.getOperand(2).getValueType());
    VTs.push_back(MVT::Flag);
    SDOperand CompNode = DAG.getNode(PPCISD::VCMPo, VTs, Ops);

    // Now that we have the comparison, emit a copy from the CR to a GPR.
    // This is flagged to the above dot comparison.
    SDOperand Flags = DAG.getNode(PPCISD::MFCR, MVT::i32,
                                  DAG.getRegister(PPC::CR6, MVT::i32),
                                  CompNode.getValue(1)); 

    // Unpack the result based on how the target uses it.
    unsigned BitNo;   // Bit # of CR6.
    bool InvertBit;   // Invert result?
    switch (cast<ConstantSDNode>(Op.getOperand(1))->getValue()) {
    default:  // Can't happen, don't crash on invalid number though.
    case 0:   // Return the value of the EQ bit of CR6.
      BitNo = 0; InvertBit = false;
      break;
    case 1:   // Return the inverted value of the EQ bit of CR6.
      BitNo = 0; InvertBit = true;
      break;
    case 2:   // Return the value of the LT bit of CR6.
      BitNo = 2; InvertBit = false;
      break;
    case 3:   // Return the inverted value of the LT bit of CR6.
      BitNo = 2; InvertBit = true;
      break;
    }
    
    // Shift the bit into the low position.
    Flags = DAG.getNode(ISD::SRL, MVT::i32, Flags,
                        DAG.getConstant(8-(3-BitNo), MVT::i32));
    // Isolate the bit.
    Flags = DAG.getNode(ISD::AND, MVT::i32, Flags,
                        DAG.getConstant(1, MVT::i32));
    
    // If we are supposed to, toggle the bit.
    if (InvertBit)
      Flags = DAG.getNode(ISD::XOR, MVT::i32, Flags,
                          DAG.getConstant(1, MVT::i32));
    return Flags;
  }
  }
  return SDOperand();
}

std::vector<SDOperand>
PPCTargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  //
  // add beautiful description of PPC stack frame format, or at least some docs
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock& BB = MF.front();
  SSARegMap *RegMap = MF.getSSARegMap();
  std::vector<SDOperand> ArgValues;
  
  unsigned ArgOffset = 24;
  unsigned GPR_remaining = 8;
  unsigned FPR_remaining = 13;
  unsigned GPR_idx = 0, FPR_idx = 0;
  static const unsigned GPR[] = {
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const unsigned FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };
  
  // Add DAG nodes to load the arguments...  On entry to a function on PPC,
  // the arguments start at offset 24, although they are likely to be passed
  // in registers.
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    SDOperand newroot, argt;
    unsigned ObjSize;
    bool needsLoad = false;
    bool ArgLive = !I->use_empty();
    MVT::ValueType ObjectVT = getValueType(I->getType());
    
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      ObjSize = 4;
      if (!ArgLive) break;
      if (GPR_remaining > 0) {
        unsigned VReg = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
        MF.addLiveIn(GPR[GPR_idx], VReg);
        argt = newroot = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);
        if (ObjectVT != MVT::i32) {
          unsigned AssertOp = I->getType()->isSigned() ? ISD::AssertSext 
                                                       : ISD::AssertZext;
          argt = DAG.getNode(AssertOp, MVT::i32, argt, 
                             DAG.getValueType(ObjectVT));
          argt = DAG.getNode(ISD::TRUNCATE, ObjectVT, argt);
        }
      } else {
        needsLoad = true;
      }
      break;
    case MVT::i64:
      ObjSize = 8;
      if (!ArgLive) break;
      if (GPR_remaining > 0) {
        SDOperand argHi, argLo;
        unsigned VReg = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
        MF.addLiveIn(GPR[GPR_idx], VReg);
        argHi = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);
        // If we have two or more remaining argument registers, then both halves
        // of the i64 can be sourced from there.  Otherwise, the lower half will
        // have to come off the stack.  This can happen when an i64 is preceded
        // by 28 bytes of arguments.
        if (GPR_remaining > 1) {
          unsigned VReg = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
          MF.addLiveIn(GPR[GPR_idx+1], VReg);
          argLo = DAG.getCopyFromReg(argHi, VReg, MVT::i32);
        } else {
          int FI = MFI->CreateFixedObject(4, ArgOffset+4);
          SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
          argLo = DAG.getLoad(MVT::i32, DAG.getEntryNode(), FIN,
                              DAG.getSrcValue(NULL));
        }
        // Build the outgoing arg thingy
        argt = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, argLo, argHi);
        newroot = argLo;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::f32:
    case MVT::f64:
      ObjSize = (ObjectVT == MVT::f64) ? 8 : 4;
      if (!ArgLive) {
        if (FPR_remaining > 0) {
          --FPR_remaining;
          ++FPR_idx;
        }        
        break;
      }
      if (FPR_remaining > 0) {
        unsigned VReg;
        if (ObjectVT == MVT::f32)
          VReg = RegMap->createVirtualRegister(&PPC::F4RCRegClass);
        else
          VReg = RegMap->createVirtualRegister(&PPC::F8RCRegClass);
        MF.addLiveIn(FPR[FPR_idx], VReg);
        argt = newroot = DAG.getCopyFromReg(DAG.getRoot(), VReg, ObjectVT);
        --FPR_remaining;
        ++FPR_idx;
      } else {
        needsLoad = true;
      }
      break;
    }
    
    // We need to load the argument to a virtual register if we determined above
    // that we ran out of physical registers of the appropriate type
    if (needsLoad) {
      unsigned SubregOffset = 0;
      if (ObjectVT == MVT::i8 || ObjectVT == MVT::i1) SubregOffset = 3;
      if (ObjectVT == MVT::i16) SubregOffset = 2;
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
      FIN = DAG.getNode(ISD::ADD, MVT::i32, FIN,
                        DAG.getConstant(SubregOffset, MVT::i32));
      argt = newroot = DAG.getLoad(ObjectVT, DAG.getEntryNode(), FIN,
                                   DAG.getSrcValue(NULL));
    }
    
    // Every 4 bytes of argument space consumes one of the GPRs available for
    // argument passing.
    if (GPR_remaining > 0) {
      unsigned delta = (GPR_remaining > 1 && ObjSize == 8) ? 2 : 1;
      GPR_remaining -= delta;
      GPR_idx += delta;
    }
    ArgOffset += ObjSize;
    if (newroot.Val)
      DAG.setRoot(newroot.getValue(1));
    
    ArgValues.push_back(argt);
  }
  
  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (F.isVarArg()) {
    VarArgsFrameIndex = MFI->CreateFixedObject(4, ArgOffset);
    SDOperand FIN = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
    // If this function is vararg, store any remaining integer argument regs
    // to their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    std::vector<SDOperand> MemOps;
    for (; GPR_remaining > 0; --GPR_remaining, ++GPR_idx) {
      unsigned VReg = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
      MF.addLiveIn(GPR[GPR_idx], VReg);
      SDOperand Val = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);
      SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1),
                                    Val, FIN, DAG.getSrcValue(NULL));
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDOperand PtrOff = DAG.getConstant(4, getPointerTy());
      FIN = DAG.getNode(ISD::ADD, MVT::i32, FIN, PtrOff);
    }
    if (!MemOps.empty()) {
      MemOps.push_back(DAG.getRoot());
      DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps));
    }
  }
  
  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
    default: assert(0 && "Unknown type!");
    case MVT::isVoid: break;
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      MF.addLiveOut(PPC::R3);
      break;
    case MVT::i64:
      MF.addLiveOut(PPC::R3);
      MF.addLiveOut(PPC::R4);
      break;
    case MVT::f32:
    case MVT::f64:
      MF.addLiveOut(PPC::F1);
      break;
  }
  
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
PPCTargetLowering::LowerCallTo(SDOperand Chain,
                               const Type *RetTy, bool isVarArg,
                               unsigned CallingConv, bool isTailCall,
                               SDOperand Callee, ArgListTy &Args,
                               SelectionDAG &DAG) {
  // args_to_use will accumulate outgoing args for the PPCISD::CALL case in
  // SelectExpr to use to put the arguments in the appropriate registers.
  std::vector<SDOperand> args_to_use;
  
  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.
  unsigned NumBytes = 24;
  
  if (Args.empty()) {
    Chain = DAG.getCALLSEQ_START(Chain,
                                 DAG.getConstant(NumBytes, getPointerTy()));
  } else {
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      switch (getValueType(Args[i].second)) {
      default: assert(0 && "Unknown value type!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      case MVT::f32:
        NumBytes += 4;
        break;
      case MVT::i64:
      case MVT::f64:
        NumBytes += 8;
        break;
      }
    }
        
    // Just to be safe, we'll always reserve the full 24 bytes of linkage area
    // plus 32 bytes of argument space in case any called code gets funky on us.
    // (Required by ABI to support var arg)
    if (NumBytes < 56) NumBytes = 56;
    
    // Adjust the stack pointer for the new arguments...
    // These operations are automatically eliminated by the prolog/epilog pass
    Chain = DAG.getCALLSEQ_START(Chain,
                                 DAG.getConstant(NumBytes, getPointerTy()));
    
    // Set up a copy of the stack pointer for use loading and storing any
    // arguments that may not fit in the registers available for argument
    // passing.
    SDOperand StackPtr = DAG.getRegister(PPC::R1, MVT::i32);
    
    // Figure out which arguments are going to go in registers, and which in
    // memory.  Also, if this is a vararg function, floating point operations
    // must be stored to our stack, and loaded into integer regs as well, if
    // any integer regs are available for argument passing.
    unsigned ArgOffset = 24;
    unsigned GPR_remaining = 8;
    unsigned FPR_remaining = 13;
    
    std::vector<SDOperand> MemOps;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      // PtrOff will be used to store the current argument to the stack if a
      // register cannot be found for it.
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      MVT::ValueType ArgVT = getValueType(Args[i].second);
      
      switch (ArgVT) {
      default: assert(0 && "Unexpected ValueType for argument!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
        // Promote the integer to 32 bits.  If the input type is signed use a
        // sign extend, otherwise use a zero extend.
        if (Args[i].second->isSigned())
          Args[i].first =DAG.getNode(ISD::SIGN_EXTEND, MVT::i32, Args[i].first);
        else
          Args[i].first =DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Args[i].first);
        // FALL THROUGH
      case MVT::i32:
        if (GPR_remaining > 0) {
          args_to_use.push_back(Args[i].first);
          --GPR_remaining;
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                       Args[i].first, PtrOff,
                                       DAG.getSrcValue(NULL)));
        }
        ArgOffset += 4;
        break;
      case MVT::i64:
        // If we have one free GPR left, we can place the upper half of the i64
        // in it, and store the other half to the stack.  If we have two or more
        // free GPRs, then we can pass both halves of the i64 in registers.
        if (GPR_remaining > 0) {
          SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                     Args[i].first, DAG.getConstant(1, MVT::i32));
          SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                     Args[i].first, DAG.getConstant(0, MVT::i32));
          args_to_use.push_back(Hi);
          --GPR_remaining;
          if (GPR_remaining > 0) {
            args_to_use.push_back(Lo);
            --GPR_remaining;
          } else {
            SDOperand ConstFour = DAG.getConstant(4, getPointerTy());
            PtrOff = DAG.getNode(ISD::ADD, MVT::i32, PtrOff, ConstFour);
            MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                         Lo, PtrOff, DAG.getSrcValue(NULL)));
          }
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                       Args[i].first, PtrOff,
                                       DAG.getSrcValue(NULL)));
        }
        ArgOffset += 8;
        break;
      case MVT::f32:
      case MVT::f64:
        if (FPR_remaining > 0) {
          args_to_use.push_back(Args[i].first);
          --FPR_remaining;
          if (isVarArg) {
            SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                          Args[i].first, PtrOff,
                                          DAG.getSrcValue(NULL));
            MemOps.push_back(Store);
            // Float varargs are always shadowed in available integer registers
            if (GPR_remaining > 0) {
              SDOperand Load = DAG.getLoad(MVT::i32, Store, PtrOff,
                                           DAG.getSrcValue(NULL));
              MemOps.push_back(Load.getValue(1));
              args_to_use.push_back(Load);
              --GPR_remaining;
            }
            if (GPR_remaining > 0 && MVT::f64 == ArgVT) {
              SDOperand ConstFour = DAG.getConstant(4, getPointerTy());
              PtrOff = DAG.getNode(ISD::ADD, MVT::i32, PtrOff, ConstFour);
              SDOperand Load = DAG.getLoad(MVT::i32, Store, PtrOff,
                                           DAG.getSrcValue(NULL));
              MemOps.push_back(Load.getValue(1));
              args_to_use.push_back(Load);
              --GPR_remaining;
            }
          } else {
            // If we have any FPRs remaining, we may also have GPRs remaining.
            // Args passed in FPRs consume either 1 (f32) or 2 (f64) available
            // GPRs.
            if (GPR_remaining > 0) {
              args_to_use.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
              --GPR_remaining;
            }
            if (GPR_remaining > 0 && MVT::f64 == ArgVT) {
              args_to_use.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
              --GPR_remaining;
            }
          }
        } else {
          MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                       Args[i].first, PtrOff,
                                       DAG.getSrcValue(NULL)));
        }
        ArgOffset += (ArgVT == MVT::f32) ? 4 : 8;
        break;
      }
    }
    if (!MemOps.empty())
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps);
  }
  
  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  MVT::ValueType ActualRetTyVT = RetTyVT;
  if (RetTyVT >= MVT::i1 && RetTyVT <= MVT::i16)
    ActualRetTyVT = MVT::i32;   // Promote result to i32.
    
  if (RetTyVT == MVT::i64) {
    RetVals.push_back(MVT::i32);
    RetVals.push_back(MVT::i32);
  } else if (RetTyVT != MVT::isVoid) {
    RetVals.push_back(ActualRetTyVT);
  }
  RetVals.push_back(MVT::Other);
  
  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i32);
  
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  Ops.insert(Ops.end(), args_to_use.begin(), args_to_use.end());
  SDOperand TheCall = DAG.getNode(PPCISD::CALL, RetVals, Ops);
  Chain = TheCall.getValue(TheCall.Val->getNumValues()-1);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  SDOperand RetVal = TheCall;
  
  // If the result is a small value, add a note so that we keep track of the
  // information about whether it is sign or zero extended.
  if (RetTyVT != ActualRetTyVT) {
    RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext : ISD::AssertZext,
                         MVT::i32, RetVal, DAG.getValueType(RetTyVT));
    RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
  } else if (RetTyVT == MVT::i64) {
    RetVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, RetVal, RetVal.getValue(1));
  }
  
  return std::make_pair(RetVal, Chain);
}

MachineBasicBlock *
PPCTargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                           MachineBasicBlock *BB) {
  assert((MI->getOpcode() == PPC::SELECT_CC_Int ||
          MI->getOpcode() == PPC::SELECT_CC_F4 ||
          MI->getOpcode() == PPC::SELECT_CC_F8) &&
         "Unexpected instr type to insert");
  
  // To "insert" a SELECT_CC instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;
  
  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   cmpTY ccX, r1, r2
  //   bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  BuildMI(BB, MI->getOperand(4).getImmedValue(), 2)
    .addReg(MI->getOperand(1).getReg()).addMBB(sinkMBB);
  MachineFunction *F = BB->getParent();
  F->getBasicBlockList().insert(It, copy0MBB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges by first adding all successors of the current
  // block to the new block which will contain the Phi node for the select.
  for(MachineBasicBlock::succ_iterator i = BB->succ_begin(), 
      e = BB->succ_end(); i != e; ++i)
    sinkMBB->addSuccessor(*i);
  // Next, remove all successors of the current block, and add the true
  // and fallthrough blocks as its successors.
  while(!BB->succ_empty())
    BB->removeSuccessor(BB->succ_begin());
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);
  
  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copy0MBB;
  
  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);
  
  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, PPC::PHI, 4, MI->getOperand(0).getReg())
    .addReg(MI->getOperand(3).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

  delete MI;   // The pseudo instruction is gone now.
  return BB;
}

SDOperand PPCTargetLowering::PerformDAGCombine(SDNode *N, 
                                               DAGCombinerInfo &DCI) const {
  TargetMachine &TM = getTargetMachine();
  SelectionDAG &DAG = DCI.DAG;
  switch (N->getOpcode()) {
  default: break;
  case ISD::SINT_TO_FP:
    if (TM.getSubtarget<PPCSubtarget>().is64Bit()) {
      if (N->getOperand(0).getOpcode() == ISD::FP_TO_SINT) {
        // Turn (sint_to_fp (fp_to_sint X)) -> fctidz/fcfid without load/stores.
        // We allow the src/dst to be either f32/f64, but the intermediate
        // type must be i64.
        if (N->getOperand(0).getValueType() == MVT::i64) {
          SDOperand Val = N->getOperand(0).getOperand(0);
          if (Val.getValueType() == MVT::f32) {
            Val = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Val);
            DCI.AddToWorklist(Val.Val);
          }
            
          Val = DAG.getNode(PPCISD::FCTIDZ, MVT::f64, Val);
          DCI.AddToWorklist(Val.Val);
          Val = DAG.getNode(PPCISD::FCFID, MVT::f64, Val);
          DCI.AddToWorklist(Val.Val);
          if (N->getValueType(0) == MVT::f32) {
            Val = DAG.getNode(ISD::FP_ROUND, MVT::f32, Val);
            DCI.AddToWorklist(Val.Val);
          }
          return Val;
        } else if (N->getOperand(0).getValueType() == MVT::i32) {
          // If the intermediate type is i32, we can avoid the load/store here
          // too.
        }
      }
    }
    break;
  case ISD::STORE:
    // Turn STORE (FP_TO_SINT F) -> STFIWX(FCTIWZ(F)).
    if (TM.getSubtarget<PPCSubtarget>().hasSTFIWX() &&
        N->getOperand(1).getOpcode() == ISD::FP_TO_SINT &&
        N->getOperand(1).getValueType() == MVT::i32) {
      SDOperand Val = N->getOperand(1).getOperand(0);
      if (Val.getValueType() == MVT::f32) {
        Val = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Val);
        DCI.AddToWorklist(Val.Val);
      }
      Val = DAG.getNode(PPCISD::FCTIWZ, MVT::f64, Val);
      DCI.AddToWorklist(Val.Val);

      Val = DAG.getNode(PPCISD::STFIWX, MVT::Other, N->getOperand(0), Val,
                        N->getOperand(2), N->getOperand(3));
      DCI.AddToWorklist(Val.Val);
      return Val;
    }
    break;
  case PPCISD::VCMP: {
    // If a VCMPo node already exists with exactly the same operands as this
    // node, use its result instead of this node (VCMPo computes both a CR6 and
    // a normal output).
    //
    if (!N->getOperand(0).hasOneUse() &&
        !N->getOperand(1).hasOneUse() &&
        !N->getOperand(2).hasOneUse()) {
      
      // Scan all of the users of the LHS, looking for VCMPo's that match.
      SDNode *VCMPoNode = 0;
      
      SDNode *LHSN = N->getOperand(0).Val;
      for (SDNode::use_iterator UI = LHSN->use_begin(), E = LHSN->use_end();
           UI != E; ++UI)
        if ((*UI)->getOpcode() == PPCISD::VCMPo &&
            (*UI)->getOperand(1) == N->getOperand(1) &&
            (*UI)->getOperand(2) == N->getOperand(2) &&
            (*UI)->getOperand(0) == N->getOperand(0)) {
          VCMPoNode = *UI;
          break;
        }
      
      // If there are non-zero uses of the flag value, use the VCMPo node!
      if (VCMPoNode && !VCMPoNode->hasNUsesOfValue(0, 1))
        return SDOperand(VCMPoNode, 0);
    }
    break;
  }
  }
  
  return SDOperand();
}

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
PPCTargetLowering::ConstraintType 
PPCTargetLowering::getConstraintType(char ConstraintLetter) const {
  switch (ConstraintLetter) {
  default: break;
  case 'b':
  case 'r':
  case 'f':
  case 'v':
  case 'y':
    return C_RegisterClass;
  }  
  return TargetLowering::getConstraintType(ConstraintLetter);
}


std::vector<unsigned> PPCTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {      // GCC RS6000 Constraint Letters
    default: break;  // Unknown constriant letter
    case 'b': 
      return make_vector<unsigned>(/*no R0*/ PPC::R1 , PPC::R2 , PPC::R3 ,
                                   PPC::R4 , PPC::R5 , PPC::R6 , PPC::R7 ,
                                   PPC::R8 , PPC::R9 , PPC::R10, PPC::R11, 
                                   PPC::R12, PPC::R13, PPC::R14, PPC::R15, 
                                   PPC::R16, PPC::R17, PPC::R18, PPC::R19, 
                                   PPC::R20, PPC::R21, PPC::R22, PPC::R23, 
                                   PPC::R24, PPC::R25, PPC::R26, PPC::R27, 
                                   PPC::R28, PPC::R29, PPC::R30, PPC::R31, 
                                   0);
    case 'r': 
      return make_vector<unsigned>(PPC::R0 , PPC::R1 , PPC::R2 , PPC::R3 ,
                                   PPC::R4 , PPC::R5 , PPC::R6 , PPC::R7 ,
                                   PPC::R8 , PPC::R9 , PPC::R10, PPC::R11, 
                                   PPC::R12, PPC::R13, PPC::R14, PPC::R15, 
                                   PPC::R16, PPC::R17, PPC::R18, PPC::R19, 
                                   PPC::R20, PPC::R21, PPC::R22, PPC::R23, 
                                   PPC::R24, PPC::R25, PPC::R26, PPC::R27, 
                                   PPC::R28, PPC::R29, PPC::R30, PPC::R31, 
                                   0);
    case 'f': 
      return make_vector<unsigned>(PPC::F0 , PPC::F1 , PPC::F2 , PPC::F3 ,
                                   PPC::F4 , PPC::F5 , PPC::F6 , PPC::F7 ,
                                   PPC::F8 , PPC::F9 , PPC::F10, PPC::F11, 
                                   PPC::F12, PPC::F13, PPC::F14, PPC::F15, 
                                   PPC::F16, PPC::F17, PPC::F18, PPC::F19, 
                                   PPC::F20, PPC::F21, PPC::F22, PPC::F23, 
                                   PPC::F24, PPC::F25, PPC::F26, PPC::F27, 
                                   PPC::F28, PPC::F29, PPC::F30, PPC::F31, 
                                   0);
    case 'v': 
      return make_vector<unsigned>(PPC::V0 , PPC::V1 , PPC::V2 , PPC::V3 ,
                                   PPC::V4 , PPC::V5 , PPC::V6 , PPC::V7 ,
                                   PPC::V8 , PPC::V9 , PPC::V10, PPC::V11, 
                                   PPC::V12, PPC::V13, PPC::V14, PPC::V15, 
                                   PPC::V16, PPC::V17, PPC::V18, PPC::V19, 
                                   PPC::V20, PPC::V21, PPC::V22, PPC::V23, 
                                   PPC::V24, PPC::V25, PPC::V26, PPC::V27, 
                                   PPC::V28, PPC::V29, PPC::V30, PPC::V31, 
                                   0);
    case 'y': 
      return make_vector<unsigned>(PPC::CR0, PPC::CR1, PPC::CR2, PPC::CR3,
                                   PPC::CR4, PPC::CR5, PPC::CR6, PPC::CR7,
                                   0);
    }
  }
  
  return std::vector<unsigned>();
}

// isOperandValidForConstraint
bool PPCTargetLowering::
isOperandValidForConstraint(SDOperand Op, char Letter) {
  switch (Letter) {
  default: break;
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P': {
    if (!isa<ConstantSDNode>(Op)) return false;  // Must be an immediate.
    unsigned Value = cast<ConstantSDNode>(Op)->getValue();
    switch (Letter) {
    default: assert(0 && "Unknown constraint letter!");
    case 'I':  // "I" is a signed 16-bit constant.
      return (short)Value == (int)Value;
    case 'J':  // "J" is a constant with only the high-order 16 bits nonzero.
    case 'L':  // "L" is a signed 16-bit constant shifted left 16 bits.
      return (short)Value == 0;
    case 'K':  // "K" is a constant with only the low-order 16 bits nonzero.
      return (Value >> 16) == 0;
    case 'M':  // "M" is a constant that is greater than 31.
      return Value > 31;
    case 'N':  // "N" is a positive constant that is an exact power of two.
      return (int)Value > 0 && isPowerOf2_32(Value);
    case 'O':  // "O" is the constant zero. 
      return Value == 0;
    case 'P':  // "P" is a constant whose negation is a signed 16-bit constant.
      return (short)-Value == (int)-Value;
    }
    break;
  }
  }
  
  // Handle standard constraint letters.
  return TargetLowering::isOperandValidForConstraint(Op, Letter);
}

/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode.
bool PPCTargetLowering::isLegalAddressImmediate(int64_t V) const {
  // PPC allows a sign-extended 16-bit immediate field.
  return (V > -(1 << 16) && V < (1 << 16)-1);
}

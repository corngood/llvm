//===-- ARMISelLowering.cpp - ARM DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that ARM uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMConstantPoolValue.h"
#include "ARMISelLowering.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
    : TargetLowering(TM), ARMPCLabelIndex(0) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();

  if (Subtarget->isTargetDarwin()) {
    // Don't have these.
    setLibcallName(RTLIB::UINTTOFP_I64_F32, NULL);
    setLibcallName(RTLIB::UINTTOFP_I64_F64, NULL);

    // Uses VFP for Thumb libfuncs if available.
    if (Subtarget->isThumb() && Subtarget->hasVFP2()) {
      // Single-precision floating-point arithmetic.
      setLibcallName(RTLIB::ADD_F32, "__addsf3vfp");
      setLibcallName(RTLIB::SUB_F32, "__subsf3vfp");
      setLibcallName(RTLIB::MUL_F32, "__mulsf3vfp");
      setLibcallName(RTLIB::DIV_F32, "__divsf3vfp");

      // Double-precision floating-point arithmetic.
      setLibcallName(RTLIB::ADD_F64, "__adddf3vfp");
      setLibcallName(RTLIB::SUB_F64, "__subdf3vfp");
      setLibcallName(RTLIB::MUL_F64, "__muldf3vfp");
      setLibcallName(RTLIB::DIV_F64, "__divdf3vfp");

      // Single-precision comparisons.
      setLibcallName(RTLIB::OEQ_F32, "__eqsf2vfp");
      setLibcallName(RTLIB::UNE_F32, "__nesf2vfp");
      setLibcallName(RTLIB::OLT_F32, "__ltsf2vfp");
      setLibcallName(RTLIB::OLE_F32, "__lesf2vfp");
      setLibcallName(RTLIB::OGE_F32, "__gesf2vfp");
      setLibcallName(RTLIB::OGT_F32, "__gtsf2vfp");
      setLibcallName(RTLIB::UO_F32,  "__unordsf2vfp");
      setLibcallName(RTLIB::O_F32,   "__unordsf2vfp");

      setCmpLibcallCC(RTLIB::OEQ_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UNE_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLT_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLE_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGE_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGT_F32, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UO_F32,  ISD::SETNE);
      setCmpLibcallCC(RTLIB::O_F32,   ISD::SETEQ);

      // Double-precision comparisons.
      setLibcallName(RTLIB::OEQ_F64, "__eqdf2vfp");
      setLibcallName(RTLIB::UNE_F64, "__nedf2vfp");
      setLibcallName(RTLIB::OLT_F64, "__ltdf2vfp");
      setLibcallName(RTLIB::OLE_F64, "__ledf2vfp");
      setLibcallName(RTLIB::OGE_F64, "__gedf2vfp");
      setLibcallName(RTLIB::OGT_F64, "__gtdf2vfp");
      setLibcallName(RTLIB::UO_F64,  "__unorddf2vfp");
      setLibcallName(RTLIB::O_F64,   "__unorddf2vfp");

      setCmpLibcallCC(RTLIB::OEQ_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UNE_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLT_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OLE_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGE_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::OGT_F64, ISD::SETNE);
      setCmpLibcallCC(RTLIB::UO_F64,  ISD::SETNE);
      setCmpLibcallCC(RTLIB::O_F64,   ISD::SETEQ);

      // Floating-point to integer conversions.
      // i64 conversions are done via library routines even when generating VFP
      // instructions, so use the same ones.
      setLibcallName(RTLIB::FPTOSINT_F64_I32, "__fixdfsivfp");
      setLibcallName(RTLIB::FPTOUINT_F64_I32, "__fixunsdfsivfp");
      setLibcallName(RTLIB::FPTOSINT_F32_I32, "__fixsfsivfp");
      setLibcallName(RTLIB::FPTOUINT_F32_I32, "__fixunssfsivfp");

      // Conversions between floating types.
      setLibcallName(RTLIB::FPROUND_F64_F32, "__truncdfsf2vfp");
      setLibcallName(RTLIB::FPEXT_F32_F64,   "__extendsfdf2vfp");

      // Integer to floating-point conversions.
      // i64 conversions are done via library routines even when generating VFP
      // instructions, so use the same ones.
      // FIXME: There appears to be some naming inconsistency in ARM libgcc: e.g.
      // __floatunsidf vs. __floatunssidfvfp.
      setLibcallName(RTLIB::SINTTOFP_I32_F64, "__floatsidfvfp");
      setLibcallName(RTLIB::UINTTOFP_I32_F64, "__floatunssidfvfp");
      setLibcallName(RTLIB::SINTTOFP_I32_F32, "__floatsisfvfp");
      setLibcallName(RTLIB::UINTTOFP_I32_F32, "__floatunssisfvfp");
    }
  }

  addRegisterClass(MVT::i32, ARM::GPRRegisterClass);
  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb()) {
    addRegisterClass(MVT::f32, ARM::SPRRegisterClass);
    addRegisterClass(MVT::f64, ARM::DPRRegisterClass);
  }

  // ARM does not have f32 extending load.
  setLoadXAction(ISD::EXTLOAD, MVT::f32, Expand);

  // ARM supports all 4 flavors of integer indexed load / store.
  for (unsigned im = (unsigned)ISD::PRE_INC;
       im != (unsigned)ISD::LAST_INDEXED_MODE; ++im) {
    setIndexedLoadAction(im,  MVT::i1,  Legal);
    setIndexedLoadAction(im,  MVT::i8,  Legal);
    setIndexedLoadAction(im,  MVT::i16, Legal);
    setIndexedLoadAction(im,  MVT::i32, Legal);
    setIndexedStoreAction(im, MVT::i1,  Legal);
    setIndexedStoreAction(im, MVT::i8,  Legal);
    setIndexedStoreAction(im, MVT::i16, Legal);
    setIndexedStoreAction(im, MVT::i32, Legal);
  }

  // i64 operation support.
  if (Subtarget->isThumb()) {
    setOperationAction(ISD::MUL,     MVT::i64, Expand);
    setOperationAction(ISD::MULHU,   MVT::i32, Expand);
    setOperationAction(ISD::MULHS,   MVT::i32, Expand);
  } else {
    setOperationAction(ISD::MUL,     MVT::i64, Custom);
    setOperationAction(ISD::MULHU,   MVT::i32, Custom);
    if (!Subtarget->hasV6Ops())
      setOperationAction(ISD::MULHS, MVT::i32, Custom);
  }
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL,       MVT::i64, Custom);
  setOperationAction(ISD::SRA,       MVT::i64, Custom);

  // ARM does not have ROTL.
  setOperationAction(ISD::ROTL,  MVT::i32, Expand);
  setOperationAction(ISD::CTTZ , MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  if (!Subtarget->hasV5TOps() || Subtarget->isThumb())
    setOperationAction(ISD::CTLZ, MVT::i32, Expand);

  // Only ARMv6 has BSWAP.
  if (!Subtarget->hasV6Ops())
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  // These are expanded into libcalls.
  setOperationAction(ISD::SDIV,  MVT::i32, Expand);
  setOperationAction(ISD::UDIV,  MVT::i32, Expand);
  setOperationAction(ISD::SREM,  MVT::i32, Expand);
  setOperationAction(ISD::UREM,  MVT::i32, Expand);
  
  // Support label based line numbers.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  // FIXME - use subtarget debug flags
  if (!Subtarget->isTargetDarwin())
    setOperationAction(ISD::LABEL, MVT::Other, Expand);

  setOperationAction(ISD::RET,           MVT::Other, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32,   Custom);
  setOperationAction(ISD::GLOBAL_OFFSET_TABLE, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);

  // Expand mem operations genericly.
  setOperationAction(ISD::MEMSET          , MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY          , MVT::Other, Expand);
  setOperationAction(ISD::MEMMOVE         , MVT::Other, Expand);
  
  // Use the default implementation.
  setOperationAction(ISD::VASTART           , MVT::Other, Expand);
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE,          MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE,       MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);

  if (!Subtarget->hasV6Ops()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8,  Expand);
  }
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  if (!UseSoftFloat && Subtarget->hasVFP2() && !Subtarget->isThumb())
    // Turn f64->i64 into FMRRD iff target supports vfp2.
    setOperationAction(ISD::BIT_CONVERT, MVT::i64, Custom);
  
  setOperationAction(ISD::SETCC    , MVT::i32, Expand);
  setOperationAction(ISD::SETCC    , MVT::f32, Expand);
  setOperationAction(ISD::SETCC    , MVT::f64, Expand);
  setOperationAction(ISD::SELECT   , MVT::i32, Expand);
  setOperationAction(ISD::SELECT   , MVT::f32, Expand);
  setOperationAction(ISD::SELECT   , MVT::f64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  setOperationAction(ISD::BRCOND   , MVT::Other, Expand);
  setOperationAction(ISD::BR_CC    , MVT::i32,   Custom);
  setOperationAction(ISD::BR_CC    , MVT::f32,   Custom);
  setOperationAction(ISD::BR_CC    , MVT::f64,   Custom);
  setOperationAction(ISD::BR_JT    , MVT::Other, Custom);

  setOperationAction(ISD::VASTART,       MVT::Other, Custom);
  setOperationAction(ISD::VACOPY,        MVT::Other, Expand); 
  setOperationAction(ISD::VAEND,         MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE,     MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE,  MVT::Other, Expand);

  // FP Constants can't be immediates.
  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);

  // We don't support sin/cos/fmod/copysign
  setOperationAction(ISD::FSIN     , MVT::f64, Expand);
  setOperationAction(ISD::FSIN     , MVT::f32, Expand);
  setOperationAction(ISD::FCOS     , MVT::f32, Expand);
  setOperationAction(ISD::FCOS     , MVT::f64, Expand);
  setOperationAction(ISD::FREM     , MVT::f64, Expand);
  setOperationAction(ISD::FREM     , MVT::f32, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);
  
  // int <-> fp are custom expanded into bit_convert + ARMISD ops.
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);

  setStackPointerRegisterToSaveRestore(ARM::SP);

  setSchedulingPreference(SchedulingForRegPressure);
  computeRegisterProperties();
}


const char *ARMTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case ARMISD::Wrapper:       return "ARMISD::Wrapper";
  case ARMISD::WrapperJT:     return "ARMISD::WrapperJT";
  case ARMISD::CALL:          return "ARMISD::CALL";
  case ARMISD::CALL_NOLINK:   return "ARMISD::CALL_NOLINK";
  case ARMISD::tCALL:         return "ARMISD::tCALL";
  case ARMISD::BRCOND:        return "ARMISD::BRCOND";
  case ARMISD::BR_JT:         return "ARMISD::BR_JT";
  case ARMISD::RET_FLAG:      return "ARMISD::RET_FLAG";
  case ARMISD::PIC_ADD:       return "ARMISD::PIC_ADD";
  case ARMISD::CMP:           return "ARMISD::CMP";
  case ARMISD::CMPNZ:         return "ARMISD::CMPNZ";
  case ARMISD::CMPFP:         return "ARMISD::CMPFP";
  case ARMISD::CMPFPw0:       return "ARMISD::CMPFPw0";
  case ARMISD::FMSTAT:        return "ARMISD::FMSTAT";
  case ARMISD::CMOV:          return "ARMISD::CMOV";
  case ARMISD::CNEG:          return "ARMISD::CNEG";
    
  case ARMISD::FTOSI:         return "ARMISD::FTOSI";
  case ARMISD::FTOUI:         return "ARMISD::FTOUI";
  case ARMISD::SITOF:         return "ARMISD::SITOF";
  case ARMISD::UITOF:         return "ARMISD::UITOF";
  case ARMISD::MULHILOU:      return "ARMISD::MULHILOU";
  case ARMISD::MULHILOS:      return "ARMISD::MULHILOS";

  case ARMISD::SRL_FLAG:      return "ARMISD::SRL_FLAG";
  case ARMISD::SRA_FLAG:      return "ARMISD::SRA_FLAG";
  case ARMISD::RRX:           return "ARMISD::RRX";
      
  case ARMISD::FMRRD:         return "ARMISD::FMRRD";
  case ARMISD::FMDRR:         return "ARMISD::FMDRR";

  case ARMISD::THREAD_POINTER:return "ARMISD::THREAD_POINTER";
  }
}

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//


/// IntCCToARMCC - Convert a DAG integer condition code to an ARM CC
static ARMCC::CondCodes IntCCToARMCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition code!");
  case ISD::SETNE:  return ARMCC::NE;
  case ISD::SETEQ:  return ARMCC::EQ;
  case ISD::SETGT:  return ARMCC::GT;
  case ISD::SETGE:  return ARMCC::GE;
  case ISD::SETLT:  return ARMCC::LT;
  case ISD::SETLE:  return ARMCC::LE;
  case ISD::SETUGT: return ARMCC::HI;
  case ISD::SETUGE: return ARMCC::HS;
  case ISD::SETULT: return ARMCC::LO;
  case ISD::SETULE: return ARMCC::LS;
  }
}

/// FPCCToARMCC - Convert a DAG fp condition code to an ARM CC. It
/// returns true if the operands should be inverted to form the proper
/// comparison.
static bool FPCCToARMCC(ISD::CondCode CC, ARMCC::CondCodes &CondCode,
                        ARMCC::CondCodes &CondCode2) {
  bool Invert = false;
  CondCode2 = ARMCC::AL;
  switch (CC) {
  default: assert(0 && "Unknown FP condition!");
  case ISD::SETEQ:
  case ISD::SETOEQ: CondCode = ARMCC::EQ; break;
  case ISD::SETGT:
  case ISD::SETOGT: CondCode = ARMCC::GT; break;
  case ISD::SETGE:
  case ISD::SETOGE: CondCode = ARMCC::GE; break;
  case ISD::SETOLT: CondCode = ARMCC::MI; break;
  case ISD::SETOLE: CondCode = ARMCC::GT; Invert = true; break;
  case ISD::SETONE: CondCode = ARMCC::MI; CondCode2 = ARMCC::GT; break;
  case ISD::SETO:   CondCode = ARMCC::VC; break;
  case ISD::SETUO:  CondCode = ARMCC::VS; break;
  case ISD::SETUEQ: CondCode = ARMCC::EQ; CondCode2 = ARMCC::VS; break;
  case ISD::SETUGT: CondCode = ARMCC::HI; break;
  case ISD::SETUGE: CondCode = ARMCC::PL; break;
  case ISD::SETLT:
  case ISD::SETULT: CondCode = ARMCC::LT; break;
  case ISD::SETLE:
  case ISD::SETULE: CondCode = ARMCC::LE; break;
  case ISD::SETNE:
  case ISD::SETUNE: CondCode = ARMCC::NE; break;
  }
  return Invert;
}

static void
HowToPassArgument(MVT::ValueType ObjectVT, unsigned NumGPRs,
                  unsigned StackOffset, unsigned &NeededGPRs,
                  unsigned &NeededStackSize, unsigned &GPRPad,
                  unsigned &StackPad, unsigned Flags) {
  NeededStackSize = 0;
  NeededGPRs = 0;
  StackPad = 0;
  GPRPad = 0;
  unsigned align = (Flags >> ISD::ParamFlags::OrigAlignmentOffs);
  GPRPad = NumGPRs % ((align + 3)/4);
  StackPad = StackOffset % align;
  unsigned firstGPR = NumGPRs + GPRPad;
  switch (ObjectVT) {
  default: assert(0 && "Unhandled argument type!");
  case MVT::i32:
  case MVT::f32:
    if (firstGPR < 4)
      NeededGPRs = 1;
    else
      NeededStackSize = 4;
    break;
  case MVT::i64:
  case MVT::f64:
    if (firstGPR < 3)
      NeededGPRs = 2;
    else if (firstGPR == 3) {
      NeededGPRs = 1;
      NeededStackSize = 4;
    } else
      NeededStackSize = 8;
  }
}

/// LowerCALL - Lowering a ISD::CALL node into a callseq_start <-
/// ARMISD:CALL <- callseq_end chain. Also add input and output parameter
/// nodes.
SDOperand ARMTargetLowering::LowerCALL(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType RetVT= Op.Val->getValueType(0);
  SDOperand Chain    = Op.getOperand(0);
  unsigned CallConv  = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  assert((CallConv == CallingConv::C ||
          CallConv == CallingConv::Fast) && "unknown calling convention");
  SDOperand Callee   = Op.getOperand(4);
  unsigned NumOps    = (Op.getNumOperands() - 5) / 2;
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  unsigned NumGPRs = 0;     // GPRs used for parameter passing.

  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;

  // Add up all the space actually used.
  for (unsigned i = 0; i < NumOps; ++i) {
    unsigned ObjSize;
    unsigned ObjGPRs;
    unsigned StackPad;
    unsigned GPRPad;
    MVT::ValueType ObjectVT = Op.getOperand(5+2*i).getValueType();
    unsigned Flags = Op.getConstantOperandVal(5+2*i+1);
    HowToPassArgument(ObjectVT, NumGPRs, NumBytes, ObjGPRs, ObjSize,
                      GPRPad, StackPad, Flags);
    NumBytes += ObjSize + StackPad;
    NumGPRs += ObjGPRs + GPRPad;
  }

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getConstant(NumBytes, MVT::i32));

  SDOperand StackPtr = DAG.getRegister(ARM::SP, MVT::i32);

  static const unsigned GPRArgRegs[] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  NumGPRs = 0;
  std::vector<std::pair<unsigned, SDOperand> > RegsToPass;
  std::vector<SDOperand> MemOpChains;
  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);
    unsigned Flags = Op.getConstantOperandVal(5+2*i+1);
    MVT::ValueType ArgVT = Arg.getValueType();

    unsigned ObjSize;
    unsigned ObjGPRs;
    unsigned GPRPad;
    unsigned StackPad;
    HowToPassArgument(ArgVT, NumGPRs, ArgOffset, ObjGPRs,
                      ObjSize, GPRPad, StackPad, Flags);
    NumGPRs += GPRPad;
    ArgOffset += StackPad;
    if (ObjGPRs > 0) {
      switch (ArgVT) {
      default: assert(0 && "Unexpected ValueType for argument!");
      case MVT::i32:
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs], Arg));
        break;
      case MVT::f32:
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs],
                                 DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Arg)));
        break;
      case MVT::i64: {
        SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Arg,
                                   DAG.getConstant(0, getPointerTy()));
        SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Arg,
                                   DAG.getConstant(1, getPointerTy()));
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs], Lo));
        if (ObjGPRs == 2)
          RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs+1], Hi));
        else {
          SDOperand PtrOff= DAG.getConstant(ArgOffset, StackPtr.getValueType());
          PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
          MemOpChains.push_back(DAG.getStore(Chain, Hi, PtrOff, NULL, 0));
        }
        break;
      }
      case MVT::f64: {
        SDOperand Cvt = DAG.getNode(ARMISD::FMRRD,
                                    DAG.getVTList(MVT::i32, MVT::i32),
                                    &Arg, 1);
        RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs], Cvt));
        if (ObjGPRs == 2)
          RegsToPass.push_back(std::make_pair(GPRArgRegs[NumGPRs+1],
                                              Cvt.getValue(1)));
        else {
          SDOperand PtrOff= DAG.getConstant(ArgOffset, StackPtr.getValueType());
          PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
          MemOpChains.push_back(DAG.getStore(Chain, Cvt.getValue(1), PtrOff,
                                             NULL, 0));
        }
        break;
      }
      }
    } else {
      assert(ObjSize != 0);
      SDOperand PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
    }

    NumGPRs += ObjGPRs;
    ArgOffset += ObjSize;
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  bool isDirect = false;
  bool isARMFunc = false;
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    GlobalValue *GV = G->getGlobal();
    isDirect = true;
    bool isExt = (GV->isDeclaration() || GV->hasWeakLinkage() ||
                  GV->hasLinkOnceLinkage());
    bool isStub = (isExt && Subtarget->isTargetDarwin()) &&
                   getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // tBX takes a register source operand.
    if (isARMFunc && Subtarget->isThumb() && !Subtarget->hasV5TOps()) {
      ARMConstantPoolValue *CPV = new ARMConstantPoolValue(GV, ARMPCLabelIndex,
                                                           ARMCP::CPStub, 4);
      SDOperand CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 2);
      CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), CPAddr, NULL, 0); 
      SDOperand PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, getPointerTy(), Callee, PICLabel);
   } else
      Callee = DAG.getTargetGlobalAddress(GV, getPointerTy());
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    isDirect = true;
    bool isStub = Subtarget->isTargetDarwin() &&
                  getTargetMachine().getRelocationModel() != Reloc::Static;
    isARMFunc = !Subtarget->isThumb() || isStub;
    // tBX takes a register source operand.
    const char *Sym = S->getSymbol();
    if (isARMFunc && Subtarget->isThumb() && !Subtarget->hasV5TOps()) {
      ARMConstantPoolValue *CPV = new ARMConstantPoolValue(Sym, ARMPCLabelIndex,
                                                           ARMCP::CPStub, 4);
      SDOperand CPAddr = DAG.getTargetConstantPool(CPV, getPointerTy(), 2);
      CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
      Callee = DAG.getLoad(getPointerTy(), DAG.getEntryNode(), CPAddr, NULL, 0); 
      SDOperand PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
      Callee = DAG.getNode(ARMISD::PIC_ADD, getPointerTy(), Callee, PICLabel);
    } else
      Callee = DAG.getTargetExternalSymbol(Sym, getPointerTy());
  }

  // FIXME: handle tail calls differently.
  unsigned CallOpc;
  if (Subtarget->isThumb()) {
    if (!Subtarget->hasV5TOps() && (!isDirect || isARMFunc))
      CallOpc = ARMISD::CALL_NOLINK;
    else
      CallOpc = isARMFunc ? ARMISD::CALL : ARMISD::tCALL;
  } else {
    CallOpc = (isDirect || Subtarget->hasV5TOps())
      ? ARMISD::CALL : ARMISD::CALL_NOLINK;
  }
  if (CallOpc == ARMISD::CALL_NOLINK && !Subtarget->isThumb()) {
    // implicit def LR - LR mustn't be allocated as GRP:$dst of CALL_NOLINK
    Chain = DAG.getCopyToReg(Chain, ARM::LR,
                             DAG.getNode(ISD::UNDEF, MVT::i32), InFlag);
    InFlag = Chain.getValue(1);
  }

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.

  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.Val)
    Ops.push_back(InFlag);
  Chain = DAG.getNode(CallOpc, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  SDOperand CSOps[] = { Chain, DAG.getConstant(NumBytes, MVT::i32), InFlag };
  Chain = DAG.getNode(ISD::CALLSEQ_END, 
                      DAG.getNodeValueTypes(MVT::Other, MVT::Flag),
                      ((RetVT != MVT::Other) ? 2 : 1), CSOps, 3);
  if (RetVT != MVT::Other)
    InFlag = Chain.getValue(1);

  std::vector<SDOperand> ResultVals;
  NodeTys.clear();

  // If the call has results, copy the values out of the ret val registers.
  switch (RetVT) {
  default: assert(0 && "Unexpected ret value!");
  case MVT::Other:
    break;
  case MVT::i32:
    Chain = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    if (Op.Val->getValueType(1) == MVT::i32) {
      // Returns a i64 value.
      Chain = DAG.getCopyFromReg(Chain, ARM::R1, MVT::i32,
                                 Chain.getValue(2)).getValue(1);
      ResultVals.push_back(Chain.getValue(0));
      NodeTys.push_back(MVT::i32);
    }
    NodeTys.push_back(MVT::i32);
    break;
  case MVT::f32:
    Chain = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag).getValue(1);
    ResultVals.push_back(DAG.getNode(ISD::BIT_CONVERT, MVT::f32,
                                     Chain.getValue(0)));
    NodeTys.push_back(MVT::f32);
    break;
  case MVT::f64: {
    SDOperand Lo = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag);
    SDOperand Hi = DAG.getCopyFromReg(Lo, ARM::R1, MVT::i32, Lo.getValue(2));
    ResultVals.push_back(DAG.getNode(ARMISD::FMDRR, MVT::f64, Lo, Hi));
    NodeTys.push_back(MVT::f64);
    break;
  }
  }

  NodeTys.push_back(MVT::Other);

  if (ResultVals.empty())
    return Chain;

  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys, &ResultVals[0],
                              ResultVals.size());
  return Res.getValue(Op.ResNo);
}

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Copy;
  SDOperand Chain = Op.getOperand(0);
  switch(Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: {
    SDOperand LR = DAG.getRegister(ARM::LR, MVT::i32);
    return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Chain);
  }
  case 3:
    Op = Op.getOperand(1);
    if (Op.getValueType() == MVT::f32) {
      Op = DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Op);
    } else if (Op.getValueType() == MVT::f64) {
      // Recursively legalize f64 -> i64.
      Op = DAG.getNode(ISD::BIT_CONVERT, MVT::i64, Op);
      return DAG.getNode(ISD::RET, MVT::Other, Chain, Op,
                         DAG.getConstant(0, MVT::i32));
    }
    Copy = DAG.getCopyToReg(Chain, ARM::R0, Op, SDOperand());
    if (DAG.getMachineFunction().liveout_empty())
      DAG.getMachineFunction().addLiveOut(ARM::R0);
    break;
  case 5:
    Copy = DAG.getCopyToReg(Chain, ARM::R1, Op.getOperand(3), SDOperand());
    Copy = DAG.getCopyToReg(Copy, ARM::R0, Op.getOperand(1), Copy.getValue(1));
    // If we haven't noted the R0+R1 are live out, do so now.
    if (DAG.getMachineFunction().liveout_empty()) {
      DAG.getMachineFunction().addLiveOut(ARM::R0);
      DAG.getMachineFunction().addLiveOut(ARM::R1);
    }
    break;
  }

  //We must use RET_FLAG instead of BRIND because BRIND doesn't have a flag
  return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
}

// ConstantPool, JumpTable, GlobalAddress, and ExternalSymbol are lowered as 
// their target countpart wrapped in the ARMISD::Wrapper node. Suppose N is
// one of the above mentioned nodes. It has to be wrapped because otherwise
// Select(N) returns N. So the raw TargetGlobalAddress nodes, etc. can only
// be used to form addressing mode. These wrapped nodes will be selected
// into MOVi.
static SDOperand LowerConstantPool(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  SDOperand Res;
  if (CP->isMachineConstantPoolEntry())
    Res = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
                                    CP->getAlignment());
  else
    Res = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
                                    CP->getAlignment());
  return DAG.getNode(ARMISD::Wrapper, MVT::i32, Res);
}

// Lower ISD::GlobalTLSAddress using the "general dynamic" model
SDOperand
ARMTargetLowering::LowerToTLSGeneralDynamicModel(GlobalAddressSDNode *GA,
                                                 SelectionDAG &DAG) {
  MVT::ValueType PtrVT = getPointerTy();
  unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
  ARMConstantPoolValue *CPV =
    new ARMConstantPoolValue(GA->getGlobal(), ARMPCLabelIndex, ARMCP::CPValue,
                             PCAdj, "tlsgd", true);
  SDOperand Argument = DAG.getTargetConstantPool(CPV, PtrVT, 2);
  Argument = DAG.getNode(ARMISD::Wrapper, MVT::i32, Argument);
  Argument = DAG.getLoad(PtrVT, DAG.getEntryNode(), Argument, NULL, 0);
  SDOperand Chain = Argument.getValue(1);

  SDOperand PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
  Argument = DAG.getNode(ARMISD::PIC_ADD, PtrVT, Argument, PICLabel);

  // call __tls_get_addr.
  ArgListTy Args;
  ArgListEntry Entry;
  Entry.Node = Argument;
  Entry.Ty = (const Type *) Type::Int32Ty;
  Args.push_back(Entry);
  std::pair<SDOperand, SDOperand> CallResult =
    LowerCallTo(Chain, (const Type *) Type::Int32Ty, false, false,
                CallingConv::C, false,
                DAG.getExternalSymbol("__tls_get_addr", PtrVT), Args, DAG);
  return CallResult.first;
}

// Lower ISD::GlobalTLSAddress using the "initial exec" or
// "local exec" model.
SDOperand
ARMTargetLowering::LowerToTLSExecModels(GlobalAddressSDNode *GA,
                                            SelectionDAG &DAG) {
  GlobalValue *GV = GA->getGlobal();
  SDOperand Offset;
  SDOperand Chain = DAG.getEntryNode();
  MVT::ValueType PtrVT = getPointerTy();
  // Get the Thread Pointer
  SDOperand ThreadPointer = DAG.getNode(ARMISD::THREAD_POINTER, PtrVT);

  if (GV->isDeclaration()){
    // initial exec model
    unsigned char PCAdj = Subtarget->isThumb() ? 4 : 8;
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GA->getGlobal(), ARMPCLabelIndex, ARMCP::CPValue,
                               PCAdj, "gottpoff", true);
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 2);
    Offset = DAG.getNode(ARMISD::Wrapper, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, Chain, Offset, NULL, 0);
    Chain = Offset.getValue(1);

    SDOperand PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
    Offset = DAG.getNode(ARMISD::PIC_ADD, PtrVT, Offset, PICLabel);

    Offset = DAG.getLoad(PtrVT, Chain, Offset, NULL, 0);
  } else {
    // local exec model
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GV, ARMCP::CPValue, "tpoff");
    Offset = DAG.getTargetConstantPool(CPV, PtrVT, 2);
    Offset = DAG.getNode(ARMISD::Wrapper, MVT::i32, Offset);
    Offset = DAG.getLoad(PtrVT, Chain, Offset, NULL, 0);
  }

  // The address of the thread local variable is the add of the thread
  // pointer with the offset of the variable.
  return DAG.getNode(ISD::ADD, PtrVT, ThreadPointer, Offset);
}

SDOperand
ARMTargetLowering::LowerGlobalTLSAddress(SDOperand Op, SelectionDAG &DAG) {
  // TODO: implement the "local dynamic" model
  assert(Subtarget->isTargetELF() &&
         "TLS not implemented for non-ELF targets");
  GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);
  // If the relocation model is PIC, use the "General Dynamic" TLS Model,
  // otherwise use the "Local Exec" TLS Model
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_)
    return LowerToTLSGeneralDynamicModel(GA, DAG);
  else
    return LowerToTLSExecModels(GA, DAG);
}

SDOperand ARMTargetLowering::LowerGlobalAddressELF(SDOperand Op,
                                                   SelectionDAG &DAG) {
  MVT::ValueType PtrVT = getPointerTy();
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  if (RelocM == Reloc::PIC_) {
    bool UseGOTOFF = GV->hasInternalLinkage();
    ARMConstantPoolValue *CPV =
      new ARMConstantPoolValue(GV, ARMCP::CPValue, UseGOTOFF ? "GOTOFF":"GOT");
    SDOperand CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 2);
    CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
    SDOperand Result = DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
    SDOperand Chain = Result.getValue(1);
    SDOperand GOT = DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, PtrVT);
    Result = DAG.getNode(ISD::ADD, PtrVT, Result, GOT);
    if (!UseGOTOFF)
      Result = DAG.getLoad(PtrVT, Chain, Result, NULL, 0);
    return Result;
  } else {
    SDOperand CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 2);
    CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
    return DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
  }
}

/// GVIsIndirectSymbol - true if the GV will be accessed via an indirect symbol
/// even in dynamic-no-pic mode.
static bool GVIsIndirectSymbol(GlobalValue *GV) {
  return (GV->hasWeakLinkage() || GV->hasLinkOnceLinkage() ||
          (GV->isDeclaration() && !GV->hasNotBeenReadFromBytecode()));
}

SDOperand ARMTargetLowering::LowerGlobalAddressDarwin(SDOperand Op,
                                                      SelectionDAG &DAG) {
  MVT::ValueType PtrVT = getPointerTy();
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  bool IsIndirect = GVIsIndirectSymbol(GV);
  SDOperand CPAddr;
  if (RelocM == Reloc::Static)
    CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 2);
  else {
    unsigned PCAdj = (RelocM != Reloc::PIC_)
      ? 0 : (Subtarget->isThumb() ? 4 : 8);
    ARMCP::ARMCPKind Kind = IsIndirect ? ARMCP::CPNonLazyPtr
      : ARMCP::CPValue;
    ARMConstantPoolValue *CPV = new ARMConstantPoolValue(GV, ARMPCLabelIndex,
                                                         Kind, PCAdj);
    CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 2);
  }
  CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);

  SDOperand Result = DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
  SDOperand Chain = Result.getValue(1);

  if (RelocM == Reloc::PIC_) {
    SDOperand PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
    Result = DAG.getNode(ARMISD::PIC_ADD, PtrVT, Result, PICLabel);
  }
  if (IsIndirect)
    Result = DAG.getLoad(PtrVT, Chain, Result, NULL, 0);

  return Result;
}

SDOperand ARMTargetLowering::LowerGLOBAL_OFFSET_TABLE(SDOperand Op,
                                                      SelectionDAG &DAG){
  assert(Subtarget->isTargetELF() &&
         "GLOBAL OFFSET TABLE not implemented for non-ELF targets");
  MVT::ValueType PtrVT = getPointerTy();
  unsigned PCAdj = Subtarget->isThumb() ? 4 : 8;
  ARMConstantPoolValue *CPV = new ARMConstantPoolValue("_GLOBAL_OFFSET_TABLE_",
                                                       ARMPCLabelIndex,
                                                       ARMCP::CPValue, PCAdj);
  SDOperand CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, 2);
  CPAddr = DAG.getNode(ARMISD::Wrapper, MVT::i32, CPAddr);
  SDOperand Result = DAG.getLoad(PtrVT, DAG.getEntryNode(), CPAddr, NULL, 0);
  SDOperand PICLabel = DAG.getConstant(ARMPCLabelIndex++, MVT::i32);
  return DAG.getNode(ARMISD::PIC_ADD, PtrVT, Result, PICLabel);
}

static SDOperand LowerVASTART(SDOperand Op, SelectionDAG &DAG,
                              unsigned VarArgsFrameIndex) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  MVT::ValueType PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
  SrcValueSDNode *SV = cast<SrcValueSDNode>(Op.getOperand(2));
  return DAG.getStore(Op.getOperand(0), FR, Op.getOperand(1), SV->getValue(),
                      SV->getOffset());
}

static SDOperand LowerFORMAL_ARGUMENT(SDOperand Op, SelectionDAG &DAG,
                                      unsigned *vRegs, unsigned ArgNo,
                                      unsigned &NumGPRs, unsigned &ArgOffset) {
  MachineFunction &MF = DAG.getMachineFunction();
  MVT::ValueType ObjectVT = Op.getValue(ArgNo).getValueType();
  SDOperand Root = Op.getOperand(0);
  std::vector<SDOperand> ArgValues;
  SSARegMap *RegMap = MF.getSSARegMap();

  static const unsigned GPRArgRegs[] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  unsigned ObjSize;
  unsigned ObjGPRs;
  unsigned GPRPad;
  unsigned StackPad;
  unsigned Flags = Op.getConstantOperandVal(ArgNo + 3);
  HowToPassArgument(ObjectVT, NumGPRs, ArgOffset, ObjGPRs,
                    ObjSize, GPRPad, StackPad, Flags);
  NumGPRs += GPRPad;
  ArgOffset += StackPad;

  SDOperand ArgValue;
  if (ObjGPRs == 1) {
    unsigned VReg = RegMap->createVirtualRegister(&ARM::GPRRegClass);
    MF.addLiveIn(GPRArgRegs[NumGPRs], VReg);
    vRegs[NumGPRs] = VReg;
    ArgValue = DAG.getCopyFromReg(Root, VReg, MVT::i32);
    if (ObjectVT == MVT::f32)
      ArgValue = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, ArgValue);
  } else if (ObjGPRs == 2) {
    unsigned VReg = RegMap->createVirtualRegister(&ARM::GPRRegClass);
    MF.addLiveIn(GPRArgRegs[NumGPRs], VReg);
    vRegs[NumGPRs] = VReg;
    ArgValue = DAG.getCopyFromReg(Root, VReg, MVT::i32);

    VReg = RegMap->createVirtualRegister(&ARM::GPRRegClass);
    MF.addLiveIn(GPRArgRegs[NumGPRs+1], VReg);
    vRegs[NumGPRs+1] = VReg;
    SDOperand ArgValue2 = DAG.getCopyFromReg(Root, VReg, MVT::i32);

    if (ObjectVT == MVT::i64)
      ArgValue = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, ArgValue, ArgValue2);
    else
      ArgValue = DAG.getNode(ARMISD::FMDRR, MVT::f64, ArgValue, ArgValue2);
  }
  NumGPRs += ObjGPRs;

  if (ObjSize) {
    // If the argument is actually used, emit a load from the right stack
    // slot.
    if (!Op.Val->hasNUsesOfValue(0, ArgNo)) {
      MachineFrameInfo *MFI = MF.getFrameInfo();
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
      if (ObjGPRs == 0)
        ArgValue = DAG.getLoad(ObjectVT, Root, FIN, NULL, 0);
      else {
        SDOperand ArgValue2 =
          DAG.getLoad(MVT::i32, Root, FIN, NULL, 0);
        if (ObjectVT == MVT::i64)
          ArgValue= DAG.getNode(ISD::BUILD_PAIR, MVT::i64, ArgValue, ArgValue2);
        else
          ArgValue= DAG.getNode(ARMISD::FMDRR, MVT::f64, ArgValue, ArgValue2);
      }
    } else {
      // Don't emit a dead load.
      ArgValue = DAG.getNode(ISD::UNDEF, ObjectVT);
    }

    ArgOffset += ObjSize;   // Move on to the next argument.
  }

  return ArgValue;
}

SDOperand
ARMTargetLowering::LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG) {
  std::vector<SDOperand> ArgValues;
  SDOperand Root = Op.getOperand(0);
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  unsigned NumGPRs = 0;     // GPRs used for parameter passing.
  unsigned VRegs[4];

  unsigned NumArgs = Op.Val->getNumValues()-1;
  for (unsigned ArgNo = 0; ArgNo < NumArgs; ++ArgNo)
    ArgValues.push_back(LowerFORMAL_ARGUMENT(Op, DAG, VRegs, ArgNo,
                                             NumGPRs, ArgOffset));

  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (isVarArg) {
    static const unsigned GPRArgRegs[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3
    };

    MachineFunction &MF = DAG.getMachineFunction();
    SSARegMap *RegMap = MF.getSSARegMap();
    MachineFrameInfo *MFI = MF.getFrameInfo();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
    unsigned VARegSize = (4 - NumGPRs) * 4;
    unsigned VARegSaveSize = (VARegSize + Align - 1) & ~(Align - 1);
    if (VARegSaveSize) {
      // If this function is vararg, store any remaining integer argument regs
      // to their spots on the stack so that they may be loaded by deferencing
      // the result of va_next.
      AFI->setVarArgsRegSaveSize(VARegSaveSize);
      VarArgsFrameIndex = MFI->CreateFixedObject(VARegSaveSize, ArgOffset +
                                                 VARegSaveSize - VARegSize);
      SDOperand FIN = DAG.getFrameIndex(VarArgsFrameIndex, getPointerTy());

      SmallVector<SDOperand, 4> MemOps;
      for (; NumGPRs < 4; ++NumGPRs) {
        unsigned VReg = RegMap->createVirtualRegister(&ARM::GPRRegClass);
        MF.addLiveIn(GPRArgRegs[NumGPRs], VReg);
        SDOperand Val = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        SDOperand Store = DAG.getStore(Val.getValue(1), Val, FIN, NULL, 0);
        MemOps.push_back(Store);
        FIN = DAG.getNode(ISD::ADD, getPointerTy(), FIN,
                          DAG.getConstant(4, getPointerTy()));
      }
      if (!MemOps.empty())
        Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                           &MemOps[0], MemOps.size());
    } else
      // This will point to the next argument passed via stack.
      VarArgsFrameIndex = MFI->CreateFixedObject(4, ArgOffset);
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVT(Op.Val->value_begin(),
                                    Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, &ArgValues[0], ArgValues.size());
}

/// isFloatingPointZero - Return true if this is +0.0.
static bool isFloatingPointZero(SDOperand Op) {
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op))
    return CFP->isExactlyValue(0.0);
  else if (ISD::isEXTLoad(Op.Val) || ISD::isNON_EXTLoad(Op.Val)) {
    // Maybe this has already been legalized into the constant pool?
    if (Op.getOperand(1).getOpcode() == ARMISD::Wrapper) {
      SDOperand WrapperOp = Op.getOperand(1).getOperand(0);
      if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(WrapperOp))
        if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->getConstVal()))
          return CFP->isExactlyValue(0.0);
    }
  }
  return false;
}

static bool isLegalCmpImmediate(unsigned C, bool isThumb) {
  return ( isThumb && (C & ~255U) == 0) ||
         (!isThumb && ARM_AM::getSOImmVal(C) != -1);
}

/// Returns appropriate ARM CMP (cmp) and corresponding condition code for
/// the given operands.
static SDOperand getARMCmp(SDOperand LHS, SDOperand RHS, ISD::CondCode CC,
                           SDOperand &ARMCC, SelectionDAG &DAG, bool isThumb) {
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS.Val)) {
    unsigned C = RHSC->getValue();
    if (!isLegalCmpImmediate(C, isThumb)) {
      // Constant does not fit, try adjusting it by one?
      switch (CC) {
      default: break;
      case ISD::SETLT:
      case ISD::SETGE:
        if (isLegalCmpImmediate(C-1, isThumb)) {
          CC = (CC == ISD::SETLT) ? ISD::SETLE : ISD::SETGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETULT:
      case ISD::SETUGE:
        if (C > 0 && isLegalCmpImmediate(C-1, isThumb)) {
          CC = (CC == ISD::SETULT) ? ISD::SETULE : ISD::SETUGT;
          RHS = DAG.getConstant(C-1, MVT::i32);
        }
        break;
      case ISD::SETLE:
      case ISD::SETGT:
        if (isLegalCmpImmediate(C+1, isThumb)) {
          CC = (CC == ISD::SETLE) ? ISD::SETLT : ISD::SETGE;
          RHS = DAG.getConstant(C+1, MVT::i32);
        }
        break;
      case ISD::SETULE:
      case ISD::SETUGT:
        if (C < 0xffffffff && isLegalCmpImmediate(C+1, isThumb)) {
          CC = (CC == ISD::SETULE) ? ISD::SETULT : ISD::SETUGE;
          RHS = DAG.getConstant(C+1, MVT::i32);
        }
        break;
      }
    }
  }

  ARMCC::CondCodes CondCode = IntCCToARMCC(CC);
  ARMISD::NodeType CompareType;
  switch (CondCode) {
  default:
    CompareType = ARMISD::CMP;
    break;
  case ARMCC::EQ:
  case ARMCC::NE:
  case ARMCC::MI:
  case ARMCC::PL:
    // Uses only N and Z Flags
    CompareType = ARMISD::CMPNZ;
    break;
  }
  ARMCC = DAG.getConstant(CondCode, MVT::i32);
  return DAG.getNode(CompareType, MVT::Flag, LHS, RHS);
}

/// Returns a appropriate VFP CMP (fcmp{s|d}+fmstat) for the given operands.
static SDOperand getVFPCmp(SDOperand LHS, SDOperand RHS, SelectionDAG &DAG) {
  SDOperand Cmp;
  if (!isFloatingPointZero(RHS))
    Cmp = DAG.getNode(ARMISD::CMPFP, MVT::Flag, LHS, RHS);
  else
    Cmp = DAG.getNode(ARMISD::CMPFPw0, MVT::Flag, LHS);
  return DAG.getNode(ARMISD::FMSTAT, MVT::Flag, Cmp);
}

static SDOperand LowerSELECT_CC(SDOperand Op, SelectionDAG &DAG,
                                const ARMSubtarget *ST) {
  MVT::ValueType VT = Op.getValueType();
  SDOperand LHS = Op.getOperand(0);
  SDOperand RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDOperand TrueVal = Op.getOperand(2);
  SDOperand FalseVal = Op.getOperand(3);

  if (LHS.getValueType() == MVT::i32) {
    SDOperand ARMCC;
    SDOperand Cmp = getARMCmp(LHS, RHS, CC, ARMCC, DAG, ST->isThumb());
    return DAG.getNode(ARMISD::CMOV, VT, FalseVal, TrueVal, ARMCC, Cmp);
  }

  ARMCC::CondCodes CondCode, CondCode2;
  if (FPCCToARMCC(CC, CondCode, CondCode2))
    std::swap(TrueVal, FalseVal);

  SDOperand ARMCC = DAG.getConstant(CondCode, MVT::i32);
  SDOperand Cmp = getVFPCmp(LHS, RHS, DAG);
  SDOperand Result = DAG.getNode(ARMISD::CMOV, VT, FalseVal, TrueVal,
                                 ARMCC, Cmp);
  if (CondCode2 != ARMCC::AL) {
    SDOperand ARMCC2 = DAG.getConstant(CondCode2, MVT::i32);
    // FIXME: Needs another CMP because flag can have but one use.
    SDOperand Cmp2 = getVFPCmp(LHS, RHS, DAG);
    Result = DAG.getNode(ARMISD::CMOV, VT, Result, TrueVal, ARMCC2, Cmp2);
  }
  return Result;
}

static SDOperand LowerBR_CC(SDOperand Op, SelectionDAG &DAG,
                            const ARMSubtarget *ST) {
  SDOperand  Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDOperand    LHS = Op.getOperand(2);
  SDOperand    RHS = Op.getOperand(3);
  SDOperand   Dest = Op.getOperand(4);

  if (LHS.getValueType() == MVT::i32) {
    SDOperand ARMCC;
    SDOperand Cmp = getARMCmp(LHS, RHS, CC, ARMCC, DAG, ST->isThumb());
    return DAG.getNode(ARMISD::BRCOND, MVT::Other, Chain, Dest, ARMCC, Cmp);
  }

  assert(LHS.getValueType() == MVT::f32 || LHS.getValueType() == MVT::f64);
  ARMCC::CondCodes CondCode, CondCode2;
  if (FPCCToARMCC(CC, CondCode, CondCode2))
    // Swap the LHS/RHS of the comparison if needed.
    std::swap(LHS, RHS);
  
  SDOperand Cmp = getVFPCmp(LHS, RHS, DAG);
  SDOperand ARMCC = DAG.getConstant(CondCode, MVT::i32);
  SDVTList VTList = DAG.getVTList(MVT::Other, MVT::Flag);
  SDOperand Ops[] = { Chain, Dest, ARMCC, Cmp };
  SDOperand Res = DAG.getNode(ARMISD::BRCOND, VTList, Ops, 4);
  if (CondCode2 != ARMCC::AL) {
    ARMCC = DAG.getConstant(CondCode2, MVT::i32);
    SDOperand Ops[] = { Res, Dest, ARMCC, Res.getValue(1) };
    Res = DAG.getNode(ARMISD::BRCOND, VTList, Ops, 4);
  }
  return Res;
}

SDOperand ARMTargetLowering::LowerBR_JT(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain = Op.getOperand(0);
  SDOperand Table = Op.getOperand(1);
  SDOperand Index = Op.getOperand(2);

  MVT::ValueType PTy = getPointerTy();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Table);
  ARMFunctionInfo *AFI = DAG.getMachineFunction().getInfo<ARMFunctionInfo>();
  SDOperand UId =  DAG.getConstant(AFI->createJumpTableUId(), PTy);
  SDOperand JTI = DAG.getTargetJumpTable(JT->getIndex(), PTy);
  Table = DAG.getNode(ARMISD::WrapperJT, MVT::i32, JTI, UId);
  Index = DAG.getNode(ISD::MUL, PTy, Index, DAG.getConstant(4, PTy));
  SDOperand Addr = DAG.getNode(ISD::ADD, PTy, Index, Table);
  bool isPIC = getTargetMachine().getRelocationModel() == Reloc::PIC_;
  Addr = DAG.getLoad(isPIC ? MVT::i32 : PTy, Chain, Addr, NULL, 0);
  Chain = Addr.getValue(1);
  if (isPIC)
    Addr = DAG.getNode(ISD::ADD, PTy, Addr, Table);
  return DAG.getNode(ARMISD::BR_JT, MVT::Other, Chain, Addr, JTI, UId);
}

static SDOperand LowerFP_TO_INT(SDOperand Op, SelectionDAG &DAG) {
  unsigned Opc =
    Op.getOpcode() == ISD::FP_TO_SINT ? ARMISD::FTOSI : ARMISD::FTOUI;
  Op = DAG.getNode(Opc, MVT::f32, Op.getOperand(0));
  return DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Op);
}

static SDOperand LowerINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType VT = Op.getValueType();
  unsigned Opc =
    Op.getOpcode() == ISD::SINT_TO_FP ? ARMISD::SITOF : ARMISD::UITOF;

  Op = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Op.getOperand(0));
  return DAG.getNode(Opc, VT, Op);
}

static SDOperand LowerFCOPYSIGN(SDOperand Op, SelectionDAG &DAG) {
  // Implement fcopysign with a fabs and a conditional fneg.
  SDOperand Tmp0 = Op.getOperand(0);
  SDOperand Tmp1 = Op.getOperand(1);
  MVT::ValueType VT = Op.getValueType();
  MVT::ValueType SrcVT = Tmp1.getValueType();
  SDOperand AbsVal = DAG.getNode(ISD::FABS, VT, Tmp0);
  SDOperand Cmp = getVFPCmp(Tmp1, DAG.getConstantFP(0.0, SrcVT), DAG);
  SDOperand ARMCC = DAG.getConstant(ARMCC::LT, MVT::i32);
  return DAG.getNode(ARMISD::CNEG, VT, AbsVal, AbsVal, ARMCC, Cmp);
}

static SDOperand LowerBIT_CONVERT(SDOperand Op, SelectionDAG &DAG) {
  // Turn f64->i64 into FMRRD.
  assert(Op.getValueType() == MVT::i64 &&
         Op.getOperand(0).getValueType() == MVT::f64);

  Op = Op.getOperand(0);
  SDOperand Cvt = DAG.getNode(ARMISD::FMRRD, DAG.getVTList(MVT::i32, MVT::i32),
                              &Op, 1);
  
  // Merge the pieces into a single i64 value.
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Cvt, Cvt.getValue(1));
}

static SDOperand LowerMUL(SDOperand Op, SelectionDAG &DAG) {
  // FIXME: All this code is target-independent.  Create a new target-indep
  // MULHILO node and move this code to the legalizer.
  //
  assert(Op.getValueType() == MVT::i64 && "Only handles i64 expand right now!");
  
  SDOperand LL = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(0, MVT::i32));
  SDOperand RL = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(1),
                             DAG.getConstant(0, MVT::i32));

  const TargetLowering &TL = DAG.getTargetLoweringInfo();
  unsigned LHSSB = TL.ComputeNumSignBits(Op.getOperand(0));
  unsigned RHSSB = TL.ComputeNumSignBits(Op.getOperand(1));
  
  SDOperand Lo, Hi;
  // Figure out how to lower this multiply.
  if (LHSSB >= 33 && RHSSB >= 33) {
    // If the input values are both sign extended, we can emit a mulhs+mul.
    Lo = DAG.getNode(ISD::MUL, MVT::i32, LL, RL);
    Hi = DAG.getNode(ISD::MULHS, MVT::i32, LL, RL);
  } else if (LHSSB == 32 && RHSSB == 32 &&
             TL.MaskedValueIsZero(Op.getOperand(0), 0xFFFFFFFF00000000ULL) &&
             TL.MaskedValueIsZero(Op.getOperand(1), 0xFFFFFFFF00000000ULL)) {
    // If the inputs are zero extended, use mulhu.
    Lo = DAG.getNode(ISD::MUL, MVT::i32, LL, RL);
    Hi = DAG.getNode(ISD::MULHU, MVT::i32, LL, RL);
  } else {
    SDOperand LH = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                               DAG.getConstant(1, MVT::i32));
    SDOperand RH = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(1),
                               DAG.getConstant(1, MVT::i32));
  
    // Lo,Hi = umul LHS, RHS.
    SDOperand Ops[] = { LL, RL };
    SDOperand UMul64 = DAG.getNode(ARMISD::MULHILOU,
                                   DAG.getVTList(MVT::i32, MVT::i32), Ops, 2);
    Lo = UMul64;
    Hi = UMul64.getValue(1);
    RH = DAG.getNode(ISD::MUL, MVT::i32, LL, RH);
    LH = DAG.getNode(ISD::MUL, MVT::i32, LH, RL);
    Hi = DAG.getNode(ISD::ADD, MVT::i32, Hi, RH);
    Hi = DAG.getNode(ISD::ADD, MVT::i32, Hi, LH);
  }
  
  // Merge the pieces into a single i64 value.
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
}

static SDOperand LowerMULHU(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Ops[] = { Op.getOperand(0), Op.getOperand(1) };
  return DAG.getNode(ARMISD::MULHILOU,
                     DAG.getVTList(MVT::i32, MVT::i32), Ops, 2).getValue(1);
}

static SDOperand LowerMULHS(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Ops[] = { Op.getOperand(0), Op.getOperand(1) };
  return DAG.getNode(ARMISD::MULHILOS,
                     DAG.getVTList(MVT::i32, MVT::i32), Ops, 2).getValue(1);
}

static SDOperand LowerSRx(SDOperand Op, SelectionDAG &DAG,
                          const ARMSubtarget *ST) {
  assert(Op.getValueType() == MVT::i64 &&
         (Op.getOpcode() == ISD::SRL || Op.getOpcode() == ISD::SRA) &&
         "Unknown shift to lower!");
  
  // We only lower SRA, SRL of 1 here, all others use generic lowering.
  if (!isa<ConstantSDNode>(Op.getOperand(1)) ||
      cast<ConstantSDNode>(Op.getOperand(1))->getValue() != 1)
    return SDOperand();
  
  // If we are in thumb mode, we don't have RRX.
  if (ST->isThumb()) return SDOperand();
  
  // Okay, we have a 64-bit SRA or SRL of 1.  Lower this to an RRX expr.
  SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(0, MVT::i32));
  SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op.getOperand(0),
                             DAG.getConstant(1, MVT::i32));

  // First, build a SRA_FLAG/SRL_FLAG op, which shifts the top part by one and
  // captures the result into a carry flag.
  unsigned Opc = Op.getOpcode() == ISD::SRL ? ARMISD::SRL_FLAG:ARMISD::SRA_FLAG;
  Hi = DAG.getNode(Opc, DAG.getVTList(MVT::i32, MVT::Flag), &Hi, 1);
  
  // The low part is an ARMISD::RRX operand, which shifts the carry in.
  Lo = DAG.getNode(ARMISD::RRX, MVT::i32, Lo, Hi.getValue(1));
  
  // Merge the pieces into a single i64 value.
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
}

SDOperand ARMTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Don't know how to custom lower this!"); abort();
  case ISD::ConstantPool:  return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:
    return Subtarget->isTargetDarwin() ? LowerGlobalAddressDarwin(Op, DAG) :
      LowerGlobalAddressELF(Op, DAG);
  case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
  case ISD::CALL:          return LowerCALL(Op, DAG);
  case ISD::RET:           return LowerRET(Op, DAG);
  case ISD::SELECT_CC:     return LowerSELECT_CC(Op, DAG, Subtarget);
  case ISD::BR_CC:         return LowerBR_CC(Op, DAG, Subtarget);
  case ISD::BR_JT:         return LowerBR_JT(Op, DAG);
  case ISD::VASTART:       return LowerVASTART(Op, DAG, VarArgsFrameIndex);
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:    return LowerINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:    return LowerFP_TO_INT(Op, DAG);
  case ISD::FCOPYSIGN:     return LowerFCOPYSIGN(Op, DAG);
  case ISD::BIT_CONVERT:   return LowerBIT_CONVERT(Op, DAG);
  case ISD::MUL:           return LowerMUL(Op, DAG);
  case ISD::MULHU:         return LowerMULHU(Op, DAG);
  case ISD::MULHS:         return LowerMULHS(Op, DAG);
  case ISD::SRL:
  case ISD::SRA:           return LowerSRx(Op, DAG, Subtarget);
  case ISD::FORMAL_ARGUMENTS:
    return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::RETURNADDR:    break;
  case ISD::FRAMEADDR:     break;
  case ISD::GLOBAL_OFFSET_TABLE: return LowerGLOBAL_OFFSET_TABLE(Op, DAG);
  }
  return SDOperand();
}

//===----------------------------------------------------------------------===//
//                           ARM Scheduler Hooks
//===----------------------------------------------------------------------===//

MachineBasicBlock *
ARMTargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                           MachineBasicBlock *BB) {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  switch (MI->getOpcode()) {
  default: assert(false && "Unexpected instr type to insert");
  case ARM::tMOVCCr: {
    // To "insert" a SELECT_CC instruction, we actually have to insert the
    // diamond control-flow pattern.  The incoming instruction knows the
    // destination vreg to set, the condition code register to branch on, the
    // true/false values to select between, and a branch opcode to use.
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    ilist<MachineBasicBlock>::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   cmpTY ccX, r1, r2
    //   bCC copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *thisMBB  = BB;
    MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB  = new MachineBasicBlock(LLVM_BB);
    BuildMI(BB, TII->get(ARM::tBcc)).addMBB(sinkMBB)
      .addImm(MI->getOperand(3).getImm());
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
    BuildMI(BB, TII->get(ARM::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

    delete MI;   // The pseudo instruction is gone now.
    return BB;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           ARM Optimization Hooks
//===----------------------------------------------------------------------===//

/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode for load / store of the
/// given type.
static bool isLegalAddressImmediate(int64_t V, MVT::ValueType VT,
                                    const ARMSubtarget *Subtarget) {
  if (V == 0)
    return true;

  if (Subtarget->isThumb()) {
    if (V < 0)
      return false;

    unsigned Scale = 1;
    switch (VT) {
    default: return false;
    case MVT::i1:
    case MVT::i8:
      // Scale == 1;
      break;
    case MVT::i16:
      // Scale == 2;
      Scale = 2;
      break;
    case MVT::i32:
      // Scale == 4;
      Scale = 4;
      break;
    }

    if ((V & (Scale - 1)) != 0)
      return false;
    V /= Scale;
    return V == V & ((1LL << 5) - 1);
  }

  if (V < 0)
    V = - V;
  switch (VT) {
  default: return false;
  case MVT::i1:
  case MVT::i8:
  case MVT::i32:
    // +- imm12
    return V == V & ((1LL << 12) - 1);
  case MVT::i16:
    // +- imm8
    return V == V & ((1LL << 8) - 1);
  case MVT::f32:
  case MVT::f64:
    if (!Subtarget->hasVFP2())
      return false;
    if ((V % 3) != 0)
      return false;
    V >>= 2;
    return V == V & ((1LL << 8) - 1);
  }
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool ARMTargetLowering::isLegalAddressingMode(const AddrMode &AM, 
                                              const Type *Ty) const {
  if (!isLegalAddressImmediate(AM.BaseOffs, getValueType(Ty), Subtarget))
    return false;
  
  // Can never fold addr of global into load/store.
  if (AM.BaseGV) 
    return false;
  
  switch (AM.Scale) {
  case 0:  // no scale reg, must be "r+i" or "r", or "i".
    break;
  case 1:
    if (Subtarget->isThumb())
      return false;
    // FALL THROUGH.
  default:
    // ARM doesn't support any R+R*scale+imm addr modes.
    if (AM.BaseOffs)
      return false;
    
    int Scale = AM.Scale;
    switch (getValueType(Ty)) {
    default: return false;
    case MVT::i1:
    case MVT::i8:
    case MVT::i32:
    case MVT::i64:
      // This assumes i64 is legalized to a pair of i32. If not (i.e.
      // ldrd / strd are used, then its address mode is same as i16.
      // r + r
      if (Scale < 0) Scale = -Scale;
      if (Scale == 1)
        return true;
      // r + r << imm
      return isPowerOf2_32(Scale & ~1);
    case MVT::i16:
      // r + r
      if (((unsigned)AM.HasBaseReg + Scale) <= 2)
        return true;
      return false;
      
    case MVT::isVoid:
      // Note, we allow "void" uses (basically, uses that aren't loads or
      // stores), because arm allows folding a scale into many arithmetic
      // operations.  This should be made more precise and revisited later.
      
      // Allow r << imm, but the imm has to be a multiple of two.
      if (AM.Scale & 1) return false;
      return isPowerOf2_32(AM.Scale);
    }
    break;
  }
  return true;
}


static bool getIndexedAddressParts(SDNode *Ptr, MVT::ValueType VT,
                                   bool isSEXTLoad, SDOperand &Base,
                                   SDOperand &Offset, bool &isInc,
                                   SelectionDAG &DAG) {
  if (Ptr->getOpcode() != ISD::ADD && Ptr->getOpcode() != ISD::SUB)
    return false;

  if (VT == MVT::i16 || ((VT == MVT::i8 || VT == MVT::i1) && isSEXTLoad)) {
    // AddressingMode 3
    Base = Ptr->getOperand(0);
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Ptr->getOperand(1))) {
      int RHSC = (int)RHS->getValue();
      if (RHSC < 0 && RHSC > -256) {
        isInc = false;
        Offset = DAG.getConstant(-RHSC, RHS->getValueType(0));
        return true;
      }
    }
    isInc = (Ptr->getOpcode() == ISD::ADD);
    Offset = Ptr->getOperand(1);
    return true;
  } else if (VT == MVT::i32 || VT == MVT::i8 || VT == MVT::i1) {
    // AddressingMode 2
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(Ptr->getOperand(1))) {
      int RHSC = (int)RHS->getValue();
      if (RHSC < 0 && RHSC > -0x1000) {
        isInc = false;
        Offset = DAG.getConstant(-RHSC, RHS->getValueType(0));
        Base = Ptr->getOperand(0);
        return true;
      }
    }

    if (Ptr->getOpcode() == ISD::ADD) {
      isInc = true;
      ARM_AM::ShiftOpc ShOpcVal= ARM_AM::getShiftOpcForNode(Ptr->getOperand(0));
      if (ShOpcVal != ARM_AM::no_shift) {
        Base = Ptr->getOperand(1);
        Offset = Ptr->getOperand(0);
      } else {
        Base = Ptr->getOperand(0);
        Offset = Ptr->getOperand(1);
      }
      return true;
    }

    isInc = (Ptr->getOpcode() == ISD::ADD);
    Base = Ptr->getOperand(0);
    Offset = Ptr->getOperand(1);
    return true;
  }

  // FIXME: Use FLDM / FSTM to emulate indexed FP load / store.
  return false;
}

/// getPreIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if the node's address
/// can be legally represented as pre-indexed load / store address.
bool
ARMTargetLowering::getPreIndexedAddressParts(SDNode *N, SDOperand &Base,
                                             SDOperand &Offset,
                                             ISD::MemIndexedMode &AM,
                                             SelectionDAG &DAG) {
  if (Subtarget->isThumb())
    return false;

  MVT::ValueType VT;
  SDOperand Ptr;
  bool isSEXTLoad = false;
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    Ptr = LD->getBasePtr();
    VT  = LD->getLoadedVT();
    isSEXTLoad = LD->getExtensionType() == ISD::SEXTLOAD;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    Ptr = ST->getBasePtr();
    VT  = ST->getStoredVT();
  } else
    return false;

  bool isInc;
  bool isLegal = getIndexedAddressParts(Ptr.Val, VT, isSEXTLoad, Base, Offset,
                                        isInc, DAG);
  if (isLegal) {
    AM = isInc ? ISD::PRE_INC : ISD::PRE_DEC;
    return true;
  }
  return false;
}

/// getPostIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if this node can be
/// combined with a load / store to form a post-indexed load / store.
bool ARMTargetLowering::getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                                   SDOperand &Base,
                                                   SDOperand &Offset,
                                                   ISD::MemIndexedMode &AM,
                                                   SelectionDAG &DAG) {
  if (Subtarget->isThumb())
    return false;

  MVT::ValueType VT;
  SDOperand Ptr;
  bool isSEXTLoad = false;
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    VT  = LD->getLoadedVT();
    isSEXTLoad = LD->getExtensionType() == ISD::SEXTLOAD;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT  = ST->getStoredVT();
  } else
    return false;

  bool isInc;
  bool isLegal = getIndexedAddressParts(Op, VT, isSEXTLoad, Base, Offset,
                                        isInc, DAG);
  if (isLegal) {
    AM = isInc ? ISD::POST_INC : ISD::POST_DEC;
    return true;
  }
  return false;
}

void ARMTargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op,
                                                       uint64_t Mask,
                                                       uint64_t &KnownZero, 
                                                       uint64_t &KnownOne,
                                                       unsigned Depth) const {
  KnownZero = 0;
  KnownOne = 0;
  switch (Op.getOpcode()) {
  default: break;
  case ARMISD::CMOV: {
    // Bits are known zero/one if known on the LHS and RHS.
    ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero, KnownOne, Depth+1);
    if (KnownZero == 0 && KnownOne == 0) return;

    uint64_t KnownZeroRHS, KnownOneRHS;
    ComputeMaskedBits(Op.getOperand(1), Mask,
                      KnownZeroRHS, KnownOneRHS, Depth+1);
    KnownZero &= KnownZeroRHS;
    KnownOne  &= KnownOneRHS;
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
//                           ARM Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
ARMTargetLowering::ConstraintType
ARMTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:  break;
    case 'l': return C_RegisterClass;
    case 'w': return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass*> 
ARMTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    // GCC RS6000 Constraint Letters
    switch (Constraint[0]) {
    case 'l':
    // FIXME: in thumb mode, 'l' is only low-regs.
    // FALL THROUGH.
    case 'r':
      return std::make_pair(0U, ARM::GPRRegisterClass);
    case 'w':
      if (VT == MVT::f32)
        return std::make_pair(0U, ARM::SPRRegisterClass);
      if (VT == MVT::f64)
        return std::make_pair(0U, ARM::DPRRegisterClass);
      break;
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

std::vector<unsigned> ARMTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  switch (Constraint[0]) {      // GCC ARM Constraint Letters
  default: break;
  case 'l':
  case 'r':
    return make_vector<unsigned>(ARM::R0, ARM::R1, ARM::R2, ARM::R3,
                                 ARM::R4, ARM::R5, ARM::R6, ARM::R7,
                                 ARM::R8, ARM::R9, ARM::R10, ARM::R11,
                                 ARM::R12, ARM::LR, 0);
  case 'w':
    if (VT == MVT::f32)
      return make_vector<unsigned>(ARM::S0, ARM::S1, ARM::S2, ARM::S3,
                                   ARM::S4, ARM::S5, ARM::S6, ARM::S7,
                                   ARM::S8, ARM::S9, ARM::S10, ARM::S11,
                                   ARM::S12,ARM::S13,ARM::S14,ARM::S15,
                                   ARM::S16,ARM::S17,ARM::S18,ARM::S19,
                                   ARM::S20,ARM::S21,ARM::S22,ARM::S23,
                                   ARM::S24,ARM::S25,ARM::S26,ARM::S27,
                                   ARM::S28,ARM::S29,ARM::S30,ARM::S31, 0);
    if (VT == MVT::f64)
      return make_vector<unsigned>(ARM::D0, ARM::D1, ARM::D2, ARM::D3,
                                   ARM::D4, ARM::D5, ARM::D6, ARM::D7,
                                   ARM::D8, ARM::D9, ARM::D10,ARM::D11,
                                   ARM::D12,ARM::D13,ARM::D14,ARM::D15, 0);
      break;
  }

  return std::vector<unsigned>();
}

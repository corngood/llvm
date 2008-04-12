//===-- AlphaISelLowering.cpp - Alpha DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AlphaISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "AlphaISelLowering.h"
#include "AlphaTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

/// AddLiveIn - This helper function adds the specified physical register to the
/// MachineFunction as a live in value.  It also creates a corresponding virtual
/// register for it.
static unsigned AddLiveIn(MachineFunction &MF, unsigned PReg,
                          TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
  MF.getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

AlphaTargetLowering::AlphaTargetLowering(TargetMachine &TM) : TargetLowering(TM) {
  // Set up the TargetLowering object.
  //I am having problems with shr n ubyte 1
  setShiftAmountType(MVT::i64);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  
  setUsesGlobalOffsetTable(true);
  
  addRegisterClass(MVT::i64, Alpha::GPRCRegisterClass);
  addRegisterClass(MVT::f64, Alpha::F8RCRegisterClass);
  addRegisterClass(MVT::f32, Alpha::F4RCRegisterClass);
  
  setLoadXAction(ISD::EXTLOAD, MVT::i1,  Promote);
  setLoadXAction(ISD::EXTLOAD, MVT::f32, Expand);
  
  setLoadXAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setLoadXAction(ISD::ZEXTLOAD, MVT::i32, Expand);
  
  setLoadXAction(ISD::SEXTLOAD, MVT::i1,  Promote);
  setLoadXAction(ISD::SEXTLOAD, MVT::i8,  Expand);
  setLoadXAction(ISD::SEXTLOAD, MVT::i16, Expand);

  //  setOperationAction(ISD::BRIND,        MVT::Other,   Expand);
  setOperationAction(ISD::BR_JT,        MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,        MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,    MVT::Other, Expand);  

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  setOperationAction(ISD::FREM, MVT::f32, Expand);
  setOperationAction(ISD::FREM, MVT::f64, Expand);
  
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);

  if (!TM.getSubtarget<AlphaSubtarget>().hasCT()) {
    setOperationAction(ISD::CTPOP    , MVT::i64  , Expand);
    setOperationAction(ISD::CTTZ     , MVT::i64  , Expand);
    setOperationAction(ISD::CTLZ     , MVT::i64  , Expand);
  }
  setOperationAction(ISD::BSWAP    , MVT::i64, Expand);
  setOperationAction(ISD::ROTL     , MVT::i64, Expand);
  setOperationAction(ISD::ROTR     , MVT::i64, Expand);
  
  setOperationAction(ISD::SREM     , MVT::i64, Custom);
  setOperationAction(ISD::UREM     , MVT::i64, Custom);
  setOperationAction(ISD::SDIV     , MVT::i64, Custom);
  setOperationAction(ISD::UDIV     , MVT::i64, Custom);

  // We don't support sin/cos/sqrt/pow
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);

  setOperationAction(ISD::FSQRT, MVT::f64, Expand);
  setOperationAction(ISD::FSQRT, MVT::f32, Expand);

  setOperationAction(ISD::FPOW , MVT::f32, Expand);
  setOperationAction(ISD::FPOW , MVT::f64, Expand);
  
  setOperationAction(ISD::SETCC, MVT::f32, Promote);

  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Promote);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::LABEL, MVT::Other, Expand);

  // Not implemented yet.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);

  // We want to legalize GlobalAddress and ConstantPool and
  // ExternalSymbols nodes into the appropriate instructions to
  // materialize the address.
  setOperationAction(ISD::GlobalAddress,  MVT::i64, Custom);
  setOperationAction(ISD::ConstantPool,   MVT::i64, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i64, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i64, Custom);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAEND,   MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,  MVT::Other, Custom);
  setOperationAction(ISD::VAARG,   MVT::Other, Custom);
  setOperationAction(ISD::VAARG,   MVT::i32,   Custom);

  setOperationAction(ISD::RET,     MVT::Other, Custom);

  setOperationAction(ISD::JumpTable, MVT::i64, Custom);
  setOperationAction(ISD::JumpTable, MVT::i32, Custom);

  setStackPointerRegisterToSaveRestore(Alpha::R30);

  addLegalFPImmediate(APFloat(+0.0)); //F31
  addLegalFPImmediate(APFloat(+0.0f)); //F31
  addLegalFPImmediate(APFloat(-0.0)); //-F31
  addLegalFPImmediate(APFloat(-0.0f)); //-F31

  setJumpBufSize(272);
  setJumpBufAlignment(16);

  computeRegisterProperties();
}

MVT::ValueType
AlphaTargetLowering::getSetCCResultType(const SDOperand &) const {
  return MVT::i64;
}

const char *AlphaTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case AlphaISD::CVTQT_: return "Alpha::CVTQT_";
  case AlphaISD::CVTQS_: return "Alpha::CVTQS_";
  case AlphaISD::CVTTQ_: return "Alpha::CVTTQ_";
  case AlphaISD::GPRelHi: return "Alpha::GPRelHi";
  case AlphaISD::GPRelLo: return "Alpha::GPRelLo";
  case AlphaISD::RelLit: return "Alpha::RelLit";
  case AlphaISD::GlobalRetAddr: return "Alpha::GlobalRetAddr";
  case AlphaISD::CALL:   return "Alpha::CALL";
  case AlphaISD::DivCall: return "Alpha::DivCall";
  case AlphaISD::RET_FLAG: return "Alpha::RET_FLAG";
  case AlphaISD::COND_BRANCH_I: return "Alpha::COND_BRANCH_I";
  case AlphaISD::COND_BRANCH_F: return "Alpha::COND_BRANCH_F";
  }
}

static SDOperand LowerJumpTable(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType PtrVT = Op.getValueType();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDOperand JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);
  SDOperand Zero = DAG.getConstant(0, PtrVT);
  
  SDOperand Hi = DAG.getNode(AlphaISD::GPRelHi,  MVT::i64, JTI,
                             DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, MVT::i64));
  SDOperand Lo = DAG.getNode(AlphaISD::GPRelLo, MVT::i64, JTI, Hi);
  return Lo;
}

//http://www.cs.arizona.edu/computer.help/policy/DIGITAL_unix/
//AA-PY8AC-TET1_html/callCH3.html#BLOCK21

//For now, just use variable size stack frame format

//In a standard call, the first six items are passed in registers $16
//- $21 and/or registers $f16 - $f21. (See Section 4.1.2 for details
//of argument-to-register correspondence.) The remaining items are
//collected in a memory argument list that is a naturally aligned
//array of quadwords. In a standard call, this list, if present, must
//be passed at 0(SP).
//7 ... n         0(SP) ... (n-7)*8(SP)

// //#define FP    $15
// //#define RA    $26
// //#define PV    $27
// //#define GP    $29
// //#define SP    $30

static SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG,
                                       int &VarArgsBase,
                                       int &VarArgsOffset) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  std::vector<SDOperand> ArgValues;
  SDOperand Root = Op.getOperand(0);

  AddLiveIn(MF, Alpha::R29, &Alpha::GPRCRegClass); //GP
  AddLiveIn(MF, Alpha::R26, &Alpha::GPRCRegClass); //RA

  unsigned args_int[] = {
    Alpha::R16, Alpha::R17, Alpha::R18, Alpha::R19, Alpha::R20, Alpha::R21};
  unsigned args_float[] = {
    Alpha::F16, Alpha::F17, Alpha::F18, Alpha::F19, Alpha::F20, Alpha::F21};
  
  for (unsigned ArgNo = 0, e = Op.Val->getNumValues()-1; ArgNo != e; ++ArgNo) {
    SDOperand argt;
    MVT::ValueType ObjectVT = Op.getValue(ArgNo).getValueType();
    SDOperand ArgVal;

    if (ArgNo  < 6) {
      switch (ObjectVT) {
      default:
        cerr << "Unknown Type " << ObjectVT << "\n";
        abort();
      case MVT::f64:
        args_float[ArgNo] = AddLiveIn(MF, args_float[ArgNo], 
                                      &Alpha::F8RCRegClass);
        ArgVal = DAG.getCopyFromReg(Root, args_float[ArgNo], ObjectVT);
        break;
      case MVT::f32:
        args_float[ArgNo] = AddLiveIn(MF, args_float[ArgNo], 
                                      &Alpha::F4RCRegClass);
        ArgVal = DAG.getCopyFromReg(Root, args_float[ArgNo], ObjectVT);
        break;
      case MVT::i64:
        args_int[ArgNo] = AddLiveIn(MF, args_int[ArgNo], 
                                    &Alpha::GPRCRegClass);
        ArgVal = DAG.getCopyFromReg(Root, args_int[ArgNo], MVT::i64);
        break;
      }
    } else { //more args
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(8, 8 * (ArgNo - 6));

      // Create the SelectionDAG nodes corresponding to a load
      //from this parameter
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
      ArgVal = DAG.getLoad(ObjectVT, Root, FIN, NULL, 0);
    }
    ArgValues.push_back(ArgVal);
  }

  // If the functions takes variable number of arguments, copy all regs to stack
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (isVarArg) {
    VarArgsOffset = (Op.Val->getNumValues()-1) * 8;
    std::vector<SDOperand> LS;
    for (int i = 0; i < 6; ++i) {
      if (TargetRegisterInfo::isPhysicalRegister(args_int[i]))
        args_int[i] = AddLiveIn(MF, args_int[i], &Alpha::GPRCRegClass);
      SDOperand argt = DAG.getCopyFromReg(Root, args_int[i], MVT::i64);
      int FI = MFI->CreateFixedObject(8, -8 * (6 - i));
      if (i == 0) VarArgsBase = FI;
      SDOperand SDFI = DAG.getFrameIndex(FI, MVT::i64);
      LS.push_back(DAG.getStore(Root, argt, SDFI, NULL, 0));

      if (TargetRegisterInfo::isPhysicalRegister(args_float[i]))
        args_float[i] = AddLiveIn(MF, args_float[i], &Alpha::F8RCRegClass);
      argt = DAG.getCopyFromReg(Root, args_float[i], MVT::f64);
      FI = MFI->CreateFixedObject(8, - 8 * (12 - i));
      SDFI = DAG.getFrameIndex(FI, MVT::i64);
      LS.push_back(DAG.getStore(Root, argt, SDFI, NULL, 0));
    }

    //Set up a token factor with all the stack traffic
    Root = DAG.getNode(ISD::TokenFactor, MVT::Other, &LS[0], LS.size());
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVT(Op.Val->value_begin(),
                                    Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, &ArgValues[0], ArgValues.size());
}

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Copy = DAG.getCopyToReg(Op.getOperand(0), Alpha::R26, 
                                    DAG.getNode(AlphaISD::GlobalRetAddr, 
                                                MVT::i64),
                                    SDOperand());
  switch (Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: 
    break;
    //return SDOperand(); // ret void is legal
  case 3: {
    MVT::ValueType ArgVT = Op.getOperand(1).getValueType();
    unsigned ArgReg;
    if (MVT::isInteger(ArgVT))
      ArgReg = Alpha::R0;
    else {
      assert(MVT::isFloatingPoint(ArgVT));
      ArgReg = Alpha::F0;
    }
    Copy = DAG.getCopyToReg(Copy, ArgReg, Op.getOperand(1), Copy.getValue(1));
    if (DAG.getMachineFunction().getRegInfo().liveout_empty())
      DAG.getMachineFunction().getRegInfo().addLiveOut(ArgReg);
    break;
  }
  }
  return DAG.getNode(AlphaISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy, 
                                 bool RetSExt, bool RetZExt, bool isVarArg,
                                 unsigned CallingConv, bool isTailCall,
                                 SDOperand Callee, ArgListTy &Args,
                                 SelectionDAG &DAG) {
  int NumBytes = 0;
  if (Args.size() > 6)
    NumBytes = (Args.size() - 6) * 8;

  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getConstant(NumBytes, getPointerTy()));
  std::vector<SDOperand> args_to_use;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
  {
    switch (getValueType(Args[i].Ty)) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      // Promote the integer to 64 bits.  If the input type is signed use a
      // sign extend, otherwise use a zero extend.
      if (Args[i].isSExt)
        Args[i].Node = DAG.getNode(ISD::SIGN_EXTEND, MVT::i64, Args[i].Node);
      else if (Args[i].isZExt)
        Args[i].Node = DAG.getNode(ISD::ZERO_EXTEND, MVT::i64, Args[i].Node);
      else
        Args[i].Node = DAG.getNode(ISD::ANY_EXTEND, MVT::i64, Args[i].Node);
      break;
    case MVT::i64:
    case MVT::f64:
    case MVT::f32:
      break;
    }
    args_to_use.push_back(Args[i].Node);
  }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  MVT::ValueType ActualRetTyVT = RetTyVT;
  if (RetTyVT >= MVT::i1 && RetTyVT <= MVT::i32)
    ActualRetTyVT = MVT::i64;

  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(ActualRetTyVT);
  RetVals.push_back(MVT::Other);

  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  Ops.insert(Ops.end(), args_to_use.begin(), args_to_use.end());
  SDOperand TheCall = DAG.getNode(AlphaISD::CALL, RetVals, &Ops[0], Ops.size());
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, getPointerTy()),
                             DAG.getConstant(0, getPointerTy()),
                             SDOperand());
  SDOperand RetVal = TheCall;

  if (RetTyVT != ActualRetTyVT) {
    ISD::NodeType AssertKind = ISD::DELETED_NODE;
    if (RetSExt)
      AssertKind = ISD::AssertSext;
    else if (RetZExt)
      AssertKind = ISD::AssertZext;

    if (AssertKind != ISD::DELETED_NODE)
      RetVal = DAG.getNode(AssertKind, MVT::i64, RetVal,
                           DAG.getValueType(RetTyVT));

    RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
  }

  return std::make_pair(RetVal, Chain);
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand AlphaTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Wasn't expecting to be able to lower this!");
  case ISD::FORMAL_ARGUMENTS: return LowerFORMAL_ARGUMENTS(Op, DAG, 
                                                           VarArgsBase,
                                                           VarArgsOffset);

  case ISD::RET: return LowerRET(Op,DAG);
  case ISD::JumpTable: return LowerJumpTable(Op, DAG);

  case ISD::SINT_TO_FP: {
    assert(MVT::i64 == Op.getOperand(0).getValueType() && 
           "Unhandled SINT_TO_FP type in custom expander!");
    SDOperand LD;
    bool isDouble = MVT::f64 == Op.getValueType();
    LD = DAG.getNode(ISD::BIT_CONVERT, MVT::f64, Op.getOperand(0));
    SDOperand FP = DAG.getNode(isDouble?AlphaISD::CVTQT_:AlphaISD::CVTQS_,
                               isDouble?MVT::f64:MVT::f32, LD);
    return FP;
  }
  case ISD::FP_TO_SINT: {
    bool isDouble = MVT::f64 == Op.getOperand(0).getValueType();
    SDOperand src = Op.getOperand(0);

    if (!isDouble) //Promote
      src = DAG.getNode(ISD::FP_EXTEND, MVT::f64, src);
    
    src = DAG.getNode(AlphaISD::CVTTQ_, MVT::f64, src);

    return DAG.getNode(ISD::BIT_CONVERT, MVT::i64, src);
  }
  case ISD::ConstantPool: {
    ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
    Constant *C = CP->getConstVal();
    SDOperand CPI = DAG.getTargetConstantPool(C, MVT::i64, CP->getAlignment());
    
    SDOperand Hi = DAG.getNode(AlphaISD::GPRelHi,  MVT::i64, CPI,
                               DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, MVT::i64));
    SDOperand Lo = DAG.getNode(AlphaISD::GPRelLo, MVT::i64, CPI, Hi);
    return Lo;
  }
  case ISD::GlobalTLSAddress:
    assert(0 && "TLS not implemented for Alpha.");
  case ISD::GlobalAddress: {
    GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
    GlobalValue *GV = GSDN->getGlobal();
    SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i64, GSDN->getOffset());

    //    if (!GV->hasWeakLinkage() && !GV->isDeclaration() && !GV->hasLinkOnceLinkage()) {
    if (GV->hasInternalLinkage()) {
      SDOperand Hi = DAG.getNode(AlphaISD::GPRelHi,  MVT::i64, GA,
                                DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, MVT::i64));
      SDOperand Lo = DAG.getNode(AlphaISD::GPRelLo, MVT::i64, GA, Hi);
      return Lo;
    } else
      return DAG.getNode(AlphaISD::RelLit, MVT::i64, GA, 
                         DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, MVT::i64));
  }
  case ISD::ExternalSymbol: {
    return DAG.getNode(AlphaISD::RelLit, MVT::i64, 
                       DAG.getTargetExternalSymbol(cast<ExternalSymbolSDNode>(Op)
                                                   ->getSymbol(), MVT::i64),
                       DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, MVT::i64));
  }

  case ISD::UREM:
  case ISD::SREM:
    //Expand only on constant case
    if (Op.getOperand(1).getOpcode() == ISD::Constant) {
      MVT::ValueType VT = Op.Val->getValueType(0);
      SDOperand Tmp1 = Op.Val->getOpcode() == ISD::UREM ?
        BuildUDIV(Op.Val, DAG, NULL) :
        BuildSDIV(Op.Val, DAG, NULL);
      Tmp1 = DAG.getNode(ISD::MUL, VT, Tmp1, Op.getOperand(1));
      Tmp1 = DAG.getNode(ISD::SUB, VT, Op.getOperand(0), Tmp1);
      return Tmp1;
    }
    //fall through
  case ISD::SDIV:
  case ISD::UDIV:
    if (MVT::isInteger(Op.getValueType())) {
      if (Op.getOperand(1).getOpcode() == ISD::Constant)
        return Op.getOpcode() == ISD::SDIV ? BuildSDIV(Op.Val, DAG, NULL) 
          : BuildUDIV(Op.Val, DAG, NULL);
      const char* opstr = 0;
      switch (Op.getOpcode()) {
      case ISD::UREM: opstr = "__remqu"; break;
      case ISD::SREM: opstr = "__remq";  break;
      case ISD::UDIV: opstr = "__divqu"; break;
      case ISD::SDIV: opstr = "__divq";  break;
      }
      SDOperand Tmp1 = Op.getOperand(0),
        Tmp2 = Op.getOperand(1),
        Addr = DAG.getExternalSymbol(opstr, MVT::i64);
      return DAG.getNode(AlphaISD::DivCall, MVT::i64, Addr, Tmp1, Tmp2);
    }
    break;

  case ISD::VAARG: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand VAListP = Op.getOperand(1);
    const Value *VAListS = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
    
    SDOperand Base = DAG.getLoad(MVT::i64, Chain, VAListP, VAListS, 0);
    SDOperand Tmp = DAG.getNode(ISD::ADD, MVT::i64, VAListP,
                                DAG.getConstant(8, MVT::i64));
    SDOperand Offset = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Base.getValue(1),
                                      Tmp, NULL, 0, MVT::i32);
    SDOperand DataPtr = DAG.getNode(ISD::ADD, MVT::i64, Base, Offset);
    if (MVT::isFloatingPoint(Op.getValueType()))
    {
      //if fp && Offset < 6*8, then subtract 6*8 from DataPtr
      SDOperand FPDataPtr = DAG.getNode(ISD::SUB, MVT::i64, DataPtr,
                                        DAG.getConstant(8*6, MVT::i64));
      SDOperand CC = DAG.getSetCC(MVT::i64, Offset,
                                  DAG.getConstant(8*6, MVT::i64), ISD::SETLT);
      DataPtr = DAG.getNode(ISD::SELECT, MVT::i64, CC, FPDataPtr, DataPtr);
    }

    SDOperand NewOffset = DAG.getNode(ISD::ADD, MVT::i64, Offset,
                                      DAG.getConstant(8, MVT::i64));
    SDOperand Update = DAG.getTruncStore(Offset.getValue(1), NewOffset,
                                         Tmp, NULL, 0, MVT::i32);
    
    SDOperand Result;
    if (Op.getValueType() == MVT::i32)
      Result = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Update, DataPtr,
                              NULL, 0, MVT::i32);
    else
      Result = DAG.getLoad(Op.getValueType(), Update, DataPtr, NULL, 0);
    return Result;
  }
  case ISD::VACOPY: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand DestP = Op.getOperand(1);
    SDOperand SrcP = Op.getOperand(2);
    const Value *DestS = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();
    const Value *SrcS = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();
    
    SDOperand Val = DAG.getLoad(getPointerTy(), Chain, SrcP, SrcS, 0);
    SDOperand Result = DAG.getStore(Val.getValue(1), Val, DestP, DestS, 0);
    SDOperand NP = DAG.getNode(ISD::ADD, MVT::i64, SrcP, 
                               DAG.getConstant(8, MVT::i64));
    Val = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Result, NP, NULL,0, MVT::i32);
    SDOperand NPD = DAG.getNode(ISD::ADD, MVT::i64, DestP,
                                DAG.getConstant(8, MVT::i64));
    return DAG.getTruncStore(Val.getValue(1), Val, NPD, NULL, 0, MVT::i32);
  }
  case ISD::VASTART: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand VAListP = Op.getOperand(1);
    const Value *VAListS = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
    
    // vastart stores the address of the VarArgsBase and VarArgsOffset
    SDOperand FR  = DAG.getFrameIndex(VarArgsBase, MVT::i64);
    SDOperand S1  = DAG.getStore(Chain, FR, VAListP, VAListS, 0);
    SDOperand SA2 = DAG.getNode(ISD::ADD, MVT::i64, VAListP,
                                DAG.getConstant(8, MVT::i64));
    return DAG.getTruncStore(S1, DAG.getConstant(VarArgsOffset, MVT::i64),
                             SA2, NULL, 0, MVT::i32);
  }
  case ISD::RETURNADDR:        
    return DAG.getNode(AlphaISD::GlobalRetAddr, MVT::i64);
      //FIXME: implement
  case ISD::FRAMEADDR:          break;
  }
  
  return SDOperand();
}

SDOperand AlphaTargetLowering::CustomPromoteOperation(SDOperand Op, 
                                                      SelectionDAG &DAG) {
  assert(Op.getValueType() == MVT::i32 && 
         Op.getOpcode() == ISD::VAARG &&
         "Unknown node to custom promote!");
  
  // The code in LowerOperation already handles i32 vaarg
  return LowerOperation(Op, DAG);
}


//Inline Asm

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
AlphaTargetLowering::ConstraintType 
AlphaTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'f':
    case 'r':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::vector<unsigned> AlphaTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;  // Unknown constriant letter
    case 'f': 
      return make_vector<unsigned>(Alpha::F0 , Alpha::F1 , Alpha::F2 ,
                                   Alpha::F3 , Alpha::F4 , Alpha::F5 ,
                                   Alpha::F6 , Alpha::F7 , Alpha::F8 , 
                                   Alpha::F9 , Alpha::F10, Alpha::F11, 
                                   Alpha::F12, Alpha::F13, Alpha::F14, 
                                   Alpha::F15, Alpha::F16, Alpha::F17, 
                                   Alpha::F18, Alpha::F19, Alpha::F20, 
                                   Alpha::F21, Alpha::F22, Alpha::F23, 
                                   Alpha::F24, Alpha::F25, Alpha::F26, 
                                   Alpha::F27, Alpha::F28, Alpha::F29, 
                                   Alpha::F30, Alpha::F31, 0);
    case 'r': 
      return make_vector<unsigned>(Alpha::R0 , Alpha::R1 , Alpha::R2 , 
                                   Alpha::R3 , Alpha::R4 , Alpha::R5 , 
                                   Alpha::R6 , Alpha::R7 , Alpha::R8 , 
                                   Alpha::R9 , Alpha::R10, Alpha::R11, 
                                   Alpha::R12, Alpha::R13, Alpha::R14, 
                                   Alpha::R15, Alpha::R16, Alpha::R17, 
                                   Alpha::R18, Alpha::R19, Alpha::R20, 
                                   Alpha::R21, Alpha::R22, Alpha::R23, 
                                   Alpha::R24, Alpha::R25, Alpha::R26, 
                                   Alpha::R27, Alpha::R28, Alpha::R29, 
                                   Alpha::R30, Alpha::R31, 0);
    }
  }
  
  return std::vector<unsigned>();
}
//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

MachineBasicBlock *
AlphaTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *BB) {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  assert((MI->getOpcode() == Alpha::CAS32 ||
          MI->getOpcode() == Alpha::CAS64 ||
          MI->getOpcode() == Alpha::LAS32 ||
          MI->getOpcode() == Alpha::LAS64 ||
          MI->getOpcode() == Alpha::SWAP32 ||
          MI->getOpcode() == Alpha::SWAP64) &&
         "Unexpected instr type to insert");

  bool is32 = MI->getOpcode() == Alpha::CAS32 || 
    MI->getOpcode() == Alpha::LAS32 ||
    MI->getOpcode() == Alpha::SWAP32;
  
  //Load locked store conditional for atomic ops take on the same form
  //start:
  //ll
  //do stuff (maybe branch to exit)
  //sc
  //test sc and maybe branck to start
  //exit:
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  ilist<MachineBasicBlock>::iterator It = BB;
  ++It;
  
  MachineBasicBlock *thisMBB = BB;
  MachineBasicBlock *llscMBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);

  for(MachineBasicBlock::succ_iterator i = thisMBB->succ_begin(), 
        e = thisMBB->succ_end(); i != e; ++i)
    sinkMBB->addSuccessor(*i);
  while(!thisMBB->succ_empty())
    thisMBB->removeSuccessor(thisMBB->succ_begin());

  MachineFunction *F = BB->getParent();
  F->getBasicBlockList().insert(It, llscMBB);
  F->getBasicBlockList().insert(It, sinkMBB);

  BuildMI(thisMBB, TII->get(Alpha::BR)).addMBB(llscMBB);
  
  unsigned reg_res = MI->getOperand(0).getReg(),
    reg_ptr = MI->getOperand(1).getReg(),
    reg_v2 = MI->getOperand(2).getReg(),
    reg_store = F->getRegInfo().createVirtualRegister(&Alpha::GPRCRegClass);

  BuildMI(llscMBB, TII->get(is32 ? Alpha::LDL_L : Alpha::LDQ_L), 
          reg_res).addImm(0).addReg(reg_ptr);
  switch (MI->getOpcode()) {
  case Alpha::CAS32:
  case Alpha::CAS64: {
    unsigned reg_cmp 
      = F->getRegInfo().createVirtualRegister(&Alpha::GPRCRegClass);
    BuildMI(llscMBB, TII->get(Alpha::CMPEQ), reg_cmp)
      .addReg(reg_v2).addReg(reg_res);
    BuildMI(llscMBB, TII->get(Alpha::BEQ))
      .addImm(0).addReg(reg_cmp).addMBB(sinkMBB);
    BuildMI(llscMBB, TII->get(Alpha::BISr), reg_store)
      .addReg(Alpha::R31).addReg(MI->getOperand(3).getReg());
    break;
  }
  case Alpha::LAS32:
  case Alpha::LAS64: {
    BuildMI(llscMBB, TII->get(is32 ? Alpha::ADDLr : Alpha::ADDQr), reg_store)
      .addReg(reg_res).addReg(reg_v2);
    break;
  }
  case Alpha::SWAP32:
  case Alpha::SWAP64: {
    BuildMI(llscMBB, TII->get(Alpha::BISr), reg_store)
      .addReg(reg_v2).addReg(reg_v2);
    break;
  }
  }
  BuildMI(llscMBB, TII->get(is32 ? Alpha::STL_C : Alpha::STQ_C), reg_store)
    .addReg(reg_store).addImm(0).addReg(reg_ptr);
  BuildMI(llscMBB, TII->get(Alpha::BEQ))
    .addImm(0).addReg(reg_store).addMBB(llscMBB);
  BuildMI(llscMBB, TII->get(Alpha::BR)).addMBB(sinkMBB);

  thisMBB->addSuccessor(llscMBB);
  llscMBB->addSuccessor(llscMBB);
  llscMBB->addSuccessor(sinkMBB);
  delete MI;   // The pseudo instruction is gone now.

  return sinkMBB;
}

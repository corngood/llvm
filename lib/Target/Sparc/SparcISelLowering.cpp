//===-- SparcISelLowering.cpp - Sparc DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces that Sparc uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "SparcISelLowering.h"
#include "SparcTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;


//===----------------------------------------------------------------------===//
// Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "SparcGenCallingConv.inc"

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG) {
  // CCValAssign - represent the assignment of the return value to locations.
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC   = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  
  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CC, isVarArg, DAG.getTarget(), RVLocs);
  
  // Analize return values of ISD::RET
  CCInfo.AnalyzeReturn(Op.Val, RetCC_Sparc32);
  
  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }
  
  SDOperand Chain = Op.getOperand(0);
  SDOperand Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    
    // ISD::RET => ret chain, (regnum1,val1), ...
    // So i*2+1 index only the regnums.
    Chain = DAG.getCopyToReg(Chain, VA.getLocReg(), Op.getOperand(i*2+1), Flag);
    
    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
  }
  
  if (Flag.Val)
    return DAG.getNode(SPISD::RET_FLAG, MVT::Other, Chain, Flag);
  return DAG.getNode(SPISD::RET_FLAG, MVT::Other, Chain);
}

/// LowerArguments - V8 uses a very simple ABI, where all values are passed in
/// either one or two GPRs, including FP values.  TODO: we should pass FP values
/// in FP registers for fastcc functions.
std::vector<SDOperand>
SparcTargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  std::vector<SDOperand> ArgValues;
  
  static const unsigned ArgRegs[] = {
    SP::I0, SP::I1, SP::I2, SP::I3, SP::I4, SP::I5
  };
  
  const unsigned *CurArgReg = ArgRegs, *ArgRegEnd = ArgRegs+6;
  unsigned ArgOffset = 68;
  
  SDOperand Root = DAG.getRoot();
  std::vector<SDOperand> OutChains;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    MVT::ValueType ObjectVT = getValueType(I->getType());
    
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      if (I->use_empty()) {                // Argument is dead.
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        ArgValues.push_back(DAG.getNode(ISD::UNDEF, ObjectVT));
      } else if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
        unsigned VReg = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
        MF.getRegInfo().addLiveIn(*CurArgReg++, VReg);
        SDOperand Arg = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        if (ObjectVT != MVT::i32) {
          unsigned AssertOp = ISD::AssertSext;
          Arg = DAG.getNode(AssertOp, MVT::i32, Arg, 
                            DAG.getValueType(ObjectVT));
          Arg = DAG.getNode(ISD::TRUNCATE, ObjectVT, Arg);
        }
        ArgValues.push_back(Arg);
      } else {
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        SDOperand Load;
        if (ObjectVT == MVT::i32) {
          Load = DAG.getLoad(MVT::i32, Root, FIPtr, NULL, 0);
        } else {
          ISD::LoadExtType LoadOp = ISD::SEXTLOAD;

          // Sparc is big endian, so add an offset based on the ObjectVT.
          unsigned Offset = 4-std::max(1U, MVT::getSizeInBits(ObjectVT)/8);
          FIPtr = DAG.getNode(ISD::ADD, MVT::i32, FIPtr,
                              DAG.getConstant(Offset, MVT::i32));
          Load = DAG.getExtLoad(LoadOp, MVT::i32, Root, FIPtr,
                                NULL, 0, ObjectVT);
          Load = DAG.getNode(ISD::TRUNCATE, ObjectVT, Load);
        }
        ArgValues.push_back(Load);
      }
      
      ArgOffset += 4;
      break;
    case MVT::f32:
      if (I->use_empty()) {                // Argument is dead.
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        ArgValues.push_back(DAG.getNode(ISD::UNDEF, ObjectVT));
      } else if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
        // FP value is passed in an integer register.
        unsigned VReg = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
        MF.getRegInfo().addLiveIn(*CurArgReg++, VReg);
        SDOperand Arg = DAG.getCopyFromReg(Root, VReg, MVT::i32);

        Arg = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Arg);
        ArgValues.push_back(Arg);
      } else {
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        SDOperand Load = DAG.getLoad(MVT::f32, Root, FIPtr, NULL, 0);
        ArgValues.push_back(Load);
      }
      ArgOffset += 4;
      break;

    case MVT::i64:
    case MVT::f64:
      if (I->use_empty()) {                // Argument is dead.
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        if (CurArgReg < ArgRegEnd) ++CurArgReg;
        ArgValues.push_back(DAG.getNode(ISD::UNDEF, ObjectVT));
      } else if (/* FIXME: Apparently this isn't safe?? */
                 0 && CurArgReg == ArgRegEnd && ObjectVT == MVT::f64 &&
                 ((CurArgReg-ArgRegs) & 1) == 0) {
        // If this is a double argument and the whole thing lives on the stack,
        // and the argument is aligned, load the double straight from the stack.
        // We can't do a load in cases like void foo([6ints], int,double),
        // because the double wouldn't be aligned!
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(8, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        ArgValues.push_back(DAG.getLoad(MVT::f64, Root, FIPtr, NULL, 0));
      } else {
        SDOperand HiVal;
        if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
          unsigned VRegHi = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
          MF.getRegInfo().addLiveIn(*CurArgReg++, VRegHi);
          HiVal = DAG.getCopyFromReg(Root, VRegHi, MVT::i32);
        } else {
          int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
          SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          HiVal = DAG.getLoad(MVT::i32, Root, FIPtr, NULL, 0);
        }
        
        SDOperand LoVal;
        if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
          unsigned VRegLo = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
          MF.getRegInfo().addLiveIn(*CurArgReg++, VRegLo);
          LoVal = DAG.getCopyFromReg(Root, VRegLo, MVT::i32);
        } else {
          int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset+4);
          SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          LoVal = DAG.getLoad(MVT::i32, Root, FIPtr, NULL, 0);
        }
        
        // Compose the two halves together into an i64 unit.
        SDOperand WholeValue = 
          DAG.getNode(ISD::BUILD_PAIR, MVT::i64, LoVal, HiVal);
        
        // If we want a double, do a bit convert.
        if (ObjectVT == MVT::f64)
          WholeValue = DAG.getNode(ISD::BIT_CONVERT, MVT::f64, WholeValue);
        
        ArgValues.push_back(WholeValue);
      }
      ArgOffset += 8;
      break;
    }
  }
  
  // Store remaining ArgRegs to the stack if this is a varargs function.
  if (F.isVarArg()) {
    // Remember the vararg offset for the va_start implementation.
    VarArgsFrameOffset = ArgOffset;
    
    for (; CurArgReg != ArgRegEnd; ++CurArgReg) {
      unsigned VReg = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
      MF.getRegInfo().addLiveIn(*CurArgReg, VReg);
      SDOperand Arg = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);

      int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
      SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);

      OutChains.push_back(DAG.getStore(DAG.getRoot(), Arg, FIPtr, NULL, 0));
      ArgOffset += 4;
    }
  }
  
  if (!OutChains.empty())
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other,
                            &OutChains[0], OutChains.size()));
  
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
SparcTargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy,
                                 bool RetSExt, bool RetZExt, bool isVarArg,
                                 unsigned CC, bool isTailCall, SDOperand Callee,
                                 ArgListTy &Args, SelectionDAG &DAG) {
  // Count the size of the outgoing arguments.
  unsigned ArgsSize = 0;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    switch (getValueType(Args[i].Ty)) {
    default: assert(0 && "Unknown value type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::f32:
      ArgsSize += 4;
      break;
    case MVT::i64:
    case MVT::f64:
      ArgsSize += 8;
      break;
    }
  }
  if (ArgsSize > 4*6)
    ArgsSize -= 4*6;    // Space for first 6 arguments is prereserved.
  else
    ArgsSize = 0;

  // Keep stack frames 8-byte aligned.
  ArgsSize = (ArgsSize+7) & ~7;

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(ArgsSize, getPointerTy()));
  
  SDOperand StackPtr;
  std::vector<SDOperand> Stores;
  std::vector<SDOperand> RegValuesToPass;
  unsigned ArgOffset = 68;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    SDOperand Val = Args[i].Node;
    MVT::ValueType ObjectVT = Val.getValueType();
    SDOperand ValToStore(0, 0);
    unsigned ObjSize;
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16: {
      // Promote the integer to 32-bits.  If the input type is signed, use a
      // sign extend, otherwise use a zero extend.
      ISD::NodeType ExtendKind = ISD::ANY_EXTEND;
      if (Args[i].isSExt)
        ExtendKind = ISD::SIGN_EXTEND;
      else if (Args[i].isZExt)
        ExtendKind = ISD::ZERO_EXTEND;
      Val = DAG.getNode(ExtendKind, MVT::i32, Val);
      // FALL THROUGH
    }
    case MVT::i32:
      ObjSize = 4;

      if (RegValuesToPass.size() >= 6) {
        ValToStore = Val;
      } else {
        RegValuesToPass.push_back(Val);
      }
      break;
    case MVT::f32:
      ObjSize = 4;
      if (RegValuesToPass.size() >= 6) {
        ValToStore = Val;
      } else {
        // Convert this to a FP value in an int reg.
        Val = DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Val);
        RegValuesToPass.push_back(Val);
      }
      break;
    case MVT::f64:
      ObjSize = 8;
      // If we can store this directly into the outgoing slot, do so.  We can
      // do this when all ArgRegs are used and if the outgoing slot is aligned.
      // FIXME: McGill/misr fails with this.
      if (0 && RegValuesToPass.size() >= 6 && ((ArgOffset-68) & 7) == 0) {
        ValToStore = Val;
        break;
      }
      
      // Otherwise, convert this to a FP value in int regs.
      Val = DAG.getNode(ISD::BIT_CONVERT, MVT::i64, Val);
      // FALL THROUGH
    case MVT::i64:
      ObjSize = 8;
      if (RegValuesToPass.size() >= 6) {
        ValToStore = Val;    // Whole thing is passed in memory.
        break;
      }
      
      // Split the value into top and bottom part.  Top part goes in a reg.
      SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, getPointerTy(), Val, 
                                 DAG.getConstant(1, MVT::i32));
      SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, getPointerTy(), Val,
                                 DAG.getConstant(0, MVT::i32));
      RegValuesToPass.push_back(Hi);
      
      if (RegValuesToPass.size() >= 6) {
        ValToStore = Lo;
        ArgOffset += 4;
        ObjSize = 4;
      } else {
        RegValuesToPass.push_back(Lo);
      }
      break;
    }
    
    if (ValToStore.Val) {
      if (!StackPtr.Val) {
        StackPtr = DAG.getRegister(SP::O6, MVT::i32);
      }
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      Stores.push_back(DAG.getStore(Chain, ValToStore, PtrOff, NULL, 0));
    }
    ArgOffset += ObjSize;
  }
  
  // Emit all stores, make sure the occur before any copies into physregs.
  if (!Stores.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Stores[0],Stores.size());
  
  static const unsigned ArgRegs[] = {
    SP::O0, SP::O1, SP::O2, SP::O3, SP::O4, SP::O5
  };
  
  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into O[0-5].
  SDOperand InFlag;
  for (unsigned i = 0, e = RegValuesToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, ArgRegs[i], RegValuesToPass[i], InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  SDOperand Ops[] = { Chain, Callee, InFlag };
  Chain = DAG.getNode(SPISD::CALL, NodeTys, Ops, InFlag.Val ? 3 : 2);
  InFlag = Chain.getValue(1);
  
  MVT::ValueType RetTyVT = getValueType(RetTy);
  
  SDOperand RetVal;
  if (RetTyVT != MVT::isVoid) {
    switch (RetTyVT) {
    default: assert(0 && "Unknown value type to return!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16: {
      RetVal = DAG.getCopyFromReg(Chain, SP::O0, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      
      // Add a note to keep track of whether it is sign or zero extended.
      ISD::NodeType AssertKind = ISD::DELETED_NODE;
      if (RetSExt)
        AssertKind = ISD::AssertSext;
      else if (RetZExt)
        AssertKind = ISD::AssertZext;

      if (AssertKind != ISD::DELETED_NODE)
        RetVal = DAG.getNode(AssertKind, MVT::i32, RetVal,
                             DAG.getValueType(RetTyVT));

      RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
      break;
    }
    case MVT::i32:
      RetVal = DAG.getCopyFromReg(Chain, SP::O0, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::f32:
      RetVal = DAG.getCopyFromReg(Chain, SP::F0, MVT::f32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::f64:
      RetVal = DAG.getCopyFromReg(Chain, SP::D0, MVT::f64, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::i64:
      SDOperand Lo = DAG.getCopyFromReg(Chain, SP::O1, MVT::i32, InFlag);
      SDOperand Hi = DAG.getCopyFromReg(Lo.getValue(1), SP::O0, MVT::i32, 
                                        Lo.getValue(2));
      RetVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
      Chain = Hi.getValue(1);
      break;
    }
  }
  
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(ArgsSize, getPointerTy()),
                             DAG.getConstant(0, getPointerTy()),
                             SDOperand());
  return std::make_pair(RetVal, Chain);
}



//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

/// IntCondCCodeToICC - Convert a DAG integer condition code to a SPARC ICC
/// condition.
static SPCC::CondCodes IntCondCCodeToICC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown integer condition code!");
  case ISD::SETEQ:  return SPCC::ICC_E;
  case ISD::SETNE:  return SPCC::ICC_NE;
  case ISD::SETLT:  return SPCC::ICC_L;
  case ISD::SETGT:  return SPCC::ICC_G;
  case ISD::SETLE:  return SPCC::ICC_LE;
  case ISD::SETGE:  return SPCC::ICC_GE;
  case ISD::SETULT: return SPCC::ICC_CS;
  case ISD::SETULE: return SPCC::ICC_LEU;
  case ISD::SETUGT: return SPCC::ICC_GU;
  case ISD::SETUGE: return SPCC::ICC_CC;
  }
}

/// FPCondCCodeToFCC - Convert a DAG floatingp oint condition code to a SPARC
/// FCC condition.
static SPCC::CondCodes FPCondCCodeToFCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown fp condition code!");
  case ISD::SETEQ:
  case ISD::SETOEQ: return SPCC::FCC_E;
  case ISD::SETNE:
  case ISD::SETUNE: return SPCC::FCC_NE;
  case ISD::SETLT:
  case ISD::SETOLT: return SPCC::FCC_L;
  case ISD::SETGT:
  case ISD::SETOGT: return SPCC::FCC_G;
  case ISD::SETLE:
  case ISD::SETOLE: return SPCC::FCC_LE;
  case ISD::SETGE:
  case ISD::SETOGE: return SPCC::FCC_GE;
  case ISD::SETULT: return SPCC::FCC_UL;
  case ISD::SETULE: return SPCC::FCC_ULE;
  case ISD::SETUGT: return SPCC::FCC_UG;
  case ISD::SETUGE: return SPCC::FCC_UGE;
  case ISD::SETUO:  return SPCC::FCC_U;
  case ISD::SETO:   return SPCC::FCC_O;
  case ISD::SETONE: return SPCC::FCC_LG;
  case ISD::SETUEQ: return SPCC::FCC_UE;
  }
}


SparcTargetLowering::SparcTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  
  // Set up the register classes.
  addRegisterClass(MVT::i32, SP::IntRegsRegisterClass);
  addRegisterClass(MVT::f32, SP::FPRegsRegisterClass);
  addRegisterClass(MVT::f64, SP::DFPRegsRegisterClass);

  // Turn FP extload into load/fextend
  setLoadXAction(ISD::EXTLOAD, MVT::f32, Expand);
  // Sparc doesn't have i1 sign extending load
  setLoadXAction(ISD::SEXTLOAD, MVT::i1, Promote);
  // Turn FP truncstore into trunc + store.
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // Custom legalize GlobalAddress nodes into LO/HI parts.
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool , MVT::i32, Custom);
  
  // Sparc doesn't have sext_inreg, replace them with shl/sra
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1 , Expand);

  // Sparc has no REM or DIVREM operations.
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  // Custom expand fp<->sint
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

  // Expand fp<->uint
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
  
  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Expand);
  setOperationAction(ISD::BIT_CONVERT, MVT::i32, Expand);
  
  // Sparc has no select or setcc: expand to SELECT_CC.
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);
  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SETCC, MVT::f32, Expand);
  setOperationAction(ISD::SETCC, MVT::f64, Expand);
  
  // Sparc doesn't have BRCOND either, it has BR_CC.
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f64, Custom);
  
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);
  
  // SPARC has no intrinsics for these particular operations.
  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);
  setOperationAction(ISD::MEMBARRIER, MVT::Other, Expand);

  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FREM , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FREM , MVT::f32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ , MVT::i32, Expand);
  setOperationAction(ISD::CTLZ , MVT::i32, Expand);
  setOperationAction(ISD::ROTL , MVT::i32, Expand);
  setOperationAction(ISD::ROTR , MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
  setOperationAction(ISD::FPOW , MVT::f64, Expand);
  setOperationAction(ISD::FPOW , MVT::f32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

  // FIXME: Sparc provides these multiplies, but we don't have them yet.
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
    
  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::LABEL, MVT::Other, Expand);

  // RET must be custom lowered, to meet ABI requirements
  setOperationAction(ISD::RET               , MVT::Other, Custom);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex.
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  // VAARG needs to be lowered to not do unaligned accesses for doubles.
  setOperationAction(ISD::VAARG             , MVT::Other, Custom);
  
  // Use the default implementation.
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Custom);

  // No debug info support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::LABEL, MVT::Other, Expand);
  setOperationAction(ISD::DECLARE, MVT::Other, Expand);
    
  setStackPointerRegisterToSaveRestore(SP::O6);

  if (TM.getSubtarget<SparcSubtarget>().isV9())
    setOperationAction(ISD::CTPOP, MVT::i32, Legal);
  
  computeRegisterProperties();
}

const char *SparcTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case SPISD::CMPICC:     return "SPISD::CMPICC";
  case SPISD::CMPFCC:     return "SPISD::CMPFCC";
  case SPISD::BRICC:      return "SPISD::BRICC";
  case SPISD::BRFCC:      return "SPISD::BRFCC";
  case SPISD::SELECT_ICC: return "SPISD::SELECT_ICC";
  case SPISD::SELECT_FCC: return "SPISD::SELECT_FCC";
  case SPISD::Hi:         return "SPISD::Hi";
  case SPISD::Lo:         return "SPISD::Lo";
  case SPISD::FTOI:       return "SPISD::FTOI";
  case SPISD::ITOF:       return "SPISD::ITOF";
  case SPISD::CALL:       return "SPISD::CALL";
  case SPISD::RET_FLAG:   return "SPISD::RET_FLAG";
  }
}

/// isMaskedValueZeroForTargetNode - Return true if 'Op & Mask' is known to
/// be zero. Op is expected to be a target specific node. Used by DAG
/// combiner.
void SparcTargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op,
                                                         const APInt &Mask,
                                                         APInt &KnownZero, 
                                                         APInt &KnownOne,
                                                         const SelectionDAG &DAG,
                                                         unsigned Depth) const {
  APInt KnownZero2, KnownOne2;
  KnownZero = KnownOne = APInt(Mask.getBitWidth(), 0);   // Don't know anything.
  
  switch (Op.getOpcode()) {
  default: break;
  case SPISD::SELECT_ICC:
  case SPISD::SELECT_FCC:
    DAG.ComputeMaskedBits(Op.getOperand(1), Mask, KnownZero, KnownOne,
                          Depth+1);
    DAG.ComputeMaskedBits(Op.getOperand(0), Mask, KnownZero2, KnownOne2,
                          Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  }
}

// Look at LHS/RHS/CC and see if they are a lowered setcc instruction.  If so
// set LHS/RHS and SPCC to the LHS/RHS of the setcc and SPCC to the condition.
static void LookThroughSetCC(SDOperand &LHS, SDOperand &RHS,
                             ISD::CondCode CC, unsigned &SPCC) {
  if (isa<ConstantSDNode>(RHS) && cast<ConstantSDNode>(RHS)->getValue() == 0 &&
      CC == ISD::SETNE && 
      ((LHS.getOpcode() == SPISD::SELECT_ICC &&
        LHS.getOperand(3).getOpcode() == SPISD::CMPICC) ||
       (LHS.getOpcode() == SPISD::SELECT_FCC &&
        LHS.getOperand(3).getOpcode() == SPISD::CMPFCC)) &&
      isa<ConstantSDNode>(LHS.getOperand(0)) &&
      isa<ConstantSDNode>(LHS.getOperand(1)) &&
      cast<ConstantSDNode>(LHS.getOperand(0))->getValue() == 1 &&
      cast<ConstantSDNode>(LHS.getOperand(1))->getValue() == 0) {
    SDOperand CMPCC = LHS.getOperand(3);
    SPCC = cast<ConstantSDNode>(LHS.getOperand(2))->getValue();
    LHS = CMPCC.getOperand(0);
    RHS = CMPCC.getOperand(1);
  }
}

static SDOperand LowerGLOBALADDRESS(SDOperand Op, SelectionDAG &DAG) {
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i32);
  SDOperand Hi = DAG.getNode(SPISD::Hi, MVT::i32, GA);
  SDOperand Lo = DAG.getNode(SPISD::Lo, MVT::i32, GA);
  return DAG.getNode(ISD::ADD, MVT::i32, Lo, Hi);
}

static SDOperand LowerCONSTANTPOOL(SDOperand Op, SelectionDAG &DAG) {
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);
  Constant *C = N->getConstVal();
  SDOperand CP = DAG.getTargetConstantPool(C, MVT::i32, N->getAlignment());
  SDOperand Hi = DAG.getNode(SPISD::Hi, MVT::i32, CP);
  SDOperand Lo = DAG.getNode(SPISD::Lo, MVT::i32, CP);
  return DAG.getNode(ISD::ADD, MVT::i32, Lo, Hi);
}

static SDOperand LowerFP_TO_SINT(SDOperand Op, SelectionDAG &DAG) {
  // Convert the fp value to integer in an FP register.
  assert(Op.getValueType() == MVT::i32);
  Op = DAG.getNode(SPISD::FTOI, MVT::f32, Op.getOperand(0));
  return DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Op);
}

static SDOperand LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  assert(Op.getOperand(0).getValueType() == MVT::i32);
  SDOperand Tmp = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Op.getOperand(0));
  // Convert the int value to FP in an FP register.
  return DAG.getNode(SPISD::ITOF, Op.getValueType(), Tmp);
}

static SDOperand LowerBR_CC(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDOperand LHS = Op.getOperand(2);
  SDOperand RHS = Op.getOperand(3);
  SDOperand Dest = Op.getOperand(4);
  unsigned Opc, SPCC = ~0U;
  
  // If this is a br_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  LookThroughSetCC(LHS, RHS, CC, SPCC);
  
  // Get the condition flag.
  SDOperand CompareFlag;
  if (LHS.getValueType() == MVT::i32) {
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(MVT::i32);
    VTs.push_back(MVT::Flag);
    SDOperand Ops[2] = { LHS, RHS };
    CompareFlag = DAG.getNode(SPISD::CMPICC, VTs, Ops, 2).getValue(1);
    if (SPCC == ~0U) SPCC = IntCondCCodeToICC(CC);
    Opc = SPISD::BRICC;
  } else {
    CompareFlag = DAG.getNode(SPISD::CMPFCC, MVT::Flag, LHS, RHS);
    if (SPCC == ~0U) SPCC = FPCondCCodeToFCC(CC);
    Opc = SPISD::BRFCC;
  }
  return DAG.getNode(Opc, MVT::Other, Chain, Dest,
                     DAG.getConstant(SPCC, MVT::i32), CompareFlag);
}

static SDOperand LowerSELECT_CC(SDOperand Op, SelectionDAG &DAG) {
  SDOperand LHS = Op.getOperand(0);
  SDOperand RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDOperand TrueVal = Op.getOperand(2);
  SDOperand FalseVal = Op.getOperand(3);
  unsigned Opc, SPCC = ~0U;
  
  // If this is a select_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  LookThroughSetCC(LHS, RHS, CC, SPCC);
  
  SDOperand CompareFlag;
  if (LHS.getValueType() == MVT::i32) {
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(LHS.getValueType());   // subcc returns a value
    VTs.push_back(MVT::Flag);
    SDOperand Ops[2] = { LHS, RHS };
    CompareFlag = DAG.getNode(SPISD::CMPICC, VTs, Ops, 2).getValue(1);
    Opc = SPISD::SELECT_ICC;
    if (SPCC == ~0U) SPCC = IntCondCCodeToICC(CC);
  } else {
    CompareFlag = DAG.getNode(SPISD::CMPFCC, MVT::Flag, LHS, RHS);
    Opc = SPISD::SELECT_FCC;
    if (SPCC == ~0U) SPCC = FPCondCCodeToFCC(CC);
  }
  return DAG.getNode(Opc, TrueVal.getValueType(), TrueVal, FalseVal, 
                     DAG.getConstant(SPCC, MVT::i32), CompareFlag);
}

static SDOperand LowerVASTART(SDOperand Op, SelectionDAG &DAG,
                              SparcTargetLowering &TLI) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  SDOperand Offset = DAG.getNode(ISD::ADD, MVT::i32,
                                 DAG.getRegister(SP::I6, MVT::i32),
                                 DAG.getConstant(TLI.getVarArgsFrameOffset(),
                                                 MVT::i32));
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), Offset, Op.getOperand(1), SV, 0);
}

static SDOperand LowerVAARG(SDOperand Op, SelectionDAG &DAG) {
  SDNode *Node = Op.Val;
  MVT::ValueType VT = Node->getValueType(0);
  SDOperand InChain = Node->getOperand(0);
  SDOperand VAListPtr = Node->getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  SDOperand VAList = DAG.getLoad(MVT::i32, InChain, VAListPtr, SV, 0);
  // Increment the pointer, VAList, to the next vaarg
  SDOperand NextPtr = DAG.getNode(ISD::ADD, MVT::i32, VAList, 
                                  DAG.getConstant(MVT::getSizeInBits(VT)/8, 
                                                  MVT::i32));
  // Store the incremented VAList to the legalized pointer
  InChain = DAG.getStore(VAList.getValue(1), NextPtr,
                         VAListPtr, SV, 0);
  // Load the actual argument out of the pointer VAList, unless this is an
  // f64 load.
  if (VT != MVT::f64)
    return DAG.getLoad(VT, InChain, VAList, NULL, 0);
  
  // Otherwise, load it as i64, then do a bitconvert.
  SDOperand V = DAG.getLoad(MVT::i64, InChain, VAList, NULL, 0);
  
  // Bit-Convert the value to f64.
  SDOperand Ops[2] = {
    DAG.getNode(ISD::BIT_CONVERT, MVT::f64, V),
    V.getValue(1)
  };
  return DAG.getNode(ISD::MERGE_VALUES, DAG.getVTList(MVT::f64, MVT::Other),
                     Ops, 2);
}

static SDOperand LowerDYNAMIC_STACKALLOC(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain = Op.getOperand(0);  // Legalize the chain.
  SDOperand Size  = Op.getOperand(1);  // Legalize the size.
  
  unsigned SPReg = SP::O6;
  SDOperand SP = DAG.getCopyFromReg(Chain, SPReg, MVT::i32);
  SDOperand NewSP = DAG.getNode(ISD::SUB, MVT::i32, SP, Size);    // Value
  Chain = DAG.getCopyToReg(SP.getValue(1), SPReg, NewSP);      // Output chain
  
  // The resultant pointer is actually 16 words from the bottom of the stack,
  // to provide a register spill area.
  SDOperand NewVal = DAG.getNode(ISD::ADD, MVT::i32, NewSP,
                                 DAG.getConstant(96, MVT::i32));
  std::vector<MVT::ValueType> Tys;
  Tys.push_back(MVT::i32);
  Tys.push_back(MVT::Other);
  SDOperand Ops[2] = { NewVal, Chain };
  return DAG.getNode(ISD::MERGE_VALUES, Tys, Ops, 2);
}


SDOperand SparcTargetLowering::
LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  // Frame & Return address.  Currently unimplemented
  case ISD::RETURNADDR: return SDOperand();
  case ISD::FRAMEADDR:  return SDOperand();
  case ISD::GlobalTLSAddress:
    assert(0 && "TLS not implemented for Sparc.");
  case ISD::GlobalAddress:      return LowerGLOBALADDRESS(Op, DAG);
  case ISD::ConstantPool:       return LowerCONSTANTPOOL(Op, DAG);
  case ISD::FP_TO_SINT:         return LowerFP_TO_SINT(Op, DAG);
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);
  case ISD::BR_CC:              return LowerBR_CC(Op, DAG);
  case ISD::SELECT_CC:          return LowerSELECT_CC(Op, DAG);
  case ISD::VASTART:            return LowerVASTART(Op, DAG, *this);
  case ISD::VAARG:              return LowerVAARG(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::RET:                return LowerRET(Op, DAG);
  }
}

MachineBasicBlock *
SparcTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *getTargetMachine().getInstrInfo();
  unsigned BROpcode;
  unsigned CC;
  // Figure out the conditional branch opcode to use for this select_cc.
  switch (MI->getOpcode()) {
  default: assert(0 && "Unknown SELECT_CC!");
  case SP::SELECT_CC_Int_ICC:
  case SP::SELECT_CC_FP_ICC:
  case SP::SELECT_CC_DFP_ICC:
    BROpcode = SP::BCOND;
    break;
  case SP::SELECT_CC_Int_FCC:
  case SP::SELECT_CC_FP_FCC:
  case SP::SELECT_CC_DFP_FCC:
    BROpcode = SP::FBCOND;
    break;
  }

  CC = (SPCC::CondCodes)MI->getOperand(3).getImm();
  
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
  //   [f]bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineBasicBlock *copy0MBB = new MachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock(LLVM_BB);
  BuildMI(BB, TII.get(BROpcode)).addMBB(sinkMBB).addImm(CC);
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
  BuildMI(BB, TII.get(SP::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);
  
  delete MI;   // The pseudo instruction is gone now.
  return BB;
}


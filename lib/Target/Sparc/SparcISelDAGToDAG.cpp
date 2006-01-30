//===-- SparcV8ISelDAGToDAG.cpp - A dag to dag inst selector for SparcV8 --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the V8 target
//
//===----------------------------------------------------------------------===//

#include "SparcV8.h"
#include "SparcV8TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

namespace V8ISD {
  enum {
    FIRST_NUMBER = ISD::BUILTIN_OP_END+V8::INSTRUCTION_LIST_END,
    CMPICC,      // Compare two GPR operands, set icc.
    CMPFCC,      // Compare two FP operands, set fcc.
    BRICC,       // Branch to dest on icc condition
    BRFCC,       // Branch to dest on fcc condition
    SELECT_ICC,  // Select between two values using the current ICC flags.
    SELECT_FCC,  // Select between two values using the current FCC flags.
    
    Hi, Lo,      // Hi/Lo operations, typically on a global address.
    
    FTOI,        // FP to Int within a FP register.
    ITOF,        // Int to FP within a FP register.

    CALL,        // A V8 call instruction.
    RET_FLAG,    // Return with a flag operand.
  };
}

// Enums corresponding to SparcV8 condition codes, both icc's and fcc's.  These
// values must be kept in sync with the ones in the .td file.
namespace V8CC {
  enum CondCodes {
  //ICC_A   =  8   ,  // Always
  //ICC_N   =  0   ,  // Never
    ICC_NE  =  9   ,  // Not Equal
    ICC_E   =  1   ,  // Equal
    ICC_G   = 10   ,  // Greater
    ICC_LE  =  2   ,  // Less or Equal
    ICC_GE  = 11   ,  // Greater or Equal
    ICC_L   =  3   ,  // Less
    ICC_GU  = 12   ,  // Greater Unsigned
    ICC_LEU =  4   ,  // Less or Equal Unsigned
    ICC_CC  = 13   ,  // Carry Clear/Great or Equal Unsigned
    ICC_CS  =  5   ,  // Carry Set/Less Unsigned
    ICC_POS = 14   ,  // Positive
    ICC_NEG =  6   ,  // Negative
    ICC_VC  = 15   ,  // Overflow Clear
    ICC_VS  =  7   ,  // Overflow Set
    
  //FCC_A   =  8+16,  // Always
  //FCC_N   =  0+16,  // Never
    FCC_U   =  7+16,  // Unordered
    FCC_G   =  6+16,  // Greater
    FCC_UG  =  5+16,  // Unordered or Greater
    FCC_L   =  4+16,  // Less
    FCC_UL  =  3+16,  // Unordered or Less
    FCC_LG  =  2+16,  // Less or Greater
    FCC_NE  =  1+16,  // Not Equal
    FCC_E   =  9+16,  // Equal
    FCC_UE  = 10+16,  // Unordered or Equal
    FCC_GE  = 11+16,  // Greater or Equal
    FCC_UGE = 12+16,  // Unordered or Greater or Equal
    FCC_LE  = 13+16,  // Less or Equal
    FCC_ULE = 14+16,  // Unordered or Less or Equal
    FCC_O   = 15+16,  // Ordered
  };
}


/// IntCondCCodeToICC - Convert a DAG integer condition code to a SPARC ICC
/// condition.
static V8CC::CondCodes IntCondCCodeToICC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown integer condition code!");
  case ISD::SETEQ:  return V8CC::ICC_E;
  case ISD::SETNE:  return V8CC::ICC_NE;
  case ISD::SETLT:  return V8CC::ICC_L;
  case ISD::SETGT:  return V8CC::ICC_G;
  case ISD::SETLE:  return V8CC::ICC_LE;
  case ISD::SETGE:  return V8CC::ICC_GE;
  case ISD::SETULT: return V8CC::ICC_CS;
  case ISD::SETULE: return V8CC::ICC_LEU;
  case ISD::SETUGT: return V8CC::ICC_GU;
  case ISD::SETUGE: return V8CC::ICC_CC;
  }
}

/// FPCondCCodeToFCC - Convert a DAG floatingp oint condition code to a SPARC
/// FCC condition.
static V8CC::CondCodes FPCondCCodeToFCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown fp condition code!");
  case ISD::SETEQ:  return V8CC::FCC_E;
  case ISD::SETNE:  return V8CC::FCC_NE;
  case ISD::SETLT:  return V8CC::FCC_L;
  case ISD::SETGT:  return V8CC::FCC_G;
  case ISD::SETLE:  return V8CC::FCC_LE;
  case ISD::SETGE:  return V8CC::FCC_GE;
  case ISD::SETULT: return V8CC::FCC_UL;
  case ISD::SETULE: return V8CC::FCC_ULE;
  case ISD::SETUGT: return V8CC::FCC_UG;
  case ISD::SETUGE: return V8CC::FCC_UGE;
  case ISD::SETUO:  return V8CC::FCC_U;
  case ISD::SETO:   return V8CC::FCC_O;
  case ISD::SETONE: return V8CC::FCC_LG;
  case ISD::SETUEQ: return V8CC::FCC_UE;
  }
}
  

static unsigned SPARCCondCodeToBranchInstr(V8CC::CondCodes CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition code");
  case V8CC::ICC_NE:  return V8::BNE;
  case V8CC::ICC_E:   return V8::BE;
  case V8CC::ICC_G:   return V8::BG;
  case V8CC::ICC_LE:  return V8::BLE;
  case V8CC::ICC_GE:  return V8::BGE;
  case V8CC::ICC_L:   return V8::BL;
  case V8CC::ICC_GU:  return V8::BGU;
  case V8CC::ICC_LEU: return V8::BLEU;
  case V8CC::ICC_CC:  return V8::BCC;
  case V8CC::ICC_CS:  return V8::BCS;
  case V8CC::ICC_POS: return V8::BPOS;
  case V8CC::ICC_NEG: return V8::BNEG;
  case V8CC::ICC_VC:  return V8::BVC;
  case V8CC::ICC_VS:  return V8::BVS;
  case V8CC::FCC_U:   return V8::FBU;
  case V8CC::FCC_G:   return V8::FBG;
  case V8CC::FCC_UG:  return V8::FBUG;
  case V8CC::FCC_L:   return V8::FBL;
  case V8CC::FCC_UL:  return V8::FBUL;
  case V8CC::FCC_LG:  return V8::FBLG;
  case V8CC::FCC_NE:  return V8::FBNE;
  case V8CC::FCC_E:   return V8::FBE;
  case V8CC::FCC_UE:  return V8::FBUE;
  case V8CC::FCC_GE:  return V8::FBGE;
  case V8CC::FCC_UGE: return V8::FBUGE;
  case V8CC::FCC_LE:  return V8::FBLE;
  case V8CC::FCC_ULE: return V8::FBULE;
  case V8CC::FCC_O:   return V8::FBO;
  }       
}


namespace {
  class SparcV8TargetLowering : public TargetLowering {
    int VarArgsFrameOffset;   // Frame offset to start of varargs area.
  public:
    SparcV8TargetLowering(TargetMachine &TM);
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    
    /// isMaskedValueZeroForTargetNode - Return true if 'Op & Mask' is known to
    /// be zero. Op is expected to be a target specific node. Used by DAG
    /// combiner.
    virtual bool isMaskedValueZeroForTargetNode(const SDOperand &Op,
                                                uint64_t Mask) const;
    
    virtual std::vector<SDOperand>
      LowerArguments(Function &F, SelectionDAG &DAG);
    virtual std::pair<SDOperand, SDOperand>
      LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                  unsigned CC,
                  bool isTailCall, SDOperand Callee, ArgListTy &Args,
                  SelectionDAG &DAG);
    virtual std::pair<SDOperand, SDOperand>
      LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                              SelectionDAG &DAG);
    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB);
    
    virtual const char *getTargetNodeName(unsigned Opcode) const;
  };
}

SparcV8TargetLowering::SparcV8TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  
  // Set up the register classes.
  addRegisterClass(MVT::i32, V8::IntRegsRegisterClass);
  addRegisterClass(MVT::f32, V8::FPRegsRegisterClass);
  addRegisterClass(MVT::f64, V8::DFPRegsRegisterClass);

  // Custom legalize GlobalAddress nodes into LO/HI parts.
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool , MVT::i32, Custom);
  
  // Sparc doesn't have sext_inreg, replace them with shl/sra
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1 , Expand);

  // Sparc has no REM operation.
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);

  // Custom expand fp<->sint
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

  // Expand fp<->uint
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
  
  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Expand);
  setOperationAction(ISD::BIT_CONVERT, MVT::i32, Expand);
  
  // Turn FP extload into load/fextend
  setOperationAction(ISD::EXTLOAD, MVT::f32, Expand);
  
  // Sparc has no select or setcc: expand to SELECT_CC.
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);
  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SETCC, MVT::f32, Expand);
  setOperationAction(ISD::SETCC, MVT::f64, Expand);
  
  // Sparc doesn't have BRCOND either, it has BR_CC.
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BRCONDTWOWAY, MVT::Other, Expand);
  setOperationAction(ISD::BRTWOWAY_CC, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f64, Custom);
  
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);
  
  // V8 has no intrinsics for these particular operations.
  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);
  
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ , MVT::i32, Expand);
  setOperationAction(ISD::CTLZ , MVT::i32, Expand);
  setOperationAction(ISD::ROTL , MVT::i32, Expand);
  setOperationAction(ISD::ROTR , MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);

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

  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
  
  setStackPointerRegisterToSaveRestore(V8::O6);

  if (TM.getSubtarget<SparcV8Subtarget>().isV9()) {
    setOperationAction(ISD::CTPOP, MVT::i32, Legal);
  }
  
  computeRegisterProperties();
}

const char *SparcV8TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case V8ISD::CMPICC:     return "V8ISD::CMPICC";
  case V8ISD::CMPFCC:     return "V8ISD::CMPFCC";
  case V8ISD::BRICC:      return "V8ISD::BRICC";
  case V8ISD::BRFCC:      return "V8ISD::BRFCC";
  case V8ISD::SELECT_ICC: return "V8ISD::SELECT_ICC";
  case V8ISD::SELECT_FCC: return "V8ISD::SELECT_FCC";
  case V8ISD::Hi:         return "V8ISD::Hi";
  case V8ISD::Lo:         return "V8ISD::Lo";
  case V8ISD::FTOI:       return "V8ISD::FTOI";
  case V8ISD::ITOF:       return "V8ISD::ITOF";
  case V8ISD::CALL:       return "V8ISD::CALL";
  case V8ISD::RET_FLAG:   return "V8ISD::RET_FLAG";
  }
}

/// isMaskedValueZeroForTargetNode - Return true if 'Op & Mask' is known to
/// be zero. Op is expected to be a target specific node. Used by DAG
/// combiner.
bool SparcV8TargetLowering::
isMaskedValueZeroForTargetNode(const SDOperand &Op, uint64_t Mask) const {
  switch (Op.getOpcode()) {
  default: return false; 
  case V8ISD::SELECT_ICC:
  case V8ISD::SELECT_FCC:
    assert(MVT::isInteger(Op.getValueType()) && "Not an integer select!");
    // These operations are masked zero if both the left and the right are zero.
    return MaskedValueIsZero(Op.getOperand(0), Mask) &&
           MaskedValueIsZero(Op.getOperand(1), Mask);
  }
}


/// LowerArguments - V8 uses a very simple ABI, where all values are passed in
/// either one or two GPRs, including FP values.  TODO: we should pass FP values
/// in FP registers for fastcc functions.
std::vector<SDOperand>
SparcV8TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  SSARegMap *RegMap = MF.getSSARegMap();
  std::vector<SDOperand> ArgValues;
  
  static const unsigned ArgRegs[] = {
    V8::I0, V8::I1, V8::I2, V8::I3, V8::I4, V8::I5
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
        unsigned VReg = RegMap->createVirtualRegister(&V8::IntRegsRegClass);
        MF.addLiveIn(*CurArgReg++, VReg);
        SDOperand Arg = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        if (ObjectVT != MVT::i32) {
          unsigned AssertOp = I->getType()->isSigned() ? ISD::AssertSext 
                                                       : ISD::AssertZext;
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
          Load = DAG.getLoad(MVT::i32, Root, FIPtr, DAG.getSrcValue(0));
        } else {
          unsigned LoadOp =
            I->getType()->isSigned() ? ISD::SEXTLOAD : ISD::ZEXTLOAD;

          // Sparc is big endian, so add an offset based on the ObjectVT.
          unsigned Offset = 4-std::max(1U, MVT::getSizeInBits(ObjectVT)/8);
          FIPtr = DAG.getNode(ISD::ADD, MVT::i32, FIPtr,
                              DAG.getConstant(Offset, MVT::i32));
          Load = DAG.getExtLoad(LoadOp, MVT::i32, Root, FIPtr,
                                DAG.getSrcValue(0), ObjectVT);
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
        unsigned VReg = RegMap->createVirtualRegister(&V8::IntRegsRegClass);
        MF.addLiveIn(*CurArgReg++, VReg);
        SDOperand Arg = DAG.getCopyFromReg(Root, VReg, MVT::i32);

        Arg = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Arg);
        ArgValues.push_back(Arg);
      } else {
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        SDOperand Load = DAG.getLoad(MVT::f32, Root, FIPtr, DAG.getSrcValue(0));
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
      } else if (CurArgReg == ArgRegEnd && ObjectVT == MVT::f64 &&
                 ((CurArgReg-ArgRegs) & 1) == 0) {
        // If this is a double argument and the whole thing lives on the stack,
        // and the argument is aligned, load the double straight from the stack.
        // We can't do a load in cases like void foo([6ints], int,double),
        // because the double wouldn't be aligned!
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(8, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
        ArgValues.push_back(DAG.getLoad(MVT::f64, Root, FIPtr, 
                                        DAG.getSrcValue(0)));
      } else {
        SDOperand HiVal;
        if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
          unsigned VRegHi = RegMap->createVirtualRegister(&V8::IntRegsRegClass);
          MF.addLiveIn(*CurArgReg++, VRegHi);
          HiVal = DAG.getCopyFromReg(Root, VRegHi, MVT::i32);
        } else {
          int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
          SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          HiVal = DAG.getLoad(MVT::i32, Root, FIPtr, DAG.getSrcValue(0));
        }
        
        SDOperand LoVal;
        if (CurArgReg < ArgRegEnd) {  // Lives in an incoming GPR
          unsigned VRegLo = RegMap->createVirtualRegister(&V8::IntRegsRegClass);
          MF.addLiveIn(*CurArgReg++, VRegLo);
          LoVal = DAG.getCopyFromReg(Root, VRegLo, MVT::i32);
        } else {
          int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset+4);
          SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          LoVal = DAG.getLoad(MVT::i32, Root, FIPtr, DAG.getSrcValue(0));
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
  if (F.getFunctionType()->isVarArg()) {
    // Remember the vararg offset for the va_start implementation.
    VarArgsFrameOffset = ArgOffset;
    
    for (; CurArgReg != ArgRegEnd; ++CurArgReg) {
      unsigned VReg = RegMap->createVirtualRegister(&V8::IntRegsRegClass);
      MF.addLiveIn(*CurArgReg, VReg);
      SDOperand Arg = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);

      int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
      SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);

      OutChains.push_back(DAG.getNode(ISD::STORE, MVT::Other, DAG.getRoot(),
                                      Arg, FIPtr, DAG.getSrcValue(0)));
      ArgOffset += 4;
    }
  }
  
  if (!OutChains.empty())
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, OutChains));
  
  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(V8::I0);
    break;
  case MVT::i64:
    MF.addLiveOut(V8::I0);
    MF.addLiveOut(V8::I1);
    break;
  case MVT::f32:
    MF.addLiveOut(V8::F0);
    break;
  case MVT::f64:
    MF.addLiveOut(V8::D0);
    break;
  }
  
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
SparcV8TargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy,
                                   bool isVarArg, unsigned CC,
                                   bool isTailCall, SDOperand Callee, 
                                   ArgListTy &Args, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  // Count the size of the outgoing arguments.
  unsigned ArgsSize = 0;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    switch (getValueType(Args[i].second)) {
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

  Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                      DAG.getConstant(ArgsSize, getPointerTy()));
  
  SDOperand StackPtr, NullSV;
  std::vector<SDOperand> Stores;
  std::vector<SDOperand> RegValuesToPass;
  unsigned ArgOffset = 68;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    SDOperand Val = Args[i].first;
    MVT::ValueType ObjectVT = Val.getValueType();
    SDOperand ValToStore(0, 0);
    unsigned ObjSize;
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
      // Promote the integer to 32-bits.  If the input type is signed, use a
      // sign extend, otherwise use a zero extend.
      if (Args[i].second->isSigned())
        Val = DAG.getNode(ISD::SIGN_EXTEND, MVT::i32, Val);
      else
        Val = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Val);
      // FALL THROUGH
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
      SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Val, 
                                 DAG.getConstant(1, MVT::i32));
      SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Val,
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
        StackPtr = DAG.getRegister(V8::O6, MVT::i32);
        NullSV = DAG.getSrcValue(NULL);
      }
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                   ValToStore, PtrOff, NullSV));
    }
    ArgOffset += ObjSize;
  }
  
  // Emit all stores, make sure the occur before any copies into physregs.
  if (!Stores.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, Stores);
  
  static const unsigned ArgRegs[] = {
    V8::O0, V8::O1, V8::O2, V8::O3, V8::O4, V8::O5
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
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i32);

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  if (InFlag.Val)
    Ops.push_back(InFlag);
  Chain = DAG.getNode(V8ISD::CALL, NodeTys, Ops);
  InFlag = Chain.getValue(1);
  
  MVT::ValueType RetTyVT = getValueType(RetTy);
  SDOperand RetVal;
  if (RetTyVT != MVT::isVoid) {
    switch (RetTyVT) {
    default: assert(0 && "Unknown value type to return!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
      RetVal = DAG.getCopyFromReg(Chain, V8::O0, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      
      // Add a note to keep track of whether it is sign or zero extended.
      RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext :ISD::AssertZext,
                           MVT::i32, RetVal, DAG.getValueType(RetTyVT));
      RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
      break;
    case MVT::i32:
      RetVal = DAG.getCopyFromReg(Chain, V8::O0, MVT::i32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::f32:
      RetVal = DAG.getCopyFromReg(Chain, V8::F0, MVT::f32, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::f64:
      RetVal = DAG.getCopyFromReg(Chain, V8::D0, MVT::f64, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::i64:
      SDOperand Lo = DAG.getCopyFromReg(Chain, V8::O1, MVT::i32, InFlag);
      SDOperand Hi = DAG.getCopyFromReg(Lo.getValue(1), V8::O0, MVT::i32, 
                                        Lo.getValue(2));
      RetVal = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
      Chain = Hi.getValue(1);
      break;
    }
  }
  
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(ArgsSize, getPointerTy()));
  
  return std::make_pair(RetVal, Chain);
}

std::pair<SDOperand, SDOperand> SparcV8TargetLowering::
LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  assert(0 && "Unimp");
  abort();
}

SDOperand SparcV8TargetLowering::
LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
    SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i32);
    SDOperand Hi = DAG.getNode(V8ISD::Hi, MVT::i32, GA);
    SDOperand Lo = DAG.getNode(V8ISD::Lo, MVT::i32, GA);
    return DAG.getNode(ISD::ADD, MVT::i32, Lo, Hi);
  }
  case ISD::ConstantPool: {
    Constant *C = cast<ConstantPoolSDNode>(Op)->get();
    SDOperand CP = DAG.getTargetConstantPool(C, MVT::i32);
    SDOperand Hi = DAG.getNode(V8ISD::Hi, MVT::i32, CP);
    SDOperand Lo = DAG.getNode(V8ISD::Lo, MVT::i32, CP);
    return DAG.getNode(ISD::ADD, MVT::i32, Lo, Hi);
  }
  case ISD::FP_TO_SINT:
    // Convert the fp value to integer in an FP register.
    assert(Op.getValueType() == MVT::i32);
    Op = DAG.getNode(V8ISD::FTOI, MVT::f32, Op.getOperand(0));
    return DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Op);
  case ISD::SINT_TO_FP: {
    assert(Op.getOperand(0).getValueType() == MVT::i32);
    SDOperand Tmp = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, Op.getOperand(0));
    // Convert the int value to FP in an FP register.
    return DAG.getNode(V8ISD::ITOF, Op.getValueType(), Tmp);
  }
  case ISD::BR_CC: {
    SDOperand Chain = Op.getOperand(0);
    ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
    SDOperand LHS = Op.getOperand(2);
    SDOperand RHS = Op.getOperand(3);
    SDOperand Dest = Op.getOperand(4);
    
    // Get the condition flag.
    if (LHS.getValueType() == MVT::i32) {
      std::vector<MVT::ValueType> VTs;
      VTs.push_back(MVT::i32);
      VTs.push_back(MVT::Flag);
      std::vector<SDOperand> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      SDOperand Cond = DAG.getNode(V8ISD::CMPICC, VTs, Ops).getValue(1);
      SDOperand CCN = DAG.getConstant(IntCondCCodeToICC(CC), MVT::i32);
      return DAG.getNode(V8ISD::BRICC, MVT::Other, Chain, Dest, CCN, Cond);
    } else {
      SDOperand Cond = DAG.getNode(V8ISD::CMPFCC, MVT::Flag, LHS, RHS);
      SDOperand CCN = DAG.getConstant(FPCondCCodeToFCC(CC), MVT::i32);
      return DAG.getNode(V8ISD::BRFCC, MVT::Other, Chain, Dest, CCN, Cond);
    }
  }
  case ISD::SELECT_CC: {
    SDOperand LHS = Op.getOperand(0);
    SDOperand RHS = Op.getOperand(1);
    ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
    SDOperand TrueVal = Op.getOperand(2);
    SDOperand FalseVal = Op.getOperand(3);
    unsigned Opc, V8CC = ~0U;

    // If this is a select_cc of a "setcc", and if the setcc got lowered into
    // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
    if (isa<ConstantSDNode>(RHS) && cast<ConstantSDNode>(RHS)->getValue() == 0&&
        CC == ISD::SETNE && 
        ((LHS.getOpcode() == V8ISD::SELECT_ICC &&
          LHS.getOperand(3).getOpcode() == V8ISD::CMPICC) ||
         (LHS.getOpcode() == V8ISD::SELECT_FCC &&
          LHS.getOperand(3).getOpcode() == V8ISD::CMPFCC)) &&
        isa<ConstantSDNode>(LHS.getOperand(0)) &&
        isa<ConstantSDNode>(LHS.getOperand(1)) &&
        cast<ConstantSDNode>(LHS.getOperand(0))->getValue() == 1 &&
        cast<ConstantSDNode>(LHS.getOperand(1))->getValue() == 0) {
      SDOperand CMPCC = LHS.getOperand(3);
      V8CC = cast<ConstantSDNode>(LHS.getOperand(2))->getValue();
      LHS = CMPCC.getOperand(0);
      RHS = CMPCC.getOperand(1);
    }
    
    SDOperand CompareFlag;
    if (LHS.getValueType() == MVT::i32) {
      std::vector<MVT::ValueType> VTs;
      VTs.push_back(LHS.getValueType());   // subcc returns a value
      VTs.push_back(MVT::Flag);
      std::vector<SDOperand> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      CompareFlag = DAG.getNode(V8ISD::CMPICC, VTs, Ops).getValue(1);
      Opc = V8ISD::SELECT_ICC;
      if (V8CC == ~0U) V8CC = IntCondCCodeToICC(CC);
    } else {
      CompareFlag = DAG.getNode(V8ISD::CMPFCC, MVT::Flag, LHS, RHS);
      Opc = V8ISD::SELECT_FCC;
      if (V8CC == ~0U) V8CC = FPCondCCodeToFCC(CC);
    }
    return DAG.getNode(Opc, TrueVal.getValueType(), TrueVal, FalseVal, 
                       DAG.getConstant(V8CC, MVT::i32), CompareFlag);
  }
  case ISD::VASTART: {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDOperand Offset = DAG.getNode(ISD::ADD, MVT::i32,
                                   DAG.getRegister(V8::I6, MVT::i32),
                                   DAG.getConstant(VarArgsFrameOffset, MVT::i32));
    return DAG.getNode(ISD::STORE, MVT::Other, Op.getOperand(0), Offset, 
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
      unsigned ArgReg;
      switch(Op.getOperand(1).getValueType()) {
      default: assert(0 && "Unknown type to return!");
      case MVT::i32: ArgReg = V8::I0; break;
      case MVT::f32: ArgReg = V8::F0; break;
      case MVT::f64: ArgReg = V8::D0; break;
      }
      Copy = DAG.getCopyToReg(Op.getOperand(0), ArgReg, Op.getOperand(1),
                              SDOperand());
      break;
    }
    case 3:
      Copy = DAG.getCopyToReg(Op.getOperand(0), V8::I0, Op.getOperand(2), 
                              SDOperand());
      Copy = DAG.getCopyToReg(Copy, V8::I1, Op.getOperand(1), Copy.getValue(1));
      break;
    }
    return DAG.getNode(V8ISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
  }
  }
}

MachineBasicBlock *
SparcV8TargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                               MachineBasicBlock *BB) {
  unsigned BROpcode;
  // Figure out the conditional branch opcode to use for this select_cc.
  switch (MI->getOpcode()) {
  default: assert(0 && "Unknown SELECT_CC!");
  case V8::SELECT_CC_Int_ICC:
  case V8::SELECT_CC_FP_ICC:
  case V8::SELECT_CC_DFP_ICC:
  case V8::SELECT_CC_Int_FCC:
  case V8::SELECT_CC_FP_FCC:
  case V8::SELECT_CC_DFP_FCC:
    V8CC::CondCodes CC = (V8CC::CondCodes)MI->getOperand(3).getImmedValue();
    BROpcode = SPARCCondCodeToBranchInstr(CC);
    break;
  }
  
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
  BuildMI(BB, BROpcode, 1).addMBB(sinkMBB);
  MachineFunction *F = BB->getParent();
  F->getBasicBlockList().insert(It, copy0MBB);
  F->getBasicBlockList().insert(It, sinkMBB);
  // Update machine-CFG edges
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
  BuildMI(BB, V8::PHI, 4, MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);
  
  delete MI;   // The pseudo instruction is gone now.
  return BB;
}
  
//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
/// SparcV8DAGToDAGISel - SPARC specific code to select Sparc V8 machine
/// instructions for SelectionDAG operations.
///
namespace {
class SparcV8DAGToDAGISel : public SelectionDAGISel {
  SparcV8TargetLowering V8Lowering;

  /// Subtarget - Keep a pointer to the Sparc Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const SparcV8Subtarget &Subtarget;
public:
  SparcV8DAGToDAGISel(TargetMachine &TM)
    : SelectionDAGISel(V8Lowering), V8Lowering(TM),
      Subtarget(TM.getSubtarget<SparcV8Subtarget>()) {
  }

  SDOperand Select(SDOperand Op);

  // Complex Pattern Selectors.
  bool SelectADDRrr(SDOperand N, SDOperand &R1, SDOperand &R2);
  bool SelectADDRri(SDOperand N, SDOperand &Base, SDOperand &Offset);
  
  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
  
  virtual const char *getPassName() const {
    return "SparcV8 DAG->DAG Pattern Instruction Selection";
  } 
  
  // Include the pieces autogenerated from the target description.
#include "SparcV8GenDAGISel.inc"
};
}  // end anonymous namespace

/// InstructionSelectBasicBlock - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void SparcV8DAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());
  
  // Select target instructions for the DAG.
  DAG.setRoot(Select(DAG.getRoot()));
  CodeGenMap.clear();
  DAG.RemoveDeadNodes();
  
  // Emit machine code to BB. 
  ScheduleAndEmitDAG(DAG);
}

bool SparcV8DAGToDAGISel::SelectADDRri(SDOperand Addr, SDOperand &Base,
                                       SDOperand &Offset) {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
    Offset = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }
  
  if (Addr.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      if (Predicate_simm13(CN)) {
        if (FrameIndexSDNode *FIN = 
                dyn_cast<FrameIndexSDNode>(Addr.getOperand(0))) {
          // Constant offset from frame ref.
          Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
        } else {
          Base = Select(Addr.getOperand(0));
        }
        Offset = CurDAG->getTargetConstant(CN->getValue(), MVT::i32);
        return true;
      }
    }
    if (Addr.getOperand(0).getOpcode() == V8ISD::Lo) {
      Base = Select(Addr.getOperand(1));
      Offset = Addr.getOperand(0).getOperand(0);
      return true;
    }
    if (Addr.getOperand(1).getOpcode() == V8ISD::Lo) {
      Base = Select(Addr.getOperand(0));
      Offset = Addr.getOperand(1).getOperand(0);
      return true;
    }
  }
  Base = Select(Addr);
  Offset = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool SparcV8DAGToDAGISel::SelectADDRrr(SDOperand Addr, SDOperand &R1, 
                                       SDOperand &R2) {
  if (Addr.getOpcode() == ISD::FrameIndex) return false; 
  if (Addr.getOpcode() == ISD::ADD) {
    if (isa<ConstantSDNode>(Addr.getOperand(1)) &&
        Predicate_simm13(Addr.getOperand(1).Val))
      return false;  // Let the reg+imm pattern catch this!
    if (Addr.getOperand(0).getOpcode() == V8ISD::Lo ||
        Addr.getOperand(1).getOpcode() == V8ISD::Lo)
      return false;  // Let the reg+imm pattern catch this!
    R1 = Select(Addr.getOperand(0));
    R2 = Select(Addr.getOperand(1));
    return true;
  }

  R1 = Select(Addr);
  R2 = CurDAG->getRegister(V8::G0, MVT::i32);
  return true;
}

SDOperand SparcV8DAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&
      N->getOpcode() < V8ISD::FIRST_NUMBER)
    return Op;   // Already selected.
                 // If this has already been converted, use it.
  std::map<SDOperand, SDOperand>::iterator CGMI = CodeGenMap.find(Op);
  if (CGMI != CodeGenMap.end()) return CGMI->second;
  
  switch (N->getOpcode()) {
  default: break;
  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    if (N->hasOneUse())
      return CurDAG->SelectNodeTo(N, V8::ADDri, MVT::i32,
                                  CurDAG->getTargetFrameIndex(FI, MVT::i32),
                                  CurDAG->getTargetConstant(0, MVT::i32));
    return CodeGenMap[Op] = 
      CurDAG->getTargetNode(V8::ADDri, MVT::i32,
                            CurDAG->getTargetFrameIndex(FI, MVT::i32),
                            CurDAG->getTargetConstant(0, MVT::i32));
  }
  case ISD::ADD_PARTS: {
    SDOperand LHSL = Select(N->getOperand(0));
    SDOperand LHSH = Select(N->getOperand(1));
    SDOperand RHSL = Select(N->getOperand(2));
    SDOperand RHSH = Select(N->getOperand(3));
    // FIXME, handle immediate RHS.
    SDOperand Low = CurDAG->getTargetNode(V8::ADDCCrr, MVT::i32, MVT::Flag,
                                          LHSL, RHSL);
    SDOperand Hi  = CurDAG->getTargetNode(V8::ADDXrr, MVT::i32, LHSH, RHSH, 
                                          Low.getValue(1));
    CodeGenMap[SDOperand(N, 0)] = Low;
    CodeGenMap[SDOperand(N, 1)] = Hi;
    return Op.ResNo ? Hi : Low;
  }
  case ISD::SUB_PARTS: {
    SDOperand LHSL = Select(N->getOperand(0));
    SDOperand LHSH = Select(N->getOperand(1));
    SDOperand RHSL = Select(N->getOperand(2));
    SDOperand RHSH = Select(N->getOperand(3));
    // FIXME, handle immediate RHS.
    SDOperand Low = CurDAG->getTargetNode(V8::SUBCCrr, MVT::i32, MVT::Flag,
                                          LHSL, RHSL);
    SDOperand Hi  = CurDAG->getTargetNode(V8::SUBXrr, MVT::i32, LHSH, RHSH, 
                                          Low.getValue(1));
    CodeGenMap[SDOperand(N, 0)] = Low;
    CodeGenMap[SDOperand(N, 1)] = Hi;
    return Op.ResNo ? Hi : Low;
  }
  case ISD::SDIV:
  case ISD::UDIV: {
    // FIXME: should use a custom expander to expose the SRA to the dag.
    SDOperand DivLHS = Select(N->getOperand(0));
    SDOperand DivRHS = Select(N->getOperand(1));
    
    // Set the Y register to the high-part.
    SDOperand TopPart;
    if (N->getOpcode() == ISD::SDIV) {
      TopPart = CurDAG->getTargetNode(V8::SRAri, MVT::i32, DivLHS,
                                      CurDAG->getTargetConstant(31, MVT::i32));
    } else {
      TopPart = CurDAG->getRegister(V8::G0, MVT::i32);
    }
    TopPart = CurDAG->getTargetNode(V8::WRYrr, MVT::Flag, TopPart,
                                    CurDAG->getRegister(V8::G0, MVT::i32));

    // FIXME: Handle div by immediate.
    unsigned Opcode = N->getOpcode() == ISD::SDIV ? V8::SDIVrr : V8::UDIVrr;
    return CurDAG->SelectNodeTo(N, Opcode, MVT::i32, DivLHS, DivRHS, TopPart);
  }    
  case ISD::MULHU:
  case ISD::MULHS: {
    // FIXME: Handle mul by immediate.
    SDOperand MulLHS = Select(N->getOperand(0));
    SDOperand MulRHS = Select(N->getOperand(1));
    unsigned Opcode = N->getOpcode() == ISD::MULHU ? V8::UMULrr : V8::SMULrr;
    SDOperand Mul = CurDAG->getTargetNode(Opcode, MVT::i32, MVT::Flag,
                                          MulLHS, MulRHS);
    // The high part is in the Y register.
    return CurDAG->SelectNodeTo(N, V8::RDY, MVT::i32, Mul.getValue(1));
  }
  case V8ISD::CALL:
    // FIXME: This is a workaround for a bug in tblgen.
  { // Pattern #47: (call:Flag (tglobaladdr:i32):$dst, ICC:Flag)
    // Emits: (CALL:void (tglobaladdr:i32):$dst)
    // Pattern complexity = 2  cost = 1
    SDOperand N1 = N->getOperand(1);
    if (N1.getOpcode() != ISD::TargetGlobalAddress &&
        N1.getOpcode() != ISD::ExternalSymbol) goto P47Fail;
    SDOperand InFlag = SDOperand(0, 0);
    SDOperand Chain = N->getOperand(0);
    SDOperand Tmp0 = N1;
    Chain = Select(Chain);
    SDOperand Result;
    if (N->getNumOperands() == 3) {
      InFlag = Select(N->getOperand(2));
      Result = CurDAG->getTargetNode(V8::CALL, MVT::Other, MVT::Flag, Tmp0, 
                                     Chain, InFlag);
    } else {
      Result = CurDAG->getTargetNode(V8::CALL, MVT::Other, MVT::Flag, Tmp0, 
                                     Chain);
    }
    Chain = CodeGenMap[SDOperand(N, 0)] = Result.getValue(0);
     CodeGenMap[SDOperand(N, 1)] = Result.getValue(1);
    return Result.getValue(Op.ResNo);
  }
    P47Fail:;
    
  }
  
  return SelectCode(Op);
}


/// createSparcV8ISelDag - This pass converts a legalized DAG into a 
/// SPARC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createSparcV8ISelDag(TargetMachine &TM) {
  return new SparcV8DAGToDAGISel(TM);
}

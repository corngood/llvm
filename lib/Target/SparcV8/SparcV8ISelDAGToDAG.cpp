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
    CMPICC,   // Compare two GPR operands, set icc.
    CMPFCC,   // Compare two FP operands, set fcc.
    BRICC,    // Branch to dest on icc condition
    BRFCC,    // Branch to dest on fcc condition
    
    Hi, Lo,   // Hi/Lo operations, typically on a global address.
    
    FTOI,     // FP to Int within a FP register.
    ITOF,     // Int to FP within a FP register.
    
    SELECT_ICC, // Select between two values using the current ICC flags.
    SELECT_FCC, // Select between two values using the current FCC flags.
  };
}

namespace {
  class SparcV8TargetLowering : public TargetLowering {
  public:
    SparcV8TargetLowering(TargetMachine &TM);
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    virtual std::vector<SDOperand>
      LowerArguments(Function &F, SelectionDAG &DAG);
    virtual std::pair<SDOperand, SDOperand>
      LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                  unsigned CC,
                  bool isTailCall, SDOperand Callee, ArgListTy &Args,
                  SelectionDAG &DAG);
    
    virtual SDOperand LowerReturnTo(SDOperand Chain, SDOperand Op,
                                    SelectionDAG &DAG);
    virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                   Value *VAListV, SelectionDAG &DAG);
    virtual std::pair<SDOperand,SDOperand>
      LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
                 const Type *ArgTy, SelectionDAG &DAG);
    virtual std::pair<SDOperand, SDOperand>
      LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                              SelectionDAG &DAG);
    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB);
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
  
  computeRegisterProperties();
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
    // TODO: FP
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

          Load = DAG.getExtLoad(LoadOp, MVT::i32, Root, FIPtr,
                                DAG.getSrcValue(0), ObjectVT);
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

        // We use the stack space that is already reserved for this reg.
        int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset);
        SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);

        SDOperand SV = DAG.getSrcValue(0);
        SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Root,
                                      Arg, FIPtr, SV);
        ArgValues.push_back(DAG.getLoad(MVT::f32, Store, FIPtr, SV));
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
                      
        if (ObjectVT == MVT::i64) {
          // If we are emitting an i64, this is what we want.
          ArgValues.push_back(WholeValue);
        } else {
          assert(ObjectVT == MVT::f64);
          // Otherwise, emit a store to the stack and reload into FPR.
          int FrameIdx = MF.getFrameInfo()->CreateStackObject(8, 8);
          SDOperand FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          SDOperand SV = DAG.getSrcValue(0);
          SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Root,
                                        WholeValue, FIPtr, SV);
          ArgValues.push_back(DAG.getLoad(MVT::f64, Store, FIPtr, SV));
        }
      }
      ArgOffset += 8;
      break;
    }
  }
  
  // Store remaining ArgRegs to the stack if this is a varargs function.
  if (F.getFunctionType()->isVarArg()) {
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
  assert(0 && "Unimp");
  abort();
}

SDOperand SparcV8TargetLowering::LowerReturnTo(SDOperand Chain, SDOperand Op,
                                               SelectionDAG &DAG) {
  if (Op.getValueType() == MVT::i64) {
    SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op, 
                               DAG.getConstant(1, MVT::i32));
    SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, Op,
                               DAG.getConstant(0, MVT::i32));
    return DAG.getNode(ISD::RET, MVT::Other, Chain, Lo, Hi);
  } else {
    return DAG.getNode(ISD::RET, MVT::Other, Chain, Op);
  }
}

SDOperand SparcV8TargetLowering::
LowerVAStart(SDOperand Chain, SDOperand VAListP, Value *VAListV, 
             SelectionDAG &DAG) {
             
  assert(0 && "Unimp");
  abort();
}

std::pair<SDOperand,SDOperand> SparcV8TargetLowering::
LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
           const Type *ArgTy, SelectionDAG &DAG) {
  assert(0 && "Unimp");
  abort();
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
  case ISD::FP_TO_SINT: {
    // Convert the fp value to integer in an FP register.
    Op = DAG.getNode(V8ISD::FTOI, Op.getOperand(0).getValueType(),
                     Op.getOperand(0));
    int Size = Op.getOperand(0).getValueType() == MVT::f32 ? 4 : 8;
    int FrameIdx =
      DAG.getMachineFunction().getFrameInfo()->CreateStackObject(Size, Size);
    SDOperand FI = DAG.getFrameIndex(FrameIdx, MVT::i32);
    SDOperand ST = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                               Op, FI, DAG.getSrcValue(0));
    return DAG.getLoad(MVT::i32, ST, FI, DAG.getSrcValue(0));
  }
  case ISD::SINT_TO_FP: {
    int Size = Op.getOperand(0).getValueType() == MVT::f32 ? 4 : 8;
    int FrameIdx =
      DAG.getMachineFunction().getFrameInfo()->CreateStackObject(Size, Size);
    SDOperand FI = DAG.getFrameIndex(FrameIdx, MVT::i32);
    SDOperand ST = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                               Op.getOperand(0), FI, DAG.getSrcValue(0));
    
    Op = DAG.getLoad(Op.getValueType(), ST, FI, DAG.getSrcValue(0));
    
    // Convert the int value to FP in an FP register.
    return DAG.getNode(V8ISD::ITOF, Op.getValueType(), Op);
  }
  case ISD::BR_CC: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand CC = Op.getOperand(1);
    SDOperand LHS = Op.getOperand(2);
    SDOperand RHS = Op.getOperand(3);
    SDOperand Dest = Op.getOperand(4);
    
    // Get the condition flag.
    if (LHS.getValueType() == MVT::i32) {
      SDOperand Cond = DAG.getNode(V8ISD::CMPICC, MVT::Flag, LHS, RHS);
      return DAG.getNode(V8ISD::BRICC, MVT::Other, Chain, Dest, CC, Cond);
    } else {
      SDOperand Cond = DAG.getNode(V8ISD::CMPFCC, MVT::Flag, LHS, RHS);
      return DAG.getNode(V8ISD::BRFCC, MVT::Other, Chain, Dest, CC, Cond);
    }
  }
  case ISD::SELECT_CC: {
    SDOperand LHS = Op.getOperand(0);
    SDOperand RHS = Op.getOperand(1);
    unsigned CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
    SDOperand TrueVal = Op.getOperand(2);
    SDOperand FalseVal = Op.getOperand(3);
    
    unsigned Opc;
    Opc = LHS.getValueType() == MVT::i32 ? V8ISD::CMPICC : V8ISD::CMPFCC;
    SDOperand CompareFlag = DAG.getNode(Opc, MVT::Flag, LHS, RHS);
    
    Opc = LHS.getValueType() == MVT::i32 ? 
      V8ISD::SELECT_ICC : V8ISD::SELECT_FCC;
    return DAG.getNode(Opc, TrueVal.getValueType(), TrueVal, FalseVal, 
                       DAG.getConstant(CC, MVT::i32), CompareFlag);
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
    // Integer compare.
    switch ((ISD::CondCode)MI->getOperand(3).getImmedValue()) {
    default: assert(0 && "Unknown integer condition code!");
    case ISD::SETEQ:  BROpcode = V8::BE; break;
    case ISD::SETNE:  BROpcode = V8::BNE; break;
    case ISD::SETLT:  BROpcode = V8::BL; break;
    case ISD::SETGT:  BROpcode = V8::BG; break;
    case ISD::SETLE:  BROpcode = V8::BLE; break;
    case ISD::SETGE:  BROpcode = V8::BGE; break;
    case ISD::SETULT: BROpcode = V8::BCS; break;
    case ISD::SETULE: BROpcode = V8::BLEU; break;
    case ISD::SETUGT: BROpcode = V8::BGU; break;
    case ISD::SETUGE: BROpcode = V8::BCC; break;
    }
    break;
  case V8::SELECT_CC_Int_FCC:
  case V8::SELECT_CC_FP_FCC:
  case V8::SELECT_CC_DFP_FCC:
    // FP compare.
    switch ((ISD::CondCode)MI->getOperand(3).getImmedValue()) {
    default: assert(0 && "Unknown fp condition code!");
    case ISD::SETEQ:  BROpcode = V8::FBE; break;
    case ISD::SETNE:  BROpcode = V8::FBNE; break;
    case ISD::SETLT:  BROpcode = V8::FBL; break;
    case ISD::SETGT:  BROpcode = V8::FBG; break;
    case ISD::SETLE:  BROpcode = V8::FBLE; break;
    case ISD::SETGE:  BROpcode = V8::FBGE; break;
    case ISD::SETULT: BROpcode = V8::FBUL; break;
    case ISD::SETULE: BROpcode = V8::FBULE; break;
    case ISD::SETUGT: BROpcode = V8::FBUG; break;
    case ISD::SETUGE: BROpcode = V8::FBUGE; break;
    case ISD::SETUO:  BROpcode = V8::FBU; break;
    case ISD::SETO:   BROpcode = V8::FBO; break;
    case ISD::SETONE: BROpcode = V8::FBLG; break;
    case ISD::SETUEQ: BROpcode = V8::FBUE; break;
    }
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
/// SparcV8DAGToDAGISel - PPC specific code to select Sparc V8 machine
/// instructions for SelectionDAG operations.
///
namespace {
class SparcV8DAGToDAGISel : public SelectionDAGISel {
  SparcV8TargetLowering V8Lowering;
public:
  SparcV8DAGToDAGISel(TargetMachine &TM)
    : SelectionDAGISel(V8Lowering), V8Lowering(TM) {}

  SDOperand Select(SDOperand Op);

  // Complex Pattern Selectors.
  bool SelectADDRrr(SDOperand N, SDOperand &R1, SDOperand &R2);
  bool SelectADDRri(SDOperand N, SDOperand &Base, SDOperand &Offset);
  
  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
  
  virtual const char *getPassName() const {
    return "PowerPC DAG->DAG Pattern Instruction Selection";
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
  case ISD::BasicBlock:         return CodeGenMap[Op] = Op;
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
  case V8ISD::CMPICC: {
    // FIXME: Handle compare with immediate.
    SDOperand LHS = Select(N->getOperand(0));
    SDOperand RHS = Select(N->getOperand(1));
    SDOperand Result = CurDAG->getTargetNode(V8::SUBCCrr, MVT::i32, MVT::Flag,
                                             LHS, RHS);
    return CodeGenMap[Op] = Result.getValue(1);
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
    
  case ISD::RET: {
    if (N->getNumOperands() == 2) {
      SDOperand Chain = Select(N->getOperand(0));     // Token chain.
      SDOperand Val = Select(N->getOperand(1));
      if (N->getOperand(1).getValueType() == MVT::i32) {
        Chain = CurDAG->getCopyToReg(Chain, V8::I0, Val);
      } else if (N->getOperand(1).getValueType() == MVT::f32) {
        Chain = CurDAG->getCopyToReg(Chain, V8::F0, Val);
      } else {
        assert(N->getOperand(1).getValueType() == MVT::f64);
        Chain = CurDAG->getCopyToReg(Chain, V8::D0, Val);
      }
      return CurDAG->SelectNodeTo(N, V8::RETL, MVT::Other, Chain);
    } else if (N->getNumOperands() > 1) {
      SDOperand Chain = Select(N->getOperand(0));     // Token chain.
      assert(N->getOperand(1).getValueType() == MVT::i32 &&
             N->getOperand(2).getValueType() == MVT::i32 &&
             N->getNumOperands() == 3 && "Unknown two-register ret value!");
      Chain = CurDAG->getCopyToReg(Chain, V8::I1, Select(N->getOperand(1)));
      Chain = CurDAG->getCopyToReg(Chain, V8::I0, Select(N->getOperand(2)));
      return CurDAG->SelectNodeTo(N, V8::RETL, MVT::Other, Chain);
    }
    break;  // Generated code handles the void case.
  }
  }
  
  return SelectCode(Op);
}


/// createPPCISelDag - This pass converts a legalized DAG into a 
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createSparcV8ISelDag(TargetMachine &TM) {
  return new SparcV8DAGToDAGISel(TM);
}

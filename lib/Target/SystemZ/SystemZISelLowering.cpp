//===-- SystemZISelLowering.cpp - SystemZ DAG Lowering Implementation  -----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZTargetLowering class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-lower"

#include "SystemZISelLowering.h"
#include "SystemZ.h"
#include "SystemZTargetMachine.h"
#include "SystemZSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/VectorExtras.h"
using namespace llvm;

SystemZTargetLowering::SystemZTargetLowering(SystemZTargetMachine &tm) :
  TargetLowering(tm), Subtarget(*tm.getSubtargetImpl()), TM(tm) {

  // Set up the register classes.
  addRegisterClass(MVT::i32, SystemZ::GR32RegisterClass);
  addRegisterClass(MVT::i64, SystemZ::GR64RegisterClass);

  // Compute derived properties from the register classes
  computeRegisterProperties();

  // Set shifts properties
  setShiftAmountFlavor(Extend);
  setShiftAmountType(MVT::i32);

  // Provide all sorts of operation actions

  setStackPointerRegisterToSaveRestore(SystemZ::R15D);
  setSchedulingPreference(SchedulingForLatency);

  setOperationAction(ISD::RET,              MVT::Other, Custom);
}

SDValue SystemZTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  case ISD::FORMAL_ARGUMENTS: return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::RET:              return LowerRET(Op, DAG);
  default:
    assert(0 && "unimplemented operand");
    return SDValue();
  }
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "SystemZGenCallingConv.inc"

SDValue SystemZTargetLowering::LowerFORMAL_ARGUMENTS(SDValue Op,
                                                     SelectionDAG &DAG) {
  unsigned CC = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  switch (CC) {
  default:
    assert(0 && "Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    return LowerCCCArguments(Op, DAG);
  }
}

/// LowerCCCArguments - transform physical registers into virtual registers and
/// generate load operations for arguments places on the stack.
// FIXME: struct return stuff
// FIXME: varargs
SDValue SystemZTargetLowering::LowerCCCArguments(SDValue Op,
                                                 SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  SDValue Root = Op.getOperand(0);
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue() != 0;
  unsigned CC = MF.getFunction()->getCallingConv();
  DebugLoc dl = Op.getDebugLoc();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);
  CCInfo.AnalyzeFormalArguments(Op.getNode(), CC_SystemZ);

  assert(!isVarArg && "Varargs not supported yet");

  SmallVector<SDValue, 16> ArgValues;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      MVT RegVT = VA.getLocVT();
      switch (RegVT.getSimpleVT()) {
      default:
        cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
             << RegVT.getSimpleVT()
             << "\n";
        abort();
      case MVT::i64:
        unsigned VReg =
          RegInfo.createVirtualRegister(SystemZ::GR64RegisterClass);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        SDValue ArgValue = DAG.getCopyFromReg(Root, dl, VReg, RegVT);

        // If this is an 8/16/32-bit value, it is really passed promoted to 64
        // bits. Insert an assert[sz]ext to capture this, then truncate to the
        // right size.
        if (VA.getLocInfo() == CCValAssign::SExt)
          ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));

        if (VA.getLocInfo() != CCValAssign::Full)
          ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);

        ArgValues.push_back(ArgValue);
      }
    } else {
      // Sanity check
      assert(VA.isMemLoc());
      // Load the argument to a virtual register
      unsigned ObjSize = VA.getLocVT().getSizeInBits()/8;
      if (ObjSize > 8) {
        cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
             << VA.getLocVT().getSimpleVT()
             << "\n";
      }
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(ObjSize, VA.getLocMemOffset());

      // Create the SelectionDAG nodes corresponding to a load
      //from this parameter
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i64);
      ArgValues.push_back(DAG.getLoad(VA.getLocVT(), dl, Root, FIN,
                                      PseudoSourceValue::getFixedStack(FI), 0));
    }
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getNode(ISD::MERGE_VALUES, dl, Op.getNode()->getVTList(),
                     &ArgValues[0], ArgValues.size()).getValue(Op.getResNo());
}

SDValue SystemZTargetLowering::LowerRET(SDValue Op, SelectionDAG &DAG) {
  // CCValAssign - represent the assignment of the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC   = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  DebugLoc dl = Op.getDebugLoc();

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CC, isVarArg, getTargetMachine(), RVLocs);

  // Analize return values of ISD::RET
  CCInfo.AnalyzeReturn(Op.getNode(), RetCC_SystemZ);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  // The chain is always operand #0
  SDValue Chain = Op.getOperand(0);
  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    SDValue ResValue = Op.getOperand(i*2+1);
    assert(VA.isRegLoc() && "Can only return in registers!");

    // If this is an 8/16/32-bit value, it is really should be passed promoted
    // to 64 bits.
    if (VA.getLocInfo() == CCValAssign::SExt)
      ResValue = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), ResValue);
    else if (VA.getLocInfo() == CCValAssign::ZExt)
      ResValue = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), ResValue);
    else if (VA.getLocInfo() == CCValAssign::AExt)
      ResValue = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), ResValue);

    // ISD::RET => ret chain, (regnum1,val1), ...
    // So i*2+1 index only the regnums
    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), ResValue, Flag);

    // Guarantee that all emitted copies are stuck together,
    // avoiding something bad.
    Flag = Chain.getValue(1);
  }

  if (Flag.getNode())
    return DAG.getNode(SystemZISD::RET_FLAG, dl, MVT::Other, Chain, Flag);

  // Return Void
  return DAG.getNode(SystemZISD::RET_FLAG, dl, MVT::Other, Chain);
}

const char *SystemZTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case SystemZISD::RET_FLAG:           return "SystemZISD::RET_FLAG";
  default: return NULL;
  }
}


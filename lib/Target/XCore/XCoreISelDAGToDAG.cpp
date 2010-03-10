//===-- XCoreISelDAGToDAG.cpp - A dag to dag inst selector for XCore ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the XCore target.
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "XCoreISelLowering.h"
#include "XCoreTargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <set>
using namespace llvm;

/// XCoreDAGToDAGISel - XCore specific code to select XCore machine
/// instructions for SelectionDAG operations.
///
namespace {
  class XCoreDAGToDAGISel : public SelectionDAGISel {
    XCoreTargetLowering &Lowering;
    const XCoreSubtarget &Subtarget;

  public:
    XCoreDAGToDAGISel(XCoreTargetMachine &TM)
      : SelectionDAGISel(TM),
        Lowering(*TM.getTargetLowering()), 
        Subtarget(*TM.getSubtargetImpl()) { }

    SDNode *Select(SDNode *N);
    
    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDValue getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }

    // Complex Pattern Selectors.
    bool SelectADDRspii(SDNode *Op, SDValue Addr, SDValue &Base,
                        SDValue &Offset);
    bool SelectADDRdpii(SDNode *Op, SDValue Addr, SDValue &Base,
                        SDValue &Offset);
    bool SelectADDRcpii(SDNode *Op, SDValue Addr, SDValue &Base,
                        SDValue &Offset);
    
    virtual const char *getPassName() const {
      return "XCore DAG->DAG Pattern Instruction Selection";
    } 
    
    // Include the pieces autogenerated from the target description.
  #include "XCoreGenDAGISel.inc"
  };
}  // end anonymous namespace

/// createXCoreISelDag - This pass converts a legalized DAG into a 
/// XCore-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createXCoreISelDag(XCoreTargetMachine &TM) {
  return new XCoreDAGToDAGISel(TM);
}

bool XCoreDAGToDAGISel::SelectADDRspii(SDNode *Op, SDValue Addr,
                                  SDValue &Base, SDValue &Offset) {
  FrameIndexSDNode *FIN = 0;
  if ((FIN = dyn_cast<FrameIndexSDNode>(Addr))) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
    Offset = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }
  if (Addr.getOpcode() == ISD::ADD) {
    ConstantSDNode *CN = 0;
    if ((FIN = dyn_cast<FrameIndexSDNode>(Addr.getOperand(0)))
      && (CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))
      && (CN->getSExtValue() % 4 == 0 && CN->getSExtValue() >= 0)) {
      // Constant positive word offset from frame index
      Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
      Offset = CurDAG->getTargetConstant(CN->getSExtValue(), MVT::i32);
      return true;
    }
  }
  return false;
}

bool XCoreDAGToDAGISel::SelectADDRdpii(SDNode *Op, SDValue Addr,
                                  SDValue &Base, SDValue &Offset) {
  if (Addr.getOpcode() == XCoreISD::DPRelativeWrapper) {
    Base = Addr.getOperand(0);
    Offset = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }
  if (Addr.getOpcode() == ISD::ADD) {
    ConstantSDNode *CN = 0;
    if ((Addr.getOperand(0).getOpcode() == XCoreISD::DPRelativeWrapper)
      && (CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))
      && (CN->getSExtValue() % 4 == 0)) {
      // Constant word offset from a object in the data region
      Base = Addr.getOperand(0).getOperand(0);
      Offset = CurDAG->getTargetConstant(CN->getSExtValue(), MVT::i32);
      return true;
    }
  }
  return false;
}

bool XCoreDAGToDAGISel::SelectADDRcpii(SDNode *Op, SDValue Addr,
                                  SDValue &Base, SDValue &Offset) {
  if (Addr.getOpcode() == XCoreISD::CPRelativeWrapper) {
    Base = Addr.getOperand(0);
    Offset = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }
  if (Addr.getOpcode() == ISD::ADD) {
    ConstantSDNode *CN = 0;
    if ((Addr.getOperand(0).getOpcode() == XCoreISD::CPRelativeWrapper)
      && (CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))
      && (CN->getSExtValue() % 4 == 0)) {
      // Constant word offset from a object in the data region
      Base = Addr.getOperand(0).getOperand(0);
      Offset = CurDAG->getTargetConstant(CN->getSExtValue(), MVT::i32);
      return true;
    }
  }
  return false;
}

SDNode *XCoreDAGToDAGISel::Select(SDNode *N) {
  DebugLoc dl = N->getDebugLoc();
  EVT NVT = N->getValueType(0);
  if (NVT == MVT::i32) {
    switch (N->getOpcode()) {
      default: break;
      case ISD::Constant: {
        if (Predicate_immMskBitp(N)) {
          // Transformation function: get the size of a mask
          int64_t MaskVal = cast<ConstantSDNode>(N)->getZExtValue();
          assert(isMask_32(MaskVal));
          // Look for the first non-zero bit
          SDValue MskSize = getI32Imm(32 - CountLeadingZeros_32(MaskVal));
          return CurDAG->getMachineNode(XCore::MKMSK_rus, dl,
                                        MVT::i32, MskSize);
        }
        else if (! Predicate_immU16(N)) {
          unsigned Val = cast<ConstantSDNode>(N)->getZExtValue();
          SDValue CPIdx =
            CurDAG->getTargetConstantPool(ConstantInt::get(
                                  Type::getInt32Ty(*CurDAG->getContext()), Val),
                                          TLI.getPointerTy());
          return CurDAG->getMachineNode(XCore::LDWCP_lru6, dl, MVT::i32, 
                                        MVT::Other, CPIdx, 
                                        CurDAG->getEntryNode());
        }
        break;
      }
      case ISD::SMUL_LOHI: {
        // FIXME fold addition into the macc instruction
        SDValue Zero(CurDAG->getMachineNode(XCore::LDC_ru6, dl, MVT::i32,
                                CurDAG->getTargetConstant(0, MVT::i32)), 0);
        SDValue Ops[] = { Zero, Zero, N->getOperand(0), N->getOperand(1) };
        SDNode *ResNode = CurDAG->getMachineNode(XCore::MACCS_l4r, dl,
                                                 MVT::i32, MVT::i32, Ops, 4);
        ReplaceUses(SDValue(N, 0), SDValue(ResNode, 1));
        ReplaceUses(SDValue(N, 1), SDValue(ResNode, 0));
        return NULL;
      }
      case ISD::UMUL_LOHI: {
        // FIXME fold addition into the macc / lmul instruction
        SDValue Zero(CurDAG->getMachineNode(XCore::LDC_ru6, dl, MVT::i32,
                                  CurDAG->getTargetConstant(0, MVT::i32)), 0);
        SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                            Zero, Zero };
        SDNode *ResNode = CurDAG->getMachineNode(XCore::LMUL_l6r, dl, MVT::i32,
                                                 MVT::i32, Ops, 4);
        ReplaceUses(SDValue(N, 0), SDValue(ResNode, 1));
        ReplaceUses(SDValue(N, 1), SDValue(ResNode, 0));
        return NULL;
      }
      case XCoreISD::LADD: {
        SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                            N->getOperand(2) };
        return CurDAG->getMachineNode(XCore::LADD_l5r, dl, MVT::i32, MVT::i32,
                                      Ops, 3);
      }
      case XCoreISD::LSUB: {
        SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                            N->getOperand(2) };
        return CurDAG->getMachineNode(XCore::LSUB_l5r, dl, MVT::i32, MVT::i32,
                                      Ops, 3);
      }
      case XCoreISD::MACCU: {
        SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                          N->getOperand(2), N->getOperand(3) };
        return CurDAG->getMachineNode(XCore::MACCU_l4r, dl, MVT::i32, MVT::i32,
                                      Ops, 4);
      }
      case XCoreISD::MACCS: {
        SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                          N->getOperand(2), N->getOperand(3) };
        return CurDAG->getMachineNode(XCore::MACCS_l4r, dl, MVT::i32, MVT::i32,
                                      Ops, 4);
      }
      // Other cases are autogenerated.
    }
  }
  return SelectCode(N);
}

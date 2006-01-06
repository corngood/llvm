//===- X86ISelDAGToDAG.cpp - A DAG pattern matching inst selector for X86 -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DAG pattern matching instruction selector for X86,
// converting from a legalized dag to a X86 dag.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86Subtarget.h"
#include "X86ISelLowering.h"
#include "llvm/GlobalValue.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                      Pattern Matcher Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// X86ISelAddressMode - This corresponds to X86AddressMode, but uses
  /// SDOperand's instead of register numbers for the leaves of the matched
  /// tree.
  struct X86ISelAddressMode {
    enum {
      RegBase,
      FrameIndexBase,
      ConstantPoolBase
    } BaseType;

    struct {            // This is really a union, discriminated by BaseType!
      SDOperand Reg;
      int FrameIndex;
    } Base;

    unsigned Scale;
    SDOperand IndexReg; 
    unsigned Disp;
    GlobalValue *GV;

    X86ISelAddressMode()
      : BaseType(RegBase), Scale(1), IndexReg(), Disp(0), GV(0) {
    }
  };
}

namespace {
  Statistic<>
  NumFPKill("x86-codegen", "Number of FP_REG_KILL instructions added");

  //===--------------------------------------------------------------------===//
  /// ISel - X86 specific code to select X86 machine instructions for
  /// SelectionDAG operations.
  ///
  class X86DAGToDAGISel : public SelectionDAGISel {
    /// ContainsFPCode - Every instruction we select that uses or defines a FP
    /// register should set this to true.
    bool ContainsFPCode;

    /// X86Lowering - This object fully describes how to lower LLVM code to an
    /// X86-specific SelectionDAG.
    X86TargetLowering X86Lowering;

    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;
  public:
    X86DAGToDAGISel(TargetMachine &TM)
      : SelectionDAGISel(X86Lowering), X86Lowering(TM) {
      Subtarget = &TM.getSubtarget<X86Subtarget>();
    }

    virtual const char *getPassName() const {
      return "X86 DAG->DAG Instruction Selection";
    }

    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);

// Include the pieces autogenerated from the target description.
#include "X86GenDAGISel.inc"

  private:
    SDOperand Select(SDOperand N);

    bool MatchAddress(SDOperand N, X86ISelAddressMode &AM);
    bool SelectAddr(SDOperand N, SDOperand &Base, SDOperand &Scale,
                    SDOperand &Index, SDOperand &Disp);
    bool SelectLEAAddr(SDOperand N, SDOperand &Base, SDOperand &Scale,
                       SDOperand &Index, SDOperand &Disp);

    inline void getAddressOperands(X86ISelAddressMode &AM, SDOperand &Base, 
                                   SDOperand &Scale, SDOperand &Index,
                                   SDOperand &Disp) {
      Base  = (AM.BaseType == X86ISelAddressMode::FrameIndexBase) ?
        CurDAG->getTargetFrameIndex(AM.Base.FrameIndex, MVT::i32) : AM.Base.Reg;
      Scale = getI8Imm(AM.Scale);
      Index = AM.IndexReg;
      Disp  = AM.GV ? CurDAG->getTargetGlobalAddress(AM.GV, MVT::i32, AM.Disp)
        : getI32Imm(AM.Disp);
    }

    /// getI8Imm - Return a target constant with the specified value, of type
    /// i8.
    inline SDOperand getI8Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i8);
    }

    /// getI16Imm - Return a target constant with the specified value, of type
    /// i16.
    inline SDOperand getI16Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i16);
    }

    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDOperand getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }
  };
}

/// InstructionSelectBasicBlock - This callback is invoked by SelectionDAGISel
/// when it has created a SelectionDAG for us to codegen.
void X86DAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());

  // Codegen the basic block.
  DAG.setRoot(Select(DAG.getRoot()));
  CodeGenMap.clear();
  DAG.RemoveDeadNodes();

  // Emit machine code to BB. 
  ScheduleAndEmitDAG(DAG);
}

/// FIXME: copied from X86ISelPattern.cpp
/// MatchAddress - Add the specified node to the specified addressing mode,
/// returning true if it cannot be done.  This just pattern matches for the
/// addressing mode
bool X86DAGToDAGISel::MatchAddress(SDOperand N, X86ISelAddressMode &AM) {
  switch (N.getOpcode()) {
  default: break;
  case ISD::FrameIndex:
    if (AM.BaseType == X86ISelAddressMode::RegBase && AM.Base.Reg.Val == 0) {
      AM.BaseType = X86ISelAddressMode::FrameIndexBase;
      AM.Base.FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return false;
    }
    break;

  case ISD::ConstantPool:
    if (AM.BaseType == X86ISelAddressMode::RegBase && AM.Base.Reg.Val == 0) {
      if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(N)) {
        AM.BaseType = X86ISelAddressMode::ConstantPoolBase;
        AM.Base.Reg = CurDAG->getTargetConstantPool(CP->get(), MVT::i32);
        return false;
      }
    }
    break;

  case ISD::GlobalAddress:
  case ISD::TargetGlobalAddress:
    if (AM.GV == 0) {
      AM.GV = cast<GlobalAddressSDNode>(N)->getGlobal();
      return false;
    }
    break;

  case ISD::Constant:
    AM.Disp += cast<ConstantSDNode>(N)->getValue();
    return false;

  case ISD::SHL:
    if (AM.IndexReg.Val == 0 && AM.Scale == 1)
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.Val->getOperand(1))) {
        unsigned Val = CN->getValue();
        if (Val == 1 || Val == 2 || Val == 3) {
          AM.Scale = 1 << Val;
          SDOperand ShVal = N.Val->getOperand(0);

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (ShVal.Val->getOpcode() == ISD::ADD && ShVal.hasOneUse() &&
              isa<ConstantSDNode>(ShVal.Val->getOperand(1))) {
            AM.IndexReg = ShVal.Val->getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(ShVal.Val->getOperand(1));
            AM.Disp += AddVal->getValue() << Val;
          } else {
            AM.IndexReg = ShVal;
          }
          return false;
        }
      }
    break;

  case ISD::MUL:
    // X*[3,5,9] -> X+X*[2,4,8]
    if (AM.IndexReg.Val == 0 && AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base.Reg.Val == 0)
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.Val->getOperand(1)))
        if (CN->getValue() == 3 || CN->getValue() == 5 || CN->getValue() == 9) {
          AM.Scale = unsigned(CN->getValue())-1;

          SDOperand MulVal = N.Val->getOperand(0);
          SDOperand Reg;

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (MulVal.Val->getOpcode() == ISD::ADD && MulVal.hasOneUse() &&
              isa<ConstantSDNode>(MulVal.Val->getOperand(1))) {
            Reg = MulVal.Val->getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(MulVal.Val->getOperand(1));
            AM.Disp += AddVal->getValue() * CN->getValue();
          } else {
            Reg = N.Val->getOperand(0);
          }

          AM.IndexReg = AM.Base.Reg = Reg;
          return false;
        }
    break;

  case ISD::ADD: {
    X86ISelAddressMode Backup = AM;
    if (!MatchAddress(N.Val->getOperand(0), AM) &&
        !MatchAddress(N.Val->getOperand(1), AM))
      return false;
    AM = Backup;
    if (!MatchAddress(N.Val->getOperand(1), AM) &&
        !MatchAddress(N.Val->getOperand(0), AM))
      return false;
    AM = Backup;
    break;
  }
  }

  // Is the base register already occupied?
  if (AM.BaseType != X86ISelAddressMode::RegBase || AM.Base.Reg.Val) {
    // If so, check to see if the scale index register is set.
    if (AM.IndexReg.Val == 0) {
      AM.IndexReg = N;
      AM.Scale = 1;
      return false;
    }

    // Otherwise, we cannot select it.
    return true;
  }

  // Default, generate it as a register.
  AM.BaseType = X86ISelAddressMode::RegBase;
  AM.Base.Reg = N;
  return false;
}

/// SelectAddr - returns true if it is able pattern match an addressing mode.
/// It returns the operands which make up the maximal addressing mode it can
/// match by reference.
bool X86DAGToDAGISel::SelectAddr(SDOperand N, SDOperand &Base, SDOperand &Scale,
                                 SDOperand &Index, SDOperand &Disp) {
  X86ISelAddressMode AM;
  if (!MatchAddress(N, AM)) {
    if (AM.BaseType == X86ISelAddressMode::RegBase) {
      if (AM.Base.Reg.Val)
        AM.Base.Reg = Select(AM.Base.Reg);
      else
        AM.Base.Reg = CurDAG->getRegister(0, MVT::i32);
    }
    if (AM.IndexReg.Val)
      AM.IndexReg = Select(AM.IndexReg);
    else
      AM.IndexReg = CurDAG->getRegister(0, MVT::i32);

    getAddressOperands(AM, Base, Scale, Index, Disp);
    return true;
  }
  return false;
}

static bool isRegister0(SDOperand Op)
{
  if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(Op))
    return (R->getReg() == 0);
  return false;
}

/// SelectLEAAddr - it calls SelectAddr and determines if the maximal addressing
/// mode it matches can be cost effectively emitted as an LEA instruction.
/// For X86, it always is unless it's just a (Reg + const).
bool X86DAGToDAGISel::SelectLEAAddr(SDOperand N, SDOperand &Base, SDOperand &Scale,
                                    SDOperand &Index, SDOperand &Disp) {
  X86ISelAddressMode AM;
  if (!MatchAddress(N, AM)) {
    bool SelectBase  = false;
    bool SelectIndex = false;
    bool Check       = false;
    if (AM.BaseType == X86ISelAddressMode::RegBase) {
      if (AM.Base.Reg.Val) {
        Check      = true;
        SelectBase = true;
      } else {
        AM.Base.Reg = CurDAG->getRegister(0, MVT::i32);
      }
    }

    if (AM.IndexReg.Val) {
      SelectIndex = true;
    } else {
      AM.IndexReg = CurDAG->getRegister(0, MVT::i32);
    }

    if (Check) {
      unsigned Complexity = 0;
      if (AM.Scale > 1)
        Complexity++;
      if (SelectIndex)
        Complexity++;
      if (AM.GV)
        Complexity++;
      else if (AM.Disp > 1)
        Complexity++;
      if (Complexity <= 1)
        return false;
    }

    if (SelectBase)
      AM.Base.Reg = Select(AM.Base.Reg);
    if (SelectIndex)
      AM.IndexReg = Select(AM.IndexReg);

    getAddressOperands(AM, Base, Scale, Index, Disp);
    return true;
  }
  return false;
}

SDOperand X86DAGToDAGISel::Select(SDOperand N) {
  SDNode *Node = N.Val;
  MVT::ValueType NVT = Node->getValueType(0);
  unsigned Opc;

  if (Node->getOpcode() >= ISD::BUILTIN_OP_END &&
      Node->getOpcode() < X86ISD::FIRST_NUMBER)
    return N;   // Already selected.
  
  switch (Node->getOpcode()) {
    default: break;

    case ISD::TRUNCATE: {
      unsigned Reg;
      MVT::ValueType VT;
      switch (Node->getOperand(0).getValueType()) {
        default: assert(0 && "Unknown truncate!");
        case MVT::i16: Reg = X86::AX;  Opc = X86::MOV16rr; VT = MVT::i16; break;
        case MVT::i32: Reg = X86::EAX; Opc = X86::MOV32rr; VT = MVT::i32; break;
      }
      SDOperand Tmp0 = Select(Node->getOperand(0));
      SDOperand Tmp1 = CurDAG->getTargetNode(Opc, VT, Tmp0);
      SDOperand InFlag = SDOperand(0,0);
      SDOperand Result = CurDAG->getCopyToReg(CurDAG->getEntryNode(),
                                              Reg, Tmp1, InFlag).getValue(1);
      SDOperand Chain = Result.getValue(0);
      InFlag = Result.getValue(1);

      switch (NVT) {
        default: assert(0 && "Unknown truncate!");
        case MVT::i8:  Reg = X86::AL;  Opc = X86::MOV8rr;  VT = MVT::i8;  break;
        case MVT::i16: Reg = X86::AX;  Opc = X86::MOV16rr; VT = MVT::i16; break;
      }

      Result = CurDAG->getCopyFromReg(Chain,
                                      Reg, VT, InFlag);
      return CodeGenMap[N] = CurDAG->getTargetNode(Opc, VT, Result);
      break;
    }
  }

  return SelectCode(N);
}

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createX86ISelDag(TargetMachine &TM) {
  return new X86DAGToDAGISel(TM);
}

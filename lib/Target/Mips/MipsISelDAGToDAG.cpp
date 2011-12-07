//===-- MipsISelDAGToDAG.cpp - A dag to dag inst selector for Mips --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the MIPS target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-isel"
#include "Mips.h"
#include "MipsMachineFunction.h"
#include "MipsRegisterInfo.h"
#include "MipsSubtarget.h"
#include "MipsTargetMachine.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/CFG.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MipsDAGToDAGISel - MIPS specific code to select MIPS machine
// instructions for SelectionDAG operations.
//===----------------------------------------------------------------------===//
namespace {

class MipsDAGToDAGISel : public SelectionDAGISel {

  /// TM - Keep a reference to MipsTargetMachine.
  MipsTargetMachine &TM;

  /// Subtarget - Keep a pointer to the MipsSubtarget around so that we can
  /// make the right decision when generating code for different targets.
  const MipsSubtarget &Subtarget;

public:
  explicit MipsDAGToDAGISel(MipsTargetMachine &tm) :
  SelectionDAGISel(tm),
  TM(tm), Subtarget(tm.getSubtarget<MipsSubtarget>()) {}

  // Pass Name
  virtual const char *getPassName() const {
    return "MIPS DAG->DAG Pattern Instruction Selection";
  }


private:
  // Include the pieces autogenerated from the target description.
  #include "MipsGenDAGISel.inc"

  /// getTargetMachine - Return a reference to the TargetMachine, casted
  /// to the target-specific type.
  const MipsTargetMachine &getTargetMachine() {
    return static_cast<const MipsTargetMachine &>(TM);
  }

  /// getInstrInfo - Return a reference to the TargetInstrInfo, casted
  /// to the target-specific type.
  const MipsInstrInfo *getInstrInfo() {
    return getTargetMachine().getInstrInfo();
  }

  SDNode *getGlobalBaseReg();
  SDNode *Select(SDNode *N);

  // Complex Pattern.
  bool SelectAddr(SDValue N, SDValue &Base, SDValue &Offset);

  // getImm - Return a target constant with the specified value.
  inline SDValue getImm(const SDNode *Node, unsigned Imm) {
    return CurDAG->getTargetConstant(Imm, Node->getValueType(0));
  }

  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps);
};

}


/// getGlobalBaseReg - Output the instructions required to put the
/// GOT address into a register.
SDNode *MipsDAGToDAGISel::getGlobalBaseReg() {
  unsigned GlobalBaseReg = getInstrInfo()->getGlobalBaseReg(MF);
  return CurDAG->getRegister(GlobalBaseReg, TLI.getPointerTy()).getNode();
}

/// ComplexPattern used on MipsInstrInfo
/// Used on Mips Load/Store instructions
bool MipsDAGToDAGISel::
SelectAddr(SDValue Addr, SDValue &Base, SDValue &Offset) {
  EVT ValTy = Addr.getValueType();
  unsigned GPReg = ValTy == MVT::i32 ? Mips::GP : Mips::GP_64;

  // if Address is FI, get the TargetFrameIndex.
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base   = CurDAG->getTargetFrameIndex(FIN->getIndex(), ValTy);
    Offset = CurDAG->getTargetConstant(0, ValTy);
    return true;
  }

  // on PIC code Load GA
  if (TM.getRelocationModel() == Reloc::PIC_) {
    if (Addr.getOpcode() == MipsISD::WrapperPIC) {
      Base   = CurDAG->getRegister(GPReg, ValTy);
      Offset = Addr.getOperand(0);
      return true;
    }
  } else {
    if ((Addr.getOpcode() == ISD::TargetExternalSymbol ||
        Addr.getOpcode() == ISD::TargetGlobalAddress))
      return false;
    else if (Addr.getOpcode() == ISD::TargetGlobalTLSAddress) {
      Base   = CurDAG->getRegister(GPReg, ValTy);
      Offset = Addr;
      return true;
    }
  }

  // Addresses of the form FI+const or FI|const
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1));
    if (isInt<16>(CN->getSExtValue())) {

      // If the first operand is a FI, get the TargetFI Node
      if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>
                                  (Addr.getOperand(0)))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), ValTy);
      else
        Base = Addr.getOperand(0);

      Offset = CurDAG->getTargetConstant(CN->getZExtValue(), ValTy);
      return true;
    }
  }

  // Operand is a result from an ADD.
  if (Addr.getOpcode() == ISD::ADD) {
    // When loading from constant pools, load the lower address part in
    // the instruction itself. Example, instead of:
    //  lui $2, %hi($CPI1_0)
    //  addiu $2, $2, %lo($CPI1_0)
    //  lwc1 $f0, 0($2)
    // Generate:
    //  lui $2, %hi($CPI1_0)
    //  lwc1 $f0, %lo($CPI1_0)($2)
    if ((Addr.getOperand(0).getOpcode() == MipsISD::Hi ||
         Addr.getOperand(0).getOpcode() == ISD::LOAD) &&
        Addr.getOperand(1).getOpcode() == MipsISD::Lo) {
      SDValue LoVal = Addr.getOperand(1);
      if (isa<ConstantPoolSDNode>(LoVal.getOperand(0)) || 
          isa<GlobalAddressSDNode>(LoVal.getOperand(0))) {
        Base = Addr.getOperand(0);
        Offset = LoVal.getOperand(0);
        return true;
      }
    }
  }

  Base   = Addr;
  Offset = CurDAG->getTargetConstant(0, ValTy);
  return true;
}

/// Select instructions not customized! Used for
/// expanded, promoted and normal instructions
SDNode* MipsDAGToDAGISel::Select(SDNode *Node) {
  unsigned Opcode = Node->getOpcode();
  DebugLoc dl = Node->getDebugLoc();

  // Dump information about the Node being selected
  DEBUG(errs() << "Selecting: "; Node->dump(CurDAG); errs() << "\n");

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    return NULL;
  }

  ///
  // Instruction Selection not handled by the auto-generated
  // tablegen selection should be handled here.
  ///
  switch(Opcode) {
    default: break;

    case ISD::SUBE:
    case ISD::ADDE: {
      SDValue InFlag = Node->getOperand(2), CmpLHS;
      unsigned Opc = InFlag.getOpcode(); (void)Opc;
      assert(((Opc == ISD::ADDC || Opc == ISD::ADDE) ||
              (Opc == ISD::SUBC || Opc == ISD::SUBE)) &&
             "(ADD|SUB)E flag operand must come from (ADD|SUB)C/E insn");

      unsigned MOp;
      if (Opcode == ISD::ADDE) {
        CmpLHS = InFlag.getValue(0);
        MOp = Mips::ADDu;
      } else {
        CmpLHS = InFlag.getOperand(0);
        MOp = Mips::SUBu;
      }

      SDValue Ops[] = { CmpLHS, InFlag.getOperand(1) };

      SDValue LHS = Node->getOperand(0);
      SDValue RHS = Node->getOperand(1);

      EVT VT = LHS.getValueType();
      SDNode *Carry = CurDAG->getMachineNode(Mips::SLTu, dl, VT, Ops, 2);
      SDNode *AddCarry = CurDAG->getMachineNode(Mips::ADDu, dl, VT,
                                                SDValue(Carry,0), RHS);

      return CurDAG->SelectNodeTo(Node, MOp, VT, MVT::Glue,
                                  LHS, SDValue(AddCarry,0));
    }

    /// Mul with two results
    case ISD::SMUL_LOHI:
    case ISD::UMUL_LOHI: {
      assert(Node->getValueType(0) != MVT::i64 &&
             "64-bit multiplication with two results not handled.");
      SDValue Op1 = Node->getOperand(0);
      SDValue Op2 = Node->getOperand(1);

      unsigned Op;
      Op = (Opcode == ISD::UMUL_LOHI ? Mips::MULTu : Mips::MULT);

      SDNode *Mul = CurDAG->getMachineNode(Op, dl, MVT::Glue, Op1, Op2);

      SDValue InFlag = SDValue(Mul, 0);
      SDNode *Lo = CurDAG->getMachineNode(Mips::MFLO, dl, MVT::i32,
                                          MVT::Glue, InFlag);
      InFlag = SDValue(Lo,1);
      SDNode *Hi = CurDAG->getMachineNode(Mips::MFHI, dl, MVT::i32, InFlag);

      if (!SDValue(Node, 0).use_empty())
        ReplaceUses(SDValue(Node, 0), SDValue(Lo,0));

      if (!SDValue(Node, 1).use_empty())
        ReplaceUses(SDValue(Node, 1), SDValue(Hi,0));

      return NULL;
    }

    /// Special Muls
    case ISD::MUL:
      // Mips32 has a 32-bit three operand mul instruction.
      if (Subtarget.hasMips32() && Node->getValueType(0) == MVT::i32)
        break;
    case ISD::MULHS:
    case ISD::MULHU: {
      assert((Opcode == ISD::MUL || Node->getValueType(0) != MVT::i64) &&
             "64-bit MULH* not handled.");
      EVT Ty = Node->getValueType(0);
      SDValue MulOp1 = Node->getOperand(0);
      SDValue MulOp2 = Node->getOperand(1);

      unsigned MulOp  = (Opcode == ISD::MULHU ?
                         Mips::MULTu :
                         (Ty == MVT::i32 ? Mips::MULT : Mips::DMULT));
      SDNode *MulNode = CurDAG->getMachineNode(MulOp, dl,
                                               MVT::Glue, MulOp1, MulOp2);

      SDValue InFlag = SDValue(MulNode, 0);

      if (Opcode == ISD::MUL) {
        unsigned Opc = (Ty == MVT::i32 ? Mips::MFLO : Mips::MFLO64);
        return CurDAG->getMachineNode(Opc, dl, Ty, InFlag);
      }
      else
        return CurDAG->getMachineNode(Mips::MFHI, dl, MVT::i32, InFlag);
    }

    // Get target GOT address.
    case ISD::GLOBAL_OFFSET_TABLE:
      return getGlobalBaseReg();

    case ISD::ConstantFP: {
      ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(Node);
      if (Node->getValueType(0) == MVT::f64 && CN->isExactlyValue(+0.0)) {
        SDValue Zero = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                        Mips::ZERO, MVT::i32);
        return CurDAG->getMachineNode(Mips::BuildPairF64, dl, MVT::f64, Zero,
                                      Zero);
      }
      break;
    }

    case MipsISD::ThreadPointer: {
      unsigned SrcReg = Mips::HWR29;
      unsigned DestReg = Mips::V1;
      SDNode *Rdhwr = CurDAG->getMachineNode(Mips::RDHWR, Node->getDebugLoc(),
          Node->getValueType(0), CurDAG->getRegister(SrcReg, MVT::i32));
      SDValue Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, DestReg,
          SDValue(Rdhwr, 0));
      SDValue ResNode = CurDAG->getCopyFromReg(Chain, dl, DestReg, MVT::i32);
      ReplaceUses(SDValue(Node, 0), ResNode);
      return ResNode.getNode();
    }
  }

  // Select the default instruction
  SDNode *ResNode = SelectCode(Node);

  DEBUG(errs() << "=> ");
  if (ResNode == NULL || ResNode == Node)
    DEBUG(Node->dump(CurDAG));
  else
    DEBUG(ResNode->dump(CurDAG));
  DEBUG(errs() << "\n");
  return ResNode;
}

bool MipsDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op, char ConstraintCode,
                             std::vector<SDValue> &OutOps) {
  assert(ConstraintCode == 'm' && "unexpected asm memory constraint");
  OutOps.push_back(Op);
  return false;
}

/// createMipsISelDag - This pass converts a legalized DAG into a
/// MIPS-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createMipsISelDag(MipsTargetMachine &TM) {
  return new MipsDAGToDAGISel(TM);
}

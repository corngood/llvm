//===-- MachineInstr.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods common to all machine instructions.
//
// FIXME: Now that MachineInstrs have parent pointers, they should always
// print themselves using their MachineFunction's TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Value.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Support/LeakDetector.h"
#include <iostream>

using namespace llvm;

// Global variable holding an array of descriptors for machine instructions.
// The actual object needs to be created separately for each target machine.
// This variable is initialized and reset by class TargetInstrInfo.
//
// FIXME: This should be a property of the target so that more than one target
// at a time can be active...
//
namespace llvm {
  extern const TargetInstrDescriptor *TargetInstrDescriptors;
}

/// MachineInstr ctor - This constructor only does a _reserve_ of the operands,
/// not a resize for them.  It is expected that if you use this that you call
/// add* methods below to fill up the operands, instead of the Set methods.
/// Eventually, the "resizing" ctors will be phased out.
///
MachineInstr::MachineInstr(short opcode, unsigned numOperands)
  : Opcode(opcode), parent(0) {
  Operands.reserve(numOperands);
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
}

/// MachineInstr ctor - Work exactly the same as the ctor above, except that the
/// MachineInstr is created and added to the end of the specified basic block.
///
MachineInstr::MachineInstr(MachineBasicBlock *MBB, short opcode,
                           unsigned numOperands)
  : Opcode(opcode), parent(0) {
  assert(MBB && "Cannot use inserting ctor with null basic block!");
  Operands.reserve(numOperands);
  // Make sure that we get added to a machine basicblock
  LeakDetector::addGarbageObject(this);
  MBB->push_back(this);  // Add instruction to end of basic block!
}

/// MachineInstr ctor - Copies MachineInstr arg exactly
///
MachineInstr::MachineInstr(const MachineInstr &MI) {
  Opcode = MI.getOpcode();
  Operands.reserve(MI.getNumOperands());

  // Add operands
  for (unsigned i = 0; i != MI.getNumOperands(); ++i)
    Operands.push_back(MI.getOperand(i));

  // Set parent, next, and prev to null
  parent = 0;
  prev = 0;
  next = 0;
}


MachineInstr::~MachineInstr() {
  LeakDetector::removeGarbageObject(this);
}

/// removeFromParent - This method unlinks 'this' from the containing basic
/// block, and returns it, but does not delete it.
MachineInstr *MachineInstr::removeFromParent() {
  assert(getParent() && "Not embedded in a basic block!");
  getParent()->remove(this);
  return this;
}


/// OperandComplete - Return true if it's illegal to add a new operand
///
bool MachineInstr::OperandsComplete() const {
  int NumOperands = TargetInstrDescriptors[Opcode].numOperands;
  if (NumOperands >= 0 && getNumOperands() >= (unsigned)NumOperands)
    return true;  // Broken: we have all the operands of this instruction!
  return false;
}

void MachineInstr::dump() const {
  std::cerr << "  " << *this;
}

static inline void OutputReg(std::ostream &os, unsigned RegNo,
                             const MRegisterInfo *MRI = 0) {
  if (!RegNo || MRegisterInfo::isPhysicalRegister(RegNo)) {
    if (MRI)
      os << "%" << MRI->get(RegNo).Name;
    else
      os << "%mreg(" << RegNo << ")";
  } else
    os << "%reg" << RegNo;
}

static void print(const MachineOperand &MO, std::ostream &OS,
                  const TargetMachine *TM) {
  const MRegisterInfo *MRI = 0;

  if (TM) MRI = TM->getRegisterInfo();

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    OutputReg(OS, MO.getReg(), MRI);
    break;
  case MachineOperand::MO_Immediate:
    OS << (long)MO.getImmedValue();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    OS << "mbb<"
       << ((Value*)MO.getMachineBasicBlock()->getBasicBlock())->getName()
       << "," << (void*)MO.getMachineBasicBlock() << ">";
    break;
  case MachineOperand::MO_FrameIndex:
    OS << "<fi#" << MO.getFrameIndex() << ">";
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    OS << "<cp#" << MO.getConstantPoolIndex() << ">";
    break;
  case MachineOperand::MO_JumpTableIndex:
    OS << "<jt#" << MO.getJumpTableIndex() << ">";
    break;
  case MachineOperand::MO_GlobalAddress:
    OS << "<ga:" << ((Value*)MO.getGlobal())->getName();
    if (MO.getOffset()) OS << "+" << MO.getOffset();
    OS << ">";
    break;
  case MachineOperand::MO_ExternalSymbol:
    OS << "<es:" << MO.getSymbolName();
    if (MO.getOffset()) OS << "+" << MO.getOffset();
    OS << ">";
    break;
  default:
    assert(0 && "Unrecognized operand type");
  }
}

void MachineInstr::print(std::ostream &OS, const TargetMachine *TM) const {
  unsigned StartOp = 0;

   // Specialize printing if op#0 is definition
  if (getNumOperands() && getOperand(0).isDef() && !getOperand(0).isUse()) {
    ::print(getOperand(0), OS, TM);
    OS << " = ";
    ++StartOp;   // Don't print this operand again!
  }

  // Must check if Target machine is not null because machine BB could not
  // be attached to a Machine function yet
  if (TM)
    OS << TM->getInstrInfo()->getName(getOpcode());

  for (unsigned i = StartOp, e = getNumOperands(); i != e; ++i) {
    const MachineOperand& mop = getOperand(i);
    if (i != StartOp)
      OS << ",";
    OS << " ";
    ::print(mop, OS, TM);

    if (mop.isDef())
      if (mop.isUse())
        OS << "<def&use>";
      else
        OS << "<def>";
  }

  OS << "\n";
}

std::ostream &llvm::operator<<(std::ostream &os, const MachineInstr &MI) {
  // If the instruction is embedded into a basic block, we can find the target
  // info for the instruction.
  if (const MachineBasicBlock *MBB = MI.getParent()) {
    const MachineFunction *MF = MBB->getParent();
    if (MF)
      MI.print(os, &MF->getTarget());
    else
      MI.print(os, 0);
    return os;
  }

  // Otherwise, print it out in the "raw" format without symbolic register names
  // and such.
  os << TargetInstrDescriptors[MI.getOpcode()].Name;

  for (unsigned i = 0, N = MI.getNumOperands(); i < N; i++) {
    os << "\t" << MI.getOperand(i);
    if (MI.getOperand(i).isDef())
      if (MI.getOperand(i).isUse())
        os << "<d&u>";
      else
        os << "<d>";
  }

  return os << "\n";
}

std::ostream &llvm::operator<<(std::ostream &OS, const MachineOperand &MO) {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    OutputReg(OS, MO.getReg());
    break;
  case MachineOperand::MO_Immediate:
    OS << (long)MO.getImmedValue();
    break;
  case MachineOperand::MO_MachineBasicBlock:
    OS << "<mbb:"
       << ((Value*)MO.getMachineBasicBlock()->getBasicBlock())->getName()
       << "@" << (void*)MO.getMachineBasicBlock() << ">";
    break;
  case MachineOperand::MO_FrameIndex:
    OS << "<fi#" << MO.getFrameIndex() << ">";
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    OS << "<cp#" << MO.getConstantPoolIndex() << ">";
    break;
  case MachineOperand::MO_JumpTableIndex:
    OS << "<jt#" << MO.getJumpTableIndex() << ">";
    break;
  case MachineOperand::MO_GlobalAddress:
    OS << "<ga:" << ((Value*)MO.getGlobal())->getName() << ">";
    break;
  case MachineOperand::MO_ExternalSymbol:
    OS << "<es:" << MO.getSymbolName() << ">";
    break;
  default:
    assert(0 && "Unrecognized operand type");
    break;
  }

  return OS;
}

//===-- TargetInstrInfoImpl.cpp - Target Instruction Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfoImpl class, it just provides default
// implementations of various methods.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
using namespace llvm;

// commuteInstruction - The default implementation of this method just exchanges
// operand 1 and 2.
MachineInstr *TargetInstrInfoImpl::commuteInstruction(MachineInstr *MI) const {
  assert(MI->getOperand(1).isRegister() && MI->getOperand(2).isRegister() &&
         "This only knows how to commute register operands so far");
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  MachineOperand &MO = MI->getOperand(0);
  bool UpdateReg0 = MO.isReg() && MO.getReg() == Reg1;
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  if (UpdateReg0) {
    // Must be two address instruction!
    assert(MI->getDesc().getOperandConstraint(0, TOI::TIED_TO) &&
           "Expecting a two-address instruction!");
    Reg2IsKill = false;
    MI->getOperand(0).setReg(Reg2);
  }
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  MI->getOperand(2).setIsKill(Reg1IsKill);
  MI->getOperand(1).setIsKill(Reg2IsKill);
  return MI;
}

bool TargetInstrInfoImpl::PredicateInstruction(MachineInstr *MI,
                                               const std::vector<MachineOperand> &Pred) const {
  bool MadeChange = false;
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isPredicable())
    return false;
  
  for (unsigned j = 0, i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (TID.OpInfo[i].isPredicate()) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) {
        MO.setReg(Pred[j].getReg());
        MadeChange = true;
      } else if (MO.isImm()) {
        MO.setImm(Pred[j].getImm());
        MadeChange = true;
      } else if (MO.isMBB()) {
        MO.setMBB(Pred[j].getMBB());
        MadeChange = true;
      }
      ++j;
    }
  }
  return MadeChange;
}

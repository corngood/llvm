//===- ARMInstrInfo.cpp - ARM Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "ARMGenInstrInfo.inc"
using namespace llvm;

ARMInstrInfo::ARMInstrInfo()
  : TargetInstrInfo(ARMInsts, sizeof(ARMInsts)/sizeof(ARMInsts[0])) {
}

const TargetRegisterClass *ARMInstrInfo::getPointerRegClass() const {
  return &ARM::IntRegsRegClass;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool ARMInstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg, unsigned &DstReg) const {
  MachineOpCode oc = MI.getOpcode();
  switch (oc) {
  case ARM::MOV: {
    assert(MI.getNumOperands() == 4 &&
	   MI.getOperand(0).isRegister() &&
	   "Invalid ARM MOV instruction");
    const MachineOperand   &Arg = MI.getOperand(1);
    const MachineOperand &Shift = MI.getOperand(2);
    if (Arg.isRegister() && Shift.isImmediate() && Shift.getImmedValue() == 0) {
      SrcReg = MI.getOperand(1).getReg();
      DstReg = MI.getOperand(0).getReg();
      return true;
    }
  }
  }
  return false;
}

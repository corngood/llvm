//===- SparcV8InstrInfo.cpp - SparcV8 Instruction Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SparcV8 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SparcV8InstrInfo.h"
#include "SparcV8.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "SparcV8GenInstrInfo.inc"
using namespace llvm;

SparcV8InstrInfo::SparcV8InstrInfo()
  : TargetInstrInfo(SparcV8Insts, sizeof(SparcV8Insts)/sizeof(SparcV8Insts[0])){
}

static bool isZeroImmed (const MachineOperand &op) {
  return (op.isImmediate() && op.getImmedValue() == 0);
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool SparcV8InstrInfo::isMoveInstr(const MachineInstr &MI,
                                   unsigned &SrcReg, unsigned &DstReg) const {
  // We look for 3 kinds of patterns here:
  // or with G0 or 0
  // add with G0 or 0
  // fmovs or FpMOVD (pseudo double move).
  if (MI.getOpcode() == V8::ORrr || MI.getOpcode() == V8::ADDrr) {
    if (MI.getOperand(1).getReg() == V8::G0) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
      return true;
    } else if (MI.getOperand (2).getReg() == V8::G0) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  } else if (MI.getOpcode() == V8::ORri || MI.getOpcode() == V8::ADDri) {
    if (isZeroImmed(MI.getOperand(2)) && MI.getOperand(1).isRegister()) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  } else if (MI.getOpcode() == V8::FMOVS || MI.getOpcode() == V8::FpMOVD) {
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}

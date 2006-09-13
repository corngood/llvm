//===- ARMRegisterInfo.cpp - ARM Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
using namespace llvm;

ARMRegisterInfo::ARMRegisterInfo()
  : ARMGenRegisterInfo(ARM::ADJCALLSTACKDOWN, ARM::ADJCALLSTACKUP) {
}

void ARMRegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, int FI,
                    const TargetRegisterClass *RC) const {
  assert (RC == ARM::IntRegsRegisterClass);
  BuildMI(MBB, I, ARM::str, 3).addReg(SrcReg).addImm(0).addFrameIndex(FI);
}

void ARMRegisterInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  assert (RC == ARM::IntRegsRegisterClass);
  BuildMI(MBB, I, ARM::ldr, 2, DestReg).addImm(0).addFrameIndex(FI);
}

void ARMRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I,
                                     unsigned DestReg, unsigned SrcReg,
                                     const TargetRegisterClass *RC) const {
  assert (RC == ARM::IntRegsRegisterClass);
  BuildMI(MBB, I, ARM::MOV, 3, DestReg).addReg(SrcReg).addImm(0)
	  .addImm(ARMShift::LSL);
}

MachineInstr *ARMRegisterInfo::foldMemoryOperand(MachineInstr* MI,
                                                   unsigned OpNum,
                                                   int FI) const {
  return NULL;
}

const unsigned* ARMRegisterInfo::getCalleeSaveRegs() const {
  static const unsigned CalleeSaveRegs[] = {
    ARM::R4,  ARM::R5, ARM::R6,  ARM::R7,
    ARM::R8,  ARM::R9, ARM::R10, ARM::R11,
    ARM::R14, 0
  };
  return CalleeSaveRegs;
}

const TargetRegisterClass* const *
ARMRegisterInfo::getCalleeSaveRegClasses() const {
  static const TargetRegisterClass * const CalleeSaveRegClasses[] = {
    &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass,
    &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass, &ARM::IntRegsRegClass,
    &ARM::IntRegsRegClass, 0
  };
  return CalleeSaveRegClasses;
}

void ARMRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MBB.erase(I);
}

void
ARMRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();

  assert (MI.getOpcode() == ARM::ldr ||
	  MI.getOpcode() == ARM::str ||
	  MI.getOpcode() == ARM::lea_addri);

  unsigned FrameIdx = 2;
  unsigned OffIdx = 1;

  int FrameIndex = MI.getOperand(FrameIdx).getFrameIndex();

  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);
  assert (MI.getOperand(OffIdx).getImmedValue() == 0);

  unsigned StackSize = MF.getFrameInfo()->getStackSize();

  Offset += StackSize;

  assert (Offset >= 0);
  if (Offset < 4096) {
    // Replace the FrameIndex with r13
    MI.getOperand(FrameIdx).ChangeToRegister(ARM::R13, false);
    // Replace the ldr offset with Offset
    MI.getOperand(OffIdx).ChangeToImmediate(Offset);
  } else {
    // Insert a set of r12 with the full address
    // r12 = r13 + offset
    MachineBasicBlock *MBB2 = MI.getParent();
    BuildMI(*MBB2, II, ARM::ADD, 4, ARM::R12).addReg(ARM::R13).addImm(Offset)
	    .addImm(0).addImm(ARMShift::LSL);

    // Replace the FrameIndex with r12
    MI.getOperand(FrameIdx).ChangeToRegister(ARM::R12, false);
  }
}

void ARMRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {}

void ARMRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  int           NumBytes = (int) MFI->getStackSize();

  if (MFI->hasCalls()) {
    // We reserve argument space for call sites in the function immediately on
    // entry to the current function.  This eliminates the need for add/sub
    // brackets around call sites.
    NumBytes += MFI->getMaxCallFrameSize();
  }

  MFI->setStackSize(NumBytes);

  //sub sp, sp, #NumBytes
  BuildMI(MBB, MBBI, ARM::SUB, 4, ARM::R13).addReg(ARM::R13).addImm(NumBytes)
	  .addImm(0).addImm(ARMShift::LSL);
}

void ARMRegisterInfo::emitEpilogue(MachineFunction &MF,
				   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == ARM::bx &&
         "Can only insert epilog into returning blocks");

  MachineFrameInfo *MFI = MF.getFrameInfo();
  int          NumBytes = (int) MFI->getStackSize();

  //add sp, sp, #NumBytes
  BuildMI(MBB, MBBI, ARM::ADD, 4, ARM::R13).addReg(ARM::R13).addImm(NumBytes)
	  .addImm(0).addImm(ARMShift::LSL);
}

unsigned ARMRegisterInfo::getRARegister() const {
  return ARM::R14;
}

unsigned ARMRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return ARM::R13;
}

#include "ARMGenRegisterInfo.inc"


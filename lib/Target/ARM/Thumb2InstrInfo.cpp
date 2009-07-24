//===- Thumb2InstrInfo.cpp - Thumb-2 Instruction Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "Thumb2InstrInfo.h"

using namespace llvm;

Thumb2InstrInfo::Thumb2InstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

unsigned Thumb2InstrInfo::getUnindexedOpcode(unsigned Opc) const {
  // FIXME
  return 0;
}

unsigned Thumb2InstrInfo::getOpcode(ARMII::Op Op) const {
  switch (Op) {
  case ARMII::ADDri: return ARM::t2ADDri;
  case ARMII::ADDrs: return ARM::t2ADDrs;
  case ARMII::ADDrr: return ARM::t2ADDrr;
  case ARMII::B: return ARM::t2B;
  case ARMII::Bcc: return ARM::t2Bcc;
  case ARMII::BR_JTr: return ARM::t2BR_JTr;
  case ARMII::BR_JTm: return ARM::t2BR_JTm;
  case ARMII::BR_JTadd: return ARM::t2BR_JTadd;
  case ARMII::BX_RET: return ARM::tBX_RET;
  case ARMII::FCPYS: return ARM::FCPYS;
  case ARMII::FCPYD: return ARM::FCPYD;
  case ARMII::FLDD: return ARM::FLDD;
  case ARMII::FLDS: return ARM::FLDS;
  case ARMII::FSTD: return ARM::FSTD;
  case ARMII::FSTS: return ARM::FSTS;
  case ARMII::LDRrr: return ARM::t2LDRs;
  case ARMII::LDRri: return ARM::t2LDRi12;
  case ARMII::MOVr: return ARM::t2MOVr;
  case ARMII::STRrr: return ARM::t2STRs;
  case ARMII::STRri: return ARM::t2STRi12;
  case ARMII::SUBri: return ARM::t2SUBri;
  case ARMII::SUBrs: return ARM::t2SUBrs;
  case ARMII::SUBrr: return ARM::t2SUBrr;
  case ARMII::VMOVD: return ARM::VMOVD;
  case ARMII::VMOVQ: return ARM::VMOVQ;
  default:
    break;
  }

  return 0;
}

bool
Thumb2InstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;

  // FIXME
  switch (MBB.back().getOpcode()) {
  case ARM::t2LDM_RET:
  case ARM::t2B:        // Uncond branch.
  case ARM::t2BR_JTr:   // Jumptable branch.
  case ARM::t2BR_JTm:   // Jumptable branch through mem.
  case ARM::t2BR_JTadd: // Jumptable branch add to pc.
    return true;
  case ARM::tBX_RET:
  case ARM::tBX_RET_vararg:
  case ARM::tPOP_RET:
  case ARM::tB:
  case ARM::tBR_JTr:
    return true;
  default:
    break;
  }

  return false;
}

bool
Thumb2InstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I,
                              unsigned DestReg, unsigned SrcReg,
                              const TargetRegisterClass *DestRC,
                              const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if ((DestRC == ARM::GPRRegisterClass &&
       SrcRC == ARM::tGPRRegisterClass) ||
      (DestRC == ARM::tGPRRegisterClass &&
       SrcRC == ARM::GPRRegisterClass)) {
    AddDefaultCC(AddDefaultPred(BuildMI(MBB, I, DL, get(getOpcode(ARMII::MOVr)),
                                        DestReg).addReg(SrcReg)));
    return true;
  }

  return ARMBaseInstrInfo::copyRegToReg(MBB, I, DestReg, SrcReg, DestRC, SrcRC);
}

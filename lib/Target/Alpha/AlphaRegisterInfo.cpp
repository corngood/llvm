//===- AlphaRegisterInfo.cpp - Alpha Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reginfo"
#include "Alpha.h"
#include "AlphaRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>
#include <iostream>
using namespace llvm;

namespace llvm {
  extern cl::opt<bool> EnableAlphaLSMark;
}

//These describe LDAx
static const int IMM_LOW  = -32768;
static const int IMM_HIGH = 32767;
static const int IMM_MULT = 65536;

static long getUpper16(long l)
{
  long y = l / IMM_MULT;
  if (l % IMM_MULT > IMM_HIGH)
    ++y;
  return y;
}

static long getLower16(long l)
{
  long h = getUpper16(l);
  return l - h * IMM_MULT;
}

static int getUID()
{
  static int id = 0;
  return ++id;
}

AlphaRegisterInfo::AlphaRegisterInfo()
  : AlphaGenRegisterInfo(Alpha::ADJUSTSTACKDOWN, Alpha::ADJUSTSTACKUP)
{
}

static const TargetRegisterClass *getClass(unsigned SrcReg) {
  if (Alpha::FPRCRegisterClass->contains(SrcReg))
    return Alpha::FPRCRegisterClass;
  assert(Alpha::GPRCRegisterClass->contains(SrcReg) && "Reg not FPR or GPR");
  return Alpha::GPRCRegisterClass;
}

void
AlphaRegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned SrcReg, int FrameIdx,
                                       const TargetRegisterClass *RC) const {
  //std::cerr << "Trying to store " << getPrettyName(SrcReg) << " to " << FrameIdx << "\n";
  //BuildMI(MBB, MI, Alpha::WTF, 0).addReg(SrcReg);
  if (EnableAlphaLSMark)
    BuildMI(MBB, MI, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(1)
      .addImm(getUID());
  if (getClass(SrcReg) == Alpha::FPRCRegisterClass)
    BuildMI(MBB, MI, Alpha::STT, 3).addReg(SrcReg).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else if (getClass(SrcReg) == Alpha::GPRCRegisterClass)
    BuildMI(MBB, MI, Alpha::STQ, 3).addReg(SrcReg).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else
    abort();
}

void
AlphaRegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC) const {
  //std::cerr << "Trying to load " << getPrettyName(DestReg) << " to " << FrameIdx << "\n";
  if (EnableAlphaLSMark)
    BuildMI(MBB, MI, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(2)
      .addImm(getUID());
  if (getClass(DestReg) == Alpha::FPRCRegisterClass)
    BuildMI(MBB, MI, Alpha::LDT, 2, DestReg).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else if (getClass(DestReg) == Alpha::GPRCRegisterClass)
    BuildMI(MBB, MI, Alpha::LDQ, 2, DestReg).addFrameIndex(FrameIdx).addReg(Alpha::F31);
  else
    abort();
}

unsigned 
AlphaRegisterInfo::isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const
{
  switch (MI->getOpcode()) {
  case Alpha::LDL:
  case Alpha::LDQ:
  case Alpha::LDBU:
  case Alpha::LDWU:
  case Alpha::LDS:
  case Alpha::LDT:
    if (MI->getOperand(1).isFrameIndex()) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

void AlphaRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI,
                                     unsigned DestReg, unsigned SrcReg,
                                     const TargetRegisterClass *RC) const {
  //  std::cerr << "copyRegToReg " << DestReg << " <- " << SrcReg << "\n";
  if (RC == Alpha::GPRCRegisterClass) {
    BuildMI(MBB, MI, Alpha::BIS, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (RC == Alpha::FPRCRegisterClass) {
    BuildMI(MBB, MI, Alpha::CPYS, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else {
    std::cerr << "Attempt to copy register that is not GPR or FPR";
     abort();
  }
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(MachineFunction &MF) {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->hasVarSizedObjects();
}

void AlphaRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have a frame pointer, turn the adjcallstackup instruction into a
    // 'sub ESP, <amt>' and the adjcallstackdown instruction into 'add ESP,
    // <amt>'
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      MachineInstr *New;
      if (Old->getOpcode() == Alpha::ADJUSTSTACKDOWN) {
         New=BuildMI(Alpha::LDA, 2, Alpha::R30)
          .addImm(-Amount).addReg(Alpha::R30);
      } else {
         assert(Old->getOpcode() == Alpha::ADJUSTSTACKUP);
         New=BuildMI(Alpha::LDA, 2, Alpha::R30)
          .addImm(Amount).addReg(Alpha::R30);
      }

      // Replace the pseudo instruction with a new instruction...
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

//Alpha has a slightly funny stack:
//Args
//<- incoming SP
//fixed locals (and spills, callee saved, etc)
//<- FP
//variable locals
//<- SP

void
AlphaRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  bool FP = hasFP(MF);

  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // Add the base register of R30 (SP) or R15 (FP).
  MI.SetMachineOperandReg(i + 1, FP ? Alpha::R15 : Alpha::R30);

  // Now add the frame object offset to the offset from the virtual frame index.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  DEBUG(std::cerr << "FI: " << FrameIndex << " Offset: " << Offset << "\n");

  Offset += MF.getFrameInfo()->getStackSize();

  DEBUG(std::cerr << "Corrected Offset " << Offset <<
        " for stack size: " << MF.getFrameInfo()->getStackSize() << "\n");

  if (Offset > IMM_HIGH || Offset < IMM_LOW) {
    //so in this case, we need to use a temporary register, and move the original
    //inst off the SP/FP
    //fix up the old:
    MI.SetMachineOperandReg(i + 1, Alpha::R28);
    MI.SetMachineOperandConst(i, MachineOperand::MO_SignExtendedImmed,
                              getLower16(Offset));
    //insert the new
    MachineInstr* nMI=BuildMI(Alpha::LDAH, 2, Alpha::R28)
      .addImm(getUpper16(Offset)).addReg(FP ? Alpha::R15 : Alpha::R30);
    MBB.insert(II, nMI);
  } else {
    MI.SetMachineOperandConst(i, MachineOperand::MO_SignExtendedImmed, Offset);
  }
}


void AlphaRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool FP = hasFP(MF);

  static int curgpdist = 0;

  //handle GOP offset
  BuildMI(MBB, MBBI, Alpha::LDAHg, 3, Alpha::R29)
    .addGlobalAddress(const_cast<Function*>(MF.getFunction()))
    .addReg(Alpha::R27).addImm(++curgpdist);
  BuildMI(MBB, MBBI, Alpha::LDAg, 3, Alpha::R29)
    .addGlobalAddress(const_cast<Function*>(MF.getFunction()))
    .addReg(Alpha::R29).addImm(curgpdist);

  //evil const_cast until MO stuff setup to handle const
  BuildMI(MBB, MBBI, Alpha::ALTENT, 1).addGlobalAddress(const_cast<Function*>(MF.getFunction()), true);

  // Get the number of bytes to allocate from the FrameInfo
  long NumBytes = MFI->getStackSize();

  if (MFI->hasCalls() && !FP) {
    // We reserve argument space for call sites in the function immediately on
    // entry to the current function.  This eliminates the need for add/sub
    // brackets around call sites.
    //If there is a frame pointer, then we don't do this
    NumBytes += MFI->getMaxCallFrameSize();
    DEBUG(std::cerr << "Added " << MFI->getMaxCallFrameSize()
          << " to the stack due to calls\n");
  }

  if (FP)
    NumBytes += 8; //reserve space for the old FP

  // Do we need to allocate space on the stack?
  if (NumBytes == 0) return;

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);

  // adjust stack pointer: r30 -= numbytes
  NumBytes = -NumBytes;
  if (NumBytes >= IMM_LOW) {
    BuildMI(MBB, MBBI, Alpha::LDA, 2, Alpha::R30).addImm(NumBytes)
      .addReg(Alpha::R30);
  } else if (getUpper16(NumBytes) >= IMM_LOW) {
    BuildMI(MBB, MBBI, Alpha::LDAH, 2, Alpha::R30).addImm(getUpper16(NumBytes))
      .addReg(Alpha::R30);
    BuildMI(MBB, MBBI, Alpha::LDA, 2, Alpha::R30).addImm(getLower16(NumBytes))
      .addReg(Alpha::R30);
  } else {
    std::cerr << "Too big a stack frame at " << NumBytes << "\n";
    abort();
  }

  //now if we need to, save the old FP and set the new
  if (FP)
  {
    if (EnableAlphaLSMark)
      BuildMI(MBB, MBBI, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(1)
        .addImm(getUID());
    BuildMI(MBB, MBBI, Alpha::STQ, 3).addReg(Alpha::R15).addImm(0).addReg(Alpha::R30);
    //this must be the last instr in the prolog
    BuildMI(MBB, MBBI, Alpha::BIS, 2, Alpha::R15).addReg(Alpha::R30).addReg(Alpha::R30);
  }

}

void AlphaRegisterInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(((MBBI->getOpcode() == Alpha::RET) || (MBBI->getOpcode() == Alpha::RETDAG))
         && "Can only insert epilog into returning blocks");

  bool FP = hasFP(MF);

  // Get the number of bytes allocated from the FrameInfo...
  long NumBytes = MFI->getStackSize();

  //now if we need to, restore the old FP
  if (FP)
  {
    //copy the FP into the SP (discards allocas)
    BuildMI(MBB, MBBI, Alpha::BIS, 2, Alpha::R30).addReg(Alpha::R15)
      .addReg(Alpha::R15);
    //restore the FP
    if (EnableAlphaLSMark)
      BuildMI(MBB, MBBI, Alpha::MEMLABEL, 4).addImm(4).addImm(0).addImm(2)
        .addImm(getUID());
    BuildMI(MBB, MBBI, Alpha::LDQ, 2, Alpha::R15).addImm(0).addReg(Alpha::R15);
  }

   if (NumBytes != 0)
     {
       if (NumBytes <= IMM_HIGH) {
         BuildMI(MBB, MBBI, Alpha::LDA, 2, Alpha::R30).addImm(NumBytes)
           .addReg(Alpha::R30);
       } else if (getUpper16(NumBytes) <= IMM_HIGH) {
         BuildMI(MBB, MBBI, Alpha::LDAH, 2, Alpha::R30)
           .addImm(getUpper16(NumBytes)).addReg(Alpha::R30);
         BuildMI(MBB, MBBI, Alpha::LDA, 2, Alpha::R30)
           .addImm(getLower16(NumBytes)).addReg(Alpha::R30);
       } else {
         std::cerr << "Too big a stack frame at " << NumBytes << "\n";
         abort();
       }
     }
}

#include "AlphaGenRegisterInfo.inc"

std::string AlphaRegisterInfo::getPrettyName(unsigned reg)
{
  std::string s(RegisterDescriptors[reg].Name);
  return s;
}

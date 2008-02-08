//===- PPCInstrInfo.cpp - PowerPC32 Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "PPCInstrBuilder.h"
#include "PPCPredicates.h"
#include "PPCGenInstrInfo.inc"
#include "PPCTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

PPCInstrInfo::PPCInstrInfo(PPCTargetMachine &tm)
  : TargetInstrInfoImpl(PPCInsts, array_lengthof(PPCInsts)), TM(tm),
    RI(*TM.getSubtargetImpl(), *this) {}

/// getPointerRegClass - Return the register class to use to hold pointers.
/// This is used for addressing modes.
const TargetRegisterClass *PPCInstrInfo::getPointerRegClass() const {
  if (TM.getSubtargetImpl()->isPPC64())
    return &PPC::G8RCRegClass;
  else
    return &PPC::GPRCRegClass;
}


bool PPCInstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg) const {
  unsigned oc = MI.getOpcode();
  if (oc == PPC::OR || oc == PPC::OR8 || oc == PPC::VOR ||
      oc == PPC::OR4To8 || oc == PPC::OR8To4) {                // or r1, r2, r2
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           MI.getOperand(2).isRegister() &&
           "invalid PPC OR instruction!");
    if (MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::ADDI) {             // addi r1, r2, 0
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(2).isImmediate() &&
           "invalid PPC ADDI instruction!");
    if (MI.getOperand(1).isRegister() && MI.getOperand(2).getImm() == 0) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::ORI) {             // ori r1, r2, 0
    assert(MI.getNumOperands() >= 3 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           MI.getOperand(2).isImmediate() &&
           "invalid PPC ORI instruction!");
    if (MI.getOperand(2).getImm() == 0) {
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
    }
  } else if (oc == PPC::FMRS || oc == PPC::FMRD ||
             oc == PPC::FMRSD) {      // fmr r1, r2
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           "invalid PPC FMR instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  } else if (oc == PPC::MCRF) {             // mcrf cr1, cr2
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isRegister() &&
           MI.getOperand(1).isRegister() &&
           "invalid PPC MCRF instruction");
    sourceReg = MI.getOperand(1).getReg();
    destReg = MI.getOperand(0).getReg();
    return true;
  }
  return false;
}

unsigned PPCInstrInfo::isLoadFromStackSlot(MachineInstr *MI, 
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::LD:
  case PPC::LWZ:
  case PPC::LFS:
  case PPC::LFD:
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned PPCInstrInfo::isStoreToStackSlot(MachineInstr *MI, 
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::STD:
  case PPC::STW:
  case PPC::STFS:
  case PPC::STFD:
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

// commuteInstruction - We can commute rlwimi instructions, but only if the
// rotate amt is zero.  We also have to munge the immediates a bit.
MachineInstr *PPCInstrInfo::commuteInstruction(MachineInstr *MI) const {
  // Normal instructions can be commuted the obvious way.
  if (MI->getOpcode() != PPC::RLWIMI)
    return TargetInstrInfoImpl::commuteInstruction(MI);
  
  // Cannot commute if it has a non-zero rotate count.
  if (MI->getOperand(3).getImm() != 0)
    return 0;
  
  // If we have a zero rotate count, we have:
  //   M = mask(MB,ME)
  //   Op0 = (Op1 & ~M) | (Op2 & M)
  // Change this to:
  //   M = mask((ME+1)&31, (MB-1)&31)
  //   Op0 = (Op2 & ~M) | (Op1 & M)

  // Swap op1/op2
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  MI->getOperand(2).setIsKill(Reg1IsKill);
  MI->getOperand(1).setIsKill(Reg2IsKill);
  
  // Swap the mask around.
  unsigned MB = MI->getOperand(4).getImm();
  unsigned ME = MI->getOperand(5).getImm();
  MI->getOperand(4).setImm((ME+1) & 31);
  MI->getOperand(5).setImm((MB-1) & 31);
  return MI;
}

void PPCInstrInfo::insertNoop(MachineBasicBlock &MBB, 
                              MachineBasicBlock::iterator MI) const {
  BuildMI(MBB, MI, get(PPC::NOP));
}


// Branch analysis.
bool PPCInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 std::vector<MachineOperand> &Cond) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastInst->getOpcode() == PPC::B) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (LastInst->getOpcode() == PPC::BCC) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(2).getMBB();
      Cond.push_back(LastInst->getOperand(0));
      Cond.push_back(LastInst->getOperand(1));
      return false;
    }
    // Otherwise, don't know what this is.
    return true;
  }
  
  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isUnpredicatedTerminator(--I))
    return true;
  
  // If the block ends with PPC::B and PPC:BCC, handle it.
  if (SecondLastInst->getOpcode() == PPC::BCC && 
      LastInst->getOpcode() == PPC::B) {
    TBB =  SecondLastInst->getOperand(2).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two PPC:Bs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == PPC::B && 
      LastInst->getOpcode() == PPC::B) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned PPCInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != PPC::B && I->getOpcode() != PPC::BCC)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != PPC::BCC)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
PPCInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const std::vector<MachineOperand> &Cond) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) && 
         "PPC branch conditions have two components!");
  
  // One-way branch.
  if (FBB == 0) {
    if (Cond.empty())   // Unconditional branch
      BuildMI(&MBB, get(PPC::B)).addMBB(TBB);
    else                // Conditional branch
      BuildMI(&MBB, get(PPC::BCC))
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
    return 1;
  }
  
  // Two-way Conditional Branch.
  BuildMI(&MBB, get(PPC::BCC))
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
  BuildMI(&MBB, get(PPC::B)).addMBB(FBB);
  return 2;
}

void PPCInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  if (DestRC != SrcRC) {
    cerr << "Not yet supported!";
    abort();
  }

  if (DestRC == PPC::GPRCRegisterClass) {
    BuildMI(MBB, MI, get(PPC::OR), DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (DestRC == PPC::G8RCRegisterClass) {
    BuildMI(MBB, MI, get(PPC::OR8), DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (DestRC == PPC::F4RCRegisterClass) {
    BuildMI(MBB, MI, get(PPC::FMRS), DestReg).addReg(SrcReg);
  } else if (DestRC == PPC::F8RCRegisterClass) {
    BuildMI(MBB, MI, get(PPC::FMRD), DestReg).addReg(SrcReg);
  } else if (DestRC == PPC::CRRCRegisterClass) {
    BuildMI(MBB, MI, get(PPC::MCRF), DestReg).addReg(SrcReg);
  } else if (DestRC == PPC::VRRCRegisterClass) {
    BuildMI(MBB, MI, get(PPC::VOR), DestReg).addReg(SrcReg).addReg(SrcReg);
  } else {
    cerr << "Attempt to copy register that is not GPR or FPR";
    abort();
  }
}

static void StoreRegToStackSlot(const TargetInstrInfo &TII,
                                unsigned SrcReg, bool isKill, int FrameIdx,
                                const TargetRegisterClass *RC,
                                SmallVectorImpl<MachineInstr*> &NewMIs) {
  if (RC == PPC::GPRCRegisterClass) {
    if (SrcReg != PPC::LR) {
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::STW))
                                .addReg(SrcReg, false, false, isKill), FrameIdx));
    } else {
      // FIXME: this spills LR immediately to memory in one step.  To do this,
      // we use R11, which we know cannot be used in the prolog/epilog.  This is
      // a hack.
      NewMIs.push_back(BuildMI(TII.get(PPC::MFLR), PPC::R11));
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::STW))
                              .addReg(PPC::R11, false, false, isKill), FrameIdx));
    }
  } else if (RC == PPC::G8RCRegisterClass) {
    if (SrcReg != PPC::LR8) {
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::STD))
                                .addReg(SrcReg, false, false, isKill), FrameIdx));
    } else {
      // FIXME: this spills LR immediately to memory in one step.  To do this,
      // we use R11, which we know cannot be used in the prolog/epilog.  This is
      // a hack.
      NewMIs.push_back(BuildMI(TII.get(PPC::MFLR8), PPC::X11));
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::STD))
                              .addReg(PPC::X11, false, false, isKill), FrameIdx));
    }
  } else if (RC == PPC::F8RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::STFD))
                                .addReg(SrcReg, false, false, isKill), FrameIdx));
  } else if (RC == PPC::F4RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::STFS))
                                .addReg(SrcReg, false, false, isKill), FrameIdx));
  } else if (RC == PPC::CRRCRegisterClass) {
    // FIXME: We use R0 here, because it isn't available for RA.
    // We need to store the CR in the low 4-bits of the saved value.  First,
    // issue a MFCR to save all of the CRBits.
    NewMIs.push_back(BuildMI(TII.get(PPC::MFCR), PPC::R0));
    
    // If the saved register wasn't CR0, shift the bits left so that they are in
    // CR0's slot.
    if (SrcReg != PPC::CR0) {
      unsigned ShiftBits = PPCRegisterInfo::getRegisterNumbering(SrcReg)*4;
      // rlwinm r0, r0, ShiftBits, 0, 31.
      NewMIs.push_back(BuildMI(TII.get(PPC::RLWINM), PPC::R0)
                       .addReg(PPC::R0).addImm(ShiftBits).addImm(0).addImm(31));
    }
    
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::STW))
                               .addReg(PPC::R0, false, false, isKill), FrameIdx));
  } else if (RC == PPC::VRRCRegisterClass) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R0 = ADDI FI#
    // STVX VAL, 0, R0
    // 
    // FIXME: We use R0 here, because it isn't available for RA.
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::ADDI), PPC::R0),
                                       FrameIdx, 0, 0));
    NewMIs.push_back(BuildMI(TII.get(PPC::STVX))
           .addReg(SrcReg, false, false, isKill).addReg(PPC::R0).addReg(PPC::R0));
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
}

void
PPCInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI,
                                     unsigned SrcReg, bool isKill, int FrameIdx,
                                     const TargetRegisterClass *RC) const {
  SmallVector<MachineInstr*, 4> NewMIs;
  StoreRegToStackSlot(*this, SrcReg, isKill, FrameIdx, RC, NewMIs);
  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);
}

void PPCInstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                     bool isKill,
                                     SmallVectorImpl<MachineOperand> &Addr,
                                     const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  if (Addr[0].isFrameIndex()) {
    StoreRegToStackSlot(*this, SrcReg, isKill, Addr[0].getIndex(), RC, NewMIs);
    return;
  }

  unsigned Opc = 0;
  if (RC == PPC::GPRCRegisterClass) {
    Opc = PPC::STW;
  } else if (RC == PPC::G8RCRegisterClass) {
    Opc = PPC::STD;
  } else if (RC == PPC::F8RCRegisterClass) {
    Opc = PPC::STFD;
  } else if (RC == PPC::F4RCRegisterClass) {
    Opc = PPC::STFS;
  } else if (RC == PPC::VRRCRegisterClass) {
    Opc = PPC::STVX;
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
  MachineInstrBuilder MIB = BuildMI(get(Opc))
    .addReg(SrcReg, false, false, isKill);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isRegister())
      MIB.addReg(MO.getReg());
    else if (MO.isImmediate())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  NewMIs.push_back(MIB);
  return;
}

static void LoadRegFromStackSlot(const TargetInstrInfo &TII,
                                 unsigned DestReg, int FrameIdx,
                                 const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) {
  if (RC == PPC::GPRCRegisterClass) {
    if (DestReg != PPC::LR) {
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::LWZ), DestReg),
                                         FrameIdx));
    } else {
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::LWZ), PPC::R11),
                                         FrameIdx));
      NewMIs.push_back(BuildMI(TII.get(PPC::MTLR)).addReg(PPC::R11));
    }
  } else if (RC == PPC::G8RCRegisterClass) {
    if (DestReg != PPC::LR8) {
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::LD), DestReg),
                                         FrameIdx));
    } else {
      NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::LD), PPC::R11),
                                         FrameIdx));
      NewMIs.push_back(BuildMI(TII.get(PPC::MTLR8)).addReg(PPC::R11));
    }
  } else if (RC == PPC::F8RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::LFD), DestReg),
                                       FrameIdx));
  } else if (RC == PPC::F4RCRegisterClass) {
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::LFS), DestReg),
                                       FrameIdx));
  } else if (RC == PPC::CRRCRegisterClass) {
    // FIXME: We use R0 here, because it isn't available for RA.
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::LWZ), PPC::R0),
                                       FrameIdx));
    
    // If the reloaded register isn't CR0, shift the bits right so that they are
    // in the right CR's slot.
    if (DestReg != PPC::CR0) {
      unsigned ShiftBits = PPCRegisterInfo::getRegisterNumbering(DestReg)*4;
      // rlwinm r11, r11, 32-ShiftBits, 0, 31.
      NewMIs.push_back(BuildMI(TII.get(PPC::RLWINM), PPC::R0)
                    .addReg(PPC::R0).addImm(32-ShiftBits).addImm(0).addImm(31));
    }
    
    NewMIs.push_back(BuildMI(TII.get(PPC::MTCRF), DestReg).addReg(PPC::R0));
  } else if (RC == PPC::VRRCRegisterClass) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R0 = ADDI FI#
    // Dest = LVX 0, R0
    // 
    // FIXME: We use R0 here, because it isn't available for RA.
    NewMIs.push_back(addFrameReference(BuildMI(TII.get(PPC::ADDI), PPC::R0),
                                       FrameIdx, 0, 0));
    NewMIs.push_back(BuildMI(TII.get(PPC::LVX),DestReg).addReg(PPC::R0)
                     .addReg(PPC::R0));
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
}

void
PPCInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned DestReg, int FrameIdx,
                                      const TargetRegisterClass *RC) const {
  SmallVector<MachineInstr*, 4> NewMIs;
  LoadRegFromStackSlot(*this, DestReg, FrameIdx, RC, NewMIs);
  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);
}

void PPCInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                      SmallVectorImpl<MachineOperand> &Addr,
                                      const TargetRegisterClass *RC,
                                  SmallVectorImpl<MachineInstr*> &NewMIs) const{
  if (Addr[0].isFrameIndex()) {
    LoadRegFromStackSlot(*this, DestReg, Addr[0].getIndex(), RC, NewMIs);
    return;
  }

  unsigned Opc = 0;
  if (RC == PPC::GPRCRegisterClass) {
    assert(DestReg != PPC::LR && "Can't handle this yet!");
    Opc = PPC::LWZ;
  } else if (RC == PPC::G8RCRegisterClass) {
    assert(DestReg != PPC::LR8 && "Can't handle this yet!");
    Opc = PPC::LD;
  } else if (RC == PPC::F8RCRegisterClass) {
    Opc = PPC::LFD;
  } else if (RC == PPC::F4RCRegisterClass) {
    Opc = PPC::LFS;
  } else if (RC == PPC::VRRCRegisterClass) {
    Opc = PPC::LVX;
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
  MachineInstrBuilder MIB = BuildMI(get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isRegister())
      MIB.addReg(MO.getReg());
    else if (MO.isImmediate())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  NewMIs.push_back(MIB);
  return;
}

/// foldMemoryOperand - PowerPC (like most RISC's) can only fold spills into
/// copy instructions, turning them into load/store instructions.
MachineInstr *PPCInstrInfo::foldMemoryOperand(MachineFunction &MF,
                                              MachineInstr *MI,
                                              SmallVectorImpl<unsigned> &Ops,
                                              int FrameIndex) const {
  if (Ops.size() != 1) return NULL;

  // Make sure this is a reg-reg copy.  Note that we can't handle MCRF, because
  // it takes more than one instruction to store it.
  unsigned Opc = MI->getOpcode();
  unsigned OpNum = Ops[0];

  MachineInstr *NewMI = NULL;
  if ((Opc == PPC::OR &&
       MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::STW)).addReg(InReg),
                                FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::LWZ), OutReg),
                                FrameIndex);
    }
  } else if ((Opc == PPC::OR8 &&
              MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::STD)).addReg(InReg),
                                FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::LD), OutReg), FrameIndex);
    }
  } else if (Opc == PPC::FMRD) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::STFD)).addReg(InReg),
                                FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::LFD), OutReg), FrameIndex);
    }
  } else if (Opc == PPC::FMRS) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::STFS)).addReg(InReg),
                                FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      NewMI = addFrameReference(BuildMI(get(PPC::LFS), OutReg), FrameIndex);
    }
  }

  if (NewMI)
    NewMI->copyKillDeadInfo(MI);
  return NewMI;
}

bool PPCInstrInfo::canFoldMemoryOperand(MachineInstr *MI,
                                        SmallVectorImpl<unsigned> &Ops) const {
  if (Ops.size() != 1) return false;

  // Make sure this is a reg-reg copy.  Note that we can't handle MCRF, because
  // it takes more than one instruction to store it.
  unsigned Opc = MI->getOpcode();

  if ((Opc == PPC::OR &&
       MI->getOperand(1).getReg() == MI->getOperand(2).getReg()))
    return true;
  else if ((Opc == PPC::OR8 &&
              MI->getOperand(1).getReg() == MI->getOperand(2).getReg()))
    return true;
  else if (Opc == PPC::FMRD || Opc == PPC::FMRS)
    return true;

  return false;
}


bool PPCInstrInfo::BlockHasNoFallThrough(MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case PPC::BLR:   // Return.
  case PPC::B:     // Uncond branch.
  case PPC::BCTR:  // Indirect branch.
    return true;
  default: return false;
  }
}

bool PPCInstrInfo::
ReverseBranchCondition(std::vector<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid PPC branch opcode!");
  // Leave the CR# the same, but invert the condition.
  Cond[0].setImm(PPC::InvertPredicate((PPC::Predicate)Cond[0].getImm()));
  return false;
}

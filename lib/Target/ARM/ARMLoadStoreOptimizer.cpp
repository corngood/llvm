//===-- ARMLoadStoreOptimizer.cpp - ARM load / store opt. pass ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that performs load / store related peephole
// optimizations. This pass should be run after register allocation.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-ldst-opt"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMRegisterInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

STATISTIC(NumLDMGened , "Number of ldm instructions generated");
STATISTIC(NumSTMGened , "Number of stm instructions generated");
STATISTIC(NumFLDMGened, "Number of fldm instructions generated");
STATISTIC(NumFSTMGened, "Number of fstm instructions generated");

namespace {
  struct VISIBILITY_HIDDEN ARMLoadStoreOpt : public MachineFunctionPass {
    const TargetInstrInfo *TII;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM load / store optimization pass";
    }

  private:
    struct MemOpQueueEntry {
      int Offset;
      unsigned Position;
      MachineBasicBlock::iterator MBBI;
      bool Merged;
      MemOpQueueEntry(int o, int p, MachineBasicBlock::iterator i)
        : Offset(o), Position(p), MBBI(i), Merged(false) {};
    };
    typedef SmallVector<MemOpQueueEntry,8> MemOpQueue;
    typedef MemOpQueue::iterator MemOpQueueIter;

    SmallVector<MachineBasicBlock::iterator, 4>
    MergeLDR_STR(MachineBasicBlock &MBB, unsigned SIndex, unsigned Base,
                 int Opcode, unsigned Size, MemOpQueue &MemOps);

    bool LoadStoreMultipleOpti(MachineBasicBlock &MBB);
    bool MergeReturnIntoLDM(MachineBasicBlock &MBB);
  };
}

/// createARMLoadStoreOptimizationPass - returns an instance of the load / store
/// optimization pass.
FunctionPass *llvm::createARMLoadStoreOptimizationPass() {
  return new ARMLoadStoreOpt();
}

static int getLoadStoreMultipleOpcode(int Opcode) {
  switch (Opcode) {
  case ARM::LDR:
    NumLDMGened++;
    return ARM::LDM;
  case ARM::STR:
    NumSTMGened++;
    return ARM::STM;
  case ARM::FLDS:
    NumFLDMGened++;
    return ARM::FLDMS;
  case ARM::FSTS:
    NumFSTMGened++;
    return ARM::FSTMS;
  case ARM::FLDD:
    NumFLDMGened++;
    return ARM::FLDMD;
  case ARM::FSTD:
    NumFSTMGened++;
    return ARM::FSTMD;
  default: abort();
  }
  return 0;
}

/// mergeOps - Create and insert a LDM or STM with Base as base register and
/// registers in Regs as the register operands that would be loaded / stored.
/// It returns true if the transformation is done. 
static bool mergeOps(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     int Offset, unsigned Base, int Opcode,
                     SmallVector<unsigned, 8> &Regs,
                     const TargetInstrInfo *TII) {
  // Only a single register to load / store. Don't bother.
  unsigned NumRegs = Regs.size();
  if (NumRegs <= 1)
    return false;

  ARM_AM::AMSubMode Mode = ARM_AM::ia;
  bool isAM4 = Opcode == ARM::LDR || Opcode == ARM::STR;
  if (isAM4 && Offset == 4)
    Mode = ARM_AM::ib;
  else if (isAM4 && Offset == -4 * (int)NumRegs + 4)
    Mode = ARM_AM::da;
  else if (isAM4 && Offset == -4 * (int)NumRegs)
    Mode = ARM_AM::db;
  else if (Offset != 0) {
    // If starting offset isn't zero, insert a MI to materialize a new base.
    // But only do so if it is cost effective, i.e. merging more than two
    // loads / stores.
    if (NumRegs <= 2)
      return false;

    unsigned NewBase;
    if (Opcode == ARM::LDR)
      // If it is a load, then just use one of the destination register to
      // use as the new base.
      NewBase = Regs[NumRegs-1];
    else {
      // FIXME: Try scavenging a register to use as a new base.
      NewBase = ARM::R12;
    }
    int BaseOpc = ARM::ADDri;
    if (Offset < 0) {
      BaseOpc = ARM::SUBri;
      Offset = - Offset;
    }
    int ImmedOffset = ARM_AM::getSOImmVal(Offset);
    if (ImmedOffset == -1)
      return false;  // Probably not worth it then.
    BuildMI(MBB, MBBI, TII->get(BaseOpc), NewBase).addReg(Base).addImm(ImmedOffset);
    Base = NewBase;
  }

  bool isDPR = Opcode == ARM::FLDD || Opcode == ARM::FSTD;
  bool isDef = Opcode == ARM::LDR || Opcode == ARM::FLDS || Opcode == ARM::FLDD;
  Opcode = getLoadStoreMultipleOpcode(Opcode);
  MachineInstrBuilder MIB = (isAM4)
    ? BuildMI(MBB, MBBI, TII->get(Opcode)).addReg(Base)
        .addImm(ARM_AM::getAM4ModeImm(Mode))
    : BuildMI(MBB, MBBI, TII->get(Opcode)).addReg(Base)
        .addImm(ARM_AM::getAM5Opc(Mode, false, isDPR ? NumRegs<<1 : NumRegs));
  for (unsigned i = 0; i != NumRegs; ++i)
    MIB = MIB.addReg(Regs[i], Opcode == isDef);

  return true;
}

SmallVector<MachineBasicBlock::iterator, 4>
ARMLoadStoreOpt::MergeLDR_STR(MachineBasicBlock &MBB,
                              unsigned SIndex, unsigned Base, int Opcode,
                              unsigned Size, MemOpQueue &MemOps) {
  bool isAM4 = Opcode == ARM::LDR || Opcode == ARM::STR;
  SmallVector<MachineBasicBlock::iterator, 4> Merges;
  int Offset = MemOps[SIndex].Offset;
  int SOffset = Offset;
  unsigned Pos = MemOps[SIndex].Position;
  MachineBasicBlock::iterator Loc = MemOps[SIndex].MBBI;
  SmallVector<unsigned, 8> Regs;
  unsigned PReg = MemOps[SIndex].MBBI->getOperand(0).getReg();
  unsigned PRegNum = ARMRegisterInfo::getRegisterNumbering(PReg);
  Regs.push_back(PReg);
  for (unsigned i = SIndex+1, e = MemOps.size(); i != e; ++i) {
    int NewOffset = MemOps[i].Offset;
    unsigned Reg = MemOps[i].MBBI->getOperand(0).getReg();
    unsigned RegNum = ARMRegisterInfo::getRegisterNumbering(Reg);
    // AM4 - register numbers in ascending order.
    // AM5 - consecutive register numbers in ascending order.
    if (NewOffset == Offset + (int)Size &&
        ((isAM4 && RegNum > PRegNum) || RegNum == PRegNum+1)) {
      Offset += Size;
      Regs.push_back(Reg);
      PRegNum = RegNum;
    } else {
      // Can't merge this in. Try merge the earlier ones first.
      if (mergeOps(MBB, ++Loc, SOffset, Base, Opcode, Regs, TII)) {
        Merges.push_back(prior(Loc));
        for (unsigned j = SIndex; j < i; ++j) {
          MBB.erase(MemOps[j].MBBI);
          MemOps[j].Merged = true;
        }
      }
      SmallVector<MachineBasicBlock::iterator, 4> Merges2 =
        MergeLDR_STR(MBB, i, Base, Opcode, Size, MemOps);
      Merges.append(Merges2.begin(), Merges2.end());
      return Merges;
    }

    if (MemOps[i].Position > Pos) {
      Pos = MemOps[i].Position;
      Loc = MemOps[i].MBBI;
    }
  }

  if (mergeOps(MBB, ++Loc, SOffset, Base, Opcode, Regs, TII)) {
    Merges.push_back(prior(Loc));
    for (unsigned i = SIndex, e = MemOps.size(); i != e; ++i) {
      MBB.erase(MemOps[i].MBBI);
      MemOps[i].Merged = true;
    }
  }

  return Merges;
}

static inline bool isMatchingDecrement(MachineInstr *MI, unsigned Base,
                                       unsigned Bytes) {
  return (MI && MI->getOpcode() == ARM::SUBri &&
          MI->getOperand(0).getReg() == Base &&
          MI->getOperand(1).getReg() == Base &&
          ARM_AM::getAM2Offset(MI->getOperand(2).getImm()) == Bytes);
}

static inline bool isMatchingIncrement(MachineInstr *MI, unsigned Base,
                                       unsigned Bytes) {
  return (MI && MI->getOpcode() == ARM::ADDri &&
          MI->getOperand(0).getReg() == Base &&
          MI->getOperand(1).getReg() == Base &&
          ARM_AM::getAM2Offset(MI->getOperand(2).getImm()) == Bytes);
}

static inline unsigned getLSMultipleTransferSize(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default: return 0;
  case ARM::LDR:
  case ARM::STR:
  case ARM::FLDS:
  case ARM::FSTS:
    return 4;
  case ARM::FLDD:
  case ARM::FSTD:
    return 8;
  case ARM::LDM:
  case ARM::STM:
    return (MI->getNumOperands() - 2) * 4;
  case ARM::FLDMS:
  case ARM::FSTMS:
  case ARM::FLDMD:
  case ARM::FSTMD:
    return ARM_AM::getAM5Offset(MI->getOperand(1).getImm()) * 4;
  }
}

/// mergeBaseUpdateLSMultiple - Fold proceeding/trailing inc/dec of base
/// register into the LDM/STM/FLDM{D|S}/FSTM{D|S} op when possible:
///
/// stmia rn, <ra, rb, rc>
/// rn := rn + 4 * 3;
/// =>
/// stmia rn!, <ra, rb, rc>
///
/// rn := rn - 4 * 3;
/// ldmia rn, <ra, rb, rc>
/// =>
/// ldmdb rn!, <ra, rb, rc>
static bool mergeBaseUpdateLSMultiple(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  MachineInstr *MI = MBBI;
  unsigned Base = MI->getOperand(0).getReg();
  unsigned Bytes = getLSMultipleTransferSize(MI);
  int Opcode = MI->getOpcode();
  bool isAM4 = Opcode == ARM::LDM || Opcode == ARM::STM;

  if (isAM4) {
    if (ARM_AM::getAM4WBFlag(MI->getOperand(1).getImm()))
      return false;

    // Can't use the updating AM4 sub-mode if the base register is also a dest
    // register. e.g. ldmdb r0!, {r0, r1, r2}. The behavior is undefined.
    for (unsigned i = 2, e = MI->getNumOperands(); i != e; ++i) {
      if (MI->getOperand(i).getReg() == Base)
        return false;
    }

    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MI->getOperand(1).getImm());
    if (MBBI != MBB.begin()) {
      MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
      if (Mode == ARM_AM::ia &&
          isMatchingDecrement(PrevMBBI, Base, Bytes)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(ARM_AM::db, true));
        MBB.erase(PrevMBBI);
        return true;
      } else if (Mode == ARM_AM::ib &&
                 isMatchingDecrement(PrevMBBI, Base, Bytes)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(ARM_AM::da, true));
        MBB.erase(PrevMBBI);
        return true;
      }
    }

    if (MBBI != MBB.end()) {
      MachineBasicBlock::iterator NextMBBI = next(MBBI);
      if ((Mode == ARM_AM::ia || Mode == ARM_AM::ib) &&
          isMatchingIncrement(NextMBBI, Base, Bytes)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(Mode, true));
        MBB.erase(NextMBBI);
        return true;
      } else if ((Mode == ARM_AM::da || Mode == ARM_AM::db) &&
                 isMatchingDecrement(NextMBBI, Base, Bytes)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(Mode, true));
        MBB.erase(NextMBBI);
        return true;
      }
    }
  } else {
    // FLDM{D|S}, FSTM{D|S} addressing mode 5 ops.
    if (ARM_AM::getAM5WBFlag(MI->getOperand(1).getImm()))
      return false;

    ARM_AM::AMSubMode Mode = ARM_AM::getAM5SubMode(MI->getOperand(1).getImm());
    unsigned Offset = ARM_AM::getAM5Offset(MI->getOperand(1).getImm());
    if (MBBI != MBB.begin()) {
      MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
      if (Mode == ARM_AM::ia &&
          isMatchingDecrement(PrevMBBI, Base, Bytes)) {
        MI->getOperand(1).setImm(ARM_AM::getAM5Opc(ARM_AM::db, true, Offset));
        MBB.erase(PrevMBBI);
        return true;
      }
    }

    if (MBBI != MBB.end()) {
      MachineBasicBlock::iterator NextMBBI = next(MBBI);
      if (Mode == ARM_AM::ia &&
          isMatchingIncrement(NextMBBI, Base, Bytes)) {
        MI->getOperand(1).setImm(ARM_AM::getAM5Opc(ARM_AM::ia, true, Offset));
        MBB.erase(NextMBBI);
      }
      return true;
    }
  }

  return false;
}

static unsigned getPreIndexedLoadStoreOpcode(unsigned Opc) {
  switch (Opc) {
  case ARM::LDR: return ARM::LDR_PRE;
  case ARM::STR: return ARM::STR_PRE;
  case ARM::FLDS: return ARM::FLDMS;
  case ARM::FLDD: return ARM::FLDMD;
  case ARM::FSTS: return ARM::FSTMS;
  case ARM::FSTD: return ARM::FSTMD;
  default: abort();
  }
  return 0;
}

static unsigned getPostIndexedLoadStoreOpcode(unsigned Opc) {
  switch (Opc) {
  case ARM::LDR: return ARM::LDR_POST;
  case ARM::STR: return ARM::STR_POST;
  case ARM::FLDS: return ARM::FLDMS;
  case ARM::FLDD: return ARM::FLDMD;
  case ARM::FSTS: return ARM::FSTMS;
  case ARM::FSTD: return ARM::FSTMD;
  default: abort();
  }
  return 0;
}

/// mergeBaseUpdateLoadStore - Fold proceeding/trailing inc/dec of base
/// register into the LDR/STR/FLD{D|S}/FST{D|S} op when possible:
static bool mergeBaseUpdateLoadStore(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     const TargetInstrInfo *TII) {
  MachineInstr *MI = MBBI;
  unsigned Base = MI->getOperand(1).getReg();
  unsigned Bytes = getLSMultipleTransferSize(MI);
  int Opcode = MI->getOpcode();
  bool isAM2 = Opcode == ARM::LDR || Opcode == ARM::STR;
  if ((isAM2 && ARM_AM::getAM2Offset(MI->getOperand(3).getImm()) != 0) ||
      (!isAM2 && ARM_AM::getAM5Offset(MI->getOperand(2).getImm()) != 0))
    return false;

  bool isLd = Opcode == ARM::LDR || Opcode == ARM::FLDS || Opcode == ARM::FLDD;
  // Can't do the merge if the destination register is the same as the would-be
  // writeback register.
  if (isLd && MI->getOperand(0).getReg() == Base)
    return false;

  bool DoMerge = false;
  ARM_AM::AddrOpc AddSub = ARM_AM::add;
  unsigned NewOpc = 0;
  if (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
    if (isMatchingDecrement(PrevMBBI, Base, Bytes)) {
      DoMerge = true;
      AddSub = ARM_AM::sub;
      NewOpc = getPreIndexedLoadStoreOpcode(Opcode);
    } else if (isAM2 && isMatchingIncrement(PrevMBBI, Base, Bytes)) {
      DoMerge = true;
      NewOpc = getPreIndexedLoadStoreOpcode(Opcode);
    }
    if (DoMerge)
      MBB.erase(PrevMBBI);
  }

  if (!DoMerge && MBBI != MBB.end()) {
    MachineBasicBlock::iterator NextMBBI = next(MBBI);
    if (isAM2 && isMatchingDecrement(NextMBBI, Base, Bytes)) {
      DoMerge = true;
      AddSub = ARM_AM::sub;
      NewOpc = getPostIndexedLoadStoreOpcode(Opcode);
    } else if (isMatchingIncrement(NextMBBI, Base, Bytes)) {
      DoMerge = true;
      NewOpc = getPostIndexedLoadStoreOpcode(Opcode);
    }
    if (DoMerge)
      MBB.erase(NextMBBI);
  }

  if (!DoMerge)
    return false;

  bool isDPR = NewOpc == ARM::FLDMD || NewOpc == ARM::FSTMD;
  unsigned Offset = isAM2 ? ARM_AM::getAM2Opc(AddSub, Bytes, ARM_AM::no_shift)
    : ARM_AM::getAM5Opc((AddSub == ARM_AM::sub) ? ARM_AM::db : ARM_AM::ia,
                        true, isDPR ? 2 : 1);
  if (isLd) {
    if (isAM2)
      BuildMI(MBB, MBBI, TII->get(NewOpc), MI->getOperand(0).getReg())
        .addReg(Base, true).addReg(Base).addReg(0).addImm(Offset);
    else
      BuildMI(MBB, MBBI, TII->get(NewOpc)).addReg(Base)
        .addImm(Offset).addReg(MI->getOperand(0).getReg(), true);
  } else {
    if (isAM2)
      BuildMI(MBB, MBBI, TII->get(NewOpc), Base).addReg(MI->getOperand(0).getReg())
        .addReg(Base).addReg(0).addImm(Offset);
    else
      BuildMI(MBB, MBBI, TII->get(NewOpc)).addReg(Base)
        .addImm(Offset).addReg(MI->getOperand(0).getReg(), false);
  }
  MBB.erase(MBBI);

  return true;
}

/// LoadStoreMultipleOpti - An optimization pass to turn multiple LDR / STR
/// ops of the same base and incrementing offset into LDM / STM ops.
bool ARMLoadStoreOpt::LoadStoreMultipleOpti(MachineBasicBlock &MBB) {
  unsigned NumMerges = 0;
  unsigned NumMemOps = 0;
  MemOpQueue MemOps;
  unsigned CurrBase = 0;
  int CurrOpc = -1;
  unsigned CurrSize = 0;
  unsigned Position = 0;
  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    bool Advance  = false;
    bool TryMerge = false;
    bool Clobber  = false;

    int Opcode = MBBI->getOpcode();
    bool isMemOp = false;
    bool isAM2 = false;
    unsigned Size = 4;
    switch (Opcode) {
    case ARM::LDR:
    case ARM::STR:
      isMemOp =
        (MBBI->getOperand(1).isRegister() && MBBI->getOperand(2).getReg() == 0);
      isAM2 = true;
      break;
    case ARM::FLDS:
    case ARM::FSTS:
      isMemOp = MBBI->getOperand(1).isRegister();
      break;
    case ARM::FLDD:
    case ARM::FSTD:
      isMemOp = MBBI->getOperand(1).isRegister();
      Size = 8;
      break;
    }
    if (isMemOp) {
      unsigned Base = MBBI->getOperand(1).getReg();
      unsigned OffIdx = MBBI->getNumOperands()-1;
      unsigned OffField = MBBI->getOperand(OffIdx).getImm();
      int Offset = isAM2
        ? ARM_AM::getAM2Offset(OffField) : ARM_AM::getAM5Offset(OffField) * 4;
      if (isAM2) {
        if (ARM_AM::getAM2Op(OffField) == ARM_AM::sub)
          Offset = -Offset;
      } else {
        if (ARM_AM::getAM5Op(OffField) == ARM_AM::sub)
          Offset = -Offset;
      }
      // Watch out for:
      // r4 := ldr [r5]
      // r5 := ldr [r5, #4]
      // r6 := ldr [r5, #8]
      //
      // The second ldr has effectively broken the chain even though it
      // looks like the later ldr(s) use the same base register. Try to
      // merge the ldr's so far, including this one. But don't try to
      // combine the following ldr(s).
      Clobber = (Opcode == ARM::LDR && Base == MBBI->getOperand(0).getReg());
      if (CurrBase == 0 && !Clobber) {
        // Start of a new chain.
        CurrBase = Base;
        CurrOpc  = Opcode;
        CurrSize = Size;
        MemOps.push_back(MemOpQueueEntry(Offset, Position, MBBI));
        NumMemOps++;
        Advance = true;
      } else {
        if (Clobber) {
          TryMerge = true;
          Advance = true;
        }

        if (CurrOpc == Opcode && CurrBase == Base) {
          // Continue adding to the queue.
          if (Offset > MemOps.back().Offset) {
            MemOps.push_back(MemOpQueueEntry(Offset, Position, MBBI));
            NumMemOps++;
            Advance = true;
          } else {
            for (MemOpQueueIter I = MemOps.begin(), E = MemOps.end();
                 I != E; ++I) {
              if (Offset < I->Offset) {
                MemOps.insert(I, MemOpQueueEntry(Offset, Position, MBBI));
                NumMemOps++;
                Advance = true;
                break;
              } else if (Offset == I->Offset) {
                // Collision! This can't be merged!
                break;
              }
            }
          }
        }
      }
    }

    if (Advance) {
      ++Position;
      ++MBBI;
    } else
      TryMerge = true;

    if (TryMerge) {
      if (NumMemOps > 1) {
        SmallVector<MachineBasicBlock::iterator,4> MBBII =
          MergeLDR_STR(MBB, 0, CurrBase, CurrOpc, CurrSize,MemOps);
        // Try folding preceeding/trailing base inc/dec into the generated
        // LDM/STM ops.
        for (unsigned i = 0, e = MBBII.size(); i < e; ++i)
          if (mergeBaseUpdateLSMultiple(MBB, MBBII[i]))
            NumMerges++;
        NumMerges += MBBII.size();
      }

      // Try folding preceeding/trailing base inc/dec into those load/store
      // that were not merged to form LDM/STM ops.
      for (unsigned i = 0; i != NumMemOps; ++i)
        if (!MemOps[i].Merged)
          if (mergeBaseUpdateLoadStore(MBB, MemOps[i].MBBI, TII))
            NumMerges++;

      CurrBase = 0;
      CurrOpc = -1;
      if (NumMemOps) {
        MemOps.clear();
        NumMemOps = 0;
      }

      // If iterator hasn't been advanced and this is not a memory op, skip it.
      // It can't start a new chain anyway.
      if (!Advance && !isMemOp && MBBI != E) {
        ++Position;
        ++MBBI;
      }
    }
  }
  return NumMerges > 0;
}

/// MergeReturnIntoLDM - If this is a exit BB, try merging the return op
/// (bx lr) into the preceeding stack restore so it directly restore the value
/// of LR into pc.
///   ldmfd sp!, {r7, lr}
///   bx lr
/// =>
///   ldmfd sp!, {r7, pc}
bool ARMLoadStoreOpt::MergeReturnIntoLDM(MachineBasicBlock &MBB) {
  if (MBB.empty()) return false;

  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  if (MBBI->getOpcode() == ARM::BX_RET && MBBI != MBB.begin()) {
    MachineInstr *PrevMI = prior(MBBI);
    if (PrevMI->getOpcode() == ARM::LDM) {
      MachineOperand &MO = PrevMI->getOperand(PrevMI->getNumOperands()-1);
      if (MO.getReg() == ARM::LR) {
        PrevMI->setInstrDescriptor(TII->get(ARM::LDM_RET));
        MO.setReg(ARM::PC);
        MBB.erase(MBBI);
        return true;
      }
    }
  }
  return false;
}

bool ARMLoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  TII = Fn.getTarget().getInstrInfo();
  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= LoadStoreMultipleOpti(MBB);
    Modified |= MergeReturnIntoLDM(MBB);
  }
  return Modified;
}

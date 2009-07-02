//===- ARMInstrInfo.cpp - ARM Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool>
EnableARM3Addr("enable-arm-3-addr-conv", cl::Hidden,
               cl::desc("Enable ARM 2-addr to 3-addr conv"));

static inline
const MachineInstrBuilder &AddDefaultPred(const MachineInstrBuilder &MIB) {
  return MIB.addImm((int64_t)ARMCC::AL).addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultCC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

ARMBaseInstrInfo::ARMBaseInstrInfo(const ARMSubtarget &STI)
  : TargetInstrInfoImpl(ARMInsts, array_lengthof(ARMInsts)) {
}

ARMInstrInfo::ARMInstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

void ARMInstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I,
                                 unsigned DestReg,
                                 const MachineInstr *Orig) const {
  DebugLoc dl = Orig->getDebugLoc();
  if (Orig->getOpcode() == ARM::MOVi2pieces) {
    RI.emitLoadConstPool(MBB, I, this, dl,
                         DestReg,
                         Orig->getOperand(1).getImm(),
                         (ARMCC::CondCodes)Orig->getOperand(2).getImm(),
                         Orig->getOperand(3).getReg());
    return;
  }

  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->getOperand(0).setReg(DestReg);
  MBB.insert(I, MI);
}

static unsigned getUnindexedOpcode(unsigned Opc) {
  switch (Opc) {
  default: break;
  case ARM::LDR_PRE:
  case ARM::LDR_POST:
    return ARM::LDR;
  case ARM::LDRH_PRE:
  case ARM::LDRH_POST:
    return ARM::LDRH;
  case ARM::LDRB_PRE:
  case ARM::LDRB_POST:
    return ARM::LDRB;
  case ARM::LDRSH_PRE:
  case ARM::LDRSH_POST:
    return ARM::LDRSH;
  case ARM::LDRSB_PRE:
  case ARM::LDRSB_POST:
    return ARM::LDRSB;
  case ARM::STR_PRE:
  case ARM::STR_POST:
    return ARM::STR;
  case ARM::STRH_PRE:
  case ARM::STRH_POST:
    return ARM::STRH;
  case ARM::STRB_PRE:
  case ARM::STRB_POST:
    return ARM::STRB;
  }
  return 0;
}

MachineInstr *
ARMBaseInstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                        MachineBasicBlock::iterator &MBBI,
                                        LiveVariables *LV) const {
  if (!EnableARM3Addr)
    return NULL;

  MachineInstr *MI = MBBI;
  MachineFunction &MF = *MI->getParent()->getParent();
  unsigned TSFlags = MI->getDesc().TSFlags;
  bool isPre = false;
  switch ((TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift) {
  default: return NULL;
  case ARMII::IndexModePre:
    isPre = true;
    break;
  case ARMII::IndexModePost:
    break;
  }

  // Try splitting an indexed load/store to an un-indexed one plus an add/sub
  // operation.
  unsigned MemOpc = getUnindexedOpcode(MI->getOpcode());
  if (MemOpc == 0)
    return NULL;

  MachineInstr *UpdateMI = NULL;
  MachineInstr *MemMI = NULL;
  unsigned AddrMode = (TSFlags & ARMII::AddrModeMask);
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned NumOps = TID.getNumOperands();
  bool isLoad = !TID.mayStore();
  const MachineOperand &WB = isLoad ? MI->getOperand(1) : MI->getOperand(0);
  const MachineOperand &Base = MI->getOperand(2);
  const MachineOperand &Offset = MI->getOperand(NumOps-3);
  unsigned WBReg = WB.getReg();
  unsigned BaseReg = Base.getReg();
  unsigned OffReg = Offset.getReg();
  unsigned OffImm = MI->getOperand(NumOps-2).getImm();
  ARMCC::CondCodes Pred = (ARMCC::CondCodes)MI->getOperand(NumOps-1).getImm();
  switch (AddrMode) {
  default:
    assert(false && "Unknown indexed op!");
    return NULL;
  case ARMII::AddrMode2: {
    bool isSub = ARM_AM::getAM2Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM2Offset(OffImm);
    if (OffReg == 0) {
      int SOImmVal = ARM_AM::getSOImmVal(Amt);
      if (SOImmVal == -1)
        // Can't encode it in a so_imm operand. This transformation will
        // add more than 1 instruction. Abandon!
        return NULL;
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(SOImmVal)
        .addImm(Pred).addReg(0).addReg(0);
    } else if (Amt != 0) {
      ARM_AM::ShiftOpc ShOpc = ARM_AM::getAM2ShiftOpc(OffImm);
      unsigned SOOpc = ARM_AM::getSORegOpc(ShOpc, Amt);
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrs : ARM::ADDrs), WBReg)
        .addReg(BaseReg).addReg(OffReg).addReg(0).addImm(SOOpc)
        .addImm(Pred).addReg(0).addReg(0);
    } else
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  case ARMII::AddrMode3 : {
    bool isSub = ARM_AM::getAM3Op(OffImm) == ARM_AM::sub;
    unsigned Amt = ARM_AM::getAM3Offset(OffImm);
    if (OffReg == 0)
      // Immediate is 8-bits. It's guaranteed to fit in a so_imm operand.
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBri : ARM::ADDri), WBReg)
        .addReg(BaseReg).addImm(Amt)
        .addImm(Pred).addReg(0).addReg(0);
    else
      UpdateMI = BuildMI(MF, MI->getDebugLoc(),
                         get(isSub ? ARM::SUBrr : ARM::ADDrr), WBReg)
        .addReg(BaseReg).addReg(OffReg)
        .addImm(Pred).addReg(0).addReg(0);
    break;
  }
  }

  std::vector<MachineInstr*> NewMIs;
  if (isPre) {
    if (isLoad)
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc), MI->getOperand(0).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(WBReg).addReg(0).addImm(0).addImm(Pred);
    NewMIs.push_back(MemMI);
    NewMIs.push_back(UpdateMI);
  } else {
    if (isLoad)
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc), MI->getOperand(0).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    else
      MemMI = BuildMI(MF, MI->getDebugLoc(),
                      get(MemOpc)).addReg(MI->getOperand(1).getReg())
        .addReg(BaseReg).addReg(0).addImm(0).addImm(Pred);
    if (WB.isDead())
      UpdateMI->getOperand(0).setIsDead();
    NewMIs.push_back(UpdateMI);
    NewMIs.push_back(MemMI);
  }

  // Transfer LiveVariables states, kill / dead info.
  if (LV) {
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.getReg() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        unsigned Reg = MO.getReg();

        LiveVariables::VarInfo &VI = LV->getVarInfo(Reg);
        if (MO.isDef()) {
          MachineInstr *NewMI = (Reg == WBReg) ? UpdateMI : MemMI;
          if (MO.isDead())
            LV->addVirtualRegisterDead(Reg, NewMI);
        }
        if (MO.isUse() && MO.isKill()) {
          for (unsigned j = 0; j < 2; ++j) {
            // Look at the two new MI's in reverse order.
            MachineInstr *NewMI = NewMIs[j];
            if (!NewMI->readsRegister(Reg))
              continue;
            LV->addVirtualRegisterKilled(Reg, NewMI);
            if (VI.removeKill(MI))
              VI.Kills.push_back(NewMI);
            break;
          }
        }
      }
    }
  }

  MFI->insert(MBBI, NewMIs[1]);
  MFI->insert(MBBI, NewMIs[0]);
  return NewMIs[0];
}

// Branch analysis.
bool
ARMBaseInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                MachineBasicBlock *&FBB,
                                SmallVectorImpl<MachineOperand> &Cond,
                                bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastOpc == ARM::B || LastOpc == ARM::tB || LastOpc == ARM::t2B) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (LastOpc == ARM::Bcc || LastOpc == ARM::tBcc || LastOpc == ARM::t2Bcc) {
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(LastInst->getOperand(1));
      Cond.push_back(LastInst->getOperand(2));
      return false;
    }
    return true;  // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with ARM::B/ARM::tB/ARM::t2B and a 
  // ARM::Bcc/ARM::tBcc/ARM::t2Bcc, handle it.
  unsigned SecondLastOpc = SecondLastInst->getOpcode();
  if ((SecondLastOpc == ARM::Bcc && LastOpc == ARM::B) ||
      (SecondLastOpc == ARM::tBcc && LastOpc == ARM::tB) ||
      (SecondLastOpc == ARM::t2Bcc && LastOpc == ARM::t2B)) {
    TBB =  SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(SecondLastInst->getOperand(1));
    Cond.push_back(SecondLastInst->getOperand(2));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if ((SecondLastOpc == ARM::B || SecondLastOpc==ARM::tB || 
       SecondLastOpc==ARM::t2B) &&
      (LastOpc == ARM::B || LastOpc == ARM::tB || LastOpc == ARM::t2B)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // ...likewise if it ends with a branch table followed by an unconditional
  // branch. The branch folder can create these, and we must get rid of them for
  // correctness of Thumb constant islands.
  if ((SecondLastOpc == ARM::BR_JTr || SecondLastOpc==ARM::BR_JTm ||
       SecondLastOpc == ARM::BR_JTadd || SecondLastOpc==ARM::tBR_JTr ||
       SecondLastOpc == ARM::t2BR_JTr || SecondLastOpc==ARM::t2BR_JTm ||
       SecondLastOpc == ARM::t2BR_JTadd) &&
      (LastOpc == ARM::B || LastOpc == ARM::tB || LastOpc == ARM::t2B)) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}


unsigned ARMBaseInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? 
    (AFI->isThumb2Function() ? ARM::t2B : ARM::tB) : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? 
    (AFI->isThumb2Function() ? ARM::t2Bcc : ARM::tBcc) : ARM::Bcc;

  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != BOpc && I->getOpcode() != BccOpc)
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != BccOpc)
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
ARMBaseInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                               MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond) const {
  // FIXME this should probably have a DebugLoc argument
  DebugLoc dl = DebugLoc::getUnknownLoc();
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  int BOpc   = AFI->isThumbFunction() ? 
    (AFI->isThumb2Function() ? ARM::t2B : ARM::tB) : ARM::B;
  int BccOpc = AFI->isThumbFunction() ? 
    (AFI->isThumb2Function() ? ARM::t2Bcc : ARM::tBcc) : ARM::Bcc;

  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "ARM branch conditions have two components!");

  if (FBB == 0) {
    if (Cond.empty()) // Unconditional branch?
      BuildMI(&MBB, dl, get(BOpc)).addMBB(TBB);
    else
      BuildMI(&MBB, dl, get(BccOpc)).addMBB(TBB)
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
    return 1;
  }

  // Two-way conditional branch.
  BuildMI(&MBB, dl, get(BccOpc)).addMBB(TBB)
    .addImm(Cond[0].getImm()).addReg(Cond[1].getReg());
  BuildMI(&MBB, dl, get(BOpc)).addMBB(FBB);
  return 2;
}

bool
ARMBaseInstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;

  switch (MBB.back().getOpcode()) {
  case ARM::BX_RET:   // Return.
  case ARM::LDM_RET:
  case ARM::tBX_RET:
  case ARM::tBX_RET_vararg:
  case ARM::tPOP_RET:
  case ARM::B:
  case ARM::tB:
  case ARM::t2B:      // Uncond branch.
  case ARM::tBR_JTr:
  case ARM::t2BR_JTr:
  case ARM::BR_JTr:   // Jumptable branch.
  case ARM::t2BR_JTm:
  case ARM::BR_JTm:   // Jumptable branch through mem.
  case ARM::t2BR_JTadd:
  case ARM::BR_JTadd: // Jumptable branch add to pc.
    return true;
  default: return false;
  }
}

bool ARMBaseInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)(int)Cond[0].getImm();
  Cond[0].setImm(ARMCC::getOppositeCondition(CC));
  return false;
}

bool ARMBaseInstrInfo::isPredicated(const MachineInstr *MI) const {
  int PIdx = MI->findFirstPredOperandIdx();
  return PIdx != -1 && MI->getOperand(PIdx).getImm() != ARMCC::AL;
}

bool ARMBaseInstrInfo::
PredicateInstruction(MachineInstr *MI,
                     const SmallVectorImpl<MachineOperand> &Pred) const {
  unsigned Opc = MI->getOpcode();
  if (Opc == ARM::B || Opc == ARM::tB || Opc == ARM::t2B) {
    MI->setDesc(get((Opc == ARM::B) ? ARM::Bcc :
                    ((Opc == ARM::tB) ? ARM::tBcc : ARM::t2Bcc)));
    MI->addOperand(MachineOperand::CreateImm(Pred[0].getImm()));
    MI->addOperand(MachineOperand::CreateReg(Pred[1].getReg(), false));
    return true;
  }

  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx != -1) {
    MachineOperand &PMO = MI->getOperand(PIdx);
    PMO.setImm(Pred[0].getImm());
    MI->getOperand(PIdx+1).setReg(Pred[1].getReg());
    return true;
  }
  return false;
}

bool ARMBaseInstrInfo::
SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                  const SmallVectorImpl<MachineOperand> &Pred2) const {
  if (Pred1.size() > 2 || Pred2.size() > 2)
    return false;

  ARMCC::CondCodes CC1 = (ARMCC::CondCodes)Pred1[0].getImm();
  ARMCC::CondCodes CC2 = (ARMCC::CondCodes)Pred2[0].getImm();
  if (CC1 == CC2)
    return true;

  switch (CC1) {
  default:
    return false;
  case ARMCC::AL:
    return true;
  case ARMCC::HS:
    return CC2 == ARMCC::HI;
  case ARMCC::LS:
    return CC2 == ARMCC::LO || CC2 == ARMCC::EQ;
  case ARMCC::GE:
    return CC2 == ARMCC::GT;
  case ARMCC::LE:
    return CC2 == ARMCC::LT;
  }
}

bool ARMBaseInstrInfo::DefinesPredicate(MachineInstr *MI,
                                    std::vector<MachineOperand> &Pred) const {
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.getImplicitDefs() && !TID.hasOptionalDef())
    return false;

  bool Found = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == ARM::CPSR) {
      Pred.push_back(MO);
      Found = true;
    }
  }

  return Found;
}


/// FIXME: Works around a gcc miscompilation with -fstrict-aliasing
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) DISABLE_INLINE;
static unsigned getNumJTEntries(const std::vector<MachineJumpTableEntry> &JT,
                                unsigned JTI) {
  return JT[JTI].MBBs.size();
}

/// GetInstSize - Return the size of the specified MachineInstr.
///
unsigned ARMBaseInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  const MachineBasicBlock &MBB = *MI->getParent();
  const MachineFunction *MF = MBB.getParent();
  const TargetAsmInfo *TAI = MF->getTarget().getTargetAsmInfo();

  // Basic size info comes from the TSFlags field.
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned TSFlags = TID.TSFlags;

  switch ((TSFlags & ARMII::SizeMask) >> ARMII::SizeShift) {
  default: {
    // If this machine instr is an inline asm, measure it.
    if (MI->getOpcode() == ARM::INLINEASM)
      return TAI->getInlineAsmLength(MI->getOperand(0).getSymbolName());
    if (MI->isLabel())
      return 0;
    switch (MI->getOpcode()) {
    default:
      assert(0 && "Unknown or unset size field for instr!");
      break;
    case TargetInstrInfo::IMPLICIT_DEF:
    case TargetInstrInfo::DECLARE:
    case TargetInstrInfo::DBG_LABEL:
    case TargetInstrInfo::EH_LABEL:
      return 0;
    }
    break;
  }
  case ARMII::Size8Bytes: return 8;          // Arm instruction x 2.
  case ARMII::Size4Bytes: return 4;          // Arm instruction.
  case ARMII::Size2Bytes: return 2;          // Thumb instruction.
  case ARMII::SizeSpecial: {
    switch (MI->getOpcode()) {
    case ARM::CONSTPOOL_ENTRY:
      // If this machine instr is a constant pool entry, its size is recorded as
      // operand #2.
      return MI->getOperand(2).getImm();
    case ARM::Int_eh_sjlj_setjmp: return 12;
    case ARM::BR_JTr:
    case ARM::BR_JTm:
    case ARM::BR_JTadd:
    case ARM::t2BR_JTr:
    case ARM::t2BR_JTm:
    case ARM::t2BR_JTadd:
    case ARM::tBR_JTr: {
      // These are jumptable branches, i.e. a branch followed by an inlined
      // jumptable. The size is 4 + 4 * number of entries.
      unsigned NumOps = TID.getNumOperands();
      MachineOperand JTOP =
        MI->getOperand(NumOps - (TID.isPredicable() ? 3 : 2));
      unsigned JTI = JTOP.getIndex();
      const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
      const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
      assert(JTI < JT.size());
      // Thumb instructions are 2 byte aligned, but JT entries are 4 byte
      // 4 aligned. The assembler / linker may add 2 byte padding just before
      // the JT entries.  The size does not include this padding; the
      // constant islands pass does separate bookkeeping for it.
      // FIXME: If we know the size of the function is less than (1 << 16) *2
      // bytes, we can use 16-bit entries instead. Then there won't be an
      // alignment issue.
      return getNumJTEntries(JT, JTI) * 4 +
        ((MI->getOpcode()==ARM::tBR_JTr) ? 2 : 4);
    }
    default:
      // Otherwise, pseudo-instruction sizes are zero.
      return 0;
    }
  }
  }
  return 0; // Not reached
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool
ARMBaseInstrInfo::isMoveInstr(const MachineInstr &MI,
                              unsigned &SrcReg, unsigned &DstReg,
                              unsigned& SrcSubIdx, unsigned& DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers.

  unsigned oc = MI.getOpcode();
  switch (oc) {
  default:
    return false;
  case ARM::FCPYS:
  case ARM::FCPYD:
  case ARM::VMOVD:
  case ARM::VMOVQ:
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  case ARM::MOVr:
    assert(MI.getDesc().getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "Invalid ARM MOV instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
}

unsigned 
ARMBaseInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::LDR:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FLDD:
  case ARM::FLDS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned
ARMBaseInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                     int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case ARM::STR:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isReg() &&
        MI->getOperand(3).isImm() &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  case ARM::FSTD:
  case ARM::FSTS:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() &&
        MI->getOperand(2).getImm() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }

  return 0;
}

bool
ARMBaseInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I,
                               unsigned DestReg, unsigned SrcReg,
                               const TargetRegisterClass *DestRC,
                               const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (DestRC != SrcRC) {
    // Not yet supported!
    return false;
  }

  if (DestRC == ARM::GPRRegisterClass)
    AddDefaultCC(AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::MOVr), DestReg)
                                .addReg(SrcReg)));
  else if (DestRC == ARM::SPRRegisterClass)
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FCPYS), DestReg)
                   .addReg(SrcReg));
  else if (DestRC == ARM::DPRRegisterClass)
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FCPYD), DestReg)
                   .addReg(SrcReg));
  else if (DestRC == ARM::QPRRegisterClass)
    BuildMI(MBB, I, DL, get(ARM::VMOVQ), DestReg).addReg(SrcReg);
  else
    return false;

  return true;
}

void ARMBaseInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (RC == ARM::GPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::STR))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addReg(0).addImm(0));
  } else if (RC == ARM::DPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FSTD))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0));
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FSTS))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0));
  }
}

void 
ARMBaseInstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                 bool isKill,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const{
  DebugLoc DL = DebugLoc::getUnknownLoc();
  unsigned Opc = 0;
  if (RC == ARM::GPRRegisterClass) {
    Opc = ARM::STR;
  } else if (RC == ARM::DPRRegisterClass) {
    Opc = ARM::FSTD;
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    Opc = ARM::FSTS;
  }

  MachineInstrBuilder MIB =
    BuildMI(MF, DL, get(Opc)).addReg(SrcReg, getKillRegState(isKill));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  AddDefaultPred(MIB);
  NewMIs.push_back(MIB);
  return;
}

void ARMBaseInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (RC == ARM::GPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::LDR), DestReg)
                   .addFrameIndex(FI).addReg(0).addImm(0));
  } else if (RC == ARM::DPRRegisterClass) {
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FLDD), DestReg)
                   .addFrameIndex(FI).addImm(0));
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::FLDS), DestReg)
                   .addFrameIndex(FI).addImm(0));
  }
}

void ARMBaseInstrInfo::
loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                SmallVectorImpl<MachineOperand> &Addr,
                const TargetRegisterClass *RC,
                SmallVectorImpl<MachineInstr*> &NewMIs) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  unsigned Opc = 0;
  if (RC == ARM::GPRRegisterClass) {
    Opc = ARM::LDR;
  } else if (RC == ARM::DPRRegisterClass) {
    Opc = ARM::FLDD;
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    Opc = ARM::FLDS;
  }

  MachineInstrBuilder MIB =  BuildMI(MF, DL, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB.addOperand(Addr[i]);
  AddDefaultPred(MIB);
  NewMIs.push_back(MIB);
  return;
}

MachineInstr *ARMBaseInstrInfo::
foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                      const SmallVectorImpl<unsigned> &Ops, int FI) const {
  if (Ops.size() != 1) return NULL;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  MachineInstr *NewMI = NULL;
  switch (Opc) {
  default: break;
  case ARM::MOVr: {
    if (MI->getOperand(4).getReg() == ARM::CPSR)
      // If it is updating CPSR, then it cannot be folded.
      break;
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::STR))
        .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef))
        .addFrameIndex(FI).addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::LDR))
        .addReg(DstReg,
                RegState::Define |
                getDeadRegState(isDead) |
                getUndefRegState(isUndef))
        .addFrameIndex(FI).addReg(0).addImm(0).addImm(Pred).addReg(PredReg);
    }
    break;
  }
  case ARM::FCPYS: {
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FSTS))
        .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef))
        .addFrameIndex(FI)
        .addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FLDS))
        .addReg(DstReg,
                RegState::Define |
                getDeadRegState(isDead) |
                getUndefRegState(isUndef))
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    }
    break;
  }
  case ARM::FCPYD: {
    unsigned Pred = MI->getOperand(2).getImm();
    unsigned PredReg = MI->getOperand(3).getReg();
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      bool isUndef = MI->getOperand(1).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FSTD))
        .addReg(SrcReg, getKillRegState(isKill) | getUndefRegState(isUndef))
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      bool isDead = MI->getOperand(0).isDead();
      bool isUndef = MI->getOperand(0).isUndef();
      NewMI = BuildMI(MF, MI->getDebugLoc(), get(ARM::FLDD))
        .addReg(DstReg,
                RegState::Define |
                getDeadRegState(isDead) |
                getUndefRegState(isUndef))
        .addFrameIndex(FI).addImm(0).addImm(Pred).addReg(PredReg);
    }
    break;
  }
  }

  return NewMI;
}

MachineInstr* 
ARMBaseInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                        MachineInstr* MI,
                                        const SmallVectorImpl<unsigned> &Ops,
                                        MachineInstr* LoadMI) const {
  return 0;
}

bool
ARMBaseInstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                       const SmallVectorImpl<unsigned> &Ops) const {
  if (Ops.size() != 1) return false;

  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default: break;
  case ARM::MOVr:
    // If it is updating CPSR, then it cannot be folded.
    return MI->getOperand(4).getReg() != ARM::CPSR;
  case ARM::FCPYS:
  case ARM::FCPYD:
    return true;

  case ARM::VMOVD:
  case ARM::VMOVQ:
    return false; // FIXME
  }

  return false;
}

//===--- LiveRangeEdit.cpp - Basic tools for editing a register live range --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LiveRangeEdit class represents changes done to a virtual register when it
// is spilled or split.
//===----------------------------------------------------------------------===//

#include "LiveRangeEdit.h"
#include "VirtRegMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

LiveInterval &LiveRangeEdit::create(MachineRegisterInfo &mri,
                                    LiveIntervals &lis,
                                    VirtRegMap &vrm) {
  const TargetRegisterClass *RC = mri.getRegClass(getReg());
  unsigned VReg = mri.createVirtualRegister(RC);
  vrm.grow();
  vrm.setIsSplitFromReg(VReg, vrm.getOriginal(getReg()));
  LiveInterval &li = lis.getOrCreateInterval(VReg);
  newRegs_.push_back(&li);
  return li;
}

void LiveRangeEdit::scanRemattable(LiveIntervals &lis,
                                   const TargetInstrInfo &tii,
                                   AliasAnalysis *aa) {
  for (LiveInterval::vni_iterator I = parent_.vni_begin(),
       E = parent_.vni_end(); I != E; ++I) {
    VNInfo *VNI = *I;
    if (VNI->isUnused())
      continue;
    MachineInstr *DefMI = lis.getInstructionFromIndex(VNI->def);
    if (!DefMI)
      continue;
    if (tii.isTriviallyReMaterializable(DefMI, aa))
      remattable_.insert(VNI);
  }
  scannedRemattable_ = true;
}

bool LiveRangeEdit::anyRematerializable(LiveIntervals &lis,
                                        const TargetInstrInfo &tii,
                                        AliasAnalysis *aa) {
  if (!scannedRemattable_)
    scanRemattable(lis, tii, aa);
  return !remattable_.empty();
}

/// allUsesAvailableAt - Return true if all registers used by OrigMI at
/// OrigIdx are also available with the same value at UseIdx.
bool LiveRangeEdit::allUsesAvailableAt(const MachineInstr *OrigMI,
                                       SlotIndex OrigIdx,
                                       SlotIndex UseIdx,
                                       LiveIntervals &lis) {
  OrigIdx = OrigIdx.getUseIndex();
  UseIdx = UseIdx.getUseIndex();
  for (unsigned i = 0, e = OrigMI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = OrigMI->getOperand(i);
    if (!MO.isReg() || !MO.getReg() || MO.getReg() == getReg())
      continue;
    // Reserved registers are OK.
    if (MO.isUndef() || !lis.hasInterval(MO.getReg()))
      continue;
    // We don't want to move any defs.
    if (MO.isDef())
      return false;
    // We cannot depend on virtual registers in uselessRegs_.
    if (uselessRegs_)
      for (unsigned ui = 0, ue = uselessRegs_->size(); ui != ue; ++ui)
        if ((*uselessRegs_)[ui]->reg == MO.getReg())
          return false;

    LiveInterval &li = lis.getInterval(MO.getReg());
    const VNInfo *OVNI = li.getVNInfoAt(OrigIdx);
    if (!OVNI)
      continue;
    if (OVNI != li.getVNInfoAt(UseIdx))
      return false;
  }
  return true;
}

bool LiveRangeEdit::canRematerializeAt(Remat &RM,
                                       SlotIndex UseIdx,
                                       bool cheapAsAMove,
                                       LiveIntervals &lis) {
  assert(scannedRemattable_ && "Call anyRematerializable first");

  // Use scanRemattable info.
  if (!remattable_.count(RM.ParentVNI))
    return false;

  // No defining instruction.
  RM.OrigMI = lis.getInstructionFromIndex(RM.ParentVNI->def);
  assert(RM.OrigMI && "Defining instruction for remattable value disappeared");

  // If only cheap remats were requested, bail out early.
  if (cheapAsAMove && !RM.OrigMI->getDesc().isAsCheapAsAMove())
    return false;

  // Verify that all used registers are available with the same values.
  if (!allUsesAvailableAt(RM.OrigMI, RM.ParentVNI->def, UseIdx, lis))
    return false;

  return true;
}

SlotIndex LiveRangeEdit::rematerializeAt(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned DestReg,
                                         const Remat &RM,
                                         LiveIntervals &lis,
                                         const TargetInstrInfo &tii,
                                         const TargetRegisterInfo &tri) {
  assert(RM.OrigMI && "Invalid remat");
  tii.reMaterialize(MBB, MI, DestReg, 0, RM.OrigMI, tri);
  rematted_.insert(RM.ParentVNI);
  return lis.InsertMachineInstrInMaps(--MI).getDefIndex();
}

void LiveRangeEdit::eliminateDeadDefs(SmallVectorImpl<MachineInstr*> &Dead,
                                      LiveIntervals &LIS,
                                      const TargetInstrInfo &TII) {
  SetVector<LiveInterval*,
            SmallVector<LiveInterval*, 8>,
            SmallPtrSet<LiveInterval*, 8> > ToShrink;

  for (;;) {
    // Erase all dead defs.
    while (!Dead.empty()) {
      MachineInstr *MI = Dead.pop_back_val();
      assert(MI->allDefsAreDead() && "Def isn't really dead");

      // Never delete inline asm.
      if (MI->isInlineAsm())
        continue;

      // Use the same criteria as DeadMachineInstructionElim.
      bool SawStore = false;
      if (!MI->isSafeToMove(&TII, 0, SawStore))
        continue;

      SlotIndex Idx = LIS.getInstructionIndex(MI).getDefIndex();
      DEBUG(dbgs() << "Deleting dead def " << Idx << '\t' << *MI);

      // Check for live intervals that may shrink
      for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
             MOE = MI->operands_end(); MOI != MOE; ++MOI) {
        if (!MOI->isReg())
          continue;
        unsigned Reg = MOI->getReg();
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          continue;
        LiveInterval &LI = LIS.getInterval(Reg);
        // Remove defined value.
        if (MOI->isDef())
          if (VNInfo *VNI = LI.getVNInfoAt(Idx))
            LI.removeValNo(VNI);
        // Shrink read registers.
        if (MI->readsVirtualRegister(Reg))
          ToShrink.insert(&LI);
      }

      if (delegate_)
        delegate_->LRE_WillEraseInstruction(MI);
      LIS.RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
    }

    if (ToShrink.empty())
      break;

    // Shrink just one live interval. Then delete new dead defs.
    LIS.shrinkToUses(ToShrink.back(), &Dead);
    ToShrink.pop_back();
  }
}


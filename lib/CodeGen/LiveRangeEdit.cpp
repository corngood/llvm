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
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

LiveInterval &LiveRangeEdit::create(MachineRegisterInfo &mri,
                                    LiveIntervals &lis,
                                    VirtRegMap &vrm) {
  const TargetRegisterClass *RC = mri.getRegClass(parent_.reg);
  unsigned VReg = mri.createVirtualRegister(RC);
  vrm.grow();
  LiveInterval &li = lis.getOrCreateInterval(VReg);
  newRegs_.push_back(&li);
  return li;
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
    for (unsigned ui = 0, ue = uselessRegs_.size(); ui != ue; ++ui)
      if (uselessRegs_[ui]->reg == MO.getReg())
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


//===---------- SplitKit.cpp - Toolkit for splitting live ranges ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SplitAnalysis class as well as mutator functions for
// live range splitting.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "splitter"
#include "SplitKit.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static cl::opt<bool>
AllowSplit("spiller-splits-edges",
           cl::desc("Allow critical edge splitting during spilling"));

//===----------------------------------------------------------------------===//
//                                 Split Analysis
//===----------------------------------------------------------------------===//

SplitAnalysis::SplitAnalysis(const MachineFunction &mf,
                             const LiveIntervals &lis,
                             const MachineLoopInfo &mli)
  : mf_(mf),
    lis_(lis),
    loops_(mli),
    tii_(*mf.getTarget().getInstrInfo()),
    curli_(0) {}

void SplitAnalysis::clear() {
  usingInstrs_.clear();
  usingBlocks_.clear();
  usingLoops_.clear();
  curli_ = 0;
}

bool SplitAnalysis::canAnalyzeBranch(const MachineBasicBlock *MBB) {
  MachineBasicBlock *T, *F;
  SmallVector<MachineOperand, 4> Cond;
  return !tii_.AnalyzeBranch(const_cast<MachineBasicBlock&>(*MBB), T, F, Cond);
}

/// analyzeUses - Count instructions, basic blocks, and loops using curli.
void SplitAnalysis::analyzeUses() {
  const MachineRegisterInfo &MRI = mf_.getRegInfo();
  for (MachineRegisterInfo::reg_iterator I = MRI.reg_begin(curli_->reg);
       MachineInstr *MI = I.skipInstruction();) {
    if (MI->isDebugValue() || !usingInstrs_.insert(MI))
      continue;
    MachineBasicBlock *MBB = MI->getParent();
    if (usingBlocks_[MBB]++)
      continue;
    if (MachineLoop *Loop = loops_.getLoopFor(MBB))
      usingLoops_[Loop]++;
  }
  DEBUG(dbgs() << "  counted "
               << usingInstrs_.size() << " instrs, "
               << usingBlocks_.size() << " blocks, "
               << usingLoops_.size()  << " loops.\n");
}

/// removeUse - Update statistics by noting that MI no longer uses curli.
void SplitAnalysis::removeUse(const MachineInstr *MI) {
  if (!usingInstrs_.erase(MI))
    return;

  // Decrement MBB count.
  const MachineBasicBlock *MBB = MI->getParent();
  BlockCountMap::iterator bi = usingBlocks_.find(MBB);
  assert(bi != usingBlocks_.end() && "MBB missing");
  assert(bi->second && "0 count in map");
  if (--bi->second)
    return;
  // No more uses in MBB.
  usingBlocks_.erase(bi);

  // Decrement loop count.
  MachineLoop *Loop = loops_.getLoopFor(MBB);
  if (!Loop)
    return;
  LoopCountMap::iterator li = usingLoops_.find(Loop);
  assert(li != usingLoops_.end() && "Loop missing");
  assert(li->second && "0 count in map");
  if (--li->second)
    return;
  // No more blocks in Loop.
  usingLoops_.erase(li);
}

// Get three sets of basic blocks surrounding a loop: Blocks inside the loop,
// predecessor blocks, and exit blocks.
void SplitAnalysis::getLoopBlocks(const MachineLoop *Loop, LoopBlocks &Blocks) {
  Blocks.clear();

  // Blocks in the loop.
  Blocks.Loop.insert(Loop->block_begin(), Loop->block_end());

  // Predecessor blocks.
  const MachineBasicBlock *Header = Loop->getHeader();
  for (MachineBasicBlock::const_pred_iterator I = Header->pred_begin(),
       E = Header->pred_end(); I != E; ++I)
    if (!Blocks.Loop.count(*I))
      Blocks.Preds.insert(*I);

  // Exit blocks.
  for (MachineLoop::block_iterator I = Loop->block_begin(),
       E = Loop->block_end(); I != E; ++I) {
    const MachineBasicBlock *MBB = *I;
    for (MachineBasicBlock::const_succ_iterator SI = MBB->succ_begin(),
       SE = MBB->succ_end(); SI != SE; ++SI)
      if (!Blocks.Loop.count(*SI))
        Blocks.Exits.insert(*SI);
  }
}

/// analyzeLoopPeripheralUse - Return an enum describing how curli_ is used in
/// and around the Loop.
SplitAnalysis::LoopPeripheralUse SplitAnalysis::
analyzeLoopPeripheralUse(const SplitAnalysis::LoopBlocks &Blocks) {
  LoopPeripheralUse use = ContainedInLoop;
  for (BlockCountMap::iterator I = usingBlocks_.begin(), E = usingBlocks_.end();
       I != E; ++I) {
    const MachineBasicBlock *MBB = I->first;
    // Is this a peripheral block?
    if (use < MultiPeripheral &&
        (Blocks.Preds.count(MBB) || Blocks.Exits.count(MBB))) {
      if (I->second > 1) use = MultiPeripheral;
      else               use = SinglePeripheral;
      continue;
    }
    // Is it a loop block?
    if (Blocks.Loop.count(MBB))
      continue;
    // It must be an unrelated block.
    return OutsideLoop;
  }
  return use;
}

/// getCriticalExits - It may be necessary to partially break critical edges
/// leaving the loop if an exit block has phi uses of curli. Collect the exit
/// blocks that need special treatment into CriticalExits.
void SplitAnalysis::getCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  CriticalExits.clear();

  // A critical exit block contains a phi def of curli, and has a predecessor
  // that is not in the loop nor a loop predecessor.
  // For such an exit block, the edges carrying the new variable must be moved
  // to a new pre-exit block.
  for (BlockPtrSet::iterator I = Blocks.Exits.begin(), E = Blocks.Exits.end();
       I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    SlotIndex SuccIdx = lis_.getMBBStartIdx(Succ);
    VNInfo *SuccVNI = curli_->getVNInfoAt(SuccIdx);
    // This exit may not have curli live in at all. No need to split.
    if (!SuccVNI)
      continue;
    // If this is not a PHI def, it is either using a value from before the
    // loop, or a value defined inside the loop. Both are safe.
    if (!SuccVNI->isPHIDef() || SuccVNI->def.getBaseIndex() != SuccIdx)
      continue;
    // This exit block does have a PHI. Does it also have a predecessor that is
    // not a loop block or loop predecessor?
    for (MachineBasicBlock::const_pred_iterator PI = Succ->pred_begin(),
         PE = Succ->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      if (Blocks.Loop.count(Pred) || Blocks.Preds.count(Pred))
        continue;
      // This is a critical exit block, and we need to split the exit edge.
      CriticalExits.insert(Succ);
      break;
    }
  }
}

/// canSplitCriticalExits - Return true if it is possible to insert new exit
/// blocks before the blocks in CriticalExits.
bool
SplitAnalysis::canSplitCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  // If we don't allow critical edge splitting, require no critical exits.
  if (!AllowSplit)
    return CriticalExits.empty();

  for (BlockPtrSet::iterator I = CriticalExits.begin(), E = CriticalExits.end();
       I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    // We want to insert a new pre-exit MBB before Succ, and change all the
    // in-loop blocks to branch to the pre-exit instead of Succ.
    // Check that all the in-loop predecessors can be changed.
    for (MachineBasicBlock::const_pred_iterator PI = Succ->pred_begin(),
         PE = Succ->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      // The external predecessors won't be altered.
      if (!Blocks.Loop.count(Pred) && !Blocks.Preds.count(Pred))
        continue;
      if (!canAnalyzeBranch(Pred))
        return false;
    }

    // If Succ's layout predecessor falls through, that too must be analyzable.
    // We need to insert the pre-exit block in the gap.
    MachineFunction::const_iterator MFI = Succ;
    if (MFI == mf_.begin())
      continue;
    if (!canAnalyzeBranch(--MFI))
      return false;
  }
  // No problems found.
  return true;
}

void SplitAnalysis::analyze(const LiveInterval *li) {
  clear();
  curli_ = li;
  analyzeUses();
}

const MachineLoop *SplitAnalysis::getBestSplitLoop() {
  assert(curli_ && "Call analyze() before getBestSplitLoop");
  if (usingLoops_.empty())
    return 0;

  LoopPtrSet Loops, SecondLoops;
  LoopBlocks Blocks;
  BlockPtrSet CriticalExits;

  // Find first-class and second class candidate loops.
  // We prefer to split around loops where curli is used outside the periphery.
  for (LoopCountMap::const_iterator I = usingLoops_.begin(),
       E = usingLoops_.end(); I != E; ++I) {
    const MachineLoop *Loop = I->first;
    getLoopBlocks(Loop, Blocks);

    // FIXME: We need an SSA updater to properly handle multiple exit blocks.
    if (Blocks.Exits.size() > 1) {
      DEBUG(dbgs() << "  multiple exits from " << *Loop);
      continue;
    }

    LoopPtrSet *LPS = 0;
    switch(analyzeLoopPeripheralUse(Blocks)) {
    case OutsideLoop:
      LPS = &Loops;
      break;
    case MultiPeripheral:
      LPS = &SecondLoops;
      break;
    case ContainedInLoop:
      DEBUG(dbgs() << "  contained in " << *Loop);
      continue;
    case SinglePeripheral:
      DEBUG(dbgs() << "  single peripheral use in " << *Loop);
      continue;
    }
    // Will it be possible to split around this loop?
    getCriticalExits(Blocks, CriticalExits);
    DEBUG(dbgs() << "  " << CriticalExits.size() << " critical exits from "
                 << *Loop);
    if (!canSplitCriticalExits(Blocks, CriticalExits))
      continue;
    // This is a possible split.
    assert(LPS);
    LPS->insert(Loop);
  }

  DEBUG(dbgs() << "  getBestSplitLoop found " << Loops.size() << " + "
               << SecondLoops.size() << " candidate loops.\n");

  // If there are no first class loops available, look at second class loops.
  if (Loops.empty())
    Loops = SecondLoops;

  if (Loops.empty())
    return 0;

  // Pick the earliest loop.
  // FIXME: Are there other heuristics to consider?
  const MachineLoop *Best = 0;
  SlotIndex BestIdx;
  for (LoopPtrSet::const_iterator I = Loops.begin(), E = Loops.end(); I != E;
       ++I) {
    SlotIndex Idx = lis_.getMBBStartIdx((*I)->getHeader());
    if (!Best || Idx < BestIdx)
      Best = *I, BestIdx = Idx;
  }
  DEBUG(dbgs() << "  getBestSplitLoop found " << *Best);
  return Best;
}

/// getMultiUseBlocks - if curli has more than one use in a basic block, it
/// may be an advantage to split curli for the duration of the block.
bool SplitAnalysis::getMultiUseBlocks(BlockPtrSet &Blocks) {
  // If curli is local to one block, there is no point to splitting it.
  if (usingBlocks_.size() <= 1)
    return false;
  // Add blocks with multiple uses.
  for (BlockCountMap::iterator I = usingBlocks_.begin(), E = usingBlocks_.end();
       I != E; ++I)
    switch (I->second) {
    case 0:
    case 1:
      continue;
    case 2: {
      // It doesn't pay to split a 2-instr block if it redefines curli.
      VNInfo *VN1 = curli_->getVNInfoAt(lis_.getMBBStartIdx(I->first));
      VNInfo *VN2 =
        curli_->getVNInfoAt(lis_.getMBBEndIdx(I->first).getPrevIndex());
      // live-in and live-out with a different value.
      if (VN1 && VN2 && VN1 != VN2)
        continue;
    } // Fall through.
    default:
      Blocks.insert(I->first);
    }
  return !Blocks.empty();
}

//===----------------------------------------------------------------------===//
//                               Split Editor
//===----------------------------------------------------------------------===//

/// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
SplitEditor::SplitEditor(SplitAnalysis &sa, LiveIntervals &lis, VirtRegMap &vrm,
                         std::vector<LiveInterval*> &intervals)
  : sa_(sa), lis_(lis), vrm_(vrm),
    mri_(vrm.getMachineFunction().getRegInfo()),
    tii_(*vrm.getMachineFunction().getTarget().getInstrInfo()),
    curli_(sa_.getCurLI()),
    dupli_(0), openli_(0),
    intervals_(intervals),
    firstInterval(intervals_.size())
{
  assert(curli_ && "SplitEditor created from empty SplitAnalysis");

  // Make sure curli_ is assigned a stack slot, so all our intervals get the
  // same slot as curli_.
  if (vrm_.getStackSlot(curli_->reg) == VirtRegMap::NO_STACK_SLOT)
    vrm_.assignVirt2StackSlot(curli_->reg);

}

LiveInterval *SplitEditor::createInterval() {
  unsigned curli = sa_.getCurLI()->reg;
  unsigned Reg = mri_.createVirtualRegister(mri_.getRegClass(curli));
  LiveInterval &Intv = lis_.getOrCreateInterval(Reg);
  vrm_.grow();
  vrm_.assignVirt2StackSlot(Reg, vrm_.getStackSlot(curli));
  return &Intv;
}

LiveInterval *SplitEditor::getDupLI() {
  if (!dupli_) {
    // Create an interval for dupli that is a copy of curli.
    dupli_ = createInterval();
    dupli_->Copy(*curli_, &mri_, lis_.getVNInfoAllocator());
  }
  return dupli_;
}

VNInfo *SplitEditor::mapValue(const VNInfo *curliVNI) {
  VNInfo *&VNI = valueMap_[curliVNI];
  if (!VNI)
    VNI = openli_->createValueCopy(curliVNI, lis_.getVNInfoAllocator());
  return VNI;
}

/// Insert a COPY instruction curli -> li. Allocate a new value from li
/// defined by the COPY. Note that rewrite() will deal with the curli
/// register, so this function can be used to copy from any interval - openli,
/// curli, or dupli.
VNInfo *SplitEditor::insertCopy(LiveInterval &LI,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) {
  MachineInstr *MI = BuildMI(MBB, I, DebugLoc(), tii_.get(TargetOpcode::COPY),
                             LI.reg).addReg(curli_->reg);
  SlotIndex DefIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();
  return LI.getNextValue(DefIdx, MI, true, lis_.getVNInfoAllocator());
}

/// Create a new virtual register and live interval.
void SplitEditor::openIntv() {
  assert(!openli_ && "Previous LI not closed before openIntv");
  openli_ = createInterval();
  intervals_.push_back(openli_);
  liveThrough_ = false;
}

/// enterIntvBefore - Enter openli before the instruction at Idx. If curli is
/// not live before Idx, a COPY is not inserted.
void SplitEditor::enterIntvBefore(SlotIndex Idx) {
  assert(openli_ && "openIntv not called before enterIntvBefore");

  // Copy from curli_ if it is live.
  if (VNInfo *CurVNI = curli_->getVNInfoAt(Idx.getUseIndex())) {
    MachineInstr *MI = lis_.getInstructionFromIndex(Idx);
    assert(MI && "enterIntvBefore called with invalid index");
    VNInfo *VNI = insertCopy(*openli_, *MI->getParent(), MI);
    openli_->addRange(LiveRange(VNI->def, Idx.getDefIndex(), VNI));

    // Make sure CurVNI is properly mapped.
    VNInfo *&mapVNI = valueMap_[CurVNI];
    // We dont have SSA update yet, so only one entry per value is allowed.
    assert(!mapVNI && "enterIntvBefore called more than once for the same value");
    mapVNI = VNI;
  }
  DEBUG(dbgs() << "    enterIntvBefore " << Idx << ": " << *openli_ << '\n');
}

/// enterIntvAtEnd - Enter openli at the end of MBB.
/// PhiMBB is a successor inside openli where a PHI value is created.
/// Currently, all entries must share the same PhiMBB.
void SplitEditor::enterIntvAtEnd(MachineBasicBlock &A, MachineBasicBlock &B) {
  assert(openli_ && "openIntv not called before enterIntvAtEnd");

  SlotIndex EndA = lis_.getMBBEndIdx(&A);
  VNInfo *CurVNIA = curli_->getVNInfoAt(EndA.getPrevIndex());
  if (!CurVNIA) {
    DEBUG(dbgs() << "    enterIntvAtEnd, curli not live out of BB#"
                 << A.getNumber() << ".\n");
    return;
  }

  // Add a phi kill value and live range out of A.
  VNInfo *VNIA = insertCopy(*openli_, A, A.getFirstTerminator());
  openli_->addRange(LiveRange(VNIA->def, EndA, VNIA));

  // FIXME: If this is the only entry edge, we don't need the extra PHI value.
  // FIXME: If there are multiple entry blocks (so not a loop), we need proper
  // SSA update.

  // Now look at the start of B.
  SlotIndex StartB = lis_.getMBBStartIdx(&B);
  SlotIndex EndB = lis_.getMBBEndIdx(&B);
  const LiveRange *CurB = curli_->getLiveRangeContaining(StartB);
  if (!CurB) {
    DEBUG(dbgs() << "    enterIntvAtEnd: curli not live in to BB#"
                 << B.getNumber() << ".\n");
    return;
  }

  VNInfo *VNIB = openli_->getVNInfoAt(StartB);
  if (!VNIB) {
    // Create a phi value.
    VNIB = openli_->getNextValue(SlotIndex(StartB, true), 0, false,
                                 lis_.getVNInfoAllocator());
    VNIB->setIsPHIDef(true);
    VNInfo *&mapVNI = valueMap_[CurB->valno];
    if (mapVNI) {
      // Multiple copies - must create PHI value.
      abort();
    } else {
      // This is the first copy of dupLR. Mark the mapping.
      mapVNI = VNIB;
    }

  }

  DEBUG(dbgs() << "    enterIntvAtEnd: " << *openli_ << '\n');
}

/// useIntv - indicate that all instructions in MBB should use openli.
void SplitEditor::useIntv(const MachineBasicBlock &MBB) {
  useIntv(lis_.getMBBStartIdx(&MBB), lis_.getMBBEndIdx(&MBB));
}

void SplitEditor::useIntv(SlotIndex Start, SlotIndex End) {
  assert(openli_ && "openIntv not called before useIntv");

  // Map the curli values from the interval into openli_
  LiveInterval::const_iterator B = curli_->begin(), E = curli_->end();
  LiveInterval::const_iterator I = std::lower_bound(B, E, Start);

  if (I != B) {
    --I;
    // I begins before Start, but overlaps.
    if (I->end > Start)
      openli_->addRange(LiveRange(Start, std::min(End, I->end),
                        mapValue(I->valno)));
    ++I;
  }

  // The remaining ranges begin after Start.
  for (;I != E && I->start < End; ++I)
    openli_->addRange(LiveRange(I->start, std::min(End, I->end),
                                mapValue(I->valno)));
  DEBUG(dbgs() << "    use [" << Start << ';' << End << "): " << *openli_
               << '\n');
}

/// leaveIntvAfter - Leave openli after the instruction at Idx.
void SplitEditor::leaveIntvAfter(SlotIndex Idx) {
  assert(openli_ && "openIntv not called before leaveIntvAfter");

  const LiveRange *CurLR = curli_->getLiveRangeContaining(Idx.getDefIndex());
  if (!CurLR || CurLR->end <= Idx.getBoundaryIndex()) {
    DEBUG(dbgs() << "    leaveIntvAfter " << Idx << ": not live\n");
    return;
  }

  // Was this value of curli live through openli?
  if (!openli_->liveAt(CurLR->valno->def)) {
    DEBUG(dbgs() << "    leaveIntvAfter " << Idx << ": using external value\n");
    liveThrough_ = true;
    return;
  }

  // We are going to insert a back copy, so we must have a dupli_.
  LiveRange *DupLR = getDupLI()->getLiveRangeContaining(Idx.getDefIndex());
  assert(DupLR && "dupli not live into black, but curli is?");

  // Insert the COPY instruction.
  MachineBasicBlock::iterator I = lis_.getInstructionFromIndex(Idx);
  MachineInstr *MI = BuildMI(*I->getParent(), llvm::next(I), I->getDebugLoc(),
                             tii_.get(TargetOpcode::COPY), dupli_->reg)
                       .addReg(openli_->reg);
  SlotIndex CopyIdx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();
  openli_->addRange(LiveRange(Idx.getDefIndex(), CopyIdx,
                    mapValue(CurLR->valno)));
  DupLR->valno->def = CopyIdx;
  DEBUG(dbgs() << "    leaveIntvAfter " << Idx << ": " << *openli_ << '\n');
}

/// leaveIntvAtTop - Leave the interval at the top of MBB.
/// Currently, only one value can leave the interval.
void SplitEditor::leaveIntvAtTop(MachineBasicBlock &MBB) {
  assert(openli_ && "openIntv not called before leaveIntvAtTop");

  SlotIndex Start = lis_.getMBBStartIdx(&MBB);
  const LiveRange *CurLR = curli_->getLiveRangeContaining(Start);

  // Is curli even live-in to MBB?
  if (!CurLR) {
    DEBUG(dbgs() << "    leaveIntvAtTop at " << Start << ": not live\n");
    return;
  }

  // Is curli defined by PHI at the beginning of MBB?
  bool isPHIDef = CurLR->valno->isPHIDef() &&
                  CurLR->valno->def.getBaseIndex() == Start;

  // If MBB is using a value of curli that was defined outside the openli range,
  // we don't want to copy it back here.
  if (!isPHIDef && !openli_->liveAt(CurLR->valno->def)) {
    DEBUG(dbgs() << "    leaveIntvAtTop at " << Start
                 << ": using external value\n");
    liveThrough_ = true;
    return;
  }

  // We are going to insert a back copy, so we must have a dupli_.
  LiveRange *DupLR = getDupLI()->getLiveRangeContaining(Start);
  assert(DupLR && "dupli not live into black, but curli is?");

  // Insert the COPY instruction.
  MachineInstr *MI = BuildMI(MBB, MBB.begin(), DebugLoc(),
                             tii_.get(TargetOpcode::COPY), dupli_->reg)
                       .addReg(openli_->reg);
  SlotIndex Idx = lis_.InsertMachineInstrInMaps(MI).getDefIndex();

  // Adjust dupli and openli values.
  if (isPHIDef) {
    // dupli was already a PHI on entry to MBB. Simply insert an openli PHI,
    // and shift the dupli def down to the COPY.
    VNInfo *VNI = openli_->getNextValue(SlotIndex(Start, true), 0, false,
                                        lis_.getVNInfoAllocator());
    VNI->setIsPHIDef(true);
    openli_->addRange(LiveRange(VNI->def, Idx, VNI));

    dupli_->removeRange(Start, Idx);
    DupLR->valno->def = Idx;
    DupLR->valno->setIsPHIDef(false);
  } else {
    // The dupli value was defined somewhere inside the openli range.
    DEBUG(dbgs() << "    leaveIntvAtTop source value defined at "
                 << DupLR->valno->def << "\n");
    // FIXME: We may not need a PHI here if all predecessors have the same
    // value.
    VNInfo *VNI = openli_->getNextValue(SlotIndex(Start, true), 0, false,
                                        lis_.getVNInfoAllocator());
    VNI->setIsPHIDef(true);
    openli_->addRange(LiveRange(VNI->def, Idx, VNI));

    // FIXME: What if DupLR->valno is used by multiple exits? SSA Update.

    // closeIntv is going to remove the superfluous live ranges.
    DupLR->valno->def = Idx;
    DupLR->valno->setIsPHIDef(false);
  }

  DEBUG(dbgs() << "    leaveIntvAtTop at " << Idx << ": " << *openli_ << '\n');
}

/// closeIntv - Indicate that we are done editing the currently open
/// LiveInterval, and ranges can be trimmed.
void SplitEditor::closeIntv() {
  assert(openli_ && "openIntv not called before closeIntv");

  DEBUG(dbgs() << "    closeIntv cleaning up\n");
  DEBUG(dbgs() << "    open " << *openli_ << '\n');

  if (liveThrough_) {
    DEBUG(dbgs() << "    value live through region, leaving dupli as is.\n");
  } else {
    // live out with copies inserted, or killed by region. Either way we need to
    // remove the overlapping region from dupli.
    getDupLI();
    for (LiveInterval::iterator I = openli_->begin(), E = openli_->end();
         I != E; ++I) {
      dupli_->removeRange(I->start, I->end);
    }
    // FIXME: A block branching to the entry block may also branch elsewhere
    // curli is live. We need both openli and curli to be live in that case.
    DEBUG(dbgs() << "    dup2 " << *dupli_ << '\n');
  }
  openli_ = 0;
  valueMap_.clear();
}

/// rewrite - after all the new live ranges have been created, rewrite
/// instructions using curli to use the new intervals.
void SplitEditor::rewrite() {
  assert(!openli_ && "Previous LI not closed before rewrite");
  const LiveInterval *curli = sa_.getCurLI();
  for (MachineRegisterInfo::reg_iterator RI = mri_.reg_begin(curli->reg),
       RE = mri_.reg_end(); RI != RE;) {
    MachineOperand &MO = RI.getOperand();
    MachineInstr *MI = MO.getParent();
    ++RI;
    if (MI->isDebugValue()) {
      DEBUG(dbgs() << "Zapping " << *MI);
      // FIXME: We can do much better with debug values.
      MO.setReg(0);
      continue;
    }
    SlotIndex Idx = lis_.getInstructionIndex(MI);
    Idx = MO.isUse() ? Idx.getUseIndex() : Idx.getDefIndex();
    LiveInterval *LI = dupli_;
    for (unsigned i = firstInterval, e = intervals_.size(); i != e; ++i) {
      LiveInterval *testli = intervals_[i];
      if (testli->liveAt(Idx)) {
        LI = testli;
        break;
      }
    }
    if (LI) {
      MO.setReg(LI->reg);
      sa_.removeUse(MI);
      DEBUG(dbgs() << "  rewrite " << Idx << '\t' << *MI);
    }
  }

  // dupli_ goes in last, after rewriting.
  if (dupli_) {
    if (dupli_->empty()) {
      DEBUG(dbgs() << "  dupli became empty?\n");
      lis_.removeInterval(dupli_->reg);
      dupli_ = 0;
    } else {
      dupli_->RenumberValues(lis_);
      intervals_.push_back(dupli_);
    }
  }

  // Calculate spill weight and allocation hints for new intervals.
  VirtRegAuxInfo vrai(vrm_.getMachineFunction(), lis_, sa_.loops_);
  for (unsigned i = firstInterval, e = intervals_.size(); i != e; ++i) {
    LiveInterval &li = *intervals_[i];
    vrai.CalculateRegClass(li.reg);
    vrai.CalculateWeightAndHint(li);
    DEBUG(dbgs() << "  new interval " << mri_.getRegClass(li.reg)->getName()
                 << ":" << li << '\n');
  }
}


//===----------------------------------------------------------------------===//
//                               Loop Splitting
//===----------------------------------------------------------------------===//

bool SplitEditor::splitAroundLoop(const MachineLoop *Loop) {
  SplitAnalysis::LoopBlocks Blocks;
  sa_.getLoopBlocks(Loop, Blocks);

  // Break critical edges as needed.
  SplitAnalysis::BlockPtrSet CriticalExits;
  sa_.getCriticalExits(Blocks, CriticalExits);
  assert(CriticalExits.empty() && "Cannot break critical exits yet");

  // Create new live interval for the loop.
  openIntv();

  // Insert copies in the predecessors.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Preds.begin(),
       E = Blocks.Preds.end(); I != E; ++I) {
    MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
    enterIntvAtEnd(MBB, *Loop->getHeader());
  }

  // Switch all loop blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Loop.begin(),
       E = Blocks.Loop.end(); I != E; ++I)
     useIntv(**I);

  // Insert back copies in the exit blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Exits.begin(),
       E = Blocks.Exits.end(); I != E; ++I) {
    MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
    leaveIntvAtTop(MBB);
  }

  // Done.
  closeIntv();
  rewrite();
  return dupli_;
}


//===----------------------------------------------------------------------===//
//                            Single Block Splitting
//===----------------------------------------------------------------------===//

/// splitSingleBlocks - Split curli into a separate live interval inside each
/// basic block in Blocks. Return true if curli has been completely replaced,
/// false if curli is still intact, and needs to be spilled or split further.
bool SplitEditor::splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks) {
  DEBUG(dbgs() << "  splitSingleBlocks for " << Blocks.size() << " blocks.\n");
  // Determine the first and last instruction using curli in each block.
  typedef std::pair<SlotIndex,SlotIndex> IndexPair;
  typedef DenseMap<const MachineBasicBlock*,IndexPair> IndexPairMap;
  IndexPairMap MBBRange;
  for (SplitAnalysis::InstrPtrSet::const_iterator I = sa_.usingInstrs_.begin(),
       E = sa_.usingInstrs_.end(); I != E; ++I) {
    const MachineBasicBlock *MBB = (*I)->getParent();
    if (!Blocks.count(MBB))
      continue;
    SlotIndex Idx = lis_.getInstructionIndex(*I);
    DEBUG(dbgs() << "  BB#" << MBB->getNumber() << '\t' << Idx << '\t' << **I);
    IndexPair &IP = MBBRange[MBB];
    if (!IP.first.isValid() || Idx < IP.first)
      IP.first = Idx;
    if (!IP.second.isValid() || Idx > IP.second)
      IP.second = Idx;
  }

  // Create a new interval for each block.
  for (SplitAnalysis::BlockPtrSet::const_iterator I = Blocks.begin(),
       E = Blocks.end(); I != E; ++I) {
    IndexPair &IP = MBBRange[*I];
    DEBUG(dbgs() << "  splitting for BB#" << (*I)->getNumber() << ": ["
                 << IP.first << ';' << IP.second << ")\n");
    assert(IP.first.isValid() && IP.second.isValid());

    openIntv();
    enterIntvBefore(IP.first);
    useIntv(IP.first.getBaseIndex(), IP.second.getBoundaryIndex());
    leaveIntvAfter(IP.second);
    closeIntv();
  }
  rewrite();
  return dupli_;
}


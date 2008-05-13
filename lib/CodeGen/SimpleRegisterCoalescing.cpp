//===-- SimpleRegisterCoalescing.cpp - Register Coalescing ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register coalescing pass that attempts to
// aggressively coalesce every register copy that it can.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regcoalescing"
#include "SimpleRegisterCoalescing.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/Value.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

STATISTIC(numJoins    , "Number of interval joins performed");
STATISTIC(numCommutes , "Number of instruction commuting performed");
STATISTIC(numExtends  , "Number of copies extended");
STATISTIC(numPeep     , "Number of identity moves eliminated after coalescing");
STATISTIC(numAborts   , "Number of times interval joining aborted");

char SimpleRegisterCoalescing::ID = 0;
static cl::opt<bool>
EnableJoining("join-liveintervals",
              cl::desc("Coalesce copies (default=true)"),
              cl::init(true));

static cl::opt<bool>
NewHeuristic("new-coalescer-heuristic",
              cl::desc("Use new coalescer heuristic"),
              cl::init(false));

static RegisterPass<SimpleRegisterCoalescing> 
X("simple-register-coalescing", "Simple Register Coalescing");

// Declare that we implement the RegisterCoalescer interface
static RegisterAnalysisGroup<RegisterCoalescer, true/*The Default*/> V(X);

const PassInfo *llvm::SimpleRegisterCoalescingID = X.getPassInfo();

void SimpleRegisterCoalescing::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<LiveIntervals>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreservedID(MachineDominatorsID);
  AU.addPreservedID(PHIEliminationID);
  AU.addPreservedID(TwoAddressInstructionPassID);
  AU.addRequired<LiveVariables>();
  AU.addRequired<LiveIntervals>();
  AU.addRequired<MachineLoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// AdjustCopiesBackFrom - We found a non-trivially-coalescable copy with IntA
/// being the source and IntB being the dest, thus this defines a value number
/// in IntB.  If the source value number (in IntA) is defined by a copy from B,
/// see if we can merge these two pieces of B into a single value number,
/// eliminating a copy.  For example:
///
///  A3 = B0
///    ...
///  B1 = A3      <- this copy
///
/// In this case, B0 can be extended to where the B1 copy lives, allowing the B1
/// value number to be replaced with B0 (which simplifies the B liveinterval).
///
/// This returns true if an interval was modified.
///
bool SimpleRegisterCoalescing::AdjustCopiesBackFrom(LiveInterval &IntA,
                                                    LiveInterval &IntB,
                                                    MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));

  // BValNo is a value number in B that is defined by a copy from A.  'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  if (BLR == IntB.end()) // Should never happen!
    return false;
  VNInfo *BValNo = BLR->valno;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  if (!BValNo->copy) return false;
  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A3 in the example.
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyIdx-1);
  if (ALR == IntA.end()) // Should never happen!
    return false;
  VNInfo *AValNo = ALR->valno;
  
  // If AValNo is defined as a copy from IntB, we can potentially process this.  
  // Get the instruction that defines this value number.
  unsigned SrcReg = li_->getVNInfoSourceReg(AValNo);
  if (!SrcReg) return false;  // Not defined by a copy.
    
  // If the value number is not defined by a copy instruction, ignore it.

  // If the source register comes from an interval other than IntB, we can't
  // handle this.
  if (SrcReg != IntB.reg) return false;
  
  // Get the LiveRange in IntB that this value number starts with.
  LiveInterval::iterator ValLR = IntB.FindLiveRangeContaining(AValNo->def-1);
  if (ValLR == IntB.end()) // Should never happen!
    return false;
  
  // Make sure that the end of the live range is inside the same block as
  // CopyMI.
  MachineInstr *ValLREndInst = li_->getInstructionFromIndex(ValLR->end-1);
  if (!ValLREndInst || 
      ValLREndInst->getParent() != CopyMI->getParent()) return false;

  // Okay, we now know that ValLR ends in the same block that the CopyMI
  // live-range starts.  If there are no intervening live ranges between them in
  // IntB, we can merge them.
  if (ValLR+1 != BLR) return false;

  // If a live interval is a physical register, conservatively check if any
  // of its sub-registers is overlapping the live interval of the virtual
  // register. If so, do not coalesce.
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg) &&
      *tri_->getSubRegisters(IntB.reg)) {
    for (const unsigned* SR = tri_->getSubRegisters(IntB.reg); *SR; ++SR)
      if (li_->hasInterval(*SR) && IntA.overlaps(li_->getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
        return false;
      }
  }
  
  DOUT << "\nExtending: "; IntB.print(DOUT, tri_);
  
  unsigned FillerStart = ValLR->end, FillerEnd = BLR->start;
  // We are about to delete CopyMI, so need to remove it as the 'instruction
  // that defines this value #'. Update the the valnum with the new defining
  // instruction #.
  BValNo->def  = FillerStart;
  BValNo->copy = NULL;
  
  // Okay, we can merge them.  We need to insert a new liverange:
  // [ValLR.end, BLR.begin) of either value number, then we merge the
  // two value numbers.
  IntB.addRange(LiveRange(FillerStart, FillerEnd, BValNo));

  // If the IntB live range is assigned to a physical register, and if that
  // physreg has aliases, 
  if (TargetRegisterInfo::isPhysicalRegister(IntB.reg)) {
    // Update the liveintervals of sub-registers.
    for (const unsigned *AS = tri_->getSubRegisters(IntB.reg); *AS; ++AS) {
      LiveInterval &AliasLI = li_->getInterval(*AS);
      AliasLI.addRange(LiveRange(FillerStart, FillerEnd,
              AliasLI.getNextValue(FillerStart, 0, li_->getVNInfoAllocator())));
    }
  }

  // Okay, merge "B1" into the same value number as "B0".
  if (BValNo != ValLR->valno)
    IntB.MergeValueNumberInto(BValNo, ValLR->valno);
  DOUT << "   result = "; IntB.print(DOUT, tri_);
  DOUT << "\n";

  // If the source instruction was killing the source register before the
  // merge, unset the isKill marker given the live range has been extended.
  int UIdx = ValLREndInst->findRegisterUseOperandIdx(IntB.reg, true);
  if (UIdx != -1)
    ValLREndInst->getOperand(UIdx).setIsKill(false);

  ++numExtends;
  return true;
}

/// HasOtherReachingDefs - Return true if there are definitions of IntB
/// other than BValNo val# that can reach uses of AValno val# of IntA.
bool SimpleRegisterCoalescing::HasOtherReachingDefs(LiveInterval &IntA,
                                                    LiveInterval &IntB,
                                                    VNInfo *AValNo,
                                                    VNInfo *BValNo) {
  for (LiveInterval::iterator AI = IntA.begin(), AE = IntA.end();
       AI != AE; ++AI) {
    if (AI->valno != AValNo) continue;
    LiveInterval::Ranges::iterator BI =
      std::upper_bound(IntB.ranges.begin(), IntB.ranges.end(), AI->start);
    if (BI != IntB.ranges.begin())
      --BI;
    for (; BI != IntB.ranges.end() && AI->end >= BI->start; ++BI) {
      if (BI->valno == BValNo)
        continue;
      if (BI->start <= AI->start && BI->end > AI->start)
        return true;
      if (BI->start > AI->start && BI->start < AI->end)
        return true;
    }
  }
  return false;
}

/// RemoveCopyByCommutingDef - We found a non-trivially-coalescable copy with IntA
/// being the source and IntB being the dest, thus this defines a value number
/// in IntB.  If the source value number (in IntA) is defined by a commutable
/// instruction and its other operand is coalesced to the copy dest register,
/// see if we can transform the copy into a noop by commuting the definition. For
/// example,
///
///  A3 = op A2 B0<kill>
///    ...
///  B1 = A3      <- this copy
///    ...
///     = op A3   <- more uses
///
/// ==>
///
///  B2 = op B0 A2<kill>
///    ...
///  B1 = B2      <- now an identify copy
///    ...
///     = op B2   <- more uses
///
/// This returns true if an interval was modified.
///
bool SimpleRegisterCoalescing::RemoveCopyByCommutingDef(LiveInterval &IntA,
                                                        LiveInterval &IntB,
                                                        MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));

  // FIXME: For now, only eliminate the copy by commuting its def when the
  // source register is a virtual register. We want to guard against cases
  // where the copy is a back edge copy and commuting the def lengthen the
  // live interval of the source register to the entire loop.
  if (TargetRegisterInfo::isPhysicalRegister(IntA.reg))
    return false;

  // BValNo is a value number in B that is defined by a copy from A. 'B3' in
  // the example above.
  LiveInterval::iterator BLR = IntB.FindLiveRangeContaining(CopyIdx);
  if (BLR == IntB.end()) // Should never happen!
    return false;
  VNInfo *BValNo = BLR->valno;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  if (!BValNo->copy) return false;
  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A3 in the example.
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyIdx-1);
  if (ALR == IntA.end()) // Should never happen!
    return false;
  VNInfo *AValNo = ALR->valno;
  // If other defs can reach uses of this def, then it's not safe to perform
  // the optimization.
  if (AValNo->def == ~0U || AValNo->def == ~1U || AValNo->hasPHIKill)
    return false;
  MachineInstr *DefMI = li_->getInstructionFromIndex(AValNo->def);
  const TargetInstrDesc &TID = DefMI->getDesc();
  unsigned NewDstIdx;
  if (!TID.isCommutable() ||
      !tii_->CommuteChangesDestination(DefMI, NewDstIdx))
    return false;

  MachineOperand &NewDstMO = DefMI->getOperand(NewDstIdx);
  unsigned NewReg = NewDstMO.getReg();
  if (NewReg != IntB.reg || !NewDstMO.isKill())
    return false;

  // Make sure there are no other definitions of IntB that would reach the
  // uses which the new definition can reach.
  if (HasOtherReachingDefs(IntA, IntB, AValNo, BValNo))
    return false;

  // If some of the uses of IntA.reg is already coalesced away, return false.
  // It's not possible to determine whether it's safe to perform the coalescing.
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(IntA.reg),
         UE = mri_->use_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    unsigned UseIdx = li_->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end())
      continue;
    if (ULR->valno == AValNo && JoinedCopies.count(UseMI))
      return false;
  }

  // At this point we have decided that it is legal to do this
  // transformation.  Start by commuting the instruction.
  MachineBasicBlock *MBB = DefMI->getParent();
  MachineInstr *NewMI = tii_->commuteInstruction(DefMI);
  if (!NewMI)
    return false;
  if (NewMI != DefMI) {
    li_->ReplaceMachineInstrInMaps(DefMI, NewMI);
    MBB->insert(DefMI, NewMI);
    MBB->erase(DefMI);
  }
  unsigned OpIdx = NewMI->findRegisterUseOperandIdx(IntA.reg, false);
  NewMI->getOperand(OpIdx).setIsKill();

  bool BHasPHIKill = BValNo->hasPHIKill;
  SmallVector<VNInfo*, 4> BDeadValNos;
  SmallVector<unsigned, 4> BKills;
  std::map<unsigned, unsigned> BExtend;

  // If ALR and BLR overlaps and end of BLR extends beyond end of ALR, e.g.
  // A = or A, B
  // ...
  // B = A
  // ...
  // C = A<kill>
  // ...
  //   = B
  //
  // then do not add kills of A to the newly created B interval.
  bool Extended = BLR->end > ALR->end && ALR->end != ALR->start;
  if (Extended)
    BExtend[ALR->end] = BLR->end;

  // Update uses of IntA of the specific Val# with IntB.
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(IntA.reg),
         UE = mri_->use_end(); UI != UE;) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    ++UI;
    if (JoinedCopies.count(UseMI))
      continue;
    unsigned UseIdx = li_->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR == IntA.end() || ULR->valno != AValNo)
      continue;
    UseMO.setReg(NewReg);
    if (UseMI == CopyMI)
      continue;
    if (UseMO.isKill()) {
      if (Extended)
        UseMO.setIsKill(false);
      else
        BKills.push_back(li_->getUseIndex(UseIdx)+1);
    }
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*UseMI, SrcReg, DstReg))
      continue;
    if (DstReg == IntB.reg) {
      // This copy will become a noop. If it's defining a new val#,
      // remove that val# as well. However this live range is being
      // extended to the end of the existing live range defined by the copy.
      unsigned DefIdx = li_->getDefIndex(UseIdx);
      const LiveRange *DLR = IntB.getLiveRangeContaining(DefIdx);
      BHasPHIKill |= DLR->valno->hasPHIKill;
      assert(DLR->valno->def == DefIdx);
      BDeadValNos.push_back(DLR->valno);
      BExtend[DLR->start] = DLR->end;
      JoinedCopies.insert(UseMI);
      // If this is a kill but it's going to be removed, the last use
      // of the same val# is the new kill.
      if (UseMO.isKill())
        BKills.pop_back();
    }
  }

  // We need to insert a new liverange: [ALR.start, LastUse). It may be we can
  // simply extend BLR if CopyMI doesn't end the range.
  DOUT << "\nExtending: "; IntB.print(DOUT, tri_);

  IntB.removeValNo(BValNo);
  for (unsigned i = 0, e = BDeadValNos.size(); i != e; ++i)
    IntB.removeValNo(BDeadValNos[i]);
  VNInfo *ValNo = IntB.getNextValue(AValNo->def, 0, li_->getVNInfoAllocator());
  for (LiveInterval::iterator AI = IntA.begin(), AE = IntA.end();
       AI != AE; ++AI) {
    if (AI->valno != AValNo) continue;
    unsigned End = AI->end;
    std::map<unsigned, unsigned>::iterator EI = BExtend.find(End);
    if (EI != BExtend.end())
      End = EI->second;
    IntB.addRange(LiveRange(AI->start, End, ValNo));
  }
  IntB.addKills(ValNo, BKills);
  ValNo->hasPHIKill = BHasPHIKill;

  DOUT << "   result = "; IntB.print(DOUT, tri_);
  DOUT << "\n";

  DOUT << "\nShortening: "; IntA.print(DOUT, tri_);
  IntA.removeValNo(AValNo);
  DOUT << "   result = "; IntA.print(DOUT, tri_);
  DOUT << "\n";

  ++numCommutes;
  return true;
}

/// isBackEdgeCopy - Returns true if CopyMI is a back edge copy.
///
bool SimpleRegisterCoalescing::isBackEdgeCopy(MachineInstr *CopyMI,
                                              unsigned DstReg) const {
  MachineBasicBlock *MBB = CopyMI->getParent();
  const MachineLoop *L = loopInfo->getLoopFor(MBB);
  if (!L)
    return false;
  if (MBB != L->getLoopLatch())
    return false;

  LiveInterval &LI = li_->getInterval(DstReg);
  unsigned DefIdx = li_->getInstructionIndex(CopyMI);
  LiveInterval::const_iterator DstLR =
    LI.FindLiveRangeContaining(li_->getDefIndex(DefIdx));
  if (DstLR == LI.end())
    return false;
  unsigned KillIdx = li_->getInstructionIndex(&MBB->back()) + InstrSlots::NUM;
  if (DstLR->valno->kills.size() == 1 &&
      DstLR->valno->kills[0] == KillIdx && DstLR->valno->hasPHIKill)
    return true;
  return false;
}

/// UpdateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
/// update the subregister number if it is not zero. If DstReg is a
/// physical register and the existing subregister number of the def / use
/// being updated is not zero, make sure to set it to the correct physical
/// subregister.
void
SimpleRegisterCoalescing::UpdateRegDefsUses(unsigned SrcReg, unsigned DstReg,
                                            unsigned SubIdx) {
  bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
  if (DstIsPhys && SubIdx) {
    // Figure out the real physical register we are updating with.
    DstReg = tri_->getSubReg(DstReg, SubIdx);
    SubIdx = 0;
  }

  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(SrcReg),
         E = mri_->reg_end(); I != E; ) {
    MachineOperand &O = I.getOperand();
    MachineInstr *UseMI = &*I;
    ++I;
    unsigned OldSubIdx = O.getSubReg();
    if (DstIsPhys) {
      unsigned UseDstReg = DstReg;
      if (OldSubIdx)
          UseDstReg = tri_->getSubReg(DstReg, OldSubIdx);
      O.setReg(UseDstReg);
      O.setSubReg(0);
    } else {
      // Sub-register indexes goes from small to large. e.g.
      // RAX: 1 -> AL, 2 -> AX, 3 -> EAX
      // EAX: 1 -> AL, 2 -> AX
      // So RAX's sub-register 2 is AX, RAX's sub-regsiter 3 is EAX, whose
      // sub-register 2 is also AX.
      if (SubIdx && OldSubIdx && SubIdx != OldSubIdx)
        assert(OldSubIdx < SubIdx && "Conflicting sub-register index!");
      else if (SubIdx)
        O.setSubReg(SubIdx);
      // Remove would-be duplicated kill marker.
      if (O.isKill() && UseMI->killsRegister(DstReg))
        O.setIsKill(false);
      O.setReg(DstReg);
    }
  }
}

/// RemoveDeadImpDef - Remove implicit_def instructions which are "re-defining"
/// registers due to insert_subreg coalescing. e.g.
/// r1024 = op
/// r1025 = implicit_def
/// r1025 = insert_subreg r1025, r1024
///       = op r1025
/// =>
/// r1025 = op
/// r1025 = implicit_def
/// r1025 = insert_subreg r1025, r1025
///       = op r1025
void
SimpleRegisterCoalescing::RemoveDeadImpDef(unsigned Reg, LiveInterval &LI) {
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(Reg),
         E = mri_->reg_end(); I != E; ) {
    MachineOperand &O = I.getOperand();
    MachineInstr *DefMI = &*I;
    ++I;
    if (!O.isDef())
      continue;
    if (DefMI->getOpcode() != TargetInstrInfo::IMPLICIT_DEF)
      continue;
    if (!LI.liveBeforeAndAt(li_->getInstructionIndex(DefMI)))
      continue;
    li_->RemoveMachineInstrFromMaps(DefMI);
    DefMI->eraseFromParent();
  }
}

/// RemoveUnnecessaryKills - Remove kill markers that are no longer accurate
/// due to live range lengthening as the result of coalescing.
void SimpleRegisterCoalescing::RemoveUnnecessaryKills(unsigned Reg,
                                                      LiveInterval &LI) {
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(Reg),
         UE = mri_->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    if (UseMO.isKill()) {
      MachineInstr *UseMI = UseMO.getParent();
      unsigned SReg, DReg;
      if (!tii_->isMoveInstr(*UseMI, SReg, DReg))
        continue;
      unsigned UseIdx = li_->getUseIndex(li_->getInstructionIndex(UseMI));
      if (JoinedCopies.count(UseMI))
        continue;
      const LiveRange *UI = LI.getLiveRangeContaining(UseIdx);
      if (!LI.isKill(UI->valno, UseIdx+1))
        UseMO.setIsKill(false);
    }
  }
}

/// removeRange - Wrapper for LiveInterval::removeRange. This removes a range
/// from a physical register live interval as well as from the live intervals
/// of its sub-registers.
static void removeRange(LiveInterval &li, unsigned Start, unsigned End,
                        LiveIntervals *li_, const TargetRegisterInfo *tri_) {
  li.removeRange(Start, End, true);
  if (TargetRegisterInfo::isPhysicalRegister(li.reg)) {
    for (const unsigned* SR = tri_->getSubRegisters(li.reg); *SR; ++SR) {
      if (!li_->hasInterval(*SR))
        continue;
      LiveInterval &sli = li_->getInterval(*SR);
      unsigned RemoveEnd = Start;
      while (RemoveEnd != End) {
        LiveInterval::iterator LR = sli.FindLiveRangeContaining(Start);
        if (LR == sli.end())
          break;
        RemoveEnd = (LR->end < End) ? LR->end : End;
        sli.removeRange(Start, RemoveEnd, true);
        Start = RemoveEnd;
      }
    }
  }
}

/// removeIntervalIfEmpty - Check if the live interval of a physical register
/// is empty, if so remove it and also remove the empty intervals of its
/// sub-registers. Return true if live interval is removed.
static bool removeIntervalIfEmpty(LiveInterval &li, LiveIntervals *li_,
                                  const TargetRegisterInfo *tri_) {
  if (li.empty()) {
    if (TargetRegisterInfo::isPhysicalRegister(li.reg))
      for (const unsigned* SR = tri_->getSubRegisters(li.reg); *SR; ++SR) {
        if (!li_->hasInterval(*SR))
          continue;
        LiveInterval &sli = li_->getInterval(*SR);
        if (sli.empty())
          li_->removeInterval(*SR);
      }
    li_->removeInterval(li.reg);
    return true;
  }
  return false;
}

/// ShortenDeadCopyLiveRange - Shorten a live range defined by a dead copy.
/// Return true if live interval is removed.
bool SimpleRegisterCoalescing::ShortenDeadCopyLiveRange(LiveInterval &li,
                                                        MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getInstructionIndex(CopyMI);
  LiveInterval::iterator MLR =
    li.FindLiveRangeContaining(li_->getDefIndex(CopyIdx));
  if (MLR == li.end())
    return false;  // Already removed by ShortenDeadCopySrcLiveRange.
  unsigned RemoveStart = MLR->start;
  unsigned RemoveEnd = MLR->end;
  // Remove the liverange that's defined by this.
  if (RemoveEnd == li_->getDefIndex(CopyIdx)+1) {
    removeRange(li, RemoveStart, RemoveEnd, li_, tri_);
    return removeIntervalIfEmpty(li, li_, tri_);
  }
  return false;
}

/// PropagateDeadness - Propagate the dead marker to the instruction which
/// defines the val#.
static void PropagateDeadness(LiveInterval &li, MachineInstr *CopyMI,
                              unsigned &LRStart, LiveIntervals *li_,
                              const TargetRegisterInfo* tri_) {
  MachineInstr *DefMI =
    li_->getInstructionFromIndex(li_->getDefIndex(LRStart));
  if (DefMI && DefMI != CopyMI) {
    int DeadIdx = DefMI->findRegisterDefOperandIdx(li.reg, false, tri_);
    if (DeadIdx != -1) {
      DefMI->getOperand(DeadIdx).setIsDead();
      // A dead def should have a single cycle interval.
      ++LRStart;
    }
  }
}

/// isSameOrFallThroughBB - Return true if MBB == SuccMBB or MBB simply
/// fallthoughs to SuccMBB.
static bool isSameOrFallThroughBB(MachineBasicBlock *MBB,
                                  MachineBasicBlock *SuccMBB,
                                  const TargetInstrInfo *tii_) {
  if (MBB == SuccMBB)
    return true;
  MachineBasicBlock *TBB = 0, *FBB = 0;
  std::vector<MachineOperand> Cond;
  return !tii_->AnalyzeBranch(*MBB, TBB, FBB, Cond) && !TBB && !FBB &&
    MBB->isSuccessor(SuccMBB);
}

/// ShortenDeadCopySrcLiveRange - Shorten a live range as it's artificially
/// extended by a dead copy. Mark the last use (if any) of the val# as kill as
/// ends the live range there. If there isn't another use, then this live range
/// is dead. Return true if live interval is removed.
bool
SimpleRegisterCoalescing::ShortenDeadCopySrcLiveRange(LiveInterval &li,
                                                      MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getInstructionIndex(CopyMI);
  if (CopyIdx == 0) {
    // FIXME: special case: function live in. It can be a general case if the
    // first instruction index starts at > 0 value.
    assert(TargetRegisterInfo::isPhysicalRegister(li.reg));
    // Live-in to the function but dead. Remove it from entry live-in set.
    if (mf_->begin()->isLiveIn(li.reg))
      mf_->begin()->removeLiveIn(li.reg);
    const LiveRange *LR = li.getLiveRangeContaining(CopyIdx);
    removeRange(li, LR->start, LR->end, li_, tri_);
    return removeIntervalIfEmpty(li, li_, tri_);
  }

  LiveInterval::iterator LR = li.FindLiveRangeContaining(CopyIdx-1);
  if (LR == li.end())
    // Livein but defined by a phi.
    return false;

  unsigned RemoveStart = LR->start;
  unsigned RemoveEnd = li_->getDefIndex(CopyIdx)+1;
  if (LR->end > RemoveEnd)
    // More uses past this copy? Nothing to do.
    return false;

  MachineBasicBlock *CopyMBB = CopyMI->getParent();
  unsigned MBBStart = li_->getMBBStartIdx(CopyMBB);
  unsigned LastUseIdx;
  MachineOperand *LastUse = lastRegisterUse(LR->start, CopyIdx-1, li.reg,
                                            LastUseIdx);
  if (LastUse) {
    MachineInstr *LastUseMI = LastUse->getParent();
    if (!isSameOrFallThroughBB(LastUseMI->getParent(), CopyMBB, tii_)) {
      // r1024 = op
      // ...
      // BB1:
      //       = r1024
      //
      // BB2:
      // r1025<dead> = r1024<kill>
      if (MBBStart < LR->end)
        removeRange(li, MBBStart, LR->end, li_, tri_);
      return false;
    }

    // There are uses before the copy, just shorten the live range to the end
    // of last use.
    LastUse->setIsKill();
    removeRange(li, li_->getDefIndex(LastUseIdx), LR->end, li_, tri_);
    unsigned SrcReg, DstReg;
    if (tii_->isMoveInstr(*LastUseMI, SrcReg, DstReg) &&
        DstReg == li.reg) {
      // Last use is itself an identity code.
      int DeadIdx = LastUseMI->findRegisterDefOperandIdx(li.reg, false, tri_);
      LastUseMI->getOperand(DeadIdx).setIsDead();
    }
    return false;
  }

  // Is it livein?
  if (LR->start <= MBBStart && LR->end > MBBStart) {
    if (LR->start == 0) {
      assert(TargetRegisterInfo::isPhysicalRegister(li.reg));
      // Live-in to the function but dead. Remove it from entry live-in set.
      mf_->begin()->removeLiveIn(li.reg);
    }
    // FIXME: Shorten intervals in BBs that reaches this BB.
  }

  if (LR->valno->def == RemoveStart)
    // If the def MI defines the val#, propagate the dead marker.
    PropagateDeadness(li, CopyMI, RemoveStart, li_, tri_);

  removeRange(li, RemoveStart, LR->end, li_, tri_);
  return removeIntervalIfEmpty(li, li_, tri_);
}

/// CanCoalesceWithImpDef - Returns true if the specified copy instruction
/// from an implicit def to another register can be coalesced away.
bool SimpleRegisterCoalescing::CanCoalesceWithImpDef(MachineInstr *CopyMI,
                                                     LiveInterval &li,
                                                     LiveInterval &ImpLi) const{
  if (!CopyMI->killsRegister(ImpLi.reg))
    return false;
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));
  LiveInterval::iterator LR = li.FindLiveRangeContaining(CopyIdx);
  if (LR == li.end())
    return false;
  if (LR->valno->hasPHIKill)
    return false;
  if (LR->valno->def != CopyIdx)
    return false;
  // Make sure all of val# uses are copies.
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(li.reg),
         UE = mri_->use_end(); UI != UE;) {
    MachineInstr *UseMI = &*UI;
    ++UI;
    if (JoinedCopies.count(UseMI))
      continue;
    unsigned UseIdx = li_->getUseIndex(li_->getInstructionIndex(UseMI));
    LiveInterval::iterator ULR = li.FindLiveRangeContaining(UseIdx);
    if (ULR == li.end() || ULR->valno != LR->valno)
      continue;
    // If the use is not a use, then it's not safe to coalesce the move.
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*UseMI, SrcReg, DstReg)) {
      if (UseMI->getOpcode() == TargetInstrInfo::INSERT_SUBREG &&
          UseMI->getOperand(1).getReg() == li.reg)
        continue;
      return false;
    }
  }
  return true;
}


/// RemoveCopiesFromValNo - The specified value# is defined by an implicit
/// def and it is being removed. Turn all copies from this value# into
/// identity copies so they will be removed.
void SimpleRegisterCoalescing::RemoveCopiesFromValNo(LiveInterval &li,
                                                     VNInfo *VNI) {
  MachineInstr *ImpDef = NULL;
  MachineOperand *LastUse = NULL;
  unsigned LastUseIdx = li_->getUseIndex(VNI->def);
  for (MachineRegisterInfo::reg_iterator RI = mri_->reg_begin(li.reg),
         RE = mri_->reg_end(); RI != RE;) {
    MachineOperand *MO = &RI.getOperand();
    MachineInstr *MI = &*RI;
    ++RI;
    if (MO->isDef()) {
      if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
        assert(!ImpDef && "Multiple implicit_def defining same register?");
        ImpDef = MI;
      }
      continue;
    }
    if (JoinedCopies.count(MI))
      continue;
    unsigned UseIdx = li_->getUseIndex(li_->getInstructionIndex(MI));
    LiveInterval::iterator ULR = li.FindLiveRangeContaining(UseIdx);
    if (ULR == li.end() || ULR->valno != VNI)
      continue;
    // If the use is a copy, turn it into an identity copy.
    unsigned SrcReg, DstReg;
    if (tii_->isMoveInstr(*MI, SrcReg, DstReg) && SrcReg == li.reg) {
      // Each use MI may have multiple uses of this register. Change them all.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.getReg() == li.reg)
          MO.setReg(DstReg);
      }
      JoinedCopies.insert(MI);
    } else if (UseIdx > LastUseIdx) {
      LastUseIdx = UseIdx;
      LastUse = MO;
    }
  }
  if (LastUse)
    LastUse->setIsKill();
  else {
    // Remove dead implicit_def.
    li_->RemoveMachineInstrFromMaps(ImpDef);
    ImpDef->eraseFromParent();
  }
}

static unsigned getMatchingSuperReg(unsigned Reg, unsigned SubIdx, 
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo* TRI) {
  for (const unsigned *SRs = TRI->getSuperRegisters(Reg);
       unsigned SR = *SRs; ++SRs)
    if (Reg == TRI->getSubReg(SR, SubIdx) && RC->contains(SR))
      return SR;
  return 0;
}

/// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
/// which are the src/dst of the copy instruction CopyMI.  This returns true
/// if the copy was successfully coalesced away. If it is not currently
/// possible to coalesce this interval, but it may be possible if other
/// things get coalesced, then it returns true by reference in 'Again'.
bool SimpleRegisterCoalescing::JoinCopy(CopyRec &TheCopy, bool &Again) {
  MachineInstr *CopyMI = TheCopy.MI;

  Again = false;
  if (JoinedCopies.count(CopyMI))
    return false; // Already done.

  DOUT << li_->getInstructionIndex(CopyMI) << '\t' << *CopyMI;

  unsigned SrcReg;
  unsigned DstReg;
  bool isExtSubReg = CopyMI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG;
  bool isInsSubReg = CopyMI->getOpcode() == TargetInstrInfo::INSERT_SUBREG;
  unsigned SubIdx = 0;
  if (isExtSubReg) {
    DstReg = CopyMI->getOperand(0).getReg();
    SrcReg = CopyMI->getOperand(1).getReg();
  } else if (isInsSubReg) {
    if (CopyMI->getOperand(2).getSubReg()) {
      DOUT << "\tSource of insert_subreg is already coalesced "
           << "to another register.\n";
      return false;  // Not coalescable.
    }
    DstReg = CopyMI->getOperand(0).getReg();
    SrcReg = CopyMI->getOperand(2).getReg();
  } else if (!tii_->isMoveInstr(*CopyMI, SrcReg, DstReg)) {
    assert(0 && "Unrecognized copy instruction!");
    return false;
  }

  // If they are already joined we continue.
  if (SrcReg == DstReg) {
    DOUT << "\tCopy already coalesced.\n";
    return false;  // Not coalescable.
  }
  
  bool SrcIsPhys = TargetRegisterInfo::isPhysicalRegister(SrcReg);
  bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);

  // If they are both physical registers, we cannot join them.
  if (SrcIsPhys && DstIsPhys) {
    DOUT << "\tCan not coalesce physregs.\n";
    return false;  // Not coalescable.
  }
  
  // We only join virtual registers with allocatable physical registers.
  if (SrcIsPhys && !allocatableRegs_[SrcReg]) {
    DOUT << "\tSrc reg is unallocatable physreg.\n";
    return false;  // Not coalescable.
  }
  if (DstIsPhys && !allocatableRegs_[DstReg]) {
    DOUT << "\tDst reg is unallocatable physreg.\n";
    return false;  // Not coalescable.
  }

  unsigned RealDstReg = 0;
  unsigned RealSrcReg = 0;
  if (isExtSubReg || isInsSubReg) {
    SubIdx = CopyMI->getOperand(isExtSubReg ? 2 : 3).getImm();
    if (SrcIsPhys && isExtSubReg) {
      // r1024 = EXTRACT_SUBREG EAX, 0 then r1024 is really going to be
      // coalesced with AX.
      unsigned DstSubIdx = CopyMI->getOperand(0).getSubReg();
      if (DstSubIdx) {
        // r1024<2> = EXTRACT_SUBREG EAX, 2. Then r1024 has already been
        // coalesced to a larger register so the subreg indices cancel out.
        if (DstSubIdx != SubIdx) {
          DOUT << "\t Sub-register indices mismatch.\n";
          return false; // Not coalescable.
        }
      } else
        SrcReg = tri_->getSubReg(SrcReg, SubIdx);
      SubIdx = 0;
    } else if (DstIsPhys && isInsSubReg) {
      // EAX = INSERT_SUBREG EAX, r1024, 0
      unsigned SrcSubIdx = CopyMI->getOperand(2).getSubReg();
      if (SrcSubIdx) {
        // EAX = INSERT_SUBREG EAX, r1024<2>, 2 Then r1024 has already been
        // coalesced to a larger register so the subreg indices cancel out.
        if (SrcSubIdx != SubIdx) {
          DOUT << "\t Sub-register indices mismatch.\n";
          return false; // Not coalescable.
        }
      } else
        DstReg = tri_->getSubReg(DstReg, SubIdx);
      SubIdx = 0;
    } else if ((DstIsPhys && isExtSubReg) || (SrcIsPhys && isInsSubReg)) {
      // If this is a extract_subreg where dst is a physical register, e.g.
      // cl = EXTRACT_SUBREG reg1024, 1
      // then create and update the actual physical register allocated to RHS.
      // Ditto for
      // reg1024 = INSERT_SUBREG r1024, cl, 1
      if (CopyMI->getOperand(1).getSubReg()) {
        DOUT << "\tSrc of extract_ / insert_subreg already coalesced with reg"
             << " of a super-class.\n";
        return false; // Not coalescable.
      }
      const TargetRegisterClass *RC =
        mri_->getRegClass(isExtSubReg ? SrcReg : DstReg);
      if (isExtSubReg) {
        RealDstReg = getMatchingSuperReg(DstReg, SubIdx, RC, tri_);
        assert(RealDstReg && "Invalid extra_subreg instruction!");
      } else {
        RealSrcReg = getMatchingSuperReg(SrcReg, SubIdx, RC, tri_);
        assert(RealSrcReg && "Invalid extra_subreg instruction!");
      }

      // For this type of EXTRACT_SUBREG, conservatively
      // check if the live interval of the source register interfere with the
      // actual super physical register we are trying to coalesce with.
      unsigned PhysReg = isExtSubReg ? RealDstReg : RealSrcReg;
      LiveInterval &RHS = li_->getInterval(isExtSubReg ? SrcReg : DstReg);
      if (li_->hasInterval(PhysReg) &&
          RHS.overlaps(li_->getInterval(PhysReg))) {
        DOUT << "Interfere with register ";
        DEBUG(li_->getInterval(PhysReg).print(DOUT, tri_));
        return false; // Not coalescable
      }
      for (const unsigned* SR = tri_->getSubRegisters(PhysReg); *SR; ++SR)
        if (li_->hasInterval(*SR) && RHS.overlaps(li_->getInterval(*SR))) {
          DOUT << "Interfere with sub-register ";
          DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
          return false; // Not coalescable
        }
      SubIdx = 0;
    } else {
      unsigned OldSubIdx = isExtSubReg ? CopyMI->getOperand(0).getSubReg()
        : CopyMI->getOperand(2).getSubReg();
      if (OldSubIdx) {
        if (OldSubIdx == SubIdx && !differingRegisterClasses(SrcReg, DstReg))
          // r1024<2> = EXTRACT_SUBREG r1025, 2. Then r1024 has already been
          // coalesced to a larger register so the subreg indices cancel out.
          // Also check if the other larger register is of the same register
          // class as the would be resulting register.
          SubIdx = 0;
        else {
          DOUT << "\t Sub-register indices mismatch.\n";
          return false; // Not coalescable.
        }
      }
      if (SubIdx) {
        unsigned LargeReg = isExtSubReg ? SrcReg : DstReg;
        unsigned SmallReg = isExtSubReg ? DstReg : SrcReg;
        unsigned LargeRegSize =
          li_->getInterval(LargeReg).getSize() / InstrSlots::NUM;
        unsigned SmallRegSize =
          li_->getInterval(SmallReg).getSize() / InstrSlots::NUM;
        const TargetRegisterClass *RC = mri_->getRegClass(SmallReg);
        unsigned Threshold = allocatableRCRegs_[RC].count();
        // Be conservative. If both sides are virtual registers, do not coalesce
        // if this will cause a high use density interval to target a smaller
        // set of registers.
        if (SmallRegSize > Threshold || LargeRegSize > Threshold) {
          LiveVariables::VarInfo &svi = lv_->getVarInfo(LargeReg);
          LiveVariables::VarInfo &dvi = lv_->getVarInfo(SmallReg);
          if ((float)dvi.NumUses / SmallRegSize <
              (float)svi.NumUses / LargeRegSize) {
            Again = true;  // May be possible to coalesce later.
            return false;
          }
        }
      }
    }
  } else if (differingRegisterClasses(SrcReg, DstReg)) {
    // FIXME: What if the resul of a EXTRACT_SUBREG is then coalesced
    // with another? If it's the resulting destination register, then
    // the subidx must be propagated to uses (but only those defined
    // by the EXTRACT_SUBREG). If it's being coalesced into another
    // register, it should be safe because register is assumed to have
    // the register class of the super-register.

    // If they are not of the same register class, we cannot join them.
    DOUT << "\tSrc/Dest are different register classes.\n";
    // Allow the coalescer to try again in case either side gets coalesced to
    // a physical register that's compatible with the other side. e.g.
    // r1024 = MOV32to32_ r1025
    // but later r1024 is assigned EAX then r1025 may be coalesced with EAX.
    Again = true;  // May be possible to coalesce later.
    return false;
  }
  
  LiveInterval &SrcInt = li_->getInterval(SrcReg);
  LiveInterval &DstInt = li_->getInterval(DstReg);
  assert(SrcInt.reg == SrcReg && DstInt.reg == DstReg &&
         "Register mapping is horribly broken!");

  DOUT << "\t\tInspecting "; SrcInt.print(DOUT, tri_);
  DOUT << " and "; DstInt.print(DOUT, tri_);
  DOUT << ": ";

  // Check if it is necessary to propagate "isDead" property.
  if (!isExtSubReg && !isInsSubReg) {
    MachineOperand *mopd = CopyMI->findRegisterDefOperand(DstReg, false);
    bool isDead = mopd->isDead();

    // We need to be careful about coalescing a source physical register with a
    // virtual register. Once the coalescing is done, it cannot be broken and
    // these are not spillable! If the destination interval uses are far away,
    // think twice about coalescing them!
    if (!isDead && (SrcIsPhys || DstIsPhys)) {
      LiveInterval &JoinVInt = SrcIsPhys ? DstInt : SrcInt;
      unsigned JoinVReg = SrcIsPhys ? DstReg : SrcReg;
      unsigned JoinPReg = SrcIsPhys ? SrcReg : DstReg;
      const TargetRegisterClass *RC = mri_->getRegClass(JoinVReg);
      unsigned Threshold = allocatableRCRegs_[RC].count() * 2;
      if (TheCopy.isBackEdge)
        Threshold *= 2; // Favors back edge copies.

      // If the virtual register live interval is long but it has low use desity,
      // do not join them, instead mark the physical register as its allocation
      // preference.
      unsigned Length = JoinVInt.getSize() / InstrSlots::NUM;
      LiveVariables::VarInfo &vi = lv_->getVarInfo(JoinVReg);
      if (Length > Threshold &&
          (((float)vi.NumUses / Length) < (1.0 / Threshold))) {
        JoinVInt.preference = JoinPReg;
        ++numAborts;
        DOUT << "\tMay tie down a physical register, abort!\n";
        Again = true;  // May be possible to coalesce later.
        return false;
      }
    }
  }

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  bool Swapped = false;
  // If SrcInt is implicitly defined, it's safe to coalesce.
  bool isEmpty = SrcInt.empty();
  if (isEmpty && !CanCoalesceWithImpDef(CopyMI, DstInt, SrcInt)) {
    // Only coalesce an empty interval (defined by implicit_def) with
    // another interval which has a valno defined by the CopyMI and the CopyMI
    // is a kill of the implicit def.
    DOUT << "Not profitable!\n";
    return false;
  }

  if (!isEmpty && !JoinIntervals(DstInt, SrcInt, Swapped)) {
    // Coalescing failed.
    
    // If we can eliminate the copy without merging the live ranges, do so now.
    if (!isExtSubReg && !isInsSubReg &&
        (AdjustCopiesBackFrom(SrcInt, DstInt, CopyMI) ||
         RemoveCopyByCommutingDef(SrcInt, DstInt, CopyMI))) {
      JoinedCopies.insert(CopyMI);
      return true;
    }
    
    // Otherwise, we are unable to join the intervals.
    DOUT << "Interference!\n";
    Again = true;  // May be possible to coalesce later.
    return false;
  }

  LiveInterval *ResSrcInt = &SrcInt;
  LiveInterval *ResDstInt = &DstInt;
  if (Swapped) {
    std::swap(SrcReg, DstReg);
    std::swap(ResSrcInt, ResDstInt);
  }
  assert(TargetRegisterInfo::isVirtualRegister(SrcReg) &&
         "LiveInterval::join didn't work right!");
                               
  // If we're about to merge live ranges into a physical register live range,
  // we have to update any aliased register's live ranges to indicate that they
  // have clobbered values for this range.
  if (TargetRegisterInfo::isPhysicalRegister(DstReg)) {
    // If this is a extract_subreg where dst is a physical register, e.g.
    // cl = EXTRACT_SUBREG reg1024, 1
    // then create and update the actual physical register allocated to RHS.
    if (RealDstReg || RealSrcReg) {
      LiveInterval &RealInt =
        li_->getOrCreateInterval(RealDstReg ? RealDstReg : RealSrcReg);
      SmallSet<const VNInfo*, 4> CopiedValNos;
      for (LiveInterval::Ranges::const_iterator I = ResSrcInt->ranges.begin(),
             E = ResSrcInt->ranges.end(); I != E; ++I) {
        const LiveRange *DstLR = ResDstInt->getLiveRangeContaining(I->start);
        assert(DstLR  && "Invalid joined interval!");
        const VNInfo *DstValNo = DstLR->valno;
        if (CopiedValNos.insert(DstValNo)) {
          VNInfo *ValNo = RealInt.getNextValue(DstValNo->def, DstValNo->copy,
                                               li_->getVNInfoAllocator());
          ValNo->hasPHIKill = DstValNo->hasPHIKill;
          RealInt.addKills(ValNo, DstValNo->kills);
          RealInt.MergeValueInAsValue(*ResDstInt, DstValNo, ValNo);
        }
      }
      
      DstReg = RealDstReg ? RealDstReg : RealSrcReg;
    }

    // Update the liveintervals of sub-registers.
    for (const unsigned *AS = tri_->getSubRegisters(DstReg); *AS; ++AS)
      li_->getOrCreateInterval(*AS).MergeInClobberRanges(*ResSrcInt,
                                                 li_->getVNInfoAllocator());
  } else {
    // Merge use info if the destination is a virtual register.
    LiveVariables::VarInfo& dVI = lv_->getVarInfo(DstReg);
    LiveVariables::VarInfo& sVI = lv_->getVarInfo(SrcReg);
    dVI.NumUses += sVI.NumUses;
  }

  // If this is a EXTRACT_SUBREG, make sure the result of coalescing is the
  // larger super-register.
  if ((isExtSubReg || isInsSubReg) && !SrcIsPhys && !DstIsPhys) {
    if ((isExtSubReg && !Swapped) || (isInsSubReg && Swapped)) {
      ResSrcInt->Copy(*ResDstInt, li_->getVNInfoAllocator());
      std::swap(SrcReg, DstReg);
      std::swap(ResSrcInt, ResDstInt);
    }
  }

  if (NewHeuristic) {
    // Add all copies that define val# in the source interval into the queue.
    for (LiveInterval::const_vni_iterator i = ResSrcInt->vni_begin(),
           e = ResSrcInt->vni_end(); i != e; ++i) {
      const VNInfo *vni = *i;
      if (!vni->def || vni->def == ~1U || vni->def == ~0U)
        continue;
      MachineInstr *CopyMI = li_->getInstructionFromIndex(vni->def);
      unsigned NewSrcReg, NewDstReg;
      if (CopyMI &&
          JoinedCopies.count(CopyMI) == 0 &&
          tii_->isMoveInstr(*CopyMI, NewSrcReg, NewDstReg)) {
        unsigned LoopDepth = loopInfo->getLoopDepth(CopyMI->getParent());
        JoinQueue->push(CopyRec(CopyMI, LoopDepth,
                                isBackEdgeCopy(CopyMI, DstReg)));
      }
    }
  }

  // Remember to delete the copy instruction.
  JoinedCopies.insert(CopyMI);

  // Some live range has been lengthened due to colaescing, eliminate the
  // unnecessary kills.
  RemoveUnnecessaryKills(SrcReg, *ResDstInt);
  if (TargetRegisterInfo::isVirtualRegister(DstReg))
    RemoveUnnecessaryKills(DstReg, *ResDstInt);

  // SrcReg is guarateed to be the register whose live interval that is
  // being merged.
  li_->removeInterval(SrcReg);
  if (isInsSubReg)
    // Avoid:
    // r1024 = op
    // r1024 = implicit_def
    // ...
    //       = r1024
    RemoveDeadImpDef(DstReg, *ResDstInt);
  UpdateRegDefsUses(SrcReg, DstReg, SubIdx);

  if (isEmpty) {
    // Now the copy is being coalesced away, the val# previously defined
    // by the copy is being defined by an IMPLICIT_DEF which defines a zero
    // length interval. Remove the val#.
    unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));
    const LiveRange *LR = ResDstInt->getLiveRangeContaining(CopyIdx);
    VNInfo *ImpVal = LR->valno;
    assert(ImpVal->def == CopyIdx);
    unsigned NextDef = LR->end;
    RemoveCopiesFromValNo(*ResDstInt, ImpVal);
    ResDstInt->removeValNo(ImpVal);
    LR = ResDstInt->FindLiveRangeContaining(NextDef);
    if (LR != ResDstInt->end() && LR->valno->def == NextDef) {
      // Special case: vr1024 = implicit_def
      //               vr1024 = insert_subreg vr1024, vr1025, c
      // The insert_subreg becomes a "copy" that defines a val# which can itself
      // be coalesced away.
      MachineInstr *DefMI = li_->getInstructionFromIndex(NextDef);
      if (DefMI->getOpcode() == TargetInstrInfo::INSERT_SUBREG)
        LR->valno->copy = DefMI;
    }
  }

  DOUT << "\n\t\tJoined.  Result = "; ResDstInt->print(DOUT, tri_);
  DOUT << "\n";

  ++numJoins;
  return true;
}

/// ComputeUltimateVN - Assuming we are going to join two live intervals,
/// compute what the resultant value numbers for each value in the input two
/// ranges will be.  This is complicated by copies between the two which can
/// and will commonly cause multiple value numbers to be merged into one.
///
/// VN is the value number that we're trying to resolve.  InstDefiningValue
/// keeps track of the new InstDefiningValue assignment for the result
/// LiveInterval.  ThisFromOther/OtherFromThis are sets that keep track of
/// whether a value in this or other is a copy from the opposite set.
/// ThisValNoAssignments/OtherValNoAssignments keep track of value #'s that have
/// already been assigned.
///
/// ThisFromOther[x] - If x is defined as a copy from the other interval, this
/// contains the value number the copy is from.
///
static unsigned ComputeUltimateVN(VNInfo *VNI,
                                  SmallVector<VNInfo*, 16> &NewVNInfo,
                                  DenseMap<VNInfo*, VNInfo*> &ThisFromOther,
                                  DenseMap<VNInfo*, VNInfo*> &OtherFromThis,
                                  SmallVector<int, 16> &ThisValNoAssignments,
                                  SmallVector<int, 16> &OtherValNoAssignments) {
  unsigned VN = VNI->id;

  // If the VN has already been computed, just return it.
  if (ThisValNoAssignments[VN] >= 0)
    return ThisValNoAssignments[VN];
//  assert(ThisValNoAssignments[VN] != -2 && "Cyclic case?");

  // If this val is not a copy from the other val, then it must be a new value
  // number in the destination.
  DenseMap<VNInfo*, VNInfo*>::iterator I = ThisFromOther.find(VNI);
  if (I == ThisFromOther.end()) {
    NewVNInfo.push_back(VNI);
    return ThisValNoAssignments[VN] = NewVNInfo.size()-1;
  }
  VNInfo *OtherValNo = I->second;

  // Otherwise, this *is* a copy from the RHS.  If the other side has already
  // been computed, return it.
  if (OtherValNoAssignments[OtherValNo->id] >= 0)
    return ThisValNoAssignments[VN] = OtherValNoAssignments[OtherValNo->id];
  
  // Mark this value number as currently being computed, then ask what the
  // ultimate value # of the other value is.
  ThisValNoAssignments[VN] = -2;
  unsigned UltimateVN =
    ComputeUltimateVN(OtherValNo, NewVNInfo, OtherFromThis, ThisFromOther,
                      OtherValNoAssignments, ThisValNoAssignments);
  return ThisValNoAssignments[VN] = UltimateVN;
}

static bool InVector(VNInfo *Val, const SmallVector<VNInfo*, 8> &V) {
  return std::find(V.begin(), V.end(), Val) != V.end();
}

/// RangeIsDefinedByCopyFromReg - Return true if the specified live range of
/// the specified live interval is defined by a copy from the specified
/// register.
bool SimpleRegisterCoalescing::RangeIsDefinedByCopyFromReg(LiveInterval &li,
                                                           LiveRange *LR,
                                                           unsigned Reg) {
  unsigned SrcReg = li_->getVNInfoSourceReg(LR->valno);
  if (SrcReg == Reg)
    return true;
  if (LR->valno->def == ~0U &&
      TargetRegisterInfo::isPhysicalRegister(li.reg) &&
      *tri_->getSuperRegisters(li.reg)) {
    // It's a sub-register live interval, we may not have precise information.
    // Re-compute it.
    MachineInstr *DefMI = li_->getInstructionFromIndex(LR->start);
    unsigned SrcReg, DstReg;
    if (tii_->isMoveInstr(*DefMI, SrcReg, DstReg) &&
        DstReg == li.reg && SrcReg == Reg) {
      // Cache computed info.
      LR->valno->def  = LR->start;
      LR->valno->copy = DefMI;
      return true;
    }
  }
  return false;
}

/// SimpleJoin - Attempt to joint the specified interval into this one. The
/// caller of this method must guarantee that the RHS only contains a single
/// value number and that the RHS is not defined by a copy from this
/// interval.  This returns false if the intervals are not joinable, or it
/// joins them and returns true.
bool SimpleRegisterCoalescing::SimpleJoin(LiveInterval &LHS, LiveInterval &RHS){
  assert(RHS.containsOneValue());
  
  // Some number (potentially more than one) value numbers in the current
  // interval may be defined as copies from the RHS.  Scan the overlapping
  // portions of the LHS and RHS, keeping track of this and looking for
  // overlapping live ranges that are NOT defined as copies.  If these exist, we
  // cannot coalesce.
  
  LiveInterval::iterator LHSIt = LHS.begin(), LHSEnd = LHS.end();
  LiveInterval::iterator RHSIt = RHS.begin(), RHSEnd = RHS.end();
  
  if (LHSIt->start < RHSIt->start) {
    LHSIt = std::upper_bound(LHSIt, LHSEnd, RHSIt->start);
    if (LHSIt != LHS.begin()) --LHSIt;
  } else if (RHSIt->start < LHSIt->start) {
    RHSIt = std::upper_bound(RHSIt, RHSEnd, LHSIt->start);
    if (RHSIt != RHS.begin()) --RHSIt;
  }
  
  SmallVector<VNInfo*, 8> EliminatedLHSVals;
  
  while (1) {
    // Determine if these live intervals overlap.
    bool Overlaps = false;
    if (LHSIt->start <= RHSIt->start)
      Overlaps = LHSIt->end > RHSIt->start;
    else
      Overlaps = RHSIt->end > LHSIt->start;
    
    // If the live intervals overlap, there are two interesting cases: if the
    // LHS interval is defined by a copy from the RHS, it's ok and we record
    // that the LHS value # is the same as the RHS.  If it's not, then we cannot
    // coalesce these live ranges and we bail out.
    if (Overlaps) {
      // If we haven't already recorded that this value # is safe, check it.
      if (!InVector(LHSIt->valno, EliminatedLHSVals)) {
        // Copy from the RHS?
        if (!RangeIsDefinedByCopyFromReg(LHS, LHSIt, RHS.reg))
          return false;    // Nope, bail out.
        
        EliminatedLHSVals.push_back(LHSIt->valno);
      }
      
      // We know this entire LHS live range is okay, so skip it now.
      if (++LHSIt == LHSEnd) break;
      continue;
    }
    
    if (LHSIt->end < RHSIt->end) {
      if (++LHSIt == LHSEnd) break;
    } else {
      // One interesting case to check here.  It's possible that we have
      // something like "X3 = Y" which defines a new value number in the LHS,
      // and is the last use of this liverange of the RHS.  In this case, we
      // want to notice this copy (so that it gets coalesced away) even though
      // the live ranges don't actually overlap.
      if (LHSIt->start == RHSIt->end) {
        if (InVector(LHSIt->valno, EliminatedLHSVals)) {
          // We already know that this value number is going to be merged in
          // if coalescing succeeds.  Just skip the liverange.
          if (++LHSIt == LHSEnd) break;
        } else {
          // Otherwise, if this is a copy from the RHS, mark it as being merged
          // in.
          if (RangeIsDefinedByCopyFromReg(LHS, LHSIt, RHS.reg)) {
            EliminatedLHSVals.push_back(LHSIt->valno);

            // We know this entire LHS live range is okay, so skip it now.
            if (++LHSIt == LHSEnd) break;
          }
        }
      }
      
      if (++RHSIt == RHSEnd) break;
    }
  }
  
  // If we got here, we know that the coalescing will be successful and that
  // the value numbers in EliminatedLHSVals will all be merged together.  Since
  // the most common case is that EliminatedLHSVals has a single number, we
  // optimize for it: if there is more than one value, we merge them all into
  // the lowest numbered one, then handle the interval as if we were merging
  // with one value number.
  VNInfo *LHSValNo;
  if (EliminatedLHSVals.size() > 1) {
    // Loop through all the equal value numbers merging them into the smallest
    // one.
    VNInfo *Smallest = EliminatedLHSVals[0];
    for (unsigned i = 1, e = EliminatedLHSVals.size(); i != e; ++i) {
      if (EliminatedLHSVals[i]->id < Smallest->id) {
        // Merge the current notion of the smallest into the smaller one.
        LHS.MergeValueNumberInto(Smallest, EliminatedLHSVals[i]);
        Smallest = EliminatedLHSVals[i];
      } else {
        // Merge into the smallest.
        LHS.MergeValueNumberInto(EliminatedLHSVals[i], Smallest);
      }
    }
    LHSValNo = Smallest;
  } else if (EliminatedLHSVals.empty()) {
    if (TargetRegisterInfo::isPhysicalRegister(LHS.reg) &&
        *tri_->getSuperRegisters(LHS.reg))
      // Imprecise sub-register information. Can't handle it.
      return false;
    assert(0 && "No copies from the RHS?");
  } else {
    LHSValNo = EliminatedLHSVals[0];
  }
  
  // Okay, now that there is a single LHS value number that we're merging the
  // RHS into, update the value number info for the LHS to indicate that the
  // value number is defined where the RHS value number was.
  const VNInfo *VNI = RHS.getValNumInfo(0);
  LHSValNo->def  = VNI->def;
  LHSValNo->copy = VNI->copy;
  
  // Okay, the final step is to loop over the RHS live intervals, adding them to
  // the LHS.
  LHSValNo->hasPHIKill |= VNI->hasPHIKill;
  LHS.addKills(LHSValNo, VNI->kills);
  LHS.MergeRangesInAsValue(RHS, LHSValNo);
  LHS.weight += RHS.weight;
  if (RHS.preference && !LHS.preference)
    LHS.preference = RHS.preference;
  
  return true;
}

/// JoinIntervals - Attempt to join these two intervals.  On failure, this
/// returns false.  Otherwise, if one of the intervals being joined is a
/// physreg, this method always canonicalizes LHS to be it.  The output
/// "RHS" will not have been modified, so we can use this information
/// below to update aliases.
bool SimpleRegisterCoalescing::JoinIntervals(LiveInterval &LHS,
                                             LiveInterval &RHS, bool &Swapped) {
  // Compute the final value assignment, assuming that the live ranges can be
  // coalesced.
  SmallVector<int, 16> LHSValNoAssignments;
  SmallVector<int, 16> RHSValNoAssignments;
  DenseMap<VNInfo*, VNInfo*> LHSValsDefinedFromRHS;
  DenseMap<VNInfo*, VNInfo*> RHSValsDefinedFromLHS;
  SmallVector<VNInfo*, 16> NewVNInfo;
                          
  // If a live interval is a physical register, conservatively check if any
  // of its sub-registers is overlapping the live interval of the virtual
  // register. If so, do not coalesce.
  if (TargetRegisterInfo::isPhysicalRegister(LHS.reg) &&
      *tri_->getSubRegisters(LHS.reg)) {
    for (const unsigned* SR = tri_->getSubRegisters(LHS.reg); *SR; ++SR)
      if (li_->hasInterval(*SR) && RHS.overlaps(li_->getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
        return false;
      }
  } else if (TargetRegisterInfo::isPhysicalRegister(RHS.reg) &&
             *tri_->getSubRegisters(RHS.reg)) {
    for (const unsigned* SR = tri_->getSubRegisters(RHS.reg); *SR; ++SR)
      if (li_->hasInterval(*SR) && LHS.overlaps(li_->getInterval(*SR))) {
        DOUT << "Interfere with sub-register ";
        DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
        return false;
      }
  }
                          
  // Compute ultimate value numbers for the LHS and RHS values.
  if (RHS.containsOneValue()) {
    // Copies from a liveinterval with a single value are simple to handle and
    // very common, handle the special case here.  This is important, because
    // often RHS is small and LHS is large (e.g. a physreg).
    
    // Find out if the RHS is defined as a copy from some value in the LHS.
    int RHSVal0DefinedFromLHS = -1;
    int RHSValID = -1;
    VNInfo *RHSValNoInfo = NULL;
    VNInfo *RHSValNoInfo0 = RHS.getValNumInfo(0);
    unsigned RHSSrcReg = li_->getVNInfoSourceReg(RHSValNoInfo0);
    if ((RHSSrcReg == 0 || RHSSrcReg != LHS.reg)) {
      // If RHS is not defined as a copy from the LHS, we can use simpler and
      // faster checks to see if the live ranges are coalescable.  This joiner
      // can't swap the LHS/RHS intervals though.
      if (!TargetRegisterInfo::isPhysicalRegister(RHS.reg)) {
        return SimpleJoin(LHS, RHS);
      } else {
        RHSValNoInfo = RHSValNoInfo0;
      }
    } else {
      // It was defined as a copy from the LHS, find out what value # it is.
      RHSValNoInfo = LHS.getLiveRangeContaining(RHSValNoInfo0->def-1)->valno;
      RHSValID = RHSValNoInfo->id;
      RHSVal0DefinedFromLHS = RHSValID;
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    NewVNInfo.resize(LHS.getNumValNums(), NULL);
    
    // Okay, *all* of the values in LHS that are defined as a copy from RHS
    // should now get updated.
    for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      unsigned VN = VNI->id;
      if (unsigned LHSSrcReg = li_->getVNInfoSourceReg(VNI)) {
        if (LHSSrcReg != RHS.reg) {
          // If this is not a copy from the RHS, its value number will be
          // unmodified by the coalescing.
          NewVNInfo[VN] = VNI;
          LHSValNoAssignments[VN] = VN;
        } else if (RHSValID == -1) {
          // Otherwise, it is a copy from the RHS, and we don't already have a
          // value# for it.  Keep the current value number, but remember it.
          LHSValNoAssignments[VN] = RHSValID = VN;
          NewVNInfo[VN] = RHSValNoInfo;
          LHSValsDefinedFromRHS[VNI] = RHSValNoInfo0;
        } else {
          // Otherwise, use the specified value #.
          LHSValNoAssignments[VN] = RHSValID;
          if (VN == (unsigned)RHSValID) {  // Else this val# is dead.
            NewVNInfo[VN] = RHSValNoInfo;
            LHSValsDefinedFromRHS[VNI] = RHSValNoInfo0;
          }
        }
      } else {
        NewVNInfo[VN] = VNI;
        LHSValNoAssignments[VN] = VN;
      }
    }
    
    assert(RHSValID != -1 && "Didn't find value #?");
    RHSValNoAssignments[0] = RHSValID;
    if (RHSVal0DefinedFromLHS != -1) {
      // This path doesn't go through ComputeUltimateVN so just set
      // it to anything.
      RHSValsDefinedFromLHS[RHSValNoInfo0] = (VNInfo*)1;
    }
  } else {
    // Loop over the value numbers of the LHS, seeing if any are defined from
    // the RHS.
    for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      if (VNI->def == ~1U || VNI->copy == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the LHS interval.  If the src is
      // from the RHS interval, we can use its value #.
      if (li_->getVNInfoSourceReg(VNI) != RHS.reg)
        continue;
      
      // Figure out the value # from the RHS.
      LHSValsDefinedFromRHS[VNI]=RHS.getLiveRangeContaining(VNI->def-1)->valno;
    }
    
    // Loop over the value numbers of the RHS, seeing if any are defined from
    // the LHS.
    for (LiveInterval::vni_iterator i = RHS.vni_begin(), e = RHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      if (VNI->def == ~1U || VNI->copy == 0)  // Src not defined by a copy?
        continue;
      
      // DstReg is known to be a register in the RHS interval.  If the src is
      // from the LHS interval, we can use its value #.
      if (li_->getVNInfoSourceReg(VNI) != LHS.reg)
        continue;
      
      // Figure out the value # from the LHS.
      RHSValsDefinedFromLHS[VNI]=LHS.getLiveRangeContaining(VNI->def-1)->valno;
    }
    
    LHSValNoAssignments.resize(LHS.getNumValNums(), -1);
    RHSValNoAssignments.resize(RHS.getNumValNums(), -1);
    NewVNInfo.reserve(LHS.getNumValNums() + RHS.getNumValNums());
    
    for (LiveInterval::vni_iterator i = LHS.vni_begin(), e = LHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      unsigned VN = VNI->id;
      if (LHSValNoAssignments[VN] >= 0 || VNI->def == ~1U) 
        continue;
      ComputeUltimateVN(VNI, NewVNInfo,
                        LHSValsDefinedFromRHS, RHSValsDefinedFromLHS,
                        LHSValNoAssignments, RHSValNoAssignments);
    }
    for (LiveInterval::vni_iterator i = RHS.vni_begin(), e = RHS.vni_end();
         i != e; ++i) {
      VNInfo *VNI = *i;
      unsigned VN = VNI->id;
      if (RHSValNoAssignments[VN] >= 0 || VNI->def == ~1U)
        continue;
      // If this value number isn't a copy from the LHS, it's a new number.
      if (RHSValsDefinedFromLHS.find(VNI) == RHSValsDefinedFromLHS.end()) {
        NewVNInfo.push_back(VNI);
        RHSValNoAssignments[VN] = NewVNInfo.size()-1;
        continue;
      }
      
      ComputeUltimateVN(VNI, NewVNInfo,
                        RHSValsDefinedFromLHS, LHSValsDefinedFromRHS,
                        RHSValNoAssignments, LHSValNoAssignments);
    }
  }
  
  // Armed with the mappings of LHS/RHS values to ultimate values, walk the
  // interval lists to see if these intervals are coalescable.
  LiveInterval::const_iterator I = LHS.begin();
  LiveInterval::const_iterator IE = LHS.end();
  LiveInterval::const_iterator J = RHS.begin();
  LiveInterval::const_iterator JE = RHS.end();
  
  // Skip ahead until the first place of potential sharing.
  if (I->start < J->start) {
    I = std::upper_bound(I, IE, J->start);
    if (I != LHS.begin()) --I;
  } else if (J->start < I->start) {
    J = std::upper_bound(J, JE, I->start);
    if (J != RHS.begin()) --J;
  }
  
  while (1) {
    // Determine if these two live ranges overlap.
    bool Overlaps;
    if (I->start < J->start) {
      Overlaps = I->end > J->start;
    } else {
      Overlaps = J->end > I->start;
    }

    // If so, check value # info to determine if they are really different.
    if (Overlaps) {
      // If the live range overlap will map to the same value number in the
      // result liverange, we can still coalesce them.  If not, we can't.
      if (LHSValNoAssignments[I->valno->id] !=
          RHSValNoAssignments[J->valno->id])
        return false;
    }
    
    if (I->end < J->end) {
      ++I;
      if (I == IE) break;
    } else {
      ++J;
      if (J == JE) break;
    }
  }

  // Update kill info. Some live ranges are extended due to copy coalescing.
  for (DenseMap<VNInfo*, VNInfo*>::iterator I = LHSValsDefinedFromRHS.begin(),
         E = LHSValsDefinedFromRHS.end(); I != E; ++I) {
    VNInfo *VNI = I->first;
    unsigned LHSValID = LHSValNoAssignments[VNI->id];
    LiveInterval::removeKill(NewVNInfo[LHSValID], VNI->def);
    NewVNInfo[LHSValID]->hasPHIKill |= VNI->hasPHIKill;
    RHS.addKills(NewVNInfo[LHSValID], VNI->kills);
  }

  // Update kill info. Some live ranges are extended due to copy coalescing.
  for (DenseMap<VNInfo*, VNInfo*>::iterator I = RHSValsDefinedFromLHS.begin(),
         E = RHSValsDefinedFromLHS.end(); I != E; ++I) {
    VNInfo *VNI = I->first;
    unsigned RHSValID = RHSValNoAssignments[VNI->id];
    LiveInterval::removeKill(NewVNInfo[RHSValID], VNI->def);
    NewVNInfo[RHSValID]->hasPHIKill |= VNI->hasPHIKill;
    LHS.addKills(NewVNInfo[RHSValID], VNI->kills);
  }

  // If we get here, we know that we can coalesce the live ranges.  Ask the
  // intervals to coalesce themselves now.
  if ((RHS.ranges.size() > LHS.ranges.size() &&
      TargetRegisterInfo::isVirtualRegister(LHS.reg)) ||
      TargetRegisterInfo::isPhysicalRegister(RHS.reg)) {
    RHS.join(LHS, &RHSValNoAssignments[0], &LHSValNoAssignments[0], NewVNInfo);
    Swapped = true;
  } else {
    LHS.join(RHS, &LHSValNoAssignments[0], &RHSValNoAssignments[0], NewVNInfo);
    Swapped = false;
  }
  return true;
}

namespace {
  // DepthMBBCompare - Comparison predicate that sort first based on the loop
  // depth of the basic block (the unsigned), and then on the MBB number.
  struct DepthMBBCompare {
    typedef std::pair<unsigned, MachineBasicBlock*> DepthMBBPair;
    bool operator()(const DepthMBBPair &LHS, const DepthMBBPair &RHS) const {
      if (LHS.first > RHS.first) return true;   // Deeper loops first
      return LHS.first == RHS.first &&
        LHS.second->getNumber() < RHS.second->getNumber();
    }
  };
}

/// getRepIntervalSize - Returns the size of the interval that represents the
/// specified register.
template<class SF>
unsigned JoinPriorityQueue<SF>::getRepIntervalSize(unsigned Reg) {
  return Rc->getRepIntervalSize(Reg);
}

/// CopyRecSort::operator - Join priority queue sorting function.
///
bool CopyRecSort::operator()(CopyRec left, CopyRec right) const {
  // Inner loops first.
  if (left.LoopDepth > right.LoopDepth)
    return false;
  else if (left.LoopDepth == right.LoopDepth)
    if (left.isBackEdge && !right.isBackEdge)
      return false;
  return true;
}

void SimpleRegisterCoalescing::CopyCoalesceInMBB(MachineBasicBlock *MBB,
                                               std::vector<CopyRec> &TryAgain) {
  DOUT << ((Value*)MBB->getBasicBlock())->getName() << ":\n";

  std::vector<CopyRec> VirtCopies;
  std::vector<CopyRec> PhysCopies;
  std::vector<CopyRec> ImpDefCopies;
  unsigned LoopDepth = loopInfo->getLoopDepth(MBB);
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E;) {
    MachineInstr *Inst = MII++;
    
    // If this isn't a copy nor a extract_subreg, we can't join intervals.
    unsigned SrcReg, DstReg;
    if (Inst->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
      DstReg = Inst->getOperand(0).getReg();
      SrcReg = Inst->getOperand(1).getReg();
    } else if (Inst->getOpcode() == TargetInstrInfo::INSERT_SUBREG) {
      DstReg = Inst->getOperand(0).getReg();
      SrcReg = Inst->getOperand(2).getReg();
    } else if (!tii_->isMoveInstr(*Inst, SrcReg, DstReg))
      continue;

    bool SrcIsPhys = TargetRegisterInfo::isPhysicalRegister(SrcReg);
    bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
    if (NewHeuristic) {
      JoinQueue->push(CopyRec(Inst, LoopDepth, isBackEdgeCopy(Inst, DstReg)));
    } else {
      if (li_->hasInterval(SrcReg) && li_->getInterval(SrcReg).empty())
        ImpDefCopies.push_back(CopyRec(Inst, 0, false));
      else if (SrcIsPhys || DstIsPhys)
        PhysCopies.push_back(CopyRec(Inst, 0, false));
      else
        VirtCopies.push_back(CopyRec(Inst, 0, false));
    }
  }

  if (NewHeuristic)
    return;

  // Try coalescing implicit copies first, followed by copies to / from
  // physical registers, then finally copies from virtual registers to
  // virtual registers.
  for (unsigned i = 0, e = ImpDefCopies.size(); i != e; ++i) {
    CopyRec &TheCopy = ImpDefCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = PhysCopies.size(); i != e; ++i) {
    CopyRec &TheCopy = PhysCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
  for (unsigned i = 0, e = VirtCopies.size(); i != e; ++i) {
    CopyRec &TheCopy = VirtCopies[i];
    bool Again = false;
    if (!JoinCopy(TheCopy, Again))
      if (Again)
        TryAgain.push_back(TheCopy);
  }
}

void SimpleRegisterCoalescing::joinIntervals() {
  DOUT << "********** JOINING INTERVALS ***********\n";

  if (NewHeuristic)
    JoinQueue = new JoinPriorityQueue<CopyRecSort>(this);

  std::vector<CopyRec> TryAgainList;
  if (loopInfo->begin() == loopInfo->end()) {
    // If there are no loops in the function, join intervals in function order.
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();
         I != E; ++I)
      CopyCoalesceInMBB(I, TryAgainList);
  } else {
    // Otherwise, join intervals in inner loops before other intervals.
    // Unfortunately we can't just iterate over loop hierarchy here because
    // there may be more MBB's than BB's.  Collect MBB's for sorting.

    // Join intervals in the function prolog first. We want to join physical
    // registers with virtual registers before the intervals got too long.
    std::vector<std::pair<unsigned, MachineBasicBlock*> > MBBs;
    for (MachineFunction::iterator I = mf_->begin(), E = mf_->end();I != E;++I){
      MachineBasicBlock *MBB = I;
      MBBs.push_back(std::make_pair(loopInfo->getLoopDepth(MBB), I));
    }

    // Sort by loop depth.
    std::sort(MBBs.begin(), MBBs.end(), DepthMBBCompare());

    // Finally, join intervals in loop nest order.
    for (unsigned i = 0, e = MBBs.size(); i != e; ++i)
      CopyCoalesceInMBB(MBBs[i].second, TryAgainList);
  }
  
  // Joining intervals can allow other intervals to be joined.  Iteratively join
  // until we make no progress.
  if (NewHeuristic) {
    SmallVector<CopyRec, 16> TryAgain;
    bool ProgressMade = true;
    while (ProgressMade) {
      ProgressMade = false;
      while (!JoinQueue->empty()) {
        CopyRec R = JoinQueue->pop();
        bool Again = false;
        bool Success = JoinCopy(R, Again);
        if (Success)
          ProgressMade = true;
        else if (Again)
          TryAgain.push_back(R);
      }

      if (ProgressMade) {
        while (!TryAgain.empty()) {
          JoinQueue->push(TryAgain.back());
          TryAgain.pop_back();
        }
      }
    }
  } else {
    bool ProgressMade = true;
    while (ProgressMade) {
      ProgressMade = false;

      for (unsigned i = 0, e = TryAgainList.size(); i != e; ++i) {
        CopyRec &TheCopy = TryAgainList[i];
        if (TheCopy.MI) {
          bool Again = false;
          bool Success = JoinCopy(TheCopy, Again);
          if (Success || !Again) {
            TheCopy.MI = 0;   // Mark this one as done.
            ProgressMade = true;
          }
        }
      }
    }
  }

  if (NewHeuristic)
    delete JoinQueue;  
}

/// Return true if the two specified registers belong to different register
/// classes.  The registers may be either phys or virt regs.
bool SimpleRegisterCoalescing::differingRegisterClasses(unsigned RegA,
                                                        unsigned RegB) const {

  // Get the register classes for the first reg.
  if (TargetRegisterInfo::isPhysicalRegister(RegA)) {
    assert(TargetRegisterInfo::isVirtualRegister(RegB) &&
           "Shouldn't consider two physregs!");
    return !mri_->getRegClass(RegB)->contains(RegA);
  }

  // Compare against the regclass for the second reg.
  const TargetRegisterClass *RegClass = mri_->getRegClass(RegA);
  if (TargetRegisterInfo::isVirtualRegister(RegB))
    return RegClass != mri_->getRegClass(RegB);
  else
    return !RegClass->contains(RegB);
}

/// lastRegisterUse - Returns the last use of the specific register between
/// cycles Start and End or NULL if there are no uses.
MachineOperand *
SimpleRegisterCoalescing::lastRegisterUse(unsigned Start, unsigned End,
                                          unsigned Reg, unsigned &UseIdx) const{
  UseIdx = 0;
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    MachineOperand *LastUse = NULL;
    for (MachineRegisterInfo::use_iterator I = mri_->use_begin(Reg),
           E = mri_->use_end(); I != E; ++I) {
      MachineOperand &Use = I.getOperand();
      MachineInstr *UseMI = Use.getParent();
      unsigned SrcReg, DstReg;
      if (tii_->isMoveInstr(*UseMI, SrcReg, DstReg) && SrcReg == DstReg)
        // Ignore identity copies.
        continue;
      unsigned Idx = li_->getInstructionIndex(UseMI);
      if (Idx >= Start && Idx < End && Idx >= UseIdx) {
        LastUse = &Use;
        UseIdx = Idx;
      }
    }
    return LastUse;
  }

  int e = (End-1) / InstrSlots::NUM * InstrSlots::NUM;
  int s = Start;
  while (e >= s) {
    // Skip deleted instructions
    MachineInstr *MI = li_->getInstructionFromIndex(e);
    while ((e - InstrSlots::NUM) >= s && !MI) {
      e -= InstrSlots::NUM;
      MI = li_->getInstructionFromIndex(e);
    }
    if (e < s || MI == NULL)
      return NULL;

    // Ignore identity copies.
    unsigned SrcReg, DstReg;
    if (!(tii_->isMoveInstr(*MI, SrcReg, DstReg) && SrcReg == DstReg))
      for (unsigned i = 0, NumOps = MI->getNumOperands(); i != NumOps; ++i) {
        MachineOperand &Use = MI->getOperand(i);
        if (Use.isRegister() && Use.isUse() && Use.getReg() &&
            tri_->regsOverlap(Use.getReg(), Reg)) {
          UseIdx = e;
          return &Use;
        }
      }

    e -= InstrSlots::NUM;
  }

  return NULL;
}


void SimpleRegisterCoalescing::printRegName(unsigned reg) const {
  if (TargetRegisterInfo::isPhysicalRegister(reg))
    cerr << tri_->getName(reg);
  else
    cerr << "%reg" << reg;
}

void SimpleRegisterCoalescing::releaseMemory() {
  JoinedCopies.clear();
}

static bool isZeroLengthInterval(LiveInterval *li) {
  for (LiveInterval::Ranges::const_iterator
         i = li->ranges.begin(), e = li->ranges.end(); i != e; ++i)
    if (i->end - i->start > LiveIntervals::InstrSlots::NUM)
      return false;
  return true;
}

/// TurnCopyIntoImpDef - If source of the specified copy is an implicit def,
/// turn the copy into an implicit def.
bool
SimpleRegisterCoalescing::TurnCopyIntoImpDef(MachineBasicBlock::iterator &I,
                                             MachineBasicBlock *MBB,
                                             unsigned DstReg, unsigned SrcReg) {
  MachineInstr *CopyMI = &*I;
  unsigned CopyIdx = li_->getDefIndex(li_->getInstructionIndex(CopyMI));
  if (!li_->hasInterval(SrcReg))
    return false;
  LiveInterval &SrcInt = li_->getInterval(SrcReg);
  if (!SrcInt.empty())
    return false;
  if (!li_->hasInterval(DstReg))
    return false;
  LiveInterval &DstInt = li_->getInterval(DstReg);
  const LiveRange *DstLR = DstInt.getLiveRangeContaining(CopyIdx);
  DstInt.removeValNo(DstLR->valno);
  CopyMI->setDesc(tii_->get(TargetInstrInfo::IMPLICIT_DEF));
  for (int i = CopyMI->getNumOperands() - 1, e = 0; i > e; --i)
    CopyMI->RemoveOperand(i);
  bool NoUse = mri_->use_begin(SrcReg) == mri_->use_end();
  if (NoUse) {
    for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(SrcReg),
           E = mri_->reg_end(); I != E; ) {
      assert(I.getOperand().isDef());
      MachineInstr *DefMI = &*I;
      ++I;
      // The implicit_def source has no other uses, delete it.
      assert(DefMI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF);
      li_->RemoveMachineInstrFromMaps(DefMI);
      DefMI->eraseFromParent();
    }
  }
  ++I;
  return true;
}


bool SimpleRegisterCoalescing::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &fn.getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  li_ = &getAnalysis<LiveIntervals>();
  lv_ = &getAnalysis<LiveVariables>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  DOUT << "********** SIMPLE REGISTER COALESCING **********\n"
       << "********** Function: "
       << ((Value*)mf_->getFunction())->getName() << '\n';

  allocatableRegs_ = tri_->getAllocatableSet(fn);
  for (TargetRegisterInfo::regclass_iterator I = tri_->regclass_begin(),
         E = tri_->regclass_end(); I != E; ++I)
    allocatableRCRegs_.insert(std::make_pair(*I,
                                             tri_->getAllocatableSet(fn, *I)));

  // Join (coalesce) intervals if requested.
  if (EnableJoining) {
    joinIntervals();
    DOUT << "********** INTERVALS POST JOINING **********\n";
    for (LiveIntervals::iterator I = li_->begin(), E = li_->end(); I != E; ++I){
      I->second.print(DOUT, tri_);
      DOUT << "\n";
    }
  }

  // Perform a final pass over the instructions and compute spill weights
  // and remove identity moves.
  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    MachineBasicBlock* mbb = mbbi;
    unsigned loopDepth = loopInfo->getLoopDepth(mbb);

    for (MachineBasicBlock::iterator mii = mbb->begin(), mie = mbb->end();
         mii != mie; ) {
      MachineInstr *MI = mii;
      unsigned SrcReg, DstReg;
      if (JoinedCopies.count(MI)) {
        // Delete all coalesced copies.
        if (!tii_->isMoveInstr(*MI, SrcReg, DstReg)) {
          assert((MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
                  MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG) &&
                 "Unrecognized copy instruction");
          DstReg = MI->getOperand(0).getReg();
        }
        if (MI->registerDefIsDead(DstReg)) {
          LiveInterval &li = li_->getInterval(DstReg);
          if (!ShortenDeadCopySrcLiveRange(li, MI))
            ShortenDeadCopyLiveRange(li, MI);
        }
        li_->RemoveMachineInstrFromMaps(MI);
        mii = mbbi->erase(mii);
        ++numPeep;
        continue;
      }

      // If the move will be an identity move delete it
      bool isMove = tii_->isMoveInstr(*mii, SrcReg, DstReg);
      if (isMove && SrcReg == DstReg) {
        if (li_->hasInterval(SrcReg)) {
          LiveInterval &RegInt = li_->getInterval(SrcReg);
          // If def of this move instruction is dead, remove its live range
          // from the dstination register's live interval.
          if (mii->registerDefIsDead(DstReg)) {
            if (!ShortenDeadCopySrcLiveRange(RegInt, mii))
              ShortenDeadCopyLiveRange(RegInt, mii);
          }
        }
        li_->RemoveMachineInstrFromMaps(mii);
        mii = mbbi->erase(mii);
        ++numPeep;
      } else if (!isMove || !TurnCopyIntoImpDef(mii, mbb, DstReg, SrcReg)) {
        SmallSet<unsigned, 4> UniqueUses;
        for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
          const MachineOperand &mop = mii->getOperand(i);
          if (mop.isRegister() && mop.getReg() &&
              TargetRegisterInfo::isVirtualRegister(mop.getReg())) {
            unsigned reg = mop.getReg();
            // Multiple uses of reg by the same instruction. It should not
            // contribute to spill weight again.
            if (UniqueUses.count(reg) != 0)
              continue;
            LiveInterval &RegInt = li_->getInterval(reg);
            RegInt.weight +=
              li_->getSpillWeight(mop.isDef(), mop.isUse(), loopDepth);
            UniqueUses.insert(reg);
          }
        }
        ++mii;
      }
    }
  }

  for (LiveIntervals::iterator I = li_->begin(), E = li_->end(); I != E; ++I) {
    LiveInterval &LI = I->second;
    if (TargetRegisterInfo::isVirtualRegister(LI.reg)) {
      // If the live interval length is essentially zero, i.e. in every live
      // range the use follows def immediately, it doesn't make sense to spill
      // it and hope it will be easier to allocate for this li.
      if (isZeroLengthInterval(&LI))
        LI.weight = HUGE_VALF;
      else {
        bool isLoad = false;
        if (li_->isReMaterializable(LI, isLoad)) {
          // If all of the definitions of the interval are re-materializable,
          // it is a preferred candidate for spilling. If non of the defs are
          // loads, then it's potentially very cheap to re-materialize.
          // FIXME: this gets much more complicated once we support non-trivial
          // re-materialization.
          if (isLoad)
            LI.weight *= 0.9F;
          else
            LI.weight *= 0.5F;
        }
      }

      // Slightly prefer live interval that has been assigned a preferred reg.
      if (LI.preference)
        LI.weight *= 1.01F;

      // Divide the weight of the interval by its size.  This encourages 
      // spilling of intervals that are large and have few uses, and
      // discourages spilling of small intervals with many uses.
      LI.weight /= LI.getSize();
    }
  }

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void SimpleRegisterCoalescing::print(std::ostream &O, const Module* m) const {
   li_->print(O, m);
}

RegisterCoalescer* llvm::createSimpleRegisterCoalescer() {
  return new SimpleRegisterCoalescing();
}

// Make sure that anything that uses RegisterCoalescer pulls in this file...
DEFINING_FILE_FOR(SimpleRegisterCoalescing)

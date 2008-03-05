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
namespace {
  static cl::opt<bool>
  EnableJoining("join-liveintervals",
                cl::desc("Coalesce copies (default=true)"),
                cl::init(true));

  static cl::opt<bool>
  NewHeuristic("new-coalescer-heuristic",
                cl::desc("Use new coalescer heuristic"),
                cl::init(false));

  static cl::opt<bool>
  CommuteDef("coalescer-commute-instrs",
             cl::init(true), cl::Hidden);

  static cl::opt<int>
  CommuteLimit("commute-limit",
               cl::init(-1), cl::Hidden);

  RegisterPass<SimpleRegisterCoalescing> 
  X("simple-register-coalescing", "Simple Register Coalescing");

  // Declare that we implement the RegisterCoalescer interface
  RegisterAnalysisGroup<RegisterCoalescer, true/*The Default*/> V(X);
}

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
  VNInfo *BValNo = BLR->valno;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  if (!BValNo->copy) return false;
  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A3 in the example.
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyIdx-1);
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
  if (!CommuteDef) return false;

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
  VNInfo *BValNo = BLR->valno;
  
  // Get the location that B is defined at.  Two options: either this value has
  // an unknown definition point or it is defined at CopyIdx.  If unknown, we 
  // can't process it.
  if (!BValNo->copy) return false;
  assert(BValNo->def == CopyIdx && "Copy doesn't define the value?");
  
  // AValNo is the value number in A that defines the copy, A3 in the example.
  LiveInterval::iterator ALR = IntA.FindLiveRangeContaining(CopyIdx-1);
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

  if (CommuteLimit >= 0 && numCommutes >= (unsigned)CommuteLimit)
    return false;

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

  // Update uses of IntA of the specific Val# with IntB.
  bool BHasPHIKill = BValNo->hasPHIKill;
  SmallVector<VNInfo*, 4> BDeadValNos;
  SmallVector<unsigned, 4> BKills;
  std::map<unsigned, unsigned> BExtend;
  for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(IntA.reg),
         UE = mri_->use_end(); UI != UE;) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    ++UI;
    if (JoinedCopies.count(UseMI))
      continue;
    unsigned UseIdx = li_->getInstructionIndex(UseMI);
    LiveInterval::iterator ULR = IntA.FindLiveRangeContaining(UseIdx);
    if (ULR->valno != AValNo)
      continue;
    UseMO.setReg(NewReg);
    if (UseMI == CopyMI)
      continue;
    if (UseMO.isKill())
      BKills.push_back(li_->getUseIndex(UseIdx)+1);
    unsigned SrcReg, DstReg;
    if (!tii_->isMoveInstr(*UseMI, SrcReg, DstReg))
      continue;
    if (DstReg == IntB.reg) {
      // This copy will become a noop. If it's defining a new val#,
      // remove that val# as well. However this live range is being
      // extended to the end of the existing live range defined by the copy.
      unsigned DefIdx = li_->getDefIndex(UseIdx);
      LiveInterval::iterator DLR = IntB.FindLiveRangeContaining(DefIdx);
      BHasPHIKill |= DLR->valno->hasPHIKill;
      assert(DLR->valno->def == DefIdx);
      BDeadValNos.push_back(DLR->valno);
      BExtend[DLR->start] = DLR->end;
      JoinedCopies.insert(UseMI);
      // If this is a kill but it's going to be removed, the last use
      // of the same val# is the new kill.
      if (UseMO.isKill()) {
        BKills.pop_back();
      }
    }
  }

  // We need to insert a new liverange: [ALR.start, LastUse). It may be we can
  // simply extend BLR if CopyMI doesn't end the range.
  DOUT << "\nExtending: "; IntB.print(DOUT, tri_);

  IntB.removeValNo(BValNo);
  for (unsigned i = 0, e = BDeadValNos.size(); i != e; ++i)
    IntB.removeValNo(BDeadValNos[i]);
  VNInfo *ValNo = IntB.getNextValue(ALR->start, 0, li_->getVNInfoAllocator());
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
                                              unsigned DstReg) {
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
    ++I;
    if (DstIsPhys) {
      unsigned UseSubIdx = O.getSubReg();
      unsigned UseDstReg = DstReg;
      if (UseSubIdx)
        UseDstReg = tri_->getSubReg(DstReg, UseSubIdx);
      O.setReg(UseDstReg);
      O.setSubReg(0);
    } else {
      unsigned OldSubIdx = O.getSubReg();
      // Sub-register indexes goes from small to large. e.g.
      // RAX: 0 -> AL, 1 -> AH, 2 -> AX, 3 -> EAX
      // EAX: 0 -> AL, 1 -> AH, 2 -> AX
      // So RAX's sub-register 2 is AX, RAX's sub-regsiter 3 is EAX, whose
      // sub-register 2 is also AX.
      if (SubIdx && OldSubIdx && SubIdx != OldSubIdx)
        assert(OldSubIdx < SubIdx && "Conflicting sub-register index!");
      else if (SubIdx)
        O.setSubReg(SubIdx);
      O.setReg(DstReg);
    }
  }
}

/// ShortenDeadCopyLiveRange - Shorten a live range as it's artificially
/// extended by a dead copy. Mark the last use (if any) of the val# as kill
/// as ends the live range there. If there isn't another use, then this
/// live range is dead.
void SimpleRegisterCoalescing::ShortenDeadCopyLiveRange(LiveInterval &li,
                                                        MachineInstr *CopyMI) {
  unsigned CopyIdx = li_->getInstructionIndex(CopyMI);
  LiveInterval::iterator MLR =
    li.FindLiveRangeContaining(li_->getDefIndex(CopyIdx));
  unsigned RemoveStart = MLR->start;
  unsigned RemoveEnd = MLR->end;
  unsigned LastUseIdx;
  MachineOperand *LastUse = lastRegisterUse(RemoveStart, CopyIdx, li.reg,
                                            LastUseIdx);
  if (LastUse) {
    // Shorten the liveinterval to the end of last use.
    LastUse->setIsKill();
    RemoveStart = li_->getDefIndex(LastUseIdx);
  }
  li.removeRange(RemoveStart, RemoveEnd, true);
  if (li.empty())
    li_->removeInterval(li.reg);
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
  unsigned SubIdx = 0;
  if (isExtSubReg) {
    DstReg = CopyMI->getOperand(0).getReg();
    SrcReg = CopyMI->getOperand(1).getReg();
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
  if (isExtSubReg) {
    SubIdx = CopyMI->getOperand(2).getImm();
    if (SrcIsPhys) {
      // r1024 = EXTRACT_SUBREG EAX, 0 then r1024 is really going to be
      // coalesced with AX.
      SrcReg = tri_->getSubReg(SrcReg, SubIdx);
      SubIdx = 0;
    } else if (DstIsPhys) {
      // If this is a extract_subreg where dst is a physical register, e.g.
      // cl = EXTRACT_SUBREG reg1024, 1
      // then create and update the actual physical register allocated to RHS.
      const TargetRegisterClass *RC = mri_->getRegClass(SrcReg);
      for (const unsigned *SRs = tri_->getSuperRegisters(DstReg);
           unsigned SR = *SRs; ++SRs) {
        if (DstReg == tri_->getSubReg(SR, SubIdx) &&
            RC->contains(SR)) {
          RealDstReg = SR;
          break;
        }
      }
      assert(RealDstReg && "Invalid extra_subreg instruction!");

      // For this type of EXTRACT_SUBREG, conservatively
      // check if the live interval of the source register interfere with the
      // actual super physical register we are trying to coalesce with.
      LiveInterval &RHS = li_->getInterval(SrcReg);
      if (li_->hasInterval(RealDstReg) &&
          RHS.overlaps(li_->getInterval(RealDstReg))) {
        DOUT << "Interfere with register ";
        DEBUG(li_->getInterval(RealDstReg).print(DOUT, tri_));
        return false; // Not coalescable
      }
      for (const unsigned* SR = tri_->getSubRegisters(RealDstReg); *SR; ++SR)
        if (li_->hasInterval(*SR) && RHS.overlaps(li_->getInterval(*SR))) {
          DOUT << "Interfere with sub-register ";
          DEBUG(li_->getInterval(*SR).print(DOUT, tri_));
          return false; // Not coalescable
        }
      SubIdx = 0;
    } else {
      unsigned SrcSize= li_->getInterval(SrcReg).getSize() / InstrSlots::NUM;
      unsigned DstSize= li_->getInterval(DstReg).getSize() / InstrSlots::NUM;
      const TargetRegisterClass *RC = mri_->getRegClass(DstReg);
      unsigned Threshold = allocatableRCRegs_[RC].count();
      // Be conservative. If both sides are virtual registers, do not coalesce
      // if this will cause a high use density interval to target a smaller set
      // of registers.
      if (DstSize > Threshold || SrcSize > Threshold) {
        LiveVariables::VarInfo &svi = lv_->getVarInfo(SrcReg);
        LiveVariables::VarInfo &dvi = lv_->getVarInfo(DstReg);
        if ((float)dvi.NumUses / DstSize < (float)svi.NumUses / SrcSize) {
          Again = true;  // May be possible to coalesce later.
          return false;
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

  // Check if it is necessary to propagate "isDead" property before intervals
  // are joined.
  MachineOperand *mopd = CopyMI->findRegisterDefOperand(DstReg, false);
  bool isDead = mopd->isDead();
  bool isShorten = false;
  unsigned SrcStart = 0, RemoveStart = 0;
  unsigned SrcEnd = 0, RemoveEnd = 0;
  if (isDead) {
    unsigned CopyIdx = li_->getInstructionIndex(CopyMI);
    LiveInterval::iterator SrcLR =
      SrcInt.FindLiveRangeContaining(li_->getUseIndex(CopyIdx));
    RemoveStart = SrcStart = SrcLR->start;
    RemoveEnd   = SrcEnd   = SrcLR->end;
    if (SrcEnd > li_->getDefIndex(CopyIdx)) {
      // If there are other uses of SrcReg beyond the copy, there is nothing to do.
      isDead = false;
    } else {
      unsigned LastUseIdx;
      MachineOperand *LastUse =
        lastRegisterUse(SrcStart, CopyIdx, SrcReg, LastUseIdx);
      if (LastUse) {
        // There are uses before the copy, just shorten the live range to the end
        // of last use.
        LastUse->setIsKill();
        isDead = false;
        isShorten = true;
        RemoveStart = li_->getDefIndex(LastUseIdx);
      } else {
        // This live range is truly dead. Remove it.
        MachineInstr *SrcMI = li_->getInstructionFromIndex(SrcStart);
        if (SrcMI && SrcMI->modifiesRegister(SrcReg, tri_))
          // A dead def should have a single cycle interval.
          ++RemoveStart;
      }
    }
  }

  // We need to be careful about coalescing a source physical register with a
  // virtual register. Once the coalescing is done, it cannot be broken and
  // these are not spillable! If the destination interval uses are far away,
  // think twice about coalescing them!
  if (!mopd->isDead() && (SrcIsPhys || DstIsPhys) && !isExtSubReg) {
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

  // Okay, attempt to join these two intervals.  On failure, this returns false.
  // Otherwise, if one of the intervals being joined is a physreg, this method
  // always canonicalizes DstInt to be it.  The output "SrcInt" will not have
  // been modified, so we can use this information below to update aliases.
  bool Swapped = false;
  if (JoinIntervals(DstInt, SrcInt, Swapped)) {
    if (isDead) {
      // Result of the copy is dead. Propagate this property.
      if (SrcStart == 0) {
        assert(TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
               "Live-in must be a physical register!");
        // Live-in to the function but dead. Remove it from entry live-in set.
        // JoinIntervals may end up swapping the two intervals.
        mf_->begin()->removeLiveIn(SrcReg);
      } else {
        MachineInstr *SrcMI = li_->getInstructionFromIndex(SrcStart);
        if (SrcMI) {
          int DeadIdx = SrcMI->findRegisterDefOperandIdx(SrcReg, false, tri_);
          if (DeadIdx != -1)
            SrcMI->getOperand(DeadIdx).setIsDead();
        }
      }
    }

    if (isShorten || isDead) {
      // Shorten the destination live interval.
      if (Swapped)
        SrcInt.removeRange(RemoveStart, RemoveEnd, true);
    }
  } else {
    // Coalescing failed.
    
    // If we can eliminate the copy without merging the live ranges, do so now.
    if (!isExtSubReg &&
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
    if (RealDstReg) {
      LiveInterval &RealDstInt = li_->getOrCreateInterval(RealDstReg);
      SmallSet<const VNInfo*, 4> CopiedValNos;
      for (LiveInterval::Ranges::const_iterator I = ResSrcInt->ranges.begin(),
             E = ResSrcInt->ranges.end(); I != E; ++I) {
        LiveInterval::const_iterator DstLR =
          ResDstInt->FindLiveRangeContaining(I->start);
        assert(DstLR != ResDstInt->end() && "Invalid joined interval!");
        const VNInfo *DstValNo = DstLR->valno;
        if (CopiedValNos.insert(DstValNo)) {
          VNInfo *ValNo = RealDstInt.getNextValue(DstValNo->def, DstValNo->copy,
                                                  li_->getVNInfoAllocator());
          ValNo->hasPHIKill = DstValNo->hasPHIKill;
          RealDstInt.addKills(ValNo, DstValNo->kills);
          RealDstInt.MergeValueInAsValue(*ResDstInt, DstValNo, ValNo);
        }
      }
      DstReg = RealDstReg;
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
  if (isExtSubReg && !SrcIsPhys && !DstIsPhys) {
    if (!Swapped) {
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

  DOUT << "\n\t\tJoined.  Result = "; ResDstInt->print(DOUT, tri_);
  DOUT << "\n";

  // Remember to delete the copy instruction.
  JoinedCopies.insert(CopyMI);

  // SrcReg is guarateed to be the register whose live interval that is
  // being merged.
  li_->removeInterval(SrcReg);
  UpdateRegDefsUses(SrcReg, DstReg, SubIdx);

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
        unsigned SrcReg = li_->getVNInfoSourceReg(LHSIt->valno);
        if (SrcReg != RHS.reg)
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
          if (li_->getVNInfoSourceReg(LHSIt->valno) == RHS.reg) {
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
  } else {
    assert(!EliminatedLHSVals.empty() && "No copies from the RHS?");
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
  unsigned LoopDepth = loopInfo->getLoopDepth(MBB);
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E;) {
    MachineInstr *Inst = MII++;
    
    // If this isn't a copy nor a extract_subreg, we can't join intervals.
    unsigned SrcReg, DstReg;
    if (Inst->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
      DstReg = Inst->getOperand(0).getReg();
      SrcReg = Inst->getOperand(1).getReg();
    } else if (!tii_->isMoveInstr(*Inst, SrcReg, DstReg))
      continue;

    bool SrcIsPhys = TargetRegisterInfo::isPhysicalRegister(SrcReg);
    bool DstIsPhys = TargetRegisterInfo::isPhysicalRegister(DstReg);
    if (NewHeuristic) {
      JoinQueue->push(CopyRec(Inst, LoopDepth, isBackEdgeCopy(Inst, DstReg)));
    } else {
      if (SrcIsPhys || DstIsPhys)
        PhysCopies.push_back(CopyRec(Inst, 0, false));
      else
        VirtCopies.push_back(CopyRec(Inst, 0, false));
    }
  }

  if (NewHeuristic)
    return;

  // Try coalescing physical register + virtual register first.
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


/// RemoveUnnecessaryKills - Remove kill markers that are no longer accurate
/// due to live range lengthening as the result of coalescing.
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

    // Delete all coalesced copies.
    for (SmallPtrSet<MachineInstr*,32>::iterator I = JoinedCopies.begin(),
           E = JoinedCopies.end(); I != E; ++I) {
      li_->RemoveMachineInstrFromMaps(*I);
      (*I)->eraseFromParent();
      ++numPeep;
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
      // if the move will be an identity move delete it
      unsigned srcReg, dstReg;
      if (tii_->isMoveInstr(*mii, srcReg, dstReg) && srcReg == dstReg) {
        // remove from def list
        LiveInterval &RegInt = li_->getOrCreateInterval(srcReg);
        // If def of this move instruction is dead, remove its live range from
        // the dstination register's live interval.
        if (mii->registerDefIsDead(dstReg))
          ShortenDeadCopyLiveRange(RegInt, mii);
        li_->RemoveMachineInstrFromMaps(mii);
        mii = mbbi->erase(mii);
        ++numPeep;
      } else {
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

//===-- MachineLICM.cpp - Machine Loop Invariant Code Motion Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs loop invariant code motion on machine instructions. We
// attempt to remove as much code from the body of a loop as possible.
//
// This pass does not attempt to throttle itself to limit register pressure.
// The register allocation phases are expected to perform rematerialization
// to recover when register pressure is high.
//
// This pass is not intended to be a replacement or a complete alternative
// for the LLVM-IR-level LICM pass. It is only designed to hoist simple
// constructs that are not exposed before lowering and instruction selection.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-licm"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumHoisted, "Number of machine instructions hoisted out of loops");
STATISTIC(NumCSEed,   "Number of hoisted machine instructions CSEed");
STATISTIC(NumPostRAHoisted,
          "Number of machine instructions hoisted out of loops post regalloc");

namespace {
  class MachineLICM : public MachineFunctionPass {
    bool PreRegAlloc;

    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const MachineFrameInfo *MFI;
    MachineRegisterInfo *RegInfo;

    // Various analyses that we use...
    AliasAnalysis        *AA;      // Alias analysis info.
    MachineLoopInfo      *LI;      // Current MachineLoopInfo
    MachineDominatorTree *DT;      // Machine dominator tree for the cur loop

    // State that is updated as we process loops
    bool         Changed;          // True if a loop is changed.
    bool         FirstInLoop;      // True if it's the first LICM in the loop.
    MachineLoop *CurLoop;          // The current loop we are working on.
    MachineBasicBlock *CurPreheader; // The preheader for CurLoop.

    BitVector AllocatableSet;

    // For each opcode, keep a list of potentail CSE instructions.
    DenseMap<unsigned, std::vector<const MachineInstr*> > CSEMap;

  public:
    static char ID; // Pass identification, replacement for typeid
    MachineLICM() :
      MachineFunctionPass(&ID), PreRegAlloc(true) {}

    explicit MachineLICM(bool PreRA) :
      MachineFunctionPass(&ID), PreRegAlloc(PreRA) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const { return "Machine Instruction LICM"; }

    // FIXME: Loop preheaders?
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    virtual void releaseMemory() {
      CSEMap.clear();
    }

  private:
    /// IsLoopInvariantInst - Returns true if the instruction is loop
    /// invariant. I.e., all virtual register operands are defined outside of
    /// the loop, physical registers aren't accessed (explicitly or implicitly),
    /// and the instruction is hoistable.
    /// 
    bool IsLoopInvariantInst(MachineInstr &I);

    /// IsProfitableToHoist - Return true if it is potentially profitable to
    /// hoist the given loop invariant.
    bool IsProfitableToHoist(MachineInstr &MI);

    /// HoistRegion - Walk the specified region of the CFG (defined by all
    /// blocks dominated by the specified block, and that are in the current
    /// loop) in depth first order w.r.t the DominatorTree. This allows us to
    /// visit definitions before uses, allowing us to hoist a loop body in one
    /// pass without iteration.
    ///
    void HoistRegion(MachineDomTreeNode *N);
    void HoistRegionPostRA(MachineDomTreeNode *N);

    /// isLoadFromConstantMemory - Return true if the given instruction is a
    /// load from constant memory.
    bool isLoadFromConstantMemory(MachineInstr *MI);

    /// ExtractHoistableLoad - Unfold a load from the given machineinstr if
    /// the load itself could be hoisted. Return the unfolded and hoistable
    /// load, or null if the load couldn't be unfolded or if it wouldn't
    /// be hoistable.
    MachineInstr *ExtractHoistableLoad(MachineInstr *MI);

    /// LookForDuplicate - Find an instruction amount PrevMIs that is a
    /// duplicate of MI. Return this instruction if it's found.
    const MachineInstr *LookForDuplicate(const MachineInstr *MI,
                                     std::vector<const MachineInstr*> &PrevMIs);

    /// EliminateCSE - Given a LICM'ed instruction, look for an instruction on
    /// the preheader that compute the same value. If it's found, do a RAU on
    /// with the definition of the existing instruction rather than hoisting
    /// the instruction to the preheader.
    bool EliminateCSE(MachineInstr *MI,
           DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator &CI);

    /// Hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    ///
    void Hoist(MachineInstr *MI);
    void HoistPostRA(MachineInstr *MI);

    /// InitCSEMap - Initialize the CSE map with instructions that are in the
    /// current loop preheader that may become duplicates of instructions that
    /// are hoisted out of the loop.
    void InitCSEMap(MachineBasicBlock *BB);
  };
} // end anonymous namespace

char MachineLICM::ID = 0;
static RegisterPass<MachineLICM>
X("machinelicm", "Machine Loop Invariant Code Motion");

FunctionPass *llvm::createMachineLICMPass(bool PreRegAlloc) {
  return new MachineLICM(PreRegAlloc);
}

/// LoopIsOuterMostWithPreheader - Test if the given loop is the outer-most
/// loop that has a preheader.
static bool LoopIsOuterMostWithPreheader(MachineLoop *CurLoop) {
  for (MachineLoop *L = CurLoop->getParentLoop(); L; L = L->getParentLoop())
    if (L->getLoopPreheader())
      return false;
  return true;
}

/// Hoist expressions out of the specified loop. Note, alias info for inner loop
/// is not preserved so it is not a good idea to run LICM multiple times on one
/// loop.
///
bool MachineLICM::runOnMachineFunction(MachineFunction &MF) {
  if (PreRegAlloc)
    DEBUG(dbgs() << "******** Pre-regalloc Machine LICM ********\n");
  else
    DEBUG(dbgs() << "******** Post-regalloc Machine LICM ********\n");

  Changed = FirstInLoop = false;
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  TRI = TM->getRegisterInfo();
  MFI = MF.getFrameInfo();
  RegInfo = &MF.getRegInfo();
  AllocatableSet = TRI->getAllocatableSet(MF);

  // Get our Loop information...
  LI = &getAnalysis<MachineLoopInfo>();
  DT = &getAnalysis<MachineDominatorTree>();
  AA = &getAnalysis<AliasAnalysis>();

  for (MachineLoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I) {
    CurLoop = *I;

    // Only visit outer-most preheader-sporting loops.
    if (!LoopIsOuterMostWithPreheader(CurLoop))
      continue;

    // Determine the block to which to hoist instructions. If we can't find a
    // suitable loop preheader, we can't do any hoisting.
    //
    // FIXME: We are only hoisting if the basic block coming into this loop
    // has only one successor. This isn't the case in general because we haven't
    // broken critical edges or added preheaders.
    CurPreheader = CurLoop->getLoopPreheader();
    if (!CurPreheader)
      continue;

    // CSEMap is initialized for loop header when the first instruction is
    // being hoisted.
    FirstInLoop = true;
    MachineDomTreeNode *N = DT->getNode(CurLoop->getHeader());
    if (!PreRegAlloc)
      HoistRegionPostRA(N);
    else {
      HoistRegion(N);
      CSEMap.clear();
    }
  }

  return Changed;
}

void MachineLICM::HoistRegionPostRA(MachineDomTreeNode *N) {
  assert(N != 0 && "Null dominator tree node?");

  unsigned NumRegs = TRI->getNumRegs();
  unsigned *PhysRegDefs = new unsigned[NumRegs];
  std::fill(PhysRegDefs, PhysRegDefs + NumRegs, 0);

  SmallVector<std::pair<MachineInstr*, int>, 32> Candidates;
  SmallSet<int, 32> StoredFIs;

  // Walk the entire region, count number of defs for each register, and
  // return potential LICM candidates.
  SmallVector<MachineDomTreeNode*, 8> WorkList;
  WorkList.push_back(N);
  do {
    N = WorkList.pop_back_val();
    MachineBasicBlock *BB = N->getBlock();

    if (!CurLoop->contains(BB))
      continue;
    // Conservatively treat live-in's as an external def.
    for (MachineBasicBlock::const_livein_iterator I = BB->livein_begin(),
           E = BB->livein_end(); I != E; ++I) {
      unsigned Reg = *I;
      ++PhysRegDefs[Reg];
      for (const unsigned *SR = TRI->getSubRegisters(Reg); *SR; ++SR)
        ++PhysRegDefs[*SR];
    }

    for (MachineBasicBlock::iterator
           MII = BB->begin(), E = BB->end(); MII != E; ++MII) {
      bool RuledOut = false;
      bool SeenDef = false;
      MachineInstr *MI = &*MII;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        assert(TargetRegisterInfo::isPhysicalRegister(Reg) &&
               "Not expecting virtual register!");

        if (MO.isDef()) {
          SeenDef = true;
          if (++PhysRegDefs[Reg] > 1)
            // MI defined register is seen defined by another instruction in
            // the loop, it cannot be a LICM candidate.
            RuledOut = true;
          for (const unsigned *SR = TRI->getSubRegisters(Reg); *SR; ++SR)
            if (++PhysRegDefs[*SR] > 1)
              RuledOut = true;
        }
      }

      // FIXME: Only consider reloads for now.
      bool SkipCheck = false;
      int FI;
      if (SeenDef && !RuledOut) {
        if (TII->isLoadFromStackSlot(MI, FI) &&
            MFI->isSpillSlotObjectIndex(FI)) {
          Candidates.push_back(std::make_pair(MI, FI));
          SkipCheck = true;
        }
      }

      // If MI is a store to a stack slot, remember the slot. An instruction
      // loads from this slot cannot be a LICM candidate.
      if (!SkipCheck && TII->isStoreToStackSlot(MI, FI))
        StoredFIs.insert(FI);
    }

    const std::vector<MachineDomTreeNode*> &Children = N->getChildren();
    for (unsigned I = 0, E = Children.size(); I != E; ++I)
      WorkList.push_back(Children[I]);
  } while (!WorkList.empty());

  // Now evaluate whether the potential candidates qualify.
  // 1. Check if the candidate defined register is defined by another
  //    instruction in the loop.
  // 2. If the candidate is a load from stack slot (always true for now),
  //    check if the slot is stored anywhere in the loop.
  for (unsigned i = 0, e = Candidates.size(); i != e; ++i) {
    bool Safe = true;
    int FI = Candidates[i].second;
    if (StoredFIs.count(FI))
      continue;

    MachineInstr *MI = Candidates[i].first;
    for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      if (MO.isDef() && PhysRegDefs[Reg] > 1) {
        Safe = false;
        break;
      }
    }

    if (Safe)
      HoistPostRA(MI);
  }
}

void MachineLICM::HoistPostRA(MachineInstr *MI) {
  // Now move the instructions to the predecessor, inserting it before any
  // terminator instructions.
  DEBUG({
      dbgs() << "Hoisting " << *MI;
      if (CurPreheader->getBasicBlock())
        dbgs() << " to MachineBasicBlock "
               << CurPreheader->getName();
      if (MI->getParent()->getBasicBlock())
        dbgs() << " from MachineBasicBlock "
               << MI->getParent()->getName();
      dbgs() << "\n";
    });

  // Splice the instruction to the preheader.
  CurPreheader->splice(CurPreheader->getFirstTerminator(),MI->getParent(),MI);

  ++NumPostRAHoisted;
  Changed = true;
}

/// HoistRegion - Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree. This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
///
void MachineLICM::HoistRegion(MachineDomTreeNode *N) {
  assert(N != 0 && "Null dominator tree node?");
  MachineBasicBlock *BB = N->getBlock();

  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return;

  for (MachineBasicBlock::iterator
         MII = BB->begin(), E = BB->end(); MII != E; ) {
    MachineBasicBlock::iterator NextMII = MII; ++NextMII;
    Hoist(&*MII);
    MII = NextMII;
  }

  const std::vector<MachineDomTreeNode*> &Children = N->getChildren();
  for (unsigned I = 0, E = Children.size(); I != E; ++I)
    HoistRegion(Children[I]);
}

/// IsLoopInvariantInst - Returns true if the instruction is loop
/// invariant. I.e., all virtual register operands are defined outside of the
/// loop, physical registers aren't accessed explicitly, and there are no side
/// effects that aren't captured by the operands or other flags.
/// 
bool MachineLICM::IsLoopInvariantInst(MachineInstr &I) {
  const TargetInstrDesc &TID = I.getDesc();
  
  // Ignore stuff that we obviously can't hoist.
  if (TID.mayStore() || TID.isCall() || TID.isTerminator() ||
      TID.hasUnmodeledSideEffects())
    return false;

  if (TID.mayLoad()) {
    // Okay, this instruction does a load. As a refinement, we allow the target
    // to decide whether the loaded value is actually a constant. If so, we can
    // actually use it as a load.
    if (!I.isInvariantLoad(AA))
      // FIXME: we should be able to hoist loads with no other side effects if
      // there are no other instructions which can change memory in this loop.
      // This is a trivial form of alias analysis.
      return false;
  }

  // The instruction is loop invariant if all of its operands are.
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = I.getOperand(i);

    if (!MO.isReg())
      continue;

    unsigned Reg = MO.getReg();
    if (Reg == 0) continue;

    // Don't hoist an instruction that uses or defines a physical register.
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse()) {
        // If the physreg has no defs anywhere, it's just an ambient register
        // and we can freely move its uses. Alternatively, if it's allocatable,
        // it could get allocated to something with a def during allocation.
        if (!RegInfo->def_empty(Reg))
          return false;
        if (AllocatableSet.test(Reg))
          return false;
        // Check for a def among the register's aliases too.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          if (!RegInfo->def_empty(AliasReg))
            return false;
          if (AllocatableSet.test(AliasReg))
            return false;
        }
        // Otherwise it's safe to move.
        continue;
      } else if (!MO.isDead()) {
        // A def that isn't dead. We can't move it.
        return false;
      } else if (CurLoop->getHeader()->isLiveIn(Reg)) {
        // If the reg is live into the loop, we can't hoist an instruction
        // which would clobber it.
        return false;
      }
    }

    if (!MO.isUse())
      continue;

    assert(RegInfo->getVRegDef(Reg) &&
           "Machine instr not mapped for this vreg?!");

    // If the loop contains the definition of an operand, then the instruction
    // isn't loop invariant.
    if (CurLoop->contains(RegInfo->getVRegDef(Reg)))
      return false;
  }

  // If we got this far, the instruction is loop invariant!
  return true;
}


/// HasPHIUses - Return true if the specified register has any PHI use.
static bool HasPHIUses(unsigned Reg, MachineRegisterInfo *RegInfo) {
  for (MachineRegisterInfo::use_iterator UI = RegInfo->use_begin(Reg),
         UE = RegInfo->use_end(); UI != UE; ++UI) {
    MachineInstr *UseMI = &*UI;
    if (UseMI->isPHI())
      return true;
  }
  return false;
}

/// isLoadFromConstantMemory - Return true if the given instruction is a
/// load from constant memory. Machine LICM will hoist these even if they are
/// not re-materializable.
bool MachineLICM::isLoadFromConstantMemory(MachineInstr *MI) {
  if (!MI->getDesc().mayLoad()) return false;
  if (!MI->hasOneMemOperand()) return false;
  MachineMemOperand *MMO = *MI->memoperands_begin();
  if (MMO->isVolatile()) return false;
  if (!MMO->getValue()) return false;
  const PseudoSourceValue *PSV = dyn_cast<PseudoSourceValue>(MMO->getValue());
  if (PSV) {
    MachineFunction &MF = *MI->getParent()->getParent();
    return PSV->isConstant(MF.getFrameInfo());
  } else {
    return AA->pointsToConstantMemory(MMO->getValue());
  }
}

/// IsProfitableToHoist - Return true if it is potentially profitable to hoist
/// the given loop invariant.
bool MachineLICM::IsProfitableToHoist(MachineInstr &MI) {
  if (MI.isImplicitDef())
    return false;

  // FIXME: For now, only hoist re-materilizable instructions. LICM will
  // increase register pressure. We want to make sure it doesn't increase
  // spilling.
  // Also hoist loads from constant memory, e.g. load from stubs, GOT. Hoisting
  // these tend to help performance in low register pressure situation. The
  // trade off is it may cause spill in high pressure situation. It will end up
  // adding a store in the loop preheader. But the reload is no more expensive.
  // The side benefit is these loads are frequently CSE'ed.
  if (!TII->isTriviallyReMaterializable(&MI, AA)) {
    if (!isLoadFromConstantMemory(&MI))
      return false;
  }

  // If result(s) of this instruction is used by PHIs, then don't hoist it.
  // The presence of joins makes it difficult for current register allocator
  // implementation to perform remat.
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.isDef())
      continue;
    if (HasPHIUses(MO.getReg(), RegInfo))
      return false;
  }

  return true;
}

MachineInstr *MachineLICM::ExtractHoistableLoad(MachineInstr *MI) {
  // If not, we may be able to unfold a load and hoist that.
  // First test whether the instruction is loading from an amenable
  // memory location.
  if (!isLoadFromConstantMemory(MI))
    return 0;

  // Next determine the register class for a temporary register.
  unsigned LoadRegIndex;
  unsigned NewOpc =
    TII->getOpcodeAfterMemoryUnfold(MI->getOpcode(),
                                    /*UnfoldLoad=*/true,
                                    /*UnfoldStore=*/false,
                                    &LoadRegIndex);
  if (NewOpc == 0) return 0;
  const TargetInstrDesc &TID = TII->get(NewOpc);
  if (TID.getNumDefs() != 1) return 0;
  const TargetRegisterClass *RC = TID.OpInfo[LoadRegIndex].getRegClass(TRI);
  // Ok, we're unfolding. Create a temporary register and do the unfold.
  unsigned Reg = RegInfo->createVirtualRegister(RC);

  MachineFunction &MF = *MI->getParent()->getParent();
  SmallVector<MachineInstr *, 2> NewMIs;
  bool Success =
    TII->unfoldMemoryOperand(MF, MI, Reg,
                             /*UnfoldLoad=*/true, /*UnfoldStore=*/false,
                             NewMIs);
  (void)Success;
  assert(Success &&
         "unfoldMemoryOperand failed when getOpcodeAfterMemoryUnfold "
         "succeeded!");
  assert(NewMIs.size() == 2 &&
         "Unfolded a load into multiple instructions!");
  MachineBasicBlock *MBB = MI->getParent();
  MBB->insert(MI, NewMIs[0]);
  MBB->insert(MI, NewMIs[1]);
  // If unfolding produced a load that wasn't loop-invariant or profitable to
  // hoist, discard the new instructions and bail.
  if (!IsLoopInvariantInst(*NewMIs[0]) || !IsProfitableToHoist(*NewMIs[0])) {
    NewMIs[0]->eraseFromParent();
    NewMIs[1]->eraseFromParent();
    return 0;
  }
  // Otherwise we successfully unfolded a load that we can hoist.
  MI->eraseFromParent();
  return NewMIs[0];
}

void MachineLICM::InitCSEMap(MachineBasicBlock *BB) {
  for (MachineBasicBlock::iterator I = BB->begin(),E = BB->end(); I != E; ++I) {
    const MachineInstr *MI = &*I;
    // FIXME: For now, only hoist re-materilizable instructions. LICM will
    // increase register pressure. We want to make sure it doesn't increase
    // spilling.
    if (TII->isTriviallyReMaterializable(MI, AA)) {
      unsigned Opcode = MI->getOpcode();
      DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator
        CI = CSEMap.find(Opcode);
      if (CI != CSEMap.end())
        CI->second.push_back(MI);
      else {
        std::vector<const MachineInstr*> CSEMIs;
        CSEMIs.push_back(MI);
        CSEMap.insert(std::make_pair(Opcode, CSEMIs));
      }
    }
  }
}

const MachineInstr*
MachineLICM::LookForDuplicate(const MachineInstr *MI,
                              std::vector<const MachineInstr*> &PrevMIs) {
  for (unsigned i = 0, e = PrevMIs.size(); i != e; ++i) {
    const MachineInstr *PrevMI = PrevMIs[i];
    if (TII->produceSameValue(MI, PrevMI))
      return PrevMI;
  }
  return 0;
}

bool MachineLICM::EliminateCSE(MachineInstr *MI,
          DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator &CI) {
  if (CI == CSEMap.end())
    return false;

  if (const MachineInstr *Dup = LookForDuplicate(MI, CI->second)) {
    DEBUG(dbgs() << "CSEing " << *MI << " with " << *Dup);

    // Replace virtual registers defined by MI by their counterparts defined
    // by Dup.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);

      // Physical registers may not differ here.
      assert((!MO.isReg() || MO.getReg() == 0 ||
              !TargetRegisterInfo::isPhysicalRegister(MO.getReg()) ||
              MO.getReg() == Dup->getOperand(i).getReg()) &&
             "Instructions with different phys regs are not identical!");

      if (MO.isReg() && MO.isDef() &&
          !TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
        RegInfo->replaceRegWith(MO.getReg(), Dup->getOperand(i).getReg());
    }
    MI->eraseFromParent();
    ++NumCSEed;
    return true;
  }
  return false;
}

/// Hoist - When an instruction is found to use only loop invariant operands
/// that are safe to hoist, this instruction is called to do the dirty work.
///
void MachineLICM::Hoist(MachineInstr *MI) {
  // First check whether we should hoist this instruction.
  if (!IsLoopInvariantInst(*MI) || !IsProfitableToHoist(*MI)) {
    // If not, try unfolding a hoistable load.
    MI = ExtractHoistableLoad(MI);
    if (!MI) return;
  }

  // Now move the instructions to the predecessor, inserting it before any
  // terminator instructions.
  DEBUG({
      dbgs() << "Hoisting " << *MI;
      if (CurPreheader->getBasicBlock())
        dbgs() << " to MachineBasicBlock "
               << CurPreheader->getName();
      if (MI->getParent()->getBasicBlock())
        dbgs() << " from MachineBasicBlock "
               << MI->getParent()->getName();
      dbgs() << "\n";
    });

  // If this is the first instruction being hoisted to the preheader,
  // initialize the CSE map with potential common expressions.
  InitCSEMap(CurPreheader);

  // Look for opportunity to CSE the hoisted instruction.
  unsigned Opcode = MI->getOpcode();
  DenseMap<unsigned, std::vector<const MachineInstr*> >::iterator
    CI = CSEMap.find(Opcode);
  if (!EliminateCSE(MI, CI)) {
    // Otherwise, splice the instruction to the preheader.
    CurPreheader->splice(CurPreheader->getFirstTerminator(),MI->getParent(),MI);

    // Add to the CSE map.
    if (CI != CSEMap.end())
      CI->second.push_back(MI);
    else {
      std::vector<const MachineInstr*> CSEMIs;
      CSEMIs.push_back(MI);
      CSEMap.insert(std::make_pair(Opcode, CSEMIs));
    }
  }

  ++NumHoisted;
  Changed = true;
}

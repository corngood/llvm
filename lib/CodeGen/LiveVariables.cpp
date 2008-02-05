//===-- LiveVariables.cpp - Live Variable Analysis for Machine Code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveVariable analysis pass.  For each machine
// instruction in the function, this pass calculates the set of registers that
// are immediately dead after the instruction (i.e., the instruction calculates
// the value, but it is never used) and the set of registers that are used by
// the instruction, but are never used after the instruction (i.e., they are
// killed).
//
// This class computes live variables using are sparse implementation based on
// the machine code SSA form.  This class computes live variable information for
// each virtual and _register allocatable_ physical register in a function.  It
// uses the dominance properties of SSA form to efficiently compute live
// variables for virtual registers, and assumes that physical registers are only
// live within a single basic block (allowing it to do a single local analysis
// to resolve physical register lifetimes in each basic block).  If a physical
// register is not register allocatable, it is not tracked.  This is useful for
// things like the stack pointer and condition codes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/alloca.h"
#include <algorithm>
using namespace llvm;

char LiveVariables::ID = 0;
static RegisterPass<LiveVariables> X("livevars", "Live Variable Analysis");

void LiveVariables::VarInfo::dump() const {
  cerr << "  Alive in blocks: ";
  for (unsigned i = 0, e = AliveBlocks.size(); i != e; ++i)
    if (AliveBlocks[i]) cerr << i << ", ";
  cerr << "  Used in blocks: ";
  for (unsigned i = 0, e = UsedBlocks.size(); i != e; ++i)
    if (UsedBlocks[i]) cerr << i << ", ";
  cerr << "\n  Killed by:";
  if (Kills.empty())
    cerr << " No instructions.\n";
  else {
    for (unsigned i = 0, e = Kills.size(); i != e; ++i)
      cerr << "\n    #" << i << ": " << *Kills[i];
    cerr << "\n";
  }
}

LiveVariables::VarInfo &LiveVariables::getVarInfo(unsigned RegIdx) {
  assert(MRegisterInfo::isVirtualRegister(RegIdx) &&
         "getVarInfo: not a virtual register!");
  RegIdx -= MRegisterInfo::FirstVirtualRegister;
  if (RegIdx >= VirtRegInfo.size()) {
    if (RegIdx >= 2*VirtRegInfo.size())
      VirtRegInfo.resize(RegIdx*2);
    else
      VirtRegInfo.resize(2*VirtRegInfo.size());
  }
  VarInfo &VI = VirtRegInfo[RegIdx];
  VI.AliveBlocks.resize(MF->getNumBlockIDs());
  VI.UsedBlocks.resize(MF->getNumBlockIDs());
  return VI;
}

bool LiveVariables::KillsRegister(MachineInstr *MI, unsigned Reg) const {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegister() && MO.isKill()) {
      if ((MO.getReg() == Reg) ||
          (MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           MRegisterInfo::isPhysicalRegister(Reg) &&
           RegInfo->isSubRegister(MO.getReg(), Reg)))
        return true;
    }
  }
  return false;
}

bool LiveVariables::RegisterDefIsDead(MachineInstr *MI, unsigned Reg) const {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegister() && MO.isDead()) {
      if ((MO.getReg() == Reg) ||
          (MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
           MRegisterInfo::isPhysicalRegister(Reg) &&
           RegInfo->isSubRegister(MO.getReg(), Reg)))
        return true;
    }
  }
  return false;
}

bool LiveVariables::ModifiesRegister(MachineInstr *MI, unsigned Reg) const {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegister() && MO.isDef() && MO.getReg() == Reg)
      return true;
  }
  return false;
}

void LiveVariables::MarkVirtRegAliveInBlock(VarInfo& VRInfo,
                                            MachineBasicBlock *DefBlock,
                                            MachineBasicBlock *MBB,
                                    std::vector<MachineBasicBlock*> &WorkList) {
  unsigned BBNum = MBB->getNumber();
  
  // Check to see if this basic block is one of the killing blocks.  If so,
  // remove it...
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    if (VRInfo.Kills[i]->getParent() == MBB) {
      VRInfo.Kills.erase(VRInfo.Kills.begin()+i);  // Erase entry
      break;
    }
  
  if (MBB == DefBlock) return;  // Terminate recursion

  if (VRInfo.AliveBlocks[BBNum])
    return;  // We already know the block is live

  // Mark the variable known alive in this bb
  VRInfo.AliveBlocks[BBNum] = true;

  for (MachineBasicBlock::const_pred_reverse_iterator PI = MBB->pred_rbegin(),
         E = MBB->pred_rend(); PI != E; ++PI)
    WorkList.push_back(*PI);
}

void LiveVariables::MarkVirtRegAliveInBlock(VarInfo& VRInfo,
                                            MachineBasicBlock *DefBlock,
                                            MachineBasicBlock *MBB) {
  std::vector<MachineBasicBlock*> WorkList;
  MarkVirtRegAliveInBlock(VRInfo, DefBlock, MBB, WorkList);
  while (!WorkList.empty()) {
    MachineBasicBlock *Pred = WorkList.back();
    WorkList.pop_back();
    MarkVirtRegAliveInBlock(VRInfo, DefBlock, Pred, WorkList);
  }
}


void LiveVariables::HandleVirtRegUse(unsigned reg, MachineBasicBlock *MBB,
                                     MachineInstr *MI) {
  MachineRegisterInfo& MRI = MBB->getParent()->getRegInfo();
  assert(MRI.getVRegDef(reg) && "Register use before def!");

  unsigned BBNum = MBB->getNumber();

  VarInfo& VRInfo = getVarInfo(reg);
  VRInfo.UsedBlocks[BBNum] = true;
  VRInfo.NumUses++;

  // Check to see if this basic block is already a kill block...
  if (!VRInfo.Kills.empty() && VRInfo.Kills.back()->getParent() == MBB) {
    // Yes, this register is killed in this basic block already.  Increase the
    // live range by updating the kill instruction.
    VRInfo.Kills.back() = MI;
    return;
  }

#ifndef NDEBUG
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    assert(VRInfo.Kills[i]->getParent() != MBB && "entry should be at end!");
#endif

  assert(MBB != MRI.getVRegDef(reg)->getParent() &&
         "Should have kill for defblock!");

  // Add a new kill entry for this basic block.
  // If this virtual register is already marked as alive in this basic block,
  // that means it is alive in at least one of the successor block, it's not
  // a kill.
  if (!VRInfo.AliveBlocks[BBNum])
    VRInfo.Kills.push_back(MI);

  // Update all dominating blocks to mark them known live.
  for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         E = MBB->pred_end(); PI != E; ++PI)
    MarkVirtRegAliveInBlock(VRInfo, MRI.getVRegDef(reg)->getParent(), *PI);
}

void LiveVariables::HandlePhysRegUse(unsigned Reg, MachineInstr *MI) {
  // Turn previous partial def's into read/mod/write.
  for (unsigned i = 0, e = PhysRegPartDef[Reg].size(); i != e; ++i) {
    MachineInstr *Def = PhysRegPartDef[Reg][i];
    // First one is just a def. This means the use is reading some undef bits.
    if (i != 0)
      Def->addOperand(MachineOperand::CreateReg(Reg, false/*IsDef*/,
                                                true/*IsImp*/,true/*IsKill*/));
    Def->addOperand(MachineOperand::CreateReg(Reg,true/*IsDef*/,true/*IsImp*/));
  }
  PhysRegPartDef[Reg].clear();

  // There was an earlier def of a super-register. Add implicit def to that MI.
  // A: EAX = ...
  // B:     = AX
  // Add implicit def to A.
  if (PhysRegInfo[Reg] && PhysRegInfo[Reg] != PhysRegPartUse[Reg] &&
      !PhysRegUsed[Reg]) {
    MachineInstr *Def = PhysRegInfo[Reg];
    if (!Def->findRegisterDefOperand(Reg))
      Def->addOperand(MachineOperand::CreateReg(Reg, true/*IsDef*/,
                                                true/*IsImp*/));
  }

  // There is a now a proper use, forget about the last partial use.
  PhysRegPartUse[Reg] = NULL;
  PhysRegInfo[Reg] = MI;
  PhysRegUsed[Reg] = true;

  for (const unsigned *SubRegs = RegInfo->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs) {
    PhysRegInfo[SubReg] = MI;
    PhysRegUsed[SubReg] = true;
  }

  for (const unsigned *SuperRegs = RegInfo->getSuperRegisters(Reg);
       unsigned SuperReg = *SuperRegs; ++SuperRegs) {
    // Remember the partial use of this superreg if it was previously defined.
    bool HasPrevDef = PhysRegInfo[SuperReg] != NULL;
    if (!HasPrevDef) {
      for (const unsigned *SSRegs = RegInfo->getSuperRegisters(SuperReg);
           unsigned SSReg = *SSRegs; ++SSRegs) {
        if (PhysRegInfo[SSReg] != NULL) {
          HasPrevDef = true;
          break;
        }
      }
    }
    if (HasPrevDef) {
      PhysRegInfo[SuperReg] = MI;
      PhysRegPartUse[SuperReg] = MI;
    }
  }
}

bool LiveVariables::HandlePhysRegKill(unsigned Reg, MachineInstr *RefMI,
                                      SmallSet<unsigned, 4> &SubKills) {
  for (const unsigned *SubRegs = RegInfo->getImmediateSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs) {
    MachineInstr *LastRef = PhysRegInfo[SubReg];
    if (LastRef != RefMI ||
        !HandlePhysRegKill(SubReg, RefMI, SubKills))
      SubKills.insert(SubReg);
  }

  if (*RegInfo->getImmediateSubRegisters(Reg) == 0) {
    // No sub-registers, just check if reg is killed by RefMI.
    if (PhysRegInfo[Reg] == RefMI)
      return true;
  } else if (SubKills.empty())
    // None of the sub-registers are killed elsewhere...
    return true;
  return false;
}

void LiveVariables::addRegisterKills(unsigned Reg, MachineInstr *MI,
                                     SmallSet<unsigned, 4> &SubKills) {
  if (SubKills.count(Reg) == 0)
    MI->addRegisterKilled(Reg, RegInfo, true);
  else {
    for (const unsigned *SubRegs = RegInfo->getImmediateSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs)
      addRegisterKills(SubReg, MI, SubKills);
  }
}

bool LiveVariables::HandlePhysRegKill(unsigned Reg, MachineInstr *RefMI) {
  SmallSet<unsigned, 4> SubKills;
  if (HandlePhysRegKill(Reg, RefMI, SubKills)) {
    RefMI->addRegisterKilled(Reg, RegInfo, true);
    return true;
  } else {
    // Some sub-registers are killed by another MI.
    for (const unsigned *SubRegs = RegInfo->getImmediateSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs)
      addRegisterKills(SubReg, RefMI, SubKills);
    return false;
  }
}

void LiveVariables::HandlePhysRegDef(unsigned Reg, MachineInstr *MI) {
  // Does this kill a previous version of this register?
  if (MachineInstr *LastRef = PhysRegInfo[Reg]) {
    if (PhysRegUsed[Reg]) {
      if (!HandlePhysRegKill(Reg, LastRef)) {
        if (PhysRegPartUse[Reg])
          PhysRegPartUse[Reg]->addRegisterKilled(Reg, RegInfo, true);
      }
    } else if (PhysRegPartUse[Reg])
      // Add implicit use / kill to last partial use.
      PhysRegPartUse[Reg]->addRegisterKilled(Reg, RegInfo, true);
    else if (LastRef != MI)
      // Defined, but not used. However, watch out for cases where a super-reg
      // is also defined on the same MI.
      LastRef->addRegisterDead(Reg, RegInfo);
  }

  for (const unsigned *SubRegs = RegInfo->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs) {
    if (MachineInstr *LastRef = PhysRegInfo[SubReg]) {
      if (PhysRegUsed[SubReg]) {
        if (!HandlePhysRegKill(SubReg, LastRef)) {
          if (PhysRegPartUse[SubReg])
            PhysRegPartUse[SubReg]->addRegisterKilled(SubReg, RegInfo, true);
        }
      } else if (PhysRegPartUse[SubReg])
        // Add implicit use / kill to last use of a sub-register.
        PhysRegPartUse[SubReg]->addRegisterKilled(SubReg, RegInfo, true);
      else if (LastRef != MI)
        // This must be a def of the subreg on the same MI.
        LastRef->addRegisterDead(SubReg, RegInfo);
    }
  }

  if (MI) {
    for (const unsigned *SuperRegs = RegInfo->getSuperRegisters(Reg);
         unsigned SuperReg = *SuperRegs; ++SuperRegs) {
      if (PhysRegInfo[SuperReg] && PhysRegInfo[SuperReg] != MI) {
        // The larger register is previously defined. Now a smaller part is
        // being re-defined. Treat it as read/mod/write.
        // EAX =
        // AX  =        EAX<imp-use,kill>, EAX<imp-def>
        MI->addOperand(MachineOperand::CreateReg(SuperReg, false/*IsDef*/,
                                                 true/*IsImp*/,true/*IsKill*/));
        MI->addOperand(MachineOperand::CreateReg(SuperReg, true/*IsDef*/,
                                                 true/*IsImp*/));
        PhysRegInfo[SuperReg] = MI;
        PhysRegUsed[SuperReg] = false;
        PhysRegPartUse[SuperReg] = NULL;
      } else {
        // Remember this partial def.
        PhysRegPartDef[SuperReg].push_back(MI);
      }
    }

    PhysRegInfo[Reg] = MI;
    PhysRegUsed[Reg] = false;
    PhysRegPartDef[Reg].clear();
    PhysRegPartUse[Reg] = NULL;
    for (const unsigned *SubRegs = RegInfo->getSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs) {
      PhysRegInfo[SubReg] = MI;
      PhysRegUsed[SubReg] = false;
      PhysRegPartDef[SubReg].clear();
      PhysRegPartUse[SubReg] = NULL;
    }
  }
}

bool LiveVariables::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  RegInfo = MF->getTarget().getRegisterInfo();
  MachineRegisterInfo& MRI = mf.getRegInfo();
  assert(RegInfo && "Target doesn't have register information?");

  ReservedRegisters = RegInfo->getReservedRegs(mf);

  unsigned NumRegs = RegInfo->getNumRegs();
  PhysRegInfo = new MachineInstr*[NumRegs];
  PhysRegUsed = new bool[NumRegs];
  PhysRegPartUse = new MachineInstr*[NumRegs];
  PhysRegPartDef = new SmallVector<MachineInstr*,4>[NumRegs];
  PHIVarInfo = new SmallVector<unsigned, 4>[MF->getNumBlockIDs()];
  std::fill(PhysRegInfo, PhysRegInfo + NumRegs, (MachineInstr*)0);
  std::fill(PhysRegUsed, PhysRegUsed + NumRegs, false);
  std::fill(PhysRegPartUse, PhysRegPartUse + NumRegs, (MachineInstr*)0);

  /// Get some space for a respectable number of registers...
  VirtRegInfo.resize(64);

  analyzePHINodes(mf);

  // Calculate live variable information in depth first order on the CFG of the
  // function.  This guarantees that we will see the definition of a virtual
  // register before its uses due to dominance properties of SSA (except for PHI
  // nodes, which are treated as a special case).
  //
  MachineBasicBlock *Entry = MF->begin();
  SmallPtrSet<MachineBasicBlock*,16> Visited;
  for (df_ext_iterator<MachineBasicBlock*, SmallPtrSet<MachineBasicBlock*,16> >
         DFI = df_ext_begin(Entry, Visited), E = df_ext_end(Entry, Visited);
       DFI != E; ++DFI) {
    MachineBasicBlock *MBB = *DFI;

    // Mark live-in registers as live-in.
    for (MachineBasicBlock::const_livein_iterator II = MBB->livein_begin(),
           EE = MBB->livein_end(); II != EE; ++II) {
      assert(MRegisterInfo::isPhysicalRegister(*II) &&
             "Cannot have a live-in virtual register!");
      HandlePhysRegDef(*II, 0);
    }

    // Loop over all of the instructions, processing them.
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      MachineInstr *MI = I;

      // Process all of the operands of the instruction...
      unsigned NumOperandsToProcess = MI->getNumOperands();

      // Unless it is a PHI node.  In this case, ONLY process the DEF, not any
      // of the uses.  They will be handled in other basic blocks.
      if (MI->getOpcode() == TargetInstrInfo::PHI)
        NumOperandsToProcess = 1;

      // Process all uses...
      for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isRegister() && MO.isUse() && MO.getReg()) {
          if (MRegisterInfo::isVirtualRegister(MO.getReg())){
            HandleVirtRegUse(MO.getReg(), MBB, MI);
          } else if (MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
                     !ReservedRegisters[MO.getReg()]) {
            HandlePhysRegUse(MO.getReg(), MI);
          }
        }
      }

      // Process all defs...
      for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isRegister() && MO.isDef() && MO.getReg()) {
          if (MRegisterInfo::isVirtualRegister(MO.getReg())) {
            VarInfo &VRInfo = getVarInfo(MO.getReg());
            if (VRInfo.AliveBlocks.none())
              // If vr is not alive in any block, then defaults to dead.
              VRInfo.Kills.push_back(MI);
          } else if (MRegisterInfo::isPhysicalRegister(MO.getReg()) &&
                     !ReservedRegisters[MO.getReg()]) {
            HandlePhysRegDef(MO.getReg(), MI);
          }
        }
      }
    }

    // Handle any virtual assignments from PHI nodes which might be at the
    // bottom of this basic block.  We check all of our successor blocks to see
    // if they have PHI nodes, and if so, we simulate an assignment at the end
    // of the current block.
    if (!PHIVarInfo[MBB->getNumber()].empty()) {
      SmallVector<unsigned, 4>& VarInfoVec = PHIVarInfo[MBB->getNumber()];

      for (SmallVector<unsigned, 4>::iterator I = VarInfoVec.begin(),
             E = VarInfoVec.end(); I != E; ++I) {
        // Only mark it alive only in the block we are representing.
        MarkVirtRegAliveInBlock(getVarInfo(*I), MRI.getVRegDef(*I)->getParent(),
                                MBB);
      }
    }

    // Finally, if the last instruction in the block is a return, make sure to mark
    // it as using all of the live-out values in the function.
    if (!MBB->empty() && MBB->back().getDesc().isReturn()) {
      MachineInstr *Ret = &MBB->back();
      for (MachineRegisterInfo::liveout_iterator
           I = MF->getRegInfo().liveout_begin(),
           E = MF->getRegInfo().liveout_end(); I != E; ++I) {
        assert(MRegisterInfo::isPhysicalRegister(*I) &&
               "Cannot have a live-in virtual register!");
        HandlePhysRegUse(*I, Ret);
        // Add live-out registers as implicit uses.
        if (Ret->findRegisterUseOperandIdx(*I) == -1)
          Ret->addOperand(MachineOperand::CreateReg(*I, false, true));
      }
    }

    // Loop over PhysRegInfo, killing any registers that are available at the
    // end of the basic block.  This also resets the PhysRegInfo map.
    for (unsigned i = 0; i != NumRegs; ++i)
      if (PhysRegInfo[i])
        HandlePhysRegDef(i, 0);

    // Clear some states between BB's. These are purely local information.
    for (unsigned i = 0; i != NumRegs; ++i)
      PhysRegPartDef[i].clear();
    std::fill(PhysRegInfo, PhysRegInfo + NumRegs, (MachineInstr*)0);
    std::fill(PhysRegUsed, PhysRegUsed + NumRegs, false);
    std::fill(PhysRegPartUse, PhysRegPartUse + NumRegs, (MachineInstr*)0);
  }

  // Convert and transfer the dead / killed information we have gathered into
  // VirtRegInfo onto MI's.
  //
  for (unsigned i = 0, e1 = VirtRegInfo.size(); i != e1; ++i)
    for (unsigned j = 0, e2 = VirtRegInfo[i].Kills.size(); j != e2; ++j) {
      if (VirtRegInfo[i].Kills[j] == MRI.getVRegDef(i + 
                                           MRegisterInfo::FirstVirtualRegister))
        VirtRegInfo[i].Kills[j]->addRegisterDead(i +
                                            MRegisterInfo::FirstVirtualRegister,
                                                 RegInfo);
      else
        VirtRegInfo[i].Kills[j]->addRegisterKilled(i +
                                            MRegisterInfo::FirstVirtualRegister,
                                                  RegInfo);
    }

  // Check to make sure there are no unreachable blocks in the MC CFG for the
  // function.  If so, it is due to a bug in the instruction selector or some
  // other part of the code generator if this happens.
#ifndef NDEBUG
  for(MachineFunction::iterator i = MF->begin(), e = MF->end(); i != e; ++i)
    assert(Visited.count(&*i) != 0 && "unreachable basic block found");
#endif

  delete[] PhysRegInfo;
  delete[] PhysRegUsed;
  delete[] PhysRegPartUse;
  delete[] PhysRegPartDef;
  delete[] PHIVarInfo;

  return false;
}

/// instructionChanged - When the address of an instruction changes, this
/// method should be called so that live variables can update its internal
/// data structures.  This removes the records for OldMI, transfering them to
/// the records for NewMI.
void LiveVariables::instructionChanged(MachineInstr *OldMI,
                                       MachineInstr *NewMI) {
  // If the instruction defines any virtual registers, update the VarInfo,
  // kill and dead information for the instruction.
  for (unsigned i = 0, e = OldMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = OldMI->getOperand(i);
    if (MO.isRegister() && MO.getReg() &&
        MRegisterInfo::isVirtualRegister(MO.getReg())) {
      unsigned Reg = MO.getReg();
      VarInfo &VI = getVarInfo(Reg);
      if (MO.isDef()) {
        if (MO.isDead()) {
          MO.setIsDead(false);
          addVirtualRegisterDead(Reg, NewMI);
        }
      }
      if (MO.isKill()) {
        MO.setIsKill(false);
        addVirtualRegisterKilled(Reg, NewMI);
      }
      // If this is a kill of the value, update the VI kills list.
      if (VI.removeKill(OldMI))
        VI.Kills.push_back(NewMI);   // Yes, there was a kill of it
    }
  }
}

/// removeVirtualRegistersKilled - Remove all killed info for the specified
/// instruction.
void LiveVariables::removeVirtualRegistersKilled(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegister() && MO.isKill()) {
      MO.setIsKill(false);
      unsigned Reg = MO.getReg();
      if (MRegisterInfo::isVirtualRegister(Reg)) {
        bool removed = getVarInfo(Reg).removeKill(MI);
        assert(removed && "kill not in register's VarInfo?");
      }
    }
  }
}

/// removeVirtualRegistersDead - Remove all of the dead registers for the
/// specified instruction from the live variable information.
void LiveVariables::removeVirtualRegistersDead(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegister() && MO.isDead()) {
      MO.setIsDead(false);
      unsigned Reg = MO.getReg();
      if (MRegisterInfo::isVirtualRegister(Reg)) {
        bool removed = getVarInfo(Reg).removeKill(MI);
        assert(removed && "kill not in register's VarInfo?");
      }
    }
  }
}

/// analyzePHINodes - Gather information about the PHI nodes in here. In
/// particular, we want to map the variable information of a virtual
/// register which is used in a PHI node. We map that to the BB the vreg is
/// coming from.
///
void LiveVariables::analyzePHINodes(const MachineFunction& Fn) {
  for (MachineFunction::const_iterator I = Fn.begin(), E = Fn.end();
       I != E; ++I)
    for (MachineBasicBlock::const_iterator BBI = I->begin(), BBE = I->end();
         BBI != BBE && BBI->getOpcode() == TargetInstrInfo::PHI; ++BBI)
      for (unsigned i = 1, e = BBI->getNumOperands(); i != e; i += 2)
        PHIVarInfo[BBI->getOperand(i + 1).getMBB()->getNumber()].
          push_back(BBI->getOperand(i).getReg());
}

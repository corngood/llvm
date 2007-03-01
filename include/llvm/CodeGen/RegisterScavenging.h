//===-- RegisterScavenging.h - Machine register scavenging ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the machine register scavenger class. It can provide
// information such as unused register at any point in a machine basic block.
// It also provides a mechanism to make registers availbale by evicting them
// to spill slots.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTER_SCAVENGING_H
#define LLVM_CODEGEN_REGISTER_SCAVENGING_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/BitVector.h"

namespace llvm {

class TargetRegisterClass;

class RegScavenger {
  MachineBasicBlock *MBB;
  MachineBasicBlock::iterator MBBI;
  unsigned NumPhysRegs;

  /// Tracking - True if RegScavenger is currently tracking the liveness of 
  /// registers.
  bool Tracking;

  /// RegStates - The current state of all the physical registers immediately
  /// before MBBI. One bit per physical register. If bit is set that means it's
  /// available, unset means the register is currently being used.
  BitVector RegStates;

public:
  RegScavenger()
    : MBB(NULL), NumPhysRegs(0), Tracking(false) {};

  RegScavenger(MachineBasicBlock *mbb)
    : MBB(mbb), NumPhysRegs(0), Tracking(false) {};

  /// enterBasicBlock - Start tracking liveness from the begin of the specific
  /// basic block.
  void enterBasicBlock(MachineBasicBlock *mbb);

  /// forward / backward - Move the internal MBB iterator and update register
  /// states.
  void forward();
  void backward();

  /// forward / backward - Move the internal MBB iterator and update register
  /// states until it has reached but not processed the specific iterator.
  void forward(MachineBasicBlock::iterator I) {
    while (MBBI != I) forward();
  }
  void backward(MachineBasicBlock::iterator I) {
    while (MBBI != I) backward();
  }

  /// isReserved - Returns true if a register is reserved. It is never "unused".
  bool isReserved(unsigned Reg) const { return ReservedRegs[Reg]; }

  /// isUsed / isUsed - Test if a register is currently being used.
  ///
  bool isUsed(unsigned Reg) const   { return !RegStates[Reg]; }
  bool isUnused(unsigned Reg) const { return RegStates[Reg]; }

  /// setUsed / setUnused - Mark the state of one or a number of registers.
  ///
  void setUsed(unsigned Reg)     { RegStates.reset(Reg); }
  void setUsed(BitVector Regs)   { RegStates &= ~Regs; }
  void setUnused(unsigned Reg)   { RegStates.set(Reg); }
  void setUnused(BitVector Regs) { RegStates |= Regs; }

  /// FindUnusedReg - Find a unused register of the specified register class
  /// from the specified set of registers. It return 0 is none is found.
  unsigned FindUnusedReg(const TargetRegisterClass *RegClass,
                         const BitVector &Candidates) const;

  /// FindUnusedReg - Find a unused register of the specified register class.
  /// Exclude callee saved registers if directed. It return 0 is none is found.
  unsigned FindUnusedReg(const TargetRegisterClass *RegClass,
                         bool ExCalleeSaved = false) const;

private:
  /// CalleeSavedrRegs - A bitvector of callee saved registers for the target.
  ///
  BitVector CalleeSavedRegs;

  /// ReservedRegs - A bitvector of reserved registers.
  ///
  BitVector ReservedRegs;
};
 
} // End llvm namespace

#endif

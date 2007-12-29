//===- AlphaInstrInfo.h - Alpha Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Alpha implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHAINSTRUCTIONINFO_H
#define ALPHAINSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "AlphaRegisterInfo.h"

namespace llvm {

class AlphaInstrInfo : public TargetInstrInfo {
  const AlphaRegisterInfo RI;
public:
  AlphaInstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// Return true if the instruction is a register to register move and
  /// leave the source and dest operands in the passed parameters.
  ///
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg) const;
  
  virtual unsigned isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const;
  virtual unsigned isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const;
  
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const std::vector<MachineOperand> &Cond) const;
  bool AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     std::vector<MachineOperand> &Cond) const;
  unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  void insertNoop(MachineBasicBlock &MBB, 
                  MachineBasicBlock::iterator MI) const;
  bool BlockHasNoFallThrough(MachineBasicBlock &MBB) const;
  bool ReverseBranchCondition(std::vector<MachineOperand> &Cond) const;
};

}

#endif

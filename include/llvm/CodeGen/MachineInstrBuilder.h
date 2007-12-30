//===-- CodeGen/MachineInstBuilder.h - Simplify creation of MIs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a function named BuildMI, which is useful for dramatically
// simplifying how MachineInstr's are created.  It allows use of code like this:
//
//   M = BuildMI(X86::ADDrr8, 2).addReg(argVal1).addReg(argVal2);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUILDER_H
#define LLVM_CODEGEN_MACHINEINSTRBUILDER_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class TargetInstrDescriptor;

class MachineInstrBuilder {
  MachineInstr *MI;
public:
  explicit MachineInstrBuilder(MachineInstr *mi) : MI(mi) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  ///
  operator MachineInstr*() const { return MI; }
  operator MachineBasicBlock::iterator() const { return MI; }

  /// addReg - Add a new virtual register operand...
  ///
  const
  MachineInstrBuilder &addReg(unsigned RegNo, bool isDef = false, 
                              bool isImp = false, bool isKill = false, 
                              bool isDead = false, unsigned SubReg = 0) const {
    MI->addOperand(MachineOperand::CreateReg(RegNo, isDef, isImp, isKill,
                                             isDead, SubReg));
    return *this;
  }

  /// addImm - Add a new immediate operand.
  ///
  const MachineInstrBuilder &addImm(int64_t Val) const {
    MI->addOperand(MachineOperand::CreateImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addMBB(MachineBasicBlock *MBB) const {
    MI->addOperand(MachineOperand::CreateMBB(MBB));
    return *this;
  }

  const MachineInstrBuilder &addFrameIndex(unsigned Idx) const {
    MI->addOperand(MachineOperand::CreateFI(Idx));
    return *this;
  }

  const MachineInstrBuilder &addConstantPoolIndex(unsigned Idx,
                                                  int Offset = 0) const {
    MI->addOperand(MachineOperand::CreateCPI(Idx, Offset));
    return *this;
  }

  const MachineInstrBuilder &addJumpTableIndex(unsigned Idx) const {
    MI->addOperand(MachineOperand::CreateJTI(Idx));
    return *this;
  }

  const MachineInstrBuilder &addGlobalAddress(GlobalValue *GV,
                                              int Offset = 0) const {
    MI->addOperand(MachineOperand::CreateGA(GV, Offset));
    return *this;
  }

  const MachineInstrBuilder &addExternalSymbol(const char *FnName) const{
    MI->addOperand(MachineOperand::CreateES(FnName, 0));
    return *this;
  }
};

/// BuildMI - Builder interface.  Specify how to create the initial instruction
/// itself.
///
inline MachineInstrBuilder BuildMI(const TargetInstrDescriptor &TID) {
  return MachineInstrBuilder(new MachineInstr(TID));
}

/// BuildMI - This version of the builder sets up the first operand as a
/// destination virtual register.
///
inline MachineInstrBuilder  BuildMI(const TargetInstrDescriptor &TID,
                                    unsigned DestReg) {
  return MachineInstrBuilder(new MachineInstr(TID)).addReg(DestReg, true);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// sets up the first operand as a destination virtual register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   const TargetInstrDescriptor &TID,
                                   unsigned DestReg) {
  MachineInstr *MI = new MachineInstr(TID);
  BB.insert(I, MI);
  return MachineInstrBuilder(MI).addReg(DestReg, true);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// does NOT take a destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   const TargetInstrDescriptor &TID) {
  MachineInstr *MI = new MachineInstr(TID);
  BB.insert(I, MI);
  return MachineInstrBuilder(MI);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and does NOT take a
/// destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   const TargetInstrDescriptor &TID) {
  return BuildMI(*BB, BB->end(), TID);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and sets up the first
/// operand as a destination virtual register. 
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   const TargetInstrDescriptor &TID,
                                   unsigned DestReg) {
  return BuildMI(*BB, BB->end(), TID, DestReg);
}

} // End llvm namespace

#endif

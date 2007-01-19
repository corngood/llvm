//===- ARMRegisterInfo.h - ARM Register Information Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMREGISTERINFO_H
#define ARMREGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"
#include "ARMGenRegisterInfo.h.inc"
#include <set>

namespace llvm {
  class TargetInstrInfo;
  class ARMSubtarget;
  class Type;

struct ARMRegisterInfo : public ARMGenRegisterInfo {
  const TargetInstrInfo &TII;
  const ARMSubtarget &STI;
private:
  /// FramePtr - ARM physical register used as frame ptr.
  unsigned FramePtr;

public:
  ARMRegisterInfo(const TargetInstrInfo &tii, const ARMSubtarget &STI);

  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// ARM::LR, return the number that it corresponds to (e.g. 14).
  static unsigned getRegisterNumbering(unsigned RegEnum);

  /// Code Generation virtual methods...
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;

  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           unsigned SrcReg, int FrameIndex,
                           const TargetRegisterClass *RC) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC) const;

  void copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, unsigned SrcReg,
                    const TargetRegisterClass *RC) const;

  MachineInstr* foldMemoryOperand(MachineInstr* MI, unsigned OpNum,
                                  int FrameIndex) const;

  const unsigned *getCalleeSavedRegs() const;

  const TargetRegisterClass* const* getCalleeSavedRegClasses() const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;
};

} // end namespace llvm

#endif

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

namespace llvm {
  class ARMSubtarget;
  class TargetInstrInfo;
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

  void storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                      SmallVectorImpl<MachineOperand> &Addr,
                      const TargetRegisterClass *RC,
                      SmallVectorImpl<MachineInstr*> &NewMIs) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC) const;

  void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                       SmallVectorImpl<MachineOperand> &Addr,
                       const TargetRegisterClass *RC,
                       SmallVectorImpl<MachineInstr*> &NewMIs) const;

  void copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, unsigned SrcReg,
                    const TargetRegisterClass *DestRC,
                    const TargetRegisterClass *SrcRC) const;

  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, const MachineInstr *Orig) const;

  MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                  SmallVectorImpl<unsigned> &Ops,
                                  int FrameIndex) const;

  MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                  SmallVectorImpl<unsigned> &Ops,
                                  MachineInstr* LoadMI) const {
    return 0;
  }

  const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool isReservedReg(const MachineFunction &MF, unsigned Reg) const;

  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  bool hasReservedCallFrame(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
};

} // end namespace llvm

#endif

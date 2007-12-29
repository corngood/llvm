//===- PPCRegisterInfo.h - PowerPC Register Information Impl -----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC32_REGISTERINFO_H
#define POWERPC32_REGISTERINFO_H

#include "PPC.h"
#include "PPCGenRegisterInfo.h.inc"
#include <map>

namespace llvm {
class PPCSubtarget;
class TargetInstrInfo;
class Type;

class PPCRegisterInfo : public PPCGenRegisterInfo {
  std::map<unsigned, unsigned> ImmToIdxMap;
  const PPCSubtarget &Subtarget;
  const TargetInstrInfo &TII;
public:
  PPCRegisterInfo(const PPCSubtarget &SubTarget, const TargetInstrInfo &tii);
  
  /// getRegisterNumbering - Given the enum value for some register, e.g.
  /// PPC::F14, return the number that it corresponds to (e.g. 14).
  static unsigned getRegisterNumbering(unsigned RegEnum);

  /// Code Generation virtual methods...
  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           unsigned SrcReg, bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC) const;

  void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
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

  /// foldMemoryOperand - PowerPC (like most RISC's) can only fold spills into
  /// copy instructions, turning them into load/store instructions.
  virtual MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                          SmallVectorImpl<unsigned> &Ops,
                                          int FrameIndex) const;

  virtual MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                          SmallVectorImpl<unsigned> &Ops,
                                          MachineInstr* LoadMI) const {
    return 0;
  }

  virtual bool canFoldMemoryOperand(MachineInstr *MI,
                                    SmallVectorImpl<unsigned> &Ops) const;

  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  /// targetHandlesStackFrameRounding - Returns true if the target is
  /// responsible for rounding up the stack frame (probably at emitPrologue
  /// time).
  bool targetHandlesStackFrameRounding() const { return true; }

  bool hasFP(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void lowerDynamicAlloc(MachineBasicBlock::iterator II) const;
  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  /// determineFrameLayout - Determine the size of the frame and maximum call
  /// frame size.
  void determineFrameLayout(MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;
  void getInitialFrameState(std::vector<MachineMove> &Moves) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
};

} // end namespace llvm

#endif

//====- ARMMachineFuctionInfo.h - ARM machine function info -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares ARM-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef ARMMACHINEFUNCTIONINFO_H
#define ARMMACHINEFUNCTIONINFO_H

#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// ARMFunctionInfo - This class is derived from MachineFunction private
/// ARM target-specific information for each MachineFunction.
class ARMFunctionInfo : public MachineFunctionInfo {

  /// isThumb - True if this function is compiled under Thumb mode.
  ///
  bool isThumb;

  /// VarArgsRegSaveSize - Size of the register save area for vararg functions.
  ///
  unsigned VarArgsRegSaveSize;

  /// HasStackFrame - True if this function has a stack frame. Set by
  /// processFunctionBeforeCalleeSavedScan().
  bool HasStackFrame;

  /// FramePtrSpillOffset - If HasStackFrame, this records the frame pointer
  /// spill stack offset.
  unsigned FramePtrSpillOffset;

  /// GPRCS1Offset, GPRCS2Offset, DPRCSOffset - Starting offset of callee saved
  /// register spills areas. For Mac OS X:
  ///
  /// GPR callee-saved (1) : r4, r5, r6, r7, lr
  /// --------------------------------------------
  /// GPR callee-saved (2) : r8, r10, r11
  /// --------------------------------------------
  /// DPR callee-saved : d8 - d15
  unsigned GPRCS1Offset;
  unsigned GPRCS2Offset;
  unsigned DPRCSOffset;

  /// GPRCS1Size, GPRCS2Size, DPRCSSize - Sizes of callee saved register spills
  /// areas.
  unsigned GPRCS1Size;
  unsigned GPRCS2Size;
  unsigned DPRCSSize;

  /// GPRCS1Frames, GPRCS2Frames, DPRCSFrames - Keeps track of frame indices
  /// which belong to these spill areas.
  std::set<int> GPRCS1Frames;
  std::set<int> GPRCS2Frames;
  std::set<int> DPRCSFrames;

  /// JumpTableUId - Unique id for jumptables.
  ///
  unsigned JumpTableUId;

public:
  ARMFunctionInfo() :
    isThumb(false),
    VarArgsRegSaveSize(0), HasStackFrame(false), FramePtrSpillOffset(0),
    GPRCS1Offset(0), GPRCS2Offset(0), DPRCSOffset(0),
    GPRCS1Size(0), GPRCS2Size(0), DPRCSSize(0), JumpTableUId(0) {}

  ARMFunctionInfo(MachineFunction &MF) :
    isThumb(MF.getTarget().getSubtarget<ARMSubtarget>().isThumb()),
    VarArgsRegSaveSize(0), HasStackFrame(false), FramePtrSpillOffset(0),
    GPRCS1Offset(0), GPRCS2Offset(0), DPRCSOffset(0),
    GPRCS1Size(0), GPRCS2Size(0), DPRCSSize(0), JumpTableUId(0) {}

  bool isThumbFunction() const { return isThumb; }

  unsigned getVarArgsRegSaveSize() const { return VarArgsRegSaveSize; }
  void setVarArgsRegSaveSize(unsigned s) { VarArgsRegSaveSize = s; }

  bool hasStackFrame() const { return HasStackFrame; }
  void setHasStackFrame(bool s) { HasStackFrame = s; }
  unsigned getFramePtrSpillOffset() const { return FramePtrSpillOffset; }
  void setFramePtrSpillOffset(unsigned o) { FramePtrSpillOffset = o; }
  
  unsigned getGPRCalleeSavedArea1Offset() const { return GPRCS1Offset; }
  unsigned getGPRCalleeSavedArea2Offset() const { return GPRCS2Offset; }
  unsigned getDPRCalleeSavedAreaOffset()  const { return DPRCSOffset; }

  void setGPRCalleeSavedArea1Offset(unsigned o) { GPRCS1Offset = o; }
  void setGPRCalleeSavedArea2Offset(unsigned o) { GPRCS2Offset = o; }
  void setDPRCalleeSavedAreaOffset(unsigned o)  { DPRCSOffset = o; }

  unsigned getGPRCalleeSavedArea1Size() const { return GPRCS1Size; }
  unsigned getGPRCalleeSavedArea2Size() const { return GPRCS2Size; }
  unsigned getDPRCalleeSavedAreaSize()  const { return DPRCSSize; }

  void setGPRCalleeSavedArea1Size(unsigned s) { GPRCS1Size = s; }
  void setGPRCalleeSavedArea2Size(unsigned s) { GPRCS2Size = s; }
  void setDPRCalleeSavedAreaSize(unsigned s)  { DPRCSSize = s; }

  bool isGPRCalleeSavedArea1Frame(unsigned fi) const {
    return GPRCS1Frames.count(fi);
  }
  bool isGPRCalleeSavedArea2Frame(unsigned fi) const {
    return GPRCS2Frames.count(fi);
  }
  bool isDPRCalleeSavedAreaFrame(unsigned fi) const {
    return DPRCSFrames.count(fi);
  }

  void addGPRCalleeSavedArea1Frame(unsigned fi) {
    GPRCS1Frames.insert(fi);
  }
  void addGPRCalleeSavedArea2Frame(unsigned fi) {
    GPRCS2Frames.insert(fi);
  }
  void addDPRCalleeSavedAreaFrame(unsigned fi) {
    DPRCSFrames.insert(fi);
  }

  unsigned createJumpTableUId() {
    return JumpTableUId++;
  }
};
} // End llvm namespace

#endif // ARMMACHINEFUNCTIONINFO_H

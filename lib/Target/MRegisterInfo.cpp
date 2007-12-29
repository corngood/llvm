//===- MRegisterInfo.cpp - Target Register Information Implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MRegisterInfo interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/ADT/BitVector.h"

using namespace llvm;

MRegisterInfo::MRegisterInfo(const TargetRegisterDesc *D, unsigned NR,
                             regclass_iterator RCB, regclass_iterator RCE,
                             int CFSO, int CFDO)
  : Desc(D), NumRegs(NR), RegClassBegin(RCB), RegClassEnd(RCE) {
  assert(NumRegs < FirstVirtualRegister &&
         "Target has too many physical registers!");

  CallFrameSetupOpcode   = CFSO;
  CallFrameDestroyOpcode = CFDO;
}

MRegisterInfo::~MRegisterInfo() {}

/// getPhysicalRegisterRegClass - Returns the Register Class of a physical
/// register.
const TargetRegisterClass *
MRegisterInfo::getPhysicalRegisterRegClass(MVT::ValueType VT,
                                           unsigned reg) const {
  assert(isPhysicalRegister(reg) && "reg must be a physical register");
  // Pick the register class of the right type that contains this physreg.
  for (regclass_iterator I = regclass_begin(), E = regclass_end(); I != E; ++I)
    if ((*I)->hasType(VT) && (*I)->contains(reg))
      return *I;
  assert(false && "Couldn't find the register class");
  return 0;
}


/// getAllocatableSetForRC - Toggle the bits that represent allocatable
/// registers for the specific register class.
static void getAllocatableSetForRC(MachineFunction &MF,
                                   const TargetRegisterClass *RC, BitVector &R){  
  for (TargetRegisterClass::iterator I = RC->allocation_order_begin(MF),
         E = RC->allocation_order_end(MF); I != E; ++I)
    R.set(*I);
}

BitVector MRegisterInfo::getAllocatableSet(MachineFunction &MF,
                                           const TargetRegisterClass *RC) const {
  BitVector Allocatable(NumRegs);
  if (RC) {
    getAllocatableSetForRC(MF, RC, Allocatable);
    return Allocatable;
  }

  for (MRegisterInfo::regclass_iterator I = regclass_begin(),
         E = regclass_end(); I != E; ++I)
    getAllocatableSetForRC(MF, *I, Allocatable);
  return Allocatable;
}

/// getLocation - This method should return the actual location of a frame
/// variable given the frame index.  The location is returned in ML.
/// Subclasses should override this method for special handling of frame
/// variables and then call MRegisterInfo::getLocation for the default action.
void MRegisterInfo::getLocation(MachineFunction &MF, unsigned Index,
                        MachineLocation &ML) const {
  const TargetFrameInfo &TFI = *MF.getTarget().getFrameInfo();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  ML.set(getFrameRegister(MF),
         MFI->getObjectOffset(Index) +
         MFI->getStackSize() -
         TFI.getOffsetOfLocalArea() +
         MFI->getOffsetAdjustment());
}

/// getInitialFrameState - Returns a list of machine moves that are assumed
/// on entry to a function.
void
MRegisterInfo::getInitialFrameState(std::vector<MachineMove> &Moves) const {
  // Default is to do nothing.
}


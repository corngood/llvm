//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_TARGETMACHINE_H
#define PPC_TARGETMACHINE_H

#include "PPCFrameInfo.h"
#include "PPCSubtarget.h"
#include "PPCJITInfo.h"
#include "PPCInstrInfo.h"
#include "PPCISelLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {
class PassManager;
class GlobalValue;

/// PPCTargetMachine - Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
  PPCSubtarget        Subtarget;
  const TargetData    DataLayout;       // Calculates type size & alignment
  PPCInstrInfo        InstrInfo;
  PPCFrameInfo        FrameInfo;
  PPCJITInfo          JITInfo;
  PPCTargetLowering   TLInfo;
  InstrItineraryData  InstrItins;
public:
  PPCTargetMachine(const Module &M, const std::string &FS, bool is64Bit);

  virtual const PPCInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual       TargetJITInfo    *getJITInfo()         { return &JITInfo; }
  virtual       PPCTargetLowering *getTargetLowering() const { 
   return const_cast<PPCTargetLowering*>(&TLInfo); 
  }
  virtual const MRegisterInfo    *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  
  virtual const TargetData    *getTargetData() const    { return &DataLayout; }
  virtual const PPCSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItins;
  }
  
  
  // Pass Pipeline Configuration
  virtual bool addInstSelector(FunctionPassManager &PM, bool Fast);
  virtual bool addPreEmitPass(FunctionPassManager &PM, bool Fast);
  virtual bool addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                  std::ostream &Out);
  virtual bool addObjectWriter(FunctionPassManager &PM, bool Fast,
                               std::ostream &Out);
  virtual bool addCodeEmitter(FunctionPassManager &PM, bool Fast,
                              MachineCodeEmitter &MCE);
};

/// PPC32TargetMachine - PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
public:
  PPC32TargetMachine(const Module &M, const std::string &FS);
  
  static unsigned getJITMatchQuality();
  static unsigned getModuleMatchQuality(const Module &M);
};

/// PPC64TargetMachine - PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
public:
  PPC64TargetMachine(const Module &M, const std::string &FS);
  
  static unsigned getJITMatchQuality();
  static unsigned getModuleMatchQuality(const Module &M);
};

} // end namespace llvm

#endif

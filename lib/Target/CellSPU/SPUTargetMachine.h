//===-- SPUTargetMachine.h - Define TargetMachine for Cell SPU ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by a team from the Computer Systems Research
// Department at The Aerospace Corporation.
//
// See README.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CellSPU-specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_TARGETMACHINE_H
#define SPU_TARGETMACHINE_H

#include "SPUSubtarget.h"
#include "SPUInstrInfo.h"
#include "SPUISelLowering.h"
#include "SPUFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {
class PassManager;
class GlobalValue;
class TargetFrameInfo;

/// SPUTargetMachine
///
class SPUTargetMachine : public LLVMTargetMachine {
  SPUSubtarget        Subtarget;
  const TargetData    DataLayout;
  SPUInstrInfo        InstrInfo;
  SPUFrameInfo        FrameInfo;
  SPUTargetLowering   TLInfo;
  InstrItineraryData  InstrItins;
  
protected:
  virtual const TargetAsmInfo *createTargetAsmInfo() const;
  
public:
  SPUTargetMachine(const Module &M, const std::string &FS);

  /// Return the subtarget implementation object
  virtual const SPUSubtarget     *getSubtargetImpl() const {
    return &Subtarget;
  }
  virtual const SPUInstrInfo     *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual const TargetFrameInfo  *getFrameInfo() const {
    return &FrameInfo;
  }
  /*!
    \note Cell SPU does not support JIT today. It could support JIT at some
    point.
   */
  virtual       TargetJITInfo    *getJITInfo() {
    return NULL;
  }
  
  //! Module match function
  /*!
    Module matching function called by TargetMachineRegistry().
   */
  static unsigned getModuleMatchQuality(const Module &M);

  virtual       SPUTargetLowering *getTargetLowering() const { 
   return const_cast<SPUTargetLowering*>(&TLInfo); 
  }

  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  
  virtual const TargetData *getTargetData() const {
    return &DataLayout;
  }

  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItins;
  }
  
  // Pass Pipeline Configuration
  virtual bool addInstSelector(FunctionPassManager &PM, bool Fast);
  virtual bool addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                  std::ostream &Out);
};

} // end namespace llvm

#endif

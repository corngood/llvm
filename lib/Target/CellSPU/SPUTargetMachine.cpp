//===-- SPUTargetMachine.cpp - Define TargetMachine for Cell SPU ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the Cell SPU target.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPURegisterNames.h"
#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

extern "C" void LLVMInitializeCellSPUTarget() { 
  // Register the target.
  RegisterTargetMachine<SPUTargetMachine> X(TheCellSPUTarget);
}

const std::pair<unsigned, int> *
SPUFrameInfo::getCalleeSaveSpillSlots(unsigned &NumEntries) const {
  NumEntries = 1;
  return &LR[0];
}

const TargetAsmInfo *SPUTargetMachine::createTargetAsmInfo() const {
  return new SPULinuxTargetAsmInfo();
}

SPUTargetMachine::SPUTargetMachine(const Target &T, const Module &M, 
                                   const std::string &FS)
  : LLVMTargetMachine(T),
    Subtarget(M.getTargetTriple(), FS),
    DataLayout(Subtarget.getTargetDataString()),
    InstrInfo(*this),
    FrameInfo(*this),
    TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {
  // For the time being, use static relocations, since there's really no
  // support for PIC yet.
  setRelocationModel(Reloc::Static);
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool SPUTargetMachine::addInstSelector(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createSPUISelDag(*this));
  return false;
}

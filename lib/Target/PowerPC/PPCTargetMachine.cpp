//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

namespace {
  // Register the targets
  RegisterTarget<PPC32TargetMachine>
  X("ppc32", "  PowerPC 32");
  RegisterTarget<PPC64TargetMachine>
  Y("ppc64", "  PowerPC 64");
}

unsigned PPC32TargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  if (sizeof(void*) == 4)
    return 10;
#endif
  return 0;
}
unsigned PPC64TargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  if (sizeof(void*) == 8)
    return 10;
#endif
  return 0;
}

unsigned PPC32TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "powerpc-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "powerpc-")
    return 20;
  
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target
  
  return getJITMatchQuality()/2;
}

unsigned PPC64TargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "powerpc64-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 10 && std::string(TT.begin(), TT.begin()+10) == "powerpc64-")
    return 20;
  
  if (M.getEndianness()  == Module::BigEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target
  
  return getJITMatchQuality()/2;
}


PPCTargetMachine::PPCTargetMachine(const Module &M, const std::string &FS,
                                   bool is64Bit)
  : Subtarget(M, FS, is64Bit),
    DataLayout(Subtarget.getTargetDataString()), InstrInfo(*this),
    FrameInfo(*this, false), JITInfo(*this, is64Bit), TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData()) {

  if (getRelocationModel() == Reloc::Default)
    if (Subtarget.isDarwin())
      setRelocationModel(Reloc::DynamicNoPIC);
    else
      setRelocationModel(Reloc::PIC_);
}

PPC32TargetMachine::PPC32TargetMachine(const Module &M, const std::string &FS) 
  : PPCTargetMachine(M, FS, false) {
}


PPC64TargetMachine::PPC64TargetMachine(const Module &M, const std::string &FS)
  : PPCTargetMachine(M, FS, true) {
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool PPCTargetMachine::addInstSelector(FunctionPassManager &PM, bool Fast) {
  // Install an instruction selector.
  PM.add(createPPCISelDag(*this));
  return false;
}

bool PPCTargetMachine::addPreEmitPass(FunctionPassManager &PM, bool Fast) {
  
  // Must run branch selection immediately preceding the asm printer.
  PM.add(createPPCBranchSelectionPass());
  return false;
}

bool PPCTargetMachine::addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                          std::ostream &Out) {
  PM.add(createDarwinCodePrinterPass(Out, *this));
  return false;
}

bool PPCTargetMachine::addObjectWriter(FunctionPassManager &PM, bool Fast,
                                       std::ostream &Out) {
  // FIXME: support PPC ELF files at some point
  addPPCMachOObjectWriterPass(PM, Out, *this);
  return true;
}

bool PPCTargetMachine::addCodeEmitter(FunctionPassManager &PM, bool Fast,
                                      MachineCodeEmitter &MCE) {
  // The JIT should use the static relocation model.
  // FIXME: This should be moved to TargetJITInfo!!
  setRelocationModel(Reloc::Static);

  
  
  // Machine code emitter pass for PowerPC.
  PM.add(createPPCCodeEmitterPass(*this, MCE));
  return false;
}


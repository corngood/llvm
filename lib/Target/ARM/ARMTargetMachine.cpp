//===-- ARMTargetMachine.cpp - Define TargetMachine for ARM ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ARMTargetMachine.h"
#include "ARMTargetAsmInfo.h"
#include "ARMFrameInfo.h"
#include "ARM.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

static cl::opt<bool> DisableLdStOpti("disable-arm-loadstore-opti", cl::Hidden,
                              cl::desc("Disable load store optimization pass"));
static cl::opt<bool> DisableIfConversion("disable-arm-if-conversion",cl::Hidden,
                              cl::desc("Disable if-conversion pass"));

namespace {
  // Register the target.
  RegisterTarget<ARMTargetMachine>   X("arm",   "  ARM");
  RegisterTarget<ThumbTargetMachine> Y("thumb", "  Thumb");
}

/// ThumbTargetMachine - Create an Thumb architecture model.
///
unsigned ThumbTargetMachine::getJITMatchQuality() {
#if defined(__thumb__)
  return 10;
#endif
  return 0;
}

unsigned ThumbTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "thumb-")
    return 20;

  // If the target triple is something non-thumb, we don't match.
  if (!TT.empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}

ThumbTargetMachine::ThumbTargetMachine(const Module &M, const std::string &FS) 
  : ARMTargetMachine(M, FS, true) {
}

/// TargetMachine ctor - Create an ARM architecture model.
///
ARMTargetMachine::ARMTargetMachine(const Module &M, const std::string &FS,
                                   bool isThumb)
  : Subtarget(M, FS, isThumb),
    DataLayout(Subtarget.isAPCS_ABI() ?
               // APCS ABI
          (isThumb ?
           std::string("e-p:32:32-f64:32:32-i64:32:32-"
                       "i16:16:32-i8:8:32-i1:8:32-a:0:32") :
           std::string("e-p:32:32-f64:32:32-i64:32:32")) :
               // AAPCS ABI
          (isThumb ?
           std::string("e-p:32:32-f64:64:64-i64:64:64-"
                       "i16:16:32-i8:8:32-i1:8:32-a:0:32") :
           std::string("e-p:32:32-f64:64:64-i64:64:64"))),
    InstrInfo(Subtarget),
    FrameInfo(Subtarget),
    JITInfo(*this),
    TLInfo(*this) {}

unsigned ARMTargetMachine::getJITMatchQuality() {
#if defined(__arm__)
  return 10;
#endif
  return 0;
}

unsigned ARMTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 4 && // Match arm-foo-bar, as well as things like armv5blah-*
      (TT.substr(0, 4) == "arm-" || TT.substr(0, 4) == "armv"))
    return 20;
  // If the target triple is something non-arm, we don't match.
  if (!TT.empty()) return 0;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer32)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return getJITMatchQuality()/2;
}


const TargetAsmInfo *ARMTargetMachine::createTargetAsmInfo() const {
  return new ARMTargetAsmInfo(*this);
}


// Pass Pipeline Configuration
bool ARMTargetMachine::addInstSelector(PassManagerBase &PM, bool Fast) {
  PM.add(createARMISelDag(*this));
  return false;
}

bool ARMTargetMachine::addPreEmitPass(PassManagerBase &PM, bool Fast) {
  // FIXME: temporarily disabling load / store optimization pass for Thumb mode.
  if (!Fast && !DisableLdStOpti && !Subtarget.isThumb())
    PM.add(createARMLoadStoreOptimizationPass());
  
  if (!Fast && !DisableIfConversion && !Subtarget.isThumb())
    PM.add(createIfConverterPass());

  PM.add(createARMConstantIslandPass());
  return true;
}

bool ARMTargetMachine::addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                                          std::ostream &Out) {
  // Output assembly language.
  PM.add(createARMCodePrinterPass(Out, *this));
  return false;
}


bool ARMTargetMachine::addCodeEmitter(PassManagerBase &PM, bool Fast,
                                      bool DumpAsm, MachineCodeEmitter &MCE) {
  // FIXME: Move this to TargetJITInfo!
  setRelocationModel(Reloc::Static);

  // Machine code emitter pass for ARM.
  PM.add(createARMCodeEmitterPass(*this, MCE));
  if (DumpAsm)
    PM.add(createARMCodePrinterPass(*cerr.stream(), *this));
  return false;
}

bool ARMTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM, bool Fast,
                                        bool DumpAsm, MachineCodeEmitter &MCE) {
  // Machine code emitter pass for ARM.
  PM.add(createARMCodeEmitterPass(*this, MCE));
  if (DumpAsm)
    PM.add(createARMCodePrinterPass(*cerr.stream(), *this));
  return false;
}

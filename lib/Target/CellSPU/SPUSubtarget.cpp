//===- SPUSubtarget.cpp - STI Cell SPU Subtarget Information --------------===//
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
// This file implements the CellSPU-specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "SPUSubtarget.h"
#include "SPU.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "SPUGenSubtarget.inc"

using namespace llvm;

SPUSubtarget::SPUSubtarget(const TargetMachine &tm, const Module &M,
                           const std::string &FS) :
  TM(tm),
  StackAlignment(16),
  ProcDirective(SPU::DEFAULT_PROC),
  UseLargeMem(false)
{
  // Should be the target SPU processor type. For now, since there's only
  // one, simply default to the current "v0" default:
  std::string default_cpu("v0");

  // Parse features string.
  ParseSubtargetFeatures(FS, default_cpu);
}

/// SetJITMode - This is called to inform the subtarget info that we are
/// producing code for the JIT.
void SPUSubtarget::SetJITMode() {
}

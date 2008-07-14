//===- MipsSubtarget.cpp - Mips Subtarget Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mips specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "MipsSubtarget.h"
#include "Mips.h"
#include "MipsGenSubtarget.inc"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

cl::opt<bool> NotABICall("disable-mips-abicall", cl::Hidden,
              cl::desc("Disable code for SVR4-style dynamic objects"));
cl::opt<bool> AbsoluteCall("enable-mips-absolute-call", cl::Hidden,
              cl::desc("Enable absolute call within abicall"));

MipsSubtarget::MipsSubtarget(const TargetMachine &TM, const Module &M, 
                             const std::string &FS, bool little) : 
  MipsArchVersion(Mips1), MipsABI(O32), IsLittle(little), IsSingleFloat(false),
  IsFP64bit(false), IsGP64bit(false), HasVFPU(false), HasSEInReg(false),
  HasABICall(true), HasAbsoluteCall(false), IsLinux(true)
{
  std::string CPU = "mips1";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
  const std::string& TT = M.getTargetTriple();

  // Is the target system Linux ?
  if (TT.find("linux") == std::string::npos)
    IsLinux = false;

  // When only the target triple is specified and is 
  // a allegrex target, set the features. We also match
  // big and little endian allegrex cores (dont really
  // know if a big one exists)
  if (TT.find("mipsallegrex") != std::string::npos) {
    MipsABI = EABI;
    IsSingleFloat = true;
    MipsArchVersion = Mips2;
    HasVFPU = true; // Enables Allegrex Vector FPU (not supported yet)
    HasSEInReg = true;
  }

  // Abicall is the default for O32 ABI and is ignored 
  // for EABI.
  if (NotABICall || isABI_EABI())
    HasABICall = false;

  // TODO: disable when handling 64 bit symbols in the future.
  if (HasABICall && AbsoluteCall)
    HasAbsoluteCall = true;
}

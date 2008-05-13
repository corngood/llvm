//===-- MipsTargetMachine.cpp - Define TargetMachine for Mips -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Mips target spec.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsTargetAsmInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

// Register the target.
static RegisterTarget<MipsTargetMachine> X("mips", "  Mips");

const TargetAsmInfo *MipsTargetMachine::
createTargetAsmInfo() const 
{
  return new MipsTargetAsmInfo(*this);
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, so StackGrowsUp is used.
// When using CodeModel::Large the behaviour 
//
// 
MipsTargetMachine::
MipsTargetMachine(const Module &M, const std::string &FS): 
  Subtarget(*this, M, FS), DataLayout("E-p:32:32:32"), 
  InstrInfo(*this), FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0),
  TLInfo(*this) 
{
  if (getRelocationModel() != Reloc::Static)
    setRelocationModel(Reloc::PIC_);  
  if (getCodeModel() == CodeModel::Default)
    setCodeModel(CodeModel::Small);
}

// return 0 and must specify -march to gen MIPS code.
unsigned MipsTargetMachine::
getModuleMatchQuality(const Module &M) 
{
  // We strongly match "mips-*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && std::string(TT.begin(), TT.begin()+5) == "mips-")
    return 20;
  
  return 0;
}

// Install an instruction selector pass using 
// the ISelDag to gen Mips code.
bool MipsTargetMachine::
addInstSelector(PassManagerBase &PM, bool Fast) 
{
  PM.add(createMipsISelDag(*this));
  return false;
}

// Implemented by targets that want to run passes immediately before 
// machine code is emitted. return true if -print-machineinstrs should 
// print out the code after the passes.
bool MipsTargetMachine::
addPreEmitPass(PassManagerBase &PM, bool Fast) 
{
  PM.add(createMipsDelaySlotFillerPass(*this));
  return true;
}

// Implements the AssemblyEmitter for the target. Must return
// true if AssemblyEmitter is supported
bool MipsTargetMachine::
addAssemblyEmitter(PassManagerBase &PM, bool Fast, 
                   std::ostream &Out) 
{
  // Output assembly language.
  PM.add(createMipsCodePrinterPass(Out, *this));
  return false;
}

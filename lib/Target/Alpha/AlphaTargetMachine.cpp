//===-- AlphaTargetMachine.cpp - Define TargetMachine for Alpha -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <iostream>

using namespace llvm;

namespace {
  // Register the targets
  RegisterTarget<AlphaTargetMachine> X("alpha", "  Alpha (incomplete)");
}

namespace llvm {
  cl::opt<bool> EnableAlphaLSR("enable-lsr-for-alpha",
                             cl::desc("Enable LSR for Alpha (beta option!)"),
                             cl::Hidden);
}

unsigned AlphaTargetMachine::getModuleMatchQuality(const Module &M) {
  // We strongly match "alpha*".
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 5 && TT[0] == 'a' && TT[1] == 'l' && TT[2] == 'p' &&
      TT[3] == 'h' && TT[4] == 'a')
    return 20;

  if (M.getEndianness()  == Module::LittleEndian &&
      M.getPointerSize() == Module::Pointer64)
    return 10;                                   // Weak match
  else if (M.getEndianness() != Module::AnyEndianness ||
           M.getPointerSize() != Module::AnyPointerSize)
    return 0;                                    // Match for some other target

  return 0;
}

AlphaTargetMachine::AlphaTargetMachine( const Module &M, IntrinsicLowering *IL)
  : TargetMachine("alpha", IL, true),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) //TODO: check these
{}

/// addPassesToEmitFile - Add passes to the specified pass manager to implement
/// a static compiler for this target.
///
bool AlphaTargetMachine::addPassesToEmitFile(PassManager &PM,
                                             std::ostream &Out,
                                             CodeGenFileType FileType) {
  if (FileType != TargetMachine::AssemblyFile) return true;

  if (EnableAlphaLSR) {
    PM.add(createLoopStrengthReducePass());
    PM.add(createCFGSimplificationPass());
  }

  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  PM.add(createAlphaPatternInstructionSelector(*this));

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createRegisterAllocator());

  if (PrintMachineCode)
    PM.add(createMachineFunctionPrinterPass(&std::cerr));

  PM.add(createPrologEpilogCodeInserter());

  // Must run branch selection immediately preceding the asm printer
  //PM.add(createAlphaBranchSelectionPass());

  PM.add(createAlphaCodePrinterPass(Out, *this));

  PM.add(createMachineCodeDeleter());
  return false;
}

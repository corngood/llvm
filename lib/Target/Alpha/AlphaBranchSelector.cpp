//===-- AlphaBranchSelector.cpp - Convert Pseudo branchs ----------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Replace Pseudo COND_BRANCH_* with their appropriate real branch
// Simplified version of the PPC Branch Selector
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN AlphaBSel : public MachineFunctionPass {

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "Alpha Branch Selection";
    }
  };
}

/// createAlphaBranchSelectionPass - returns an instance of the Branch Selection
/// Pass
///
FunctionPass *llvm::createAlphaBranchSelectionPass() {
  return new AlphaBSel();
}

bool AlphaBSel::runOnMachineFunction(MachineFunction &Fn) {

  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock *MBB = MFI;
    
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), EE = MBB->end();
         MBBI != EE; ++MBBI) {
      if (MBBI->getOpcode() == Alpha::COND_BRANCH_I ||
          MBBI->getOpcode() == Alpha::COND_BRANCH_F) {
        
        // condbranch operands:
        // 0. bc opcode
        // 1. reg
        // 2. target MBB
        MBBI->setOpcode(MBBI->getOperand(0).getImm());
      }
    }
  }
  
  return true;
}


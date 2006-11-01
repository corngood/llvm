//===-- TargetInstrInfo.cpp - Target Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;

namespace llvm {
  // External object describing the machine instructions Initialized only when
  // the TargetMachine class is created and reset when that class is destroyed.
  //
  // FIXME: UGLY SPARCV9 HACK!
  const TargetInstrDescriptor* TargetInstrDescriptors = 0;
}

TargetInstrInfo::TargetInstrInfo(const TargetInstrDescriptor* Desc,
                                 unsigned numOpcodes)
  : desc(Desc), NumOpcodes(numOpcodes) {
  // FIXME: TargetInstrDescriptors should not be global
  assert(TargetInstrDescriptors == NULL && desc != NULL
         && "TargetMachine data structure corrupt; maybe you tried to create another TargetMachine? (only one may exist in a program)");
  TargetInstrDescriptors = desc; // initialize global variable
}

TargetInstrInfo::~TargetInstrInfo() {
  TargetInstrDescriptors = NULL; // reset global variable
}

/// findTiedToSrcOperand - Returns the operand that is tied to the specified
/// dest operand. Returns -1 if there isn't one.
int
TargetInstrInfo::findTiedToSrcOperand(MachineOpCode Opc, unsigned OpNum) const {
  for (unsigned i = 0, e = getNumOperands(Opc); i != e; ++i) {
    if (i == OpNum)
      continue;
    int ti = getOperandConstraint(Opc, i, TIED_TO);
    if (ti == (int)OpNum)
      return i;
  }
  return -1;
}


// commuteInstruction - The default implementation of this method just exchanges
// operand 1 and 2.
MachineInstr *TargetInstrInfo::commuteInstruction(MachineInstr *MI) const {
  assert(MI->getOperand(1).isRegister() && MI->getOperand(2).isRegister() &&
         "This only knows how to commute register operands so far");
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  return MI;
}

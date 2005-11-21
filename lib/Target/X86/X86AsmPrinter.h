//===-- X86AsmPrinter.h - Convert X86 LLVM code to Intel assembly ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file the shared super class printer that converts from our internal
// representation of machine-dependent LLVM code to Intel and AT&T format
// assembly language.  This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#ifndef X86ASMPRINTER_H
#define X86ASMPRINTER_H

#include "X86.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/ADT/Statistic.h"
#include <set>


namespace llvm {
namespace x86 {

extern Statistic<> EmittedInsts;

struct X86SharedAsmPrinter : public AsmPrinter {
  X86SharedAsmPrinter(std::ostream &O, TargetMachine &TM)
    : AsmPrinter(O, TM), forCygwin(false), forDarwin(false) { }

  bool doInitialization(Module &M);
  void printConstantPool(MachineConstantPool *MCP);
  bool doFinalization(Module &M);

  bool forCygwin;
  bool forDarwin;

  // Necessary for Darwin to print out the apprioriate types of linker stubs
  std::set<std::string> FnStubs, GVStubs, LinkOnceStubs;

  inline static bool isScale(const MachineOperand &MO) {
    return MO.isImmediate() &&
          (MO.getImmedValue() == 1 || MO.getImmedValue() == 2 ||
          MO.getImmedValue() == 4 || MO.getImmedValue() == 8);
  }

  inline static bool isMem(const MachineInstr *MI, unsigned Op) {
    if (MI->getOperand(Op).isFrameIndex()) return true;
    if (MI->getOperand(Op).isConstantPoolIndex()) return true;
    return Op+4 <= MI->getNumOperands() &&
      MI->getOperand(Op  ).isRegister() && isScale(MI->getOperand(Op+1)) &&
      MI->getOperand(Op+2).isRegister() && (MI->getOperand(Op+3).isImmediate()||
      MI->getOperand(Op+3).isGlobalAddress());
  }
};

} // end namespace x86
} // end namespace llvm

#endif

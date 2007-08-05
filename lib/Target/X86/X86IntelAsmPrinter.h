//===-- X86IntelAsmPrinter.h - Convert X86 LLVM code to Intel assembly ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Intel assembly code printer class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INTELASMPRINTER_H
#define X86INTELASMPRINTER_H

#include "X86AsmPrinter.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Target/MRegisterInfo.h"

namespace llvm {

struct X86IntelAsmPrinter : public X86SharedAsmPrinter {
  X86IntelAsmPrinter(std::ostream &O, X86TargetMachine &TM,
                     const TargetAsmInfo *T)
      : X86SharedAsmPrinter(O, TM, T) {
  }

  virtual const char *getPassName() const {
    return "X86 Intel-Style Assembly Printer";
  }

  /// printInstruction - This method is automatically generated by tablegen
  /// from the instruction set description.  This method returns true if the
  /// machine instruction was sufficiently described to print it, otherwise it
  /// returns false.
  bool printInstruction(const MachineInstr *MI);

  // This method is used by the tablegen'erated instruction printer.
  void printOperand(const MachineInstr *MI, unsigned OpNo,
                    const char *Modifier = 0) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    if (MO.isRegister()) {
      assert(MRegisterInfo::isPhysicalRegister(MO.getReg()) && "Not physreg??");
      O << TM.getRegisterInfo()->get(MO.getReg()).Name;
    } else {
      printOp(MO, Modifier);
    }
  }

  void printi8mem(const MachineInstr *MI, unsigned OpNo) {
    O << "BYTE PTR ";
    printMemReference(MI, OpNo);
  }
  void printi16mem(const MachineInstr *MI, unsigned OpNo) {
    O << "WORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printi32mem(const MachineInstr *MI, unsigned OpNo) {
    O << "DWORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printi64mem(const MachineInstr *MI, unsigned OpNo) {
    O << "QWORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printi128mem(const MachineInstr *MI, unsigned OpNo) {
    O << "XMMWORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printf32mem(const MachineInstr *MI, unsigned OpNo) {
    O << "DWORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printf64mem(const MachineInstr *MI, unsigned OpNo) {
    O << "QWORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printf80mem(const MachineInstr *MI, unsigned OpNo) {
    O << "XWORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printf128mem(const MachineInstr *MI, unsigned OpNo) {
    O << "XMMWORD PTR ";
    printMemReference(MI, OpNo);
  }
  void printlea64_32mem(const MachineInstr *MI, unsigned OpNo) {
    O << "QWORD PTR ";
    printMemReference(MI, OpNo, "subreg64");
  }

  bool printAsmMRegister(const MachineOperand &MO, const char Mode);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode);
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             unsigned AsmVariant, const char *ExtraCode);
  void printMachineInstruction(const MachineInstr *MI);
  void printOp(const MachineOperand &MO, const char *Modifier = 0);
  void printSSECC(const MachineInstr *MI, unsigned Op);
  void printMemReference(const MachineInstr *MI, unsigned Op,
                         const char *Modifier=NULL);
  void printPICLabel(const MachineInstr *MI, unsigned Op);
  bool runOnMachineFunction(MachineFunction &F);
  bool doInitialization(Module &M);
  bool doFinalization(Module &M);
  
  /// getSectionForFunction - Return the section that we should emit the
  /// specified function body into.
  virtual std::string getSectionForFunction(const Function &F) const;

  virtual void EmitString(const ConstantArray *CVA) const;
};

} // end namespace llvm

#endif

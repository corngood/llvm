//===--- lib/CodeGen/DwarfPrinter.h - Dwarf Printer -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Emit general DWARF directives.
// 
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFPRINTER_H__
#define CODEGEN_ASMPRINTER_DWARFPRINTER_H__

#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormattedStream.h"
#include <vector>

namespace llvm {
class AsmPrinter;
class MachineFunction;
class MachineModuleInfo;
class Module;
class MCAsmInfo;
class TargetData;
class TargetRegisterInfo;
class GlobalValue;
class MCSymbol;
class Twine;

class DwarfPrinter {
protected:
  ~DwarfPrinter() {}

  //===-------------------------------------------------------------==---===//
  // Core attributes used by the DWARF printer.
  //

  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  /// MAI - Target asm information.
  const MCAsmInfo *MAI;

  /// TD - Target data.
  const TargetData *TD;

  /// RI - Register Information.
  const TargetRegisterInfo *RI;

  /// M - Current module.
  Module *M;

  /// MF - Current machine function.
  const MachineFunction *MF;

  /// MMI - Collected machine module information.
  MachineModuleInfo *MMI;

  /// SubprogramCount - The running count of functions being compiled.
  unsigned SubprogramCount;

  DwarfPrinter(AsmPrinter *A);
public:
  
  //===------------------------------------------------------------------===//
  // Accessors.
  //
  const AsmPrinter *getAsm() const { return Asm; }
  MachineModuleInfo *getMMI() const { return MMI; }
  const MCAsmInfo *getMCAsmInfo() const { return MAI; }
  const TargetData *getTargetData() const { return TD; }

  /// EmitSectionOffset - Emit a 4-byte "Label-Section" value or use a special
  /// purpose directive to emit a section offset if the target has one.
  void EmitSectionOffset(const MCSymbol *Label, const MCSymbol *Section);
  
  /// EmitFrameMoves - Emit frame instructions to describe the layout of the
  /// frame.
  void EmitFrameMoves(MCSymbol *BaseLabel,
                      const std::vector<MachineMove> &Moves, bool isEH);
};

} // end llvm namespace

#endif

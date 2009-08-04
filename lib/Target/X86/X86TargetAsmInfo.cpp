//===-- X86TargetAsmInfo.cpp - X86 asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "X86TargetAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86Subtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::dwarf;

const char *const llvm::x86_asm_table[] = {
  "{si}", "S",
  "{di}", "D",
  "{ax}", "a",
  "{cx}", "c",
  "{memory}", "memory",
  "{flags}", "",
  "{dirflag}", "",
  "{fpsr}", "",
  "{cc}", "cc",
  0,0};

X86DarwinTargetAsmInfo::X86DarwinTargetAsmInfo(const X86TargetMachine &TM):
  X86TargetAsmInfo<DarwinTargetAsmInfo>(TM) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  bool is64Bit = Subtarget->is64Bit();

  AlignmentIsInBytes = false;
  TextAlignFillValue = 0x90;


  if (!is64Bit)
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit
  ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
  ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
  LCOMMDirective = "\t.lcomm\t";

  // Leopard and above support aligned common symbols.
  COMMDirectiveTakesAlignment = (Subtarget->getDarwinVers() >= 9);
  HasDotTypeDotSizeDirective = false;

  if (is64Bit) {
    PersonalityPrefix = "";
    PersonalitySuffix = "+4@GOTPCREL";
  } else {
    PersonalityPrefix = "L";
    PersonalitySuffix = "$non_lazy_ptr";
  }

  InlineAsmStart = "## InlineAsm Start";
  InlineAsmEnd = "## InlineAsm End";
  CommentString = "##";
  SetDirective = "\t.set";
  PCSymbol = ".";
  UsedDirective = "\t.no_dead_strip\t";
  ProtectedDirective = "\t.globl\t";

  SupportsDebugInformation = true;
  DwarfUsesInlineInfoSection = true;

  // Exceptions handling
  SupportsExceptionHandling = true;
  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
  AbsoluteEHSectionOffsets = false;
}

X86ELFTargetAsmInfo::X86ELFTargetAsmInfo(const X86TargetMachine &TM) :
  X86TargetAsmInfo<TargetAsmInfo>(TM) {

  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  // Debug Information
  AbsoluteDebugSectionOffsets = true;
  SupportsDebugInformation = true;

  // Exceptions handling
  SupportsExceptionHandling = true;
  AbsoluteEHSectionOffsets = false;

  // On Linux we must declare when we can use a non-executable stack.
  if (TM.getSubtarget<X86Subtarget>().isLinux())
    NonexecutableStackDirective = "\t.section\t.note.GNU-stack,\"\",@progbits";
}


X86WinTargetAsmInfo::X86WinTargetAsmInfo(const X86TargetMachine &TM):
  X86TargetAsmInfo<TargetAsmInfo>(TM) {
  GlobalPrefix = "_";
  CommentString = ";";

  InlineAsmStart = "; InlineAsm Start";
  InlineAsmEnd   = "; InlineAsm End";

  PrivateGlobalPrefix = "$";
  AlignDirective = "\tALIGN\t";
  ZeroDirective = "\tdb\t";
  ZeroDirectiveSuffix = " dup(0)";
  AsciiDirective = "\tdb\t";
  AscizDirective = 0;
  Data8bitsDirective = "\tdb\t";
  Data16bitsDirective = "\tdw\t";
  Data32bitsDirective = "\tdd\t";
  Data64bitsDirective = "\tdq\t";
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;

  AlignmentIsInBytes = true;

  SwitchToSectionDirective = "";
  TextSectionStartSuffix = "\tSEGMENT PARA 'CODE'";
  DataSectionStartSuffix = "\tSEGMENT PARA 'DATA'";
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class X86TargetAsmInfo<TargetAsmInfo>);

//===-- SPUTargetAsmInfo.cpp - Cell SPU asm properties ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

SPULinuxTargetAsmInfo::SPULinuxTargetAsmInfo(const SPUTargetMachine &TM) :
    SPUTargetAsmInfo<ELFTargetAsmInfo>(TM) {
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";

  // Has leb128, .loc and .file
  HasLEB128 = true;
  HasDotLocAndDotFile = true;

  SupportsDebugInformation = true;
  NeedsSet = true;
  DwarfAbbrevSection =  "\t.section        .debug_abbrev,\"\",@progbits";
  DwarfInfoSection =    "\t.section        .debug_info,\"\",@progbits";
  DwarfLineSection =    "\t.section        .debug_line,\"\",@progbits";
  DwarfFrameSection =   "\t.section        .debug_frame,\"\",@progbits";
  DwarfPubNamesSection = "\t.section        .debug_pubnames,\"\",@progbits";
  DwarfPubTypesSection = "\t.section        .debug_pubtypes,\"\",progbits";
  DwarfStrSection =     "\t.section        .debug_str,\"MS\",@progbits,1";
  DwarfLocSection =     "\t.section        .debug_loc,\"\",@progbits";
  DwarfARangesSection = "\t.section        .debug_aranges,\"\",@progbits";
  DwarfRangesSection =  "\t.section        .debug_ranges,\"\",@progbits";
  DwarfMacroInfoSection = 0;  // macro info not supported.

  // Exception handling is not supported on CellSPU (think about it: you only
  // have 256K for code+data. Would you support exception handling?)
  SupportsExceptionHandling = false;
}


// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class SPUTargetAsmInfo<TargetAsmInfo>);

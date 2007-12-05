//===-- SPUTargetAsmInfo.cpp - Cell SPU asm properties ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by a team from the Computer Systems Research
// Department at The Aerospace Corporation.
//
// See README.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Function.h"
using namespace llvm;

SPUTargetAsmInfo::SPUTargetAsmInfo(const SPUTargetMachine &TM) {
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  Data64bitsDirective = "\t.quad\t";  
  AlignmentIsInBytes = false;
  SwitchToSectionDirective = "\t.section\t";
  ConstantPoolSection = "\t.const\t";
  JumpTableDataSection = ".const";
  CStringSection = "\t.cstring";
  LCOMMDirective = "\t.lcomm\t";
  StaticCtorsSection = ".mod_init_func";
  StaticDtorsSection = ".mod_term_func";
  FourByteConstantSection = ".const";
  SixteenByteConstantSection = "\t.section\t.rodata.cst16,\"aM\",@progbits,16";
  UsedDirective = "\t.no_dead_strip\t";
  WeakRefDirective = "\t.weak_reference\t";
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  
  NeedsSet = true;
  /* FIXME: Need actual assembler syntax for DWARF info: */
  DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
  DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
  DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
  DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
  DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
  DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
  DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
  DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
  DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
  DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
  DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
}

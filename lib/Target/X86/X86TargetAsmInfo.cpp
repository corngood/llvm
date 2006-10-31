//===-- X86TargetAsmInfo.cpp - X86 asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "X86TargetAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86Subtarget.h"

using namespace llvm;

X86TargetAsmInfo::X86TargetAsmInfo(const X86TargetMachine &TM) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  
  // FIXME - Should be simplified.
   
  switch (Subtarget->TargetType) {
  case X86Subtarget::isDarwin:
    AlignmentIsInBytes = false;
    GlobalPrefix = "_";
    if (!Subtarget->is64Bit())
      Data64bitsDirective = 0;       // we can't emit a 64-bit unit
    ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
    PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
    ConstantPoolSection = "\t.const\n";
    JumpTableDataSection = "\t.const\n";
    CStringSection = "\t.cstring";
    FourByteConstantSection = "\t.literal4\n";
    EightByteConstantSection = "\t.literal8\n";
    if (Subtarget->is64Bit())
      SixteenByteConstantSection = "\t.literal16\n";
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
    InlineAsmStart = "# InlineAsm Start";
    InlineAsmEnd = "# InlineAsm End";
    SetDirective = "\t.set";
    UsedDirective = "\t.no_dead_strip\t";
    
    NeedsSet = true;
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
    break;

  case X86Subtarget::isELF:
    // Set up DWARF directives
    HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
    // bool HasLEB128; // Defaults to false.
    // hasDotLoc - True if target asm supports .loc directives.
    // bool HasDotLoc; // Defaults to false.
    // HasDotFile - True if target asm supports .file directives.
    // bool HasDotFile; // Defaults to false.
    PrivateGlobalPrefix = ".";  // Prefix for private global symbols
    DwarfRequiresFrameSection = false;
    DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",@progbits";
    DwarfInfoSection =    "\t.section\t.debug_info,\"\",@progbits";
    DwarfLineSection =    "\t.section\t.debug_line,\"\",@progbits";
    DwarfFrameSection =   "\t.section\t.debug_frame,\"\",@progbits";
    DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",@progbits";
    DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",@progbits";
    DwarfStrSection =     "\t.section\t.debug_str,\"\",@progbits";
    DwarfLocSection =     "\t.section\t.debug_loc,\"\",@progbits";
    DwarfARangesSection = "\t.section\t.debug_aranges,\"\",@progbits";
    DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",@progbits";
    DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";
    break;

  case X86Subtarget::isCygwin:
    GlobalPrefix = "_";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    StaticCtorsSection = "\t.section .ctors,\"aw\"";
    StaticDtorsSection = "\t.section .dtors,\"aw\"";

    // Set up DWARF directives
    HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
    PrivateGlobalPrefix = "L";  // Prefix for private global symbols
    DwarfRequiresFrameSection = false;
    DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"dr\"";
    DwarfInfoSection =    "\t.section\t.debug_info,\"dr\"";
    DwarfLineSection =    "\t.section\t.debug_line,\"dr\"";
    DwarfFrameSection =   "\t.section\t.debug_frame,\"dr\"";
    DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"dr\"";
    DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"dr\"";
    DwarfStrSection =     "\t.section\t.debug_str,\"dr\"";
    DwarfLocSection =     "\t.section\t.debug_loc,\"dr\"";
    DwarfARangesSection = "\t.section\t.debug_aranges,\"dr\"";
    DwarfRangesSection =  "\t.section\t.debug_ranges,\"dr\"";
    DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"dr\"";
    break;
    
    break;
  case X86Subtarget::isWindows:
    GlobalPrefix = "_";
    HasDotTypeDotSizeDirective = false;
    break;
  default: break;
  }
  
  if (Subtarget->isFlavorIntel()) {
    GlobalPrefix = "_";
    CommentString = ";";
  
    PrivateGlobalPrefix = "$";
    AlignDirective = "\talign\t";
    ZeroDirective = "\tdb\t";
    ZeroDirectiveSuffix = " dup(0)";
    AsciiDirective = "\tdb\t";
    AscizDirective = 0;
    Data8bitsDirective = "\tdb\t";
    Data16bitsDirective = "\tdw\t";
    Data32bitsDirective = "\tdd\t";
    Data64bitsDirective = "\tdq\t";
    HasDotTypeDotSizeDirective = false;
    
    TextSection = "_text";
    DataSection = "_data";
    SwitchToSectionDirective = "";
    TextSectionStartSuffix = "\tsegment 'CODE'";
    DataSectionStartSuffix = "\tsegment 'DATA'";
    SectionEndDirectiveSuffix = "\tends\n";
  }
}


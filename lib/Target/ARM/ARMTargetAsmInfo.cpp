//===-- ARMTargetAsmInfo.cpp - ARM asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetAsmInfo.h"
#include "ARMTargetMachine.h"
using namespace llvm;

ARMTargetAsmInfo::ARMTargetAsmInfo(const ARMTargetMachine &TM) {
  const ARMSubtarget *Subtarget = &TM.getSubtarget<ARMSubtarget>();
  if (Subtarget->isDarwin()) {
    HasDotTypeDotSizeDirective = false;
    PrivateGlobalPrefix = "L";
    GlobalPrefix = "_";
    ZeroDirective = "\t.space\t";
    SetDirective = "\t.set";
    WeakRefDirective = "\t.weak_reference\t";
    JumpTableDataSection = ".const";
    CStringSection = "\t.cstring";
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
    InlineAsmStart = "@ InlineAsm Start";
    InlineAsmEnd = "@ InlineAsm End";
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    
    // In non-PIC modes, emit a special label before jump tables so that the
    // linker can perform more accurate dead code stripping.
    if (TM.getRelocationModel() != Reloc::PIC_) {
      // Emit a local label that is preserved until the linker runs.
      JumpTableSpecialLabelPrefix = "l";
    }
    
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
  } else {
    Data16bitsDirective = "\t.half\t";
    Data32bitsDirective = "\t.word\t";
    ZeroDirective = "\t.skip\t";
    WeakRefDirective = "\t.weak\t";
    StaticCtorsSection = "\t.section .ctors,\"aw\",%progbits";
    StaticDtorsSection = "\t.section .dtors,\"aw\",%progbits";
  }
  AlignmentIsInBytes = false; 
  Data64bitsDirective = 0;
  CommentString = "@";
  DataSection = "\t.data";
  ConstantPoolSection = "\t.text\n";
}

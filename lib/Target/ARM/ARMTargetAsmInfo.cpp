//===-- ARMTargetAsmInfo.cpp - ARM asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetAsmInfo.h"
#include "ARMTargetMachine.h"
#include <cstring>
#include <cctype>
using namespace llvm;

const char *const llvm::arm_asm_table[] = {
  "{r0}", "r0",
  "{r1}", "r1",
  "{r2}", "r2",
  "{r3}", "r3",
  "{r4}", "r4",
  "{r5}", "r5",
  "{r6}", "r6",
  "{r7}", "r7",
  "{r8}", "r8",
  "{r9}", "r9",
  "{r10}", "r10",
  "{r11}", "r11",
  "{r12}", "r12",
  "{r13}", "r13",
  "{r14}", "r14",
  "{lr}", "lr",
  "{sp}", "sp",
  "{ip}", "ip",
  "{fp}", "fp",
  "{sl}", "sl",
  "{memory}", "memory",
  "{cc}", "cc",
  0,0
};

ARMDarwinTargetAsmInfo::ARMDarwinTargetAsmInfo(const ARMBaseTargetMachine &TM):
  ARMTargetAsmInfo<DarwinTargetAsmInfo>(TM) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();

  ZeroDirective = "\t.space\t";
  ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
  SetDirective = "\t.set\t";
  ProtectedDirective = NULL;
  HasDotTypeDotSizeDirective = false;
  SupportsDebugInformation = true;
}

ARMELFTargetAsmInfo::ARMELFTargetAsmInfo(const ARMBaseTargetMachine &TM):
  ARMTargetAsmInfo<TargetAsmInfo>(TM) {
  Subtarget = &TM.getSubtarget<ARMSubtarget>();

  NeedsSet = false;
  HasLEB128 = true;
  AbsoluteDebugSectionOffsets = true;
  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  DwarfRequiresFrameSection = false;
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",%progbits";
  DwarfInfoSection =    "\t.section\t.debug_info,\"\",%progbits";
  DwarfLineSection =    "\t.section\t.debug_line,\"\",%progbits";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"\",%progbits";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",%progbits";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",%progbits";
  DwarfStrSection =     "\t.section\t.debug_str,\"\",%progbits";
  DwarfLocSection =     "\t.section\t.debug_loc,\"\",%progbits";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"\",%progbits";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",%progbits";
  DwarfMacroInfoSection = "\t.section\t.debug_macinfo,\"\",%progbits";

  SupportsDebugInformation = true;
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class ARMTargetAsmInfo<TargetAsmInfo>);

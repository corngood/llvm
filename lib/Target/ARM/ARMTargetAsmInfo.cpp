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

ARMDarwinTargetAsmInfo::ARMDarwinTargetAsmInfo() {
  ZeroDirective = "\t.space\t";
  ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
  SetDirective = "\t.set\t";
  ProtectedDirective = NULL;
  HasDotTypeDotSizeDirective = false;
  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::SjLj;
  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
  AbsoluteEHSectionOffsets = false;
}

ARMELFTargetAsmInfo::ARMELFTargetAsmInfo() {
  NeedsSet = false;
  HasLEB128 = true;
  AbsoluteDebugSectionOffsets = true;
  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  DwarfRequiresFrameSection = false;

  SupportsDebugInformation = true;
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class ARMTargetAsmInfo<DarwinTargetAsmInfo>);
TEMPLATE_INSTANTIATION(class ARMTargetAsmInfo<TargetAsmInfo>);

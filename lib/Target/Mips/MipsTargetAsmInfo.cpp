//===-- MipsTargetAsmInfo.cpp - Mips asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MipsTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetAsmInfo.h"
#include "MipsTargetMachine.h"
using namespace llvm;

MipsTargetAsmInfo::MipsTargetAsmInfo(const MipsTargetMachine &TM) {
  AlignmentIsInBytes          = false;
  COMMDirectiveTakesAlignment = true;
  Data16bitsDirective         = "\t.half\t";
  Data32bitsDirective         = "\t.word\t";
  Data64bitsDirective         = NULL;
  PrivateGlobalPrefix         = "$";
  CommentString               = "#";
  ZeroDirective               = "\t.space\t";
  PICJumpTableDirective       = "\t.gpword\t";
}

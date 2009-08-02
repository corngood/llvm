//===-- BlackfinTargetAsmInfo.cpp - Blackfin asm properties -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the BlackfinTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "BlackfinTargetAsmInfo.h"

using namespace llvm;

BlackfinTargetAsmInfo::BlackfinTargetAsmInfo() {
  GlobalPrefix = "_";
  CommentString = "//";
  InlineAsmStart = "// APP";
  InlineAsmEnd = "// NO_APP";
}

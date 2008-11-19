//===-- PIC16TargetAsmInfo.cpp - PIC16 asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the PIC16TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PIC16TargetAsmInfo.h"
#include "PIC16TargetMachine.h"

using namespace llvm;

PIC16TargetAsmInfo::
PIC16TargetAsmInfo(const PIC16TargetMachine &TM) 
  : TargetAsmInfo(TM) {
  CommentString = ";";
  Data8bitsDirective = " db ";
  Data16bitsDirective = " db ";
  Data32bitsDirective = " db ";
  DataSectionStartSuffix = " IDATA ";
  UDataSectionStartSuffix = " UDATA ";
  TextSectionStartSuffix = " CODE ";
  RomDataSectionStartSuffix = " ROMDATA ";
  ZeroDirective = NULL;
}

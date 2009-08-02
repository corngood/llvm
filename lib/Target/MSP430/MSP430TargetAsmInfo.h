//=====-- MSP430TargetAsmInfo.h - MSP430 asm properties -------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MSP430TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MSP430TARGETASMINFO_H
#define MSP430TARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class TargetMachine;
  struct MSP430TargetAsmInfo : public TargetAsmInfo {
    explicit MSP430TargetAsmInfo(const TargetMachine &TM);
  };

} // namespace llvm

#endif

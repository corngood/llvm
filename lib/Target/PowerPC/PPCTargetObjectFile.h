//===-- llvm/Target/PPCTargetObjectFile.h - PowerPC Object Info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_PPC_TARGETOBJECTFILE_H
#define LLVM_TARGET_PPC_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

class MCContext;
class TargetMachine;

// FIXME: This subclass isn't 100% necessary. It will become obsolete once we
//        can place all LSDAs into the TEXT section. See
//        <rdar://problem/6804645>.
class PPCMachOTargetObjectFile : public TargetLoweringObjectFileMachO {
public:
  PPCMachOTargetObjectFile() : TargetLoweringObjectFileMachO() {}

  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

  virtual unsigned getTTypeEncoding() const;
};

} // end namespace llvm

#endif

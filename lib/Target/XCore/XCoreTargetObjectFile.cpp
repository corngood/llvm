//===-- XCoreTargetObjectFile.cpp - XCore object files --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetObjectFile.h"
#include "XCoreSubtarget.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;


void XCoreTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);

  TextSection = getOrCreateSection("\t.text", true, SectionKind::Text);
  DataSection = getOrCreateSection("\t.dp.data", false, SectionKind::DataRel);
  BSSSection_ = getOrCreateSection("\t.dp.bss", false, SectionKind::BSS);
  
  // TLS globals are lowered in the backend to arrays indexed by the current
  // thread id. After lowering they require no special handling by the linker
  // and can be placed in the standard data / bss sections.
  TLSDataSection = DataSection;
  TLSBSSSection = BSSSection_;
  
  if (TM.getSubtarget<XCoreSubtarget>().isXS1A())
    // FIXME: Why is this writable ("datarel")???
    ReadOnlySection = getOrCreateSection("\t.dp.rodata", false,
                                         SectionKind::DataRel);
  else
    ReadOnlySection = getOrCreateSection("\t.cp.rodata", false,
                                         SectionKind::ReadOnly);
}
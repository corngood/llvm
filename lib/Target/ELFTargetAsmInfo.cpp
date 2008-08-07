//===-- ELFTargetAsmInfo.cpp - ELF asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on ELF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

ELFTargetAsmInfo::ELFTargetAsmInfo(const TargetMachine &TM) {
  ETM = &TM;

  TextSection_ = getUnnamedSection("\t.text", SectionFlags::Code);
  DataSection_ = getUnnamedSection("\t.data", SectionFlags::Writeable);
  BSSSection_  = getUnnamedSection("\t.bss",
                                   SectionFlags::Writeable | SectionFlags::BSS);
  ReadOnlySection_ = getNamedSection("\t.rodata", SectionFlags::None);
  TLSDataSection_ = getNamedSection("\t.tdata",
                                    SectionFlags::Writeable | SectionFlags::TLS);
  TLSBSSSection_ = getNamedSection("\t.tbss",
                SectionFlags::Writeable | SectionFlags::TLS | SectionFlags::BSS);

}

const Section*
ELFTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);

  if (const Function *F = dyn_cast<Function>(GV)) {
    switch (F->getLinkage()) {
     default: assert(0 && "Unknown linkage type!");
     case Function::InternalLinkage:
     case Function::DLLExportLinkage:
     case Function::ExternalLinkage:
      return getTextSection_();
     case Function::WeakLinkage:
     case Function::LinkOnceLinkage:
      std::string Name = UniqueSectionForGlobal(GV, Kind);
      unsigned Flags = SectionFlagsForGlobal(GV, Name.c_str());
      return getNamedSection(Name.c_str(), Flags);
    }
  } else if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
    if (GVar->isWeakForLinker()) {
      std::string Name = UniqueSectionForGlobal(GVar, Kind);
      unsigned Flags = SectionFlagsForGlobal(GVar, Name.c_str());
      return getNamedSection(Name.c_str(), Flags);
    } else {
      switch (Kind) {
       case SectionKind::Data:
       case SectionKind::SmallData:
        return getDataSection_();
       case SectionKind::BSS:
       case SectionKind::SmallBSS:
        // ELF targets usually have BSS sections
        return getBSSSection_();
       case SectionKind::ROData:
       case SectionKind::SmallROData:
        return getReadOnlySection_();
       case SectionKind::RODataMergeStr:
        return MergeableStringSection(GVar);
       case SectionKind::RODataMergeConst:
        return MergeableConstSection(GVar);
       case SectionKind::ThreadData:
        // ELF targets usually support TLS stuff
        return getTLSDataSection_();
       case SectionKind::ThreadBSS:
        return getTLSBSSSection_();
       default:
        assert(0 && "Unsuported section kind for global");
      }
    }
  } else
    assert(0 && "Unsupported global");
}

const Section*
ELFTargetAsmInfo::SelectSectionForMachineConst(const Type *Ty) const {
  // FIXME: Support data.rel stuff someday
  return MergeableConstSection(Ty);
}

const Section*
ELFTargetAsmInfo::MergeableConstSection(const GlobalVariable *GV) const {
  Constant *C = cast<GlobalVariable>(GV)->getInitializer();
  const Type *Ty = C->getType();

  return MergeableConstSection(Ty);
}

inline const Section*
ELFTargetAsmInfo::MergeableConstSection(const Type *Ty) const {
  const TargetData *TD = ETM->getTargetData();

  // FIXME: string here is temporary, until stuff will fully land in.
  // We cannot use {Four,Eight,Sixteen}ByteConstantSection here, since it's
  // currently directly used by asmprinter.
  unsigned Size = TD->getABITypeSize(Ty);
  if (Size == 4 || Size == 8 || Size == 16) {
    std::string Name =  ".rodata.cst" + utostr(Size);

    return getNamedSection(Name.c_str(),
                           SectionFlags::setEntitySize(SectionFlags::Mergeable,
                                                       Size));
  }

  return getReadOnlySection_();
}

const Section*
ELFTargetAsmInfo::MergeableStringSection(const GlobalVariable *GV) const {
  const TargetData *TD = ETM->getTargetData();
  Constant *C = cast<GlobalVariable>(GV)->getInitializer();
  const ConstantArray *CVA = cast<ConstantArray>(C);
  const Type *Ty = CVA->getType()->getElementType();

  unsigned Size = TD->getABITypeSize(Ty);
  if (Size <= 16) {
    // We also need alignment here
    const TargetData *TD = ETM->getTargetData();
    unsigned Align = TD->getPrefTypeAlignment(Ty);
    if (Align < Size)
      Align = Size;

    std::string Name = getCStringSection() + utostr(Size) + '.' + utostr(Align);
    unsigned Flags = SectionFlags::setEntitySize(SectionFlags::Mergeable |
                                                 SectionFlags::Strings,
                                                 Size);
    return getNamedSection(Name.c_str(), Flags);
  }

  return getReadOnlySection_();
}

std::string ELFTargetAsmInfo::PrintSectionFlags(unsigned flags) const {
  std::string Flags = ",\"";

  if (!(flags & SectionFlags::Debug))
    Flags += 'a';
  if (flags & SectionFlags::Code)
    Flags += 'x';
  if (flags & SectionFlags::Writeable)
    Flags += 'w';
  if (flags & SectionFlags::Mergeable)
    Flags += 'M';
  if (flags & SectionFlags::Strings)
    Flags += 'S';
  if (flags & SectionFlags::TLS)
    Flags += 'T';
  if (flags & SectionFlags::Small)
    Flags += 's';

  Flags += "\"";

  // FIXME: There can be exceptions here
  if (flags & SectionFlags::BSS)
    Flags += ",@nobits";
  else
    Flags += ",@progbits";

  if (unsigned entitySize = SectionFlags::getEntitySize(flags))
    Flags += "," + utostr(entitySize);

  return Flags;
}


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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

ELFTargetAsmInfo::ELFTargetAsmInfo(const TargetMachine &TM)
  : TargetAsmInfo(TM) {

  BSSSection_  = getUnnamedSection("\t.bss",
                                   SectionFlags::Writeable | SectionFlags::BSS);
  ReadOnlySection = getNamedSection("\t.rodata", SectionFlags::None);
  TLSDataSection = getNamedSection("\t.tdata",
                                   SectionFlags::Writeable | SectionFlags::TLS);
  TLSBSSSection = getNamedSection("\t.tbss",
                SectionFlags::Writeable | SectionFlags::TLS | SectionFlags::BSS);

  DataRelSection = getNamedSection("\t.data.rel", SectionFlags::Writeable);
  DataRelLocalSection = getNamedSection("\t.data.rel.local",
                                        SectionFlags::Writeable);
  DataRelROSection = getNamedSection("\t.data.rel.ro",
                                     SectionFlags::Writeable);
  DataRelROLocalSection = getNamedSection("\t.data.rel.ro.local",
                                          SectionFlags::Writeable);
}

SectionKind::Kind
ELFTargetAsmInfo::SectionKindForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = TargetAsmInfo::SectionKindForGlobal(GV);

  if (Kind != SectionKind::Data)
    return Kind;

  // Decide, whether we need data.rel stuff
  const GlobalVariable* GVar = dyn_cast<GlobalVariable>(GV);
  if (GVar->hasInitializer() && TM.getRelocationModel() != Reloc::Static) {
    Constant *C = GVar->getInitializer();
    bool isConstant = GVar->isConstant();
    
    // By default - all relocations in PIC mode would force symbol to be
    // placed in r/w section.
    switch (C->getRelocationInfo()) {
    default: break;
    case 1:
      return isConstant ? SectionKind::DataRelROLocal :
                          SectionKind::DataRelLocal;
    case 2:
      return isConstant ? SectionKind::DataRelRO : SectionKind::DataRel;
    }
  }

  return Kind;
}

const Section*
ELFTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);

  if (const Function *F = dyn_cast<Function>(GV)) {
    switch (F->getLinkage()) {
     default: llvm_unreachable("Unknown linkage type!");
     case Function::PrivateLinkage:
     case Function::LinkerPrivateLinkage:
     case Function::InternalLinkage:
     case Function::DLLExportLinkage:
     case Function::ExternalLinkage:
      return TextSection;
     case Function::WeakAnyLinkage:
     case Function::WeakODRLinkage:
     case Function::LinkOnceAnyLinkage:
     case Function::LinkOnceODRLinkage:
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
      case SectionKind::DataRel:
        return DataRelSection;
      case SectionKind::DataRelLocal:
        return DataRelLocalSection;
      case SectionKind::DataRelRO:
        return DataRelROSection;
      case SectionKind::DataRelROLocal:
        return DataRelROLocalSection;
      case SectionKind::BSS:
        return getBSSSection_();
      case SectionKind::ROData:
        return getReadOnlySection();
      case SectionKind::RODataMergeStr:
        return MergeableStringSection(GVar);
      case SectionKind::RODataMergeConst: {
        const Type *Ty = GVar->getInitializer()->getType();
        const TargetData *TD = TM.getTargetData();
        return getSectionForMergableConstant(TD->getTypeAllocSize(Ty), 0);
      }
      case SectionKind::ThreadData:
        // ELF targets usually support TLS stuff
        return TLSDataSection;
      case SectionKind::ThreadBSS:
        return TLSBSSSection;
      default:
        llvm_unreachable("Unsuported section kind for global");
      }
    }
  } else
    llvm_unreachable("Unsupported global");

  return NULL;
}

/// getSectionForMergableConstant - Given a mergable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const Section *
ELFTargetAsmInfo::getSectionForMergableConstant(uint64_t Size,
                                                unsigned ReloInfo) const {
  // FIXME: IF this global requires a relocation, can we really put it in
  // rodata???  This should check ReloInfo like darwin.
  
  const char *SecName = 0;
  switch (Size) {
  default: break;
  case 4:  SecName = ".rodata.cst4"; break;
  case 8:  SecName = ".rodata.cst8"; break;
  case 16: SecName = ".rodata.cst16"; break;
  }
  
  if (SecName)
    return getNamedSection(SecName,
                           SectionFlags::setEntitySize(SectionFlags::Mergeable,
                                                       Size));
  
  return getReadOnlySection();  // .rodata
}


const Section*
ELFTargetAsmInfo::MergeableStringSection(const GlobalVariable *GV) const {
  const TargetData *TD = TM.getTargetData();
  Constant *C = cast<GlobalVariable>(GV)->getInitializer();
  const Type *Ty = cast<ArrayType>(C->getType())->getElementType();

  unsigned Size = TD->getTypeAllocSize(Ty);
  if (Size <= 16) {
    assert(getCStringSection() && "Should have string section prefix");

    // We also need alignment here
    unsigned Align = TD->getPrefTypeAlignment(Ty);
    if (Align < Size)
      Align = Size;

    std::string Name = getCStringSection() + utostr(Size) + '.' + utostr(Align);
    unsigned Flags = SectionFlags::setEntitySize(SectionFlags::Mergeable |
                                                 SectionFlags::Strings,
                                                 Size);
    return getNamedSection(Name.c_str(), Flags);
  }

  return getReadOnlySection();
}

std::string ELFTargetAsmInfo::printSectionFlags(unsigned flags) const {
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

  Flags += "\",";

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (strcmp(CommentString, "@") == 0)
    Flags += '%';
  else
    Flags += '@';

  // FIXME: There can be exceptions here
  if (flags & SectionFlags::BSS)
    Flags += "nobits";
  else
    Flags += "progbits";

  if (unsigned entitySize = SectionFlags::getEntitySize(flags))
    Flags += "," + utostr(entitySize);

  return Flags;
}

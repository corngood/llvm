//===-- DarwinTargetAsmInfo.cpp - Darwin asm properties ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

DarwinTargetAsmInfo::DarwinTargetAsmInfo(const TargetMachine &TM) {
  DTM = &TM;

  CStringSection_ = getUnnamedSection("\t.cstring",
                                SectionFlags::Mergeable | SectionFlags::Strings);
  FourByteConstantSection_ = getUnnamedSection("\t.literal4\n",
                                               SectionFlags::Mergeable);
  EightByteConstantSection_ = getUnnamedSection("\t.literal8\n",
                                                SectionFlags::Mergeable);

  // Note: 16-byte constant section is subtarget specific and should be provided
  // there.

  ReadOnlySection_ = getUnnamedSection("\t.const\n", SectionFlags::None);

  // FIXME: These should be named sections, really.
  TextCoalSection =
  getUnnamedSection(".section __TEXT,__textcoal_nt,coalesced,pure_instructions",
                    SectionFlags::Code);
  ConstDataCoalSection =
    getUnnamedSection(".section __DATA,__const_coal,coalesced",
                      SectionFlags::None);
  ConstDataSection = getUnnamedSection(".const_data", SectionFlags::None);
  DataCoalSection = getUnnamedSection(".section __DATA,__datacoal_nt,coalesced",
                                      SectionFlags::Writeable);
}

/// emitUsedDirectiveFor - On Darwin, internally linked data beginning with
/// the PrivateGlobalPrefix or the LessPrivateGlobalPrefix does not have the
/// directive emitted (this occurs in ObjC metadata).

bool
DarwinTargetAsmInfo::emitUsedDirectiveFor(const GlobalValue* GV,
                                          Mangler *Mang) const {
  if (GV==0)
    return false;
  if (GV->hasInternalLinkage() && !isa<Function>(GV) &&
      ((strlen(getPrivateGlobalPrefix()) != 0 &&
        Mang->getValueName(GV).substr(0,strlen(getPrivateGlobalPrefix())) ==
          getPrivateGlobalPrefix()) ||
       (strlen(getLessPrivateGlobalPrefix()) != 0 &&
        Mang->getValueName(GV).substr(0,strlen(getLessPrivateGlobalPrefix())) ==
          getLessPrivateGlobalPrefix())))
    return false;
  return true;
}

const Section*
DarwinTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);
  bool isWeak = GV->isWeakForLinker();
  bool isNonStatic = (DTM->getRelocationModel() != Reloc::Static);

  switch (Kind) {
   case SectionKind::Text:
    if (isWeak)
      return TextCoalSection;
    else
      return TextSection;
   case SectionKind::Data:
   case SectionKind::ThreadData:
   case SectionKind::BSS:
   case SectionKind::ThreadBSS:
    if (cast<GlobalVariable>(GV)->isConstant())
      return (isWeak ? ConstDataCoalSection : ConstDataSection);
    else
      return (isWeak ? DataCoalSection : DataSection);
   case SectionKind::ROData:
    return (isWeak ? ConstDataCoalSection :
            (isNonStatic ? ConstDataSection : getReadOnlySection_()));
   case SectionKind::RODataMergeStr:
    return (isWeak ?
            ConstDataCoalSection :
            MergeableStringSection(cast<GlobalVariable>(GV)));
   case SectionKind::RODataMergeConst:
    return (isWeak ?
            ConstDataCoalSection:
            MergeableConstSection(cast<GlobalVariable>(GV)));
   default:
    assert(0 && "Unsuported section kind for global");
  }

  // FIXME: Do we have any extra special weird cases?
}

const Section*
DarwinTargetAsmInfo::MergeableStringSection(const GlobalVariable *GV) const {
  const TargetData *TD = DTM->getTargetData();
  Constant *C = cast<GlobalVariable>(GV)->getInitializer();
  const Type *Type = cast<ConstantArray>(C)->getType()->getElementType();

  unsigned Size = TD->getABITypeSize(Type);
  if (Size) {
    const TargetData *TD = DTM->getTargetData();
    unsigned Align = TD->getPreferredAlignment(GV);
    if (Align <= 32)
      return getCStringSection_();
  }

  return getReadOnlySection_();
}

const Section*
DarwinTargetAsmInfo::MergeableConstSection(const GlobalVariable *GV) const {
  Constant *C = GV->getInitializer();

  return MergeableConstSection(C->getType());
}

inline const Section*
DarwinTargetAsmInfo::MergeableConstSection(const Type *Ty) const {
  const TargetData *TD = DTM->getTargetData();

  unsigned Size = TD->getABITypeSize(Ty);
  if (Size == 4)
    return FourByteConstantSection_;
  else if (Size == 8)
    return EightByteConstantSection_;
  else if (Size == 16 && SixteenByteConstantSection_)
    return SixteenByteConstantSection_;

  return getReadOnlySection_();
}

const Section*
DarwinTargetAsmInfo::SelectSectionForMachineConst(const Type *Ty) const {
  const Section* S = MergeableConstSection(Ty);

  // Handle weird special case, when compiling PIC stuff.
  if (S == getReadOnlySection_() &&
      DTM->getRelocationModel() != Reloc::Static)
    return ConstDataSection;

  return S;
}

std::string
DarwinTargetAsmInfo::UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const {
  assert(0 && "Darwin does not use unique sections");
  return "";
}

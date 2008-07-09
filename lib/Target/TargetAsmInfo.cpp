//===-- TargetAsmInfo.cpp - Asm Info ---------------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Dwarf.h"
#include <cctype>
#include <cstring>

using namespace llvm;

TargetAsmInfo::TargetAsmInfo() :
  TextSection("\t.text"),
  DataSection("\t.data"),
  BSSSection("\t.bss"),
  TLSDataSection("\t.section .tdata,\"awT\",@progbits"),
  TLSBSSSection("\t.section .tbss,\"awT\",@nobits"),
  ZeroFillDirective(0),
  NonexecutableStackDirective(0),
  NeedsSet(false),
  MaxInstLength(4),
  PCSymbol("$"),
  SeparatorChar(';'),
  CommentString("#"),
  GlobalPrefix(""),
  PrivateGlobalPrefix("."),
  JumpTableSpecialLabelPrefix(0),
  GlobalVarAddrPrefix(""),
  GlobalVarAddrSuffix(""),
  FunctionAddrPrefix(""),
  FunctionAddrSuffix(""),
  PersonalityPrefix(""),
  PersonalitySuffix(""),
  NeedsIndirectEncoding(false),
  InlineAsmStart("#APP"),
  InlineAsmEnd("#NO_APP"),
  AssemblerDialect(0),
  StringConstantPrefix(".str"),
  ZeroDirective("\t.zero\t"),
  ZeroDirectiveSuffix(0),
  AsciiDirective("\t.ascii\t"),
  AscizDirective("\t.asciz\t"),
  Data8bitsDirective("\t.byte\t"),
  Data16bitsDirective("\t.short\t"),
  Data32bitsDirective("\t.long\t"),
  Data64bitsDirective("\t.quad\t"),
  AlignDirective("\t.align\t"),
  AlignmentIsInBytes(true),
  TextAlignFillValue(0),
  SwitchToSectionDirective("\t.section\t"),
  TextSectionStartSuffix(""),
  DataSectionStartSuffix(""),
  SectionEndDirectiveSuffix(0),
  ConstantPoolSection("\t.section .rodata"),
  JumpTableDataSection("\t.section .rodata"),
  JumpTableDirective(0),
  CStringSection(0),
  StaticCtorsSection("\t.section .ctors,\"aw\",@progbits"),
  StaticDtorsSection("\t.section .dtors,\"aw\",@progbits"),
  FourByteConstantSection(0),
  EightByteConstantSection(0),
  SixteenByteConstantSection(0),
  ReadOnlySection(0),
  GlobalDirective("\t.globl\t"),
  SetDirective(0),
  LCOMMDirective(0),
  COMMDirective("\t.comm\t"),
  COMMDirectiveTakesAlignment(true),
  HasDotTypeDotSizeDirective(true),
  UsedDirective(0),
  WeakRefDirective(0),
  WeakDefDirective(0),
  HiddenDirective("\t.hidden\t"),
  ProtectedDirective("\t.protected\t"),
  AbsoluteDebugSectionOffsets(false),
  AbsoluteEHSectionOffsets(false),
  HasLEB128(false),
  HasDotLocAndDotFile(false),
  SupportsDebugInformation(false),
  SupportsExceptionHandling(false),
  DwarfRequiresFrameSection(true),
  GlobalEHDirective(0),
  SupportsWeakOmittedEHFrame(true),
  DwarfSectionOffsetDirective(0),
  DwarfAbbrevSection(".debug_abbrev"),
  DwarfInfoSection(".debug_info"),
  DwarfLineSection(".debug_line"),
  DwarfFrameSection(".debug_frame"),
  DwarfPubNamesSection(".debug_pubnames"),
  DwarfPubTypesSection(".debug_pubtypes"),
  DwarfStrSection(".debug_str"),
  DwarfLocSection(".debug_loc"),
  DwarfARangesSection(".debug_aranges"),
  DwarfRangesSection(".debug_ranges"),
  DwarfMacInfoSection(".debug_macinfo"),
  DwarfEHFrameSection(".eh_frame"),
  DwarfExceptionSection(".gcc_except_table"),
  AsmTransCBE(0) {
}

TargetAsmInfo::~TargetAsmInfo() {
}

/// Measure the specified inline asm to determine an approximation of its
/// length.
/// Comments (which run till the next SeparatorChar or newline) do not
/// count as an instruction.
/// Any other non-whitespace text is considered an instruction, with
/// multiple instructions separated by SeparatorChar or newlines.
/// Variable-length instructions are not handled here; this function
/// may be overloaded in the target code to do that.
unsigned TargetAsmInfo::getInlineAsmLength(const char *Str) const {
  // Count the number of instructions in the asm.
  bool atInsnStart = true;
  unsigned Length = 0;
  for (; *Str; ++Str) {
    if (*Str == '\n' || *Str == SeparatorChar)
      atInsnStart = true;
    if (atInsnStart && !isspace(*Str)) {
      Length += MaxInstLength;
      atInsnStart = false;
    }
    if (atInsnStart && strncmp(Str, CommentString, strlen(CommentString))==0)
      atInsnStart = false;
  }

  return Length;
}

unsigned TargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                              bool Global) const {
  return dwarf::DW_EH_PE_absptr;
}

static bool isSuitableForBSS(const GlobalVariable *GV) {
  if (!GV->hasInitializer())
    return true;

  // Leave constant zeros in readonly constant sections, so they can be shared
  Constant *C = GV->getInitializer();
  return (C->isNullValue() && !GV->isConstant() && !NoZerosInBSS);
}

SectionKind::Kind
TargetAsmInfo::SectionKindForGlobal(const GlobalValue *GV) const {
  // Early exit - functions should be always in text sections.
  if (isa<Function>(GV))
    return SectionKind::Text;

  const GlobalVariable* GVar = dyn_cast<GlobalVariable>(GV);
  bool isThreadLocal = GVar->isThreadLocal();
  assert(GVar && "Invalid global value for section selection");

  if (isSuitableForBSS(GVar)) {
    // Variable can be easily put to BSS section.
    return (isThreadLocal ? SectionKind::ThreadBSS : SectionKind::BSS);
  } else if (GVar->isConstant() && !isThreadLocal) {
    // Now we know, that varible has initializer and it is constant. We need to
    // check its initializer to decide, which section to output it into. Also
    // note, there is no thread-local r/o section.
    Constant *C = GVar->getInitializer();
    if (C->ContainsRelocations())
      return SectionKind::ROData;
    else {
      const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
      // Check, if initializer is a null-terminated string
      if (CVA && CVA->isCString())
        return SectionKind::RODataMergeStr;
      else
        return SectionKind::RODataMergeConst;
    }
  }

  // Variable is not constant or thread-local - emit to generic data section.
  return (isThreadLocal ? SectionKind::ThreadData : SectionKind::Data);
}

unsigned
TargetAsmInfo::SectionFlagsForGlobal(const GlobalValue *GV,
                                     const char* Name) const {
  unsigned Flags = SectionFlags::None;

  // Decode flags from global itself.
  if (GV) {
    SectionKind::Kind Kind = SectionKindForGlobal(GV);
    switch (Kind) {
     case SectionKind::Text:
      Flags |= SectionFlags::Code;
      break;
     case SectionKind::ThreadData:
      Flags |= SectionFlags::TLS;
      // FALLS THROUGH
     case SectionKind::Data:
      Flags |= SectionFlags::Writeable;
      break;
     case SectionKind::ThreadBSS:
      Flags |= SectionFlags::TLS;
      // FALLS THROUGH
     case SectionKind::BSS:
      Flags |= SectionFlags::BSS;
      break;
     case SectionKind::ROData:
      // No additional flags here
      break;
     case SectionKind::RODataMergeStr:
      Flags |= SectionFlags::Strings;
      // FALLS THROUGH
     case SectionKind::RODataMergeConst:
      Flags |= SectionFlags::Mergeable;
      break;
     default:
      assert(0 && "Unexpected section kind!");
    }

    if (GV->hasLinkOnceLinkage() ||
        GV->hasWeakLinkage() ||
        GV->hasCommonLinkage())
      Flags |= SectionFlags::Linkonce;
  }

  // Add flags from sections, if any.
  if (Name) {
    Flags |= SectionFlags::Named;

    // Some lame default implementation based on some magic section names.
    if (strncmp(Name, ".gnu.linkonce.b.", 16) == 0 ||
        strncmp(Name, ".llvm.linkonce.b.", 17) == 0)
      Flags |= SectionFlags::BSS;
    else if (strcmp(Name, ".tdata") == 0 ||
             strncmp(Name, ".tdata.", 7) == 0 ||
             strncmp(Name, ".gnu.linkonce.td.", 17) == 0 ||
             strncmp(Name, ".llvm.linkonce.td.", 18) == 0)
      Flags |= SectionFlags::TLS;
    else if (strcmp(Name, ".tbss") == 0 ||
             strncmp(Name, ".tbss.", 6) == 0 ||
             strncmp(Name, ".gnu.linkonce.tb.", 17) == 0 ||
             strncmp(Name, ".llvm.linkonce.tb.", 18) == 0)
      Flags |= SectionFlags::BSS | SectionFlags::TLS;
  }

  return Flags;
}

std::string
TargetAsmInfo::SectionForGlobal(const GlobalValue *GV) const {
  unsigned Flags = SectionFlagsForGlobal(GV, GV->getSection().c_str());

  std::string Name;

  // FIXME: Should we use some hashing based on section name and just check
  // flags?

  // Select section name
  if (GV->hasSection()) {
    // Honour section already set, if any
    Name = GV->getSection();
  } else {
    // Use default section depending on the 'type' of global
    Name = SelectSectionForGlobal(GV);
  }

  // If section is named we need to switch into it via special '.section'
  // directive and also append funky flags. Otherwise - section name is just
  // some magic assembler directive.
  if (Flags & SectionFlags::Named)
    Name = SwitchToSectionDirective + Name + PrintSectionFlags(Flags);

  return Name;
}

// Lame default implementation. Calculate the section name for global.
std::string
TargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);

  if (GV->hasLinkOnceLinkage() ||
      GV->hasWeakLinkage() ||
      GV->hasCommonLinkage())
    return UniqueSectionForGlobal(GV, Kind);
  else {
    if (Kind == SectionKind::Text)
      return getTextSection();
    else if (Kind == SectionKind::BSS && getBSSSection())
      return getBSSSection();
    else if (getReadOnlySection() &&
             (Kind == SectionKind::ROData ||
              Kind == SectionKind::RODataMergeConst ||
              Kind == SectionKind::RODataMergeStr))
      return getReadOnlySection();
  }

  return getDataSection();
}

std::string
TargetAsmInfo::UniqueSectionForGlobal(const GlobalValue* GV,
                                      SectionKind::Kind Kind) const {
  switch (Kind) {
   case SectionKind::Text:
    return ".gnu.linkonce.t." + GV->getName();
   case SectionKind::Data:
    return ".gnu.linkonce.d." + GV->getName();
   case SectionKind::BSS:
    return ".gnu.linkonce.b." + GV->getName();
   case SectionKind::ROData:
   case SectionKind::RODataMergeConst:
   case SectionKind::RODataMergeStr:
    return ".gnu.linkonce.r." + GV->getName();
   case SectionKind::ThreadData:
    return ".gnu.linkonce.td." + GV->getName();
   case SectionKind::ThreadBSS:
    return ".gnu.linkonce.tb." + GV->getName();
   default:
    assert(0 && "Unknown section kind");
  }
}

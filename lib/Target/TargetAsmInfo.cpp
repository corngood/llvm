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
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <cstring>
using namespace llvm;

TargetAsmInfo::TargetAsmInfo(const TargetMachine &tm)
  : TM(tm) 
{
  BSSSection = "\t.bss";
  BSSSection_ = 0;
  ReadOnlySection = 0;
  TLSDataSection = 0;
  TLSBSSSection = 0;
  ZeroFillDirective = 0;
  NonexecutableStackDirective = 0;
  NeedsSet = false;
  MaxInstLength = 4;
  PCSymbol = "$";
  SeparatorChar = ';';
  CommentColumn = 60;
  CommentString = "#";
  FirstOperandColumn = 0;
  MaxOperandLength = 0;
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".";
  LinkerPrivateGlobalPrefix = "";
  JumpTableSpecialLabelPrefix = 0;
  GlobalVarAddrPrefix = "";
  GlobalVarAddrSuffix = "";
  FunctionAddrPrefix = "";
  FunctionAddrSuffix = "";
  PersonalityPrefix = "";
  PersonalitySuffix = "";
  NeedsIndirectEncoding = false;
  InlineAsmStart = "#APP";
  InlineAsmEnd = "#NO_APP";
  AssemblerDialect = 0;
  AllowQuotesInName = false;
  ZeroDirective = "\t.zero\t";
  ZeroDirectiveSuffix = 0;
  AsciiDirective = "\t.ascii\t";
  AscizDirective = "\t.asciz\t";
  Data8bitsDirective = "\t.byte\t";
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = "\t.quad\t";
  AlignDirective = "\t.align\t";
  AlignmentIsInBytes = true;
  TextAlignFillValue = 0;
  SwitchToSectionDirective = "\t.section\t";
  TextSectionStartSuffix = "";
  DataSectionStartSuffix = "";
  SectionEndDirectiveSuffix = 0;
  ConstantPoolSection = "\t.section .rodata";
  JumpTableDataSection = "\t.section .rodata";
  JumpTableDirective = 0;
  CStringSection = 0;
  CStringSection_ = 0;
  // FIXME: Flags are ELFish - replace with normal section stuff.
  StaticCtorsSection = "\t.section .ctors,\"aw\",@progbits";
  StaticDtorsSection = "\t.section .dtors,\"aw\",@progbits";
  GlobalDirective = "\t.globl\t";
  SetDirective = 0;
  LCOMMDirective = 0;
  COMMDirective = "\t.comm\t";
  COMMDirectiveTakesAlignment = true;
  HasDotTypeDotSizeDirective = true;
  HasSingleParameterDotFile = true;
  UsedDirective = 0;
  WeakRefDirective = 0;
  WeakDefDirective = 0;
  // FIXME: These are ELFish - move to ELFTAI.
  HiddenDirective = "\t.hidden\t";
  ProtectedDirective = "\t.protected\t";
  AbsoluteDebugSectionOffsets = false;
  AbsoluteEHSectionOffsets = false;
  HasLEB128 = false;
  HasDotLocAndDotFile = false;
  SupportsDebugInformation = false;
  SupportsExceptionHandling = false;
  DwarfRequiresFrameSection = true;
  DwarfUsesInlineInfoSection = false;
  Is_EHSymbolPrivate = true;
  GlobalEHDirective = 0;
  SupportsWeakOmittedEHFrame = true;
  DwarfSectionOffsetDirective = 0;
  DwarfAbbrevSection = ".debug_abbrev";
  DwarfInfoSection = ".debug_info";
  DwarfLineSection = ".debug_line";
  DwarfFrameSection = ".debug_frame";
  DwarfPubNamesSection = ".debug_pubnames";
  DwarfPubTypesSection = ".debug_pubtypes";
  DwarfDebugInlineSection = ".debug_inlined";
  DwarfStrSection = ".debug_str";
  DwarfLocSection = ".debug_loc";
  DwarfARangesSection = ".debug_aranges";
  DwarfRangesSection = ".debug_ranges";
  DwarfMacroInfoSection = ".debug_macinfo";
  DwarfEHFrameSection = ".eh_frame";
  DwarfExceptionSection = ".gcc_except_table";
  AsmTransCBE = 0;
  TextSection = getUnnamedSection("\t.text", SectionFlags::Code);
  DataSection = getUnnamedSection("\t.data", SectionFlags::Writeable);
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

static bool isConstantString(const Constant *C) {
  // First check: is we have constant array of i8 terminated with zero
  const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
  // Check, if initializer is a null-terminated string
  if (CVA && CVA->isCString())
    return true;

  // Another possibility: [1 x i8] zeroinitializer
  if (isa<ConstantAggregateZero>(C)) {
    if (const ArrayType *Ty = dyn_cast<ArrayType>(C->getType())) {
      return (Ty->getElementType() == Type::Int8Ty &&
              Ty->getNumElements() == 1);
    }
  }

  return false;
}

static unsigned SectionFlagsForGlobal(const GlobalValue *GV,
                                      SectionKind::Kind Kind) {
  unsigned Flags = SectionFlags::None;

  // Decode flags from global itself.
  switch (Kind) {
  case SectionKind::Text:
    Flags |= SectionFlags::Code;
    break;
  case SectionKind::ThreadData:
  case SectionKind::ThreadBSS:
    Flags |= SectionFlags::TLS;
    // FALLS THROUGH
  case SectionKind::Data:
  case SectionKind::DataRel:
  case SectionKind::DataRelLocal:
  case SectionKind::DataRelRO:
  case SectionKind::DataRelROLocal:
  case SectionKind::BSS:
    Flags |= SectionFlags::Writeable;
    break;
  case SectionKind::ROData:
  case SectionKind::RODataMergeStr:
  case SectionKind::RODataMergeConst:
    // No additional flags here
    break;
  default:
    llvm_unreachable("Unexpected section kind!");
  }

  if (GV->isWeakForLinker())
    Flags |= SectionFlags::Linkonce;

  return Flags;
}

static unsigned GetSectionFlagsForNamedELFSection(const char *Name) {
  unsigned Flags = 0;
  // Some lame default implementation based on some magic section names.
  if (strncmp(Name, ".gnu.linkonce.b.", 16) == 0 ||
      strncmp(Name, ".llvm.linkonce.b.", 17) == 0 ||
      strncmp(Name, ".gnu.linkonce.sb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.sb.", 18) == 0)
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
  
  return Flags;
}



SectionKind::Kind
TargetAsmInfo::SectionKindForGlobal(const GlobalValue *GV) const {
  // Early exit - functions should be always in text sections.
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (GVar == 0)
    return SectionKind::Text;

  bool isThreadLocal = GVar->isThreadLocal();
  assert(GVar && "Invalid global value for section selection");

  if (isSuitableForBSS(GVar)) {
    // Variable can be easily put to BSS section.
    return isThreadLocal ? SectionKind::ThreadBSS : SectionKind::BSS;
  } else if (GVar->isConstant() && !isThreadLocal) {
    // Now we know, that variable has initializer and it is constant. We need to
    // check its initializer to decide, which section to output it into. Also
    // note, there is no thread-local r/o section.
    Constant *C = GVar->getInitializer();
    if (C->getRelocationInfo() != 0) {
      // Decide whether it is still possible to put symbol into r/o section.
      if (TM.getRelocationModel() != Reloc::Static)
        return SectionKind::Data;
      else
        return SectionKind::ROData;
    } else {
      // Check, if initializer is a null-terminated string
      if (isConstantString(C))
        return SectionKind::RODataMergeStr;
      else
        return SectionKind::RODataMergeConst;
    }
  }

  // Variable either is not constant or thread-local - output to data section.
  return isThreadLocal ? SectionKind::ThreadData : SectionKind::Data;
}


const Section *TargetAsmInfo::SectionForGlobal(const GlobalValue *GV) const {
  // Select section name
  if (GV->hasSection()) {
    // Honour section already set, if any.
    unsigned Flags = SectionFlagsForGlobal(GV, SectionKindForGlobal(GV));

    // This is an explicitly named section.
    Flags |= SectionFlags::Named;
    
    // If the target has magic semantics for certain section names, make sure to
    // pick up the flags.  This allows the user to write things with attribute
    // section and still get the appropriate section flags printed.
    Flags |= GetSectionFlagsForNamedELFSection(GV->getSection().c_str());
    
    return getNamedSection(GV->getSection().c_str(), Flags);
  }

  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (GV->isWeakForLinker()) {
    if (const char *Prefix =
          getSectionPrefixForUniqueGlobal(SectionKindForGlobal(GV))) {
      unsigned Flags = SectionFlagsForGlobal(GV, SectionKindForGlobal(GV));

      // FIXME: Use mangler interface (PR4584).
      std::string Name = Prefix+GV->getNameStr();
      return getNamedSection(Name.c_str(), Flags);
    }
  }
  
  // Use default section depending on the 'type' of global
  return SelectSectionForGlobal(GV);
}

// Lame default implementation. Calculate the section name for global.
const Section*
TargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);

  if (Kind == SectionKind::Text)
    return getTextSection();
  
  if (isBSS(Kind))
    if (const Section *S = getBSSSection_())
      return S;
  
  if (SectionKind::isReadOnly(Kind))
    if (const Section *S = getReadOnlySection())
      return S;

  return getDataSection();
}

/// getSectionForMergableConstant - Given a mergable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const Section *
TargetAsmInfo::getSectionForMergableConstant(uint64_t Size,
                                             unsigned ReloInfo) const {
  // FIXME: Support data.rel stuff someday
  // Lame default implementation. Calculate the section name for machine const.
  return getDataSection();
}




const char *
TargetAsmInfo::getSectionPrefixForUniqueGlobal(SectionKind::Kind Kind) const {
  switch (Kind) {
  default: llvm_unreachable("Unknown section kind");
  case SectionKind::Text:             return ".gnu.linkonce.t.";
  case SectionKind::Data:             return ".gnu.linkonce.d.";
  case SectionKind::DataRel:          return ".gnu.linkonce.d.rel.";
  case SectionKind::DataRelLocal:     return ".gnu.linkonce.d.rel.local.";
  case SectionKind::DataRelRO:        return ".gnu.linkonce.d.rel.ro.";
  case SectionKind::DataRelROLocal:   return ".gnu.linkonce.d.rel.ro.local.";
  case SectionKind::BSS:              return ".gnu.linkonce.b.";
  case SectionKind::ROData:
  case SectionKind::RODataMergeConst:
  case SectionKind::RODataMergeStr:   return ".gnu.linkonce.r.";
  case SectionKind::ThreadData:       return ".gnu.linkonce.td.";
  case SectionKind::ThreadBSS:        return ".gnu.linkonce.tb.";
  }
}

const Section *TargetAsmInfo::getNamedSection(const char *Name, unsigned Flags,
                                              bool Override) const {
  Section &S = Sections[Name];

  // This is newly-created section, set it up properly.
  if (S.Flags == SectionFlags::Invalid || Override) {
    S.Flags = Flags | SectionFlags::Named;
    S.Name = Name;
  }

  return &S;
}

const Section*
TargetAsmInfo::getUnnamedSection(const char *Directive, unsigned Flags,
                                 bool Override) const {
  Section& S = Sections[Directive];

  // This is newly-created section, set it up properly.
  if (S.Flags == SectionFlags::Invalid || Override) {
    S.Flags = Flags & ~SectionFlags::Named;
    S.Name = Directive;
  }

  return &S;
}

const std::string&
TargetAsmInfo::getSectionFlags(unsigned Flags) const {
  SectionFlags::FlagsStringsMapType::iterator I = FlagsStrings.find(Flags);

  // We didn't print these flags yet, print and save them to map. This reduces
  // amount of heap trashing due to std::string construction / concatenation.
  if (I == FlagsStrings.end())
    I = FlagsStrings.insert(std::make_pair(Flags,
                                           printSectionFlags(Flags))).first;

  return I->second;
}

unsigned TargetAsmInfo::getULEB128Size(unsigned Value) {
  unsigned Size = 0;
  do {
    Value >>= 7;
    Size += sizeof(int8_t);
  } while (Value);
  return Size;
}

unsigned TargetAsmInfo::getSLEB128Size(int Value) {
  unsigned Size = 0;
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;

  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    Size += sizeof(int8_t);
  } while (IsMore);
  return Size;
}

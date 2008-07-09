//===-- X86TargetAsmInfo.cpp - X86 asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "X86TargetAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86Subtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

static const char *const x86_asm_table[] = {
                                      "{si}", "S",
                                      "{di}", "D",
                                      "{ax}", "a",
                                      "{cx}", "c",
                                      "{memory}", "memory",
                                      "{flags}", "",
                                      "{dirflag}", "",
                                      "{fpsr}", "",
                                      "{cc}", "cc",
                                      0,0};

X86TargetAsmInfo::X86TargetAsmInfo(const X86TargetMachine &TM) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  X86TM = &TM;

  // FIXME - Should be simplified.

  AsmTransCBE = x86_asm_table;
  
  switch (Subtarget->TargetType) {
  case X86Subtarget::isDarwin:
    AlignmentIsInBytes = false;
    TextAlignFillValue = 0x90;
    GlobalPrefix = "_";
    if (!Subtarget->is64Bit())
      Data64bitsDirective = 0;       // we can't emit a 64-bit unit
    ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
    PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
    BSSSection = 0;                       // no BSS section.
    ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
    ConstantPoolSection = "\t.const\n";
    JumpTableDataSection = "\t.const\n";
    CStringSection = "\t.cstring";
    FourByteConstantSection = "\t.literal4\n";
    EightByteConstantSection = "\t.literal8\n";
    if (Subtarget->is64Bit())
      SixteenByteConstantSection = "\t.literal16\n";
    ReadOnlySection = "\t.const\n";
    LCOMMDirective = "\t.lcomm\t";
    SwitchToSectionDirective = "\t.section ";
    StringConstantPrefix = "\1LC";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    if (TM.getRelocationModel() == Reloc::Static) {
      StaticCtorsSection = ".constructor";
      StaticDtorsSection = ".destructor";
    } else {
      StaticCtorsSection = ".mod_init_func";
      StaticDtorsSection = ".mod_term_func";
    }
    if (Subtarget->is64Bit()) {
      PersonalityPrefix = "";
      PersonalitySuffix = "+4@GOTPCREL";
    } else {
      PersonalityPrefix = "L";
      PersonalitySuffix = "$non_lazy_ptr";
    }
    NeedsIndirectEncoding = true;
    InlineAsmStart = "## InlineAsm Start";
    InlineAsmEnd = "## InlineAsm End";
    CommentString = "##";
    SetDirective = "\t.set";
    PCSymbol = ".";
    UsedDirective = "\t.no_dead_strip\t";
    WeakDefDirective = "\t.weak_definition ";
    WeakRefDirective = "\t.weak_reference ";
    HiddenDirective = "\t.private_extern ";
    ProtectedDirective = "\t.globl\t";
    
    // In non-PIC modes, emit a special label before jump tables so that the
    // linker can perform more accurate dead code stripping.
    if (TM.getRelocationModel() != Reloc::PIC_) {
      // Emit a local label that is preserved until the linker runs.
      JumpTableSpecialLabelPrefix = "l";
    }

    SupportsDebugInformation = true;
    NeedsSet = true;
    DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
    DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
    DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
    DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
    DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
    DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
    DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
    DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
    DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
    DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
    DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";

    // Exceptions handling
    SupportsExceptionHandling = true;
    GlobalEHDirective = "\t.globl\t";
    SupportsWeakOmittedEHFrame = false;
    AbsoluteEHSectionOffsets = false;
    DwarfEHFrameSection =
    ".section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support";
    DwarfExceptionSection = ".section __DATA,__gcc_except_tab";
    break;

  case X86Subtarget::isELF:
    ReadOnlySection = "\t.section\t.rodata";
    FourByteConstantSection = "\t.section\t.rodata.cst4,\"aM\",@progbits,4";
    EightByteConstantSection = "\t.section\t.rodata.cst8,\"aM\",@progbits,8";
    SixteenByteConstantSection = "\t.section\t.rodata.cst16,\"aM\",@progbits,16";
    CStringSection = "\t.section\t.rodata.str1.1,\"aMS\",@progbits,1";
    PrivateGlobalPrefix = ".L";
    WeakRefDirective = "\t.weak\t";
    SetDirective = "\t.set\t";
    PCSymbol = ".";

    // Set up DWARF directives
    HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

    // Debug Information
    AbsoluteDebugSectionOffsets = true;
    SupportsDebugInformation = true;
    DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",@progbits";
    DwarfInfoSection =    "\t.section\t.debug_info,\"\",@progbits";
    DwarfLineSection =    "\t.section\t.debug_line,\"\",@progbits";
    DwarfFrameSection =   "\t.section\t.debug_frame,\"\",@progbits";
    DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",@progbits";
    DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",@progbits";
    DwarfStrSection =     "\t.section\t.debug_str,\"\",@progbits";
    DwarfLocSection =     "\t.section\t.debug_loc,\"\",@progbits";
    DwarfARangesSection = "\t.section\t.debug_aranges,\"\",@progbits";
    DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",@progbits";
    DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";

    // Exceptions handling
    if (!Subtarget->is64Bit())
      SupportsExceptionHandling = true;
    AbsoluteEHSectionOffsets = false;
    DwarfEHFrameSection = "\t.section\t.eh_frame,\"aw\",@progbits";
    DwarfExceptionSection = "\t.section\t.gcc_except_table,\"a\",@progbits";
    break;

  case X86Subtarget::isCygwin:
  case X86Subtarget::isMingw:
    GlobalPrefix = "_";
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    StaticCtorsSection = "\t.section .ctors,\"aw\"";
    StaticDtorsSection = "\t.section .dtors,\"aw\"";
    HiddenDirective = NULL;
    PrivateGlobalPrefix = "L";  // Prefix for private global symbols
    WeakRefDirective = "\t.weak\t";
    SetDirective = "\t.set\t";

    // Set up DWARF directives
    HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
    AbsoluteDebugSectionOffsets = true;
    AbsoluteEHSectionOffsets = false;
    SupportsDebugInformation = true;
    DwarfSectionOffsetDirective = "\t.secrel32\t";
    DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"dr\"";
    DwarfInfoSection =    "\t.section\t.debug_info,\"dr\"";
    DwarfLineSection =    "\t.section\t.debug_line,\"dr\"";
    DwarfFrameSection =   "\t.section\t.debug_frame,\"dr\"";
    DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"dr\"";
    DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"dr\"";
    DwarfStrSection =     "\t.section\t.debug_str,\"dr\"";
    DwarfLocSection =     "\t.section\t.debug_loc,\"dr\"";
    DwarfARangesSection = "\t.section\t.debug_aranges,\"dr\"";
    DwarfRangesSection =  "\t.section\t.debug_ranges,\"dr\"";
    DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"dr\"";
    break;

  case X86Subtarget::isWindows:
    GlobalPrefix = "_";
    HasDotTypeDotSizeDirective = false;
    break;

  default: break;
  }
  
  if (Subtarget->isFlavorIntel()) {
    GlobalPrefix = "_";
    CommentString = ";";
  
    PrivateGlobalPrefix = "$";
    AlignDirective = "\talign\t";
    ZeroDirective = "\tdb\t";
    ZeroDirectiveSuffix = " dup(0)";
    AsciiDirective = "\tdb\t";
    AscizDirective = 0;
    Data8bitsDirective = "\tdb\t";
    Data16bitsDirective = "\tdw\t";
    Data32bitsDirective = "\tdd\t";
    Data64bitsDirective = "\tdq\t";
    HasDotTypeDotSizeDirective = false;
    
    TextSection = "_text";
    DataSection = "_data";
    JumpTableDataSection = NULL;
    SwitchToSectionDirective = "";
    TextSectionStartSuffix = "\tsegment 'CODE'";
    DataSectionStartSuffix = "\tsegment 'DATA'";
    SectionEndDirectiveSuffix = "\tends\n";
  }

  // On Linux we must declare when we can use a non-executable stack.
  if (Subtarget->isLinux())
    NonexecutableStackDirective = "\t.section\t.note.GNU-stack,\"\",@progbits";

  AssemblerDialect = Subtarget->getAsmFlavor();
}

bool X86TargetAsmInfo::LowerToBSwap(CallInst *CI) const {
  // FIXME: this should verify that we are targetting a 486 or better.  If not,
  // we will turn this bswap into something that will be lowered to logical ops
  // instead of emitting the bswap asm.  For now, we don't support 486 or lower
  // so don't worry about this.
  
  // Verify this is a simple bswap.
  if (CI->getNumOperands() != 2 ||
      CI->getType() != CI->getOperand(1)->getType() ||
      !CI->getType()->isInteger())
    return false;
  
  const IntegerType *Ty = dyn_cast<IntegerType>(CI->getType());
  if (!Ty || Ty->getBitWidth() % 16 != 0)
    return false;
  
  // Okay, we can do this xform, do so now.
  const Type *Tys[] = { Ty };
  Module *M = CI->getParent()->getParent()->getParent();
  Constant *Int = Intrinsic::getDeclaration(M, Intrinsic::bswap, Tys, 1);
  
  Value *Op = CI->getOperand(1);
  Op = CallInst::Create(Int, Op, CI->getName(), CI);
  
  CI->replaceAllUsesWith(Op);
  CI->eraseFromParent();
  return true;
}


bool X86TargetAsmInfo::ExpandInlineAsm(CallInst *CI) const {
  InlineAsm *IA = cast<InlineAsm>(CI->getCalledValue());
  std::vector<InlineAsm::ConstraintInfo> Constraints = IA->ParseConstraints();
  
  std::string AsmStr = IA->getAsmString();
  
  // TODO: should remove alternatives from the asmstring: "foo {a|b}" -> "foo a"
  std::vector<std::string> AsmPieces;
  SplitString(AsmStr, AsmPieces, "\n");  // ; as separator?
  
  switch (AsmPieces.size()) {
  default: return false;    
  case 1:
    AsmStr = AsmPieces[0];
    AsmPieces.clear();
    SplitString(AsmStr, AsmPieces, " \t");  // Split with whitespace.
    
    // bswap $0
    if (AsmPieces.size() == 2 && 
        AsmPieces[0] == "bswap" && AsmPieces[1] == "$0") {
      // No need to check constraints, nothing other than the equivalent of
      // "=r,0" would be valid here.
      return LowerToBSwap(CI);
    }
    break;
  case 3:
    if (CI->getType() == Type::Int64Ty && Constraints.size() >= 2 &&
        Constraints[0].Codes.size() == 1 && Constraints[0].Codes[0] == "A" &&
        Constraints[1].Codes.size() == 1 && Constraints[1].Codes[0] == "0") {
      // bswap %eax / bswap %edx / xchgl %eax, %edx  -> llvm.bswap.i64
      std::vector<std::string> Words;
      SplitString(AsmPieces[0], Words, " \t");
      if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%eax") {
        Words.clear();
        SplitString(AsmPieces[1], Words, " \t");
        if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%edx") {
          Words.clear();
          SplitString(AsmPieces[2], Words, " \t,");
          if (Words.size() == 3 && Words[0] == "xchgl" && Words[1] == "%eax" &&
              Words[2] == "%edx") {
            return LowerToBSwap(CI);
          }
        }
      }
    }
    break;
  }
  return false;
}

/// PreferredEHDataFormat - This hook allows the target to select data
/// format used for encoding pointers in exception handling data. Reason is
/// 0 for data, 1 for code labels, 2 for function pointers. Global is true
/// if the symbol can be relocated.
unsigned X86TargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                                 bool Global) const {
  const X86Subtarget *Subtarget = &X86TM->getSubtarget<X86Subtarget>();

  switch (Subtarget->TargetType) {
  case X86Subtarget::isDarwin:
   if (Reason == DwarfEncoding::Functions && Global)
     return (DW_EH_PE_pcrel | DW_EH_PE_indirect | DW_EH_PE_sdata4);
   else if (Reason == DwarfEncoding::CodeLabels || !Global)
     return DW_EH_PE_pcrel;
   else
     return DW_EH_PE_absptr;

  case X86Subtarget::isELF:
  case X86Subtarget::isCygwin:
  case X86Subtarget::isMingw: {
    CodeModel::Model CM = X86TM->getCodeModel();

    if (X86TM->getRelocationModel() == Reloc::PIC_) {
      unsigned Format = 0;

      if (!Subtarget->is64Bit())
        // 32 bit targets always encode pointers as 4 bytes
        Format = DW_EH_PE_sdata4;
      else {
        // 64 bit targets encode pointers in 4 bytes iff:
        // - code model is small OR
        // - code model is medium and we're emitting externally visible symbols
        //   or any code symbols
        if (CM == CodeModel::Small ||
            (CM == CodeModel::Medium && (Global ||
                                         Reason != DwarfEncoding::Data)))
          Format = DW_EH_PE_sdata4;
        else
          Format = DW_EH_PE_sdata8;
      }

      if (Global)
        Format |= DW_EH_PE_indirect;

      return (Format | DW_EH_PE_pcrel);
    } else {
      if (Subtarget->is64Bit() &&
          (CM == CodeModel::Small ||
           (CM == CodeModel::Medium && Reason != DwarfEncoding::Data)))
        return DW_EH_PE_udata4;
      else
        return DW_EH_PE_absptr;
    }
  }

  default:
   return TargetAsmInfo::PreferredEHDataFormat(Reason, Global);
  }
}

std::string X86TargetAsmInfo::UniqueSectionForGlobal(const GlobalValue* GV,
                                                SectionKind::Kind kind) const {
  const X86Subtarget *Subtarget = &X86TM->getSubtarget<X86Subtarget>();

  switch (Subtarget->TargetType) {
   case X86Subtarget::isDarwin:
    if (kind == SectionKind::Text)
      return "__TEXT,__textcoal_nt,coalesced,pure_instructions";
    else
      return "__DATA,__datacoal_nt,coalesced";
   case X86Subtarget::isCygwin:
   case X86Subtarget::isMingw:
    switch (kind) {
     case SectionKind::Text:
      return ".text$linkonce" + GV->getName();
     case SectionKind::Data:
     case SectionKind::BSS:
     case SectionKind::ThreadData:
     case SectionKind::ThreadBSS:
      return ".data$linkonce" + GV->getName();
     case SectionKind::ROData:
     case SectionKind::RODataMergeConst:
     case SectionKind::RODataMergeStr:
      return ".rdata$linkonce" + GV->getName();
     default:
      assert(0 && "Unknown section kind");
    }
   case X86Subtarget::isELF:
    return TargetAsmInfo::UniqueSectionForGlobal(GV, kind);
   default:
    return "";
  }
}


std::string X86TargetAsmInfo::SectionForGlobal(const GlobalValue *GV) const {
  const X86Subtarget *Subtarget = &X86TM->getSubtarget<X86Subtarget>();
  SectionKind::Kind kind = SectionKindForGlobal(GV);
  unsigned flags = SectionFlagsForGlobal(GV, GV->getSection().c_str());
  std::string Name;

  // FIXME: Should we use some hashing based on section name and just check
  // flags?
  // FIXME: It seems, that Darwin uses much more sections.

  // Select section name
  if (GV->hasSection()) {
    // Honour section already set, if any
    Name = GV->getSection();
  } else {
    // Use default section depending on the 'type' of global
    if (const Function *F = dyn_cast<Function>(GV)) {
      switch (F->getLinkage()) {
       default: assert(0 && "Unknown linkage type!");
       case Function::InternalLinkage:
       case Function::DLLExportLinkage:
       case Function::ExternalLinkage:
        Name = TextSection;
        break;
       case Function::WeakLinkage:
       case Function::LinkOnceLinkage:
        Name = UniqueSectionForGlobal(F, kind);
        break;
      }
    } else if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
      if (GVar->hasCommonLinkage() ||
          GVar->hasLinkOnceLinkage() ||
          GVar->hasWeakLinkage())
        Name = UniqueSectionForGlobal(GVar, kind);
      else {
        switch (kind) {
         case SectionKind::Data:
          Name = DataSection;
          break;
         case SectionKind::BSS:
          Name = (BSSSection ? BSSSection : DataSection);
          break;
         case SectionKind::ROData:
         case SectionKind::RODataMergeStr:
         case SectionKind::RODataMergeConst:
          // FIXME: Temporary
          Name = DataSection;
          break;
         case SectionKind::ThreadData:
          Name = (TLSDataSection ? TLSDataSection : DataSection);
          break;
         case SectionKind::ThreadBSS:
          Name = (TLSBSSSection ? TLSBSSSection : DataSection);
         default:
          assert(0 && "Unsuported section kind for global");
        }
      }
    } else
      assert(0 && "Unsupported global");
  }

  // Add all special flags, etc
  switch (Subtarget->TargetType) {
   case X86Subtarget::isELF:
    Name += ",\"";

    if (!(flags & SectionFlags::Debug))
      Name += 'a';
    if (flags & SectionFlags::Code)
      Name += 'x';
    if (flags & SectionFlags::Writeable)
      Name += 'w';
    if (flags & SectionFlags::Mergeable)
      Name += 'M';
    if (flags & SectionFlags::Strings)
      Name += 'S';
    if (flags & SectionFlags::TLS)
      Name += 'T';

    Name += "\"";

    // FIXME: There can be exceptions here
    if (flags & SectionFlags::BSS)
      Name += ",@nobits";
    else
      Name += ",@progbits";

    // FIXME: entity size for mergeable sections
    break;
   case X86Subtarget::isCygwin:
   case X86Subtarget::isMingw:
    Name += ",\"";

    if (flags & SectionFlags::Code)
      Name += 'x';
    if (flags & SectionFlags::Writeable)
      Name += 'w';

    Name += "\"";

    break;
   case X86Subtarget::isDarwin:
    // Darwin does not use any special flags
   default:
    break;
  }

  return Name;
}


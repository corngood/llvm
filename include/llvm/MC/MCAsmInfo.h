//===-- llvm/MC/MCAsmInfo.h - Asm info --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the basis for target specific
// asm writers.  This class primarily takes care of global printing constants,
// which are used in very similar ways across all targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ASM_INFO_H
#define LLVM_TARGET_ASM_INFO_H

#include "llvm/MC/MCDirectives.h"
#include <cassert>

namespace llvm {
  class MCSection;
  class MCContext;

  /// MCAsmInfo - This class is intended to be used as a base class for asm
  /// properties and features specific to the target.
  namespace ExceptionHandling { enum ExceptionsType { None, Dwarf, SjLj }; }

  class MCAsmInfo {
  protected:
    //===------------------------------------------------------------------===//
    // Properties to be set by the target writer, used to configure asm printer.
    //

    /// HasSubsectionsViaSymbols - True if this target has the MachO
    /// .subsections_via_symbols directive.
    bool HasSubsectionsViaSymbols;           // Default is false.

    /// HasMachoZeroFillDirective - True if this is a MachO target that supports
    /// the macho-specific .zerofill directive for emitting BSS Symbols.
    bool HasMachoZeroFillDirective;               // Default is false.

    /// HasMachoTBSSDirective - True if this is a MachO target that supports
    /// the macho-specific .tbss directive for emitting thread local BSS Symbols
    bool HasMachoTBSSDirective;                 // Default is false.

    /// HasStaticCtorDtorReferenceInStaticMode - True if the compiler should
    /// emit a ".reference .constructors_used" or ".reference .destructors_used"
    /// directive after the a static ctor/dtor list.  This directive is only
    /// emitted in Static relocation model.
    bool HasStaticCtorDtorReferenceInStaticMode;  // Default is false.

    /// LinkerRequiresNonEmptyDwarfLines - True if the linker has a bug and
    /// requires that the debug_line section be of a minimum size. In practice
    /// such a linker requires a non empty line sequence if a file is present.
    bool LinkerRequiresNonEmptyDwarfLines; // Default to false.

    /// MaxInstLength - This is the maximum possible length of an instruction,
    /// which is needed to compute the size of an inline asm.
    unsigned MaxInstLength;                  // Defaults to 4.

    /// PCSymbol - The symbol used to represent the current PC.  Used in PC
    /// relative expressions.
    const char *PCSymbol;                    // Defaults to "$".

    /// SeparatorChar - This character, if specified, is used to separate
    /// instructions from each other when on the same line.  This is used to
    /// measure inline asm instructions.
    char SeparatorChar;                      // Defaults to ';'

    /// CommentColumn - This indicates the comment num (zero-based) at
    /// which asm comments should be printed.
    unsigned CommentColumn;                  // Defaults to 40

    /// CommentString - This indicates the comment character used by the
    /// assembler.
    const char *CommentString;               // Defaults to "#"

    /// LabelSuffix - This is appended to emitted labels.
    const char *LabelSuffix;                 // Defaults to ":"

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;                // Defaults to ""

    /// PrivateGlobalPrefix - This prefix is used for globals like constant
    /// pool entries that are completely private to the .s file and should not
    /// have names in the .o file.  This is often "." or "L".
    const char *PrivateGlobalPrefix;         // Defaults to "."

    /// LinkerPrivateGlobalPrefix - This prefix is used for symbols that should
    /// be passed through the assembler but be removed by the linker.  This
    /// is "l" on Darwin, currently used for some ObjC metadata.
    const char *LinkerPrivateGlobalPrefix;   // Defaults to ""

    /// InlineAsmStart/End - If these are nonempty, they contain a directive to
    /// emit before and after an inline assembly statement.
    const char *InlineAsmStart;              // Defaults to "#APP\n"
    const char *InlineAsmEnd;                // Defaults to "#NO_APP\n"

    /// AssemblerDialect - Which dialect of an assembler variant to use.
    unsigned AssemblerDialect;               // Defaults to 0

    /// AllowQuotesInName - This is true if the assembler allows for complex
    /// symbol names to be surrounded in quotes.  This defaults to false.
    bool AllowQuotesInName;

    /// AllowNameToStartWithDigit - This is true if the assembler allows symbol
    /// names to start with a digit (e.g., "0x0021").  This defaults to false.
    bool AllowNameToStartWithDigit;

    /// AllowPeriodsInName - This is true if the assembler allows periods in
    /// symbol names.  This defaults to true.
    bool AllowPeriodsInName;

    //===--- Data Emission Directives -------------------------------------===//

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;               // Defaults to "\t.zero\t"

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;              // Defaults to "\t.ascii\t"

    /// AscizDirective - If not null, this allows for special handling of
    /// zero terminated strings on this target.  This is commonly supported as
    /// ".asciz".  If a target doesn't support this, it can be set to null.
    const char *AscizDirective;              // Defaults to "\t.asciz\t"

    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;          // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;         // Defaults to "\t.short\t"
    const char *Data32bitsDirective;         // Defaults to "\t.long\t"
    const char *Data64bitsDirective;         // Defaults to "\t.quad\t"

    /// GPRel32Directive - if non-null, a directive that is used to emit a word
    /// which should be relocated as a 32-bit GP-relative offset, e.g. .gpword
    /// on Mips or .gprel32 on Alpha.
    const char *GPRel32Directive;            // Defaults to NULL.

    /// getDataASDirective - Return the directive that should be used to emit
    /// data of the specified size to the specified numeric address space.
    virtual const char *getDataASDirective(unsigned Size, unsigned AS) const {
      assert(AS != 0 && "Don't know the directives for default addr space");
      return 0;
    }

    /// SunStyleELFSectionSwitchSyntax - This is true if this target uses "Sun
    /// Style" syntax for section switching ("#alloc,#write" etc) instead of the
    /// normal ELF syntax (,"a,w") in .section directives.
    bool SunStyleELFSectionSwitchSyntax;     // Defaults to false.

    /// UsesELFSectionDirectiveForBSS - This is true if this target uses ELF
    /// '.section' directive before the '.bss' one. It's used for PPC/Linux
    /// which doesn't support the '.bss' directive only.
    bool UsesELFSectionDirectiveForBSS;      // Defaults to false.

    /// HasMicrosoftFastStdCallMangling - True if this target uses microsoft
    /// style mangling for functions with X86_StdCall/X86_FastCall calling
    /// convention.
    bool HasMicrosoftFastStdCallMangling;    // Defaults to false.

    //===--- Alignment Information ----------------------------------------===//

    /// AlignDirective - The directive used to emit round up to an alignment
    /// boundary.
    ///
    const char *AlignDirective;              // Defaults to "\t.align\t"

    /// AlignmentIsInBytes - If this is true (the default) then the asmprinter
    /// emits ".align N" directives, where N is the number of bytes to align to.
    /// Otherwise, it emits ".align log2(N)", e.g. 3 to align to an 8 byte
    /// boundary.
    bool AlignmentIsInBytes;                 // Defaults to true

    /// TextAlignFillValue - If non-zero, this is used to fill the executable
    /// space created as the result of a alignment directive.
    unsigned TextAlignFillValue;             // Defaults to 0

    //===--- Global Variable Emission Directives --------------------------===//

    /// GlobalDirective - This is the directive used to declare a global entity.
    ///
    const char *GlobalDirective;             // Defaults to NULL.

    /// ExternDirective - This is the directive used to declare external
    /// globals.
    ///
    const char *ExternDirective;             // Defaults to NULL.

    /// HasSetDirective - True if the assembler supports the .set directive.
    bool HasSetDirective;                    // Defaults to true.

    /// HasAggressiveSymbolFolding - False if the assembler requires that we use
    /// Lc = a - b
    /// .long Lc
    /// instead of
    /// .long a - b
    bool HasAggressiveSymbolFolding;           // Defaults to true.

    /// HasLCOMMDirective - This is true if the target supports the .lcomm
    /// directive.
    bool HasLCOMMDirective;                  // Defaults to false.

    /// COMMDirectiveAlignmentIsInBytes - True is COMMDirective's optional
    /// alignment is to be specified in bytes instead of log2(n).
    bool COMMDirectiveAlignmentIsInBytes;    // Defaults to true;

    /// HasDotTypeDotSizeDirective - True if the target has .type and .size
    /// directives, this is true for most ELF targets.
    bool HasDotTypeDotSizeDirective;         // Defaults to true.

    /// HasSingleParameterDotFile - True if the target has a single parameter
    /// .file directive, this is true for ELF targets.
    bool HasSingleParameterDotFile;          // Defaults to true.

    /// HasNoDeadStrip - True if this target supports the MachO .no_dead_strip
    /// directive.
    bool HasNoDeadStrip;                     // Defaults to false.

    /// HasSymbolResolver - True if this target supports the MachO
    /// .symbol_resolver directive.
    bool HasSymbolResolver;                     // Defaults to false.

    /// WeakRefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak undefined symbol.
    const char *WeakRefDirective;            // Defaults to NULL.

    /// WeakDefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak defined symbol.
    const char *WeakDefDirective;            // Defaults to NULL.

    /// LinkOnceDirective - This directive, if non-null is used to declare a
    /// global as being a weak defined symbol.  This is used on cygwin/mingw.
    const char *LinkOnceDirective;           // Defaults to NULL.

    /// HiddenVisibilityAttr - This attribute, if not MCSA_Invalid, is used to
    /// declare a symbol as having hidden visibility.
    MCSymbolAttr HiddenVisibilityAttr;       // Defaults to MCSA_Hidden.

    /// ProtectedVisibilityAttr - This attribute, if not MCSA_Invalid, is used
    /// to declare a symbol as having protected visibility.
    MCSymbolAttr ProtectedVisibilityAttr;    // Defaults to MCSA_Protected

    //===--- Dwarf Emission Directives -----------------------------------===//

    /// HasLEB128 - True if target asm supports leb128 directives.
    bool HasLEB128;                          // Defaults to false.

    /// SupportsDebugInformation - True if target supports emission of debugging
    /// information.
    bool SupportsDebugInformation;           // Defaults to false.

    /// SupportsExceptionHandling - True if target supports exception handling.
    ExceptionHandling::ExceptionsType ExceptionsType; // Defaults to None

    /// RequiresFrameSection - true if the Dwarf2 output needs a frame section
    bool DwarfRequiresFrameSection;          // Defaults to true.

    /// DwarfUsesInlineInfoSection - True if DwarfDebugInlineSection is used to
    /// encode inline subroutine information.
    bool DwarfUsesInlineInfoSection;         // Defaults to false.

    /// DwarfSectionOffsetDirective - Special section offset directive.
    const char* DwarfSectionOffsetDirective; // Defaults to NULL

    /// DwarfUsesAbsoluteLabelForStmtList - True if DW_AT_stmt_list needs
    /// absolute label instead of offset.
    bool DwarfUsesAbsoluteLabelForStmtList;  // Defaults to true;

    // DwarfUsesLabelOffsetDifference - True if Dwarf2 output can
    // use EmitLabelOffsetDifference.
    bool DwarfUsesLabelOffsetForRanges;

    //===--- CBE Asm Translation Table -----------------------------------===//

    const char *const *AsmTransCBE;          // Defaults to empty

  public:
    explicit MCAsmInfo();
    virtual ~MCAsmInfo();

    // FIXME: move these methods to DwarfPrinter when the JIT stops using them.
    static unsigned getSLEB128Size(int Value);
    static unsigned getULEB128Size(unsigned Value);

    bool hasSubsectionsViaSymbols() const { return HasSubsectionsViaSymbols; }

    // Data directive accessors.
    //
    const char *getData8bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data8bitsDirective : getDataASDirective(8, AS);
    }
    const char *getData16bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data16bitsDirective : getDataASDirective(16, AS);
    }
    const char *getData32bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data32bitsDirective : getDataASDirective(32, AS);
    }
    const char *getData64bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data64bitsDirective : getDataASDirective(64, AS);
    }
    const char *getGPRel32Directive() const { return GPRel32Directive; }

    /// getNonexecutableStackSection - Targets can implement this method to
    /// specify a section to switch to if the translation unit doesn't have any
    /// trampolines that require an executable stack.
    virtual const MCSection *getNonexecutableStackSection(MCContext &Ctx) const{
      return 0;
    }

    bool usesSunStyleELFSectionSwitchSyntax() const {
      return SunStyleELFSectionSwitchSyntax;
    }

    bool usesELFSectionDirectiveForBSS() const {
      return UsesELFSectionDirectiveForBSS;
    }

    bool hasMicrosoftFastStdCallMangling() const {
      return HasMicrosoftFastStdCallMangling;
    }

    // Accessors.
    //
    bool hasMachoZeroFillDirective() const { return HasMachoZeroFillDirective; }
    bool hasMachoTBSSDirective() const { return HasMachoTBSSDirective; }
    bool hasStaticCtorDtorReferenceInStaticMode() const {
      return HasStaticCtorDtorReferenceInStaticMode;
    }
    bool getLinkerRequiresNonEmptyDwarfLines() const {
      return LinkerRequiresNonEmptyDwarfLines;
    }
    unsigned getMaxInstLength() const {
      return MaxInstLength;
    }
    const char *getPCSymbol() const {
      return PCSymbol;
    }
    char getSeparatorChar() const {
      return SeparatorChar;
    }
    unsigned getCommentColumn() const {
      return CommentColumn;
    }
    const char *getCommentString() const {
      return CommentString;
    }
    const char *getLabelSuffix() const {
      return LabelSuffix;
    }
    const char *getGlobalPrefix() const {
      return GlobalPrefix;
    }
    const char *getPrivateGlobalPrefix() const {
      return PrivateGlobalPrefix;
    }
    const char *getLinkerPrivateGlobalPrefix() const {
      return LinkerPrivateGlobalPrefix;
    }
    const char *getInlineAsmStart() const {
      return InlineAsmStart;
    }
    const char *getInlineAsmEnd() const {
      return InlineAsmEnd;
    }
    unsigned getAssemblerDialect() const {
      return AssemblerDialect;
    }
    bool doesAllowQuotesInName() const {
      return AllowQuotesInName;
    }
    bool doesAllowNameToStartWithDigit() const {
      return AllowNameToStartWithDigit;
    }
    bool doesAllowPeriodsInName() const {
      return AllowPeriodsInName;
    }
    const char *getZeroDirective() const {
      return ZeroDirective;
    }
    const char *getAsciiDirective() const {
      return AsciiDirective;
    }
    const char *getAscizDirective() const {
      return AscizDirective;
    }
    const char *getAlignDirective() const {
      return AlignDirective;
    }
    bool getAlignmentIsInBytes() const {
      return AlignmentIsInBytes;
    }
    unsigned getTextAlignFillValue() const {
      return TextAlignFillValue;
    }
    const char *getGlobalDirective() const {
      return GlobalDirective;
    }
    const char *getExternDirective() const {
      return ExternDirective;
    }
    bool hasSetDirective() const { return HasSetDirective; }
    bool hasAggressiveSymbolFolding() const {
      return HasAggressiveSymbolFolding;
    }
    bool hasLCOMMDirective() const { return HasLCOMMDirective; }
    bool hasDotTypeDotSizeDirective() const {return HasDotTypeDotSizeDirective;}
    bool getCOMMDirectiveAlignmentIsInBytes() const {
      return COMMDirectiveAlignmentIsInBytes;
    }
    bool hasSingleParameterDotFile() const { return HasSingleParameterDotFile; }
    bool hasNoDeadStrip() const { return HasNoDeadStrip; }
    bool hasSymbolResolver() const { return HasSymbolResolver; }
    const char *getWeakRefDirective() const { return WeakRefDirective; }
    const char *getWeakDefDirective() const { return WeakDefDirective; }
    const char *getLinkOnceDirective() const { return LinkOnceDirective; }

    MCSymbolAttr getHiddenVisibilityAttr() const { return HiddenVisibilityAttr;}
    MCSymbolAttr getProtectedVisibilityAttr() const {
      return ProtectedVisibilityAttr;
    }
    bool hasLEB128() const {
      return HasLEB128;
    }
    bool doesSupportDebugInformation() const {
      return SupportsDebugInformation;
    }
    bool doesSupportExceptionHandling() const {
      return ExceptionsType != ExceptionHandling::None;
    }
    ExceptionHandling::ExceptionsType getExceptionHandlingType() const {
      return ExceptionsType;
    }
    bool doesDwarfRequireFrameSection() const {
      return DwarfRequiresFrameSection;
    }
    bool doesDwarfUsesInlineInfoSection() const {
      return DwarfUsesInlineInfoSection;
    }
    const char *getDwarfSectionOffsetDirective() const {
      return DwarfSectionOffsetDirective;
    }
    bool doesDwarfUsesAbsoluteLabelForStmtList() const {
      return DwarfUsesAbsoluteLabelForStmtList;
    }
    bool doesDwarfUsesLabelOffsetForRanges() const {
      return DwarfUsesLabelOffsetForRanges;
    }
    const char *const *getAsmCBE() const {
      return AsmTransCBE;
    }
  };
}

#endif

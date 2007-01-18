//===-- llvm/Target/TargetAsmInfo.h - Asm info ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class TargetMachine;
  class CallInst;

  /// TargetAsmInfo - This class is intended to be used as a base class for asm
  /// properties and features specific to the target.
  class TargetAsmInfo {
  protected:
    //===------------------------------------------------------------------===//
    // Properties to be set by the target writer, used to configure asm printer.
    //
    
    /// TextSection - Section directive for standard text.
    ///
    const char *TextSection;              // Defaults to ".text".
    
    /// DataSection - Section directive for standard data.
    ///
    const char *DataSection;              // Defaults to ".data".

    /// BSSSection - Section directive for uninitialized data.  Null if this
    /// target doesn't support a BSS section.
    ///
    const char *BSSSection;               // Default to ".bss".
    
    /// ZeroFillDirective - Directive for emitting a global to the ZeroFill
    /// section on this target.  Null if this target doesn't support zerofill.
    const char *ZeroFillDirective;        // Default is null.
    
    /// AddressSize - Size of addresses used in file.
    ///
    unsigned AddressSize;                 // Defaults to 4.

    /// NeedsSet - True if target asm can't compute addresses on data
    /// directives.
    bool NeedsSet;                        // Defaults to false.
    
    /// MaxInstLength - This is the maximum possible length of an instruction,
    /// which is needed to compute the size of an inline asm.
    unsigned MaxInstLength;               // Defaults to 4.
    
    /// SeparatorChar - This character, if specified, is used to separate
    /// instructions from each other when on the same line.  This is used to
    /// measure inline asm instructions.
    char SeparatorChar;                   // Defaults to ';'

    /// CommentString - This indicates the comment character used by the
    /// assembler.
    const char *CommentString;            // Defaults to "#"

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;             // Defaults to ""

    /// PrivateGlobalPrefix - This prefix is used for globals like constant
    /// pool entries that are completely private to the .o file and should not
    /// have names in the .o file.  This is often "." or "L".
    const char *PrivateGlobalPrefix;      // Defaults to "."
    
    /// JumpTableSpecialLabelPrefix - If not null, a extra (dead) label is
    /// emitted before jump tables with the specified prefix.
    const char *JumpTableSpecialLabelPrefix;  // Default to null.
    
    /// GlobalVarAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable (that isn't a function)
    ///
    const char *GlobalVarAddrPrefix;      // Defaults to ""
    const char *GlobalVarAddrSuffix;      // Defaults to ""

    /// FunctionAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable that points to a function.
    /// For example, this is used by the IA64 backend to materialize
    /// function descriptors, by decorating the ".data8" object with the
    /// \literal @fptr( ) \endliteral
    /// link-relocation operator.
    ///
    const char *FunctionAddrPrefix;       // Defaults to ""
    const char *FunctionAddrSuffix;       // Defaults to ""

    /// InlineAsmStart/End - If these are nonempty, they contain a directive to
    /// emit before and after an inline assembly statement.
    const char *InlineAsmStart;           // Defaults to "#APP\n"
    const char *InlineAsmEnd;             // Defaults to "#NO_APP\n"

    /// AssemblerDialect - Which dialect of an assembler variant to use.
    unsigned AssemblerDialect;            // Defaults to 0

    //===--- Data Emission Directives -------------------------------------===//

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;            // Defaults to "\t.zero\t"
    const char *ZeroDirectiveSuffix;      // Defaults to ""

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;           // Defaults to "\t.ascii\t"
    
    /// AscizDirective - If not null, this allows for special handling of
    /// zero terminated strings on this target.  This is commonly supported as
    /// ".asciz".  If a target doesn't support this, it can be set to null.
    const char *AscizDirective;           // Defaults to "\t.asciz\t"

    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;       // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;      // Defaults to "\t.short\t"
    const char *Data32bitsDirective;      // Defaults to "\t.long\t"
    const char *Data64bitsDirective;      // Defaults to "\t.quad\t"

    //===--- Alignment Information ----------------------------------------===//

    /// AlignDirective - The directive used to emit round up to an alignment
    /// boundary.
    ///
    const char *AlignDirective;           // Defaults to "\t.align\t"

    /// AlignmentIsInBytes - If this is true (the default) then the asmprinter
    /// emits ".align N" directives, where N is the number of bytes to align to.
    /// Otherwise, it emits ".align log2(N)", e.g. 3 to align to an 8 byte
    /// boundary.
    bool AlignmentIsInBytes;              // Defaults to true

    //===--- Section Switching Directives ---------------------------------===//
    
    /// SwitchToSectionDirective - This is the directive used when we want to
    /// emit a global to an arbitrary section.  The section name is emited after
    /// this.
    const char *SwitchToSectionDirective; // Defaults to "\t.section\t"
    
    /// TextSectionStartSuffix - This is printed after each start of section
    /// directive for text sections.
    const char *TextSectionStartSuffix;   // Defaults to "".

    /// DataSectionStartSuffix - This is printed after each start of section
    /// directive for data sections.
    const char *DataSectionStartSuffix;   // Defaults to "".
    
    /// SectionEndDirectiveSuffix - If non-null, the asm printer will close each
    /// section with the section name and this suffix printed.
    const char *SectionEndDirectiveSuffix;// Defaults to null.
    
    /// ConstantPoolSection - This is the section that we SwitchToSection right
    /// before emitting the constant pool for a function.
    const char *ConstantPoolSection;      // Defaults to "\t.section .rodata\n"

    /// JumpTableDataSection - This is the section that we SwitchToSection right
    /// before emitting the jump tables for a function when the relocation model
    /// is not PIC.
    const char *JumpTableDataSection;     // Defaults to "\t.section .rodata\n"
    
    /// JumpTableDirective - if non-null, the directive to emit before a jump
    /// table.
    const char *JumpTableDirective;

    /// CStringSection - If not null, this allows for special handling of
    /// cstring constants (\0 terminated string that does not contain any
    /// other null bytes) on this target. This is commonly supported as
    /// ".cstring".
    const char *CStringSection;           // Defaults to NULL

    /// StaticCtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static constructor list.
    /// Defaults to "\t.section .ctors,\"aw\",@progbits".
    const char *StaticCtorsSection;

    /// StaticDtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static destructor list.
    /// Defaults to "\t.section .dtors,\"aw\",@progbits".
    const char *StaticDtorsSection;

    /// FourByteConstantSection, EightByteConstantSection,
    /// SixteenByteConstantSection - These are special sections where we place
    /// 4-, 8-, and 16- byte constant literals.
    const char *FourByteConstantSection;
    const char *EightByteConstantSection;
    const char *SixteenByteConstantSection;
    
    //===--- Global Variable Emission Directives --------------------------===//
    
    /// SetDirective - This is the name of a directive that can be used to tell
    /// the assembler to set the value of a variable to some expression.
    const char *SetDirective;             // Defaults to null.
    
    /// LCOMMDirective - This is the name of a directive (if supported) that can
    /// be used to efficiently declare a local (internal) block of zero
    /// initialized data in the .bss/.data section.  The syntax expected is:
    /// \literal <LCOMMDirective> SYMBOLNAME LENGTHINBYTES, ALIGNMENT
    /// \endliteral
    const char *LCOMMDirective;           // Defaults to null.
    
    const char *COMMDirective;            // Defaults to "\t.comm\t".

    /// COMMDirectiveTakesAlignment - True if COMMDirective take a third
    /// argument that specifies the alignment of the declaration.
    bool COMMDirectiveTakesAlignment;     // Defaults to true.
    
    /// HasDotTypeDotSizeDirective - True if the target has .type and .size
    /// directives, this is true for most ELF targets.
    bool HasDotTypeDotSizeDirective;      // Defaults to true.
    
    /// UsedDirective - This directive, if non-null, is used to declare a global
    /// as being used somehow that the assembler can't see.  This prevents dead
    /// code elimination on some targets.
    const char *UsedDirective;            // Defaults to null.

    /// WeakRefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak undefined symbol.
    const char *WeakRefDirective;         // Defaults to null.
    
    /// HiddenDirective - This directive, if non-null, is used to declare a
    /// global or function as having hidden visibility.
    const char *HiddenDirective;          // Defaults to "\t.hidden\t".
    
    //===--- Dwarf Emission Directives -----------------------------------===//

    /// HasLEB128 - True if target asm supports leb128 directives.
    ///
    bool HasLEB128; // Defaults to false.
    
    /// hasDotLoc - True if target asm supports .loc directives.
    ///
    bool HasDotLoc; // Defaults to false.
    
    /// HasDotFile - True if target asm supports .file directives.
    ///
    bool HasDotFile; // Defaults to false.
    
    /// RequiresFrameSection - true if the Dwarf2 output needs a frame section
    ///
    bool DwarfRequiresFrameSection; // Defaults to true.

    /// DwarfAbbrevSection - Section directive for Dwarf abbrev.
    ///
    const char *DwarfAbbrevSection; // Defaults to ".debug_abbrev".

    /// DwarfInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfInfoSection; // Defaults to ".debug_info".

    /// DwarfLineSection - Section directive for Dwarf info.
    ///
    const char *DwarfLineSection; // Defaults to ".debug_line".
    
    /// DwarfFrameSection - Section directive for Dwarf info.
    ///
    const char *DwarfFrameSection; // Defaults to ".debug_frame".
    
    /// DwarfPubNamesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubNamesSection; // Defaults to ".debug_pubnames".
    
    /// DwarfPubTypesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubTypesSection; // Defaults to ".debug_pubtypes".
    
    /// DwarfStrSection - Section directive for Dwarf info.
    ///
    const char *DwarfStrSection; // Defaults to ".debug_str".

    /// DwarfLocSection - Section directive for Dwarf info.
    ///
    const char *DwarfLocSection; // Defaults to ".debug_loc".

    /// DwarfARangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfARangesSection; // Defaults to ".debug_aranges".

    /// DwarfRangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfRangesSection; // Defaults to ".debug_ranges".

    /// DwarfMacInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfMacInfoSection; // Defaults to ".debug_macinfo".

    //===--- CBE Asm Translation Table -----------------------------------===//

    const char** AsmTransCBE; // Defaults to empty

  public:
    TargetAsmInfo();
    virtual ~TargetAsmInfo();

    /// Measure the specified inline asm to determine an approximation of its
    /// length.
    unsigned getInlineAsmLength(const char *Str) const;

    /// ExpandInlineAsm - This hook allows the target to expand an inline asm
    /// call to be explicit llvm code if it wants to.  This is useful for
    /// turning simple inline asms into LLVM intrinsics, which gives the
    /// compiler more information about the behavior of the code.
    virtual bool ExpandInlineAsm(CallInst *CI) const {
      return false;
    }
    
    // Accessors.
    //
    const char *getTextSection() const {
      return TextSection;
    }
    const char *getDataSection() const {
      return DataSection;
    }
    const char *getBSSSection() const {
      return BSSSection;
    }
    const char *getZeroFillDirective() const {
      return ZeroFillDirective;
    }
    unsigned getAddressSize() const {
      return AddressSize;
    }
    bool needsSet() const {
      return NeedsSet;
    }
    const char *getCommentString() const {
      return CommentString;
    }
    const char *getGlobalPrefix() const {
      return GlobalPrefix;
    }
    const char *getPrivateGlobalPrefix() const {
      return PrivateGlobalPrefix;
    }
    const char *getJumpTableSpecialLabelPrefix() const {
      return JumpTableSpecialLabelPrefix;
    }
    const char *getGlobalVarAddrPrefix() const {
      return GlobalVarAddrPrefix;
    }
    const char *getGlobalVarAddrSuffix() const {
      return GlobalVarAddrSuffix;
    }
    const char *getFunctionAddrPrefix() const {
      return FunctionAddrPrefix;
    }
    const char *getFunctionAddrSuffix() const {
      return FunctionAddrSuffix;
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
    const char *getZeroDirective() const {
      return ZeroDirective;
    }
    const char *getZeroDirectiveSuffix() const {
      return ZeroDirectiveSuffix;
    }
    const char *getAsciiDirective() const {
      return AsciiDirective;
    }
    const char *getAscizDirective() const {
      return AscizDirective;
    }
    const char *getData8bitsDirective() const {
      return Data8bitsDirective;
    }
    const char *getData16bitsDirective() const {
      return Data16bitsDirective;
    }
    const char *getData32bitsDirective() const {
      return Data32bitsDirective;
    }
    const char *getData64bitsDirective() const {
      return Data64bitsDirective;
    }
    const char *getJumpTableDirective() const {
      return JumpTableDirective;
    }
    const char *getAlignDirective() const {
      return AlignDirective;
    }
    bool getAlignmentIsInBytes() const {
      return AlignmentIsInBytes;
    }
    const char *getSwitchToSectionDirective() const {
      return SwitchToSectionDirective;
    }
    const char *getTextSectionStartSuffix() const {
      return TextSectionStartSuffix;
    }
    const char *getDataSectionStartSuffix() const {
      return DataSectionStartSuffix;
    }
    const char *getSectionEndDirectiveSuffix() const {
      return SectionEndDirectiveSuffix;
    }
    const char *getConstantPoolSection() const {
      return ConstantPoolSection;
    }
    const char *getJumpTableDataSection() const {
      return JumpTableDataSection;
    }
    const char *getCStringSection() const {
      return CStringSection;
    }
    const char *getStaticCtorsSection() const {
      return StaticCtorsSection;
    }
    const char *getStaticDtorsSection() const {
      return StaticDtorsSection;
    }
    const char *getFourByteConstantSection() const {
      return FourByteConstantSection;
    }
    const char *getEightByteConstantSection() const {
      return EightByteConstantSection;
    }
    const char *getSixteenByteConstantSection() const {
      return SixteenByteConstantSection;
    }
    const char *getSetDirective() const {
      return SetDirective;
    }
    const char *getLCOMMDirective() const {
      return LCOMMDirective;
    }
    const char *getCOMMDirective() const {
      return COMMDirective;
    }
    bool getCOMMDirectiveTakesAlignment() const {
      return COMMDirectiveTakesAlignment;
    }
    bool hasDotTypeDotSizeDirective() const {
      return HasDotTypeDotSizeDirective;
    }
    const char *getUsedDirective() const {
      return UsedDirective;
    }
    const char *getWeakRefDirective() const {
      return WeakRefDirective;
    }
    const char *getHiddenDirective() const {
      return HiddenDirective;
    }
    bool hasLEB128() const {
      return HasLEB128;
    }
    bool hasDotLoc() const {
      return HasDotLoc;
    }
    bool hasDotFile() const {
      return HasDotFile;
    }
    bool getDwarfRequiresFrameSection() const {
      return DwarfRequiresFrameSection;
    }
    const char *getDwarfAbbrevSection() const {
      return DwarfAbbrevSection;
    }
    const char *getDwarfInfoSection() const {
      return DwarfInfoSection;
    }
    const char *getDwarfLineSection() const {
      return DwarfLineSection;
    }
    const char *getDwarfFrameSection() const {
      return DwarfFrameSection;
    }
    const char *getDwarfPubNamesSection() const {
      return DwarfPubNamesSection;
    }
    const char *getDwarfPubTypesSection() const {
      return DwarfPubTypesSection;
    }
    const char *getDwarfStrSection() const {
      return DwarfStrSection;
    }
    const char *getDwarfLocSection() const {
      return DwarfLocSection;
    }
    const char *getDwarfARangesSection() const {
      return DwarfARangesSection;
    }
    const char *getDwarfRangesSection() const {
      return DwarfRangesSection;
    }
    const char *getDwarfMacInfoSection() const {
      return DwarfMacInfoSection;
    }
    const char** getAsmCBE() const {
      return AsmTransCBE;
    }
  };
}

#endif


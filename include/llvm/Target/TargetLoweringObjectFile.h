//===-- llvm/Target/TargetLoweringObjectFile.h - Object Info ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used to handle lowerings specific to common
// object file formats.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETLOWERINGOBJECTFILE_H
#define LLVM_TARGET_TARGETLOWERINGOBJECTFILE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {
  class MCSection;
  class MCContext;
  class GlobalValue;
  class Mangler;
  class TargetMachine;
  
  
/// SectionInfo - This class is a target-independent classification of a global
/// which is used to simplify target-specific code by exposing common
/// predicates.
class SectionInfo : public SectionKind {
  /// Weak - This is true if the referenced symbol is weak (i.e. linkonce,
  /// weak, weak_odr, etc).  This is orthogonal from the categorization.
  bool Weak : 1;
  
public:
  
  /// Weak - This is true if the referenced symbol is weak (i.e. linkonce,
  /// weak, weak_odr, etc).  This is orthogonal from the categorization.
  bool isWeak() const { return Weak; }
  
  static SectionInfo get(Kind K, bool isWeak = false) {
    SectionInfo Res;
    Res.K = K;
    Res.Weak = isWeak;
    return Res;
  }
  static SectionInfo get(SectionKind K, bool isWeak = false) {
    SectionInfo Res;
    *(SectionKind*)&Res = K;
    Res.Weak = isWeak;
    return Res;
  }
};
  
class TargetLoweringObjectFile {
  MCContext *Ctx;
protected:
  
  TargetLoweringObjectFile();
  
  /// TextSection - Section directive for standard text.
  ///
  const MCSection *TextSection;           // Defaults to ".text".
  
  /// DataSection - Section directive for standard data.
  ///
  const MCSection *DataSection;           // Defaults to ".data".
  
  
  
  // FIXME: SINK THESE.
  const MCSection *BSSSection_;

  /// ReadOnlySection - This is the directive that is emitted to switch to a
  /// read-only section for constant data (e.g. data declared const,
  /// jump tables).
  const MCSection *ReadOnlySection;       // Defaults to NULL
  
  /// TLSDataSection - Section directive for Thread Local data.
  ///
  const MCSection *TLSDataSection;        // Defaults to ".tdata".
  
  /// TLSBSSSection - Section directive for Thread Local uninitialized data.
  /// Null if this target doesn't support a BSS section.
  ///
  const MCSection *TLSBSSSection;         // Defaults to ".tbss".
  
  const MCSection *CStringSection_;
  
public:
  // FIXME: NONPUB.
  const MCSection *getOrCreateSection(const char *Name,
                                      bool isDirective,
                                      SectionKind K) const;
public:
  
  virtual ~TargetLoweringObjectFile();
  
  /// Initialize - this method must be called before any actual lowering is
  /// done.  This specifies the current context for codegen, and gives the
  /// lowering implementations a chance to set up their default sections.
  virtual void Initialize(MCContext &ctx, const TargetMachine &TM) {
    Ctx = &ctx;
  }
  
  
  const MCSection *getTextSection() const { return TextSection; }
  const MCSection *getDataSection() const { return DataSection; }
  
  /// shouldEmitUsedDirectiveFor - This hook allows targets to selectively
  /// decide not to emit the UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  virtual bool shouldEmitUsedDirectiveFor(const GlobalValue *GV,
                                          Mangler *) const {
    return (GV!=0);
  }
  
  /// getSectionForMergeableConstant - Given a mergeable constant with the
  /// specified size and relocation information, return a section that it
  /// should be placed in.
  virtual const MCSection *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  /// getKindForNamedSection - If this target wants to be able to override
  /// section flags based on the name of the section specified for a global
  /// variable, it can implement this.  This is used on ELF systems so that
  /// ".tbss" gets the TLS bit set etc.
  virtual SectionKind getKindForNamedSection(const char *Section,
                                             SectionKind K) const {
    return K;
  }
  
  /// SectionForGlobal - This method computes the appropriate section to emit
  /// the specified global variable or function definition.  This should not
  /// be passed external (or available externally) globals.
  const MCSection *SectionForGlobal(const GlobalValue *GV,
                                    Mangler *Mang,
                                    const TargetMachine &TM) const;
  
  /// getSpecialCasedSectionGlobals - Allow the target to completely override
  /// section assignment of a global.
  /// FIXME: ELIMINATE this by making PIC16 implement ADDRESS with
  /// getFlagsForNamedSection.
  virtual const MCSection *
  getSpecialCasedSectionGlobals(const GlobalValue *GV, Mangler *Mang,
                                SectionInfo Kind) const {
    return 0;
  }
  
  /// getSectionFlagsAsString - Turn the flags in the specified SectionKind
  /// into a string that can be printed to the assembly file after the
  /// ".section foo" part of a section directive.
  virtual void getSectionFlagsAsString(SectionKind Kind,
                                       SmallVectorImpl<char> &Str) const {
  }
  
protected:
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};
  
  
  

class TargetLoweringObjectFileELF : public TargetLoweringObjectFile {
  bool AtIsCommentChar;  // True if @ is the comment character on this target.
  bool HasCrazyBSS;
public:
  /// ELF Constructor - AtIsCommentChar is true if the CommentCharacter from TAI
  /// is "@".
  TargetLoweringObjectFileELF(bool atIsCommentChar = false,
                              // FIXME: REMOVE AFTER UNIQUING IS FIXED.
                              bool hasCrazyBSS = false)
    : AtIsCommentChar(atIsCommentChar), HasCrazyBSS(hasCrazyBSS) {}
    
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
  
  
  /// getSectionForMergeableConstant - Given a mergeable constant with the
  /// specified size and relocation information, return a section that it
  /// should be placed in.
  virtual const MCSection *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  virtual SectionKind getKindForNamedSection(const char *Section,
                                             SectionKind K) const;
  void getSectionFlagsAsString(SectionKind Kind,
                               SmallVectorImpl<char> &Str) const;
  
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
protected:
  const MCSection *DataRelSection;
  const MCSection *DataRelLocalSection;
  const MCSection *DataRelROSection;
  const MCSection *DataRelROLocalSection;
  
  const MCSection *MergeableConst4Section;
  const MCSection *MergeableConst8Section;
  const MCSection *MergeableConst16Section;
};

  
  
class TargetLoweringObjectFileMachO : public TargetLoweringObjectFile {
  const MCSection *TextCoalSection;
  const MCSection *ConstTextCoalSection;
  const MCSection *ConstDataCoalSection;
  const MCSection *ConstDataSection;
  const MCSection *DataCoalSection;
  const MCSection *FourByteConstantSection;
  const MCSection *EightByteConstantSection;
  const MCSection *SixteenByteConstantSection;
public:
  
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
  
  virtual const MCSection *
  getSectionForMergeableConstant(SectionKind Kind) const;
  
  /// shouldEmitUsedDirectiveFor - This hook allows targets to selectively
  /// decide not to emit the UsedDirective for some symbols in llvm.used.
  /// FIXME: REMOVE this (rdar://7071300)
  virtual bool shouldEmitUsedDirectiveFor(const GlobalValue *GV,
                                          Mangler *) const;
};



class TargetLoweringObjectFileCOFF : public TargetLoweringObjectFile {
public:
  virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
  
  virtual void getSectionFlagsAsString(SectionKind Kind,
                                       SmallVectorImpl<char> &Str) const;
  
  virtual const MCSection *
  SelectSectionForGlobal(const GlobalValue *GV, SectionInfo Kind,
                         Mangler *Mang, const TargetMachine &TM) const;
};

} // end namespace llvm

#endif

//===- MCSection.h - Machine Code Sections ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSection class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTION_H
#define LLVM_MC_MCSECTION_H

#include <string>
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {
  class MCContext;
  class MCAsmInfo;
  class raw_ostream;
  
  /// MCSection - Instances of this class represent a uniqued identifier for a
  /// section in the current translation unit.  The MCContext class uniques and
  /// creates these.
  class MCSection {
    MCSection(const MCSection&);      // DO NOT IMPLEMENT
    void operator=(const MCSection&); // DO NOT IMPLEMENT
  protected:
    MCSection(SectionKind K) : Kind(K) {}
    SectionKind Kind;
  public:
    virtual ~MCSection();

    SectionKind getKind() const { return Kind; }
    
    virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                      raw_ostream &OS) const = 0;

    /// isBaseAddressKnownZero - Return true if we know that this section will
    /// get a base address of zero.  In cases where we know that this is true we
    /// can emit section offsets as direct references to avoid a subtraction
    /// from the base of the section, saving a relocation.
    virtual bool isBaseAddressKnownZero() const {
      return false;
    }
  };

  class MCSectionCOFF : public MCSection {
    // The memory for this string is stored in the same MCContext as *this.
    StringRef Name;
    
    /// IsDirective - This is true if the section name is a directive, not
    /// something that should be printed with ".section".
    ///
    /// FIXME: This is a hack.  Switch to a semantic view of the section instead
    /// of a syntactic one.
    bool IsDirective;
    
    MCSectionCOFF(StringRef name, bool isDirective, SectionKind K)
      : MCSection(K), Name(name), IsDirective(isDirective) {
    }
  public:
    
    static MCSectionCOFF *Create(StringRef Name, bool IsDirective, 
                                 SectionKind K, MCContext &Ctx);

    StringRef getName() const { return Name; }
    bool isDirective() const { return IsDirective; }
    
    virtual void PrintSwitchToSection(const MCAsmInfo &MAI,
                                      raw_ostream &OS) const;
  };
  
} // end namespace llvm

#endif

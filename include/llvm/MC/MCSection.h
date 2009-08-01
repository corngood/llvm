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

// FIXME: HORRIBLE HACK: major layering violation to get an enum.
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
  class MCContext;
  
  /// MCSection - Instances of this class represent a uniqued identifier for a
  /// section in the current translation unit.  The MCContext class uniques and
  /// creates these.
  class MCSection {
    std::string Name;
    MCSection(const MCSection&);      // DO NOT IMPLEMENT
    void operator=(const MCSection&); // DO NOT IMPLEMENT
  protected:
    MCSection(const StringRef &Name, SectionKind K, MCContext &Ctx);
    SectionKind Kind;
  public:
    virtual ~MCSection();

    static MCSection *Create(const StringRef &Name, SectionKind K,
                             MCContext &Ctx);
    
    const std::string &getName() const { return Name; }
    SectionKind getKind() const { return Kind; }
  };

  
  typedef MCSection MCSectionELF;
  
} // end namespace llvm

#endif

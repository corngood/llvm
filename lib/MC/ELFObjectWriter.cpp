//===- lib/MC/ELFObjectWriter.cpp - ELF File Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/ELFObjectWriter.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetAsmBackend.h"

#include "../Target/X86/X86FixupKinds.h"

#include <vector>
using namespace llvm;

static unsigned GetType(const MCSymbolData &SD) {
  uint32_t Type = (SD.getFlags() & (0xf << ELF_STT_Shift)) >> ELF_STT_Shift;
  assert(Type == ELF::STT_NOTYPE || Type == ELF::STT_OBJECT ||
         Type == ELF::STT_FUNC || Type == ELF::STT_SECTION ||
         Type == ELF::STT_FILE || Type == ELF::STT_COMMON ||
         Type == ELF::STT_TLS);
  return Type;
}

static unsigned GetBinding(const MCSymbolData &SD) {
  uint32_t Binding = (SD.getFlags() & (0xf << ELF_STB_Shift)) >> ELF_STB_Shift;
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  return Binding;
}

static void SetBinding(MCSymbolData &SD, unsigned Binding) {
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  uint32_t OtherFlags = SD.getFlags() & ~(0xf << ELF_STB_Shift);
  SD.setFlags(OtherFlags | (Binding << ELF_STB_Shift));
}

static unsigned GetVisibility(MCSymbolData &SD) {
  unsigned Visibility =
    (SD.getFlags() & (0xf << ELF_STV_Shift)) >> ELF_STV_Shift;
  assert(Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_INTERNAL ||
         Visibility == ELF::STV_HIDDEN || Visibility == ELF::STV_PROTECTED);
  return Visibility;
}

static bool isFixupKindX86PCRel(unsigned Kind) {
  switch (Kind) {
  default:
    return false;
  case X86::reloc_pcrel_1byte:
  case X86::reloc_pcrel_4byte:
  case X86::reloc_riprel_4byte:
  case X86::reloc_riprel_4byte_movq_load:
    return true;
  }
}

static bool RelocNeedsGOT(unsigned Type) {
  switch (Type) {
  default:
    return false;
  case ELF::R_X86_64_GOT32:
  case ELF::R_X86_64_PLT32:
  case ELF::R_X86_64_GOTPCREL:
    return true;
  }
}

namespace {

  class ELFObjectWriterImpl {
    /*static bool isFixupKindX86RIPRel(unsigned Kind) {
      return Kind == X86::reloc_riprel_4byte ||
        Kind == X86::reloc_riprel_4byte_movq_load;
    }*/


    /// ELFSymbolData - Helper struct for containing some precomputed information
    /// on symbols.
    struct ELFSymbolData {
      MCSymbolData *SymbolData;
      uint64_t StringIndex;
      uint32_t SectionIndex;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        if (GetType(*SymbolData) == ELF::STT_FILE)
          return true;
        if (GetType(*RHS.SymbolData) == ELF::STT_FILE)
          return false;
        return SymbolData->getSymbol().getName() <
               RHS.SymbolData->getSymbol().getName();
      }
    };

    /// @name Relocation Data
    /// @{

    struct ELFRelocationEntry {
      // Make these big enough for both 32-bit and 64-bit
      uint64_t r_offset;
      int Index;
      unsigned Type;
      const MCSymbol *Symbol;
      uint64_t r_addend;

      // Support lexicographic sorting.
      bool operator<(const ELFRelocationEntry &RE) const {
        return RE.r_offset < r_offset;
      }
    };

    SmallPtrSet<const MCSymbol *, 16> UsedInReloc;

    llvm::DenseMap<const MCSectionData*,
                   std::vector<ELFRelocationEntry> > Relocations;
    DenseMap<const MCSection*, uint64_t> SectionStringTableIndex;

    /// @}
    /// @name Symbol Table Data
    /// @{

    SmallString<256> StringTable;
    std::vector<ELFSymbolData> LocalSymbolData;
    std::vector<ELFSymbolData> ExternalSymbolData;
    std::vector<ELFSymbolData> UndefinedSymbolData;

    /// @}

    int NumRegularSections;

    bool NeedsGOT;

    ELFObjectWriter *Writer;

    raw_ostream &OS;

    unsigned Is64Bit : 1;

    bool HasRelocationAddend;

    Triple::OSType OSType;

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;

    unsigned ShstrtabIndex;

  public:
    ELFObjectWriterImpl(ELFObjectWriter *_Writer, bool _Is64Bit,
                        bool _HasRelAddend, Triple::OSType _OSType)
      : NeedsGOT(false), Writer(_Writer), OS(Writer->getStream()),
        Is64Bit(_Is64Bit), HasRelocationAddend(_HasRelAddend),
        OSType(_OSType) {
    }

    void Write8(uint8_t Value) { Writer->Write8(Value); }
    void Write16(uint16_t Value) { Writer->Write16(Value); }
    void Write32(uint32_t Value) { Writer->Write32(Value); }
    //void Write64(uint64_t Value) { Writer->Write64(Value); }
    void WriteZeros(unsigned N) { Writer->WriteZeros(N); }
    //void WriteBytes(StringRef Str, unsigned ZeroFillSize = 0) {
    //  Writer->WriteBytes(Str, ZeroFillSize);
    //}

    void WriteWord(uint64_t W) {
      if (Is64Bit)
        Writer->Write64(W);
      else
        Writer->Write32(W);
    }

    void String8(char *buf, uint8_t Value) {
      buf[0] = Value;
    }

    void StringLE16(char *buf, uint16_t Value) {
      buf[0] = char(Value >> 0);
      buf[1] = char(Value >> 8);
    }

    void StringLE32(char *buf, uint32_t Value) {
      StringLE16(buf, uint16_t(Value >> 0));
      StringLE16(buf + 2, uint16_t(Value >> 16));
    }

    void StringLE64(char *buf, uint64_t Value) {
      StringLE32(buf, uint32_t(Value >> 0));
      StringLE32(buf + 4, uint32_t(Value >> 32));
    }

    void StringBE16(char *buf ,uint16_t Value) {
      buf[0] = char(Value >> 8);
      buf[1] = char(Value >> 0);
    }

    void StringBE32(char *buf, uint32_t Value) {
      StringBE16(buf, uint16_t(Value >> 16));
      StringBE16(buf + 2, uint16_t(Value >> 0));
    }

    void StringBE64(char *buf, uint64_t Value) {
      StringBE32(buf, uint32_t(Value >> 32));
      StringBE32(buf + 4, uint32_t(Value >> 0));
    }

    void String16(char *buf, uint16_t Value) {
      if (Writer->isLittleEndian())
        StringLE16(buf, Value);
      else
        StringBE16(buf, Value);
    }

    void String32(char *buf, uint32_t Value) {
      if (Writer->isLittleEndian())
        StringLE32(buf, Value);
      else
        StringBE32(buf, Value);
    }

    void String64(char *buf, uint64_t Value) {
      if (Writer->isLittleEndian())
        StringLE64(buf, Value);
      else
        StringBE64(buf, Value);
    }

    void WriteHeader(uint64_t SectionDataSize, unsigned NumberOfSections);

    void WriteSymbolEntry(MCDataFragment *F, uint64_t name, uint8_t info,
                          uint64_t value, uint64_t size,
                          uint8_t other, uint16_t shndx);

    void WriteSymbol(MCDataFragment *F, ELFSymbolData &MSD,
                     const MCAsmLayout &Layout);

    void WriteSymbolTable(MCDataFragment *F, const MCAssembler &Asm,
                          const MCAsmLayout &Layout,
                          unsigned NumRegularSections);

    void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                          const MCFragment *Fragment, const MCFixup &Fixup,
                          MCValue Target, uint64_t &FixedValue);

    uint64_t getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                         const MCSymbol *S);

    /// ComputeSymbolTable - Compute the symbol table data
    ///
    /// \param StringTable [out] - The string table data.
    /// \param StringIndexMap [out] - Map from symbol names to offsets in the
    /// string table.
    void ComputeSymbolTable(MCAssembler &Asm);

    void WriteRelocation(MCAssembler &Asm, MCAsmLayout &Layout,
                         const MCSectionData &SD);

    void WriteRelocations(MCAssembler &Asm, MCAsmLayout &Layout) {
      for (MCAssembler::const_iterator it = Asm.begin(),
             ie = Asm.end(); it != ie; ++it) {
        WriteRelocation(Asm, Layout, *it);
      }
    }

    void CreateMetadataSections(MCAssembler &Asm, MCAsmLayout &Layout);

    void ExecutePostLayoutBinding(MCAssembler &Asm) {
    }

    void WriteSecHdrEntry(uint32_t Name, uint32_t Type, uint64_t Flags,
                          uint64_t Address, uint64_t Offset,
                          uint64_t Size, uint32_t Link, uint32_t Info,
                          uint64_t Alignment, uint64_t EntrySize);

    void WriteRelocationsFragment(const MCAssembler &Asm, MCDataFragment *F,
                                  const MCSectionData *SD);

    bool IsFixupFullyResolved(const MCAssembler &Asm,
                              const MCValue Target,
                              bool IsPCRel,
                              const MCFragment *DF) const;

    void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout);
  };

}

// Emit the ELF header.
void ELFObjectWriterImpl::WriteHeader(uint64_t SectionDataSize,
                                      unsigned NumberOfSections) {
  // ELF Header
  // ----------
  //
  // Note
  // ----
  // emitWord method behaves differently for ELF32 and ELF64, writing
  // 4 bytes in the former and 8 in the latter.

  Write8(0x7f); // e_ident[EI_MAG0]
  Write8('E');  // e_ident[EI_MAG1]
  Write8('L');  // e_ident[EI_MAG2]
  Write8('F');  // e_ident[EI_MAG3]

  Write8(Is64Bit ? ELF::ELFCLASS64 : ELF::ELFCLASS32); // e_ident[EI_CLASS]

  // e_ident[EI_DATA]
  Write8(Writer->isLittleEndian() ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);

  Write8(ELF::EV_CURRENT);        // e_ident[EI_VERSION]
  // e_ident[EI_OSABI]
  switch (OSType) {
    case Triple::FreeBSD:  Write8(ELF::ELFOSABI_FREEBSD); break;
    case Triple::Linux:    Write8(ELF::ELFOSABI_LINUX); break;
    default:               Write8(ELF::ELFOSABI_NONE); break;
  }
  Write8(0);                  // e_ident[EI_ABIVERSION]

  WriteZeros(ELF::EI_NIDENT - ELF::EI_PAD);

  Write16(ELF::ET_REL);             // e_type

  // FIXME: Make this configurable
  Write16(Is64Bit ? ELF::EM_X86_64 : ELF::EM_386); // e_machine = target

  Write32(ELF::EV_CURRENT);         // e_version
  WriteWord(0);                    // e_entry, no entry point in .o file
  WriteWord(0);                    // e_phoff, no program header for .o
  WriteWord(SectionDataSize + (Is64Bit ? sizeof(ELF::Elf64_Ehdr) :
            sizeof(ELF::Elf32_Ehdr)));  // e_shoff = sec hdr table off in bytes

  // FIXME: Make this configurable.
  Write32(0);   // e_flags = whatever the target wants

  // e_ehsize = ELF header size
  Write16(Is64Bit ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr));

  Write16(0);                  // e_phentsize = prog header entry size
  Write16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  Write16(Is64Bit ? sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr));

  // e_shnum     = # of section header ents
  Write16(NumberOfSections);

  // e_shstrndx  = Section # of '.shstrtab'
  Write16(ShstrtabIndex);
}

void ELFObjectWriterImpl::WriteSymbolEntry(MCDataFragment *F, uint64_t name,
                                           uint8_t info, uint64_t value,
                                           uint64_t size, uint8_t other,
                                           uint16_t shndx) {
  if (Is64Bit) {
    char buf[8];

    String32(buf, name);
    F->getContents() += StringRef(buf, 4); // st_name

    String8(buf, info);
    F->getContents() += StringRef(buf, 1);  // st_info

    String8(buf, other);
    F->getContents() += StringRef(buf, 1); // st_other

    String16(buf, shndx);
    F->getContents() += StringRef(buf, 2); // st_shndx

    String64(buf, value);
    F->getContents() += StringRef(buf, 8); // st_value

    String64(buf, size);
    F->getContents() += StringRef(buf, 8);  // st_size
  } else {
    char buf[4];

    String32(buf, name);
    F->getContents() += StringRef(buf, 4);  // st_name

    String32(buf, value);
    F->getContents() += StringRef(buf, 4); // st_value

    String32(buf, size);
    F->getContents() += StringRef(buf, 4);  // st_size

    String8(buf, info);
    F->getContents() += StringRef(buf, 1);  // st_info

    String8(buf, other);
    F->getContents() += StringRef(buf, 1); // st_other

    String16(buf, shndx);
    F->getContents() += StringRef(buf, 2); // st_shndx
  }
}

static uint64_t SymbolValue(MCSymbolData &Data, const MCAsmLayout &Layout) {
  if (Data.isCommon() && Data.isExternal())
    return Data.getCommonAlignment();

  const MCSymbol &Symbol = Data.getSymbol();
  if (!Symbol.isInSection())
    return 0;

  if (!Data.isCommon() && !(Data.getFlags() & ELF_STB_Weak))
    if (MCFragment *FF = Data.getFragment())
      return Layout.getSymbolAddress(&Data) -
             Layout.getSectionAddress(FF->getParent());

  return 0;
}

static const MCSymbol &AliasedSymbol(const MCSymbol &Symbol) {
  const MCSymbol *S = &Symbol;
  while (S->isVariable()) {
    const MCExpr *Value = S->getVariableValue();
    assert (Value->getKind() == MCExpr::SymbolRef && "Unimplemented");
    const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr*>(Value);
    S = &Ref->getSymbol();
  }
  return *S;
}

void ELFObjectWriterImpl::WriteSymbol(MCDataFragment *F, ELFSymbolData &MSD,
                                      const MCAsmLayout &Layout) {
  MCSymbolData &OrigData = *MSD.SymbolData;
  MCSymbolData &Data =
    Layout.getAssembler().getSymbolData(AliasedSymbol(OrigData.getSymbol()));

  uint8_t Binding = GetBinding(OrigData);
  uint8_t Visibility = GetVisibility(OrigData);
  uint8_t Type = GetType(Data);

  uint8_t Info = (Binding << ELF_STB_Shift) | (Type << ELF_STT_Shift);
  uint8_t Other = Visibility;

  uint64_t Value = SymbolValue(Data, Layout);
  uint64_t Size = 0;
  const MCExpr *ESize;

  assert(!(Data.isCommon() && !Data.isExternal()));

  ESize = Data.getSize();
  if (Data.getSize()) {
    MCValue Res;
    if (ESize->getKind() == MCExpr::Binary) {
      const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(ESize);

      if (BE->EvaluateAsRelocatable(Res, &Layout)) {
        assert(!Res.getSymA() || !Res.getSymA()->getSymbol().isDefined());
        assert(!Res.getSymB() || !Res.getSymB()->getSymbol().isDefined());
        Size = Res.getConstant();
      }
    } else if (ESize->getKind() == MCExpr::Constant) {
      Size = static_cast<const MCConstantExpr *>(ESize)->getValue();
    } else {
      assert(0 && "Unsupported size expression");
    }
  }

  // Write out the symbol table entry
  WriteSymbolEntry(F, MSD.StringIndex, Info, Value,
                   Size, Other, MSD.SectionIndex);
}

void ELFObjectWriterImpl::WriteSymbolTable(MCDataFragment *F,
                                           const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           unsigned NumRegularSections) {
  // The string table must be emitted first because we need the index
  // into the string table for all the symbol names.
  assert(StringTable.size() && "Missing string table");

  // FIXME: Make sure the start of the symbol table is aligned.

  // The first entry is the undefined symbol entry.
  unsigned EntrySize = Is64Bit ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;
  F->getContents().append(EntrySize, '\x00');

  // Write the symbol table entries.
  LastLocalSymbolIndex = LocalSymbolData.size() + 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = LocalSymbolData[i];
    WriteSymbol(F, MSD, Layout);
  }

  // Write out a symbol table entry for each regular section.
  unsigned Index = 1;
  for (MCAssembler::const_iterator it = Asm.begin();
       Index <= NumRegularSections; ++it, ++Index) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    // Leave out relocations so we don't have indexes within
    // the relocations messed up
    if (Section.getType() == ELF::SHT_RELA || Section.getType() == ELF::SHT_REL)
      continue;
    WriteSymbolEntry(F, 0, ELF::STT_SECTION, 0, 0, ELF::STV_DEFAULT, Index);
    LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = ExternalSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    assert(((Data.getFlags() & ELF_STB_Global) ||
            (Data.getFlags() & ELF_STB_Weak)) &&
           "External symbol requires STB_GLOBAL or STB_WEAK flag");
    WriteSymbol(F, MSD, Layout);
    if (GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = UndefinedSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    WriteSymbol(F, MSD, Layout);
    if (GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }
}

static bool ShouldRelocOnSymbol(const MCSymbolData &SD,
                                const MCValue &Target,
                                const MCFragment &F) {
  const MCSymbol &Symbol = SD.getSymbol();
  if (Symbol.isUndefined())
    return true;

  const MCSectionELF &Section =
    static_cast<const MCSectionELF&>(Symbol.getSection());

  if (SD.isExternal())
    return true;

  MCSymbolRefExpr::VariantKind Kind = Target.getSymA()->getKind();
  const MCSectionELF &Sec2 =
    static_cast<const MCSectionELF&>(F.getParent()->getSection());

  if (&Sec2 != &Section &&
      (Kind == MCSymbolRefExpr::VK_PLT ||
       Kind == MCSymbolRefExpr::VK_GOTPCREL ||
       Kind == MCSymbolRefExpr::VK_GOTOFF))
    return true;

  if (Section.getFlags() & MCSectionELF::SHF_MERGE)
    return Target.getConstant() != 0;

  return false;
}

// FIXME: this is currently X86/X86_64 only
void ELFObjectWriterImpl::RecordRelocation(const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCFragment *Fragment,
                                           const MCFixup &Fixup,
                                           MCValue Target,
                                           uint64_t &FixedValue) {
  int64_t Addend = 0;
  int Index = 0;
  int64_t Value = Target.getConstant();
  const MCSymbol *Symbol = 0;

  bool IsPCRel = isFixupKindX86PCRel(Fixup.getKind());
  if (!Target.isAbsolute()) {
    Symbol = &AliasedSymbol(Target.getSymA()->getSymbol());
    MCSymbolData &SD = Asm.getSymbolData(*Symbol);
    MCFragment *F = SD.getFragment();

    if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
      const MCSymbol &SymbolB = RefB->getSymbol();
      MCSymbolData &SDB = Asm.getSymbolData(SymbolB);
      IsPCRel = true;
      MCSectionData *Sec = Fragment->getParent();

      // Offset of the symbol in the section
      int64_t a = Layout.getSymbolAddress(&SDB) - Layout.getSectionAddress(Sec);

      // Ofeset of the relocation in the section
      int64_t b = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
      Value += b - a;
    }

    // Check that this case has already been fully resolved before we get
    // here.
    if (Symbol->isDefined() && !SD.isExternal() &&
        IsPCRel &&
        &Fragment->getParent()->getSection() == &Symbol->getSection()) {
      llvm_unreachable("We don't need a relocation in this case.");
      return;
    }

    bool RelocOnSymbol = ShouldRelocOnSymbol(SD, Target, *Fragment);
    if (!RelocOnSymbol) {
      Index = F->getParent()->getOrdinal();

      MCSectionData *FSD = F->getParent();
      // Offset of the symbol in the section
      Value += Layout.getSymbolAddress(&SD) - Layout.getSectionAddress(FSD);
    } else {
      UsedInReloc.insert(Symbol);
      Index = -1;
    }
    Addend = Value;
    // Compensate for the addend on i386.
    if (Is64Bit)
      Value = 0;
  }

  FixedValue = Value;

  // determine the type of the relocation

  MCSymbolRefExpr::VariantKind Modifier = Target.getSymA()->getKind();
  unsigned Type;
  if (Is64Bit) {
    if (IsPCRel) {
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_X86_64_PC32;
        break;
      case MCSymbolRefExpr::VK_PLT:
        Type = ELF::R_X86_64_PLT32;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTPCREL:
        Type = ELF::R_X86_64_GOTPCREL;
        break;
      }
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
      case FK_Data_8: Type = ELF::R_X86_64_64; break;
      case X86::reloc_signed_4byte:
      case X86::reloc_pcrel_4byte:
        assert(isInt<32>(Target.getConstant()));
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_X86_64_32S;
          break;
        case MCSymbolRefExpr::VK_GOT:
          Type = ELF::R_X86_64_GOT32;
          break;
        case MCSymbolRefExpr::VK_GOTPCREL:
          Type = ELF::R_X86_64_GOTPCREL;
          break;
        }
        break;
      case FK_Data_4:
        Type = ELF::R_X86_64_32;
        break;
      case FK_Data_2: Type = ELF::R_X86_64_16; break;
      case X86::reloc_pcrel_1byte:
      case FK_Data_1: Type = ELF::R_X86_64_8; break;
      }
    }
  } else {
    if (IsPCRel) {
      switch (Modifier) {
      default:
        Type = ELF::R_386_PC32;
        //llvm_unreachable("Unimplemented");
        break;
      case MCSymbolRefExpr::VK_PLT:
        Type = ELF::R_386_PLT32;
        break;
      }
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");

      // FIXME: Should we avoid selecting reloc_signed_4byte in 32 bit mode
      // instead?
      case X86::reloc_signed_4byte:
      case X86::reloc_pcrel_4byte:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_GOTOFF:
          Type = ELF::R_386_GOTOFF;
          break;
        }
        break;
      case FK_Data_4:
        if (Symbol->getName() == "_GLOBAL_OFFSET_TABLE_")
          Type = ELF::R_386_GOTPC;
        else
          Type = ELF::R_386_32;
        break;
      case FK_Data_2: Type = ELF::R_386_16; break;
      case X86::reloc_pcrel_1byte:
      case FK_Data_1: Type = ELF::R_386_8; break;
      }
    }
  }

  if (RelocNeedsGOT(Type))
    NeedsGOT = true;

  ELFRelocationEntry ERE;

  ERE.Index = Index;
  ERE.Type = Type;
  ERE.Symbol = Symbol;

  ERE.r_offset = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();

  if (HasRelocationAddend)
    ERE.r_addend = Addend;
  else
    ERE.r_addend = 0; // Silence compiler warning.

  Relocations[Fragment->getParent()].push_back(ERE);
}

uint64_t
ELFObjectWriterImpl::getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                                 const MCSymbol *S) {
  MCSymbolData &SD = Asm.getSymbolData(*S);

  // Local symbol.
  if (!SD.isExternal() && !S->isUndefined())
    return SD.getIndex() + /* empty symbol */ 1;

  // External or undefined symbol.
  return SD.getIndex() + NumRegularSections + /* empty symbol */ 1;
}

static bool isInSymtab(const MCAssembler &Asm, const MCSymbolData &Data,
                       bool Used) {
  const MCSymbol &Symbol = Data.getSymbol();
  if (!Asm.isSymbolLinkerVisible(Symbol) && !Symbol.isUndefined())
    return false;

  if (!Used && Symbol.isTemporary())
    return false;

  return true;
}

static bool isLocal(const MCSymbolData &Data) {
  if (Data.isExternal())
    return false;

  const MCSymbol &Symbol = Data.getSymbol();
  if (Symbol.isUndefined() && !Symbol.isVariable())
    return false;

  return true;
}

void ELFObjectWriterImpl::ComputeSymbolTable(MCAssembler &Asm) {
  // FIXME: Is this the correct place to do this?
  if (NeedsGOT) {
    llvm::StringRef Name = "_GLOBAL_OFFSET_TABLE_";
    MCSymbol *Sym = Asm.getContext().GetOrCreateSymbol(Name);
    MCSymbolData &Data = Asm.getOrCreateSymbolData(*Sym);
    Data.setExternal(true);
  }

  // Build section lookup table.
  NumRegularSections = Asm.size();
  DenseMap<const MCSection*, uint32_t> SectionIndexMap;
  unsigned Index = 1;
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it, ++Index)
    SectionIndexMap[&it->getSection()] = Index;

  // Index 0 is always the empty string.
  StringMap<uint64_t> StringIndexMap;
  StringTable += '\x00';

  // Add the data for the symbols.
  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Symbol = it->getSymbol();

    if (!isInSymtab(Asm, *it, UsedInReloc.count(&Symbol)))
      continue;

    ELFSymbolData MSD;
    MSD.SymbolData = it;
    bool Local = isLocal(*it);

    bool Add = false;
    if (it->isCommon()) {
      assert(!Local);
      MSD.SectionIndex = ELF::SHN_COMMON;
      Add = true;
    } else if (Symbol.isAbsolute()) {
      MSD.SectionIndex = ELF::SHN_ABS;
      Add = true;
    } else if (Symbol.isVariable()) {
      const MCSymbol &RefSymbol = AliasedSymbol(Symbol);
      if (RefSymbol.isDefined()) {
        MSD.SectionIndex = SectionIndexMap.lookup(&RefSymbol.getSection());
        assert(MSD.SectionIndex && "Invalid section index!");
        Add = true;
      }
    } else if (Symbol.isUndefined()) {
      assert(!Local);
      MSD.SectionIndex = ELF::SHN_UNDEF;
      // FIXME: Undefined symbols are global, but this is the first place we
      // are able to set it.
      if (GetBinding(*it) == ELF::STB_LOCAL)
        SetBinding(*it, ELF::STB_GLOBAL);
      Add = true;
    } else {
      MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
      assert(MSD.SectionIndex && "Invalid section index!");
      Add = true;
    }

    if (Add) {
      uint64_t &Entry = StringIndexMap[Symbol.getName()];
      if (!Entry) {
        Entry = StringTable.size();
        StringTable += Symbol.getName();
        StringTable += '\x00';
      }
      MSD.StringIndex = Entry;
      if (MSD.SectionIndex == ELF::SHN_UNDEF)
        UndefinedSymbolData.push_back(MSD);
      else if (Local)
        LocalSymbolData.push_back(MSD);
      else
        ExternalSymbolData.push_back(MSD);
    }
  }

  // Symbols are required to be in lexicographic order.
  array_pod_sort(LocalSymbolData.begin(), LocalSymbolData.end());
  array_pod_sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  array_pod_sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices. Local symbols must come before all other
  // symbols with non-local bindings.
  Index = 0;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
    LocalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
    ExternalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
    UndefinedSymbolData[i].SymbolData->setIndex(Index++);
}

void ELFObjectWriterImpl::WriteRelocation(MCAssembler &Asm, MCAsmLayout &Layout,
                                          const MCSectionData &SD) {
  if (!Relocations[&SD].empty()) {
    MCContext &Ctx = Asm.getContext();
    const MCSection *RelaSection;
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    const StringRef SectionName = Section.getSectionName();
    std::string RelaSectionName = HasRelocationAddend ? ".rela" : ".rel";
    RelaSectionName += SectionName;

    unsigned EntrySize;
    if (HasRelocationAddend)
      EntrySize = Is64Bit ? sizeof(ELF::Elf64_Rela) : sizeof(ELF::Elf32_Rela);
    else
      EntrySize = Is64Bit ? sizeof(ELF::Elf64_Rel) : sizeof(ELF::Elf32_Rel);

    RelaSection = Ctx.getELFSection(RelaSectionName, HasRelocationAddend ?
                                    ELF::SHT_RELA : ELF::SHT_REL, 0,
                                    SectionKind::getReadOnly(),
                                    false, EntrySize);

    MCSectionData &RelaSD = Asm.getOrCreateSectionData(*RelaSection);
    RelaSD.setAlignment(Is64Bit ? 8 : 4);

    MCDataFragment *F = new MCDataFragment(&RelaSD);

    WriteRelocationsFragment(Asm, F, &SD);

    Asm.AddSectionToTheEnd(*Writer, RelaSD, Layout);
  }
}

void ELFObjectWriterImpl::WriteSecHdrEntry(uint32_t Name, uint32_t Type,
                                           uint64_t Flags, uint64_t Address,
                                           uint64_t Offset, uint64_t Size,
                                           uint32_t Link, uint32_t Info,
                                           uint64_t Alignment,
                                           uint64_t EntrySize) {
  Write32(Name);        // sh_name: index into string table
  Write32(Type);        // sh_type
  WriteWord(Flags);     // sh_flags
  WriteWord(Address);   // sh_addr
  WriteWord(Offset);    // sh_offset
  WriteWord(Size);      // sh_size
  Write32(Link);        // sh_link
  Write32(Info);        // sh_info
  WriteWord(Alignment); // sh_addralign
  WriteWord(EntrySize); // sh_entsize
}

void ELFObjectWriterImpl::WriteRelocationsFragment(const MCAssembler &Asm,
                                                   MCDataFragment *F,
                                                   const MCSectionData *SD) {
  std::vector<ELFRelocationEntry> &Relocs = Relocations[SD];
  // sort by the r_offset just like gnu as does
  array_pod_sort(Relocs.begin(), Relocs.end());

  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    ELFRelocationEntry entry = Relocs[e - i - 1];

    if (entry.Index < 0)
      entry.Index = getSymbolIndexInSymbolTable(Asm, entry.Symbol);
    else
      entry.Index += LocalSymbolData.size() + 1;
    if (Is64Bit) {
      char buf[8];

      String64(buf, entry.r_offset);
      F->getContents() += StringRef(buf, 8);

      struct ELF::Elf64_Rela ERE64;
      ERE64.setSymbolAndType(entry.Index, entry.Type);
      String64(buf, ERE64.r_info);
      F->getContents() += StringRef(buf, 8);

      if (HasRelocationAddend) {
        String64(buf, entry.r_addend);
        F->getContents() += StringRef(buf, 8);
      }
    } else {
      char buf[4];

      String32(buf, entry.r_offset);
      F->getContents() += StringRef(buf, 4);

      struct ELF::Elf32_Rela ERE32;
      ERE32.setSymbolAndType(entry.Index, entry.Type);
      String32(buf, ERE32.r_info);
      F->getContents() += StringRef(buf, 4);

      if (HasRelocationAddend) {
        String32(buf, entry.r_addend);
        F->getContents() += StringRef(buf, 4);
      }
    }
  }
}

void ELFObjectWriterImpl::CreateMetadataSections(MCAssembler &Asm,
                                                 MCAsmLayout &Layout) {
  MCContext &Ctx = Asm.getContext();
  MCDataFragment *F;

  const MCSection *SymtabSection;
  unsigned EntrySize = Is64Bit ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;

  unsigned NumRegularSections = Asm.size();

  // We construct .shstrtab, .symtab and .strtab in this order to match gnu as.
  const MCSection *ShstrtabSection;
  ShstrtabSection = Ctx.getELFSection(".shstrtab", ELF::SHT_STRTAB, 0,
                                      SectionKind::getReadOnly(), false);
  MCSectionData &ShstrtabSD = Asm.getOrCreateSectionData(*ShstrtabSection);
  ShstrtabSD.setAlignment(1);
  ShstrtabIndex = Asm.size();

  SymtabSection = Ctx.getELFSection(".symtab", ELF::SHT_SYMTAB, 0,
                                    SectionKind::getReadOnly(),
                                    false, EntrySize);
  MCSectionData &SymtabSD = Asm.getOrCreateSectionData(*SymtabSection);
  SymtabSD.setAlignment(Is64Bit ? 8 : 4);

  const MCSection *StrtabSection;
  StrtabSection = Ctx.getELFSection(".strtab", ELF::SHT_STRTAB, 0,
                                    SectionKind::getReadOnly(), false);
  MCSectionData &StrtabSD = Asm.getOrCreateSectionData(*StrtabSection);
  StrtabSD.setAlignment(1);
  StringTableIndex = Asm.size();

  WriteRelocations(Asm, Layout);

  // Symbol table
  F = new MCDataFragment(&SymtabSD);
  WriteSymbolTable(F, Asm, Layout, NumRegularSections);
  Asm.AddSectionToTheEnd(*Writer, SymtabSD, Layout);

  F = new MCDataFragment(&StrtabSD);
  F->getContents().append(StringTable.begin(), StringTable.end());
  Asm.AddSectionToTheEnd(*Writer, StrtabSD, Layout);

  F = new MCDataFragment(&ShstrtabSD);

  // Section header string table.
  //
  // The first entry of a string table holds a null character so skip
  // section 0.
  uint64_t Index = 1;
  F->getContents() += '\x00';

  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    // FIXME: We could merge suffixes like in .text and .rela.text.

    // Remember the index into the string table so we can write it
    // into the sh_name field of the section header table.
    SectionStringTableIndex[&it->getSection()] = Index;

    Index += Section.getSectionName().size() + 1;
    F->getContents() += Section.getSectionName();
    F->getContents() += '\x00';
  }

  Asm.AddSectionToTheEnd(*Writer, ShstrtabSD, Layout);
}

bool ELFObjectWriterImpl::IsFixupFullyResolved(const MCAssembler &Asm,
                                               const MCValue Target,
                                               bool IsPCRel,
                                               const MCFragment *DF) const {
  // If this is a PCrel relocation, find the section this fixup value is
  // relative to.
  const MCSection *BaseSection = 0;
  if (IsPCRel) {
    BaseSection = &DF->getParent()->getSection();
    assert(BaseSection);
  }

  const MCSection *SectionA = 0;
  const MCSymbol *SymbolA = 0;
  if (const MCSymbolRefExpr *A = Target.getSymA()) {
    SymbolA = &A->getSymbol();
    SectionA = &SymbolA->getSection();
  }

  const MCSection *SectionB = 0;
  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    SectionB = &B->getSymbol().getSection();
  }

  if (!BaseSection)
    return SectionA == SectionB;

  const MCSymbolData &DataA = Asm.getSymbolData(*SymbolA);
  if (DataA.isExternal())
    return false;

  return !SectionB && BaseSection == SectionA;
}

void ELFObjectWriterImpl::WriteObject(MCAssembler &Asm,
                                      const MCAsmLayout &Layout) {
  // Compute symbol table information.
  ComputeSymbolTable(Asm);

  CreateMetadataSections(const_cast<MCAssembler&>(Asm),
                         const_cast<MCAsmLayout&>(Layout));

  // Add 1 for the null section.
  unsigned NumSections = Asm.size() + 1;
  uint64_t NaturalAlignment = Is64Bit ? 8 : 4;
  uint64_t HeaderSize = Is64Bit ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr);
  uint64_t FileOff = HeaderSize;

  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;

    FileOff = RoundUpToAlignment(FileOff, SD.getAlignment());

    // Get the size of the section in the output file (including padding).
    uint64_t Size = Layout.getSectionFileSize(&SD);

    FileOff += Size;
  }

  FileOff = RoundUpToAlignment(FileOff, NaturalAlignment);

  // Write out the ELF header ...
  WriteHeader(FileOff - HeaderSize, NumSections);

  FileOff = HeaderSize;

  // ... then all of the sections ...
  DenseMap<const MCSection*, uint64_t> SectionOffsetMap;

  DenseMap<const MCSection*, uint32_t> SectionIndexMap;

  unsigned Index = 1;
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;

    uint64_t Padding = OffsetToAlignment(FileOff, SD.getAlignment());
    WriteZeros(Padding);
    FileOff += Padding;

    // Remember the offset into the file for this section.
    SectionOffsetMap[&it->getSection()] = FileOff;
    SectionIndexMap[&it->getSection()] = Index++;

    FileOff += Layout.getSectionFileSize(&SD);

    Asm.WriteSectionData(it, Layout, Writer);
  }

  uint64_t Padding = OffsetToAlignment(FileOff, NaturalAlignment);
  WriteZeros(Padding);
  FileOff += Padding;

  // ... and then the section header table.
  // Should we align the section header table?
  //
  // Null section first.
  WriteSecHdrEntry(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    uint64_t sh_link = 0;
    uint64_t sh_info = 0;

    switch(Section.getType()) {
    case ELF::SHT_DYNAMIC:
      sh_link = SectionStringTableIndex[&it->getSection()];
      sh_info = 0;
      break;

    case ELF::SHT_REL:
    case ELF::SHT_RELA: {
      const MCSection *SymtabSection;
      const MCSection *InfoSection;

      SymtabSection = Asm.getContext().getELFSection(".symtab", ELF::SHT_SYMTAB, 0,
                                                     SectionKind::getReadOnly(),
                                                     false);
      sh_link = SectionIndexMap[SymtabSection];

      // Remove ".rel" and ".rela" prefixes.
      unsigned SecNameLen = (Section.getType() == ELF::SHT_REL) ? 4 : 5;
      StringRef SectionName = Section.getSectionName().substr(SecNameLen);

      InfoSection = Asm.getContext().getELFSection(SectionName,
                                                   ELF::SHT_PROGBITS, 0,
                                                   SectionKind::getReadOnly(),
                                                   false);
      sh_info = SectionIndexMap[InfoSection];
      break;
    }

    case ELF::SHT_SYMTAB:
    case ELF::SHT_DYNSYM:
      sh_link = StringTableIndex;
      sh_info = LastLocalSymbolIndex;
      break;

    case ELF::SHT_PROGBITS:
    case ELF::SHT_STRTAB:
    case ELF::SHT_NOBITS:
    case ELF::SHT_NULL:
      // Nothing to do.
      break;

    case ELF::SHT_HASH:
    case ELF::SHT_GROUP:
    case ELF::SHT_SYMTAB_SHNDX:
    default:
      assert(0 && "FIXME: sh_type value not supported!");
      break;
    }

    WriteSecHdrEntry(SectionStringTableIndex[&it->getSection()],
                     Section.getType(), Section.getFlags(),
                     0,
                     SectionOffsetMap.lookup(&SD.getSection()),
                     Layout.getSectionSize(&SD), sh_link,
                     sh_info, SD.getAlignment(),
                     Section.getEntrySize());
  }
}

ELFObjectWriter::ELFObjectWriter(raw_ostream &OS,
                                 bool Is64Bit,
                                 Triple::OSType OSType,
                                 bool IsLittleEndian,
                                 bool HasRelocationAddend)
  : MCObjectWriter(OS, IsLittleEndian)
{
  Impl = new ELFObjectWriterImpl(this, Is64Bit, HasRelocationAddend, OSType);
}

ELFObjectWriter::~ELFObjectWriter() {
  delete (ELFObjectWriterImpl*) Impl;
}

void ELFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm) {
  ((ELFObjectWriterImpl*) Impl)->ExecutePostLayoutBinding(Asm);
}

void ELFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFragment *Fragment,
                                       const MCFixup &Fixup, MCValue Target,
                                       uint64_t &FixedValue) {
  ((ELFObjectWriterImpl*) Impl)->RecordRelocation(Asm, Layout, Fragment, Fixup,
                                                  Target, FixedValue);
}

bool ELFObjectWriter::IsFixupFullyResolved(const MCAssembler &Asm,
                                           const MCValue Target,
                                           bool IsPCRel,
                                           const MCFragment *DF) const {
  return ((ELFObjectWriterImpl*) Impl)->IsFixupFullyResolved(Asm, Target,
                                                             IsPCRel, DF);
}

void ELFObjectWriter::WriteObject(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) {
  ((ELFObjectWriterImpl*) Impl)->WriteObject(Asm, Layout);
}

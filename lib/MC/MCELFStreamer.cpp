//===- lib/MC/MCELFStreamer.cpp - ELF Object Output ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits ELF .o object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"

using namespace llvm;

namespace {

class MCELFStreamer : public MCObjectStreamer {
  void EmitInstToFragment(const MCInst &Inst);
  void EmitInstToData(const MCInst &Inst);
public:
  MCELFStreamer(MCContext &Context, TargetAsmBackend &TAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter)
    : MCObjectStreamer(Context, TAB, OS, Emitter, false) {}

  ~MCELFStreamer() {}

  /// @name MCStreamer Interface
  /// @{

  virtual void InitSections();
  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EmitCOFFSymbolStorageClass(int StorageClass) {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EmitCOFFSymbolType(int Type) {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EndCOFFSymbolDef() {
    assert(0 && "ELF doesn't support this directive");
  }

  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
     MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
     SD.setSize(Value);
  }

  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);
  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace);
  virtual void EmitGPRel32Value(const MCExpr *Value) {
    assert(0 && "ELF doesn't support this directive");
  }
  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);
  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);

  virtual void EmitFileDirective(StringRef Filename);
  virtual void EmitDwarfFileDirective(unsigned FileNo, StringRef Filename) {
    DEBUG(dbgs() << "FIXME: MCELFStreamer:EmitDwarfFileDirective not implemented\n");
  }

  virtual void EmitInstruction(const MCInst &Inst);
  virtual void Finish();

private:
  SmallPtrSet<MCSymbol *, 16> BindingExplicitlySet;
  /// @}
  void SetSection(StringRef Section, unsigned Type, unsigned Flags,
                  SectionKind Kind) {
    SwitchSection(getContext().getELFSection(Section, Type, Flags, Kind));
  }

  void SetSectionData() {
    SetSection(".data", MCSectionELF::SHT_PROGBITS,
               MCSectionELF::SHF_WRITE |MCSectionELF::SHF_ALLOC,
               SectionKind::getDataRel());
    EmitCodeAlignment(4, 0);
  }
  void SetSectionText() {
    SetSection(".text", MCSectionELF::SHT_PROGBITS,
               MCSectionELF::SHF_EXECINSTR |
               MCSectionELF::SHF_ALLOC, SectionKind::getText());
    EmitCodeAlignment(4, 0);
  }
  void SetSectionBss() {
    SetSection(".bss", MCSectionELF::SHT_NOBITS,
               MCSectionELF::SHF_WRITE |
               MCSectionELF::SHF_ALLOC, SectionKind::getBSS());
    EmitCodeAlignment(4, 0);
  }
};

} // end anonymous namespace.

void MCELFStreamer::InitSections() {
  // This emulates the same behavior of GNU as. This makes it easier
  // to compare the output as the major sections are in the same order.
  SetSectionText();
  SetSectionData();
  SetSectionBss();
  SetSectionText();
}

static bool isSymbolLinkerVisible(const MCAssembler &Asm,
                                  const MCSymbolData &Data) {
  const MCSymbol &Symbol = Data.getSymbol();
  // Absolute temporary labels are never visible.
  if (!Symbol.isInSection())
    return false;

  if (Asm.getBackend().doesSectionRequireSymbols(Symbol.getSection()))
    return true;

  if (!Data.isExternal())
    return false;

  return Asm.isSymbolLinkerVisible(Symbol);
}

void MCELFStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  Symbol->setSection(*CurSection);

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // We have to create a new fragment if this is an atom defining symbol,
  // fragments cannot span atoms.
  if (isSymbolLinkerVisible(getAssembler(), SD))
    new MCDataFragment(getCurrentSectionData());

  // FIXME: This is wasteful, we don't necessarily need to create a data
  // fragment. Instead, we should mark the symbol as pointing into the data
  // fragment if it exists, otherwise we should just queue the label and set its
  // fragment pointer when we emit the next fragment.
  MCDataFragment *F = getOrCreateDataFragment();

  assert(!SD.getFragment() && "Unexpected fragment on symbol data!");
  SD.setFragment(F);
  SD.setOffset(F->getContents().size());
}

void MCELFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  case MCAF_SubsectionsViaSymbols:
    getAssembler().setSubsectionsViaSymbols(true);
    return;
  }

  assert(0 && "invalid assembler flag!");
}

void MCELFStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  // FIXME: Lift context changes into super class.
  getAssembler().getOrCreateSymbolData(*Symbol);
  Symbol->setVariableValue(AddValueSymbols(Value));
}

static void SetBinding(MCSymbolData &SD, unsigned Binding) {
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  uint32_t OtherFlags = SD.getFlags() & ~(0xf << ELF_STB_Shift);
  SD.setFlags(OtherFlags | (Binding << ELF_STB_Shift));
}

static unsigned GetBinding(const MCSymbolData &SD) {
  uint32_t Binding = (SD.getFlags() & (0xf << ELF_STB_Shift)) >> ELF_STB_Shift;
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  return Binding;
}

static void SetType(MCSymbolData &SD, unsigned Type) {
  assert(Type == ELF::STT_NOTYPE || Type == ELF::STT_OBJECT ||
         Type == ELF::STT_FUNC || Type == ELF::STT_SECTION ||
         Type == ELF::STT_FILE || Type == ELF::STT_COMMON ||
         Type == ELF::STT_TLS);

  uint32_t OtherFlags = SD.getFlags() & ~(0xf << ELF_STT_Shift);
  SD.setFlags(OtherFlags | (Type << ELF_STT_Shift));
}

static void SetVisibility(MCSymbolData &SD, unsigned Visibility) {
  assert(Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_INTERNAL ||
         Visibility == ELF::STV_HIDDEN || Visibility == ELF::STV_PROTECTED);

  uint32_t OtherFlags = SD.getFlags() & ~(0xf << ELF_STV_Shift);
  SD.setFlags(OtherFlags | (Visibility << ELF_STV_Shift));
}

void MCELFStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
  // Indirect symbols are handled differently, to match how 'as' handles
  // them. This makes writing matching .o files easier.
  if (Attribute == MCSA_IndirectSymbol) {
    // Note that we intentionally cannot use the symbol data here; this is
    // important for matching the string table that 'as' generates.
    IndirectSymbolData ISD;
    ISD.Symbol = Symbol;
    ISD.SectionData = getCurrentSectionData();
    getAssembler().getIndirectSymbols().push_back(ISD);
    return;
  }

  // Adding a symbol attribute always introduces the symbol, note that an
  // important side effect of calling getOrCreateSymbolData here is to register
  // the symbol with the assembler.
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // The implementation of symbol attributes is designed to match 'as', but it
  // leaves much to desired. It doesn't really make sense to arbitrarily add and
  // remove flags, but 'as' allows this (in particular, see .desc).
  //
  // In the future it might be worth trying to make these operations more well
  // defined.
  switch (Attribute) {
  case MCSA_LazyReference:
  case MCSA_Reference:
  case MCSA_NoDeadStrip:
  case MCSA_PrivateExtern:
  case MCSA_WeakDefinition:
  case MCSA_WeakDefAutoPrivate:
  case MCSA_Invalid:
  case MCSA_ELF_TypeIndFunction:
  case MCSA_IndirectSymbol:
    assert(0 && "Invalid symbol attribute for ELF!");
    break;

  case MCSA_Global:
    SetBinding(SD, ELF::STB_GLOBAL);
    SD.setExternal(true);
    BindingExplicitlySet.insert(Symbol);
    break;

  case MCSA_WeakReference:
  case MCSA_Weak:
    SetBinding(SD, ELF::STB_WEAK);
    BindingExplicitlySet.insert(Symbol);
    break;

  case MCSA_Local:
    SetBinding(SD, ELF::STB_LOCAL);
    SD.setExternal(false);
    BindingExplicitlySet.insert(Symbol);
    break;

  case MCSA_ELF_TypeFunction:
    SetType(SD, ELF::STT_FUNC);
    break;

  case MCSA_ELF_TypeObject:
    SetType(SD, ELF::STT_OBJECT);
    break;

  case MCSA_ELF_TypeTLS:
    SetType(SD, ELF::STT_TLS);
    break;

  case MCSA_ELF_TypeCommon:
    SetType(SD, ELF::STT_COMMON);
    break;

  case MCSA_ELF_TypeNoType:
    SetType(SD, ELF::STT_NOTYPE);
    break;

  case MCSA_Protected:
    SetVisibility(SD, ELF::STV_PROTECTED);
    break;

  case MCSA_Hidden:
    SetVisibility(SD, ELF::STV_HIDDEN);
    break;

  case MCSA_Internal:
    SetVisibility(SD, ELF::STV_INTERNAL);
    break;
  }
}

void MCELFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  if (!BindingExplicitlySet.count(Symbol)) {
    SetBinding(SD, ELF::STB_GLOBAL);
    SD.setExternal(true);
  }

  if (GetBinding(SD) == ELF_STB_Local) {
    const MCSection *Section = getAssembler().getContext().getELFSection(".bss",
                                                                    MCSectionELF::SHT_NOBITS,
                                                                    MCSectionELF::SHF_WRITE |
                                                                    MCSectionELF::SHF_ALLOC,
                                                                    SectionKind::getBSS());

    MCSectionData &SectData = getAssembler().getOrCreateSectionData(*Section);
    new MCAlignFragment(ByteAlignment, 0, 1, ByteAlignment, &SectData);

    MCFragment *F = new MCFillFragment(0, 0, Size, &SectData);
    SD.setFragment(F);
    Symbol->setSection(*Section);

    // Update the maximum alignment of the section if necessary.
    if (ByteAlignment > SectData.getAlignment())
      SectData.setAlignment(ByteAlignment);
  } else {
    SD.setCommon(Size, ByteAlignment);
  }

  SD.setSize(MCConstantExpr::Create(Size, getContext()));
}

void MCELFStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  getOrCreateDataFragment()->getContents().append(Data.begin(), Data.end());
}

void MCELFStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                                unsigned AddrSpace) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  MCDataFragment *DF = getOrCreateDataFragment();

  // Avoid fixups when possible.
  int64_t AbsValue;
  if (AddValueSymbols(Value)->EvaluateAsAbsolute(AbsValue)) {
    // FIXME: Endianness assumption.
    for (unsigned i = 0; i != Size; ++i)
      DF->getContents().push_back(uint8_t(AbsValue >> (i * 8)));
  } else {
    DF->addFixup(MCFixup::Create(DF->getContents().size(), AddValueSymbols(Value),
                                 MCFixup::getKindForSize(Size)));
    DF->getContents().resize(DF->getContents().size() + Size, 0);
  }
}

void MCELFStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                           int64_t Value, unsigned ValueSize,
                                           unsigned MaxBytesToEmit) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  new MCAlignFragment(ByteAlignment, Value, ValueSize, MaxBytesToEmit,
                      getCurrentSectionData());

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void MCELFStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                        unsigned MaxBytesToEmit) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  MCAlignFragment *F = new MCAlignFragment(ByteAlignment, 0, 1, MaxBytesToEmit,
                                           getCurrentSectionData());
  F->setEmitNops(true);

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void MCELFStreamer::EmitValueToOffset(const MCExpr *Offset,
                                        unsigned char Value) {
  // TODO: This is exactly the same as MCMachOStreamer. Consider merging into
  // MCObjectStreamer.
  new MCOrgFragment(*Offset, Value, getCurrentSectionData());
}

// Add a symbol for the file name of this module. This is the second
// entry in the module's symbol table (the first being the null symbol).
void MCELFStreamer::EmitFileDirective(StringRef Filename) {
  MCSymbol *Symbol = getAssembler().getContext().GetOrCreateSymbol(Filename);
  Symbol->setSection(*CurSection);
  Symbol->setAbsolute();

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  SD.setFlags(ELF_STT_File | ELF_STB_Local | ELF_STV_Default);
}

void MCELFStreamer::EmitInstToFragment(const MCInst &Inst) {
  MCInstFragment *IF = new MCInstFragment(Inst, getCurrentSectionData());

  // Add the fixups and data.
  //
  // FIXME: Revisit this design decision when relaxation is done, we may be
  // able to get away with not storing any extra data in the MCInst.
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  IF->getCode() = Code;
  IF->getFixups() = Fixups;
}

void MCELFStreamer::EmitInstToData(const MCInst &Inst) {
  MCDataFragment *DF = getOrCreateDataFragment();

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  // Add the fixups and data.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    Fixups[i].setOffset(Fixups[i].getOffset() + DF->getContents().size());
    DF->addFixup(Fixups[i]);
  }
  DF->getContents().append(Code.begin(), Code.end());
}

void MCELFStreamer::EmitInstruction(const MCInst &Inst) {
  // Scan for values.
  for (unsigned i = 0; i != Inst.getNumOperands(); ++i)
    if (Inst.getOperand(i).isExpr())
      AddValueSymbols(Inst.getOperand(i).getExpr());

  getCurrentSectionData()->setHasInstructions(true);

  // If this instruction doesn't need relaxation, just emit it as data.
  if (!getAssembler().getBackend().MayNeedRelaxation(Inst)) {
    EmitInstToData(Inst);
    return;
  }

  // Otherwise, if we are relaxing everything, relax the instruction as much as
  // possible and emit it as data.
  if (getAssembler().getRelaxAll()) {
    MCInst Relaxed;
    getAssembler().getBackend().RelaxInstruction(Inst, Relaxed);
    while (getAssembler().getBackend().MayNeedRelaxation(Relaxed))
      getAssembler().getBackend().RelaxInstruction(Relaxed, Relaxed);
    EmitInstToData(Relaxed);
    return;
  }

  // Otherwise emit to a separate fragment.
  EmitInstToFragment(Inst);
}

void MCELFStreamer::Finish() {
  // FIXME: We create more atoms than it is necessary. Some relocations to
  // merge sections can be implemented with section address + offset,
  // figure out which ones and why.

  // First, scan the symbol table to build a lookup table from fragments to
  // defining symbols.
  DenseMap<const MCFragment*, MCSymbolData*> DefiningSymbolMap;
  for (MCAssembler::symbol_iterator it = getAssembler().symbol_begin(),
         ie = getAssembler().symbol_end(); it != ie; ++it) {
    if (isSymbolLinkerVisible(getAssembler(), *it) &&
        it->getFragment()) {
      // An atom defining symbol should never be internal to a fragment.
      assert(it->getOffset() == 0 && "Invalid offset in atom defining symbol!");
      DefiningSymbolMap[it->getFragment()] = it;
    }
  }

  // Set the fragment atom associations by tracking the last seen atom defining
  // symbol.
  for (MCAssembler::iterator it = getAssembler().begin(),
         ie = getAssembler().end(); it != ie; ++it) {
    MCSymbolData *CurrentAtom = 0;
    for (MCSectionData::iterator it2 = it->begin(),
           ie2 = it->end(); it2 != ie2; ++it2) {
      if (MCSymbolData *SD = DefiningSymbolMap.lookup(it2))
        CurrentAtom = SD;
      it2->setAtom(CurrentAtom);
    }
  }

  this->MCObjectStreamer::Finish();
}

MCStreamer *llvm::createELFStreamer(MCContext &Context, TargetAsmBackend &TAB,
                                      raw_ostream &OS, MCCodeEmitter *CE,
                                      bool RelaxAll) {
  MCELFStreamer *S = new MCELFStreamer(Context, TAB, OS, CE);
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}

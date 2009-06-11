//===-- ELFWriter.h - Target-independent ELF writer support -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ELFWriter class.
//
//===----------------------------------------------------------------------===//

#ifndef ELFWRITER_H
#define ELFWRITER_H

#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/OutputBuffer.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetELFWriterInfo.h"
#include "ELF.h"
#include <list>
#include <map>

namespace llvm {
  class ConstantStruct;
  class ELFCodeEmitter;
  class GlobalVariable;
  class Mangler;
  class MachineCodeEmitter;
  class raw_ostream;

  /// ELFWriter - This class implements the common target-independent code for
  /// writing ELF files.  Targets should derive a class from this to
  /// parameterize the output format.
  ///
  class ELFWriter : public MachineFunctionPass {
    friend class ELFCodeEmitter;
  public:
    static char ID;

    MachineCodeEmitter &getMachineCodeEmitter() const {
      return *(MachineCodeEmitter*)MCE;
    }

    ELFWriter(raw_ostream &O, TargetMachine &TM);
    ~ELFWriter();

    typedef std::vector<unsigned char> DataBuffer;

  protected:
    /// Output stream to send the resultant object file to.
    raw_ostream &O;

    /// Target machine description.
    TargetMachine &TM;

    /// Mang - The object used to perform name mangling for this module.
    Mangler *Mang;

    /// MCE - The MachineCodeEmitter object that we are exposing to emit machine
    /// code for functions to the .o file.
    ELFCodeEmitter *MCE;

    /// TAI - Target Asm Info, provide information about section names for
    /// globals and other target specific stuff.
    const TargetAsmInfo *TAI;

    //===------------------------------------------------------------------===//
    // Properties inferred automatically from the target machine.
    //===------------------------------------------------------------------===//

    /// is64Bit/isLittleEndian - This information is inferred from the target
    /// machine directly, indicating whether to emit a 32- or 64-bit ELF file.
    bool is64Bit, isLittleEndian;

    /// doInitialization - Emit the file header and all of the global variables
    /// for the module to the ELF file.
    bool doInitialization(Module &M);
    bool runOnMachineFunction(MachineFunction &MF);

    /// doFinalization - Now that the module has been completely processed, emit
    /// the ELF file to 'O'.
    bool doFinalization(Module &M);

  private:
    // The buffer we accumulate the file header into.  Note that this should be
    // changed into something much more efficient later (and the bitcode writer
    // as well!).
    DataBuffer FileHeader;

    /// ElfHdr - Hold information about the ELF Header
    ELFHeader *ElfHdr;

    /// SectionList - This is the list of sections that we have emitted to the
    /// file.  Once the file has been completely built, the section header table
    /// is constructed from this info.
    std::list<ELFSection> SectionList;
    unsigned NumSections;   // Always = SectionList.size()

    /// SectionLookup - This is a mapping from section name to section number in
    /// the SectionList.
    std::map<std::string, ELFSection*> SectionLookup;

    /// getSection - Return the section with the specified name, creating a new
    /// section if one does not already exist.
    ELFSection &getSection(const std::string &Name, unsigned Type, 
                           unsigned Flags = 0, unsigned Align = 0) {
      ELFSection *&SN = SectionLookup[Name];
      if (SN) return *SN;

      SectionList.push_back(Name);
      SN = &SectionList.back();
      SN->SectionIdx = NumSections++;
      SN->Type = Type;
      SN->Flags = Flags;
      SN->Link = ELFSection::SHN_UNDEF;
      SN->Align = Align;
      return *SN;
    }

    ELFSection &getTextSection() {
      return getSection(".text", ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_EXECINSTR | ELFSection::SHF_ALLOC);
    }

    ELFSection &getSymbolTableSection() {
      return getSection(".symtab", ELFSection::SHT_SYMTAB, 0);
    }

    ELFSection &getStringTableSection() {
      return getSection(".strtab", ELFSection::SHT_STRTAB, 0, 1);
    }

    ELFSection &getDataSection() {
      return getSection(".data", ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC);
    }

    ELFSection &getBSSSection() {
      return getSection(".bss", ELFSection::SHT_NOBITS,
                        ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC);
    }

    /// SymbolTable - This is the list of symbols we have emitted to the file.
    /// This actually gets rearranged before emission to the file (to put the
    /// local symbols first in the list).
    std::vector<ELFSym> SymbolTable;

    /// PendingSyms - This is a list of externally defined symbols that we have
    /// been asked to emit, but have not seen a reference to.  When a reference
    /// is seen, the symbol will move from this list to the SymbolTable.
    SetVector<GlobalValue*> PendingGlobals;

    // As we complete the ELF file, we need to update fields in the ELF header
    // (e.g. the location of the section table).  These members keep track of
    // the offset in ELFHeader of these various pieces to update and other
    // locations in the file.
    unsigned ELFHdr_e_shoff_Offset;     // e_shoff    in ELF header.
    unsigned ELFHdr_e_shstrndx_Offset;  // e_shstrndx in ELF header.
    unsigned ELFHdr_e_shnum_Offset;     // e_shnum    in ELF header.
  private:
    void EmitGlobal(GlobalVariable *GV);
    void EmitGlobalConstant(const Constant *C, OutputBuffer &GblCstTab);
    void EmitGlobalConstantStruct(const ConstantStruct *CVS,
                                  OutputBuffer &GblCstTab);
    void EmitRelocations();
    void EmitSectionHeader(OutputBuffer &TableOut, const ELFSection &Section);
    void EmitSectionTableStringTable();
    void EmitSymbol(OutputBuffer &SymTabOut, ELFSym &Sym);
    void EmitSymbolTable();
    void OutputSectionsAndSectionTable();
  };
}

#endif

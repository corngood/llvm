//===- llvm-readobj.cpp - Dump contents of an Object File -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like traditional Unix "readelf",
// except that it can handle any type of object file recognized by lib/Object.
//
// It makes use of the generic ObjectFile interface.
//
// Caution: This utility is new, experimental, unsupported, and incomplete.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input object>"), cl::init(""));

static void dumpSymbolHeader() {
  outs() << format("  %-32s", (const char*)"Name")
         << format("  %-4s", (const char*)"Type")
         << format("  %-16s", (const char*)"Address")
         << format("  %-16s", (const char*)"Size")
         << format("  %-16s", (const char*)"FileOffset")
         << format("  %-26s", (const char*)"Flags")
         << "\n";
}

static void dumpSectionHeader() {
  outs() << format("  %-24s", (const char*)"Name")
         << format("  %-16s", (const char*)"Address")
         << format("  %-16s", (const char*)"Size")
         << format("  %-8s", (const char*)"Align")
         << format("  %-26s", (const char*)"Flags")
         << "\n";
}

static const char *getTypeStr(SymbolRef::Type Type) {
  switch (Type) {
  case SymbolRef::ST_Unknown: return "?";
  case SymbolRef::ST_Data: return "DATA";
  case SymbolRef::ST_Debug: return "DBG";
  case SymbolRef::ST_File: return "FILE";
  case SymbolRef::ST_Function: return "FUNC";
  case SymbolRef::ST_Other: return "-";
  }
  return "INV";
}

static std::string getSymbolFlagStr(uint32_t Flags) {
  std::string result;
  if (Flags & SymbolRef::SF_Undefined)
    result += "undef,";
  if (Flags & SymbolRef::SF_Global)
    result += "global,";
  if (Flags & SymbolRef::SF_Weak)
    result += "weak,";
  if (Flags & SymbolRef::SF_Absolute)
    result += "absolute,";
  if (Flags & SymbolRef::SF_ThreadLocal)
    result += "threadlocal,";
  if (Flags & SymbolRef::SF_Common)
    result += "common,";
  if (Flags & SymbolRef::SF_FormatSpecific)
    result += "formatspecific,";

  // Remove trailing comma
  if (result.size() > 0) {
    result.erase(result.size() - 1);
  }
  return result;
}

static void checkError(error_code ec, const char *msg) {
  if (ec)
    report_fatal_error(std::string(msg) + ": " + ec.message());
}

static std::string getSectionFlagStr(const SectionRef &Section) {
  const struct {
    error_code (SectionRef::*MemF)(bool &) const;
    const char *FlagStr, *ErrorStr;
  } Work[] =
      {{ &SectionRef::isText, "text,", "Section.isText() failed" },
       { &SectionRef::isData, "data,", "Section.isData() failed" },
       { &SectionRef::isBSS, "bss,", "Section.isBSS() failed"  },
       { &SectionRef::isRequiredForExecution, "required,",
         "Section.isRequiredForExecution() failed" },
       { &SectionRef::isVirtual, "virtual,", "Section.isVirtual() failed" },
       { &SectionRef::isZeroInit, "zeroinit,", "Section.isZeroInit() failed" },
       { &SectionRef::isReadOnlyData, "rodata,",
         "Section.isReadOnlyData() failed" }};

  std::string result;
  for (uint32_t I = 0; I < sizeof(Work)/sizeof(*Work); ++I) {
    bool B;
    checkError((Section.*Work[I].MemF)(B), Work[I].ErrorStr);
    if (B)
      result += Work[I].FlagStr;
  }

  // Remove trailing comma
  if (result.size() > 0) {
    result.erase(result.size() - 1);
  }
  return result;
}

static void
dumpSymbol(const SymbolRef &Sym, const ObjectFile *obj, bool IsDynamic) {
  StringRef Name;
  SymbolRef::Type Type;
  uint32_t Flags;
  uint64_t Address;
  uint64_t Size;
  uint64_t FileOffset;
  checkError(Sym.getName(Name), "SymbolRef.getName() failed");
  checkError(Sym.getAddress(Address), "SymbolRef.getAddress() failed");
  checkError(Sym.getSize(Size), "SymbolRef.getSize() failed");
  checkError(Sym.getFileOffset(FileOffset),
             "SymbolRef.getFileOffset() failed");
  checkError(Sym.getType(Type), "SymbolRef.getType() failed");
  checkError(Sym.getFlags(Flags), "SymbolRef.getFlags() failed");
  std::string FullName = Name;

  // If this is a dynamic symbol from an ELF object, append
  // the symbol's version to the name.
  if (IsDynamic && obj->isELF()) {
    StringRef Version;
    bool IsDefault;
    GetELFSymbolVersion(obj, Sym, Version, IsDefault);
    if (!Version.empty()) {
      FullName += (IsDefault ? "@@" : "@");
      FullName += Version;
    }
  }

  // format() can't handle StringRefs
  outs() << format("  %-32s", FullName.c_str())
         << format("  %-4s", getTypeStr(Type))
         << format("  %16" PRIx64, Address)
         << format("  %16" PRIx64, Size)
         << format("  %16" PRIx64, FileOffset)
         << "  " << getSymbolFlagStr(Flags)
         << "\n";
}

// Iterate through the normal symbols in the ObjectFile
static void dumpSymbols(const ObjectFile *obj) {
  error_code ec;
  uint32_t count = 0;
  outs() << "Symbols:\n";
  dumpSymbolHeader();
  symbol_iterator it = obj->begin_symbols();
  symbol_iterator ie = obj->end_symbols();
  while (it != ie) {
    dumpSymbol(*it, obj, false);
    it.increment(ec);
    if (ec)
      report_fatal_error("Symbol iteration failed");
    ++count;
  }
  outs() << "  Total: " << count << "\n\n";
}

// Iterate through the dynamic symbols in the ObjectFile.
static void dumpDynamicSymbols(const ObjectFile *obj) {
  error_code ec;
  uint32_t count = 0;
  outs() << "Dynamic Symbols:\n";
  dumpSymbolHeader();
  symbol_iterator it = obj->begin_dynamic_symbols();
  symbol_iterator ie = obj->end_dynamic_symbols();
  while (it != ie) {
    dumpSymbol(*it, obj, true);
    it.increment(ec);
    if (ec)
      report_fatal_error("Symbol iteration failed");
    ++count;
  }
  outs() << "  Total: " << count << "\n\n";
}

static void dumpSection(const SectionRef &Section, const ObjectFile *obj) {
  StringRef Name;
  checkError(Section.getName(Name), "SectionRef::getName() failed");
  uint64_t Addr, Size, Align;
  checkError(Section.getAddress(Addr), "SectionRef::getAddress() failed");
  checkError(Section.getSize(Size), "SectionRef::getSize() failed");
  checkError(Section.getAlignment(Align), "SectionRef::getAlignment() failed");
  outs() << format("  %-24s", std::string(Name).c_str())
         << format("  %16" PRIx64, Addr)
         << format("  %16" PRIx64, Size)
         << format("  %8" PRIx64, Align)
         << "  " << getSectionFlagStr(Section)
         << "\n";
}

static void dumpLibrary(const LibraryRef &lib) {
  StringRef path;
  lib.getPath(path);
  outs() << "  " << path << "\n";
}

template<typename Iterator, typename Func>
static void dump(const ObjectFile *obj, Func f, Iterator begin, Iterator end,
                 const char *errStr) {
  error_code ec;
  uint32_t count = 0;
  Iterator it = begin, ie = end;
  while (it != ie) {
    f(*it, obj);
    it.increment(ec);
    if (ec)
      report_fatal_error(errStr);
    ++count;
  }
  outs() << "  Total: " << count << "\n\n";
}

// Iterate through needed libraries
static void dumpLibrariesNeeded(const ObjectFile *obj) {
  error_code ec;
  uint32_t count = 0;
  library_iterator it = obj->begin_libraries_needed();
  library_iterator ie = obj->end_libraries_needed();
  outs() << "Libraries needed:\n";
  while (it != ie) {
    dumpLibrary(*it);
    it.increment(ec);
    if (ec)
      report_fatal_error("Needed libraries iteration failed");
    ++count;
  }
  outs() << "  Total: " << count << "\n\n";
}

static void dumpHeaders(const ObjectFile *obj) {
  outs() << "File Format : " << obj->getFileFormatName() << "\n";
  outs() << "Arch        : "
         << Triple::getArchTypeName((llvm::Triple::ArchType)obj->getArch())
         << "\n";
  outs() << "Address Size: " << (8*obj->getBytesInAddress()) << " bits\n";
  outs() << "Load Name   : " << obj->getLoadName() << "\n";
  outs() << "\n";
}

int main(int argc, char** argv) {
  error_code ec;
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv,
                              "LLVM Object Reader\n");

  if (InputFilename.empty()) {
    errs() << "Please specify an input filename\n";
    return 1;
  }

  // Open the object file
  OwningPtr<MemoryBuffer> File;
  if (MemoryBuffer::getFile(InputFilename, File)) {
    errs() << InputFilename << ": Open failed\n";
    return 1;
  }

  ObjectFile *obj = ObjectFile::createObjectFile(File.take());
  if (!obj) {
    errs() << InputFilename << ": Object type not recognized\n";
  }

  dumpHeaders(obj);
  dumpSymbols(obj);
  dumpDynamicSymbols(obj);

  outs() << "Sections:\n";
  dumpSectionHeader();
  dump(obj, &dumpSection, obj->begin_sections(), obj->end_sections(),
       "Section iteration failed");

  dumpLibrariesNeeded(obj);
  return 0;
}


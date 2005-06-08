//===- ReaderWrappers.cpp - Parse bytecode from file or buffer  -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements loading and parsing a bytecode file and parsing a
// bytecode module from a given buffer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Bytecode/Reader.h"
#include "Reader.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/System/MappedFile.h"
#include <cerrno>
#include <iostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
// BytecodeFileReader - Read from an mmap'able file descriptor.
//

namespace {
  /// BytecodeFileReader - parses a bytecode file from a file
  ///
  class BytecodeFileReader : public BytecodeReader {
  private:
    sys::MappedFile mapFile;

    BytecodeFileReader(const BytecodeFileReader&); // Do not implement
    void operator=(const BytecodeFileReader &BFR); // Do not implement

  public:
    BytecodeFileReader(const std::string &Filename, llvm::BytecodeHandler* H=0);
  };
}

BytecodeFileReader::BytecodeFileReader(const std::string &Filename,
                                       llvm::BytecodeHandler* H )
  : BytecodeReader(H)
  , mapFile( sys::Path(Filename))
{
  mapFile.map();
  unsigned char* buffer = reinterpret_cast<unsigned char*>(mapFile.base());
  ParseBytecode(buffer, mapFile.size(), Filename);
}

//===----------------------------------------------------------------------===//
// BytecodeBufferReader - Read from a memory buffer
//

namespace {
  /// BytecodeBufferReader - parses a bytecode file from a buffer
  ///
  class BytecodeBufferReader : public BytecodeReader {
  private:
    const unsigned char *Buffer;
    bool MustDelete;

    BytecodeBufferReader(const BytecodeBufferReader&); // Do not implement
    void operator=(const BytecodeBufferReader &BFR);   // Do not implement

  public:
    BytecodeBufferReader(const unsigned char *Buf, unsigned Length,
                         const std::string &ModuleID,
                         llvm::BytecodeHandler* Handler = 0);
    ~BytecodeBufferReader();

  };
}

BytecodeBufferReader::BytecodeBufferReader(const unsigned char *Buf,
                                           unsigned Length,
                                           const std::string &ModuleID,
                                           llvm::BytecodeHandler* H )
  : BytecodeReader(H)
{
  // If not aligned, allocate a new buffer to hold the bytecode...
  const unsigned char *ParseBegin = 0;
  if (reinterpret_cast<uint64_t>(Buf) & 3) {
    Buffer = new unsigned char[Length+4];
    unsigned Offset = 4 - ((intptr_t)Buffer & 3);   // Make sure it's aligned
    ParseBegin = Buffer + Offset;
    memcpy((unsigned char*)ParseBegin, Buf, Length);    // Copy it over
    MustDelete = true;
  } else {
    // If we don't need to copy it over, just use the caller's copy
    ParseBegin = Buffer = Buf;
    MustDelete = false;
  }
  try {
    ParseBytecode(ParseBegin, Length, ModuleID);
  } catch (...) {
    if (MustDelete) delete [] Buffer;
    throw;
  }
}

BytecodeBufferReader::~BytecodeBufferReader() {
  if (MustDelete) delete [] Buffer;
}

//===----------------------------------------------------------------------===//
//  BytecodeStdinReader - Read bytecode from Standard Input
//

namespace {
  /// BytecodeStdinReader - parses a bytecode file from stdin
  ///
  class BytecodeStdinReader : public BytecodeReader {
  private:
    std::vector<unsigned char> FileData;
    unsigned char *FileBuf;

    BytecodeStdinReader(const BytecodeStdinReader&); // Do not implement
    void operator=(const BytecodeStdinReader &BFR);  // Do not implement

  public:
    BytecodeStdinReader( llvm::BytecodeHandler* H = 0 );
  };
}

BytecodeStdinReader::BytecodeStdinReader( BytecodeHandler* H )
  : BytecodeReader(H)
{
  char Buffer[4096*4];

  // Read in all of the data from stdin, we cannot mmap stdin...
  while (std::cin.good()) {
    std::cin.read(Buffer, 4096*4);
    int BlockSize = std::cin.gcount();
    if (0 >= BlockSize)
      break;
    FileData.insert(FileData.end(), Buffer, Buffer+BlockSize);
  }

  if (FileData.empty())
    throw std::string("Standard Input empty!");

  FileBuf = &FileData[0];
  ParseBytecode(FileBuf, FileData.size(), "<stdin>");
}

//===----------------------------------------------------------------------===//
// Wrapper functions
//===----------------------------------------------------------------------===//

/// getBytecodeBufferModuleProvider - lazy function-at-a-time loading from a
/// buffer
ModuleProvider*
llvm::getBytecodeBufferModuleProvider(const unsigned char *Buffer,
                                      unsigned Length,
                                      const std::string &ModuleID,
                                      BytecodeHandler* H ) {
  return new BytecodeBufferReader(Buffer, Length, ModuleID, H);
}

/// ParseBytecodeBuffer - Parse a given bytecode buffer
///
Module *llvm::ParseBytecodeBuffer(const unsigned char *Buffer, unsigned Length,
                                  const std::string &ModuleID,
                                  std::string *ErrorStr){
  try {
    std::auto_ptr<ModuleProvider>
      AMP(getBytecodeBufferModuleProvider(Buffer, Length, ModuleID));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

/// getBytecodeModuleProvider - lazy function-at-a-time loading from a file
///
ModuleProvider *llvm::getBytecodeModuleProvider(const std::string &Filename,
                                                BytecodeHandler* H) {
  if (Filename != std::string("-"))        // Read from a file...
    return new BytecodeFileReader(Filename,H);
  else                                     // Read from stdin
    return new BytecodeStdinReader(H);
}

/// ParseBytecodeFile - Parse the given bytecode file
///
Module *llvm::ParseBytecodeFile(const std::string &Filename,
                                std::string *ErrorStr) {
  try {
    std::auto_ptr<ModuleProvider> AMP(getBytecodeModuleProvider(Filename));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

// AnalyzeBytecodeFile - analyze one file
Module* llvm::AnalyzeBytecodeFile(
  const std::string &Filename,  ///< File to analyze
  BytecodeAnalysis& bca,        ///< Statistical output
  std::string *ErrorStr,        ///< Error output
  std::ostream* output          ///< Dump output
)
{
  try {
    BytecodeHandler* analyzerHandler =createBytecodeAnalyzerHandler(bca,output);
    std::auto_ptr<ModuleProvider> AMP(
      getBytecodeModuleProvider(Filename,analyzerHandler));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

// AnalyzeBytecodeBuffer - analyze a buffer
Module* llvm::AnalyzeBytecodeBuffer(
  const unsigned char* Buffer, ///< Pointer to start of bytecode buffer
  unsigned Length,             ///< Size of the bytecode buffer
  const std::string& ModuleID, ///< Identifier for the module
  BytecodeAnalysis& bca,       ///< The results of the analysis
  std::string* ErrorStr,       ///< Errors, if any.
  std::ostream* output         ///< Dump output, if any
)
{
  try {
    BytecodeHandler* hdlr = createBytecodeAnalyzerHandler(bca, output);
    std::auto_ptr<ModuleProvider>
      AMP(getBytecodeBufferModuleProvider(Buffer, Length, ModuleID, hdlr));
    return AMP->releaseModule();
  } catch (std::string &err) {
    if (ErrorStr) *ErrorStr = err;
    return 0;
  }
}

bool llvm::GetBytecodeDependentLibraries(const std::string &fname,
                                         Module::LibraryListType& deplibs) {
  try {
    std::auto_ptr<ModuleProvider> AMP( getBytecodeModuleProvider(fname));
    Module* M = AMP->releaseModule();

    deplibs = M->getLibraries();
    delete M;
    return true;
  } catch (...) {
    deplibs.clear();
    return false;
  }
}

static void getSymbols(Module*M, std::vector<std::string>& symbols) {
  // Loop over global variables
  for (Module::global_iterator GI = M->global_begin(), GE=M->global_end(); GI != GE; ++GI)
    if (!GI->isExternal() && !GI->hasInternalLinkage())
      if (!GI->getName().empty())
        symbols.push_back(GI->getName());

  // Loop over functions.
  for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI)
    if (!FI->isExternal() && !FI->hasInternalLinkage())
      if (!FI->getName().empty())
        symbols.push_back(FI->getName());
}

// Get just the externally visible defined symbols from the bytecode
bool llvm::GetBytecodeSymbols(const sys::Path& fName,
                              std::vector<std::string>& symbols) {
  try {
    std::auto_ptr<ModuleProvider> AMP(
        getBytecodeModuleProvider(fName.toString()));

    // Get the module from the provider
    Module* M = AMP->materializeModule();

    // Get the symbols
    getSymbols(M, symbols);

    // Done with the module
    return true;

  } catch (...) {
    return false;
  }
}

ModuleProvider*
llvm::GetBytecodeSymbols(const unsigned char*Buffer, unsigned Length,
                         const std::string& ModuleID,
                         std::vector<std::string>& symbols) {

  ModuleProvider* MP = 0;
  try {
    // Get the module provider
    MP = getBytecodeBufferModuleProvider(Buffer, Length, ModuleID);

    // Get the module from the provider
    Module* M = MP->materializeModule();

    // Get the symbols
    getSymbols(M, symbols);

    // Done with the module. Note that ModuleProvider will delete the
    // Module when it is deleted. Also note that its the caller's responsibility
    // to delete the ModuleProvider.
    return MP;

  } catch (...) {
    // We delete only the ModuleProvider here because its destructor will
    // also delete the Module (we used materializeModule not releaseModule).
    delete MP;
  }
  return 0;
}

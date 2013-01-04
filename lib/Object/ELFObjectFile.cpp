//===- ELFObjectFile.cpp - ELF object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the ELFObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {

using namespace object;

// Creates an in-memory object-file by default: createELFObjectFile(Buffer)
ObjectFile *ObjectFile::createELFObjectFile(MemoryBuffer *Object) {
  std::pair<unsigned char, unsigned char> Ident = getElfArchType(Object);
  error_code ec;

  std::size_t MaxAlignment =
    1ULL << CountTrailingZeros_64(uintptr_t(Object->getBufferStart()));

  if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2LSB)
    if (MaxAlignment >= 4)
      return new ELFObjectFile<support::little, 4, false>(Object, ec);
    else if (MaxAlignment >= 2)
      return new ELFObjectFile<support::little, 2, false>(Object, ec);
    else
      llvm_unreachable("Invalid alignment for ELF file!");
  else if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2MSB)
    if (MaxAlignment >= 4)
      return new ELFObjectFile<support::big, 4, false>(Object, ec);
    else if (MaxAlignment >= 2)
      return new ELFObjectFile<support::big, 2, false>(Object, ec);
    else
      llvm_unreachable("Invalid alignment for ELF file!");
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2MSB)
    if (MaxAlignment >= 8)
      return new ELFObjectFile<support::big, 8, true>(Object, ec);
    else if (MaxAlignment >= 2)
      return new ELFObjectFile<support::big, 2, true>(Object, ec);
    else
      llvm_unreachable("Invalid alignment for ELF file!");
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2LSB) {
    if (MaxAlignment >= 8)
      return new ELFObjectFile<support::little, 8, true>(Object, ec);
    else if (MaxAlignment >= 2)
      return new ELFObjectFile<support::little, 2, true>(Object, ec);
    else
      llvm_unreachable("Invalid alignment for ELF file!");
  }

  report_fatal_error("Buffer is not an ELF object file!");
}

} // end namespace llvm

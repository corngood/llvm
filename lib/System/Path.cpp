//===-- Path.cpp - Implement OS Path Concept --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Path concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/Path.h"
#include "llvm/Config/config.h"
#include <cassert>
#include <ostream>
using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

std::ostream& llvm::operator<<(std::ostream &strm, const sys::Path &aPath) {
  strm << aPath.toString();
  return strm;
}

Path
Path::GetLLVMConfigDir() {
  Path result;
#ifdef LLVM_ETCDIR
  if (result.set(LLVM_ETCDIR))
    return result;
#endif
  return GetLLVMDefaultConfigDir();
}

LLVMFileType
sys::IdentifyFileType(const char*magic, unsigned length) {
  assert(magic && "Invalid magic number string");
  assert(length >=4 && "Invalid magic number length");
  switch (magic[0]) {
    case 'B':
      if (magic[1] == 'C' && magic[2] == (char)0xC0 && magic[3] == (char)0xDE)
        return Bitcode_FileType;
      break;
    case '!':
      if (length >= 8)
        if (memcmp(magic,"!<arch>\n",8) == 0)
          return Archive_FileType;
      break;
      
    case '\177':
      if (magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F') {
        if (length >= 18 && magic[17] == 0)
          switch (magic[16]) {
            default: break;
            case 1: return ELF_Relocatable_FileType;
            case 2: return ELF_Executable_FileType;
            case 3: return ELF_SharedObject_FileType;
            case 4: return ELF_Core_FileType;
          }
      }
      break;

    case 0xCA:
      // This is complicated by an overlap with Java class files. 
      // See the Mach-O section in /usr/share/file/magic for details.
      if (magic[1] == char(0xFE) && magic[2] == char(0xBA) && 
          magic[3] == char(0xBE)) {
        return Mach_O_DynamicallyLinkedSharedLib_FileType;
        
        // FIXME: How does this work?
        if (length >= 14 && magic[13] == 0)
          switch (magic[12]) {
            default: break;
            case 1: return Mach_O_Object_FileType;
            case 2: return Mach_O_Executable_FileType;
            case 3: return Mach_O_FixedVirtualMemorySharedLib_FileType;
            case 4: return Mach_O_Core_FileType;
            case 5: return Mach_O_PreloadExectuable_FileType;
            case 6: return Mach_O_DynamicallyLinkedSharedLib_FileType;
            case 7: return Mach_O_DynamicLinker_FileType;
            case 8: return Mach_O_Bundle_FileType;
            case 9: return Mach_O_DynamicallyLinkedSharedLibStub_FileType;
          }
      }
      break;

    case 0xF0: // PowerPC Windows
    case 0x83: // Alpha 32-bit
    case 0x84: // Alpha 64-bit
    case 0x66: // MPS R4000 Windows
    case 0x50: // mc68K
    case 0x4c: // 80386 Windows
      if (magic[1] == 0x01)
        return COFF_FileType;

    case 0x90: // PA-RISC Windows
    case 0x68: // mc68K Windows
      if (magic[1] == 0x02)
        return COFF_FileType;
      break;

    default:
      break;
  }
  return Unknown_FileType;
}

bool
Path::isArchive() const {
  if (canRead())
    return hasMagicNumber("!<arch>\012");
  return false;
}

bool
Path::isDynamicLibrary() const {
  if (canRead()) {
    std::string Magic;
    if (getMagicNumber(Magic, 64))
      switch (IdentifyFileType(Magic.c_str(), Magic.length())) {
        default: return false;
        case Mach_O_FixedVirtualMemorySharedLib_FileType:
        case Mach_O_DynamicallyLinkedSharedLib_FileType:
        case Mach_O_DynamicallyLinkedSharedLibStub_FileType:
        case ELF_SharedObject_FileType:
        case COFF_FileType:  return true;
      }
  }
  return false;
}

Path
Path::FindLibrary(std::string& name) {
  std::vector<sys::Path> LibPaths;
  GetSystemLibraryPaths(LibPaths);
  for (unsigned i = 0; i < LibPaths.size(); ++i) {
    sys::Path FullPath(LibPaths[i]);
    FullPath.appendComponent("lib" + name + LTDL_SHLIB_EXT);
    if (FullPath.isDynamicLibrary())
      return FullPath;
    FullPath.eraseSuffix();
    FullPath.appendSuffix("a");
    if (FullPath.isArchive())
      return FullPath;
  }
  return sys::Path();
}

std::string Path::GetDLLSuffix() {
  return LTDL_SHLIB_EXT;
}

bool
Path::isBitcodeFile() const {
  std::string actualMagic;
  if (!getMagicNumber(actualMagic, 4))
    return false;
  return actualMagic == "BC\xC0\xDE";
}

bool Path::hasMagicNumber(const std::string &Magic) const {
  std::string actualMagic;
  if (getMagicNumber(actualMagic, Magic.size()))
    return Magic == actualMagic;
  return false;
}



// Include the truly platform-specific parts of this class.
#if defined(LLVM_ON_UNIX)
#include "Unix/Path.inc"
#endif
#if defined(LLVM_ON_WIN32)
#include "Win32/Path.inc"
#endif


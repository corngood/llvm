//===--- Triple.cpp - Target triple helper class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"

#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstring>
using namespace llvm;

//

const char *Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case InvalidArch: return "<invalid>";
  case UnknownArch: return "unknown";
    
  case alpha:   return "alpha";
  case arm:     return "arm";
  case cellspu: return "cellspu";
  case mips:    return "mips";
  case mipsel:  return "mipsel";
  case msp430:  return "msp430";
  case sparc:   return "sparc";
  case systemz: return "s390x";
  case thumb:   return "thumb";
  case x86:     return "i386";
  case x86_64:  return "x86_64";
  case ppc:     return "powerpc";
  case ppc64:   return "powerpc64";
  }

  return "<invalid>";
}

const char *Triple::getVendorTypeName(VendorType Kind) {
  switch (Kind) {
  case UnknownVendor: return "unknown";

  case Apple: return "apple";
  case PC: return "PC";
  }

  return "<invalid>";
}

const char *Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS: return "unknown";

  case AuroraUX: return "auroraux";
  case Cygwin: return "cygwin";
  case Darwin: return "darwin";
  case DragonFly: return "dragonfly";
  case FreeBSD: return "freebsd";
  case Linux: return "linux";
  case MinGW32: return "mingw32";
  case NetBSD: return "netbsd";
  case OpenBSD: return "openbsd";
  case Win32: return "win32";
  }

  return "<invalid>";
}

//

void Triple::Parse() const {
  assert(!isInitialized() && "Invalid parse call.");

  StringRef ArchName = getArchName();
  if (ArchName.size() == 4 && ArchName[0] == 'i' && 
      ArchName[2] == '8' && ArchName[3] == '6' && 
      ArchName[1] - '3' < 6) // i[3-9]86
    Arch = x86;
  else if (ArchName == "amd64" || ArchName == "x86_64")
    Arch = x86_64;
  else if (ArchName == "powerpc")
    Arch = ppc;
  else if (ArchName == "powerpc64")
    Arch = ppc64;
  else if (ArchName == "arm" ||
           ArchName.startswith("armv"))
    Arch = arm;
  else if (ArchName == "thumb" ||
           ArchName.startswith("thumbv"))
    Arch = thumb;
  else if (ArchName.startswith("alpha"))
    Arch = alpha;
  else if (ArchName == "spu" || ArchName == "cellspu")
    Arch = cellspu;
  else if (ArchName == "msp430")
    Arch = msp430;
  else if (ArchName == "mips" || ArchName == "mipsallegrex")
    Arch = mips;
  else if (ArchName == "mipsel" || ArchName == "mipsallegrexel" ||
           ArchName == "psp")
    Arch = mipsel;
  else if (ArchName == "sparc")
    Arch = sparc;
  else if (ArchName == "s390x")
    Arch = systemz;
  else
    Arch = UnknownArch;

  StringRef VendorName = getVendorName();
  if (VendorName == "apple")
    Vendor = Apple;
  else if (VendorName == "pc")
    Vendor = PC;
  else
    Vendor = UnknownVendor;

  StringRef OSName = getOSName();
  if (OSName.startswith("auroraux"))
    OS = AuroraUX;
  else if (OSName.startswith("cygwin"))
    OS = Cygwin;
  else if (OSName.startswith("darwin"))
    OS = Darwin;
  else if (OSName.startswith("dragonfly"))
    OS = DragonFly;
  else if (OSName.startswith("freebsd"))
    OS = FreeBSD;
  else if (OSName.startswith("linux"))
    OS = Linux;
  else if (OSName.startswith("mingw32"))
    OS = MinGW32;
  else if (OSName.startswith("netbsd"))
    OS = NetBSD;
  else if (OSName.startswith("openbsd"))
    OS = OpenBSD;
  else if (OSName.startswith("win32"))
    OS = Win32;
  else
    OS = UnknownOS;

  assert(isInitialized() && "Failed to initialize!");
}

StringRef Triple::getArchName() const {
  return StringRef(Data).split('-').first;           // Isolate first component
}

StringRef Triple::getVendorName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').first;                       // Isolate second component
}

StringRef Triple::getOSName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').first;                       // Isolate third component
}

StringRef Triple::getEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').second;                      // Strip third component
}

StringRef Triple::getOSAndEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').second;                      // Strip second component
}

void Triple::setTriple(const Twine &Str) {
  Data = Str.str();
  Arch = InvalidArch;
}

void Triple::setArch(ArchType Kind) {
  setArchName(getArchTypeName(Kind));
}

void Triple::setVendor(VendorType Kind) {
  setVendorName(getVendorTypeName(Kind));
}

void Triple::setOS(OSType Kind) {
  setOSName(getOSTypeName(Kind));
}

void Triple::setArchName(const StringRef &Str) {
  setTriple(Str + "-" + getVendorName() + "-" + getOSAndEnvironmentName());
}

void Triple::setVendorName(const StringRef &Str) {
  setTriple(getArchName() + "-" + Str + "-" + getOSAndEnvironmentName());
}

void Triple::setOSName(const StringRef &Str) {
  if (hasEnvironment())
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str +
              "-" + getEnvironmentName());
  else
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

void Triple::setEnvironmentName(const StringRef &Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + getOSName() + 
            "-" + Str);
}

void Triple::setOSAndEnvironmentName(const StringRef &Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

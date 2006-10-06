//===-- X86Subtarget.cpp - X86 Subtarget Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "X86Subtarget.h"
//#include "X86GenSubtarget.inc"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
using namespace llvm;

cl::opt<X86Subtarget::AsmWriterFlavorTy>
AsmWriterFlavor("x86-asm-syntax", cl::init(X86Subtarget::unset),
  cl::desc("Choose style of code to emit from X86 backend:"),
  cl::values(
    clEnumValN(X86Subtarget::att,   "att",   "  Emit AT&T-style assembly"),
    clEnumValN(X86Subtarget::intel, "intel", "  Emit Intel-style assembly"),
    clEnumValEnd));

/// GetCpuIDAndInfo - Execute the specified cpuid and return the 4 values in the
/// specified arguments.  If we can't run cpuid on the host, return true.
static inline bool GetCpuIDAndInfo(unsigned value, unsigned *rEAX, unsigned *rEBX,
                            unsigned *rECX, unsigned *rEDX) {
#if defined(__x86_64__)
  asm ("pushq\t%%rbx\n\t"
       "cpuid\n\t"
       "movl\t%%ebx, %%esi\n\t"
       "popq\t%%rbx"
       : "=a" (*rEAX),
         "=S" (*rEBX),
         "=c" (*rECX),
         "=d" (*rEDX)
       :  "a" (value));
  return false;
#elif defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
#if defined(__GNUC__)
  asm ("pushl\t%%ebx\n\t"
       "cpuid\n\t"
       "movl\t%%ebx, %%esi\n\t"
       "popl\t%%ebx"
       : "=a" (*rEAX),
         "=S" (*rEBX),
         "=c" (*rECX),
         "=d" (*rEDX)
       :  "a" (value));
  return false;
#elif defined(_MSC_VER)
  __asm {
    mov   eax,value
    cpuid
    mov   esi,rEAX
    mov   dword ptr [esi],eax
    mov   esi,rEBX
    mov   dword ptr [esi],ebx
    mov   esi,rECX
    mov   dword ptr [esi],ecx
    mov   esi,rEDX
    mov   dword ptr [esi],edx
  }
  return false;
#endif
#endif
  return true;
}

void X86Subtarget::DetectSubtargetFeatures() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  union {
    unsigned u[3];
    char     c[12];
  } text;

  if (GetCpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1))
    return;

  // FIXME: support for AMD family of processors.
  if (memcmp(text.c, "GenuineIntel", 12) == 0) {
    GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX);

    if ((EDX >> 23) & 0x1) X86SSELevel = MMX;
    if ((EDX >> 25) & 0x1) X86SSELevel = SSE1;
    if ((EDX >> 26) & 0x1) X86SSELevel = SSE2;
    if (ECX & 0x1)         X86SSELevel = SSE3;

    GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
    HasX86_64 = (EDX >> 29) & 0x1;
  }
}

X86Subtarget::X86Subtarget(const Module &M, const std::string &FS, bool is64Bit)
  : AsmFlavor(AsmWriterFlavor)
  , X86SSELevel(NoMMXSSE)
  , HasX86_64(false)
  , stackAlignment(8)
  // FIXME: this is a known good value for Yonah. How about others?
  , MinRepStrSizeThreshold(128)
  , Is64Bit(is64Bit)
  , TargetType(isELF) { // Default to ELF unless otherwise specified.

  // Determine default and user specified characteristics
  DetectSubtargetFeatures();
  if (Is64Bit && !HasX86_64) {
      std::cerr << "Warning: Generation of 64-bit code for a 32-bit processor "
                   "requested.\n";
      HasX86_64 = true;
  }

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    if (TT.find("cygwin") != std::string::npos ||
        TT.find("mingw")  != std::string::npos)
      TargetType = isCygwin;
    else if (TT.find("darwin") != std::string::npos)
      TargetType = isDarwin;
    else if (TT.find("win32") != std::string::npos)
      TargetType = isWindows;
  } else if (TT.empty()) {
#if defined(__CYGWIN__) || defined(__MINGW32__)
    TargetType = isCygwin;
#elif defined(__APPLE__)
    TargetType = isDarwin;
#elif defined(_WIN32)
    TargetType = isWindows;
#endif
  }

  // If the asm syntax hasn't been overridden on the command line, use whatever
  // the target wants.
  if (AsmFlavor == X86Subtarget::unset) {
    if (TargetType == isWindows) {
      AsmFlavor = X86Subtarget::intel;
    } else {
      AsmFlavor = X86Subtarget::att;
    }
  }

  if (TargetType == isDarwin || TargetType == isCygwin)
    stackAlignment = 16;
}

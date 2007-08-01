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
#include "X86GenSubtarget.inc"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

cl::opt<X86Subtarget::AsmWriterFlavorTy>
AsmWriterFlavor("x86-asm-syntax", cl::init(X86Subtarget::Unset),
  cl::desc("Choose style of code to emit from X86 backend:"),
  cl::values(
    clEnumValN(X86Subtarget::ATT,   "att",   "  Emit AT&T-style assembly"),
    clEnumValN(X86Subtarget::Intel, "intel", "  Emit Intel-style assembly"),
    clEnumValEnd));


/// True if accessing the GV requires an extra load. For Windows, dllimported
/// symbols are indirect, loading the value at address GV rather then the
/// value of GV itself. This means that the GlobalAddress must be in the base
/// or index register of the address, not the GV offset field.
bool X86Subtarget::GVRequiresExtraLoad(const GlobalValue* GV,
                                       const TargetMachine& TM,
                                       bool isDirectCall) const
{
  // FIXME: PIC
  if (TM.getRelocationModel() != Reloc::Static)
    if (isTargetDarwin()) {
      return (!isDirectCall &&
              (GV->hasWeakLinkage() || GV->hasLinkOnceLinkage() ||
               (GV->isDeclaration() && !GV->hasNotBeenReadFromBitcode())));
    } else if (TM.getRelocationModel() == Reloc::PIC_ && isPICStyleGOT()) {
      // Extra load is needed for all non-statics.
      return (!isDirectCall &&
              (GV->isDeclaration() || !GV->hasInternalLinkage()));
    } else if (isTargetCygMing() || isTargetWindows()) {
      return (GV->hasDLLImportLinkage());
    }
  
  return false;
}

/// GetCpuIDAndInfo - Execute the specified cpuid and return the 4 values in the
/// specified arguments.  If we can't run cpuid on the host, return true.
bool X86::GetCpuIDAndInfo(unsigned value, unsigned *rEAX, unsigned *rEBX,
                          unsigned *rECX, unsigned *rEDX) {
#if defined(__x86_64__)
  // gcc doesn't know cpuid would clobber ebx/rbx. Preseve it manually.
  asm ("movq\t%%rbx, %%rsi\n\t"
       "cpuid\n\t"
       "xchgq\t%%rbx, %%rsi\n\t"
       : "=a" (*rEAX),
         "=S" (*rEBX),
         "=c" (*rECX),
         "=d" (*rEDX)
       :  "a" (value));
  return false;
#elif defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
#if defined(__GNUC__)
  asm ("movl\t%%ebx, %%esi\n\t"
       "cpuid\n\t"
       "xchgl\t%%ebx, %%esi\n\t"
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

void X86Subtarget::AutoDetectSubtargetFeatures() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  union {
    unsigned u[3];
    char     c[12];
  } text;
  
  if (X86::GetCpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1))
    return;

  X86::GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX);
  
  if ((EDX >> 23) & 0x1) X86SSELevel = MMX;
  if ((EDX >> 25) & 0x1) X86SSELevel = SSE1;
  if ((EDX >> 26) & 0x1) X86SSELevel = SSE2;
  if (ECX & 0x1)         X86SSELevel = SSE3;
  if ((ECX >> 9)  & 0x1) X86SSELevel = SSSE3;

  if (memcmp(text.c, "GenuineIntel", 12) == 0 ||
      memcmp(text.c, "AuthenticAMD", 12) == 0) {
    X86::GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
    HasX86_64 = (EDX >> 29) & 0x1;
  }
}

static const char *GetCurrentX86CPU() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  if (X86::GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX))
    return "generic";
  unsigned Family  = (EAX >> 8) & 0xf; // Bits 8 - 11
  unsigned Model   = (EAX >> 4) & 0xf; // Bits 4 - 7
  X86::GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  bool Em64T = (EDX >> 29) & 0x1;

  union {
    unsigned u[3];
    char     c[12];
  } text;

  X86::GetCpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1);
  if (memcmp(text.c, "GenuineIntel", 12) == 0) {
    switch (Family) {
      case 3:
        return "i386";
      case 4:
        return "i486";
      case 5:
        switch (Model) {
        case 4:  return "pentium-mmx";
        default: return "pentium";
        }
      case 6:
        switch (Model) {
        case 1:  return "pentiumpro";
        case 3:
        case 5:
        case 6:  return "pentium2";
        case 7:
        case 8:
        case 10:
        case 11: return "pentium3";
        case 9:
        case 13: return "pentium-m";
        case 14: return "yonah";
        case 15: return "core2";
        default: return "i686";
        }
      case 15: {
        switch (Model) {
        case 3:  
        case 4:
          return (Em64T) ? "nocona" : "prescott";
        default:
          return (Em64T) ? "x86-64" : "pentium4";
        }
      }
        
    default:
      return "generic";
    }
  } else if (memcmp(text.c, "AuthenticAMD", 12) == 0) {
    // FIXME: this poorly matches the generated SubtargetFeatureKV table.  There
    // appears to be no way to generate the wide variety of AMD-specific targets
    // from the information returned from CPUID.
    switch (Family) {
      case 4:
        return "i486";
      case 5:
        switch (Model) {
        case 6:
        case 7:  return "k6";
        case 8:  return "k6-2";
        case 9:
        case 13: return "k6-3";
        default: return "pentium";
        }
      case 6:
        switch (Model) {
        case 4:  return "athlon-tbird";
        case 6:
        case 7:
        case 8:  return "athlon-mp";
        case 10: return "athlon-xp";
        default: return "athlon";
        }
      case 15:
        switch (Model) {
        case 1:  return "opteron";
        case 5:  return "athlon-fx"; // also opteron
        default: return "athlon64";
        }
    default:
      return "generic";
    }
  } else {
    return "generic";
  }
}

X86Subtarget::X86Subtarget(const Module &M, const std::string &FS, bool is64Bit)
  : AsmFlavor(AsmWriterFlavor)
  , PICStyle(PICStyle::None)
  , X86SSELevel(NoMMXSSE)
  , HasX86_64(false)
  , stackAlignment(8)
  // FIXME: this is a known good value for Yonah. How about others?
  , MinRepStrSizeThreshold(128)
  , Is64Bit(is64Bit)
  , HasLow4GUserAddress(true)
  , TargetType(isELF) { // Default to ELF unless otherwise specified.

  // Determine default and user specified characteristics
  if (!FS.empty()) {
    // If feature string is not empty, parse features string.
    std::string CPU = GetCurrentX86CPU();
    ParseSubtargetFeatures(FS, CPU);
    
    if (Is64Bit && !HasX86_64)
      cerr << "Warning: Generation of 64-bit code for a 32-bit processor "
           << "requested.\n";
    if (Is64Bit && X86SSELevel < SSE2)
      cerr << "Warning: 64-bit processors all have at least SSE2.\n";
  } else {
    // Otherwise, use CPUID to auto-detect feature set.
    AutoDetectSubtargetFeatures();
  }
    
  // If requesting codegen for X86-64, make sure that 64-bit and SSE2 features
  // are enabled.  These are available on all x86-64 CPUs.
  if (Is64Bit) {
    HasX86_64 = true;
    if (X86SSELevel < SSE2)
      X86SSELevel = SSE2;
  }

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    if (TT.find("cygwin") != std::string::npos)
      TargetType = isCygwin;
    else if (TT.find("mingw") != std::string::npos)
      TargetType = isMingw;
    else if (TT.find("darwin") != std::string::npos)
      TargetType = isDarwin;
    else if (TT.find("win32") != std::string::npos)
      TargetType = isWindows;
  } else if (TT.empty()) {
#if defined(__CYGWIN__)
    TargetType = isCygwin;
#elif defined(__MINGW32__)
    TargetType = isMingw;
#elif defined(__APPLE__)
    TargetType = isDarwin;
#elif defined(_WIN32)
    TargetType = isWindows;
#endif
  }

  // If the asm syntax hasn't been overridden on the command line, use whatever
  // the target wants.
  if (AsmFlavor == X86Subtarget::Unset) {
    if (TargetType == isWindows) {
      AsmFlavor = X86Subtarget::Intel;
    } else {
      AsmFlavor = X86Subtarget::ATT;
    }
  }

  if (TargetType == isDarwin && Is64Bit)
    HasLow4GUserAddress = false;

  if (TargetType == isDarwin ||
      TargetType == isCygwin ||
      TargetType == isMingw  ||
      (TargetType == isELF && Is64Bit))
    stackAlignment = 16;
}

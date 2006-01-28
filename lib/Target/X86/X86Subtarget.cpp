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
#include "llvm/Module.h"
#include "X86GenSubtarget.inc"
using namespace llvm;

// FIXME: temporary.
#include "llvm/Support/CommandLine.h"
namespace {
  cl::opt<bool> EnableSSE("enable-x86-sse", cl::Hidden,
                          cl::desc("Enable sse on X86"));
}

/// GetCpuIDAndInfo - Execute the specified cpuid and return the 4 values in the
/// specified arguments.  If we can't run cpuid on the host, return true.
static bool GetCpuIDAndInfo(unsigned value, unsigned *rEAX, unsigned *rEBX,
                            unsigned *rECX, unsigned *rEDX) {
#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
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

static const char *GetCurrentX86CPU() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  if (GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX))
    return "generic";
  unsigned Family  = (EAX & (0xffffffff >> (32 - 4)) << 8) >> 8; // Bits 8 - 11
  unsigned Model   = (EAX & (0xffffffff >> (32 - 4)) << 4) >> 4; // Bits 4 - 7
  GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  bool Em64T = EDX & (1 << 29);

  unsigned text[12];
  GetCpuIDAndInfo(0x80000002, text+0, text+1, text+2, text+3);
  GetCpuIDAndInfo(0x80000003, text+4, text+5, text+6, text+7);
  GetCpuIDAndInfo(0x80000004, text+8, text+9, text+10, text+11);
  char *t = reinterpret_cast<char *>(&text[0]);

  if (memcmp(t, "Intel", 5) == 0) {
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
  } else if (memcmp(t, "AMD", 3) == 0) {
    // FIXME: fill in remaining family/model combinations
    switch (Family) {
      case 15:
        return (Em64T) ? "athlon64" : "athlon";

    default:
      return "generic";
    }
  } else {
    return "generic";
  }
}

X86Subtarget::X86Subtarget(const Module &M, const std::string &FS) {
  stackAlignment = 8;
  indirectExternAndWeakGlobals = false;
  X86SSELevel = NoMMXSSE;
  X863DNowLevel = NoThreeDNow;
  Is64Bit = false;

  // Determine default and user specified characteristics
  std::string CPU = GetCurrentX86CPU();

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

  // FIXME: Just because the CPU supports 64-bit operation doesn't mean it isn't
  // currently running a 32-bit operating system.  This must be taken into account.
  // This hack will do for now, though obviously it breaks cross-compilation.
  if (sizeof(void *) == 4)
    Is64Bit = false;

  // Default to ELF unless otherwise specified.
  TargetType = isELF;
  
  // FIXME: Force these off until they work.  An llc-beta option should turn
  // them back on.
  if (!EnableSSE) {
    X86SSELevel = NoMMXSSE;
    X863DNowLevel = NoThreeDNow;
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

  if (TargetType == isDarwin) {
    stackAlignment = 16;
    indirectExternAndWeakGlobals = true;
  }
}

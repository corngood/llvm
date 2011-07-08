//===-- X86Subtarget.cpp - X86 Subtarget Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "subtarget"
#include "X86Subtarget.h"
#include "X86InstrInfo.h"
#include "llvm/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/SmallVector.h"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "X86GenSubtargetInfo.inc"

using namespace llvm;

#if defined(_MSC_VER)
#include <intrin.h>
#endif

/// ClassifyBlockAddressReference - Classify a blockaddress reference for the
/// current subtarget according to how we should reference it in a non-pcrel
/// context.
unsigned char X86Subtarget::
ClassifyBlockAddressReference() const {
  if (isPICStyleGOT())    // 32-bit ELF targets.
    return X86II::MO_GOTOFF;
  
  if (isPICStyleStubPIC())   // Darwin/32 in PIC mode.
    return X86II::MO_PIC_BASE_OFFSET;
  
  // Direct static reference to label.
  return X86II::MO_NO_FLAG;
}

/// ClassifyGlobalReference - Classify a global variable reference for the
/// current subtarget according to how we should reference it in a non-pcrel
/// context.
unsigned char X86Subtarget::
ClassifyGlobalReference(const GlobalValue *GV, const TargetMachine &TM) const {
  // DLLImport only exists on windows, it is implemented as a load from a
  // DLLIMPORT stub.
  if (GV->hasDLLImportLinkage())
    return X86II::MO_DLLIMPORT;

  // Determine whether this is a reference to a definition or a declaration.
  // Materializable GVs (in JIT lazy compilation mode) do not require an extra
  // load from stub.
  bool isDecl = GV->hasAvailableExternallyLinkage();
  if (GV->isDeclaration() && !GV->isMaterializable())
    isDecl = true;

  // X86-64 in PIC mode.
  if (isPICStyleRIPRel()) {
    // Large model never uses stubs.
    if (TM.getCodeModel() == CodeModel::Large)
      return X86II::MO_NO_FLAG;
      
    if (isTargetDarwin()) {
      // If symbol visibility is hidden, the extra load is not needed if
      // target is x86-64 or the symbol is definitely defined in the current
      // translation unit.
      if (GV->hasDefaultVisibility() &&
          (isDecl || GV->isWeakForLinker()))
        return X86II::MO_GOTPCREL;
    } else if (!isTargetWin64()) {
      assert(isTargetELF() && "Unknown rip-relative target");

      // Extra load is needed for all externally visible.
      if (!GV->hasLocalLinkage() && GV->hasDefaultVisibility())
        return X86II::MO_GOTPCREL;
    }

    return X86II::MO_NO_FLAG;
  }
  
  if (isPICStyleGOT()) {   // 32-bit ELF targets.
    // Extra load is needed for all externally visible.
    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return X86II::MO_GOTOFF;
    return X86II::MO_GOT;
  }
  
  if (isPICStyleStubPIC()) {  // Darwin/32 in PIC mode.
    // Determine whether we have a stub reference and/or whether the reference
    // is relative to the PIC base or not.
    
    // If this is a strong reference to a definition, it is definitely not
    // through a stub.
    if (!isDecl && !GV->isWeakForLinker())
      return X86II::MO_PIC_BASE_OFFSET;

    // Unless we have a symbol with hidden visibility, we have to go through a
    // normal $non_lazy_ptr stub because this symbol might be resolved late.
    if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_NONLAZY_PIC_BASE;
    
    // If symbol visibility is hidden, we have a stub for common symbol
    // references and external declarations.
    if (isDecl || GV->hasCommonLinkage()) {
      // Hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE;
    }
    
    // Otherwise, no stub.
    return X86II::MO_PIC_BASE_OFFSET;
  }
  
  if (isPICStyleStubNoDynamic()) {  // Darwin/32 in -mdynamic-no-pic mode.
    // Determine whether we have a stub reference.
    
    // If this is a strong reference to a definition, it is definitely not
    // through a stub.
    if (!isDecl && !GV->isWeakForLinker())
      return X86II::MO_NO_FLAG;
    
    // Unless we have a symbol with hidden visibility, we have to go through a
    // normal $non_lazy_ptr stub because this symbol might be resolved late.
    if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_NONLAZY;

    // Otherwise, no stub.
    return X86II::MO_NO_FLAG;
  }
  
  // Direct static reference to global.
  return X86II::MO_NO_FLAG;
}


/// getBZeroEntry - This function returns the name of a function which has an
/// interface like the non-standard bzero function, if such a function exists on
/// the current subtarget and it is considered prefereable over memset with zero
/// passed as the second argument. Otherwise it returns null.
const char *X86Subtarget::getBZeroEntry() const {
  // Darwin 10 has a __bzero entry point for this purpose.
  if (getTargetTriple().isMacOSX() &&
      !getTargetTriple().isMacOSXVersionLT(10, 6))
    return "__bzero";

  return 0;
}

/// IsLegalToCallImmediateAddr - Return true if the subtarget allows calls
/// to immediate address.
bool X86Subtarget::IsLegalToCallImmediateAddr(const TargetMachine &TM) const {
  if (In64BitMode)
    return false;
  return isTargetELF() || TM.getRelocationModel() == Reloc::Static;
}

/// getSpecialAddressLatency - For targets where it is beneficial to
/// backschedule instructions that compute addresses, return a value
/// indicating the number of scheduling cycles of backscheduling that
/// should be attempted.
unsigned X86Subtarget::getSpecialAddressLatency() const {
  // For x86 out-of-order targets, back-schedule address computations so
  // that loads and stores aren't blocked.
  // This value was chosen arbitrarily.
  return 200;
}

void X86Subtarget::AutoDetectSubtargetFeatures() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  union {
    unsigned u[3];
    char     c[12];
  } text;
  
  if (X86_MC::GetCpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1))
    return;

  X86_MC::GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX);
  
  if ((EDX >> 15) & 1) HasCMov = true;
  if ((EDX >> 23) & 1) X86SSELevel = MMX;
  if ((EDX >> 25) & 1) X86SSELevel = SSE1;
  if ((EDX >> 26) & 1) X86SSELevel = SSE2;
  if (ECX & 0x1)       X86SSELevel = SSE3;
  if ((ECX >> 9)  & 1) X86SSELevel = SSSE3;
  if ((ECX >> 19) & 1) X86SSELevel = SSE41;
  if ((ECX >> 20) & 1) X86SSELevel = SSE42;
  // FIXME: AVX codegen support is not ready.
  //if ((ECX >> 28) & 1) { HasAVX = true; X86SSELevel = NoMMXSSE; }

  bool IsIntel = memcmp(text.c, "GenuineIntel", 12) == 0;
  bool IsAMD   = !IsIntel && memcmp(text.c, "AuthenticAMD", 12) == 0;

  HasCLMUL = IsIntel && ((ECX >> 1) & 0x1);
  HasFMA3  = IsIntel && ((ECX >> 12) & 0x1);
  HasPOPCNT = IsIntel && ((ECX >> 23) & 0x1);
  HasAES   = IsIntel && ((ECX >> 25) & 0x1);

  if (IsIntel || IsAMD) {
    // Determine if bit test memory instructions are slow.
    unsigned Family = 0;
    unsigned Model  = 0;
    X86_MC::DetectFamilyModel(EAX, Family, Model);
    IsBTMemSlow = IsAMD || (Family == 6 && Model >= 13);
    // If it's Nehalem, unaligned memory access is fast.
    if (Family == 15 && Model == 26)
      IsUAMemFast = true;

    X86_MC::GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
    HasX86_64 = (EDX >> 29) & 0x1;
    HasSSE4A = IsAMD && ((ECX >> 6) & 0x1);
    HasFMA4 = IsAMD && ((ECX >> 16) & 0x1);
  }
}

X86Subtarget::X86Subtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS, 
                           unsigned StackAlignOverride, bool is64Bit)
  : X86GenSubtargetInfo(TT, CPU, FS)
  , PICStyle(PICStyles::None)
  , X86SSELevel(NoMMXSSE)
  , X863DNowLevel(NoThreeDNow)
  , HasCMov(false)
  , HasX86_64(false)
  , HasPOPCNT(false)
  , HasSSE4A(false)
  , HasAVX(false)
  , HasAES(false)
  , HasCLMUL(false)
  , HasFMA3(false)
  , HasFMA4(false)
  , IsBTMemSlow(false)
  , IsUAMemFast(false)
  , HasVectorUAMem(false)
  , stackAlignment(8)
  // FIXME: this is a known good value for Yonah. How about others?
  , MaxInlineSizeThreshold(128)
  , TargetTriple(TT)
  , In64BitMode(is64Bit) {
  // Determine default and user specified characteristics
  if (!FS.empty() || !CPU.empty()) {
    std::string CPUName = CPU;
    if (CPUName.empty()) {
#if defined (__x86_64__) || defined(__i386__)
      CPUName = sys::getHostCPUName();
#else
      CPUName = "generic";
#endif
    }

    // If feature string is not empty, parse features string.
    ParseSubtargetFeatures(CPUName, FS);

    // All X86-64 CPUs also have SSE2, however user might request no SSE via 
    // -mattr, so don't force SSELevel here.
    if (HasAVX)
      X86SSELevel = NoMMXSSE;
  } else {
    // Otherwise, use CPUID to auto-detect feature set.
    AutoDetectSubtargetFeatures();

    // Make sure 64-bit features are available in 64-bit mode.
    if (In64BitMode) {
      HasX86_64 = true;
      HasCMov = true;

      if (!HasAVX && X86SSELevel < SSE2)
        X86SSELevel = SSE2;
    }
  }
    
  DEBUG(dbgs() << "Subtarget features: SSELevel " << X86SSELevel
               << ", 3DNowLevel " << X863DNowLevel
               << ", 64bit " << HasX86_64 << "\n");
  assert((!In64BitMode || HasX86_64) &&
         "64-bit code requested on a subtarget that doesn't support it!");

  // Stack alignment is 16 bytes on Darwin, FreeBSD, Linux and Solaris (both
  // 32 and 64 bit) and for all 64-bit targets.
  if (StackAlignOverride)
    stackAlignment = StackAlignOverride;
  else if (isTargetDarwin() || isTargetFreeBSD() || isTargetLinux() ||
           isTargetSolaris() || In64BitMode)
    stackAlignment = 16;
}

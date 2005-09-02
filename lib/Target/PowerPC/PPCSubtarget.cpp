//===- PowerPCSubtarget.cpp - PPC Subtarget Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPC specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "PowerPCSubtarget.h"
#include "PowerPC.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/SubtargetFeature.h"

using namespace llvm;
PPCTargetEnum llvm::PPCTarget = TargetDefault;

namespace llvm {
  cl::opt<PPCTargetEnum, true>
  PPCTargetArg(cl::desc("Force generation of code for a specific PPC target:"),
               cl::values(
                          clEnumValN(TargetAIX,  "aix", "  Enable AIX codegen"),
                          clEnumValN(TargetDarwin,"darwin",
                                     "  Enable Darwin codegen"),
                          clEnumValEnd),
               cl::location(PPCTarget), cl::init(TargetDefault));
}

enum PowerPCFeature {
  PowerPCFeature64Bit   = 1 << 0,
  PowerPCFeatureAltivec = 1 << 1,
  PowerPCFeatureFSqrt   = 1 << 2,
  PowerPCFeatureGPUL    = 1 << 3,
};

/// Sorted (by key) array of values for CPU subtype.
static const SubtargetFeatureKV PowerPCSubTypeKV[] = {
  { "601"    , 0 },
  { "602"    , 0 },
  { "603"    , 0 },
  { "603e"   , 0 },
  { "603ev"  , 0 },
  { "604"    , 0 },
  { "604e"   , 0 },
  { "620"    , 0 },
  { "7400"   , PowerPCFeatureAltivec },
  { "7450"   , PowerPCFeatureAltivec },
  { "750"    , 0 },
  { "970"    , PowerPCFeature64Bit | PowerPCFeatureAltivec |
               PowerPCFeatureFSqrt | PowerPCFeatureGPUL },
  { "g3"     , 0 },
  { "g4"     , PowerPCFeatureAltivec },
  { "g4+"    , PowerPCFeatureAltivec },
  { "g5"     , PowerPCFeature64Bit | PowerPCFeatureAltivec |
               PowerPCFeatureFSqrt | PowerPCFeatureGPUL },
  { "generic", 0 }
};
/// Length of PowerPCSubTypeKV.
static const unsigned PowerPCSubTypeKVSize = sizeof(PowerPCSubTypeKV)
                                             / sizeof(SubtargetFeatureKV);

/// Sorted (by key) array of values for CPU features.
static SubtargetFeatureKV PowerPCFeatureKV[] = {
  { "64bit"  , PowerPCFeature64Bit   },
  { "altivec", PowerPCFeatureAltivec },
  { "fsqrt"  , PowerPCFeatureFSqrt },
  { "gpul"   , PowerPCFeatureGPUL    }
 };
/// Length of PowerPCFeatureKV.
static const unsigned PowerPCFeatureKVSize = sizeof(PowerPCFeatureKV)
                                          / sizeof(SubtargetFeatureKV);


#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#include <mach/machine.h>

/// GetCurrentPowerPCFeatures - Returns the current CPUs features.
static const char *GetCurrentPowerPCCPU() {
  host_basic_info_data_t hostInfo;
  mach_msg_type_number_t infoCount;

  infoCount = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&hostInfo, 
            &infoCount);
            
  if (hostInfo.cpu_type != CPU_TYPE_POWERPC) return "generic";

  switch(hostInfo.cpu_subtype) {
  case CPU_SUBTYPE_POWERPC_601:   return "601";
  case CPU_SUBTYPE_POWERPC_602:   return "602";
  case CPU_SUBTYPE_POWERPC_603:   return "603";
  case CPU_SUBTYPE_POWERPC_603e:  return "603e";
  case CPU_SUBTYPE_POWERPC_603ev: return "603ev";
  case CPU_SUBTYPE_POWERPC_604:   return "604";
  case CPU_SUBTYPE_POWERPC_604e:  return "604e";
  case CPU_SUBTYPE_POWERPC_620:   return "620";
  case CPU_SUBTYPE_POWERPC_750:   return "750";
  case CPU_SUBTYPE_POWERPC_7400:  return "7400";
  case CPU_SUBTYPE_POWERPC_7450:  return "7450";
  case CPU_SUBTYPE_POWERPC_970:   return "970";
  default: ;
  }
  
  return "generic";
}
#endif

PPCSubtarget::PPCSubtarget(const Module &M, const std::string &FS)
  : StackAlignment(16), IsGigaProcessor(false), IsAIX(false), IsDarwin(false) {

  // Determine default and user specified characteristics
  std::string CPU;
#if defined(__APPLE__)
  CPU = GetCurrentPowerPCCPU();
#endif
  uint32_t Bits =
  SubtargetFeatures::Parse(FS, CPU,
                           PowerPCSubTypeKV, PowerPCSubTypeKVSize,
                           PowerPCFeatureKV, PowerPCFeatureKVSize);
  IsGigaProcessor = (Bits & PowerPCFeatureGPUL ) != 0;
  HasFSQRT        = (Bits & PowerPCFeatureFSqrt) != 0;

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    IsDarwin = TT.find("darwin") != std::string::npos;
  } else if (TT.empty()) {
#if defined(_POWER)
    IsAIX = true;
#elif defined(__APPLE__)
    IsDarwin = true;
#endif
  }
}

//===-- TargetLibraryInfo.cpp - Runtime library information ----------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetLibraryInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

// Register the default implementation.
INITIALIZE_PASS(TargetLibraryInfo, "targetlibinfo",
                "Target Library Information", false, true)
char TargetLibraryInfo::ID = 0;

/// initialize - Initialize the set of available library functions based on the
/// specified target triple.  This should be carefully written so that a missing
/// target triple gets a sane set of defaults.
static void initialize(TargetLibraryInfo &TLI, const Triple &T) {
  initializeTargetLibraryInfoPass(*PassRegistry::getPassRegistry());

  
  // memset_pattern16 is only available on iOS 3.0 and Mac OS/X 10.5 and later.
  if (T.getOS() != Triple::Darwin || T.getDarwinMajorNumber() < 9)
    TLI.setUnavailable(LibFunc::memset_pattern16);
  
}


TargetLibraryInfo::TargetLibraryInfo() : ImmutablePass(ID) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));

  initialize(*this, Triple());
}

TargetLibraryInfo::TargetLibraryInfo(const Triple &T) : ImmutablePass(ID) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));
  
  initialize(*this, T);
}

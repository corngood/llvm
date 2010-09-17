//===- llvm/System/FEnv.h - Host floating-point exceptions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides an operating system independent interface to
// floating-point exception interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_FENV_H
#define LLVM_SYSTEM_FENV_H

#include "llvm/Config/config.h"
#include <cerrno>
#ifdef HAVE_FENV_H
#include <fenv.h>
#endif

namespace llvm {
namespace sys {

/// llvm_fenv_clearexcept - Clear the floating-point exception state.
static inline void llvm_fenv_clearexcept() {
#ifdef HAVE_FENV_H
  feclearexcept(FE_ALL_EXCEPT);
#endif
  errno = 0;
}

/// llvm_fenv_testexcept - Test if a floating-point exception was raised.
static inline bool llvm_fenv_testexcept() {
  int errno_val = errno;
  if (errno_val == ERANGE || errno_val == EDOM)
    return true;
#ifdef HAVE_FENV_H
  if (fetestexcept(FE_ALL_EXCEPT))
    return true;
#endif
  return false;
}

} // End sys namespace
} // End llvm namespace

#endif

//===-- llvm/Target/TargetOptions.h - Target Options ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines command line option flags that are shared across various
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPTIONS_H
#define LLVM_TARGET_TARGETOPTIONS_H

namespace llvm {
  /// PrintMachineCode - This flag is enabled when the -print-machineinstrs
  /// option is specified on the command line, and should enable debugging
  /// output from the code generator.
  extern bool PrintMachineCode;

  /// NoFramePointerElim - This flag is enabled when the -disable-fp-elim is
  /// specified on the command line.  If the target supports the frame pointer
  /// elimination optimization, this option should disable it.
  extern bool NoFramePointerElim;

  /// NoExcessFPPrecision - This flag is enabled when the
  /// -disable-excess-fp-precision flag is specified on the command line.  When
  /// this flag is off (the default), the code generator is allowed to produce
  /// results that are "more precise" than IEEE allows.  This includes use of
  /// FMA-like operations and use of the X86 FP registers without rounding all
  /// over the place.
  extern bool NoExcessFPPrecision;

  /// UnsafeFPMath - This flag is enabled when the
  /// -enable-unsafe-fp-math flag is specified on the command line.  When
  /// this flag is off (the default), the code generator is not allowed to
  /// produce results that are "less precise" than IEEE allows.  This includes
  /// use of X86 instructions like FSIN and FCOS instead of libcalls.
  /// UnsafeFPMath implies FiniteOnlyFPMath.
  extern bool UnsafeFPMath;

  /// FiniteOnlyFPMath - This returns true when the -enable-finite-only-fp-math
  /// option is specified on the command line. If this returns false (default),
  /// the code generator is not allowed to assume that FP arithmetic arguments
  /// and results are never NaNs or +-Infs.
  extern bool FiniteOnlyFPMathOption;
  extern bool FiniteOnlyFPMath();
  
  /// UseSoftFloat - This flag is enabled when the -soft-float flag is specified
  /// on the command line.  When this flag is on, the code generator will
  /// generate libcalls to the software floating point library instead of
  /// target FP instructions.
  extern bool UseSoftFloat;
} // End llvm namespace

#endif

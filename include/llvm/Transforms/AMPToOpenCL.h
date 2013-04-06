//===-- Vectorize.h - Vectorization Transformations -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Vectorize transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_AMPTOOPENCL_H
#define LLVM_TRANSFORMS_AMPTOOPENCL_H

namespace llvm {
class Pass;

Pass *createAMPToOpenCLPass();

} // End llvm namespace

#endif

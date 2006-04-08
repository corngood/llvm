//===-- ConstantFolding.h - Internal Constant Folding Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the (internal) constant folding interfaces for LLVM.  These
// interfaces are used by the ConstantExpr::get* methods to automatically fold
// constants when possible.
//
// These operators may return a null object if I don't know how to perform the
// specified operation on the specified constant types.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTANTFOLDING_H
#define CONSTANTFOLDING_H

#include <vector>

namespace llvm {
  class Value;
  class Constant;
  class Type;

  // Constant fold various types of instruction...
  Constant *ConstantFoldCastInstruction(const Constant *V, const Type *DestTy);
  Constant *ConstantFoldSelectInstruction(const Constant *Cond,
                                          const Constant *V1,
                                          const Constant *V2);
  Constant *ConstantFoldExtractElementInstruction(const Constant *Val,
                                                  const Constant *Idx);
  Constant *ConstantFoldInsertElementInstruction(const Constant *Val,
                                                 const Constant *Elt,
                                                 const Constant *Idx);
  Constant *ConstantFoldShuffleVectorInstruction(const Constant *V1,
                                                 const Constant *V2,
                                                 const Constant *Mask);
  Constant *ConstantFoldBinaryInstruction(unsigned Opcode, const Constant *V1,
                                          const Constant *V2);
  Constant *ConstantFoldGetElementPtr(const Constant *C,
                                      const std::vector<Value*> &IdxList);
} // End llvm namespace

#endif

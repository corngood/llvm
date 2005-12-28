//===-- CodeGen/MachineConstantPool.h - Abstract Constant Pool --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The MachineConstantPool class keeps track of constants referenced by a
// function which must be spilled to memory.  This is used for constants which
// are unable to be used directly as operands to instructions, which typically
// include floating point and large integer constants.
//
// Instructions reference the address of these constant pool constants through
// the use of MO_ConstantPoolIndex values.  When emitting assembly or machine
// code, these virtual address references are converted to refer to the
// address of the function constant pool values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECONSTANTPOOL_H
#define LLVM_CODEGEN_MACHINECONSTANTPOOL_H

#include <vector>
#include <iosfwd>

namespace llvm {

class Constant;

class MachineConstantPool {
  std::vector<Constant*> Constants;
public:

  /// getConstantPoolIndex - Create a new entry in the constant pool or return
  /// an existing one.
  ///
  unsigned getConstantPoolIndex(Constant *C) {
    // Check to see if we already have this constant.
    //
    // FIXME, this could be made much more efficient for large constant pools.
    for (unsigned i = 0, e = Constants.size(); i != e; ++i)
      if (Constants[i] == C)
        return i;
    Constants.push_back(C);
    return Constants.size()-1;
  }

  /// isEmpty - Return true if this constant pool contains no constants.
  ///
  bool isEmpty() const { return Constants.empty(); }

  const std::vector<Constant*> &getConstants() const { return Constants; }

  /// print - Used by the MachineFunction printer to print information about
  /// stack objects.  Implemented in MachineFunction.cpp
  ///
  void print(std::ostream &OS) const;

  /// dump - Call print(std::cerr) to be called from the debugger.
  void dump() const;
};

} // End llvm namespace

#endif

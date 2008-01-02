//===-- llvm/Support/CallSite.h - Abstract Call & Invoke instrs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CallSite class, which is a handy wrapper for code that
// wants to treat Call and Invoke instructions in a generic way.
//
// NOTE: This class is supposed to have "value semantics". So it should be
// passed by value, not by reference; it should not be "new"ed or "delete"d. It
// is efficiently copyable, assignable and constructable, with cost equivalent
// to copying a pointer (notice that it has only a single data member).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CALLSITE_H
#define LLVM_SUPPORT_CALLSITE_H

#include "llvm/Instruction.h"
#include "llvm/BasicBlock.h"

namespace llvm {

class CallInst;
class InvokeInst;
class ParamAttrsList;

class CallSite {
  Instruction *I;
public:
  CallSite() : I(0) {}
  CallSite(CallInst *CI) : I(reinterpret_cast<Instruction*>(CI)) {}
  CallSite(InvokeInst *II) : I(reinterpret_cast<Instruction*>(II)) {}
  CallSite(const CallSite &CS) : I(CS.I) {}
  CallSite &operator=(const CallSite &CS) { I = CS.I; return *this; }

  /// CallSite::get - This static method is sort of like a constructor.  It will
  /// create an appropriate call site for a Call or Invoke instruction, but it
  /// can also create a null initialized CallSite object for something which is
  /// NOT a call site.
  ///
  static CallSite get(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (I->getOpcode() == Instruction::Call)
        return CallSite(reinterpret_cast<CallInst*>(I));
      else if (I->getOpcode() == Instruction::Invoke)
        return CallSite(reinterpret_cast<InvokeInst*>(I));
    }
    return CallSite();
  }

  /// getCallingConv/setCallingConv - get or set the calling convention of the
  /// call.
  unsigned getCallingConv() const;
  void setCallingConv(unsigned CC);

  /// getParamAttrs/setParamAttrs - get or set the parameter attributes of
  /// the call.
  const ParamAttrsList *getParamAttrs() const;
  void setParamAttrs(const ParamAttrsList *PAL);

  /// paramHasAttr - whether the call or the callee has the given attribute.
  bool paramHasAttr(uint16_t i, unsigned attr) const;

  /// @brief Determine if the call does not access memory.
  bool doesNotAccessMemory() const;

  /// @brief Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const;

  /// @brief Determine if the call cannot unwind.
  bool doesNotThrow() const;
  void setDoesNotThrow(bool doesNotThrow = true);

  /// getType - Return the type of the instruction that generated this call site
  ///
  const Type *getType() const { return I->getType(); }

  /// getInstruction - Return the instruction this call site corresponds to
  ///
  Instruction *getInstruction() const { return I; }

  /// getCaller - Return the caller function for this call site
  ///
  Function *getCaller() const { return I->getParent()->getParent(); }

  /// getCalledValue - Return the pointer to function that is being called...
  ///
  Value *getCalledValue() const {
    assert(I && "Not a call or invoke instruction!");
    return I->getOperand(0);
  }

  /// getCalledFunction - Return the function being called if this is a direct
  /// call, otherwise return null (if it's an indirect call).
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(getCalledValue());
  }

  /// setCalledFunction - Set the callee to the specified value...
  ///
  void setCalledFunction(Value *V) {
    assert(I && "Not a call or invoke instruction!");
    I->setOperand(0, V);
  }

  Value *getArgument(unsigned ArgNo) const {
    assert(arg_begin() + ArgNo < arg_end() && "Argument # out of range!");
    return *(arg_begin()+ArgNo);
  }

  /// arg_iterator - The type of iterator to use when looping over actual
  /// arguments at this call site...
  typedef User::op_iterator arg_iterator;

  /// arg_begin/arg_end - Return iterators corresponding to the actual argument
  /// list for a call site.
  ///
  arg_iterator arg_begin() const {
    assert(I && "Not a call or invoke instruction!");
    if (I->getOpcode() == Instruction::Call)
      return I->op_begin()+1; // Skip Function
    else
      return I->op_begin()+3; // Skip Function, BB, BB
  }
  arg_iterator arg_end() const { return I->op_end(); }
  bool arg_empty() const { return arg_end() == arg_begin(); }
  unsigned arg_size() const { return unsigned(arg_end() - arg_begin()); }

  bool operator<(const CallSite &CS) const {
    return getInstruction() < CS.getInstruction();
  }
};

} // End llvm namespace

#endif

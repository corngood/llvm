//===-- llvm/iOther.h - "Other" instruction node definitions ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for instructions that fall into the 
// grandiose 'other' catagory...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IOTHER_H
#define LLVM_IOTHER_H

#include "llvm/InstrTypes.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//                                 CastInst Class
//===----------------------------------------------------------------------===//

/// CastInst - This class represents a cast from Operand[0] to the type of
/// the instruction (i->getType()).
///
class CastInst : public Instruction {
  CastInst(const CastInst &CI) : Instruction(CI.getType(), Cast) {
    Operands.reserve(1);
    Operands.push_back(Use(CI.Operands[0], this));
  }
  void init(Value *S) {
    Operands.reserve(1);
    Operands.push_back(Use(S, this));
  }
public:
  CastInst(Value *S, const Type *Ty, const std::string &Name = "",
           Instruction *InsertBefore = 0)
    : Instruction(Ty, Cast, Name, InsertBefore) {
    init(S);
  }
  CastInst(Value *S, const Type *Ty, const std::string &Name,
           BasicBlock *InsertAtEnd)
    : Instruction(Ty, Cast, Name, InsertAtEnd) {
    init(S);
  }

  virtual Instruction *clone() const { return new CastInst(*this); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CastInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Cast;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                 CallInst Class
//===----------------------------------------------------------------------===//

/// CallInst - This class represents a function call, abstracting a target
/// machine's calling convention.
///
class CallInst : public Instruction {
  CallInst(const CallInst &CI);
  void init(Value *Func, const std::vector<Value*> &Params);
  void init(Value *Func, Value *Actual);
  void init(Value *Func);

public:
  CallInst(Value *F, const std::vector<Value*> &Par,
           const std::string &Name = "", Instruction *InsertBefore = 0);
  CallInst(Value *F, const std::vector<Value*> &Par,
           const std::string &Name, BasicBlock *InsertAtEnd);

  // Alternate CallInst ctors w/ one actual & no actuals, respectively.
  CallInst(Value *F, Value *Actual, const std::string& Name = "",
           Instruction *InsertBefore = 0);
  CallInst(Value *F, Value *Actual, const std::string& Name,
           BasicBlock *InsertAtEnd);
  explicit CallInst(Value *F, const std::string &Name = "", 
                    Instruction *InsertBefore = 0);
  explicit CallInst(Value *F, const std::string &Name, 
                    BasicBlock *InsertAtEnd);

  virtual Instruction *clone() const { return new CallInst(*this); }
  bool mayWriteToMemory() const { return true; }

  // FIXME: These methods should be inline once we eliminate
  // ConstantPointerRefs!
  const Function *getCalledFunction() const;
  Function *getCalledFunction();

  // getCalledValue - Get a pointer to a method that is invoked by this inst.
  inline const Value *getCalledValue() const { return Operands[0]; }
  inline       Value *getCalledValue()       { return Operands[0]; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CallInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Call; 
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                 ShiftInst Class
//===----------------------------------------------------------------------===//

/// ShiftInst - This class represents left and right shift instructions.
///
class ShiftInst : public Instruction {
  ShiftInst(const ShiftInst &SI) : Instruction(SI.getType(), SI.getOpcode()) {
    Operands.reserve(2);
    Operands.push_back(Use(SI.Operands[0], this));
    Operands.push_back(Use(SI.Operands[1], this));
  }
  void init(OtherOps Opcode, Value *S, Value *SA) {
    assert((Opcode == Shl || Opcode == Shr) && "ShiftInst Opcode invalid!");
    Operands.reserve(2);
    Operands.push_back(Use(S, this));
    Operands.push_back(Use(SA, this));
  }

public:
  ShiftInst(OtherOps Opcode, Value *S, Value *SA, const std::string &Name = "",
            Instruction *InsertBefore = 0)
    : Instruction(S->getType(), Opcode, Name, InsertBefore) {
    init(Opcode, S, SA);
  }
  ShiftInst(OtherOps Opcode, Value *S, Value *SA, const std::string &Name,
            BasicBlock *InsertAtEnd)
    : Instruction(S->getType(), Opcode, Name, InsertAtEnd) {
    init(Opcode, S, SA);
  }

  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  virtual Instruction *clone() const { return new ShiftInst(*this); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ShiftInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Shr) | 
           (I->getOpcode() == Instruction::Shl);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               SelectInst Class
//===----------------------------------------------------------------------===//

/// SelectInst - This class represents the LLVM 'select' instruction.
///
class SelectInst : public Instruction {
  SelectInst(const SelectInst &SI) : Instruction(SI.getType(), SI.getOpcode()) {
    Operands.reserve(3);
    Operands.push_back(Use(SI.Operands[0], this));
    Operands.push_back(Use(SI.Operands[1], this));
    Operands.push_back(Use(SI.Operands[2], this));
  }
  void init(Value *C, Value *S1, Value *S2) {
    Operands.reserve(3);
    Operands.push_back(Use(C, this));
    Operands.push_back(Use(S1, this));
    Operands.push_back(Use(S2, this));
  }

public:
  SelectInst(Value *C, Value *S1, Value *S2, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(S1->getType(), Instruction::Select, Name, InsertBefore) {
    init(C, S1, S2);
  }
  SelectInst(Value *C, Value *S1, Value *S2, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : Instruction(S1->getType(), Instruction::Select, Name, InsertAtEnd) {
    init(C, S1, S2);
  }

  Value *getCondition() const { return Operands[0]; }
  Value *getTrueValue() const { return Operands[1]; }
  Value *getFalseValue() const { return Operands[2]; }

  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  virtual Instruction *clone() const { return new SelectInst(*this); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SelectInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Select;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                VANextInst Class
//===----------------------------------------------------------------------===//

/// VANextInst - This class represents the va_next llvm instruction, which
/// advances a vararg list passed an argument of the specified type, returning
/// the resultant list.
///
class VANextInst : public Instruction {
  PATypeHolder ArgTy;
  void init(Value *List) {
    Operands.reserve(1);
    Operands.push_back(Use(List, this));
  }
  VANextInst(const VANextInst &VAN)
    : Instruction(VAN.getType(), VANext), ArgTy(VAN.getArgType()) {
    init(VAN.Operands[0]);
  }

public:
  VANextInst(Value *List, const Type *Ty, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(List->getType(), VANext, Name, InsertBefore), ArgTy(Ty) {
    init(List);
  }
  VANextInst(Value *List, const Type *Ty, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : Instruction(List->getType(), VANext, Name, InsertAtEnd), ArgTy(Ty) {
    init(List);
  }

  const Type *getArgType() const { return ArgTy; }

  virtual Instruction *clone() const { return new VANextInst(*this); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VANextInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VANext;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                VAArgInst Class
//===----------------------------------------------------------------------===//

/// VAArgInst - This class represents the va_arg llvm instruction, which returns
/// an argument of the specified type given a va_list.
///
class VAArgInst : public Instruction {
  void init(Value* List) {
    Operands.reserve(1);
    Operands.push_back(Use(List, this));
  }
  VAArgInst(const VAArgInst &VAA)
    : Instruction(VAA.getType(), VAArg) {
    init(VAA.Operands[0]);
  }
public:
  VAArgInst(Value *List, const Type *Ty, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(Ty, VAArg, Name, InsertBefore) {
    init(List);
  }
  VAArgInst(Value *List, const Type *Ty, const std::string &Name,
            BasicBlock *InsertAtEnd)
    : Instruction(Ty, VAArg, Name, InsertAtEnd) {
    init(List);
  }

  virtual Instruction *clone() const { return new VAArgInst(*this); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VAArgInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VAArg;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif

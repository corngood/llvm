//===-- iBinaryOperators.cpp - Implement the BinaryOperators -----*- C++ -*--=//
//
// This file implements the nontrivial binary operator instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/iBinary.h"
#include "llvm/Type.h"

UnaryOperator *UnaryOperator::create(UnaryOps Op, Value *Source,
				     const Type *DestTy = 0) {
  if (DestTy == 0) DestTy = Source->getType();
  switch (Op) {
  case Not:  assert(DestTy == Source->getType());
  case Cast: return new GenericUnaryInst(Op, Source, DestTy);
  default:
    cerr << "Don't know how to GetUnaryOperator " << Op << endl;
    return 0;
  }
}

const char *GenericUnaryInst::getOpcodeName() const {
  switch (getOpcode()) {
  case Not: return "not";
  case Cast: return "cast";
  default:
    cerr << "Invalid unary operator type!" << getOpcode() << endl;
    abort();
  }
}

//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
				       const string &Name) {
  switch (Op) {
  // Binary comparison operators...
  case SetLT: case SetGT: case SetLE:
  case SetGE: case SetEQ: case SetNE:
    return new SetCondInst(Op, S1, S2, Name);

  default:
    return new GenericBinaryInst(Op, S1, S2, Name);
  }
}

//===----------------------------------------------------------------------===//
//                            GenericBinaryInst Class
//===----------------------------------------------------------------------===//

const char *GenericBinaryInst::getOpcodeName() const {
  switch (getOpcode()) {
  // Standard binary operators...
  case Add: return "add";
  case Sub: return "sub";
  case Mul: return "mul";
  case Div: return "div";
  case Rem: return "rem";

  // Logical operators...
  case And: return "and";
  case Or : return "or";
  case Xor: return "xor";
  default:
    cerr << "Invalid binary operator type!" << getOpcode() << endl;
    abort();
  }
}

//===----------------------------------------------------------------------===//
//                             SetCondInst Class
//===----------------------------------------------------------------------===//

SetCondInst::SetCondInst(BinaryOps opType, Value *S1, Value *S2, 
                         const string &Name) 
  : BinaryOperator(opType, S1, S2, Name) {

  OpType = opType;
  setType(Type::BoolTy);   // setcc instructions always return bool type.

  // Make sure it's a valid type...
  assert(getOpcodeName() != 0);
}

const char *SetCondInst::getOpcodeName() const {
  switch (OpType) {
  case SetLE:  return "setle";
  case SetGE:  return "setge";
  case SetLT:  return "setlt";
  case SetGT:  return "setgt";
  case SetEQ:  return "seteq";
  case SetNE:  return "setne";
  default:
    assert(0 && "Invalid opcode type to SetCondInst class!");
    return "invalid opcode type to SetCondInst";
  }
}

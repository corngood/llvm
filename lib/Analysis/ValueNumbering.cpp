//===- ValueNumbering.cpp - Value #'ing Implementation ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the non-abstract Value Numbering methods as well as a
// default implementation for the analysis group.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ValueNumbering.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
using namespace llvm;

// Register the ValueNumbering interface, providing a nice name to refer to.
static RegisterAnalysisGroup<ValueNumbering> X("Value Numbering");

/// ValueNumbering destructor: DO NOT move this to the header file for
/// ValueNumbering or else clients of the ValueNumbering class may not depend on
/// the ValueNumbering.o file in the current .a file, causing alias analysis
/// support to not be included in the tool correctly!
///
ValueNumbering::~ValueNumbering() {}

//===----------------------------------------------------------------------===//
// Basic ValueNumbering Pass Implementation
//===----------------------------------------------------------------------===//
//
// Because of the way .a files work, the implementation of the BasicVN class
// MUST be in the ValueNumbering file itself, or else we run the risk of
// ValueNumbering being used, but the default implementation not being linked
// into the tool that uses it.  As such, we register and implement the class
// here.
//

namespace {
  /// BasicVN - This class is the default implementation of the ValueNumbering
  /// interface.  It walks the SSA def-use chains to trivially identify
  /// lexically identical expressions.  This does not require any ahead of time
  /// analysis, so it is a very fast default implementation.
  ///
  struct BasicVN : public ImmutablePass, public ValueNumbering {
    /// getEqualNumberNodes - Return nodes with the same value number as the
    /// specified Value.  This fills in the argument vector with any equal
    /// values.
    ///
    /// This is where our implementation is.
    ///
    virtual void getEqualNumberNodes(Value *V1,
                                     std::vector<Value*> &RetVals) const;
  };

  // Register this pass...
  RegisterOpt<BasicVN>
  X("basicvn", "Basic Value Numbering (default GVN impl)");

  // Declare that we implement the ValueNumbering interface
  RegisterAnalysisGroup<ValueNumbering, BasicVN, true> Y;

  /// BVNImpl - Implement BasicVN in terms of a visitor class that
  /// handles the different types of instructions as appropriate.
  ///
  struct BVNImpl : public InstVisitor<BVNImpl> {
    std::vector<Value*> &RetVals;
    BVNImpl(std::vector<Value*> &RV) : RetVals(RV) {}

    void visitCastInst(CastInst &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);

    void handleBinaryInst(Instruction &I);
    void visitBinaryOperator(Instruction &I)     { handleBinaryInst(I); }
    void visitShiftInst(Instruction &I)          { handleBinaryInst(I); }
    void visitExtractElementInst(Instruction &I) { handleBinaryInst(I); }

    void handleTernaryInst(Instruction &I);
    void visitSelectInst(Instruction &I)         { handleTernaryInst(I); }
    void visitInsertElementInst(Instruction &I)  { handleTernaryInst(I); }
    void visitShuffleVectorInst(Instruction &I)  { handleTernaryInst(I); }
    void visitInstruction(Instruction &) {
      // Cannot value number calls or terminator instructions.
    }
  };
}

ImmutablePass *llvm::createBasicVNPass() { return new BasicVN(); }

// getEqualNumberNodes - Return nodes with the same value number as the
// specified Value.  This fills in the argument vector with any equal values.
//
void BasicVN::getEqualNumberNodes(Value *V, std::vector<Value*> &RetVals) const{
  assert(V->getType() != Type::VoidTy &&
         "Can only value number non-void values!");
  // We can only handle the case where I is an instruction!
  if (Instruction *I = dyn_cast<Instruction>(V))
    BVNImpl(RetVals).visit(I);
}

void BVNImpl::visitCastInst(CastInst &CI) {
  Instruction &I = (Instruction&)CI;
  Value *Op = I.getOperand(0);
  Function *F = I.getParent()->getParent();

  for (Value::use_iterator UI = Op->use_begin(), UE = Op->use_end();
       UI != UE; ++UI)
    if (CastInst *Other = dyn_cast<CastInst>(*UI))
      // Check that the types are the same, since this code handles casts...
      if (Other->getType() == I.getType() &&
          // Is it embedded in the same function?  (This could be false if LHS
          // is a constant or global!)
          Other->getParent()->getParent() == F &&
          // Check to see if this new cast is not I.
          Other != &I) {
        // These instructions are identical.  Add to list...
        RetVals.push_back(Other);
      }
}


// isIdenticalBinaryInst - Return true if the two binary instructions are
// identical.
//
static inline bool isIdenticalBinaryInst(const Instruction &I1,
                                         const Instruction *I2) {
  // Is it embedded in the same function?  (This could be false if LHS
  // is a constant or global!)
  if (I1.getOpcode() != I2->getOpcode() ||
      I1.getParent()->getParent() != I2->getParent()->getParent())
    return false;

  // They are identical if both operands are the same!
  if (I1.getOperand(0) == I2->getOperand(0) &&
      I1.getOperand(1) == I2->getOperand(1))
    return true;

  // If the instruction is commutative, the instruction can match if the
  // operands are swapped!
  //
  if ((I1.getOperand(0) == I2->getOperand(1) &&
       I1.getOperand(1) == I2->getOperand(0)) &&
      I1.isCommutative())
    return true;

  return false;
}

// isIdenticalTernaryInst - Return true if the two ternary instructions are
// identical.
//
static inline bool isIdenticalTernaryInst(const Instruction &I1,
                                          const Instruction *I2) {
  // Is it embedded in the same function?  (This could be false if LHS
  // is a constant or global!)
  if (I1.getParent()->getParent() != I2->getParent()->getParent())
    return false;
  
  // They are identical if all operands are the same!
  return I1.getOperand(0) == I2->getOperand(0) &&
         I1.getOperand(1) == I2->getOperand(1) &&
         I1.getOperand(2) == I2->getOperand(2);
}



void BVNImpl::handleBinaryInst(Instruction &I) {
  Value *LHS = I.getOperand(0);

  for (Value::use_iterator UI = LHS->use_begin(), UE = LHS->use_end();
       UI != UE; ++UI)
    if (Instruction *Other = dyn_cast<Instruction>(*UI))
      // Check to see if this new binary operator is not I, but same operand...
      if (Other != &I && isIdenticalBinaryInst(I, Other)) {
        // These instructions are identical.  Handle the situation.
        RetVals.push_back(Other);
      }
}

// IdenticalComplexInst - Return true if the two instructions are the same, by
// using a brute force comparison.  This is useful for instructions with an
// arbitrary number of arguments.
//
static inline bool IdenticalComplexInst(const Instruction *I1,
                                        const Instruction *I2) {
  assert(I1->getOpcode() == I2->getOpcode());
  // Equal if they are in the same function...
  return I1->getParent()->getParent() == I2->getParent()->getParent() &&
    // And return the same type...
    I1->getType() == I2->getType() &&
    // And have the same number of operands...
    I1->getNumOperands() == I2->getNumOperands() &&
    // And all of the operands are equal.
    std::equal(I1->op_begin(), I1->op_end(), I2->op_begin());
}

void BVNImpl::visitGetElementPtrInst(GetElementPtrInst &I) {
  Value *Op = I.getOperand(0);

  // Try to pick a local operand if possible instead of a constant or a global
  // that might have a lot of uses.
  for (unsigned i = 1, e = I.getNumOperands(); i != e; ++i)
    if (isa<Instruction>(I.getOperand(i)) || isa<Argument>(I.getOperand(i))) {
      Op = I.getOperand(i);
      break;
    }

  for (Value::use_iterator UI = Op->use_begin(), UE = Op->use_end();
       UI != UE; ++UI)
    if (GetElementPtrInst *Other = dyn_cast<GetElementPtrInst>(*UI))
      // Check to see if this new getelementptr is not I, but same operand...
      if (Other != &I && IdenticalComplexInst(&I, Other)) {
        // These instructions are identical.  Handle the situation.
        RetVals.push_back(Other);
      }
}

void BVNImpl::handleTernaryInst(Instruction &I) {
  Value *Op0 = I.getOperand(0);
  Instruction *OtherInst;
  
  for (Value::use_iterator UI = Op0->use_begin(), UE = Op0->use_end();
       UI != UE; ++UI)
    if ((OtherInst = dyn_cast<Instruction>(*UI)) && 
        OtherInst->getOpcode() == I.getOpcode()) {
      // Check to see if this new select is not I, but has the same operands.
      if (OtherInst != &I && isIdenticalTernaryInst(I, OtherInst)) {
        // These instructions are identical.  Handle the situation.
        RetVals.push_back(OtherInst);
      }
        
    }
}


// Ensure that users of ValueNumbering.h will link with this file
DEFINING_FILE_FOR(BasicValueNumbering)

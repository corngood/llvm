//===-- Local.h - Functions to perform local transformations ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOCAL_H
#define LLVM_TRANSFORMS_UTILS_LOCAL_H

#include "llvm/Function.h"

namespace llvm {

class Pass;
class PHINode;
class AllocaInst;

//===----------------------------------------------------------------------===//
//  Local constant propagation...
//

/// doConstantPropagation - Constant prop a specific instruction.  Returns true
/// and potentially moves the iterator if constant propagation was performed.
///
bool doConstantPropagation(BasicBlock::iterator &I);

/// ConstantFoldTerminator - If a terminator instruction is predicated on a
/// constant value, convert it into an unconditional branch to the constant
/// destination.  This is a nontrivial operation because the successors of this
/// basic block must have their PHI nodes updated.
///
bool ConstantFoldTerminator(BasicBlock *BB);

/// ConstantFoldInstruction - Attempt to constant fold the specified
/// instruction.  If successful, the constant result is returned, if not, null
/// is returned.  Note that this function can only fail when attempting to fold
/// instructions like loads and stores, which have no constant expression form.
///
Constant *ConstantFoldInstruction(Instruction *I);


/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool canConstantFoldCallTo(Function *F);

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *ConstantFoldCall(Function *F, const std::vector<Constant*> &Operands);


//===----------------------------------------------------------------------===//
//  Local dead code elimination...
//

/// isInstructionTriviallyDead - Return true if the result produced by the
/// instruction is not used, and the instruction has no side effects.
///
bool isInstructionTriviallyDead(Instruction *I);


/// dceInstruction - Inspect the instruction at *BBI and figure out if it
/// isTriviallyDead.  If so, remove the instruction and update the iterator to
/// point to the instruction that immediately succeeded the original
/// instruction.
///
bool dceInstruction(BasicBlock::iterator &BBI);

//===----------------------------------------------------------------------===//
//  Control Flow Graph Restructuring...
//

/// SimplifyCFG - This function is used to do simplification of a CFG.  For
/// example, it adjusts branches to branches to eliminate the extra hop, it
/// eliminates unreachable basic blocks, and does other "peephole" optimization
/// of the CFG.  It returns true if a modification was made, possibly deleting
/// the basic block that was pointed to.
///
/// WARNING:  The entry node of a method may not be simplified.
///
bool SimplifyCFG(BasicBlock *BB);

/// DemoteRegToStack - This function takes a virtual register computed by an
/// Instruction and replaces it with a slot in the stack frame, allocated via
/// alloca.  This allows the CFG to be changed around without fear of
/// invalidating the SSA information for the value.  It returns the pointer to
/// the alloca inserted to create a stack slot for X.
///
AllocaInst *DemoteRegToStack(Instruction &X);

} // End llvm namespace

#endif

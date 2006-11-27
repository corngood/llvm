//===- ProfilingUtils.cpp - Helper functions shared by profilers ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files implements a few helper functions which are used by profile
// instrumentation code to instrument the code.  This allows the profiler pass
// to worry about *what* to insert, and these functions take care of *how* to do
// it.
//
//===----------------------------------------------------------------------===//

#include "ProfilingUtils.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"

void llvm::InsertProfilingInitCall(Function *MainFn, const char *FnName,
                                   GlobalValue *Array) {
  const Type *ArgVTy = PointerType::get(PointerType::get(Type::SByteTy));
  const PointerType *UIntPtr = PointerType::get(Type::UIntTy);
  Module &M = *MainFn->getParent();
  Function *InitFn = M.getOrInsertFunction(FnName, Type::IntTy, Type::IntTy,
                                           ArgVTy, UIntPtr, Type::UIntTy,
                                           (Type *)0);

  // This could force argc and argv into programs that wouldn't otherwise have
  // them, but instead we just pass null values in.
  std::vector<Value*> Args(4);
  Args[0] = Constant::getNullValue(Type::IntTy);
  Args[1] = Constant::getNullValue(ArgVTy);

  // Skip over any allocas in the entry block.
  BasicBlock *Entry = MainFn->begin();
  BasicBlock::iterator InsertPos = Entry->begin();
  while (isa<AllocaInst>(InsertPos)) ++InsertPos;

  std::vector<Constant*> GEPIndices(2, Constant::getNullValue(Type::IntTy));
  unsigned NumElements = 0;
  if (Array) {
    Args[2] = ConstantExpr::getGetElementPtr(Array, GEPIndices);
    NumElements =
      cast<ArrayType>(Array->getType()->getElementType())->getNumElements();
  } else {
    // If this profiling instrumentation doesn't have a constant array, just
    // pass null.
    Args[2] = ConstantPointerNull::get(UIntPtr);
  }
  Args[3] = ConstantInt::get(Type::UIntTy, NumElements);

  Instruction *InitCall = new CallInst(InitFn, Args, "newargc", InsertPos);

  // If argc or argv are not available in main, just pass null values in.
  Function::arg_iterator AI;
  switch (MainFn->arg_size()) {
  default:
  case 2:
    AI = MainFn->arg_begin(); ++AI;
    if (AI->getType() != ArgVTy) {
      InitCall->setOperand(2, 
          CastInst::createInferredCast(AI, ArgVTy, "argv.cast", InitCall));
    } else {
      InitCall->setOperand(2, AI);
    }

  case 1:
    AI = MainFn->arg_begin();
    // If the program looked at argc, have it look at the return value of the
    // init call instead.
    if (AI->getType() != Type::IntTy) {
      if (!AI->use_empty())
        AI->replaceAllUsesWith(
          CastInst::createInferredCast(InitCall, AI->getType(), "", InsertPos));
      InitCall->setOperand(1, 
          CastInst::createInferredCast(AI, Type::IntTy, "argc.cast", InitCall));
    } else {
      AI->replaceAllUsesWith(InitCall);
      InitCall->setOperand(1, AI);
    }

  case 0: break;
  }
}

void llvm::IncrementCounterInBlock(BasicBlock *BB, unsigned CounterNum,
                                   GlobalValue *CounterArray) {
  // Insert the increment after any alloca or PHI instructions...
  BasicBlock::iterator InsertPos = BB->begin();
  while (isa<AllocaInst>(InsertPos) || isa<PHINode>(InsertPos))
    ++InsertPos;

  // Create the getelementptr constant expression
  std::vector<Constant*> Indices(2);
  Indices[0] = Constant::getNullValue(Type::IntTy);
  Indices[1] = ConstantInt::get(Type::IntTy, CounterNum);
  Constant *ElementPtr = ConstantExpr::getGetElementPtr(CounterArray, Indices);

  // Load, increment and store the value back.
  Value *OldVal = new LoadInst(ElementPtr, "OldFuncCounter", InsertPos);
  Value *NewVal = BinaryOperator::create(Instruction::Add, OldVal,
                                         ConstantInt::get(Type::UIntTy, 1),
                                         "NewFuncCounter", InsertPos);
  new StoreInst(NewVal, ElementPtr, InsertPos);
}

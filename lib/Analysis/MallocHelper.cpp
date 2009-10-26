//===-- MallocFreeHelper.cpp - Identify calls to malloc and free builtins -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to malloc, bitcasts of malloc
// calls, and the types and array sizes associated with them.  It also
// identifies calls to the free builtin.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MallocHelper.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ConstantFolding.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  malloc Call Utility Functions.
//

/// isMalloc - Returns true if the the value is either a malloc call or a
/// bitcast of the result of a malloc call.
bool llvm::isMalloc(const Value* I) {
  return extractMallocCall(I) || extractMallocCallFromBitCast(I);
}

static bool isMallocCall(const CallInst *CI) {
  if (!CI)
    return false;

  const Module* M = CI->getParent()->getParent()->getParent();
  Function *MallocFunc = M->getFunction("malloc");

  if (CI->getOperand(0) != MallocFunc)
    return false;

  // Check malloc prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin 
  // attribute will exist.
  const FunctionType *FTy = MallocFunc->getFunctionType();
  if (FTy->getNumParams() != 1)
    return false;
  if (IntegerType *ITy = dyn_cast<IntegerType>(FTy->param_begin()->get())) {
    if (ITy->getBitWidth() != 32 && ITy->getBitWidth() != 64)
      return false;
    return true;
  }

  return false;
}

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst* llvm::extractMallocCall(const Value* I) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  return (isMallocCall(CI)) ? CI : NULL;
}

CallInst* llvm::extractMallocCall(Value* I) {
  CallInst *CI = dyn_cast<CallInst>(I);
  return (isMallocCall(CI)) ? CI : NULL;
}

static bool isBitCastOfMallocCall(const BitCastInst* BCI) {
  if (!BCI)
    return false;
    
  return isMallocCall(dyn_cast<CallInst>(BCI->getOperand(0)));
}

/// extractMallocCallFromBitCast - Returns the corresponding CallInst if the
/// instruction is a bitcast of the result of a malloc call.
CallInst* llvm::extractMallocCallFromBitCast(Value* I) {
  BitCastInst *BCI = dyn_cast<BitCastInst>(I);
  return (isBitCastOfMallocCall(BCI)) ? cast<CallInst>(BCI->getOperand(0))
                                      : NULL;
}

const CallInst* llvm::extractMallocCallFromBitCast(const Value* I) {
  const BitCastInst *BCI = dyn_cast<BitCastInst>(I);
  return (isBitCastOfMallocCall(BCI)) ? cast<CallInst>(BCI->getOperand(0))
                                      : NULL;
}

static bool isArrayMallocHelper(const CallInst *CI, LLVMContext &Context,
                                const TargetData* TD) {
  if (!CI)
    return false;

  const Type* T = getMallocAllocatedType(CI);

  // We can only indentify an array malloc if we know the type of the malloc 
  // call.
  if (!T) return false;

  Value* MallocArg = CI->getOperand(1);
  Constant *ElementSize = ConstantExpr::getSizeOf(T);
  ElementSize = ConstantExpr::getTruncOrBitCast(ElementSize, 
                                                MallocArg->getType());
  Constant *FoldedElementSize = ConstantFoldConstantExpression(
                                       cast<ConstantExpr>(ElementSize), 
                                       Context, TD);


  if (isa<ConstantExpr>(MallocArg))
    return (MallocArg != ElementSize);

  BinaryOperator *BI = dyn_cast<BinaryOperator>(MallocArg);
  if (!BI)
    return false;

  if (BI->getOpcode() == Instruction::Mul)
    // ArraySize * ElementSize
    if (BI->getOperand(1) == ElementSize ||
        (FoldedElementSize && BI->getOperand(1) == FoldedElementSize))
      return true;

  // TODO: Detect case where MallocArg mul has been transformed to shl.

  return false;
}

/// isArrayMalloc - Returns the corresponding CallInst if the instruction 
/// matches the malloc call IR generated by CallInst::CreateMalloc().  This 
/// means that it is a malloc call with one bitcast use AND the malloc call's 
/// size argument is:
///  1. a constant not equal to the size of the malloced type
/// or
///  2. the result of a multiplication by the size of the malloced type
/// Otherwise it returns NULL.
/// The unique bitcast is needed to determine the type/size of the array
/// allocation.
CallInst* llvm::isArrayMalloc(Value* I, LLVMContext &Context,
                              const TargetData* TD) {
  CallInst *CI = extractMallocCall(I);
  return (isArrayMallocHelper(CI, Context, TD)) ? CI : NULL;
}

const CallInst* llvm::isArrayMalloc(const Value* I, LLVMContext &Context,
                                    const TargetData* TD) {
  const CallInst *CI = extractMallocCall(I);
  return (isArrayMallocHelper(CI, Context, TD)) ? CI : NULL;
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// This PointerType is the result type of the call's only bitcast use.
/// If there is no unique bitcast use, then return NULL.
const PointerType* llvm::getMallocType(const CallInst* CI) {
  assert(isMalloc(CI) && "GetMallocType and not malloc call");
  
  const BitCastInst* BCI = NULL;
  
  // Determine if CallInst has a bitcast use.
  for (Value::use_const_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; )
    if ((BCI = dyn_cast<BitCastInst>(cast<Instruction>(*UI++))))
      break;

  // Malloc call has 1 bitcast use and no other uses, so type is the bitcast's
  // destination type.
  if (BCI && CI->hasOneUse())
    return cast<PointerType>(BCI->getDestTy());

  // Malloc call was not bitcast, so type is the malloc function's return type.
  if (!BCI)
    return cast<PointerType>(CI->getType());

  // Type could not be determined.
  return NULL;
}

/// getMallocAllocatedType - Returns the Type allocated by malloc call. This
/// Type is the result type of the call's only bitcast use. If there is no
/// unique bitcast use, then return NULL.
const Type* llvm::getMallocAllocatedType(const CallInst* CI) {
  const PointerType* PT = getMallocType(CI);
  return PT ? PT->getElementType() : NULL;
}

/// isSafeToGetMallocArraySize - Returns true if the array size of a malloc can
/// be determined.  It can be determined in these 3 cases of malloc codegen:
/// 1. non-array malloc: The malloc's size argument is a constant and equals the ///    size of the type being malloced.
/// 2. array malloc: This is a malloc call with one bitcast use AND the malloc
///    call's size argument is a constant multiple of the size of the malloced
///    type.
/// 3. array malloc: This is a malloc call with one bitcast use AND the malloc
///    call's size argument is the result of a multiplication by the size of the
///    malloced type.
/// Otherwise returns false.
static bool isSafeToGetMallocArraySize(const CallInst *CI,
                                       LLVMContext &Context,
                                       const TargetData* TD) {
  if (!CI)
    return false;

  // Type must be known to determine array size.
  const Type* T = getMallocAllocatedType(CI);
  if (!T) return false;

  Value* MallocArg = CI->getOperand(1);
  Constant *ElementSize = ConstantExpr::getSizeOf(T);
  ElementSize = ConstantExpr::getTruncOrBitCast(ElementSize, 
                                                MallocArg->getType());

  // First, check if it is a non-array malloc.
  if (isa<ConstantExpr>(MallocArg) && (MallocArg == ElementSize))
    return true;

  // Second, check if it can be determined that this is an array malloc.
  return isArrayMallocHelper(CI, Context, TD);
}

/// isConstantOne - Return true only if val is constant int 1.
static bool isConstantOne(Value *val) {
  return isa<ConstantInt>(val) && cast<ConstantInt>(val)->isOne();
}

/// getMallocArraySize - Returns the array size of a malloc call.  For array
/// mallocs, the size is computated in 1 of 3 ways:
///  1. If the element type is of size 1, then array size is the argument to 
///     malloc.
///  2. Else if the malloc's argument is a constant, the array size is that
///     argument divided by the element type's size.
///  3. Else the malloc argument must be a multiplication and the array size is
///     the first operand of the multiplication.
/// For non-array mallocs, the computed size is constant 1. 
/// This function returns NULL for all mallocs whose array size cannot be
/// determined.
Value* llvm::getMallocArraySize(CallInst* CI, LLVMContext &Context,
                                const TargetData* TD) {
  if (!isSafeToGetMallocArraySize(CI, Context, TD))
    return NULL;

  // Match CreateMalloc's use of constant 1 array-size for non-array mallocs.
  if (!isArrayMalloc(CI, Context, TD))
    return ConstantInt::get(CI->getOperand(1)->getType(), 1);

  Value* MallocArg = CI->getOperand(1);
  assert(getMallocAllocatedType(CI) && "getMallocArraySize and no type");
  Constant *ElementSize = ConstantExpr::getSizeOf(getMallocAllocatedType(CI));
  ElementSize = ConstantExpr::getTruncOrBitCast(ElementSize, 
                                                MallocArg->getType());

  Constant* CO = dyn_cast<Constant>(MallocArg);
  BinaryOperator* BO = dyn_cast<BinaryOperator>(MallocArg);
  assert((isConstantOne(ElementSize) || CO || BO) &&
         "getMallocArraySize and malformed malloc IR");
      
  if (isConstantOne(ElementSize))
    return MallocArg;
    
  if (CO)
    return CO->getOperand(0);
    
  // TODO: Detect case where MallocArg mul has been transformed to shl.

  assert(BO && "getMallocArraySize not constant but not multiplication either");
  return BO->getOperand(0);
}

//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isFreeCall - Returns true if the the value is a call to the builtin free()
bool llvm::isFreeCall(const Value* I) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI)
    return false;

  const Module* M = CI->getParent()->getParent()->getParent();
  Function *FreeFunc = M->getFunction("free");

  if (CI->getOperand(0) != FreeFunc)
    return false;

  // Check free prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin 
  // attribute will exist.
  const FunctionType *FTy = FreeFunc->getFunctionType();
  if (FTy->getReturnType() != Type::getVoidTy(M->getContext()))
    return false;
  if (FTy->getNumParams() != 1)
    return false;
  if (FTy->param_begin()->get() != Type::getInt8PtrTy(M->getContext()))
    return false;

  return true;
}

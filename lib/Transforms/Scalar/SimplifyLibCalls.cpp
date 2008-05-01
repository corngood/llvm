//===- SimplifyLibCalls.cpp - Optimize specific well-known library calls --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple pass that applies a variety of small
// optimizations for calls to specific well-known function calls (e.g. runtime
// library functions). For example, a call to the function "exit(3)" that
// occurs within the main() function can be transformed into a simple "return 3"
// instruction. Any optimization that takes this form (replace call to library
// function with simpler code that provides the same result) belongs in this
// file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplify-libcalls"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Config/config.h"
using namespace llvm;

STATISTIC(NumSimplified, "Number of library calls simplified");

//===----------------------------------------------------------------------===//
// Optimizer Base Class
//===----------------------------------------------------------------------===//

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class VISIBILITY_HIDDEN LibCallOptimization {
protected:
  Function *Caller;
  const TargetData *TD;
public:
  LibCallOptimization() { }
  virtual ~LibCallOptimization() {}

  /// CallOptimizer - This pure virtual method is implemented by base classes to
  /// do various optimizations.  If this returns null then no transformation was
  /// performed.  If it returns CI, then it transformed the call and CI is to be
  /// deleted.  If it returns something else, replace CI with the new value and
  /// delete CI.
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) =0;
  
  Value *OptimizeCall(CallInst *CI, const TargetData &TD, IRBuilder &B) {
    Caller = CI->getParent()->getParent();
    this->TD = &TD;
    return CallOptimizer(CI->getCalledFunction(), CI, B);
  }

  /// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
  Value *CastToCStr(Value *V, IRBuilder &B);

  /// EmitStrLen - Emit a call to the strlen function to the builder, for the
  /// specified pointer.  Ptr is required to be some pointer type, and the
  /// return value has 'intptr_t' type.
  Value *EmitStrLen(Value *Ptr, IRBuilder &B);
  
  /// EmitMemCpy - Emit a call to the memcpy function to the builder.  This
  /// always expects that the size has type 'intptr_t' and Dst/Src are pointers.
  Value *EmitMemCpy(Value *Dst, Value *Src, Value *Len, 
                    unsigned Align, IRBuilder &B);
  
  /// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
  /// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
  Value *EmitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilder &B);
    
  /// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name' (e.g.
  /// 'floor').  This function is known to take a single of type matching 'Op'
  /// and returns one value with the same type.  If 'Op' is a long double, 'l'
  /// is added as the suffix of name, if 'Op' is a float, we add a 'f' suffix.
  Value *EmitUnaryFloatFnCall(Value *Op, const char *Name, IRBuilder &B);
  
  /// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
  /// is an integer.
  void EmitPutChar(Value *Char, IRBuilder &B);
  
  /// EmitPutS - Emit a call to the puts function.  This assumes that Str is
  /// some pointer.
  void EmitPutS(Value *Str, IRBuilder &B);
    
  /// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
  /// an i32, and File is a pointer to FILE.
  void EmitFPutC(Value *Char, Value *File, IRBuilder &B);
  
  /// EmitFPutS - Emit a call to the puts function.  Str is required to be a
  /// pointer and File is a pointer to FILE.
  void EmitFPutS(Value *Str, Value *File, IRBuilder &B);
  
  /// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
  /// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
  void EmitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilder &B);
    
};
} // End anonymous namespace.

/// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
Value *LibCallOptimization::CastToCStr(Value *V, IRBuilder &B) {
  return B.CreateBitCast(V, PointerType::getUnqual(Type::Int8Ty), "cstr");
}

/// EmitStrLen - Emit a call to the strlen function to the builder, for the
/// specified pointer.  This always returns an integer value of size intptr_t.
Value *LibCallOptimization::EmitStrLen(Value *Ptr, IRBuilder &B) {
  Module *M = Caller->getParent();
  Constant *StrLen =M->getOrInsertFunction("strlen", TD->getIntPtrType(),
                                           PointerType::getUnqual(Type::Int8Ty),
                                           NULL);
  return B.CreateCall(StrLen, CastToCStr(Ptr, B), "strlen");
}

/// EmitMemCpy - Emit a call to the memcpy function to the builder.  This always
/// expects that the size has type 'intptr_t' and Dst/Src are pointers.
Value *LibCallOptimization::EmitMemCpy(Value *Dst, Value *Src, Value *Len,
                                       unsigned Align, IRBuilder &B) {
  Module *M = Caller->getParent();
  Intrinsic::ID IID = TD->getIntPtrType() == Type::Int32Ty ?
                           Intrinsic::memcpy_i32 : Intrinsic::memcpy_i64;
  Value *MemCpy = Intrinsic::getDeclaration(M, IID);
  return B.CreateCall4(MemCpy, CastToCStr(Dst, B), CastToCStr(Src, B), Len,
                       ConstantInt::get(Type::Int32Ty, Align));
}

/// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
/// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
Value *LibCallOptimization::EmitMemChr(Value *Ptr, Value *Val,
                                       Value *Len, IRBuilder &B) {
  Module *M = Caller->getParent();
  Value *MemChr = M->getOrInsertFunction("memchr",
                                         PointerType::getUnqual(Type::Int8Ty),
                                         PointerType::getUnqual(Type::Int8Ty),
                                         Type::Int32Ty, TD->getIntPtrType(),
                                         NULL);
  return B.CreateCall3(MemChr, CastToCStr(Ptr, B), Val, Len, "memchr");
}

/// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name' (e.g.
/// 'floor').  This function is known to take a single of type matching 'Op' and
/// returns one value with the same type.  If 'Op' is a long double, 'l' is
/// added as the suffix of name, if 'Op' is a float, we add a 'f' suffix.
Value *LibCallOptimization::EmitUnaryFloatFnCall(Value *Op, const char *Name,
                                                 IRBuilder &B) {
  char NameBuffer[20];
  if (Op->getType() != Type::DoubleTy) {
    // If we need to add a suffix, copy into NameBuffer.
    unsigned NameLen = strlen(Name);
    assert(NameLen < sizeof(NameBuffer)-2);
    memcpy(NameBuffer, Name, NameLen);
    if (Op->getType() == Type::FloatTy)
      NameBuffer[NameLen] = 'f';  // floorf
    else
      NameBuffer[NameLen] = 'l';  // floorl
    NameBuffer[NameLen+1] = 0;
    Name = NameBuffer;
  }
  
  Module *M = Caller->getParent();
  Value *Callee = M->getOrInsertFunction(Name, Op->getType(), 
                                         Op->getType(), NULL);
  return B.CreateCall(Callee, Op, Name);
}

/// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
/// is an integer.
void LibCallOptimization::EmitPutChar(Value *Char, IRBuilder &B) {
  Module *M = Caller->getParent();
  Value *F = M->getOrInsertFunction("putchar", Type::Int32Ty,
                                    Type::Int32Ty, NULL);
  B.CreateCall(F, B.CreateIntCast(Char, Type::Int32Ty, "chari"), "putchar");
}

/// EmitPutS - Emit a call to the puts function.  This assumes that Str is
/// some pointer.
void LibCallOptimization::EmitPutS(Value *Str, IRBuilder &B) {
  Module *M = Caller->getParent();
  Value *F = M->getOrInsertFunction("puts", Type::Int32Ty,
                                    PointerType::getUnqual(Type::Int8Ty), NULL);
  B.CreateCall(F, CastToCStr(Str, B), "puts");
}

/// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
/// an integer and File is a pointer to FILE.
void LibCallOptimization::EmitFPutC(Value *Char, Value *File, IRBuilder &B) {
  Module *M = Caller->getParent();
  Constant *F = M->getOrInsertFunction("fputc", Type::Int32Ty, Type::Int32Ty,
                                       File->getType(), NULL);
  Char = B.CreateIntCast(Char, Type::Int32Ty, "chari");
  B.CreateCall2(F, Char, File, "fputc");
}

/// EmitFPutS - Emit a call to the puts function.  Str is required to be a
/// pointer and File is a pointer to FILE.
void LibCallOptimization::EmitFPutS(Value *Str, Value *File, IRBuilder &B) {
  Module *M = Caller->getParent();
  Constant *F = M->getOrInsertFunction("fputs", Type::Int32Ty,
                                       PointerType::getUnqual(Type::Int8Ty),
                                       File->getType(), NULL);
  B.CreateCall2(F, CastToCStr(Str, B), File, "fputs");
}

/// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
/// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
void LibCallOptimization::EmitFWrite(Value *Ptr, Value *Size, Value *File,
                                     IRBuilder &B) {
  Module *M = Caller->getParent();
  Constant *F = M->getOrInsertFunction("fwrite", TD->getIntPtrType(),
                                       PointerType::getUnqual(Type::Int8Ty),
                                       TD->getIntPtrType(), TD->getIntPtrType(),
                                       File->getType(), NULL);
  B.CreateCall4(F, CastToCStr(Ptr, B), Size, 
                ConstantInt::get(TD->getIntPtrType(), 1), File);
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// GetConstantStringInfo - This function computes the length of a
/// null-terminated C string pointed to by V.  If successful, it returns true
/// and returns the string in Str.  If unsuccessful, it returns false.
static bool GetConstantStringInfo(Value *V, std::string &Str) {
  // Look bitcast instructions.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(V))
    return GetConstantStringInfo(BCI->getOperand(0), Str);
  
  // If the value is not a GEP instruction nor a constant expression with a
  // GEP instruction, then return false because ConstantArray can't occur
  // any other way
  User *GEP = 0;
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    GEP = GEPI;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() != Instruction::GetElementPtr)
      return false;
    GEP = CE;
  } else {
    return false;
  }
  
  // Make sure the GEP has exactly three arguments.
  if (GEP->getNumOperands() != 3)
    return false;
  
  // Check to make sure that the first operand of the GEP is an integer and
  // has value 0 so that we are sure we're indexing into the initializer.
  if (ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (!Idx->isZero())
      return false;
  } else
    return false;
  
  // If the second index isn't a ConstantInt, then this is a variable index
  // into the array.  If this occurs, we can't say anything meaningful about
  // the string.
  uint64_t StartIdx = 0;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
    StartIdx = CI->getZExtValue();
  else
    return false;
  
  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  GlobalVariable* GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasInitializer())
    return false;
  Constant *GlobalInit = GV->getInitializer();
  
  // Handle the ConstantAggregateZero case
  if (isa<ConstantAggregateZero>(GlobalInit)) {
    // This is a degenerate case. The initializer is constant zero so the
    // length of the string must be zero.
    Str.clear();
    return true;
  }
  
  // Must be a Constant Array
  ConstantArray *Array = dyn_cast<ConstantArray>(GlobalInit);
  if (Array == 0 || Array->getType()->getElementType() != Type::Int8Ty)
    return false;
  
  // Get the number of elements in the array
  uint64_t NumElts = Array->getType()->getNumElements();
  
  // Traverse the constant array from StartIdx (derived above) which is
  // the place the GEP refers to in the array.
  for (unsigned i = StartIdx; i < NumElts; ++i) {
    Constant *Elt = Array->getOperand(i);
    ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
    if (!CI) // This array isn't suitable, non-int initializer.
      return false;
    if (CI->isZero())
      return true; // we found end of string, success!
    Str += (char)CI->getZExtValue();
  }
  
  return false; // The array isn't null terminated.
}

/// GetStringLengthH - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLengthH(Value *V, SmallPtrSet<PHINode*, 32> &PHIs) {
  // Look through noop bitcast instructions.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(V))
    return GetStringLengthH(BCI->getOperand(0), PHIs);
  
  // If this is a PHI node, there are two cases: either we have already seen it
  // or we haven't.
  if (PHINode *PN = dyn_cast<PHINode>(V)) {
    if (!PHIs.insert(PN))
      return ~0ULL;  // already in the set.
    
    // If it was new, see if all the input strings are the same length.
    uint64_t LenSoFar = ~0ULL;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      uint64_t Len = GetStringLengthH(PN->getIncomingValue(i), PHIs);
      if (Len == 0) return 0; // Unknown length -> unknown.
      
      if (Len == ~0ULL) continue;
      
      if (Len != LenSoFar && LenSoFar != ~0ULL)
        return 0;    // Disagree -> unknown.
      LenSoFar = Len;
    }
    
    // Success, all agree.
    return LenSoFar;
  }
  
  // strlen(select(c,x,y)) -> strlen(x) ^ strlen(y)
  if (SelectInst *SI = dyn_cast<SelectInst>(V)) {
    uint64_t Len1 = GetStringLengthH(SI->getTrueValue(), PHIs);
    if (Len1 == 0) return 0;
    uint64_t Len2 = GetStringLengthH(SI->getFalseValue(), PHIs);
    if (Len2 == 0) return 0;
    if (Len1 == ~0ULL) return Len2;
    if (Len2 == ~0ULL) return Len1;
    if (Len1 != Len2) return 0;
    return Len1;
  }
  
  // If the value is not a GEP instruction nor a constant expression with a
  // GEP instruction, then return unknown.
  User *GEP = 0;
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    GEP = GEPI;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() != Instruction::GetElementPtr)
      return 0;
    GEP = CE;
  } else {
    return 0;
  }
  
  // Make sure the GEP has exactly three arguments.
  if (GEP->getNumOperands() != 3)
    return 0;
  
  // Check to make sure that the first operand of the GEP is an integer and
  // has value 0 so that we are sure we're indexing into the initializer.
  if (ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (!Idx->isZero())
      return 0;
  } else
    return 0;
  
  // If the second index isn't a ConstantInt, then this is a variable index
  // into the array.  If this occurs, we can't say anything meaningful about
  // the string.
  uint64_t StartIdx = 0;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
    StartIdx = CI->getZExtValue();
  else
    return 0;
  
  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  GlobalVariable* GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasInitializer())
    return 0;
  Constant *GlobalInit = GV->getInitializer();
  
  // Handle the ConstantAggregateZero case, which is a degenerate case. The
  // initializer is constant zero so the length of the string must be zero.
  if (isa<ConstantAggregateZero>(GlobalInit))
    return 1;  // Len = 0 offset by 1.
  
  // Must be a Constant Array
  ConstantArray *Array = dyn_cast<ConstantArray>(GlobalInit);
  if (!Array || Array->getType()->getElementType() != Type::Int8Ty)
    return false;
  
  // Get the number of elements in the array
  uint64_t NumElts = Array->getType()->getNumElements();
  
  // Traverse the constant array from StartIdx (derived above) which is
  // the place the GEP refers to in the array.
  for (unsigned i = StartIdx; i != NumElts; ++i) {
    Constant *Elt = Array->getOperand(i);
    ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
    if (!CI) // This array isn't suitable, non-int initializer.
      return 0;
    if (CI->isZero())
      return i-StartIdx+1; // We found end of string, success!
  }
  
  return 0; // The array isn't null terminated, conservatively return 'unknown'.
}

/// GetStringLength - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLength(Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;
  
  SmallPtrSet<PHINode*, 32> PHIs;
  uint64_t Len = GetStringLengthH(V, PHIs);
  // If Len is ~0ULL, we had an infinite phi cycle: this is dead code, so return
  // an empty string as a length.
  return Len == ~0ULL ? 1 : Len;
}

/// IsOnlyUsedInZeroEqualityComparison - Return true if it only matters that the
/// value is equal or not-equal to zero. 
static bool IsOnlyUsedInZeroEqualityComparison(Value *V) {
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(*UI))
      if (IC->isEquality())
        if (Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
          if (C->isNullValue())
            continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Miscellaneous LibCall Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'exit' Optimizations

/// ExitOpt - int main() { exit(4); } --> int main() { return 4; }
struct VISIBILITY_HIDDEN ExitOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Verify we have a reasonable prototype for exit.
    if (Callee->arg_size() == 0 || !CI->use_empty())
      return 0;

    // Verify the caller is main, and that the result type of main matches the
    // argument type of exit.
    if (!Caller->isName("main") || !Caller->hasExternalLinkage() ||
        Caller->getReturnType() != CI->getOperand(1)->getType())
      return 0;

    TerminatorInst *OldTI = CI->getParent()->getTerminator();
    
    // Create the return after the call.
    ReturnInst *RI = B.CreateRet(CI->getOperand(1));

    // Drop all successor phi node entries.
    for (unsigned i = 0, e = OldTI->getNumSuccessors(); i != e; ++i)
      OldTI->getSuccessor(i)->removePredecessor(CI->getParent());
    
    // Erase all instructions from after our return instruction until the end of
    // the block.
    BasicBlock::iterator FirstDead = RI; ++FirstDead;
    CI->getParent()->getInstList().erase(FirstDead, CI->getParent()->end());
    return CI;
  }
};

//===----------------------------------------------------------------------===//
// String and Memory LibCall Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'strcat' Optimizations

struct VISIBILITY_HIDDEN StrCatOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Verify the "strcat" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != PointerType::getUnqual(Type::Int8Ty) ||
        FT->getParamType(0) != FT->getReturnType() ||
        FT->getParamType(1) != FT->getReturnType())
      return 0;
    
    // Extract some information from the instruction
    Value *Dst = CI->getOperand(1);
    Value *Src = CI->getOperand(2);
    
    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return false;
    --Len;  // Unbias length.
    
    // Handle the simple, do-nothing case: strcat(x, "") -> x
    if (Len == 0)
      return Dst;
    
    // We need to find the end of the destination string.  That's where the
    // memory is to be moved to. We just generate a call to strlen.
    Value *DstLen = EmitStrLen(Dst, B);
    
    // Now that we have the destination's length, we must index into the
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    Dst = B.CreateGEP(Dst, DstLen, "endptr");
    
    // We have enough information to now generate the memcpy call to do the
    // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
    EmitMemCpy(Dst, Src, ConstantInt::get(TD->getIntPtrType(), Len+1), 1, B);
    return Dst;
  }
};

//===---------------------------------------===//
// 'strchr' Optimizations

struct VISIBILITY_HIDDEN StrChrOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Verify the "strchr" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 ||
        FT->getReturnType() != PointerType::getUnqual(Type::Int8Ty) ||
        FT->getParamType(0) != FT->getReturnType())
      return 0;
    
    Value *SrcStr = CI->getOperand(1);
    
    // If the second operand is non-constant, see if we can compute the length
    // of the input string and turn this into memchr.
    ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getOperand(2));
    if (CharC == 0) {
      uint64_t Len = GetStringLength(SrcStr);
      if (Len == 0 || FT->getParamType(1) != Type::Int32Ty) // memchr needs i32.
        return 0;
      
      return EmitMemChr(SrcStr, CI->getOperand(2), // include nul.
                        ConstantInt::get(TD->getIntPtrType(), Len), B);
    }

    // Otherwise, the character is a constant, see if the first argument is
    // a string literal.  If so, we can constant fold.
    std::string Str;
    if (!GetConstantStringInfo(SrcStr, Str))
      return false;
    
    // strchr can find the nul character.
    Str += '\0';
    char CharValue = CharC->getSExtValue();
    
    // Compute the offset.
    uint64_t i = 0;
    while (1) {
      if (i == Str.size())    // Didn't find the char.  strchr returns null.
        return Constant::getNullValue(CI->getType());
      // Did we find our match?
      if (Str[i] == CharValue)
        break;
      ++i;
    }
    
    // strchr(s+n,c)  -> gep(s+n+i,c)
    Value *Idx = ConstantInt::get(Type::Int64Ty, i);
    return B.CreateGEP(SrcStr, Idx, "strchr");
  }
};

//===---------------------------------------===//
// 'strcmp' Optimizations

struct VISIBILITY_HIDDEN StrCmpOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Verify the "strcmp" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || FT->getReturnType() != Type::Int32Ty ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty))
      return 0;
    
    Value *Str1P = CI->getOperand(1), *Str2P = CI->getOperand(2);
    if (Str1P == Str2P)      // strcmp(x,x)  -> 0
      return ConstantInt::get(CI->getType(), 0);
    
    std::string Str1, Str2;
    bool HasStr1 = GetConstantStringInfo(Str1P, Str1);
    bool HasStr2 = GetConstantStringInfo(Str2P, Str2);
    
    if (HasStr1 && Str1.empty()) // strcmp("", x) -> *x
      return B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"), CI->getType());
    
    if (HasStr2 && Str2.empty()) // strcmp(x,"") -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());
    
    // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2)
      return ConstantInt::get(CI->getType(), strcmp(Str1.c_str(),Str2.c_str()));
    return 0;
  }
};

//===---------------------------------------===//
// 'strncmp' Optimizations

struct VISIBILITY_HIDDEN StrNCmpOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Verify the "strncmp" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != Type::Int32Ty ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty) ||
        !isa<IntegerType>(FT->getParamType(2)))
      return 0;
    
    Value *Str1P = CI->getOperand(1), *Str2P = CI->getOperand(2);
    if (Str1P == Str2P)      // strncmp(x,x,n)  -> 0
      return ConstantInt::get(CI->getType(), 0);
    
    // Get the length argument if it is constant.
    uint64_t Length;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getOperand(3)))
      Length = LengthArg->getZExtValue();
    else
      return 0;
    
    if (Length == 0) // strncmp(x,y,0)   -> 0
      return ConstantInt::get(CI->getType(), 0);
    
    std::string Str1, Str2;
    bool HasStr1 = GetConstantStringInfo(Str1P, Str1);
    bool HasStr2 = GetConstantStringInfo(Str2P, Str2);
    
    if (HasStr1 && Str1.empty())  // strncmp("", x, n) -> *x
      return B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"), CI->getType());
    
    if (HasStr2 && Str2.empty())  // strncmp(x, "", n) -> *x
      return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());
    
    // strncmp(x, y)  -> cnst  (if both x and y are constant strings)
    if (HasStr1 && HasStr2)
      return ConstantInt::get(CI->getType(),
                              strncmp(Str1.c_str(), Str2.c_str(), Length));
    return 0;
  }
};


//===---------------------------------------===//
// 'strcpy' Optimizations

struct VISIBILITY_HIDDEN StrCpyOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Verify the "strcpy" function prototype.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty))
      return 0;
    
    Value *Dst = CI->getOperand(1), *Src = CI->getOperand(2);
    if (Dst == Src)      // strcpy(x,x)  -> x
      return Src;
    
    // See if we can get the length of the input string.
    uint64_t Len = GetStringLength(Src);
    if (Len == 0) return false;
    
    // We have enough information to now generate the memcpy call to do the
    // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
    EmitMemCpy(Dst, Src, ConstantInt::get(TD->getIntPtrType(), Len), 1, B);
    return Dst;
  }
};



//===---------------------------------------===//
// 'strlen' Optimizations

struct VISIBILITY_HIDDEN StrLenOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 ||
        FT->getParamType(0) != PointerType::getUnqual(Type::Int8Ty) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;
    
    Value *Src = CI->getOperand(1);

    // Constant folding: strlen("xyz") -> 3
    if (uint64_t Len = GetStringLength(Src))
      return ConstantInt::get(CI->getType(), Len-1);

    // Handle strlen(p) != 0.
    if (!IsOnlyUsedInZeroEqualityComparison(CI)) return 0;

    // strlen(x) != 0 --> *x != 0
    // strlen(x) == 0 --> *x == 0
    return B.CreateZExt(B.CreateLoad(Src, "strlenfirst"), CI->getType());
  }
};

//===---------------------------------------===//
// 'memcmp' Optimizations

struct VISIBILITY_HIDDEN MemCmpOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        FT->getReturnType() != Type::Int32Ty)
      return 0;
    
    Value *LHS = CI->getOperand(1), *RHS = CI->getOperand(2);
    
    if (LHS == RHS)  // memcmp(s,s,x) -> 0
      return Constant::getNullValue(CI->getType());
    
    // Make sure we have a constant length.
    ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getOperand(3));
    if (!LenC) return false;
    uint64_t Len = LenC->getZExtValue();
    
    if (Len == 0) // memcmp(s1,s2,0) -> 0
      return Constant::getNullValue(CI->getType());

    if (Len == 1) { // memcmp(S1,S2,1) -> *LHS - *RHS
      Value *LHSV = B.CreateLoad(CastToCStr(LHS, B), "lhsv");
      Value *RHSV = B.CreateLoad(CastToCStr(RHS, B), "rhsv");
      return B.CreateZExt(B.CreateSub(LHSV, RHSV, "chardiff"), CI->getType());
    }
    
    // memcmp(S1,S2,2) != 0 -> (*(short*)LHS ^ *(short*)RHS)  != 0
    // memcmp(S1,S2,4) != 0 -> (*(int*)LHS ^ *(int*)RHS)  != 0
    if ((Len == 2 || Len == 4) && IsOnlyUsedInZeroEqualityComparison(CI)) {
      LHS = B.CreateBitCast(LHS, PointerType::getUnqual(Type::Int16Ty), "tmp");
      RHS = B.CreateBitCast(RHS, LHS->getType(), "tmp");
      LoadInst *LHSV = B.CreateLoad(LHS, "lhsv");
      LoadInst *RHSV = B.CreateLoad(RHS, "rhsv");
      LHSV->setAlignment(1); RHSV->setAlignment(1);  // Unaligned loads.
      return B.CreateZExt(B.CreateXor(LHSV, RHSV, "shortdiff"), CI->getType());
    }
    
    return 0;
  }
};

//===---------------------------------------===//
// 'memcpy' Optimizations

struct VISIBILITY_HIDDEN MemCpyOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 3 || FT->getReturnType() != FT->getParamType(0) ||
        !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        FT->getParamType(2) != TD->getIntPtrType())
      return 0;

    // memcpy(x, y, n) -> llvm.memcpy(x, y, n, 1)
    EmitMemCpy(CI->getOperand(1), CI->getOperand(2), CI->getOperand(3), 1, B);
    return CI->getOperand(1);
  }
};

//===----------------------------------------------------------------------===//
// Math Library Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'pow*' Optimizations

struct VISIBILITY_HIDDEN PowOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != FT->getParamType(1) ||
        !FT->getParamType(0)->isFloatingPoint())
      return 0;
    
    Value *Op1 = CI->getOperand(1), *Op2 = CI->getOperand(2);
    if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
      if (Op1C->isExactlyValue(1.0))  // pow(1.0, x) -> 1.0
        return Op1C;
      if (Op1C->isExactlyValue(2.0))  // pow(2.0, x) -> exp2(x)
        return EmitUnaryFloatFnCall(Op2, "exp2", B);
    }
    
    ConstantFP *Op2C = dyn_cast<ConstantFP>(Op2);
    if (Op2C == 0) return 0;
    
    if (Op2C->getValueAPF().isZero())  // pow(x, 0.0) -> 1.0
      return ConstantFP::get(CI->getType(), 1.0);
    
    if (Op2C->isExactlyValue(0.5)) {
      // FIXME: This is not safe for -0.0 and -inf.  This can only be done when
      // 'unsafe' math optimizations are allowed.
      // x    pow(x, 0.5)  sqrt(x)
      // ---------------------------------------------
      // -0.0    +0.0       -0.0
      // -inf    +inf       NaN
#if 0
      // pow(x, 0.5) -> sqrt(x)
      return B.CreateCall(get_sqrt(), Op1, "sqrt");
#endif
    }
    
    if (Op2C->isExactlyValue(1.0))  // pow(x, 1.0) -> x
      return Op1;
    if (Op2C->isExactlyValue(2.0))  // pow(x, 2.0) -> x*x
      return B.CreateMul(Op1, Op1, "pow2");
    if (Op2C->isExactlyValue(-1.0)) // pow(x, -1.0) -> 1.0/x
      return B.CreateFDiv(ConstantFP::get(CI->getType(), 1.0), Op1, "powrecip");
    return 0;
  }
};

//===---------------------------------------===//
// Double -> Float Shrinking Optimizations for Unary Functions like 'floor'

struct VISIBILITY_HIDDEN UnaryDoubleFPOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 1 || FT->getReturnType() != Type::DoubleTy ||
        FT->getParamType(0) != Type::DoubleTy)
      return 0;
    
    // If this is something like 'floor((double)floatval)', convert to floorf.
    FPExtInst *Cast = dyn_cast<FPExtInst>(CI->getOperand(1));
    if (Cast == 0 || Cast->getOperand(0)->getType() != Type::FloatTy)
      return 0;

    // floor((double)floatval) -> (double)floorf(floatval)
    Value *V = Cast->getOperand(0);
    V = EmitUnaryFloatFnCall(V, Callee->getNameStart(), B);
    return B.CreateFPExt(V, Type::DoubleTy);
  }
};

//===----------------------------------------------------------------------===//
// Integer Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'ffs*' Optimizations

struct VISIBILITY_HIDDEN FFSOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    if (FT->getNumParams() != 1 || FT->getReturnType() != Type::Int32Ty ||
        !isa<IntegerType>(FT->getParamType(0)))
      return 0;
    
    Value *Op = CI->getOperand(1);
    
    // Constant fold.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op)) {
      if (CI->getValue() == 0)  // ffs(0) -> 0.
        return Constant::getNullValue(CI->getType());
      return ConstantInt::get(Type::Int32Ty, // ffs(c) -> cttz(c)+1
                              CI->getValue().countTrailingZeros()+1);
    }
    
    // ffs(x) -> x != 0 ? (i32)llvm.cttz(x)+1 : 0
    const Type *ArgType = Op->getType();
    Value *F = Intrinsic::getDeclaration(Callee->getParent(),
                                         Intrinsic::cttz, &ArgType, 1);
    Value *V = B.CreateCall(F, Op, "cttz");
    V = B.CreateAdd(V, ConstantInt::get(Type::Int32Ty, 1), "tmp");
    V = B.CreateIntCast(V, Type::Int32Ty, false, "tmp");
    
    Value *Cond = B.CreateICmpNE(Op, Constant::getNullValue(ArgType), "tmp");
    return B.CreateSelect(Cond, V, ConstantInt::get(Type::Int32Ty, 0));
  }
};

//===---------------------------------------===//
// 'isdigit' Optimizations

struct VISIBILITY_HIDDEN IsDigitOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !isa<IntegerType>(FT->getReturnType()) ||
        FT->getParamType(0) != Type::Int32Ty)
      return 0;
    
    // isdigit(c) -> (c-'0') <u 10
    Value *Op = CI->getOperand(1);
    Op = B.CreateSub(Op, ConstantInt::get(Type::Int32Ty, '0'), "isdigittmp");
    Op = B.CreateICmpULT(Op, ConstantInt::get(Type::Int32Ty, 10), "isdigit");
    return B.CreateZExt(Op, CI->getType());
  }
};

//===---------------------------------------===//
// 'isascii' Optimizations

struct VISIBILITY_HIDDEN IsAsciiOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // We require integer(i32)
    if (FT->getNumParams() != 1 || !isa<IntegerType>(FT->getReturnType()) ||
        FT->getParamType(0) != Type::Int32Ty)
      return 0;
    
    // isascii(c) -> c <u 128
    Value *Op = CI->getOperand(1);
    Op = B.CreateICmpULT(Op, ConstantInt::get(Type::Int32Ty, 128), "isascii");
    return B.CreateZExt(Op, CI->getType());
  }
};

//===---------------------------------------===//
// 'toascii' Optimizations

struct VISIBILITY_HIDDEN ToAsciiOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    const FunctionType *FT = Callee->getFunctionType();
    // We require i32(i32)
    if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
        FT->getParamType(0) != Type::Int32Ty)
      return 0;
    
    // isascii(c) -> c & 0x7f
    return B.CreateAnd(CI->getOperand(1), ConstantInt::get(CI->getType(),0x7F));
  }
};

//===----------------------------------------------------------------------===//
// Formatting and IO Optimizations
//===----------------------------------------------------------------------===//

//===---------------------------------------===//
// 'printf' Optimizations

struct VISIBILITY_HIDDEN PrintFOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Require one fixed pointer argument and an integer/void result.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() < 1 || !isa<PointerType>(FT->getParamType(0)) ||
        !(isa<IntegerType>(FT->getReturnType()) ||
          FT->getReturnType() == Type::VoidTy))
      return 0;
    
    // Check for a fixed format string.
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(1), FormatStr))
      return false;

    // Empty format string -> noop.
    if (FormatStr.empty())  // Tolerate printf's declared void.
      return CI->use_empty() ? (Value*)CI : ConstantInt::get(CI->getType(), 0);
    
    // printf("x") -> putchar('x'), even for '%'.
    if (FormatStr.size() == 1) {
      EmitPutChar(ConstantInt::get(Type::Int32Ty, FormatStr[0]), B);
      return CI->use_empty() ? (Value*)CI : ConstantInt::get(CI->getType(), 1);
    }
    
    // printf("foo\n") --> puts("foo")
    if (FormatStr[FormatStr.size()-1] == '\n' &&
        FormatStr.find('%') == std::string::npos) {  // no format characters.
      // Create a string literal with no \n on it.  We expect the constant merge
      // pass to be run after this pass, to merge duplicate strings.
      FormatStr.erase(FormatStr.end()-1);
      Constant *C = ConstantArray::get(FormatStr, true);
      C = new GlobalVariable(C->getType(), true,GlobalVariable::InternalLinkage,
                             C, "str", Callee->getParent());
      EmitPutS(C, B);
      return CI->use_empty() ? (Value*)CI : 
                          ConstantInt::get(CI->getType(), FormatStr.size()+1);
    }
    
    // Optimize specific format strings.
    // printf("%c", chr) --> putchar(*(i8*)dst)
    if (FormatStr == "%c" && CI->getNumOperands() > 2 &&
        isa<IntegerType>(CI->getOperand(2)->getType())) {
      EmitPutChar(CI->getOperand(2), B);
      return CI->use_empty() ? (Value*)CI : ConstantInt::get(CI->getType(), 1);
    }
    
    // printf("%s\n", str) --> puts(str)
    if (FormatStr == "%s\n" && CI->getNumOperands() > 2 &&
        isa<PointerType>(CI->getOperand(2)->getType()) &&
        CI->use_empty()) {
      EmitPutS(CI->getOperand(2), B);
      return CI;
    }
    return 0;
  }
};

//===---------------------------------------===//
// 'sprintf' Optimizations

struct VISIBILITY_HIDDEN SPrintFOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Require two fixed pointer arguments and an integer result.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;

    // Check for a fixed format string.
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(2), FormatStr))
      return false;
    
    // If we just have a format string (nothing else crazy) transform it.
    if (CI->getNumOperands() == 3) {
      // Make sure there's no % in the constant array.  We could try to handle
      // %% -> % in the future if we cared.
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')
          return 0; // we found a format specifier, bail out.
      
      // sprintf(str, fmt) -> llvm.memcpy(str, fmt, strlen(fmt)+1, 1)
      EmitMemCpy(CI->getOperand(1), CI->getOperand(2), // Copy the nul byte.
                 ConstantInt::get(TD->getIntPtrType(), FormatStr.size()+1),1,B);
      return ConstantInt::get(CI->getType(), FormatStr.size());
    }
    
    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' || CI->getNumOperands() <4)
      return 0;
    
    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // sprintf(dst, "%c", chr) --> *(i8*)dst = chr
      if (!isa<IntegerType>(CI->getOperand(3)->getType())) return 0;
      Value *V = B.CreateTrunc(CI->getOperand(3), Type::Int8Ty, "char");
      B.CreateStore(V, CastToCStr(CI->getOperand(1), B));
      return ConstantInt::get(CI->getType(), 1);
    }
    
    if (FormatStr[1] == 's') {
      // sprintf(dest, "%s", str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
      if (!isa<PointerType>(CI->getOperand(3)->getType())) return 0;

      Value *Len = EmitStrLen(CI->getOperand(3), B);
      Value *IncLen = B.CreateAdd(Len, ConstantInt::get(Len->getType(), 1),
                                  "leninc");
      EmitMemCpy(CI->getOperand(1), CI->getOperand(3), IncLen, 1, B);
      
      // The sprintf result is the unincremented number of bytes in the string.
      return B.CreateIntCast(Len, CI->getType(), false);
    }
    return 0;
  }
};

//===---------------------------------------===//
// 'fwrite' Optimizations

struct VISIBILITY_HIDDEN FWriteOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Require a pointer, an integer, an integer, a pointer, returning integer.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 4 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<IntegerType>(FT->getParamType(1)) ||
        !isa<IntegerType>(FT->getParamType(2)) ||
        !isa<PointerType>(FT->getParamType(3)) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;
    
    // Get the element size and count.
    ConstantInt *SizeC = dyn_cast<ConstantInt>(CI->getOperand(2));
    ConstantInt *CountC = dyn_cast<ConstantInt>(CI->getOperand(3));
    if (!SizeC || !CountC) return 0;
    uint64_t Bytes = SizeC->getZExtValue()*CountC->getZExtValue();
    
    // If this is writing zero records, remove the call (it's a noop).
    if (Bytes == 0)
      return ConstantInt::get(CI->getType(), 0);
    
    // If this is writing one byte, turn it into fputc.
    if (Bytes == 1) {  // fwrite(S,1,1,F) -> fputc(S[0],F)
      Value *Char = B.CreateLoad(CastToCStr(CI->getOperand(1), B), "char");
      EmitFPutC(Char, CI->getOperand(4), B);
      return ConstantInt::get(CI->getType(), 1);
    }

    return 0;
  }
};

//===---------------------------------------===//
// 'fputs' Optimizations

struct VISIBILITY_HIDDEN FPutsOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Require two pointers.  Also, we can't optimize if return value is used.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        !CI->use_empty())
      return 0;
    
    // fputs(s,F) --> fwrite(s,1,strlen(s),F)
    uint64_t Len = GetStringLength(CI->getOperand(1));
    if (!Len) return false;
    EmitFWrite(CI->getOperand(1), ConstantInt::get(TD->getIntPtrType(), Len-1),
               CI->getOperand(2), B);
    return CI;  // Known to have no uses (see above).
  }
};

//===---------------------------------------===//
// 'fprintf' Optimizations

struct VISIBILITY_HIDDEN FPrintFOpt : public LibCallOptimization {
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder &B) {
    // Require two fixed paramters as pointers and integer result.
    const FunctionType *FT = Callee->getFunctionType();
    if (FT->getNumParams() != 2 || !isa<PointerType>(FT->getParamType(0)) ||
        !isa<PointerType>(FT->getParamType(1)) ||
        !isa<IntegerType>(FT->getReturnType()))
      return 0;
    
    // All the optimizations depend on the format string.
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(2), FormatStr))
      return false;

    // fprintf(F, "foo") --> fwrite("foo", 3, 1, F)
    if (CI->getNumOperands() == 3) {
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')  // Could handle %% -> % if we cared.
          return false; // We found a format specifier.
      
      EmitFWrite(CI->getOperand(2), ConstantInt::get(TD->getIntPtrType(),
                                                     FormatStr.size()),
                 CI->getOperand(1), B);
      return ConstantInt::get(CI->getType(), FormatStr.size());
    }
    
    // The remaining optimizations require the format string to be "%s" or "%c"
    // and have an extra operand.
    if (FormatStr.size() != 2 || FormatStr[0] != '%' || CI->getNumOperands() <4)
      return 0;
    
    // Decode the second character of the format string.
    if (FormatStr[1] == 'c') {
      // fprintf(F, "%c", chr) --> *(i8*)dst = chr
      if (!isa<IntegerType>(CI->getOperand(3)->getType())) return 0;
      EmitFPutC(CI->getOperand(3), CI->getOperand(1), B);
      return ConstantInt::get(CI->getType(), 1);
    }
    
    if (FormatStr[1] == 's') {
      // fprintf(F, "%s", str) -> fputs(str, F)
      if (!isa<PointerType>(CI->getOperand(3)->getType()) || !CI->use_empty())
        return 0;
      EmitFPutS(CI->getOperand(3), CI->getOperand(1), B);
      return CI;
    }
    return 0;
  }
};


//===----------------------------------------------------------------------===//
// SimplifyLibCalls Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// This pass optimizes well known library functions from libc and libm.
  ///
  class VISIBILITY_HIDDEN SimplifyLibCalls : public FunctionPass {
    StringMap<LibCallOptimization*> Optimizations;
    // Miscellaneous LibCall Optimizations
    ExitOpt Exit; 
    // String and Memory LibCall Optimizations
    StrCatOpt StrCat; StrChrOpt StrChr; StrCmpOpt StrCmp; StrNCmpOpt StrNCmp;
    StrCpyOpt StrCpy; StrLenOpt StrLen; MemCmpOpt MemCmp; MemCpyOpt  MemCpy;
    // Math Library Optimizations
    PowOpt Pow; UnaryDoubleFPOpt UnaryDoubleFP;
    // Integer Optimizations
    FFSOpt FFS; IsDigitOpt IsDigit; IsAsciiOpt IsAscii; ToAsciiOpt ToAscii;
    // Formatting and IO Optimizations
    SPrintFOpt SPrintF; PrintFOpt PrintF;
    FWriteOpt FWrite; FPutsOpt FPuts; FPrintFOpt FPrintF;
  public:
    static char ID; // Pass identification
    SimplifyLibCalls() : FunctionPass((intptr_t)&ID) {}

    void InitOptimizations();
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
    }
  };
  char SimplifyLibCalls::ID = 0;
} // end anonymous namespace.

static RegisterPass<SimplifyLibCalls>
X("simplify-libcalls", "Simplify well-known library calls");

// Public interface to the Simplify LibCalls pass.
FunctionPass *llvm::createSimplifyLibCallsPass() {
  return new SimplifyLibCalls(); 
}

/// Optimizations - Populate the Optimizations map with all the optimizations
/// we know.
void SimplifyLibCalls::InitOptimizations() {
  // Miscellaneous LibCall Optimizations
  Optimizations["exit"] = &Exit;
  
  // String and Memory LibCall Optimizations
  Optimizations["strcat"] = &StrCat;
  Optimizations["strchr"] = &StrChr;
  Optimizations["strcmp"] = &StrCmp;
  Optimizations["strncmp"] = &StrNCmp;
  Optimizations["strcpy"] = &StrCpy;
  Optimizations["strlen"] = &StrLen;
  Optimizations["memcmp"] = &MemCmp;
  Optimizations["memcpy"] = &MemCpy;
  
  // Math Library Optimizations
  Optimizations["powf"] = &Pow;
  Optimizations["pow"] = &Pow;
  Optimizations["powl"] = &Pow;
#ifdef HAVE_FLOORF
  Optimizations["floor"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_CEILF
  Optimizations["ceil"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_ROUNDF
  Optimizations["round"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_RINTF
  Optimizations["rint"] = &UnaryDoubleFP;
#endif
#ifdef HAVE_NEARBYINTF
  Optimizations["nearbyint"] = &UnaryDoubleFP;
#endif
  
  // Integer Optimizations
  Optimizations["ffs"] = &FFS;
  Optimizations["ffsl"] = &FFS;
  Optimizations["ffsll"] = &FFS;
  Optimizations["isdigit"] = &IsDigit;
  Optimizations["isascii"] = &IsAscii;
  Optimizations["toascii"] = &ToAscii;
  
  // Formatting and IO Optimizations
  Optimizations["sprintf"] = &SPrintF;
  Optimizations["printf"] = &PrintF;
  Optimizations["fwrite"] = &FWrite;
  Optimizations["fputs"] = &FPuts;
  Optimizations["fprintf"] = &FPrintF;
}


/// runOnFunction - Top level algorithm.
///
bool SimplifyLibCalls::runOnFunction(Function &F) {
  if (Optimizations.empty())
    InitOptimizations();
  
  const TargetData &TD = getAnalysis<TargetData>();
  
  IRBuilder Builder;

  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      // Ignore non-calls.
      CallInst *CI = dyn_cast<CallInst>(I++);
      if (!CI) continue;
      
      // Ignore indirect calls and calls to non-external functions.
      Function *Callee = CI->getCalledFunction();
      if (Callee == 0 || !Callee->isDeclaration() ||
          !(Callee->hasExternalLinkage() || Callee->hasDLLImportLinkage()))
        continue;
      
      // Ignore unknown calls.
      const char *CalleeName = Callee->getNameStart();
      StringMap<LibCallOptimization*>::iterator OMI =
        Optimizations.find(CalleeName, CalleeName+Callee->getNameLen());
      if (OMI == Optimizations.end()) continue;
      
      // Set the builder to the instruction after the call.
      Builder.SetInsertPoint(BB, I);
      
      // Try to optimize this call.
      Value *Result = OMI->second->OptimizeCall(CI, TD, Builder);
      if (Result == 0) continue;

      // Something changed!
      Changed = true;
      ++NumSimplified;
      
      // Inspect the instruction after the call (which was potentially just
      // added) next.
      I = CI; ++I;
      
      if (CI != Result && !CI->use_empty()) {
        CI->replaceAllUsesWith(Result);
        if (!Result->hasName())
          Result->takeName(CI);
      }
      CI->eraseFromParent();
    }
  }
  return Changed;
}


// TODO:
//   Additional cases that we need to add to this file:
//
// cbrt:
//   * cbrt(expN(X))  -> expN(x/3)
//   * cbrt(sqrt(x))  -> pow(x,1/6)
//   * cbrt(sqrt(x))  -> pow(x,1/9)
//
// cos, cosf, cosl:
//   * cos(-x)  -> cos(x)
//
// exp, expf, expl:
//   * exp(log(x))  -> x
//
// log, logf, logl:
//   * log(exp(x))   -> x
//   * log(x**y)     -> y*log(x)
//   * log(exp(y))   -> y*log(e)
//   * log(exp2(y))  -> y*log(2)
//   * log(exp10(y)) -> y*log(10)
//   * log(sqrt(x))  -> 0.5*log(x)
//   * log(pow(x,y)) -> y*log(x)
//
// lround, lroundf, lroundl:
//   * lround(cnst) -> cnst'
//
// memcmp:
//   * memcmp(x,y,l)   -> cnst
//      (if all arguments are constant and strlen(x) <= l and strlen(y) <= l)
//
// memmove:
//   * memmove(d,s,l,a) -> memcpy(d,s,l,a)
//       (if s is a global constant array)
//
// pow, powf, powl:
//   * pow(exp(x),y)  -> exp(x*y)
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// puts:
//   * puts("") -> putchar("\n")
//
// round, roundf, roundl:
//   * round(cnst) -> cnst'
//
// signbit:
//   * signbit(cnst) -> cnst'
//   * signbit(nncst) -> 0 (if pstv is a non-negative constant)
//
// sqrt, sqrtf, sqrtl:
//   * sqrt(expN(x))  -> expN(x*0.5)
//   * sqrt(Nroot(x)) -> pow(x,1/(2*N))
//   * sqrt(pow(x,y)) -> pow(|x|,y*0.5)
//
// stpcpy:
//   * stpcpy(str, "literal") ->
//           llvm.memcpy(str,"literal",strlen("literal")+1,1)
// strrchr:
//   * strrchr(s,c) -> reverse_offset_of_in(c,s)
//      (if c is a constant integer and s is a constant string)
//   * strrchr(s1,0) -> strchr(s1,0)
//
// strncat:
//   * strncat(x,y,0) -> x
//   * strncat(x,y,0) -> x (if strlen(y) = 0)
//   * strncat(x,y,l) -> strcat(x,y) (if y and l are constants an l > strlen(y))
//
// strncpy:
//   * strncpy(d,s,0) -> d
//   * strncpy(d,s,l) -> memcpy(d,s,l,1)
//      (if s and l are constants)
//
// strpbrk:
//   * strpbrk(s,a) -> offset_in_for(s,a)
//      (if s and a are both constant strings)
//   * strpbrk(s,"") -> 0
//   * strpbrk(s,a) -> strchr(s,a[0]) (if a is constant string of length 1)
//
// strspn, strcspn:
//   * strspn(s,a)   -> const_int (if both args are constant)
//   * strspn("",a)  -> 0
//   * strspn(s,"")  -> 0
//   * strcspn(s,a)  -> const_int (if both args are constant)
//   * strcspn("",a) -> 0
//   * strcspn(s,"") -> strlen(a)
//
// strstr:
//   * strstr(x,x)  -> x
//   * strstr(s1,s2) -> offset_of_s2_in(s1)
//       (if s1 and s2 are constant strings)
//
// tan, tanf, tanl:
//   * tan(atan(x)) -> x
//
// trunc, truncf, truncl:
//   * trunc(cnst) -> cnst'
//
//

//===-- llvm/LLVMContext.h - Class for managing "global" state --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVMContext, a container of "global" state in LLVM, such
// as the global type and constant uniquing tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LLVMCONTEXT_H
#define LLVM_LLVMCONTEXT_H

#include "llvm/Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm {

class APFloat;
class APInt;
class ArrayType;
class Constant;
class ConstantAggregateZero;
class ConstantArray;
class ConstantFP;
class ConstantInt;
class ConstantPointerNull;
class ConstantStruct;
class ConstantVector;
class FunctionType;
class IntegerType;
class LLVMContextImpl;
class MDNode;
class MDString;
class OpaqueType;
class PointerType;
class StringRef;
class StructType;
class Type;
class UndefValue;
class Use;
class Value;
class VectorType;

/// This is an important class for using LLVM in a threaded context.  It
/// (opaquely) owns and manages the core "global" data of LLVM's core 
/// infrastructure, including the type and constant uniquing tables.
/// LLVMContext itself provides no locking guarantees, so you should be careful
/// to have one context per thread.
class LLVMContext {
  LLVMContextImpl* pImpl;
  
  friend class ConstantInt;
  friend class ConstantFP;
  friend class ConstantStruct;
  friend class ConstantArray;
  friend class ConstantVector;
public:
  LLVMContext();
  ~LLVMContext();
  
  // Constant accessors
  Constant* getNullValue(const Type* Ty);
  
  /// @returns the value for an integer constant of the given type that has all
  /// its bits set to true.
  /// @brief Get the all ones value
  Constant* getAllOnesValue(const Type* Ty);
  
  // UndefValue accessors
  UndefValue* getUndef(const Type* Ty);
  
  // ConstantInt accessors
  ConstantInt* getTrue();
  ConstantInt* getFalse();
  
  // ConstantPointerNull accessors
  ConstantPointerNull* getConstantPointerNull(const PointerType* T);
                              
  // ConstantAggregateZero accessors
  ConstantAggregateZero* getConstantAggregateZero(const Type* Ty);
                             
  // MDNode accessors
  MDNode* getMDNode(Value* const* Vals, unsigned NumVals);
  
  // MDString accessors
  MDString* getMDString(const StringRef &Str);
  
  // FunctionType accessors
  FunctionType* getFunctionType(const Type* Result, bool isVarArg);
  FunctionType* getFunctionType(const Type* Result,
                                const std::vector<const Type*>& Params,
                                bool isVarArg);
                                
  // IntegerType accessors
  const IntegerType* getIntegerType(unsigned NumBits);
  
  // OpaqueType accessors
  OpaqueType* getOpaqueType();
  
  // StructType accessors
  StructType* getStructType(bool isPacked=false);
  StructType* getStructType(const std::vector<const Type*>& Params,
                            bool isPacked = false);
  StructType* getStructType(const Type* type, ...);
  
  // ArrayType accessors
  ArrayType* getArrayType(const Type* ElementType, uint64_t NumElements);
  
  // PointerType accessors
  PointerType* getPointerType(const Type* ElementType, unsigned AddressSpace);
  PointerType* getPointerTypeUnqual(const Type* ElementType);
  
  // VectorType accessors
  VectorType* getVectorType(const Type* ElementType, unsigned NumElements);
  VectorType* getVectorTypeInteger(const VectorType* VTy);
  VectorType* getVectorTypeExtendedElement(const VectorType* VTy);
  VectorType* getVectorTypeTruncatedElement(const VectorType* VTy);
  
  // Other helpers
  /// @brief Create a result type for fcmp/icmp
  const Type* makeCmpResultType(const Type* opnd_type);
  
  // Methods for erasing constants
  void erase(MDString *M);
  void erase(MDNode *M);
  void erase(ConstantAggregateZero *Z);
};

/// FOR BACKWARDS COMPATIBILITY - Returns a global context.
extern LLVMContext& getGlobalContext();

}

#endif

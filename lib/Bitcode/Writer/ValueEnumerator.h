//===-- Bitcode/Writer/ValueEnumerator.h - Number values --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class gives values and types Unique ID's.
//
//===----------------------------------------------------------------------===//

#ifndef VALUE_ENUMERATOR_H
#define VALUE_ENUMERATOR_H

#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {

class Value;
class Type;
class Module;
class Function;
class TypeSymbolTable;
class ValueSymbolTable;
class ConstantArray;

class ValueEnumerator {
public:
  // For each type, we remember its Type* and occurrence frequency.
  typedef std::vector<std::pair<const Type*, unsigned> > TypeList;

  // For each value, we remember its Value* and occurrence frequency.
  typedef std::vector<std::pair<const Value*, unsigned> > ValueList;
private:
  TypeList Types;
  
  typedef DenseMap<const Type*, unsigned> TypeMapType;
  TypeMapType TypeMap;

  ValueList Values;
  
  typedef DenseMap<const Value*, unsigned> ValueMapType;
  ValueMapType ValueMap;
  
  
  ValueEnumerator(const ValueEnumerator &);  // DO NOT IMPLEMENT
  void operator=(const ValueEnumerator &);   // DO NOT IMPLEMENT
public:
  ValueEnumerator(const Module *M);

  unsigned getValueID(const Value *V) const {
    ValueMapType::const_iterator I = ValueMap.find(V);
    assert(I != ValueMap.end() && "Value not in slotcalculator!");
    return I->second;
  }
  
  unsigned getTypeID(const Type *T) const {
    TypeMapType::const_iterator I = TypeMap.find(T);
    assert(I != TypeMap.end() && "Type not in ValueEnumerator!");
    return I->second-1;
  }


  const TypeList &getTypes() const { return Types; }

  /// incorporateFunction/purgeFunction - If you'd like to deal with a function,
  /// use these two methods to get its data into the ValueEnumerator!
  ///
  void incorporateFunction(const Function *F);
  void purgeFunction();

private:
  void EnumerateValue(const Value *V);
  void EnumerateType(const Type *T);
  
  void EnumerateTypeSymbolTable(const TypeSymbolTable &ST);
  void EnumerateValueSymbolTable(const ValueSymbolTable &ST);
};

} // End llvm namespace

#endif

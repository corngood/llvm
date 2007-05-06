//===-- ValueEnumerator.cpp - Number values and types for bitcode writer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueEnumerator class.
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ValueSymbolTable.h"
#include <algorithm>
using namespace llvm;

static bool isFirstClassType(const std::pair<const llvm::Type*,
                             unsigned int> &P) {
  return P.first->isFirstClassType();
}

static bool isIntegerValue(const std::pair<const Value*, unsigned> &V) {
  return isa<IntegerType>(V.first->getType());
}

static bool CompareByFrequency(const std::pair<const llvm::Type*,
                               unsigned int> &P1,
                               const std::pair<const llvm::Type*,
                               unsigned int> &P2) {
  return P1.second > P2.second;
}

/// ValueEnumerator - Enumerate module-level information.
ValueEnumerator::ValueEnumerator(const Module *M) {
  // Enumerate the global variables.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    EnumerateValue(I);

  // Enumerate the functions.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    EnumerateValue(I);

  // Enumerate the aliases.
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    EnumerateValue(I);
  
  // Remember what is the cutoff between globalvalue's and other constants.
  unsigned FirstConstant = Values.size();
  
  // Enumerate the global variable initializers.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    if (I->hasInitializer())
      EnumerateValue(I->getInitializer());

  // Enumerate the aliasees.
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    EnumerateValue(I->getAliasee());
  
  // Enumerate types used by the type symbol table.
  EnumerateTypeSymbolTable(M->getTypeSymbolTable());

  // Insert constants that are named at module level into the slot pool so that
  // the module symbol table can refer to them...
  EnumerateValueSymbolTable(M->getValueSymbolTable());
  
  // Enumerate types used by function bodies and argument lists.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    
    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I)
      EnumerateType(I->getType());
    
    for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E;++I){
        for (User::const_op_iterator OI = I->op_begin(), E = I->op_end(); 
             OI != E; ++OI)
          EnumerateType((*OI)->getType());
        EnumerateType(I->getType());
      }
  }
  
  // Optimize constant ordering.
  OptimizeConstants(FirstConstant, Values.size());
    
  // Sort the type table by frequency so that most commonly used types are early
  // in the table (have low bit-width).
  std::stable_sort(Types.begin(), Types.end(), CompareByFrequency);
    
  // Partition the Type ID's so that the first-class types occur before the
  // aggregate types.  This allows the aggregate types to be dropped from the
  // type table after parsing the global variable initializers.
  std::partition(Types.begin(), Types.end(), isFirstClassType);

  // Now that we rearranged the type table, rebuild TypeMap.
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    TypeMap[Types[i].first] = i+1;
}

// Optimize constant ordering.
struct CstSortPredicate {
  ValueEnumerator &VE;
  CstSortPredicate(ValueEnumerator &ve) : VE(ve) {}
  bool operator()(const std::pair<const Value*, unsigned> &LHS,
                  const std::pair<const Value*, unsigned> &RHS) {
    // Sort by plane.
    if (LHS.first->getType() != RHS.first->getType())
      return VE.getTypeID(LHS.first->getType()) < 
             VE.getTypeID(RHS.first->getType());
    // Then by frequency.
    return LHS.second > RHS.second;
  }
};

/// OptimizeConstants - Reorder constant pool for denser encoding.
void ValueEnumerator::OptimizeConstants(unsigned CstStart, unsigned CstEnd) {
  if (CstStart == CstEnd || CstStart+1 == CstEnd) return;
  
  CstSortPredicate P(*this);
  std::stable_sort(Values.begin()+CstStart, Values.begin()+CstEnd, P);
  
  // Ensure that integer constants are at the start of the constant pool.  This
  // is important so that GEP structure indices come before gep constant exprs.
  std::partition(Values.begin()+CstStart, Values.begin()+CstEnd,
                 isIntegerValue);
  
  // Rebuild the modified portion of ValueMap.
  for (; CstStart != CstEnd; ++CstStart)
    ValueMap[Values[CstStart].first] = CstStart+1;
}


/// EnumerateTypeSymbolTable - Insert all of the types in the specified symbol
/// table.
void ValueEnumerator::EnumerateTypeSymbolTable(const TypeSymbolTable &TST) {
  for (TypeSymbolTable::const_iterator TI = TST.begin(), TE = TST.end(); 
       TI != TE; ++TI)
    EnumerateType(TI->second);
}

/// EnumerateValueSymbolTable - Insert all of the values in the specified symbol
/// table into the values table.
void ValueEnumerator::EnumerateValueSymbolTable(const ValueSymbolTable &VST) {
  for (ValueSymbolTable::const_iterator VI = VST.begin(), VE = VST.end(); 
       VI != VE; ++VI)
    EnumerateValue(VI->getValue());
}

void ValueEnumerator::EnumerateValue(const Value *V) {
  assert(V->getType() != Type::VoidTy && "Can't insert void values!");
  
  // Check to see if it's already in!
  unsigned &ValueID = ValueMap[V];
  if (ValueID) {
    // Increment use count.
    Values[ValueID-1].second++;
    return;
  }
  
  // Add the value.
  Values.push_back(std::make_pair(V, 1U));
  ValueID = Values.size();

  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (isa<GlobalValue>(C)) {
      // Initializers for globals are handled explicitly elsewhere.
    } else if (isa<ConstantArray>(C) && cast<ConstantArray>(C)->isString()) {
      // Do not enumerate the initializers for an array of simple characters.
      // The initializers just polute the value table, and we emit the strings
      // specially.
    } else {
      // This makes sure that if a constant has uses (for example an array of
      // const ints), that they are inserted also.
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        EnumerateValue(*I);
    }
  }

  EnumerateType(V->getType());
}


void ValueEnumerator::EnumerateType(const Type *Ty) {
  unsigned &TypeID = TypeMap[Ty];
  
  if (TypeID) {
    // If we've already seen this type, just increase its occurrence count.
    Types[TypeID-1].second++;
    return;
  }
  
  // First time we saw this type, add it.
  Types.push_back(std::make_pair(Ty, 1U));
  TypeID = Types.size();
  
  // Enumerate subtypes.
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    EnumerateType(*I);
  
  // If this is a function type, enumerate the param attrs.
  if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty))
    EnumerateParamAttrs(FTy->getParamAttrs());
}

void ValueEnumerator::EnumerateParamAttrs(const ParamAttrsList *PAL) {
  if (PAL == 0) return;  // null is always 0.
  // Do a lookup.
  unsigned &Entry = ParamAttrMap[PAL];
  if (Entry == 0) {
    // Never saw this before, add it.
    ParamAttrs.push_back(PAL);
    Entry = ParamAttrs.size();
  }
}


/// PurgeAggregateValues - If there are any aggregate values at the end of the
/// value list, remove them and return the count of the remaining values.  If
/// there are none, return -1.
int ValueEnumerator::PurgeAggregateValues() {
  // If there are no aggregate values at the end of the list, return -1.
  if (Values.empty() || Values.back().first->getType()->isFirstClassType())
    return -1;
  
  // Otherwise, remove aggregate values...
  while (!Values.empty() && !Values.back().first->getType()->isFirstClassType())
    Values.pop_back();
  
  // ... and return the new size.
  return Values.size();
}

void ValueEnumerator::incorporateFunction(const Function &F) {
  NumModuleValues = Values.size();
  
  // Adding function arguments to the value table.
  for(Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end();
      I != E; ++I)
    EnumerateValue(I);

  FirstFuncConstantID = Values.size();
  
  // Add all function-level constants to the value table.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I)
      for (User::const_op_iterator OI = I->op_begin(), E = I->op_end(); 
           OI != E; ++OI) {
        if ((isa<Constant>(*OI) && !isa<GlobalValue>(*OI)) ||
            isa<InlineAsm>(*OI))
          EnumerateValue(*OI);
      }
    BasicBlocks.push_back(BB);
    ValueMap[BB] = BasicBlocks.size();
  }
  
  // Optimize the constant layout.
  OptimizeConstants(FirstFuncConstantID, Values.size());
  
  FirstInstID = Values.size();
  
  // Add all of the instructions.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      if (I->getType() != Type::VoidTy)
        EnumerateValue(I);
    }
  }
}

void ValueEnumerator::purgeFunction() {
  /// Remove purged values from the ValueMap.
  for (unsigned i = NumModuleValues, e = Values.size(); i != e; ++i)
    ValueMap.erase(Values[i].first);
  for (unsigned i = 0, e = BasicBlocks.size(); i != e; ++i)
    ValueMap.erase(BasicBlocks[i]);
    
  Values.resize(NumModuleValues);
  BasicBlocks.clear();
}


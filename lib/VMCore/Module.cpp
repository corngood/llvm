//===-- Module.cpp - Implement the Module class ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Module class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/InstrTypes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/LeakDetector.h"
#include "SymbolTableListTraitsImpl.h"
#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <map>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Methods to implement the globals and functions lists.
//

Function *ilist_traits<Function>::createSentinel() {
  FunctionType *FTy =
    FunctionType::get(Type::VoidTy, std::vector<const Type*>(), false);
  Function *Ret = new Function(FTy, GlobalValue::ExternalLinkage);
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}
GlobalVariable *ilist_traits<GlobalVariable>::createSentinel() {
  GlobalVariable *Ret = new GlobalVariable(Type::IntTy, false,
                                           GlobalValue::ExternalLinkage);
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}

iplist<Function> &ilist_traits<Function>::getList(Module *M) {
  return M->getFunctionList();
}
iplist<GlobalVariable> &ilist_traits<GlobalVariable>::getList(Module *M) {
  return M->getGlobalList();
}

// Explicit instantiations of SymbolTableListTraits since some of the methods
// are not in the public header file.
template class SymbolTableListTraits<GlobalVariable, Module, Module>;
template class SymbolTableListTraits<Function, Module, Module>;

//===----------------------------------------------------------------------===//
// Primitive Module methods.
//

Module::Module(const std::string &MID)
  : ModuleID(MID), DataLayout("") {
  FunctionList.setItemParent(this);
  FunctionList.setParent(this);
  GlobalList.setItemParent(this);
  GlobalList.setParent(this);
  SymTab = new SymbolTable();
}

Module::~Module() {
  dropAllReferences();
  GlobalList.clear();
  GlobalList.setParent(0);
  FunctionList.clear();
  FunctionList.setParent(0);
  LibraryList.clear();
  delete SymTab;
}

// Module::dump() - Allow printing from debugger
void Module::dump() const {
  print(std::cerr);
}

/// Target endian information...
Module::Endianness Module::getEndianness() const {
  std::string temp = DataLayout;
  
  while (temp.length() > 0) {
    std::string token = getToken(temp, "-");
    
    if (token[0] == 'e') {
      return LittleEndian;
    } else if (token[0] == 'E') {
      return BigEndian;
    }
  }
  
  return AnyEndianness;
}

void Module::setEndianness(Endianness E) {
  if (DataLayout.compare("") != 0 && E != AnyEndianness)
    DataLayout.insert(0, "-");
  
  if (E == LittleEndian)
    DataLayout.insert(0, "e");
  else if (E == BigEndian)
    DataLayout.insert(0, "E");
}

/// Target Pointer Size information...
Module::PointerSize Module::getPointerSize() const {
  std::string temp = DataLayout;
  
  while (temp.length() > 0) {
    std::string token = getToken(temp, "-");
    char signal = getToken(token, ":")[0];
    
    if (signal == 'p') {
      int size = atoi(getToken(token, ":").c_str());
      if (size == 32)
        return Pointer32;
      else if (size == 64)
        return Pointer64;
    }
  }
  
  return AnyPointerSize;
}

void Module::setPointerSize(PointerSize PS) {
  if (DataLayout.compare("") != 0 && PS != AnyPointerSize)
    DataLayout.insert(0, "-");
  
  if (PS == Pointer32)
    DataLayout.insert(0, "p:32:32");
  else if (PS == Pointer64)
    DataLayout.insert(0, "p:64:64");
}

//===----------------------------------------------------------------------===//
// Methods for easy access to the functions in the module.
//

// getOrInsertFunction - Look up the specified function in the module symbol
// table.  If it does not exist, add a prototype for the function and return
// it.  This is nice because it allows most passes to get away with not handling
// the symbol table directly for this common task.
//
Function *Module::getOrInsertFunction(const std::string &Name,
                                      const FunctionType *Ty) {
  SymbolTable &SymTab = getSymbolTable();

  // See if we have a definitions for the specified function already...
  if (Value *V = SymTab.lookup(PointerType::get(Ty), Name)) {
    return cast<Function>(V);      // Yup, got it
  } else {                         // Nope, add one
    Function *New = new Function(Ty, GlobalVariable::ExternalLinkage, Name);
    FunctionList.push_back(New);
    return New;                    // Return the new prototype...
  }
}

// getOrInsertFunction - Look up the specified function in the module symbol
// table.  If it does not exist, add a prototype for the function and return it.
// This version of the method takes a null terminated list of function
// arguments, which makes it easier for clients to use.
//
Function *Module::getOrInsertFunction(const std::string &Name,
                                      const Type *RetTy, ...) {
  va_list Args;
  va_start(Args, RetTy);

  // Build the list of argument types...
  std::vector<const Type*> ArgTys;
  while (const Type *ArgTy = va_arg(Args, const Type*))
    ArgTys.push_back(ArgTy);

  va_end(Args);

  // Build the function type and chain to the other getOrInsertFunction...
  return getOrInsertFunction(Name, FunctionType::get(RetTy, ArgTys, false));
}


// getFunction - Look up the specified function in the module symbol table.
// If it does not exist, return null.
//
Function *Module::getFunction(const std::string &Name, const FunctionType *Ty) {
  SymbolTable &SymTab = getSymbolTable();
  return cast_or_null<Function>(SymTab.lookup(PointerType::get(Ty), Name));
}


/// getMainFunction - This function looks up main efficiently.  This is such a
/// common case, that it is a method in Module.  If main cannot be found, a
/// null pointer is returned.
///
Function *Module::getMainFunction() {
  std::vector<const Type*> Params;

  // int main(void)...
  if (Function *F = getFunction("main", FunctionType::get(Type::IntTy,
                                                          Params, false)))
    return F;

  // void main(void)...
  if (Function *F = getFunction("main", FunctionType::get(Type::VoidTy,
                                                          Params, false)))
    return F;

  Params.push_back(Type::IntTy);

  // int main(int argc)...
  if (Function *F = getFunction("main", FunctionType::get(Type::IntTy,
                                                          Params, false)))
    return F;

  // void main(int argc)...
  if (Function *F = getFunction("main", FunctionType::get(Type::VoidTy,
                                                          Params, false)))
    return F;

  for (unsigned i = 0; i != 2; ++i) {  // Check argv and envp
    Params.push_back(PointerType::get(PointerType::get(Type::SByteTy)));

    // int main(int argc, char **argv)...
    if (Function *F = getFunction("main", FunctionType::get(Type::IntTy,
                                                            Params, false)))
      return F;

    // void main(int argc, char **argv)...
    if (Function *F = getFunction("main", FunctionType::get(Type::VoidTy,
                                                            Params, false)))
      return F;
  }

  // Ok, try to find main the hard way...
  return getNamedFunction("main");
}

/// getNamedFunction - Return the first function in the module with the
/// specified name, of arbitrary type.  This method returns null if a function
/// with the specified name is not found.
///
Function *Module::getNamedFunction(const std::string &Name) {
  // Loop over all of the functions, looking for the function desired
  Function *Found = 0;
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (I->getName() == Name)
      if (I->isExternal())
        Found = I;
      else
        return I;
  return Found; // Non-external function not found...
}

//===----------------------------------------------------------------------===//
// Methods for easy access to the global variables in the module.
//

/// getGlobalVariable - Look up the specified global variable in the module
/// symbol table.  If it does not exist, return null.  The type argument
/// should be the underlying type of the global, i.e., it should not have
/// the top-level PointerType, which represents the address of the global.
/// If AllowInternal is set to true, this function will return types that
/// have InternalLinkage. By default, these types are not returned.
///
GlobalVariable *Module::getGlobalVariable(const std::string &Name,
                                          const Type *Ty, bool AllowInternal) {
  if (Value *V = getSymbolTable().lookup(PointerType::get(Ty), Name)) {
    GlobalVariable *Result = cast<GlobalVariable>(V);
    if (AllowInternal || !Result->hasInternalLinkage())
      return Result;
  }
  return 0;
}

/// getNamedGlobal - Return the first global variable in the module with the
/// specified name, of arbitrary type.  This method returns null if a global
/// with the specified name is not found.
///
GlobalVariable *Module::getNamedGlobal(const std::string &Name) {
  // FIXME: This would be much faster with a symbol table that doesn't
  // discriminate based on type!
  for (global_iterator I = global_begin(), E = global_end();
       I != E; ++I)
    if (I->getName() == Name) 
      return I;
  return 0;
}



//===----------------------------------------------------------------------===//
// Methods for easy access to the types in the module.
//


// addTypeName - Insert an entry in the symbol table mapping Str to Type.  If
// there is already an entry for this name, true is returned and the symbol
// table is not modified.
//
bool Module::addTypeName(const std::string &Name, const Type *Ty) {
  SymbolTable &ST = getSymbolTable();

  if (ST.lookupType(Name)) return true;  // Already in symtab...

  // Not in symbol table?  Set the name with the Symtab as an argument so the
  // type knows what to update...
  ST.insert(Name, Ty);

  return false;
}

/// getTypeByName - Return the type with the specified name in this module, or
/// null if there is none by that name.
const Type *Module::getTypeByName(const std::string &Name) const {
  const SymbolTable &ST = getSymbolTable();
  return cast_or_null<Type>(ST.lookupType(Name));
}

// getTypeName - If there is at least one entry in the symbol table for the
// specified type, return it.
//
std::string Module::getTypeName(const Type *Ty) const {
  const SymbolTable &ST = getSymbolTable();

  SymbolTable::type_const_iterator TI = ST.type_begin();
  SymbolTable::type_const_iterator TE = ST.type_end();
  if ( TI == TE ) return ""; // No names for types

  while (TI != TE && TI->second != Ty)
    ++TI;

  if (TI != TE)  // Must have found an entry!
    return TI->first;
  return "";     // Must not have found anything...
}

//===----------------------------------------------------------------------===//
// Other module related stuff.
//


// dropAllReferences() - This function causes all the subelementss to "let go"
// of all references that they are maintaining.  This allows one to 'delete' a
// whole module at a time, even though there may be circular references... first
// all references are dropped, and all use counts go to zero.  Then everything
// is deleted for real.  Note that no operations are valid on an object that
// has "dropped all references", except operator delete.
//
void Module::dropAllReferences() {
  for(Module::iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();

  for(Module::global_iterator I = global_begin(), E = global_end(); I != E; ++I)
    I->dropAllReferences();
}


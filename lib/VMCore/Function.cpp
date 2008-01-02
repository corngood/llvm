//===-- Function.cpp - Implement the Global object classes ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Function class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/StringPool.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

BasicBlock *ilist_traits<BasicBlock>::createSentinel() {
  BasicBlock *Ret = new BasicBlock();
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}

iplist<BasicBlock> &ilist_traits<BasicBlock>::getList(Function *F) {
  return F->getBasicBlockList();
}

Argument *ilist_traits<Argument>::createSentinel() {
  Argument *Ret = new Argument(Type::Int32Ty);
  // This should not be garbage monitored.
  LeakDetector::removeGarbageObject(Ret);
  return Ret;
}

iplist<Argument> &ilist_traits<Argument>::getList(Function *F) {
  return F->getArgumentList();
}

// Explicit instantiations of SymbolTableListTraits since some of the methods
// are not in the public header file...
template class SymbolTableListTraits<Argument, Function>;
template class SymbolTableListTraits<BasicBlock, Function>;

//===----------------------------------------------------------------------===//
// Argument Implementation
//===----------------------------------------------------------------------===//

Argument::Argument(const Type *Ty, const std::string &Name, Function *Par)
  : Value(Ty, Value::ArgumentVal) {
  Parent = 0;

  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (Par)
    Par->getArgumentList().push_back(this);
  setName(Name);
}

void Argument::setParent(Function *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

//===----------------------------------------------------------------------===//
// Helper Methods in Function
//===----------------------------------------------------------------------===//

const FunctionType *Function::getFunctionType() const {
  return cast<FunctionType>(getType()->getElementType());
}

bool Function::isVarArg() const {
  return getFunctionType()->isVarArg();
}

const Type *Function::getReturnType() const {
  return getFunctionType()->getReturnType();
}

void Function::removeFromParent() {
  getParent()->getFunctionList().remove(this);
}

void Function::eraseFromParent() {
  getParent()->getFunctionList().erase(this);
}

/// @brief Determine whether the function has the given attribute.
bool Function::paramHasAttr(uint16_t i, unsigned attr) const {
  return ParamAttrs && ParamAttrs->paramHasAttr(i, (ParameterAttributes)attr);
}

/// @brief Determine if the function cannot return.
bool Function::doesNotReturn() const {
  return paramHasAttr(0, ParamAttr::NoReturn);
}

/// @brief Determine if the function cannot unwind.
bool Function::doesNotThrow() const {
  return paramHasAttr(0, ParamAttr::NoUnwind);
}

/// @brief Determine if the function does not access memory.
bool Function::doesNotAccessMemory() const {
  return paramHasAttr(0, ParamAttr::ReadNone);
}

/// @brief Determine if the function does not access or only reads memory.
bool Function::onlyReadsMemory() const {
  return doesNotAccessMemory() || paramHasAttr(0, ParamAttr::ReadOnly);
}

/// @brief Determine if the function returns a structure.
bool Function::isStructReturn() const {
  return paramHasAttr(1, ParamAttr::StructRet);
}

//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//

Function::Function(const FunctionType *Ty, LinkageTypes Linkage,
                   const std::string &name, Module *ParentModule)
  : GlobalValue(PointerType::getUnqual(Ty), 
                Value::FunctionVal, 0, 0, Linkage, name),
    ParamAttrs(0) {
  SymTab = new ValueSymbolTable();

  assert((getReturnType()->isFirstClassType() ||getReturnType() == Type::VoidTy)
         && "LLVM functions cannot return aggregate values!");

  // If the function has arguments, mark them as lazily built.
  if (Ty->getNumParams())
    SubclassData = 1;   // Set the "has lazy arguments" bit.
  
  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getFunctionList().push_back(this);
}

Function::~Function() {
  dropAllReferences();    // After this it is safe to delete instructions.

  // Delete all of the method arguments and unlink from symbol table...
  ArgumentList.clear();
  delete SymTab;

  // Drop our reference to the parameter attributes, if any.
  if (ParamAttrs)
    ParamAttrs->dropRef();
  
  // Remove the function from the on-the-side collector table.
  clearCollector();
}

void Function::BuildLazyArguments() const {
  // Create the arguments vector, all arguments start out unnamed.
  const FunctionType *FT = getFunctionType();
  for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
    assert(FT->getParamType(i) != Type::VoidTy &&
           "Cannot have void typed arguments!");
    ArgumentList.push_back(new Argument(FT->getParamType(i)));
  }
  
  // Clear the lazy arguments bit.
  const_cast<Function*>(this)->SubclassData &= ~1;
}

size_t Function::arg_size() const {
  return getFunctionType()->getNumParams();
}
bool Function::arg_empty() const {
  return getFunctionType()->getNumParams() == 0;
}

void Function::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void Function::setParamAttrs(const ParamAttrsList *attrs) {
  // Avoid deleting the ParamAttrsList if they are setting the
  // attributes to the same list.
  if (ParamAttrs == attrs)
    return;

  // Drop reference on the old ParamAttrsList
  if (ParamAttrs)
    ParamAttrs->dropRef();

  // Add reference to the new ParamAttrsList
  if (attrs)
    attrs->addRef();

  // Set the new ParamAttrsList.
  ParamAttrs = attrs; 
}

// dropAllReferences() - This function causes all the subinstructions to "let
// go" of all references that they are maintaining.  This allows one to
// 'delete' a whole class at a time, even though there may be circular
// references... first all references are dropped, and all use counts go to
// zero.  Then everything is deleted for real.  Note that no operations are
// valid on an object that has "dropped all references", except operator
// delete.
//
void Function::dropAllReferences() {
  for (iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();
  BasicBlocks.clear();    // Delete all basic blocks...
}

// Maintain the collector name for each function in an on-the-side table. This
// saves allocating an additional word in Function for programs which do not use
// GC (i.e., most programs) at the cost of increased overhead for clients which
// do use GC.
static DenseMap<const Function*,PooledStringPtr> *CollectorNames;
static StringPool *CollectorNamePool;

bool Function::hasCollector() const {
  return CollectorNames && CollectorNames->count(this);
}

const char *Function::getCollector() const {
  assert(hasCollector() && "Function has no collector");
  return *(*CollectorNames)[this];
}

void Function::setCollector(const char *Str) {
  if (!CollectorNamePool)
    CollectorNamePool = new StringPool();
  if (!CollectorNames)
    CollectorNames = new DenseMap<const Function*,PooledStringPtr>();
  (*CollectorNames)[this] = CollectorNamePool->intern(Str);
}

void Function::clearCollector() {
  if (CollectorNames) {
    CollectorNames->erase(this);
    if (CollectorNames->empty()) {
      delete CollectorNames;
      CollectorNames = 0;
      if (CollectorNamePool->empty()) {
        delete CollectorNamePool;
        CollectorNamePool = 0;
      }
    }
  }
}

/// getIntrinsicID - This method returns the ID number of the specified
/// function, or Intrinsic::not_intrinsic if the function is not an
/// intrinsic, or if the pointer is null.  This value is always defined to be
/// zero to allow easy checking for whether a function is intrinsic or not.  The
/// particular intrinsic functions which correspond to this value are defined in
/// llvm/Intrinsics.h.
///
unsigned Function::getIntrinsicID(bool noAssert) const {
  const ValueName *ValName = this->getValueName();
  if (!ValName)
    return 0;
  unsigned Len = ValName->getKeyLength();
  const char *Name = ValName->getKeyData();
  
  if (Len < 5 || Name[4] != '.' || Name[0] != 'l' || Name[1] != 'l'
      || Name[2] != 'v' || Name[3] != 'm')
    return 0;  // All intrinsics start with 'llvm.'

  assert((Len != 5 || noAssert) && "'llvm.' is an invalid intrinsic name!");

#define GET_FUNCTION_RECOGNIZER
#include "llvm/Intrinsics.gen"
#undef GET_FUNCTION_RECOGNIZER
  assert(noAssert && "Invalid LLVM intrinsic name");
  return 0;
}

std::string Intrinsic::getName(ID id, const Type **Tys, unsigned numTys) { 
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  const char * const Table[] = {
    "not_intrinsic",
#define GET_INTRINSIC_NAME_TABLE
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_NAME_TABLE
  };
  if (numTys == 0)
    return Table[id];
  std::string Result(Table[id]);
  for (unsigned i = 0; i < numTys; ++i) 
    if (Tys[i])
      Result += "." + MVT::getValueTypeString(MVT::getValueType(Tys[i]));
  return Result;
}

const FunctionType *Intrinsic::getType(ID id, const Type **Tys, 
                                       unsigned numTys) {
  const Type *ResultTy = NULL;
  std::vector<const Type*> ArgTys;
  bool IsVarArg = false;
  
#define GET_INTRINSIC_GENERATOR
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_GENERATOR

  return FunctionType::get(ResultTy, ArgTys, IsVarArg); 
}

const ParamAttrsList *Intrinsic::getParamAttrs(ID id) {
  static const ParamAttrsList *IntrinsicAttributes[Intrinsic::num_intrinsics];

  if (IntrinsicAttributes[id])
    return IntrinsicAttributes[id];

  ParamAttrsVector Attrs;
  uint16_t Attr = ParamAttr::None;

#define GET_INTRINSIC_ATTRIBUTES
#include "llvm/Intrinsics.gen"
#undef GET_INTRINSIC_ATTRIBUTES

  // Intrinsics cannot throw exceptions.
  Attr |= ParamAttr::NoUnwind;

  Attrs.push_back(ParamAttrsWithIndex::get(0, Attr));
  IntrinsicAttributes[id] = ParamAttrsList::get(Attrs);
  return IntrinsicAttributes[id];
}

Function *Intrinsic::getDeclaration(Module *M, ID id, const Type **Tys, 
                                    unsigned numTys) {
  // There can never be multiple globals with the same name of different types,
  // because intrinsics must be a specific type.
  Function *F =
    cast<Function>(M->getOrInsertFunction(getName(id, Tys, numTys),
                                          getType(id, Tys, numTys)));
  F->setParamAttrs(getParamAttrs(id));
  return F;
}

Value *IntrinsicInst::StripPointerCasts(Value *Ptr) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
    if (CE->getOpcode() == Instruction::BitCast) {
      if (isa<PointerType>(CE->getOperand(0)->getType()))
        return StripPointerCasts(CE->getOperand(0));
    } else if (CE->getOpcode() == Instruction::GetElementPtr) {
      for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
        if (!CE->getOperand(i)->isNullValue())
          return Ptr;
      return StripPointerCasts(CE->getOperand(0));
    }
    return Ptr;
  }

  if (BitCastInst *CI = dyn_cast<BitCastInst>(Ptr)) {
    if (isa<PointerType>(CI->getOperand(0)->getType()))
      return StripPointerCasts(CI->getOperand(0));
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    if (GEP->hasAllZeroIndices())
      return StripPointerCasts(GEP->getOperand(0));
  }
  return Ptr;
}

// vim: sw=2 ai

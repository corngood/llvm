//===-- Instructions.cpp - Implement the LLVM instructions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements all of the non-inline methods for the LLVM instruction
// classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/ParamAttrsList.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                            CallSite Class
//===----------------------------------------------------------------------===//

CallSite::CallSite(Instruction *C) {
  assert((isa<CallInst>(C) || isa<InvokeInst>(C)) && "Not a call!");
  I = C;
}
unsigned CallSite::getCallingConv() const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->getCallingConv();
  else
    return cast<InvokeInst>(I)->getCallingConv();
}
void CallSite::setCallingConv(unsigned CC) {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    CI->setCallingConv(CC);
  else
    cast<InvokeInst>(I)->setCallingConv(CC);
}
const ParamAttrsList* CallSite::getParamAttrs() const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->getParamAttrs();
  else
    return cast<InvokeInst>(I)->getParamAttrs();
}
void CallSite::setParamAttrs(const ParamAttrsList *PAL) {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    CI->setParamAttrs(PAL);
  else
    cast<InvokeInst>(I)->setParamAttrs(PAL);
}
bool CallSite::paramHasAttr(uint16_t i, ParameterAttributes attr) const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->paramHasAttr(i, attr);
  else
    return cast<InvokeInst>(I)->paramHasAttr(i, attr);
}
uint16_t CallSite::getParamAlignment(uint16_t i) const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->getParamAlignment(i);
  else
    return cast<InvokeInst>(I)->getParamAlignment(i);
}

bool CallSite::doesNotAccessMemory() const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->doesNotAccessMemory();
  else
    return cast<InvokeInst>(I)->doesNotAccessMemory();
}
bool CallSite::onlyReadsMemory() const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->onlyReadsMemory();
  else
    return cast<InvokeInst>(I)->onlyReadsMemory();
}
bool CallSite::doesNotThrow() const {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    return CI->doesNotThrow();
  else
    return cast<InvokeInst>(I)->doesNotThrow();
}
void CallSite::setDoesNotThrow(bool doesNotThrow) {
  if (CallInst *CI = dyn_cast<CallInst>(I))
    CI->setDoesNotThrow(doesNotThrow);
  else
    cast<InvokeInst>(I)->setDoesNotThrow(doesNotThrow);
}

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

// Out of line virtual method, so the vtable, etc has a home.
TerminatorInst::~TerminatorInst() {
}

// Out of line virtual method, so the vtable, etc has a home.
UnaryInstruction::~UnaryInstruction() {
}


//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

PHINode::PHINode(const PHINode &PN)
  : Instruction(PN.getType(), Instruction::PHI,
                new Use[PN.getNumOperands()], PN.getNumOperands()),
    ReservedSpace(PN.getNumOperands()) {
  Use *OL = OperandList;
  for (unsigned i = 0, e = PN.getNumOperands(); i != e; i+=2) {
    OL[i].init(PN.getOperand(i), this);
    OL[i+1].init(PN.getOperand(i+1), this);
  }
}

PHINode::~PHINode() {
  delete [] OperandList;
}

// removeIncomingValue - Remove an incoming value.  This is useful if a
// predecessor basic block is deleted.
Value *PHINode::removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty) {
  unsigned NumOps = getNumOperands();
  Use *OL = OperandList;
  assert(Idx*2 < NumOps && "BB not in PHI node!");
  Value *Removed = OL[Idx*2];

  // Move everything after this operand down.
  //
  // FIXME: we could just swap with the end of the list, then erase.  However,
  // client might not expect this to happen.  The code as it is thrashes the
  // use/def lists, which is kinda lame.
  for (unsigned i = (Idx+1)*2; i != NumOps; i += 2) {
    OL[i-2] = OL[i];
    OL[i-2+1] = OL[i+1];
  }

  // Nuke the last value.
  OL[NumOps-2].set(0);
  OL[NumOps-2+1].set(0);
  NumOperands = NumOps-2;

  // If the PHI node is dead, because it has zero entries, nuke it now.
  if (NumOps == 2 && DeletePHIIfEmpty) {
    // If anyone is using this PHI, make them use a dummy value instead...
    replaceAllUsesWith(UndefValue::get(getType()));
    eraseFromParent();
  }
  return Removed;
}

/// resizeOperands - resize operands - This adjusts the length of the operands
/// list according to the following behavior:
///   1. If NumOps == 0, grow the operand list in response to a push_back style
///      of operation.  This grows the number of ops by 1.5 times.
///   2. If NumOps > NumOperands, reserve space for NumOps operands.
///   3. If NumOps == NumOperands, trim the reserved space.
///
void PHINode::resizeOperands(unsigned NumOps) {
  if (NumOps == 0) {
    NumOps = (getNumOperands())*3/2;
    if (NumOps < 4) NumOps = 4;      // 4 op PHI nodes are VERY common.
  } else if (NumOps*2 > NumOperands) {
    // No resize needed.
    if (ReservedSpace >= NumOps) return;
  } else if (NumOps == NumOperands) {
    if (ReservedSpace == NumOps) return;
  } else {
    return;
  }

  ReservedSpace = NumOps;
  Use *NewOps = new Use[NumOps];
  Use *OldOps = OperandList;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      NewOps[i].init(OldOps[i], this);
      OldOps[i].set(0);
  }
  delete [] OldOps;
  OperandList = NewOps;
}

/// hasConstantValue - If the specified PHI node always merges together the same
/// value, return the value, otherwise return null.
///
Value *PHINode::hasConstantValue(bool AllowNonDominatingInstruction) const {
  // If the PHI node only has one incoming value, eliminate the PHI node...
  if (getNumIncomingValues() == 1) {
    if (getIncomingValue(0) != this)   // not  X = phi X
      return getIncomingValue(0);
    else
      return UndefValue::get(getType());  // Self cycle is dead.
  }
      
  // Otherwise if all of the incoming values are the same for the PHI, replace
  // the PHI node with the incoming value.
  //
  Value *InVal = 0;
  bool HasUndefInput = false;
  for (unsigned i = 0, e = getNumIncomingValues(); i != e; ++i)
    if (isa<UndefValue>(getIncomingValue(i))) {
      HasUndefInput = true;
    } else if (getIncomingValue(i) != this) { // Not the PHI node itself...
      if (InVal && getIncomingValue(i) != InVal)
        return 0;  // Not the same, bail out.
      else
        InVal = getIncomingValue(i);
    }
  
  // The only case that could cause InVal to be null is if we have a PHI node
  // that only has entries for itself.  In this case, there is no entry into the
  // loop, so kill the PHI.
  //
  if (InVal == 0) InVal = UndefValue::get(getType());
  
  // If we have a PHI node like phi(X, undef, X), where X is defined by some
  // instruction, we cannot always return X as the result of the PHI node.  Only
  // do this if X is not an instruction (thus it must dominate the PHI block),
  // or if the client is prepared to deal with this possibility.
  if (HasUndefInput && !AllowNonDominatingInstruction)
    if (Instruction *IV = dyn_cast<Instruction>(InVal))
      // If it's in the entry block, it dominates everything.
      if (IV->getParent() != &IV->getParent()->getParent()->getEntryBlock() ||
          isa<InvokeInst>(IV))
        return 0;   // Cannot guarantee that InVal dominates this PHINode.

  // All of the incoming values are the same, return the value now.
  return InVal;
}


//===----------------------------------------------------------------------===//
//                        CallInst Implementation
//===----------------------------------------------------------------------===//

CallInst::~CallInst() {
  delete [] OperandList;
  if (ParamAttrs)
    ParamAttrs->dropRef();
}

void CallInst::init(Value *Func, Value* const *Params, unsigned NumParams) {
  ParamAttrs = 0;
  NumOperands = NumParams+1;
  Use *OL = OperandList = new Use[NumParams+1];
  OL[0].init(Func, this);

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert((NumParams == FTy->getNumParams() ||
          (FTy->isVarArg() && NumParams > FTy->getNumParams())) &&
         "Calling a function with bad signature!");
  for (unsigned i = 0; i != NumParams; ++i) {
    assert((i >= FTy->getNumParams() || 
            FTy->getParamType(i) == Params[i]->getType()) &&
           "Calling a function with a bad signature!");
    OL[i+1].init(Params[i], this);
  }
}

void CallInst::init(Value *Func, Value *Actual1, Value *Actual2) {
  ParamAttrs = 0;
  NumOperands = 3;
  Use *OL = OperandList = new Use[3];
  OL[0].init(Func, this);
  OL[1].init(Actual1, this);
  OL[2].init(Actual2, this);

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert((FTy->getNumParams() == 2 ||
          (FTy->isVarArg() && FTy->getNumParams() < 2)) &&
         "Calling a function with bad signature");
  assert((0 >= FTy->getNumParams() || 
          FTy->getParamType(0) == Actual1->getType()) &&
         "Calling a function with a bad signature!");
  assert((1 >= FTy->getNumParams() || 
          FTy->getParamType(1) == Actual2->getType()) &&
         "Calling a function with a bad signature!");
}

void CallInst::init(Value *Func, Value *Actual) {
  ParamAttrs = 0;
  NumOperands = 2;
  Use *OL = OperandList = new Use[2];
  OL[0].init(Func, this);
  OL[1].init(Actual, this);

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert((FTy->getNumParams() == 1 ||
          (FTy->isVarArg() && FTy->getNumParams() == 0)) &&
         "Calling a function with bad signature");
  assert((0 == FTy->getNumParams() || 
          FTy->getParamType(0) == Actual->getType()) &&
         "Calling a function with a bad signature!");
}

void CallInst::init(Value *Func) {
  ParamAttrs = 0;
  NumOperands = 1;
  Use *OL = OperandList = new Use[1];
  OL[0].init(Func, this);

  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Func->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert(FTy->getNumParams() == 0 && "Calling a function with bad signature");
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, InsertBefore) {
  init(Func, Actual);
  setName(Name);
}

CallInst::CallInst(Value *Func, Value* Actual, const std::string &Name,
                   BasicBlock  *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, InsertAtEnd) {
  init(Func, Actual);
  setName(Name);
}
CallInst::CallInst(Value *Func, const std::string &Name,
                   Instruction *InsertBefore)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, InsertBefore) {
  init(Func);
  setName(Name);
}

CallInst::CallInst(Value *Func, const std::string &Name,
                   BasicBlock *InsertAtEnd)
  : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                   ->getElementType())->getReturnType(),
                Instruction::Call, 0, 0, InsertAtEnd) {
  init(Func);
  setName(Name);
}

CallInst::CallInst(const CallInst &CI)
  : Instruction(CI.getType(), Instruction::Call, new Use[CI.getNumOperands()],
                CI.getNumOperands()),
    ParamAttrs(0) {
  setParamAttrs(CI.getParamAttrs());
  SubclassData = CI.SubclassData;
  Use *OL = OperandList;
  Use *InOL = CI.OperandList;
  for (unsigned i = 0, e = CI.getNumOperands(); i != e; ++i)
    OL[i].init(InOL[i], this);
}

void CallInst::setParamAttrs(const ParamAttrsList *newAttrs) {
  if (ParamAttrs == newAttrs)
    return;

  if (ParamAttrs)
    ParamAttrs->dropRef();

  if (newAttrs)
    newAttrs->addRef();

  ParamAttrs = newAttrs; 
}

bool CallInst::paramHasAttr(uint16_t i, ParameterAttributes attr) const {
  if (ParamAttrs && ParamAttrs->paramHasAttr(i, attr))
    return true;
  if (const Function *F = getCalledFunction())
    return F->paramHasAttr(i, attr);
  return false;
}

uint16_t CallInst::getParamAlignment(uint16_t i) const {
  if (ParamAttrs && ParamAttrs->getParamAlignment(i))
    return ParamAttrs->getParamAlignment(i);
  if (const Function *F = getCalledFunction())
    return F->getParamAlignment(i);
  return 0;
}

/// @brief Determine if the call does not access memory.
bool CallInst::doesNotAccessMemory() const {
  return paramHasAttr(0, ParamAttr::ReadNone);
}

/// @brief Determine if the call does not access or only reads memory.
bool CallInst::onlyReadsMemory() const {
  return doesNotAccessMemory() || paramHasAttr(0, ParamAttr::ReadOnly);
}

/// @brief Determine if the call cannot return.
bool CallInst::doesNotReturn() const {
  return paramHasAttr(0, ParamAttr::NoReturn);
}

/// @brief Determine if the call cannot unwind.
bool CallInst::doesNotThrow() const {
  return paramHasAttr(0, ParamAttr::NoUnwind);
}

/// @brief Determine if the call returns a structure.
bool CallInst::isStructReturn() const {
  // Be friendly and also check the callee.
  return paramHasAttr(1, ParamAttr::StructRet);
}

/// @brief Determine if any call argument is an aggregate passed by value.
bool CallInst::hasByValArgument() const {
  if (ParamAttrs && ParamAttrs->hasAttrSomewhere(ParamAttr::ByVal))
    return true;
  // Be consistent with other methods and check the callee too.
  if (const Function *F = getCalledFunction())
    if (const ParamAttrsList *PAL = F->getParamAttrs())
      return PAL->hasAttrSomewhere(ParamAttr::ByVal);
  return false;
}

void CallInst::setDoesNotThrow(bool doesNotThrow) {
  const ParamAttrsList *PAL = getParamAttrs();
  if (doesNotThrow)
    PAL = ParamAttrsList::includeAttrs(PAL, 0, ParamAttr::NoUnwind);
  else
    PAL = ParamAttrsList::excludeAttrs(PAL, 0, ParamAttr::NoUnwind);
  setParamAttrs(PAL);
}


//===----------------------------------------------------------------------===//
//                        InvokeInst Implementation
//===----------------------------------------------------------------------===//

InvokeInst::~InvokeInst() {
  delete [] OperandList;
  if (ParamAttrs)
    ParamAttrs->dropRef();
}

void InvokeInst::init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
                      Value* const *Args, unsigned NumArgs) {
  ParamAttrs = 0;
  NumOperands = 3+NumArgs;
  Use *OL = OperandList = new Use[3+NumArgs];
  OL[0].init(Fn, this);
  OL[1].init(IfNormal, this);
  OL[2].init(IfException, this);
  const FunctionType *FTy =
    cast<FunctionType>(cast<PointerType>(Fn->getType())->getElementType());
  FTy = FTy;  // silence warning.

  assert(((NumArgs == FTy->getNumParams()) ||
          (FTy->isVarArg() && NumArgs > FTy->getNumParams())) &&
         "Calling a function with bad signature");

  for (unsigned i = 0, e = NumArgs; i != e; i++) {
    assert((i >= FTy->getNumParams() || 
            FTy->getParamType(i) == Args[i]->getType()) &&
           "Invoking a function with a bad signature!");
    
    OL[i+3].init(Args[i], this);
  }
}

InvokeInst::InvokeInst(const InvokeInst &II)
  : TerminatorInst(II.getType(), Instruction::Invoke,
                   new Use[II.getNumOperands()], II.getNumOperands()),
    ParamAttrs(0) {
  setParamAttrs(II.getParamAttrs());
  SubclassData = II.SubclassData;
  Use *OL = OperandList, *InOL = II.OperandList;
  for (unsigned i = 0, e = II.getNumOperands(); i != e; ++i)
    OL[i].init(InOL[i], this);
}

BasicBlock *InvokeInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned InvokeInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void InvokeInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  return setSuccessor(idx, B);
}

void InvokeInst::setParamAttrs(const ParamAttrsList *newAttrs) {
  if (ParamAttrs == newAttrs)
    return;

  if (ParamAttrs)
    ParamAttrs->dropRef();

  if (newAttrs)
    newAttrs->addRef();

  ParamAttrs = newAttrs; 
}

bool InvokeInst::paramHasAttr(uint16_t i, ParameterAttributes attr) const {
  if (ParamAttrs && ParamAttrs->paramHasAttr(i, attr))
    return true;
  if (const Function *F = getCalledFunction())
    return F->paramHasAttr(i, attr);
  return false;
}

uint16_t InvokeInst::getParamAlignment(uint16_t i) const {
  if (ParamAttrs && ParamAttrs->getParamAlignment(i))
    return ParamAttrs->getParamAlignment(i);
  if (const Function *F = getCalledFunction())
    return F->getParamAlignment(i);
  return 0;
}

/// @brief Determine if the call does not access memory.
bool InvokeInst::doesNotAccessMemory() const {
  return paramHasAttr(0, ParamAttr::ReadNone);
}

/// @brief Determine if the call does not access or only reads memory.
bool InvokeInst::onlyReadsMemory() const {
  return doesNotAccessMemory() || paramHasAttr(0, ParamAttr::ReadOnly);
}

/// @brief Determine if the call cannot return.
bool InvokeInst::doesNotReturn() const {
  return paramHasAttr(0, ParamAttr::NoReturn);
}

/// @brief Determine if the call cannot unwind.
bool InvokeInst::doesNotThrow() const {
  return paramHasAttr(0, ParamAttr::NoUnwind);
}

void InvokeInst::setDoesNotThrow(bool doesNotThrow) {
  const ParamAttrsList *PAL = getParamAttrs();
  if (doesNotThrow)
    PAL = ParamAttrsList::includeAttrs(PAL, 0, ParamAttr::NoUnwind);
  else
    PAL = ParamAttrsList::excludeAttrs(PAL, 0, ParamAttr::NoUnwind);
  setParamAttrs(PAL);
}

/// @brief Determine if the call returns a structure.
bool InvokeInst::isStructReturn() const {
  // Be friendly and also check the callee.
  return paramHasAttr(1, ParamAttr::StructRet);
}


//===----------------------------------------------------------------------===//
//                        ReturnInst Implementation
//===----------------------------------------------------------------------===//

ReturnInst::ReturnInst(const ReturnInst &RI)
  : TerminatorInst(Type::VoidTy, Instruction::Ret,
                   OperandList, RI.getNumOperands()) {
  unsigned N = RI.getNumOperands();
  Use *OL = OperandList = new Use[N];
  for (unsigned i = 0; i < N; ++i)
    OL[i].init(RI.getOperand(i), this);
}

ReturnInst::ReturnInst(Value *retVal, Instruction *InsertBefore)
  : TerminatorInst(Type::VoidTy, Instruction::Ret, OperandList, 0, InsertBefore) {
  init(retVal);
}
ReturnInst::ReturnInst(Value *retVal, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Ret, OperandList, 0, InsertAtEnd) {
  init(retVal);
}
ReturnInst::ReturnInst(BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Ret, OperandList, 0, InsertAtEnd) {
}

ReturnInst::ReturnInst(std::vector<Value *> &retVals, Instruction *InsertBefore)
  : TerminatorInst(Type::VoidTy, Instruction::Ret, OperandList, retVals.size(), InsertBefore) {
  init(retVals);
}
ReturnInst::ReturnInst(std::vector<Value *> &retVals, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Ret, OperandList, retVals.size(), InsertAtEnd) {
  init(retVals);
}
ReturnInst::ReturnInst(std::vector<Value *> &retVals)
  : TerminatorInst(Type::VoidTy, Instruction::Ret, OperandList, retVals.size()) {
  init(retVals);
}

void ReturnInst::init(Value *retVal) {
  if (retVal && retVal->getType() != Type::VoidTy) {
    assert(!isa<BasicBlock>(retVal) &&
           "Cannot return basic block.  Probably using the incorrect ctor");
    NumOperands = 1;
    Use *OL = OperandList = new Use[1];
    OL[0].init(retVal, this);
  }
}

void ReturnInst::init(std::vector<Value *> &retVals) {
  if (retVals.empty())
    return;

  NumOperands = retVals.size();
  if (NumOperands == 1) {
    Value *V = retVals[0];
    if (V->getType() == Type::VoidTy)
      return;
  }

  Use *OL = OperandList = new Use[NumOperands];
  for (unsigned i = 0; i < NumOperands; ++i) {
    Value *V = retVals[i];
    assert(!isa<BasicBlock>(V) &&
           "Cannot return basic block.  Probably using the incorrect ctor");
    OL[i].init(V, this);
  }
}

Value *ReturnInst::getReturnValue(unsigned n) const {
  if (NumOperands)
    return OperandList[n];
  else
    return 0;
}

unsigned ReturnInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

// Out-of-line ReturnInst method, put here so the C++ compiler can choose to
// emit the vtable for the class in this translation unit.
void ReturnInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "ReturnInst has no successors!");
}

BasicBlock *ReturnInst::getSuccessorV(unsigned idx) const {
  assert(0 && "ReturnInst has no successors!");
  abort();
  return 0;
}

ReturnInst::~ReturnInst() {
  if (NumOperands)
    delete [] OperandList;
}

//===----------------------------------------------------------------------===//
//                        UnwindInst Implementation
//===----------------------------------------------------------------------===//

UnwindInst::UnwindInst(Instruction *InsertBefore)
  : TerminatorInst(Type::VoidTy, Instruction::Unwind, 0, 0, InsertBefore) {
}
UnwindInst::UnwindInst(BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Unwind, 0, 0, InsertAtEnd) {
}


unsigned UnwindInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

void UnwindInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "UnwindInst has no successors!");
}

BasicBlock *UnwindInst::getSuccessorV(unsigned idx) const {
  assert(0 && "UnwindInst has no successors!");
  abort();
  return 0;
}

//===----------------------------------------------------------------------===//
//                      UnreachableInst Implementation
//===----------------------------------------------------------------------===//

UnreachableInst::UnreachableInst(Instruction *InsertBefore)
  : TerminatorInst(Type::VoidTy, Instruction::Unreachable, 0, 0, InsertBefore) {
}
UnreachableInst::UnreachableInst(BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Unreachable, 0, 0, InsertAtEnd) {
}

unsigned UnreachableInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}

void UnreachableInst::setSuccessorV(unsigned idx, BasicBlock *NewSucc) {
  assert(0 && "UnwindInst has no successors!");
}

BasicBlock *UnreachableInst::getSuccessorV(unsigned idx) const {
  assert(0 && "UnwindInst has no successors!");
  abort();
  return 0;
}

//===----------------------------------------------------------------------===//
//                        BranchInst Implementation
//===----------------------------------------------------------------------===//

void BranchInst::AssertOK() {
  if (isConditional())
    assert(getCondition()->getType() == Type::Int1Ty &&
           "May only branch on boolean predicates!");
}

BranchInst::BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore)
  : TerminatorInst(Type::VoidTy, Instruction::Br, Ops, 1, InsertBefore) {
  assert(IfTrue != 0 && "Branch destination may not be null!");
  Ops[0].init(reinterpret_cast<Value*>(IfTrue), this);
}
BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
                       Instruction *InsertBefore)
: TerminatorInst(Type::VoidTy, Instruction::Br, Ops, 3, InsertBefore) {
  Ops[0].init(reinterpret_cast<Value*>(IfTrue), this);
  Ops[1].init(reinterpret_cast<Value*>(IfFalse), this);
  Ops[2].init(Cond, this);
#ifndef NDEBUG
  AssertOK();
#endif
}

BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Br, Ops, 1, InsertAtEnd) {
  assert(IfTrue != 0 && "Branch destination may not be null!");
  Ops[0].init(reinterpret_cast<Value*>(IfTrue), this);
}

BranchInst::BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
           BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Br, Ops, 3, InsertAtEnd) {
  Ops[0].init(reinterpret_cast<Value*>(IfTrue), this);
  Ops[1].init(reinterpret_cast<Value*>(IfFalse), this);
  Ops[2].init(Cond, this);
#ifndef NDEBUG
  AssertOK();
#endif
}


BranchInst::BranchInst(const BranchInst &BI) :
  TerminatorInst(Type::VoidTy, Instruction::Br, Ops, BI.getNumOperands()) {
  OperandList[0].init(BI.getOperand(0), this);
  if (BI.getNumOperands() != 1) {
    assert(BI.getNumOperands() == 3 && "BR can have 1 or 3 operands!");
    OperandList[1].init(BI.getOperand(1), this);
    OperandList[2].init(BI.getOperand(2), this);
  }
}

BasicBlock *BranchInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned BranchInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void BranchInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  setSuccessor(idx, B);
}


//===----------------------------------------------------------------------===//
//                        AllocationInst Implementation
//===----------------------------------------------------------------------===//

static Value *getAISize(Value *Amt) {
  if (!Amt)
    Amt = ConstantInt::get(Type::Int32Ty, 1);
  else {
    assert(!isa<BasicBlock>(Amt) &&
           "Passed basic block into allocation size parameter! Use other ctor");
    assert(Amt->getType() == Type::Int32Ty &&
           "Malloc/Allocation array size is not a 32-bit integer!");
  }
  return Amt;
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy,
                               unsigned Align, const std::string &Name,
                               Instruction *InsertBefore)
  : UnaryInstruction(PointerType::getUnqual(Ty), iTy, getAISize(ArraySize),
                     InsertBefore), Alignment(Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  assert(Ty != Type::VoidTy && "Cannot allocate void!");
  setName(Name);
}

AllocationInst::AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy,
                               unsigned Align, const std::string &Name,
                               BasicBlock *InsertAtEnd)
  : UnaryInstruction(PointerType::getUnqual(Ty), iTy, getAISize(ArraySize),
                     InsertAtEnd), Alignment(Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  assert(Ty != Type::VoidTy && "Cannot allocate void!");
  setName(Name);
}

// Out of line virtual method, so the vtable, etc has a home.
AllocationInst::~AllocationInst() {
}

bool AllocationInst::isArrayAllocation() const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(getOperand(0)))
    return CI->getZExtValue() != 1;
  return true;
}

const Type *AllocationInst::getAllocatedType() const {
  return getType()->getElementType();
}

AllocaInst::AllocaInst(const AllocaInst &AI)
  : AllocationInst(AI.getType()->getElementType(), (Value*)AI.getOperand(0),
                   Instruction::Alloca, AI.getAlignment()) {
}

MallocInst::MallocInst(const MallocInst &MI)
  : AllocationInst(MI.getType()->getElementType(), (Value*)MI.getOperand(0),
                   Instruction::Malloc, MI.getAlignment()) {
}

//===----------------------------------------------------------------------===//
//                             FreeInst Implementation
//===----------------------------------------------------------------------===//

void FreeInst::AssertOK() {
  assert(isa<PointerType>(getOperand(0)->getType()) &&
         "Can not free something of nonpointer type!");
}

FreeInst::FreeInst(Value *Ptr, Instruction *InsertBefore)
  : UnaryInstruction(Type::VoidTy, Free, Ptr, InsertBefore) {
  AssertOK();
}

FreeInst::FreeInst(Value *Ptr, BasicBlock *InsertAtEnd)
  : UnaryInstruction(Type::VoidTy, Free, Ptr, InsertAtEnd) {
  AssertOK();
}


//===----------------------------------------------------------------------===//
//                           LoadInst Implementation
//===----------------------------------------------------------------------===//

void LoadInst::AssertOK() {
  assert(isa<PointerType>(getOperand(0)->getType()) &&
         "Ptr must have pointer type.");
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
                   Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile, 
                   unsigned Align, Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile, 
                   unsigned Align, BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
  setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
                   BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  setName(Name);
}



LoadInst::LoadInst(Value *Ptr, const char *Name, Instruction *InsertBef)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertBef) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const char *Name, BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(false);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const char *Name, bool isVolatile,
                   Instruction *InsertBef)
: UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                   Load, Ptr, InsertBef) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

LoadInst::LoadInst(Value *Ptr, const char *Name, bool isVolatile,
                   BasicBlock *InsertAE)
  : UnaryInstruction(cast<PointerType>(Ptr->getType())->getElementType(),
                     Load, Ptr, InsertAE) {
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
  if (Name && Name[0]) setName(Name);
}

void LoadInst::setAlignment(unsigned Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  SubclassData = (SubclassData & 1) | ((Log2_32(Align)+1)<<1);
}

//===----------------------------------------------------------------------===//
//                           StoreInst Implementation
//===----------------------------------------------------------------------===//

void StoreInst::AssertOK() {
  assert(isa<PointerType>(getOperand(1)->getType()) &&
         "Ptr must have pointer type!");
  assert(getOperand(0)->getType() ==
                 cast<PointerType>(getOperand(1)->getType())->getElementType()
         && "Ptr must be a pointer to Val type!");
}


StoreInst::StoreInst(Value *val, Value *addr, Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, Ops, 2, InsertBefore) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(false);
  setAlignment(0);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Store, Ops, 2, InsertAtEnd) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(false);
  setAlignment(0);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, Ops, 2, InsertBefore) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     unsigned Align, Instruction *InsertBefore)
  : Instruction(Type::VoidTy, Store, Ops, 2, InsertBefore) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     unsigned Align, BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Store, Ops, 2, InsertAtEnd) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(isVolatile);
  setAlignment(Align);
  AssertOK();
}

StoreInst::StoreInst(Value *val, Value *addr, bool isVolatile,
                     BasicBlock *InsertAtEnd)
  : Instruction(Type::VoidTy, Store, Ops, 2, InsertAtEnd) {
  Ops[0].init(val, this);
  Ops[1].init(addr, this);
  setVolatile(isVolatile);
  setAlignment(0);
  AssertOK();
}

void StoreInst::setAlignment(unsigned Align) {
  assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
  SubclassData = (SubclassData & 1) | ((Log2_32(Align)+1)<<1);
}

//===----------------------------------------------------------------------===//
//                       GetElementPtrInst Implementation
//===----------------------------------------------------------------------===//

static unsigned retrieveAddrSpace(const Value *Val) {
  return cast<PointerType>(Val->getType())->getAddressSpace();
}

void GetElementPtrInst::init(Value *Ptr, Value* const *Idx, unsigned NumIdx) {
  NumOperands = 1+NumIdx;
  Use *OL = OperandList = new Use[NumOperands];
  OL[0].init(Ptr, this);

  for (unsigned i = 0; i != NumIdx; ++i)
    OL[i+1].init(Idx[i], this);
}

void GetElementPtrInst::init(Value *Ptr, Value *Idx) {
  NumOperands = 2;
  Use *OL = OperandList = new Use[2];
  OL[0].init(Ptr, this);
  OL[1].init(Idx, this);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx,
                                     const std::string &Name, Instruction *InBe)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),Idx)),
                                 retrieveAddrSpace(Ptr)),
                GetElementPtr, 0, 0, InBe) {
  init(Ptr, Idx);
  setName(Name);
}

GetElementPtrInst::GetElementPtrInst(Value *Ptr, Value *Idx,
                                     const std::string &Name, BasicBlock *IAE)
  : Instruction(PointerType::get(checkType(getIndexedType(Ptr->getType(),Idx)),
                                 retrieveAddrSpace(Ptr)),
                GetElementPtr, 0, 0, IAE) {
  init(Ptr, Idx);
  setName(Name);
}

GetElementPtrInst::~GetElementPtrInst() {
  delete[] OperandList;
}

// getIndexedType - Returns the type of the element that would be loaded with
// a load instruction with the specified parameters.
//
// A null type is returned if the indices are invalid for the specified
// pointer type.
//
const Type* GetElementPtrInst::getIndexedType(const Type *Ptr,
                                              Value* const *Idxs,
                                              unsigned NumIdx,
                                              bool AllowCompositeLeaf) {
  if (!isa<PointerType>(Ptr)) return 0;   // Type isn't a pointer type!

  // Handle the special case of the empty set index set...
  if (NumIdx == 0) {
    if (AllowCompositeLeaf ||
        cast<PointerType>(Ptr)->getElementType()->isFirstClassType())
      return cast<PointerType>(Ptr)->getElementType();
    else
      return 0;
  }

  unsigned CurIdx = 0;
  while (const CompositeType *CT = dyn_cast<CompositeType>(Ptr)) {
    if (NumIdx == CurIdx) {
      if (AllowCompositeLeaf || CT->isFirstClassType()) return Ptr;
      return 0;   // Can't load a whole structure or array!?!?
    }

    Value *Index = Idxs[CurIdx++];
    if (isa<PointerType>(CT) && CurIdx != 1)
      return 0;  // Can only index into pointer types at the first index!
    if (!CT->indexValid(Index)) return 0;
    Ptr = CT->getTypeAtIndex(Index);

    // If the new type forwards to another type, then it is in the middle
    // of being refined to another type (and hence, may have dropped all
    // references to what it was using before).  So, use the new forwarded
    // type.
    if (const Type * Ty = Ptr->getForwardedType()) {
      Ptr = Ty;
    }
  }
  return CurIdx == NumIdx ? Ptr : 0;
}

const Type* GetElementPtrInst::getIndexedType(const Type *Ptr, Value *Idx) {
  const PointerType *PTy = dyn_cast<PointerType>(Ptr);
  if (!PTy) return 0;   // Type isn't a pointer type!

  // Check the pointer index.
  if (!PTy->indexValid(Idx)) return 0;

  return PTy->getElementType();
}


/// hasAllZeroIndices - Return true if all of the indices of this GEP are
/// zeros.  If so, the result pointer and the first operand have the same
/// value, just potentially different types.
bool GetElementPtrInst::hasAllZeroIndices() const {
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(getOperand(i))) {
      if (!CI->isZero()) return false;
    } else {
      return false;
    }
  }
  return true;
}

/// hasAllConstantIndices - Return true if all of the indices of this GEP are
/// constant integers.  If so, the result pointer and the first operand have
/// a constant offset between them.
bool GetElementPtrInst::hasAllConstantIndices() const {
  for (unsigned i = 1, e = getNumOperands(); i != e; ++i) {
    if (!isa<ConstantInt>(getOperand(i)))
      return false;
  }
  return true;
}


//===----------------------------------------------------------------------===//
//                           ExtractElementInst Implementation
//===----------------------------------------------------------------------===//

ExtractElementInst::ExtractElementInst(Value *Val, Value *Index,
                                       const std::string &Name,
                                       Instruction *InsertBef)
  : Instruction(cast<VectorType>(Val->getType())->getElementType(),
                ExtractElement, Ops, 2, InsertBef) {
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");
  Ops[0].init(Val, this);
  Ops[1].init(Index, this);
  setName(Name);
}

ExtractElementInst::ExtractElementInst(Value *Val, unsigned IndexV,
                                       const std::string &Name,
                                       Instruction *InsertBef)
  : Instruction(cast<VectorType>(Val->getType())->getElementType(),
                ExtractElement, Ops, 2, InsertBef) {
  Constant *Index = ConstantInt::get(Type::Int32Ty, IndexV);
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");
  Ops[0].init(Val, this);
  Ops[1].init(Index, this);
  setName(Name);
}


ExtractElementInst::ExtractElementInst(Value *Val, Value *Index,
                                       const std::string &Name,
                                       BasicBlock *InsertAE)
  : Instruction(cast<VectorType>(Val->getType())->getElementType(),
                ExtractElement, Ops, 2, InsertAE) {
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");

  Ops[0].init(Val, this);
  Ops[1].init(Index, this);
  setName(Name);
}

ExtractElementInst::ExtractElementInst(Value *Val, unsigned IndexV,
                                       const std::string &Name,
                                       BasicBlock *InsertAE)
  : Instruction(cast<VectorType>(Val->getType())->getElementType(),
                ExtractElement, Ops, 2, InsertAE) {
  Constant *Index = ConstantInt::get(Type::Int32Ty, IndexV);
  assert(isValidOperands(Val, Index) &&
         "Invalid extractelement instruction operands!");
  
  Ops[0].init(Val, this);
  Ops[1].init(Index, this);
  setName(Name);
}


bool ExtractElementInst::isValidOperands(const Value *Val, const Value *Index) {
  if (!isa<VectorType>(Val->getType()) || Index->getType() != Type::Int32Ty)
    return false;
  return true;
}


//===----------------------------------------------------------------------===//
//                           InsertElementInst Implementation
//===----------------------------------------------------------------------===//

InsertElementInst::InsertElementInst(const InsertElementInst &IE)
    : Instruction(IE.getType(), InsertElement, Ops, 3) {
  Ops[0].init(IE.Ops[0], this);
  Ops[1].init(IE.Ops[1], this);
  Ops[2].init(IE.Ops[2], this);
}
InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, Value *Index,
                                     const std::string &Name,
                                     Instruction *InsertBef)
  : Instruction(Vec->getType(), InsertElement, Ops, 3, InsertBef) {
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");
  Ops[0].init(Vec, this);
  Ops[1].init(Elt, this);
  Ops[2].init(Index, this);
  setName(Name);
}

InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, unsigned IndexV,
                                     const std::string &Name,
                                     Instruction *InsertBef)
  : Instruction(Vec->getType(), InsertElement, Ops, 3, InsertBef) {
  Constant *Index = ConstantInt::get(Type::Int32Ty, IndexV);
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");
  Ops[0].init(Vec, this);
  Ops[1].init(Elt, this);
  Ops[2].init(Index, this);
  setName(Name);
}


InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, Value *Index,
                                     const std::string &Name,
                                     BasicBlock *InsertAE)
  : Instruction(Vec->getType(), InsertElement, Ops, 3, InsertAE) {
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");

  Ops[0].init(Vec, this);
  Ops[1].init(Elt, this);
  Ops[2].init(Index, this);
  setName(Name);
}

InsertElementInst::InsertElementInst(Value *Vec, Value *Elt, unsigned IndexV,
                                     const std::string &Name,
                                     BasicBlock *InsertAE)
: Instruction(Vec->getType(), InsertElement, Ops, 3, InsertAE) {
  Constant *Index = ConstantInt::get(Type::Int32Ty, IndexV);
  assert(isValidOperands(Vec, Elt, Index) &&
         "Invalid insertelement instruction operands!");
  
  Ops[0].init(Vec, this);
  Ops[1].init(Elt, this);
  Ops[2].init(Index, this);
  setName(Name);
}

bool InsertElementInst::isValidOperands(const Value *Vec, const Value *Elt, 
                                        const Value *Index) {
  if (!isa<VectorType>(Vec->getType()))
    return false;   // First operand of insertelement must be vector type.
  
  if (Elt->getType() != cast<VectorType>(Vec->getType())->getElementType())
    return false;// Second operand of insertelement must be vector element type.
    
  if (Index->getType() != Type::Int32Ty)
    return false;  // Third operand of insertelement must be uint.
  return true;
}


//===----------------------------------------------------------------------===//
//                      ShuffleVectorInst Implementation
//===----------------------------------------------------------------------===//

ShuffleVectorInst::ShuffleVectorInst(const ShuffleVectorInst &SV) 
    : Instruction(SV.getType(), ShuffleVector, Ops, 3) {
  Ops[0].init(SV.Ops[0], this);
  Ops[1].init(SV.Ops[1], this);
  Ops[2].init(SV.Ops[2], this);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const std::string &Name,
                                     Instruction *InsertBefore)
  : Instruction(V1->getType(), ShuffleVector, Ops, 3, InsertBefore) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");
  Ops[0].init(V1, this);
  Ops[1].init(V2, this);
  Ops[2].init(Mask, this);
  setName(Name);
}

ShuffleVectorInst::ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                                     const std::string &Name, 
                                     BasicBlock *InsertAtEnd)
  : Instruction(V1->getType(), ShuffleVector, Ops, 3, InsertAtEnd) {
  assert(isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector instruction operands!");

  Ops[0].init(V1, this);
  Ops[1].init(V2, this);
  Ops[2].init(Mask, this);
  setName(Name);
}

bool ShuffleVectorInst::isValidOperands(const Value *V1, const Value *V2, 
                                        const Value *Mask) {
  if (!isa<VectorType>(V1->getType())) return false;
  if (V1->getType() != V2->getType()) return false;
  if (!isa<VectorType>(Mask->getType()) ||
         cast<VectorType>(Mask->getType())->getElementType() != Type::Int32Ty ||
         cast<VectorType>(Mask->getType())->getNumElements() !=
         cast<VectorType>(V1->getType())->getNumElements())
    return false;
  return true;
}


//===----------------------------------------------------------------------===//
//                             BinaryOperator Class
//===----------------------------------------------------------------------===//

BinaryOperator::BinaryOperator(BinaryOps iType, Value *S1, Value *S2,
                               const Type *Ty, const std::string &Name,
                               Instruction *InsertBefore)
  : Instruction(Ty, iType, Ops, 2, InsertBefore) {
  Ops[0].init(S1, this);
  Ops[1].init(S2, this);
  init(iType);
  setName(Name);
}

BinaryOperator::BinaryOperator(BinaryOps iType, Value *S1, Value *S2, 
                               const Type *Ty, const std::string &Name,
                               BasicBlock *InsertAtEnd)
  : Instruction(Ty, iType, Ops, 2, InsertAtEnd) {
  Ops[0].init(S1, this);
  Ops[1].init(S2, this);
  init(iType);
  setName(Name);
}


void BinaryOperator::init(BinaryOps iType) {
  Value *LHS = getOperand(0), *RHS = getOperand(1);
  LHS = LHS; RHS = RHS; // Silence warnings.
  assert(LHS->getType() == RHS->getType() &&
         "Binary operator operand types must match!");
#ifndef NDEBUG
  switch (iType) {
  case Add: case Sub:
  case Mul: 
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isInteger() || getType()->isFloatingPoint() ||
            isa<VectorType>(getType())) &&
          "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case UDiv: 
  case SDiv: 
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isInteger() || (isa<VectorType>(getType()) && 
            cast<VectorType>(getType())->getElementType()->isInteger())) &&
           "Incorrect operand type (not integer) for S/UDIV");
    break;
  case FDiv:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isFloatingPoint() || (isa<VectorType>(getType()) &&
            cast<VectorType>(getType())->getElementType()->isFloatingPoint())) 
            && "Incorrect operand type (not floating point) for FDIV");
    break;
  case URem: 
  case SRem: 
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isInteger() || (isa<VectorType>(getType()) && 
            cast<VectorType>(getType())->getElementType()->isInteger())) &&
           "Incorrect operand type (not integer) for S/UREM");
    break;
  case FRem:
    assert(getType() == LHS->getType() &&
           "Arithmetic operation should return same type as operands!");
    assert((getType()->isFloatingPoint() || (isa<VectorType>(getType()) &&
            cast<VectorType>(getType())->getElementType()->isFloatingPoint())) 
            && "Incorrect operand type (not floating point) for FREM");
    break;
  case Shl:
  case LShr:
  case AShr:
    assert(getType() == LHS->getType() &&
           "Shift operation should return same type as operands!");
    assert(getType()->isInteger() && 
           "Shift operation requires integer operands");
    break;
  case And: case Or:
  case Xor:
    assert(getType() == LHS->getType() &&
           "Logical operation should return same type as operands!");
    assert((getType()->isInteger() ||
            (isa<VectorType>(getType()) && 
             cast<VectorType>(getType())->getElementType()->isInteger())) &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  default:
    break;
  }
#endif
}

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
                                       const std::string &Name,
                                       Instruction *InsertBefore) {
  assert(S1->getType() == S2->getType() &&
         "Cannot create binary operator with two operands of differing type!");
  return new BinaryOperator(Op, S1, S2, S1->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::create(BinaryOps Op, Value *S1, Value *S2,
                                       const std::string &Name,
                                       BasicBlock *InsertAtEnd) {
  BinaryOperator *Res = create(Op, S1, S2, Name);
  InsertAtEnd->getInstList().push_back(Res);
  return Res;
}

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  Value *zero = ConstantExpr::getZeroValueForNegationExpr(Op->getType());
  return new BinaryOperator(Instruction::Sub,
                            zero, Op,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::createNeg(Value *Op, const std::string &Name,
                                          BasicBlock *InsertAtEnd) {
  Value *zero = ConstantExpr::getZeroValueForNegationExpr(Op->getType());
  return new BinaryOperator(Instruction::Sub,
                            zero, Op,
                            Op->getType(), Name, InsertAtEnd);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name,
                                          Instruction *InsertBefore) {
  Constant *C;
  if (const VectorType *PTy = dyn_cast<VectorType>(Op->getType())) {
    C = ConstantInt::getAllOnesValue(PTy->getElementType());
    C = ConstantVector::get(std::vector<Constant*>(PTy->getNumElements(), C));
  } else {
    C = ConstantInt::getAllOnesValue(Op->getType());
  }
  
  return new BinaryOperator(Instruction::Xor, Op, C,
                            Op->getType(), Name, InsertBefore);
}

BinaryOperator *BinaryOperator::createNot(Value *Op, const std::string &Name,
                                          BasicBlock *InsertAtEnd) {
  Constant *AllOnes;
  if (const VectorType *PTy = dyn_cast<VectorType>(Op->getType())) {
    // Create a vector of all ones values.
    Constant *Elt = ConstantInt::getAllOnesValue(PTy->getElementType());
    AllOnes = 
      ConstantVector::get(std::vector<Constant*>(PTy->getNumElements(), Elt));
  } else {
    AllOnes = ConstantInt::getAllOnesValue(Op->getType());
  }
  
  return new BinaryOperator(Instruction::Xor, Op, AllOnes,
                            Op->getType(), Name, InsertAtEnd);
}


// isConstantAllOnes - Helper function for several functions below
static inline bool isConstantAllOnes(const Value *V) {
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(V))
    return CI->isAllOnesValue();
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(V))
    return CV->isAllOnesValue();
  return false;
}

bool BinaryOperator::isNeg(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    if (Bop->getOpcode() == Instruction::Sub)
      return Bop->getOperand(0) ==
             ConstantExpr::getZeroValueForNegationExpr(Bop->getType());
  return false;
}

bool BinaryOperator::isNot(const Value *V) {
  if (const BinaryOperator *Bop = dyn_cast<BinaryOperator>(V))
    return (Bop->getOpcode() == Instruction::Xor &&
            (isConstantAllOnes(Bop->getOperand(1)) ||
             isConstantAllOnes(Bop->getOperand(0))));
  return false;
}

Value *BinaryOperator::getNegArgument(Value *BinOp) {
  assert(isNeg(BinOp) && "getNegArgument from non-'neg' instruction!");
  return cast<BinaryOperator>(BinOp)->getOperand(1);
}

const Value *BinaryOperator::getNegArgument(const Value *BinOp) {
  return getNegArgument(const_cast<Value*>(BinOp));
}

Value *BinaryOperator::getNotArgument(Value *BinOp) {
  assert(isNot(BinOp) && "getNotArgument on non-'not' instruction!");
  BinaryOperator *BO = cast<BinaryOperator>(BinOp);
  Value *Op0 = BO->getOperand(0);
  Value *Op1 = BO->getOperand(1);
  if (isConstantAllOnes(Op0)) return Op1;

  assert(isConstantAllOnes(Op1));
  return Op0;
}

const Value *BinaryOperator::getNotArgument(const Value *BinOp) {
  return getNotArgument(const_cast<Value*>(BinOp));
}


// swapOperands - Exchange the two operands to this instruction.  This
// instruction is safe to use on any binary instruction and does not
// modify the semantics of the instruction.  If the instruction is
// order dependent (SetLT f.e.) the opcode is changed.
//
bool BinaryOperator::swapOperands() {
  if (!isCommutative())
    return true; // Can't commute operands
  std::swap(Ops[0], Ops[1]);
  return false;
}

//===----------------------------------------------------------------------===//
//                                CastInst Class
//===----------------------------------------------------------------------===//

// Just determine if this cast only deals with integral->integral conversion.
bool CastInst::isIntegerCast() const {
  switch (getOpcode()) {
    default: return false;
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::Trunc:
      return true;
    case Instruction::BitCast:
      return getOperand(0)->getType()->isInteger() && getType()->isInteger();
  }
}

bool CastInst::isLosslessCast() const {
  // Only BitCast can be lossless, exit fast if we're not BitCast
  if (getOpcode() != Instruction::BitCast)
    return false;

  // Identity cast is always lossless
  const Type* SrcTy = getOperand(0)->getType();
  const Type* DstTy = getType();
  if (SrcTy == DstTy)
    return true;
  
  // Pointer to pointer is always lossless.
  if (isa<PointerType>(SrcTy))
    return isa<PointerType>(DstTy);
  return false;  // Other types have no identity values
}

/// This function determines if the CastInst does not require any bits to be
/// changed in order to effect the cast. Essentially, it identifies cases where
/// no code gen is necessary for the cast, hence the name no-op cast.  For 
/// example, the following are all no-op casts:
/// # bitcast uint %X, int
/// # bitcast uint* %x, sbyte*
/// # bitcast vector< 2 x int > %x, vector< 4 x short> 
/// # ptrtoint uint* %x, uint     ; on 32-bit plaforms only
/// @brief Determine if a cast is a no-op.
bool CastInst::isNoopCast(const Type *IntPtrTy) const {
  switch (getOpcode()) {
    default:
      assert(!"Invalid CastOp");
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt: 
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      return false; // These always modify bits
    case Instruction::BitCast:
      return true;  // BitCast never modifies bits.
    case Instruction::PtrToInt:
      return IntPtrTy->getPrimitiveSizeInBits() ==
            getType()->getPrimitiveSizeInBits();
    case Instruction::IntToPtr:
      return IntPtrTy->getPrimitiveSizeInBits() ==
             getOperand(0)->getType()->getPrimitiveSizeInBits();
  }
}

/// This function determines if a pair of casts can be eliminated and what 
/// opcode should be used in the elimination. This assumes that there are two 
/// instructions like this:
/// *  %F = firstOpcode SrcTy %x to MidTy
/// *  %S = secondOpcode MidTy %F to DstTy
/// The function returns a resultOpcode so these two casts can be replaced with:
/// *  %Replacement = resultOpcode %SrcTy %x to DstTy
/// If no such cast is permited, the function returns 0.
unsigned CastInst::isEliminableCastPair(
  Instruction::CastOps firstOp, Instruction::CastOps secondOp,
  const Type *SrcTy, const Type *MidTy, const Type *DstTy, const Type *IntPtrTy)
{
  // Define the 144 possibilities for these two cast instructions. The values
  // in this matrix determine what to do in a given situation and select the
  // case in the switch below.  The rows correspond to firstOp, the columns 
  // correspond to secondOp.  In looking at the table below, keep in  mind
  // the following cast properties:
  //
  //          Size Compare       Source               Destination
  // Operator  Src ? Size   Type       Sign         Type       Sign
  // -------- ------------ -------------------   ---------------------
  // TRUNC         >       Integer      Any        Integral     Any
  // ZEXT          <       Integral   Unsigned     Integer      Any
  // SEXT          <       Integral    Signed      Integer      Any
  // FPTOUI       n/a      FloatPt      n/a        Integral   Unsigned
  // FPTOSI       n/a      FloatPt      n/a        Integral    Signed 
  // UITOFP       n/a      Integral   Unsigned     FloatPt      n/a   
  // SITOFP       n/a      Integral    Signed      FloatPt      n/a   
  // FPTRUNC       >       FloatPt      n/a        FloatPt      n/a   
  // FPEXT         <       FloatPt      n/a        FloatPt      n/a   
  // PTRTOINT     n/a      Pointer      n/a        Integral   Unsigned
  // INTTOPTR     n/a      Integral   Unsigned     Pointer      n/a
  // BITCONVERT    =       FirstClass   n/a       FirstClass    n/a   
  //
  // NOTE: some transforms are safe, but we consider them to be non-profitable.
  // For example, we could merge "fptoui double to uint" + "zext uint to ulong",
  // into "fptoui double to ulong", but this loses information about the range
  // of the produced value (we no longer know the top-part is all zeros). 
  // Further this conversion is often much more expensive for typical hardware,
  // and causes issues when building libgcc.  We disallow fptosi+sext for the 
  // same reason.
  const unsigned numCastOps = 
    Instruction::CastOpsEnd - Instruction::CastOpsBegin;
  static const uint8_t CastResults[numCastOps][numCastOps] = {
    // T        F  F  U  S  F  F  P  I  B   -+
    // R  Z  S  P  P  I  I  T  P  2  N  T    |
    // U  E  E  2  2  2  2  R  E  I  T  C    +- secondOp
    // N  X  X  U  S  F  F  N  X  N  2  V    |
    // C  T  T  I  I  P  P  C  T  T  P  T   -+
    {  1, 0, 0,99,99, 0, 0,99,99,99, 0, 3 }, // Trunc      -+
    {  8, 1, 9,99,99, 2, 0,99,99,99, 2, 3 }, // ZExt        |
    {  8, 0, 1,99,99, 0, 2,99,99,99, 0, 3 }, // SExt        |
    {  0, 0, 0,99,99, 0, 0,99,99,99, 0, 3 }, // FPToUI      |
    {  0, 0, 0,99,99, 0, 0,99,99,99, 0, 3 }, // FPToSI      |
    { 99,99,99, 0, 0,99,99, 0, 0,99,99, 4 }, // UIToFP      +- firstOp
    { 99,99,99, 0, 0,99,99, 0, 0,99,99, 4 }, // SIToFP      |
    { 99,99,99, 0, 0,99,99, 1, 0,99,99, 4 }, // FPTrunc     |
    { 99,99,99, 2, 2,99,99,10, 2,99,99, 4 }, // FPExt       |
    {  1, 0, 0,99,99, 0, 0,99,99,99, 7, 3 }, // PtrToInt    |
    { 99,99,99,99,99,99,99,99,99,13,99,12 }, // IntToPtr    |
    {  5, 5, 5, 6, 6, 5, 5, 6, 6,11, 5, 1 }, // BitCast    -+
  };

  int ElimCase = CastResults[firstOp-Instruction::CastOpsBegin]
                            [secondOp-Instruction::CastOpsBegin];
  switch (ElimCase) {
    case 0: 
      // categorically disallowed
      return 0;
    case 1: 
      // allowed, use first cast's opcode
      return firstOp;
    case 2: 
      // allowed, use second cast's opcode
      return secondOp;
    case 3: 
      // no-op cast in second op implies firstOp as long as the DestTy 
      // is integer
      if (DstTy->isInteger())
        return firstOp;
      return 0;
    case 4:
      // no-op cast in second op implies firstOp as long as the DestTy
      // is floating point
      if (DstTy->isFloatingPoint())
        return firstOp;
      return 0;
    case 5: 
      // no-op cast in first op implies secondOp as long as the SrcTy
      // is an integer
      if (SrcTy->isInteger())
        return secondOp;
      return 0;
    case 6:
      // no-op cast in first op implies secondOp as long as the SrcTy
      // is a floating point
      if (SrcTy->isFloatingPoint())
        return secondOp;
      return 0;
    case 7: { 
      // ptrtoint, inttoptr -> bitcast (ptr -> ptr) if int size is >= ptr size
      unsigned PtrSize = IntPtrTy->getPrimitiveSizeInBits();
      unsigned MidSize = MidTy->getPrimitiveSizeInBits();
      if (MidSize >= PtrSize)
        return Instruction::BitCast;
      return 0;
    }
    case 8: {
      // ext, trunc -> bitcast,    if the SrcTy and DstTy are same size
      // ext, trunc -> ext,        if sizeof(SrcTy) < sizeof(DstTy)
      // ext, trunc -> trunc,      if sizeof(SrcTy) > sizeof(DstTy)
      unsigned SrcSize = SrcTy->getPrimitiveSizeInBits();
      unsigned DstSize = DstTy->getPrimitiveSizeInBits();
      if (SrcSize == DstSize)
        return Instruction::BitCast;
      else if (SrcSize < DstSize)
        return firstOp;
      return secondOp;
    }
    case 9: // zext, sext -> zext, because sext can't sign extend after zext
      return Instruction::ZExt;
    case 10:
      // fpext followed by ftrunc is allowed if the bit size returned to is
      // the same as the original, in which case its just a bitcast
      if (SrcTy == DstTy)
        return Instruction::BitCast;
      return 0; // If the types are not the same we can't eliminate it.
    case 11:
      // bitcast followed by ptrtoint is allowed as long as the bitcast
      // is a pointer to pointer cast.
      if (isa<PointerType>(SrcTy) && isa<PointerType>(MidTy))
        return secondOp;
      return 0;
    case 12:
      // inttoptr, bitcast -> intptr  if bitcast is a ptr to ptr cast
      if (isa<PointerType>(MidTy) && isa<PointerType>(DstTy))
        return firstOp;
      return 0;
    case 13: {
      // inttoptr, ptrtoint -> bitcast if SrcSize<=PtrSize and SrcSize==DstSize
      unsigned PtrSize = IntPtrTy->getPrimitiveSizeInBits();
      unsigned SrcSize = SrcTy->getPrimitiveSizeInBits();
      unsigned DstSize = DstTy->getPrimitiveSizeInBits();
      if (SrcSize <= PtrSize && SrcSize == DstSize)
        return Instruction::BitCast;
      return 0;
    }
    case 99: 
      // cast combination can't happen (error in input). This is for all cases
      // where the MidTy is not the same for the two cast instructions.
      assert(!"Invalid Cast Combination");
      return 0;
    default:
      assert(!"Error in CastResults table!!!");
      return 0;
  }
  return 0;
}

CastInst *CastInst::create(Instruction::CastOps op, Value *S, const Type *Ty, 
  const std::string &Name, Instruction *InsertBefore) {
  // Construct and return the appropriate CastInst subclass
  switch (op) {
    case Trunc:    return new TruncInst    (S, Ty, Name, InsertBefore);
    case ZExt:     return new ZExtInst     (S, Ty, Name, InsertBefore);
    case SExt:     return new SExtInst     (S, Ty, Name, InsertBefore);
    case FPTrunc:  return new FPTruncInst  (S, Ty, Name, InsertBefore);
    case FPExt:    return new FPExtInst    (S, Ty, Name, InsertBefore);
    case UIToFP:   return new UIToFPInst   (S, Ty, Name, InsertBefore);
    case SIToFP:   return new SIToFPInst   (S, Ty, Name, InsertBefore);
    case FPToUI:   return new FPToUIInst   (S, Ty, Name, InsertBefore);
    case FPToSI:   return new FPToSIInst   (S, Ty, Name, InsertBefore);
    case PtrToInt: return new PtrToIntInst (S, Ty, Name, InsertBefore);
    case IntToPtr: return new IntToPtrInst (S, Ty, Name, InsertBefore);
    case BitCast:  return new BitCastInst  (S, Ty, Name, InsertBefore);
    default:
      assert(!"Invalid opcode provided");
  }
  return 0;
}

CastInst *CastInst::create(Instruction::CastOps op, Value *S, const Type *Ty,
  const std::string &Name, BasicBlock *InsertAtEnd) {
  // Construct and return the appropriate CastInst subclass
  switch (op) {
    case Trunc:    return new TruncInst    (S, Ty, Name, InsertAtEnd);
    case ZExt:     return new ZExtInst     (S, Ty, Name, InsertAtEnd);
    case SExt:     return new SExtInst     (S, Ty, Name, InsertAtEnd);
    case FPTrunc:  return new FPTruncInst  (S, Ty, Name, InsertAtEnd);
    case FPExt:    return new FPExtInst    (S, Ty, Name, InsertAtEnd);
    case UIToFP:   return new UIToFPInst   (S, Ty, Name, InsertAtEnd);
    case SIToFP:   return new SIToFPInst   (S, Ty, Name, InsertAtEnd);
    case FPToUI:   return new FPToUIInst   (S, Ty, Name, InsertAtEnd);
    case FPToSI:   return new FPToSIInst   (S, Ty, Name, InsertAtEnd);
    case PtrToInt: return new PtrToIntInst (S, Ty, Name, InsertAtEnd);
    case IntToPtr: return new IntToPtrInst (S, Ty, Name, InsertAtEnd);
    case BitCast:  return new BitCastInst  (S, Ty, Name, InsertAtEnd);
    default:
      assert(!"Invalid opcode provided");
  }
  return 0;
}

CastInst *CastInst::createZExtOrBitCast(Value *S, const Type *Ty, 
                                        const std::string &Name,
                                        Instruction *InsertBefore) {
  if (S->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return create(Instruction::ZExt, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::createZExtOrBitCast(Value *S, const Type *Ty, 
                                        const std::string &Name,
                                        BasicBlock *InsertAtEnd) {
  if (S->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return create(Instruction::ZExt, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::createSExtOrBitCast(Value *S, const Type *Ty, 
                                        const std::string &Name,
                                        Instruction *InsertBefore) {
  if (S->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return create(Instruction::SExt, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::createSExtOrBitCast(Value *S, const Type *Ty, 
                                        const std::string &Name,
                                        BasicBlock *InsertAtEnd) {
  if (S->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return create(Instruction::SExt, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::createTruncOrBitCast(Value *S, const Type *Ty,
                                         const std::string &Name,
                                         Instruction *InsertBefore) {
  if (S->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return create(Instruction::BitCast, S, Ty, Name, InsertBefore);
  return create(Instruction::Trunc, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::createTruncOrBitCast(Value *S, const Type *Ty,
                                         const std::string &Name, 
                                         BasicBlock *InsertAtEnd) {
  if (S->getType()->getPrimitiveSizeInBits() == Ty->getPrimitiveSizeInBits())
    return create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
  return create(Instruction::Trunc, S, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::createPointerCast(Value *S, const Type *Ty,
                                      const std::string &Name,
                                      BasicBlock *InsertAtEnd) {
  assert(isa<PointerType>(S->getType()) && "Invalid cast");
  assert((Ty->isInteger() || isa<PointerType>(Ty)) &&
         "Invalid cast");

  if (Ty->isInteger())
    return create(Instruction::PtrToInt, S, Ty, Name, InsertAtEnd);
  return create(Instruction::BitCast, S, Ty, Name, InsertAtEnd);
}

/// @brief Create a BitCast or a PtrToInt cast instruction
CastInst *CastInst::createPointerCast(Value *S, const Type *Ty, 
                                      const std::string &Name, 
                                      Instruction *InsertBefore) {
  assert(isa<PointerType>(S->getType()) && "Invalid cast");
  assert((Ty->isInteger() || isa<PointerType>(Ty)) &&
         "Invalid cast");

  if (Ty->isInteger())
    return create(Instruction::PtrToInt, S, Ty, Name, InsertBefore);
  return create(Instruction::BitCast, S, Ty, Name, InsertBefore);
}

CastInst *CastInst::createIntegerCast(Value *C, const Type *Ty, 
                                      bool isSigned, const std::string &Name,
                                      Instruction *InsertBefore) {
  assert(C->getType()->isInteger() && Ty->isInteger() && "Invalid cast");
  unsigned SrcBits = C->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = Ty->getPrimitiveSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return create(opcode, C, Ty, Name, InsertBefore);
}

CastInst *CastInst::createIntegerCast(Value *C, const Type *Ty, 
                                      bool isSigned, const std::string &Name,
                                      BasicBlock *InsertAtEnd) {
  assert(C->getType()->isInteger() && Ty->isInteger() && "Invalid cast");
  unsigned SrcBits = C->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = Ty->getPrimitiveSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return create(opcode, C, Ty, Name, InsertAtEnd);
}

CastInst *CastInst::createFPCast(Value *C, const Type *Ty, 
                                 const std::string &Name, 
                                 Instruction *InsertBefore) {
  assert(C->getType()->isFloatingPoint() && Ty->isFloatingPoint() && 
         "Invalid cast");
  unsigned SrcBits = C->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = Ty->getPrimitiveSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt));
  return create(opcode, C, Ty, Name, InsertBefore);
}

CastInst *CastInst::createFPCast(Value *C, const Type *Ty, 
                                 const std::string &Name, 
                                 BasicBlock *InsertAtEnd) {
  assert(C->getType()->isFloatingPoint() && Ty->isFloatingPoint() && 
         "Invalid cast");
  unsigned SrcBits = C->getType()->getPrimitiveSizeInBits();
  unsigned DstBits = Ty->getPrimitiveSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt));
  return create(opcode, C, Ty, Name, InsertAtEnd);
}

// Check whether it is valid to call getCastOpcode for these types.
// This routine must be kept in sync with getCastOpcode.
bool CastInst::isCastable(const Type *SrcTy, const Type *DestTy) {
  if (!SrcTy->isFirstClassType() || !DestTy->isFirstClassType())
    return false;

  if (SrcTy == DestTy)
    return true;

  // Get the bit sizes, we'll need these
  unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();   // 0 for ptr/vector
  unsigned DestBits = DestTy->getPrimitiveSizeInBits(); // 0 for ptr/vector

  // Run through the possibilities ...
  if (DestTy->isInteger()) {                      // Casting to integral
    if (SrcTy->isInteger()) {                     // Casting from integral
        return true;
    } else if (SrcTy->isFloatingPoint()) {        // Casting from floating pt
      return true;
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
                                                  // Casting from vector
      return DestBits == PTy->getBitWidth();
    } else {                                      // Casting from something else
      return isa<PointerType>(SrcTy);
    }
  } else if (DestTy->isFloatingPoint()) {         // Casting to floating pt
    if (SrcTy->isInteger()) {                     // Casting from integral
      return true;
    } else if (SrcTy->isFloatingPoint()) {        // Casting from floating pt
      return true;
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
                                                  // Casting from vector
      return DestBits == PTy->getBitWidth();
    } else {                                      // Casting from something else
      return false;
    }
  } else if (const VectorType *DestPTy = dyn_cast<VectorType>(DestTy)) {
                                                   // Casting to vector
    if (const VectorType *SrcPTy = dyn_cast<VectorType>(SrcTy)) {
                                                   // Casting from vector
      return DestPTy->getBitWidth() == SrcPTy->getBitWidth();
    } else {                                       // Casting from something else
      return DestPTy->getBitWidth() == SrcBits;
    }
  } else if (isa<PointerType>(DestTy)) {           // Casting to pointer
    if (isa<PointerType>(SrcTy)) {                 // Casting from pointer
      return true;
    } else if (SrcTy->isInteger()) {               // Casting from integral
      return true;
    } else {                                       // Casting from something else
      return false;
    }
  } else {                                         // Casting to something else
    return false;
  }
}

// Provide a way to get a "cast" where the cast opcode is inferred from the 
// types and size of the operand. This, basically, is a parallel of the 
// logic in the castIsValid function below.  This axiom should hold:
//   castIsValid( getCastOpcode(Val, Ty), Val, Ty)
// should not assert in castIsValid. In other words, this produces a "correct"
// casting opcode for the arguments passed to it.
// This routine must be kept in sync with isCastable.
Instruction::CastOps
CastInst::getCastOpcode(
  const Value *Src, bool SrcIsSigned, const Type *DestTy, bool DestIsSigned) {
  // Get the bit sizes, we'll need these
  const Type *SrcTy = Src->getType();
  unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();   // 0 for ptr/vector
  unsigned DestBits = DestTy->getPrimitiveSizeInBits(); // 0 for ptr/vector

  assert(SrcTy->isFirstClassType() && DestTy->isFirstClassType() &&
         "Only first class types are castable!");

  // Run through the possibilities ...
  if (DestTy->isInteger()) {                       // Casting to integral
    if (SrcTy->isInteger()) {                      // Casting from integral
      if (DestBits < SrcBits)
        return Trunc;                               // int -> smaller int
      else if (DestBits > SrcBits) {                // its an extension
        if (SrcIsSigned)
          return SExt;                              // signed -> SEXT
        else
          return ZExt;                              // unsigned -> ZEXT
      } else {
        return BitCast;                             // Same size, No-op cast
      }
    } else if (SrcTy->isFloatingPoint()) {          // Casting from floating pt
      if (DestIsSigned) 
        return FPToSI;                              // FP -> sint
      else
        return FPToUI;                              // FP -> uint 
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
      assert(DestBits == PTy->getBitWidth() &&
               "Casting vector to integer of different width");
      return BitCast;                             // Same size, no-op cast
    } else {
      assert(isa<PointerType>(SrcTy) &&
             "Casting from a value that is not first-class type");
      return PtrToInt;                              // ptr -> int
    }
  } else if (DestTy->isFloatingPoint()) {           // Casting to floating pt
    if (SrcTy->isInteger()) {                      // Casting from integral
      if (SrcIsSigned)
        return SIToFP;                              // sint -> FP
      else
        return UIToFP;                              // uint -> FP
    } else if (SrcTy->isFloatingPoint()) {          // Casting from floating pt
      if (DestBits < SrcBits) {
        return FPTrunc;                             // FP -> smaller FP
      } else if (DestBits > SrcBits) {
        return FPExt;                               // FP -> larger FP
      } else  {
        return BitCast;                             // same size, no-op cast
      }
    } else if (const VectorType *PTy = dyn_cast<VectorType>(SrcTy)) {
      assert(DestBits == PTy->getBitWidth() &&
             "Casting vector to floating point of different width");
        return BitCast;                             // same size, no-op cast
    } else {
      assert(0 && "Casting pointer or non-first class to float");
    }
  } else if (const VectorType *DestPTy = dyn_cast<VectorType>(DestTy)) {
    if (const VectorType *SrcPTy = dyn_cast<VectorType>(SrcTy)) {
      assert(DestPTy->getBitWidth() == SrcPTy->getBitWidth() &&
             "Casting vector to vector of different widths");
      return BitCast;                             // vector -> vector
    } else if (DestPTy->getBitWidth() == SrcBits) {
      return BitCast;                               // float/int -> vector
    } else {
      assert(!"Illegal cast to vector (wrong type or size)");
    }
  } else if (isa<PointerType>(DestTy)) {
    if (isa<PointerType>(SrcTy)) {
      return BitCast;                               // ptr -> ptr
    } else if (SrcTy->isInteger()) {
      return IntToPtr;                              // int -> ptr
    } else {
      assert(!"Casting pointer to other than pointer or int");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }

  // If we fall through to here we probably hit an assertion cast above
  // and assertions are not turned on. Anything we return is an error, so
  // BitCast is as good a choice as any.
  return BitCast;
}

//===----------------------------------------------------------------------===//
//                    CastInst SubClass Constructors
//===----------------------------------------------------------------------===//

/// Check that the construction parameters for a CastInst are correct. This
/// could be broken out into the separate constructors but it is useful to have
/// it in one place and to eliminate the redundant code for getting the sizes
/// of the types involved.
bool 
CastInst::castIsValid(Instruction::CastOps op, Value *S, const Type *DstTy) {

  // Check for type sanity on the arguments
  const Type *SrcTy = S->getType();
  if (!SrcTy->isFirstClassType() || !DstTy->isFirstClassType())
    return false;

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DstBitSize = DstTy->getPrimitiveSizeInBits();

  // Switch on the opcode provided
  switch (op) {
  default: return false; // This is an input error
  case Instruction::Trunc:
    return SrcTy->isInteger() && DstTy->isInteger()&& SrcBitSize > DstBitSize;
  case Instruction::ZExt:
    return SrcTy->isInteger() && DstTy->isInteger()&& SrcBitSize < DstBitSize;
  case Instruction::SExt: 
    return SrcTy->isInteger() && DstTy->isInteger()&& SrcBitSize < DstBitSize;
  case Instruction::FPTrunc:
    return SrcTy->isFloatingPoint() && DstTy->isFloatingPoint() && 
      SrcBitSize > DstBitSize;
  case Instruction::FPExt:
    return SrcTy->isFloatingPoint() && DstTy->isFloatingPoint() && 
      SrcBitSize < DstBitSize;
  case Instruction::UIToFP:
  case Instruction::SIToFP:
    if (const VectorType *SVTy = dyn_cast<VectorType>(SrcTy)) {
      if (const VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
        return SVTy->getElementType()->isInteger() &&
               DVTy->getElementType()->isFloatingPoint() &&
               SVTy->getNumElements() == DVTy->getNumElements();
      }
    }
    return SrcTy->isInteger() && DstTy->isFloatingPoint();
  case Instruction::FPToUI:
  case Instruction::FPToSI:
    if (const VectorType *SVTy = dyn_cast<VectorType>(SrcTy)) {
      if (const VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
        return SVTy->getElementType()->isFloatingPoint() &&
               DVTy->getElementType()->isInteger() &&
               SVTy->getNumElements() == DVTy->getNumElements();
      }
    }
    return SrcTy->isFloatingPoint() && DstTy->isInteger();
  case Instruction::PtrToInt:
    return isa<PointerType>(SrcTy) && DstTy->isInteger();
  case Instruction::IntToPtr:
    return SrcTy->isInteger() && isa<PointerType>(DstTy);
  case Instruction::BitCast:
    // BitCast implies a no-op cast of type only. No bits change.
    // However, you can't cast pointers to anything but pointers.
    if (isa<PointerType>(SrcTy) != isa<PointerType>(DstTy))
      return false;

    // Now we know we're not dealing with a pointer/non-pointer mismatch. In all
    // these cases, the cast is okay if the source and destination bit widths
    // are identical.
    return SrcBitSize == DstBitSize;
  }
}

TruncInst::TruncInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, Trunc, S, Name, InsertBefore) {
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal Trunc");
}

TruncInst::TruncInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, Trunc, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal Trunc");
}

ZExtInst::ZExtInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
)  : CastInst(Ty, ZExt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal ZExt");
}

ZExtInst::ZExtInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
)  : CastInst(Ty, ZExt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal ZExt");
}
SExtInst::SExtInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, SExt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SExt");
}

SExtInst::SExtInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
)  : CastInst(Ty, SExt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SExt");
}

FPTruncInst::FPTruncInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, FPTrunc, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPTrunc");
}

FPTruncInst::FPTruncInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPTrunc, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPTrunc");
}

FPExtInst::FPExtInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, FPExt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPExt");
}

FPExtInst::FPExtInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPExt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPExt");
}

UIToFPInst::UIToFPInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, UIToFP, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal UIToFP");
}

UIToFPInst::UIToFPInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, UIToFP, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal UIToFP");
}

SIToFPInst::SIToFPInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, SIToFP, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SIToFP");
}

SIToFPInst::SIToFPInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, SIToFP, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal SIToFP");
}

FPToUIInst::FPToUIInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, FPToUI, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToUI");
}

FPToUIInst::FPToUIInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPToUI, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToUI");
}

FPToSIInst::FPToSIInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, FPToSI, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToSI");
}

FPToSIInst::FPToSIInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, FPToSI, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal FPToSI");
}

PtrToIntInst::PtrToIntInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, PtrToInt, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal PtrToInt");
}

PtrToIntInst::PtrToIntInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, PtrToInt, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal PtrToInt");
}

IntToPtrInst::IntToPtrInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, IntToPtr, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal IntToPtr");
}

IntToPtrInst::IntToPtrInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, IntToPtr, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal IntToPtr");
}

BitCastInst::BitCastInst(
  Value *S, const Type *Ty, const std::string &Name, Instruction *InsertBefore
) : CastInst(Ty, BitCast, S, Name, InsertBefore) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal BitCast");
}

BitCastInst::BitCastInst(
  Value *S, const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd
) : CastInst(Ty, BitCast, S, Name, InsertAtEnd) { 
  assert(castIsValid(getOpcode(), S, Ty) && "Illegal BitCast");
}

//===----------------------------------------------------------------------===//
//                               CmpInst Classes
//===----------------------------------------------------------------------===//

CmpInst::CmpInst(OtherOps op, unsigned short predicate, Value *LHS, Value *RHS,
                 const std::string &Name, Instruction *InsertBefore)
  : Instruction(Type::Int1Ty, op, Ops, 2, InsertBefore) {
    Ops[0].init(LHS, this);
    Ops[1].init(RHS, this);
  SubclassData = predicate;
  setName(Name);
  if (op == Instruction::ICmp) {
    assert(predicate >= ICmpInst::FIRST_ICMP_PREDICATE &&
           predicate <= ICmpInst::LAST_ICMP_PREDICATE &&
           "Invalid ICmp predicate value");
    const Type* Op0Ty = getOperand(0)->getType();
    const Type* Op1Ty = getOperand(1)->getType();
    assert(Op0Ty == Op1Ty &&
           "Both operands to ICmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert((Op0Ty->isInteger() || isa<PointerType>(Op0Ty)) &&
           "Invalid operand types for ICmp instruction");
    return;
  }
  assert(op == Instruction::FCmp && "Invalid CmpInst opcode");
  assert(predicate <= FCmpInst::LAST_FCMP_PREDICATE &&
         "Invalid FCmp predicate value");
  const Type* Op0Ty = getOperand(0)->getType();
  const Type* Op1Ty = getOperand(1)->getType();
  assert(Op0Ty == Op1Ty &&
         "Both operands to FCmp instruction are not of the same type!");
  // Check that the operands are the right type
  assert(Op0Ty->isFloatingPoint() &&
         "Invalid operand types for FCmp instruction");
}
  
CmpInst::CmpInst(OtherOps op, unsigned short predicate, Value *LHS, Value *RHS,
                 const std::string &Name, BasicBlock *InsertAtEnd)
  : Instruction(Type::Int1Ty, op, Ops, 2, InsertAtEnd) {
  Ops[0].init(LHS, this);
  Ops[1].init(RHS, this);
  SubclassData = predicate;
  setName(Name);
  if (op == Instruction::ICmp) {
    assert(predicate >= ICmpInst::FIRST_ICMP_PREDICATE &&
           predicate <= ICmpInst::LAST_ICMP_PREDICATE &&
           "Invalid ICmp predicate value");

    const Type* Op0Ty = getOperand(0)->getType();
    const Type* Op1Ty = getOperand(1)->getType();
    assert(Op0Ty == Op1Ty &&
          "Both operands to ICmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert((Op0Ty->isInteger() || isa<PointerType>(Op0Ty)) &&
           "Invalid operand types for ICmp instruction");
    return;
  }
  assert(op == Instruction::FCmp && "Invalid CmpInst opcode");
  assert(predicate <= FCmpInst::LAST_FCMP_PREDICATE &&
         "Invalid FCmp predicate value");
  const Type* Op0Ty = getOperand(0)->getType();
  const Type* Op1Ty = getOperand(1)->getType();
  assert(Op0Ty == Op1Ty &&
          "Both operands to FCmp instruction are not of the same type!");
  // Check that the operands are the right type
  assert(Op0Ty->isFloatingPoint() &&
        "Invalid operand types for FCmp instruction");
}

CmpInst *
CmpInst::create(OtherOps Op, unsigned short predicate, Value *S1, Value *S2, 
                const std::string &Name, Instruction *InsertBefore) {
  if (Op == Instruction::ICmp) {
    return new ICmpInst(ICmpInst::Predicate(predicate), S1, S2, Name, 
                        InsertBefore);
  }
  return new FCmpInst(FCmpInst::Predicate(predicate), S1, S2, Name, 
                      InsertBefore);
}

CmpInst *
CmpInst::create(OtherOps Op, unsigned short predicate, Value *S1, Value *S2, 
                const std::string &Name, BasicBlock *InsertAtEnd) {
  if (Op == Instruction::ICmp) {
    return new ICmpInst(ICmpInst::Predicate(predicate), S1, S2, Name, 
                        InsertAtEnd);
  }
  return new FCmpInst(FCmpInst::Predicate(predicate), S1, S2, Name, 
                      InsertAtEnd);
}

void CmpInst::swapOperands() {
  if (ICmpInst *IC = dyn_cast<ICmpInst>(this))
    IC->swapOperands();
  else
    cast<FCmpInst>(this)->swapOperands();
}

bool CmpInst::isCommutative() {
  if (ICmpInst *IC = dyn_cast<ICmpInst>(this))
    return IC->isCommutative();
  return cast<FCmpInst>(this)->isCommutative();
}

bool CmpInst::isEquality() {
  if (ICmpInst *IC = dyn_cast<ICmpInst>(this))
    return IC->isEquality();
  return cast<FCmpInst>(this)->isEquality();
}


ICmpInst::Predicate ICmpInst::getInversePredicate(Predicate pred) {
  switch (pred) {
    default:
      assert(!"Unknown icmp predicate!");
    case ICMP_EQ: return ICMP_NE;
    case ICMP_NE: return ICMP_EQ;
    case ICMP_UGT: return ICMP_ULE;
    case ICMP_ULT: return ICMP_UGE;
    case ICMP_UGE: return ICMP_ULT;
    case ICMP_ULE: return ICMP_UGT;
    case ICMP_SGT: return ICMP_SLE;
    case ICMP_SLT: return ICMP_SGE;
    case ICMP_SGE: return ICMP_SLT;
    case ICMP_SLE: return ICMP_SGT;
  }
}

ICmpInst::Predicate ICmpInst::getSwappedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(! "Unknown icmp predicate!");
    case ICMP_EQ: case ICMP_NE:
      return pred;
    case ICMP_SGT: return ICMP_SLT;
    case ICMP_SLT: return ICMP_SGT;
    case ICMP_SGE: return ICMP_SLE;
    case ICMP_SLE: return ICMP_SGE;
    case ICMP_UGT: return ICMP_ULT;
    case ICMP_ULT: return ICMP_UGT;
    case ICMP_UGE: return ICMP_ULE;
    case ICMP_ULE: return ICMP_UGE;
  }
}

ICmpInst::Predicate ICmpInst::getSignedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(! "Unknown icmp predicate!");
    case ICMP_EQ: case ICMP_NE: 
    case ICMP_SGT: case ICMP_SLT: case ICMP_SGE: case ICMP_SLE: 
       return pred;
    case ICMP_UGT: return ICMP_SGT;
    case ICMP_ULT: return ICMP_SLT;
    case ICMP_UGE: return ICMP_SGE;
    case ICMP_ULE: return ICMP_SLE;
  }
}

ICmpInst::Predicate ICmpInst::getUnsignedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(! "Unknown icmp predicate!");
    case ICMP_EQ: case ICMP_NE: 
    case ICMP_UGT: case ICMP_ULT: case ICMP_UGE: case ICMP_ULE: 
       return pred;
    case ICMP_SGT: return ICMP_UGT;
    case ICMP_SLT: return ICMP_ULT;
    case ICMP_SGE: return ICMP_UGE;
    case ICMP_SLE: return ICMP_ULE;
  }
}

bool ICmpInst::isSignedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(! "Unknown icmp predicate!");
    case ICMP_SGT: case ICMP_SLT: case ICMP_SGE: case ICMP_SLE: 
      return true;
    case ICMP_EQ:  case ICMP_NE: case ICMP_UGT: case ICMP_ULT: 
    case ICMP_UGE: case ICMP_ULE:
      return false;
  }
}

/// Initialize a set of values that all satisfy the condition with C.
///
ConstantRange 
ICmpInst::makeConstantRange(Predicate pred, const APInt &C) {
  APInt Lower(C);
  APInt Upper(C);
  uint32_t BitWidth = C.getBitWidth();
  switch (pred) {
  default: assert(0 && "Invalid ICmp opcode to ConstantRange ctor!");
  case ICmpInst::ICMP_EQ: Upper++; break;
  case ICmpInst::ICMP_NE: Lower++; break;
  case ICmpInst::ICMP_ULT: Lower = APInt::getMinValue(BitWidth); break;
  case ICmpInst::ICMP_SLT: Lower = APInt::getSignedMinValue(BitWidth); break;
  case ICmpInst::ICMP_UGT: 
    Lower++; Upper = APInt::getMinValue(BitWidth);        // Min = Next(Max)
    break;
  case ICmpInst::ICMP_SGT:
    Lower++; Upper = APInt::getSignedMinValue(BitWidth);  // Min = Next(Max)
    break;
  case ICmpInst::ICMP_ULE: 
    Lower = APInt::getMinValue(BitWidth); Upper++; 
    break;
  case ICmpInst::ICMP_SLE: 
    Lower = APInt::getSignedMinValue(BitWidth); Upper++; 
    break;
  case ICmpInst::ICMP_UGE:
    Upper = APInt::getMinValue(BitWidth);        // Min = Next(Max)
    break;
  case ICmpInst::ICMP_SGE:
    Upper = APInt::getSignedMinValue(BitWidth);  // Min = Next(Max)
    break;
  }
  return ConstantRange(Lower, Upper);
}

FCmpInst::Predicate FCmpInst::getInversePredicate(Predicate pred) {
  switch (pred) {
    default:
      assert(!"Unknown icmp predicate!");
    case FCMP_OEQ: return FCMP_UNE;
    case FCMP_ONE: return FCMP_UEQ;
    case FCMP_OGT: return FCMP_ULE;
    case FCMP_OLT: return FCMP_UGE;
    case FCMP_OGE: return FCMP_ULT;
    case FCMP_OLE: return FCMP_UGT;
    case FCMP_UEQ: return FCMP_ONE;
    case FCMP_UNE: return FCMP_OEQ;
    case FCMP_UGT: return FCMP_OLE;
    case FCMP_ULT: return FCMP_OGE;
    case FCMP_UGE: return FCMP_OLT;
    case FCMP_ULE: return FCMP_OGT;
    case FCMP_ORD: return FCMP_UNO;
    case FCMP_UNO: return FCMP_ORD;
    case FCMP_TRUE: return FCMP_FALSE;
    case FCMP_FALSE: return FCMP_TRUE;
  }
}

FCmpInst::Predicate FCmpInst::getSwappedPredicate(Predicate pred) {
  switch (pred) {
    default: assert(!"Unknown fcmp predicate!");
    case FCMP_FALSE: case FCMP_TRUE:
    case FCMP_OEQ: case FCMP_ONE:
    case FCMP_UEQ: case FCMP_UNE:
    case FCMP_ORD: case FCMP_UNO:
      return pred;
    case FCMP_OGT: return FCMP_OLT;
    case FCMP_OLT: return FCMP_OGT;
    case FCMP_OGE: return FCMP_OLE;
    case FCMP_OLE: return FCMP_OGE;
    case FCMP_UGT: return FCMP_ULT;
    case FCMP_ULT: return FCMP_UGT;
    case FCMP_UGE: return FCMP_ULE;
    case FCMP_ULE: return FCMP_UGE;
  }
}

bool CmpInst::isUnsigned(unsigned short predicate) {
  switch (predicate) {
    default: return false;
    case ICmpInst::ICMP_ULT: case ICmpInst::ICMP_ULE: case ICmpInst::ICMP_UGT: 
    case ICmpInst::ICMP_UGE: return true;
  }
}

bool CmpInst::isSigned(unsigned short predicate){
  switch (predicate) {
    default: return false;
    case ICmpInst::ICMP_SLT: case ICmpInst::ICMP_SLE: case ICmpInst::ICMP_SGT: 
    case ICmpInst::ICMP_SGE: return true;
  }
}

bool CmpInst::isOrdered(unsigned short predicate) {
  switch (predicate) {
    default: return false;
    case FCmpInst::FCMP_OEQ: case FCmpInst::FCMP_ONE: case FCmpInst::FCMP_OGT: 
    case FCmpInst::FCMP_OLT: case FCmpInst::FCMP_OGE: case FCmpInst::FCMP_OLE: 
    case FCmpInst::FCMP_ORD: return true;
  }
}
      
bool CmpInst::isUnordered(unsigned short predicate) {
  switch (predicate) {
    default: return false;
    case FCmpInst::FCMP_UEQ: case FCmpInst::FCMP_UNE: case FCmpInst::FCMP_UGT: 
    case FCmpInst::FCMP_ULT: case FCmpInst::FCMP_UGE: case FCmpInst::FCMP_ULE: 
    case FCmpInst::FCMP_UNO: return true;
  }
}

//===----------------------------------------------------------------------===//
//                        SwitchInst Implementation
//===----------------------------------------------------------------------===//

void SwitchInst::init(Value *Value, BasicBlock *Default, unsigned NumCases) {
  assert(Value && Default);
  ReservedSpace = 2+NumCases*2;
  NumOperands = 2;
  OperandList = new Use[ReservedSpace];

  OperandList[0].init(Value, this);
  OperandList[1].init(Default, this);
}

/// SwitchInst ctor - Create a new switch instruction, specifying a value to
/// switch on and a default destination.  The number of additional cases can
/// be specified here to make memory allocation more efficient.  This
/// constructor can also autoinsert before another instruction.
SwitchInst::SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
                       Instruction *InsertBefore)
  : TerminatorInst(Type::VoidTy, Instruction::Switch, 0, 0, InsertBefore) {
  init(Value, Default, NumCases);
}

/// SwitchInst ctor - Create a new switch instruction, specifying a value to
/// switch on and a default destination.  The number of additional cases can
/// be specified here to make memory allocation more efficient.  This
/// constructor also autoinserts at the end of the specified BasicBlock.
SwitchInst::SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
                       BasicBlock *InsertAtEnd)
  : TerminatorInst(Type::VoidTy, Instruction::Switch, 0, 0, InsertAtEnd) {
  init(Value, Default, NumCases);
}

SwitchInst::SwitchInst(const SwitchInst &SI)
  : TerminatorInst(Type::VoidTy, Instruction::Switch,
                   new Use[SI.getNumOperands()], SI.getNumOperands()) {
  Use *OL = OperandList, *InOL = SI.OperandList;
  for (unsigned i = 0, E = SI.getNumOperands(); i != E; i+=2) {
    OL[i].init(InOL[i], this);
    OL[i+1].init(InOL[i+1], this);
  }
}

SwitchInst::~SwitchInst() {
  delete [] OperandList;
}


/// addCase - Add an entry to the switch instruction...
///
void SwitchInst::addCase(ConstantInt *OnVal, BasicBlock *Dest) {
  unsigned OpNo = NumOperands;
  if (OpNo+2 > ReservedSpace)
    resizeOperands(0);  // Get more space!
  // Initialize some new operands.
  assert(OpNo+1 < ReservedSpace && "Growing didn't work!");
  NumOperands = OpNo+2;
  OperandList[OpNo].init(OnVal, this);
  OperandList[OpNo+1].init(Dest, this);
}

/// removeCase - This method removes the specified successor from the switch
/// instruction.  Note that this cannot be used to remove the default
/// destination (successor #0).
///
void SwitchInst::removeCase(unsigned idx) {
  assert(idx != 0 && "Cannot remove the default case!");
  assert(idx*2 < getNumOperands() && "Successor index out of range!!!");

  unsigned NumOps = getNumOperands();
  Use *OL = OperandList;

  // Move everything after this operand down.
  //
  // FIXME: we could just swap with the end of the list, then erase.  However,
  // client might not expect this to happen.  The code as it is thrashes the
  // use/def lists, which is kinda lame.
  for (unsigned i = (idx+1)*2; i != NumOps; i += 2) {
    OL[i-2] = OL[i];
    OL[i-2+1] = OL[i+1];
  }

  // Nuke the last value.
  OL[NumOps-2].set(0);
  OL[NumOps-2+1].set(0);
  NumOperands = NumOps-2;
}

/// resizeOperands - resize operands - This adjusts the length of the operands
/// list according to the following behavior:
///   1. If NumOps == 0, grow the operand list in response to a push_back style
///      of operation.  This grows the number of ops by 1.5 times.
///   2. If NumOps > NumOperands, reserve space for NumOps operands.
///   3. If NumOps == NumOperands, trim the reserved space.
///
void SwitchInst::resizeOperands(unsigned NumOps) {
  if (NumOps == 0) {
    NumOps = getNumOperands()/2*6;
  } else if (NumOps*2 > NumOperands) {
    // No resize needed.
    if (ReservedSpace >= NumOps) return;
  } else if (NumOps == NumOperands) {
    if (ReservedSpace == NumOps) return;
  } else {
    return;
  }

  ReservedSpace = NumOps;
  Use *NewOps = new Use[NumOps];
  Use *OldOps = OperandList;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      NewOps[i].init(OldOps[i], this);
      OldOps[i].set(0);
  }
  delete [] OldOps;
  OperandList = NewOps;
}


BasicBlock *SwitchInst::getSuccessorV(unsigned idx) const {
  return getSuccessor(idx);
}
unsigned SwitchInst::getNumSuccessorsV() const {
  return getNumSuccessors();
}
void SwitchInst::setSuccessorV(unsigned idx, BasicBlock *B) {
  setSuccessor(idx, B);
}

//===----------------------------------------------------------------------===//
//                           GetResultInst Implementation
//===----------------------------------------------------------------------===//

GetResultInst::GetResultInst(Value *Aggregate, unsigned Index,
                             const std::string &Name,
                             Instruction *InsertBef)
  : Instruction(cast<StructType>(Aggregate->getType())->getElementType(Index),
                GetResult, &Aggr, 1, InsertBef) {
  assert(isValidOperands(Aggregate, Index) && "Invalid GetResultInst operands!");
  Aggr.init(Aggregate, this);
  Idx = Index;
  setName(Name);
}

bool GetResultInst::isValidOperands(const Value *Aggregate, unsigned Index) {
  if (!Aggregate)
    return false;

  if (const StructType *STy = dyn_cast<StructType>(Aggregate->getType())) {
    unsigned NumElements = STy->getNumElements();
    if (Index >= NumElements)
      return false;

    // getresult aggregate value's element types are restricted to
    // avoid nested aggregates.
    for (unsigned i = 0; i < NumElements; ++i)
      if (!STy->getElementType(i)->isFirstClassType())
        return false;

    // Otherwise, Aggregate is valid.
    return true;
  }
  return false;
}

// Define these methods here so vtables don't get emitted into every translation
// unit that uses these classes.

GetElementPtrInst *GetElementPtrInst::clone() const {
  return new GetElementPtrInst(*this);
}

BinaryOperator *BinaryOperator::clone() const {
  return create(getOpcode(), Ops[0], Ops[1]);
}

FCmpInst* FCmpInst::clone() const {
  return new FCmpInst(getPredicate(), Ops[0], Ops[1]);
}
ICmpInst* ICmpInst::clone() const {
  return new ICmpInst(getPredicate(), Ops[0], Ops[1]);
}

MallocInst *MallocInst::clone()   const { return new MallocInst(*this); }
AllocaInst *AllocaInst::clone()   const { return new AllocaInst(*this); }
FreeInst   *FreeInst::clone()     const { return new FreeInst(getOperand(0)); }
LoadInst   *LoadInst::clone()     const { return new LoadInst(*this); }
StoreInst  *StoreInst::clone()    const { return new StoreInst(*this); }
CastInst   *TruncInst::clone()    const { return new TruncInst(*this); }
CastInst   *ZExtInst::clone()     const { return new ZExtInst(*this); }
CastInst   *SExtInst::clone()     const { return new SExtInst(*this); }
CastInst   *FPTruncInst::clone()  const { return new FPTruncInst(*this); }
CastInst   *FPExtInst::clone()    const { return new FPExtInst(*this); }
CastInst   *UIToFPInst::clone()   const { return new UIToFPInst(*this); }
CastInst   *SIToFPInst::clone()   const { return new SIToFPInst(*this); }
CastInst   *FPToUIInst::clone()   const { return new FPToUIInst(*this); }
CastInst   *FPToSIInst::clone()   const { return new FPToSIInst(*this); }
CastInst   *PtrToIntInst::clone() const { return new PtrToIntInst(*this); }
CastInst   *IntToPtrInst::clone() const { return new IntToPtrInst(*this); }
CastInst   *BitCastInst::clone()  const { return new BitCastInst(*this); }
CallInst   *CallInst::clone()     const { return new CallInst(*this); }
SelectInst *SelectInst::clone()   const { return new SelectInst(*this); }
VAArgInst  *VAArgInst::clone()    const { return new VAArgInst(*this); }

ExtractElementInst *ExtractElementInst::clone() const {
  return new ExtractElementInst(*this);
}
InsertElementInst *InsertElementInst::clone() const {
  return new InsertElementInst(*this);
}
ShuffleVectorInst *ShuffleVectorInst::clone() const {
  return new ShuffleVectorInst(*this);
}
PHINode    *PHINode::clone()    const { return new PHINode(*this); }
ReturnInst *ReturnInst::clone() const { return new ReturnInst(*this); }
BranchInst *BranchInst::clone() const { return new BranchInst(*this); }
SwitchInst *SwitchInst::clone() const { return new SwitchInst(*this); }
InvokeInst *InvokeInst::clone() const { return new InvokeInst(*this); }
UnwindInst *UnwindInst::clone() const { return new UnwindInst(); }
UnreachableInst *UnreachableInst::clone() const { return new UnreachableInst();}
GetResultInst *GetResultInst::clone() const { return new GetResultInst(*this); }

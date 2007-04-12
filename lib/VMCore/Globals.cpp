//===-- Globals.cpp - Implement the GlobalValue & GlobalVariable class ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the GlobalValue & GlobalVariable classes for the VMCore
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Support/LeakDetector.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                            GlobalValue Class
//===----------------------------------------------------------------------===//

/// removeDeadUsersOfConstant - If the specified constantexpr is dead, remove
/// it.  This involves recursively eliminating any dead users of the
/// constantexpr.
static bool removeDeadUsersOfConstant(Constant *C) {
  if (isa<GlobalValue>(C)) return false; // Cannot remove this

  while (!C->use_empty()) {
    Constant *User = dyn_cast<Constant>(C->use_back());
    if (!User) return false; // Non-constant usage;
    if (!removeDeadUsersOfConstant(User))
      return false; // Constant wasn't dead
  }

  C->destroyConstant();
  return true;
}

/// removeDeadConstantUsers - If there are any dead constant users dangling
/// off of this global value, remove them.  This method is useful for clients
/// that want to check to see if a global is unused, but don't want to deal
/// with potentially dead constants hanging off of the globals.
void GlobalValue::removeDeadConstantUsers() {
  Value::use_iterator I = use_begin(), E = use_end();
  Value::use_iterator LastNonDeadUser = E;
  while (I != E) {
    if (Constant *User = dyn_cast<Constant>(*I)) {
      if (!removeDeadUsersOfConstant(User)) {
        // If the constant wasn't dead, remember that this was the last live use
        // and move on to the next constant.
        LastNonDeadUser = I;
        ++I;
      } else {
        // If the constant was dead, then the iterator is invalidated.
        if (LastNonDeadUser == E) {
          I = use_begin();
          if (I == E) break;
        } else {
          I = LastNonDeadUser;
          ++I;
        }
      }
    } else {
      LastNonDeadUser = I;
      ++I;
    }
  }
}

/// Override destroyConstant to make sure it doesn't get called on
/// GlobalValue's because they shouldn't be treated like other constants.
void GlobalValue::destroyConstant() {
  assert(0 && "You can't GV->destroyConstant()!");
  abort();
}
//===----------------------------------------------------------------------===//
// GlobalVariable Implementation
//===----------------------------------------------------------------------===//

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, LinkageTypes Link,
                               Constant *InitVal, const std::string &Name,
                               Module *ParentModule, bool ThreadLocal)
  : GlobalValue(PointerType::get(Ty), Value::GlobalVariableVal,
                &Initializer, InitVal != 0, Link, Name),
    isConstantGlobal(constant), isThreadLocalSymbol(ThreadLocal) {
  if (InitVal) {
    assert(InitVal->getType() == Ty &&
           "Initializer should be the same type as the GlobalVariable!");
    Initializer.init(InitVal, this);
  } else {
    Initializer.init(0, this);
  }

  LeakDetector::addGarbageObject(this);

  if (ParentModule)
    ParentModule->getGlobalList().push_back(this);
}

GlobalVariable::GlobalVariable(const Type *Ty, bool constant, LinkageTypes Link,
                               Constant *InitVal, const std::string &Name,
                               GlobalVariable *Before, bool ThreadLocal)
  : GlobalValue(PointerType::get(Ty), Value::GlobalVariableVal,
                &Initializer, InitVal != 0, Link, Name), 
    isConstantGlobal(constant), isThreadLocalSymbol(ThreadLocal) {
  if (InitVal) {
    assert(InitVal->getType() == Ty &&
           "Initializer should be the same type as the GlobalVariable!");
    Initializer.init(InitVal, this);
  } else {
    Initializer.init(0, this);
  }
  
  LeakDetector::addGarbageObject(this);
  
  if (Before)
    Before->getParent()->getGlobalList().insert(Before, this);
}


void GlobalVariable::setParent(Module *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);
  Parent = parent;
  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void GlobalVariable::removeFromParent() {
  getParent()->getGlobalList().remove(this);
}

void GlobalVariable::eraseFromParent() {
  getParent()->getGlobalList().erase(this);
}

void GlobalVariable::replaceUsesOfWithOnConstant(Value *From, Value *To,
                                                 Use *U) {
  // If you call this, then you better know this GVar has a constant
  // initializer worth replacing. Enforce that here.
  assert(getNumOperands() == 1 &&
         "Attempt to replace uses of Constants on a GVar with no initializer");

  // And, since you know it has an initializer, the From value better be
  // the initializer :)
  assert(getOperand(0) == From &&
         "Attempt to replace wrong constant initializer in GVar");

  // And, you better have a constant for the replacement value
  assert(isa<Constant>(To) &&
         "Attempt to replace GVar initializer with non-constant");

  // Okay, preconditions out of the way, replace the constant initializer.
  this->setOperand(0, cast<Constant>(To));
}

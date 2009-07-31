//===-- LLVMContext.cpp - Implement LLVMContext -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
// class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/Metadata.h"
#include "llvm/Support/ManagedStatic.h"
#include "LLVMContextImpl.h"
#include <cstdarg>

using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) { }
LLVMContext::~LLVMContext() { delete pImpl; }

// MDNode accessors
MDNode* LLVMContext::getMDNode(Value* const* Vals, unsigned NumVals) {
  return pImpl->getMDNode(Vals, NumVals);
}

// MDString accessors
MDString* LLVMContext::getMDString(const StringRef &Str) {
  return pImpl->getMDString(Str.data(), Str.size());
}

void LLVMContext::erase(MDString *M) {
  pImpl->erase(M);
}

void LLVMContext::erase(MDNode *M) {
  pImpl->erase(M);
}

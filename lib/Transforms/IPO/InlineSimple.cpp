//===- InlineSimple.cpp - Code to perform simple function inlining --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "inline"
#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO/InlinerPass.h"

using namespace llvm;

namespace {

/// \brief Actaul inliner pass implementation.
///
/// The common implementation of the inlining logic is shared between this
/// inliner pass and the always inliner pass. The two passes use different cost
/// analyses to determine when to inline.
class SimpleInliner : public Inliner {
  InlineCostAnalyzer CA;

public:
  SimpleInliner() : Inliner(ID) {
    initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
  }

  SimpleInliner(int Threshold)
      : Inliner(ID, Threshold, /*InsertLifetime*/ true) {
    initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
  }

  static char ID; // Pass identification, replacement for typeid

  InlineCost getInlineCost(CallSite CS) {
    return CA.getInlineCost(CS, getInlineThreshold(CS));
  }

  using llvm::Pass::doInitialization;
  virtual bool doInitialization(CallGraph &CG);
};

} // end anonymous namespace

char SimpleInliner::ID = 0;
INITIALIZE_PASS_BEGIN(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)
INITIALIZE_AG_DEPENDENCY(CallGraph)
INITIALIZE_PASS_END(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)

Pass *llvm::createFunctionInliningPass() { return new SimpleInliner(); }

Pass *llvm::createFunctionInliningPass(int Threshold) {
  return new SimpleInliner(Threshold);
}

// doInitialization - Initializes the vector of functions that have been
// annotated with the noinline attribute.
bool SimpleInliner::doInitialization(CallGraph &CG) {
  CA.setDataLayout(getAnalysisIfAvailable<DataLayout>());
  return false;
}


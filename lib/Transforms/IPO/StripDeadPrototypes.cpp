//===-- StripDeadPrototypes.cpp - Removed unused function declarations ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass loops over all of the functions in the input module, looking for 
// dead declarations and removes them.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strip-dead-prototypes"
#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumDeadPrototypes, "Number of dead prototypes removed");

namespace {

/// @brief Pass to remove unused function declarations.
class VISIBILITY_HIDDEN StripDeadPrototypesPass : public ModulePass {
public:
  static char ID; // Pass identifcation, replacement for typeid
  StripDeadPrototypesPass() : ModulePass((intptr_t)&ID) { }
  virtual bool runOnModule(Module &M);
};

char StripDeadPrototypesPass::ID = 0;
RegisterPass<StripDeadPrototypesPass> X("strip-dead-prototypes", 
                                        "Strip Unused Function Prototypes");

} // end anonymous namespace

bool StripDeadPrototypesPass::runOnModule(Module &M) {
  bool MadeChange = false;
  
  // Erase dead function prototypes.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    Function *F = I++;
    // Function must be a prototype and unused.
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      ++NumDeadPrototypes;
      MadeChange = true;
    }
  }

  // Erase dead global var prototypes.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ) {
    GlobalVariable *GV = I++;
    // Global must be a prototype and unused.
    if (GV->isDeclaration() && GV->use_empty())
      GV->eraseFromParent();
  }
  
  // Return an indication of whether we changed anything or not.
  return MadeChange;
}

ModulePass *llvm::createStripDeadPrototypesPass() {
  return new StripDeadPrototypesPass();
}

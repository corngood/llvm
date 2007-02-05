//===-- ExtractFunction.cpp - Function extraction pass --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass extracts
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO.h"
using namespace llvm;

namespace {
  class FunctionExtractorPass : public ModulePass {
    Function *Named;
    bool deleteFunc;
    bool reLink;
  public:
    /// FunctionExtractorPass - If deleteFn is true, this pass deletes as the
    /// specified function. Otherwise, it deletes as much of the module as
    /// possible, except for the function specified.
    ///
    FunctionExtractorPass(Function *F = 0, bool deleteFn = true,
                          bool relinkCallees = false)
      : Named(F), deleteFunc(deleteFn), reLink(relinkCallees) {}

    bool runOnModule(Module &M) {
      if (Named == 0) {
        Named = M.getFunction("main");
        if (Named == 0) return false;  // No function to extract
      }
      
      if (deleteFunc)
        return deleteFunction();
      M.setModuleInlineAsm("");
      return isolateFunction(M);
    }

    bool deleteFunction() {
      // If we're in relinking mode, set linkage of all internal callees to
      // external. This will allow us extract function, and then - link
      // everything together
      if (reLink) {
        for (Function::iterator B = Named->begin(), BE = Named->end();
             B != BE; ++B) {
          for (BasicBlock::iterator I = B->begin(), E = B->end();
               I != E; ++I) {
            if (CallInst* callInst = dyn_cast<CallInst>(&*I)) {
              Function* Callee = callInst->getCalledFunction();
              if (Callee && Callee->hasInternalLinkage())
                Callee->setLinkage(GlobalValue::ExternalLinkage);
            }
          }
        }
      }
      
      Named->setLinkage(GlobalValue::ExternalLinkage);
      Named->deleteBody();
      assert(Named->isDeclaration() && "This didn't make the function external!");
      return true;
    }

    bool isolateFunction(Module &M) {
      // Make sure our result is globally accessible...
      Named->setLinkage(GlobalValue::ExternalLinkage);

      // Mark all global variables internal
      for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
        if (!I->isDeclaration()) {
          I->setInitializer(0);  // Make all variables external
          I->setLinkage(GlobalValue::ExternalLinkage);
        }

      // All of the functions may be used by global variables or the named
      // function.  Loop through them and create a new, external functions that
      // can be "used", instead of ones with bodies.
      std::vector<Function*> NewFunctions;

      Function *Last = --M.end();  // Figure out where the last real fn is.

      for (Module::iterator I = M.begin(); ; ++I) {
        if (&*I != Named) {
          Function *New = new Function(I->getFunctionType(),
                                       GlobalValue::ExternalLinkage,
                                       I->getName());
          New->setCallingConv(I->getCallingConv());
          I->setName("");  // Remove Old name

          // If it's not the named function, delete the body of the function
          I->dropAllReferences();

          M.getFunctionList().push_back(New);
          NewFunctions.push_back(New);
        }

        if (&*I == Last) break;  // Stop after processing the last function
      }

      // Now that we have replacements all set up, loop through the module,
      // deleting the old functions, replacing them with the newly created
      // functions.
      if (!NewFunctions.empty()) {
        unsigned FuncNum = 0;
        Module::iterator I = M.begin();
        do {
          if (&*I != Named) {
            // Make everything that uses the old function use the new dummy fn
            I->replaceAllUsesWith(NewFunctions[FuncNum++]);

            Function *Old = I;
            ++I;  // Move the iterator to the new function

            // Delete the old function!
            M.getFunctionList().erase(Old);

          } else {
            ++I;  // Skip the function we are extracting
          }
        } while (&*I != NewFunctions[0]);
      }

      return true;
    }
  };

  RegisterPass<FunctionExtractorPass> X("extract", "Function Extractor");
}

ModulePass *llvm::createFunctionExtractionPass(Function *F, bool deleteFn,
                                               bool relinkCallees) {
  return new FunctionExtractorPass(F, deleteFn, relinkCallees);
}

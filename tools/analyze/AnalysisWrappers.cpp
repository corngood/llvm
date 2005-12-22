//===- AnalysisWrappers.cpp - Wrappers around non-pass analyses -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines pass wrappers around LLVM analyses that don't make sense to
// be passes.  It provides a nice standard pass interface to these classes so
// that they can be printed out by analyze.
//
// These classes are separated out of analyze.cpp so that it is more clear which
// code is the integral part of the analyze tool, and which part of the code is
// just making it so more passes are available.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Analysis/CallGraph.h"
#include <iostream>
using namespace llvm;

namespace {
  /// ExternalFunctionsPassedConstants - This pass prints out call sites to
  /// external functions that are called with constant arguments.  This can be
  /// useful when looking for standard library functions we should constant fold
  /// or handle in alias analyses.
  struct ExternalFunctionsPassedConstants : public ModulePass {
    virtual bool runOnModule(Module &M) {
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
        if (I->isExternal()) {
          bool PrintedFn = false;
          for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
               UI != E; ++UI)
            if (Instruction *User = dyn_cast<Instruction>(*UI)) {
              CallSite CS = CallSite::get(User);
              if (CS.getInstruction()) {
                for (CallSite::arg_iterator AI = CS.arg_begin(),
                       E = CS.arg_end(); AI != E; ++AI)
                  if (isa<Constant>(*AI)) {
                    if (!PrintedFn) {
                      std::cerr << "Function '" << I->getName() << "':\n";
                      PrintedFn = true;
                    }
                    std::cerr << *User;
                    break;
                  }
              }
            }
        }

      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  RegisterAnalysis<ExternalFunctionsPassedConstants>
  P1("externalfnconstants", "Print external fn callsites passed constants");
  
  struct CallGraphPrinter : public ModulePass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<CallGraph>();
    }
    virtual bool runOnModule(Module &M) { return false; }

    void print(std::ostream &OS, Module *M) const {
      getAnalysis<CallGraph>().print(OS, M);
    }
  };
  
  RegisterAnalysis<CallGraphPrinter>
    P2("callgraph", "Print a call graph");
}

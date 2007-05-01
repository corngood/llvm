//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hello"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(HelloCounter, "Counts number of functions greeted");

namespace {
  // Hello - The first implementation, without getAnalysisUsage.
  struct Hello : public FunctionPass {
    static const int ID; // Pass identifcation, replacement for typeid
    Hello() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F) {
      HelloCounter++;
      std::string fname = F.getName();
      EscapeString(fname);
      cerr << "Hello: " << fname << "\n";
      return false;
    }
  };

  const int Hello::ID = 0;
  RegisterPass<Hello> X("hello", "Hello World Pass");

  // Hello2 - The second implementation with getAnalysisUsage implemented.
  struct Hello2 : public FunctionPass {
    static const int ID; // Pass identifcation, replacement for typeid
    Hello2() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F) {
      HelloCounter++;
      std::string fname = F.getName();
      EscapeString(fname);
      cerr << "Hello: " << fname << "\n";
      return false;
    }

    // We don't modify the program, so we preserve all analyses
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    };
  };
  const int Hello2::ID = 0;
  RegisterPass<Hello2> Y("hello2",
                        "Hello World Pass (with getAnalysisUsage implemented)");
}

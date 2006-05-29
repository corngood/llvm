//===- llvm/Assembly/PrintModulePass.h - Printing Pass ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two passes to print out a module.  The PrintModulePass pass
// simply prints out the entire module when it is executed.  The
// PrintFunctionPass class is designed to be pipelined with other
// FunctionPass's, and prints out the functions of the class as they are
// processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PRINTMODULEPASS_H
#define LLVM_ASSEMBLY_PRINTMODULEPASS_H

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include <iostream>

namespace llvm {

class PrintModulePass : public ModulePass {
  std::ostream *Out;      // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
public:
  PrintModulePass() : Out(&std::cerr), DeleteStream(false) {}
  PrintModulePass(std::ostream *o, bool DS = false)
    : Out(o), DeleteStream(DS) {
  }

  ~PrintModulePass() {
    if (DeleteStream) delete Out;
  }

  bool runOnModule(Module &M) {
    (*Out) << M << std::flush;
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

class PrintFunctionPass : public FunctionPass {
  std::string Banner;     // String to print before each function
  std::ostream *Out;      // ostream to print on
  bool DeleteStream;      // Delete the ostream in our dtor?
public:
  PrintFunctionPass() : Banner(""), Out(&std::cerr), DeleteStream(false) {}
  PrintFunctionPass(const std::string &B, std::ostream *o = &std::cout,
                    bool DS = false)
    : Banner(B), Out(o), DeleteStream(DS) {
  }

  inline ~PrintFunctionPass() {
    if (DeleteStream) delete Out;
  }

  // runOnFunction - This pass just prints a banner followed by the function as
  // it's processed.
  //
  bool runOnFunction(Function &F) {
    (*Out) << Banner << static_cast<Value&>(F);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

} // End llvm namespace

#endif

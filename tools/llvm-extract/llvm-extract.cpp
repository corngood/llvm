//===- llvm-extract.cpp - LLVM function extraction utility ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility changes the input module to only contain a single function,
// which is primarily used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bytecode/Reader.h"
#include "llvm/Bytecode/WriteBytecodePass.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Signals.h"
#include <iostream>
#include <memory>
#include <fstream>
using namespace llvm;

// InputFilename - The filename to read from.
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode file>"),
              cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Specify output filename"),
               cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DeleteFn("delete", cl::desc("Delete specified function from Module"));

// ExtractFunc - The function to extract from the module... defaults to main.
static cl::opt<std::string>
ExtractFunc("func", cl::desc("Specify function to extract"), cl::init("main"),
            cl::value_desc("function"));

int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  try {
    cl::ParseCommandLineOptions(argc, argv, " llvm extractor\n");
    sys::PrintStackTraceOnErrorSignal();

    std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
    if (M.get() == 0) {
      llvm_cerr << argv[0] << ": bytecode didn't read correctly.\n";
      return 1;
    }

    // Figure out which function we should extract
    Function *F = M.get()->getNamedFunction(ExtractFunc);
    if (F == 0) {
      llvm_cerr << argv[0] << ": program doesn't contain function named '"
                << ExtractFunc << "'!\n";
      return 1;
    }

    // In addition to deleting all other functions, we also want to spiff it
    // up a little bit.  Do this now.
    PassManager Passes;
    Passes.add(new TargetData(M.get())); // Use correct TargetData
    // Either isolate the function or delete it from the Module
    Passes.add(createFunctionExtractionPass(F, DeleteFn));
    Passes.add(createGlobalDCEPass());             // Delete unreachable globals
    Passes.add(createFunctionResolvingPass());     // Delete prototypes
    Passes.add(createDeadTypeEliminationPass());   // Remove dead types...

    std::ostream *Out = 0;

    if (OutputFilename != "-") {  // Not stdout?
      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        llvm_cerr << argv[0] << ": error opening '" << OutputFilename
                  << "': file exists!\n"
                  << "Use -f command line argument to force output\n";
        return 1;
      }
      std::ios::openmode io_mode = std::ios::out | std::ios::trunc |
                                   std::ios::binary;
      Out = new std::ofstream(OutputFilename.c_str(), io_mode);
    } else {                      // Specified stdout
      // FIXME: cout is not binary!
      Out = &std::cout;
    }

    llvm_ostream L(*Out);
    Passes.add(new WriteBytecodePass(&L));  // Write bytecode to file...
    Passes.run(*M.get());

    if (Out != &std::cout)
      delete Out;
    return 0;
  } catch (const std::string& msg) {
    llvm_cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    llvm_cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}

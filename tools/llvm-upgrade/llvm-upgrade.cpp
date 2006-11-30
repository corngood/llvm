//===--- llvm-upgrade.cpp - The LLVM Assembly Upgrader --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This utility will upgrade LLVM 1.9 Assembly to 2.0 format. It may be 
//  invoked as a filter, like this:
//    llvm-1.9/bin/llvm-dis < 1.9.bc | llvm-upgrade | llvm-as > 2.0.bc
//  
//  or, you can directly upgrade, like this:
//    llvm-upgrade -o 2.0.ll < 1.9.ll
//  
//  llvm-upgrade won't overwrite files by default. Use -f to force it to
//  overwrite the output file.
//
//===----------------------------------------------------------------------===//

#include "ParserInternals.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Signals.h"
#include <fstream>
#include <iostream>
#include <memory>
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input .llvm file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, " llvm .ll -> .bc assembler\n");
  sys::PrintStackTraceOnErrorSignal();

  int exitCode = 0;
  std::ostream *Out = 0;
  try {
    if (OutputFilename != "") {   // Specified an output filename?
      if (OutputFilename != "-") {  // Not stdout?
        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          llvm_cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }
        Out = new std::ofstream(OutputFilename.c_str(), std::ios::out |
                                std::ios::trunc);
      } else {                      // Specified stdout
        Out = &std::cout;
      }
    } else {
      if (InputFilename == "-") {
        OutputFilename = "-";
        Out = &std::cout;
      } else {
        std::string IFN = InputFilename;
        int Len = IFN.length();
        if (IFN[Len-3] == '.' && IFN[Len-2] == 'l' && IFN[Len-1] == 'l') {
          // Source ends in .ll
          OutputFilename = std::string(IFN.begin(), IFN.end()-3);
        } else {
          OutputFilename = IFN;   // Append to it
        }
        OutputFilename += ".llu";

        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          llvm_cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }

        Out = new std::ofstream(OutputFilename.c_str(), std::ios::out |
                                std::ios::trunc | std::ios::binary);
        // Make sure that the Out file gets unlinked from the disk if we get a
        // SIGINT
        sys::RemoveFileOnSignal(sys::Path(OutputFilename));
      }
    }

    if (!Out->good()) {
      llvm_cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
      return 1;
    }

    UpgradeAssembly(InputFilename, *Out);

    /*
  } catch (const std::string& caught_message) {
    llvm_cerr << argv[0] << ": " << caught_message << "\n";
    exitCode = 1;
    */
  } catch (...) {
    llvm_cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
    exitCode = 1;
  }

  if (Out != &std::cout) delete Out;
  return exitCode;
}


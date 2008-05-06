//===--- llvmcc.cpp - The LLVM Compiler Driver ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This tool provides a single point of access to the LLVM
//  compilation tools.  It has many options. To discover the options
//  supported please refer to the tools' manual page or run the tool
//  with the --help option.
//
//===----------------------------------------------------------------------===//

#include "CompilationGraph.h"
#include "Tool.h"

#include "llvm/System/Path.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace cl = llvm::cl;
namespace sys = llvm::sys;
using namespace llvmcc;

// Built-in command-line options.
// External linkage here is intentional.

cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input file>"),
                                     cl::ZeroOrMore);
cl::opt<std::string> OutputFilename("o", cl::desc("Output file name"),
                                    cl::value_desc("file"));
cl::opt<bool> VerboseMode("v",
                          cl::desc("Enable verbose mode"));
cl::opt<bool> WriteGraph("write-graph",
                         cl::desc("Write CompilationGraph.dot file"),
                         cl::Hidden);
cl::opt<bool> ViewGraph("view-graph",
                         cl::desc("Show compilation graph in GhostView"),
                         cl::Hidden);

namespace {
  int BuildTargets(const CompilationGraph& graph) {
    int ret;
    sys::Path tempDir(sys::Path::GetTemporaryDirectory());

    try {
      ret = graph.Build(tempDir);
    }
    catch(...) {
      tempDir.eraseFromDisk(true);
      throw;
    }

    tempDir.eraseFromDisk(true);
    return ret;
  }
}

int main(int argc, char** argv) {
  try {
    CompilationGraph graph;

    cl::ParseCommandLineOptions(argc, argv,
                                "LLVM Compiler Driver(Work In Progress)");
    PopulateCompilationGraph(graph);

    if (WriteGraph) {
      graph.writeGraph();
      return 0;
    }

    if (ViewGraph) {
      graph.viewGraph();
      return 0;
    }

    if (InputFilenames.empty()) {
      std::cerr << "No input files.\n";
      return 1;
    }

    return BuildTargets(graph);
  }
  catch(const std::exception& ex) {
    std::cerr << ex.what() << '\n';
  }
  catch(...) {
    std::cerr << "Unknown error!\n";
  }
  return 1;
}

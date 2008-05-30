//===--- Tools.h - The LLVM Compiler Driver ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Action - encapsulates a single shell command.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC2_ACTION_H
#define LLVM_TOOLS_LLVMC2_ACTION_H

#include <string>
#include <vector>

namespace llvmc {

  typedef std::vector<std::string> StringVector;

  /// Action - A class that encapsulates a single shell command.
  class Action {
    /// Command_ - The actual command (for example, 'ls').
    std::string Command_;
    /// Args_ - Command arguments. Stdout redirection ("> file") is allowed.
    std::vector<std::string> Args_;
  public:
    Action() {}
    Action (const std::string& C,
            const StringVector& A)
      : Command_(C), Args_(A)
    {}

    /// Execute - Executes the represented action.
    int Execute() const;
  };

}

#endif // LLVM_TOOLS_LLVMC2_ACTION_H

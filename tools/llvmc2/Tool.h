//===--- Tools.h - The LLVM Compiler Driver ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Tool abstract base class - an interface to tool descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC2_TOOL_H
#define LLVM_TOOLS_LLVMC2_TOOL_H

#include "Action.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/System/Path.h"

#include <string>
#include <vector>

namespace llvmc {

  typedef std::vector<llvm::sys::Path> PathVector;

  class Tool : public llvm::RefCountedBaseVPTR<Tool> {
  public:

    virtual ~Tool() {}

    virtual Action GenerateAction (const PathVector& inFiles,
                                   const llvm::sys::Path& outFile) const = 0;

    virtual Action GenerateAction (const llvm::sys::Path& inFile,
                                   const llvm::sys::Path& outFile) const = 0;

    virtual const char* Name() const = 0;
    virtual const char* InputLanguage() const = 0;
    virtual const char* OutputLanguage() const = 0;
    virtual const char* OutputSuffix() const = 0;

    virtual bool IsLast() const = 0;
    virtual bool IsJoin() const = 0;

    // Helper function that is called by the auto-generated code
    // Splits strings of the form ",-foo,-bar,-baz"
    // TOFIX: find a better name
    static void UnpackValues (std::string const& from,
                              std::vector<std::string>& to);
  };

  // Join tools have an input file list associated with them.
  class JoinTool : public Tool {
  public:
    void AddToJoinList(const llvm::sys::Path& P) { JoinList_.push_back(P); }
    void ClearJoinList() { JoinList_.clear(); }
    bool JoinListEmpty() const { return JoinList_.empty(); }

    Action GenerateAction(const llvm::sys::Path& outFile) const
    { return GenerateAction(JoinList_, outFile); }
    // We shouldn't shadow base class's version of GenerateAction.
    using Tool::GenerateAction;

  private:
    PathVector JoinList_;
  };

}

#endif //LLVM_TOOLS_LLVMC2_TOOL_H

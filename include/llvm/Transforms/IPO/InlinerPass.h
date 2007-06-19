//===- InlinerPass.h - Code common to all inliners --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple policy-based bottom-up inliner.  This file
// implements all of the boring mechanics of the bottom-up inlining, while the
// subclass determines WHAT to inline, which is the much more interesting
// component.
//
//===----------------------------------------------------------------------===//

#ifndef INLINER_H
#define INLINER_H

#include "llvm/CallGraphSCCPass.h"

namespace llvm {
  class CallSite;

/// Inliner - This class contains all of the helper code which is used to
/// perform the inlining operations that does not depend on the policy.
///
struct Inliner : public CallGraphSCCPass {
  Inliner(const void *ID);

  /// getAnalysisUsage - For this class, we declare that we require and preserve
  /// the call graph.  If the derived class implements this method, it should
  /// always explicitly call the implementation here.
  virtual void getAnalysisUsage(AnalysisUsage &Info) const;

  // Main run interface method, this implements the interface required by the
  // Pass class.
  virtual bool runOnSCC(const std::vector<CallGraphNode *> &SCC);

  // doFinalization - Remove now-dead linkonce functions at the end of
  // processing to avoid breaking the SCC traversal.
  virtual bool doFinalization(CallGraph &CG);


  /// This method returns the value specified by the -inline-threshold value,
  /// specified on the command line.  This is typically not directly needed.
  ///
  unsigned getInlineThreshold() const { return InlineThreshold; }

  /// getInlineCost - This method must be implemented by the subclass to
  /// determine the cost of inlining the specified call site.  If the cost
  /// returned is greater than the current inline threshold, the call site is
  /// not inlined.
  ///
  virtual int getInlineCost(CallSite CS) = 0;

private:
  // InlineThreshold - Cache the value here for easy access.
  unsigned InlineThreshold;
};

} // End llvm namespace

#endif

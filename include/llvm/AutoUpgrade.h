//===-- llvm/AutoUpgrade.h - AutoUpgrade Helpers ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chandler Carruth is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These functions are implemented by lib/VMCore/AutoUpgrade.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AUTOUPGRADE_H
#define LLVM_AUTOUPGRADE_H

namespace llvm {
  class Function;
  class CallInst;
  class BasicBlock;

  /// This is a more granular function that simply checks an intrinsic function 
  /// for upgrading, and if it requires upgrading provides the new function.
  Function* UpgradeIntrinsicFunction(Function *F);

  /// This is the complement to the above, replacing a specific call to an 
  /// intrinsic function with a call to the specified new function.
  void UpgradeIntrinsicCall(CallInst *CI, Function *NewFn);
  
  /// This is an auto-upgrade hook for any old intrinsic function syntaxes 
  /// which need to have both the function updated as well as all calls updated 
  /// to the new function. This should only be run in a post-processing fashion 
  /// so that it can update all calls to the old function.
  void UpgradeCallsToIntrinsic(Function* F);

} // End llvm namespace

#endif

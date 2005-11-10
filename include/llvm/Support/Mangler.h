//===-- llvm/Support/Mangler.h - Self-contained name mangler ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unified name mangler for various backends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MANGLER_H
#define LLVM_SUPPORT_MANGLER_H

#include <map>
#include <set>
#include <string>

namespace llvm {
class Type;
class Module;
class Value;
class GlobalValue;

class Mangler {
  /// This keeps track of which global values have had their names
  /// mangled in the current module.
  ///
  std::set<const GlobalValue*> MangledGlobals;

  Module &M;
  const char *Prefix;

  unsigned TypeCounter;
  std::map<const Type*, unsigned> TypeMap;

  typedef std::map<const Value*, std::string> ValueMap;
  ValueMap Memo;

  unsigned Count;

  void InsertName(GlobalValue *GV, std::map<std::string, GlobalValue*> &Names);
public:

  // Mangler ctor - if a prefix is specified, it will be prepended onto all
  // symbols.
  Mangler(Module &M, const char *Prefix = "");

  /// getTypeID - Return a unique ID for the specified LLVM type.
  ///
  unsigned getTypeID(const Type *Ty);

  /// getValueName - Returns the mangled name of V, an LLVM Value,
  /// in the current module.
  ///
  std::string getValueName(const GlobalValue *V);
  std::string getValueName(const Value *V);

  /// makeNameProper - We don't want identifier names with ., space, or
  /// - in them, so we mangle these characters into the strings "d_",
  /// "s_", and "D_", respectively. This is a very simple mangling that
  /// doesn't guarantee unique names for values. getValueName already
  /// does this for you, so there's no point calling it on the result
  /// from getValueName.
  ///
  static std::string makeNameProper(const std::string &x,
                                    const char *Prefix = "");
};

} // End llvm namespace

#endif // LLVM_SUPPORT_MANGLER_H

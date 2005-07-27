//===-- SparcV9FunctionInfo.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class keeps track of information about the stack frame and about the
// per-function constant pool.
//
// FIXME: This class is completely SparcV9 specific.  Do not use it for future
// targets.  This file will be eliminated in future versions of LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef MACHINEFUNCTIONINFO_H
#define MACHINEFUNCTIONINFO_H

#include "MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ADT/HashExtras.h"
#include "llvm/ADT/hash_set"

namespace llvm {

class MachineFunction;
class Constant;
class Type;

class SparcV9FunctionInfo : public MachineFunctionInfo {
  hash_set<const Constant*> constantsForConstPool;
  hash_map<const Value*, int> offsets;

  unsigned      staticStackSize;
  unsigned      automaticVarsSize;
  unsigned      regSpillsSize;
  unsigned      maxOptionalArgsSize;
  unsigned      maxOptionalNumArgs;
  unsigned      currentTmpValuesSize;
  unsigned      maxTmpValuesSize;
  bool          compiledAsLeaf;
  bool          spillsAreaFrozen;
  bool          automaticVarsAreaFrozen;

  MachineFunction &MF;
public:
  hash_map<const Instruction*, MachineCodeForInstruction> MCFIEntries;

  SparcV9FunctionInfo(MachineFunction &mf) : MF(mf) {
    staticStackSize = automaticVarsSize = regSpillsSize = 0;
    maxOptionalArgsSize = maxOptionalNumArgs = currentTmpValuesSize = 0;
    maxTmpValuesSize = 0;
    compiledAsLeaf = spillsAreaFrozen = automaticVarsAreaFrozen = false;
  }

  /// CalculateArgSize - Call this method to fill in the maxOptionalArgsSize &
  /// staticStackSize fields...
  ///
  void CalculateArgSize();

  //
  // Accessors for global information about generated code for a method.
  //
  bool     isCompiledAsLeafMethod() const { return compiledAsLeaf; }
  unsigned getStaticStackSize()     const { return staticStackSize; }
  unsigned getAutomaticVarsSize()   const { return automaticVarsSize; }
  unsigned getRegSpillsSize()       const { return regSpillsSize; }
  unsigned getMaxOptionalArgsSize() const { return maxOptionalArgsSize;}
  unsigned getMaxOptionalNumArgs()  const { return maxOptionalNumArgs;}
  const hash_set<const Constant*> &getConstantPoolValues() const {
    return constantsForConstPool;
  }

  //
  // Modifiers used during code generation
  //
  void            initializeFrameLayout    ();

  void            addToConstantPool        (const Constant* constVal) {
    constantsForConstPool.insert(constVal);
  }

  void markAsLeafMethod() { compiledAsLeaf = true; }

  int             computeOffsetforLocalVar (const Value*  local,
                                            unsigned& getPaddedSize,
                                            unsigned  sizeToUse = 0);
  int             allocateLocalVar         (const Value* local,
                                            unsigned sizeToUse = 0);

  int             allocateSpilledValue     (const Type* type);
  int             pushTempValue            (unsigned size);
  void            popAllTempValues         ();

  void            freezeSpillsArea         () { spillsAreaFrozen = true; }
  void            freezeAutomaticVarsArea  () { automaticVarsAreaFrozen=true; }

private:
  void incrementAutomaticVarsSize(int incr) {
    automaticVarsSize+= incr;
    staticStackSize += incr;
  }
  void incrementRegSpillsSize(int incr) {
    regSpillsSize+= incr;
    staticStackSize += incr;
  }
  void incrementTmpAreaSize(int incr) {
    currentTmpValuesSize += incr;
    if (maxTmpValuesSize < currentTmpValuesSize)
      {
        staticStackSize += currentTmpValuesSize - maxTmpValuesSize;
        maxTmpValuesSize = currentTmpValuesSize;
      }
  }
  void resetTmpAreaSize() {
    currentTmpValuesSize = 0;
  }
  int allocateOptionalArg(const Type* type);
};

} // End llvm namespace

#endif

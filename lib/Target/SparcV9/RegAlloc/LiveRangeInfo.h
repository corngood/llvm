//===-- LiveRangeInfo.h - Track all LiveRanges for a Function ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the class LiveRangeInfo which constructs and keeps
// the LiveRangeMap which contains all the live ranges used in a method.
//
// Assumptions:
//
// All variables (llvm Values) are defined before they are used. However, a
// constant may not be defined in the machine instruction stream if it can be
// used as an immediate value within a machine instruction. However, register
// allocation does not have to worry about immediate constants since they
// do not require registers.
//
// Since an llvm Value has a list of uses associated, it is sufficient to
// record only the defs in a Live Range.
//
//===----------------------------------------------------------------------===//

#ifndef LIVERANGEINFO_H
#define LIVERANGEINFO_H

#include "llvm/CodeGen/ValueSet.h"
#include "llvm/ADT/hash_map"

namespace llvm {

class V9LiveRange;
class MachineInstr;
class RegClass;
class SparcV9RegInfo;
class TargetMachine;
class Value;
class Function;
class Instruction;

typedef hash_map<const Value*, V9LiveRange*> LiveRangeMapType;

//----------------------------------------------------------------------------
// Class LiveRangeInfo
//
// Constructs and keeps the LiveRangeMap which contains all the live
// ranges used in a method. Also contain methods to coalesce live ranges.
//----------------------------------------------------------------------------

class LiveRangeInfo {
  const Function *const Meth;       // Func for which live range info is held
  LiveRangeMapType  LiveRangeMap;   // A map from Value * to V9LiveRange * to
                                    // record all live ranges in a method
                                    // created by constructLiveRanges

  const TargetMachine& TM;          // target machine description

  std::vector<RegClass *> & RegClassList;// vector containing register classess

  const SparcV9RegInfo& MRI;        // machine reg info

  std::vector<MachineInstr*> CallRetInstrList;  // a list of all call/ret instrs

  //------------ Private methods (see LiveRangeInfo.cpp for description)-------

  V9LiveRange* createNewLiveRange         (const Value* Def,
                                         bool isCC = false);

  V9LiveRange* createOrAddToLiveRange     (const Value* Def,
                                         bool isCC = false);

  void unionAndUpdateLRs                (V9LiveRange *L1,
                                         V9LiveRange *L2);

  void suggestRegs4CallRets             ();
public:

  LiveRangeInfo(const Function *F,
		const TargetMachine& tm,
		std::vector<RegClass *> & RCList);


  /// Destructor to destroy all LiveRanges in the V9LiveRange Map
  ///
  ~LiveRangeInfo();

  // Main entry point for live range construction
  //
  void constructLiveRanges();

  /// return the common live range map for this method
  ///
  inline const LiveRangeMapType *getLiveRangeMap() const
    { return &LiveRangeMap; }

  /// Method used to get the live range containing a Value.
  /// This may return NULL if no live range exists for a Value (eg, some consts)
  ///
  inline V9LiveRange *getLiveRangeForValue(const Value *Val) {
    return LiveRangeMap[Val];
  }
  inline const V9LiveRange *getLiveRangeForValue(const Value *Val) const {
    LiveRangeMapType::const_iterator I = LiveRangeMap.find(Val);
    return I->second;
  }

  /// Method for coalescing live ranges. Called only after interference info
  /// is calculated.
  ///
  void coalesceLRs();

  /// debugging method to print the live ranges
  ///
  void printLiveRanges();
};

} // End llvm namespace

#endif

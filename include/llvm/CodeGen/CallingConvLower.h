//===-- llvm/CallingConvLower.h - Calling Conventions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CCState and CCValAssign classes, used for lowering
// and implementing calling conventions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_CALLINGCONVLOWER_H
#define LLVM_CODEGEN_CALLINGCONVLOWER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {
  class MRegisterInfo;
  class TargetMachine;
  class CCState;
  class SDNode;

/// CCValAssign - Represent assignment of one arg/retval to a location.
class CCValAssign {
public:
  enum LocInfo {
    Full,   // The value fills the full location.
    SExt,   // The value is sign extended in the location.
    ZExt,   // The value is zero extended in the location.
    AExt    // The value is extended with undefined upper bits.
    // TODO: a subset of the value is in the location.
  };
private:
  /// ValNo - This is the value number begin assigned (e.g. an argument number).
  unsigned ValNo;
  
  /// Loc is either a stack offset or a register number.
  unsigned Loc;
  
  /// isMem - True if this is a memory loc, false if it is a register loc.
  bool isMem : 1;
  
  /// Information about how the value is assigned.
  LocInfo HTP : 7;
  
  /// ValVT - The type of the value being assigned.
  MVT::ValueType ValVT;

  /// LocVT - The type of the location being assigned to.
  MVT::ValueType LocVT;
public:
    
  static CCValAssign getReg(unsigned ValNo, MVT::ValueType ValVT,
                            unsigned RegNo, MVT::ValueType LocVT,
                            LocInfo HTP) {
    CCValAssign Ret;
    Ret.ValNo = ValNo;
    Ret.Loc = RegNo;
    Ret.isMem = false;
    Ret.HTP = HTP;
    Ret.ValVT = ValVT;
    Ret.LocVT = LocVT;
    return Ret;
  }
  static CCValAssign getMem(unsigned ValNo, MVT::ValueType ValVT,
                            unsigned Offset, MVT::ValueType LocVT,
                            LocInfo HTP) {
    CCValAssign Ret;
    Ret.ValNo = ValNo;
    Ret.Loc = Offset;
    Ret.isMem = true;
    Ret.HTP = HTP;
    Ret.ValVT = ValVT;
    Ret.LocVT = LocVT;
    return Ret;
  }
  
  unsigned getValNo() const { return ValNo; }
  MVT::ValueType getValVT() const { return ValVT; }

  bool isRegLoc() const { return !isMem; }
  bool isMemLoc() const { return isMem; }
  
  unsigned getLocReg() const { assert(isRegLoc()); return Loc; }
  unsigned getLocMemOffset() const { assert(isMemLoc()); return Loc; }
  MVT::ValueType getLocVT() const { return LocVT; }
  
  LocInfo getLocInfo() const { return HTP; }
};


/// CCAssignFn - This function assigns a location for Val, updating State to
/// reflect the change.
typedef bool CCAssignFn(unsigned ValNo, MVT::ValueType ValVT,
                        MVT::ValueType LocVT, CCValAssign::LocInfo LocInfo,
                        unsigned ArgFlags, CCState &State);

  
/// CCState - This class holds information needed while lowering arguments and
/// return values.  It captures which registers are already assigned and which
/// stack slots are used.  It provides accessors to allocate these values.
class CCState {
  unsigned CallingConv;
  bool IsVarArg;
  const TargetMachine &TM;
  const MRegisterInfo &MRI;
  SmallVector<CCValAssign, 16> &Locs;
  
  unsigned StackOffset;
  SmallVector<uint32_t, 16> UsedRegs;
public:
  CCState(unsigned CC, bool isVarArg, const TargetMachine &TM,
          SmallVector<CCValAssign, 16> &locs);
  
  void addLoc(const CCValAssign &V) {
    Locs.push_back(V);
  }
  
  const TargetMachine &getTarget() const { return TM; }
  unsigned getCallingConv() const { return CallingConv; }
  bool isVarArg() const { return IsVarArg; }
  
  unsigned getNextStackOffset() const { return StackOffset; }

  /// isAllocated - Return true if the specified register (or an alias) is
  /// allocated.
  bool isAllocated(unsigned Reg) const {
    return UsedRegs[Reg/32] & (1 << (Reg&31));
  }
  
  /// AnalyzeFormalArguments - Analyze an ISD::FORMAL_ARGUMENTS node,
  /// incorporating info about the formals into this state.
  void AnalyzeFormalArguments(SDNode *TheArgs, CCAssignFn Fn);
  
  /// AnalyzeReturn - Analyze the returned values of an ISD::RET node,
  /// incorporating info about the result values into this state.
  void AnalyzeReturn(SDNode *TheRet, CCAssignFn Fn);
  
  /// AnalyzeCallOperands - Analyze an ISD::CALL node, incorporating info
  /// about the passed values into this state.
  void AnalyzeCallOperands(SDNode *TheCall, CCAssignFn Fn);

  /// AnalyzeCallResult - Analyze the return values of an ISD::CALL node,
  /// incorporating info about the passed values into this state.
  void AnalyzeCallResult(SDNode *TheCall, CCAssignFn Fn);
  

  /// getFirstUnallocated - Return the first unallocated register in the set, or
  /// NumRegs if they are all allocated.
  unsigned getFirstUnallocated(const unsigned *Regs, unsigned NumRegs) const {
    for (unsigned i = 0; i != NumRegs; ++i)
      if (!isAllocated(Regs[i]))
        return i;
    return NumRegs;
  }
  
  /// AllocateReg - Attempt to allocate one register.  If it is not available,
  /// return zero.  Otherwise, return the register, marking it and any aliases
  /// as allocated.
  unsigned AllocateReg(unsigned Reg) {
    if (isAllocated(Reg)) return 0;
    MarkAllocated(Reg);
    return Reg;
  }
  
  /// AllocateReg - Attempt to allocate one of the specified registers.  If none
  /// are available, return zero.  Otherwise, return the first one available,
  /// marking it and any aliases as allocated.
  unsigned AllocateReg(const unsigned *Regs, unsigned NumRegs) {
    unsigned FirstUnalloc = getFirstUnallocated(Regs, NumRegs);
    if (FirstUnalloc == NumRegs)
      return 0;    // Didn't find the reg.
     
    // Mark the register and any aliases as allocated.
    unsigned Reg = Regs[FirstUnalloc];
    MarkAllocated(Reg);
    return Reg;
  }
  
  /// AllocateStack - Allocate a chunk of stack space with the specified size
  /// and alignment.
  unsigned AllocateStack(unsigned Size, unsigned Align) {
    assert(Align && ((Align-1) & Align) == 0); // Align is power of 2.
    StackOffset = ((StackOffset + Align-1) & ~(Align-1));
    unsigned Result = StackOffset;
    StackOffset += Size;
    return Result;
  }

  // HandleByVal - Allocate a stack slot large enough to pass an argument by
  // value. The size and alignment information of the argument is encoded in its
  // parameter attribute.
  void HandleByVal(unsigned ValNo, MVT::ValueType ValVT,
                   MVT::ValueType LocVT, CCValAssign::LocInfo LocInfo,
                   int MinSize, int MinAlign, unsigned ArgFlags);

private:
  /// MarkAllocated - Mark a register and all of its aliases as allocated.
  void MarkAllocated(unsigned Reg);
};



} // end namespace llvm

#endif

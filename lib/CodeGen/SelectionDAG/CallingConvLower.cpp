//===-- llvm/CallingConvLower.cpp - Calling Conventions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CCState class, used for lowering and implementing
// calling conventions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

CCState::CCState(unsigned CC, bool isVarArg, const TargetMachine &tm,
                 SmallVector<CCValAssign, 16> &locs)
  : CallingConv(CC), IsVarArg(isVarArg), TM(tm),
    TRI(*TM.getRegisterInfo()), Locs(locs) {
  // No stack is used.
  StackOffset = 0;
  
  UsedRegs.resize(TRI.getNumRegs());
}

// HandleByVal - Allocate a stack slot large enough to pass an argument by
// value. The size and alignment information of the argument is encoded in its
// parameter attribute.
void CCState::HandleByVal(unsigned ValNo, MVT::ValueType ValVT,
                          MVT::ValueType LocVT, CCValAssign::LocInfo LocInfo,
                          int MinSize, int MinAlign,
                          ISD::ParamFlags::ParamFlagsTy ArgFlags) {
  unsigned Align  = 1 << ((ArgFlags & ISD::ParamFlags::ByValAlign) >>
                          ISD::ParamFlags::ByValAlignOffs);
  unsigned Size   = (ArgFlags & ISD::ParamFlags::ByValSize) >>
      ISD::ParamFlags::ByValSizeOffs;
  if (MinSize > (int)Size)
    Size = MinSize;
  if (MinAlign > (int)Align)
    Align = MinAlign;
  unsigned Offset = AllocateStack(Size, Align);

  addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
}

/// MarkAllocated - Mark a register and all of its aliases as allocated.
void CCState::MarkAllocated(unsigned Reg) {
  UsedRegs[Reg/32] |= 1 << (Reg&31);
  
  if (const unsigned *RegAliases = TRI.getAliasSet(Reg))
    for (; (Reg = *RegAliases); ++RegAliases)
      UsedRegs[Reg/32] |= 1 << (Reg&31);
}

/// AnalyzeFormalArguments - Analyze an ISD::FORMAL_ARGUMENTS node,
/// incorporating info about the formals into this state.
void CCState::AnalyzeFormalArguments(SDNode *TheArgs, CCAssignFn Fn) {
  unsigned NumArgs = TheArgs->getNumValues()-1;
  
  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT::ValueType ArgVT = TheArgs->getValueType(i);
    SDOperand FlagOp = TheArgs->getOperand(3+i);
    ISD::ParamFlags::ParamFlagsTy ArgFlags = 
              cast<ConstantSDNode>(FlagOp)->getValue();
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
      cerr << "Formal argument #" << i << " has unhandled type "
           << MVT::getValueTypeString(ArgVT) << "\n";
      abort();
    }
  }
}

/// AnalyzeReturn - Analyze the returned values of an ISD::RET node,
/// incorporating info about the result values into this state.
void CCState::AnalyzeReturn(SDNode *TheRet, CCAssignFn Fn) {
  // Determine which register each value should be copied into.
  for (unsigned i = 0, e = TheRet->getNumOperands() / 2; i != e; ++i) {
    MVT::ValueType VT = TheRet->getOperand(i*2+1).getValueType();
    if (Fn(i, VT, VT, CCValAssign::Full,
           cast<ConstantSDNode>(TheRet->getOperand(i*2+2))->getValue(), *this)){
      cerr << "Return operand #" << i << " has unhandled type "
           << MVT::getValueTypeString(VT) << "\n";
      abort();
    }
  }
}


/// AnalyzeCallOperands - Analyze an ISD::CALL node, incorporating info
/// about the passed values into this state.
void CCState::AnalyzeCallOperands(SDNode *TheCall, CCAssignFn Fn) {
  unsigned NumOps = (TheCall->getNumOperands() - 5) / 2;
  for (unsigned i = 0; i != NumOps; ++i) {
    MVT::ValueType ArgVT = TheCall->getOperand(5+2*i).getValueType();
    SDOperand FlagOp = TheCall->getOperand(5+2*i+1);
    ISD::ParamFlags::ParamFlagsTy ArgFlags =
                cast<ConstantSDNode>(FlagOp)->getValue();
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
      cerr << "Call operand #" << i << " has unhandled type "
           << MVT::getValueTypeString(ArgVT) << "\n";
      abort();
    }
  }
}

/// AnalyzeCallResult - Analyze the return values of an ISD::CALL node,
/// incorporating info about the passed values into this state.
void CCState::AnalyzeCallResult(SDNode *TheCall, CCAssignFn Fn) {
  for (unsigned i = 0, e = TheCall->getNumValues() - 1; i != e; ++i) {
    MVT::ValueType VT = TheCall->getValueType(i);
    if (Fn(i, VT, VT, CCValAssign::Full, 0, *this)) {
      cerr << "Call result #" << i << " has unhandled type "
           << MVT::getValueTypeString(VT) << "\n";
      abort();
    }
  }
}

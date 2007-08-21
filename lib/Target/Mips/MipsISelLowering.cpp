//===-- MipsISelLowering.cpp - Mips DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Mips uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-lower"

#include "MipsISelLowering.h"
#include "MipsTargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <set>

using namespace llvm;

const char *MipsTargetLowering::
getTargetNodeName(unsigned Opcode) const 
{
  switch (Opcode) 
  {
    case MipsISD::JmpLink   : return "MipsISD::JmpLink";
    case MipsISD::Hi        : return "MipsISD::Hi";
    case MipsISD::Lo        : return "MipsISD::Lo";
    case MipsISD::Ret       : return "MipsISD::Ret";
    case MipsISD::Add       : return "MipsISD::Add";
    default                 : return NULL;
  }
}

MipsTargetLowering::
MipsTargetLowering(MipsTargetMachine &TM): TargetLowering(TM) 
{
  // Mips does not have i1 type, so use i32 for
  // setcc operations results (slt, sgt, ...). 
  setSetCCResultType(MVT::i32);
  setSetCCResultContents(ZeroOrOneSetCCResult);

  // Set up the register classes
  addRegisterClass(MVT::i32, Mips::CPURegsRegisterClass);

  // Custom
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::RET, MVT::Other, Custom);

  // Load extented operations for i1 types must be promoted 
  setLoadXAction(ISD::EXTLOAD,  MVT::i1,  Promote);
  setLoadXAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setLoadXAction(ISD::SEXTLOAD, MVT::i1,  Promote);

  // Store operations for i1 types must be promoted
  setStoreXAction(MVT::i1, Promote);

  // Mips does not have these NodeTypes below.
  setOperationAction(ISD::BR_JT,     MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,     MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SELECT, MVT::i32, Expand);

  // Mips not supported intrinsics.
  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);

  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ , MVT::i32, Expand);
  setOperationAction(ISD::CTLZ , MVT::i32, Expand);
  setOperationAction(ISD::ROTL , MVT::i32, Expand);
  setOperationAction(ISD::ROTR , MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::LABEL, MVT::Other, Expand);

  // Use the default for now
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setOperationAction(ISD::ADJUST_TRAMP, MVT::i32, Expand);

  setStackPointerRegisterToSaveRestore(Mips::SP);
  computeRegisterProperties();
}


SDOperand MipsTargetLowering::
LowerOperation(SDOperand Op, SelectionDAG &DAG) 
{
  switch (Op.getOpcode()) 
  {
    case ISD::CALL:             return LowerCALL(Op, DAG);
    case ISD::FORMAL_ARGUMENTS: return LowerFORMAL_ARGUMENTS(Op, DAG);
    case ISD::RET:              return LowerRET(Op, DAG);
    case ISD::GlobalAddress:    return LowerGlobalAddress(Op, DAG);
    case ISD::GlobalTLSAddress: return LowerGlobalTLSAddress(Op, DAG);
  }
  return SDOperand();
}

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

// AddLiveIn - This helper function adds the specified physical register to the
// MachineFunction as a live in value.  It also creates a corresponding
// virtual register for it.
static unsigned
AddLiveIn(MachineFunction &MF, unsigned PReg, TargetRegisterClass *RC) 
{
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getSSARegMap()->createVirtualRegister(RC);
  MF.addLiveIn(PReg, VReg);
  return VReg;
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//
SDOperand MipsTargetLowering::
LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG) 
{
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i32);

  const MVT::ValueType *VTs = DAG.getNodeValueTypes(MVT::i32, MVT::Flag);
  SDOperand Ops[] = { GA };

  SDOperand Hi = DAG.getNode(MipsISD::Hi, VTs, 2, Ops, 1);
  SDOperand Lo = DAG.getNode(MipsISD::Lo, MVT::i32, GA);

  SDOperand InFlag = Hi.getValue(1);
  return DAG.getNode(MipsISD::Add, MVT::i32, Lo, Hi, InFlag);
}

SDOperand MipsTargetLowering::
LowerGlobalTLSAddress(SDOperand Op, SelectionDAG &DAG)
{
  assert(0 && "TLS not implemented for MIPS.");
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//
//  The lower operations present on calling convention works on this order:
//      LowerCALL (virt regs --> phys regs, virt regs --> stack) 
//      LowerFORMAL_ARGUMENTS (phys --> virt regs, stack --> virt regs)
//      LowerRET (virt regs --> phys regs)
//      LowerCALL (phys regs --> virt regs)
//
//===----------------------------------------------------------------------===//

#include "MipsGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//                  CALL Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// Mips custom CALL implementation
SDOperand MipsTargetLowering::
LowerCALL(SDOperand Op, SelectionDAG &DAG)
{
  unsigned CallingConv= cast<ConstantSDNode>(Op.getOperand(1))->getValue();

  // By now, only CallingConv::C implemented
  switch (CallingConv) 
  {
    default:
      assert(0 && "Unsupported calling convention");
    case CallingConv::Fast:
    case CallingConv::C:
      return LowerCCCCallTo(Op, DAG, CallingConv);
  }
}

/// LowerCCCCallTo - functions arguments are copied from virtual
/// regs to (physical regs)/(stack frame), CALLSEQ_START and
/// CALLSEQ_END are emitted.
/// TODO: isVarArg, isTailCall, sret, GOT, linkage types.
SDOperand MipsTargetLowering::
LowerCCCCallTo(SDOperand Op, SelectionDAG &DAG, unsigned CC) 
{
  MachineFunction &MF = DAG.getMachineFunction();
  unsigned StackReg   = MF.getTarget().getRegisterInfo()->getFrameRegister(MF);

  SDOperand Chain  = Op.getOperand(0);
  SDOperand Callee = Op.getOperand(4);
  bool isVarArg    = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;

  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);

  // To meet ABI, Mips must always allocate 16 bytes on
  // the stack (even if less than 4 are used as arguments)
  int VTsize = MVT::getSizeInBits(MVT::i32)/8;
  MFI->CreateFixedObject(VTsize, -(VTsize*3));

  CCInfo.AnalyzeCallOperands(Op.Val, CC_Mips);
  
  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();
  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes, 
                                 getPointerTy()));

  SmallVector<std::pair<unsigned, SDOperand>, 8> RegsToPass;
  SmallVector<SDOperand, 8> MemOpChains;

  SDOperand StackPtr;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // Arguments start after the 5 first operands of ISD::CALL
    SDOperand Arg = Op.getOperand(5+2*VA.getValNo());
    
    // Promote the value if needed.
    switch (VA.getLocInfo()) {
      default: assert(0 && "Unknown loc info!");
      case CCValAssign::Full: break;
      case CCValAssign::SExt:
        Arg = DAG.getNode(ISD::SIGN_EXTEND, VA.getLocVT(), Arg);
        break;
      case CCValAssign::ZExt:
        Arg = DAG.getNode(ISD::ZERO_EXTEND, VA.getLocVT(), Arg);
        break;
      case CCValAssign::AExt:
        Arg = DAG.getNode(ISD::ANY_EXTEND, VA.getLocVT(), Arg);
        break;
    }
    
    // Arguments that can be passed on register, 
    // must be kept at RegsToPass vector
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {

      assert(VA.isMemLoc());

      if (StackPtr.Val == 0)
        StackPtr = DAG.getRegister(StackReg, getPointerTy());
     
      // Create the frame index object for this incoming parameter
      // This guarantees that when allocating Local Area our room
      // will not be overwritten.
      int FI = MFI->CreateFixedObject(MVT::getSizeInBits(VA.getValVT())/8,
                                      -(16 + VA.getLocMemOffset()) );

      SDOperand PtrOff = DAG.getFrameIndex(FI,getPointerTy());

      // emit ISD::STORE whichs stores the 
      // parameter value to a stack Location
      MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
    }
  }

  // Transform all store nodes into one single node because
  // all store nodes are independent of each other.
  if (!MemOpChains.empty())     
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, 
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token 
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emited instructions must be
  // stuck together.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, 
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct 
  // call is) turn it into a TargetGlobalAddress node so that legalize 
  // doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  } else 
  if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());

  // MipsJmpLink = #chain, #target_address, #opt_in_flags...
  //             = Chain, Callee, Reg#1, Reg#2, ...  
  //
  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  SmallVector<SDOperand, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are 
  // known live into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.Val)
    Ops.push_back(InFlag);

  Chain  = DAG.getNode(MipsISD::JmpLink, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  Ops.clear();
  Ops.push_back(Chain);
  Ops.push_back(DAG.getConstant(NumBytes, getPointerTy()));
  Ops.push_back(InFlag);
  Chain  = DAG.getNode(ISD::CALLSEQ_END, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return SDOperand(LowerCallResult(Chain, InFlag, Op.Val, CC, DAG), Op.ResNo);
}

/// LowerCallResult - Lower the result values of an ISD::CALL into the
/// appropriate copies out of appropriate physical registers.  This assumes that
/// Chain/InFlag are the input chain/flag to use, and that TheCall is the call
/// being lowered. Returns a SDNode with the same number of values as the 
/// ISD::CALL.
SDNode *MipsTargetLowering::
LowerCallResult(SDOperand Chain, SDOperand InFlag, SDNode *TheCall, 
        unsigned CallingConv, SelectionDAG &DAG) {
  
  bool isVarArg = cast<ConstantSDNode>(TheCall->getOperand(2))->getValue() != 0;

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallingConv, isVarArg, getTargetMachine(), RVLocs);

  CCInfo.AnalyzeCallResult(TheCall, RetCC_Mips);
  SmallVector<SDOperand, 8> ResultVals;

  // Returns void
  //if (!RVLocs.size())
  //  return Chain.Val;

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    Chain = DAG.getCopyFromReg(Chain, RVLocs[i].getLocReg(),
                                 RVLocs[i].getValVT(), InFlag).getValue(1);
    InFlag = Chain.getValue(2);
    ResultVals.push_back(Chain.getValue(0));
  }
  
  // Merge everything together with a MERGE_VALUES node.
  ResultVals.push_back(Chain);
  return DAG.getNode(ISD::MERGE_VALUES, TheCall->getVTList(),
                       &ResultVals[0], ResultVals.size()).Val;
}

//===----------------------------------------------------------------------===//
//             FORMAL_ARGUMENTS Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// Mips custom FORMAL_ARGUMENTS implementation
SDOperand MipsTargetLowering::
LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG) 
{
  unsigned CC = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  switch(CC) 
  {
    default:
      assert(0 && "Unsupported calling convention");
    case CallingConv::C:
      return LowerCCCArguments(Op, DAG);
  }
}

/// LowerCCCArguments - transform physical registers into
/// virtual registers and generate load operations for
/// arguments places on the stack.
/// TODO: isVarArg, sret
SDOperand MipsTargetLowering::
LowerCCCArguments(SDOperand Op, SelectionDAG &DAG) 
{
  SDOperand Root        = Op.getOperand(0);
  MachineFunction &MF   = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  unsigned CC   = DAG.getMachineFunction().getFunction()->getCallingConv();

  unsigned StackReg = MF.getTarget().getRegisterInfo()->getFrameRegister(MF);

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);

  CCInfo.AnalyzeFormalArguments(Op.Val, CC_Mips);
  SmallVector<SDOperand, 8> ArgValues;
  SDOperand StackPtr;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {

    CCValAssign &VA = ArgLocs[i];

    // Arguments stored on registers
    if (VA.isRegLoc()) {
      MVT::ValueType RegVT = VA.getLocVT();
      TargetRegisterClass *RC;
            
      if (RegVT == MVT::i32)
        RC = Mips::CPURegsRegisterClass;
      else
        assert(0 && "support only Mips::CPURegsRegisterClass");
      

      // Transform the arguments stored on 
      // physical registers into virtual ones
      unsigned Reg = AddLiveIn(DAG.getMachineFunction(), VA.getLocReg(), RC);
      SDOperand ArgValue = DAG.getCopyFromReg(Root, Reg, RegVT);
      
      // If this is an 8 or 16-bit value, it is really passed promoted 
      // to 32 bits.  Insert an assert[sz]ext to capture this, then 
      // truncate to the right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      
      if (VA.getLocInfo() != CCValAssign::Full)
        ArgValue = DAG.getNode(ISD::TRUNCATE, VA.getValVT(), ArgValue);

      ArgValues.push_back(ArgValue);

      // To meet ABI, when VARARGS are passed on registers, the registers
      // containt must be written to the their always reserved home location 
      // on the stack.
      if (isVarArg) {

        if (StackPtr.Val == 0)
          StackPtr = DAG.getRegister(StackReg, getPointerTy());
     
        // Create the frame index object for this incoming parameter
        // The first 16 bytes are reserved.
        int FI = MFI->CreateFixedObject(MVT::getSizeInBits(VA.getValVT())/8,
                                        i*4);
        SDOperand PtrOff = DAG.getFrameIndex(FI, getPointerTy());
      
        // emit ISD::STORE whichs stores the 
        // parameter value to a stack Location
        ArgValues.push_back(DAG.getStore(Root, ArgValue, PtrOff, NULL, 0));
      }

    } else {
      // sanity check
      assert(VA.isMemLoc());
      
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(MVT::getSizeInBits(VA.getValVT())/8,
                                      (16 + VA.getLocMemOffset()));

      // Create load nodes to retrieve arguments from the stack
      SDOperand FIN = DAG.getFrameIndex(FI, getPointerTy());
      ArgValues.push_back(DAG.getLoad(VA.getValVT(), Root, FIN, NULL, 0));
    }
  }
  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getNode(ISD::MERGE_VALUES, Op.Val->getVTList(),
                     &ArgValues[0], ArgValues.size()).getValue(Op.ResNo);
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

SDOperand MipsTargetLowering::
LowerRET(SDOperand Op, SelectionDAG &DAG)
{
  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC   = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CC, isVarArg, getTargetMachine(), RVLocs);

  // Analize return values of ISD::RET
  CCInfo.AnalyzeReturn(Op.Val, RetCC_Mips);

  // If this is the first return lowered for this function, add 
  // the regs to the liveout set for the function.
  if (DAG.getMachineFunction().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().addLiveOut(RVLocs[i].getLocReg());
  }

  // The chain is always operand #0
  SDOperand Chain = Op.getOperand(0);
  SDOperand Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // ISD::RET => ret chain, (regnum1,val1), ...
    // So i*2+1 index only the regnums
    Chain = DAG.getCopyToReg(Chain, VA.getLocReg(), 
                                 Op.getOperand(i*2+1), Flag);

    // guarantee that all emitted copies are
    // stuck together, avoiding something bad
    Flag = Chain.getValue(1);
  }

  // Return on Mips is always a "jr $ra"
  if (Flag.Val)
    return DAG.getNode(MipsISD::Ret, MVT::Other, 
                           Chain, DAG.getRegister(Mips::RA, MVT::i32), Flag);
  else // Return Void
    return DAG.getNode(MipsISD::Ret, MVT::Other, 
                           Chain, DAG.getRegister(Mips::RA, MVT::i32));
}

//===----------------------------------------------------------------------===//
//                           Mips Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
MipsTargetLowering::ConstraintType MipsTargetLowering::
getConstraintType(const std::string &Constraint) const 
{
  if (Constraint.size() == 1) {
    // Mips specific constrainy 
    // GCC config/mips/constraints.md
    //
    // 'd' : An address register. Equivalent to r 
    //       unless generating MIPS16 code. 
    // 'y' : Equivalent to r; retained for 
    //       backwards compatibility. 
    //
    switch (Constraint[0]) {
      default : break;
      case 'd':     
      case 'y': 
        return C_RegisterClass;
        break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass*> MipsTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint,
                             MVT::ValueType VT) const 
{
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, Mips::CPURegsRegisterClass);
      break;
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

std::vector<unsigned> MipsTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const 
{
  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  switch (Constraint[0]) {         
    default : break;
    case 'r':
    // GCC Mips Constraint Letters
    case 'd':     
    case 'y': 
      return make_vector<unsigned>(Mips::V0, Mips::V1, Mips::A0, 
                                   Mips::A1, Mips::A2, Mips::A3, 
                                   Mips::T0, Mips::T1, Mips::T2, 
                                   Mips::T3, Mips::T4, Mips::T5, 
                                   Mips::T6, Mips::T7, Mips::S0, 
                                   Mips::S1, Mips::S2, Mips::S3, 
                                   Mips::S4, Mips::S5, Mips::S6, 
                                   Mips::S7, Mips::T8, Mips::T9, 0);
      break;
  }
  return std::vector<unsigned>();
}

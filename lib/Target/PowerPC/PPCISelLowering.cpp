//===-- PPC32ISelLowering.cpp - PPC32 DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPC32ISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "PPC32ISelLowering.h"
#include "PPC32TargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Function.h"
using namespace llvm;

PPC32TargetLowering::PPC32TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
    
  // Fold away setcc operations if possible.
  setSetCCIsExpensive();
  
  // Set up the register classes.
  addRegisterClass(MVT::i32, PPC32::GPRCRegisterClass);
  addRegisterClass(MVT::f32, PPC32::FPRCRegisterClass);
  addRegisterClass(MVT::f64, PPC32::FPRCRegisterClass);
  
  // PowerPC has no intrinsics for these particular operations
  setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
  setOperationAction(ISD::MEMSET, MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY, MVT::Other, Expand);
  
  // PowerPC has an i16 but no i8 (or i1) SEXTLOAD
  setOperationAction(ISD::SEXTLOAD, MVT::i1, Expand);
  setOperationAction(ISD::SEXTLOAD, MVT::i8, Expand);
  
  // PowerPC has no SREM/UREM instructions
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  
  // We don't support sin/cos/sqrt/fmod
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::SREM , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::SREM , MVT::f32, Expand);
  
  // If we're enabling GP optimizations, use hardware square root
  if (!TM.getSubtarget<PPCSubtarget>().isGigaProcessor()) {
    setOperationAction(ISD::FSQRT, MVT::f64, Expand);
    setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  }
  
  // PowerPC does not have CTPOP or CTTZ
  setOperationAction(ISD::CTPOP, MVT::i32  , Expand);
  setOperationAction(ISD::CTTZ , MVT::i32  , Expand);
  
  // PowerPC does not have Select
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);

  // PowerPC does not have BRCOND* which requires SetCC
  setOperationAction(ISD::BRCOND,       MVT::Other, Expand);
  setOperationAction(ISD::BRCONDTWOWAY, MVT::Other, Expand);
  
  // PowerPC does not have FP_TO_UINT
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  
  // PowerPC does not have [U|S]INT_TO_FP
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);

  setSetCCResultContents(ZeroOrOneSetCCResult);
  addLegalFPImmediate(+0.0); // Necessary for FSEL
  addLegalFPImmediate(-0.0); //
  
  computeRegisterProperties();
}

std::vector<SDOperand>
PPC32TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  //
  // add beautiful description of PPC stack frame format, or at least some docs
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock& BB = MF.front();
  std::vector<SDOperand> ArgValues;
  
  // Due to the rather complicated nature of the PowerPC ABI, rather than a
  // fixed size array of physical args, for the sake of simplicity let the STL
  // handle tracking them for us.
  std::vector<unsigned> argVR, argPR, argOp;
  unsigned ArgOffset = 24;
  unsigned GPR_remaining = 8;
  unsigned FPR_remaining = 13;
  unsigned GPR_idx = 0, FPR_idx = 0;
  static const unsigned GPR[] = {
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const unsigned FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };
  
  // Add DAG nodes to load the arguments...  On entry to a function on PPC,
  // the arguments start at offset 24, although they are likely to be passed
  // in registers.
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    SDOperand newroot, argt;
    unsigned ObjSize;
    bool needsLoad = false;
    bool ArgLive = !I->use_empty();
    MVT::ValueType ObjectVT = getValueType(I->getType());
    
    switch (ObjectVT) {
      default: assert(0 && "Unhandled argument type!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
        ObjSize = 4;
        if (!ArgLive) break;
          if (GPR_remaining > 0) {
            MF.addLiveIn(GPR[GPR_idx]);
            argt = newroot = DAG.getCopyFromReg(DAG.getRoot(),
                                                GPR[GPR_idx], MVT::i32);
            if (ObjectVT != MVT::i32)
              argt = DAG.getNode(ISD::TRUNCATE, ObjectVT, newroot);
          } else {
            needsLoad = true;
          }
            break;
      case MVT::i64: ObjSize = 8;
        if (!ArgLive) break;
          if (GPR_remaining > 0) {
            SDOperand argHi, argLo;
            MF.addLiveIn(GPR[GPR_idx]);
            argHi = DAG.getCopyFromReg(DAG.getRoot(), GPR[GPR_idx], MVT::i32);
            // If we have two or more remaining argument registers, then both halves
            // of the i64 can be sourced from there.  Otherwise, the lower half will
            // have to come off the stack.  This can happen when an i64 is preceded
            // by 28 bytes of arguments.
            if (GPR_remaining > 1) {
              MF.addLiveIn(GPR[GPR_idx+1]);
              argLo = DAG.getCopyFromReg(argHi, GPR[GPR_idx+1], MVT::i32);
            } else {
              int FI = MFI->CreateFixedObject(4, ArgOffset+4);
              SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
              argLo = DAG.getLoad(MVT::i32, DAG.getEntryNode(), FIN,
                                  DAG.getSrcValue(NULL));
            }
            // Build the outgoing arg thingy
            argt = DAG.getNode(ISD::BUILD_PAIR, MVT::i64, argLo, argHi);
            newroot = argLo;
          } else {
            needsLoad = true;
          }
            break;
      case MVT::f32:
      case MVT::f64:
        ObjSize = (ObjectVT == MVT::f64) ? 8 : 4;
        if (!ArgLive) break;
          if (FPR_remaining > 0) {
            MF.addLiveIn(FPR[FPR_idx]);
            argt = newroot = DAG.getCopyFromReg(DAG.getRoot(), 
                                                FPR[FPR_idx], ObjectVT);
            --FPR_remaining;
            ++FPR_idx;
          } else {
            needsLoad = true;
          }
            break;
    }
    
    // We need to load the argument to a virtual register if we determined above
    // that we ran out of physical registers of the appropriate type
    if (needsLoad) {
      unsigned SubregOffset = 0;
      if (ObjectVT == MVT::i8 || ObjectVT == MVT::i1) SubregOffset = 3;
      if (ObjectVT == MVT::i16) SubregOffset = 2;
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
      FIN = DAG.getNode(ISD::ADD, MVT::i32, FIN,
                        DAG.getConstant(SubregOffset, MVT::i32));
      argt = newroot = DAG.getLoad(ObjectVT, DAG.getEntryNode(), FIN,
                                   DAG.getSrcValue(NULL));
    }
    
    // Every 4 bytes of argument space consumes one of the GPRs available for
    // argument passing.
    if (GPR_remaining > 0) {
      unsigned delta = (GPR_remaining > 1 && ObjSize == 8) ? 2 : 1;
      GPR_remaining -= delta;
      GPR_idx += delta;
    }
    ArgOffset += ObjSize;
    if (newroot.Val)
      DAG.setRoot(newroot.getValue(1));
    
    ArgValues.push_back(argt);
  }
  
  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (F.isVarArg()) {
    VarArgsFrameIndex = MFI->CreateFixedObject(4, ArgOffset);
    SDOperand FIN = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
    // If this function is vararg, store any remaining integer argument regs
    // to their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    std::vector<SDOperand> MemOps;
    for (; GPR_remaining > 0; --GPR_remaining, ++GPR_idx) {
      MF.addLiveIn(GPR[GPR_idx]);
      SDOperand Val = DAG.getCopyFromReg(DAG.getRoot(), GPR[GPR_idx], MVT::i32);
      SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1),
                                    Val, FIN, DAG.getSrcValue(NULL));
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDOperand PtrOff = DAG.getConstant(4, getPointerTy());
      FIN = DAG.getNode(ISD::ADD, MVT::i32, FIN, PtrOff);
    }
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps));
  }
  
  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
    default: assert(0 && "Unknown type!");
    case MVT::isVoid: break;
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      MF.addLiveOut(PPC::R3);
      break;
    case MVT::i64:
      MF.addLiveOut(PPC::R3);
      MF.addLiveOut(PPC::R4);
      break;
    case MVT::f32:
    case MVT::f64:
      MF.addLiveOut(PPC::F1);
      break;
  }
  
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
PPC32TargetLowering::LowerCallTo(SDOperand Chain,
                                 const Type *RetTy, bool isVarArg,
                                 unsigned CallingConv, bool isTailCall,
                                 SDOperand Callee, ArgListTy &Args,
                                 SelectionDAG &DAG) {
  // args_to_use will accumulate outgoing args for the ISD::CALL case in
  // SelectExpr to use to put the arguments in the appropriate registers.
  std::vector<SDOperand> args_to_use;
  
  // Count how many bytes are to be pushed on the stack, including the linkage
  // area, and parameter passing area.
  unsigned NumBytes = 24;
  
  if (Args.empty()) {
    Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));
  } else {
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
      switch (getValueType(Args[i].second)) {
        default: assert(0 && "Unknown value type!");
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::f32:
          NumBytes += 4;
          break;
        case MVT::i64:
        case MVT::f64:
          NumBytes += 8;
          break;
      }
        
        // Just to be safe, we'll always reserve the full 24 bytes of linkage area
        // plus 32 bytes of argument space in case any called code gets funky on us.
        // (Required by ABI to support var arg)
        if (NumBytes < 56) NumBytes = 56;
    
    // Adjust the stack pointer for the new arguments...
    // These operations are automatically eliminated by the prolog/epilog pass
    Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));
    
    // Set up a copy of the stack pointer for use loading and storing any
    // arguments that may not fit in the registers available for argument
    // passing.
    SDOperand StackPtr = DAG.getCopyFromReg(DAG.getEntryNode(),
                                            PPC::R1, MVT::i32);
    
    // Figure out which arguments are going to go in registers, and which in
    // memory.  Also, if this is a vararg function, floating point operations
    // must be stored to our stack, and loaded into integer regs as well, if
    // any integer regs are available for argument passing.
    unsigned ArgOffset = 24;
    unsigned GPR_remaining = 8;
    unsigned FPR_remaining = 13;
    
    std::vector<SDOperand> MemOps;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      // PtrOff will be used to store the current argument to the stack if a
      // register cannot be found for it.
      SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      MVT::ValueType ArgVT = getValueType(Args[i].second);
      
      switch (ArgVT) {
        default: assert(0 && "Unexpected ValueType for argument!");
        case MVT::i1:
        case MVT::i8:
        case MVT::i16:
          // Promote the integer to 32 bits.  If the input type is signed use a
          // sign extend, otherwise use a zero extend.
          if (Args[i].second->isSigned())
            Args[i].first =DAG.getNode(ISD::SIGN_EXTEND, MVT::i32, Args[i].first);
          else
            Args[i].first =DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Args[i].first);
          // FALL THROUGH
        case MVT::i32:
          if (GPR_remaining > 0) {
            args_to_use.push_back(Args[i].first);
            --GPR_remaining;
          } else {
            MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                         Args[i].first, PtrOff,
                                         DAG.getSrcValue(NULL)));
          }
          ArgOffset += 4;
          break;
        case MVT::i64:
          // If we have one free GPR left, we can place the upper half of the i64
          // in it, and store the other half to the stack.  If we have two or more
          // free GPRs, then we can pass both halves of the i64 in registers.
          if (GPR_remaining > 0) {
            SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                       Args[i].first, DAG.getConstant(1, MVT::i32));
            SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32,
                                       Args[i].first, DAG.getConstant(0, MVT::i32));
            args_to_use.push_back(Hi);
            --GPR_remaining;
            if (GPR_remaining > 0) {
              args_to_use.push_back(Lo);
              --GPR_remaining;
            } else {
              SDOperand ConstFour = DAG.getConstant(4, getPointerTy());
              PtrOff = DAG.getNode(ISD::ADD, MVT::i32, PtrOff, ConstFour);
              MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                           Lo, PtrOff, DAG.getSrcValue(NULL)));
            }
          } else {
            MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                         Args[i].first, PtrOff,
                                         DAG.getSrcValue(NULL)));
          }
          ArgOffset += 8;
          break;
        case MVT::f32:
        case MVT::f64:
          if (FPR_remaining > 0) {
            args_to_use.push_back(Args[i].first);
            --FPR_remaining;
            if (isVarArg) {
              SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                            Args[i].first, PtrOff,
                                            DAG.getSrcValue(NULL));
              MemOps.push_back(Store);
              // Float varargs are always shadowed in available integer registers
              if (GPR_remaining > 0) {
                SDOperand Load = DAG.getLoad(MVT::i32, Store, PtrOff,
                                             DAG.getSrcValue(NULL));
                MemOps.push_back(Load);
                args_to_use.push_back(Load);
                --GPR_remaining;
              }
              if (GPR_remaining > 0 && MVT::f64 == ArgVT) {
                SDOperand ConstFour = DAG.getConstant(4, getPointerTy());
                PtrOff = DAG.getNode(ISD::ADD, MVT::i32, PtrOff, ConstFour);
                SDOperand Load = DAG.getLoad(MVT::i32, Store, PtrOff,
                                             DAG.getSrcValue(NULL));
                MemOps.push_back(Load);
                args_to_use.push_back(Load);
                --GPR_remaining;
              }
            } else {
              // If we have any FPRs remaining, we may also have GPRs remaining.
              // Args passed in FPRs consume either 1 (f32) or 2 (f64) available
              // GPRs.
              if (GPR_remaining > 0) {
                args_to_use.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
                --GPR_remaining;
              }
              if (GPR_remaining > 0 && MVT::f64 == ArgVT) {
                args_to_use.push_back(DAG.getNode(ISD::UNDEF, MVT::i32));
                --GPR_remaining;
              }
            }
          } else {
            MemOps.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                         Args[i].first, PtrOff,
                                         DAG.getSrcValue(NULL)));
          }
          ArgOffset += (ArgVT == MVT::f32) ? 4 : 8;
          break;
      }
    }
    if (!MemOps.empty())
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, MemOps);
  }
  
  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(RetTyVT);
  RetVals.push_back(MVT::Other);
  
  SDOperand TheCall = SDOperand(DAG.getCall(RetVals,
                                            Chain, Callee, args_to_use), 0);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  return std::make_pair(TheCall, Chain);
}

SDOperand PPC32TargetLowering::LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                            Value *VAListV, SelectionDAG &DAG) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i32);
  return DAG.getNode(ISD::STORE, MVT::Other, Chain, FR, VAListP,
                     DAG.getSrcValue(VAListV));
}

std::pair<SDOperand,SDOperand>
PPC32TargetLowering::LowerVAArg(SDOperand Chain,
                                SDOperand VAListP, Value *VAListV,
                                const Type *ArgTy, SelectionDAG &DAG) {
  MVT::ValueType ArgVT = getValueType(ArgTy);
  
  SDOperand VAList =
    DAG.getLoad(MVT::i32, Chain, VAListP, DAG.getSrcValue(VAListV));
  SDOperand Result = DAG.getLoad(ArgVT, Chain, VAList, DAG.getSrcValue(NULL));
  unsigned Amt;
  if (ArgVT == MVT::i32 || ArgVT == MVT::f32)
    Amt = 4;
  else {
    assert((ArgVT == MVT::i64 || ArgVT == MVT::f64) &&
           "Other types should have been promoted for varargs!");
    Amt = 8;
  }
  VAList = DAG.getNode(ISD::ADD, VAList.getValueType(), VAList,
                       DAG.getConstant(Amt, VAList.getValueType()));
  Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                      VAList, VAListP, DAG.getSrcValue(VAListV));
  return std::make_pair(Result, Chain);
}


std::pair<SDOperand, SDOperand> PPC32TargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  assert(0 && "LowerFrameReturnAddress unimplemented");
  abort();
}

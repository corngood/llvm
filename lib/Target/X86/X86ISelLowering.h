//===-- X86ISelLowering.h - X86 DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that X86 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef X86ISELLOWERING_H
#define X86ISELLOWERING_H

#include "X86Subtarget.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {
  namespace X86ISD {
    // X86 Specific DAG Nodes
    enum NodeType {
      // Start the numbering where the builtin ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+X86::INSTRUCTION_LIST_END,

      /// SHLD, SHRD - Double shift instructions. These correspond to
      /// X86::SHLDxx and X86::SHRDxx instructions.
      SHLD,
      SHRD,

      /// FAND - Bitwise logical AND of floating point values. This corresponds
      /// to X86::ANDPS or X86::ANDPD.
      FAND,

      /// FXOR - Bitwise logical XOR of floating point values. This corresponds
      /// to X86::XORPS or X86::XORPD.
      FXOR,

      /// FILD, FILD_FLAG - This instruction implements SINT_TO_FP with the
      /// integer source in memory and FP reg result.  This corresponds to the
      /// X86::FILD*m instructions. It has three inputs (token chain, address,
      /// and source type) and two outputs (FP value and token chain). FILD_FLAG
      /// also produces a flag).
      FILD,
      FILD_FLAG,

      /// FP_TO_INT*_IN_MEM - This instruction implements FP_TO_SINT with the
      /// integer destination in memory and a FP reg source.  This corresponds
      /// to the X86::FIST*m instructions and the rounding mode change stuff. It
      /// has two inputs (token chain and address) and two outputs (int value and
      /// token chain).
      FP_TO_INT16_IN_MEM,
      FP_TO_INT32_IN_MEM,
      FP_TO_INT64_IN_MEM,

      /// FLD - This instruction implements an extending load to FP stack slots.
      /// This corresponds to the X86::FLD32m / X86::FLD64m. It takes a chain
      /// operand, ptr to load from, and a ValueType node indicating the type
      /// to load to.
      FLD,

      /// FST - This instruction implements a truncating store to FP stack
      /// slots. This corresponds to the X86::FST32m / X86::FST64m. It takes a
      /// chain operand, value to store, address, and a ValueType to store it
      /// as.
      FST,

      /// FP_SET_RESULT - This corresponds to FpGETRESULT pseudo instrcuction
      /// which copies from ST(0) to the destination. It takes a chain and writes
      /// a RFP result and a chain.
      FP_GET_RESULT,

      /// FP_SET_RESULT - This corresponds to FpSETRESULT pseudo instrcuction
      /// which copies the source operand to ST(0). It takes a chain and writes
      /// a chain and a flag.
      FP_SET_RESULT,

      /// CALL/TAILCALL - These operations represent an abstract X86 call
      /// instruction, which includes a bunch of information.  In particular the
      /// operands of these node are:
      ///
      ///     #0 - The incoming token chain
      ///     #1 - The callee
      ///     #2 - The number of arg bytes the caller pushes on the stack.
      ///     #3 - The number of arg bytes the callee pops off the stack.
      ///     #4 - The value to pass in AL/AX/EAX (optional)
      ///     #5 - The value to pass in DL/DX/EDX (optional)
      ///
      /// The result values of these nodes are:
      ///
      ///     #0 - The outgoing token chain
      ///     #1 - The first register result value (optional)
      ///     #2 - The second register result value (optional)
      ///
      /// The CALL vs TAILCALL distinction boils down to whether the callee is
      /// known not to modify the caller's stack frame, as is standard with
      /// LLVM.
      CALL,
      TAILCALL,
      
      /// RDTSC_DAG - This operation implements the lowering for 
      /// readcyclecounter
      RDTSC_DAG,

      /// X86 compare and logical compare instructions.
      CMP, TEST,

      /// X86 SetCC. Operand 1 is condition code, and operand 2 is the flag
      /// operand produced by a CMP instruction.
      SETCC,

      /// X86 conditional moves. Operand 1 and operand 2 are the two values
      /// to select from (operand 1 is a R/W operand). Operand 3 is the condition
      /// code, and operand 4 is the flag operand produced by a CMP or TEST
      /// instruction. It also writes a flag result.
      CMOV,

      /// X86 conditional branches. Operand 1 is the chain operand, operand 2
      /// is the block to branch if condition is true, operand 3 is the
      /// condition code, and operand 4 is the flag operand produced by a CMP
      /// or TEST instruction.
      BRCOND,

      /// Return with a flag operand. Operand 1 is the chain operand, operand
      /// 2 is the number of bytes of stack to pop.
      RET_FLAG,

      /// REP_STOS - Repeat fill, corresponds to X86::REP_STOSx.
      REP_STOS,

      /// REP_MOVS - Repeat move, corresponds to X86::REP_MOVSx.
      REP_MOVS,

      /// LOAD_PACK Load a 128-bit packed float / double value. It has the same
      /// operands as a normal load.
      LOAD_PACK,

      /// GlobalBaseReg - On Darwin, this node represents the result of the popl
      /// at function entry, used for PIC code.
      GlobalBaseReg,

      /// TCPWrapper - A wrapper node for TargetConstantPool,
      /// TargetExternalSymbol, and TargetGlobalAddress.
      Wrapper,

      /// S2VEC - X86 version of SCALAR_TO_VECTOR. The destination base does not
      /// have to match the operand type.
      S2VEC,

      /// ZEXT_S2VEC - SCALAR_TO_VECTOR with zero extension. The destination base
      /// does not have to match the operand type.
      ZEXT_S2VEC,
    };

    // X86 specific condition code. These correspond to X86_*_COND in
    // X86InstrInfo.td. They must be kept in synch.
    enum CondCode {
      COND_A  = 0,
      COND_AE = 1,
      COND_B  = 2,
      COND_BE = 3,
      COND_E  = 4,
      COND_G  = 5,
      COND_GE = 6,
      COND_L  = 7,
      COND_LE = 8,
      COND_NE = 9,
      COND_NO = 10,
      COND_NP = 11,
      COND_NS = 12,
      COND_O  = 13,
      COND_P  = 14,
      COND_S  = 15,
      COND_INVALID
    };
  }

 /// Define some predicates that are used for node matching.
 namespace X86 {
   /// isPSHUFDMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to PSHUFD.
   bool isPSHUFDMask(SDNode *N);

   /// isSHUFPMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to SHUFP*.
   bool isSHUFPMask(SDNode *N);

   /// isMOVLHPSMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVHLPS.
   bool isMOVLHPSMask(SDNode *N);

   /// isMOVHLPSMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to MOVHLPS.
   bool isMOVHLPSMask(SDNode *N);

   /// isUNPCKLMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to UNPCKL.
   bool isUNPCKLMask(SDNode *N);

   /// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a shuffle of elements that is suitable for input to UNPCKH.
   bool isUNPCKHMask(SDNode *N);

   /// isSplatMask - Return true if the specified VECTOR_SHUFFLE operand
   /// specifies a splat of a single element.
   bool isSplatMask(SDNode *N);

   /// getShuffleSHUFImmediate - Return the appropriate immediate to shuffle
   /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUF* and SHUFP*
   /// instructions.
   unsigned getShuffleSHUFImmediate(SDNode *N);
 }

  //===----------------------------------------------------------------------===//
  //  X86TargetLowering - X86 Implementation of the TargetLowering interface
  class X86TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int ReturnAddrIndex;              // FrameIndex for return slot.
    int BytesToPopOnReturn;           // Number of arg bytes ret should pop.
    int BytesCallerReserves;          // Number of arg bytes caller makes.
  public:
    X86TargetLowering(TargetMachine &TM);

    // Return the number of bytes that a function should pop when it returns (in
    // addition to the space used by the return address).
    //
    unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }

    // Return the number of bytes that the caller reserves for arguments passed
    // to this function.
    unsigned getBytesCallerReserves() const { return BytesCallerReserves; }
 
    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg, unsigned CC,
                bool isTailCall, SDOperand Callee, ArgListTy &Args,
                SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);

    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// computeMaskedBitsForTargetNode - Determine which of the bits specified 
    /// in Mask are known to be either zero or one and return them in the 
    /// KnownZero/KnownOne bitsets.
    virtual void computeMaskedBitsForTargetNode(const SDOperand Op,
                                                uint64_t Mask,
                                                uint64_t &KnownZero, 
                                                uint64_t &KnownOne,
                                                unsigned Depth = 0) const;
    
    SDOperand getReturnAddressFrameIndex(SelectionDAG &DAG);

    std::vector<unsigned> 
      getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                        MVT::ValueType VT) const;

    /// isLegalAddressImmediate - Return true if the integer value or
    /// GlobalValue can be used as the offset of the target addressing mode.
    virtual bool isLegalAddressImmediate(int64_t V) const;
    virtual bool isLegalAddressImmediate(GlobalValue *GV) const;

    /// isShuffleMaskLegal - Targets can use this to indicate that they only
    /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
    /// By default, if a target supports the VECTOR_SHUFFLE node, all mask values
    /// are assumed to be legal.
    virtual bool isShuffleMaskLegal(SDOperand Mask, MVT::ValueType VT) const;
  private:
    // C Calling Convention implementation.
    std::vector<SDOperand> LowerCCCArguments(Function &F, SelectionDAG &DAG);
    std::pair<SDOperand, SDOperand>
    LowerCCCCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                   bool isTailCall,
                   SDOperand Callee, ArgListTy &Args, SelectionDAG &DAG);

    // Fast Calling Convention implementation.
    std::vector<SDOperand> LowerFastCCArguments(Function &F, SelectionDAG &DAG);
    std::pair<SDOperand, SDOperand>
    LowerFastCCCallTo(SDOperand Chain, const Type *RetTy, bool isTailCall,
                      SDOperand Callee, ArgListTy &Args, SelectionDAG &DAG);

    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;

    /// X86ScalarSSE - Select between SSE2 or x87 floating point ops.
    bool X86ScalarSSE;
  };
}

#endif    // X86ISELLOWERING_H

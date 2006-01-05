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

#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {
  // X86 Specific DAG Nodes
  namespace X86ISD {
    enum NodeType {
      // Start the numbering where the builtin ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+X86::INSTRUCTION_LIST_END,

      /// FILD64m - This instruction implements SINT_TO_FP with a
      /// 64-bit source in memory and a FP reg result.  This corresponds to
      /// the X86::FILD64m instruction.  It has two inputs (token chain and
      /// address) and two outputs (FP value and token chain).
      FILD64m,

      /// FP_TO_INT*_IN_MEM - This instruction implements FP_TO_SINT with the
      /// integer destination in memory and a FP reg source.  This corresponds
      /// to the X86::FIST*m instructions and the rounding mode change stuff. It
      /// has two inputs (token chain and address) and two outputs (FP value and
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
      /// instruction.
      CMOV,

      /// X86 conditional branches. Operand 1 is the chain operand, operand 2
      /// is the block to branch if condition is true, operand 3 is the
      /// condition code, and operand 4 is the flag operand produced by a CMP
      /// or TEST instruction.
      BRCOND,

      /// Return with a flag operand. Operand 1 is the number of bytes of stack
      /// to pop, operand 2 is the chain and operand 3 is a flag operand.
      RET_FLAG,
    };
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

    virtual SDOperand LowerReturnTo(SDOperand Chain, SDOperand Op,
                                    SelectionDAG &DAG);
    
    virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                   Value *VAListV, SelectionDAG &DAG);
    virtual std::pair<SDOperand,SDOperand>
    LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
               const Type *ArgTy, SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// isMaskedValueZeroForTargetNode - Return true if 'Op & Mask' is known to
    /// be zero. Op is expected to be a target specific node. Used by DAG
    /// combiner.
    virtual bool isMaskedValueZeroForTargetNode(const SDOperand &Op,
                                                uint64_t Mask) const;

    SDOperand getReturnAddressFrameIndex(SelectionDAG &DAG);

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
  };
}

#endif    // X86ISELLOWERING_H

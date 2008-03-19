//===-- X86ISelLowering.h - X86 DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "X86RegisterInfo.h"
#include "X86MachineFunctionInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {
  namespace X86ISD {
    // X86 Specific DAG Nodes
    enum NodeType {
      // Start the numbering where the builtin ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+X86::INSTRUCTION_LIST_END,

      /// BSF - Bit scan forward.
      /// BSR - Bit scan reverse.
      BSF,
      BSR,

      /// SHLD, SHRD - Double shift instructions. These correspond to
      /// X86::SHLDxx and X86::SHRDxx instructions.
      SHLD,
      SHRD,

      /// FAND - Bitwise logical AND of floating point values. This corresponds
      /// to X86::ANDPS or X86::ANDPD.
      FAND,

      /// FOR - Bitwise logical OR of floating point values. This corresponds
      /// to X86::ORPS or X86::ORPD.
      FOR,

      /// FXOR - Bitwise logical XOR of floating point values. This corresponds
      /// to X86::XORPS or X86::XORPD.
      FXOR,

      /// FSRL - Bitwise logical right shift of floating point values. These
      /// corresponds to X86::PSRLDQ.
      FSRL,

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
      /// has two inputs (token chain and address) and two outputs (int value
      /// and token chain).
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

      /// FP_GET_ST0_ST1 - Same as FP_GET_ST0 except it copies two values
      /// ST(0) and ST(1).
      FP_GET_ST0_ST1,

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
      CMP, COMI, UCOMI,

      /// X86 SetCC. Operand 1 is condition code, and operand 2 is the flag
      /// operand produced by a CMP instruction.
      SETCC,

      /// X86 conditional moves. Operand 1 and operand 2 are the two values
      /// to select from (operand 1 is a R/W operand). Operand 3 is the
      /// condition code, and operand 4 is the flag operand produced by a CMP
      /// or TEST instruction. It also writes a flag result.
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

      /// GlobalBaseReg - On Darwin, this node represents the result of the popl
      /// at function entry, used for PIC code.
      GlobalBaseReg,

      /// Wrapper - A wrapper node for TargetConstantPool,
      /// TargetExternalSymbol, and TargetGlobalAddress.
      Wrapper,

      /// WrapperRIP - Special wrapper used under X86-64 PIC mode for RIP
      /// relative displacements.
      WrapperRIP,

      /// PEXTRB - Extract an 8-bit value from a vector and zero extend it to
      /// i32, corresponds to X86::PEXTRB.
      PEXTRB,

      /// PEXTRW - Extract a 16-bit value from a vector and zero extend it to
      /// i32, corresponds to X86::PEXTRW.
      PEXTRW,

      /// INSERTPS - Insert any element of a 4 x float vector into any element
      /// of a destination 4 x floatvector.
      INSERTPS,

      /// PINSRB - Insert the lower 8-bits of a 32-bit value to a vector,
      /// corresponds to X86::PINSRB.
      PINSRB,

      /// PINSRW - Insert the lower 16-bits of a 32-bit value to a vector,
      /// corresponds to X86::PINSRW.
      PINSRW,

      /// FMAX, FMIN - Floating point max and min.
      ///
      FMAX, FMIN,

      /// FRSQRT, FRCP - Floating point reciprocal-sqrt and reciprocal
      /// approximation.  Note that these typically require refinement
      /// in order to obtain suitable precision.
      FRSQRT, FRCP,

      // Thread Local Storage
      TLSADDR, THREAD_POINTER,

      // Exception Handling helpers
      EH_RETURN,
      
      /// TC_RETURN - Tail call return.
      ///   operand #0 chain
      ///   operand #1 callee (register or absolute)
      ///   operand #2 stack adjustment
      ///   operand #3 optional in flag
      TC_RETURN,

      // compare and swap
      LCMPXCHG_DAG,
      LCMPXCHG8_DAG,

      // Store FP control world into i16 memory
      FNSTCW16m
    };
  }

  /// Define some predicates that are used for node matching.
  namespace X86 {
    /// isPSHUFDMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to PSHUFD.
    bool isPSHUFDMask(SDNode *N);

    /// isPSHUFHWMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to PSHUFD.
    bool isPSHUFHWMask(SDNode *N);

    /// isPSHUFLWMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to PSHUFD.
    bool isPSHUFLWMask(SDNode *N);

    /// isSHUFPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to SHUFP*.
    bool isSHUFPMask(SDNode *N);

    /// isMOVHLPSMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVHLPS.
    bool isMOVHLPSMask(SDNode *N);

    /// isMOVHLPS_v_undef_Mask - Special case of isMOVHLPSMask for canonical form
    /// of vector_shuffle v, v, <2, 3, 2, 3>, i.e. vector_shuffle v, undef,
    /// <2, 3, 2, 3>
    bool isMOVHLPS_v_undef_Mask(SDNode *N);

    /// isMOVLPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVLP{S|D}.
    bool isMOVLPMask(SDNode *N);

    /// isMOVHPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVHP{S|D}
    /// as well as MOVLHPS.
    bool isMOVHPMask(SDNode *N);

    /// isUNPCKLMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to UNPCKL.
    bool isUNPCKLMask(SDNode *N, bool V2IsSplat = false);

    /// isUNPCKHMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to UNPCKH.
    bool isUNPCKHMask(SDNode *N, bool V2IsSplat = false);

    /// isUNPCKL_v_undef_Mask - Special case of isUNPCKLMask for canonical form
    /// of vector_shuffle v, v, <0, 4, 1, 5>, i.e. vector_shuffle v, undef,
    /// <0, 0, 1, 1>
    bool isUNPCKL_v_undef_Mask(SDNode *N);

    /// isUNPCKH_v_undef_Mask - Special case of isUNPCKHMask for canonical form
    /// of vector_shuffle v, v, <2, 6, 3, 7>, i.e. vector_shuffle v, undef,
    /// <2, 2, 3, 3>
    bool isUNPCKH_v_undef_Mask(SDNode *N);

    /// isMOVLMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVSS,
    /// MOVSD, and MOVD, i.e. setting the lowest element.
    bool isMOVLMask(SDNode *N);

    /// isMOVSHDUPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVSHDUP.
    bool isMOVSHDUPMask(SDNode *N);

    /// isMOVSLDUPMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a shuffle of elements that is suitable for input to MOVSLDUP.
    bool isMOVSLDUPMask(SDNode *N);

    /// isSplatMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a splat of a single element.
    bool isSplatMask(SDNode *N);

    /// isSplatLoMask - Return true if the specified VECTOR_SHUFFLE operand
    /// specifies a splat of zero element.
    bool isSplatLoMask(SDNode *N);

    /// getShuffleSHUFImmediate - Return the appropriate immediate to shuffle
    /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUF* and SHUFP*
    /// instructions.
    unsigned getShuffleSHUFImmediate(SDNode *N);

    /// getShufflePSHUFHWImmediate - Return the appropriate immediate to shuffle
    /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUFHW
    /// instructions.
    unsigned getShufflePSHUFHWImmediate(SDNode *N);

    /// getShufflePSHUFKWImmediate - Return the appropriate immediate to shuffle
    /// the specified isShuffleMask VECTOR_SHUFFLE mask with PSHUFLW
    /// instructions.
    unsigned getShufflePSHUFLWImmediate(SDNode *N);
  }

  namespace X86 {
   /// X86_64SRet - These represent different ways to implement x86_64 struct
   /// returns call results.
   enum X86_64SRet {
     InMemory,    // Really is sret, returns in memory.
     InGPR64,     // Returns in a pair of 64-bit integer registers.
     InSSE,       // Returns in a pair of SSE registers.
     InX87        // Returns in a pair of f80 X87 registers.
   };
  }

  //===--------------------------------------------------------------------===//
  //  X86TargetLowering - X86 Implementation of the TargetLowering interface
  class X86TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int RegSaveFrameIndex;            // X86-64 vararg func register save area.
    unsigned VarArgsGPOffset;         // X86-64 vararg func int reg offset.
    unsigned VarArgsFPOffset;         // X86-64 vararg func fp reg offset.
    int BytesToPopOnReturn;           // Number of arg bytes ret should pop.
    int BytesCallerReserves;          // Number of arg bytes caller makes.

  public:
    explicit X86TargetLowering(TargetMachine &TM);

    /// getPICJumpTableRelocaBase - Returns relocation base for the given PIC
    /// jumptable.
    SDOperand getPICJumpTableRelocBase(SDOperand Table,
                                       SelectionDAG &DAG) const;

    // Return the number of bytes that a function should pop when it returns (in
    // addition to the space used by the return address).
    //
    unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }

    // Return the number of bytes that the caller reserves for arguments passed
    // to this function.
    unsigned getBytesCallerReserves() const { return BytesCallerReserves; }
 
    /// getStackPtrReg - Return the stack pointer register we are using: either
    /// ESP or RSP.
    unsigned getStackPtrReg() const { return X86StackPtr; }

    /// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
    /// function arguments in the caller parameter area. For X86, aggregates
    /// that contains are placed at 16-byte boundaries while the rest are at
    /// 4-byte boundaries.
    virtual unsigned getByValTypeAlignment(const Type *Ty) const;
    
    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

    /// ExpandOperation - Custom lower the specified operation, splitting the
    /// value into two pieces.
    ///
    virtual SDNode *ExpandOperationResult(SDNode *N, SelectionDAG &DAG);

    
    virtual SDOperand PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual MachineBasicBlock *EmitInstrWithCustomInserter(MachineInstr *MI,
                                                        MachineBasicBlock *MBB);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - Return the ISD::SETCC ValueType
    virtual MVT::ValueType getSetCCResultType(const SDOperand &) const;

    /// computeMaskedBitsForTargetNode - Determine which of the bits specified 
    /// in Mask are known to be either zero or one and return them in the 
    /// KnownZero/KnownOne bitsets.
    virtual void computeMaskedBitsForTargetNode(const SDOperand Op,
                                                const APInt &Mask,
                                                APInt &KnownZero, 
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth = 0) const;
    
    SDOperand getReturnAddressFrameIndex(SelectionDAG &DAG);

    ConstraintType getConstraintType(const std::string &Constraint) const;
     
    std::vector<unsigned> 
      getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                        MVT::ValueType VT) const;

    virtual void lowerXConstraint(MVT::ValueType ConstraintVT, 
                                  std::string&) const;

    /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
    /// vector.  If it is invalid, don't add anything to Ops.
    virtual void LowerAsmOperandForConstraint(SDOperand Op,
                                              char ConstraintLetter,
                                              std::vector<SDOperand> &Ops,
                                              SelectionDAG &DAG);
    
    /// getRegForInlineAsmConstraint - Given a physical register constraint
    /// (e.g. {edx}), return the register number and the register class for the
    /// register.  This should only be used for C_Register constraints.  On
    /// error, this returns a register number of 0.
    std::pair<unsigned, const TargetRegisterClass*> 
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   MVT::ValueType VT) const;
    
    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    virtual bool isLegalAddressingMode(const AddrMode &AM, const Type *Ty)const;

    /// isTruncateFree - Return true if it's free to truncate a value of
    /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
    /// register EAX to i16 by referencing its sub-register AX.
    virtual bool isTruncateFree(const Type *Ty1, const Type *Ty2) const;
    virtual bool isTruncateFree(MVT::ValueType VT1, MVT::ValueType VT2) const;
  
    /// isShuffleMaskLegal - Targets can use this to indicate that they only
    /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
    /// By default, if a target supports the VECTOR_SHUFFLE node, all mask
    /// values are assumed to be legal.
    virtual bool isShuffleMaskLegal(SDOperand Mask, MVT::ValueType VT) const;

    /// isVectorClearMaskLegal - Similar to isShuffleMaskLegal. This is
    /// used by Targets can use this to indicate if there is a suitable
    /// VECTOR_SHUFFLE that can be used to replace a VAND with a constant
    /// pool entry.
    virtual bool isVectorClearMaskLegal(std::vector<SDOperand> &BVOps,
                                        MVT::ValueType EVT,
                                        SelectionDAG &DAG) const;

    /// ShouldShrinkFPConstant - If true, then instruction selection should
    /// seek to shrink the FP constant of the specified type to a smaller type
    /// in order to save space and / or reduce runtime.
    virtual bool ShouldShrinkFPConstant(MVT::ValueType VT) const {
      // Don't shrink FP constpool if SSE2 is available since cvtss2sd is more
      // expensive than a straight movsd. On the other hand, it's important to
      // shrink long double fp constant since fldt is very slow.
      return !X86ScalarSSEf64 || VT == MVT::f80;
    }
    
    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization. Target which want to do tail call
    /// optimization should implement this function.
    virtual bool IsEligibleForTailCallOptimization(SDOperand Call, 
                                                   SDOperand Ret, 
                                                   SelectionDAG &DAG) const;

    virtual const TargetSubtarget* getSubtarget() {
      return static_cast<const TargetSubtarget*>(Subtarget);
    }

    /// isScalarFPTypeInSSEReg - Return true if the specified scalar FP type is
    /// computed in an SSE register, not on the X87 floating point stack.
    bool isScalarFPTypeInSSEReg(MVT::ValueType VT) const {
      return (VT == MVT::f64 && X86ScalarSSEf64) || // f64 is when SSE2
      (VT == MVT::f32 && X86ScalarSSEf32);   // f32 is when SSE1
    }
    
  private:
    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;
    const TargetRegisterInfo *RegInfo;

    /// X86StackPtr - X86 physical register used as stack ptr.
    unsigned X86StackPtr;
   
    /// X86ScalarSSEf32, X86ScalarSSEf64 - Select between SSE or x87 
    /// floating point ops.
    /// When SSE is available, use it for f32 operations.
    /// When SSE2 is available, use it for f64 operations.
    bool X86ScalarSSEf32;
    bool X86ScalarSSEf64;

    X86::X86_64SRet ClassifyX86_64SRetCallReturn(const Function *Fn);

    void X86_64AnalyzeSRetCallOperands(SDNode*, CCAssignFn*, CCState&);

    SDNode *LowerCallResult(SDOperand Chain, SDOperand InFlag, SDNode*TheCall,
                            unsigned CallingConv, SelectionDAG &DAG);

    SDNode *LowerCallResultToTwo64BitRegs(SDOperand Chain, SDOperand InFlag,
                                          SDNode *TheCall, unsigned Reg1,
                                          unsigned Reg2, MVT::ValueType VT,
                                          SelectionDAG &DAG);        

    SDNode *LowerCallResultToTwoX87Regs(SDOperand Chain, SDOperand InFlag,
                                        SDNode *TheCall, SelectionDAG &DAG);        

    SDOperand LowerMemArgument(SDOperand Op, SelectionDAG &DAG,
                               const CCValAssign &VA,  MachineFrameInfo *MFI,
                               unsigned CC, SDOperand Root, unsigned i);

    SDOperand LowerMemOpCallTo(SDOperand Op, SelectionDAG &DAG,
                               const SDOperand &StackPtr,
                               const CCValAssign &VA, SDOperand Chain,
                               SDOperand Arg);

    // Call lowering helpers.
    bool IsCalleePop(SDOperand Op);
    bool CallRequiresGOTPtrInReg(bool Is64Bit, bool IsTailCall);
    bool CallRequiresFnAddressInReg(bool Is64Bit, bool IsTailCall);
    CCAssignFn *CCAssignFnForNode(SDOperand Op) const;
    NameDecorationStyle NameDecorationForFORMAL_ARGUMENTS(SDOperand Op);
    unsigned GetAlignedArgumentStackSize(unsigned StackSize, SelectionDAG &DAG);

    std::pair<SDOperand,SDOperand> FP_TO_SINTHelper(SDOperand Op, 
                                                    SelectionDAG &DAG);
    
    SDOperand LowerBUILD_VECTOR(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerVECTOR_SHUFFLE(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerEXTRACT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerEXTRACT_VECTOR_ELT_SSE4(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerINSERT_VECTOR_ELT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerINSERT_VECTOR_ELT_SSE4(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSCALAR_TO_VECTOR(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerConstantPool(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalTLSAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerExternalSymbol(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerShift(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFP_TO_SINT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFABS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFNEG(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFCOPYSIGN(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSETCC(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSELECT(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerBRCOND(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerMEMSET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerMEMCPYInline(SDOperand Dest, SDOperand Source,
                                SDOperand Chain, unsigned Size, unsigned Align,
                                SelectionDAG &DAG);
    SDOperand LowerJumpTable(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerDYNAMIC_STACKALLOC(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerVASTART(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerVACOPY(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerINTRINSIC_WO_CHAIN(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerRETURNADDR(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFRAMEADDR(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFRAME_TO_ARGS_OFFSET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerEH_RETURN(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerTRAMPOLINE(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFLT_ROUNDS_(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCTLZ(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCTTZ(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerLCS(SDOperand Op, SelectionDAG &DAG);
    SDNode *ExpandFP_TO_SINT(SDNode *N, SelectionDAG &DAG);
    SDNode *ExpandREADCYCLECOUNTER(SDNode *N, SelectionDAG &DAG);
    SDNode *ExpandATOMIC_LCS(SDNode *N, SelectionDAG &DAG);
  };
}

#endif    // X86ISELLOWERING_H

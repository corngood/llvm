//===-- PIC16ISelLowering.h - PIC16 DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PIC16 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16ISELLOWERING_H
#define PIC16ISELLOWERING_H

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
#include "PIC16.h"
#include "PIC16Subtarget.h"

namespace llvm {
  namespace PIC16ISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+PIC16::INSTRUCTION_LIST_END,

      // used for encapsulating the expanded nodes into one node.
      Package,

      // Get the Higher 16 bits from a 32-bit immediate
      Hi, 

      // Get the Lower 16 bits from a 32-bit immediate
      Lo,

      Cmp,	// PIC16 Generic Comparison instruction.
      Branch,	// PIC16 Generic Branch Instruction.	
      BTFSS,	// PIC16 BitTest Instruction (Skip if set).
      BTFSC,	// PIC16 BitTest Instruction (Skip if clear).

      // PIC16 comparison to be converted to either XOR or SUB
      // Following instructions cater to those convertions.
      XORCC,	
      SUBCC,	

      // Get the Global Address wrapped into a wrapper that also captures 
      // the bank or page.
      Wrapper,
      SetBank,
      SetPage
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class PIC16TargetLowering : public TargetLowering 
  {
  public:
    typedef std::map<SDNode *, SDNode *> NodeMap_t;

    explicit PIC16TargetLowering(PIC16TargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

    SDOperand LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFrameIndex(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerBR_CC(SDOperand Op, SelectionDAG &DAG);

    SDOperand RemoveHiLo(SDNode *, SelectionDAG &DAG, 
			 DAGCombinerInfo &DCI) const;
    SDOperand LowerADDSUB(SDNode *, SelectionDAG &DAG, 
			  DAGCombinerInfo &DCI) const;
    SDOperand LowerLOAD(SDNode *, SelectionDAG &DAG, 
                        DAGCombinerInfo &DCI) const;

    /// getTargetNodeName - This method returns the name of a target specific 
    //  DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;
    virtual SDOperand PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    // utility function.
    const SDOperand *findLoadi8(const SDOperand &Src, SelectionDAG &DAG) const;
  };
} // namespace llvm

#endif // PIC16ISELLOWERING_H

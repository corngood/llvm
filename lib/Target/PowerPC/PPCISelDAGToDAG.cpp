//===-- PPCISelDAGToDAG.cpp - PPC --pattern matching inst selector --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for PowerPC,
// converting from a legalized dag to a PPC dag.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCTargetMachine.h"
#include "PPCISelLowering.h"
#include "PPCHazardRecognizers.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <iostream>
#include <set>
using namespace llvm;

namespace {
  Statistic<> FrameOff("ppc-codegen", "Number of frame idx offsets collapsed");
    
  //===--------------------------------------------------------------------===//
  /// PPCDAGToDAGISel - PPC specific code to select PPC machine
  /// instructions for SelectionDAG operations.
  ///
  class PPCDAGToDAGISel : public SelectionDAGISel {
    PPCTargetMachine &TM;
    PPCTargetLowering PPCLowering;
    unsigned GlobalBaseReg;
  public:
    PPCDAGToDAGISel(PPCTargetMachine &tm)
      : SelectionDAGISel(PPCLowering), TM(tm),
        PPCLowering(*TM.getTargetLowering()) {}
    
    virtual bool runOnFunction(Function &Fn) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseReg = 0;
      SelectionDAGISel::runOnFunction(Fn);
      
      InsertVRSaveCode(Fn);
      return true;
    }
   
    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDOperand getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }

    /// getGlobalBaseReg - insert code into the entry mbb to materialize the PIC
    /// base register.  Return the virtual register that holds this value.
    SDOperand getGlobalBaseReg();
    
    // Select - Convert the specified operand from a target-independent to a
    // target-specific node if it hasn't already been changed.
    void Select(SDOperand &Result, SDOperand Op);
    
    SDNode *SelectBitfieldInsert(SDNode *N);

    /// SelectCC - Select a comparison of the specified values with the
    /// specified condition code, returning the CR# of the expression.
    SDOperand SelectCC(SDOperand LHS, SDOperand RHS, ISD::CondCode CC);

    /// SelectAddrImm - Returns true if the address N can be represented by
    /// a base register plus a signed 16-bit displacement [r+imm].
    bool SelectAddrImm(SDOperand N, SDOperand &Disp, SDOperand &Base);
      
    /// SelectAddrIdx - Given the specified addressed, check to see if it can be
    /// represented as an indexed [r+r] operation.  Returns false if it can
    /// be represented by [r+imm], which are preferred.
    bool SelectAddrIdx(SDOperand N, SDOperand &Base, SDOperand &Index);
    
    /// SelectAddrIdxOnly - Given the specified addressed, force it to be
    /// represented as an indexed [r+r] operation.
    bool SelectAddrIdxOnly(SDOperand N, SDOperand &Base, SDOperand &Index);

    /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
    /// inline asm expressions.
    virtual bool SelectInlineAsmMemoryOperand(const SDOperand &Op,
                                              char ConstraintCode,
                                              std::vector<SDOperand> &OutOps,
                                              SelectionDAG &DAG) {
      SDOperand Op0, Op1;
      switch (ConstraintCode) {
      default: return true;
      case 'm':   // memory
        if (!SelectAddrIdx(Op, Op0, Op1))
          SelectAddrImm(Op, Op0, Op1);
        break;
      case 'o':   // offsetable
        if (!SelectAddrImm(Op, Op0, Op1)) {
          Select(Op0, Op);     // r+0.
          Op1 = getI32Imm(0);
        }
        break;
      case 'v':   // not offsetable
        SelectAddrIdxOnly(Op, Op0, Op1);
        break;
      }
      
      OutOps.push_back(Op0);
      OutOps.push_back(Op1);
      return false;
    }
    
    SDOperand BuildSDIVSequence(SDNode *N);
    SDOperand BuildUDIVSequence(SDNode *N);
    
    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
    
    void InsertVRSaveCode(Function &Fn);

    virtual const char *getPassName() const {
      return "PowerPC DAG->DAG Pattern Instruction Selection";
    } 
    
    /// CreateTargetHazardRecognizer - Return the hazard recognizer to use for this
    /// target when scheduling the DAG.
    virtual HazardRecognizer *CreateTargetHazardRecognizer() {
      // Should use subtarget info to pick the right hazard recognizer.  For
      // now, always return a PPC970 recognizer.
      const TargetInstrInfo *II = PPCLowering.getTargetMachine().getInstrInfo();
      assert(II && "No InstrInfo?");
      return new PPCHazardRecognizer970(*II); 
    }

// Include the pieces autogenerated from the target description.
#include "PPCGenDAGISel.inc"
    
private:
    SDOperand SelectSETCC(SDOperand Op);
    SDOperand SelectCALL(SDOperand Op);
  };
}

/// InstructionSelectBasicBlock - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void PPCDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());
  
  // The selection process is inherently a bottom-up recursive process (users
  // select their uses before themselves).  Given infinite stack space, we
  // could just start selecting on the root and traverse the whole graph.  In
  // practice however, this causes us to run out of stack space on large basic
  // blocks.  To avoid this problem, select the entry node, then all its uses,
  // iteratively instead of recursively.
  std::vector<SDOperand> Worklist;
  Worklist.push_back(DAG.getEntryNode());
  
  // Note that we can do this in the PPC target (scanning forward across token
  // chain edges) because no nodes ever get folded across these edges.  On a
  // target like X86 which supports load/modify/store operations, this would
  // have to be more careful.
  while (!Worklist.empty()) {
    SDOperand Node = Worklist.back();
    Worklist.pop_back();
    
    // Chose from the least deep of the top two nodes.
    if (!Worklist.empty() &&
        Worklist.back().Val->getNodeDepth() < Node.Val->getNodeDepth())
      std::swap(Worklist.back(), Node);
    
    if ((Node.Val->getOpcode() >= ISD::BUILTIN_OP_END &&
         Node.Val->getOpcode() < PPCISD::FIRST_NUMBER) ||
        CodeGenMap.count(Node)) continue;
    
    for (SDNode::use_iterator UI = Node.Val->use_begin(),
         E = Node.Val->use_end(); UI != E; ++UI) {
      // Scan the values.  If this use has a value that is a token chain, add it
      // to the worklist.
      SDNode *User = *UI;
      for (unsigned i = 0, e = User->getNumValues(); i != e; ++i)
        if (User->getValueType(i) == MVT::Other) {
          Worklist.push_back(SDOperand(User, i));
          break; 
        }
    }

    // Finally, legalize this node.
    SDOperand Dummy;
    Select(Dummy, Node);
  }
    
  // Select target instructions for the DAG.
  DAG.setRoot(SelectRoot(DAG.getRoot()));
  CodeGenMap.clear();
  DAG.RemoveDeadNodes();
  
  // Emit machine code to BB.
  ScheduleAndEmitDAG(DAG);
}

/// InsertVRSaveCode - Once the entire function has been instruction selected,
/// all virtual registers are created and all machine instructions are built,
/// check to see if we need to save/restore VRSAVE.  If so, do it.
void PPCDAGToDAGISel::InsertVRSaveCode(Function &F) {
  // Check to see if this function uses vector registers, which means we have to
  // save and restore the VRSAVE register and update it with the regs we use.  
  //
  // In this case, there will be virtual registers of vector type type created
  // by the scheduler.  Detect them now.
  MachineFunction &Fn = MachineFunction::get(&F);
  SSARegMap *RegMap = Fn.getSSARegMap();
  bool HasVectorVReg = false;
  for (unsigned i = MRegisterInfo::FirstVirtualRegister, 
       e = RegMap->getLastVirtReg()+1; i != e; ++i)
    if (RegMap->getRegClass(i) == &PPC::VRRCRegClass) {
      HasVectorVReg = true;
      break;
    }
  if (!HasVectorVReg) return;  // nothing to do.
      
  // If we have a vector register, we want to emit code into the entry and exit
  // blocks to save and restore the VRSAVE register.  We do this here (instead
  // of marking all vector instructions as clobbering VRSAVE) for two reasons:
  //
  // 1. This (trivially) reduces the load on the register allocator, by not
  //    having to represent the live range of the VRSAVE register.
  // 2. This (more significantly) allows us to create a temporary virtual
  //    register to hold the saved VRSAVE value, allowing this temporary to be
  //    register allocated, instead of forcing it to be spilled to the stack.

  // Create two vregs - one to hold the VRSAVE register that is live-in to the
  // function and one for the value after having bits or'd into it.
  unsigned InVRSAVE = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
  unsigned UpdatedVRSAVE = RegMap->createVirtualRegister(&PPC::GPRCRegClass);
  
  MachineBasicBlock &EntryBB = *Fn.begin();
  // Emit the following code into the entry block:
  // InVRSAVE = MFVRSAVE
  // UpdatedVRSAVE = UPDATE_VRSAVE InVRSAVE
  // MTVRSAVE UpdatedVRSAVE
  MachineBasicBlock::iterator IP = EntryBB.begin();  // Insert Point
  BuildMI(EntryBB, IP, PPC::MFVRSAVE, 0, InVRSAVE);
  BuildMI(EntryBB, IP, PPC::UPDATE_VRSAVE, 1, UpdatedVRSAVE).addReg(InVRSAVE);
  BuildMI(EntryBB, IP, PPC::MTVRSAVE, 1).addReg(UpdatedVRSAVE);
  
  // Find all return blocks, outputting a restore in each epilog.
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    if (!BB->empty() && TII.isReturn(BB->back().getOpcode())) {
      IP = BB->end(); --IP;
      
      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = IP;
      while (I2 != BB->begin() && TII.isTerminatorInstr((--I2)->getOpcode()))
        IP = I2;
      
      // Emit: MTVRSAVE InVRSave
      BuildMI(*BB, IP, PPC::MTVRSAVE, 1).addReg(InVRSAVE);
    }        
  }
}


/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
SDOperand PPCDAGToDAGISel::getGlobalBaseReg() {
  if (!GlobalBaseReg) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    SSARegMap *RegMap = BB->getParent()->getSSARegMap();
    // FIXME: when we get to LP64, we will need to create the appropriate
    // type of register here.
    GlobalBaseReg = RegMap->createVirtualRegister(PPC::GPRCRegisterClass);
    BuildMI(FirstMBB, MBBI, PPC::MovePCtoLR, 0, PPC::LR);
    BuildMI(FirstMBB, MBBI, PPC::MFLR, 1, GlobalBaseReg);
  }
  return CurDAG->getRegister(GlobalBaseReg, MVT::i32);
}


// isIntImmediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isIntImmediate(SDNode *N, unsigned& Imm) {
  if (N->getOpcode() == ISD::Constant) {
    Imm = cast<ConstantSDNode>(N)->getValue();
    return true;
  }
  return false;
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME) {
  if (isShiftedMask_32(Val)) {
    // look for the first non-zero bit
    MB = CountLeadingZeros_32(Val);
    // look for the first zero bit after the run of ones
    ME = CountLeadingZeros_32((Val - 1) ^ Val);
    return true;
  } else {
    Val = ~Val; // invert mask
    if (isShiftedMask_32(Val)) {
      // effectively look for the first zero bit
      ME = CountLeadingZeros_32(Val) - 1;
      // effectively look for the first one bit after the run of zeros
      MB = CountLeadingZeros_32((Val - 1) ^ Val) + 1;
      return true;
    }
  }
  // no run present
  return false;
}

// isRotateAndMask - Returns true if Mask and Shift can be folded into a rotate
// and mask opcode and mask operation.
static bool isRotateAndMask(SDNode *N, unsigned Mask, bool IsShiftMask,
                            unsigned &SH, unsigned &MB, unsigned &ME) {
  // Don't even go down this path for i64, since different logic will be
  // necessary for rldicl/rldicr/rldimi.
  if (N->getValueType(0) != MVT::i32)
    return false;

  unsigned Shift  = 32;
  unsigned Indeterminant = ~0;  // bit mask marking indeterminant results
  unsigned Opcode = N->getOpcode();
  if (N->getNumOperands() != 2 ||
      !isIntImmediate(N->getOperand(1).Val, Shift) || (Shift > 31))
    return false;
  
  if (Opcode == ISD::SHL) {
    // apply shift left to mask if it comes first
    if (IsShiftMask) Mask = Mask << Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu << Shift);
  } else if (Opcode == ISD::SRL) { 
    // apply shift right to mask if it comes first
    if (IsShiftMask) Mask = Mask >> Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu >> Shift);
    // adjust for the left rotate
    Shift = 32 - Shift;
  } else {
    return false;
  }
  
  // if the mask doesn't intersect any Indeterminant bits
  if (Mask && !(Mask & Indeterminant)) {
    SH = Shift;
    // make sure the mask is still a mask (wrap arounds may not be)
    return isRunOfOnes(Mask, MB, ME);
  }
  return false;
}

// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(SDNode *N, unsigned Opc, unsigned& Imm) {
  return N->getOpcode() == Opc && isIntImmediate(N->getOperand(1).Val, Imm);
}

// isIntImmediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isIntImmediate(SDOperand N, unsigned& Imm) {
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N)) {
    Imm = (unsigned)CN->getSignExtended();
    return true;
  }
  return false;
}

/// SelectBitfieldInsert - turn an or of two masked values into
/// the rotate left word immediate then mask insert (rlwimi) instruction.
/// Returns true on success, false if the caller still needs to select OR.
///
/// Patterns matched:
/// 1. or shl, and   5. or and, and
/// 2. or and, shl   6. or shl, shr
/// 3. or shr, and   7. or shr, shl
/// 4. or and, shr
SDNode *PPCDAGToDAGISel::SelectBitfieldInsert(SDNode *N) {
  bool IsRotate = false;
  unsigned TgtMask = 0xFFFFFFFF, InsMask = 0xFFFFFFFF, SH = 0;
  unsigned Value;
  
  SDOperand Op0 = N->getOperand(0);
  SDOperand Op1 = N->getOperand(1);
  
  unsigned Op0Opc = Op0.getOpcode();
  unsigned Op1Opc = Op1.getOpcode();
  
  // Verify that we have the correct opcodes
  if (ISD::SHL != Op0Opc && ISD::SRL != Op0Opc && ISD::AND != Op0Opc)
    return false;
  if (ISD::SHL != Op1Opc && ISD::SRL != Op1Opc && ISD::AND != Op1Opc)
    return false;
  
  // Generate Mask value for Target
  if (isIntImmediate(Op0.getOperand(1), Value)) {
    switch(Op0Opc) {
    case ISD::SHL: TgtMask <<= Value; break;
    case ISD::SRL: TgtMask >>= Value; break;
    case ISD::AND: TgtMask &= Value; break;
    }
  } else {
    return 0;
  }
  
  // Generate Mask value for Insert
  if (!isIntImmediate(Op1.getOperand(1), Value))
    return 0;
  
  switch(Op1Opc) {
  case ISD::SHL:
    SH = Value;
    InsMask <<= SH;
    if (Op0Opc == ISD::SRL) IsRotate = true;
    break;
  case ISD::SRL:
    SH = Value;
    InsMask >>= SH;
    SH = 32-SH;
    if (Op0Opc == ISD::SHL) IsRotate = true;
    break;
  case ISD::AND:
    InsMask &= Value;
    break;
  }
  
  // If both of the inputs are ANDs and one of them has a logical shift by
  // constant as its input, make that AND the inserted value so that we can
  // combine the shift into the rotate part of the rlwimi instruction
  bool IsAndWithShiftOp = false;
  if (Op0Opc == ISD::AND && Op1Opc == ISD::AND) {
    if (Op1.getOperand(0).getOpcode() == ISD::SHL ||
        Op1.getOperand(0).getOpcode() == ISD::SRL) {
      if (isIntImmediate(Op1.getOperand(0).getOperand(1), Value)) {
        SH = Op1.getOperand(0).getOpcode() == ISD::SHL ? Value : 32 - Value;
        IsAndWithShiftOp = true;
      }
    } else if (Op0.getOperand(0).getOpcode() == ISD::SHL ||
               Op0.getOperand(0).getOpcode() == ISD::SRL) {
      if (isIntImmediate(Op0.getOperand(0).getOperand(1), Value)) {
        std::swap(Op0, Op1);
        std::swap(TgtMask, InsMask);
        SH = Op1.getOperand(0).getOpcode() == ISD::SHL ? Value : 32 - Value;
        IsAndWithShiftOp = true;
      }
    }
  }
  
  // Verify that the Target mask and Insert mask together form a full word mask
  // and that the Insert mask is a run of set bits (which implies both are runs
  // of set bits).  Given that, Select the arguments and generate the rlwimi
  // instruction.
  unsigned MB, ME;
  if (((TgtMask & InsMask) == 0) && isRunOfOnes(InsMask, MB, ME)) {
    bool fullMask = (TgtMask ^ InsMask) == 0xFFFFFFFF;
    bool Op0IsAND = Op0Opc == ISD::AND;
    // Check for rotlwi / rotrwi here, a special case of bitfield insert
    // where both bitfield halves are sourced from the same value.
    if (IsRotate && fullMask &&
        N->getOperand(0).getOperand(0) == N->getOperand(1).getOperand(0)) {
      SDOperand Tmp;
      Select(Tmp, N->getOperand(0).getOperand(0));
      return CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, Tmp,
                                   getI32Imm(SH), getI32Imm(0), getI32Imm(31));
    }
    SDOperand Tmp1, Tmp2;
    Select(Tmp1, ((Op0IsAND && fullMask) ? Op0.getOperand(0) : Op0));
    Select(Tmp2, (IsAndWithShiftOp ? Op1.getOperand(0).getOperand(0)
                                   : Op1.getOperand(0)));
    return CurDAG->getTargetNode(PPC::RLWIMI, MVT::i32, Tmp1, Tmp2,
                                 getI32Imm(SH), getI32Imm(MB), getI32Imm(ME));
  }
  return 0;
}

/// SelectAddrImm - Returns true if the address N can be represented by
/// a base register plus a signed 16-bit displacement [r+imm].
bool PPCDAGToDAGISel::SelectAddrImm(SDOperand N, SDOperand &Disp, 
                                    SDOperand &Base) {
  // If this can be more profitably realized as r+r, fail.
  if (SelectAddrIdx(N, Disp, Base))
    return false;

  if (N.getOpcode() == ISD::ADD) {
    unsigned imm = 0;
    if (isIntImmediate(N.getOperand(1), imm) && isInt16(imm)) {
      Disp = getI32Imm(imm & 0xFFFF);
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N.getOperand(0))) {
        Base = CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
      } else {
        Base = N.getOperand(0);
      }
      return true; // [r+i]
    } else if (N.getOperand(1).getOpcode() == PPCISD::Lo) {
      // Match LOAD (ADD (X, Lo(G))).
      assert(!cast<ConstantSDNode>(N.getOperand(1).getOperand(1))->getValue()
             && "Cannot handle constant offsets yet!");
      Disp = N.getOperand(1).getOperand(0);  // The global address.
      assert(Disp.getOpcode() == ISD::TargetGlobalAddress ||
             Disp.getOpcode() == ISD::TargetConstantPool);
      Base = N.getOperand(0);
      return true;  // [&g+r]
    }
  } else if (N.getOpcode() == ISD::OR) {
    unsigned imm = 0;
    if (isIntImmediate(N.getOperand(1), imm) && isInt16(imm)) {
      // If this is an or of disjoint bitfields, we can codegen this as an add
      // (for better address arithmetic) if the LHS and RHS of the OR are
      // provably disjoint.
      uint64_t LHSKnownZero, LHSKnownOne;
      PPCLowering.ComputeMaskedBits(N.getOperand(0), ~0U,
                                    LHSKnownZero, LHSKnownOne);
      if ((LHSKnownZero|~imm) == ~0U) {
        // If all of the bits are known zero on the LHS or RHS, the add won't
        // carry.
        Base = N.getOperand(0);
        Disp = getI32Imm(imm & 0xFFFF);
        return true;
      }
    }
  }
  Disp = getI32Imm(0);
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N))
    Base = CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
  else
    Base = N;
  return true;      // [r+0]
}

/// SelectAddrIdx - Given the specified addressed, check to see if it can be
/// represented as an indexed [r+r] operation.  Returns false if it can
/// be represented by [r+imm], which are preferred.
bool PPCDAGToDAGISel::SelectAddrIdx(SDOperand N, SDOperand &Base, 
                                    SDOperand &Index) {
  unsigned imm = 0;
  if (N.getOpcode() == ISD::ADD) {
    if (isIntImmediate(N.getOperand(1), imm) && isInt16(imm))
      return false;    // r+i
    if (N.getOperand(1).getOpcode() == PPCISD::Lo)
      return false;    // r+i
    
    Base = N.getOperand(0);
    Index = N.getOperand(1);
    return true;
  } else if (N.getOpcode() == ISD::OR) {
    if (isIntImmediate(N.getOperand(1), imm) && isInt16(imm))
      return false;    // r+i can fold it if we can.
    
    // If this is an or of disjoint bitfields, we can codegen this as an add
    // (for better address arithmetic) if the LHS and RHS of the OR are provably
    // disjoint.
    uint64_t LHSKnownZero, LHSKnownOne;
    uint64_t RHSKnownZero, RHSKnownOne;
    PPCLowering.ComputeMaskedBits(N.getOperand(0), ~0U,
                                  LHSKnownZero, LHSKnownOne);
    
    if (LHSKnownZero) {
      PPCLowering.ComputeMaskedBits(N.getOperand(1), ~0U,
                                    RHSKnownZero, RHSKnownOne);
      // If all of the bits are known zero on the LHS or RHS, the add won't
      // carry.
      if ((LHSKnownZero | RHSKnownZero) == ~0U) {
        Base = N.getOperand(0);
        Index = N.getOperand(1);
        return true;
      }
    }
  }
  
  return false;
}

/// SelectAddrIdxOnly - Given the specified addressed, force it to be
/// represented as an indexed [r+r] operation.
bool PPCDAGToDAGISel::SelectAddrIdxOnly(SDOperand N, SDOperand &Base, 
                                        SDOperand &Index) {
  // Check to see if we can easily represent this as an [r+r] address.  This
  // will fail if it thinks that the address is more profitably represented as
  // reg+imm, e.g. where imm = 0.
  if (!SelectAddrIdx(N, Base, Index)) {
    // Nope, do it the hard way.
    Base = CurDAG->getRegister(PPC::R0, MVT::i32);
    Index = N;
  }
  return true;
}

/// SelectCC - Select a comparison of the specified values with the specified
/// condition code, returning the CR# of the expression.
SDOperand PPCDAGToDAGISel::SelectCC(SDOperand LHS, SDOperand RHS,
                                    ISD::CondCode CC) {
  // Always select the LHS.
  Select(LHS, LHS);

  // Use U to determine whether the SETCC immediate range is signed or not.
  if (MVT::isInteger(LHS.getValueType())) {
    bool U = ISD::isUnsignedIntSetCC(CC);
    unsigned Imm;
    if (isIntImmediate(RHS, Imm) && 
        ((U && isUInt16(Imm)) || (!U && isInt16(Imm))))
      return SDOperand(CurDAG->getTargetNode(U ? PPC::CMPLWI : PPC::CMPWI,
                                    MVT::i32, LHS, getI32Imm(Imm & 0xFFFF)), 0);
    Select(RHS, RHS);
    return SDOperand(CurDAG->getTargetNode(U ? PPC::CMPLW : PPC::CMPW, MVT::i32,
                                           LHS, RHS), 0);
  } else if (LHS.getValueType() == MVT::f32) {
    Select(RHS, RHS);
    return SDOperand(CurDAG->getTargetNode(PPC::FCMPUS, MVT::i32, LHS, RHS), 0);
  } else {
    Select(RHS, RHS);
    return SDOperand(CurDAG->getTargetNode(PPC::FCMPUD, MVT::i32, LHS, RHS), 0);
  }
}

/// getBCCForSetCC - Returns the PowerPC condition branch mnemonic corresponding
/// to Condition.
static unsigned getBCCForSetCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETOEQ:    // FIXME: This is incorrect see PR642.
  case ISD::SETEQ:  return PPC::BEQ;
  case ISD::SETONE:    // FIXME: This is incorrect see PR642.
  case ISD::SETNE:  return PPC::BNE;
  case ISD::SETOLT:    // FIXME: This is incorrect see PR642.
  case ISD::SETULT:
  case ISD::SETLT:  return PPC::BLT;
  case ISD::SETOLE:    // FIXME: This is incorrect see PR642.
  case ISD::SETULE:
  case ISD::SETLE:  return PPC::BLE;
  case ISD::SETOGT:    // FIXME: This is incorrect see PR642.
  case ISD::SETUGT:
  case ISD::SETGT:  return PPC::BGT;
  case ISD::SETOGE:    // FIXME: This is incorrect see PR642.
  case ISD::SETUGE:
  case ISD::SETGE:  return PPC::BGE;
    
  case ISD::SETO:   return PPC::BUN;
  case ISD::SETUO:  return PPC::BNU;
  }
  return 0;
}

/// getCRIdxForSetCC - Return the index of the condition register field
/// associated with the SetCC condition, and whether or not the field is
/// treated as inverted.  That is, lt = 0; ge = 0 inverted.
static unsigned getCRIdxForSetCC(ISD::CondCode CC, bool& Inv) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETOLT:  // FIXME: This is incorrect see PR642.
  case ISD::SETULT:
  case ISD::SETLT:  Inv = false;  return 0;
  case ISD::SETOGE:  // FIXME: This is incorrect see PR642.
  case ISD::SETUGE:
  case ISD::SETGE:  Inv = true;   return 0;
  case ISD::SETOGT:  // FIXME: This is incorrect see PR642.
  case ISD::SETUGT:
  case ISD::SETGT:  Inv = false;  return 1;
  case ISD::SETOLE:  // FIXME: This is incorrect see PR642.
  case ISD::SETULE:
  case ISD::SETLE:  Inv = true;   return 1;
  case ISD::SETOEQ:  // FIXME: This is incorrect see PR642.
  case ISD::SETEQ:  Inv = false;  return 2;
  case ISD::SETONE:  // FIXME: This is incorrect see PR642.
  case ISD::SETNE:  Inv = true;   return 2;
  case ISD::SETO:   Inv = true;   return 3;
  case ISD::SETUO:  Inv = false;  return 3;
  }
  return 0;
}

SDOperand PPCDAGToDAGISel::SelectSETCC(SDOperand Op) {
  SDNode *N = Op.Val;
  unsigned Imm;
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  if (isIntImmediate(N->getOperand(1), Imm)) {
    // We can codegen setcc op, imm very efficiently compared to a brcond.
    // Check for those cases here.
    // setcc op, 0
    if (Imm == 0) {
      SDOperand Op;
      Select(Op, N->getOperand(0));
      switch (CC) {
      default: break;
      case ISD::SETEQ:
        Op = SDOperand(CurDAG->getTargetNode(PPC::CNTLZW, MVT::i32, Op), 0);
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Op, getI32Imm(27),
                                    getI32Imm(5), getI32Imm(31));
      case ISD::SETNE: {
        SDOperand AD =
          SDOperand(CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                          Op, getI32Imm(~0U)), 0);
        return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, AD, Op, 
                                    AD.getValue(1));
      }
      case ISD::SETLT:
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Op, getI32Imm(1),
                                    getI32Imm(31), getI32Imm(31));
      case ISD::SETGT: {
        SDOperand T =
          SDOperand(CurDAG->getTargetNode(PPC::NEG, MVT::i32, Op), 0);
        T = SDOperand(CurDAG->getTargetNode(PPC::ANDC, MVT::i32, T, Op), 0);
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, T, getI32Imm(1),
                                    getI32Imm(31), getI32Imm(31));
      }
      }
    } else if (Imm == ~0U) {        // setcc op, -1
      SDOperand Op;
      Select(Op, N->getOperand(0));
      switch (CC) {
      default: break;
      case ISD::SETEQ:
        Op = SDOperand(CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                             Op, getI32Imm(1)), 0);
        return CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32, 
                              SDOperand(CurDAG->getTargetNode(PPC::LI, MVT::i32,
                                                              getI32Imm(0)), 0),
                                    Op.getValue(1));
      case ISD::SETNE: {
        Op = SDOperand(CurDAG->getTargetNode(PPC::NOR, MVT::i32, Op, Op), 0);
        SDNode *AD = CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                           Op, getI32Imm(~0U));
        return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, SDOperand(AD, 0), Op, 
                                    SDOperand(AD, 1));
      }
      case ISD::SETLT: {
        SDOperand AD = SDOperand(CurDAG->getTargetNode(PPC::ADDI, MVT::i32, Op,
                                                       getI32Imm(1)), 0);
        SDOperand AN = SDOperand(CurDAG->getTargetNode(PPC::AND, MVT::i32, AD,
                                                       Op), 0);
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, AN, getI32Imm(1),
                                    getI32Imm(31), getI32Imm(31));
      }
      case ISD::SETGT:
        Op = SDOperand(CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, Op,
                                             getI32Imm(1), getI32Imm(31),
                                             getI32Imm(31)), 0);
        return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Op, getI32Imm(1));
      }
    }
  }
  
  bool Inv;
  unsigned Idx = getCRIdxForSetCC(CC, Inv);
  SDOperand CCReg = SelectCC(N->getOperand(0), N->getOperand(1), CC);
  SDOperand IntCR;
  
  // Force the ccreg into CR7.
  SDOperand CR7Reg = CurDAG->getRegister(PPC::CR7, MVT::i32);
  
  SDOperand InFlag(0, 0);  // Null incoming flag value.
  CCReg = CurDAG->getCopyToReg(CurDAG->getEntryNode(), CR7Reg, CCReg, 
                               InFlag).getValue(1);
  
  if (TLI.getTargetMachine().getSubtarget<PPCSubtarget>().isGigaProcessor())
    IntCR = SDOperand(CurDAG->getTargetNode(PPC::MFOCRF, MVT::i32, CR7Reg,
                                            CCReg), 0);
  else
    IntCR = SDOperand(CurDAG->getTargetNode(PPC::MFCR, MVT::i32, CCReg), 0);
  
  if (!Inv) {
    return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, IntCR,
                                getI32Imm((32-(3-Idx)) & 31),
                                getI32Imm(31), getI32Imm(31));
  } else {
    SDOperand Tmp =
      SDOperand(CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, IntCR,
                                      getI32Imm((32-(3-Idx)) & 31),
                                      getI32Imm(31),getI32Imm(31)), 0);
    return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Tmp, getI32Imm(1));
  }
}

/// isCallCompatibleAddress - Return true if the specified 32-bit value is
/// representable in the immediate field of a Bx instruction.
static bool isCallCompatibleAddress(ConstantSDNode *C) {
  int Addr = C->getValue();
  if (Addr & 3) return false;  // Low 2 bits are implicitly zero.
  return (Addr << 6 >> 6) == Addr;  // Top 6 bits have to be sext of immediate.
}

SDOperand PPCDAGToDAGISel::SelectCALL(SDOperand Op) {
  SDNode *N = Op.Val;
  SDOperand Chain;
  Select(Chain, N->getOperand(0));
  
  unsigned CallOpcode;
  std::vector<SDOperand> CallOperands;
  
  if (GlobalAddressSDNode *GASD =
      dyn_cast<GlobalAddressSDNode>(N->getOperand(1))) {
    CallOpcode = PPC::BL;
    CallOperands.push_back(N->getOperand(1));
  } else if (ExternalSymbolSDNode *ESSDN =
             dyn_cast<ExternalSymbolSDNode>(N->getOperand(1))) {
    CallOpcode = PPC::BL;
    CallOperands.push_back(N->getOperand(1));
  } else if (isa<ConstantSDNode>(N->getOperand(1)) &&
             isCallCompatibleAddress(cast<ConstantSDNode>(N->getOperand(1)))) {
    ConstantSDNode *C = cast<ConstantSDNode>(N->getOperand(1));
    CallOpcode = PPC::BLA;
    CallOperands.push_back(getI32Imm((int)C->getValue() >> 2));
  } else {
    // Copy the callee address into the CTR register.
    SDOperand Callee;
    Select(Callee, N->getOperand(1));
    Chain = SDOperand(CurDAG->getTargetNode(PPC::MTCTR, MVT::Other, Callee,
                                            Chain), 0);
    
    // Copy the callee address into R12 on darwin.
    SDOperand R12 = CurDAG->getRegister(PPC::R12, MVT::i32);
    Chain = CurDAG->getNode(ISD::CopyToReg, MVT::Other, Chain, R12, Callee);

    CallOperands.push_back(R12);
    CallOpcode = PPC::BCTRL;
  }
  
  unsigned GPR_idx = 0, FPR_idx = 0;
  static const unsigned GPR[] = {
    PPC::R3, PPC::R4, PPC::R5, PPC::R6,
    PPC::R7, PPC::R8, PPC::R9, PPC::R10,
  };
  static const unsigned FPR[] = {
    PPC::F1, PPC::F2, PPC::F3, PPC::F4, PPC::F5, PPC::F6, PPC::F7,
    PPC::F8, PPC::F9, PPC::F10, PPC::F11, PPC::F12, PPC::F13
  };
  
  SDOperand InFlag;  // Null incoming flag value.
  
  for (unsigned i = 2, e = N->getNumOperands(); i != e; ++i) {
    unsigned DestReg = 0;
    MVT::ValueType RegTy = N->getOperand(i).getValueType();
    if (RegTy == MVT::i32) {
      assert(GPR_idx < 8 && "Too many int args");
      DestReg = GPR[GPR_idx++];
    } else {
      assert(MVT::isFloatingPoint(N->getOperand(i).getValueType()) &&
             "Unpromoted integer arg?");
      assert(FPR_idx < 13 && "Too many fp args");
      DestReg = FPR[FPR_idx++];
    }
    
    if (N->getOperand(i).getOpcode() != ISD::UNDEF) {
      SDOperand Val;
      Select(Val, N->getOperand(i));
      Chain = CurDAG->getCopyToReg(Chain, DestReg, Val, InFlag);
      InFlag = Chain.getValue(1);
      CallOperands.push_back(CurDAG->getRegister(DestReg, RegTy));
    }
  }
  
  // Finally, once everything is in registers to pass to the call, emit the
  // call itself.
  if (InFlag.Val)
    CallOperands.push_back(InFlag);   // Strong dep on register copies.
  else
    CallOperands.push_back(Chain);    // Weak dep on whatever occurs before
  Chain = SDOperand(CurDAG->getTargetNode(CallOpcode, MVT::Other, MVT::Flag,
                                          CallOperands), 0);
  
  std::vector<SDOperand> CallResults;
  
  // If the call has results, copy the values out of the ret val registers.
  switch (N->getValueType(0)) {
    default: assert(0 && "Unexpected ret value!");
    case MVT::Other: break;
    case MVT::i32:
      if (N->getValueType(1) == MVT::i32) {
        Chain = CurDAG->getCopyFromReg(Chain, PPC::R4, MVT::i32, 
                                       Chain.getValue(1)).getValue(1);
        CallResults.push_back(Chain.getValue(0));
        Chain = CurDAG->getCopyFromReg(Chain, PPC::R3, MVT::i32,
                                       Chain.getValue(2)).getValue(1);
        CallResults.push_back(Chain.getValue(0));
      } else {
        Chain = CurDAG->getCopyFromReg(Chain, PPC::R3, MVT::i32,
                                       Chain.getValue(1)).getValue(1);
        CallResults.push_back(Chain.getValue(0));
      }
      break;
    case MVT::f32:
    case MVT::f64:
      Chain = CurDAG->getCopyFromReg(Chain, PPC::F1, N->getValueType(0),
                                     Chain.getValue(1)).getValue(1);
      CallResults.push_back(Chain.getValue(0));
      break;
  }
  
  CallResults.push_back(Chain);
  for (unsigned i = 0, e = CallResults.size(); i != e; ++i)
    CodeGenMap[Op.getValue(i)] = CallResults[i];
  return CallResults[Op.ResNo];
}

// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
void PPCDAGToDAGISel::Select(SDOperand &Result, SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&
      N->getOpcode() < PPCISD::FIRST_NUMBER) {
    Result = Op;
    return;   // Already selected.
  }

  // If this has already been converted, use it.
  std::map<SDOperand, SDOperand>::iterator CGMI = CodeGenMap.find(Op);
  if (CGMI != CodeGenMap.end()) {
    Result = CGMI->second;
    return;
  }
  
  switch (N->getOpcode()) {
  default: break;
  case ISD::VECTOR_SHUFFLE:
    // FIXME: This should be autogenerated from the .td file, it is here for now
    // due to bugs in tblgen.
    if (Op.getOperand(1).getOpcode() == ISD::UNDEF &&
        (Op.getValueType() == MVT::v4f32 || Op.getValueType() == MVT::v4i32) &&
        PPC::isSplatShuffleMask(Op.getOperand(2).Val)) {
      SDOperand N0;
      Select(N0, N->getOperand(0));
      Result = CodeGenMap[Op] = 
        SDOperand(CurDAG->getTargetNode(PPC::VSPLTW, MVT::v4f32,
                                        getI32Imm(PPC::getVSPLTImmediate(Op.getOperand(2).Val)),
                                        N0), 0);
      return;
    }
    assert(0 && "ILLEGAL VECTOR_SHUFFLE!");
      
  case ISD::SETCC:
    Result = SelectSETCC(Op);
    return;
  case PPCISD::CALL:
    Result = SelectCALL(Op);
    return;
  case PPCISD::GlobalBaseReg:
    Result = getGlobalBaseReg();
    return;
    
  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    if (N->hasOneUse()) {
      Result = CurDAG->SelectNodeTo(N, PPC::ADDI, MVT::i32,
                                    CurDAG->getTargetFrameIndex(FI, MVT::i32),
                                    getI32Imm(0));
      return;
    }
    Result = CodeGenMap[Op] = 
      SDOperand(CurDAG->getTargetNode(PPC::ADDI, MVT::i32,
                                      CurDAG->getTargetFrameIndex(FI, MVT::i32),
                                      getI32Imm(0)), 0);
    return;
  }
  case ISD::SDIV: {
    // FIXME: since this depends on the setting of the carry flag from the srawi
    //        we should really be making notes about that for the scheduler.
    // FIXME: It sure would be nice if we could cheaply recognize the 
    //        srl/add/sra pattern the dag combiner will generate for this as
    //        sra/addze rather than having to handle sdiv ourselves.  oh well.
    unsigned Imm;
    if (isIntImmediate(N->getOperand(1), Imm)) {
      SDOperand N0;
      Select(N0, N->getOperand(0));
      if ((signed)Imm > 0 && isPowerOf2_32(Imm)) {
        SDNode *Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                N0, getI32Imm(Log2_32(Imm)));
        Result = CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32, 
                                      SDOperand(Op, 0), SDOperand(Op, 1));
      } else if ((signed)Imm < 0 && isPowerOf2_32(-Imm)) {
        SDNode *Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                N0, getI32Imm(Log2_32(-Imm)));
        SDOperand PT =
          SDOperand(CurDAG->getTargetNode(PPC::ADDZE, MVT::i32,
                                          SDOperand(Op, 0), SDOperand(Op, 1)),
                    0);
        Result = CurDAG->SelectNodeTo(N, PPC::NEG, MVT::i32, PT);
      }
      return;
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::AND: {
    unsigned Imm, Imm2;
    // If this is an and of a value rotated between 0 and 31 bits and then and'd
    // with a mask, emit rlwinm
    if (isIntImmediate(N->getOperand(1), Imm) && (isShiftedMask_32(Imm) ||
                                                  isShiftedMask_32(~Imm))) {
      SDOperand Val;
      unsigned SH, MB, ME;
      if (isRotateAndMask(N->getOperand(0).Val, Imm, false, SH, MB, ME)) {
        Select(Val, N->getOperand(0).getOperand(0));
      } else if (Imm == 0) {
        // AND X, 0 -> 0, not "rlwinm 32".
        Select(Result, N->getOperand(1));
        return ;
      } else {        
        Select(Val, N->getOperand(0));
        isRunOfOnes(Imm, MB, ME);
        SH = 0;
      }
      Result = CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Val,
                                    getI32Imm(SH), getI32Imm(MB),
                                    getI32Imm(ME));
      return;
    }
    // ISD::OR doesn't get all the bitfield insertion fun.
    // (and (or x, c1), c2) where isRunOfOnes(~(c1^c2)) is a bitfield insert
    if (isIntImmediate(N->getOperand(1), Imm) && 
        N->getOperand(0).getOpcode() == ISD::OR &&
        isIntImmediate(N->getOperand(0).getOperand(1), Imm2)) {
      unsigned MB, ME;
      Imm = ~(Imm^Imm2);
      if (isRunOfOnes(Imm, MB, ME)) {
        SDOperand Tmp1, Tmp2;
        Select(Tmp1, N->getOperand(0).getOperand(0));
        Select(Tmp2, N->getOperand(0).getOperand(1));
        Result = SDOperand(CurDAG->getTargetNode(PPC::RLWIMI, MVT::i32,
                                                 Tmp1, Tmp2,
                                                 getI32Imm(0), getI32Imm(MB),
                                                 getI32Imm(ME)), 0);
        return;
      }
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::OR:
    if (SDNode *I = SelectBitfieldInsert(N)) {
      Result = CodeGenMap[Op] = SDOperand(I, 0);
      return;
    }
      
    // Other cases are autogenerated.
    break;
  case ISD::SHL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) {
      SDOperand Val;
      Select(Val, N->getOperand(0).getOperand(0));
      Result = CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, 
                                    Val, getI32Imm(SH), getI32Imm(MB),
                                    getI32Imm(ME));
      return;
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::SRL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) { 
      SDOperand Val;
      Select(Val, N->getOperand(0).getOperand(0));
      Result = CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, 
                                    Val, getI32Imm(SH & 0x1F), getI32Imm(MB),
                                    getI32Imm(ME));
      return;
    }
    
    // Other cases are autogenerated.
    break;
  }
  case ISD::SELECT_CC: {
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(4))->get();
    
    // handle the setcc cases here.  select_cc lhs, 0, 1, 0, cc
    if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N->getOperand(1)))
      if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N->getOperand(2)))
        if (ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N->getOperand(3)))
          if (N1C->isNullValue() && N3C->isNullValue() &&
              N2C->getValue() == 1ULL && CC == ISD::SETNE) {
            SDOperand LHS;
            Select(LHS, N->getOperand(0));
            SDNode *Tmp =
              CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                    LHS, getI32Imm(~0U));
            Result = CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32,
                                          SDOperand(Tmp, 0), LHS,
                                          SDOperand(Tmp, 1));
            return;
          }

    SDOperand CCReg = SelectCC(N->getOperand(0), N->getOperand(1), CC);
    unsigned BROpc = getBCCForSetCC(CC);

    bool isFP = MVT::isFloatingPoint(N->getValueType(0));
    unsigned SelectCCOp;
    if (MVT::isInteger(N->getValueType(0)))
      SelectCCOp = PPC::SELECT_CC_Int;
    else if (N->getValueType(0) == MVT::f32)
      SelectCCOp = PPC::SELECT_CC_F4;
    else
      SelectCCOp = PPC::SELECT_CC_F8;
    SDOperand N2, N3;
    Select(N2, N->getOperand(2));
    Select(N3, N->getOperand(3));
    Result = CurDAG->SelectNodeTo(N, SelectCCOp, N->getValueType(0), CCReg,
                                  N2, N3, getI32Imm(BROpc));
    return;
  }
  case ISD::BR_CC: {
    SDOperand Chain;
    Select(Chain, N->getOperand(0));
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
    SDOperand CondCode = SelectCC(N->getOperand(2), N->getOperand(3), CC);
    Result = CurDAG->SelectNodeTo(N, PPC::COND_BRANCH, MVT::Other, 
                                  CondCode, getI32Imm(getBCCForSetCC(CC)), 
                                  N->getOperand(4), Chain);
    return;
  }
  }
  
  SelectCode(Result, Op);
}


/// createPPCISelDag - This pass converts a legalized DAG into a 
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createPPCISelDag(PPCTargetMachine &TM) {
  return new PPCDAGToDAGISel(TM);
}


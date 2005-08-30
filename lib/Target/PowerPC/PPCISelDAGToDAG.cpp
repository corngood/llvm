//===-- PPC32ISelDAGToDAG.cpp - PPC32 pattern matching inst selector ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for 32 bit PowerPC,
// converting from a legalized dag to a PPC dag.
//
//===----------------------------------------------------------------------===//

#include "PowerPC.h"
#include "PPC32TargetMachine.h"
#include "PPC32ISelLowering.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

namespace {
  Statistic<> FusedFP ("ppc-codegen", "Number of fused fp operations");
  Statistic<> FrameOff("ppc-codegen", "Number of frame idx offsets collapsed");
    
  //===--------------------------------------------------------------------===//
  /// PPC32DAGToDAGISel - PPC32 specific code to select PPC32 machine
  /// instructions for SelectionDAG operations.
  ///
  class PPC32DAGToDAGISel : public SelectionDAGISel {
    PPC32TargetLowering PPC32Lowering;
    unsigned GlobalBaseReg;
  public:
    PPC32DAGToDAGISel(TargetMachine &TM)
      : SelectionDAGISel(PPC32Lowering), PPC32Lowering(TM) {}
    
    virtual bool runOnFunction(Function &Fn) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseReg = 0;
      return SelectionDAGISel::runOnFunction(Fn);
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
    SDOperand Select(SDOperand Op);
    
    SDNode *SelectIntImmediateExpr(SDOperand LHS, SDOperand RHS,
                                   unsigned OCHi, unsigned OCLo,
                                   bool IsArithmetic = false,
                                   bool Negate = false);
    SDNode *SelectBitfieldInsert(SDNode *N);

    /// SelectCC - Select a comparison of the specified values with the
    /// specified condition code, returning the CR# of the expression.
    SDOperand SelectCC(SDOperand LHS, SDOperand RHS, ISD::CondCode CC);

    /// SelectAddr - Given the specified address, return the two operands for a
    /// load/store instruction, and return true if it should be an indexed [r+r]
    /// operation.
    bool SelectAddr(SDOperand Addr, SDOperand &Op1, SDOperand &Op2);

    SDOperand BuildSDIVSequence(SDNode *N);
    SDOperand BuildUDIVSequence(SDNode *N);
    
    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
      DEBUG(BB->dump());
      // Select target instructions for the DAG.
      DAG.setRoot(Select(DAG.getRoot()));
      DAG.RemoveDeadNodes();
      
      // Emit machine code to BB. 
      ScheduleAndEmitDAG(DAG);
    }
 
    virtual const char *getPassName() const {
      return "PowerPC DAG->DAG Pattern Instruction Selection";
    } 
  };
}

/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
SDOperand PPC32DAGToDAGISel::getGlobalBaseReg() {
  if (!GlobalBaseReg) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    SSARegMap *RegMap = BB->getParent()->getSSARegMap();
    GlobalBaseReg = RegMap->createVirtualRegister(PPC32::GPRCRegisterClass);
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

// isOprShiftImm - Returns true if the specified operand is a shift opcode with
// a immediate shift count less than 32.
static bool isOprShiftImm(SDNode *N, unsigned& Opc, unsigned& SH) {
  Opc = N->getOpcode();
  return (Opc == ISD::SHL || Opc == ISD::SRL || Opc == ISD::SRA) &&
    isIntImmediate(N->getOperand(1).Val, SH) && SH < 32;
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

// isRotateAndMask - Returns true if Mask and Shift can be folded in to a rotate
// and mask opcode and mask operation.
static bool isRotateAndMask(SDNode *N, unsigned Mask, bool IsShiftMask,
                            unsigned &SH, unsigned &MB, unsigned &ME) {
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
  } else if (Opcode == ISD::SRA || Opcode == ISD::SRL) { 
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

// isOprNot - Returns true if the specified operand is an xor with immediate -1.
static bool isOprNot(SDNode *N) {
  unsigned Imm;
  return isOpcWithIntImmediate(N, ISD::XOR, Imm) && (signed)Imm == -1;
}

// Immediate constant composers.
// Lo16 - grabs the lo 16 bits from a 32 bit constant.
// Hi16 - grabs the hi 16 bits from a 32 bit constant.
// HA16 - computes the hi bits required if the lo bits are add/subtracted in
// arithmethically.
static unsigned Lo16(unsigned x)  { return x & 0x0000FFFF; }
static unsigned Hi16(unsigned x)  { return Lo16(x >> 16); }
static unsigned HA16(unsigned x)  { return Hi16((signed)x - (signed short)x); }

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
SDNode *PPC32DAGToDAGISel::SelectBitfieldInsert(SDNode *N) {
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
  if (isIntImmediate(Op1.getOperand(1), Value)) {
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
  } else {
    return 0;
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
      Op0 = CurDAG->getTargetNode(PPC::RLWINM, MVT::i32,
                                  Select(N->getOperand(0).getOperand(0)),
                                  getI32Imm(SH), getI32Imm(0), getI32Imm(31));
      return Op0.Val;
    }
    SDOperand Tmp1 = (Op0IsAND && fullMask) ? Select(Op0.getOperand(0))
                                            : Select(Op0);
    SDOperand Tmp2 = IsAndWithShiftOp ? Select(Op1.getOperand(0).getOperand(0)) 
                                      : Select(Op1.getOperand(0));
    Op0 = CurDAG->getTargetNode(PPC::RLWIMI, MVT::i32, Tmp1, Tmp2,
                                getI32Imm(SH), getI32Imm(MB), getI32Imm(ME));
    return Op0.Val;
  }
  return 0;
}

// SelectIntImmediateExpr - Choose code for integer operations with an immediate
// operand.
SDNode *PPC32DAGToDAGISel::SelectIntImmediateExpr(SDOperand LHS, SDOperand RHS,
                                                  unsigned OCHi, unsigned OCLo,
                                                  bool IsArithmetic,
                                                  bool Negate) {
  // Check to make sure this is a constant.
  ConstantSDNode *CN = dyn_cast<ConstantSDNode>(RHS);
  // Exit if not a constant.
  if (!CN) return 0;
  // Extract immediate.
  unsigned C = (unsigned)CN->getValue();
  // Negate if required (ISD::SUB).
  if (Negate) C = -C;
  // Get the hi and lo portions of constant.
  unsigned Hi = IsArithmetic ? HA16(C) : Hi16(C);
  unsigned Lo = Lo16(C);

  // If two instructions are needed and usage indicates it would be better to
  // load immediate into a register, bail out.
  if (Hi && Lo && CN->use_size() > 2) return false;

  // Select the first operand.
  SDOperand Opr0 = Select(LHS);

  if (Lo)  // Add in the lo-part.
    Opr0 = CurDAG->getTargetNode(OCLo, MVT::i32, Opr0, getI32Imm(Lo));
  if (Hi)  // Add in the hi-part.
    Opr0 = CurDAG->getTargetNode(OCHi, MVT::i32, Opr0, getI32Imm(Hi));
  return Opr0.Val;
}

/// SelectAddr - Given the specified address, return the two operands for a
/// load/store instruction, and return true if it should be an indexed [r+r]
/// operation.
bool PPC32DAGToDAGISel::SelectAddr(SDOperand Addr, SDOperand &Op1,
                                   SDOperand &Op2) {
  unsigned imm = 0;
  if (Addr.getOpcode() == ISD::ADD) {
    if (isIntImmediate(Addr.getOperand(1), imm) && isInt16(imm)) {
      Op1 = getI32Imm(Lo16(imm));
      if (FrameIndexSDNode *FI =
            dyn_cast<FrameIndexSDNode>(Addr.getOperand(0))) {
        ++FrameOff;
        Op2 = CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
      } else {
        Op2 = Select(Addr.getOperand(0));
      }
      return false;
    } else {
      Op1 = Select(Addr.getOperand(0));
      Op2 = Select(Addr.getOperand(1));
      return true;   // [r+r]
    }
  }

  // Now check if we're dealing with a global, and whether or not we should emit
  // an optimized load or store for statics.
  if (GlobalAddressSDNode *GN = dyn_cast<GlobalAddressSDNode>(Addr)) {
    GlobalValue *GV = GN->getGlobal();
    if (!GV->hasWeakLinkage() && !GV->isExternal()) {
      Op1 = CurDAG->getTargetGlobalAddress(GV, MVT::i32);
      if (PICEnabled)
        Op2 = CurDAG->getTargetNode(PPC::ADDIS, MVT::i32, getGlobalBaseReg(),
                                    Op1);
      else
        Op2 = CurDAG->getTargetNode(PPC::LIS, MVT::i32, Op1);
      return false;
    }
  } else if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Addr)) {
    Op1 = getI32Imm(0);
    Op2 = CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
    return false;
  } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Addr)) {
    Op1 = Addr;
    if (PICEnabled)
      Op2 = CurDAG->getTargetNode(PPC::ADDIS, MVT::i32, getGlobalBaseReg(),Op1);
    else
      Op2 = CurDAG->getTargetNode(PPC::LIS, MVT::i32, Op1);
    return false;
  }
  Op1 = getI32Imm(0);
  Op2 = Select(Addr);
  return false;
}

/// SelectCC - Select a comparison of the specified values with the specified
/// condition code, returning the CR# of the expression.
SDOperand PPC32DAGToDAGISel::SelectCC(SDOperand LHS, SDOperand RHS,
                                      ISD::CondCode CC) {
  // Always select the LHS.
  LHS = Select(LHS);

  // Use U to determine whether the SETCC immediate range is signed or not.
  if (MVT::isInteger(LHS.getValueType())) {
    bool U = ISD::isUnsignedIntSetCC(CC);
    unsigned Imm;
    if (isIntImmediate(RHS, Imm) && 
        ((U && isUInt16(Imm)) || (!U && isInt16(Imm))))
      return CurDAG->getTargetNode(U ? PPC::CMPLWI : PPC::CMPWI, MVT::i32,
                                   LHS, getI32Imm(Lo16(Imm)));
    return CurDAG->getTargetNode(U ? PPC::CMPLW : PPC::CMPW, MVT::i32,
                                 LHS, Select(RHS));
  } else {
    return CurDAG->getTargetNode(PPC::FCMPU, MVT::i32, LHS, Select(RHS));
  }
}

/// getBCCForSetCC - Returns the PowerPC condition branch mnemonic corresponding
/// to Condition.
static unsigned getBCCForSetCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETEQ:  return PPC::BEQ;
  case ISD::SETNE:  return PPC::BNE;
  case ISD::SETULT:
  case ISD::SETLT:  return PPC::BLT;
  case ISD::SETULE:
  case ISD::SETLE:  return PPC::BLE;
  case ISD::SETUGT:
  case ISD::SETGT:  return PPC::BGT;
  case ISD::SETUGE:
  case ISD::SETGE:  return PPC::BGE;
  }
  return 0;
}

/// getCRIdxForSetCC - Return the index of the condition register field
/// associated with the SetCC condition, and whether or not the field is
/// treated as inverted.  That is, lt = 0; ge = 0 inverted.
static unsigned getCRIdxForSetCC(ISD::CondCode CC, bool& Inv) {
  switch (CC) {
  default: assert(0 && "Unknown condition!"); abort();
  case ISD::SETULT:
  case ISD::SETLT:  Inv = false;  return 0;
  case ISD::SETUGE:
  case ISD::SETGE:  Inv = true;   return 0;
  case ISD::SETUGT:
  case ISD::SETGT:  Inv = false;  return 1;
  case ISD::SETULE:
  case ISD::SETLE:  Inv = true;   return 1;
  case ISD::SETEQ:  Inv = false;  return 2;
  case ISD::SETNE:  Inv = true;   return 2;
  }
  return 0;
}

// Structure used to return the necessary information to codegen an SDIV as
// a multiply.
struct ms {
  int m; // magic number
  int s; // shift amount
};

struct mu {
  unsigned int m; // magic number
  int a;          // add indicator
  int s;          // shift amount
};

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static struct ms magic(int d) {
  int p;
  unsigned int ad, anc, delta, q1, r1, q2, r2, t;
  const unsigned int two31 = 0x80000000U;
  struct ms mag;
  
  ad = abs(d);
  t = two31 + ((unsigned int)d >> 31);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 31;               // initialize p
  q1 = two31/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two31 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two31/ad;        // initialize q2 = 2p/abs(d)
  r2 = two31 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  
  mag.m = q2 + 1;
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 32;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static struct mu magicu(unsigned d)
{
  int p;
  unsigned int nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 31;                   // initialize p
  q1 = 0x80000000/nc;       // initialize q1 = 2p/nc
  r1 = 0x80000000 - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFF/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFF - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFF) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x80000000) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 32;  // resulting shift
  return magu;
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand PPC32DAGToDAGISel::BuildSDIVSequence(SDNode *N) {
  int d = (int)cast<ConstantSDNode>(N->getOperand(1))->getValue();
  ms magics = magic(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = CurDAG->getNode(ISD::MULHS, MVT::i32, N->getOperand(0),
                                CurDAG->getConstant(magics.m, MVT::i32));
  // If d > 0 and m < 0, add the numerator
  if (d > 0 && magics.m < 0)
    Q = CurDAG->getNode(ISD::ADD, MVT::i32, Q, N->getOperand(0));
  // If d < 0 and m > 0, subtract the numerator.
  if (d < 0 && magics.m > 0)
    Q = CurDAG->getNode(ISD::SUB, MVT::i32, Q, N->getOperand(0));
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0)
    Q = CurDAG->getNode(ISD::SRA, MVT::i32, Q,
                        CurDAG->getConstant(magics.s, MVT::i32));
  // Extract the sign bit and add it to the quotient
  SDOperand T =
    CurDAG->getNode(ISD::SRL, MVT::i32, Q, CurDAG->getConstant(31, MVT::i32));
  return CurDAG->getNode(ISD::ADD, MVT::i32, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand PPC32DAGToDAGISel::BuildUDIVSequence(SDNode *N) {
  unsigned d = (unsigned)cast<ConstantSDNode>(N->getOperand(1))->getValue();
  mu magics = magicu(d);
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = CurDAG->getNode(ISD::MULHU, MVT::i32, N->getOperand(0),
                                CurDAG->getConstant(magics.m, MVT::i32));
  if (magics.a == 0) {
    return CurDAG->getNode(ISD::SRL, MVT::i32, Q,
                           CurDAG->getConstant(magics.s, MVT::i32));
  } else {
    SDOperand NPQ = CurDAG->getNode(ISD::SUB, MVT::i32, N->getOperand(0), Q);
    NPQ = CurDAG->getNode(ISD::SRL, MVT::i32, NPQ,
                           CurDAG->getConstant(1, MVT::i32));
    NPQ = CurDAG->getNode(ISD::ADD, MVT::i32, NPQ, Q);
    return CurDAG->getNode(ISD::SRL, MVT::i32, NPQ,
                           CurDAG->getConstant(magics.s-1, MVT::i32));
  }
}

// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
SDOperand PPC32DAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&
      N->getOpcode() < PPCISD::FIRST_NUMBER)
    return Op;   // Already selected.
  
  switch (N->getOpcode()) {
  default:
    std::cerr << "Cannot yet select: ";
    N->dump();
    std::cerr << "\n";
    abort();
  case ISD::EntryToken:       // These leaves remain the same.
    return Op;
  case ISD::TokenFactor: {
    SDOperand New;
    if (N->getNumOperands() == 2) {
      SDOperand Op0 = Select(N->getOperand(0));
      SDOperand Op1 = Select(N->getOperand(1));
      New = CurDAG->getNode(ISD::TokenFactor, MVT::Other, Op0, Op1);
    } else {
      std::vector<SDOperand> Ops;
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        Ops.push_back(Select(N->getOperand(i)));
      New = CurDAG->getNode(ISD::TokenFactor, MVT::Other, Ops);
    }
    
    if (New.Val != N) {
      CurDAG->ReplaceAllUsesWith(Op, New);
      N = New.Val;
    }
    break;
  }
  case ISD::CopyFromReg: {
    SDOperand Chain = Select(N->getOperand(0));
    if (Chain == N->getOperand(0)) return Op; // No change
    SDOperand New = CurDAG->getCopyFromReg(Chain,
         cast<RegisterSDNode>(N->getOperand(1))->getReg(), N->getValueType(0));
    return New.getValue(Op.ResNo);
  }
  case ISD::CopyToReg: {
    SDOperand Chain = Select(N->getOperand(0));
    SDOperand Reg = N->getOperand(1);
    SDOperand Val = Select(N->getOperand(2));
    if (Chain != N->getOperand(0) || Val != N->getOperand(2)) {
      SDOperand New = CurDAG->getNode(ISD::CopyToReg, MVT::Other,
                                      Chain, Reg, Val);
      CurDAG->ReplaceAllUsesWith(Op, New);
      N = New.Val;
    }
    break;    
  }
  case ISD::Constant: {
    assert(N->getValueType(0) == MVT::i32);
    unsigned v = (unsigned)cast<ConstantSDNode>(N)->getValue();
    unsigned Hi = HA16(v);
    unsigned Lo = Lo16(v);

    // NOTE: This doesn't use SelectNodeTo, because doing that will prevent 
    // folding shared immediates into other the second instruction that 
    // uses it.
    if (Hi && Lo) {
      SDOperand Top = CurDAG->getTargetNode(PPC::LIS, MVT::i32, 
                                            getI32Imm(v >> 16));
      return CurDAG->getTargetNode(PPC::ORI, MVT::i32, Top, 
                                   getI32Imm(v & 0xFFFF));
    } else if (Lo) {
      return CurDAG->getTargetNode(PPC::LI, MVT::i32, getI32Imm(v));
    } else {
      return CurDAG->getTargetNode(PPC::LIS, MVT::i32, getI32Imm(v >> 16));
    }
  }
  case ISD::UNDEF:
    if (N->getValueType(0) == MVT::i32)
      CurDAG->SelectNodeTo(N, PPC::IMPLICIT_DEF_GPR, MVT::i32);
    else
      CurDAG->SelectNodeTo(N, PPC::IMPLICIT_DEF_FP, N->getValueType(0));
    break;
  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    CurDAG->SelectNodeTo(N, PPC::ADDI, MVT::i32,
                         CurDAG->getTargetFrameIndex(FI, MVT::i32),
                         getI32Imm(0));
    break;
  }
  case ISD::ConstantPool: {
    Constant *C = cast<ConstantPoolSDNode>(N)->get();
    SDOperand Tmp, CPI = CurDAG->getTargetConstantPool(C, MVT::i32);
    if (PICEnabled)
      Tmp = CurDAG->getTargetNode(PPC::ADDIS, MVT::i32, getGlobalBaseReg(),CPI);
    else
      Tmp = CurDAG->getTargetNode(PPC::LIS, MVT::i32, CPI);
    CurDAG->SelectNodeTo(N, PPC::LA, MVT::i32, Tmp, CPI);
    break;
  }
  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
    SDOperand Tmp;
    SDOperand GA = CurDAG->getTargetGlobalAddress(GV, MVT::i32);
    if (PICEnabled)
      Tmp = CurDAG->getTargetNode(PPC::ADDIS, MVT::i32, getGlobalBaseReg(), GA);
    else
      Tmp = CurDAG->getTargetNode(PPC::LIS, MVT::i32, GA);

    if (GV->hasWeakLinkage() || GV->isExternal())
      CurDAG->SelectNodeTo(N, PPC::LWZ, MVT::i32, GA, Tmp);
    else
      CurDAG->SelectNodeTo(N, PPC::LA, MVT::i32, Tmp, GA);
    break;
  }
  case ISD::DYNAMIC_STACKALLOC: {
    // FIXME: We are currently ignoring the requested alignment for handling
    // greater than the stack alignment.  This will need to be revisited at some
    // point.  Align = N.getOperand(2);
    if (!isa<ConstantSDNode>(N->getOperand(2)) ||
        cast<ConstantSDNode>(N->getOperand(2))->getValue() != 0) {
      std::cerr << "Cannot allocate stack object with greater alignment than"
                << " the stack alignment yet!";
      abort();
    }
    SDOperand Chain = Select(N->getOperand(0));
    SDOperand Amt   = Select(N->getOperand(1));
    
    SDOperand R1Reg = CurDAG->getRegister(PPC::R1, MVT::i32);
    
    // Subtract the amount (guaranteed to be a multiple of the stack alignment)
    // from the stack pointer, giving us the result pointer.
    SDOperand Result = CurDAG->getTargetNode(PPC::SUBF, MVT::i32, Amt, R1Reg);

    // Copy this result back into R1.
    Chain = CurDAG->getNode(ISD::CopyToReg, MVT::Other, Chain, R1Reg, Result);
    
    // Copy this result back out of R1 to make sure we're not using the stack
    // space without decrementing the stack pointer.
    Result = CurDAG->getCopyFromReg(Chain, PPC::R1, MVT::i32);
    
    // Finally, replace the DYNAMIC_STACKALLOC with the copyfromreg.
    CurDAG->ReplaceAllUsesWith(N, Result.Val);
    N = Result.Val;
    break;
  }      
  case ISD::SIGN_EXTEND_INREG:
    switch(cast<VTSDNode>(N->getOperand(1))->getVT()) {
    default: assert(0 && "Illegal type in SIGN_EXTEND_INREG"); break;
    case MVT::i16:
      CurDAG->SelectNodeTo(N, PPC::EXTSH, MVT::i32, Select(N->getOperand(0)));
      break;
    case MVT::i8:
      CurDAG->SelectNodeTo(N, PPC::EXTSB, MVT::i32, Select(N->getOperand(0)));
      break;
    }
    break;
  case ISD::CTLZ:
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, PPC::CNTLZW, MVT::i32, Select(N->getOperand(0)));
    break;
  case PPCISD::FSEL:
    CurDAG->SelectNodeTo(N, PPC::FSEL, N->getValueType(0),
                         Select(N->getOperand(0)),
                         Select(N->getOperand(1)),
                         Select(N->getOperand(2)));
    break;
  case ISD::ADD: {
    MVT::ValueType Ty = N->getValueType(0);
    if (Ty == MVT::i32) {
      if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), N->getOperand(1),
                                             PPC::ADDIS, PPC::ADDI, true)) {
        CurDAG->ReplaceAllUsesWith(Op, SDOperand(I, 0));
        N = I;
      } else {
        CurDAG->SelectNodeTo(N, PPC::ADD, MVT::i32, Select(N->getOperand(0)),
                             Select(N->getOperand(1)));
      }
      break;
    }
    
    if (!NoExcessFPPrecision) {  // Match FMA ops
      if (N->getOperand(0).getOpcode() == ISD::MUL &&
          N->getOperand(0).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty == MVT::f64 ? PPC::FMADD : PPC::FMADDS, Ty,
                             Select(N->getOperand(0).getOperand(0)),
                             Select(N->getOperand(0).getOperand(1)),
                             Select(N->getOperand(1)));
        break;
      } else if (N->getOperand(1).getOpcode() == ISD::MUL &&
                 N->getOperand(1).hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty == MVT::f64 ? PPC::FMADD : PPC::FMADDS, Ty,
                             Select(N->getOperand(1).getOperand(0)),
                             Select(N->getOperand(1).getOperand(1)),
                             Select(N->getOperand(0)));
        break;
      }
    }
    
    CurDAG->SelectNodeTo(N, Ty == MVT::f64 ? PPC::FADD : PPC::FADDS, Ty,
                         Select(N->getOperand(0)), Select(N->getOperand(1)));
    break;
  }
  case ISD::SUB: {
    MVT::ValueType Ty = N->getValueType(0);
    if (Ty == MVT::i32) {
      unsigned Imm;
      if (isIntImmediate(N->getOperand(0), Imm) && isInt16(Imm)) {
        if (0 == Imm)
          CurDAG->SelectNodeTo(N, PPC::NEG, Ty, Select(N->getOperand(1)));
        else
          CurDAG->SelectNodeTo(N, PPC::SUBFIC, Ty, Select(N->getOperand(1)),
                               getI32Imm(Lo16(Imm)));
        break;
      }
      if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), N->getOperand(1),
                                          PPC::ADDIS, PPC::ADDI, true, true)) {
        CurDAG->ReplaceAllUsesWith(Op, SDOperand(I, 0));
        N = I;
      } else {
        CurDAG->SelectNodeTo(N, PPC::SUBF, Ty, Select(N->getOperand(1)),
                             Select(N->getOperand(0)));
      }
      break;
    }
    
    if (!NoExcessFPPrecision) {  // Match FMA ops
      if (N->getOperand(0).getOpcode() == ISD::MUL &&
          N->getOperand(0).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty == MVT::f64 ? PPC::FMSUB : PPC::FMSUBS, Ty,
                             Select(N->getOperand(0).getOperand(0)),
                             Select(N->getOperand(0).getOperand(1)),
                             Select(N->getOperand(1)));
        break;
      } else if (N->getOperand(1).getOpcode() == ISD::MUL &&
                 N->getOperand(1).Val->hasOneUse()) {
        ++FusedFP; // Statistic
        CurDAG->SelectNodeTo(N, Ty == MVT::f64 ? PPC::FNMSUB : PPC::FNMSUBS, Ty,
                             Select(N->getOperand(1).getOperand(0)),
                             Select(N->getOperand(1).getOperand(1)),
                             Select(N->getOperand(0)));
        break;
      }
    }
    CurDAG->SelectNodeTo(N, Ty == MVT::f64 ? PPC::FSUB : PPC::FSUBS, Ty,
                         Select(N->getOperand(0)),
                         Select(N->getOperand(1)));
    break;
  }
  case ISD::MUL: {
    unsigned Imm, Opc;
    if (isIntImmediate(N->getOperand(1), Imm) && isInt16(Imm)) {
      CurDAG->SelectNodeTo(N, PPC::MULLI, MVT::i32,
                           Select(N->getOperand(0)), getI32Imm(Lo16(Imm)));
      break;
    } 
    switch (N->getValueType(0)) {
      default: assert(0 && "Unhandled multiply type!");
      case MVT::i32: Opc = PPC::MULLW; break;
      case MVT::f32: Opc = PPC::FMULS; break;
      case MVT::f64: Opc = PPC::FMUL;  break;
    }
    CurDAG->SelectNodeTo(N, Opc, N->getValueType(0), Select(N->getOperand(0)), 
                         Select(N->getOperand(1)));
    break;
  }
  case ISD::SDIV: {
    unsigned Imm;
    if (isIntImmediate(N->getOperand(1), Imm)) {
      if ((signed)Imm > 0 && isPowerOf2_32(Imm)) {
        SDOperand Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                Select(N->getOperand(0)),
                                getI32Imm(Log2_32(Imm)));
        CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32, 
                             Op.getValue(0), Op.getValue(1));
        break;
      } else if ((signed)Imm < 0 && isPowerOf2_32(-Imm)) {
        SDOperand Op =
          CurDAG->getTargetNode(PPC::SRAWI, MVT::i32, MVT::Flag,
                                Select(N->getOperand(0)),
                                getI32Imm(Log2_32(-Imm)));
        SDOperand PT =
          CurDAG->getTargetNode(PPC::ADDZE, MVT::i32, Op.getValue(0),
                                Op.getValue(1));
        CurDAG->SelectNodeTo(N, PPC::NEG, MVT::i32, PT);
        break;
      } else if (Imm) {
        SDOperand Result = Select(BuildSDIVSequence(N));
        assert(Result.ResNo == 0);
        CurDAG->ReplaceAllUsesWith(Op, Result);
        N = Result.Val;
        break;
      }
    }
    
    unsigned Opc;
    switch (N->getValueType(0)) {
    default: assert(0 && "Unknown type to ISD::SDIV");
    case MVT::i32: Opc = PPC::DIVW; break;
    case MVT::f32: Opc = PPC::FDIVS; break;
    case MVT::f64: Opc = PPC::FDIV; break;
    }
    CurDAG->SelectNodeTo(N, Opc, N->getValueType(0), Select(N->getOperand(0)),
                         Select(N->getOperand(1)));
    break;
  }
  case ISD::UDIV: {
    // If this is a divide by constant, we can emit code using some magic
    // constants to implement it as a multiply instead.
    unsigned Imm;
    if (isIntImmediate(N->getOperand(1), Imm) && Imm) {
      SDOperand Result = Select(BuildUDIVSequence(N));
      assert(Result.ResNo == 0);
      CurDAG->ReplaceAllUsesWith(Op, Result);
      N = Result.Val;
      break;
    }
    
    CurDAG->SelectNodeTo(N, PPC::DIVWU, MVT::i32, Select(N->getOperand(0)),
                         Select(N->getOperand(1)));
    break;
  }
  case ISD::MULHS:
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, PPC::MULHW, MVT::i32, Select(N->getOperand(0)), 
                         Select(N->getOperand(1)));
    break;
  case ISD::MULHU:
    assert(N->getValueType(0) == MVT::i32);
    CurDAG->SelectNodeTo(N, PPC::MULHWU, MVT::i32, Select(N->getOperand(0)),
                         Select(N->getOperand(1)));
    break;
  case ISD::AND: {
    unsigned Imm;
    // If this is an and of a value rotated between 0 and 31 bits and then and'd
    // with a mask, emit rlwinm
    if (isIntImmediate(N->getOperand(1), Imm) && (isShiftedMask_32(Imm) ||
                                                  isShiftedMask_32(~Imm))) {
      SDOperand Val;
      unsigned SH, MB, ME;
      if (isRotateAndMask(N->getOperand(0).Val, Imm, false, SH, MB, ME)) {
        Val = Select(N->getOperand(0).getOperand(0));
      } else {
        Val = Select(N->getOperand(0));
        isRunOfOnes(Imm, MB, ME);
        SH = 0;
      }
      CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Val, getI32Imm(SH),
                           getI32Imm(MB), getI32Imm(ME));
      break;
    }
    // If this is an and with an immediate that isn't a mask, then codegen it as
    // high and low 16 bit immediate ands.
    if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), 
                                           N->getOperand(1),
                                           PPC::ANDISo, PPC::ANDIo)) {
      CurDAG->ReplaceAllUsesWith(Op, SDOperand(I, 0));
      N = I;
      break;
    }
    // Finally, check for the case where we are being asked to select
    // and (not(a), b) or and (a, not(b)) which can be selected as andc.
    if (isOprNot(N->getOperand(0).Val))
      CurDAG->SelectNodeTo(N, PPC::ANDC, MVT::i32, Select(N->getOperand(1)),
                           Select(N->getOperand(0).getOperand(0)));
    else if (isOprNot(N->getOperand(1).Val))
      CurDAG->SelectNodeTo(N, PPC::ANDC, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1).getOperand(0)));
    else
      CurDAG->SelectNodeTo(N, PPC::AND, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  }
  case ISD::OR:
    if (SDNode *I = SelectBitfieldInsert(N)) {
      CurDAG->ReplaceAllUsesWith(Op, SDOperand(I, 0));
      N = I;
      break;
    }
    if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), 
                                           N->getOperand(1),
                                           PPC::ORIS, PPC::ORI)) {
      CurDAG->ReplaceAllUsesWith(Op, SDOperand(I, 0));
      N = I;
      break;
    }
    // Finally, check for the case where we are being asked to select
    // 'or (not(a), b)' or 'or (a, not(b))' which can be selected as orc.
    if (isOprNot(N->getOperand(0).Val))
      CurDAG->SelectNodeTo(N, PPC::ORC, MVT::i32, Select(N->getOperand(1)),
                           Select(N->getOperand(0).getOperand(0)));
    else if (isOprNot(N->getOperand(1).Val))
      CurDAG->SelectNodeTo(N, PPC::ORC, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1).getOperand(0)));
    else
      CurDAG->SelectNodeTo(N, PPC::OR, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  case ISD::XOR:
    // Check whether or not this node is a logical 'not'.  This is represented
    // by llvm as a xor with the constant value -1 (all bits set).  If this is a
    // 'not', then fold 'or' into 'nor', and so forth for the supported ops.
    if (isOprNot(N)) {
      unsigned Opc;
      SDOperand Val = Select(N->getOperand(0));
      switch (Val.isTargetOpcode() ? Val.getTargetOpcode() : 0) {
      default:        Opc = 0;          break;
      case PPC::OR:   Opc = PPC::NOR;   break;
      case PPC::AND:  Opc = PPC::NAND;  break;
      case PPC::XOR:  Opc = PPC::EQV;   break;
      }
      if (Opc)
        CurDAG->SelectNodeTo(N, Opc, MVT::i32, Val.getOperand(0),
                             Val.getOperand(1));
      else
        CurDAG->SelectNodeTo(N, PPC::NOR, MVT::i32, Val, Val);
      break;
    }
    // If this is a xor with an immediate other than -1, then codegen it as high
    // and low 16 bit immediate xors.
    if (SDNode *I = SelectIntImmediateExpr(N->getOperand(0), 
                                           N->getOperand(1),
                                           PPC::XORIS, PPC::XORI)) {
      CurDAG->ReplaceAllUsesWith(Op, SDOperand(I, 0));
      N = I;
      break;
    }
    // Finally, check for the case where we are being asked to select
    // xor (not(a), b) which is equivalent to not(xor a, b), which is eqv
    if (isOprNot(N->getOperand(0).Val))
      CurDAG->SelectNodeTo(N, PPC::EQV, MVT::i32, 
                           Select(N->getOperand(0).getOperand(0)),
                           Select(N->getOperand(1)));
    else
      CurDAG->SelectNodeTo(N, PPC::XOR, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  case ISD::SHL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME))
      CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, 
                           Select(N->getOperand(0).getOperand(0)),
                           getI32Imm(SH), getI32Imm(MB), getI32Imm(ME));
    else if (isIntImmediate(N->getOperand(1), Imm))
      CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Select(N->getOperand(0)),
                           getI32Imm(Imm), getI32Imm(0), getI32Imm(31-Imm));
    else
      CurDAG->SelectNodeTo(N, PPC::SLW, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  }
  case ISD::SRL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME))
      CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, 
                           Select(N->getOperand(0).getOperand(0)),
                           getI32Imm(SH), getI32Imm(MB), getI32Imm(ME));
    else if (isIntImmediate(N->getOperand(1), Imm))
      CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Select(N->getOperand(0)),
                           getI32Imm(32-Imm), getI32Imm(Imm), getI32Imm(31));
    else
      CurDAG->SelectNodeTo(N, PPC::SRW, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  }
  case ISD::SRA: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).Val, ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME))
      CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, 
                           Select(N->getOperand(0).getOperand(0)),
                           getI32Imm(SH), getI32Imm(MB), getI32Imm(ME));
    else if (isIntImmediate(N->getOperand(1), Imm))
      CurDAG->SelectNodeTo(N, PPC::SRAWI, MVT::i32, Select(N->getOperand(0)), 
                           getI32Imm(Imm));
    else
      CurDAG->SelectNodeTo(N, PPC::SRAW, MVT::i32, Select(N->getOperand(0)),
                           Select(N->getOperand(1)));
    break;
  }
  case ISD::FABS:
    CurDAG->SelectNodeTo(N, PPC::FABS, N->getValueType(0), 
                         Select(N->getOperand(0)));
    break;
  case ISD::FP_EXTEND:
    assert(MVT::f64 == N->getValueType(0) && 
           MVT::f32 == N->getOperand(0).getValueType() && "Illegal FP_EXTEND");
    // We need to emit an FMR to make sure that the result has the right value
    // type.
    CurDAG->SelectNodeTo(N, PPC::FMR, MVT::f64, Select(N->getOperand(0)));
    break;
  case ISD::FP_ROUND:
    assert(MVT::f32 == N->getValueType(0) && 
           MVT::f64 == N->getOperand(0).getValueType() && "Illegal FP_ROUND");
    CurDAG->SelectNodeTo(N, PPC::FRSP, MVT::f32, Select(N->getOperand(0)));
    break;
  case ISD::FP_TO_SINT: {
    SDOperand In = Select(N->getOperand(0));
    In = CurDAG->getTargetNode(PPC::FCTIWZ, MVT::f64, In);

    int FrameIdx = BB->getParent()->getFrameInfo()->CreateStackObject(8, 8);
    SDOperand FI = CurDAG->getTargetFrameIndex(FrameIdx, MVT::f64);
    SDOperand ST = CurDAG->getTargetNode(PPC::STFD, MVT::Other, In,
                                         getI32Imm(0), FI);
    CurDAG->SelectNodeTo(N, PPC::LWZ, MVT::i32, MVT::Other,
                         getI32Imm(4), FI, ST);
    break;
  }
  case ISD::FNEG: {
    SDOperand Val = Select(N->getOperand(0));
    MVT::ValueType Ty = N->getValueType(0);
    if (Val.Val->hasOneUse()) {
      unsigned Opc;
      switch (Val.isTargetOpcode() ? Val.getTargetOpcode() : 0) {
      default:          Opc = 0;            break;
      case PPC::FABS:   Opc = PPC::FNABS;   break;
      case PPC::FMADD:  Opc = PPC::FNMADD;  break;
      case PPC::FMADDS: Opc = PPC::FNMADDS; break;
      case PPC::FMSUB:  Opc = PPC::FNMSUB;  break;
      case PPC::FMSUBS: Opc = PPC::FNMSUBS; break;
      }
      // If we inverted the opcode, then emit the new instruction with the
      // inverted opcode and the original instruction's operands.  Otherwise, 
      // fall through and generate a fneg instruction.
      if (Opc) {
        if (PPC::FNABS == Opc)
          CurDAG->SelectNodeTo(N, Opc, Ty, Val.getOperand(0));
        else
          CurDAG->SelectNodeTo(N, Opc, Ty, Val.getOperand(0),
                               Val.getOperand(1), Val.getOperand(2));
        break;
      }
    }
    CurDAG->SelectNodeTo(N, PPC::FNEG, Ty, Val);
    break;
  }
  case ISD::FSQRT: {
    MVT::ValueType Ty = N->getValueType(0);
    CurDAG->SelectNodeTo(N, Ty == MVT::f64 ? PPC::FSQRT : PPC::FSQRTS, Ty,
                         Select(N->getOperand(0)));
    break;
  }
    
  case ISD::ADD_PARTS: {
    SDOperand LHSL = Select(N->getOperand(0));
    SDOperand LHSH = Select(N->getOperand(1));
   
    unsigned Imm;
    bool ME = false, ZE = false;
    if (isIntImmediate(N->getOperand(3), Imm)) {
      ME = (signed)Imm == -1;
      ZE = Imm == 0;
    }

    std::vector<SDOperand> Result;
    SDOperand CarryFromLo;
    if (isIntImmediate(N->getOperand(2), Imm) &&
        ((signed)Imm >= -32768 || (signed)Imm < 32768)) {
      // Codegen the low 32 bits of the add.  Interestingly, there is no
      // shifted form of add immediate carrying.
      CarryFromLo = CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                          LHSL, getI32Imm(Imm));
    } else {
      CarryFromLo = CurDAG->getTargetNode(PPC::ADDC, MVT::i32, MVT::Flag,
                                          LHSL, Select(N->getOperand(2)));
    }
    CarryFromLo = CarryFromLo.getValue(1);
    
    // Codegen the high 32 bits, adding zero, minus one, or the full value
    // along with the carry flag produced by addc/addic.
    SDOperand ResultHi;
    if (ZE)
      ResultHi = CurDAG->getTargetNode(PPC::ADDZE, MVT::i32, LHSH, CarryFromLo);
    else if (ME)
      ResultHi = CurDAG->getTargetNode(PPC::ADDME, MVT::i32, LHSH, CarryFromLo);
    else
      ResultHi = CurDAG->getTargetNode(PPC::ADDE, MVT::i32, LHSH,
                                       Select(N->getOperand(3)), CarryFromLo);
    Result.push_back(ResultHi);
    Result.push_back(CarryFromLo.getValue(0));
    CurDAG->ReplaceAllUsesWith(N, Result);
    return Result[Op.ResNo];
  }
  case ISD::SUB_PARTS: {
    SDOperand LHSL = Select(N->getOperand(0));
    SDOperand LHSH = Select(N->getOperand(1));
    SDOperand RHSL = Select(N->getOperand(2));
    SDOperand RHSH = Select(N->getOperand(3));

    std::vector<SDOperand> Result;
    Result.push_back(CurDAG->getTargetNode(PPC::SUBFC, MVT::i32, MVT::Flag,
                                           RHSL, LHSL));
    Result.push_back(CurDAG->getTargetNode(PPC::SUBFE, MVT::i32, RHSH, LHSH,
                                           Result[0].getValue(1)));
    CurDAG->ReplaceAllUsesWith(N, Result);
    return Result[Op.ResNo];
  }
  case ISD::SHL_PARTS: {
    SDOperand HI = Select(N->getOperand(0));
    SDOperand LO = Select(N->getOperand(1));
    SDOperand SH = Select(N->getOperand(2));
    SDOperand SH_LO_R = CurDAG->getTargetNode(PPC::SUBFIC, MVT::i32, MVT::Flag,
                                              SH, getI32Imm(32));
    SDOperand SH_LO_L = CurDAG->getTargetNode(PPC::ADDI, MVT::i32, SH, 
                                          getI32Imm((unsigned)-32));
    SDOperand HI_SHL = CurDAG->getTargetNode(PPC::SLW, MVT::i32, HI, SH);
    SDOperand HI_LOR = CurDAG->getTargetNode(PPC::SRW, MVT::i32, LO, SH_LO_R);
    SDOperand HI_LOL = CurDAG->getTargetNode(PPC::SLW, MVT::i32, LO, SH_LO_L);
    SDOperand HI_OR =  CurDAG->getTargetNode(PPC::OR, MVT::i32, HI_SHL, HI_LOR);

    std::vector<SDOperand> Result;
    Result.push_back(CurDAG->getTargetNode(PPC::SLW, MVT::i32, LO, SH));
    Result.push_back(CurDAG->getTargetNode(PPC::OR, MVT::i32, HI_OR, HI_LOL));
    CurDAG->ReplaceAllUsesWith(N, Result);
    return Result[Op.ResNo];
  }
  case ISD::SRL_PARTS: {
    SDOperand HI = Select(N->getOperand(0));
    SDOperand LO = Select(N->getOperand(1));
    SDOperand SH = Select(N->getOperand(2));
    SDOperand SH_HI_L = CurDAG->getTargetNode(PPC::SUBFIC, MVT::i32, MVT::Flag,
                                              SH, getI32Imm(32));
    SDOperand SH_HI_R = CurDAG->getTargetNode(PPC::ADDI, MVT::i32, SH, 
                                              getI32Imm((unsigned)-32));
    SDOperand LO_SHR = CurDAG->getTargetNode(PPC::SRW, MVT::i32, LO, SH);
    SDOperand LO_HIL = CurDAG->getTargetNode(PPC::SLW, MVT::i32, HI, SH_HI_L);
    SDOperand LO_HIR = CurDAG->getTargetNode(PPC::SRW, MVT::i32, HI, SH_HI_R);
    SDOperand LO_OR =  CurDAG->getTargetNode(PPC::OR, MVT::i32, LO_SHR, LO_HIL);

    std::vector<SDOperand> Result;
    Result.push_back(CurDAG->getTargetNode(PPC::OR, MVT::i32, LO_OR, LO_HIR));
    Result.push_back(CurDAG->getTargetNode(PPC::SRW, MVT::i32, HI, SH));
    CurDAG->ReplaceAllUsesWith(N, Result);
    return Result[Op.ResNo];
  }
    
  case ISD::LOAD:
  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD: {
    SDOperand Op1, Op2;
    bool isIdx = SelectAddr(N->getOperand(1), Op1, Op2);

    MVT::ValueType TypeBeingLoaded = (N->getOpcode() == ISD::LOAD) ?
      N->getValueType(0) : cast<VTSDNode>(N->getOperand(3))->getVT();
    unsigned Opc;
    switch (TypeBeingLoaded) {
    default: N->dump(); assert(0 && "Cannot load this type!");
    case MVT::i1:
    case MVT::i8:  Opc = isIdx ? PPC::LBZX : PPC::LBZ; break;
    case MVT::i16:
      if (N->getOpcode() == ISD::SEXTLOAD) { // SEXT load?
        Opc = isIdx ? PPC::LHAX : PPC::LHA;
      } else {
        Opc = isIdx ? PPC::LHZX : PPC::LHZ;
      }
      break;
    case MVT::i32: Opc = isIdx ? PPC::LWZX : PPC::LWZ; break;
    case MVT::f32: Opc = isIdx ? PPC::LFSX : PPC::LFS; break;
    case MVT::f64: Opc = isIdx ? PPC::LFDX : PPC::LFD; break;
    }

    CurDAG->SelectNodeTo(N, Opc, N->getValueType(0), MVT::Other,
                         Op1, Op2, Select(N->getOperand(0)));
    break;
  }

  case ISD::TRUNCSTORE:
  case ISD::STORE: {
    SDOperand AddrOp1, AddrOp2;
    bool isIdx = SelectAddr(N->getOperand(2), AddrOp1, AddrOp2);

    unsigned Opc;
    if (N->getOpcode() == ISD::STORE) {
      switch (N->getOperand(1).getValueType()) {
      default: assert(0 && "unknown Type in store");
      case MVT::i32: Opc = isIdx ? PPC::STWX  : PPC::STW; break;
      case MVT::f64: Opc = isIdx ? PPC::STFDX : PPC::STFD; break;
      case MVT::f32: Opc = isIdx ? PPC::STFSX : PPC::STFS; break;
      }
    } else { //ISD::TRUNCSTORE
      switch(cast<VTSDNode>(N->getOperand(4))->getVT()) {
      default: assert(0 && "unknown Type in store");
      case MVT::i1:
      case MVT::i8:  Opc = isIdx ? PPC::STBX : PPC::STB; break;
      case MVT::i16: Opc = isIdx ? PPC::STHX : PPC::STH; break;
      }
    }
    
    CurDAG->SelectNodeTo(N, Opc, MVT::Other, Select(N->getOperand(1)),
                         AddrOp1, AddrOp2, Select(N->getOperand(0)));
    break;
  }
    
  case ISD::SETCC: {
    unsigned Imm;
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
    if (isIntImmediate(N->getOperand(1), Imm)) {
      // We can codegen setcc op, imm very efficiently compared to a brcond.
      // Check for those cases here.
      // setcc op, 0
      if (Imm == 0) {
        SDOperand Op = Select(N->getOperand(0));
        switch (CC) {
        default: assert(0 && "Unhandled SetCC condition"); abort();
        case ISD::SETEQ:
          Op = CurDAG->getTargetNode(PPC::CNTLZW, MVT::i32, Op);
          CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Op, getI32Imm(27),
                               getI32Imm(5), getI32Imm(31));
          break;
        case ISD::SETNE: {
          SDOperand AD = CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                               Op, getI32Imm(~0U));
          CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, AD, Op, AD.getValue(1));
          break;
        }
        case ISD::SETLT:
          CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Op, getI32Imm(1),
                               getI32Imm(31), getI32Imm(31));
          break;
        case ISD::SETGT: {
          SDOperand T = CurDAG->getTargetNode(PPC::NEG, MVT::i32, Op);
          T = CurDAG->getTargetNode(PPC::ANDC, MVT::i32, T, Op);;
          CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, T, getI32Imm(1),
                               getI32Imm(31), getI32Imm(31));
          break;
        }
        }
        break;
      } else if (Imm == ~0U) {        // setcc op, -1
        SDOperand Op = Select(N->getOperand(0));
        switch (CC) {
        default: assert(0 && "Unhandled SetCC condition"); abort();
        case ISD::SETEQ:
          Op = CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                     Op, getI32Imm(1));
          CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32, 
                               CurDAG->getTargetNode(PPC::LI, MVT::i32,
                                                     getI32Imm(0)),
                               Op.getValue(1));
          break;
        case ISD::SETNE: {
          Op = CurDAG->getTargetNode(PPC::NOR, MVT::i32, Op, Op);
          SDOperand AD = CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                                Op, getI32Imm(~0U));
          CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, AD, Op, AD.getValue(1));
          break;
        }
        case ISD::SETLT: {
          SDOperand AD = CurDAG->getTargetNode(PPC::ADDI, MVT::i32, Op,
                                               getI32Imm(1));
          SDOperand AN = CurDAG->getTargetNode(PPC::AND, MVT::i32, AD, Op);
          CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, AN, getI32Imm(1),
                               getI32Imm(31), getI32Imm(31));
          break;
        }
        case ISD::SETGT:
          Op = CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, Op, getI32Imm(1),
                                     getI32Imm(31), getI32Imm(31));
          CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Op, getI32Imm(1));
          break;
        }
        break;
      }
    }
    
    bool Inv;
    unsigned Idx = getCRIdxForSetCC(CC, Inv);
    SDOperand CCReg =
      SelectCC(Select(N->getOperand(0)), Select(N->getOperand(1)), CC);
    SDOperand IntCR;

    // Force the ccreg into CR7.
    SDOperand CR7Reg = CurDAG->getRegister(PPC::CR7, MVT::i32);
    
    std::vector<MVT::ValueType> VTs;
    VTs.push_back(MVT::Other);
    VTs.push_back(MVT::Flag);    // NONSTANDARD CopyToReg node: defines a flag
    std::vector<SDOperand> Ops;
    Ops.push_back(CurDAG->getEntryNode());
    Ops.push_back(CR7Reg);
    Ops.push_back(CCReg);
    CCReg = CurDAG->getNode(ISD::CopyToReg, VTs, Ops).getValue(1);
    
    if (TLI.getTargetMachine().getSubtarget<PPCSubtarget>().isGigaProcessor())
      IntCR = CurDAG->getTargetNode(PPC::MFOCRF, MVT::i32, CR7Reg, CCReg);
    else
      IntCR = CurDAG->getTargetNode(PPC::MFCR, MVT::i32, CCReg);
    
    if (!Inv) {
      CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, IntCR,
                           getI32Imm(32-(3-Idx)), getI32Imm(31), getI32Imm(31));
    } else {
      SDOperand Tmp =
      CurDAG->getTargetNode(PPC::RLWINM, MVT::i32, IntCR,
                            getI32Imm(32-(3-Idx)), getI32Imm(31),getI32Imm(31));
      CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Tmp, getI32Imm(1));
    }
      
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
            SDOperand LHS = Select(N->getOperand(0));
            SDOperand Tmp =
              CurDAG->getTargetNode(PPC::ADDIC, MVT::i32, MVT::Flag,
                                    LHS, getI32Imm(~0U));
            CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, Tmp, LHS,
                                 Tmp.getValue(1));
            break;
          }

    SDOperand CCReg = SelectCC(Select(N->getOperand(0)),
                               Select(N->getOperand(1)), CC);
    unsigned BROpc = getBCCForSetCC(CC);

    bool isFP = MVT::isFloatingPoint(N->getValueType(0));
    unsigned SelectCCOp = isFP ? PPC::SELECT_CC_FP : PPC::SELECT_CC_Int;
    CurDAG->SelectNodeTo(N, SelectCCOp, N->getValueType(0), CCReg,
                         Select(N->getOperand(2)), Select(N->getOperand(3)),
                         getI32Imm(BROpc));
    break;
  }
    
  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END: {
    unsigned Amt = cast<ConstantSDNode>(N->getOperand(1))->getValue();
    unsigned Opc = N->getOpcode() == ISD::CALLSEQ_START ?
                       PPC::ADJCALLSTACKDOWN : PPC::ADJCALLSTACKUP;
    CurDAG->SelectNodeTo(N, Opc, MVT::Other,
                         getI32Imm(Amt), Select(N->getOperand(0)));
    break;
  }
  case ISD::CALL:
  case ISD::TAILCALL: {
    SDOperand Chain = Select(N->getOperand(0));

    unsigned CallOpcode;
    std::vector<SDOperand> CallOperands;
    
    if (GlobalAddressSDNode *GASD =
        dyn_cast<GlobalAddressSDNode>(N->getOperand(1))) {
      CallOpcode = PPC::CALLpcrel;
      CallOperands.push_back(CurDAG->getTargetGlobalAddress(GASD->getGlobal(),
                                                            MVT::i32));
    } else if (ExternalSymbolSDNode *ESSDN =
               dyn_cast<ExternalSymbolSDNode>(N->getOperand(1))) {
      CallOpcode = PPC::CALLpcrel;
      CallOperands.push_back(N->getOperand(1));
    } else {
      // Copy the callee address into the CTR register.
      SDOperand Callee = Select(N->getOperand(1));
      Chain = CurDAG->getTargetNode(PPC::MTCTR, MVT::Other, Callee, Chain);

      // Copy the callee address into R12 on darwin.
      SDOperand R12 = CurDAG->getRegister(PPC::R12, MVT::i32);
      Chain = CurDAG->getNode(ISD::CopyToReg, MVT::Other, Chain, R12, Callee);
      
      CallOperands.push_back(getI32Imm(20));  // Information to encode indcall
      CallOperands.push_back(getI32Imm(0));   // Information to encode indcall
      CallOperands.push_back(R12);
      CallOpcode = PPC::CALLindirect;
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
      MVT::ValueType RegTy;
      if (N->getOperand(i).getValueType() == MVT::i32) {
        assert(GPR_idx < 8 && "Too many int args");
        DestReg = GPR[GPR_idx++];
        RegTy = MVT::i32;
      } else {
        assert(MVT::isFloatingPoint(N->getOperand(i).getValueType()) &&
               "Unpromoted integer arg?");
        assert(FPR_idx < 13 && "Too many fp args");
        DestReg = FPR[FPR_idx++];
        RegTy = MVT::f64;   // Even if this is really f32!
      }

      if (N->getOperand(i).getOpcode() != ISD::UNDEF) {
        Chain = CurDAG->getCopyToReg(Chain, DestReg,
                                     Select(N->getOperand(i)), InFlag);
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
    Chain = CurDAG->getTargetNode(CallOpcode, MVT::Other, MVT::Flag,
                                  CallOperands);
    
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
                                       Chain.getValue(1)).getValue(1);
        CallResults.push_back(Chain.getValue(0));
      } else {
        Chain = CurDAG->getCopyFromReg(Chain, PPC::R3, MVT::i32,
                                       Chain.getValue(1)).getValue(1);
        CallResults.push_back(Chain.getValue(0));
      }
      break;
    case MVT::f32:
    case MVT::f64:
      Chain = CurDAG->getCopyFromReg(Chain, PPC::F1, MVT::f64,
                                     Chain.getValue(1)).getValue(1);
      if (N->getValueType(0) == MVT::f64)
        CallResults.push_back(Chain.getValue(0));
      else
        // Insert an FMR to convert the result to f32 from f64.
        CallResults.push_back(CurDAG->getTargetNode(PPC::FMR, MVT::f32,
                                                    Chain.getValue(0)));
      break;
    }
    
    CallResults.push_back(Chain);
    CurDAG->ReplaceAllUsesWith(N, CallResults);
    return CallResults[Op.ResNo];
  }
  case ISD::RET: {
    SDOperand Chain = Select(N->getOperand(0));     // Token chain.

    if (N->getNumOperands() > 1) {
      SDOperand Val = Select(N->getOperand(1));
      switch (N->getOperand(1).getValueType()) {
      default: assert(0 && "Unknown return type!");
      case MVT::f32:
        // Insert a copy to get the type right.
        Val = CurDAG->getTargetNode(PPC::FMR, MVT::f64, Val);
        // FALL THROUGH
      case MVT::f64:
        Chain = CurDAG->getCopyToReg(Chain, PPC::F1, Val);
        break;
      case MVT::i32:
        Chain = CurDAG->getCopyToReg(Chain, PPC::R3, Val);
        break;
      }

      if (N->getNumOperands() > 2) {
        assert(N->getOperand(1).getValueType() == MVT::i32 &&
               N->getOperand(2).getValueType() == MVT::i32 &&
               N->getNumOperands() == 3 && "Unknown two-register ret value!");
        Val = Select(N->getOperand(2));
        Chain = CurDAG->getCopyToReg(Chain, PPC::R4, Val);
      }
    }

    // Finally, select this to a blr (return) instruction.
    CurDAG->SelectNodeTo(N, PPC::BLR, MVT::Other, Chain);
    break;
  }
  case ISD::BR:
    CurDAG->SelectNodeTo(N, PPC::B, MVT::Other, N->getOperand(1),
                         Select(N->getOperand(0)));
    break;
  case ISD::BR_CC:
  case ISD::BRTWOWAY_CC: {
    SDOperand Chain = Select(N->getOperand(0));
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N->getOperand(4))->getBasicBlock();
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
    SDOperand CondCode = SelectCC(N->getOperand(2), N->getOperand(3), CC);
    unsigned Opc = getBCCForSetCC(CC);

    // If this is a two way branch, then grab the fallthrough basic block
    // argument and build a PowerPC branch pseudo-op, suitable for long branch
    // conversion if necessary by the branch selection pass.  Otherwise, emit a
    // standard conditional branch.
    if (N->getOpcode() == ISD::BRTWOWAY_CC) {
      MachineBasicBlock *Fallthrough =
        cast<BasicBlockSDNode>(N->getOperand(5))->getBasicBlock();
      SDOperand CB = CurDAG->getTargetNode(PPC::COND_BRANCH, MVT::Other,
                                           CondCode, getI32Imm(Opc),
                                           N->getOperand(4), N->getOperand(5),
                                           Chain);
      CurDAG->SelectNodeTo(N, PPC::B, MVT::Other, N->getOperand(5), CB);
    } else {
      // Iterate to the next basic block
      ilist<MachineBasicBlock>::iterator It = BB;
      ++It;

      // If the fallthrough path is off the end of the function, which would be
      // undefined behavior, set it to be the same as the current block because
      // we have nothing better to set it to, and leaving it alone will cause
      // the PowerPC Branch Selection pass to crash.
      if (It == BB->getParent()->end()) It = Dest;
      CurDAG->SelectNodeTo(N, PPC::COND_BRANCH, MVT::Other, CondCode,
                           getI32Imm(Opc), N->getOperand(4),
                           CurDAG->getBasicBlock(It), Chain);
    }
    break;
  }
  }
  return SDOperand(N, Op.ResNo);
}


/// createPPC32ISelDag - This pass converts a legalized DAG into a 
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createPPC32ISelDag(TargetMachine &TM) {
  return new PPC32DAGToDAGISel(TM);
}


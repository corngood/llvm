//===- ARMDisassemblerCore.h - ARM disassembler helpers ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
//
// The first part defines the enumeration type of ARM instruction format, which
// specifies the encoding used by the instruction, as well as a helper function
// to convert the enums to printable char strings.
//
// It also contains code to represent the concepts of Builder, Builder Factory,
// as well as the Algorithm to solve the problem of disassembling an ARM instr.
//
//===----------------------------------------------------------------------===//

#ifndef ARMDISASSEMBLERCORE_H
#define ARMDISASSEMBLERCORE_H

#include "llvm/MC/MCInst.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "ARMInstrInfo.h"
#include "ARMDisassembler.h"

namespace llvm {

class ARMUtils {
public:
  static const char *OpcodeName(unsigned Opcode);
};

/////////////////////////////////////////////////////
//                                                 //
//  Enums and Utilities for ARM Instruction Format //
//                                                 //
/////////////////////////////////////////////////////

#define ARM_FORMATS                   \
  ENTRY(ARM_FORMAT_PSEUDO,         0) \
  ENTRY(ARM_FORMAT_MULFRM,         1) \
  ENTRY(ARM_FORMAT_BRFRM,          2) \
  ENTRY(ARM_FORMAT_BRMISCFRM,      3) \
  ENTRY(ARM_FORMAT_DPFRM,          4) \
  ENTRY(ARM_FORMAT_DPSOREGFRM,     5) \
  ENTRY(ARM_FORMAT_LDFRM,          6) \
  ENTRY(ARM_FORMAT_STFRM,          7) \
  ENTRY(ARM_FORMAT_LDMISCFRM,      8) \
  ENTRY(ARM_FORMAT_STMISCFRM,      9) \
  ENTRY(ARM_FORMAT_LDSTMULFRM,    10) \
  ENTRY(ARM_FORMAT_LDSTEXFRM,     11) \
  ENTRY(ARM_FORMAT_ARITHMISCFRM,  12) \
  ENTRY(ARM_FORMAT_EXTFRM,        13) \
  ENTRY(ARM_FORMAT_VFPUNARYFRM,   14) \
  ENTRY(ARM_FORMAT_VFPBINARYFRM,  15) \
  ENTRY(ARM_FORMAT_VFPCONV1FRM,   16) \
  ENTRY(ARM_FORMAT_VFPCONV2FRM,   17) \
  ENTRY(ARM_FORMAT_VFPCONV3FRM,   18) \
  ENTRY(ARM_FORMAT_VFPCONV4FRM,   19) \
  ENTRY(ARM_FORMAT_VFPCONV5FRM,   20) \
  ENTRY(ARM_FORMAT_VFPLDSTFRM,    21) \
  ENTRY(ARM_FORMAT_VFPLDSTMULFRM, 22) \
  ENTRY(ARM_FORMAT_VFPMISCFRM,    23) \
  ENTRY(ARM_FORMAT_THUMBFRM,      24) \
  ENTRY(ARM_FORMAT_NEONFRM,       25) \
  ENTRY(ARM_FORMAT_NEONGETLNFRM,  26) \
  ENTRY(ARM_FORMAT_NEONSETLNFRM,  27) \
  ENTRY(ARM_FORMAT_NEONDUPFRM,    28) \
  ENTRY(ARM_FORMAT_MISCFRM,       29) \
  ENTRY(ARM_FORMAT_THUMBMISCFRM,  30) \
  ENTRY(ARM_FORMAT_NLdSt,         31) \
  ENTRY(ARM_FORMAT_N1RegModImm,   32) \
  ENTRY(ARM_FORMAT_N2Reg,         33) \
  ENTRY(ARM_FORMAT_NVCVT,         34) \
  ENTRY(ARM_FORMAT_NVecDupLn,     35) \
  ENTRY(ARM_FORMAT_N2RegVecShL,   36) \
  ENTRY(ARM_FORMAT_N2RegVecShR,   37) \
  ENTRY(ARM_FORMAT_N3Reg,         38) \
  ENTRY(ARM_FORMAT_N3RegVecSh,    39) \
  ENTRY(ARM_FORMAT_NVecExtract,   40) \
  ENTRY(ARM_FORMAT_NVecMulScalar, 41) \
  ENTRY(ARM_FORMAT_NVTBL,         42)

// ARM instruction format specifies the encoding used by the instruction.
#define ENTRY(n, v) n = v,
typedef enum {
  ARM_FORMATS
  ARM_FORMAT_NA
} ARMFormat;
#undef ENTRY

// Converts enum to const char*.
static const inline char *stringForARMFormat(ARMFormat form) {
#define ENTRY(n, v) case n: return #n;
  switch(form) {
    ARM_FORMATS
  case ARM_FORMAT_NA:
  default:
    return "";
  }
#undef ENTRY
}

/// Expands on the enum definitions from ARMBaseInstrInfo.h.
/// They are being used by the disassembler implementation.
namespace ARMII {
  enum {
    NEONRegMask = 15,
    GPRRegMask = 15,
    NEON_RegRdShift = 12,
    NEON_D_BitShift = 22,
    NEON_RegRnShift = 16,
    NEON_N_BitShift = 7,
    NEON_RegRmShift = 0,
    NEON_M_BitShift = 5
  };
}

/// Utility function for extracting [From, To] bits from a uint32_t.
static inline unsigned slice(uint32_t Bits, unsigned From, unsigned To) {
  assert(From < 32 && To < 32 && From >= To);
  return (Bits >> To) & ((1 << (From - To + 1)) - 1);
}

/// Utility function for setting [From, To] bits to Val for a uint32_t.
static inline void setSlice(uint32_t &Bits, unsigned From, unsigned To,
                            uint32_t Val) {
  assert(From < 32 && To < 32 && From >= To);
  uint32_t Mask = ((1 << (From - To + 1)) - 1);
  Bits &= ~(Mask << To);
  Bits |= (Val & Mask) << To;
}

/// Various utilities for checking the target specific flags.

/// A unary data processing instruction doesn't have an Rn operand.
static inline bool isUnaryDP(unsigned TSFlags) {
  return (TSFlags & ARMII::UnaryDP);
}

/// This four-bit field describes the addressing mode used.
/// See also ARMBaseInstrInfo.h.
static inline unsigned getAddrMode(unsigned TSFlags) {
  return (TSFlags & ARMII::AddrModeMask);
}

/// {IndexModePre, IndexModePost}
/// Only valid for load and store ops.
/// See also ARMBaseInstrInfo.h.
static inline unsigned getIndexMode(unsigned TSFlags) {
  return (TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift;
}

/// Pre-/post-indexed operations define an extra $base_wb in the OutOperandList.
static inline bool isPrePostLdSt(unsigned TSFlags) {
  return (TSFlags & ARMII::IndexModeMask) != 0;
}

// Forward declaration.
class ARMBasicMCBuilder;

// Builder Object is mostly ignored except in some Thumb disassemble functions.
typedef ARMBasicMCBuilder *BO;

/// DisassembleFP - DisassembleFP points to a function that disassembles an insn
/// and builds the MCOperand list upon disassembly.  It returns false on failure
/// or true on success.  The number of operands added is updated upon success.
typedef bool (*DisassembleFP)(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO Builder);

/// ARMAlgorithm - ARMAlgorithm implements the ARM/Thumb disassembly by solving
/// the problem of building the MCOperands of an MCInst.  Construction of
/// ARMAlgorithm requires passing in a function pointer with the DisassembleFP
/// data type.
class ARMAlgorithm {
public:
  /// GetInstance - GetInstance returns an instance of ARMAlgorithm given the
  /// encoding Format.  API clients should not free up the returned instance.
  static ARMAlgorithm *GetInstance(ARMFormat Format);

  /// Return true if this algorithm successfully disassembles the instruction.
  /// NumOpsAdded is updated to reflect the number of operands added by the
  /// algorithm.  NumOpsAdded may be less than NumOps, in which case, there are
  /// operands unaccounted for which need to be dealt with by the API client.
  bool Solve(MCInst &MI, unsigned Opcode, uint32_t insn, unsigned short NumOps,
      unsigned &NumOpsAdded, BO Builder) const {
    if (Disassemble == NULL)
      return false;

    return (*Disassemble)(MI, Opcode, insn, NumOps, NumOpsAdded, Builder);
  }

private:
  ARMAlgorithm(DisassembleFP fp) : Disassemble(fp) {}
  ARMAlgorithm(ARMAlgorithm &AA) : Disassemble(AA.Disassemble) {}

  virtual ~ARMAlgorithm() {}

  DisassembleFP Disassemble;
};

/// ARMBasicMCBuilder - ARMBasicMCBuilder represents an ARM MCInst builder that
/// knows how to build up the MCOperand list.
class ARMBasicMCBuilder {
  unsigned Opcode;
  ARMFormat Format;
  unsigned short NumOps;
  const ARMAlgorithm &Algo;
  Session *SP;

public:
  ARMBasicMCBuilder(ARMBasicMCBuilder &B)
    : Opcode(B.Opcode), Format(B.Format), NumOps(B.NumOps), Algo(B.Algo),
      SP(B.SP)
  {}

  /// Opcode, Format, NumOperands, and Algo make an ARM Basic MCBuilder.
  ARMBasicMCBuilder(unsigned opc, ARMFormat format, unsigned short num,
                    const ARMAlgorithm &algo)
    : Opcode(opc), Format(format), NumOps(num), Algo(algo), SP(0)
  {}

  void setSession(Session *sp) {
    SP = sp;
  }

  /// TryPredicateAndSBitModifier - TryPredicateAndSBitModifier tries to process
  /// the possible Predicate and SBitModifier, to build the remaining MCOperand
  /// constituents.
  bool TryPredicateAndSBitModifier(MCInst& MI, unsigned Opcode,
      uint32_t insn, unsigned short NumOpsRemaning);

  /// InITBlock - InITBlock returns true if we are inside an IT block.
  bool InITBlock() {
    if (SP)
      return SP->ITCounter > 0;

    return false;
  }

  /// Build - Build delegates to BuildIt to perform the heavy liftling.  After
  /// that, it invokes RunBuildAfterHook where some housekeepings can be done.
  virtual bool Build(MCInst &MI, uint32_t insn) {
    bool Status = BuildIt(MI, insn);
    return RunBuildAfterHook(Status, MI, insn);
  }

  /// BuildIt - BuildIt performs the build step for this ARM Basic MC Builder.
  /// The general idea is to set the Opcode for the MCInst, followed by adding
  /// the appropriate MCOperands to the MCInst.  ARM Basic MC Builder delegates
  /// to the Algo (ARM Disassemble Algorithm) object to perform Format-specific
  /// disassembly, followed by class method TryPredicateAndSBitModifier() to do
  /// PredicateOperand and OptionalDefOperand which follow the Dst/Src Operands.
  virtual bool BuildIt(MCInst &MI, uint32_t insn);

  /// RunBuildAfterHook - RunBuildAfterHook performs operations deemed necessary
  /// after BuildIt is finished.
  virtual bool RunBuildAfterHook(bool Status, MCInst &MI, uint32_t insn);

private:
  /// Get condition of the current IT instruction.
  unsigned GetITCond() {
    assert(SP);
    return slice(SP->ITState, 7, 4);
  }
};

/// CreateMCBuilder - Return an ARMBasicMCBuilder that can build up the MC
/// infrastructure of an MCInst given the Opcode and Format of the instr.
/// Return NULL if it fails to create/return a proper builder.  API clients
/// are responsible for freeing up of the allocated memory.  Cacheing can be
/// performed by the API clients to improve performance.
extern ARMBasicMCBuilder *CreateMCBuilder(unsigned Opcode, ARMFormat Format);

} // namespace llvm

#endif

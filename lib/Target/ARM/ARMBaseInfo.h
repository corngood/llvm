//===-- ARMBaseInfo.h - Top level definitions for ARM -------- --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the ARM target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef ARMBASEINFO_H
#define ARMBASEINFO_H

#include "llvm/Support/ErrorHandling.h"

namespace llvm {

// Enums corresponding to ARM condition codes
namespace ARMCC {
  // The CondCodes constants map directly to the 4-bit encoding of the
  // condition field for predicated instructions.
  enum CondCodes { // Meaning (integer)          Meaning (floating-point)
    EQ,            // Equal                      Equal
    NE,            // Not equal                  Not equal, or unordered
    HS,            // Carry set                  >, ==, or unordered
    LO,            // Carry clear                Less than
    MI,            // Minus, negative            Less than
    PL,            // Plus, positive or zero     >, ==, or unordered
    VS,            // Overflow                   Unordered
    VC,            // No overflow                Not unordered
    HI,            // Unsigned higher            Greater than, or unordered
    LS,            // Unsigned lower or same     Less than or equal
    GE,            // Greater than or equal      Greater than or equal
    LT,            // Less than                  Less than, or unordered
    GT,            // Greater than               Greater than
    LE,            // Less than or equal         <, ==, or unordered
    AL             // Always (unconditional)     Always (unconditional)
  };

  inline static CondCodes getOppositeCondition(CondCodes CC) {
    switch (CC) {
    default: llvm_unreachable("Unknown condition code");
    case EQ: return NE;
    case NE: return EQ;
    case HS: return LO;
    case LO: return HS;
    case MI: return PL;
    case PL: return MI;
    case VS: return VC;
    case VC: return VS;
    case HI: return LS;
    case LS: return HI;
    case GE: return LT;
    case LT: return GE;
    case GT: return LE;
    case LE: return GT;
    }
  }
} // namespace ARMCC

inline static const char *ARMCondCodeToString(ARMCC::CondCodes CC) {
  switch (CC) {
  default: llvm_unreachable("Unknown condition code");
  case ARMCC::EQ:  return "eq";
  case ARMCC::NE:  return "ne";
  case ARMCC::HS:  return "hs";
  case ARMCC::LO:  return "lo";
  case ARMCC::MI:  return "mi";
  case ARMCC::PL:  return "pl";
  case ARMCC::VS:  return "vs";
  case ARMCC::VC:  return "vc";
  case ARMCC::HI:  return "hi";
  case ARMCC::LS:  return "ls";
  case ARMCC::GE:  return "ge";
  case ARMCC::LT:  return "lt";
  case ARMCC::GT:  return "gt";
  case ARMCC::LE:  return "le";
  case ARMCC::AL:  return "al";
  }
}

namespace ARM_MB {
  // The Memory Barrier Option constants map directly to the 4-bit encoding of
  // the option field for memory barrier operations.
  enum MemBOpt {
    ST    = 14,
    ISH   = 11,
    ISHST = 10,
    NSH   = 7,
    NSHST = 6,
    OSH   = 3,
    OSHST = 2
  };

  inline static const char *MemBOptToString(unsigned val) {
    switch (val) {
    default: llvm_unreachable("Unknown memory operation");
    case ST:    return "st";
    case ISH:   return "ish";
    case ISHST: return "ishst";
    case NSH:   return "nsh";
    case NSHST: return "nshst";
    case OSH:   return "osh";
    case OSHST: return "oshst";
    }
  }
} // namespace ARM_MB
} // end namespace llvm;

// Note that the following auto-generated files only defined enum types, and
// so are safe to include here.

// Defines symbolic names for ARM registers.  This defines a mapping from
// register name to register number.
//
#include "ARMGenRegisterNames.inc"

// Defines symbolic names for the ARM instructions.
//
#include "ARMGenInstrNames.inc"

#endif

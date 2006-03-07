//===-- PPCHazardRecognizers.cpp - PowerPC Hazard Recognizer Impls --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements hazard recognizers for scheduling on PowerPC processors.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "PPCHazardRecognizers.h"
#include "PPC.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;


//===----------------------------------------------------------------------===//
// PowerPC 970 Hazard Recognizer
//
// FIXME: This is missing some significant cases:
//   0. Handling of instructions that must be the first/last in a group.
//   1. Modeling of microcoded instructions.
//   2. Handling of cracked instructions.
//   3. Handling of serialized operations.
//   4. Handling of the esoteric cases in "Resource-based Instruction Grouping",
//      e.g. integer divides that only execute in the second slot.
//
// Note: on the PPC970, logical CR operations are more expensive in their three
// address form: ops that read/write the same register are half as expensive as
//

void PPCHazardRecognizer970::EndDispatchGroup() {
  DEBUG(std::cerr << "=== Start of dispatch group\n");
  // Pipeline units.
  NumFXU = NumLSU = NumFPU = 0;
  HasCR = HasVALU = HasVPERM = false;
  NumIssued = 0;
  
  // Structural hazard info.
  HasCTRSet = false;
  StorePtr1 = StorePtr2 = SDOperand();
  StoreSize = 0;
}


PPCHazardRecognizer970::PPC970InstrType
PPCHazardRecognizer970::GetInstrType(unsigned Opcode) {
  if (Opcode < ISD::BUILTIN_OP_END)
    return PseudoInst;
  Opcode -= ISD::BUILTIN_OP_END;
  
  switch (Opcode) {
  case PPC::FMRSD: return PseudoInst;  // Usually coallesced away.
  case PPC::BCTRL:
  case PPC::BL:
  case PPC::BLA:
    return BR;
  case PPC::LFS:
  case PPC::LWZ:
    return LSU_LD;
  case PPC::STFD:
    return LSU_ST;
  case PPC::FADDS:
  case PPC::FCTIWZ:
    return FPU;
  }
  
  return FXU;
}


/// StartBasicBlock - Initiate a new dispatch group.
void PPCHazardRecognizer970::StartBasicBlock() {
  EndDispatchGroup();
}

/// isLoadOfStoredAddress - If we have a load from the previously stored pointer
/// as indicated by StorePtr1/StorePtr2/StoreSize, return true.
bool PPCHazardRecognizer970::
isLoadOfStoredAddress(unsigned LoadSize, SDOperand Ptr1, SDOperand Ptr2) const {
  // Handle exact and commuted addresses.
  if (Ptr1 == StorePtr1 && Ptr2 == StorePtr2)
    return true;
  if (Ptr2 == StorePtr1 && Ptr1 == StorePtr2)
    return true;
  
  // Okay, we don't have an exact match, if this is an indexed offset, see if we
  // have overlap (which happens during fp->int conversion for example).
  if (StorePtr2 == Ptr2) {
    if (ConstantSDNode *StoreOffset = dyn_cast<ConstantSDNode>(StorePtr1))
      if (ConstantSDNode *LoadOffset = dyn_cast<ConstantSDNode>(Ptr1)) {
        // Okay the base pointers match, so we have [c1+r] vs [c2+r].  Check to
        // see if the load and store actually overlap.
        int StoreOffs = StoreOffset->getValue();
        int LoadOffs  = LoadOffset->getValue();
        if (StoreOffs < LoadOffs) {
          if (int(StoreOffs+StoreSize) > LoadOffs) return true;
        } else {
          if (int(LoadOffs+LoadSize) > StoreOffs) return true;
        }
      }
  }
  return false;
}

/// getHazardType - We return hazard for any non-branch instruction that would
/// terminate terminate the dispatch group.  We turn NoopHazard for any
/// instructions that wouldn't terminate the dispatch group that would cause a
/// pipeline flush.
HazardRecognizer::HazardType PPCHazardRecognizer970::
getHazardType(SDNode *Node) {
  PPC970InstrType InstrType = GetInstrType(Node->getOpcode());
  if (InstrType == PseudoInst) return NoHazard;  
  unsigned Opcode = Node->getOpcode()-ISD::BUILTIN_OP_END;

  switch (InstrType) {
  default: assert(0 && "Unknown instruction type!");
  case FXU:    if (NumFXU  == 2) return Hazard;
  case LSU_ST:
  case LSU_LD: if (NumLSU  == 2) return Hazard;
  case FPU:    if (NumFPU  == 2) return Hazard;
  case CR:     if (HasCR) return Hazard;
  case VALU:   if (HasVALU) return Hazard;
  case VPERM:  if (HasVPERM) return Hazard;
  case BR:    break;
  }

  // We can only issue a branch as the last instruction in a group.
  if (NumIssued == 4 && InstrType != BR)
    return Hazard;
  
  // Do not allow MTCTR and BCTRL to be in the same dispatch group.
  if (HasCTRSet && Opcode == PPC::BCTRL)
    return NoopHazard;
  
  // If this is a load following a store, make sure it's not to the same or
  // overlapping address.
  if (InstrType == LSU_LD && StoreSize) {
    unsigned LoadSize;
    switch (Opcode) {
    default: assert(0 && "Unknown load!");
    case PPC::LFS:
    case PPC::LWZ: LoadSize = 4; break;
    }
    
    if (isLoadOfStoredAddress(LoadSize, 
                              Node->getOperand(0), Node->getOperand(1)))
      return NoopHazard;
  }
  
  return NoHazard;
}

void PPCHazardRecognizer970::EmitInstruction(SDNode *Node) {
  PPC970InstrType InstrType = GetInstrType(Node->getOpcode());
  if (InstrType == PseudoInst) return;
  unsigned Opcode = Node->getOpcode()-ISD::BUILTIN_OP_END;

  // Update structural hazard information.
  if (Opcode == PPC::MTCTR) HasCTRSet = true;
  
  // Track the address stored to.
  if (InstrType == LSU_ST) {
    StorePtr1 = Node->getOperand(1);
    StorePtr2 = Node->getOperand(2);
    switch (Opcode) {
    default: assert(0 && "Unknown store instruction!");
    case PPC::STFD: StoreSize = 8; break;
    }
  }
  
  switch (InstrType) {
  default: assert(0 && "Unknown instruction type!");
  case FXU:    ++NumFXU; break;
  case LSU_LD:
  case LSU_ST: ++NumLSU; break;
  case FPU:    ++NumFPU; break;
  case CR:     HasCR    = true; break;
  case VALU:   HasVALU  = true; break;
  case VPERM:  HasVPERM = true; break;
  case BR:     NumIssued = 4; return;  // ends a d-group.
  }
  ++NumIssued;
  
  if (NumIssued == 5)
    EndDispatchGroup();
}

void PPCHazardRecognizer970::AdvanceCycle() {
  assert(NumIssued < 5 && "Illegal dispatch group!");
  ++NumIssued;
  if (NumIssued == 5)
    EndDispatchGroup();
}

void PPCHazardRecognizer970::EmitNoop() {
  AdvanceCycle();
}

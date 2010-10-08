//===-- ARM/ARMMCCodeEmitter.cpp - Convert ARM code to machine code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARMMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-emitter"
#include "ARM.h"
#include "ARMInstrInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {
class ARMMCCodeEmitter : public MCCodeEmitter {
  ARMMCCodeEmitter(const ARMMCCodeEmitter &); // DO NOT IMPLEMENT
  void operator=(const ARMMCCodeEmitter &); // DO NOT IMPLEMENT
  const TargetMachine &TM;
  const TargetInstrInfo &TII;
  MCContext &Ctx;

public:
  ARMMCCodeEmitter(TargetMachine &tm, MCContext &ctx)
    : TM(tm), TII(*TM.getInstrInfo()), Ctx(ctx) {
  }

  ~ARMMCCodeEmitter() {}

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  unsigned getBinaryCodeForInstr(const MCInst &MI) const;

  /// getMachineOpValue - Return binary encoding of operand. If the machine
  /// operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI,const MCOperand &MO) const;
  unsigned getMachineOpValue(const MCInst &MI, unsigned OpIdx) const {
    return getMachineOpValue(MI, MI.getOperand(OpIdx));
  }

  unsigned getNumFixupKinds() const {
    assert(0 && "ARMMCCodeEmitter::getNumFixupKinds() not yet implemented.");
    return 0;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    static MCFixupKindInfo rtn;
    assert(0 && "ARMMCCodeEmitter::getFixupKindInfo() not yet implemented.");
    return rtn;
  }

  static unsigned GetARMRegNum(const MCOperand &MO) {
    // FIXME: getARMRegisterNumbering() is sufficient?
    assert(0 && "ARMMCCodeEmitter::GetARMRegNum() not yet implemented.");
    return 0;
  }

  void EmitByte(unsigned char C, unsigned &CurByte, raw_ostream &OS) const {
    OS << (char)C;
    ++CurByte;
  }

  void EmitConstant(uint64_t Val, unsigned Size, unsigned &CurByte,
                    raw_ostream &OS) const {
    // Output the constant in little endian byte order.
    for (unsigned i = 0; i != Size; ++i) {
      EmitByte(Val & 255, CurByte, OS);
      Val >>= 8;
    }
  }

  void EmitImmediate(const MCOperand &Disp,
                     unsigned ImmSize, MCFixupKind FixupKind,
                     unsigned &CurByte, raw_ostream &OS,
                     SmallVectorImpl<MCFixup> &Fixups,
                     int ImmOffset = 0) const;

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;
};

} // end anonymous namespace


MCCodeEmitter *llvm::createARMMCCodeEmitter(const Target &,
                                             TargetMachine &TM,
                                             MCContext &Ctx) {
  return new ARMMCCodeEmitter(TM, Ctx);
}

void ARMMCCodeEmitter::
EmitImmediate(const MCOperand &DispOp, unsigned Size, MCFixupKind FixupKind,
              unsigned &CurByte, raw_ostream &OS,
              SmallVectorImpl<MCFixup> &Fixups, int ImmOffset) const {
  assert(0 && "ARMMCCodeEmitter::EmitImmediate() not yet implemented.");
}

void ARMMCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = TII.get(Opcode);
  uint64_t TSFlags = Desc.TSFlags;
  // Keep track of the current byte being emitted.
  unsigned CurByte = 0;

  // Pseudo instructions don't get encoded.
  if ((TSFlags & ARMII::FormMask) == ARMII::Pseudo)
    return;

  ++MCNumEmitted;  // Keep track of the # of mi's emitted
  switch (Opcode) {
  //FIXME: Any non-pseudos that need special handling, if there are any...
  default: {
    unsigned Value = getBinaryCodeForInstr(MI);
    EmitConstant(Value, 4, CurByte, OS);
    break;
  }
  }
}

// FIXME: These #defines shouldn't be necessary. Instead, tblgen should
// be able to generate code emitter helpers for either variant, like it
// does for the AsmWriter.
#define ARMCodeEmitter ARMMCCodeEmitter
#define MachineInstr MCInst
#include "ARMGenCodeEmitter.inc"
#undef ARMCodeEmitter
#undef MachineInstr

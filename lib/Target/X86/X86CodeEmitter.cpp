//===-- X86/X86CodeEmitter.cpp - Convert X86 code to machine code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the pass that transforms the X86 machine instructions into
// relocatable machine code.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-emitter"
#include "X86InstrInfo.h"
#include "X86JITInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "X86Relocations.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(NumEmitted, "Number of machine instructions emitted");

namespace {
  class VISIBILITY_HIDDEN Emitter : public MachineFunctionPass {
    const X86InstrInfo  *II;
    const TargetData    *TD;
    TargetMachine       &TM;
    MachineCodeEmitter  &MCE;
    intptr_t PICBaseOffset;
    bool Is64BitMode;
    bool IsPIC;
  public:
    static char ID;
    explicit Emitter(TargetMachine &tm, MachineCodeEmitter &mce)
      : MachineFunctionPass((intptr_t)&ID), II(0), TD(0), TM(tm), 
      MCE(mce), PICBaseOffset(0), Is64BitMode(false),
      IsPIC(TM.getRelocationModel() == Reloc::PIC_) {}
    Emitter(TargetMachine &tm, MachineCodeEmitter &mce,
            const X86InstrInfo &ii, const TargetData &td, bool is64)
      : MachineFunctionPass((intptr_t)&ID), II(&ii), TD(&td), TM(tm), 
      MCE(mce), PICBaseOffset(0), Is64BitMode(is64),
      IsPIC(TM.getRelocationModel() == Reloc::PIC_) {}

    bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "X86 Machine Code Emitter";
    }

    void emitInstruction(const MachineInstr &MI,
                         const TargetInstrDesc *Desc);
    
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineModuleInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    void emitPCRelativeBlockAddress(MachineBasicBlock *MBB);
    void emitGlobalAddress(GlobalValue *GV, unsigned Reloc,
                           int Disp = 0, intptr_t PCAdj = 0,
                           bool NeedStub = false, bool IsLazy = false);
    void emitExternalSymbolAddress(const char *ES, unsigned Reloc);
    void emitConstPoolAddress(unsigned CPI, unsigned Reloc, int Disp = 0,
                              intptr_t PCAdj = 0);
    void emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                              intptr_t PCAdj = 0);

    void emitDisplacementField(const MachineOperand *RelocOp, int DispVal,
                               intptr_t PCAdj = 0);

    void emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeField);
    void emitSIBByte(unsigned SS, unsigned Index, unsigned Base);
    void emitConstant(uint64_t Val, unsigned Size);

    void emitMemModRMByte(const MachineInstr &MI,
                          unsigned Op, unsigned RegOpcodeField,
                          intptr_t PCAdj = 0);

    unsigned getX86RegNum(unsigned RegNo) const;
    bool isX86_64ExtendedReg(const MachineOperand &MO);
    unsigned determineREX(const MachineInstr &MI);

    bool gvNeedsLazyPtr(const GlobalValue *GV);
  };
  char Emitter::ID = 0;
}

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *llvm::createX86CodeEmitterPass(X86TargetMachine &TM,
                                             MachineCodeEmitter &MCE) {
  return new Emitter(TM, MCE);
}

bool Emitter::runOnMachineFunction(MachineFunction &MF) {
  assert((MF.getTarget().getRelocationModel() != Reloc::Default ||
          MF.getTarget().getRelocationModel() != Reloc::Static) &&
         "JIT relocation model must be set to static or default!");
  
  MCE.setModuleInfo(&getAnalysis<MachineModuleInfo>());
  
  II = ((X86TargetMachine&)TM).getInstrInfo();
  TD = ((X86TargetMachine&)TM).getTargetData();
  Is64BitMode = TM.getSubtarget<X86Subtarget>().is64Bit();
  
  do {
    MCE.startFunction(MF);
    for (MachineFunction::iterator MBB = MF.begin(), E = MF.end(); 
         MBB != E; ++MBB) {
      MCE.StartMachineBasicBlock(MBB);
      for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
           I != E; ++I) {
        const TargetInstrDesc &Desc = I->getDesc();
        emitInstruction(*I, &Desc);
        // MOVPC32r is basically a call plus a pop instruction.
        if (Desc.getOpcode() == X86::MOVPC32r)
          emitInstruction(*I, &II->get(X86::POP32r));
        NumEmitted++;  // Keep track of the # of mi's emitted
      }
    }
  } while (MCE.finishFunction(MF));

  return false;
}

/// emitPCRelativeBlockAddress - This method keeps track of the information
/// necessary to resolve the address of this block later and emits a dummy
/// value.
///
void Emitter::emitPCRelativeBlockAddress(MachineBasicBlock *MBB) {
  // Remember where this reference was and where it is to so we can
  // deal with it later.
  MCE.addRelocation(MachineRelocation::getBB(MCE.getCurrentPCOffset(),
                                             X86::reloc_pcrel_word, MBB));
  MCE.emitWordLE(0);
}

/// emitGlobalAddress - Emit the specified address to the code stream assuming
/// this is part of a "take the address of a global" instruction.
///
void Emitter::emitGlobalAddress(GlobalValue *GV, unsigned Reloc,
                                int Disp /* = 0 */, intptr_t PCAdj /* = 0 */,
                                bool NeedStub /* = false */,
                                bool isLazy /* = false */) {
  intptr_t RelocCST = 0;
  if (Reloc == X86::reloc_picrel_word)
    RelocCST = PICBaseOffset;
  else if (Reloc == X86::reloc_pcrel_word)
    RelocCST = PCAdj;
  MachineRelocation MR = isLazy 
    ? MachineRelocation::getGVLazyPtr(MCE.getCurrentPCOffset(), Reloc,
                                      GV, RelocCST, NeedStub)
    : MachineRelocation::getGV(MCE.getCurrentPCOffset(), Reloc,
                               GV, RelocCST, NeedStub);
  MCE.addRelocation(MR);
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitWordLE(0);
  MCE.emitWordLE(Disp); // The relocated value will be added to the displacement
}

/// emitExternalSymbolAddress - Arrange for the address of an external symbol to
/// be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitExternalSymbolAddress(const char *ES, unsigned Reloc) {
  intptr_t RelocCST = (Reloc == X86::reloc_picrel_word) ? PICBaseOffset : 0;
  MCE.addRelocation(MachineRelocation::getExtSym(MCE.getCurrentPCOffset(),
                                                 Reloc, ES, RelocCST));
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitWordLE(0);
  MCE.emitWordLE(0);
}

/// emitConstPoolAddress - Arrange for the address of an constant pool
/// to be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitConstPoolAddress(unsigned CPI, unsigned Reloc,
                                   int Disp /* = 0 */,
                                   intptr_t PCAdj /* = 0 */) {
  intptr_t RelocCST = 0;
  if (Reloc == X86::reloc_picrel_word)
    RelocCST = PICBaseOffset;
  else if (Reloc == X86::reloc_pcrel_word)
    RelocCST = PCAdj;
  MCE.addRelocation(MachineRelocation::getConstPool(MCE.getCurrentPCOffset(),
                                                    Reloc, CPI, RelocCST));
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitWordLE(0);
  MCE.emitWordLE(Disp); // The relocated value will be added to the displacement
}

/// emitJumpTableAddress - Arrange for the address of a jump table to
/// be emitted to the current location in the function, and allow it to be PC
/// relative.
void Emitter::emitJumpTableAddress(unsigned JTI, unsigned Reloc,
                                   intptr_t PCAdj /* = 0 */) {
  intptr_t RelocCST = 0;
  if (Reloc == X86::reloc_picrel_word)
    RelocCST = PICBaseOffset;
  else if (Reloc == X86::reloc_pcrel_word)
    RelocCST = PCAdj;
  MCE.addRelocation(MachineRelocation::getJumpTable(MCE.getCurrentPCOffset(),
                                                    Reloc, JTI, RelocCST));
  if (Reloc == X86::reloc_absolute_dword)
    MCE.emitWordLE(0);
  MCE.emitWordLE(0); // The relocated value will be added to the displacement
}

unsigned Emitter::getX86RegNum(unsigned RegNo) const {
  return ((const X86RegisterInfo&)II->getRegisterInfo()).getX86RegNum(RegNo);
}

inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                      unsigned RM) {
  assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
  return RM | (RegOpcode << 3) | (Mod << 6);
}

void Emitter::emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeFld){
  MCE.emitByte(ModRMByte(3, RegOpcodeFld, getX86RegNum(ModRMReg)));
}

void Emitter::emitSIBByte(unsigned SS, unsigned Index, unsigned Base) {
  // SIB byte is in the same format as the ModRMByte...
  MCE.emitByte(ModRMByte(SS, Index, Base));
}

void Emitter::emitConstant(uint64_t Val, unsigned Size) {
  // Output the constant in little endian byte order...
  for (unsigned i = 0; i != Size; ++i) {
    MCE.emitByte(Val & 255);
    Val >>= 8;
  }
}

/// isDisp8 - Return true if this signed displacement fits in a 8-bit 
/// sign-extended field. 
static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

bool Emitter::gvNeedsLazyPtr(const GlobalValue *GV) {
  return !Is64BitMode && 
    TM.getSubtarget<X86Subtarget>().GVRequiresExtraLoad(GV, TM, false);
}

void Emitter::emitDisplacementField(const MachineOperand *RelocOp,
                                    int DispVal, intptr_t PCAdj) {
  // If this is a simple integer displacement that doesn't require a relocation,
  // emit it now.
  if (!RelocOp) {
    emitConstant(DispVal, 4);
    return;
  }
  
  // Otherwise, this is something that requires a relocation.  Emit it as such
  // now.
  if (RelocOp->isGlobalAddress()) {
    // In 64-bit static small code model, we could potentially emit absolute.
    // But it's probably not beneficial.
    //  89 05 00 00 00 00     mov    %eax,0(%rip)  # PC-relative
    //  89 04 25 00 00 00 00  mov    %eax,0x0      # Absolute
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
      : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
    bool NeedStub = isa<Function>(RelocOp->getGlobal());
    bool isLazy = gvNeedsLazyPtr(RelocOp->getGlobal());
    emitGlobalAddress(RelocOp->getGlobal(), rt, RelocOp->getOffset(),
                      PCAdj, NeedStub, isLazy);
  } else if (RelocOp->isConstantPoolIndex()) {
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word : X86::reloc_picrel_word;
    emitConstPoolAddress(RelocOp->getIndex(), rt,
                         RelocOp->getOffset(), PCAdj);
  } else if (RelocOp->isJumpTableIndex()) {
    unsigned rt = Is64BitMode ? X86::reloc_pcrel_word : X86::reloc_picrel_word;
    emitJumpTableAddress(RelocOp->getIndex(), rt, PCAdj);
  } else {
    assert(0 && "Unknown value to relocate!");
  }
}

void Emitter::emitMemModRMByte(const MachineInstr &MI,
                               unsigned Op, unsigned RegOpcodeField,
                               intptr_t PCAdj) {
  const MachineOperand &Op3 = MI.getOperand(Op+3);
  int DispVal = 0;
  const MachineOperand *DispForReloc = 0;
  
  // Figure out what sort of displacement we have to handle here.
  if (Op3.isGlobalAddress()) {
    DispForReloc = &Op3;
  } else if (Op3.isConstantPoolIndex()) {
    if (Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal += MCE.getConstantPoolEntryAddress(Op3.getIndex());
      DispVal += Op3.getOffset();
    }
  } else if (Op3.isJumpTableIndex()) {
    if (Is64BitMode || IsPIC) {
      DispForReloc = &Op3;
    } else {
      DispVal += MCE.getJumpTableEntryAddress(Op3.getIndex());
    }
  } else {
    DispVal = Op3.getImm();
  }

  const MachineOperand &Base     = MI.getOperand(Op);
  const MachineOperand &Scale    = MI.getOperand(Op+1);
  const MachineOperand &IndexReg = MI.getOperand(Op+2);

  unsigned BaseReg = Base.getReg();

  // Is a SIB byte needed?
  if (IndexReg.getReg() == 0 &&
      (BaseReg == 0 || getX86RegNum(BaseReg) != N86::ESP)) {
    if (BaseReg == 0) {  // Just a displacement?
      // Emit special case [disp32] encoding
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      
      emitDisplacementField(DispForReloc, DispVal, PCAdj);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg);
      if (!DispForReloc && DispVal == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        MCE.emitByte(ModRMByte(0, RegOpcodeField, BaseRegNo));
      } else if (!DispForReloc && isDisp8(DispVal)) {
        // Emit the disp8 encoding... [REG+disp8]
        MCE.emitByte(ModRMByte(1, RegOpcodeField, BaseRegNo));
        emitConstant(DispVal, 1);
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        MCE.emitByte(ModRMByte(2, RegOpcodeField, BaseRegNo));
        emitDisplacementField(DispForReloc, DispVal, PCAdj);
      }
    }

  } else {  // We need a SIB byte, so start by outputting the ModR/M byte first
    assert(IndexReg.getReg() != X86::ESP &&
           IndexReg.getReg() != X86::RSP && "Cannot use ESP as index reg!");

    bool ForceDisp32 = false;
    bool ForceDisp8  = false;
    if (BaseReg == 0) {
      // If there is no base register, we emit the special case SIB byte with
      // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (DispForReloc) {
      // Emit the normal disp32 encoding.
      MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (DispVal == 0 && getX86RegNum(BaseReg) != N86::EBP) {
      // Emit no displacement ModR/M byte
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
    } else if (isDisp8(DispVal)) {
      // Emit the disp8 encoding...
      MCE.emitByte(ModRMByte(1, RegOpcodeField, 4));
      ForceDisp8 = true;           // Make sure to force 8 bit disp if Base=EBP
    } else {
      // Emit the normal disp32 encoding...
      MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
    }

    // Calculate what the SS field value should be...
    static const unsigned SSTable[] = { ~0, 0, 1, ~0, 2, ~0, ~0, ~0, 3 };
    unsigned SS = SSTable[Scale.getImm()];

    if (BaseReg == 0) {
      // Handle the SIB byte for the case where there is no base.  The
      // displacement has already been output.
      assert(IndexReg.getReg() && "Index register must be specified!");
      emitSIBByte(SS, getX86RegNum(IndexReg.getReg()), 5);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg);
      unsigned IndexRegNo;
      if (IndexReg.getReg())
        IndexRegNo = getX86RegNum(IndexReg.getReg());
      else
        IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
      emitSIBByte(SS, IndexRegNo, BaseRegNo);
    }

    // Do we need to output a displacement?
    if (ForceDisp8) {
      emitConstant(DispVal, 1);
    } else if (DispVal != 0 || ForceDisp32) {
      emitDisplacementField(DispForReloc, DispVal, PCAdj);
    }
  }
}

static unsigned sizeOfImm(const TargetInstrDesc *Desc) {
  switch (Desc->TSFlags & X86II::ImmMask) {
  case X86II::Imm8:   return 1;
  case X86II::Imm16:  return 2;
  case X86II::Imm32:  return 4;
  case X86II::Imm64:  return 8;
  default: assert(0 && "Immediate size not set!");
    return 0;
  }
}

/// isX86_64ExtendedReg - Is the MachineOperand a x86-64 extended register?
/// e.g. r8, xmm8, etc.
bool Emitter::isX86_64ExtendedReg(const MachineOperand &MO) {
  if (!MO.isRegister()) return false;
  switch (MO.getReg()) {
  default: break;
  case X86::R8:    case X86::R9:    case X86::R10:   case X86::R11:
  case X86::R12:   case X86::R13:   case X86::R14:   case X86::R15:
  case X86::R8D:   case X86::R9D:   case X86::R10D:  case X86::R11D:
  case X86::R12D:  case X86::R13D:  case X86::R14D:  case X86::R15D:
  case X86::R8W:   case X86::R9W:   case X86::R10W:  case X86::R11W:
  case X86::R12W:  case X86::R13W:  case X86::R14W:  case X86::R15W:
  case X86::R8B:   case X86::R9B:   case X86::R10B:  case X86::R11B:
  case X86::R12B:  case X86::R13B:  case X86::R14B:  case X86::R15B:
  case X86::XMM8:  case X86::XMM9:  case X86::XMM10: case X86::XMM11:
  case X86::XMM12: case X86::XMM13: case X86::XMM14: case X86::XMM15:
    return true;
  }
  return false;
}

inline static bool isX86_64NonExtLowByteReg(unsigned reg) {
  return (reg == X86::SPL || reg == X86::BPL ||
          reg == X86::SIL || reg == X86::DIL);
}

/// determineREX - Determine if the MachineInstr has to be encoded with a X86-64
/// REX prefix which specifies 1) 64-bit instructions, 2) non-default operand
/// size, and 3) use of X86-64 extended registers.
unsigned Emitter::determineREX(const MachineInstr &MI) {
  unsigned REX = 0;
  const TargetInstrDesc &Desc = MI.getDesc();

  // Pseudo instructions do not need REX prefix byte.
  if ((Desc.TSFlags & X86II::FormMask) == X86II::Pseudo)
    return 0;
  if (Desc.TSFlags & X86II::REX_W)
    REX |= 1 << 3;

  unsigned NumOps = Desc.getNumOperands();
  if (NumOps) {
    bool isTwoAddr = NumOps > 1 &&
      Desc.getOperandConstraint(1, TOI::TIED_TO) != -1;

    // If it accesses SPL, BPL, SIL, or DIL, then it requires a 0x40 REX prefix.
    unsigned i = isTwoAddr ? 1 : 0;
    for (unsigned e = NumOps; i != e; ++i) {
      const MachineOperand& MO = MI.getOperand(i);
      if (MO.isRegister()) {
        unsigned Reg = MO.getReg();
        if (isX86_64NonExtLowByteReg(Reg))
          REX |= 0x40;
      }
    }

    switch (Desc.TSFlags & X86II::FormMask) {
    case X86II::MRMInitReg:
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= (1 << 0) | (1 << 2);
      break;
    case X86II::MRMSrcReg: {
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= 1 << 2;
      i = isTwoAddr ? 2 : 1;
      for (unsigned e = NumOps; i != e; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (isX86_64ExtendedReg(MO))
          REX |= 1 << 0;
      }
      break;
    }
    case X86II::MRMSrcMem: {
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= 1 << 2;
      unsigned Bit = 0;
      i = isTwoAddr ? 2 : 1;
      for (; i != NumOps; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (MO.isRegister()) {
          if (isX86_64ExtendedReg(MO))
            REX |= 1 << Bit;
          Bit++;
        }
      }
      break;
    }
    case X86II::MRM0m: case X86II::MRM1m:
    case X86II::MRM2m: case X86II::MRM3m:
    case X86II::MRM4m: case X86II::MRM5m:
    case X86II::MRM6m: case X86II::MRM7m:
    case X86II::MRMDestMem: {
      unsigned e = isTwoAddr ? 5 : 4;
      i = isTwoAddr ? 1 : 0;
      if (NumOps > e && isX86_64ExtendedReg(MI.getOperand(e)))
        REX |= 1 << 2;
      unsigned Bit = 0;
      for (; i != e; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (MO.isRegister()) {
          if (isX86_64ExtendedReg(MO))
            REX |= 1 << Bit;
          Bit++;
        }
      }
      break;
    }
    default: {
      if (isX86_64ExtendedReg(MI.getOperand(0)))
        REX |= 1 << 0;
      i = isTwoAddr ? 2 : 1;
      for (unsigned e = NumOps; i != e; ++i) {
        const MachineOperand& MO = MI.getOperand(i);
        if (isX86_64ExtendedReg(MO))
          REX |= 1 << 2;
      }
      break;
    }
    }
  }
  return REX;
}

void Emitter::emitInstruction(const MachineInstr &MI,
                              const TargetInstrDesc *Desc) {
  unsigned Opcode = Desc->Opcode;

  // Emit the repeat opcode prefix as needed.
  if ((Desc->TSFlags & X86II::Op0Mask) == X86II::REP) MCE.emitByte(0xF3);

  // Emit the operand size opcode prefix as needed.
  if (Desc->TSFlags & X86II::OpSize) MCE.emitByte(0x66);

  // Emit the address size opcode prefix as needed.
  if (Desc->TSFlags & X86II::AdSize) MCE.emitByte(0x67);

  bool Need0FPrefix = false;
  switch (Desc->TSFlags & X86II::Op0Mask) {
  case X86II::TB:
    Need0FPrefix = true;   // Two-byte opcode prefix
    break;
  case X86II::T8:
    MCE.emitByte(0x0F);
    MCE.emitByte(0x38);
    break;
  case X86II::TA:
    MCE.emitByte(0x0F);
    MCE.emitByte(0x3A);
    break;
  case X86II::REP: break; // already handled.
  case X86II::XS:   // F3 0F
    MCE.emitByte(0xF3);
    Need0FPrefix = true;
    break;
  case X86II::XD:   // F2 0F
    MCE.emitByte(0xF2);
    Need0FPrefix = true;
    break;
  case X86II::D8: case X86II::D9: case X86II::DA: case X86II::DB:
  case X86II::DC: case X86II::DD: case X86II::DE: case X86II::DF:
    MCE.emitByte(0xD8+
                 (((Desc->TSFlags & X86II::Op0Mask)-X86II::D8)
                                   >> X86II::Op0Shift));
    break; // Two-byte opcode prefix
  default: assert(0 && "Invalid prefix!");
  case 0: break;  // No prefix!
  }

  if (Is64BitMode) {
    // REX prefix
    unsigned REX = determineREX(MI);
    if (REX)
      MCE.emitByte(0x40 | REX);
  }

  // 0x0F escape code must be emitted just before the opcode.
  if (Need0FPrefix)
    MCE.emitByte(0x0F);

  // If this is a two-address instruction, skip one of the register operands.
  unsigned NumOps = Desc->getNumOperands();
  unsigned CurOp = 0;
  if (NumOps > 1 && Desc->getOperandConstraint(1, TOI::TIED_TO) != -1)
    CurOp++;

  unsigned char BaseOpcode = II->getBaseOpcodeFor(Desc);
  switch (Desc->TSFlags & X86II::FormMask) {
  default: assert(0 && "Unknown FormMask value in X86 MachineCodeEmitter!");
  case X86II::Pseudo:
    // Remember the current PC offset, this is the PIC relocation
    // base address.
    switch (Opcode) {
    default: 
      assert(0 && "psuedo instructions should be removed before code emission");
    case TargetInstrInfo::INLINEASM:
      assert(0 && "JIT does not support inline asm!\n");
    case TargetInstrInfo::LABEL:
      MCE.emitLabel(MI.getOperand(0).getImm());
      break;
    case X86::IMPLICIT_DEF_GR8:
    case X86::IMPLICIT_DEF_GR16:
    case X86::IMPLICIT_DEF_GR32:
    case X86::IMPLICIT_DEF_GR64:
    case X86::IMPLICIT_DEF_FR32:
    case X86::IMPLICIT_DEF_FR64:
    case X86::IMPLICIT_DEF_VR64:
    case X86::IMPLICIT_DEF_VR128:
    case X86::FP_REG_KILL:
      break;
    case X86::MOVPC32r: {
      // This emits the "call" portion of this pseudo instruction.
      MCE.emitByte(BaseOpcode);
      emitConstant(0, sizeOfImm(Desc));
      // Remember PIC base.
      PICBaseOffset = MCE.getCurrentPCOffset();
      X86JITInfo *JTI = dynamic_cast<X86JITInfo*>(TM.getJITInfo());
      JTI->setPICBase(MCE.getCurrentPCValue());
      break;
    }
    }
    CurOp = NumOps;
    break;
  case X86II::RawFrm:
    MCE.emitByte(BaseOpcode);

    if (CurOp != NumOps) {
      const MachineOperand &MO = MI.getOperand(CurOp++);
      if (MO.isMachineBasicBlock()) {
        emitPCRelativeBlockAddress(MO.getMBB());
      } else if (MO.isGlobalAddress()) {
        bool NeedStub = (Is64BitMode && TM.getCodeModel() == CodeModel::Large)
          || Opcode == X86::TAILJMPd;
        emitGlobalAddress(MO.getGlobal(), X86::reloc_pcrel_word,
                          0, 0, NeedStub);
      } else if (MO.isExternalSymbol()) {
        emitExternalSymbolAddress(MO.getSymbolName(), X86::reloc_pcrel_word);
      } else if (MO.isImmediate()) {
        emitConstant(MO.getImm(), sizeOfImm(Desc));
      } else {
        assert(0 && "Unknown RawFrm operand!");
      }
    }
    break;

  case X86II::AddRegFrm:
    MCE.emitByte(BaseOpcode + getX86RegNum(MI.getOperand(CurOp++).getReg()));
    
    if (CurOp != NumOps) {
      const MachineOperand &MO1 = MI.getOperand(CurOp++);
      unsigned Size = sizeOfImm(Desc);
      if (MO1.isImmediate())
        emitConstant(MO1.getImm(), Size);
      else {
        unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
          : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
        if (Opcode == X86::MOV64ri)
          rt = X86::reloc_absolute_dword;  // FIXME: add X86II flag?
        if (MO1.isGlobalAddress()) {
          bool NeedStub = isa<Function>(MO1.getGlobal());
          bool isLazy = gvNeedsLazyPtr(MO1.getGlobal());
          emitGlobalAddress(MO1.getGlobal(), rt, MO1.getOffset(), 0,
                            NeedStub, isLazy);
        } else if (MO1.isExternalSymbol())
          emitExternalSymbolAddress(MO1.getSymbolName(), rt);
        else if (MO1.isConstantPoolIndex())
          emitConstPoolAddress(MO1.getIndex(), rt);
        else if (MO1.isJumpTableIndex())
          emitJumpTableAddress(MO1.getIndex(), rt);
      }
    }
    break;

  case X86II::MRMDestReg: {
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(CurOp).getReg(),
                     getX86RegNum(MI.getOperand(CurOp+1).getReg()));
    CurOp += 2;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), sizeOfImm(Desc));
    break;
  }
  case X86II::MRMDestMem: {
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, CurOp, getX86RegNum(MI.getOperand(CurOp+4).getReg()));
    CurOp += 5;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), sizeOfImm(Desc));
    break;
  }

  case X86II::MRMSrcReg:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(CurOp+1).getReg(),
                     getX86RegNum(MI.getOperand(CurOp).getReg()));
    CurOp += 2;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), sizeOfImm(Desc));
    break;

  case X86II::MRMSrcMem: {
    intptr_t PCAdj = (CurOp+5 != NumOps) ? sizeOfImm(Desc) : 0;

    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, CurOp+1, getX86RegNum(MI.getOperand(CurOp).getReg()),
                     PCAdj);
    CurOp += 5;
    if (CurOp != NumOps)
      emitConstant(MI.getOperand(CurOp++).getImm(), sizeOfImm(Desc));
    break;
  }

  case X86II::MRM0r: case X86II::MRM1r:
  case X86II::MRM2r: case X86II::MRM3r:
  case X86II::MRM4r: case X86II::MRM5r:
  case X86II::MRM6r: case X86II::MRM7r:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(CurOp++).getReg(),
                     (Desc->TSFlags & X86II::FormMask)-X86II::MRM0r);

    if (CurOp != NumOps) {
      const MachineOperand &MO1 = MI.getOperand(CurOp++);
      unsigned Size = sizeOfImm(Desc);
      if (MO1.isImmediate())
        emitConstant(MO1.getImm(), Size);
      else {
        unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
          : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
        if (Opcode == X86::MOV64ri32)
          rt = X86::reloc_absolute_word;  // FIXME: add X86II flag?
        if (MO1.isGlobalAddress()) {
          bool NeedStub = isa<Function>(MO1.getGlobal());
          bool isLazy = gvNeedsLazyPtr(MO1.getGlobal());
          emitGlobalAddress(MO1.getGlobal(), rt, MO1.getOffset(), 0,
                            NeedStub, isLazy);
        } else if (MO1.isExternalSymbol())
          emitExternalSymbolAddress(MO1.getSymbolName(), rt);
        else if (MO1.isConstantPoolIndex())
          emitConstPoolAddress(MO1.getIndex(), rt);
        else if (MO1.isJumpTableIndex())
          emitJumpTableAddress(MO1.getIndex(), rt);
      }
    }
    break;

  case X86II::MRM0m: case X86II::MRM1m:
  case X86II::MRM2m: case X86II::MRM3m:
  case X86II::MRM4m: case X86II::MRM5m:
  case X86II::MRM6m: case X86II::MRM7m: {
    intptr_t PCAdj = (CurOp+4 != NumOps) ?
      (MI.getOperand(CurOp+4).isImmediate() ? sizeOfImm(Desc) : 4) : 0;

    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, CurOp, (Desc->TSFlags & X86II::FormMask)-X86II::MRM0m,
                     PCAdj);
    CurOp += 4;

    if (CurOp != NumOps) {
      const MachineOperand &MO = MI.getOperand(CurOp++);
      unsigned Size = sizeOfImm(Desc);
      if (MO.isImmediate())
        emitConstant(MO.getImm(), Size);
      else {
        unsigned rt = Is64BitMode ? X86::reloc_pcrel_word
          : (IsPIC ? X86::reloc_picrel_word : X86::reloc_absolute_word);
        if (Opcode == X86::MOV64mi32)
          rt = X86::reloc_absolute_word;  // FIXME: add X86II flag?
        if (MO.isGlobalAddress()) {
          bool NeedStub = isa<Function>(MO.getGlobal());
          bool isLazy = gvNeedsLazyPtr(MO.getGlobal());
          emitGlobalAddress(MO.getGlobal(), rt, MO.getOffset(), 0,
                            NeedStub, isLazy);
        } else if (MO.isExternalSymbol())
          emitExternalSymbolAddress(MO.getSymbolName(), rt);
        else if (MO.isConstantPoolIndex())
          emitConstPoolAddress(MO.getIndex(), rt);
        else if (MO.isJumpTableIndex())
          emitJumpTableAddress(MO.getIndex(), rt);
      }
    }
    break;
  }

  case X86II::MRMInitReg:
    MCE.emitByte(BaseOpcode);
    // Duplicate register, used by things like MOV8r0 (aka xor reg,reg).
    emitRegModRMByte(MI.getOperand(CurOp).getReg(),
                     getX86RegNum(MI.getOperand(CurOp).getReg()));
    ++CurOp;
    break;
  }

  assert((Desc->isVariadic() || CurOp == NumOps) && "Unknown encoding!");
}

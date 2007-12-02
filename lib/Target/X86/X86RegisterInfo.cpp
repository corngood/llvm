//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.  This
// file is responsible for the frame pointer elimination optimization on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

namespace {
  cl::opt<bool>
  NoFusing("disable-spill-fusing",
           cl::desc("Disable fusing of spill code into instructions"));
  cl::opt<bool>
  PrintFailedFusing("print-failed-fuse-candidates",
                    cl::desc("Print instructions that the allocator wants to"
                             " fuse, but the X86 backend currently can't"),
                    cl::Hidden);
}

X86RegisterInfo::X86RegisterInfo(X86TargetMachine &tm,
                                 const TargetInstrInfo &tii)
  : X86GenRegisterInfo(X86::ADJCALLSTACKDOWN, X86::ADJCALLSTACKUP),
    TM(tm), TII(tii) {
  // Cache some information.
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  Is64Bit = Subtarget->is64Bit();
  StackAlign = TM.getFrameInfo()->getStackAlignment();
  if (Is64Bit) {
    SlotSize = 8;
    StackPtr = X86::RSP;
    FramePtr = X86::RBP;
  } else {
    SlotSize = 4;
    StackPtr = X86::ESP;
    FramePtr = X86::EBP;
  }

  SmallVector<unsigned,16> AmbEntries;
  static const unsigned OpTbl2Addr[][2] = {
    { X86::ADC32ri,     X86::ADC32mi },
    { X86::ADC32ri8,    X86::ADC32mi8 },
    { X86::ADC32rr,     X86::ADC32mr },
    { X86::ADC64ri32,   X86::ADC64mi32 },
    { X86::ADC64ri8,    X86::ADC64mi8 },
    { X86::ADC64rr,     X86::ADC64mr },
    { X86::ADD16ri,     X86::ADD16mi },
    { X86::ADD16ri8,    X86::ADD16mi8 },
    { X86::ADD16rr,     X86::ADD16mr },
    { X86::ADD32ri,     X86::ADD32mi },
    { X86::ADD32ri8,    X86::ADD32mi8 },
    { X86::ADD32rr,     X86::ADD32mr },
    { X86::ADD64ri32,   X86::ADD64mi32 },
    { X86::ADD64ri8,    X86::ADD64mi8 },
    { X86::ADD64rr,     X86::ADD64mr },
    { X86::ADD8ri,      X86::ADD8mi },
    { X86::ADD8rr,      X86::ADD8mr },
    { X86::AND16ri,     X86::AND16mi },
    { X86::AND16ri8,    X86::AND16mi8 },
    { X86::AND16rr,     X86::AND16mr },
    { X86::AND32ri,     X86::AND32mi },
    { X86::AND32ri8,    X86::AND32mi8 },
    { X86::AND32rr,     X86::AND32mr },
    { X86::AND64ri32,   X86::AND64mi32 },
    { X86::AND64ri8,    X86::AND64mi8 },
    { X86::AND64rr,     X86::AND64mr },
    { X86::AND8ri,      X86::AND8mi },
    { X86::AND8rr,      X86::AND8mr },
    { X86::DEC16r,      X86::DEC16m },
    { X86::DEC32r,      X86::DEC32m },
    { X86::DEC64_16r,   X86::DEC64_16m },
    { X86::DEC64_32r,   X86::DEC64_32m },
    { X86::DEC64r,      X86::DEC64m },
    { X86::DEC8r,       X86::DEC8m },
    { X86::INC16r,      X86::INC16m },
    { X86::INC32r,      X86::INC32m },
    { X86::INC64_16r,   X86::INC64_16m },
    { X86::INC64_32r,   X86::INC64_32m },
    { X86::INC64r,      X86::INC64m },
    { X86::INC8r,       X86::INC8m },
    { X86::NEG16r,      X86::NEG16m },
    { X86::NEG32r,      X86::NEG32m },
    { X86::NEG64r,      X86::NEG64m },
    { X86::NEG8r,       X86::NEG8m },
    { X86::NOT16r,      X86::NOT16m },
    { X86::NOT32r,      X86::NOT32m },
    { X86::NOT64r,      X86::NOT64m },
    { X86::NOT8r,       X86::NOT8m },
    { X86::OR16ri,      X86::OR16mi },
    { X86::OR16ri8,     X86::OR16mi8 },
    { X86::OR16rr,      X86::OR16mr },
    { X86::OR32ri,      X86::OR32mi },
    { X86::OR32ri8,     X86::OR32mi8 },
    { X86::OR32rr,      X86::OR32mr },
    { X86::OR64ri32,    X86::OR64mi32 },
    { X86::OR64ri8,     X86::OR64mi8 },
    { X86::OR64rr,      X86::OR64mr },
    { X86::OR8ri,       X86::OR8mi },
    { X86::OR8rr,       X86::OR8mr },
    { X86::ROL16r1,     X86::ROL16m1 },
    { X86::ROL16rCL,    X86::ROL16mCL },
    { X86::ROL16ri,     X86::ROL16mi },
    { X86::ROL32r1,     X86::ROL32m1 },
    { X86::ROL32rCL,    X86::ROL32mCL },
    { X86::ROL32ri,     X86::ROL32mi },
    { X86::ROL64r1,     X86::ROL64m1 },
    { X86::ROL64rCL,    X86::ROL64mCL },
    { X86::ROL64ri,     X86::ROL64mi },
    { X86::ROL8r1,      X86::ROL8m1 },
    { X86::ROL8rCL,     X86::ROL8mCL },
    { X86::ROL8ri,      X86::ROL8mi },
    { X86::ROR16r1,     X86::ROR16m1 },
    { X86::ROR16rCL,    X86::ROR16mCL },
    { X86::ROR16ri,     X86::ROR16mi },
    { X86::ROR32r1,     X86::ROR32m1 },
    { X86::ROR32rCL,    X86::ROR32mCL },
    { X86::ROR32ri,     X86::ROR32mi },
    { X86::ROR64r1,     X86::ROR64m1 },
    { X86::ROR64rCL,    X86::ROR64mCL },
    { X86::ROR64ri,     X86::ROR64mi },
    { X86::ROR8r1,      X86::ROR8m1 },
    { X86::ROR8rCL,     X86::ROR8mCL },
    { X86::ROR8ri,      X86::ROR8mi },
    { X86::SAR16r1,     X86::SAR16m1 },
    { X86::SAR16rCL,    X86::SAR16mCL },
    { X86::SAR16ri,     X86::SAR16mi },
    { X86::SAR32r1,     X86::SAR32m1 },
    { X86::SAR32rCL,    X86::SAR32mCL },
    { X86::SAR32ri,     X86::SAR32mi },
    { X86::SAR64r1,     X86::SAR64m1 },
    { X86::SAR64rCL,    X86::SAR64mCL },
    { X86::SAR64ri,     X86::SAR64mi },
    { X86::SAR8r1,      X86::SAR8m1 },
    { X86::SAR8rCL,     X86::SAR8mCL },
    { X86::SAR8ri,      X86::SAR8mi },
    { X86::SBB32ri,     X86::SBB32mi },
    { X86::SBB32ri8,    X86::SBB32mi8 },
    { X86::SBB32rr,     X86::SBB32mr },
    { X86::SBB64ri32,   X86::SBB64mi32 },
    { X86::SBB64ri8,    X86::SBB64mi8 },
    { X86::SBB64rr,     X86::SBB64mr },
    { X86::SHL16r1,     X86::SHL16m1 },
    { X86::SHL16rCL,    X86::SHL16mCL },
    { X86::SHL16ri,     X86::SHL16mi },
    { X86::SHL32r1,     X86::SHL32m1 },
    { X86::SHL32rCL,    X86::SHL32mCL },
    { X86::SHL32ri,     X86::SHL32mi },
    { X86::SHL64r1,     X86::SHL64m1 },
    { X86::SHL64rCL,    X86::SHL64mCL },
    { X86::SHL64ri,     X86::SHL64mi },
    { X86::SHL8r1,      X86::SHL8m1 },
    { X86::SHL8rCL,     X86::SHL8mCL },
    { X86::SHL8ri,      X86::SHL8mi },
    { X86::SHLD16rrCL,  X86::SHLD16mrCL },
    { X86::SHLD16rri8,  X86::SHLD16mri8 },
    { X86::SHLD32rrCL,  X86::SHLD32mrCL },
    { X86::SHLD32rri8,  X86::SHLD32mri8 },
    { X86::SHLD64rrCL,  X86::SHLD64mrCL },
    { X86::SHLD64rri8,  X86::SHLD64mri8 },
    { X86::SHR16r1,     X86::SHR16m1 },
    { X86::SHR16rCL,    X86::SHR16mCL },
    { X86::SHR16ri,     X86::SHR16mi },
    { X86::SHR32r1,     X86::SHR32m1 },
    { X86::SHR32rCL,    X86::SHR32mCL },
    { X86::SHR32ri,     X86::SHR32mi },
    { X86::SHR64r1,     X86::SHR64m1 },
    { X86::SHR64rCL,    X86::SHR64mCL },
    { X86::SHR64ri,     X86::SHR64mi },
    { X86::SHR8r1,      X86::SHR8m1 },
    { X86::SHR8rCL,     X86::SHR8mCL },
    { X86::SHR8ri,      X86::SHR8mi },
    { X86::SHRD16rrCL,  X86::SHRD16mrCL },
    { X86::SHRD16rri8,  X86::SHRD16mri8 },
    { X86::SHRD32rrCL,  X86::SHRD32mrCL },
    { X86::SHRD32rri8,  X86::SHRD32mri8 },
    { X86::SHRD64rrCL,  X86::SHRD64mrCL },
    { X86::SHRD64rri8,  X86::SHRD64mri8 },
    { X86::SUB16ri,     X86::SUB16mi },
    { X86::SUB16ri8,    X86::SUB16mi8 },
    { X86::SUB16rr,     X86::SUB16mr },
    { X86::SUB32ri,     X86::SUB32mi },
    { X86::SUB32ri8,    X86::SUB32mi8 },
    { X86::SUB32rr,     X86::SUB32mr },
    { X86::SUB64ri32,   X86::SUB64mi32 },
    { X86::SUB64ri8,    X86::SUB64mi8 },
    { X86::SUB64rr,     X86::SUB64mr },
    { X86::SUB8ri,      X86::SUB8mi },
    { X86::SUB8rr,      X86::SUB8mr },
    { X86::XOR16ri,     X86::XOR16mi },
    { X86::XOR16ri8,    X86::XOR16mi8 },
    { X86::XOR16rr,     X86::XOR16mr },
    { X86::XOR32ri,     X86::XOR32mi },
    { X86::XOR32ri8,    X86::XOR32mi8 },
    { X86::XOR32rr,     X86::XOR32mr },
    { X86::XOR64ri32,   X86::XOR64mi32 },
    { X86::XOR64ri8,    X86::XOR64mi8 },
    { X86::XOR64rr,     X86::XOR64mr },
    { X86::XOR8ri,      X86::XOR8mi },
    { X86::XOR8rr,      X86::XOR8mr }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl2Addr); i != e; ++i) {
    unsigned RegOp = OpTbl2Addr[i][0];
    unsigned MemOp = OpTbl2Addr[i][1];
    if (!RegOp2MemOpTable2Addr.insert(std::make_pair((unsigned*)RegOp, MemOp)))
      assert(false && "Duplicated entries?");
    unsigned AuxInfo = 0 | (1 << 4) | (1 << 5); // Index 0,folded load and store
    if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                               std::make_pair(RegOp, AuxInfo))))
      AmbEntries.push_back(MemOp);
  }

  // If the third value is 1, then it's folding either a load or a store.
  static const unsigned OpTbl0[][3] = {
    { X86::CALL32r,     X86::CALL32m, 1 },
    { X86::CALL64r,     X86::CALL64m, 1 },
    { X86::CMP16ri,     X86::CMP16mi, 1 },
    { X86::CMP16ri8,    X86::CMP16mi8, 1 },
    { X86::CMP32ri,     X86::CMP32mi, 1 },
    { X86::CMP32ri8,    X86::CMP32mi8, 1 },
    { X86::CMP64ri32,   X86::CMP64mi32, 1 },
    { X86::CMP64ri8,    X86::CMP64mi8, 1 },
    { X86::CMP8ri,      X86::CMP8mi, 1 },
    { X86::DIV16r,      X86::DIV16m, 1 },
    { X86::DIV32r,      X86::DIV32m, 1 },
    { X86::DIV64r,      X86::DIV64m, 1 },
    { X86::DIV8r,       X86::DIV8m, 1 },
    { X86::FsMOVAPDrr,  X86::MOVSDmr, 0 },
    { X86::FsMOVAPSrr,  X86::MOVSSmr, 0 },
    { X86::IDIV16r,     X86::IDIV16m, 1 },
    { X86::IDIV32r,     X86::IDIV32m, 1 },
    { X86::IDIV64r,     X86::IDIV64m, 1 },
    { X86::IDIV8r,      X86::IDIV8m, 1 },
    { X86::IMUL16r,     X86::IMUL16m, 1 },
    { X86::IMUL32r,     X86::IMUL32m, 1 },
    { X86::IMUL64r,     X86::IMUL64m, 1 },
    { X86::IMUL8r,      X86::IMUL8m, 1 },
    { X86::JMP32r,      X86::JMP32m, 1 },
    { X86::JMP64r,      X86::JMP64m, 1 },
    { X86::MOV16ri,     X86::MOV16mi, 0 },
    { X86::MOV16rr,     X86::MOV16mr, 0 },
    { X86::MOV16to16_,  X86::MOV16_mr, 0 },
    { X86::MOV32ri,     X86::MOV32mi, 0 },
    { X86::MOV32rr,     X86::MOV32mr, 0 },
    { X86::MOV32to32_,  X86::MOV32_mr, 0 },
    { X86::MOV64ri32,   X86::MOV64mi32, 0 },
    { X86::MOV64rr,     X86::MOV64mr, 0 },
    { X86::MOV8ri,      X86::MOV8mi, 0 },
    { X86::MOV8rr,      X86::MOV8mr, 0 },
    { X86::MOVAPDrr,    X86::MOVAPDmr, 0 },
    { X86::MOVAPSrr,    X86::MOVAPSmr, 0 },
    { X86::MOVPDI2DIrr, X86::MOVPDI2DImr, 0 },
    { X86::MOVPQIto64rr,X86::MOVPQIto64mr, 0 },
    { X86::MOVPS2SSrr,  X86::MOVPS2SSmr, 0 },
    { X86::MOVSDrr,     X86::MOVSDmr, 0 },
    { X86::MOVSDto64rr, X86::MOVSDto64mr, 0 },
    { X86::MOVSS2DIrr,  X86::MOVSS2DImr, 0 },
    { X86::MOVSSrr,     X86::MOVSSmr, 0 },
    { X86::MOVUPDrr,    X86::MOVUPDmr, 0 },
    { X86::MOVUPSrr,    X86::MOVUPSmr, 0 },
    { X86::MUL16r,      X86::MUL16m, 1 },
    { X86::MUL32r,      X86::MUL32m, 1 },
    { X86::MUL64r,      X86::MUL64m, 1 },
    { X86::MUL8r,       X86::MUL8m, 1 },
    { X86::SETAEr,      X86::SETAEm, 0 },
    { X86::SETAr,       X86::SETAm, 0 },
    { X86::SETBEr,      X86::SETBEm, 0 },
    { X86::SETBr,       X86::SETBm, 0 },
    { X86::SETEr,       X86::SETEm, 0 },
    { X86::SETGEr,      X86::SETGEm, 0 },
    { X86::SETGr,       X86::SETGm, 0 },
    { X86::SETLEr,      X86::SETLEm, 0 },
    { X86::SETLr,       X86::SETLm, 0 },
    { X86::SETNEr,      X86::SETNEm, 0 },
    { X86::SETNPr,      X86::SETNPm, 0 },
    { X86::SETNSr,      X86::SETNSm, 0 },
    { X86::SETPr,       X86::SETPm, 0 },
    { X86::SETSr,       X86::SETSm, 0 },
    { X86::TAILJMPr,    X86::TAILJMPm, 1 },
    { X86::TEST16ri,    X86::TEST16mi, 1 },
    { X86::TEST32ri,    X86::TEST32mi, 1 },
    { X86::TEST64ri32,  X86::TEST64mi32, 1 },
    { X86::TEST8ri,     X86::TEST8mi, 1 },
    { X86::XCHG16rr,    X86::XCHG16mr, 0 },
    { X86::XCHG32rr,    X86::XCHG32mr, 0 },
    { X86::XCHG64rr,    X86::XCHG64mr, 0 },
    { X86::XCHG8rr,     X86::XCHG8mr, 0 }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl0); i != e; ++i) {
    unsigned RegOp = OpTbl0[i][0];
    unsigned MemOp = OpTbl0[i][1];
    if (!RegOp2MemOpTable0.insert(std::make_pair((unsigned*)RegOp, MemOp)))
      assert(false && "Duplicated entries?");
    unsigned FoldedLoad = OpTbl0[i][2];
    // Index 0, folded load or store.
    unsigned AuxInfo = 0 | (FoldedLoad << 4) | ((FoldedLoad^1) << 5);
    if (RegOp != X86::FsMOVAPDrr && RegOp != X86::FsMOVAPSrr)
      if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                               std::make_pair(RegOp, AuxInfo))))
        AmbEntries.push_back(MemOp);
  }

  static const unsigned OpTbl1[][2] = {
    { X86::CMP16rr,         X86::CMP16rm },
    { X86::CMP32rr,         X86::CMP32rm },
    { X86::CMP64rr,         X86::CMP64rm },
    { X86::CMP8rr,          X86::CMP8rm },
    { X86::CVTSD2SSrr,      X86::CVTSD2SSrm },
    { X86::CVTSI2SD64rr,    X86::CVTSI2SD64rm },
    { X86::CVTSI2SDrr,      X86::CVTSI2SDrm },
    { X86::CVTSI2SS64rr,    X86::CVTSI2SS64rm },
    { X86::CVTSI2SSrr,      X86::CVTSI2SSrm },
    { X86::CVTSS2SDrr,      X86::CVTSS2SDrm },
    { X86::CVTTSD2SI64rr,   X86::CVTTSD2SI64rm },
    { X86::CVTTSD2SIrr,     X86::CVTTSD2SIrm },
    { X86::CVTTSS2SI64rr,   X86::CVTTSS2SI64rm },
    { X86::CVTTSS2SIrr,     X86::CVTTSS2SIrm },
    { X86::FsMOVAPDrr,      X86::MOVSDrm },
    { X86::FsMOVAPSrr,      X86::MOVSSrm },
    { X86::IMUL16rri,       X86::IMUL16rmi },
    { X86::IMUL16rri8,      X86::IMUL16rmi8 },
    { X86::IMUL32rri,       X86::IMUL32rmi },
    { X86::IMUL32rri8,      X86::IMUL32rmi8 },
    { X86::IMUL64rri32,     X86::IMUL64rmi32 },
    { X86::IMUL64rri8,      X86::IMUL64rmi8 },
    { X86::Int_CMPSDrr,     X86::Int_CMPSDrm },
    { X86::Int_CMPSSrr,     X86::Int_CMPSSrm },
    { X86::Int_COMISDrr,    X86::Int_COMISDrm },
    { X86::Int_COMISSrr,    X86::Int_COMISSrm },
    { X86::Int_CVTDQ2PDrr,  X86::Int_CVTDQ2PDrm },
    { X86::Int_CVTDQ2PSrr,  X86::Int_CVTDQ2PSrm },
    { X86::Int_CVTPD2DQrr,  X86::Int_CVTPD2DQrm },
    { X86::Int_CVTPD2PSrr,  X86::Int_CVTPD2PSrm },
    { X86::Int_CVTPS2DQrr,  X86::Int_CVTPS2DQrm },
    { X86::Int_CVTPS2PDrr,  X86::Int_CVTPS2PDrm },
    { X86::Int_CVTSD2SI64rr,X86::Int_CVTSD2SI64rm },
    { X86::Int_CVTSD2SIrr,  X86::Int_CVTSD2SIrm },
    { X86::Int_CVTSD2SSrr,  X86::Int_CVTSD2SSrm },
    { X86::Int_CVTSI2SD64rr,X86::Int_CVTSI2SD64rm },
    { X86::Int_CVTSI2SDrr,  X86::Int_CVTSI2SDrm },
    { X86::Int_CVTSI2SS64rr,X86::Int_CVTSI2SS64rm },
    { X86::Int_CVTSI2SSrr,  X86::Int_CVTSI2SSrm },
    { X86::Int_CVTSS2SDrr,  X86::Int_CVTSS2SDrm },
    { X86::Int_CVTSS2SI64rr,X86::Int_CVTSS2SI64rm },
    { X86::Int_CVTSS2SIrr,  X86::Int_CVTSS2SIrm },
    { X86::Int_CVTTPD2DQrr, X86::Int_CVTTPD2DQrm },
    { X86::Int_CVTTPS2DQrr, X86::Int_CVTTPS2DQrm },
    { X86::Int_CVTTSD2SI64rr,X86::Int_CVTTSD2SI64rm },
    { X86::Int_CVTTSD2SIrr, X86::Int_CVTTSD2SIrm },
    { X86::Int_CVTTSS2SI64rr,X86::Int_CVTTSS2SI64rm },
    { X86::Int_CVTTSS2SIrr, X86::Int_CVTTSS2SIrm },
    { X86::Int_UCOMISDrr,   X86::Int_UCOMISDrm },
    { X86::Int_UCOMISSrr,   X86::Int_UCOMISSrm },
    { X86::MOV16rr,         X86::MOV16rm },
    { X86::MOV16to16_,      X86::MOV16_rm },
    { X86::MOV32rr,         X86::MOV32rm },
    { X86::MOV32to32_,      X86::MOV32_rm },
    { X86::MOV64rr,         X86::MOV64rm },
    { X86::MOV64toPQIrr,    X86::MOV64toPQIrm },
    { X86::MOV64toSDrr,     X86::MOV64toSDrm },
    { X86::MOV8rr,          X86::MOV8rm },
    { X86::MOVAPDrr,        X86::MOVAPDrm },
    { X86::MOVAPSrr,        X86::MOVAPSrm },
    { X86::MOVDDUPrr,       X86::MOVDDUPrm },
    { X86::MOVDI2PDIrr,     X86::MOVDI2PDIrm },
    { X86::MOVDI2SSrr,      X86::MOVDI2SSrm },
    { X86::MOVSD2PDrr,      X86::MOVSD2PDrm },
    { X86::MOVSDrr,         X86::MOVSDrm },
    { X86::MOVSHDUPrr,      X86::MOVSHDUPrm },
    { X86::MOVSLDUPrr,      X86::MOVSLDUPrm },
    { X86::MOVSS2PSrr,      X86::MOVSS2PSrm },
    { X86::MOVSSrr,         X86::MOVSSrm },
    { X86::MOVSX16rr8,      X86::MOVSX16rm8 },
    { X86::MOVSX32rr16,     X86::MOVSX32rm16 },
    { X86::MOVSX32rr8,      X86::MOVSX32rm8 },
    { X86::MOVSX64rr16,     X86::MOVSX64rm16 },
    { X86::MOVSX64rr32,     X86::MOVSX64rm32 },
    { X86::MOVSX64rr8,      X86::MOVSX64rm8 },
    { X86::MOVUPDrr,        X86::MOVUPDrm },
    { X86::MOVUPSrr,        X86::MOVUPSrm },
    { X86::MOVZX16rr8,      X86::MOVZX16rm8 },
    { X86::MOVZX32rr16,     X86::MOVZX32rm16 },
    { X86::MOVZX32rr8,      X86::MOVZX32rm8 },
    { X86::MOVZX64rr16,     X86::MOVZX64rm16 },
    { X86::MOVZX64rr8,      X86::MOVZX64rm8 },
    { X86::PSHUFDri,        X86::PSHUFDmi },
    { X86::PSHUFHWri,       X86::PSHUFHWmi },
    { X86::PSHUFLWri,       X86::PSHUFLWmi },
    { X86::PsMOVZX64rr32,   X86::PsMOVZX64rm32 },
    { X86::RCPPSr,          X86::RCPPSm },
    { X86::RCPPSr_Int,      X86::RCPPSm_Int },
    { X86::RSQRTPSr,        X86::RSQRTPSm },
    { X86::RSQRTPSr_Int,    X86::RSQRTPSm_Int },
    { X86::RSQRTSSr,        X86::RSQRTSSm },
    { X86::RSQRTSSr_Int,    X86::RSQRTSSm_Int },
    { X86::SQRTPDr,         X86::SQRTPDm },
    { X86::SQRTPDr_Int,     X86::SQRTPDm_Int },
    { X86::SQRTPSr,         X86::SQRTPSm },
    { X86::SQRTPSr_Int,     X86::SQRTPSm_Int },
    { X86::SQRTSDr,         X86::SQRTSDm },
    { X86::SQRTSDr_Int,     X86::SQRTSDm_Int },
    { X86::SQRTSSr,         X86::SQRTSSm },
    { X86::SQRTSSr_Int,     X86::SQRTSSm_Int },
    { X86::TEST16rr,        X86::TEST16rm },
    { X86::TEST32rr,        X86::TEST32rm },
    { X86::TEST64rr,        X86::TEST64rm },
    { X86::TEST8rr,         X86::TEST8rm },
    // FIXME: TEST*rr EAX,EAX ---> CMP [mem], 0
    { X86::UCOMISDrr,       X86::UCOMISDrm },
    { X86::UCOMISSrr,       X86::UCOMISSrm },
    { X86::XCHG16rr,        X86::XCHG16rm },
    { X86::XCHG32rr,        X86::XCHG32rm },
    { X86::XCHG64rr,        X86::XCHG64rm },
    { X86::XCHG8rr,         X86::XCHG8rm }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl1); i != e; ++i) {
    unsigned RegOp = OpTbl1[i][0];
    unsigned MemOp = OpTbl1[i][1];
    if (!RegOp2MemOpTable1.insert(std::make_pair((unsigned*)RegOp, MemOp)))
      assert(false && "Duplicated entries?");
    unsigned AuxInfo = 1 | (1 << 4); // Index 1, folded load
    if (RegOp != X86::FsMOVAPDrr && RegOp != X86::FsMOVAPSrr)
      if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                               std::make_pair(RegOp, AuxInfo))))
        AmbEntries.push_back(MemOp);
  }

  static const unsigned OpTbl2[][2] = {
    { X86::ADC32rr,         X86::ADC32rm },
    { X86::ADC64rr,         X86::ADC64rm },
    { X86::ADD16rr,         X86::ADD16rm },
    { X86::ADD32rr,         X86::ADD32rm },
    { X86::ADD64rr,         X86::ADD64rm },
    { X86::ADD8rr,          X86::ADD8rm },
    { X86::ADDPDrr,         X86::ADDPDrm },
    { X86::ADDPSrr,         X86::ADDPSrm },
    { X86::ADDSDrr,         X86::ADDSDrm },
    { X86::ADDSSrr,         X86::ADDSSrm },
    { X86::ADDSUBPDrr,      X86::ADDSUBPDrm },
    { X86::ADDSUBPSrr,      X86::ADDSUBPSrm },
    { X86::AND16rr,         X86::AND16rm },
    { X86::AND32rr,         X86::AND32rm },
    { X86::AND64rr,         X86::AND64rm },
    { X86::AND8rr,          X86::AND8rm },
    { X86::ANDNPDrr,        X86::ANDNPDrm },
    { X86::ANDNPSrr,        X86::ANDNPSrm },
    { X86::ANDPDrr,         X86::ANDPDrm },
    { X86::ANDPSrr,         X86::ANDPSrm },
    { X86::CMOVA16rr,       X86::CMOVA16rm },
    { X86::CMOVA32rr,       X86::CMOVA32rm },
    { X86::CMOVA64rr,       X86::CMOVA64rm },
    { X86::CMOVAE16rr,      X86::CMOVAE16rm },
    { X86::CMOVAE32rr,      X86::CMOVAE32rm },
    { X86::CMOVAE64rr,      X86::CMOVAE64rm },
    { X86::CMOVB16rr,       X86::CMOVB16rm },
    { X86::CMOVB32rr,       X86::CMOVB32rm },
    { X86::CMOVB64rr,       X86::CMOVB64rm },
    { X86::CMOVBE16rr,      X86::CMOVBE16rm },
    { X86::CMOVBE32rr,      X86::CMOVBE32rm },
    { X86::CMOVBE64rr,      X86::CMOVBE64rm },
    { X86::CMOVE16rr,       X86::CMOVE16rm },
    { X86::CMOVE32rr,       X86::CMOVE32rm },
    { X86::CMOVE64rr,       X86::CMOVE64rm },
    { X86::CMOVG16rr,       X86::CMOVG16rm },
    { X86::CMOVG32rr,       X86::CMOVG32rm },
    { X86::CMOVG64rr,       X86::CMOVG64rm },
    { X86::CMOVGE16rr,      X86::CMOVGE16rm },
    { X86::CMOVGE32rr,      X86::CMOVGE32rm },
    { X86::CMOVGE64rr,      X86::CMOVGE64rm },
    { X86::CMOVL16rr,       X86::CMOVL16rm },
    { X86::CMOVL32rr,       X86::CMOVL32rm },
    { X86::CMOVL64rr,       X86::CMOVL64rm },
    { X86::CMOVLE16rr,      X86::CMOVLE16rm },
    { X86::CMOVLE32rr,      X86::CMOVLE32rm },
    { X86::CMOVLE64rr,      X86::CMOVLE64rm },
    { X86::CMOVNE16rr,      X86::CMOVNE16rm },
    { X86::CMOVNE32rr,      X86::CMOVNE32rm },
    { X86::CMOVNE64rr,      X86::CMOVNE64rm },
    { X86::CMOVNP16rr,      X86::CMOVNP16rm },
    { X86::CMOVNP32rr,      X86::CMOVNP32rm },
    { X86::CMOVNP64rr,      X86::CMOVNP64rm },
    { X86::CMOVNS16rr,      X86::CMOVNS16rm },
    { X86::CMOVNS32rr,      X86::CMOVNS32rm },
    { X86::CMOVNS64rr,      X86::CMOVNS64rm },
    { X86::CMOVP16rr,       X86::CMOVP16rm },
    { X86::CMOVP32rr,       X86::CMOVP32rm },
    { X86::CMOVP64rr,       X86::CMOVP64rm },
    { X86::CMOVS16rr,       X86::CMOVS16rm },
    { X86::CMOVS32rr,       X86::CMOVS32rm },
    { X86::CMOVS64rr,       X86::CMOVS64rm },
    { X86::CMPPDrri,        X86::CMPPDrmi },
    { X86::CMPPSrri,        X86::CMPPSrmi },
    { X86::CMPSDrr,         X86::CMPSDrm },
    { X86::CMPSSrr,         X86::CMPSSrm },
    { X86::DIVPDrr,         X86::DIVPDrm },
    { X86::DIVPSrr,         X86::DIVPSrm },
    { X86::DIVSDrr,         X86::DIVSDrm },
    { X86::DIVSSrr,         X86::DIVSSrm },
    { X86::HADDPDrr,        X86::HADDPDrm },
    { X86::HADDPSrr,        X86::HADDPSrm },
    { X86::HSUBPDrr,        X86::HSUBPDrm },
    { X86::HSUBPSrr,        X86::HSUBPSrm },
    { X86::IMUL16rr,        X86::IMUL16rm },
    { X86::IMUL32rr,        X86::IMUL32rm },
    { X86::IMUL64rr,        X86::IMUL64rm },
    { X86::MAXPDrr,         X86::MAXPDrm },
    { X86::MAXPDrr_Int,     X86::MAXPDrm_Int },
    { X86::MAXPSrr,         X86::MAXPSrm },
    { X86::MAXPSrr_Int,     X86::MAXPSrm_Int },
    { X86::MAXSDrr,         X86::MAXSDrm },
    { X86::MAXSDrr_Int,     X86::MAXSDrm_Int },
    { X86::MAXSSrr,         X86::MAXSSrm },
    { X86::MAXSSrr_Int,     X86::MAXSSrm_Int },
    { X86::MINPDrr,         X86::MINPDrm },
    { X86::MINPDrr_Int,     X86::MINPDrm_Int },
    { X86::MINPSrr,         X86::MINPSrm },
    { X86::MINPSrr_Int,     X86::MINPSrm_Int },
    { X86::MINSDrr,         X86::MINSDrm },
    { X86::MINSDrr_Int,     X86::MINSDrm_Int },
    { X86::MINSSrr,         X86::MINSSrm },
    { X86::MINSSrr_Int,     X86::MINSSrm_Int },
    { X86::MULPDrr,         X86::MULPDrm },
    { X86::MULPSrr,         X86::MULPSrm },
    { X86::MULSDrr,         X86::MULSDrm },
    { X86::MULSSrr,         X86::MULSSrm },
    { X86::OR16rr,          X86::OR16rm },
    { X86::OR32rr,          X86::OR32rm },
    { X86::OR64rr,          X86::OR64rm },
    { X86::OR8rr,           X86::OR8rm },
    { X86::ORPDrr,          X86::ORPDrm },
    { X86::ORPSrr,          X86::ORPSrm },
    { X86::PACKSSDWrr,      X86::PACKSSDWrm },
    { X86::PACKSSWBrr,      X86::PACKSSWBrm },
    { X86::PACKUSWBrr,      X86::PACKUSWBrm },
    { X86::PADDBrr,         X86::PADDBrm },
    { X86::PADDDrr,         X86::PADDDrm },
    { X86::PADDQrr,         X86::PADDQrm },
    { X86::PADDSBrr,        X86::PADDSBrm },
    { X86::PADDSWrr,        X86::PADDSWrm },
    { X86::PADDWrr,         X86::PADDWrm },
    { X86::PANDNrr,         X86::PANDNrm },
    { X86::PANDrr,          X86::PANDrm },
    { X86::PAVGBrr,         X86::PAVGBrm },
    { X86::PAVGWrr,         X86::PAVGWrm },
    { X86::PCMPEQBrr,       X86::PCMPEQBrm },
    { X86::PCMPEQDrr,       X86::PCMPEQDrm },
    { X86::PCMPEQWrr,       X86::PCMPEQWrm },
    { X86::PCMPGTBrr,       X86::PCMPGTBrm },
    { X86::PCMPGTDrr,       X86::PCMPGTDrm },
    { X86::PCMPGTWrr,       X86::PCMPGTWrm },
    { X86::PINSRWrri,       X86::PINSRWrmi },
    { X86::PMADDWDrr,       X86::PMADDWDrm },
    { X86::PMAXSWrr,        X86::PMAXSWrm },
    { X86::PMAXUBrr,        X86::PMAXUBrm },
    { X86::PMINSWrr,        X86::PMINSWrm },
    { X86::PMINUBrr,        X86::PMINUBrm },
    { X86::PMULHUWrr,       X86::PMULHUWrm },
    { X86::PMULHWrr,        X86::PMULHWrm },
    { X86::PMULLWrr,        X86::PMULLWrm },
    { X86::PMULUDQrr,       X86::PMULUDQrm },
    { X86::PORrr,           X86::PORrm },
    { X86::PSADBWrr,        X86::PSADBWrm },
    { X86::PSLLDrr,         X86::PSLLDrm },
    { X86::PSLLQrr,         X86::PSLLQrm },
    { X86::PSLLWrr,         X86::PSLLWrm },
    { X86::PSRADrr,         X86::PSRADrm },
    { X86::PSRAWrr,         X86::PSRAWrm },
    { X86::PSRLDrr,         X86::PSRLDrm },
    { X86::PSRLQrr,         X86::PSRLQrm },
    { X86::PSRLWrr,         X86::PSRLWrm },
    { X86::PSUBBrr,         X86::PSUBBrm },
    { X86::PSUBDrr,         X86::PSUBDrm },
    { X86::PSUBSBrr,        X86::PSUBSBrm },
    { X86::PSUBSWrr,        X86::PSUBSWrm },
    { X86::PSUBWrr,         X86::PSUBWrm },
    { X86::PUNPCKHBWrr,     X86::PUNPCKHBWrm },
    { X86::PUNPCKHDQrr,     X86::PUNPCKHDQrm },
    { X86::PUNPCKHQDQrr,    X86::PUNPCKHQDQrm },
    { X86::PUNPCKHWDrr,     X86::PUNPCKHWDrm },
    { X86::PUNPCKLBWrr,     X86::PUNPCKLBWrm },
    { X86::PUNPCKLDQrr,     X86::PUNPCKLDQrm },
    { X86::PUNPCKLQDQrr,    X86::PUNPCKLQDQrm },
    { X86::PUNPCKLWDrr,     X86::PUNPCKLWDrm },
    { X86::PXORrr,          X86::PXORrm },
    { X86::SBB32rr,         X86::SBB32rm },
    { X86::SBB64rr,         X86::SBB64rm },
    { X86::SHUFPDrri,       X86::SHUFPDrmi },
    { X86::SHUFPSrri,       X86::SHUFPSrmi },
    { X86::SUB16rr,         X86::SUB16rm },
    { X86::SUB32rr,         X86::SUB32rm },
    { X86::SUB64rr,         X86::SUB64rm },
    { X86::SUB8rr,          X86::SUB8rm },
    { X86::SUBPDrr,         X86::SUBPDrm },
    { X86::SUBPSrr,         X86::SUBPSrm },
    { X86::SUBSDrr,         X86::SUBSDrm },
    { X86::SUBSSrr,         X86::SUBSSrm },
    // FIXME: TEST*rr -> swapped operand of TEST*mr.
    { X86::UNPCKHPDrr,      X86::UNPCKHPDrm },
    { X86::UNPCKHPSrr,      X86::UNPCKHPSrm },
    { X86::UNPCKLPDrr,      X86::UNPCKLPDrm },
    { X86::UNPCKLPSrr,      X86::UNPCKLPSrm },
    { X86::XOR16rr,         X86::XOR16rm },
    { X86::XOR32rr,         X86::XOR32rm },
    { X86::XOR64rr,         X86::XOR64rm },
    { X86::XOR8rr,          X86::XOR8rm },
    { X86::XORPDrr,         X86::XORPDrm },
    { X86::XORPSrr,         X86::XORPSrm }
  };

  for (unsigned i = 0, e = array_lengthof(OpTbl2); i != e; ++i) {
    unsigned RegOp = OpTbl2[i][0];
    unsigned MemOp = OpTbl2[i][1];
    if (!RegOp2MemOpTable2.insert(std::make_pair((unsigned*)RegOp, MemOp)))
      assert(false && "Duplicated entries?");
    unsigned AuxInfo = 2 | (1 << 4); // Index 1, folded load
    if (!MemOp2RegOpTable.insert(std::make_pair((unsigned*)MemOp,
                                               std::make_pair(RegOp, AuxInfo))))
      AmbEntries.push_back(MemOp);
  }

  // Remove ambiguous entries.
  assert(AmbEntries.empty() && "Duplicated entries in unfolding maps?");
}

// getDwarfRegNum - This function maps LLVM register identifiers to the
// Dwarf specific numbering, used in debug info and exception tables.

int X86RegisterInfo::getDwarfRegNum(unsigned RegNo, bool isEH) const {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  unsigned Flavour = DWARFFlavour::X86_64;
  if (!Subtarget->is64Bit()) {
    if (Subtarget->isTargetDarwin()) {
      Flavour = DWARFFlavour::X86_32_Darwin;
    } else if (Subtarget->isTargetCygMing()) {
      // Unsupported by now, just quick fallback
      Flavour = DWARFFlavour::X86_32_ELF;
    } else {
      Flavour = DWARFFlavour::X86_32_ELF;
    }
  }

  return X86GenRegisterInfo::getDwarfRegNumFull(RegNo, Flavour);
}

// getX86RegNum - This function maps LLVM register identifiers to their X86
// specific numbering, which is used in various places encoding instructions.
//
unsigned X86RegisterInfo::getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::RAX: case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::RCX: case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::RDX: case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::RBX: case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::RSP: case X86::ESP: case X86::SP: case X86::SPL: case X86::AH:
    return N86::ESP;
  case X86::RBP: case X86::EBP: case X86::BP: case X86::BPL: case X86::CH:
    return N86::EBP;
  case X86::RSI: case X86::ESI: case X86::SI: case X86::SIL: case X86::DH:
    return N86::ESI;
  case X86::RDI: case X86::EDI: case X86::DI: case X86::DIL: case X86::BH:
    return N86::EDI;

  case X86::R8:  case X86::R8D:  case X86::R8W:  case X86::R8B:
    return N86::EAX;
  case X86::R9:  case X86::R9D:  case X86::R9W:  case X86::R9B:
    return N86::ECX;
  case X86::R10: case X86::R10D: case X86::R10W: case X86::R10B:
    return N86::EDX;
  case X86::R11: case X86::R11D: case X86::R11W: case X86::R11B:
    return N86::EBX;
  case X86::R12: case X86::R12D: case X86::R12W: case X86::R12B:
    return N86::ESP;
  case X86::R13: case X86::R13D: case X86::R13W: case X86::R13B:
    return N86::EBP;
  case X86::R14: case X86::R14D: case X86::R14W: case X86::R14B:
    return N86::ESI;
  case X86::R15: case X86::R15D: case X86::R15W: case X86::R15B:
    return N86::EDI;

  case X86::ST0: case X86::ST1: case X86::ST2: case X86::ST3:
  case X86::ST4: case X86::ST5: case X86::ST6: case X86::ST7:
    return RegNo-X86::ST0;

  case X86::XMM0: case X86::XMM8:
    return 0;
  case X86::XMM1: case X86::XMM9:
    return 1;
  case X86::XMM2: case X86::XMM10:
    return 2;
  case X86::XMM3: case X86::XMM11:
    return 3;
  case X86::XMM4: case X86::XMM12:
    return 4;
  case X86::XMM5: case X86::XMM13:
    return 5;
  case X86::XMM6: case X86::XMM14:
    return 6;
  case X86::XMM7: case X86::XMM15:
    return 7;

  default:
    assert(isVirtualRegister(RegNo) && "Unknown physical register!");
    assert(0 && "Register allocator hasn't allocated reg correctly yet!");
    return 0;
  }
}

bool X86RegisterInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                                MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  MachineFunction &MF = *MBB.getParent();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  X86FI->setCalleeSavedFrameSize(CSI.size() * SlotSize);
  unsigned Opc = Is64Bit ? X86::PUSH64r : X86::PUSH32r;
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    BuildMI(MBB, MI, TII.get(Opc)).addReg(Reg);
  }
  return true;
}

bool X86RegisterInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                                 MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  unsigned Opc = Is64Bit ? X86::POP64r : X86::POP32r;
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    BuildMI(MBB, MI, TII.get(Opc), Reg);
  }
  return true;
}

static const MachineInstrBuilder &X86InstrAddOperand(MachineInstrBuilder &MIB,
                                                     MachineOperand &MO) {
  if (MO.isRegister())
    MIB = MIB.addReg(MO.getReg(), MO.isDef(), MO.isImplicit(),
                     false, false, MO.getSubReg());
  else if (MO.isImmediate())
    MIB = MIB.addImm(MO.getImm());
  else if (MO.isFrameIndex())
    MIB = MIB.addFrameIndex(MO.getFrameIndex());
  else if (MO.isGlobalAddress())
    MIB = MIB.addGlobalAddress(MO.getGlobal(), MO.getOffset());
  else if (MO.isConstantPoolIndex())
    MIB = MIB.addConstantPoolIndex(MO.getConstantPoolIndex(), MO.getOffset());
  else if (MO.isJumpTableIndex())
    MIB = MIB.addJumpTableIndex(MO.getJumpTableIndex());
  else if (MO.isExternalSymbol())
    MIB = MIB.addExternalSymbol(MO.getSymbolName());
  else
    assert(0 && "Unknown operand for X86InstrAddOperand!");

  return MIB;
}

static unsigned getStoreRegOpcode(const TargetRegisterClass *RC,
                                  unsigned StackAlign) {
  unsigned Opc = 0;
  if (RC == &X86::GR64RegClass) {
    Opc = X86::MOV64mr;
  } else if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32mr;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16mr;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8mr;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_mr;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_mr;
  } else if (RC == &X86::RFP80RegClass) {
    Opc = X86::ST_FpP80m;   // pops
  } else if (RC == &X86::RFP64RegClass) {
    Opc = X86::ST_Fp64m;
  } else if (RC == &X86::RFP32RegClass) {
    Opc = X86::ST_Fp32m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSmr;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDmr;
  } else if (RC == &X86::VR128RegClass) {
    // FIXME: Use movaps once we are capable of selectively
    // aligning functions that spill SSE registers on 16-byte boundaries.
    Opc = StackAlign >= 16 ? X86::MOVAPSmr : X86::MOVUPSmr;
  } else if (RC == &X86::VR64RegClass) {
    Opc = X86::MMX_MOVQ64mr;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }

  return Opc;
}

void X86RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned SrcReg, int FrameIdx,
                                          const TargetRegisterClass *RC) const {
  unsigned Opc = getStoreRegOpcode(RC, StackAlign);
  addFrameReference(BuildMI(MBB, MI, TII.get(Opc)), FrameIdx)
    .addReg(SrcReg, false, false, true);
}

void X86RegisterInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                     SmallVectorImpl<MachineOperand> &Addr,
                                     const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Opc = getStoreRegOpcode(RC, StackAlign);
  MachineInstrBuilder MIB = BuildMI(TII.get(Opc));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB = X86InstrAddOperand(MIB, Addr[i]);
  MIB.addReg(SrcReg, false, false, true);
  NewMIs.push_back(MIB);
}

static unsigned getLoadRegOpcode(const TargetRegisterClass *RC,
                                 unsigned StackAlign) {
  unsigned Opc = 0;
  if (RC == &X86::GR64RegClass) {
    Opc = X86::MOV64rm;
  } else if (RC == &X86::GR32RegClass) {
    Opc = X86::MOV32rm;
  } else if (RC == &X86::GR16RegClass) {
    Opc = X86::MOV16rm;
  } else if (RC == &X86::GR8RegClass) {
    Opc = X86::MOV8rm;
  } else if (RC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_rm;
  } else if (RC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_rm;
  } else if (RC == &X86::RFP80RegClass) {
    Opc = X86::LD_Fp80m;
  } else if (RC == &X86::RFP64RegClass) {
    Opc = X86::LD_Fp64m;
  } else if (RC == &X86::RFP32RegClass) {
    Opc = X86::LD_Fp32m;
  } else if (RC == &X86::FR32RegClass) {
    Opc = X86::MOVSSrm;
  } else if (RC == &X86::FR64RegClass) {
    Opc = X86::MOVSDrm;
  } else if (RC == &X86::VR128RegClass) {
    // FIXME: Use movaps once we are capable of selectively
    // aligning functions that spill SSE registers on 16-byte boundaries.
    Opc = StackAlign >= 16 ? X86::MOVAPSrm : X86::MOVUPSrm;
  } else if (RC == &X86::VR64RegClass) {
    Opc = X86::MMX_MOVQ64rm;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }

  return Opc;
}

void X86RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                           unsigned DestReg, int FrameIdx,
                                           const TargetRegisterClass *RC) const{
  unsigned Opc = getLoadRegOpcode(RC, StackAlign);
  addFrameReference(BuildMI(MBB, MI, TII.get(Opc), DestReg), FrameIdx);
}

void X86RegisterInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                      SmallVectorImpl<MachineOperand> &Addr,
                                      const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Opc = getLoadRegOpcode(RC, StackAlign);
  MachineInstrBuilder MIB = BuildMI(TII.get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i)
    MIB = X86InstrAddOperand(MIB, Addr[i]);
  NewMIs.push_back(MIB);
}

void X86RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  if (DestRC != SrcRC) {
    // Moving EFLAGS to / from another register requires a push and a pop.
    if (SrcRC == &X86::CCRRegClass) {
      assert(SrcReg == X86::EFLAGS);
      if (DestRC == &X86::GR64RegClass) {
        BuildMI(MBB, MI, TII.get(X86::PUSHFQ));
        BuildMI(MBB, MI, TII.get(X86::POP64r), DestReg);
        return;
      } else if (DestRC == &X86::GR32RegClass) {
        BuildMI(MBB, MI, TII.get(X86::PUSHFD));
        BuildMI(MBB, MI, TII.get(X86::POP32r), DestReg);
        return;
      }
    } else if (DestRC == &X86::CCRRegClass) {
      assert(DestReg == X86::EFLAGS);
      if (SrcRC == &X86::GR64RegClass) {
        BuildMI(MBB, MI, TII.get(X86::PUSH64r)).addReg(SrcReg);
        BuildMI(MBB, MI, TII.get(X86::POPFQ));
        return;
      } else if (SrcRC == &X86::GR32RegClass) {
        BuildMI(MBB, MI, TII.get(X86::PUSH32r)).addReg(SrcReg);
        BuildMI(MBB, MI, TII.get(X86::POPFD));
        return;
      }
    }
    cerr << "Not yet supported!";
    abort();
  }

  unsigned Opc;
  if (DestRC == &X86::GR64RegClass) {
    Opc = X86::MOV64rr;
  } else if (DestRC == &X86::GR32RegClass) {
    Opc = X86::MOV32rr;
  } else if (DestRC == &X86::GR16RegClass) {
    Opc = X86::MOV16rr;
  } else if (DestRC == &X86::GR8RegClass) {
    Opc = X86::MOV8rr;
  } else if (DestRC == &X86::GR32_RegClass) {
    Opc = X86::MOV32_rr;
  } else if (DestRC == &X86::GR16_RegClass) {
    Opc = X86::MOV16_rr;
  } else if (DestRC == &X86::RFP32RegClass) {
    Opc = X86::MOV_Fp3232;
  } else if (DestRC == &X86::RFP64RegClass || DestRC == &X86::RSTRegClass) {
    Opc = X86::MOV_Fp6464;
  } else if (DestRC == &X86::RFP80RegClass) {
    Opc = X86::MOV_Fp8080;
  } else if (DestRC == &X86::FR32RegClass) {
    Opc = X86::FsMOVAPSrr;
  } else if (DestRC == &X86::FR64RegClass) {
    Opc = X86::FsMOVAPDrr;
  } else if (DestRC == &X86::VR128RegClass) {
    Opc = X86::MOVAPSrr;
  } else if (DestRC == &X86::VR64RegClass) {
    Opc = X86::MMX_MOVQ64rr;
  } else {
    assert(0 && "Unknown regclass");
    abort();
  }
  BuildMI(MBB, MI, TII.get(Opc), DestReg).addReg(SrcReg);
}

const TargetRegisterClass *
X86RegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &X86::CCRRegClass)
    if (Is64Bit)
      return &X86::GR64RegClass;
    else
      return &X86::GR32RegClass;
  return NULL;
}

void X86RegisterInfo::reMaterialize(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned DestReg,
                                    const MachineInstr *Orig) const {
  // MOV32r0 etc. are implemented with xor which clobbers condition code.
  // Re-materialize them as movri instructions to avoid side effects.
  switch (Orig->getOpcode()) {
  case X86::MOV8r0:
    BuildMI(MBB, I, TII.get(X86::MOV8ri), DestReg).addImm(0);
    break;
  case X86::MOV16r0:
    BuildMI(MBB, I, TII.get(X86::MOV16ri), DestReg).addImm(0);
    break;
  case X86::MOV32r0:
    BuildMI(MBB, I, TII.get(X86::MOV32ri), DestReg).addImm(0);
    break;
  case X86::MOV64r0:
    BuildMI(MBB, I, TII.get(X86::MOV64ri32), DestReg).addImm(0);
    break;
  default: {
    MachineInstr *MI = Orig->clone();
    MI->getOperand(0).setReg(DestReg);
    MBB.insert(I, MI);
    break;
  }
  }
}

static MachineInstr *FuseTwoAddrInst(unsigned Opcode,
                                     SmallVector<MachineOperand,4> &MOs,
                                 MachineInstr *MI, const TargetInstrInfo &TII) {
  // Create the base instruction with the memory operand as the first part.
  MachineInstr *NewMI = new MachineInstr(TII.get(Opcode), true);
  MachineInstrBuilder MIB(NewMI);
  unsigned NumAddrOps = MOs.size();
  for (unsigned i = 0; i != NumAddrOps; ++i)
    MIB = X86InstrAddOperand(MIB, MOs[i]);
  if (NumAddrOps < 4)  // FrameIndex only
    MIB.addImm(1).addReg(0).addImm(0);
  
  // Loop over the rest of the ri operands, converting them over.
  unsigned NumOps = TII.getNumOperands(MI->getOpcode())-2;
  for (unsigned i = 0; i != NumOps; ++i) {
    MachineOperand &MO = MI->getOperand(i+2);
    MIB = X86InstrAddOperand(MIB, MO);
  }
  for (unsigned i = NumOps+2, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    MIB = X86InstrAddOperand(MIB, MO);
  }
  return MIB;
}

static MachineInstr *FuseInst(unsigned Opcode, unsigned OpNo,
                              SmallVector<MachineOperand,4> &MOs,
                              MachineInstr *MI, const TargetInstrInfo &TII) {
  MachineInstr *NewMI = new MachineInstr(TII.get(Opcode), true);
  MachineInstrBuilder MIB(NewMI);
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (i == OpNo) {
      assert(MO.isRegister() && "Expected to fold into reg operand!");
      unsigned NumAddrOps = MOs.size();
      for (unsigned i = 0; i != NumAddrOps; ++i)
        MIB = X86InstrAddOperand(MIB, MOs[i]);
      if (NumAddrOps < 4)  // FrameIndex only
        MIB.addImm(1).addReg(0).addImm(0);
    } else {
      MIB = X86InstrAddOperand(MIB, MO);
    }
  }
  return MIB;
}

static MachineInstr *MakeM0Inst(const TargetInstrInfo &TII, unsigned Opcode,
                                SmallVector<MachineOperand,4> &MOs,
                                MachineInstr *MI) {
  MachineInstrBuilder MIB = BuildMI(TII.get(Opcode));

  unsigned NumAddrOps = MOs.size();
  for (unsigned i = 0; i != NumAddrOps; ++i)
    MIB = X86InstrAddOperand(MIB, MOs[i]);
  if (NumAddrOps < 4)  // FrameIndex only
    MIB.addImm(1).addReg(0).addImm(0);
  return MIB.addImm(0);
}

MachineInstr*
X86RegisterInfo::foldMemoryOperand(MachineInstr *MI, unsigned i,
                                   SmallVector<MachineOperand,4> &MOs) const {
  const DenseMap<unsigned*, unsigned> *OpcodeTablePtr = NULL;
  bool isTwoAddrFold = false;
  unsigned NumOps = TII.getNumOperands(MI->getOpcode());
  bool isTwoAddr = NumOps > 1 &&
    MI->getInstrDescriptor()->getOperandConstraint(1, TOI::TIED_TO) != -1;

  MachineInstr *NewMI = NULL;
  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  if (isTwoAddr && NumOps >= 2 && i < 2 &&
      MI->getOperand(0).isRegister() && 
      MI->getOperand(1).isRegister() &&
      MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) { 
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
    isTwoAddrFold = true;
  } else if (i == 0) { // If operand 0
    if (MI->getOpcode() == X86::MOV16r0)
      NewMI = MakeM0Inst(TII, X86::MOV16mi, MOs, MI);
    else if (MI->getOpcode() == X86::MOV32r0)
      NewMI = MakeM0Inst(TII, X86::MOV32mi, MOs, MI);
    else if (MI->getOpcode() == X86::MOV64r0)
      NewMI = MakeM0Inst(TII, X86::MOV64mi32, MOs, MI);
    else if (MI->getOpcode() == X86::MOV8r0)
      NewMI = MakeM0Inst(TII, X86::MOV8mi, MOs, MI);
    if (NewMI) {
      NewMI->copyKillDeadInfo(MI);
      return NewMI;
    }
    
    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (i == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (i == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  }
  
  // If table selected...
  if (OpcodeTablePtr) {
    // Find the Opcode to fuse
    DenseMap<unsigned*, unsigned>::iterator I =
      OpcodeTablePtr->find((unsigned*)MI->getOpcode());
    if (I != OpcodeTablePtr->end()) {
      if (isTwoAddrFold)
        NewMI = FuseTwoAddrInst(I->second, MOs, MI, TII);
      else
        NewMI = FuseInst(I->second, i, MOs, MI, TII);
      NewMI->copyKillDeadInfo(MI);
      return NewMI;
    }
  }
  
  // No fusion 
  if (PrintFailedFusing)
    cerr << "We failed to fuse ("
         << ((i == 1) ? "r" : "s") << "): " << *MI;
  return NULL;
}


MachineInstr* X86RegisterInfo::foldMemoryOperand(MachineInstr *MI,
                                              SmallVectorImpl<unsigned> &Ops,
                                              int FrameIndex) const {
  // Check switch flag 
  if (NoFusing) return NULL;

  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri32; break;
    }
    // Change to CMPXXri r, 0 first.
    MI->setInstrDescriptor(TII.get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  SmallVector<MachineOperand,4> MOs;
  MOs.push_back(MachineOperand::CreateFrameIndex(FrameIndex));
  return foldMemoryOperand(MI, Ops[0], MOs);
}

MachineInstr* X86RegisterInfo::foldMemoryOperand(MachineInstr *MI,
                                                 SmallVectorImpl<unsigned> &Ops,
                                                 MachineInstr *LoadMI) const {
  // Check switch flag 
  if (NoFusing) return NULL;

  if (Ops.size() == 2 && Ops[0] == 0 && Ops[1] == 1) {
    unsigned NewOpc = 0;
    switch (MI->getOpcode()) {
    default: return NULL;
    case X86::TEST8rr:  NewOpc = X86::CMP8ri; break;
    case X86::TEST16rr: NewOpc = X86::CMP16ri; break;
    case X86::TEST32rr: NewOpc = X86::CMP32ri; break;
    case X86::TEST64rr: NewOpc = X86::CMP64ri32; break;
    }
    // Change to CMPXXri r, 0 first.
    MI->setInstrDescriptor(TII.get(NewOpc));
    MI->getOperand(1).ChangeToImmediate(0);
  } else if (Ops.size() != 1)
    return NULL;

  SmallVector<MachineOperand,4> MOs;
  unsigned NumOps = TII.getNumOperands(LoadMI->getOpcode());
  for (unsigned i = NumOps - 4; i != NumOps; ++i)
    MOs.push_back(LoadMI->getOperand(i));
  return foldMemoryOperand(MI, Ops[0], MOs);
}


unsigned X86RegisterInfo::getOpcodeAfterMemoryFold(unsigned Opc,
                                                   unsigned OpNum) const {
  // Check switch flag 
  if (NoFusing) return 0;
  const DenseMap<unsigned*, unsigned> *OpcodeTablePtr = NULL;
  unsigned NumOps = TII.getNumOperands(Opc);
  bool isTwoAddr = NumOps > 1 &&
    TII.getOperandConstraint(Opc, 1, TOI::TIED_TO) != -1;

  // Folding a memory location into the two-address part of a two-address
  // instruction is different than folding it other places.  It requires
  // replacing the *two* registers with the memory location.
  if (isTwoAddr && NumOps >= 2 && OpNum < 2) { 
    OpcodeTablePtr = &RegOp2MemOpTable2Addr;
  } else if (OpNum == 0) { // If operand 0
    switch (Opc) {
    case X86::MOV16r0:
      return X86::MOV16mi;
    case X86::MOV32r0:
      return X86::MOV32mi;
    case X86::MOV64r0:
      return X86::MOV64mi32;
    case X86::MOV8r0:
      return X86::MOV8mi;
    default: break;
    }
    OpcodeTablePtr = &RegOp2MemOpTable0;
  } else if (OpNum == 1) {
    OpcodeTablePtr = &RegOp2MemOpTable1;
  } else if (OpNum == 2) {
    OpcodeTablePtr = &RegOp2MemOpTable2;
  }
  
  if (OpcodeTablePtr) {
    // Find the Opcode to fuse
    DenseMap<unsigned*, unsigned>::iterator I =
      OpcodeTablePtr->find((unsigned*)Opc);
    if (I != OpcodeTablePtr->end())
      return I->second;
  }
  return 0;
}

bool X86RegisterInfo::unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
    MemOp2RegOpTable.find((unsigned*)MI->getOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & 0xf;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  if (UnfoldLoad && !FoldedLoad)
    return false;
  UnfoldLoad &= FoldedLoad;
  if (UnfoldStore && !FoldedStore)
    return false;
  UnfoldStore &= FoldedStore;

  const TargetInstrDescriptor &TID = TII.get(Opc);
  const TargetOperandInfo &TOI = TID.OpInfo[Index];
  const TargetRegisterClass *RC = (TOI.Flags & M_LOOK_UP_PTR_REG_CLASS)
    ? TII.getPointerRegClass() : getRegClass(TOI.RegClass);
  SmallVector<MachineOperand,4> AddrOps;
  SmallVector<MachineOperand,2> BeforeOps;
  SmallVector<MachineOperand,2> AfterOps;
  SmallVector<MachineOperand,4> ImpOps;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &Op = MI->getOperand(i);
    if (i >= Index && i < Index+4)
      AddrOps.push_back(Op);
    else if (Op.isRegister() && Op.isImplicit())
      ImpOps.push_back(Op);
    else if (i < Index)
      BeforeOps.push_back(Op);
    else if (i > Index)
      AfterOps.push_back(Op);
  }

  // Emit the load instruction.
  if (UnfoldLoad) {
    loadRegFromAddr(MF, Reg, AddrOps, RC, NewMIs);
    if (UnfoldStore) {
      // Address operands cannot be marked isKill.
      for (unsigned i = 1; i != 5; ++i) {
        MachineOperand &MO = NewMIs[0]->getOperand(i);
        if (MO.isRegister())
          MO.unsetIsKill();
      }
    }
  }

  // Emit the data processing instruction.
  MachineInstr *DataMI = new MachineInstr(TID, true);
  MachineInstrBuilder MIB(DataMI);
  
  if (FoldedStore)
    MIB.addReg(Reg, true);
  for (unsigned i = 0, e = BeforeOps.size(); i != e; ++i)
    MIB = X86InstrAddOperand(MIB, BeforeOps[i]);
  if (FoldedLoad)
    MIB.addReg(Reg);
  for (unsigned i = 0, e = AfterOps.size(); i != e; ++i)
    MIB = X86InstrAddOperand(MIB, AfterOps[i]);
  for (unsigned i = 0, e = ImpOps.size(); i != e; ++i) {
    MachineOperand &MO = ImpOps[i];
    MIB.addReg(MO.getReg(), MO.isDef(), true, MO.isKill(), MO.isDead());
  }
  // Change CMP32ri r, 0 back to TEST32rr r, r, etc.
  unsigned NewOpc = 0;
  switch (DataMI->getOpcode()) {
  default: break;
  case X86::CMP64ri32:
  case X86::CMP32ri:
  case X86::CMP16ri:
  case X86::CMP8ri: {
    MachineOperand &MO0 = DataMI->getOperand(0);
    MachineOperand &MO1 = DataMI->getOperand(1);
    if (MO1.getImm() == 0) {
      switch (DataMI->getOpcode()) {
      default: break;
      case X86::CMP64ri32: NewOpc = X86::TEST64rr; break;
      case X86::CMP32ri:   NewOpc = X86::TEST32rr; break;
      case X86::CMP16ri:   NewOpc = X86::TEST16rr; break;
      case X86::CMP8ri:    NewOpc = X86::TEST8rr; break;
      }
      DataMI->setInstrDescriptor(TII.get(NewOpc));
      MO1.ChangeToRegister(MO0.getReg(), false);
    }
  }
  }
  NewMIs.push_back(DataMI);

  // Emit the store instruction.
  if (UnfoldStore) {
    const TargetOperandInfo &DstTOI = TID.OpInfo[0];
    const TargetRegisterClass *DstRC = (DstTOI.Flags & M_LOOK_UP_PTR_REG_CLASS)
      ? TII.getPointerRegClass() : getRegClass(DstTOI.RegClass);
    storeRegToAddr(MF, Reg, AddrOps, DstRC, NewMIs);
  }

  return true;
}


bool
X86RegisterInfo::unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                     SmallVectorImpl<SDNode*> &NewNodes) const {
  if (!N->isTargetOpcode())
    return false;

  DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
    MemOp2RegOpTable.find((unsigned*)N->getTargetOpcode());
  if (I == MemOp2RegOpTable.end())
    return false;
  unsigned Opc = I->second.first;
  unsigned Index = I->second.second & 0xf;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  const TargetInstrDescriptor &TID = TII.get(Opc);
  const TargetOperandInfo &TOI = TID.OpInfo[Index];
  const TargetRegisterClass *RC = (TOI.Flags & M_LOOK_UP_PTR_REG_CLASS)
    ? TII.getPointerRegClass() : getRegClass(TOI.RegClass);
  std::vector<SDOperand> AddrOps;
  std::vector<SDOperand> BeforeOps;
  std::vector<SDOperand> AfterOps;
  unsigned NumOps = N->getNumOperands();
  for (unsigned i = 0; i != NumOps-1; ++i) {
    SDOperand Op = N->getOperand(i);
    if (i >= Index && i < Index+4)
      AddrOps.push_back(Op);
    else if (i < Index)
      BeforeOps.push_back(Op);
    else if (i > Index)
      AfterOps.push_back(Op);
  }
  SDOperand Chain = N->getOperand(NumOps-1);
  AddrOps.push_back(Chain);

  // Emit the load instruction.
  SDNode *Load = 0;
  if (FoldedLoad) {
    MVT::ValueType VT = *RC->vt_begin();
    Load = DAG.getTargetNode(getLoadRegOpcode(RC, StackAlign), VT, MVT::Other,
                             &AddrOps[0], AddrOps.size());
    NewNodes.push_back(Load);
  }

  // Emit the data processing instruction.
  std::vector<MVT::ValueType> VTs;
  const TargetRegisterClass *DstRC = 0;
  if (TID.numDefs > 0) {
    const TargetOperandInfo &DstTOI = TID.OpInfo[0];
    DstRC = (DstTOI.Flags & M_LOOK_UP_PTR_REG_CLASS)
      ? TII.getPointerRegClass() : getRegClass(DstTOI.RegClass);
    VTs.push_back(*DstRC->vt_begin());
  }
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    MVT::ValueType VT = N->getValueType(i);
    if (VT != MVT::Other && i >= TID.numDefs)
      VTs.push_back(VT);
  }
  if (Load)
    BeforeOps.push_back(SDOperand(Load, 0));
  std::copy(AfterOps.begin(), AfterOps.end(), std::back_inserter(BeforeOps));
  SDNode *NewNode= DAG.getTargetNode(Opc, VTs, &BeforeOps[0], BeforeOps.size());
  NewNodes.push_back(NewNode);

  // Emit the store instruction.
  if (FoldedStore) {
    AddrOps.pop_back();
    AddrOps.push_back(SDOperand(NewNode, 0));
    AddrOps.push_back(Chain);
    SDNode *Store = DAG.getTargetNode(getStoreRegOpcode(DstRC, StackAlign),
                                      MVT::Other, &AddrOps[0], AddrOps.size());
    NewNodes.push_back(Store);
  }

  return true;
}

unsigned X86RegisterInfo::getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore) const {
  DenseMap<unsigned*, std::pair<unsigned,unsigned> >::iterator I =
    MemOp2RegOpTable.find((unsigned*)Opc);
  if (I == MemOp2RegOpTable.end())
    return 0;
  bool FoldedLoad = I->second.second & (1 << 4);
  bool FoldedStore = I->second.second & (1 << 5);
  if (UnfoldLoad && !FoldedLoad)
    return 0;
  if (UnfoldStore && !FoldedStore)
    return 0;
  return I->second.first;
}

const unsigned *
X86RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs32Bit[] = {
    X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs32EHRet[] = {
    X86::EAX, X86::EDX, X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs64Bit[] = {
    X86::RBX, X86::R12, X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };

  if (Is64Bit)
    return CalleeSavedRegs64Bit;
  else {
    if (MF) {
        MachineFrameInfo *MFI = MF->getFrameInfo();
        MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
        if (MMI && MMI->callsEHReturn())
          return CalleeSavedRegs32EHRet;
    }
    return CalleeSavedRegs32Bit;
  }
}

const TargetRegisterClass* const*
X86RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses32Bit[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  static const TargetRegisterClass * const CalleeSavedRegClasses32EHRet[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  static const TargetRegisterClass * const CalleeSavedRegClasses64Bit[] = {
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass, 0
  };

  if (Is64Bit)
    return CalleeSavedRegClasses64Bit;
  else {
    if (MF) {
        MachineFrameInfo *MFI = MF->getFrameInfo();
        MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
        if (MMI && MMI->callsEHReturn())
          return CalleeSavedRegClasses32EHRet;
    }
    return CalleeSavedRegClasses32Bit;
  }

}

BitVector X86RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(X86::RSP);
  Reserved.set(X86::ESP);
  Reserved.set(X86::SP);
  Reserved.set(X86::SPL);
  if (hasFP(MF)) {
    Reserved.set(X86::RBP);
    Reserved.set(X86::EBP);
    Reserved.set(X86::BP);
    Reserved.set(X86::BPL);
  }
  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
bool X86RegisterInfo::hasFP(const MachineFunction &MF) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();

  return (NoFramePointerElim || 
          MFI->hasVarSizedObjects() ||
          MF.getInfo<X86MachineFunctionInfo>()->getForceFramePointer() ||
          (MMI && MMI->callsUnwindInit()));
}

bool X86RegisterInfo::hasReservedCallFrame(MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

void X86RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    // If the stack pointer can be changed after prologue, turn the
    // adjcallstackup instruction into a 'sub ESP, <amt>' and the
    // adjcallstackdown instruction into 'add ESP, <amt>'
    // TODO: consider using push / pop instead of sub + store / add
    MachineInstr *Old = I;
    uint64_t Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      Amount = (Amount+StackAlign-1)/StackAlign*StackAlign;

      MachineInstr *New = 0;
      if (Old->getOpcode() == X86::ADJCALLSTACKDOWN) {
        New=BuildMI(TII.get(Is64Bit ? X86::SUB64ri32 : X86::SUB32ri), StackPtr)
          .addReg(StackPtr).addImm(Amount);
      } else {
        assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
        // factor out the amount the callee already popped.
        uint64_t CalleeAmt = Old->getOperand(1).getImm();
        Amount -= CalleeAmt;
        if (Amount) {
          unsigned Opc = (Amount < 128) ?
            (Is64Bit ? X86::ADD64ri8 : X86::ADD32ri8) :
            (Is64Bit ? X86::ADD64ri32 : X86::ADD32ri);
          New = BuildMI(TII.get(Opc), StackPtr).addReg(StackPtr).addImm(Amount);
        }
      }

      // Replace the pseudo instruction with a new instruction...
      if (New) MBB.insert(I, New);
    }
  } else if (I->getOpcode() == X86::ADJCALLSTACKUP) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.  We do this until we have
    // more advanced stack pointer tracking ability.
    if (uint64_t CalleeAmt = I->getOperand(1).getImm()) {
      unsigned Opc = (CalleeAmt < 128) ?
        (Is64Bit ? X86::SUB64ri8 : X86::SUB32ri8) :
        (Is64Bit ? X86::SUB64ri32 : X86::SUB32ri);
      MachineInstr *New =
        BuildMI(TII.get(Opc), StackPtr).addReg(StackPtr).addImm(CalleeAmt);
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void X86RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, RegScavenger *RS) const{
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();
  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add an offset to the offset.
  MI.getOperand(i).ChangeToRegister(hasFP(MF) ? FramePtr : StackPtr, false);

  // Now add the frame object offset to the offset from EBP.
  int64_t Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
                   MI.getOperand(i+3).getImm()+SlotSize;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else {
    Offset += SlotSize;  // Skip the saved EBP
    // Skip the RETADDR move area
    X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
    int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
    if (TailCallReturnAddrDelta < 0) Offset -= TailCallReturnAddrDelta;
  }
  
  MI.getOperand(i+3).ChangeToImmediate(Offset);
}

void
X86RegisterInfo::processFunctionBeforeFrameFinalized(MachineFunction &MF) const{
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  int32_t TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
  if (TailCallReturnAddrDelta < 0) {
    // create RETURNADDR area
    //   arg
    //   arg
    //   RETADDR
    //   { ...
    //     RETADDR area
    //     ...
    //   }
    //   [EBP]
    MF.getFrameInfo()->
      CreateFixedObject(-TailCallReturnAddrDelta,
                        (-1*SlotSize)+TailCallReturnAddrDelta);
  }
  if (hasFP(MF)) {
    assert((TailCallReturnAddrDelta <= 0) &&
           "The Delta should always be zero or negative");
    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MF.getFrameInfo()->CreateFixedObject(SlotSize,
                                                        (int)SlotSize * -2+
                                                       TailCallReturnAddrDelta);
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexBegin() &&
           "Slot for EBP register must be last in order to be found!");
  }
}

/// emitSPUpdate - Emit a series of instructions to increment / decrement the
/// stack pointer by a constant value.
static
void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                  unsigned StackPtr, int64_t NumBytes, bool Is64Bit,
                  const TargetInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  uint64_t Offset = isSub ? -NumBytes : NumBytes;
  unsigned Opc = isSub
    ? ((Offset < 128) ?
       (Is64Bit ? X86::SUB64ri8 : X86::SUB32ri8) :
       (Is64Bit ? X86::SUB64ri32 : X86::SUB32ri))
    : ((Offset < 128) ?
       (Is64Bit ? X86::ADD64ri8 : X86::ADD32ri8) :
       (Is64Bit ? X86::ADD64ri32 : X86::ADD32ri));
  uint64_t Chunk = (1LL << 31) - 1;

  while (Offset) {
    uint64_t ThisVal = (Offset > Chunk) ? Chunk : Offset;
    BuildMI(MBB, MBBI, TII.get(Opc), StackPtr).addReg(StackPtr).addImm(ThisVal);
    Offset -= ThisVal;
  }
}

// mergeSPUpdatesUp - Merge two stack-manipulating instructions upper iterator.
static
void mergeSPUpdatesUp(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                      unsigned StackPtr, uint64_t *NumBytes = NULL) {
  if (MBBI == MBB.begin()) return;
  
  MachineBasicBlock::iterator PI = prior(MBBI);
  unsigned Opc = PI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += PI->getOperand(2).getImm();
    MBB.erase(PI);
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= PI->getOperand(2).getImm();
    MBB.erase(PI);
  }
}

// mergeSPUpdatesUp - Merge two stack-manipulating instructions lower iterator.
static
void mergeSPUpdatesDown(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI,
                        unsigned StackPtr, uint64_t *NumBytes = NULL) {
  return;
  
  if (MBBI == MBB.end()) return;
  
  MachineBasicBlock::iterator NI = next(MBBI);
  if (NI == MBB.end()) return;
  
  unsigned Opc = NI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  }
}

/// mergeSPUpdates - Checks the instruction before/after the passed
/// instruction. If it is an ADD/SUB instruction it is deleted 
/// argument and the stack adjustment is returned as a positive value for ADD
/// and a negative for SUB. 
static int mergeSPUpdates(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI,
                           unsigned StackPtr,                     
                           bool doMergeWithPrevious) {

  if ((doMergeWithPrevious && MBBI == MBB.begin()) ||
      (!doMergeWithPrevious && MBBI == MBB.end()))
    return 0;

  int Offset = 0;

  MachineBasicBlock::iterator PI = doMergeWithPrevious ? prior(MBBI) : MBBI;
  MachineBasicBlock::iterator NI = doMergeWithPrevious ? 0 : next(MBBI);
  unsigned Opc = PI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr){
    Offset += PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    Offset -= PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  }   

  return Offset;
}

void X86RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function* Fn = MF.getFunction();
  const X86Subtarget* Subtarget = &MF.getTarget().getSubtarget<X86Subtarget>();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  
  // Prepare for frame info.
  unsigned FrameLabelId = 0;
  
  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI->getStackSize();
  // Add RETADDR move area to callee saved frame size.
  int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
  if (TailCallReturnAddrDelta < 0)  
    X86FI->setCalleeSavedFrameSize(
          X86FI->getCalleeSavedFrameSize() +(-TailCallReturnAddrDelta));
  uint64_t NumBytes = StackSize - X86FI->getCalleeSavedFrameSize();

  // Insert stack pointer adjustment for later moving of return addr.  Only
  // applies to tail call optimized functions where the callee argument stack
  // size is bigger than the callers.
  if (TailCallReturnAddrDelta < 0) {
    BuildMI(MBB, MBBI, TII.get(Is64Bit? X86::SUB64ri32 : X86::SUB32ri), 
            StackPtr).addReg(StackPtr).addImm(-TailCallReturnAddrDelta);
  }

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    // Update the frame offset adjustment.
    MFI->setOffsetAdjustment(SlotSize-NumBytes);

    // Save EBP into the appropriate stack slot...
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::PUSH64r : X86::PUSH32r))
      .addReg(FramePtr);
    NumBytes -= SlotSize;

    if (MMI && MMI->needsFrameInfo()) {
      // Mark effective beginning of when frame pointer becomes valid.
      FrameLabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, TII.get(X86::LABEL)).addImm(FrameLabelId);
    }

    // Update EBP with the new base value...
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr), FramePtr)
      .addReg(StackPtr);
  }
  
  unsigned ReadyLabelId = 0;
  if (MMI && MMI->needsFrameInfo()) {
    // Mark effective beginning of when frame pointer is ready.
    ReadyLabelId = MMI->NextLabelID();
    BuildMI(MBB, MBBI, TII.get(X86::LABEL)).addImm(ReadyLabelId);
  }

  // Skip the callee-saved push instructions.
  while (MBBI != MBB.end() &&
         (MBBI->getOpcode() == X86::PUSH32r ||
          MBBI->getOpcode() == X86::PUSH64r))
    ++MBBI;

  if (NumBytes) {   // adjust stack pointer: ESP -= numbytes
    if (NumBytes >= 4096 && Subtarget->isTargetCygMing()) {
      // Check, whether EAX is livein for this function
      bool isEAXAlive = false;
      for (MachineFunction::livein_iterator II = MF.livein_begin(),
             EE = MF.livein_end(); (II != EE) && !isEAXAlive; ++II) {
        unsigned Reg = II->first;
        isEAXAlive = (Reg == X86::EAX || Reg == X86::AX ||
                      Reg == X86::AH || Reg == X86::AL);
      }

      // Function prologue calls _alloca to probe the stack when allocating  
      // more than 4k bytes in one go. Touching the stack at 4K increments is  
      // necessary to ensure that the guard pages used by the OS virtual memory
      // manager are allocated in correct sequence.
      if (!isEAXAlive) {
        BuildMI(MBB, MBBI, TII.get(X86::MOV32ri), X86::EAX).addImm(NumBytes);
        BuildMI(MBB, MBBI, TII.get(X86::CALLpcrel32))
          .addExternalSymbol("_alloca");
      } else {
        // Save EAX
        BuildMI(MBB, MBBI, TII.get(X86::PUSH32r), X86::EAX);
        // Allocate NumBytes-4 bytes on stack. We'll also use 4 already
        // allocated bytes for EAX.
        BuildMI(MBB, MBBI, TII.get(X86::MOV32ri), X86::EAX).addImm(NumBytes-4);
        BuildMI(MBB, MBBI, TII.get(X86::CALLpcrel32))
          .addExternalSymbol("_alloca");
        // Restore EAX
        MachineInstr *MI = addRegOffset(BuildMI(TII.get(X86::MOV32rm),X86::EAX),
                                        StackPtr, NumBytes-4);
        MBB.insert(MBBI, MI);
      }
    } else {
      // If there is an SUB32ri of ESP immediately before this instruction,
      // merge the two. This can be the case when tail call elimination is
      // enabled and the callee has more arguments then the caller.
      NumBytes -= mergeSPUpdates(MBB, MBBI, StackPtr, true);
      // If there is an ADD32ri or SUB32ri of ESP immediately after this
      // instruction, merge the two instructions.
      mergeSPUpdatesDown(MBB, MBBI, StackPtr, &NumBytes);
      
      if (NumBytes)
        emitSPUpdate(MBB, MBBI, StackPtr, -(int64_t)NumBytes, Is64Bit, TII);
    }
  }

  if (MMI && MMI->needsFrameInfo()) {
    std::vector<MachineMove> &Moves = MMI->getFrameMoves();
    const TargetData *TD = MF.getTarget().getTargetData();

    // Calculate amount of bytes used for return address storing
    int stackGrowth =
      (MF.getTarget().getFrameInfo()->getStackGrowthDirection() ==
       TargetFrameInfo::StackGrowsUp ?
       TD->getPointerSize() : -TD->getPointerSize());

    if (StackSize) {
      // Show update of SP.
      if (hasFP(MF)) {
        // Adjust SP
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP, 2*stackGrowth);
        Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
      } else {
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP, -StackSize+stackGrowth);
        Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
      }
    } else {
      //FIXME: Verify & implement for FP
      MachineLocation SPDst(StackPtr);
      MachineLocation SPSrc(StackPtr, stackGrowth);
      Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
    }
            
    // Add callee saved registers to move list.
    const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

    // FIXME: This is dirty hack. The code itself is pretty mess right now.
    // It should be rewritten from scratch and generalized sometimes.
    
    // Determine maximum offset (minumum due to stack growth)
    int64_t MaxOffset = 0;
    for (unsigned I = 0, E = CSI.size(); I!=E; ++I)
      MaxOffset = std::min(MaxOffset,
                           MFI->getObjectOffset(CSI[I].getFrameIdx()));

    // Calculate offsets
    int64_t saveAreaOffset = (hasFP(MF) ? 3 : 2)*stackGrowth;
    for (unsigned I = 0, E = CSI.size(); I!=E; ++I) {
      int64_t Offset = MFI->getObjectOffset(CSI[I].getFrameIdx());
      unsigned Reg = CSI[I].getReg();
      Offset = (MaxOffset-Offset+saveAreaOffset);
      MachineLocation CSDst(MachineLocation::VirtualFP, Offset);
      MachineLocation CSSrc(Reg);
      Moves.push_back(MachineMove(FrameLabelId, CSDst, CSSrc));
    }
    
    if (hasFP(MF)) {
      // Save FP
      MachineLocation FPDst(MachineLocation::VirtualFP, 2*stackGrowth);
      MachineLocation FPSrc(FramePtr);
      Moves.push_back(MachineMove(ReadyLabelId, FPDst, FPSrc));
    }
    
    MachineLocation FPDst(hasFP(MF) ? FramePtr : StackPtr);
    MachineLocation FPSrc(MachineLocation::VirtualFP);
    Moves.push_back(MachineMove(ReadyLabelId, FPDst, FPSrc));
  }

  // If it's main() on Cygwin\Mingw32 we should align stack as well
  if (Fn->hasExternalLinkage() && Fn->getName() == "main" &&
      Subtarget->isTargetCygMing()) {
    BuildMI(MBB, MBBI, TII.get(X86::AND32ri), X86::ESP)
                .addReg(X86::ESP).addImm(-StackAlign);

    // Probe the stack
    BuildMI(MBB, MBBI, TII.get(X86::MOV32ri), X86::EAX).addImm(StackAlign);
    BuildMI(MBB, MBBI, TII.get(X86::CALLpcrel32)).addExternalSymbol("_alloca");
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function* Fn = MF.getFunction();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  const X86Subtarget* Subtarget = &MF.getTarget().getSubtarget<X86Subtarget>();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  unsigned RetOpcode = MBBI->getOpcode();

  switch (RetOpcode) {
  case X86::RET:
  case X86::RETI:
  case X86::TCRETURNdi:
  case X86::TCRETURNri:
  case X86::TCRETURNri64:
  case X86::TCRETURNdi64:
  case X86::EH_RETURN:
  case X86::TAILJMPd:
  case X86::TAILJMPr:
  case X86::TAILJMPm: break;  // These are ok
  default:
    assert(0 && "Can only insert epilog into returning blocks");
  }

  // Get the number of bytes to allocate from the FrameInfo
  uint64_t StackSize = MFI->getStackSize();
  unsigned CSSize = X86FI->getCalleeSavedFrameSize();
  uint64_t NumBytes = StackSize - CSSize;

  if (hasFP(MF)) {
    // pop EBP.
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::POP64r : X86::POP32r), FramePtr);
    NumBytes -= SlotSize;
  }

  // Skip the callee-saved pop instructions.
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = prior(MBBI);
    unsigned Opc = PI->getOpcode();
    if (Opc != X86::POP32r && Opc != X86::POP64r && !TII.isTerminatorInstr(Opc))
      break;
    --MBBI;
  }

  // If there is an ADD32ri or SUB32ri of ESP immediately before this
  // instruction, merge the two instructions.
  if (NumBytes || MFI->hasVarSizedObjects())
    mergeSPUpdatesUp(MBB, MBBI, StackPtr, &NumBytes);

  // If dynamic alloca is used, then reset esp to point to the last callee-saved
  // slot before popping them off!  Also, if it's main() on Cygwin/Mingw32 we
  // aligned stack in the prologue, - revert stack changes back. Note: we're
  // assuming, that frame pointer was forced for main()
  if (MFI->hasVarSizedObjects() ||
      (Fn->hasExternalLinkage() && Fn->getName() == "main" &&
       Subtarget->isTargetCygMing())) {
    unsigned Opc = Is64Bit ? X86::LEA64r : X86::LEA32r;
    if (CSSize) {
      MachineInstr *MI = addRegOffset(BuildMI(TII.get(Opc), StackPtr),
                                      FramePtr, -CSSize);
      MBB.insert(MBBI, MI);
    } else
      BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),StackPtr).
        addReg(FramePtr);

    NumBytes = 0;
  }

  // adjust stack pointer back: ESP += numbytes
  if (NumBytes)
    emitSPUpdate(MBB, MBBI, StackPtr, NumBytes, Is64Bit, TII);

  // We're returning from function via eh_return.
  if (RetOpcode == X86::EH_RETURN) {
    MBBI = prior(MBB.end());
    MachineOperand &DestAddr  = MBBI->getOperand(0);
    assert(DestAddr.isRegister() && "Offset should be in register!");
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),StackPtr).
      addReg(DestAddr.getReg()); 
  // Tail call return: adjust the stack pointer and jump to callee
  } else if (RetOpcode == X86::TCRETURNri || RetOpcode == X86::TCRETURNdi ||
             RetOpcode== X86::TCRETURNri64 || RetOpcode == X86::TCRETURNdi64) {
    MBBI = prior(MBB.end());
    MachineOperand &JumpTarget = MBBI->getOperand(0);
    MachineOperand &StackAdjust = MBBI->getOperand(1);
    assert( StackAdjust.isImmediate() && "Expecting immediate value.");
    
    // Adjust stack pointer.
    int StackAdj = StackAdjust.getImm();
    int MaxTCDelta = X86FI->getTCReturnAddrDelta();
    int Offset = 0;
    assert(MaxTCDelta <= 0 && "MaxTCDelta should never be positive");
    // Incoporate the retaddr area.
    Offset = StackAdj-MaxTCDelta;
    assert(Offset >= 0 && "Offset should never be negative");
    if (Offset) {
      // Check for possible merge with preceeding ADD instruction.
      Offset += mergeSPUpdates(MBB, MBBI, StackPtr, true);
      emitSPUpdate(MBB, MBBI, StackPtr, Offset, Is64Bit, TII);
    } 
    // Jump to label or value in register.
    if (RetOpcode == X86::TCRETURNdi|| RetOpcode == X86::TCRETURNdi64)
      BuildMI(MBB, MBBI, TII.get(X86::TAILJMPd)).
        addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset());
    else if (RetOpcode== X86::TCRETURNri64) {
      BuildMI(MBB, MBBI, TII.get(X86::TAILJMPr64), JumpTarget.getReg());
    } else
       BuildMI(MBB, MBBI, TII.get(X86::TAILJMPr), JumpTarget.getReg());
    // Delete the pseudo instruction TCRETURN.
    MBB.erase(MBBI);
  } else if ((RetOpcode == X86::RET || RetOpcode == X86::RETI) && 
             (X86FI->getTCReturnAddrDelta() < 0)) {
    // Add the return addr area delta back since we are not tail calling.
    int delta = -1*X86FI->getTCReturnAddrDelta();
    MBBI = prior(MBB.end());
    // Check for possible merge with preceeding ADD instruction.
    delta += mergeSPUpdates(MBB, MBBI, StackPtr, true);
    emitSPUpdate(MBB, MBBI, StackPtr, delta, Is64Bit, TII);
  }
}

unsigned X86RegisterInfo::getRARegister() const {
  if (Is64Bit)
    return X86::RIP;  // Should have dwarf #16
  else
    return X86::EIP;  // Should have dwarf #8
}

unsigned X86RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? FramePtr : StackPtr;
}

void X86RegisterInfo::getInitialFrameState(std::vector<MachineMove> &Moves)
                                                                         const {
  // Calculate amount of bytes used for return address storing
  int stackGrowth = (Is64Bit ? -8 : -4);

  // Initial state of the frame pointer is esp+4.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(StackPtr, stackGrowth);
  Moves.push_back(MachineMove(0, Dst, Src));

  // Add return address to move list
  MachineLocation CSDst(StackPtr, stackGrowth);
  MachineLocation CSSrc(getRARegister());
  Moves.push_back(MachineMove(0, CSDst, CSSrc));
}

unsigned X86RegisterInfo::getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned X86RegisterInfo::getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

namespace llvm {
unsigned getX86SubSuperRegister(unsigned Reg, MVT::ValueType VT, bool High) {
  switch (VT) {
  default: return Reg;
  case MVT::i8:
    if (High) {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AH;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DH;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CH;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BH;
      }
    } else {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AL;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DL;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CL;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BL;
      case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
        return X86::SIL;
      case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
        return X86::DIL;
      case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
        return X86::BPL;
      case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
        return X86::SPL;
      case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
        return X86::R8B;
      case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
        return X86::R9B;
      case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
        return X86::R10B;
      case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
        return X86::R11B;
      case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
        return X86::R12B;
      case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
        return X86::R13B;
      case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
        return X86::R14B;
      case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
        return X86::R15B;
      }
    }
  case MVT::i16:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::AX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::DX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::CX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::BX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::SI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::DI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::BP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::SP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8W;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9W;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10W;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11W;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12W;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13W;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14W;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15W;
    }
  case MVT::i32:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::EAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::EDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::ECX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::EBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::ESI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::EDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::EBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::ESP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8D;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9D;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10D;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11D;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12D;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13D;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14D;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15D;
    }
  case MVT::i64:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::RAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::RDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::RCX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::RBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::RSI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::RDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::RBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::RSP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15;
    }
  }

  return Reg;
}
}

#include "X86GenRegisterInfo.inc"


//===-- X86.h - Top-level interface for X86 representation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the x86
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_X86_H
#define TARGET_X86_H

#include <iosfwd>

namespace llvm {

class TargetMachine;
class PassManager;
class FunctionPass;
class IntrinsicLowering;
class MachineCodeEmitter;

enum X86VectorEnum {
  NoSSE, SSE, SSE2, SSE3
};

extern X86VectorEnum X86Vector;
extern bool X86ScalarSSE;

/// createX86ISelPattern - This pass converts an LLVM function into a 
/// machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *createX86ISelPattern(TargetMachine &TM);

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *createX86ISelDag(TargetMachine &TM);

/// createX86SSAPeepholeOptimizerPass - Create a pass to perform SSA-based X86
/// specific peephole optimizations.
///
FunctionPass *createX86SSAPeepholeOptimizerPass();

/// createX86PeepholeOptimizer - Create a pass to perform X86 specific peephole
/// optimizations.
///
FunctionPass *createX86PeepholeOptimizerPass();

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
FunctionPass *createX86FloatingPointStackifierPass();

/// createX86CodePrinterPass - Returns a pass that prints the X86
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.
///
FunctionPass *createX86CodePrinterPass(std::ostream &o, TargetMachine &tm);

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *createX86CodeEmitterPass(MachineCodeEmitter &MCE);

/// addX86ELFObjectWriterPass - Add passes to the FPM that output the generated
/// code as an ELF object file.
///
void addX86ELFObjectWriterPass(PassManager &FPM,
                               std::ostream &o, TargetMachine &tm);

/// createX86EmitCodeToMemory - Returns a pass that converts a register
/// allocated function into raw machine code in a dynamically
/// allocated chunk of memory.
///
FunctionPass *createEmitX86CodeToMemory();

} // End llvm namespace

// Defines symbolic names for X86 registers.  This defines a mapping from
// register name to register number.
//
#include "X86GenRegisterNames.inc"

// Defines symbolic names for the X86 instructions.
//
#include "X86GenInstrNames.inc"

#endif

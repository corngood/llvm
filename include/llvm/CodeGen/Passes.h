//===-- Passes.h - Target independent code generation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code generation
// passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PASSES_H
#define LLVM_CODEGEN_PASSES_H

#include <iosfwd>
#include <string>

namespace llvm {

  class FunctionPass;
  class PassInfo;
  class TargetMachine;

  /// createUnreachableBlockEliminationPass - The LLVM code generator does not
  /// work well with unreachable basic blocks (what live ranges make sense for a
  /// block that cannot be reached?).  As such, a code generator should either
  /// not instruction select unreachable blocks, or it can run this pass as it's
  /// last LLVM modifying pass to clean up blocks that are not reachable from
  /// the entry block.
  FunctionPass *createUnreachableBlockEliminationPass();

  /// MachineFunctionPrinter pass - This pass prints out the machine function to
  /// standard error, as a debugging tool.
  FunctionPass *createMachineFunctionPrinterPass(std::ostream *OS,
                                                 const std::string &Banner ="");

  /// PHIElimination pass - This pass eliminates machine instruction PHI nodes
  /// by inserting copy instructions.  This destroys SSA information, but is the
  /// desired input for some register allocators.  This pass is "required" by
  /// these register allocator like this: AU.addRequiredID(PHIEliminationID);
  ///
  extern const PassInfo *PHIEliminationID;

  /// TwoAddressInstruction pass - This pass reduces two-address instructions to
  /// use two operands. This destroys SSA information but it is desired by
  /// register allocators.
  extern const PassInfo *TwoAddressInstructionPassID;

  /// Creates a register allocator as the user specified on the command line.
  ///
  FunctionPass *createRegisterAllocator();

  /// SimpleRegisterAllocation Pass - This pass converts the input machine code
  /// from SSA form to use explicit registers by spilling every register.  Wow,
  /// great policy huh?
  ///
  FunctionPass *createSimpleRegisterAllocator();

  /// LocalRegisterAllocation Pass - This pass register allocates the input code
  /// a basic block at a time, yielding code better than the simple register
  /// allocator, but not as good as a global allocator.
  ///
  FunctionPass *createLocalRegisterAllocator();

  /// LinearScanRegisterAllocation Pass - This pass implements the linear scan
  /// register allocation algorithm, a global register allocator.
  ///
  FunctionPass *createLinearScanRegisterAllocator();

  /// IterativeScanRegisterAllocation Pass - This pass implements the iterative
  /// scan register allocation algorithm, a global register allocator.
  ///
  FunctionPass *createIterativeScanRegisterAllocator();

  /// PrologEpilogCodeInserter Pass - This pass inserts prolog and epilog code,
  /// and eliminates abstract frame references.
  ///
  FunctionPass *createPrologEpilogCodeInserter();

  /// BranchFolding Pass - This pass performs machine code CFG based
  /// optimizations to delete branches to branches, eliminate branches to
  /// successor blocks (creating fall throughs), and eliminating branches over
  /// branches.
  FunctionPass *createBranchFoldingPass();

  /// MachineCodeDeletion Pass - This pass deletes all of the machine code for
  /// the current function, which should happen after the function has been
  /// emitted to a .s file or to memory.
  FunctionPass *createMachineCodeDeleter();

  /// getRegisterAllocator - This creates an instance of the register allocator
  /// for the Sparc.
  FunctionPass *getRegisterAllocator(TargetMachine &T);

  //createModuloSchedulingPass - Creates the Swing Modulo Scheduling Pass
  FunctionPass *createModuloSchedulingPass(TargetMachine & targ);

  //createModuloSchedulingPass - Creates the Swing Modulo Scheduling Pass
  FunctionPass *createModuloSchedulingSBPass(TargetMachine & targ);

} // End llvm namespace

#endif

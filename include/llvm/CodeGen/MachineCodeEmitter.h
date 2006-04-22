//===-- llvm/CodeGen/MachineCodeEmitter.h - Code emission -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an abstract interface that is used by the machine code
// emission framework to output the code.  This allows machine code emission to
// be separated from concerns such as resolution of call targets, and where the
// machine code will be written (memory or disk, f.e.).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECODEEMITTER_H
#define LLVM_CODEGEN_MACHINECODEEMITTER_H

#include "llvm/Support/DataTypes.h"
#include <map>

namespace llvm {

class MachineBasicBlock;
class MachineConstantPool;
class MachineJumpTableInfo;
class MachineFunction;
class MachineRelocation;
class Value;
class GlobalValue;
class Function;

class MachineCodeEmitter {
public:
  virtual ~MachineCodeEmitter() {}

  /// startFunction - This callback is invoked when the specified function is
  /// about to be code generated.
  ///
  virtual void startFunction(MachineFunction &F) {}

  /// finishFunction - This callback is invoked when the specified function has
  /// finished code generation.
  ///
  virtual void finishFunction(MachineFunction &F) {}

  /// emitConstantPool - This callback is invoked to output the constant pool
  /// for the function.
  virtual void emitConstantPool(MachineConstantPool *MCP) {}

  /// initJumpTableInfo - This callback is invoked by the JIT to allocate the
  /// necessary memory to hold the jump tables.
  virtual void initJumpTableInfo(MachineJumpTableInfo *MJTI) {}
  
  /// emitJumpTableInfo - This callback is invoked to output the jump tables
  /// for the function.  In addition to a pointer to the MachineJumpTableInfo,
  /// this function also takes a map of MBBs to addresses, so that the final
  /// addresses of the MBBs can be written to the jump tables.
  virtual void emitJumpTableInfo(MachineJumpTableInfo *MJTI,
                                 std::map<MachineBasicBlock*,uint64_t> &MBBM) {}
  
  /// startFunctionStub - This callback is invoked when the JIT needs the
  /// address of a function that has not been code generated yet.  The StubSize
  /// specifies the total size required by the stub.  Stubs are not allowed to
  /// have constant pools, the can only use the other emit* methods.
  ///
  virtual void startFunctionStub(unsigned StubSize) {}

  /// finishFunctionStub - This callback is invoked to terminate a function
  /// stub.
  ///
  virtual void *finishFunctionStub(const Function *F) { return 0; }

  /// emitByte - This callback is invoked when a byte needs to be written to the
  /// output stream.
  ///
  virtual void emitByte(unsigned char B) {}

  /// emitWordAt - This callback is invoked when a word needs to be written to
  /// the output stream at a different position than the current PC (for
  /// instance, when performing relocations).
  ///
  virtual void emitWordAt(unsigned W, unsigned *Ptr) {}

  /// emitWord - This callback is invoked when a word needs to be written to the
  /// output stream.
  ///
  virtual void emitWord(unsigned W) = 0;

  /// getCurrentPCValue - This returns the address that the next emitted byte
  /// will be output to.
  ///
  virtual uint64_t getCurrentPCValue() = 0;


  /// getCurrentPCOffset - Return the offset from the start of the emitted
  /// buffer that we are currently writing to.
  virtual uint64_t getCurrentPCOffset() = 0;

  /// addRelocation - Whenever a relocatable address is needed, it should be
  /// noted with this interface.
  virtual void addRelocation(const MachineRelocation &MR) = 0;

  // getConstantPoolEntryAddress - Return the address of the 'Index' entry in
  // the constant pool that was last emitted with the 'emitConstantPool' method.
  //
  virtual uint64_t getConstantPoolEntryAddress(unsigned Index) = 0;

  // getJumpTablelEntryAddress - Return the address of the jump table with index
  // 'Index' in the function that last called initJumpTableInfo.
  //
  virtual uint64_t getJumpTableEntryAddress(unsigned Index) = 0;
  
  // allocateGlobal - Allocate some space for a global variable.
  virtual unsigned char* allocateGlobal(unsigned size, unsigned alignment) = 0;

  /// createDebugEmitter - Return a dynamically allocated machine
  /// code emitter, which just prints the opcodes and fields out the cout.  This
  /// can be used for debugging users of the MachineCodeEmitter interface.
  ///
  static MachineCodeEmitter *createDebugEmitter();

  /// createFilePrinterEmitter - Return a dynamically allocated
  /// machine code emitter, which prints binary code to a file.  This
  /// can be used for debugging users of the MachineCodeEmitter interface.
  ///
  static MachineCodeEmitter *createFilePrinterEmitter(MachineCodeEmitter&);
};

} // End llvm namespace

#endif

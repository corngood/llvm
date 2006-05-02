//===-- MachineCodeEmitter.cpp - Implement the MachineCodeEmitter itf -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachineCodeEmitter interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCodeEmitter.h"
#include <fstream>
#include <iostream>
using namespace llvm;

namespace {
  class FilePrinterEmitter : public MachineCodeEmitter {
    std::ofstream actual;
    std::ostream &o;
    MachineCodeEmitter &MCE;
    unsigned counter;
    unsigned values[4];

  public:
    FilePrinterEmitter(MachineCodeEmitter &M, std::ostream &os)
      : o(os), MCE(M), counter(0) {
      openActual();
    }

    ~FilePrinterEmitter() {
      o << "\n";
      actual.close();
    }

    void openActual() {
      actual.open("lli.actual.obj");
      if (!actual.good()) {
        std::cerr << "Cannot open 'lli.actual.obj' for writing\n";
        abort();
      }
    }

    void startFunction(MachineFunction &F) {
      // resolve any outstanding calls
      MCE.startFunction(F);
    }
    void finishFunction(MachineFunction &F) {
      MCE.finishFunction(F);
    }

    void emitConstantPool(MachineConstantPool *MCP) {
      MCE.emitConstantPool(MCP);
    }
    void initJumpTableInfo(MachineJumpTableInfo *MJTI) {
      MCE.initJumpTableInfo(MJTI);
    }
    void emitJumpTableInfo(MachineJumpTableInfo *MJTI,
                           std::map<MachineBasicBlock*,uint64_t> &MBBM) {
      MCE.emitJumpTableInfo(MJTI, MBBM);
    }
    
    void startFunctionStub(unsigned StubSize) {
      MCE.startFunctionStub(StubSize);
    }

    void *finishFunctionStub(const Function *F) {
      return MCE.finishFunctionStub(F);
    }

    void emitByte(unsigned char B) {
      MCE.emitByte(B);
      actual << B; actual.flush();

      values[counter] = (unsigned int) B;
      if (++counter % 4 == 0 && counter != 0) {
        o << std::hex;
        for (unsigned i=0; i<4; ++i) {
          if (values[i] < 16) o << "0";
          o << values[i] << " ";
        }

        o << std::dec << "\t";
        for (unsigned i=0; i<4; ++i) {
          for (int j=7; j>=0; --j) {
            o << ((values[i] >> j) & 1);
          }
          o << " ";
        }

        o << "\n";

        unsigned instr = 0;
        for (unsigned i=0; i<4; ++i)
          instr |= values[i] << (i*8);

        o << "--- * --- * --- * --- * ---\n";
        counter %= 4;
      }
    }

    void emitWord(unsigned W) {
      MCE.emitWord(W);
    }
    uint64_t getConstantPoolEntryAddress(unsigned Num) {
      return MCE.getConstantPoolEntryAddress(Num);
    }
    uint64_t getJumpTableEntryAddress(unsigned Num) {
      return MCE.getJumpTableEntryAddress(Num);
    }
    virtual unsigned char* allocateGlobal(unsigned size, unsigned alignment)
    { return MCE.allocateGlobal(size, alignment); }

    uint64_t getCurrentPCValue() {
      return MCE.getCurrentPCValue();
    }
    uint64_t getCurrentPCOffset() {
      return MCE.getCurrentPCOffset();
    }
    void addRelocation(const MachineRelocation &MR) {
      return MCE.addRelocation(MR);
    }
  };
}

MachineCodeEmitter *
MachineCodeEmitter::createFilePrinterEmitter(MachineCodeEmitter &MCE) {
  return new FilePrinterEmitter(MCE, std::cerr);
}

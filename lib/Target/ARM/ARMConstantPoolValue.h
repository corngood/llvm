//===- ARMConstantPoolValue.h - ARM constantpool value ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM_CONSTANTPOOLVALUE_H
#define LLVM_TARGET_ARM_CONSTANTPOOLVALUE_H

#include "llvm/CodeGen/MachineConstantPool.h"

namespace llvm {

namespace ARMCP {
  enum ARMCPKind {
    CPValue,
    CPNonLazyPtr,
    CPStub
  };
}

/// ARMConstantPoolValue - ARM specific constantpool value. This is used to
/// represent PC relative displacement between the address of the load
/// instruction and the global value being loaded, i.e. (&GV-(LPIC+8)).
class ARMConstantPoolValue : public MachineConstantPoolValue {
  GlobalValue *GV;         // GlobalValue being loaded.
  const char *S;           // ExtSymbol being loaded.
  unsigned LabelId;        // Label id of the load.
  ARMCP::ARMCPKind Kind;   // non_lazy_ptr or stub?
  unsigned char PCAdjust;  // Extra adjustment if constantpool is pc relative.
                           // 8 for ARM, 4 for Thumb.

public:
  ARMConstantPoolValue(GlobalValue *gv, unsigned id,
                       ARMCP::ARMCPKind Kind = ARMCP::CPValue,
                       unsigned char PCAdj = 0);
  ARMConstantPoolValue(const char *s, unsigned id,
                       ARMCP::ARMCPKind Kind = ARMCP::CPValue,
                       unsigned char PCAdj = 0);

  GlobalValue *getGV() const { return GV; }
  const char *getSymbol() const { return S; }
  unsigned getLabelId() const { return LabelId; }
  bool isNonLazyPointer() const { return Kind == ARMCP::CPNonLazyPtr; }
  bool isStub() const { return Kind == ARMCP::CPStub; }
  unsigned char getPCAdjustment() const { return PCAdjust; }

  virtual int getExistingMachineCPValue(MachineConstantPool *CP,
                                        unsigned Alignment);

  virtual void AddSelectionDAGCSEId(FoldingSetNodeID &ID);

  virtual void print(std::ostream &O) const;
};
  
}

#endif

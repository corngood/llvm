//===- PIC16ConstantPoolValue.cpp - PIC16 constantpool value --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PIC16 specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#include "PIC16ConstantPoolValue.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/GlobalValue.h"
#include "llvm/Type.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

PIC16ConstantPoolValue::PIC16ConstantPoolValue(GlobalValue *gv, unsigned id,
                                               PIC16CP::PIC16CPKind k,
                                               unsigned char PCAdj,
                                               const char *Modif, bool AddCA)
  : MachineConstantPoolValue((const Type*)gv->getType()),
    GV(gv), S(NULL), LabelId(id), Kind(k), PCAdjust(PCAdj),
    Modifier(Modif), AddCurrentAddress(AddCA) {}

PIC16ConstantPoolValue::PIC16ConstantPoolValue(const char *s, unsigned id,
                                               PIC16CP::PIC16CPKind k,
                                               unsigned char PCAdj,
                                               const char *Modif, bool AddCA)
  : MachineConstantPoolValue((const Type*)Type::Int32Ty),
    GV(NULL), S(s), LabelId(id), Kind(k), PCAdjust(PCAdj),
    Modifier(Modif), AddCurrentAddress(AddCA) {}

PIC16ConstantPoolValue::PIC16ConstantPoolValue(GlobalValue *gv,
                                               PIC16CP::PIC16CPKind k,
                                               const char *Modif)
  : MachineConstantPoolValue((const Type*)Type::Int32Ty),
    GV(gv), S(NULL), LabelId(0), Kind(k), PCAdjust(0),
    Modifier(Modif) {}

int PIC16ConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                      unsigned Alignment) {
  unsigned AlignMask = (1 << Alignment)-1;
  const std::vector<MachineConstantPoolEntry> Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        (Constants[i].Offset & AlignMask) == 0) {
      PIC16ConstantPoolValue *CPV =
        (PIC16ConstantPoolValue *)Constants[i].Val.MachineCPVal;
      if (CPV->GV == GV &&
          CPV->S == S &&
          CPV->LabelId == LabelId &&
          CPV->Kind == Kind &&
          CPV->PCAdjust == PCAdjust)
        return i;
    }
  }

  return -1;
}

void
PIC16ConstantPoolValue::AddSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(GV);
  ID.AddPointer(S);
  ID.AddInteger(LabelId);
  ID.AddInteger((unsigned)Kind);
  ID.AddInteger(PCAdjust);
}

void PIC16ConstantPoolValue::print(raw_ostream &O) const {
  if (GV)
    O << GV->getName();
  else
    O << S;
  if (isNonLazyPointer()) O << "$non_lazy_ptr";
  else if (isStub()) O << "$stub";
  if (Modifier) O << "(" << Modifier << ")";
  if (PCAdjust != 0) {
    O << "-(LPIC" << LabelId << "+"
      << (unsigned)PCAdjust;
    if (AddCurrentAddress)
      O << "-.";
    O << ")";
  }
}

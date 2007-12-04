//===-- SPUHazardRecognizers.h - Cell SPU Hazard Recognizer -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by a team from the Computer Systems Research
// Department at The Aerospace Corporation.
//
// See README.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling on the Cell SPU
// processor.
//
//===----------------------------------------------------------------------===//

#ifndef SPUHAZRECS_H
#define SPUHAZRECS_H

#include "llvm/CodeGen/ScheduleDAG.h"
#include "SPUInstrInfo.h"

namespace llvm {
  
/// SPUHazardRecognizer
class SPUHazardRecognizer : public HazardRecognizer
{
private:
  const TargetInstrInfo &TII;
  int EvenOdd;

public:
  SPUHazardRecognizer(const TargetInstrInfo &TII);
  virtual HazardType getHazardType(SDNode *Node);
  virtual void EmitInstruction(SDNode *Node);
  virtual void AdvanceCycle();
  virtual void EmitNoop();
};

} // end namespace llvm

#endif


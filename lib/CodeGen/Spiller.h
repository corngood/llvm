//===-- llvm/CodeGen/Spiller.h - Spiller -*- C++ -*------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPILLER_H
#define LLVM_CODEGEN_SPILLER_H

#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm {

  class LiveInterval;
  class LiveIntervals;
  class LiveStacks;
  class MachineFunction;
  class MachineInstr;
  class MachineLoopInfo;
  class SlotIndex;
  class VirtRegMap;
  class VNInfo;

  /// Spiller interface.
  ///
  /// Implementations are utility classes which insert spill or remat code on
  /// demand.
  class Spiller {
  public:
    virtual ~Spiller() = 0;

    /// spill - Spill the given live interval. The method used will depend on
    /// the Spiller implementation selected.
    ///
    /// @param li            The live interval to be spilled.
    /// @param spillIs       An essential hook into the register allocator guts
    ///                      that perhaps serves a purpose(?!)
    /// @param newIntervals  The newly created intervals will be appended here.
    /// @param earliestIndex The earliest point for splitting. (OK, it's another
    ///                      pointer to the allocator guts).
    virtual void spill(LiveInterval *li,
                       std::vector<LiveInterval*> &newIntervals,
                       SmallVectorImpl<LiveInterval*> &spillIs,
                       SlotIndex *earliestIndex = 0) = 0;

  };

  /// Create and return a spiller object, as specified on the command line.
  Spiller* createSpiller(MachineFunction *mf, LiveIntervals *li,
                         const MachineLoopInfo *loopInfo, VirtRegMap *vrm);
}

#endif

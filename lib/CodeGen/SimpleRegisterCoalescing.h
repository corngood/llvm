//===-- SimpleRegisterCoalescing.h - Register Coalescing --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register copy coalescing phase.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H
#define LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IndexedMap.h"
#include <queue>

namespace llvm {
  class SimpleRegisterCoalescing;
  class LiveVariables;
  class TargetRegisterInfo;
  class TargetInstrInfo;
  class VirtRegMap;
  class MachineLoopInfo;

  /// CopyRec - Representation for copy instructions in coalescer queue.
  ///
  struct CopyRec {
    MachineInstr *MI;
    unsigned SrcReg, DstReg;
    unsigned LoopDepth;
    bool isBackEdge;
    CopyRec(MachineInstr *mi, unsigned src, unsigned dst, unsigned depth,
            bool be)
      : MI(mi), SrcReg(src), DstReg(dst), LoopDepth(depth), isBackEdge(be) {};
  };

  template<class SF> class JoinPriorityQueue;

  /// CopyRecSort - Sorting function for coalescer queue.
  ///
  struct CopyRecSort : public std::binary_function<CopyRec,CopyRec,bool> {
    JoinPriorityQueue<CopyRecSort> *JPQ;
    explicit CopyRecSort(JoinPriorityQueue<CopyRecSort> *jpq) : JPQ(jpq) {}
    CopyRecSort(const CopyRecSort &RHS) : JPQ(RHS.JPQ) {}
    bool operator()(CopyRec left, CopyRec right) const;
  };

  /// JoinQueue - A priority queue of copy instructions the coalescer is
  /// going to process.
  template<class SF>
  class JoinPriorityQueue {
    SimpleRegisterCoalescing *Rc;
    std::priority_queue<CopyRec, std::vector<CopyRec>, SF> Queue;

  public:
    explicit JoinPriorityQueue(SimpleRegisterCoalescing *rc)
      : Rc(rc), Queue(SF(this)) {}

    bool empty() const { return Queue.empty(); }
    void push(CopyRec R) { Queue.push(R); }
    CopyRec pop() {
      if (empty()) return CopyRec(0, 0, 0, 0, false);
      CopyRec R = Queue.top();
      Queue.pop();
      return R;
    }

    // Callbacks to SimpleRegisterCoalescing.
    unsigned getRepIntervalSize(unsigned Reg);
  };

  class SimpleRegisterCoalescing : public MachineFunctionPass,
                                   public RegisterCoalescer {
    MachineFunction* mf_;
    const MachineRegisterInfo* mri_;
    const TargetMachine* tm_;
    const TargetRegisterInfo* tri_;
    const TargetInstrInfo* tii_;
    LiveIntervals *li_;
    LiveVariables *lv_;
    const MachineLoopInfo* loopInfo;
    
    BitVector allocatableRegs_;
    DenseMap<const TargetRegisterClass*, BitVector> allocatableRCRegs_;

    /// r2rMap_ - Map from register to its representative register.
    ///
    IndexedMap<unsigned> r2rMap_;

    /// r2rRevMap_ - Reverse of r2rRevMap_, i.e. Map from register to all
    /// the registers it represent.
    IndexedMap<std::vector<unsigned> > r2rRevMap_;

    /// JoinQueue - A priority queue of copy instructions the coalescer is
    /// going to process.
    JoinPriorityQueue<CopyRecSort> *JoinQueue;

    /// JoinedLIs - Keep track which register intervals have been coalesced
    /// with other intervals.
    BitVector JoinedLIs;

    /// SubRegIdxes - Keep track of sub-register and indexes.
    ///
    SmallVector<std::pair<unsigned, unsigned>, 32> SubRegIdxes;

    /// JoinedCopies - Keep track of copies eliminated due to coalescing.
    ///
    SmallPtrSet<MachineInstr*, 32> JoinedCopies;

    /// ChangedCopies - Keep track of copies modified due to commuting.
    SmallPtrSet<MachineInstr*, 32> ChangedCopies;

  public:
    static char ID; // Pass identifcation, replacement for typeid
    SimpleRegisterCoalescing() : MachineFunctionPass((intptr_t)&ID) {}

    struct InstrSlots {
      enum {
        LOAD  = 0,
        USE   = 1,
        DEF   = 2,
        STORE = 3,
        NUM   = 4
      };
    };
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    bool coalesceFunction(MachineFunction &mf, RegallocQuery &) {
      // This runs as an independent pass, so don't do anything.
      return false;
    };

    /// getRepIntervalSize - Called from join priority queue sorting function.
    /// It returns the size of the interval that represent the given register.
    unsigned getRepIntervalSize(unsigned Reg) {
      Reg = rep(Reg);
      if (!li_->hasInterval(Reg))
        return 0;
      return li_->getInterval(Reg).getSize();
    }

    /// print - Implement the dump method.
    virtual void print(std::ostream &O, const Module* = 0) const;
    void print(std::ostream *O, const Module* M = 0) const {
      if (O) print(*O, M);
    }

  private:
    /// joinIntervals - join compatible live intervals
    void joinIntervals();

    /// CopyCoalesceInMBB - Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into the "TryAgain" list.
    void CopyCoalesceInMBB(MachineBasicBlock *MBB,
                           std::vector<CopyRec> &TryAgain);

    /// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns true
    /// if the copy was successfully coalesced away. If it is not currently
    /// possible to coalesce this interval, but it may be possible if other
    /// things get coalesced, then it returns true by reference in 'Again'.
    bool JoinCopy(CopyRec &TheCopy, bool &Again);
    
    /// JoinIntervals - Attempt to join these two intervals.  On failure, this
    /// returns false.  Otherwise, if one of the intervals being joined is a
    /// physreg, this method always canonicalizes DestInt to be it.  The output
    /// "SrcInt" will not have been modified, so we can use this information
    /// below to update aliases.
    bool JoinIntervals(LiveInterval &LHS, LiveInterval &RHS, bool &Swapped);
    
    /// SimpleJoin - Attempt to join the specified interval into this one. The
    /// caller of this method must guarantee that the RHS only contains a single
    /// value number and that the RHS is not defined by a copy from this
    /// interval.  This returns false if the intervals are not joinable, or it
    /// joins them and returns true.
    bool SimpleJoin(LiveInterval &LHS, LiveInterval &RHS);
    
    /// Return true if the two specified registers belong to different
    /// register classes.  The registers may be either phys or virt regs.
    bool differingRegisterClasses(unsigned RegA, unsigned RegB) const;


    bool AdjustCopiesBackFrom(LiveInterval &IntA, LiveInterval &IntB,
                              MachineInstr *CopyMI);

    bool RemoveCopyByCommutingDef(LiveInterval &IntA, LiveInterval &IntB,
                                  MachineInstr *CopyMI);

    /// AddSubRegIdxPairs - Recursively mark all the registers represented by the
    /// specified register as sub-registers. The recursion level is expected to be
    /// shallow.
    void AddSubRegIdxPairs(unsigned Reg, unsigned SubIdx);

    /// isBackEdgeCopy - Returns true if CopyMI is a back edge copy.
    ///
    bool isBackEdgeCopy(MachineInstr *CopyMI, unsigned DstReg);

    /// lastRegisterUse - Returns the last use of the specific register between
    /// cycles Start and End. It also returns the use operand by reference. It
    /// returns NULL if there are no uses.
    MachineInstr *lastRegisterUse(unsigned Start, unsigned End, unsigned Reg,
                                  MachineOperand *&MOU);

    /// findDefOperand - Returns the MachineOperand that is a def of the specific
    /// register. It returns NULL if the def is not found.
    MachineOperand *findDefOperand(MachineInstr *MI, unsigned Reg);

    /// unsetRegisterKill - Unset IsKill property of all uses of the specific
    /// register of the specific instruction.
    void unsetRegisterKill(MachineInstr *MI, unsigned Reg);

    /// unsetRegisterKills - Unset IsKill property of all uses of specific register
    /// between cycles Start and End.
    void unsetRegisterKills(unsigned Start, unsigned End, unsigned Reg);

    /// hasRegisterDef - True if the instruction defines the specific register.
    ///
    bool hasRegisterDef(MachineInstr *MI, unsigned Reg);

    /// rep - returns the representative of this register
    unsigned rep(unsigned Reg) {
      unsigned Rep = r2rMap_[Reg];
      if (Rep)
        return r2rMap_[Reg] = rep(Rep);
      return Reg;
    }

    void printRegName(unsigned reg) const;
  };

} // End llvm namespace

#endif

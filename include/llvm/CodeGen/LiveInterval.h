//===-- llvm/CodeGen/LiveInterval.h - Interval representation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveRange and LiveInterval classes.  Given some
// numbering of each the machine instructions an interval [i, j) is said to be a
// live interval for register v if there is no instruction with number j' > j
// such that v is live at j' and there is no instruction with number i' < i such
// that v is live at i'. In this implementation intervals can have holes,
// i.e. an interval might look like [1,20), [50,65), [1000,1001).  Each
// individual range is represented as an instance of LiveRange, and the whole
// interval is represented as an instance of LiveInterval.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_H
#define LLVM_CODEGEN_LIVEINTERVAL_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Streams.h"
#include <iosfwd>
#include <vector>
#include <cassert>

namespace llvm {
  class MachineInstr;
  class MRegisterInfo;
  struct LiveInterval;

  /// VNInfo - If the value number definition is undefined (e.g. phi
  /// merge point), it contains ~0u,x. If the value number is not in use, it
  /// contains ~1u,x to indicate that the value # is not used. 
  ///   parent- LiveInterval parent.
  ///   def   - Instruction # of the definition.
  ///   reg   - Source reg iff val# is defined by a copy; zero otherwise.
  ///   kills - Instruction # of the kills. If a kill is an odd #, it means
  ///           the kill is a phi join point.
  struct VNInfo {
    LiveInterval *parent;
    unsigned id;
    unsigned def;
    unsigned reg;
    SmallVector<unsigned, 4> kills;
    VNInfo() : parent(0), id(~1U), def(~1U), reg(0) {}
    VNInfo(LiveInterval *p, unsigned i, unsigned d, unsigned r)
      : parent(p), id(i), def(d), reg(r) {}
  };

  /// LiveRange structure - This represents a simple register range in the
  /// program, with an inclusive start point and an exclusive end point.
  /// These ranges are rendered as [start,end).
  struct LiveRange {
    unsigned start;  // Start point of the interval (inclusive)
    unsigned end;    // End point of the interval (exclusive)
    VNInfo *valno;   // identifier for the value contained in this interval.

    LiveRange(unsigned S, unsigned E, VNInfo *V) : start(S), end(E), valno(V) {
      assert(S < E && "Cannot create empty or backwards range");
    }

    /// contains - Return true if the index is covered by this range.
    ///
    bool contains(unsigned I) const {
      return start <= I && I < end;
    }

    bool operator<(const LiveRange &LR) const {
      return start < LR.start || (start == LR.start && end < LR.end);
    }
    bool operator==(const LiveRange &LR) const {
      return start == LR.start && end == LR.end;
    }

    void dump() const;
    void print(std::ostream &os) const;
    void print(std::ostream *os) const { if (os) print(*os); }

  private:
    LiveRange(); // DO NOT IMPLEMENT
  };

  std::ostream& operator<<(std::ostream& os, const LiveRange &LR);


  inline bool operator<(unsigned V, const LiveRange &LR) {
    return V < LR.start;
  }

  inline bool operator<(const LiveRange &LR, unsigned V) {
    return LR.start < V;
  }

  /// LiveInterval - This class represents some number of live ranges for a
  /// register or value.  This class also contains a bit of register allocator
  /// state.
  struct LiveInterval {
    typedef SmallVector<LiveRange,4> Ranges;
    typedef SmallVector<VNInfo*,4> VNInfoList;

    unsigned reg;        // the register of this interval
    unsigned preference; // preferred register to allocate for this interval
    float weight;        // weight of this interval
    Ranges ranges;       // the ranges in which this register is live
    unsigned numvals;    // number of value#'s
    VNInfoList valnos;   // value#'s

  public:
    LiveInterval(unsigned Reg, float Weight)
      : reg(Reg), preference(0), weight(Weight), numvals(0) {
    }

    typedef Ranges::iterator iterator;
    iterator begin() { return ranges.begin(); }
    iterator end()   { return ranges.end(); }

    typedef Ranges::const_iterator const_iterator;
    const_iterator begin() const { return ranges.begin(); }
    const_iterator end() const  { return ranges.end(); }

    typedef VNInfoList::iterator vni_iterator;
    vni_iterator vni_begin() { return valnos.begin(); }
    vni_iterator vni_end() { return valnos.end(); }

    typedef VNInfoList::const_iterator const_vni_iterator;
    const_vni_iterator vni_begin() const { return valnos.begin(); }
    const_vni_iterator vni_end() const { return valnos.end(); }

    ~LiveInterval() {
      for (vni_iterator i = vni_begin(), e = vni_end(); i != e; ++i) {
        VNInfo *VNI = *i;
        if (VNI->parent == this)
          delete VNI;
      }
      valnos.clear();
    }

    /// advanceTo - Advance the specified iterator to point to the LiveRange
    /// containing the specified position, or end() if the position is past the
    /// end of the interval.  If no LiveRange contains this position, but the
    /// position is in a hole, this method returns an iterator pointing the the
    /// LiveRange immediately after the hole.
    iterator advanceTo(iterator I, unsigned Pos) {
      if (Pos >= endNumber())
        return end();
      while (I->end <= Pos) ++I;
      return I;
    }

    bool containsOneValue() const { return numvals == 1; }

    unsigned getNumValNums() const { return numvals; }
    
    /// getFirstValNumInfo - Returns pointer to the first val#.
    ///
    inline VNInfo *getFirstValNumInfo() {
      return valnos.front();
    }
    inline const VNInfo *getFirstValNumInfo() const {
      return valnos.front();
    }
    
    /// copyValNumInfo - Copy the value number info for one value number to
    /// another.
    void copyValNumInfo(VNInfo &DstValNo, VNInfo &SrcValNo) {
      DstValNo.def = SrcValNo.def;
      DstValNo.reg = SrcValNo.reg;
      DstValNo.kills = SrcValNo.kills;
    }

    /// getNextValue - Create a new value number and return it.  MIIdx specifies
    /// the instruction that defines the value number.
    VNInfo *getNextValue(unsigned MIIdx, unsigned SrcReg) {
      VNInfo *VNI = new VNInfo(this, numvals++, MIIdx, SrcReg);
      valnos.push_back(VNI);
      return VNI;
    }

    /// addKillForValNum - Add a kill instruction index to the specified value
    /// number.
    static void addKill(VNInfo &VNI, unsigned KillIdx) {
      SmallVector<unsigned, 4> &kills = VNI.kills;
      if (kills.empty()) {
        kills.push_back(KillIdx);
      } else {
        SmallVector<unsigned, 4>::iterator
          I = std::lower_bound(kills.begin(), kills.end(), KillIdx);
        kills.insert(I, KillIdx);
      }
    }

    /// addKills - Add a number of kills into the VNInfo kill vector. If this
    /// interval is live at a kill point, then the kill is not added.
    void addKills(VNInfo &VNI, const SmallVector<unsigned, 4> &kills) {
      for (unsigned i = 0, e = kills.size(); i != e; ++i) {
        unsigned KillIdx = kills[i];
        if (!liveAt(KillIdx)) {
          SmallVector<unsigned, 4>::iterator
            I = std::lower_bound(VNI.kills.begin(), VNI.kills.end(), KillIdx);
          VNI.kills.insert(I, KillIdx);
        }
      }
    }

    /// removeKill - Remove the specified kill from the list of kills of
    /// the specified val#.
    static bool removeKill(VNInfo &VNI, unsigned KillIdx) {
      SmallVector<unsigned, 4> &kills = VNI.kills;
      SmallVector<unsigned, 4>::iterator
        I = std::lower_bound(kills.begin(), kills.end(), KillIdx);
      if (I != kills.end() && *I == KillIdx) {
        kills.erase(I);
        return true;
      }
      return false;
    }

    /// removeKills - Remove all the kills in specified range
    /// [Start, End] of the specified val#.
    void removeKills(VNInfo &VNI, unsigned Start, unsigned End) {
      SmallVector<unsigned, 4> &kills = VNI.kills;
      SmallVector<unsigned, 4>::iterator
        I = std::lower_bound(kills.begin(), kills.end(), Start);
      SmallVector<unsigned, 4>::iterator
        E = std::upper_bound(kills.begin(), kills.end(), End);
      kills.erase(I, E);
    }
    
    /// MergeValueNumberInto - This method is called when two value nubmers
    /// are found to be equivalent.  This eliminates V1, replacing all
    /// LiveRanges with the V1 value number with the V2 value number.  This can
    /// cause merging of V1/V2 values numbers and compaction of the value space.
    void MergeValueNumberInto(VNInfo *V1, VNInfo *V2);

    /// MergeInClobberRanges - For any live ranges that are not defined in the
    /// current interval, but are defined in the Clobbers interval, mark them
    /// used with an unknown definition value.
    void MergeInClobberRanges(const LiveInterval &Clobbers);

    
    /// MergeRangesInAsValue - Merge all of the intervals in RHS into this live
    /// interval as the specified value number.  The LiveRanges in RHS are
    /// allowed to overlap with LiveRanges in the current interval, but only if
    /// the overlapping LiveRanges have the specified value number.
    void MergeRangesInAsValue(const LiveInterval &RHS, VNInfo *LHSValNo);
    
    bool empty() const { return ranges.empty(); }

    /// beginNumber - Return the lowest numbered slot covered by interval.
    unsigned beginNumber() const {
      assert(!empty() && "empty interval for register");
      return ranges.front().start;
    }

    /// endNumber - return the maximum point of the interval of the whole,
    /// exclusive.
    unsigned endNumber() const {
      assert(!empty() && "empty interval for register");
      return ranges.back().end;
    }

    bool expiredAt(unsigned index) const {
      return index >= endNumber();
    }

    bool liveAt(unsigned index) const;

    /// getLiveRangeContaining - Return the live range that contains the
    /// specified index, or null if there is none.
    const LiveRange *getLiveRangeContaining(unsigned Idx) const {
      const_iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    const_iterator FindLiveRangeContaining(unsigned Idx) const;

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    iterator FindLiveRangeContaining(unsigned Idx);
    
    /// getOverlapingRanges - Given another live interval which is defined as a
    /// copy from this one, return a list of all of the live ranges where the
    /// two overlap and have different value numbers.
    void getOverlapingRanges(const LiveInterval &Other, unsigned CopyIdx,
                             std::vector<LiveRange*> &Ranges);

    /// overlaps - Return true if the intersection of the two live intervals is
    /// not empty.
    bool overlaps(const LiveInterval& other) const {
      return overlapsFrom(other, other.begin());
    }

    /// overlapsFrom - Return true if the intersection of the two live intervals
    /// is not empty.  The specified iterator is a hint that we can begin
    /// scanning the Other interval starting at I.
    bool overlapsFrom(const LiveInterval& other, const_iterator I) const;

    /// addRange - Add the specified LiveRange to this interval, merging
    /// intervals as appropriate.  This returns an iterator to the inserted live
    /// range (which may have grown since it was inserted.
    void addRange(LiveRange LR) {
      addRangeFrom(LR, ranges.begin());
    }

    /// join - Join two live intervals (this, and other) together.  This applies
    /// mappings to the value numbers in the LHS/RHS intervals as specified.  If
    /// the intervals are not joinable, this aborts.
    void join(LiveInterval &Other, int *ValNoAssignments,
              int *RHSValNoAssignments, SmallVector<VNInfo*, 16> &NewVNInfo);

    /// removeRange - Remove the specified range from this interval.  Note that
    /// the range must already be in this interval in its entirety.
    void removeRange(unsigned Start, unsigned End);

    void removeRange(LiveRange LR) {
      removeRange(LR.start, LR.end);
    }

    /// getSize - Returns the sum of sizes of all the LiveRange's.
    ///
    unsigned getSize() const;

    bool operator<(const LiveInterval& other) const {
      return beginNumber() < other.beginNumber();
    }

    void print(std::ostream &OS, const MRegisterInfo *MRI = 0) const;
    void print(std::ostream *OS, const MRegisterInfo *MRI = 0) const {
      if (OS) print(*OS, MRI);
    }
    void dump() const;

  private:
    Ranges::iterator addRangeFrom(LiveRange LR, Ranges::iterator From);
    void extendIntervalEndTo(Ranges::iterator I, unsigned NewEnd);
    Ranges::iterator extendIntervalStartTo(Ranges::iterator I, unsigned NewStr);
    LiveInterval& operator=(const LiveInterval& rhs); // DO NOT IMPLEMENT
  };

  inline std::ostream &operator<<(std::ostream &OS, const LiveInterval &LI) {
    LI.print(OS);
    return OS;
  }
}

#endif

//===---------- SplitKit.cpp - Toolkit for splitting live ranges ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SplitAnalysis class as well as mutator functions for
// live range splitting.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "SplitKit.h"
#include "LiveRangeEdit.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static cl::opt<bool>
AllowSplit("spiller-splits-edges",
           cl::desc("Allow critical edge splitting during spilling"));

//===----------------------------------------------------------------------===//
//                                 Split Analysis
//===----------------------------------------------------------------------===//

SplitAnalysis::SplitAnalysis(const MachineFunction &mf,
                             const LiveIntervals &lis,
                             const MachineLoopInfo &mli)
  : MF(mf),
    LIS(lis),
    Loops(mli),
    TII(*mf.getTarget().getInstrInfo()),
    CurLI(0) {}

void SplitAnalysis::clear() {
  UseSlots.clear();
  UsingInstrs.clear();
  UsingBlocks.clear();
  UsingLoops.clear();
  CurLI = 0;
}

bool SplitAnalysis::canAnalyzeBranch(const MachineBasicBlock *MBB) {
  MachineBasicBlock *T, *F;
  SmallVector<MachineOperand, 4> Cond;
  return !TII.AnalyzeBranch(const_cast<MachineBasicBlock&>(*MBB), T, F, Cond);
}

/// analyzeUses - Count instructions, basic blocks, and loops using CurLI.
void SplitAnalysis::analyzeUses() {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineRegisterInfo::reg_iterator I = MRI.reg_begin(CurLI->reg);
       MachineInstr *MI = I.skipInstruction();) {
    if (MI->isDebugValue() || !UsingInstrs.insert(MI))
      continue;
    UseSlots.push_back(LIS.getInstructionIndex(MI).getDefIndex());
    MachineBasicBlock *MBB = MI->getParent();
    if (UsingBlocks[MBB]++)
      continue;
    for (MachineLoop *Loop = Loops.getLoopFor(MBB); Loop;
         Loop = Loop->getParentLoop())
      UsingLoops[Loop]++;
  }
  array_pod_sort(UseSlots.begin(), UseSlots.end());
  DEBUG(dbgs() << "  counted "
               << UsingInstrs.size() << " instrs, "
               << UsingBlocks.size() << " blocks, "
               << UsingLoops.size()  << " loops.\n");
}

void SplitAnalysis::print(const BlockPtrSet &B, raw_ostream &OS) const {
  for (BlockPtrSet::const_iterator I = B.begin(), E = B.end(); I != E; ++I) {
    unsigned count = UsingBlocks.lookup(*I);
    OS << " BB#" << (*I)->getNumber();
    if (count)
      OS << '(' << count << ')';
  }
}

// Get three sets of basic blocks surrounding a loop: Blocks inside the loop,
// predecessor blocks, and exit blocks.
void SplitAnalysis::getLoopBlocks(const MachineLoop *Loop, LoopBlocks &Blocks) {
  Blocks.clear();

  // Blocks in the loop.
  Blocks.Loop.insert(Loop->block_begin(), Loop->block_end());

  // Predecessor blocks.
  const MachineBasicBlock *Header = Loop->getHeader();
  for (MachineBasicBlock::const_pred_iterator I = Header->pred_begin(),
       E = Header->pred_end(); I != E; ++I)
    if (!Blocks.Loop.count(*I))
      Blocks.Preds.insert(*I);

  // Exit blocks.
  for (MachineLoop::block_iterator I = Loop->block_begin(),
       E = Loop->block_end(); I != E; ++I) {
    const MachineBasicBlock *MBB = *I;
    for (MachineBasicBlock::const_succ_iterator SI = MBB->succ_begin(),
       SE = MBB->succ_end(); SI != SE; ++SI)
      if (!Blocks.Loop.count(*SI))
        Blocks.Exits.insert(*SI);
  }
}

void SplitAnalysis::print(const LoopBlocks &B, raw_ostream &OS) const {
  OS << "Loop:";
  print(B.Loop, OS);
  OS << ", preds:";
  print(B.Preds, OS);
  OS << ", exits:";
  print(B.Exits, OS);
}

/// analyzeLoopPeripheralUse - Return an enum describing how CurLI is used in
/// and around the Loop.
SplitAnalysis::LoopPeripheralUse SplitAnalysis::
analyzeLoopPeripheralUse(const SplitAnalysis::LoopBlocks &Blocks) {
  LoopPeripheralUse use = ContainedInLoop;
  for (BlockCountMap::iterator I = UsingBlocks.begin(), E = UsingBlocks.end();
       I != E; ++I) {
    const MachineBasicBlock *MBB = I->first;
    // Is this a peripheral block?
    if (use < MultiPeripheral &&
        (Blocks.Preds.count(MBB) || Blocks.Exits.count(MBB))) {
      if (I->second > 1) use = MultiPeripheral;
      else               use = SinglePeripheral;
      continue;
    }
    // Is it a loop block?
    if (Blocks.Loop.count(MBB))
      continue;
    // It must be an unrelated block.
    DEBUG(dbgs() << ", outside: BB#" << MBB->getNumber());
    return OutsideLoop;
  }
  return use;
}

/// getCriticalExits - It may be necessary to partially break critical edges
/// leaving the loop if an exit block has predecessors from outside the loop
/// periphery.
void SplitAnalysis::getCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  CriticalExits.clear();

  // A critical exit block has CurLI live-in, and has a predecessor that is not
  // in the loop nor a loop predecessor. For such an exit block, the edges
  // carrying the new variable must be moved to a new pre-exit block.
  for (BlockPtrSet::iterator I = Blocks.Exits.begin(), E = Blocks.Exits.end();
       I != E; ++I) {
    const MachineBasicBlock *Exit = *I;
    // A single-predecessor exit block is definitely not a critical edge.
    if (Exit->pred_size() == 1)
      continue;
    // This exit may not have CurLI live in at all. No need to split.
    if (!LIS.isLiveInToMBB(*CurLI, Exit))
      continue;
    // Does this exit block have a predecessor that is not a loop block or loop
    // predecessor?
    for (MachineBasicBlock::const_pred_iterator PI = Exit->pred_begin(),
         PE = Exit->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      if (Blocks.Loop.count(Pred) || Blocks.Preds.count(Pred))
        continue;
      // This is a critical exit block, and we need to split the exit edge.
      CriticalExits.insert(Exit);
      break;
    }
  }
}

void SplitAnalysis::getCriticalPreds(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalPreds) {
  CriticalPreds.clear();

  // A critical predecessor block has CurLI live-out, and has a successor that
  // has CurLI live-in and is not in the loop nor a loop exit block. For such a
  // predecessor block, we must carry the value in both the 'inside' and
  // 'outside' registers.
  for (BlockPtrSet::iterator I = Blocks.Preds.begin(), E = Blocks.Preds.end();
       I != E; ++I) {
    const MachineBasicBlock *Pred = *I;
    // Definitely not a critical edge.
    if (Pred->succ_size() == 1)
      continue;
    // This block may not have CurLI live out at all if there is a PHI.
    if (!LIS.isLiveOutOfMBB(*CurLI, Pred))
      continue;
    // Does this block have a successor outside the loop?
    for (MachineBasicBlock::const_pred_iterator SI = Pred->succ_begin(),
         SE = Pred->succ_end(); SI != SE; ++SI) {
      const MachineBasicBlock *Succ = *SI;
      if (Blocks.Loop.count(Succ) || Blocks.Exits.count(Succ))
        continue;
      if (!LIS.isLiveInToMBB(*CurLI, Succ))
        continue;
      // This is a critical predecessor block.
      CriticalPreds.insert(Pred);
      break;
    }
  }
}

/// canSplitCriticalExits - Return true if it is possible to insert new exit
/// blocks before the blocks in CriticalExits.
bool
SplitAnalysis::canSplitCriticalExits(const SplitAnalysis::LoopBlocks &Blocks,
                                     BlockPtrSet &CriticalExits) {
  // If we don't allow critical edge splitting, require no critical exits.
  if (!AllowSplit)
    return CriticalExits.empty();

  for (BlockPtrSet::iterator I = CriticalExits.begin(), E = CriticalExits.end();
       I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    // We want to insert a new pre-exit MBB before Succ, and change all the
    // in-loop blocks to branch to the pre-exit instead of Succ.
    // Check that all the in-loop predecessors can be changed.
    for (MachineBasicBlock::const_pred_iterator PI = Succ->pred_begin(),
         PE = Succ->pred_end(); PI != PE; ++PI) {
      const MachineBasicBlock *Pred = *PI;
      // The external predecessors won't be altered.
      if (!Blocks.Loop.count(Pred) && !Blocks.Preds.count(Pred))
        continue;
      if (!canAnalyzeBranch(Pred))
        return false;
    }

    // If Succ's layout predecessor falls through, that too must be analyzable.
    // We need to insert the pre-exit block in the gap.
    MachineFunction::const_iterator MFI = Succ;
    if (MFI == MF.begin())
      continue;
    if (!canAnalyzeBranch(--MFI))
      return false;
  }
  // No problems found.
  return true;
}

void SplitAnalysis::analyze(const LiveInterval *li) {
  clear();
  CurLI = li;
  analyzeUses();
}

void SplitAnalysis::getSplitLoops(LoopPtrSet &Loops) {
  assert(CurLI && "Call analyze() before getSplitLoops");
  if (UsingLoops.empty())
    return;

  LoopBlocks Blocks;
  BlockPtrSet CriticalExits;

  // We split around loops where CurLI is used outside the periphery.
  for (LoopCountMap::const_iterator I = UsingLoops.begin(),
       E = UsingLoops.end(); I != E; ++I) {
    const MachineLoop *Loop = I->first;
    getLoopBlocks(Loop, Blocks);
    DEBUG({ dbgs() << "  "; print(Blocks, dbgs()); });

    switch(analyzeLoopPeripheralUse(Blocks)) {
    case OutsideLoop:
      break;
    case MultiPeripheral:
      // FIXME: We could split a live range with multiple uses in a peripheral
      // block and still make progress. However, it is possible that splitting
      // another live range will insert copies into a peripheral block, and
      // there is a small chance we can enter an infinite loop, inserting copies
      // forever.
      // For safety, stick to splitting live ranges with uses outside the
      // periphery.
      DEBUG(dbgs() << ": multiple peripheral uses");
      break;
    case ContainedInLoop:
      DEBUG(dbgs() << ": fully contained\n");
      continue;
    case SinglePeripheral:
      DEBUG(dbgs() << ": single peripheral use\n");
      continue;
    }
    // Will it be possible to split around this loop?
    getCriticalExits(Blocks, CriticalExits);
    DEBUG(dbgs() << ": " << CriticalExits.size() << " critical exits\n");
    if (!canSplitCriticalExits(Blocks, CriticalExits))
      continue;
    // This is a possible split.
    Loops.insert(Loop);
  }

  DEBUG(dbgs() << "  getSplitLoops found " << Loops.size()
               << " candidate loops.\n");
}

const MachineLoop *SplitAnalysis::getBestSplitLoop() {
  LoopPtrSet Loops;
  getSplitLoops(Loops);
  if (Loops.empty())
    return 0;

  // Pick the earliest loop.
  // FIXME: Are there other heuristics to consider?
  const MachineLoop *Best = 0;
  SlotIndex BestIdx;
  for (LoopPtrSet::const_iterator I = Loops.begin(), E = Loops.end(); I != E;
       ++I) {
    SlotIndex Idx = LIS.getMBBStartIdx((*I)->getHeader());
    if (!Best || Idx < BestIdx)
      Best = *I, BestIdx = Idx;
  }
  DEBUG(dbgs() << "  getBestSplitLoop found " << *Best);
  return Best;
}

/// isBypassLoop - Return true if CurLI is live through Loop and has no uses
/// inside the loop. Bypass loops are candidates for splitting because it can
/// prevent interference inside the loop.
bool SplitAnalysis::isBypassLoop(const MachineLoop *Loop) {
  // If CurLI is live into the loop header and there are no uses in the loop, it
  // must be live in the entire loop and live on at least one exiting edge.
  return !UsingLoops.count(Loop) &&
         LIS.isLiveInToMBB(*CurLI, Loop->getHeader());
}

/// getBypassLoops - Get all the maximal bypass loops. These are the bypass
/// loops whose parent is not a bypass loop.
void SplitAnalysis::getBypassLoops(LoopPtrSet &BypassLoops) {
  SmallVector<MachineLoop*, 8> Todo(Loops.begin(), Loops.end());
  while (!Todo.empty()) {
    MachineLoop *Loop = Todo.pop_back_val();
    if (!UsingLoops.count(Loop)) {
      // This is either a bypass loop or completely irrelevant.
      if (LIS.isLiveInToMBB(*CurLI, Loop->getHeader()))
        BypassLoops.insert(Loop);
      // Either way, skip the child loops.
      continue;
    }

    // The child loops may be bypass loops.
    Todo.append(Loop->begin(), Loop->end());
  }
}


//===----------------------------------------------------------------------===//
//                               LiveIntervalMap
//===----------------------------------------------------------------------===//

// Work around the fact that the std::pair constructors are broken for pointer
// pairs in some implementations. makeVV(x, 0) works.
static inline std::pair<const VNInfo*, VNInfo*>
makeVV(const VNInfo *a, VNInfo *b) {
  return std::make_pair(a, b);
}

void LiveIntervalMap::reset(LiveInterval *li) {
  LI = li;
  Values.clear();
  LiveOutCache.clear();
}

bool LiveIntervalMap::isComplexMapped(const VNInfo *ParentVNI) const {
  ValueMap::const_iterator i = Values.find(ParentVNI);
  return i != Values.end() && i->second == 0;
}

// defValue - Introduce a LI def for ParentVNI that could be later than
// ParentVNI->def.
VNInfo *LiveIntervalMap::defValue(const VNInfo *ParentVNI, SlotIndex Idx) {
  assert(LI && "call reset first");
  assert(ParentVNI && "Mapping  NULL value");
  assert(Idx.isValid() && "Invalid SlotIndex");
  assert(ParentLI.getVNInfoAt(Idx) == ParentVNI && "Bad ParentVNI");

  // Create a new value.
  VNInfo *VNI = LI->getNextValue(Idx, 0, LIS.getVNInfoAllocator());

  // Preserve the PHIDef bit.
  if (ParentVNI->isPHIDef() && Idx == ParentVNI->def)
    VNI->setIsPHIDef(true);

  // Use insert for lookup, so we can add missing values with a second lookup.
  std::pair<ValueMap::iterator,bool> InsP =
    Values.insert(makeVV(ParentVNI, Idx == ParentVNI->def ? VNI : 0));

  // This is now a complex def. Mark with a NULL in valueMap.
  if (!InsP.second)
    InsP.first->second = 0;

  return VNI;
}


// mapValue - Find the mapped value for ParentVNI at Idx.
// Potentially create phi-def values.
VNInfo *LiveIntervalMap::mapValue(const VNInfo *ParentVNI, SlotIndex Idx,
                                  bool *simple) {
  assert(LI && "call reset first");
  assert(ParentVNI && "Mapping  NULL value");
  assert(Idx.isValid() && "Invalid SlotIndex");
  assert(ParentLI.getVNInfoAt(Idx) == ParentVNI && "Bad ParentVNI");

  // Use insert for lookup, so we can add missing values with a second lookup.
  std::pair<ValueMap::iterator,bool> InsP =
    Values.insert(makeVV(ParentVNI, 0));

  // This was an unknown value. Create a simple mapping.
  if (InsP.second) {
    if (simple) *simple = true;
    return InsP.first->second = LI->createValueCopy(ParentVNI,
                                                     LIS.getVNInfoAllocator());
  }

  // This was a simple mapped value.
  if (InsP.first->second) {
    if (simple) *simple = true;
    return InsP.first->second;
  }

  // This is a complex mapped value. There may be multiple defs, and we may need
  // to create phi-defs.
  if (simple) *simple = false;
  MachineBasicBlock *IdxMBB = LIS.getMBBFromIndex(Idx);
  assert(IdxMBB && "No MBB at Idx");

  // Is there a def in the same MBB we can extend?
  if (VNInfo *VNI = extendTo(IdxMBB, Idx))
    return VNI;

  // Now for the fun part. We know that ParentVNI potentially has multiple defs,
  // and we may need to create even more phi-defs to preserve VNInfo SSA form.
  // Perform a search for all predecessor blocks where we know the dominating
  // VNInfo. Insert phi-def VNInfos along the path back to IdxMBB.
  DEBUG(dbgs() << "\n  Reaching defs for BB#" << IdxMBB->getNumber()
               << " at " << Idx << " in " << *LI << '\n');

  // Blocks where LI should be live-in.
  SmallVector<MachineDomTreeNode*, 16> LiveIn;
  LiveIn.push_back(MDT[IdxMBB]);

  // Using LiveOutCache as a visited set, perform a BFS for all reaching defs.
  for (unsigned i = 0; i != LiveIn.size(); ++i) {
    MachineBasicBlock *MBB = LiveIn[i]->getBlock();
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
           PE = MBB->pred_end(); PI != PE; ++PI) {
       MachineBasicBlock *Pred = *PI;
       // Is this a known live-out block?
       std::pair<LiveOutMap::iterator,bool> LOIP =
         LiveOutCache.insert(std::make_pair(Pred, LiveOutPair()));
       // Yes, we have been here before.
       if (!LOIP.second) {
         DEBUG(if (VNInfo *VNI = LOIP.first->second.first)
                 dbgs() << "    known valno #" << VNI->id
                        << " at BB#" << Pred->getNumber() << '\n');
         continue;
       }

       // Does Pred provide a live-out value?
       SlotIndex Last = LIS.getMBBEndIdx(Pred).getPrevSlot();
       if (VNInfo *VNI = extendTo(Pred, Last)) {
         MachineBasicBlock *DefMBB = LIS.getMBBFromIndex(VNI->def);
         DEBUG(dbgs() << "    found valno #" << VNI->id
                      << " from BB#" << DefMBB->getNumber()
                      << " at BB#" << Pred->getNumber() << '\n');
         LiveOutPair &LOP = LOIP.first->second;
         LOP.first = VNI;
         LOP.second = MDT[DefMBB];
         continue;
       }
       // No, we need a live-in value for Pred as well
       if (Pred != IdxMBB)
         LiveIn.push_back(MDT[Pred]);
    }
  }

  // We may need to add phi-def values to preserve the SSA form.
  // This is essentially the same iterative algorithm that SSAUpdater uses,
  // except we already have a dominator tree, so we don't have to recompute it.
  VNInfo *IdxVNI = 0;
  unsigned Changes;
  do {
    Changes = 0;
    DEBUG(dbgs() << "  Iterating over " << LiveIn.size() << " blocks.\n");
    // Propagate live-out values down the dominator tree, inserting phi-defs when
    // necessary. Since LiveIn was created by a BFS, going backwards makes it more
    // likely for us to visit immediate dominators before their children.
    for (unsigned i = LiveIn.size(); i; --i) {
      MachineDomTreeNode *Node = LiveIn[i-1];
      MachineBasicBlock *MBB = Node->getBlock();
      MachineDomTreeNode *IDom = Node->getIDom();
      LiveOutPair IDomValue;
      // We need a live-in value to a block with no immediate dominator?
      // This is probably an unreachable block that has survived somehow.
      bool needPHI = !IDom;

      // Get the IDom live-out value.
      if (!needPHI) {
        LiveOutMap::iterator I = LiveOutCache.find(IDom->getBlock());
        if (I != LiveOutCache.end())
          IDomValue = I->second;
        else
          // If IDom is outside our set of live-out blocks, there must be new
          // defs, and we need a phi-def here.
          needPHI = true;
      }

      // IDom dominates all of our predecessors, but it may not be the immediate
      // dominator. Check if any of them have live-out values that are properly
      // dominated by IDom. If so, we need a phi-def here.
      if (!needPHI) {
        for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
               PE = MBB->pred_end(); PI != PE; ++PI) {
          LiveOutPair Value = LiveOutCache[*PI];
          if (!Value.first || Value.first == IDomValue.first)
            continue;
          // This predecessor is carrying something other than IDomValue.
          // It could be because IDomValue hasn't propagated yet, or it could be
          // because MBB is in the dominance frontier of that value.
          if (MDT.dominates(IDom, Value.second)) {
            needPHI = true;
            break;
          }
        }
      }

      // Create a phi-def if required.
      if (needPHI) {
        ++Changes;
        SlotIndex Start = LIS.getMBBStartIdx(MBB);
        VNInfo *VNI = LI->getNextValue(Start, 0, LIS.getVNInfoAllocator());
        VNI->setIsPHIDef(true);
        DEBUG(dbgs() << "    - BB#" << MBB->getNumber()
                     << " phi-def #" << VNI->id << " at " << Start << '\n');
        // We no longer need LI to be live-in.
        LiveIn.erase(LiveIn.begin()+(i-1));
        // Blocks in LiveIn are either IdxMBB, or have a value live-through.
        if (MBB == IdxMBB)
          IdxVNI = VNI;
        // Check if we need to update live-out info.
        LiveOutMap::iterator I = LiveOutCache.find(MBB);
        if (I == LiveOutCache.end() || I->second.second == Node) {
          // We already have a live-out defined in MBB, so this must be IdxMBB.
          assert(MBB == IdxMBB && "Adding phi-def to known live-out");
          LI->addRange(LiveRange(Start, Idx.getNextSlot(), VNI));
        } else {
          // This phi-def is also live-out, so color the whole block.
          LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));
          I->second = LiveOutPair(VNI, Node);
        }
      } else if (IDomValue.first) {
        // No phi-def here. Remember incoming value for IdxMBB.
        if (MBB == IdxMBB)
          IdxVNI = IDomValue.first;
        // Propagate IDomValue if needed:
        // MBB is live-out and doesn't define its own value.
        LiveOutMap::iterator I = LiveOutCache.find(MBB);
        if (I != LiveOutCache.end() && I->second.second != Node &&
            I->second.first != IDomValue.first) {
          ++Changes;
          I->second = IDomValue;
          DEBUG(dbgs() << "    - BB#" << MBB->getNumber()
                       << " idom valno #" << IDomValue.first->id
                       << " from BB#" << IDom->getBlock()->getNumber() << '\n');
        }
      }
    }
    DEBUG(dbgs() << "  - made " << Changes << " changes.\n");
  } while (Changes);

  assert(IdxVNI && "Didn't find value for Idx");

#ifndef NDEBUG
  // Check the LiveOutCache invariants.
  for (LiveOutMap::iterator I = LiveOutCache.begin(), E = LiveOutCache.end();
         I != E; ++I) {
    assert(I->first && "Null MBB entry in cache");
    assert(I->second.first && "Null VNInfo in cache");
    assert(I->second.second && "Null DomTreeNode in cache");
    if (I->second.second->getBlock() == I->first)
      continue;
    for (MachineBasicBlock::pred_iterator PI = I->first->pred_begin(),
           PE = I->first->pred_end(); PI != PE; ++PI)
      assert(LiveOutCache.lookup(*PI) == I->second && "Bad invariant");
  }
#endif

  // Since we went through the trouble of a full BFS visiting all reaching defs,
  // the values in LiveIn are now accurate. No more phi-defs are needed
  // for these blocks, so we can color the live ranges.
  // This makes the next mapValue call much faster.
  for (unsigned i = 0, e = LiveIn.size(); i != e; ++i) {
    MachineBasicBlock *MBB = LiveIn[i]->getBlock();
    SlotIndex Start = LIS.getMBBStartIdx(MBB);
    VNInfo *VNI = LiveOutCache.lookup(MBB).first;

    // Anything in LiveIn other than IdxMBB is live-through.
    // In IdxMBB, we should stop at Idx unless the same value is live-out.
    if (MBB == IdxMBB && IdxVNI != VNI)
      LI->addRange(LiveRange(Start, Idx.getNextSlot(), IdxVNI));
    else
      LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));
  }

  return IdxVNI;
}

#ifndef NDEBUG
void LiveIntervalMap::dumpCache() {
  for (LiveOutMap::iterator I = LiveOutCache.begin(), E = LiveOutCache.end();
         I != E; ++I) {
    assert(I->first && "Null MBB entry in cache");
    assert(I->second.first && "Null VNInfo in cache");
    assert(I->second.second && "Null DomTreeNode in cache");
    dbgs() << "    cache: BB#" << I->first->getNumber()
           << " has valno #" << I->second.first->id << " from BB#"
           << I->second.second->getBlock()->getNumber() << ", preds";
    for (MachineBasicBlock::pred_iterator PI = I->first->pred_begin(),
           PE = I->first->pred_end(); PI != PE; ++PI)
      dbgs() << " BB#" << (*PI)->getNumber();
    dbgs() << '\n';
  }
  dbgs() << "    cache: " << LiveOutCache.size() << " entries.\n";
}
#endif

// extendTo - Find the last LI value defined in MBB at or before Idx. The
// ParentLI is assumed to be live at Idx. Extend the live range to Idx.
// Return the found VNInfo, or NULL.
VNInfo *LiveIntervalMap::extendTo(const MachineBasicBlock *MBB, SlotIndex Idx) {
  assert(LI && "call reset first");
  LiveInterval::iterator I = std::upper_bound(LI->begin(), LI->end(), Idx);
  if (I == LI->begin())
    return 0;
  --I;
  if (I->end <= LIS.getMBBStartIdx(MBB))
    return 0;
  if (I->end <= Idx)
    I->end = Idx.getNextSlot();
  return I->valno;
}

// addSimpleRange - Add a simple range from ParentLI to LI.
// ParentVNI must be live in the [Start;End) interval.
void LiveIntervalMap::addSimpleRange(SlotIndex Start, SlotIndex End,
                                     const VNInfo *ParentVNI) {
  assert(LI && "call reset first");
  bool simple;
  VNInfo *VNI = mapValue(ParentVNI, Start, &simple);
  // A simple mapping is easy.
  if (simple) {
    LI->addRange(LiveRange(Start, End, VNI));
    return;
  }

  // ParentVNI is a complex value. We must map per MBB.
  MachineFunction::iterator MBB = LIS.getMBBFromIndex(Start);
  MachineFunction::iterator MBBE = LIS.getMBBFromIndex(End.getPrevSlot());

  if (MBB == MBBE) {
    LI->addRange(LiveRange(Start, End, VNI));
    return;
  }

  // First block.
  LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB), VNI));

  // Run sequence of full blocks.
  for (++MBB; MBB != MBBE; ++MBB) {
    Start = LIS.getMBBStartIdx(MBB);
    LI->addRange(LiveRange(Start, LIS.getMBBEndIdx(MBB),
                            mapValue(ParentVNI, Start)));
  }

  // Final block.
  Start = LIS.getMBBStartIdx(MBB);
  if (Start != End)
    LI->addRange(LiveRange(Start, End, mapValue(ParentVNI, Start)));
}

/// addRange - Add live ranges to LI where [Start;End) intersects ParentLI.
/// All needed values whose def is not inside [Start;End) must be defined
/// beforehand so mapValue will work.
void LiveIntervalMap::addRange(SlotIndex Start, SlotIndex End) {
  assert(LI && "call reset first");
  LiveInterval::const_iterator B = ParentLI.begin(), E = ParentLI.end();
  LiveInterval::const_iterator I = std::lower_bound(B, E, Start);

  // Check if --I begins before Start and overlaps.
  if (I != B) {
    --I;
    if (I->end > Start)
      addSimpleRange(Start, std::min(End, I->end), I->valno);
    ++I;
  }

  // The remaining ranges begin after Start.
  for (;I != E && I->start < End; ++I)
    addSimpleRange(I->start, std::min(End, I->end), I->valno);
}


//===----------------------------------------------------------------------===//
//                               Split Editor
//===----------------------------------------------------------------------===//

/// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
SplitEditor::SplitEditor(SplitAnalysis &sa,
                         LiveIntervals &lis,
                         VirtRegMap &vrm,
                         MachineDominatorTree &mdt,
                         LiveRangeEdit &edit)
  : sa_(sa), LIS(lis), VRM(vrm),
    MRI(vrm.getMachineFunction().getRegInfo()),
    MDT(mdt),
    TII(*vrm.getMachineFunction().getTarget().getInstrInfo()),
    TRI(*vrm.getMachineFunction().getTarget().getRegisterInfo()),
    Edit(edit),
    OpenIdx(0),
    RegAssign(Allocator)
{
  // We don't need an AliasAnalysis since we will only be performing
  // cheap-as-a-copy remats anyway.
  Edit.anyRematerializable(LIS, TII, 0);
}

void SplitEditor::dump() const {
  if (RegAssign.empty()) {
    dbgs() << " empty\n";
    return;
  }

  for (RegAssignMap::const_iterator I = RegAssign.begin(); I.valid(); ++I)
    dbgs() << " [" << I.start() << ';' << I.stop() << "):" << I.value();
  dbgs() << '\n';
}

VNInfo *SplitEditor::defFromParent(unsigned RegIdx,
                                   VNInfo *ParentVNI,
                                   SlotIndex UseIdx,
                                   MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I) {
  MachineInstr *CopyMI = 0;
  SlotIndex Def;
  LiveInterval *LI = Edit.get(RegIdx);

  // Attempt cheap-as-a-copy rematerialization.
  LiveRangeEdit::Remat RM(ParentVNI);
  if (Edit.canRematerializeAt(RM, UseIdx, true, LIS)) {
    Def = Edit.rematerializeAt(MBB, I, LI->reg, RM, LIS, TII, TRI);
  } else {
    // Can't remat, just insert a copy from parent.
    CopyMI = BuildMI(MBB, I, DebugLoc(), TII.get(TargetOpcode::COPY), LI->reg)
               .addReg(Edit.getReg());
    Def = LIS.InsertMachineInstrInMaps(CopyMI).getDefIndex();
  }

  // Define the value in Reg.
  VNInfo *VNI = LIMappers[RegIdx].defValue(ParentVNI, Def);
  VNI->setCopy(CopyMI);

  // Add minimal liveness for the new value.
  Edit.get(RegIdx)->addRange(LiveRange(Def, Def.getNextSlot(), VNI));
  return VNI;
}

/// Create a new virtual register and live interval.
void SplitEditor::openIntv() {
  assert(!OpenIdx && "Previous LI not closed before openIntv");

  // Create the complement as index 0.
  if (Edit.empty()) {
    Edit.create(MRI, LIS, VRM);
    LIMappers.push_back(LiveIntervalMap(LIS, MDT, Edit.getParent()));
    LIMappers.back().reset(Edit.get(0));
  }

  // Create the open interval.
  OpenIdx = Edit.size();
  Edit.create(MRI, LIS, VRM);
  LIMappers.push_back(LiveIntervalMap(LIS, MDT, Edit.getParent()));
  LIMappers[OpenIdx].reset(Edit.get(OpenIdx));
}

SlotIndex SplitEditor::enterIntvBefore(SlotIndex Idx) {
  assert(OpenIdx && "openIntv not called before enterIntvBefore");
  DEBUG(dbgs() << "    enterIntvBefore " << Idx);
  Idx = Idx.getBaseIndex();
  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Idx;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id << '\n');
  MachineInstr *MI = LIS.getInstructionFromIndex(Idx);
  assert(MI && "enterIntvBefore called with invalid index");

  VNInfo *VNI = defFromParent(OpenIdx, ParentVNI, Idx, *MI->getParent(), MI);
  return VNI->def;
}

SlotIndex SplitEditor::enterIntvAtEnd(MachineBasicBlock &MBB) {
  assert(OpenIdx && "openIntv not called before enterIntvAtEnd");
  SlotIndex End = LIS.getMBBEndIdx(&MBB);
  SlotIndex Last = End.getPrevSlot();
  DEBUG(dbgs() << "    enterIntvAtEnd BB#" << MBB.getNumber() << ", " << Last);
  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Last);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return End;
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id);
  VNInfo *VNI = defFromParent(OpenIdx, ParentVNI, Last, MBB,
                              LIS.getLastSplitPoint(Edit.getParent(), &MBB));
  RegAssign.insert(VNI->def, End, OpenIdx);
  DEBUG(dump());
  return VNI->def;
}

/// useIntv - indicate that all instructions in MBB should use OpenLI.
void SplitEditor::useIntv(const MachineBasicBlock &MBB) {
  useIntv(LIS.getMBBStartIdx(&MBB), LIS.getMBBEndIdx(&MBB));
}

void SplitEditor::useIntv(SlotIndex Start, SlotIndex End) {
  assert(OpenIdx && "openIntv not called before useIntv");
  DEBUG(dbgs() << "    useIntv [" << Start << ';' << End << "):");
  RegAssign.insert(Start, End, OpenIdx);
  DEBUG(dump());
}

SlotIndex SplitEditor::leaveIntvAfter(SlotIndex Idx) {
  assert(OpenIdx && "openIntv not called before leaveIntvAfter");
  DEBUG(dbgs() << "    leaveIntvAfter " << Idx);

  // The interval must be live beyond the instruction at Idx.
  Idx = Idx.getBoundaryIndex();
  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Idx);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Idx.getNextSlot();
  }
  DEBUG(dbgs() << ": valno " << ParentVNI->id << '\n');

  MachineInstr *MI = LIS.getInstructionFromIndex(Idx);
  assert(MI && "No instruction at index");
  VNInfo *VNI = defFromParent(0, ParentVNI, Idx, *MI->getParent(),
                              llvm::next(MachineBasicBlock::iterator(MI)));
  return VNI->def;
}

SlotIndex SplitEditor::leaveIntvAtTop(MachineBasicBlock &MBB) {
  assert(OpenIdx && "openIntv not called before leaveIntvAtTop");
  SlotIndex Start = LIS.getMBBStartIdx(&MBB);
  DEBUG(dbgs() << "    leaveIntvAtTop BB#" << MBB.getNumber() << ", " << Start);

  VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Start);
  if (!ParentVNI) {
    DEBUG(dbgs() << ": not live\n");
    return Start;
  }

  VNInfo *VNI = defFromParent(0, ParentVNI, Start, MBB,
                              MBB.SkipPHIsAndLabels(MBB.begin()));
  RegAssign.insert(Start, VNI->def, OpenIdx);
  DEBUG(dump());
  return VNI->def;
}

/// closeIntv - Indicate that we are done editing the currently open
/// LiveInterval, and ranges can be trimmed.
void SplitEditor::closeIntv() {
  assert(OpenIdx && "openIntv not called before closeIntv");
  OpenIdx = 0;
}

/// rewriteAssigned - Rewrite all uses of Edit.getReg().
void SplitEditor::rewriteAssigned() {
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Edit.getReg()),
       RE = MRI.reg_end(); RI != RE;) {
    MachineOperand &MO = RI.getOperand();
    MachineInstr *MI = MO.getParent();
    ++RI;
    // LiveDebugVariables should have handled all DBG_VALUE instructions.
    if (MI->isDebugValue()) {
      DEBUG(dbgs() << "Zapping " << *MI);
      MO.setReg(0);
      continue;
    }
    SlotIndex Idx = LIS.getInstructionIndex(MI);
    Idx = MO.isUse() ? Idx.getUseIndex() : Idx.getDefIndex();

    // Rewrite to the mapped register at Idx.
    unsigned RegIdx = RegAssign.lookup(Idx);
    MO.setReg(Edit.get(RegIdx)->reg);
    DEBUG(dbgs() << "  rewr BB#" << MI->getParent()->getNumber() << '\t'
                 << Idx << ':' << RegIdx << '\t' << *MI);

    // Extend liveness to Idx.
    const VNInfo *ParentVNI = Edit.getParent().getVNInfoAt(Idx);
    LIMappers[RegIdx].mapValue(ParentVNI, Idx);
  }
}

/// rewriteSplit - Rewrite uses of Intvs[0] according to the ConEQ mapping.
void SplitEditor::rewriteComponents(const SmallVectorImpl<LiveInterval*> &Intvs,
                                    const ConnectedVNInfoEqClasses &ConEq) {
  for (MachineRegisterInfo::reg_iterator RI = MRI.reg_begin(Intvs[0]->reg),
       RE = MRI.reg_end(); RI != RE;) {
    MachineOperand &MO = RI.getOperand();
    MachineInstr *MI = MO.getParent();
    ++RI;
    if (MO.isUse() && MO.isUndef())
      continue;
    // DBG_VALUE instructions should have been eliminated earlier.
    SlotIndex Idx = LIS.getInstructionIndex(MI);
    Idx = MO.isUse() ? Idx.getUseIndex() : Idx.getDefIndex();
    DEBUG(dbgs() << "  rewr BB#" << MI->getParent()->getNumber() << '\t'
                 << Idx << ':');
    const VNInfo *VNI = Intvs[0]->getVNInfoAt(Idx);
    assert(VNI && "Interval not live at use.");
    MO.setReg(Intvs[ConEq.getEqClass(VNI)]->reg);
    DEBUG(dbgs() << VNI->id << '\t' << *MI);
  }
}

void SplitEditor::finish() {
  assert(OpenIdx == 0 && "Previous LI not closed before rewrite");

  // At this point, the live intervals in Edit contain VNInfos corresponding to
  // the inserted copies.

  // Add the original defs from the parent interval.
  for (LiveInterval::const_vni_iterator I = Edit.getParent().vni_begin(),
         E = Edit.getParent().vni_end(); I != E; ++I) {
    const VNInfo *ParentVNI = *I;
    if (ParentVNI->isUnused())
      continue;
    LiveIntervalMap &LIM = LIMappers[RegAssign.lookup(ParentVNI->def)];
    VNInfo *VNI = LIM.defValue(ParentVNI, ParentVNI->def);
    LIM.getLI()->addRange(LiveRange(ParentVNI->def,
                                    ParentVNI->def.getNextSlot(), VNI));
    // Mark all values as complex to force liveness computation.
    // This should really only be necessary for remat victims, but we are lazy.
    LIM.markComplexMapped(ParentVNI);
  }

#ifndef NDEBUG
  // Every new interval must have a def by now, otherwise the split is bogus.
  for (LiveRangeEdit::iterator I = Edit.begin(), E = Edit.end(); I != E; ++I)
    assert((*I)->hasAtLeastOneValue() && "Split interval has no value");
#endif

  // FIXME: Don't recompute the liveness of all values, infer it from the
  // overlaps between the parent live interval and RegAssign.
  // The mapValue algorithm is only necessary when:
  // - The parent value maps to multiple defs, and new phis are needed, or
  // - The value has been rematerialized before some uses, and we want to
  //   minimize the live range so it only reaches the remaining uses.
  // All other values have simple liveness that can be computed from RegAssign
  // and the parent live interval.

  // Extend live ranges to be live-out for successor PHI values.
  for (LiveInterval::const_vni_iterator I = Edit.getParent().vni_begin(),
       E = Edit.getParent().vni_end(); I != E; ++I) {
    const VNInfo *PHIVNI = *I;
    if (PHIVNI->isUnused() || !PHIVNI->isPHIDef())
      continue;
    unsigned RegIdx = RegAssign.lookup(PHIVNI->def);
    LiveIntervalMap &LIM = LIMappers[RegIdx];
    MachineBasicBlock *MBB = LIS.getMBBFromIndex(PHIVNI->def);
    DEBUG(dbgs() << "  map phi in BB#" << MBB->getNumber() << '@' << PHIVNI->def
                 << " -> " << RegIdx << '\n');
    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
      SlotIndex End = LIS.getMBBEndIdx(*PI).getPrevSlot();
      DEBUG(dbgs() << "    pred BB#" << (*PI)->getNumber() << '@' << End);
      // The predecessor may not have a live-out value. That is OK, like an
      // undef PHI operand.
      if (VNInfo *VNI = Edit.getParent().getVNInfoAt(End)) {
        DEBUG(dbgs() << " has parent valno #" << VNI->id << " live out\n");
        assert(RegAssign.lookup(End) == RegIdx &&
               "Different register assignment in phi predecessor");
        LIM.mapValue(VNI, End);
      }
      else
        DEBUG(dbgs() << " is not live-out\n");
    }
    DEBUG(dbgs() << "    " << *LIM.getLI() << '\n');
  }

  // Rewrite instructions.
  rewriteAssigned();

  // FIXME: Delete defs that were rematted everywhere.

  // Get rid of unused values and set phi-kill flags.
  for (LiveRangeEdit::iterator I = Edit.begin(), E = Edit.end(); I != E; ++I)
    (*I)->RenumberValues(LIS);

  // Now check if any registers were separated into multiple components.
  ConnectedVNInfoEqClasses ConEQ(LIS);
  for (unsigned i = 0, e = Edit.size(); i != e; ++i) {
    // Don't use iterators, they are invalidated by create() below.
    LiveInterval *li = Edit.get(i);
    unsigned NumComp = ConEQ.Classify(li);
    if (NumComp <= 1)
      continue;
    DEBUG(dbgs() << "  " << NumComp << " components: " << *li << '\n');
    SmallVector<LiveInterval*, 8> dups;
    dups.push_back(li);
    for (unsigned i = 1; i != NumComp; ++i)
      dups.push_back(&Edit.create(MRI, LIS, VRM));
    rewriteComponents(dups, ConEQ);
    ConEQ.Distribute(&dups[0]);
  }

  // Calculate spill weight and allocation hints for new intervals.
  VirtRegAuxInfo vrai(VRM.getMachineFunction(), LIS, sa_.Loops);
  for (LiveRangeEdit::iterator I = Edit.begin(), E = Edit.end(); I != E; ++I){
    LiveInterval &li = **I;
    vrai.CalculateRegClass(li.reg);
    vrai.CalculateWeightAndHint(li);
    DEBUG(dbgs() << "  new interval " << MRI.getRegClass(li.reg)->getName()
                 << ":" << li << '\n');
  }
}


//===----------------------------------------------------------------------===//
//                               Loop Splitting
//===----------------------------------------------------------------------===//

void SplitEditor::splitAroundLoop(const MachineLoop *Loop) {
  SplitAnalysis::LoopBlocks Blocks;
  sa_.getLoopBlocks(Loop, Blocks);

  DEBUG({
    dbgs() << "  splitAround"; sa_.print(Blocks, dbgs()); dbgs() << '\n';
  });

  // Break critical edges as needed.
  SplitAnalysis::BlockPtrSet CriticalExits;
  sa_.getCriticalExits(Blocks, CriticalExits);
  assert(CriticalExits.empty() && "Cannot break critical exits yet");

  // Create new live interval for the loop.
  openIntv();

  // Insert copies in the predecessors if live-in to the header.
  if (LIS.isLiveInToMBB(Edit.getParent(), Loop->getHeader())) {
    for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Preds.begin(),
           E = Blocks.Preds.end(); I != E; ++I) {
      MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
      enterIntvAtEnd(MBB);
    }
  }

  // Switch all loop blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Loop.begin(),
       E = Blocks.Loop.end(); I != E; ++I)
     useIntv(**I);

  // Insert back copies in the exit blocks.
  for (SplitAnalysis::BlockPtrSet::iterator I = Blocks.Exits.begin(),
       E = Blocks.Exits.end(); I != E; ++I) {
    MachineBasicBlock &MBB = const_cast<MachineBasicBlock&>(**I);
    leaveIntvAtTop(MBB);
  }

  // Done.
  closeIntv();
  finish();
}


//===----------------------------------------------------------------------===//
//                            Single Block Splitting
//===----------------------------------------------------------------------===//

/// getMultiUseBlocks - if CurLI has more than one use in a basic block, it
/// may be an advantage to split CurLI for the duration of the block.
bool SplitAnalysis::getMultiUseBlocks(BlockPtrSet &Blocks) {
  // If CurLI is local to one block, there is no point to splitting it.
  if (UsingBlocks.size() <= 1)
    return false;
  // Add blocks with multiple uses.
  for (BlockCountMap::iterator I = UsingBlocks.begin(), E = UsingBlocks.end();
       I != E; ++I)
    switch (I->second) {
    case 0:
    case 1:
      continue;
    case 2: {
      // When there are only two uses and CurLI is both live in and live out,
      // we don't really win anything by isolating the block since we would be
      // inserting two copies.
      // The remaing register would still have two uses in the block. (Unless it
      // separates into disconnected components).
      if (LIS.isLiveInToMBB(*CurLI, I->first) &&
          LIS.isLiveOutOfMBB(*CurLI, I->first))
        continue;
    } // Fall through.
    default:
      Blocks.insert(I->first);
    }
  return !Blocks.empty();
}

/// splitSingleBlocks - Split CurLI into a separate live interval inside each
/// basic block in Blocks.
void SplitEditor::splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks) {
  DEBUG(dbgs() << "  splitSingleBlocks for " << Blocks.size() << " blocks.\n");
  // Determine the first and last instruction using CurLI in each block.
  typedef std::pair<SlotIndex,SlotIndex> IndexPair;
  typedef DenseMap<const MachineBasicBlock*,IndexPair> IndexPairMap;
  IndexPairMap MBBRange;
  for (SplitAnalysis::InstrPtrSet::const_iterator I = sa_.UsingInstrs.begin(),
       E = sa_.UsingInstrs.end(); I != E; ++I) {
    const MachineBasicBlock *MBB = (*I)->getParent();
    if (!Blocks.count(MBB))
      continue;
    SlotIndex Idx = LIS.getInstructionIndex(*I);
    DEBUG(dbgs() << "  BB#" << MBB->getNumber() << '\t' << Idx << '\t' << **I);
    IndexPair &IP = MBBRange[MBB];
    if (!IP.first.isValid() || Idx < IP.first)
      IP.first = Idx;
    if (!IP.second.isValid() || Idx > IP.second)
      IP.second = Idx;
  }

  // Create a new interval for each block.
  for (SplitAnalysis::BlockPtrSet::const_iterator I = Blocks.begin(),
       E = Blocks.end(); I != E; ++I) {
    IndexPair &IP = MBBRange[*I];
    DEBUG(dbgs() << "  splitting for BB#" << (*I)->getNumber() << ": ["
                 << IP.first << ';' << IP.second << ")\n");
    assert(IP.first.isValid() && IP.second.isValid());

    openIntv();
    useIntv(enterIntvBefore(IP.first), leaveIntvAfter(IP.second));
    closeIntv();
  }
  finish();
}


//===----------------------------------------------------------------------===//
//                            Sub Block Splitting
//===----------------------------------------------------------------------===//

/// getBlockForInsideSplit - If CurLI is contained inside a single basic block,
/// and it wou pay to subdivide the interval inside that block, return it.
/// Otherwise return NULL. The returned block can be passed to
/// SplitEditor::splitInsideBlock.
const MachineBasicBlock *SplitAnalysis::getBlockForInsideSplit() {
  // The interval must be exclusive to one block.
  if (UsingBlocks.size() != 1)
    return 0;
  // Don't to this for less than 4 instructions. We want to be sure that
  // splitting actually reduces the instruction count per interval.
  if (UsingInstrs.size() < 4)
    return 0;
  return UsingBlocks.begin()->first;
}

/// splitInsideBlock - Split CurLI into multiple intervals inside MBB.
void SplitEditor::splitInsideBlock(const MachineBasicBlock *MBB) {
  SmallVector<SlotIndex, 32> Uses;
  Uses.reserve(sa_.UsingInstrs.size());
  for (SplitAnalysis::InstrPtrSet::const_iterator I = sa_.UsingInstrs.begin(),
       E = sa_.UsingInstrs.end(); I != E; ++I)
    if ((*I)->getParent() == MBB)
      Uses.push_back(LIS.getInstructionIndex(*I));
  DEBUG(dbgs() << "  splitInsideBlock BB#" << MBB->getNumber() << " for "
               << Uses.size() << " instructions.\n");
  assert(Uses.size() >= 3 && "Need at least 3 instructions");
  array_pod_sort(Uses.begin(), Uses.end());

  // Simple algorithm: Find the largest gap between uses as determined by slot
  // indices. Create new intervals for instructions before the gap and after the
  // gap.
  unsigned bestPos = 0;
  int bestGap = 0;
  DEBUG(dbgs() << "    dist (" << Uses[0]);
  for (unsigned i = 1, e = Uses.size(); i != e; ++i) {
    int g = Uses[i-1].distance(Uses[i]);
    DEBUG(dbgs() << ") -" << g << "- (" << Uses[i]);
    if (g > bestGap)
      bestPos = i, bestGap = g;
  }
  DEBUG(dbgs() << "), best: -" << bestGap << "-\n");

  // bestPos points to the first use after the best gap.
  assert(bestPos > 0 && "Invalid gap");

  // FIXME: Don't create intervals for low densities.

  // First interval before the gap. Don't create single-instr intervals.
  if (bestPos > 1) {
    openIntv();
    useIntv(enterIntvBefore(Uses.front()), leaveIntvAfter(Uses[bestPos-1]));
    closeIntv();
  }

  // Second interval after the gap.
  if (bestPos < Uses.size()-1) {
    openIntv();
    useIntv(enterIntvBefore(Uses[bestPos]), leaveIntvAfter(Uses.back()));
    closeIntv();
  }

  finish();
}

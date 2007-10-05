//===----- ScheduleDAGList.cpp - Reg pressure reduction list scheduler ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements bottom-up and top-down register pressure reduction list
// schedulers, using standard algorithms.  The basic approach uses a priority
// queue of available nodes to schedule.  One at a time, nodes are taken from
// the priority queue (thus in priority order), checked for legality to
// schedule, and emitted if legal.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include <climits>
#include <queue>
#include "llvm/Support/CommandLine.h"
using namespace llvm;

STATISTIC(NumBacktracks, "Number of times scheduler backtraced");
STATISTIC(NumUnfolds,    "Number of nodes unfolded");
STATISTIC(NumDups,       "Number of duplicated nodes");
STATISTIC(NumCCCopies,   "Number of cross class copies");

static RegisterScheduler
  burrListDAGScheduler("list-burr",
                       "  Bottom-up register reduction list scheduling",
                       createBURRListDAGScheduler);
static RegisterScheduler
  tdrListrDAGScheduler("list-tdrr",
                       "  Top-down register reduction list scheduling",
                       createTDRRListDAGScheduler);

namespace {
//===----------------------------------------------------------------------===//
/// ScheduleDAGRRList - The actual register reduction list scheduler
/// implementation.  This supports both top-down and bottom-up scheduling.
///
class VISIBILITY_HIDDEN ScheduleDAGRRList : public ScheduleDAG {
private:
  /// isBottomUp - This is true if the scheduling problem is bottom-up, false if
  /// it is top-down.
  bool isBottomUp;
  
  /// AvailableQueue - The priority queue to use for the available SUnits.
  ///a
  SchedulingPriorityQueue *AvailableQueue;

  /// LiveRegs / LiveRegDefs - A set of physical registers and their definition
  /// that are "live". These nodes must be scheduled before any other nodes that
  /// modifies the registers can be scheduled.
  SmallSet<unsigned, 4> LiveRegs;
  std::vector<SUnit*> LiveRegDefs;
  std::vector<unsigned> LiveRegCycles;

public:
  ScheduleDAGRRList(SelectionDAG &dag, MachineBasicBlock *bb,
                  const TargetMachine &tm, bool isbottomup,
                  SchedulingPriorityQueue *availqueue)
    : ScheduleDAG(dag, bb, tm), isBottomUp(isbottomup),
      AvailableQueue(availqueue) {
    }

  ~ScheduleDAGRRList() {
    delete AvailableQueue;
  }

  void Schedule();

private:
  void ReleasePred(SUnit*, bool, unsigned);
  void ReleaseSucc(SUnit*, bool isChain, unsigned);
  void CapturePred(SUnit*, SUnit*, bool);
  void ScheduleNodeBottomUp(SUnit*, unsigned);
  void ScheduleNodeTopDown(SUnit*, unsigned);
  void UnscheduleNodeBottomUp(SUnit*);
  void BacktrackBottomUp(SUnit*, unsigned, unsigned&);
  SUnit *CopyAndMoveSuccessors(SUnit*);
  void InsertCCCopiesAndMoveSuccs(SUnit*, unsigned,
                                  const TargetRegisterClass*,
                                  const TargetRegisterClass*,
                                  SmallVector<SUnit*, 2>&);
  bool DelayForLiveRegsBottomUp(SUnit*, SmallVector<unsigned, 4>&);
  void ListScheduleTopDown();
  void ListScheduleBottomUp();
  void CommuteNodesToReducePressure();
};
}  // end anonymous namespace


/// Schedule - Schedule the DAG using list scheduling.
void ScheduleDAGRRList::Schedule() {
  DOUT << "********** List Scheduling **********\n";

  LiveRegDefs.resize(MRI->getNumRegs(), NULL);  
  LiveRegCycles.resize(MRI->getNumRegs(), 0);

  // Build scheduling units.
  BuildSchedUnits();

  DEBUG(for (unsigned su = 0, e = SUnits.size(); su != e; ++su)
          SUnits[su].dumpAll(&DAG));
  CalculateDepths();
  CalculateHeights();

  AvailableQueue->initNodes(SUnitMap, SUnits);
  
  // Execute the actual scheduling loop Top-Down or Bottom-Up as appropriate.
  if (isBottomUp)
    ListScheduleBottomUp();
  else
    ListScheduleTopDown();
  
  AvailableQueue->releaseState();
  
  CommuteNodesToReducePressure();
  
  DOUT << "*** Final schedule ***\n";
  DEBUG(dumpSchedule());
  DOUT << "\n";
  
  // Emit in scheduled order
  EmitSchedule();
}

/// CommuteNodesToReducePressure - If a node is two-address and commutable, and
/// it is not the last use of its first operand, add it to the CommuteSet if
/// possible. It will be commuted when it is translated to a MI.
void ScheduleDAGRRList::CommuteNodesToReducePressure() {
  SmallPtrSet<SUnit*, 4> OperandSeen;
  for (unsigned i = Sequence.size()-1; i != 0; --i) {  // Ignore first node.
    SUnit *SU = Sequence[i];
    if (!SU || !SU->Node) continue;
    if (SU->isCommutable) {
      unsigned Opc = SU->Node->getTargetOpcode();
      unsigned NumRes = TII->getNumDefs(Opc);
      unsigned NumOps = CountOperands(SU->Node);
      for (unsigned j = 0; j != NumOps; ++j) {
        if (TII->getOperandConstraint(Opc, j+NumRes, TOI::TIED_TO) == -1)
          continue;

        SDNode *OpN = SU->Node->getOperand(j).Val;
        SUnit *OpSU = SUnitMap[OpN][SU->InstanceNo];
        if (OpSU && OperandSeen.count(OpSU) == 1) {
          // Ok, so SU is not the last use of OpSU, but SU is two-address so
          // it will clobber OpSU. Try to commute SU if no other source operands
          // are live below.
          bool DoCommute = true;
          for (unsigned k = 0; k < NumOps; ++k) {
            if (k != j) {
              OpN = SU->Node->getOperand(k).Val;
              OpSU = SUnitMap[OpN][SU->InstanceNo];
              if (OpSU && OperandSeen.count(OpSU) == 1) {
                DoCommute = false;
                break;
              }
            }
          }
          if (DoCommute)
            CommuteSet.insert(SU->Node);
        }

        // Only look at the first use&def node for now.
        break;
      }
    }

    for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (!I->isCtrl)
        OperandSeen.insert(I->Dep);
    }
  }
}

//===----------------------------------------------------------------------===//
//  Bottom-Up Scheduling
//===----------------------------------------------------------------------===//

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleasePred(SUnit *PredSU, bool isChain, 
                                    unsigned CurCycle) {
  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  PredSU->CycleBound = std::max(PredSU->CycleBound, CurCycle + PredSU->Latency);

  --PredSU->NumSuccsLeft;
  
#ifndef NDEBUG
  if (PredSU->NumSuccsLeft < 0) {
    cerr << "*** List scheduling failed! ***\n";
    PredSU->dump(&DAG);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if (PredSU->NumSuccsLeft == 0) {
    // EntryToken has to go last!  Special case it here.
    if (!PredSU->Node || PredSU->Node->getOpcode() != ISD::EntryToken) {
      PredSU->isAvailable = true;
      AvailableQueue->push(PredSU);
    }
  }
}

/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeBottomUp(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(&DAG));
  SU->Cycle = CurCycle;

  AvailableQueue->ScheduledNode(SU);

  // Bottom up: release predecessors
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    ReleasePred(I->Dep, I->isCtrl, CurCycle);
    if (I->Cost < 0)  {
      // This is a physical register dependency and it's impossible or
      // expensive to copy the register. Make sure nothing that can 
      // clobber the register is scheduled between the predecessor and
      // this node.
      if (LiveRegs.insert(I->Reg)) {
        LiveRegDefs[I->Reg] = I->Dep;
        LiveRegCycles[I->Reg] = CurCycle;
      }
    }
  }

  // Release all the implicit physical register defs that are live.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->Cost < 0)  {
      if (LiveRegCycles[I->Reg] == I->Dep->Cycle) {
        LiveRegs.erase(I->Reg);
        assert(LiveRegDefs[I->Reg] == SU &&
               "Physical register dependency violated?");
        LiveRegDefs[I->Reg] = NULL;
        LiveRegCycles[I->Reg] = 0;
      }
    }
  }

  SU->isScheduled = true;
}

/// CapturePred - This does the opposite of ReleasePred. Since SU is being
/// unscheduled, incrcease the succ left count of its predecessors. Remove
/// them from AvailableQueue if necessary.
void ScheduleDAGRRList::CapturePred(SUnit *PredSU, SUnit *SU, bool isChain) {
  PredSU->CycleBound = 0;
  for (SUnit::succ_iterator I = PredSU->Succs.begin(), E = PredSU->Succs.end();
       I != E; ++I) {
    if (I->Dep == SU)
      continue;
    PredSU->CycleBound = std::max(PredSU->CycleBound,
                                  I->Dep->Cycle + PredSU->Latency);
  }

  if (PredSU->isAvailable) {
    PredSU->isAvailable = false;
    if (!PredSU->isPending)
      AvailableQueue->remove(PredSU);
  }

  ++PredSU->NumSuccsLeft;
}

/// UnscheduleNodeBottomUp - Remove the node from the schedule, update its and
/// its predecessor states to reflect the change.
void ScheduleDAGRRList::UnscheduleNodeBottomUp(SUnit *SU) {
  DOUT << "*** Unscheduling [" << SU->Cycle << "]: ";
  DEBUG(SU->dump(&DAG));

  AvailableQueue->UnscheduledNode(SU);

  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    CapturePred(I->Dep, SU, I->isCtrl);
    if (I->Cost < 0 && SU->Cycle == LiveRegCycles[I->Reg])  {
      LiveRegs.erase(I->Reg);
      assert(LiveRegDefs[I->Reg] == I->Dep &&
             "Physical register dependency violated?");
      LiveRegDefs[I->Reg] = NULL;
      LiveRegCycles[I->Reg] = 0;
    }
  }

  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->Cost < 0)  {
      if (LiveRegs.insert(I->Reg)) {
        assert(!LiveRegDefs[I->Reg] &&
               "Physical register dependency violated?");
        LiveRegDefs[I->Reg] = SU;
      }
      if (I->Dep->Cycle < LiveRegCycles[I->Reg])
        LiveRegCycles[I->Reg] = I->Dep->Cycle;
    }
  }

  SU->Cycle = 0;
  SU->isScheduled = false;
  SU->isAvailable = true;
  AvailableQueue->push(SU);
}

// FIXME: This is probably too slow!
static void isReachable(SUnit *SU, SUnit *TargetSU,
                        SmallPtrSet<SUnit*, 32> &Visited, bool &Reached) {
  if (Reached) return;
  if (SU == TargetSU) {
    Reached = true;
    return;
  }
  if (!Visited.insert(SU)) return;

  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end(); I != E;
       ++I)
    isReachable(I->Dep, TargetSU, Visited, Reached);
}

static bool isReachable(SUnit *SU, SUnit *TargetSU) {
  SmallPtrSet<SUnit*, 32> Visited;
  bool Reached = false;
  isReachable(SU, TargetSU, Visited, Reached);
  return Reached;
}

/// willCreateCycle - Returns true if adding an edge from SU to TargetSU will
/// create a cycle.
static bool WillCreateCycle(SUnit *SU, SUnit *TargetSU) {
  if (isReachable(TargetSU, SU))
    return true;
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I)
    if (I->Cost < 0 && isReachable(TargetSU, I->Dep))
      return true;
  return false;
}

/// BacktrackBottomUp - Backtrack scheduling to a previous cycle specified in
/// BTCycle in order to schedule a specific node. Returns the last unscheduled
/// SUnit. Also returns if a successor is unscheduled in the process.
void ScheduleDAGRRList::BacktrackBottomUp(SUnit *SU, unsigned BtCycle,
                                          unsigned &CurCycle) {
  SUnit *OldSU = NULL;
  while (CurCycle > BtCycle) {
    OldSU = Sequence.back();
    Sequence.pop_back();
    if (SU->isSucc(OldSU))
      // Don't try to remove SU from AvailableQueue.
      SU->isAvailable = false;
    UnscheduleNodeBottomUp(OldSU);
    --CurCycle;
  }

      
  if (SU->isSucc(OldSU)) {
    assert(false && "Something is wrong!");
    abort();
  }

  ++NumBacktracks;
}

/// CopyAndMoveSuccessors - Clone the specified node and move its scheduled
/// successors to the newly created node.
SUnit *ScheduleDAGRRList::CopyAndMoveSuccessors(SUnit *SU) {
  if (SU->FlaggedNodes.size())
    return NULL;

  SDNode *N = SU->Node;
  if (!N)
    return NULL;

  SUnit *NewSU;
  bool TryUnfold = false;
  for (unsigned i = 0, e = N->getNumValues(); i != e; ++i) {
    MVT::ValueType VT = N->getValueType(i);
    if (VT == MVT::Flag)
      return NULL;
    else if (VT == MVT::Other)
      TryUnfold = true;
  }
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    const SDOperand &Op = N->getOperand(i);
    MVT::ValueType VT = Op.Val->getValueType(Op.ResNo);
    if (VT == MVT::Flag)
      return NULL;
  }

  if (TryUnfold) {
    SmallVector<SDNode*, 4> NewNodes;
    if (!MRI->unfoldMemoryOperand(DAG, N, NewNodes))
      return NULL;

    DOUT << "Unfolding SU # " << SU->NodeNum << "\n";
    assert(NewNodes.size() == 2 && "Expected a load folding node!");

    N = NewNodes[1];
    SDNode *LoadNode = NewNodes[0];
    std::vector<SDNode*> Deleted;
    unsigned NumVals = N->getNumValues();
    unsigned OldNumVals = SU->Node->getNumValues();
    for (unsigned i = 0; i != NumVals; ++i)
      DAG.ReplaceAllUsesOfValueWith(SDOperand(SU->Node, i),
                                    SDOperand(N, i), Deleted);
    DAG.ReplaceAllUsesOfValueWith(SDOperand(SU->Node, OldNumVals-1),
                                  SDOperand(LoadNode, 1), Deleted);

    SUnit *LoadSU = NewSUnit(LoadNode);
    SUnit *NewSU = NewSUnit(N);
    SUnitMap[LoadNode].push_back(LoadSU);
    SUnitMap[N].push_back(NewSU);
    const TargetInstrDescriptor *TID = &TII->get(LoadNode->getTargetOpcode());
    for (unsigned i = 0; i != TID->numOperands; ++i) {
      if (TID->getOperandConstraint(i, TOI::TIED_TO) != -1) {
        LoadSU->isTwoAddress = true;
        break;
      }
    }
    if (TID->Flags & M_COMMUTABLE)
      LoadSU->isCommutable = true;

    TID = &TII->get(N->getTargetOpcode());
    for (unsigned i = 0; i != TID->numOperands; ++i) {
      if (TID->getOperandConstraint(i, TOI::TIED_TO) != -1) {
        NewSU->isTwoAddress = true;
        break;
      }
    }
    if (TID->Flags & M_COMMUTABLE)
      NewSU->isCommutable = true;

    // FIXME: Calculate height / depth and propagate the changes?
    LoadSU->Depth = NewSU->Depth = SU->Depth;
    LoadSU->Height = NewSU->Height = SU->Height;
    ComputeLatency(LoadSU);
    ComputeLatency(NewSU);

    SUnit *ChainPred = NULL;
    SmallVector<SDep, 4> ChainSuccs;
    SmallVector<SDep, 4> LoadPreds;
    SmallVector<SDep, 4> NodePreds;
    SmallVector<SDep, 4> NodeSuccs;
    for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->isCtrl)
        ChainPred = I->Dep;
      else if (I->Dep->Node && I->Dep->Node->isOperand(LoadNode))
        LoadPreds.push_back(SDep(I->Dep, I->Reg, I->Cost, false, false));
      else
        NodePreds.push_back(SDep(I->Dep, I->Reg, I->Cost, false, false));
    }
    for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      if (I->isCtrl)
        ChainSuccs.push_back(SDep(I->Dep, I->Reg, I->Cost,
                                  I->isCtrl, I->isSpecial));
      else
        NodeSuccs.push_back(SDep(I->Dep, I->Reg, I->Cost,
                                 I->isCtrl, I->isSpecial));
    }

    SU->removePred(ChainPred, true, false);
    LoadSU->addPred(ChainPred, true, false);
    for (unsigned i = 0, e = LoadPreds.size(); i != e; ++i) {
      SDep *Pred = &LoadPreds[i];
      SU->removePred(Pred->Dep, Pred->isCtrl, Pred->isSpecial);
      LoadSU->addPred(Pred->Dep, Pred->isCtrl, Pred->isSpecial,
                      Pred->Reg, Pred->Cost);
    }
    for (unsigned i = 0, e = NodePreds.size(); i != e; ++i) {
      SDep *Pred = &NodePreds[i];
      SU->removePred(Pred->Dep, Pred->isCtrl, Pred->isSpecial);
      NewSU->addPred(Pred->Dep, Pred->isCtrl, Pred->isSpecial,
                     Pred->Reg, Pred->Cost);
    }
    for (unsigned i = 0, e = NodeSuccs.size(); i != e; ++i) {
      SDep *Succ = &NodeSuccs[i];
      Succ->Dep->removePred(SU, Succ->isCtrl, Succ->isSpecial);
      Succ->Dep->addPred(NewSU, Succ->isCtrl, Succ->isSpecial,
                         Succ->Reg, Succ->Cost);
    }
    for (unsigned i = 0, e = ChainSuccs.size(); i != e; ++i) {
      SDep *Succ = &ChainSuccs[i];
      Succ->Dep->removePred(SU, Succ->isCtrl, Succ->isSpecial);
      Succ->Dep->addPred(LoadSU, Succ->isCtrl, Succ->isSpecial,
                         Succ->Reg, Succ->Cost);
    } 
    NewSU->addPred(LoadSU, false, false);

    AvailableQueue->addNode(LoadSU);
    AvailableQueue->addNode(NewSU);

    ++NumUnfolds;

    if (NewSU->NumSuccsLeft == 0) {
      NewSU->isAvailable = true;
      return NewSU;
    } else
      SU = NewSU;
  }

  DOUT << "Duplicating SU # " << SU->NodeNum << "\n";
  NewSU = Clone(SU);

  // New SUnit has the exact same predecessors.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I)
    if (!I->isSpecial) {
      NewSU->addPred(I->Dep, I->isCtrl, false, I->Reg, I->Cost);
      NewSU->Depth = std::max(NewSU->Depth, I->Dep->Depth+1);
    }

  // Only copy scheduled successors. Cut them from old node's successor
  // list and move them over.
  SmallVector<std::pair<SUnit*, bool>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isSpecial)
      continue;
    if (I->Dep->isScheduled) {
      NewSU->Height = std::max(NewSU->Height, I->Dep->Height+1);
      I->Dep->addPred(NewSU, I->isCtrl, false, I->Reg, I->Cost);
      DelDeps.push_back(std::make_pair(I->Dep, I->isCtrl));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i) {
    SUnit *Succ = DelDeps[i].first;
    bool isCtrl = DelDeps[i].second;
    Succ->removePred(SU, isCtrl, false);
  }

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(NewSU);

  ++NumDups;
  return NewSU;
}

/// InsertCCCopiesAndMoveSuccs - Insert expensive cross register class copies
/// and move all scheduled successors of the given SUnit to the last copy.
void ScheduleDAGRRList::InsertCCCopiesAndMoveSuccs(SUnit *SU, unsigned Reg,
                                              const TargetRegisterClass *DestRC,
                                              const TargetRegisterClass *SrcRC,
                                               SmallVector<SUnit*, 2> &Copies) {
  SUnit *CopyFromSU = NewSUnit(NULL);
  CopyFromSU->CopySrcRC = SrcRC;
  CopyFromSU->CopyDstRC = DestRC;
  CopyFromSU->Depth = SU->Depth;
  CopyFromSU->Height = SU->Height;

  SUnit *CopyToSU = NewSUnit(NULL);
  CopyToSU->CopySrcRC = DestRC;
  CopyToSU->CopyDstRC = SrcRC;

  // Only copy scheduled successors. Cut them from old node's successor
  // list and move them over.
  SmallVector<std::pair<SUnit*, bool>, 4> DelDeps;
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isSpecial)
      continue;
    if (I->Dep->isScheduled) {
      CopyToSU->Height = std::max(CopyToSU->Height, I->Dep->Height+1);
      I->Dep->addPred(CopyToSU, I->isCtrl, false, I->Reg, I->Cost);
      DelDeps.push_back(std::make_pair(I->Dep, I->isCtrl));
    }
  }
  for (unsigned i = 0, e = DelDeps.size(); i != e; ++i) {
    SUnit *Succ = DelDeps[i].first;
    bool isCtrl = DelDeps[i].second;
    Succ->removePred(SU, isCtrl, false);
  }

  CopyFromSU->addPred(SU, false, false, Reg, -1);
  CopyToSU->addPred(CopyFromSU, false, false, Reg, 1);

  AvailableQueue->updateNode(SU);
  AvailableQueue->addNode(CopyFromSU);
  AvailableQueue->addNode(CopyToSU);
  Copies.push_back(CopyFromSU);
  Copies.push_back(CopyToSU);

  ++NumCCCopies;
}

/// getPhysicalRegisterVT - Returns the ValueType of the physical register
/// definition of the specified node.
/// FIXME: Move to SelectionDAG?
static MVT::ValueType getPhysicalRegisterVT(SDNode *N, unsigned Reg,
                                            const TargetInstrInfo *TII) {
  const TargetInstrDescriptor &TID = TII->get(N->getTargetOpcode());
  assert(TID.ImplicitDefs && "Physical reg def must be in implicit def list!");
  unsigned NumRes = TID.numDefs;
  for (const unsigned *ImpDef = TID.ImplicitDefs; *ImpDef; ++ImpDef) {
    if (Reg == *ImpDef)
      break;
    ++NumRes;
  }
  return N->getValueType(NumRes);
}

/// DelayForLiveRegsBottomUp - Returns true if it is necessary to delay
/// scheduling of the given node to satisfy live physical register dependencies.
/// If the specific node is the last one that's available to schedule, do
/// whatever is necessary (i.e. backtracking or cloning) to make it possible.
bool ScheduleDAGRRList::DelayForLiveRegsBottomUp(SUnit *SU,
                                                 SmallVector<unsigned, 4> &LRegs){
  if (LiveRegs.empty())
    return false;

  SmallSet<unsigned, 4> RegAdded;
  // If this node would clobber any "live" register, then it's not ready.
  for (SUnit::pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->Cost < 0)  {
      unsigned Reg = I->Reg;
      if (LiveRegs.count(Reg) && LiveRegDefs[Reg] != I->Dep) {
        if (RegAdded.insert(Reg))
          LRegs.push_back(Reg);
      }
      for (const unsigned *Alias = MRI->getAliasSet(Reg);
           *Alias; ++Alias)
        if (LiveRegs.count(*Alias) && LiveRegDefs[*Alias] != I->Dep) {
          if (RegAdded.insert(*Alias))
            LRegs.push_back(*Alias);
        }
    }
  }

  for (unsigned i = 0, e = SU->FlaggedNodes.size()+1; i != e; ++i) {
    SDNode *Node = (i == 0) ? SU->Node : SU->FlaggedNodes[i-1];
    if (!Node || !Node->isTargetOpcode())
      continue;
    const TargetInstrDescriptor &TID = TII->get(Node->getTargetOpcode());
    if (!TID.ImplicitDefs)
      continue;
    for (const unsigned *Reg = TID.ImplicitDefs; *Reg; ++Reg) {
      if (LiveRegs.count(*Reg) && LiveRegDefs[*Reg] != SU) {
        if (RegAdded.insert(*Reg))
          LRegs.push_back(*Reg);
      }
      for (const unsigned *Alias = MRI->getAliasSet(*Reg);
           *Alias; ++Alias)
        if (LiveRegs.count(*Alias) && LiveRegDefs[*Alias] != SU) {
          if (RegAdded.insert(*Alias))
            LRegs.push_back(*Alias);
        }
    }
  }
  return !LRegs.empty();
}


/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGRRList::ListScheduleBottomUp() {
  unsigned CurCycle = 0;
  // Add root to Available queue.
  SUnit *RootSU = SUnitMap[DAG.getRoot().Val].front();
  RootSU->isAvailable = true;
  AvailableQueue->push(RootSU);

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  SmallVector<SUnit*, 4> NotReady;
  while (!AvailableQueue->empty()) {
    bool Delayed = false;
    DenseMap<SUnit*, SmallVector<unsigned, 4> > LRegsMap;
    SUnit *CurSU = AvailableQueue->pop();
    while (CurSU) {
      if (CurSU->CycleBound <= CurCycle) {
        SmallVector<unsigned, 4> LRegs;
        if (!DelayForLiveRegsBottomUp(CurSU, LRegs))
          break;
        Delayed = true;
        LRegsMap.insert(std::make_pair(CurSU, LRegs));
      }

      CurSU->isPending = true;  // This SU is not in AvailableQueue right now.
      NotReady.push_back(CurSU);
      CurSU = AvailableQueue->pop();
    }

    // All candidates are delayed due to live physical reg dependencies.
    // Try backtracking, code duplication, or inserting cross class copies
    // to resolve it.
    if (Delayed && !CurSU) {
      for (unsigned i = 0, e = NotReady.size(); i != e; ++i) {
        SUnit *TrySU = NotReady[i];
        SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];

        // Try unscheduling up to the point where it's safe to schedule
        // this node.
        unsigned LiveCycle = CurCycle;
        for (unsigned j = 0, ee = LRegs.size(); j != ee; ++j) {
          unsigned Reg = LRegs[j];
          unsigned LCycle = LiveRegCycles[Reg];
          LiveCycle = std::min(LiveCycle, LCycle);
        }
        SUnit *OldSU = Sequence[LiveCycle];
        if (!WillCreateCycle(TrySU, OldSU))  {
          BacktrackBottomUp(TrySU, LiveCycle, CurCycle);
          // Force the current node to be scheduled before the node that
          // requires the physical reg dep.
          if (OldSU->isAvailable) {
            OldSU->isAvailable = false;
            AvailableQueue->remove(OldSU);
          }
          TrySU->addPred(OldSU, true, true);
          // If one or more successors has been unscheduled, then the current
          // node is no longer avaialable. Schedule a successor that's now
          // available instead.
          if (!TrySU->isAvailable)
            CurSU = AvailableQueue->pop();
          else {
            CurSU = TrySU;
            TrySU->isPending = false;
            NotReady.erase(NotReady.begin()+i);
          }
          break;
        }
      }

      if (!CurSU) {
        // Can't backtrace. Try duplicating the nodes that produces these
        // "expensive to copy" values to break the dependency. In case even
        // that doesn't work, insert cross class copies.
        SUnit *TrySU = NotReady[0];
        SmallVector<unsigned, 4> &LRegs = LRegsMap[TrySU];
        assert(LRegs.size() == 1 && "Can't handle this yet!");
        unsigned Reg = LRegs[0];
        SUnit *LRDef = LiveRegDefs[Reg];
        SUnit *NewDef = CopyAndMoveSuccessors(LRDef);
        if (!NewDef) {
          // Issue expensive cross register class copies.
          MVT::ValueType VT = getPhysicalRegisterVT(LRDef->Node, Reg, TII);
          const TargetRegisterClass *RC =
            MRI->getPhysicalRegisterRegClass(VT, Reg);
          const TargetRegisterClass *DestRC = MRI->getCrossCopyRegClass(RC);
          if (!DestRC) {
            assert(false && "Don't know how to copy this physical register!");
            abort();
          }
          SmallVector<SUnit*, 2> Copies;
          InsertCCCopiesAndMoveSuccs(LRDef, Reg, DestRC, RC, Copies);
          DOUT << "Adding an edge from SU # " << TrySU->NodeNum
               << " to SU #" << Copies.front()->NodeNum << "\n";
          TrySU->addPred(Copies.front(), true, true);
          NewDef = Copies.back();
        }

        DOUT << "Adding an edge from SU # " << NewDef->NodeNum
             << " to SU #" << TrySU->NodeNum << "\n";
        LiveRegDefs[Reg] = NewDef;
        NewDef->addPred(TrySU, true, true);
        TrySU->isAvailable = false;
        CurSU = NewDef;
      }

      if (!CurSU) {
        assert(false && "Unable to resolve live physical register dependencies!");
        abort();
      }
    }

    // Add the nodes that aren't ready back onto the available list.
    for (unsigned i = 0, e = NotReady.size(); i != e; ++i) {
      NotReady[i]->isPending = false;
      // May no longer be available due to backtracking.
      if (NotReady[i]->isAvailable)
        AvailableQueue->push(NotReady[i]);
    }
    NotReady.clear();

    if (!CurSU)
      Sequence.push_back(0);
    else {
      ScheduleNodeBottomUp(CurSU, CurCycle);
      Sequence.push_back(CurSU);
    }
    ++CurCycle;
  }

  // Add entry node last
  if (DAG.getEntryNode().Val != DAG.getRoot().Val) {
    SUnit *Entry = SUnitMap[DAG.getEntryNode().Val].front();
    Sequence.push_back(Entry);
  }

  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());
  
  
#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (SUnits[i].NumSuccsLeft != 0) {
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(&DAG);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the AvailableQueue if the count reaches zero. Also update its cycle bound.
void ScheduleDAGRRList::ReleaseSucc(SUnit *SuccSU, bool isChain, 
                                    unsigned CurCycle) {
  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  SuccSU->CycleBound = std::max(SuccSU->CycleBound, CurCycle + SuccSU->Latency);

  --SuccSU->NumPredsLeft;
  
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft < 0) {
    cerr << "*** List scheduling failed! ***\n";
    SuccSU->dump(&DAG);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if (SuccSU->NumPredsLeft == 0) {
    SuccSU->isAvailable = true;
    AvailableQueue->push(SuccSU);
  }
}


/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGRRList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(&DAG));
  SU->Cycle = CurCycle;

  AvailableQueue->ScheduledNode(SU);

  // Top down: release successors
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    ReleaseSucc(I->Dep, I->isCtrl, CurCycle);
  SU->isScheduled = true;
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGRRList::ListScheduleTopDown() {
  unsigned CurCycle = 0;
  SUnit *Entry = SUnitMap[DAG.getEntryNode().Val].front();

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.size() == 0 && &SUnits[i] != Entry) {
      AvailableQueue->push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }
  
  // Emit the entry node first.
  ScheduleNodeTopDown(Entry, CurCycle);
  Sequence.push_back(Entry);
  ++CurCycle;

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  while (!AvailableQueue->empty()) {
    SUnit *CurSU = AvailableQueue->pop();
    while (CurSU && CurSU->CycleBound > CurCycle) {
      NotReady.push_back(CurSU);
      CurSU = AvailableQueue->pop();
    }
    
    // Add the nodes that aren't ready back onto the available list.
    AvailableQueue->push_all(NotReady);
    NotReady.clear();

    if (!CurSU)
      Sequence.push_back(0);
    else {
      ScheduleNodeTopDown(CurSU, CurCycle);
      Sequence.push_back(CurSU);
    }
    CurCycle++;
  }
  
  
#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (!SUnits[i].isScheduled) {
      if (!AnyNotSched)
        cerr << "*** List scheduling failed! ***\n";
      SUnits[i].dump(&DAG);
      cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}



//===----------------------------------------------------------------------===//
//                RegReductionPriorityQueue Implementation
//===----------------------------------------------------------------------===//
//
// This is a SchedulingPriorityQueue that schedules using Sethi Ullman numbers
// to reduce register pressure.
// 
namespace {
  template<class SF>
  class RegReductionPriorityQueue;
  
  /// Sorting functions for the Available queue.
  struct bu_ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    RegReductionPriorityQueue<bu_ls_rr_sort> *SPQ;
    bu_ls_rr_sort(RegReductionPriorityQueue<bu_ls_rr_sort> *spq) : SPQ(spq) {}
    bu_ls_rr_sort(const bu_ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };

  struct td_ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    RegReductionPriorityQueue<td_ls_rr_sort> *SPQ;
    td_ls_rr_sort(RegReductionPriorityQueue<td_ls_rr_sort> *spq) : SPQ(spq) {}
    td_ls_rr_sort(const td_ls_rr_sort &RHS) : SPQ(RHS.SPQ) {}
    
    bool operator()(const SUnit* left, const SUnit* right) const;
  };
}  // end anonymous namespace

static inline bool isCopyFromLiveIn(const SUnit *SU) {
  SDNode *N = SU->Node;
  return N && N->getOpcode() == ISD::CopyFromReg &&
    N->getOperand(N->getNumOperands()-1).getValueType() != MVT::Flag;
}

namespace {
  template<class SF>
  class VISIBILITY_HIDDEN RegReductionPriorityQueue
   : public SchedulingPriorityQueue {
    std::priority_queue<SUnit*, std::vector<SUnit*>, SF> Queue;

  public:
    RegReductionPriorityQueue() :
    Queue(SF(this)) {}
    
    virtual void initNodes(DenseMap<SDNode*, std::vector<SUnit*> > &sumap,
                           std::vector<SUnit> &sunits) {}

    virtual void addNode(const SUnit *SU) {}

    virtual void updateNode(const SUnit *SU) {}

    virtual void releaseState() {}
    
    virtual unsigned getNodePriority(const SUnit *SU) const {
      return 0;
    }
    
    unsigned size() const { return Queue.size(); }

    bool empty() const { return Queue.empty(); }
    
    void push(SUnit *U) {
      Queue.push(U);
    }
    void push_all(const std::vector<SUnit *> &Nodes) {
      for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
        Queue.push(Nodes[i]);
    }
    
    SUnit *pop() {
      if (empty()) return NULL;
      SUnit *V = Queue.top();
      Queue.pop();
      return V;
    }

    /// remove - This is a really inefficient way to remove a node from a
    /// priority queue.  We should roll our own heap to make this better or
    /// something.
    void remove(SUnit *SU) {
      std::vector<SUnit*> Temp;
      
      assert(!Queue.empty() && "Not in queue!");
      while (Queue.top() != SU) {
        Temp.push_back(Queue.top());
        Queue.pop();
        assert(!Queue.empty() && "Not in queue!");
      }

      // Remove the node from the PQ.
      Queue.pop();
      
      // Add all the other nodes back.
      for (unsigned i = 0, e = Temp.size(); i != e; ++i)
        Queue.push(Temp[i]);
    }
  };

  template<class SF>
  class VISIBILITY_HIDDEN BURegReductionPriorityQueue
   : public RegReductionPriorityQueue<SF> {
    // SUnitMap SDNode to SUnit mapping (n -> n).
    DenseMap<SDNode*, std::vector<SUnit*> > *SUnitMap;

    // SUnits - The SUnits for the current graph.
    const std::vector<SUnit> *SUnits;
    
    // SethiUllmanNumbers - The SethiUllman number for each node.
    std::vector<unsigned> SethiUllmanNumbers;

    const TargetInstrInfo *TII;
  public:
    explicit BURegReductionPriorityQueue(const TargetInstrInfo *tii)
      : TII(tii) {}

    void initNodes(DenseMap<SDNode*, std::vector<SUnit*> > &sumap,
                   std::vector<SUnit> &sunits) {
      SUnitMap = &sumap;
      SUnits = &sunits;
      // Add pseudo dependency edges for two-address nodes.
      AddPseudoTwoAddrDeps();
      // Calculate node priorities.
      CalculateSethiUllmanNumbers();
    }

    void addNode(const SUnit *SU) {
      SethiUllmanNumbers.resize(SUnits->size(), 0);
      CalcNodeSethiUllmanNumber(SU);
    }

    void updateNode(const SUnit *SU) {
      SethiUllmanNumbers[SU->NodeNum] = 0;
      CalcNodeSethiUllmanNumber(SU);
    }

    void releaseState() {
      SUnits = 0;
      SethiUllmanNumbers.clear();
    }

    unsigned getNodePriority(const SUnit *SU) const {
      assert(SU->NodeNum < SethiUllmanNumbers.size());
      unsigned Opc = SU->Node ? SU->Node->getOpcode() : 0;
      if (Opc == ISD::CopyFromReg && !isCopyFromLiveIn(SU))
        // CopyFromReg should be close to its def because it restricts
        // allocation choices. But if it is a livein then perhaps we want it
        // closer to its uses so it can be coalesced.
        return 0xffff;
      else if (Opc == ISD::TokenFactor || Opc == ISD::CopyToReg)
        // CopyToReg should be close to its uses to facilitate coalescing and
        // avoid spilling.
        return 0;
      else if (SU->NumSuccs == 0)
        // If SU does not have a use, i.e. it doesn't produce a value that would
        // be consumed (e.g. store), then it terminates a chain of computation.
        // Give it a large SethiUllman number so it will be scheduled right
        // before its predecessors that it doesn't lengthen their live ranges.
        return 0xffff;
      else if (SU->NumPreds == 0)
        // If SU does not have a def, schedule it close to its uses because it
        // does not lengthen any live ranges.
        return 0;
      else
        return SethiUllmanNumbers[SU->NodeNum];
    }

  private:
    bool canClobber(SUnit *SU, SUnit *Op);
    void AddPseudoTwoAddrDeps();
    void CalculateSethiUllmanNumbers();
    unsigned CalcNodeSethiUllmanNumber(const SUnit *SU);
  };


  template<class SF>
  class VISIBILITY_HIDDEN TDRegReductionPriorityQueue
   : public RegReductionPriorityQueue<SF> {
    // SUnitMap SDNode to SUnit mapping (n -> n).
    DenseMap<SDNode*, std::vector<SUnit*> > *SUnitMap;

    // SUnits - The SUnits for the current graph.
    const std::vector<SUnit> *SUnits;
    
    // SethiUllmanNumbers - The SethiUllman number for each node.
    std::vector<unsigned> SethiUllmanNumbers;

  public:
    TDRegReductionPriorityQueue() {}

    void initNodes(DenseMap<SDNode*, std::vector<SUnit*> > &sumap,
                   std::vector<SUnit> &sunits) {
      SUnitMap = &sumap;
      SUnits = &sunits;
      // Calculate node priorities.
      CalculateSethiUllmanNumbers();
    }

    void addNode(const SUnit *SU) {
      SethiUllmanNumbers.resize(SUnits->size(), 0);
      CalcNodeSethiUllmanNumber(SU);
    }

    void updateNode(const SUnit *SU) {
      SethiUllmanNumbers[SU->NodeNum] = 0;
      CalcNodeSethiUllmanNumber(SU);
    }

    void releaseState() {
      SUnits = 0;
      SethiUllmanNumbers.clear();
    }

    unsigned getNodePriority(const SUnit *SU) const {
      assert(SU->NodeNum < SethiUllmanNumbers.size());
      return SethiUllmanNumbers[SU->NodeNum];
    }

  private:
    void CalculateSethiUllmanNumbers();
    unsigned CalcNodeSethiUllmanNumber(const SUnit *SU);
  };
}

/// closestSucc - Returns the scheduled cycle of the successor which is
/// closet to the current cycle.
static unsigned closestSucc(const SUnit *SU) {
  unsigned MaxCycle = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    unsigned Cycle = I->Dep->Cycle;
    // If there are bunch of CopyToRegs stacked up, they should be considered
    // to be at the same position.
    if (I->Dep->Node && I->Dep->Node->getOpcode() == ISD::CopyToReg)
      Cycle = closestSucc(I->Dep)+1;
    if (Cycle > MaxCycle)
      MaxCycle = Cycle;
  }
  return MaxCycle;
}

// Bottom up
bool bu_ls_rr_sort::operator()(const SUnit *left, const SUnit *right) const {
  // There used to be a special tie breaker here that looked for
  // two-address instructions and preferred the instruction with a
  // def&use operand.  The special case triggered diagnostics when
  // _GLIBCXX_DEBUG was enabled because it broke the strict weak
  // ordering that priority_queue requires. It didn't help much anyway
  // because AddPseudoTwoAddrDeps already covers many of the cases
  // where it would have applied.  In addition, it's counter-intuitive
  // that a tie breaker would be the first thing attempted.  There's a
  // "real" tie breaker below that is the operation of last resort.
  // The fact that the "special tie breaker" would trigger when there
  // wasn't otherwise a tie is what broke the strict weak ordering
  // constraint.

  unsigned LPriority = SPQ->getNodePriority(left);
  unsigned RPriority = SPQ->getNodePriority(right);
  if (LPriority > RPriority)
    return true;
  else if (LPriority == RPriority) {
    // Try schedule def + use closer when Sethi-Ullman numbers are the same.
    // e.g.
    // t1 = op t2, c1
    // t3 = op t4, c2
    //
    // and the following instructions are both ready.
    // t2 = op c3
    // t4 = op c4
    //
    // Then schedule t2 = op first.
    // i.e.
    // t4 = op c4
    // t2 = op c3
    // t1 = op t2, c1
    // t3 = op t4, c2
    //
    // This creates more short live intervals.
    unsigned LDist = closestSucc(left);
    unsigned RDist = closestSucc(right);
    if (LDist < RDist)
      return true;
    else if (LDist == RDist) {
      if (left->Height > right->Height)
        return true;
      else if (left->Height == right->Height)
        if (left->Depth < right->Depth)
          return true;
        else if (left->Depth == right->Depth)
          if (left->CycleBound > right->CycleBound) 
            return true;
    }
  }
  return false;
}

template<class SF>
bool BURegReductionPriorityQueue<SF>::canClobber(SUnit *SU, SUnit *Op) {
  if (SU->isTwoAddress) {
    unsigned Opc = SU->Node->getTargetOpcode();
    unsigned NumRes = TII->getNumDefs(Opc);
    unsigned NumOps = ScheduleDAG::CountOperands(SU->Node);
    for (unsigned i = 0; i != NumOps; ++i) {
      if (TII->getOperandConstraint(Opc, i+NumRes, TOI::TIED_TO) != -1) {
        SDNode *DU = SU->Node->getOperand(i).Val;
        if (Op == (*SUnitMap)[DU][SU->InstanceNo])
          return true;
      }
    }
  }
  return false;
}


/// hasCopyToRegUse - Return true if SU has a value successor that is a
/// CopyToReg node.
static bool hasCopyToRegUse(SUnit *SU) {
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    if (I->isCtrl) continue;
    SUnit *SuccSU = I->Dep;
    if (SuccSU->Node && SuccSU->Node->getOpcode() == ISD::CopyToReg)
      return true;
  }
  return false;
}

/// AddPseudoTwoAddrDeps - If two nodes share an operand and one of them uses
/// it as a def&use operand. Add a pseudo control edge from it to the other
/// node (if it won't create a cycle) so the two-address one will be scheduled
/// first (lower in the schedule). If both nodes are two-address, favor the
/// one that has a CopyToReg use (more likely to be a loop induction update).
/// If both are two-address, but one is commutable while the other is not
/// commutable, favor the one that's not commutable.
template<class SF>
void BURegReductionPriorityQueue<SF>::AddPseudoTwoAddrDeps() {
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i) {
    SUnit *SU = (SUnit *)&((*SUnits)[i]);
    if (!SU->isTwoAddress)
      continue;

    SDNode *Node = SU->Node;
    if (!Node || !Node->isTargetOpcode() || SU->FlaggedNodes.size() > 0)
      continue;

    unsigned Opc = Node->getTargetOpcode();
    unsigned NumRes = TII->getNumDefs(Opc);
    unsigned NumOps = ScheduleDAG::CountOperands(Node);
    for (unsigned j = 0; j != NumOps; ++j) {
      if (TII->getOperandConstraint(Opc, j+NumRes, TOI::TIED_TO) != -1) {
        SDNode *DU = SU->Node->getOperand(j).Val;
        SUnit *DUSU = (*SUnitMap)[DU][SU->InstanceNo];
        if (!DUSU) continue;
        for (SUnit::succ_iterator I = DUSU->Succs.begin(),E = DUSU->Succs.end();
             I != E; ++I) {
          if (I->isCtrl) continue;
          SUnit *SuccSU = I->Dep;
          // Don't constraint nodes with implicit defs. It can create cycles
          // plus it may increase register pressures.
          if (SuccSU == SU || SuccSU->hasPhysRegDefs)
            continue;
          // Be conservative. Ignore if nodes aren't at the same depth.
          if (SuccSU->Depth != SU->Depth)
            continue;
          if ((!canClobber(SuccSU, DUSU) ||
               (hasCopyToRegUse(SU) && !hasCopyToRegUse(SuccSU)) ||
               (!SU->isCommutable && SuccSU->isCommutable)) &&
              !isReachable(SuccSU, SU)) {
            DOUT << "Adding an edge from SU # " << SU->NodeNum
                 << " to SU #" << SuccSU->NodeNum << "\n";
            SU->addPred(SuccSU, true, true);
          }
        }
      }
    }
  }
}

/// CalcNodeSethiUllmanNumber - Priority is the Sethi Ullman number. 
/// Smaller number is the higher priority.
template<class SF>
unsigned BURegReductionPriorityQueue<SF>::
CalcNodeSethiUllmanNumber(const SUnit *SU) {
  unsigned &SethiUllmanNumber = SethiUllmanNumbers[SU->NodeNum];
  if (SethiUllmanNumber != 0)
    return SethiUllmanNumber;

  unsigned Extra = 0;
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl) continue;  // ignore chain preds
    SUnit *PredSU = I->Dep;
    unsigned PredSethiUllman = CalcNodeSethiUllmanNumber(PredSU);
    if (PredSethiUllman > SethiUllmanNumber) {
      SethiUllmanNumber = PredSethiUllman;
      Extra = 0;
    } else if (PredSethiUllman == SethiUllmanNumber && !I->isCtrl)
      ++Extra;
  }

  SethiUllmanNumber += Extra;

  if (SethiUllmanNumber == 0)
    SethiUllmanNumber = 1;
  
  return SethiUllmanNumber;
}

/// CalculateSethiUllmanNumbers - Calculate Sethi-Ullman numbers of all
/// scheduling units.
template<class SF>
void BURegReductionPriorityQueue<SF>::CalculateSethiUllmanNumbers() {
  SethiUllmanNumbers.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodeSethiUllmanNumber(&(*SUnits)[i]);
}

static unsigned SumOfUnscheduledPredsOfSuccs(const SUnit *SU) {
  unsigned Sum = 0;
  for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I) {
    SUnit *SuccSU = I->Dep;
    for (SUnit::const_pred_iterator II = SuccSU->Preds.begin(),
         EE = SuccSU->Preds.end(); II != EE; ++II) {
      SUnit *PredSU = II->Dep;
      if (!PredSU->isScheduled)
        ++Sum;
    }
  }

  return Sum;
}


// Top down
bool td_ls_rr_sort::operator()(const SUnit *left, const SUnit *right) const {
  unsigned LPriority = SPQ->getNodePriority(left);
  unsigned RPriority = SPQ->getNodePriority(right);
  bool LIsTarget = left->Node && left->Node->isTargetOpcode();
  bool RIsTarget = right->Node && right->Node->isTargetOpcode();
  bool LIsFloater = LIsTarget && left->NumPreds == 0;
  bool RIsFloater = RIsTarget && right->NumPreds == 0;
  unsigned LBonus = (SumOfUnscheduledPredsOfSuccs(left) == 1) ? 2 : 0;
  unsigned RBonus = (SumOfUnscheduledPredsOfSuccs(right) == 1) ? 2 : 0;

  if (left->NumSuccs == 0 && right->NumSuccs != 0)
    return false;
  else if (left->NumSuccs != 0 && right->NumSuccs == 0)
    return true;

  // Special tie breaker: if two nodes share a operand, the one that use it
  // as a def&use operand is preferred.
  if (LIsTarget && RIsTarget) {
    if (left->isTwoAddress && !right->isTwoAddress) {
      SDNode *DUNode = left->Node->getOperand(0).Val;
      if (DUNode->isOperand(right->Node))
        RBonus += 2;
    }
    if (!left->isTwoAddress && right->isTwoAddress) {
      SDNode *DUNode = right->Node->getOperand(0).Val;
      if (DUNode->isOperand(left->Node))
        LBonus += 2;
    }
  }
  if (LIsFloater)
    LBonus -= 2;
  if (RIsFloater)
    RBonus -= 2;
  if (left->NumSuccs == 1)
    LBonus += 2;
  if (right->NumSuccs == 1)
    RBonus += 2;

  if (LPriority+LBonus < RPriority+RBonus)
    return true;
  else if (LPriority == RPriority)
    if (left->Depth < right->Depth)
      return true;
    else if (left->Depth == right->Depth)
      if (left->NumSuccsLeft > right->NumSuccsLeft)
        return true;
      else if (left->NumSuccsLeft == right->NumSuccsLeft)
        if (left->CycleBound > right->CycleBound) 
          return true;
  return false;
}

/// CalcNodeSethiUllmanNumber - Priority is the Sethi Ullman number. 
/// Smaller number is the higher priority.
template<class SF>
unsigned TDRegReductionPriorityQueue<SF>::
CalcNodeSethiUllmanNumber(const SUnit *SU) {
  unsigned &SethiUllmanNumber = SethiUllmanNumbers[SU->NodeNum];
  if (SethiUllmanNumber != 0)
    return SethiUllmanNumber;

  unsigned Opc = SU->Node ? SU->Node->getOpcode() : 0;
  if (Opc == ISD::TokenFactor || Opc == ISD::CopyToReg)
    SethiUllmanNumber = 0xffff;
  else if (SU->NumSuccsLeft == 0)
    // If SU does not have a use, i.e. it doesn't produce a value that would
    // be consumed (e.g. store), then it terminates a chain of computation.
    // Give it a small SethiUllman number so it will be scheduled right before
    // its predecessors that it doesn't lengthen their live ranges.
    SethiUllmanNumber = 0;
  else if (SU->NumPredsLeft == 0 &&
           (Opc != ISD::CopyFromReg || isCopyFromLiveIn(SU)))
    SethiUllmanNumber = 0xffff;
  else {
    int Extra = 0;
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      if (I->isCtrl) continue;  // ignore chain preds
      SUnit *PredSU = I->Dep;
      unsigned PredSethiUllman = CalcNodeSethiUllmanNumber(PredSU);
      if (PredSethiUllman > SethiUllmanNumber) {
        SethiUllmanNumber = PredSethiUllman;
        Extra = 0;
      } else if (PredSethiUllman == SethiUllmanNumber && !I->isCtrl)
        ++Extra;
    }

    SethiUllmanNumber += Extra;
  }
  
  return SethiUllmanNumber;
}

/// CalculateSethiUllmanNumbers - Calculate Sethi-Ullman numbers of all
/// scheduling units.
template<class SF>
void TDRegReductionPriorityQueue<SF>::CalculateSethiUllmanNumbers() {
  SethiUllmanNumbers.assign(SUnits->size(), 0);
  
  for (unsigned i = 0, e = SUnits->size(); i != e; ++i)
    CalcNodeSethiUllmanNumber(&(*SUnits)[i]);
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

llvm::ScheduleDAG* llvm::createBURRListDAGScheduler(SelectionDAGISel *IS,
                                                    SelectionDAG *DAG,
                                                    MachineBasicBlock *BB) {
  const TargetInstrInfo *TII = DAG->getTarget().getInstrInfo();
  return new ScheduleDAGRRList(*DAG, BB, DAG->getTarget(), true,
                           new BURegReductionPriorityQueue<bu_ls_rr_sort>(TII));
}

llvm::ScheduleDAG* llvm::createTDRRListDAGScheduler(SelectionDAGISel *IS,
                                                    SelectionDAG *DAG,
                                                    MachineBasicBlock *BB) {
  return new ScheduleDAGRRList(*DAG, BB, DAG->getTarget(), false,
                              new TDRegReductionPriorityQueue<td_ls_rr_sort>());
}


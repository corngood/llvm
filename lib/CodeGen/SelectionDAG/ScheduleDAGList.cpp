//===---- ScheduleDAGList.cpp - Implement a list scheduler for isel DAG ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements bottom-up and top-down list schedulers, using standard
// algorithms.  The basic approach uses a priority queue of available nodes to
// schedule.  One at a time, nodes are taken from the priority queue (thus in
// priority order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <climits>
#include <iostream>
#include <queue>
#include <set>
#include <vector>
using namespace llvm;

namespace {
  Statistic<> NumNoops ("scheduler", "Number of noops inserted");
  Statistic<> NumStalls("scheduler", "Number of pipeline stalls");

/// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or a
/// group of nodes flagged together.
struct SUnit {
  SDNode *Node;                       // Representative node.
  std::vector<SDNode*> FlaggedNodes;  // All nodes flagged to Node.
  std::set<SUnit*> Preds;             // All real predecessors.
  std::set<SUnit*> ChainPreds;        // All chain predecessors.
  std::set<SUnit*> Succs;             // All real successors.
  std::set<SUnit*> ChainSuccs;        // All chain successors.
  int NumPredsLeft;                   // # of preds not scheduled.
  int NumSuccsLeft;                   // # of succs not scheduled.
  int NumChainPredsLeft;              // # of chain preds not scheduled.
  int NumChainSuccsLeft;              // # of chain succs not scheduled.
  int SethiUllman;                    // Sethi Ullman number.
  bool isTwoAddress;                  // Is a two-address instruction.
  bool isDefNUseOperand;              // Is a def&use operand.
  unsigned Latency;                   // Node latency.
  unsigned CycleBound;                // Upper/lower cycle to be scheduled at.
  unsigned Slot;                      // Cycle node is scheduled at.
  SUnit *Next;

  SUnit(SDNode *node)
    : Node(node), NumPredsLeft(0), NumSuccsLeft(0),
      NumChainPredsLeft(0), NumChainSuccsLeft(0),
      SethiUllman(INT_MIN),
      isTwoAddress(false), isDefNUseOperand(false),
      Latency(0), CycleBound(0), Slot(0), Next(NULL) {}

  void dump(const SelectionDAG *G, bool All=true) const;
};

void SUnit::dump(const SelectionDAG *G, bool All) const {
  std::cerr << "SU: ";
  Node->dump(G);
  std::cerr << "\n";
  if (FlaggedNodes.size() != 0) {
    for (unsigned i = 0, e = FlaggedNodes.size(); i != e; i++) {
      std::cerr << "    ";
      FlaggedNodes[i]->dump(G);
      std::cerr << "\n";
    }
  }

  if (All) {
    std::cerr << "  # preds left       : " << NumPredsLeft << "\n";
    std::cerr << "  # succs left       : " << NumSuccsLeft << "\n";
    std::cerr << "  # chain preds left : " << NumChainPredsLeft << "\n";
    std::cerr << "  # chain succs left : " << NumChainSuccsLeft << "\n";
    std::cerr << "  Latency            : " << Latency << "\n";
    std::cerr << "  SethiUllman        : " << SethiUllman << "\n";

    if (Preds.size() != 0) {
      std::cerr << "  Predecessors:\n";
      for (std::set<SUnit*>::const_iterator I = Preds.begin(),
             E = Preds.end(); I != E; ++I) {
        std::cerr << "    ";
        (*I)->dump(G, false);
      }
    }
    if (ChainPreds.size() != 0) {
      std::cerr << "  Chained Preds:\n";
      for (std::set<SUnit*>::const_iterator I = ChainPreds.begin(),
             E = ChainPreds.end(); I != E; ++I) {
        std::cerr << "    ";
        (*I)->dump(G, false);
      }
    }
    if (Succs.size() != 0) {
      std::cerr << "  Successors:\n";
      for (std::set<SUnit*>::const_iterator I = Succs.begin(),
             E = Succs.end(); I != E; ++I) {
        std::cerr << "    ";
        (*I)->dump(G, false);
      }
    }
    if (ChainSuccs.size() != 0) {
      std::cerr << "  Chained succs:\n";
      for (std::set<SUnit*>::const_iterator I = ChainSuccs.begin(),
             E = ChainSuccs.end(); I != E; ++I) {
        std::cerr << "    ";
        (*I)->dump(G, false);
      }
    }
  }
}

/// Sorting functions for the Available queue.
struct ls_rr_sort : public std::binary_function<SUnit*, SUnit*, bool> {
  bool operator()(const SUnit* left, const SUnit* right) const {
    int LBonus = (int)left ->isDefNUseOperand;
    int RBonus = (int)right->isDefNUseOperand;

    // Special tie breaker: if two nodes share a operand, the one that
    // use it as a def&use operand is preferred.
    if (left->isTwoAddress && !right->isTwoAddress) {
      SDNode *DUNode = left->Node->getOperand(0).Val;
      if (DUNode->isOperand(right->Node))
        LBonus++;
    }
    if (!left->isTwoAddress && right->isTwoAddress) {
      SDNode *DUNode = right->Node->getOperand(0).Val;
      if (DUNode->isOperand(left->Node))
        RBonus++;
    }

    // Priority1 is just the number of live range genned.
    int LPriority1 = left ->NumPredsLeft - LBonus;
    int RPriority1 = right->NumPredsLeft - RBonus;
    int LPriority2 = left ->SethiUllman + LBonus;
    int RPriority2 = right->SethiUllman + RBonus;

    if (LPriority1 > RPriority1)
      return true;
    else if (LPriority1 == RPriority1)
      if (LPriority2 < RPriority2)
        return true;
      else if (LPriority2 == RPriority2)
        if (left->CycleBound > right->CycleBound) 
          return true;

    return false;
  }
};


/// ScheduleDAGList - List scheduler.
class ScheduleDAGList : public ScheduleDAG {
private:
  // SDNode to SUnit mapping (many to one).
  std::map<SDNode*, SUnit*> SUnitMap;
  // The schedule.  Null SUnit*'s represent noop instructions.
  std::vector<SUnit*> Sequence;
  // Current scheduling cycle.
  unsigned CurrCycle;
  // First and last SUnit created.
  SUnit *HeadSUnit, *TailSUnit;

  /// isBottomUp - This is true if the scheduling problem is bottom-up, false if
  /// it is top-down.
  bool isBottomUp;
  
  /// HazardRec - The hazard recognizer to use.
  HazardRecognizer *HazardRec;
  
  typedef std::priority_queue<SUnit*, std::vector<SUnit*>, ls_rr_sort>
    AvailableQueueTy;

public:
  ScheduleDAGList(SelectionDAG &dag, MachineBasicBlock *bb,
                  const TargetMachine &tm, bool isbottomup,
                  HazardRecognizer *HR)
    : ScheduleDAG(listSchedulingBURR, dag, bb, tm),
      CurrCycle(0), HeadSUnit(NULL), TailSUnit(NULL), isBottomUp(isbottomup),
      HazardRec(HR) {
    }

  ~ScheduleDAGList() {
    SUnit *SU = HeadSUnit;
    while (SU) {
      SUnit *NextSU = SU->Next;
      delete SU;
      SU = NextSU;
    }
    delete HazardRec;
  }

  void Schedule();

  void dump() const;

private:
  SUnit *NewSUnit(SDNode *N);
  void ReleasePred(AvailableQueueTy &Avail,SUnit *PredSU, bool isChain = false);
  void ReleaseSucc(AvailableQueueTy &Avail,SUnit *SuccSU, bool isChain = false);
  void ScheduleNodeBottomUp(AvailableQueueTy &Avail, SUnit *SU);
  void ScheduleNodeTopDown(AvailableQueueTy &Avail, SUnit *SU);
  int  CalcNodePriority(SUnit *SU);
  void CalculatePriorities();
  void ListScheduleTopDown();
  void ListScheduleBottomUp();
  void BuildSchedUnits();
  void EmitSchedule();
};
}  // end namespace

HazardRecognizer::~HazardRecognizer() {}


/// NewSUnit - Creates a new SUnit and return a ptr to it.
SUnit *ScheduleDAGList::NewSUnit(SDNode *N) {
  SUnit *CurrSUnit = new SUnit(N);

  if (HeadSUnit == NULL)
    HeadSUnit = CurrSUnit;
  if (TailSUnit != NULL)
    TailSUnit->Next = CurrSUnit;
  TailSUnit = CurrSUnit;

  return CurrSUnit;
}

/// ReleasePred - Decrement the NumSuccsLeft count of a predecessor. Add it to
/// the Available queue is the count reaches zero. Also update its cycle bound.
void ScheduleDAGList::ReleasePred(AvailableQueueTy &Available, 
                                  SUnit *PredSU, bool isChain) {
  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  PredSU->CycleBound = std::max(PredSU->CycleBound, CurrCycle + PredSU->Latency);

  if (!isChain)
    PredSU->NumSuccsLeft--;
  else
    PredSU->NumChainSuccsLeft--;
  
#ifndef NDEBUG
  if (PredSU->NumSuccsLeft < 0 || PredSU->NumChainSuccsLeft < 0) {
    std::cerr << "*** List scheduling failed! ***\n";
    PredSU->dump(&DAG);
    std::cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  if ((PredSU->NumSuccsLeft + PredSU->NumChainSuccsLeft) == 0) {
    // EntryToken has to go last!  Special case it here.
    if (PredSU->Node->getOpcode() != ISD::EntryToken)
      Available.push(PredSU);
  }
}

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the Available queue is the count reaches zero. Also update its cycle bound.
void ScheduleDAGList::ReleaseSucc(AvailableQueueTy &Available, 
                                  SUnit *SuccSU, bool isChain) {
  // FIXME: the distance between two nodes is not always == the predecessor's
  // latency. For example, the reader can very well read the register written
  // by the predecessor later than the issue cycle. It also depends on the
  // interrupt model (drain vs. freeze).
  SuccSU->CycleBound = std::max(SuccSU->CycleBound, CurrCycle + SuccSU->Latency);
  
  if (!isChain)
    SuccSU->NumPredsLeft--;
  else
    SuccSU->NumChainPredsLeft--;
  
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft < 0 || SuccSU->NumChainPredsLeft < 0) {
    std::cerr << "*** List scheduling failed! ***\n";
    SuccSU->dump(&DAG);
    std::cerr << " has been released too many times!\n";
    abort();
  }
#endif
  
  if ((SuccSU->NumPredsLeft + SuccSU->NumChainPredsLeft) == 0)
    Available.push(SuccSU);
}

/// ScheduleNodeBottomUp - Add the node to the schedule. Decrement the pending
/// count of its predecessors. If a predecessor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGList::ScheduleNodeBottomUp(AvailableQueueTy &Available,
                                           SUnit *SU) {
  DEBUG(std::cerr << "*** Scheduling: ");
  DEBUG(SU->dump(&DAG, false));

  Sequence.push_back(SU);
  SU->Slot = CurrCycle;

  // Bottom up: release predecessors
  for (std::set<SUnit*>::iterator I1 = SU->Preds.begin(),
         E1 = SU->Preds.end(); I1 != E1; ++I1) {
    ReleasePred(Available, *I1);
    SU->NumPredsLeft--;
  }
  for (std::set<SUnit*>::iterator I2 = SU->ChainPreds.begin(),
         E2 = SU->ChainPreds.end(); I2 != E2; ++I2)
    ReleasePred(Available, *I2, true);

  CurrCycle++;
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void ScheduleDAGList::ScheduleNodeTopDown(AvailableQueueTy &Available,
                                          SUnit *SU) {
  DEBUG(std::cerr << "*** Scheduling: ");
  DEBUG(SU->dump(&DAG, false));
  
  Sequence.push_back(SU);
  SU->Slot = CurrCycle;
  
  // Bottom up: release successors.
  for (std::set<SUnit*>::iterator I1 = SU->Succs.begin(),
       E1 = SU->Succs.end(); I1 != E1; ++I1) {
    ReleaseSucc(Available, *I1);
    SU->NumSuccsLeft--;
  }
  for (std::set<SUnit*>::iterator I2 = SU->ChainSuccs.begin(),
       E2 = SU->ChainSuccs.end(); I2 != E2; ++I2)
    ReleaseSucc(Available, *I2, true);
  
  CurrCycle++;
}

/// isReady - True if node's lower cycle bound is less or equal to the current
/// scheduling cycle. Always true if all nodes have uniform latency 1.
static inline bool isReady(SUnit *SU, unsigned CurrCycle) {
  return SU->CycleBound <= CurrCycle;
}

/// ListScheduleBottomUp - The main loop of list scheduling for bottom-up
/// schedulers.
void ScheduleDAGList::ListScheduleBottomUp() {
  // Available queue.
  AvailableQueueTy Available;

  // Add root to Available queue.
  Available.push(SUnitMap[DAG.getRoot().Val]);

  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back. Schedule the node.
  std::vector<SUnit*> NotReady;
  while (!Available.empty()) {
    SUnit *CurrNode = Available.top();
    Available.pop();

    while (!isReady(CurrNode, CurrCycle)) {
      NotReady.push_back(CurrNode);
      CurrNode = Available.top();
      Available.pop();
    }
    
    // Add the nodes that aren't ready back onto the available list.
    while (!NotReady.empty()) {
      Available.push(NotReady.back());
      NotReady.pop_back();
    }

    ScheduleNodeBottomUp(Available, CurrNode);
  }

  // Add entry node last
  if (DAG.getEntryNode().Val != DAG.getRoot().Val) {
    SUnit *Entry = SUnitMap[DAG.getEntryNode().Val];
    Entry->Slot = CurrCycle;
    Sequence.push_back(Entry);
  }

  // Reverse the order if it is bottom up.
  std::reverse(Sequence.begin(), Sequence.end());
  
  
#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    if (SU->NumSuccsLeft != 0 || SU->NumChainSuccsLeft != 0) {
      if (!AnyNotSched)
        std::cerr << "*** List scheduling failed! ***\n";
      SU->dump(&DAG);
      std::cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void ScheduleDAGList::ListScheduleTopDown() {
  // Available queue.
  AvailableQueueTy Available;

  // Emit the entry node first.
  SUnit *Entry = SUnitMap[DAG.getEntryNode().Val];
  ScheduleNodeTopDown(Available, Entry);
  HazardRec->EmitInstruction(Entry->Node);
                      
  // All leaves to Available queue.
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    // It is available if it has no predecessors.
    if ((SU->Preds.size() + SU->ChainPreds.size()) == 0 && SU != Entry)
      Available.push(SU);
  }
  
  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  while (!Available.empty()) {
    SUnit *FoundNode = 0;

    bool HasNoopHazards = false;
    do {
      SUnit *CurNode = Available.top();
      Available.pop();
      
      // Get the node represented by this SUnit.
      SDNode *N = CurNode->Node;
      // If this is a pseudo op, like copyfromreg, look to see if there is a
      // real target node flagged to it.  If so, use the target node.
      for (unsigned i = 0, e = CurNode->FlaggedNodes.size(); 
           N->getOpcode() < ISD::BUILTIN_OP_END && i != e; ++i)
        N = CurNode->FlaggedNodes[i];
      
      HazardRecognizer::HazardType HT = HazardRec->getHazardType(N);
      if (HT == HazardRecognizer::NoHazard) {
        FoundNode = CurNode;
        break;
      }
      
      // Remember if this is a noop hazard.
      HasNoopHazards |= HT == HazardRecognizer::NoopHazard;
      
      NotReady.push_back(CurNode);
    } while (!Available.empty());
    
    // Add the nodes that aren't ready back onto the available list.
    while (!NotReady.empty()) {
      Available.push(NotReady.back());
      NotReady.pop_back();
    }

    // If we found a node to schedule, do it now.
    if (FoundNode) {
      ScheduleNodeTopDown(Available, FoundNode);
      HazardRec->EmitInstruction(FoundNode->Node);
    } else if (!HasNoopHazards) {
      // Otherwise, we have a pipeline stall, but no other problem, just advance
      // the current cycle and try again.
      DEBUG(std::cerr << "*** Advancing cycle, no work to do\n");
      HazardRec->AdvanceCycle();
      ++NumStalls;
    } else {
      // Otherwise, we have no instructions to issue and we have instructions
      // that will fault if we don't do this right.  This is the case for
      // processors without pipeline interlocks and other cases.
      DEBUG(std::cerr << "*** Emitting noop\n");
      HazardRec->EmitNoop();
      Sequence.push_back(0);   // NULL SUnit* -> noop
      ++NumNoops;
    }
  }

#ifndef NDEBUG
  // Verify that all SUnits were scheduled.
  bool AnyNotSched = false;
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    if (SU->NumPredsLeft != 0 || SU->NumChainPredsLeft != 0) {
      if (!AnyNotSched)
        std::cerr << "*** List scheduling failed! ***\n";
      SU->dump(&DAG);
      std::cerr << "has not been scheduled!\n";
      AnyNotSched = true;
    }
  }
  assert(!AnyNotSched);
#endif
}


/// CalcNodePriority - Priority is the Sethi Ullman number. 
/// Smaller number is the higher priority.
int ScheduleDAGList::CalcNodePriority(SUnit *SU) {
  if (SU->SethiUllman != INT_MIN)
    return SU->SethiUllman;

  if (SU->Preds.size() == 0) {
    SU->SethiUllman = 1;
  } else {
    int Extra = 0;
    for (std::set<SUnit*>::iterator I = SU->Preds.begin(),
           E = SU->Preds.end(); I != E; ++I) {
      SUnit *PredSU = *I;
      int PredSethiUllman = CalcNodePriority(PredSU);
      if (PredSethiUllman > SU->SethiUllman) {
        SU->SethiUllman = PredSethiUllman;
        Extra = 0;
      } else if (PredSethiUllman == SU->SethiUllman)
        Extra++;
    }

    if (SU->Node->getOpcode() != ISD::TokenFactor)
      SU->SethiUllman += Extra;
    else
      SU->SethiUllman = (Extra == 1) ? 0 : Extra-1;
  }

  return SU->SethiUllman;
}

/// CalculatePriorities - Calculate priorities of all scheduling units.
void ScheduleDAGList::CalculatePriorities() {
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    // FIXME: assumes uniform latency for now.
    SU->Latency = 1;
    (void)CalcNodePriority(SU);
    DEBUG(SU->dump(&DAG));
    DEBUG(std::cerr << "\n");
  }
}

void ScheduleDAGList::BuildSchedUnits() {
  // Pass 1: create the SUnit's.
  for (unsigned i = 0, NC = NodeCount; i < NC; i++) {
    NodeInfo *NI = &Info[i];
    SDNode *N = NI->Node;
    if (isPassiveNode(N))
      continue;

    SUnit *SU;
    if (NI->isInGroup()) {
      if (NI != NI->Group->getBottom())  // Bottom up, so only look at bottom
        continue;                        // node of the NodeGroup

      SU = NewSUnit(N);
      // Find the flagged nodes.
      SDOperand  FlagOp = N->getOperand(N->getNumOperands() - 1);
      SDNode    *Flag   = FlagOp.Val;
      unsigned   ResNo  = FlagOp.ResNo;
      while (Flag->getValueType(ResNo) == MVT::Flag) {
        NodeInfo *FNI = getNI(Flag);
        assert(FNI->Group == NI->Group);
        SU->FlaggedNodes.insert(SU->FlaggedNodes.begin(), Flag);
        SUnitMap[Flag] = SU;

        FlagOp = Flag->getOperand(Flag->getNumOperands() - 1);
        Flag   = FlagOp.Val;
        ResNo  = FlagOp.ResNo;
      }
    } else {
      SU = NewSUnit(N);
    }
    SUnitMap[N] = SU;
  }

  // Pass 2: add the preds, succs, etc.
  for (SUnit *SU = HeadSUnit; SU != NULL; SU = SU->Next) {
    SDNode   *N  = SU->Node;
    NodeInfo *NI = getNI(N);
    
    if (N->isTargetOpcode() && TII->isTwoAddrInstr(N->getTargetOpcode()))
      SU->isTwoAddress = true;

    if (NI->isInGroup()) {
      // Find all predecessors (of the group).
      NodeGroupOpIterator NGOI(NI);
      while (!NGOI.isEnd()) {
        SDOperand Op  = NGOI.next();
        SDNode   *OpN = Op.Val;
        MVT::ValueType VT = OpN->getValueType(Op.ResNo);
        NodeInfo *OpNI = getNI(OpN);
        if (OpNI->Group != NI->Group && !isPassiveNode(OpN)) {
          assert(VT != MVT::Flag);
          SUnit *OpSU = SUnitMap[OpN];
          if (VT == MVT::Other) {
            if (SU->ChainPreds.insert(OpSU).second)
              SU->NumChainPredsLeft++;
            if (OpSU->ChainSuccs.insert(SU).second)
              OpSU->NumChainSuccsLeft++;
          } else {
            if (SU->Preds.insert(OpSU).second)
              SU->NumPredsLeft++;
            if (OpSU->Succs.insert(SU).second)
              OpSU->NumSuccsLeft++;
          }
        }
      }
    } else {
      // Find node predecessors.
      for (unsigned j = 0, e = N->getNumOperands(); j != e; j++) {
        SDOperand Op  = N->getOperand(j);
        SDNode   *OpN = Op.Val;
        MVT::ValueType VT = OpN->getValueType(Op.ResNo);
        if (!isPassiveNode(OpN)) {
          assert(VT != MVT::Flag);
          SUnit *OpSU = SUnitMap[OpN];
          if (VT == MVT::Other) {
            if (SU->ChainPreds.insert(OpSU).second)
              SU->NumChainPredsLeft++;
            if (OpSU->ChainSuccs.insert(SU).second)
              OpSU->NumChainSuccsLeft++;
          } else {
            if (SU->Preds.insert(OpSU).second)
              SU->NumPredsLeft++;
            if (OpSU->Succs.insert(SU).second)
              OpSU->NumSuccsLeft++;
            if (j == 0 && SU->isTwoAddress) 
              OpSU->isDefNUseOperand = true;
          }
        }
      }
    }
  }
}

/// EmitSchedule - Emit the machine code in scheduled order.
void ScheduleDAGList::EmitSchedule() {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i]) {
      for (unsigned j = 0, ee = SU->FlaggedNodes.size(); j != ee; j++) {
        SDNode *N = SU->FlaggedNodes[j];
        EmitNode(getNI(N));
      }
      EmitNode(getNI(SU->Node));
    } else {
      // Null SUnit* is a noop.
      EmitNoop();
    }
  }
}

/// dump - dump the schedule.
void ScheduleDAGList::dump() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      SU->dump(&DAG, false);
    else
      std::cerr << "**** NOOP ****\n";
  }
}

/// Schedule - Schedule the DAG using list scheduling.
/// FIXME: Right now it only supports the burr (bottom up register reducing)
/// heuristic.
void ScheduleDAGList::Schedule() {
  DEBUG(std::cerr << "********** List Scheduling **********\n");

  // Build scheduling units.
  BuildSchedUnits();

  // Calculate node priorities.
  CalculatePriorities();

  // Execute the actual scheduling loop Top-Down or Bottom-Up as appropriate.
  if (isBottomUp)
    ListScheduleBottomUp();
  else
    ListScheduleTopDown();
  
  DEBUG(std::cerr << "*** Final schedule ***\n");
  DEBUG(dump());
  DEBUG(std::cerr << "\n");
  
  // Emit in scheduled order
  EmitSchedule();
}

llvm::ScheduleDAG* llvm::createBURRListDAGScheduler(SelectionDAG &DAG,
                                                    MachineBasicBlock *BB) {
  return new ScheduleDAGList(DAG, BB, DAG.getTarget(), true, 
                             new HazardRecognizer());
}

/// createTDListDAGScheduler - This creates a top-down list scheduler with the
/// specified hazard recognizer.
ScheduleDAG* llvm::createTDListDAGScheduler(SelectionDAG &DAG,
                                            MachineBasicBlock *BB,
                                            HazardRecognizer *HR) {
  return new ScheduleDAGList(DAG, BB, DAG.getTarget(), false, HR);
}

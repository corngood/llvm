//===------- llvm/CodeGen/ScheduleDAG.h - Common Base Class------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAG class, which is used as the common
// base class for SelectionDAG-based instruction scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAG_H
#define LLVM_CODEGEN_SCHEDULEDAG_H

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallSet.h"

namespace llvm {
  struct InstrStage;
  struct SUnit;
  class MachineConstantPool;
  class MachineModuleInfo;
  class MachineInstr;
  class MRegisterInfo;
  class SelectionDAG;
  class SelectionDAGISel;
  class SSARegMap;
  class TargetInstrInfo;
  class TargetInstrDescriptor;
  class TargetMachine;

  /// HazardRecognizer - This determines whether or not an instruction can be
  /// issued this cycle, and whether or not a noop needs to be inserted to handle
  /// the hazard.
  class HazardRecognizer {
  public:
    virtual ~HazardRecognizer();
    
    enum HazardType {
      NoHazard,      // This instruction can be emitted at this cycle.
      Hazard,        // This instruction can't be emitted at this cycle.
      NoopHazard     // This instruction can't be emitted, and needs noops.
    };
    
    /// getHazardType - Return the hazard type of emitting this node.  There are
    /// three possible results.  Either:
    ///  * NoHazard: it is legal to issue this instruction on this cycle.
    ///  * Hazard: issuing this instruction would stall the machine.  If some
    ///     other instruction is available, issue it first.
    ///  * NoopHazard: issuing this instruction would break the program.  If
    ///     some other instruction can be issued, do so, otherwise issue a noop.
    virtual HazardType getHazardType(SDNode *Node) {
      return NoHazard;
    }
    
    /// EmitInstruction - This callback is invoked when an instruction is
    /// emitted, to advance the hazard state.
    virtual void EmitInstruction(SDNode *Node) {
    }
    
    /// AdvanceCycle - This callback is invoked when no instructions can be
    /// issued on this cycle without a hazard.  This should increment the
    /// internal state of the hazard recognizer so that previously "Hazard"
    /// instructions will now not be hazards.
    virtual void AdvanceCycle() {
    }
    
    /// EmitNoop - This callback is invoked when a noop was added to the
    /// instruction stream.
    virtual void EmitNoop() {
    }
  };

  /// SDep - Scheduling dependency. It keeps track of dependent nodes,
  /// cost of the depdenency, etc.
  struct SDep {
    SUnit    *Dep;           // Dependent - either a predecessor or a successor.
    unsigned  Reg;           // If non-zero, this dep is a phy register dependency.
    int       Cost;          // Cost of the dependency.
    bool      isCtrl    : 1; // True iff it's a control dependency.
    bool      isSpecial : 1; // True iff it's a special ctrl dep added during sched.
    SDep(SUnit *d, unsigned r, int t, bool c, bool s)
      : Dep(d), Reg(r), Cost(t), isCtrl(c), isSpecial(s) {}
  };

  /// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
  /// a group of nodes flagged together.
  struct SUnit {
    SDNode *Node;                       // Representative node.
    SmallVector<SDNode*,4> FlaggedNodes;// All nodes flagged to Node.
    unsigned InstanceNo;                // Instance#. One SDNode can be multiple
                                        // SUnit due to cloning.
    
    // Preds/Succs - The SUnits before/after us in the graph.  The boolean value
    // is true if the edge is a token chain edge, false if it is a value edge. 
    SmallVector<SDep, 4> Preds;  // All sunit predecessors.
    SmallVector<SDep, 4> Succs;  // All sunit successors.

    typedef SmallVector<SDep, 4>::iterator pred_iterator;
    typedef SmallVector<SDep, 4>::iterator succ_iterator;
    typedef SmallVector<SDep, 4>::const_iterator const_pred_iterator;
    typedef SmallVector<SDep, 4>::const_iterator const_succ_iterator;
    
    unsigned NodeNum;                   // Entry # of node in the node vector.
    unsigned short Latency;             // Node latency.
    short NumPreds;                     // # of preds.
    short NumSuccs;                     // # of sucss.
    short NumPredsLeft;                 // # of preds not scheduled.
    short NumSuccsLeft;                 // # of succs not scheduled.
    short NumChainPredsLeft;            // # of chain preds not scheduled.
    short NumChainSuccsLeft;            // # of chain succs not scheduled.
    bool isTwoAddress     : 1;          // Is a two-address instruction.
    bool isCommutable     : 1;          // Is a commutable instruction.
    bool hasImplicitDefs  : 1;          // Has implicit physical reg defs.
    bool isPending        : 1;          // True once pending.
    bool isAvailable      : 1;          // True once available.
    bool isScheduled      : 1;          // True once scheduled.
    unsigned CycleBound;                // Upper/lower cycle to be scheduled at.
    unsigned Cycle;                     // Once scheduled, the cycle of the op.
    unsigned Depth;                     // Node depth;
    unsigned Height;                    // Node height;
    
    SUnit(SDNode *node, unsigned nodenum)
      : Node(node), InstanceNo(0), NodeNum(nodenum), Latency(0),
        NumPreds(0), NumSuccs(0), NumPredsLeft(0), NumSuccsLeft(0),
        NumChainPredsLeft(0), NumChainSuccsLeft(0),
        isTwoAddress(false), isCommutable(false), hasImplicitDefs(false),
        isPending(false), isAvailable(false), isScheduled(false),
        CycleBound(0), Cycle(0), Depth(0), Height(0) {}

    /// addPred - This adds the specified node as a pred of the current node if
    /// not already.  This returns true if this is a new pred.
    bool addPred(SUnit *N, bool isCtrl, bool isSpecial,
                 unsigned PhyReg = 0, int Cost = 1) {
      for (unsigned i = 0, e = Preds.size(); i != e; ++i)
        if (Preds[i].Dep == N &&
            Preds[i].isCtrl == isCtrl && Preds[i].isSpecial == isSpecial)
          return false;
      Preds.push_back(SDep(N, PhyReg, Cost, isCtrl, isSpecial));
      N->Succs.push_back(SDep(this, PhyReg, Cost, isCtrl, isSpecial));
      if (isCtrl) {
        if (!N->isScheduled)
          ++NumChainPredsLeft;
        if (!isScheduled)
          ++N->NumChainSuccsLeft;
      } else {
        ++NumPreds;
        ++N->NumSuccs;
        if (!N->isScheduled)
          ++NumPredsLeft;
        if (!isScheduled)
          ++N->NumSuccsLeft;
      }
      return true;
    }

    bool removePred(SUnit *N, bool isCtrl, bool isSpecial) {
      for (SmallVector<SDep, 4>::iterator I = Preds.begin(), E = Preds.end();
           I != E; ++I)
        if (I->Dep == N && I->isCtrl == isCtrl && I->isSpecial == isSpecial) {
          bool FoundSucc = false;
          for (SmallVector<SDep, 4>::iterator II = N->Succs.begin(),
                 EE = N->Succs.end(); II != EE; ++II)
            if (II->Dep == this &&
                II->isCtrl == isCtrl && II->isSpecial == isSpecial) {
              FoundSucc = true;
              N->Succs.erase(II);
              break;
            }
          assert(FoundSucc && "Mismatching preds / succs lists!");
          Preds.erase(I);
          if (isCtrl) {
            if (!N->isScheduled)
              --NumChainPredsLeft;
            if (!isScheduled)
              --NumChainSuccsLeft;
          } else {
            --NumPreds;
            --N->NumSuccs;
            if (!N->isScheduled)
              --NumPredsLeft;
            if (!isScheduled)
              --N->NumSuccsLeft;
          }
          return true;
        }
      return false;
    }

    bool isPred(SUnit *N) {
      for (unsigned i = 0, e = Preds.size(); i != e; ++i)
        if (Preds[i].Dep == N)
          return true;
      return false;
    }
    
    bool isSucc(SUnit *N) {
      for (unsigned i = 0, e = Succs.size(); i != e; ++i)
        if (Succs[i].Dep == N)
          return true;
      return false;
    }
    
    void dump(const SelectionDAG *G) const;
    void dumpAll(const SelectionDAG *G) const;
  };

  //===--------------------------------------------------------------------===//
  /// SchedulingPriorityQueue - This interface is used to plug different
  /// priorities computation algorithms into the list scheduler. It implements
  /// the interface of a standard priority queue, where nodes are inserted in 
  /// arbitrary order and returned in priority order.  The computation of the
  /// priority and the representation of the queue are totally up to the
  /// implementation to decide.
  /// 
  class SchedulingPriorityQueue {
  public:
    virtual ~SchedulingPriorityQueue() {}
  
    virtual void initNodes(DenseMap<SDNode*, std::vector<SUnit*> > &SUMap,
                           std::vector<SUnit> &SUnits) = 0;
    virtual void addNode(const SUnit *SU) = 0;
    virtual void updateNode(const SUnit *SU) = 0;
    virtual void releaseState() = 0;

    virtual unsigned size() const = 0;
    virtual bool empty() const = 0;
    virtual void push(SUnit *U) = 0;
  
    virtual void push_all(const std::vector<SUnit *> &Nodes) = 0;
    virtual SUnit *pop() = 0;

    virtual void remove(SUnit *SU) = 0;

    /// ScheduledNode - As each node is scheduled, this method is invoked.  This
    /// allows the priority function to adjust the priority of node that have
    /// already been emitted.
    virtual void ScheduledNode(SUnit *Node) {}

    virtual void UnscheduledNode(SUnit *Node) {}
  };

  class ScheduleDAG {
  public:
    SelectionDAG &DAG;                    // DAG of the current basic block
    MachineBasicBlock *BB;                // Current basic block
    const TargetMachine &TM;              // Target processor
    const TargetInstrInfo *TII;           // Target instruction information
    const MRegisterInfo *MRI;             // Target processor register info
    SSARegMap *RegMap;                    // Virtual/real register map
    MachineConstantPool *ConstPool;       // Target constant pool
    std::vector<SUnit*> Sequence;         // The schedule. Null SUnit*'s
                                          // represent noop instructions.
    DenseMap<SDNode*, std::vector<SUnit*> > SUnitMap;
                                          // SDNode to SUnit mapping (n -> n).
    std::vector<SUnit> SUnits;            // The scheduling units.
    SmallSet<SDNode*, 16> CommuteSet;     // Nodes the should be commuted.

    ScheduleDAG(SelectionDAG &dag, MachineBasicBlock *bb,
                const TargetMachine &tm)
      : DAG(dag), BB(bb), TM(tm) {}

    virtual ~ScheduleDAG() {}

    /// viewGraph - Pop up a GraphViz/gv window with the ScheduleDAG rendered
    /// using 'dot'.
    ///
    void viewGraph();
  
    /// Run - perform scheduling.
    ///
    MachineBasicBlock *Run();

    /// isPassiveNode - Return true if the node is a non-scheduled leaf.
    ///
    static bool isPassiveNode(SDNode *Node) {
      if (isa<ConstantSDNode>(Node))       return true;
      if (isa<RegisterSDNode>(Node))       return true;
      if (isa<GlobalAddressSDNode>(Node))  return true;
      if (isa<BasicBlockSDNode>(Node))     return true;
      if (isa<FrameIndexSDNode>(Node))     return true;
      if (isa<ConstantPoolSDNode>(Node))   return true;
      if (isa<JumpTableSDNode>(Node))      return true;
      if (isa<ExternalSymbolSDNode>(Node)) return true;
      return false;
    }

    /// NewSUnit - Creates a new SUnit and return a ptr to it.
    ///
    SUnit *NewSUnit(SDNode *N) {
      SUnits.push_back(SUnit(N, SUnits.size()));
      return &SUnits.back();
    }

    /// Clone - Creates a clone of the specified SUnit. It does not copy the
    /// predecessors / successors info nor the temporary scheduling states.
    SUnit *Clone(SUnit *N);
    
    /// BuildSchedUnits - Build SUnits from the selection dag that we are input.
    /// This SUnit graph is similar to the SelectionDAG, but represents flagged
    /// together nodes with a single SUnit.
    void BuildSchedUnits();

    /// CalculateDepths, CalculateHeights - Calculate node depth / height.
    ///
    void CalculateDepths();
    void CalculateHeights();

    /// CountResults - The results of target nodes have register or immediate
    /// operands first, then an optional chain, and optional flag operands
    /// (which do not go into the machine instrs.)
    static unsigned CountResults(SDNode *Node);

    /// CountOperands  The inputs to target nodes have any actual inputs first,
    /// followed by an optional chain operand, then flag operands.  Compute the
    /// number of actual operands that  will go into the machine instr.
    static unsigned CountOperands(SDNode *Node);

    /// EmitNode - Generate machine code for an node and needed dependencies.
    /// VRBaseMap contains, for each already emitted node, the first virtual
    /// register number for the results of the node.
    ///
    void EmitNode(SDNode *Node, unsigned InstNo,
                  DenseMap<SDOperand, unsigned> &VRBaseMap);
    
    /// EmitNoop - Emit a noop instruction.
    ///
    void EmitNoop();

    /// EmitCopyFromReg - Generate machine code for an CopyFromReg node or an
    /// implicit physical register output.
    void EmitCopyFromReg(SDNode *Node, unsigned ResNo, unsigned InstNo,
                         unsigned SrcReg,
                         DenseMap<SDOperand, unsigned> &VRBaseMap);
    
    void CreateVirtualRegisters(SDNode *Node, MachineInstr *MI,
                                const TargetInstrDescriptor &II,
                                DenseMap<SDOperand, unsigned> &VRBaseMap);

    void EmitSchedule();

    void dumpSchedule() const;

    /// Schedule - Order nodes according to selected style.
    ///
    virtual void Schedule() {}

  private:
    /// EmitSubregNode - Generate machine code for subreg nodes.
    ///
    void EmitSubregNode(SDNode *Node, 
                        DenseMap<SDOperand, unsigned> &VRBaseMap);
  
    void AddOperand(MachineInstr *MI, SDOperand Op, unsigned IIOpNum,
                    const TargetInstrDescriptor *II,
                    DenseMap<SDOperand, unsigned> &VRBaseMap);
  };

  /// createBFS_DAGScheduler - This creates a simple breadth first instruction
  /// scheduler.
  ScheduleDAG *createBFS_DAGScheduler(SelectionDAGISel *IS,
                                      SelectionDAG *DAG,
                                      MachineBasicBlock *BB);
  
  /// createSimpleDAGScheduler - This creates a simple two pass instruction
  /// scheduler using instruction itinerary.
  ScheduleDAG* createSimpleDAGScheduler(SelectionDAGISel *IS,
                                        SelectionDAG *DAG,
                                        MachineBasicBlock *BB);

  /// createNoItinsDAGScheduler - This creates a simple two pass instruction
  /// scheduler without using instruction itinerary.
  ScheduleDAG* createNoItinsDAGScheduler(SelectionDAGISel *IS,
                                         SelectionDAG *DAG,
                                         MachineBasicBlock *BB);

  /// createBURRListDAGScheduler - This creates a bottom up register usage
  /// reduction list scheduler.
  ScheduleDAG* createBURRListDAGScheduler(SelectionDAGISel *IS,
                                          SelectionDAG *DAG,
                                          MachineBasicBlock *BB);
  
  /// createTDRRListDAGScheduler - This creates a top down register usage
  /// reduction list scheduler.
  ScheduleDAG* createTDRRListDAGScheduler(SelectionDAGISel *IS,
                                          SelectionDAG *DAG,
                                          MachineBasicBlock *BB);
  
  /// createTDListDAGScheduler - This creates a top-down list scheduler with
  /// a hazard recognizer.
  ScheduleDAG* createTDListDAGScheduler(SelectionDAGISel *IS,
                                        SelectionDAG *DAG,
                                        MachineBasicBlock *BB);
                                        
  /// createDefaultScheduler - This creates an instruction scheduler appropriate
  /// for the target.
  ScheduleDAG* createDefaultScheduler(SelectionDAGISel *IS,
                                      SelectionDAG *DAG,
                                      MachineBasicBlock *BB);

  class SUnitIterator : public forward_iterator<SUnit, ptrdiff_t> {
    SUnit *Node;
    unsigned Operand;

    SUnitIterator(SUnit *N, unsigned Op) : Node(N), Operand(Op) {}
  public:
    bool operator==(const SUnitIterator& x) const {
      return Operand == x.Operand;
    }
    bool operator!=(const SUnitIterator& x) const { return !operator==(x); }

    const SUnitIterator &operator=(const SUnitIterator &I) {
      assert(I.Node == Node && "Cannot assign iterators to two different nodes!");
      Operand = I.Operand;
      return *this;
    }

    pointer operator*() const {
      return Node->Preds[Operand].Dep;
    }
    pointer operator->() const { return operator*(); }

    SUnitIterator& operator++() {                // Preincrement
      ++Operand;
      return *this;
    }
    SUnitIterator operator++(int) { // Postincrement
      SUnitIterator tmp = *this; ++*this; return tmp;
    }

    static SUnitIterator begin(SUnit *N) { return SUnitIterator(N, 0); }
    static SUnitIterator end  (SUnit *N) {
      return SUnitIterator(N, N->Preds.size());
    }

    unsigned getOperand() const { return Operand; }
    const SUnit *getNode() const { return Node; }
    bool isCtrlDep() const { return Node->Preds[Operand].isCtrl; }
  };

  template <> struct GraphTraits<SUnit*> {
    typedef SUnit NodeType;
    typedef SUnitIterator ChildIteratorType;
    static inline NodeType *getEntryNode(SUnit *N) { return N; }
    static inline ChildIteratorType child_begin(NodeType *N) {
      return SUnitIterator::begin(N);
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return SUnitIterator::end(N);
    }
  };

  template <> struct GraphTraits<ScheduleDAG*> : public GraphTraits<SUnit*> {
    typedef std::vector<SUnit>::iterator nodes_iterator;
    static nodes_iterator nodes_begin(ScheduleDAG *G) {
      return G->SUnits.begin();
    }
    static nodes_iterator nodes_end(ScheduleDAG *G) {
      return G->SUnits.end();
    }
  };
}

#endif

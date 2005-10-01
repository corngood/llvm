//===-- ScheduleDAG.cpp - Implement a trivial DAG scheduler ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a simple two pass scheduler.  The first pass attempts to push
// backward any lengthy instructions and critical paths.  The second pass packs
// instructions into semi-optimal time slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;

namespace {
  // Style of scheduling to use.
  enum ScheduleChoices {
    noScheduling,
    simpleScheduling,
  };
} // namespace

cl::opt<ScheduleChoices> ScheduleStyle("sched",
  cl::desc("Choose scheduling style"),
  cl::init(noScheduling),
  cl::values(
    clEnumValN(noScheduling, "none",
              "Trivial emission with no analysis"),
    clEnumValN(simpleScheduling, "simple",
              "Minimize critical path and maximize processor utilization"),
   clEnumValEnd));


#ifndef NDEBUG
static cl::opt<bool>
ViewDAGs("view-sched-dags", cl::Hidden,
         cl::desc("Pop up a window to show sched dags as they are processed"));
#else
static const bool ViewDAGs = 0;
#endif

namespace {
//===----------------------------------------------------------------------===//
///
/// BitsIterator - Provides iteration through individual bits in a bit vector.
///
template<class T>
class BitsIterator {
private:
  T Bits;                               // Bits left to iterate through

public:
  /// Ctor.
  BitsIterator(T Initial) : Bits(Initial) {}
  
  /// Next - Returns the next bit set or zero if exhausted.
  inline T Next() {
    // Get the rightmost bit set
    T Result = Bits & -Bits;
    // Remove from rest
    Bits &= ~Result;
    // Return single bit or zero
    return Result;
  }
};
  
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// ResourceTally - Manages the use of resources over time intervals.  Each
/// item (slot) in the tally vector represents the resources used at a given
/// moment.  A bit set to 1 indicates that a resource is in use, otherwise
/// available.  An assumption is made that the tally is large enough to schedule 
/// all current instructions (asserts otherwise.)
///
template<class T>
class ResourceTally {
private:
  std::vector<T> Tally;                 // Resources used per slot
  typedef typename std::vector<T>::iterator Iter;
                                        // Tally iterator 
  
  /// AllInUse - Test to see if all of the resources in the slot are busy (set.)
  inline bool AllInUse(Iter Cursor, unsigned ResourceSet) {
    return (*Cursor & ResourceSet) == ResourceSet;
  }

  /// Skip - Skip over slots that use all of the specified resource (all are
  /// set.)
  Iter Skip(Iter Cursor, unsigned ResourceSet) {
    assert(ResourceSet && "At least one resource bit needs to bet set");
    
    // Continue to the end
    while (true) {
      // Break out if one of the resource bits is not set
      if (!AllInUse(Cursor, ResourceSet)) return Cursor;
      // Try next slot
      Cursor++;
      assert(Cursor < Tally.end() && "Tally is not large enough for schedule");
    }
  }
  
  /// FindSlots - Starting from Begin, locate N consecutive slots where at least 
  /// one of the resource bits is available.  Returns the address of first slot.
  Iter FindSlots(Iter Begin, unsigned N, unsigned ResourceSet,
                                         unsigned &Resource) {
    // Track position      
    Iter Cursor = Begin;
    
    // Try all possible slots forward
    while (true) {
      // Skip full slots
      Cursor = Skip(Cursor, ResourceSet);
      // Determine end of interval
      Iter End = Cursor + N;
      assert(End <= Tally.end() && "Tally is not large enough for schedule");
      
      // Iterate thru each resource
      BitsIterator<T> Resources(ResourceSet & ~*Cursor);
      while (unsigned Res = Resources.Next()) {
        // Check if resource is available for next N slots
        // Break out if resource is busy
        Iter Interval = Cursor;
        for (; Interval < End && !(*Interval & Res); Interval++) {}
        
        // If available for interval, return where and which resource
        if (Interval == End) {
          Resource = Res;
          return Cursor;
        }
        // Otherwise, check if worth checking other resources
        if (AllInUse(Interval, ResourceSet)) {
          // Start looking beyond interval
          Cursor = Interval;
          break;
        }
      }
      Cursor++;
    }
  }
  
  /// Reserve - Mark busy (set) the specified N slots.
  void Reserve(Iter Begin, unsigned N, unsigned Resource) {
    // Determine end of interval
    Iter End = Begin + N;
    assert(End <= Tally.end() && "Tally is not large enough for schedule");
 
    // Set resource bit in each slot
    for (; Begin < End; Begin++)
      *Begin |= Resource;
  }

public:
  /// Initialize - Resize and zero the tally to the specified number of time
  /// slots.
  inline void Initialize(unsigned N) {
    Tally.assign(N, 0);   // Initialize tally to all zeros.
  }
  
  // FindAndReserve - Locate and mark busy (set) N bits started at slot I, using
  // ResourceSet for choices.
  unsigned FindAndReserve(unsigned I, unsigned N, unsigned ResourceSet) {
    // Which resource used
    unsigned Resource;
    // Find slots for instruction.
    Iter Where = FindSlots(Tally.begin() + I, N, ResourceSet, Resource);
    // Reserve the slots
    Reserve(Where, N, Resource);
    // Return time slot (index)
    return Where - Tally.begin();
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// Node group -  This struct is used to manage flagged node groups.
///
class NodeInfo;
class NodeGroup : public std::vector<NodeInfo *> {
private:
  int           Pending;                // Number of visits pending before
                                        //    adding to order  

public:
  // Ctor.
  NodeGroup() : Pending(0) {}
  
  // Accessors
  inline NodeInfo *getLeader() { return empty() ? NULL : front(); }
  inline int getPending() const { return Pending; }
  inline void setPending(int P)  { Pending = P; }
  inline int addPending(int I)  { return Pending += I; }

  static void Add(NodeInfo *D, NodeInfo *U);
  static unsigned CountInternalUses(NodeInfo *D, NodeInfo *U);
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// NodeInfo - This struct tracks information used to schedule the a node.
///
class NodeInfo {
private:
  int           Pending;                // Number of visits pending before
                                        //    adding to order
public:
  SDNode        *Node;                  // DAG node
  unsigned      Latency;                // Cycles to complete instruction
  unsigned      ResourceSet;            // Bit vector of usable resources
  unsigned      Slot;                   // Node's time slot
  NodeGroup     *Group;                 // Grouping information
  unsigned      VRBase;                 // Virtual register base
  
  // Ctor.
  NodeInfo(SDNode *N = NULL)
  : Pending(0)
  , Node(N)
  , Latency(0)
  , ResourceSet(0)
  , Slot(0)
  , Group(NULL)
  , VRBase(0)
  {}
  
  // Accessors
  inline bool isInGroup() const {
    assert(!Group || !Group->empty() && "Group with no members");
    return Group != NULL;
  }
  inline bool isGroupLeader() const {
     return isInGroup() && Group->getLeader() == this;
  }
  inline int getPending() const {
    return Group ? Group->getPending() : Pending;
  }
  inline void setPending(int P) {
    if (Group) Group->setPending(P);
    else       Pending = P;
  }
  inline int addPending(int I) {
    if (Group) return Group->addPending(I);
    else       return Pending += I;
  }
};
typedef std::vector<NodeInfo *>::iterator NIIterator;
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// NodeGroupIterator - Iterates over all the nodes indicated by the node info.
/// If the node is in a group then iterate over the members of the group,
/// otherwise just the node info.
///
class NodeGroupIterator {
private:
  NodeInfo   *NI;                       // Node info
  NIIterator NGI;                       // Node group iterator
  NIIterator NGE;                       // Node group iterator end
  
public:
  // Ctor.
  NodeGroupIterator(NodeInfo *N) : NI(N) {
    // If the node is in a group then set up the group iterator.  Otherwise
    // the group iterators will trip first time out.
    if (N->isInGroup()) {
      // get Group
      NodeGroup *Group = NI->Group;
      NGI = Group->begin();
      NGE = Group->end();
      // Prevent this node from being used (will be in members list
      NI = NULL;
    }
  }
  
  /// next - Return the next node info, otherwise NULL.
  ///
  NodeInfo *next() {
    // If members list
    if (NGI != NGE) return *NGI++;
    // Use node as the result (may be NULL)
    NodeInfo *Result = NI;
    // Only use once
    NI = NULL;
    // Return node or NULL
    return Result;
  }
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// NodeGroupOpIterator - Iterates over all the operands of a node.  If the node
/// is a member of a group, this iterates over all the operands of all the
/// members of the group.
///
class NodeGroupOpIterator {
private:
  NodeInfo            *NI;              // Node containing operands
  NodeGroupIterator   GI;               // Node group iterator
  SDNode::op_iterator OI;               // Operand iterator
  SDNode::op_iterator OE;               // Operand iterator end
  
  /// CheckNode - Test if node has more operands.  If not get the next node
  /// skipping over nodes that have no operands.
  void CheckNode() {
    // Only if operands are exhausted first
    while (OI == OE) {
      // Get next node info
      NodeInfo *NI = GI.next();
      // Exit if nodes are exhausted
      if (!NI) return;
      // Get node itself
      SDNode *Node = NI->Node;
      // Set up the operand iterators
      OI = Node->op_begin();
      OE = Node->op_end();
    }
  }
  
public:
  // Ctor.
  NodeGroupOpIterator(NodeInfo *N) : NI(N), GI(N) {}
  
  /// isEnd - Returns true when not more operands are available.
  ///
  inline bool isEnd() { CheckNode(); return OI == OE; }
  
  /// next - Returns the next available operand.
  ///
  inline SDOperand next() {
    assert(OI != OE && "Not checking for end of NodeGroupOpIterator correctly");
    return *OI++;
  }
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
///
/// SimpleSched - Simple two pass scheduler.
///
class SimpleSched {
private:
  // TODO - get ResourceSet from TII
  enum {
    RSInteger = 0x3,                    // Two integer units
    RSFloat = 0xC,                      // Two float units
    RSLoadStore = 0x30,                 // Two load store units
    RSOther = 0                         // Processing unit independent
  };
  
  MachineBasicBlock *BB;                // Current basic block
  SelectionDAG &DAG;                    // DAG of the current basic block
  const TargetMachine &TM;              // Target processor
  const TargetInstrInfo &TII;           // Target instruction information
  const MRegisterInfo &MRI;             // Target processor register information
  SSARegMap *RegMap;                    // Virtual/real register map
  MachineConstantPool *ConstPool;       // Target constant pool
  unsigned NodeCount;                   // Number of nodes in DAG
  NodeInfo *Info;                       // Info for nodes being scheduled
  std::map<SDNode *, NodeInfo *> Map;   // Map nodes to info
  std::vector<NodeInfo*> Ordering;      // Emit ordering of nodes
  ResourceTally<unsigned> Tally;        // Resource usage tally
  unsigned NSlots;                      // Total latency
  std::map<SDNode *, unsigned> VRMap;   // Node to VR map
  static const unsigned NotFound = ~0U; // Search marker
  
public:

  // Ctor.
  SimpleSched(SelectionDAG &D, MachineBasicBlock *bb)
    : BB(bb), DAG(D), TM(D.getTarget()), TII(*TM.getInstrInfo()),
      MRI(*TM.getRegisterInfo()), RegMap(BB->getParent()->getSSARegMap()),
      ConstPool(BB->getParent()->getConstantPool()),
      NSlots(0) {
    assert(&TII && "Target doesn't provide instr info?");
    assert(&MRI && "Target doesn't provide register info?");
  }
  
  // Run - perform scheduling.
  MachineBasicBlock *Run() {
    Schedule();
    return BB;
  }
  
private:
  /// getNI - Returns the node info for the specified node.
  ///
  inline NodeInfo *getNI(SDNode *Node) { return Map[Node]; }
  
  /// getVR - Returns the virtual register number of the node.
  ///
  inline unsigned getVR(SDOperand Op) {
    NodeInfo *NI = getNI(Op.Val);
    assert(NI->VRBase != 0 && "Node emitted out of order - late");
    return NI->VRBase + Op.ResNo;
  }

  static bool isFlagDefiner(SDNode *A);
  static bool isFlagUser(SDNode *A);
  static bool isDefiner(NodeInfo *A, NodeInfo *B);
  static bool isPassiveNode(SDNode *Node);
  void IncludeNode(NodeInfo *NI);
  void VisitAll();
  void Schedule();
  void GatherNodeInfo();
  bool isStrongDependency(NodeInfo *A, NodeInfo *B);
  bool isWeakDependency(NodeInfo *A, NodeInfo *B);
  void ScheduleBackward();
  void ScheduleForward();
  void EmitAll();
  void EmitNode(NodeInfo *NI);
  static unsigned CountResults(SDNode *Node);
  static unsigned CountOperands(SDNode *Node);
  unsigned CreateVirtualRegisters(MachineInstr *MI,
                                  unsigned NumResults,
                                  const TargetInstrDescriptor &II);
  unsigned EmitDAG(SDOperand A);

  void printSI(std::ostream &O, NodeInfo *NI) const;
  void print(std::ostream &O) const;
  inline void dump(const char *tag) const { std::cerr << tag; dump(); }
  void dump() const;
};
//===----------------------------------------------------------------------===//

} // namespace

//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Add - Adds a definer and user pair to a node group.
///
void NodeGroup::Add(NodeInfo *D, NodeInfo *U) {
  // Get current groups
  NodeGroup *DGroup = D->Group;
  NodeGroup *UGroup = U->Group;
  // If both are members of groups
  if (DGroup && UGroup) {
    // There may have been another edge connecting 
    if (DGroup == UGroup) return;
    // Add the pending users count
    DGroup->addPending(UGroup->getPending());
    // For each member of the users group
    NodeGroupIterator UNGI(U);
    while (NodeInfo *UNI = UNGI.next() ) {
      // Change the group
      UNI->Group = DGroup;
      // For each member of the definers group
      NodeGroupIterator DNGI(D);
      while (NodeInfo *DNI = DNGI.next() ) {
        // Remove internal edges
        DGroup->addPending(-CountInternalUses(DNI, UNI));
      }
    }
    // Merge the two lists
    DGroup->insert(DGroup->end(), UGroup->begin(), UGroup->end());
  } else if (DGroup) {
    // Make user member of definers group
    U->Group = DGroup;
    // Add users uses to definers group pending
    DGroup->addPending(U->Node->use_size());
    // For each member of the definers group
    NodeGroupIterator DNGI(D);
    while (NodeInfo *DNI = DNGI.next() ) {
      // Remove internal edges
      DGroup->addPending(-CountInternalUses(DNI, U));
    }
    DGroup->push_back(U);
  } else if (UGroup) {
    // Make definer member of users group
    D->Group = UGroup;
    // Add definers uses to users group pending
    UGroup->addPending(D->Node->use_size());
    // For each member of the users group
    NodeGroupIterator UNGI(U);
    while (NodeInfo *UNI = UNGI.next() ) {
      // Remove internal edges
      UGroup->addPending(-CountInternalUses(D, UNI));
    }
    UGroup->insert(UGroup->begin(), D);
  } else {
    D->Group = U->Group = DGroup = new NodeGroup();
    DGroup->addPending(D->Node->use_size() + U->Node->use_size() -
                       CountInternalUses(D, U));
    DGroup->push_back(D);
    DGroup->push_back(U);
  }
}

/// CountInternalUses - Returns the number of edges between the two nodes.
///
unsigned NodeGroup::CountInternalUses(NodeInfo *D, NodeInfo *U) {
  unsigned N = 0;
  for (SDNode:: use_iterator UI = D->Node->use_begin(),
                             E = D->Node->use_end(); UI != E; UI++) {
    if (*UI == U->Node) N++;
  }
  return N;
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// isFlagDefiner - Returns true if the node defines a flag result.
bool SimpleSched::isFlagDefiner(SDNode *A) {
  unsigned N = A->getNumValues();
  return N && A->getValueType(N - 1) == MVT::Flag;
}

/// isFlagUser - Returns true if the node uses a flag result.
///
bool SimpleSched::isFlagUser(SDNode *A) {
  unsigned N = A->getNumOperands();
  return N && A->getOperand(N - 1).getValueType() == MVT::Flag;
}

/// isDefiner - Return true if node A is a definer for B.
///
bool SimpleSched::isDefiner(NodeInfo *A, NodeInfo *B) {
  // While there are A nodes
  NodeGroupIterator NII(A);
  while (NodeInfo *NI = NII.next()) {
    // Extract node
    SDNode *Node = NI->Node;
    // While there operands in nodes of B
    NodeGroupOpIterator NGOI(B);
    while (!NGOI.isEnd()) {
      SDOperand Op = NGOI.next();
      // If node from A defines a node in B
      if (Node == Op.Val) return true;
    }
  }
  return false;
}

/// isPassiveNode - Return true if the node is a non-scheduled leaf.
///
bool SimpleSched::isPassiveNode(SDNode *Node) {
  if (isa<ConstantSDNode>(Node))       return true;
  if (isa<RegisterSDNode>(Node))       return true;
  if (isa<GlobalAddressSDNode>(Node))  return true;
  if (isa<BasicBlockSDNode>(Node))     return true;
  if (isa<FrameIndexSDNode>(Node))     return true;
  if (isa<ConstantPoolSDNode>(Node))   return true;
  if (isa<ExternalSymbolSDNode>(Node)) return true;
  return false;
}

/// IncludeNode - Add node to NodeInfo vector.
///
void SimpleSched::IncludeNode(NodeInfo *NI) {
  // Get node
  SDNode *Node = NI->Node;
  // Ignore entry node
if (Node->getOpcode() == ISD::EntryToken) return;
  // Check current count for node
  int Count = NI->getPending();
  // If the node is already in list
  if (Count < 0) return;
  // Decrement count to indicate a visit
  Count--;
  // If count has gone to zero then add node to list
  if (!Count) {
    // Add node
    if (NI->isInGroup()) {
      Ordering.push_back(NI->Group->getLeader());
    } else {
      Ordering.push_back(NI);
    }
    // indicate node has been added
    Count--;
  }
  // Mark as visited with new count 
  NI->setPending(Count);
}

/// VisitAll - Visit each node breadth-wise to produce an initial ordering.
/// Note that the ordering in the Nodes vector is reversed.
void SimpleSched::VisitAll() {
  // Add first element to list
  Ordering.push_back(getNI(DAG.getRoot().Val));
  
  // Iterate through all nodes that have been added
  for (unsigned i = 0; i < Ordering.size(); i++) { // note: size() varies
    // Visit all operands
    NodeGroupOpIterator NGI(Ordering[i]);
    while (!NGI.isEnd()) {
      // Get next operand
      SDOperand Op = NGI.next();
      // Get node
      SDNode *Node = Op.Val;
      // Ignore passive nodes
      if (isPassiveNode(Node)) continue;
      // Check out node
      IncludeNode(getNI(Node));
    }
  }

  // Add entry node last (IncludeNode filters entry nodes)
  if (DAG.getEntryNode().Val != DAG.getRoot().Val)
    Ordering.push_back(getNI(DAG.getEntryNode().Val));
    
  // FIXME - Reverse the order
  for (unsigned i = 0, N = Ordering.size(), Half = N >> 1; i < Half; i++) {
    unsigned j = N - i - 1;
    NodeInfo *tmp = Ordering[i];
    Ordering[i] = Ordering[j];
    Ordering[j] = tmp;
  }
}

/// GatherNodeInfo - Get latency and resource information about each node.
/// 
void SimpleSched::GatherNodeInfo() {
  // Allocate node information
  Info = new NodeInfo[NodeCount];
  // Get base of all nodes table
  SelectionDAG::allnodes_iterator AllNodes = DAG.allnodes_begin();
  
  // For each node being scheduled
  for (unsigned i = 0, N = NodeCount; i < N; i++) {
    // Get next node from DAG all nodes table
    SDNode *Node = AllNodes[i];
    // Fast reference to node schedule info
    NodeInfo* NI = &Info[i];
    // Set up map
    Map[Node] = NI;
    // Set node
    NI->Node = Node;
    // Set pending visit count
    NI->setPending(Node->use_size());    
    
    MVT::ValueType VT = Node->getValueType(0);
    if (Node->isTargetOpcode()) {
      MachineOpCode TOpc = Node->getTargetOpcode();
      // FIXME: This is an ugly (but temporary!) hack to test the scheduler
      // before we have real target info.
      // FIXME NI->Latency = std::max(1, TII.maxLatency(TOpc));
      // FIXME NI->ResourceSet = TII.resources(TOpc);
      if (TII.isCall(TOpc)) {
        NI->ResourceSet = RSInteger;
        NI->Latency = 40;
      } else if (TII.isLoad(TOpc)) {
        NI->ResourceSet = RSLoadStore;
        NI->Latency = 5;
      } else if (TII.isStore(TOpc)) {
        NI->ResourceSet = RSLoadStore;
        NI->Latency = 2;
      } else if (MVT::isInteger(VT)) {
        NI->ResourceSet = RSInteger;
        NI->Latency = 2;
      } else if (MVT::isFloatingPoint(VT)) {
        NI->ResourceSet = RSFloat;
        NI->Latency = 3;
      } else {
        NI->ResourceSet = RSOther;
        NI->Latency = 0;
      }
    } else {
      if (MVT::isInteger(VT)) {
        NI->ResourceSet = RSInteger;
        NI->Latency = 2;
      } else if (MVT::isFloatingPoint(VT)) {
        NI->ResourceSet = RSFloat;
        NI->Latency = 3;
      } else {
        NI->ResourceSet = RSOther;
        NI->Latency = 0;
      }
    }
    
    // Add one slot for the instruction itself
    NI->Latency++;
    
    // Sum up all the latencies for max tally size
    NSlots += NI->Latency;
  }

  // Put flagged nodes into groups
  for (unsigned i = 0, N = NodeCount; i < N; i++) {
    NodeInfo* NI = &Info[i];
    SDNode *Node = NI->Node;

    // For each operand (in reverse to only look at flags)
    for (unsigned N = Node->getNumOperands(); 0 < N--;) {
      // Get operand
      SDOperand Op = Node->getOperand(N);
      // No more flags to walk
      if (Op.getValueType() != MVT::Flag) break;
      // Add do node group
      NodeGroup::Add(getNI(Op.Val), NI);
    }
  }
}

/// isStrongDependency - Return true if node A has results used by node B. 
/// I.E., B must wait for latency of A.
bool SimpleSched::isStrongDependency(NodeInfo *A, NodeInfo *B) {
  // If A defines for B then it's a strong dependency
  return isDefiner(A, B);
}

/// isWeakDependency Return true if node A produces a result that will
/// conflict with operands of B.
bool SimpleSched::isWeakDependency(NodeInfo *A, NodeInfo *B) {
  // TODO check for conflicting real registers and aliases
#if 0 // FIXME - Since we are in SSA form and not checking register aliasing
  return A->Node->getOpcode() == ISD::EntryToken || isStrongDependency(B, A);
#else
  return A->Node->getOpcode() == ISD::EntryToken;
#endif
}

/// ScheduleBackward - Schedule instructions so that any long latency
/// instructions and the critical path get pushed back in time. Time is run in
/// reverse to allow code reuse of the Tally and eliminate the overhead of
/// biasing every slot indices against NSlots.
void SimpleSched::ScheduleBackward() {
  // Size and clear the resource tally
  Tally.Initialize(NSlots);
  // Get number of nodes to schedule
  unsigned N = Ordering.size();
  
  // For each node being scheduled
  for (unsigned i = N; 0 < i--;) {
    NodeInfo *NI = Ordering[i];
    // Track insertion
    unsigned Slot = NotFound;
    
    // Compare against those previously scheduled nodes
    unsigned j = i + 1;
    for (; j < N; j++) {
      // Get following instruction
      NodeInfo *Other = Ordering[j];
      
      // Check dependency against previously inserted nodes
      if (isStrongDependency(NI, Other)) {
        Slot = Other->Slot + Other->Latency;
        break;
      } else if (isWeakDependency(NI, Other)) {
        Slot = Other->Slot;
        break;
      }
    }
    
    // If independent of others (or first entry)
    if (Slot == NotFound) Slot = 0;
    
    // Find a slot where the needed resources are available
    if (NI->ResourceSet)
      Slot = Tally.FindAndReserve(Slot, NI->Latency, NI->ResourceSet);
      
    // Set node slot
    NI->Slot = Slot;
    
    // Insert sort based on slot
    j = i + 1;
    for (; j < N; j++) {
      // Get following instruction
      NodeInfo *Other = Ordering[j];
      // Should we look further
      if (Slot >= Other->Slot) break;
      // Shuffle other into ordering
      Ordering[j - 1] = Other;
    }
    // Insert node in proper slot
    if (j != i + 1) Ordering[j - 1] = NI;
  }
}

/// ScheduleForward - Schedule instructions to maximize packing.
///
void SimpleSched::ScheduleForward() {
  // Size and clear the resource tally
  Tally.Initialize(NSlots);
  // Get number of nodes to schedule
  unsigned N = Ordering.size();
  
  // For each node being scheduled
  for (unsigned i = 0; i < N; i++) {
    NodeInfo *NI = Ordering[i];
    // Track insertion
    unsigned Slot = NotFound;
    
    // Compare against those previously scheduled nodes
    unsigned j = i;
    for (; 0 < j--;) {
      // Get following instruction
      NodeInfo *Other = Ordering[j];
      
      // Check dependency against previously inserted nodes
      if (isStrongDependency(Other, NI)) {
        Slot = Other->Slot + Other->Latency;
        break;
      } else if (isWeakDependency(Other, NI)) {
        Slot = Other->Slot;
        break;
      }
    }
    
    // If independent of others (or first entry)
    if (Slot == NotFound) Slot = 0;
    
    // Find a slot where the needed resources are available
    if (NI->ResourceSet)
      Slot = Tally.FindAndReserve(Slot, NI->Latency, NI->ResourceSet);
      
    // Set node slot
    NI->Slot = Slot;
    
    // Insert sort based on slot
    j = i;
    for (; 0 < j--;) {
      // Get following instruction
      NodeInfo *Other = Ordering[j];
      // Should we look further
      if (Slot >= Other->Slot) break;
      // Shuffle other into ordering
      Ordering[j + 1] = Other;
    }
    // Insert node in proper slot
    if (j != i) Ordering[j + 1] = NI;
  }
}

/// EmitAll - Emit all nodes in schedule sorted order.
///
void SimpleSched::EmitAll() {
  // For each node in the ordering
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    // Get the scheduling info
    NodeInfo *NI = Ordering[i];
#if 0
    // Iterate through nodes
    NodeGroupIterator NGI(Ordering[i]);
    while (NodeInfo *NI = NGI.next()) EmitNode(NI);
#else
    if (NI->isInGroup()) {
      if (NI->isGroupLeader()) {
        NodeGroupIterator NGI(Ordering[i]);
        while (NodeInfo *NI = NGI.next()) EmitNode(NI);
      }
    } else {
      EmitNode(NI);
    }
#endif
  }
}

/// CountResults - The results of target nodes have register or immediate
/// operands first, then an optional chain, and optional flag operands (which do
/// not go into the machine instrs.)
unsigned SimpleSched::CountResults(SDNode *Node) {
  unsigned N = Node->getNumValues();
  while (N && Node->getValueType(N - 1) == MVT::Flag)
    --N;
  if (N && Node->getValueType(N - 1) == MVT::Other)
    --N;    // Skip over chain result.
  return N;
}

/// CountOperands  The inputs to target nodes have any actual inputs first,
/// followed by an optional chain operand, then flag operands.  Compute the
/// number of actual operands that  will go into the machine instr.
unsigned SimpleSched::CountOperands(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Flag)
    --N;
  if (N && Node->getOperand(N - 1).getValueType() == MVT::Other)
    --N; // Ignore chain if it exists.
  return N;
}

/// CreateVirtualRegisters - Add result register values for things that are
/// defined by this instruction.
unsigned SimpleSched::CreateVirtualRegisters(MachineInstr *MI,
                                             unsigned NumResults,
                                             const TargetInstrDescriptor &II) {
  // Create the result registers for this node and add the result regs to
  // the machine instruction.
  const TargetOperandInfo *OpInfo = II.OpInfo;
  unsigned ResultReg = RegMap->createVirtualRegister(OpInfo[0].RegClass);
  MI->addRegOperand(ResultReg, MachineOperand::Def);
  for (unsigned i = 1; i != NumResults; ++i) {
    assert(OpInfo[i].RegClass && "Isn't a register operand!");
    MI->addRegOperand(RegMap->createVirtualRegister(OpInfo[0].RegClass),
                      MachineOperand::Def);
  }
  return ResultReg;
}

/// EmitNode - Generate machine code for an node and needed dependencies.
///
void SimpleSched::EmitNode(NodeInfo *NI) {
  unsigned VRBase = 0;                 // First virtual register for node
  SDNode *Node = NI->Node;
  
  // If machine instruction
  if (Node->isTargetOpcode()) {
    unsigned Opc = Node->getTargetOpcode();
    const TargetInstrDescriptor &II = TII.get(Opc);

    unsigned NumResults = CountResults(Node);
    unsigned NodeOperands = CountOperands(Node);
    unsigned NumMIOperands = NodeOperands + NumResults;
#ifndef NDEBUG
    assert((unsigned(II.numOperands) == NumMIOperands || II.numOperands == -1)&&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, NumMIOperands, true, true);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults) VRBase = CreateVirtualRegisters(MI, NumResults, II);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i) {
      if (Node->getOperand(i).isTargetOpcode()) {
        // Note that this case is redundant with the final else block, but we
        // include it because it is the most common and it makes the logic
        // simpler here.
        assert(Node->getOperand(i).getValueType() != MVT::Other &&
               Node->getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        
        MI->addRegOperand(getVR(Node->getOperand(i)), MachineOperand::Use);
      } else if (ConstantSDNode *C =
                 dyn_cast<ConstantSDNode>(Node->getOperand(i))) {
        MI->addZeroExtImm64Operand(C->getValue());
      } else if (RegisterSDNode*R =
                 dyn_cast<RegisterSDNode>(Node->getOperand(i))) {
        MI->addRegOperand(R->getReg(), MachineOperand::Use);
      } else if (GlobalAddressSDNode *TGA =
                       dyn_cast<GlobalAddressSDNode>(Node->getOperand(i))) {
        MI->addGlobalAddressOperand(TGA->getGlobal(), false, 0);
      } else if (BasicBlockSDNode *BB =
                       dyn_cast<BasicBlockSDNode>(Node->getOperand(i))) {
        MI->addMachineBasicBlockOperand(BB->getBasicBlock());
      } else if (FrameIndexSDNode *FI =
                       dyn_cast<FrameIndexSDNode>(Node->getOperand(i))) {
        MI->addFrameIndexOperand(FI->getIndex());
      } else if (ConstantPoolSDNode *CP = 
                    dyn_cast<ConstantPoolSDNode>(Node->getOperand(i))) {
        unsigned Idx = ConstPool->getConstantPoolIndex(CP->get());
        MI->addConstantPoolIndexOperand(Idx);
      } else if (ExternalSymbolSDNode *ES = 
                 dyn_cast<ExternalSymbolSDNode>(Node->getOperand(i))) {
        MI->addExternalSymbolOperand(ES->getSymbol(), false);
      } else {
        assert(Node->getOperand(i).getValueType() != MVT::Other &&
               Node->getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        MI->addRegOperand(getVR(Node->getOperand(i)), MachineOperand::Use);
      }
    }
    
    // Now that we have emitted all operands, emit this instruction itself.
    if ((II.Flags & M_USES_CUSTOM_DAG_SCHED_INSERTION) == 0) {
      BB->insert(BB->end(), MI);
    } else {
      // Insert this instruction into the end of the basic block, potentially
      // taking some custom action.
      BB = DAG.getTargetLoweringInfo().InsertAtEndOfBasicBlock(MI, BB);
    }
  } else {
    switch (Node->getOpcode()) {
    default:
      Node->dump(); 
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: // fall thru
    case ISD::TokenFactor:
      break;
    case ISD::CopyToReg: {
      unsigned Val = getVR(Node->getOperand(2));
      MRI.copyRegToReg(*BB, BB->end(),
                       cast<RegisterSDNode>(Node->getOperand(1))->getReg(), Val,
                       RegMap->getRegClass(Val));
      break;
    }
    case ISD::CopyFromReg: {
      unsigned SrcReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
      
      // Figure out the register class to create for the destreg.
      const TargetRegisterClass *TRC = 0;
      if (MRegisterInfo::isVirtualRegister(SrcReg)) {
        TRC = RegMap->getRegClass(SrcReg);
      } else {
        // FIXME: we don't know what register class to generate this for.  Do
        // a brute force search and pick the first match. :(
        for (MRegisterInfo::regclass_iterator I = MRI.regclass_begin(),
               E = MRI.regclass_end(); I != E; ++I)
          if ((*I)->contains(SrcReg)) {
            TRC = *I;
            break;
          }
        assert(TRC && "Couldn't find register class for reg copy!");
      }
      
      // Create the reg, emit the copy.
      VRBase = RegMap->createVirtualRegister(TRC);
      MRI.copyRegToReg(*BB, BB->end(), VRBase, SrcReg, TRC);
      break;
    }
    }
  }

  assert(NI->VRBase == 0 && "Node emitted out of order - early");
  NI->VRBase = VRBase;
}

/// EmitDag - Generate machine code for an operand and needed dependencies.
///
unsigned SimpleSched::EmitDAG(SDOperand Op) {
  std::map<SDNode *, unsigned>::iterator OpI = VRMap.lower_bound(Op.Val);
  if (OpI != VRMap.end() && OpI->first == Op.Val)
    return OpI->second + Op.ResNo;
  unsigned &OpSlot = VRMap.insert(OpI, std::make_pair(Op.Val, 0))->second;
  
  unsigned ResultReg = 0;
  if (Op.isTargetOpcode()) {
    unsigned Opc = Op.getTargetOpcode();
    const TargetInstrDescriptor &II = TII.get(Opc);

    unsigned NumResults = CountResults(Op.Val);
    unsigned NodeOperands = CountOperands(Op.Val);
    unsigned NumMIOperands = NodeOperands + NumResults;
#ifndef NDEBUG
    assert((unsigned(II.numOperands) == NumMIOperands || II.numOperands == -1)&&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = new MachineInstr(Opc, NumMIOperands, true, true);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults) ResultReg = CreateVirtualRegisters(MI, NumResults, II);
    
    // If there is a token chain operand, emit it first, as a hack to get avoid
    // really bad cases.
    if (Op.getNumOperands() > NodeOperands &&
        Op.getOperand(NodeOperands).getValueType() == MVT::Other) {
      EmitDAG(Op.getOperand(NodeOperands));
    }
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i) {
      if (Op.getOperand(i).isTargetOpcode()) {
        // Note that this case is redundant with the final else block, but we
        // include it because it is the most common and it makes the logic
        // simpler here.
        assert(Op.getOperand(i).getValueType() != MVT::Other &&
               Op.getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        
        MI->addRegOperand(EmitDAG(Op.getOperand(i)), MachineOperand::Use);
      } else if (ConstantSDNode *C =
                                   dyn_cast<ConstantSDNode>(Op.getOperand(i))) {
        MI->addZeroExtImm64Operand(C->getValue());
      } else if (RegisterSDNode*R =dyn_cast<RegisterSDNode>(Op.getOperand(i))) {
        MI->addRegOperand(R->getReg(), MachineOperand::Use);
      } else if (GlobalAddressSDNode *TGA =
                       dyn_cast<GlobalAddressSDNode>(Op.getOperand(i))) {
        MI->addGlobalAddressOperand(TGA->getGlobal(), false, 0);
      } else if (BasicBlockSDNode *BB =
                       dyn_cast<BasicBlockSDNode>(Op.getOperand(i))) {
        MI->addMachineBasicBlockOperand(BB->getBasicBlock());
      } else if (FrameIndexSDNode *FI =
                       dyn_cast<FrameIndexSDNode>(Op.getOperand(i))) {
        MI->addFrameIndexOperand(FI->getIndex());
      } else if (ConstantPoolSDNode *CP = 
                    dyn_cast<ConstantPoolSDNode>(Op.getOperand(i))) {
        unsigned Idx = ConstPool->getConstantPoolIndex(CP->get());
        MI->addConstantPoolIndexOperand(Idx);
      } else if (ExternalSymbolSDNode *ES = 
                 dyn_cast<ExternalSymbolSDNode>(Op.getOperand(i))) {
        MI->addExternalSymbolOperand(ES->getSymbol(), false);
      } else {
        assert(Op.getOperand(i).getValueType() != MVT::Other &&
               Op.getOperand(i).getValueType() != MVT::Flag &&
               "Chain and flag operands should occur at end of operand list!");
        MI->addRegOperand(EmitDAG(Op.getOperand(i)), MachineOperand::Use);
      }
    }

    // Finally, if this node has any flag operands, we *must* emit them last, to
    // avoid emitting operations that might clobber the flags.
    if (Op.getNumOperands() > NodeOperands) {
      unsigned i = NodeOperands;
      if (Op.getOperand(i).getValueType() == MVT::Other)
        ++i;  // the chain is already selected.
      for (unsigned N = Op.getNumOperands(); i < N; i++) {
        assert(Op.getOperand(i).getValueType() == MVT::Flag &&
               "Must be flag operands!");
        EmitDAG(Op.getOperand(i));
      }
    }
    
    // Now that we have emitted all operands, emit this instruction itself.
    if ((II.Flags & M_USES_CUSTOM_DAG_SCHED_INSERTION) == 0) {
      BB->insert(BB->end(), MI);
    } else {
      // Insert this instruction into the end of the basic block, potentially
      // taking some custom action.
      BB = DAG.getTargetLoweringInfo().InsertAtEndOfBasicBlock(MI, BB);
    }
  } else {
    switch (Op.getOpcode()) {
    default:
      Op.Val->dump(); 
      assert(0 && "This target-independent node should have been selected!");
    case ISD::EntryToken: break;
    case ISD::TokenFactor:
      for (unsigned i = 0, N = Op.getNumOperands(); i < N; i++) {
        EmitDAG(Op.getOperand(i));
      }
      break;
    case ISD::CopyToReg: {
      SDOperand FlagOp;
      if (Op.getNumOperands() == 4) {
        FlagOp = Op.getOperand(3);
      }
      if (Op.getOperand(0).Val != FlagOp.Val) {
        EmitDAG(Op.getOperand(0));   // Emit the chain.
      }
      unsigned Val = EmitDAG(Op.getOperand(2));
      if (FlagOp.Val) {
        EmitDAG(FlagOp);
      }
      MRI.copyRegToReg(*BB, BB->end(),
                       cast<RegisterSDNode>(Op.getOperand(1))->getReg(), Val,
                       RegMap->getRegClass(Val));
      break;
    }
    case ISD::CopyFromReg: {
      EmitDAG(Op.getOperand(0));   // Emit the chain.
      unsigned SrcReg = cast<RegisterSDNode>(Op.getOperand(1))->getReg();
      
      // Figure out the register class to create for the destreg.
      const TargetRegisterClass *TRC = 0;
      if (MRegisterInfo::isVirtualRegister(SrcReg)) {
        TRC = RegMap->getRegClass(SrcReg);
      } else {
        // FIXME: we don't know what register class to generate this for.  Do
        // a brute force search and pick the first match. :(
        for (MRegisterInfo::regclass_iterator I = MRI.regclass_begin(),
               E = MRI.regclass_end(); I != E; ++I)
          if ((*I)->contains(SrcReg)) {
            TRC = *I;
            break;
          }
        assert(TRC && "Couldn't find register class for reg copy!");
      }
      
      // Create the reg, emit the copy.
      ResultReg = RegMap->createVirtualRegister(TRC);
      MRI.copyRegToReg(*BB, BB->end(), ResultReg, SrcReg, TRC);
      break;
    }
    }
  }

  OpSlot = ResultReg;
  return ResultReg+Op.ResNo;
}

/// Schedule - Order nodes according to selected style.
///
void SimpleSched::Schedule() {
  switch (ScheduleStyle) {
  case simpleScheduling:
    // Number the nodes
    NodeCount = DAG.allnodes_size();
    // Don't waste time if is only entry and return
    if (NodeCount > 3) {
      // Get latency and resource requirements
      GatherNodeInfo();
      // Breadth first walk of DAG
      VisitAll();
      DEBUG(dump("Pre-"));
      // Push back long instructions and critical path
      ScheduleBackward();
      DEBUG(dump("Mid-"));
      // Pack instructions to maximize resource utilization
      ScheduleForward();
      DEBUG(dump("Post-"));
      // Emit in scheduled order
      EmitAll();
      break;
    } // fall thru
  case noScheduling:
    // Emit instructions in using a DFS from the exit root
    EmitDAG(DAG.getRoot());
    break;
  }
}

/// printSI - Print schedule info.
///
void SimpleSched::printSI(std::ostream &O, NodeInfo *NI) const {
#ifndef NDEBUG
  using namespace std;
  SDNode *Node = NI->Node;
  O << " "
    << hex << Node
    << ", RS=" << NI->ResourceSet
    << ", Lat=" << NI->Latency
    << ", Slot=" << NI->Slot
    << ", ARITY=(" << Node->getNumOperands() << ","
                   << Node->getNumValues() << ")"
    << " " << Node->getOperationName(&DAG);
  if (isFlagDefiner(Node)) O << "<#";
  if (isFlagUser(Node)) O << ">#";
#endif
}

/// print - Print ordering to specified output stream.
///
void SimpleSched::print(std::ostream &O) const {
#ifndef NDEBUG
  using namespace std;
  O << "Ordering\n";
  for (unsigned i = 0, N = Ordering.size(); i < N; i++) {
    NodeInfo *NI = Ordering[i];
    printSI(O, NI);
    O << "\n";
    if (NI->isGroupLeader()) {
      NodeGroup *Group = NI->Group;
      for (NIIterator NII = Group->begin(), E = Group->end();
           NII != E; NII++) {
        O << "    ";
        printSI(O, *NII);
        O << "\n";
      }
    }
  }
#endif
}

/// dump - Print ordering to std::cerr.
///
void SimpleSched::dump() const {
  print(std::cerr);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// ScheduleAndEmitDAG - Pick a safe ordering and emit instructions for each
/// target node in the graph.
void SelectionDAGISel::ScheduleAndEmitDAG(SelectionDAG &SD) {
  if (ViewDAGs) SD.viewGraph();
  BB = SimpleSched(SD, BB).Run();  
}

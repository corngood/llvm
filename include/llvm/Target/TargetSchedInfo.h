//===- Target/MachineSchedInfo.h - Target Instruction Sched Info -*- C++ -*-==//
//
// This file describes the target machine to the instruction scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MACHINESCHEDINFO_H
#define LLVM_TARGET_MACHINESCHEDINFO_H

#include "llvm/Target/MachineInstrInfo.h"
#include <Support/hash_map>

typedef long long cycles_t; 
static const cycles_t HUGE_LATENCY = ~((long long) 1 << (sizeof(cycles_t)-2));
static const cycles_t INVALID_LATENCY = -HUGE_LATENCY; 
static const unsigned MAX_OPCODE_SIZE = 16;

class OpCodePair {
public:
  long val;			// make long by concatenating two opcodes
  OpCodePair(MachineOpCode op1, MachineOpCode op2)
    : val((op1 < 0 || op2 < 0)?
	-1 : (long)((((unsigned) op1) << MAX_OPCODE_SIZE) | (unsigned) op2)) {}
  bool operator==(const OpCodePair& op) const {
    return val == op.val;
  }
private:
  OpCodePair();			// disable for now
};

namespace HASH_NAMESPACE {
  template <> struct hash<OpCodePair> {
    size_t operator()(const OpCodePair& pair) const {
      return hash<long>()(pair.val);
    }
  };
}

//---------------------------------------------------------------------------
// class MachineResource 
// class CPUResource
// 
// Purpose:
//   Representation of a single machine resource used in specifying
//   resource usages of machine instructions for scheduling.
//---------------------------------------------------------------------------


typedef unsigned resourceId_t;

struct MachineResource {
  const std::string rname;
  resourceId_t rid;
  
  MachineResource(const std::string &resourceName)
    : rname(resourceName), rid(nextId++) {}
  
private:
  static resourceId_t nextId;
  MachineResource();			// disable
};


struct CPUResource : public MachineResource {
  int maxNumUsers;   // MAXINT if no restriction
  
  CPUResource(const std::string& rname, int maxUsers)
    : MachineResource(rname), maxNumUsers(maxUsers) {}
};


//---------------------------------------------------------------------------
// struct InstrClassRUsage
// struct InstrRUsageDelta 
// struct InstrIssueDelta 
// struct InstrRUsage 
// 
// Purpose:
//   The first three are structures used to specify machine resource 
//   usages for each instruction in a machine description file:
//    InstrClassRUsage : resource usages common to all instrs. in a class
//    InstrRUsageDelta : add/delete resource usage for individual instrs. 
//    InstrIssueDelta  : add/delete instr. issue info for individual instrs 
//   
//   The last one (InstrRUsage) is the internal representation of
//   instruction resource usage constructed from the above three.
//---------------------------------------------------------------------------

const int MAX_NUM_SLOTS  = 32;
const int MAX_NUM_CYCLES = 32;

struct InstrClassRUsage {
  InstrSchedClass schedClass;
  int		totCycles;
  
  // Issue restrictions common to instructions in this class
  unsigned      maxNumIssue;
  bool	        isSingleIssue;
  bool	        breaksGroup;
  cycles_t      numBubbles;
  
  // Feasible slots to use for instructions in this class.
  // The size of vector S[] is `numSlots'.
  unsigned      numSlots;
  unsigned      feasibleSlots[MAX_NUM_SLOTS];
  
  // Resource usages common to instructions in this class.
  // The size of vector V[] is `numRUEntries'.
  unsigned      numRUEntries;
  struct {
    resourceId_t resourceId;
    unsigned    startCycle;
    int	        numCycles;
  } V[MAX_NUM_CYCLES];
};

struct InstrRUsageDelta {
  MachineOpCode opCode;
  resourceId_t	resourceId;
  unsigned      startCycle;
  int		numCycles;
};

// Specify instruction issue restrictions for individual instructions
// that differ from the common rules for the class.
// 
struct InstrIssueDelta {
  MachineOpCode	opCode;
  bool		isSingleIssue;
  bool		breaksGroup;
  cycles_t	numBubbles;
};


struct InstrRUsage {
  bool		sameAsClass;
  
  // Issue restrictions for this instruction
  bool		isSingleIssue;
  bool		breaksGroup;
  cycles_t	numBubbles;
  
  // Feasible slots to use for this instruction.
  std::vector<bool> feasibleSlots;
  
  // Resource usages for this instruction, with one resource vector per cycle.
  cycles_t	numCycles;
  std::vector<std::vector<resourceId_t> > resourcesByCycle;
  
private:
  // Conveniences for initializing this structure
  void setTo(const InstrClassRUsage& classRU);

  void addIssueDelta(const InstrIssueDelta& delta) {
    sameAsClass = false;
    isSingleIssue = delta.isSingleIssue;
    breaksGroup = delta.breaksGroup;
    numBubbles = delta.numBubbles;
  }

  void addUsageDelta	(const InstrRUsageDelta& delta);
  void setMaxSlots	(int maxNumSlots) {
    feasibleSlots.resize(maxNumSlots);
  }
  
  friend class MachineSchedInfo;	// give access to these functions
};


//---------------------------------------------------------------------------
// class MachineSchedInfo
//
// Purpose:
//   Common interface to machine information for instruction scheduling
//---------------------------------------------------------------------------

class MachineSchedInfo : public NonCopyableV {
public:
  const TargetMachine& target;
  
  unsigned maxNumIssueTotal;
  int	longestIssueConflict;
  
  int	branchMispredictPenalty;	// 4 for SPARC IIi
  int	branchTargetUnknownPenalty;	// 2 for SPARC IIi
  int   l1DCacheMissPenalty;		// 7 or 9 for SPARC IIi
  int   l1ICacheMissPenalty;		// ? for SPARC IIi
  
  bool	inOrderLoads;			// true for SPARC IIi
  bool	inOrderIssue;			// true for SPARC IIi
  bool	inOrderExec;			// false for most architectures
  bool	inOrderRetire;			// true for most architectures
  
protected:
  inline const InstrRUsage& getInstrRUsage(MachineOpCode opCode) const {
    assert(opCode >= 0 && opCode < (int) instrRUsages.size());
    return instrRUsages[opCode];
  }
  inline const InstrClassRUsage&
			getClassRUsage(const InstrSchedClass& sc) const {
    assert(sc >= 0 && sc < numSchedClasses);
    return classRUsages[sc];
  }
  
public:
  /*ctor*/	   MachineSchedInfo	(const TargetMachine& tgt,
                                         int                  _numSchedClasses,
					 const InstrClassRUsage* _classRUsages,
					 const InstrRUsageDelta* _usageDeltas,
					 const InstrIssueDelta*  _issueDeltas,
					 unsigned _numUsageDeltas,
					 unsigned _numIssueDeltas);
  /*dtor*/ virtual ~MachineSchedInfo	() {}
  
  inline const MachineInstrInfo& getInstrInfo() const {
    return *mii;
  }
  
  inline int		getNumSchedClasses()  const {
    return numSchedClasses;
  }  
  
  inline  unsigned getMaxNumIssueTotal() const {
    return maxNumIssueTotal;
  }
  
  inline  unsigned getMaxIssueForClass(const InstrSchedClass& sc) const {
    assert(sc >= 0 && sc < numSchedClasses);
    return classRUsages[sc].maxNumIssue;
  }

  inline InstrSchedClass getSchedClass	(MachineOpCode opCode) const {
    return getInstrInfo().getSchedClass(opCode);
  } 
  
  inline  bool	instrCanUseSlot		(MachineOpCode opCode,
					 unsigned s) const {
    assert(s < getInstrRUsage(opCode).feasibleSlots.size() && "Invalid slot!");
    return getInstrRUsage(opCode).feasibleSlots[s];
  }
  
  inline int	getLongestIssueConflict	() const {
    return longestIssueConflict;
  }
  
  inline  int 	getMinIssueGap		(MachineOpCode fromOp,
					 MachineOpCode toOp)   const {
    hash_map<OpCodePair,int>::const_iterator
      I = issueGaps.find(OpCodePair(fromOp, toOp));
    return (I == issueGaps.end())? 0 : (*I).second;
  }
  
  inline const std::vector<MachineOpCode>*
		getConflictList(MachineOpCode opCode) const {
    hash_map<MachineOpCode, std::vector<MachineOpCode> >::const_iterator
      I = conflictLists.find(opCode);
    return (I == conflictLists.end())? NULL : & (*I).second;
  }
  
  inline  bool	isSingleIssue		(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).isSingleIssue;
  }
  
  inline  bool	breaksIssueGroup	(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).breaksGroup;
  }
  
  inline  unsigned numBubblesAfter	(MachineOpCode opCode) const {
    return getInstrRUsage(opCode).numBubbles;
  }
  
protected:
  virtual void	initializeResources	();
  
private:
  void computeInstrResources(const std::vector<InstrRUsage>& instrRUForClasses);
  void computeIssueGaps(const std::vector<InstrRUsage>& instrRUForClasses);
  
protected:
  int		           numSchedClasses;
  const MachineInstrInfo*  mii;
  const	InstrClassRUsage*  classRUsages;        // raw array by sclass
  const	InstrRUsageDelta*  usageDeltas;	        // raw array [1:numUsageDeltas]
  const InstrIssueDelta*   issueDeltas;	        // raw array [1:numIssueDeltas]
  unsigned 		   numUsageDeltas;
  unsigned 		   numIssueDeltas;
  
  std::vector<InstrRUsage>      instrRUsages;   // indexed by opcode
  hash_map<OpCodePair,int> issueGaps;      // indexed by opcode pair
  hash_map<MachineOpCode, std::vector<MachineOpCode> >
			   conflictLists;       // indexed by opcode
};

#endif

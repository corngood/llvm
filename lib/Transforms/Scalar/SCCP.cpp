//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse conditional constant propagation and merging:
//
// Specifically, this:
//   * Assumes values are constant unless proven otherwise
//   * Assumes BasicBlocks are dead unless proven otherwise
//   * Proves values to be constant, and replaces them with constants
//   * Proves conditional branches to be unconditional
//
// Notice that:
//   * This pass has a habit of making definitions be dead.  It is a good idea
//     to to run a DCE pass sometime after running this pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sccp"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumInstRemoved, "Number of instructions removed");
STATISTIC(NumDeadBlocks , "Number of basic blocks unreachable");

STATISTIC(IPNumInstRemoved, "Number ofinstructions removed by IPSCCP");
STATISTIC(IPNumDeadBlocks , "Number of basic blocks unreachable by IPSCCP");
STATISTIC(IPNumArgsElimed ,"Number of arguments constant propagated by IPSCCP");
STATISTIC(IPNumGlobalConst, "Number of globals found to be constant by IPSCCP");

namespace {
/// LatticeVal class - This class represents the different lattice values that
/// an LLVM value may occupy.  It is a simple class with value semantics.
///
class VISIBILITY_HIDDEN LatticeVal {
  enum {
    /// undefined - This LLVM Value has no known value yet.
    undefined,
    
    /// constant - This LLVM Value has a specific constant value.
    constant,

    /// forcedconstant - This LLVM Value was thought to be undef until
    /// ResolvedUndefsIn.  This is treated just like 'constant', but if merged
    /// with another (different) constant, it goes to overdefined, instead of
    /// asserting.
    forcedconstant,
    
    /// overdefined - This instruction is not known to be constant, and we know
    /// it has a value.
    overdefined
  } LatticeValue;    // The current lattice position
  
  Constant *ConstantVal; // If Constant value, the current value
public:
  inline LatticeVal() : LatticeValue(undefined), ConstantVal(0) {}
  
  // markOverdefined - Return true if this is a new status to be in...
  inline bool markOverdefined() {
    if (LatticeValue != overdefined) {
      LatticeValue = overdefined;
      return true;
    }
    return false;
  }

  // markConstant - Return true if this is a new status for us.
  inline bool markConstant(Constant *V) {
    if (LatticeValue != constant) {
      if (LatticeValue == undefined) {
        LatticeValue = constant;
        assert(V && "Marking constant with NULL");
        ConstantVal = V;
      } else {
        assert(LatticeValue == forcedconstant && 
               "Cannot move from overdefined to constant!");
        // Stay at forcedconstant if the constant is the same.
        if (V == ConstantVal) return false;
        
        // Otherwise, we go to overdefined.  Assumptions made based on the
        // forced value are possibly wrong.  Assuming this is another constant
        // could expose a contradiction.
        LatticeValue = overdefined;
      }
      return true;
    } else {
      assert(ConstantVal == V && "Marking constant with different value");
    }
    return false;
  }

  inline void markForcedConstant(Constant *V) {
    assert(LatticeValue == undefined && "Can't force a defined value!");
    LatticeValue = forcedconstant;
    ConstantVal = V;
  }
  
  inline bool isUndefined() const { return LatticeValue == undefined; }
  inline bool isConstant() const {
    return LatticeValue == constant || LatticeValue == forcedconstant;
  }
  inline bool isOverdefined() const { return LatticeValue == overdefined; }

  inline Constant *getConstant() const {
    assert(isConstant() && "Cannot get the constant of a non-constant!");
    return ConstantVal;
  }
};

//===----------------------------------------------------------------------===//
//
/// SCCPSolver - This class is a general purpose solver for Sparse Conditional
/// Constant Propagation.
///
class SCCPSolver : public InstVisitor<SCCPSolver> {
  SmallSet<BasicBlock*, 16> BBExecutable;// The basic blocks that are executable
  std::map<Value*, LatticeVal> ValueState;  // The state each value is in.

  /// GlobalValue - If we are tracking any values for the contents of a global
  /// variable, we keep a mapping from the constant accessor to the element of
  /// the global, to the currently known value.  If the value becomes
  /// overdefined, it's entry is simply removed from this map.
  DenseMap<GlobalVariable*, LatticeVal> TrackedGlobals;

  /// TrackedFunctionRetVals - If we are tracking arguments into and the return
  /// value out of a function, it will have an entry in this map, indicating
  /// what the known return value for the function is.
  DenseMap<Function*, LatticeVal> TrackedFunctionRetVals;

  // The reason for two worklists is that overdefined is the lowest state
  // on the lattice, and moving things to overdefined as fast as possible
  // makes SCCP converge much faster.
  // By having a separate worklist, we accomplish this because everything
  // possibly overdefined will become overdefined at the soonest possible
  // point.
  std::vector<Value*> OverdefinedInstWorkList;
  std::vector<Value*> InstWorkList;


  std::vector<BasicBlock*>  BBWorkList;  // The BasicBlock work list

  /// UsersOfOverdefinedPHIs - Keep track of any users of PHI nodes that are not
  /// overdefined, despite the fact that the PHI node is overdefined.
  std::multimap<PHINode*, Instruction*> UsersOfOverdefinedPHIs;

  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  typedef std::pair<BasicBlock*,BasicBlock*> Edge;
  std::set<Edge> KnownFeasibleEdges;
public:

  /// MarkBlockExecutable - This method can be used by clients to mark all of
  /// the blocks that are known to be intrinsically live in the processed unit.
  void MarkBlockExecutable(BasicBlock *BB) {
    DOUT << "Marking Block Executable: " << BB->getName() << "\n";
    BBExecutable.insert(BB);   // Basic block is executable!
    BBWorkList.push_back(BB);  // Add the block to the work list!
  }

  /// TrackValueOfGlobalVariable - Clients can use this method to
  /// inform the SCCPSolver that it should track loads and stores to the
  /// specified global variable if it can.  This is only legal to call if
  /// performing Interprocedural SCCP.
  void TrackValueOfGlobalVariable(GlobalVariable *GV) {
    const Type *ElTy = GV->getType()->getElementType();
    if (ElTy->isFirstClassType()) {
      LatticeVal &IV = TrackedGlobals[GV];
      if (!isa<UndefValue>(GV->getInitializer()))
        IV.markConstant(GV->getInitializer());
    }
  }

  /// AddTrackedFunction - If the SCCP solver is supposed to track calls into
  /// and out of the specified function (which cannot have its address taken),
  /// this method must be called.
  void AddTrackedFunction(Function *F) {
    assert(F->hasInternalLinkage() && "Can only track internal functions!");
    // Add an entry, F -> undef.
    TrackedFunctionRetVals[F];
  }

  /// Solve - Solve for constants and executable blocks.
  ///
  void Solve();

  /// ResolvedUndefsIn - While solving the dataflow for a function, we assume
  /// that branches on undef values cannot reach any of their successors.
  /// However, this is not a safe assumption.  After we solve dataflow, this
  /// method should be use to handle this.  If this returns true, the solver
  /// should be rerun.
  bool ResolvedUndefsIn(Function &F);

  /// getExecutableBlocks - Once we have solved for constants, return the set of
  /// blocks that is known to be executable.
  SmallSet<BasicBlock*, 16> &getExecutableBlocks() {
    return BBExecutable;
  }

  /// getValueMapping - Once we have solved for constants, return the mapping of
  /// LLVM values to LatticeVals.
  std::map<Value*, LatticeVal> &getValueMapping() {
    return ValueState;
  }

  /// getTrackedFunctionRetVals - Get the inferred return value map.
  ///
  const DenseMap<Function*, LatticeVal> &getTrackedFunctionRetVals() {
    return TrackedFunctionRetVals;
  }

  /// getTrackedGlobals - Get and return the set of inferred initializers for
  /// global variables.
  const DenseMap<GlobalVariable*, LatticeVal> &getTrackedGlobals() {
    return TrackedGlobals;
  }

  inline void markOverdefined(Value *V) {
    markOverdefined(ValueState[V], V);
  }

private:
  // markConstant - Make a value be marked as "constant".  If the value
  // is not already a constant, add it to the instruction work list so that
  // the users of the instruction are updated later.
  //
  inline void markConstant(LatticeVal &IV, Value *V, Constant *C) {
    if (IV.markConstant(C)) {
      DOUT << "markConstant: " << *C << ": " << *V;
      InstWorkList.push_back(V);
    }
  }
  
  inline void markForcedConstant(LatticeVal &IV, Value *V, Constant *C) {
    IV.markForcedConstant(C);
    DOUT << "markForcedConstant: " << *C << ": " << *V;
    InstWorkList.push_back(V);
  }
  
  inline void markConstant(Value *V, Constant *C) {
    markConstant(ValueState[V], V, C);
  }

  // markOverdefined - Make a value be marked as "overdefined". If the
  // value is not already overdefined, add it to the overdefined instruction
  // work list so that the users of the instruction are updated later.

  inline void markOverdefined(LatticeVal &IV, Value *V) {
    if (IV.markOverdefined()) {
      DEBUG(DOUT << "markOverdefined: ";
            if (Function *F = dyn_cast<Function>(V))
              DOUT << "Function '" << F->getName() << "'\n";
            else
              DOUT << *V);
      // Only instructions go on the work list
      OverdefinedInstWorkList.push_back(V);
    }
  }

  inline void mergeInValue(LatticeVal &IV, Value *V, LatticeVal &MergeWithV) {
    if (IV.isOverdefined() || MergeWithV.isUndefined())
      return;  // Noop.
    if (MergeWithV.isOverdefined())
      markOverdefined(IV, V);
    else if (IV.isUndefined())
      markConstant(IV, V, MergeWithV.getConstant());
    else if (IV.getConstant() != MergeWithV.getConstant())
      markOverdefined(IV, V);
  }
  
  inline void mergeInValue(Value *V, LatticeVal &MergeWithV) {
    return mergeInValue(ValueState[V], V, MergeWithV);
  }


  // getValueState - Return the LatticeVal object that corresponds to the value.
  // This function is necessary because not all values should start out in the
  // underdefined state... Argument's should be overdefined, and
  // constants should be marked as constants.  If a value is not known to be an
  // Instruction object, then use this accessor to get its value from the map.
  //
  inline LatticeVal &getValueState(Value *V) {
    std::map<Value*, LatticeVal>::iterator I = ValueState.find(V);
    if (I != ValueState.end()) return I->second;  // Common case, in the map

    if (Constant *C = dyn_cast<Constant>(V)) {
      if (isa<UndefValue>(V)) {
        // Nothing to do, remain undefined.
      } else {
        LatticeVal &LV = ValueState[C];
        LV.markConstant(C);          // Constants are constant
        return LV;
      }
    }
    // All others are underdefined by default...
    return ValueState[V];
  }

  // markEdgeExecutable - Mark a basic block as executable, adding it to the BB
  // work list if it is not already executable...
  //
  void markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest) {
    if (!KnownFeasibleEdges.insert(Edge(Source, Dest)).second)
      return;  // This edge is already known to be executable!

    if (BBExecutable.count(Dest)) {
      DOUT << "Marking Edge Executable: " << Source->getName()
           << " -> " << Dest->getName() << "\n";

      // The destination is already executable, but we just made an edge
      // feasible that wasn't before.  Revisit the PHI nodes in the block
      // because they have potentially new operands.
      for (BasicBlock::iterator I = Dest->begin(); isa<PHINode>(I); ++I)
        visitPHINode(*cast<PHINode>(I));

    } else {
      MarkBlockExecutable(Dest);
    }
  }

  // getFeasibleSuccessors - Return a vector of booleans to indicate which
  // successors are reachable from a given terminator instruction.
  //
  void getFeasibleSuccessors(TerminatorInst &TI, SmallVector<bool, 16> &Succs);

  // isEdgeFeasible - Return true if the control flow edge from the 'From' basic
  // block to the 'To' basic block is currently feasible...
  //
  bool isEdgeFeasible(BasicBlock *From, BasicBlock *To);

  // OperandChangedState - This method is invoked on all of the users of an
  // instruction that was just changed state somehow....  Based on this
  // information, we need to update the specified user of this instruction.
  //
  void OperandChangedState(User *U) {
    // Only instructions use other variable values!
    Instruction &I = cast<Instruction>(*U);
    if (BBExecutable.count(I.getParent()))   // Inst is executable?
      visit(I);
  }

private:
  friend class InstVisitor<SCCPSolver>;

  // visit implementations - Something changed in this instruction... Either an
  // operand made a transition, or the instruction is newly executable.  Change
  // the value type of I to reflect these changes if appropriate.
  //
  void visitPHINode(PHINode &I);

  // Terminators
  void visitReturnInst(ReturnInst &I);
  void visitTerminatorInst(TerminatorInst &TI);

  void visitCastInst(CastInst &I);
  void visitSelectInst(SelectInst &I);
  void visitBinaryOperator(Instruction &I);
  void visitCmpInst(CmpInst &I);
  void visitExtractElementInst(ExtractElementInst &I);
  void visitInsertElementInst(InsertElementInst &I);
  void visitShuffleVectorInst(ShuffleVectorInst &I);

  // Instructions that cannot be folded away...
  void visitStoreInst     (Instruction &I);
  void visitLoadInst      (LoadInst &I);
  void visitGetElementPtrInst(GetElementPtrInst &I);
  void visitCallInst      (CallInst &I) { visitCallSite(CallSite::get(&I)); }
  void visitInvokeInst    (InvokeInst &II) {
    visitCallSite(CallSite::get(&II));
    visitTerminatorInst(II);
  }
  void visitCallSite      (CallSite CS);
  void visitUnwindInst    (TerminatorInst &I) { /*returns void*/ }
  void visitUnreachableInst(TerminatorInst &I) { /*returns void*/ }
  void visitAllocationInst(Instruction &I) { markOverdefined(&I); }
  void visitVANextInst    (Instruction &I) { markOverdefined(&I); }
  void visitVAArgInst     (Instruction &I) { markOverdefined(&I); }
  void visitFreeInst      (Instruction &I) { /*returns void*/ }

  void visitInstruction(Instruction &I) {
    // If a new instruction is added to LLVM that we don't handle...
    cerr << "SCCP: Don't know how to handle: " << I;
    markOverdefined(&I);   // Just in case
  }
};

} // end anonymous namespace


// getFeasibleSuccessors - Return a vector of booleans to indicate which
// successors are reachable from a given terminator instruction.
//
void SCCPSolver::getFeasibleSuccessors(TerminatorInst &TI,
                                       SmallVector<bool, 16> &Succs) {
  Succs.resize(TI.getNumSuccessors());
  if (BranchInst *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
    } else {
      LatticeVal &BCValue = getValueState(BI->getCondition());
      if (BCValue.isOverdefined() ||
          (BCValue.isConstant() && !isa<ConstantInt>(BCValue.getConstant()))) {
        // Overdefined condition variables, and branches on unfoldable constant
        // conditions, mean the branch could go either way.
        Succs[0] = Succs[1] = true;
      } else if (BCValue.isConstant()) {
        // Constant condition variables mean the branch can only go a single way
        Succs[BCValue.getConstant() == ConstantInt::getFalse()] = true;
      }
    }
  } else if (isa<InvokeInst>(&TI)) {
    // Invoke instructions successors are always executable.
    Succs[0] = Succs[1] = true;
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(&TI)) {
    LatticeVal &SCValue = getValueState(SI->getCondition());
    if (SCValue.isOverdefined() ||   // Overdefined condition?
        (SCValue.isConstant() && !isa<ConstantInt>(SCValue.getConstant()))) {
      // All destinations are executable!
      Succs.assign(TI.getNumSuccessors(), true);
    } else if (SCValue.isConstant()) {
      Constant *CPV = SCValue.getConstant();
      // Make sure to skip the "default value" which isn't a value
      for (unsigned i = 1, E = SI->getNumSuccessors(); i != E; ++i) {
        if (SI->getSuccessorValue(i) == CPV) {// Found the right branch...
          Succs[i] = true;
          return;
        }
      }

      // Constant value not equal to any of the branches... must execute
      // default branch then...
      Succs[0] = true;
    }
  } else {
    assert(0 && "SCCP: Don't know how to handle this terminator!");
  }
}


// isEdgeFeasible - Return true if the control flow edge from the 'From' basic
// block to the 'To' basic block is currently feasible...
//
bool SCCPSolver::isEdgeFeasible(BasicBlock *From, BasicBlock *To) {
  assert(BBExecutable.count(To) && "Dest should always be alive!");

  // Make sure the source basic block is executable!!
  if (!BBExecutable.count(From)) return false;

  // Check to make sure this edge itself is actually feasible now...
  TerminatorInst *TI = From->getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isUnconditional())
      return true;
    else {
      LatticeVal &BCValue = getValueState(BI->getCondition());
      if (BCValue.isOverdefined()) {
        // Overdefined condition variables mean the branch could go either way.
        return true;
      } else if (BCValue.isConstant()) {
        // Not branching on an evaluatable constant?
        if (!isa<ConstantInt>(BCValue.getConstant())) return true;

        // Constant condition variables mean the branch can only go a single way
        return BI->getSuccessor(BCValue.getConstant() ==
                                       ConstantInt::getFalse()) == To;
      }
      return false;
    }
  } else if (isa<InvokeInst>(TI)) {
    // Invoke instructions successors are always executable.
    return true;
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    LatticeVal &SCValue = getValueState(SI->getCondition());
    if (SCValue.isOverdefined()) {  // Overdefined condition?
      // All destinations are executable!
      return true;
    } else if (SCValue.isConstant()) {
      Constant *CPV = SCValue.getConstant();
      if (!isa<ConstantInt>(CPV))
        return true;  // not a foldable constant?

      // Make sure to skip the "default value" which isn't a value
      for (unsigned i = 1, E = SI->getNumSuccessors(); i != E; ++i)
        if (SI->getSuccessorValue(i) == CPV) // Found the taken branch...
          return SI->getSuccessor(i) == To;

      // Constant value not equal to any of the branches... must execute
      // default branch then...
      return SI->getDefaultDest() == To;
    }
    return false;
  } else {
    cerr << "Unknown terminator instruction: " << *TI;
    abort();
  }
}

// visit Implementations - Something changed in this instruction... Either an
// operand made a transition, or the instruction is newly executable.  Change
// the value type of I to reflect these changes if appropriate.  This method
// makes sure to do the following actions:
//
// 1. If a phi node merges two constants in, and has conflicting value coming
//    from different branches, or if the PHI node merges in an overdefined
//    value, then the PHI node becomes overdefined.
// 2. If a phi node merges only constants in, and they all agree on value, the
//    PHI node becomes a constant value equal to that.
// 3. If V <- x (op) y && isConstant(x) && isConstant(y) V = Constant
// 4. If V <- x (op) y && (isOverdefined(x) || isOverdefined(y)) V = Overdefined
// 5. If V <- MEM or V <- CALL or V <- (unknown) then V = Overdefined
// 6. If a conditional branch has a value that is constant, make the selected
//    destination executable
// 7. If a conditional branch has a value that is overdefined, make all
//    successors executable.
//
void SCCPSolver::visitPHINode(PHINode &PN) {
  LatticeVal &PNIV = getValueState(&PN);
  if (PNIV.isOverdefined()) {
    // There may be instructions using this PHI node that are not overdefined
    // themselves.  If so, make sure that they know that the PHI node operand
    // changed.
    std::multimap<PHINode*, Instruction*>::iterator I, E;
    tie(I, E) = UsersOfOverdefinedPHIs.equal_range(&PN);
    if (I != E) {
      SmallVector<Instruction*, 16> Users;
      for (; I != E; ++I) Users.push_back(I->second);
      while (!Users.empty()) {
        visit(Users.back());
        Users.pop_back();
      }
    }
    return;  // Quick exit
  }

  // Super-extra-high-degree PHI nodes are unlikely to ever be marked constant,
  // and slow us down a lot.  Just mark them overdefined.
  if (PN.getNumIncomingValues() > 64) {
    markOverdefined(PNIV, &PN);
    return;
  }

  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  If they are all
  // constant, and they agree with each other, the PHI becomes the identical
  // constant.  If they are constant and don't agree, the PHI is overdefined.
  // If there are no executable operands, the PHI remains undefined.
  //
  Constant *OperandVal = 0;
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    LatticeVal &IV = getValueState(PN.getIncomingValue(i));
    if (IV.isUndefined()) continue;  // Doesn't influence PHI node.

    if (isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent())) {
      if (IV.isOverdefined()) {   // PHI node becomes overdefined!
        markOverdefined(PNIV, &PN);
        return;
      }

      if (OperandVal == 0) {   // Grab the first value...
        OperandVal = IV.getConstant();
      } else {                // Another value is being merged in!
        // There is already a reachable operand.  If we conflict with it,
        // then the PHI node becomes overdefined.  If we agree with it, we
        // can continue on.

        // Check to see if there are two different constants merging...
        if (IV.getConstant() != OperandVal) {
          // Yes there is.  This means the PHI node is not constant.
          // You must be overdefined poor PHI.
          //
          markOverdefined(PNIV, &PN);    // The PHI node now becomes overdefined
          return;    // I'm done analyzing you
        }
      }
    }
  }

  // If we exited the loop, this means that the PHI node only has constant
  // arguments that agree with each other(and OperandVal is the constant) or
  // OperandVal is null because there are no defined incoming arguments.  If
  // this is the case, the PHI remains undefined.
  //
  if (OperandVal)
    markConstant(PNIV, &PN, OperandVal);      // Acquire operand value
}

void SCCPSolver::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() == 0) return;  // Ret void

  // If we are tracking the return value of this function, merge it in.
  Function *F = I.getParent()->getParent();
  if (F->hasInternalLinkage() && !TrackedFunctionRetVals.empty()) {
    DenseMap<Function*, LatticeVal>::iterator TFRVI =
      TrackedFunctionRetVals.find(F);
    if (TFRVI != TrackedFunctionRetVals.end() &&
        !TFRVI->second.isOverdefined()) {
      LatticeVal &IV = getValueState(I.getOperand(0));
      mergeInValue(TFRVI->second, F, IV);
    }
  }
}


void SCCPSolver::visitTerminatorInst(TerminatorInst &TI) {
  SmallVector<bool, 16> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible);

  BasicBlock *BB = TI.getParent();

  // Mark all feasible successors executable...
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

void SCCPSolver::visitCastInst(CastInst &I) {
  Value *V = I.getOperand(0);
  LatticeVal &VState = getValueState(V);
  if (VState.isOverdefined())          // Inherit overdefinedness of operand
    markOverdefined(&I);
  else if (VState.isConstant())        // Propagate constant value
    markConstant(&I, ConstantExpr::getCast(I.getOpcode(), 
                                           VState.getConstant(), I.getType()));
}

void SCCPSolver::visitSelectInst(SelectInst &I) {
  LatticeVal &CondValue = getValueState(I.getCondition());
  if (CondValue.isUndefined())
    return;
  if (CondValue.isConstant()) {
    if (ConstantInt *CondCB = dyn_cast<ConstantInt>(CondValue.getConstant())){
      mergeInValue(&I, getValueState(CondCB->getZExtValue() ? I.getTrueValue()
                                                          : I.getFalseValue()));
      return;
    }
  }
  
  // Otherwise, the condition is overdefined or a constant we can't evaluate.
  // See if we can produce something better than overdefined based on the T/F
  // value.
  LatticeVal &TVal = getValueState(I.getTrueValue());
  LatticeVal &FVal = getValueState(I.getFalseValue());
  
  // select ?, C, C -> C.
  if (TVal.isConstant() && FVal.isConstant() && 
      TVal.getConstant() == FVal.getConstant()) {
    markConstant(&I, FVal.getConstant());
    return;
  }

  if (TVal.isUndefined()) {  // select ?, undef, X -> X.
    mergeInValue(&I, FVal);
  } else if (FVal.isUndefined()) {  // select ?, X, undef -> X.
    mergeInValue(&I, TVal);
  } else {
    markOverdefined(&I);
  }
}

// Handle BinaryOperators and Shift Instructions...
void SCCPSolver::visitBinaryOperator(Instruction &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  LatticeVal &V1State = getValueState(I.getOperand(0));
  LatticeVal &V2State = getValueState(I.getOperand(1));

  if (V1State.isOverdefined() || V2State.isOverdefined()) {
    // If this is an AND or OR with 0 or -1, it doesn't matter that the other
    // operand is overdefined.
    if (I.getOpcode() == Instruction::And || I.getOpcode() == Instruction::Or) {
      LatticeVal *NonOverdefVal = 0;
      if (!V1State.isOverdefined()) {
        NonOverdefVal = &V1State;
      } else if (!V2State.isOverdefined()) {
        NonOverdefVal = &V2State;
      }

      if (NonOverdefVal) {
        if (NonOverdefVal->isUndefined()) {
          // Could annihilate value.
          if (I.getOpcode() == Instruction::And)
            markConstant(IV, &I, Constant::getNullValue(I.getType()));
          else if (const VectorType *PT = dyn_cast<VectorType>(I.getType()))
            markConstant(IV, &I, ConstantVector::getAllOnesValue(PT));
          else
            markConstant(IV, &I, ConstantInt::getAllOnesValue(I.getType()));
          return;
        } else {
          if (I.getOpcode() == Instruction::And) {
            if (NonOverdefVal->getConstant()->isNullValue()) {
              markConstant(IV, &I, NonOverdefVal->getConstant());
              return;      // X and 0 = 0
            }
          } else {
            if (ConstantInt *CI =
                     dyn_cast<ConstantInt>(NonOverdefVal->getConstant()))
              if (CI->isAllOnesValue()) {
                markConstant(IV, &I, NonOverdefVal->getConstant());
                return;    // X or -1 = -1
              }
          }
        }
      }
    }


    // If both operands are PHI nodes, it is possible that this instruction has
    // a constant value, despite the fact that the PHI node doesn't.  Check for
    // this condition now.
    if (PHINode *PN1 = dyn_cast<PHINode>(I.getOperand(0)))
      if (PHINode *PN2 = dyn_cast<PHINode>(I.getOperand(1)))
        if (PN1->getParent() == PN2->getParent()) {
          // Since the two PHI nodes are in the same basic block, they must have
          // entries for the same predecessors.  Walk the predecessor list, and
          // if all of the incoming values are constants, and the result of
          // evaluating this expression with all incoming value pairs is the
          // same, then this expression is a constant even though the PHI node
          // is not a constant!
          LatticeVal Result;
          for (unsigned i = 0, e = PN1->getNumIncomingValues(); i != e; ++i) {
            LatticeVal &In1 = getValueState(PN1->getIncomingValue(i));
            BasicBlock *InBlock = PN1->getIncomingBlock(i);
            LatticeVal &In2 =
              getValueState(PN2->getIncomingValueForBlock(InBlock));

            if (In1.isOverdefined() || In2.isOverdefined()) {
              Result.markOverdefined();
              break;  // Cannot fold this operation over the PHI nodes!
            } else if (In1.isConstant() && In2.isConstant()) {
              Constant *V = ConstantExpr::get(I.getOpcode(), In1.getConstant(),
                                              In2.getConstant());
              if (Result.isUndefined())
                Result.markConstant(V);
              else if (Result.isConstant() && Result.getConstant() != V) {
                Result.markOverdefined();
                break;
              }
            }
          }

          // If we found a constant value here, then we know the instruction is
          // constant despite the fact that the PHI nodes are overdefined.
          if (Result.isConstant()) {
            markConstant(IV, &I, Result.getConstant());
            // Remember that this instruction is virtually using the PHI node
            // operands.
            UsersOfOverdefinedPHIs.insert(std::make_pair(PN1, &I));
            UsersOfOverdefinedPHIs.insert(std::make_pair(PN2, &I));
            return;
          } else if (Result.isUndefined()) {
            return;
          }

          // Okay, this really is overdefined now.  Since we might have
          // speculatively thought that this was not overdefined before, and
          // added ourselves to the UsersOfOverdefinedPHIs list for the PHIs,
          // make sure to clean out any entries that we put there, for
          // efficiency.
          std::multimap<PHINode*, Instruction*>::iterator It, E;
          tie(It, E) = UsersOfOverdefinedPHIs.equal_range(PN1);
          while (It != E) {
            if (It->second == &I) {
              UsersOfOverdefinedPHIs.erase(It++);
            } else
              ++It;
          }
          tie(It, E) = UsersOfOverdefinedPHIs.equal_range(PN2);
          while (It != E) {
            if (It->second == &I) {
              UsersOfOverdefinedPHIs.erase(It++);
            } else
              ++It;
          }
        }

    markOverdefined(IV, &I);
  } else if (V1State.isConstant() && V2State.isConstant()) {
    markConstant(IV, &I, ConstantExpr::get(I.getOpcode(), V1State.getConstant(),
                                           V2State.getConstant()));
  }
}

// Handle ICmpInst instruction...
void SCCPSolver::visitCmpInst(CmpInst &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  LatticeVal &V1State = getValueState(I.getOperand(0));
  LatticeVal &V2State = getValueState(I.getOperand(1));

  if (V1State.isOverdefined() || V2State.isOverdefined()) {
    // If both operands are PHI nodes, it is possible that this instruction has
    // a constant value, despite the fact that the PHI node doesn't.  Check for
    // this condition now.
    if (PHINode *PN1 = dyn_cast<PHINode>(I.getOperand(0)))
      if (PHINode *PN2 = dyn_cast<PHINode>(I.getOperand(1)))
        if (PN1->getParent() == PN2->getParent()) {
          // Since the two PHI nodes are in the same basic block, they must have
          // entries for the same predecessors.  Walk the predecessor list, and
          // if all of the incoming values are constants, and the result of
          // evaluating this expression with all incoming value pairs is the
          // same, then this expression is a constant even though the PHI node
          // is not a constant!
          LatticeVal Result;
          for (unsigned i = 0, e = PN1->getNumIncomingValues(); i != e; ++i) {
            LatticeVal &In1 = getValueState(PN1->getIncomingValue(i));
            BasicBlock *InBlock = PN1->getIncomingBlock(i);
            LatticeVal &In2 =
              getValueState(PN2->getIncomingValueForBlock(InBlock));

            if (In1.isOverdefined() || In2.isOverdefined()) {
              Result.markOverdefined();
              break;  // Cannot fold this operation over the PHI nodes!
            } else if (In1.isConstant() && In2.isConstant()) {
              Constant *V = ConstantExpr::getCompare(I.getPredicate(), 
                                                     In1.getConstant(), 
                                                     In2.getConstant());
              if (Result.isUndefined())
                Result.markConstant(V);
              else if (Result.isConstant() && Result.getConstant() != V) {
                Result.markOverdefined();
                break;
              }
            }
          }

          // If we found a constant value here, then we know the instruction is
          // constant despite the fact that the PHI nodes are overdefined.
          if (Result.isConstant()) {
            markConstant(IV, &I, Result.getConstant());
            // Remember that this instruction is virtually using the PHI node
            // operands.
            UsersOfOverdefinedPHIs.insert(std::make_pair(PN1, &I));
            UsersOfOverdefinedPHIs.insert(std::make_pair(PN2, &I));
            return;
          } else if (Result.isUndefined()) {
            return;
          }

          // Okay, this really is overdefined now.  Since we might have
          // speculatively thought that this was not overdefined before, and
          // added ourselves to the UsersOfOverdefinedPHIs list for the PHIs,
          // make sure to clean out any entries that we put there, for
          // efficiency.
          std::multimap<PHINode*, Instruction*>::iterator It, E;
          tie(It, E) = UsersOfOverdefinedPHIs.equal_range(PN1);
          while (It != E) {
            if (It->second == &I) {
              UsersOfOverdefinedPHIs.erase(It++);
            } else
              ++It;
          }
          tie(It, E) = UsersOfOverdefinedPHIs.equal_range(PN2);
          while (It != E) {
            if (It->second == &I) {
              UsersOfOverdefinedPHIs.erase(It++);
            } else
              ++It;
          }
        }

    markOverdefined(IV, &I);
  } else if (V1State.isConstant() && V2State.isConstant()) {
    markConstant(IV, &I, ConstantExpr::getCompare(I.getPredicate(), 
                                                  V1State.getConstant(), 
                                                  V2State.getConstant()));
  }
}

void SCCPSolver::visitExtractElementInst(ExtractElementInst &I) {
  // FIXME : SCCP does not handle vectors properly.
  markOverdefined(&I);
  return;

#if 0
  LatticeVal &ValState = getValueState(I.getOperand(0));
  LatticeVal &IdxState = getValueState(I.getOperand(1));

  if (ValState.isOverdefined() || IdxState.isOverdefined())
    markOverdefined(&I);
  else if(ValState.isConstant() && IdxState.isConstant())
    markConstant(&I, ConstantExpr::getExtractElement(ValState.getConstant(),
                                                     IdxState.getConstant()));
#endif
}

void SCCPSolver::visitInsertElementInst(InsertElementInst &I) {
  // FIXME : SCCP does not handle vectors properly.
  markOverdefined(&I);
  return;
#if 0
  LatticeVal &ValState = getValueState(I.getOperand(0));
  LatticeVal &EltState = getValueState(I.getOperand(1));
  LatticeVal &IdxState = getValueState(I.getOperand(2));

  if (ValState.isOverdefined() || EltState.isOverdefined() ||
      IdxState.isOverdefined())
    markOverdefined(&I);
  else if(ValState.isConstant() && EltState.isConstant() &&
          IdxState.isConstant())
    markConstant(&I, ConstantExpr::getInsertElement(ValState.getConstant(),
                                                    EltState.getConstant(),
                                                    IdxState.getConstant()));
  else if (ValState.isUndefined() && EltState.isConstant() &&
           IdxState.isConstant()) 
    markConstant(&I,ConstantExpr::getInsertElement(UndefValue::get(I.getType()),
                                                   EltState.getConstant(),
                                                   IdxState.getConstant()));
#endif
}

void SCCPSolver::visitShuffleVectorInst(ShuffleVectorInst &I) {
  // FIXME : SCCP does not handle vectors properly.
  markOverdefined(&I);
  return;
#if 0
  LatticeVal &V1State   = getValueState(I.getOperand(0));
  LatticeVal &V2State   = getValueState(I.getOperand(1));
  LatticeVal &MaskState = getValueState(I.getOperand(2));

  if (MaskState.isUndefined() ||
      (V1State.isUndefined() && V2State.isUndefined()))
    return;  // Undefined output if mask or both inputs undefined.
  
  if (V1State.isOverdefined() || V2State.isOverdefined() ||
      MaskState.isOverdefined()) {
    markOverdefined(&I);
  } else {
    // A mix of constant/undef inputs.
    Constant *V1 = V1State.isConstant() ? 
        V1State.getConstant() : UndefValue::get(I.getType());
    Constant *V2 = V2State.isConstant() ? 
        V2State.getConstant() : UndefValue::get(I.getType());
    Constant *Mask = MaskState.isConstant() ? 
      MaskState.getConstant() : UndefValue::get(I.getOperand(2)->getType());
    markConstant(&I, ConstantExpr::getShuffleVector(V1, V2, Mask));
  }
#endif
}

// Handle getelementptr instructions... if all operands are constants then we
// can turn this into a getelementptr ConstantExpr.
//
void SCCPSolver::visitGetElementPtrInst(GetElementPtrInst &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  SmallVector<Constant*, 8> Operands;
  Operands.reserve(I.getNumOperands());

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    LatticeVal &State = getValueState(I.getOperand(i));
    if (State.isUndefined())
      return;  // Operands are not resolved yet...
    else if (State.isOverdefined()) {
      markOverdefined(IV, &I);
      return;
    }
    assert(State.isConstant() && "Unknown state!");
    Operands.push_back(State.getConstant());
  }

  Constant *Ptr = Operands[0];
  Operands.erase(Operands.begin());  // Erase the pointer from idx list...

  markConstant(IV, &I, ConstantExpr::getGetElementPtr(Ptr, &Operands[0],
                                                      Operands.size()));
}

void SCCPSolver::visitStoreInst(Instruction &SI) {
  if (TrackedGlobals.empty() || !isa<GlobalVariable>(SI.getOperand(1)))
    return;
  GlobalVariable *GV = cast<GlobalVariable>(SI.getOperand(1));
  DenseMap<GlobalVariable*, LatticeVal>::iterator I = TrackedGlobals.find(GV);
  if (I == TrackedGlobals.end() || I->second.isOverdefined()) return;

  // Get the value we are storing into the global.
  LatticeVal &PtrVal = getValueState(SI.getOperand(0));

  mergeInValue(I->second, GV, PtrVal);
  if (I->second.isOverdefined())
    TrackedGlobals.erase(I);      // No need to keep tracking this!
}


// Handle load instructions.  If the operand is a constant pointer to a constant
// global, we can replace the load with the loaded constant value!
void SCCPSolver::visitLoadInst(LoadInst &I) {
  LatticeVal &IV = ValueState[&I];
  if (IV.isOverdefined()) return;

  LatticeVal &PtrVal = getValueState(I.getOperand(0));
  if (PtrVal.isUndefined()) return;   // The pointer is not resolved yet!
  if (PtrVal.isConstant() && !I.isVolatile()) {
    Value *Ptr = PtrVal.getConstant();
    // TODO: Consider a target hook for valid address spaces for this xform.
    if (isa<ConstantPointerNull>(Ptr) && 
        cast<PointerType>(Ptr->getType())->getAddressSpace() == 0) {
      // load null -> null
      markConstant(IV, &I, Constant::getNullValue(I.getType()));
      return;
    }

    // Transform load (constant global) into the value loaded.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
      if (GV->isConstant()) {
        if (!GV->isDeclaration()) {
          markConstant(IV, &I, GV->getInitializer());
          return;
        }
      } else if (!TrackedGlobals.empty()) {
        // If we are tracking this global, merge in the known value for it.
        DenseMap<GlobalVariable*, LatticeVal>::iterator It =
          TrackedGlobals.find(GV);
        if (It != TrackedGlobals.end()) {
          mergeInValue(IV, &I, It->second);
          return;
        }
      }
    }

    // Transform load (constantexpr_GEP global, 0, ...) into the value loaded.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
      if (CE->getOpcode() == Instruction::GetElementPtr)
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
      if (GV->isConstant() && !GV->isDeclaration())
        if (Constant *V =
             ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE)) {
          markConstant(IV, &I, V);
          return;
        }
  }

  // Otherwise we cannot say for certain what value this load will produce.
  // Bail out.
  markOverdefined(IV, &I);
}

void SCCPSolver::visitCallSite(CallSite CS) {
  Function *F = CS.getCalledFunction();

  // If we are tracking this function, we must make sure to bind arguments as
  // appropriate.
  DenseMap<Function*, LatticeVal>::iterator TFRVI =TrackedFunctionRetVals.end();
  if (F && F->hasInternalLinkage())
    TFRVI = TrackedFunctionRetVals.find(F);

  if (TFRVI != TrackedFunctionRetVals.end()) {
    // If this is the first call to the function hit, mark its entry block
    // executable.
    if (!BBExecutable.count(F->begin()))
      MarkBlockExecutable(F->begin());

    CallSite::arg_iterator CAI = CS.arg_begin();
    for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
         AI != E; ++AI, ++CAI) {
      LatticeVal &IV = ValueState[AI];
      if (!IV.isOverdefined())
        mergeInValue(IV, AI, getValueState(*CAI));
    }
  }
  Instruction *I = CS.getInstruction();
  if (I->getType() == Type::VoidTy) return;

  LatticeVal &IV = ValueState[I];
  if (IV.isOverdefined()) return;

  // Propagate the return value of the function to the value of the instruction.
  if (TFRVI != TrackedFunctionRetVals.end()) {
    mergeInValue(IV, I, TFRVI->second);
    return;
  }

  if (F == 0 || !F->isDeclaration() || !canConstantFoldCallTo(F)) {
    markOverdefined(IV, I);
    return;
  }

  SmallVector<Constant*, 8> Operands;
  Operands.reserve(I->getNumOperands()-1);

  for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
       AI != E; ++AI) {
    LatticeVal &State = getValueState(*AI);
    if (State.isUndefined())
      return;  // Operands are not resolved yet...
    else if (State.isOverdefined()) {
      markOverdefined(IV, I);
      return;
    }
    assert(State.isConstant() && "Unknown state!");
    Operands.push_back(State.getConstant());
  }

  if (Constant *C = ConstantFoldCall(F, &Operands[0], Operands.size()))
    markConstant(IV, I, C);
  else
    markOverdefined(IV, I);
}


void SCCPSolver::Solve() {
  // Process the work lists until they are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty() ||
         !OverdefinedInstWorkList.empty()) {
    // Process the instruction work list...
    while (!OverdefinedInstWorkList.empty()) {
      Value *I = OverdefinedInstWorkList.back();
      OverdefinedInstWorkList.pop_back();

      DOUT << "\nPopped off OI-WL: " << *I;

      // "I" got into the work list because it either made the transition from
      // bottom to constant
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined
      // Update all of the users of this instruction's value...
      //
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
           UI != E; ++UI)
        OperandChangedState(*UI);
    }
    // Process the instruction work list...
    while (!InstWorkList.empty()) {
      Value *I = InstWorkList.back();
      InstWorkList.pop_back();

      DOUT << "\nPopped off I-WL: " << *I;

      // "I" got into the work list because it either made the transition from
      // bottom to constant
      //
      // Anything on this worklist that is overdefined need not be visited
      // since all of its users will have already been marked as overdefined.
      // Update all of the users of this instruction's value...
      //
      if (!getValueState(I).isOverdefined())
        for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
             UI != E; ++UI)
          OperandChangedState(*UI);
    }

    // Process the basic block work list...
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      DOUT << "\nPopped off BBWL: " << *BB;

      // Notify all instructions in this basic block that they are newly
      // executable.
      visit(BB);
    }
  }
}

/// ResolvedUndefsIn - While solving the dataflow for a function, we assume
/// that branches on undef values cannot reach any of their successors.
/// However, this is not a safe assumption.  After we solve dataflow, this
/// method should be use to handle this.  If this returns true, the solver
/// should be rerun.
///
/// This method handles this by finding an unresolved branch and marking it one
/// of the edges from the block as being feasible, even though the condition
/// doesn't say it would otherwise be.  This allows SCCP to find the rest of the
/// CFG and only slightly pessimizes the analysis results (by marking one,
/// potentially infeasible, edge feasible).  This cannot usefully modify the
/// constraints on the condition of the branch, as that would impact other users
/// of the value.
///
/// This scan also checks for values that use undefs, whose results are actually
/// defined.  For example, 'zext i8 undef to i32' should produce all zeros
/// conservatively, as "(zext i8 X -> i32) & 0xFF00" must always return zero,
/// even if X isn't defined.
bool SCCPSolver::ResolvedUndefsIn(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (!BBExecutable.count(BB))
      continue;
    
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // Look for instructions which produce undef values.
      if (I->getType() == Type::VoidTy) continue;
      
      LatticeVal &LV = getValueState(I);
      if (!LV.isUndefined()) continue;

      // Get the lattice values of the first two operands for use below.
      LatticeVal &Op0LV = getValueState(I->getOperand(0));
      LatticeVal Op1LV;
      if (I->getNumOperands() == 2) {
        // If this is a two-operand instruction, and if both operands are
        // undefs, the result stays undef.
        Op1LV = getValueState(I->getOperand(1));
        if (Op0LV.isUndefined() && Op1LV.isUndefined())
          continue;
      }
      
      // If this is an instructions whose result is defined even if the input is
      // not fully defined, propagate the information.
      const Type *ITy = I->getType();
      switch (I->getOpcode()) {
      default: break;          // Leave the instruction as an undef.
      case Instruction::ZExt:
        // After a zero extend, we know the top part is zero.  SExt doesn't have
        // to be handled here, because we don't know whether the top part is 1's
        // or 0's.
        assert(Op0LV.isUndefined());
        markForcedConstant(LV, I, Constant::getNullValue(ITy));
        return true;
      case Instruction::Mul:
      case Instruction::And:
        // undef * X -> 0.   X could be zero.
        // undef & X -> 0.   X could be zero.
        markForcedConstant(LV, I, Constant::getNullValue(ITy));
        return true;

      case Instruction::Or:
        // undef | X -> -1.   X could be -1.
        if (const VectorType *PTy = dyn_cast<VectorType>(ITy))
          markForcedConstant(LV, I, ConstantVector::getAllOnesValue(PTy));
        else          
          markForcedConstant(LV, I, ConstantInt::getAllOnesValue(ITy));
        return true;

      case Instruction::SDiv:
      case Instruction::UDiv:
      case Instruction::SRem:
      case Instruction::URem:
        // X / undef -> undef.  No change.
        // X % undef -> undef.  No change.
        if (Op1LV.isUndefined()) break;
        
        // undef / X -> 0.   X could be maxint.
        // undef % X -> 0.   X could be 1.
        markForcedConstant(LV, I, Constant::getNullValue(ITy));
        return true;
        
      case Instruction::AShr:
        // undef >>s X -> undef.  No change.
        if (Op0LV.isUndefined()) break;
        
        // X >>s undef -> X.  X could be 0, X could have the high-bit known set.
        if (Op0LV.isConstant())
          markForcedConstant(LV, I, Op0LV.getConstant());
        else
          markOverdefined(LV, I);
        return true;
      case Instruction::LShr:
      case Instruction::Shl:
        // undef >> X -> undef.  No change.
        // undef << X -> undef.  No change.
        if (Op0LV.isUndefined()) break;
        
        // X >> undef -> 0.  X could be 0.
        // X << undef -> 0.  X could be 0.
        markForcedConstant(LV, I, Constant::getNullValue(ITy));
        return true;
      case Instruction::Select:
        // undef ? X : Y  -> X or Y.  There could be commonality between X/Y.
        if (Op0LV.isUndefined()) {
          if (!Op1LV.isConstant())  // Pick the constant one if there is any.
            Op1LV = getValueState(I->getOperand(2));
        } else if (Op1LV.isUndefined()) {
          // c ? undef : undef -> undef.  No change.
          Op1LV = getValueState(I->getOperand(2));
          if (Op1LV.isUndefined())
            break;
          // Otherwise, c ? undef : x -> x.
        } else {
          // Leave Op1LV as Operand(1)'s LatticeValue.
        }
        
        if (Op1LV.isConstant())
          markForcedConstant(LV, I, Op1LV.getConstant());
        else
          markOverdefined(LV, I);
        return true;
      }
    }
  
    TerminatorInst *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (!BI->isConditional()) continue;
      if (!getValueState(BI->getCondition()).isUndefined())
        continue;
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (!getValueState(SI->getCondition()).isUndefined())
        continue;
    } else {
      continue;
    }
    
    // If the edge to the first successor isn't thought to be feasible yet, mark
    // it so now.
    if (KnownFeasibleEdges.count(Edge(BB, TI->getSuccessor(0))))
      continue;
    
    // Otherwise, it isn't already thought to be feasible.  Mark it as such now
    // and return.  This will make other blocks reachable, which will allow new
    // values to be discovered and existing ones to be moved in the lattice.
    markEdgeExecutable(BB, TI->getSuccessor(0));
    return true;
  }

  return false;
}


namespace {
  //===--------------------------------------------------------------------===//
  //
  /// SCCP Class - This class uses the SCCPSolver to implement a per-function
  /// Sparse Conditional Constant Propagator.
  ///
  struct VISIBILITY_HIDDEN SCCP : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    SCCP() : FunctionPass((intptr_t)&ID) {}

    // runOnFunction - Run the Sparse Conditional Constant Propagation
    // algorithm, and return true if the function was modified.
    //
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
    }
  };

  char SCCP::ID = 0;
  RegisterPass<SCCP> X("sccp", "Sparse Conditional Constant Propagation");
} // end anonymous namespace


// createSCCPPass - This is the public interface to this file...
FunctionPass *llvm::createSCCPPass() {
  return new SCCP();
}


// runOnFunction() - Run the Sparse Conditional Constant Propagation algorithm,
// and return true if the function was modified.
//
bool SCCP::runOnFunction(Function &F) {
  DOUT << "SCCP on function '" << F.getName() << "'\n";
  SCCPSolver Solver;

  // Mark the first block of the function as being executable.
  Solver.MarkBlockExecutable(F.begin());

  // Mark all arguments to the function as being overdefined.
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end(); AI != E;++AI)
    Solver.markOverdefined(AI);

  // Solve for constants.
  bool ResolvedUndefs = true;
  while (ResolvedUndefs) {
    Solver.Solve();
    DOUT << "RESOLVING UNDEFs\n";
    ResolvedUndefs = Solver.ResolvedUndefsIn(F);
  }

  bool MadeChanges = false;

  // If we decided that there are basic blocks that are dead in this function,
  // delete their contents now.  Note that we cannot actually delete the blocks,
  // as we cannot modify the CFG of the function.
  //
  SmallSet<BasicBlock*, 16> &ExecutableBBs = Solver.getExecutableBlocks();
  SmallVector<Instruction*, 32> Insts;
  std::map<Value*, LatticeVal> &Values = Solver.getValueMapping();

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (!ExecutableBBs.count(BB)) {
      DOUT << "  BasicBlock Dead:" << *BB;
      ++NumDeadBlocks;

      // Delete the instructions backwards, as it has a reduced likelihood of
      // having to update as many def-use and use-def chains.
      for (BasicBlock::iterator I = BB->begin(), E = BB->getTerminator();
           I != E; ++I)
        Insts.push_back(I);
      while (!Insts.empty()) {
        Instruction *I = Insts.back();
        Insts.pop_back();
        if (!I->use_empty())
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        BB->getInstList().erase(I);
        MadeChanges = true;
        ++NumInstRemoved;
      }
    } else {
      // Iterate over all of the instructions in a function, replacing them with
      // constants if we have found them to be of constant values.
      //
      for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ) {
        Instruction *Inst = BI++;
        if (Inst->getType() != Type::VoidTy) {
          LatticeVal &IV = Values[Inst];
          if ((IV.isConstant() || IV.isUndefined()) &&
              !isa<TerminatorInst>(Inst)) {
            Constant *Const = IV.isConstant()
              ? IV.getConstant() : UndefValue::get(Inst->getType());
            DOUT << "  Constant: " << *Const << " = " << *Inst;

            // Replaces all of the uses of a variable with uses of the constant.
            Inst->replaceAllUsesWith(Const);

            // Delete the instruction.
            BB->getInstList().erase(Inst);

            // Hey, we just changed something!
            MadeChanges = true;
            ++NumInstRemoved;
          }
        }
      }
    }

  return MadeChanges;
}

namespace {
  //===--------------------------------------------------------------------===//
  //
  /// IPSCCP Class - This class implements interprocedural Sparse Conditional
  /// Constant Propagation.
  ///
  struct VISIBILITY_HIDDEN IPSCCP : public ModulePass {
    static char ID;
    IPSCCP() : ModulePass((intptr_t)&ID) {}
    bool runOnModule(Module &M);
  };

  char IPSCCP::ID = 0;
  RegisterPass<IPSCCP>
  Y("ipsccp", "Interprocedural Sparse Conditional Constant Propagation");
} // end anonymous namespace

// createIPSCCPPass - This is the public interface to this file...
ModulePass *llvm::createIPSCCPPass() {
  return new IPSCCP();
}


static bool AddressIsTaken(GlobalValue *GV) {
  // Delete any dead constantexpr klingons.
  GV->removeDeadConstantUsers();

  for (Value::use_iterator UI = GV->use_begin(), E = GV->use_end();
       UI != E; ++UI)
    if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (SI->getOperand(0) == GV || SI->isVolatile())
        return true;  // Storing addr of GV.
    } else if (isa<InvokeInst>(*UI) || isa<CallInst>(*UI)) {
      // Make sure we are calling the function, not passing the address.
      CallSite CS = CallSite::get(cast<Instruction>(*UI));
      for (CallSite::arg_iterator AI = CS.arg_begin(),
             E = CS.arg_end(); AI != E; ++AI)
        if (*AI == GV)
          return true;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      if (LI->isVolatile())
        return true;
    } else {
      return true;
    }
  return false;
}

bool IPSCCP::runOnModule(Module &M) {
  SCCPSolver Solver;

  // Loop over all functions, marking arguments to those with their addresses
  // taken or that are external as overdefined.
  //
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    if (!F->hasInternalLinkage() || AddressIsTaken(F)) {
      if (!F->isDeclaration())
        Solver.MarkBlockExecutable(F->begin());
      for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
           AI != E; ++AI)
        Solver.markOverdefined(AI);
    } else {
      Solver.AddTrackedFunction(F);
    }

  // Loop over global variables.  We inform the solver about any internal global
  // variables that do not have their 'addresses taken'.  If they don't have
  // their addresses taken, we can propagate constants through them.
  for (Module::global_iterator G = M.global_begin(), E = M.global_end();
       G != E; ++G)
    if (!G->isConstant() && G->hasInternalLinkage() && !AddressIsTaken(G))
      Solver.TrackValueOfGlobalVariable(G);

  // Solve for constants.
  bool ResolvedUndefs = true;
  while (ResolvedUndefs) {
    Solver.Solve();

    DOUT << "RESOLVING UNDEFS\n";
    ResolvedUndefs = false;
    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
      ResolvedUndefs |= Solver.ResolvedUndefsIn(*F);
  }

  bool MadeChanges = false;

  // Iterate over all of the instructions in the module, replacing them with
  // constants if we have found them to be of constant values.
  //
  SmallSet<BasicBlock*, 16> &ExecutableBBs = Solver.getExecutableBlocks();
  SmallVector<Instruction*, 32> Insts;
  SmallVector<BasicBlock*, 32> BlocksToErase;
  std::map<Value*, LatticeVal> &Values = Solver.getValueMapping();

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end();
         AI != E; ++AI)
      if (!AI->use_empty()) {
        LatticeVal &IV = Values[AI];
        if (IV.isConstant() || IV.isUndefined()) {
          Constant *CST = IV.isConstant() ?
            IV.getConstant() : UndefValue::get(AI->getType());
          DOUT << "***  Arg " << *AI << " = " << *CST <<"\n";

          // Replaces all of the uses of a variable with uses of the
          // constant.
          AI->replaceAllUsesWith(CST);
          ++IPNumArgsElimed;
        }
      }

    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      if (!ExecutableBBs.count(BB)) {
        DOUT << "  BasicBlock Dead:" << *BB;
        ++IPNumDeadBlocks;

        // Delete the instructions backwards, as it has a reduced likelihood of
        // having to update as many def-use and use-def chains.
        TerminatorInst *TI = BB->getTerminator();
        for (BasicBlock::iterator I = BB->begin(), E = TI; I != E; ++I)
          Insts.push_back(I);

        while (!Insts.empty()) {
          Instruction *I = Insts.back();
          Insts.pop_back();
          if (!I->use_empty())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          BB->getInstList().erase(I);
          MadeChanges = true;
          ++IPNumInstRemoved;
        }

        for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
          BasicBlock *Succ = TI->getSuccessor(i);
          if (!Succ->empty() && isa<PHINode>(Succ->begin()))
            TI->getSuccessor(i)->removePredecessor(BB);
        }
        if (!TI->use_empty())
          TI->replaceAllUsesWith(UndefValue::get(TI->getType()));
        BB->getInstList().erase(TI);

        if (&*BB != &F->front())
          BlocksToErase.push_back(BB);
        else
          new UnreachableInst(BB);

      } else {
        for (BasicBlock::iterator BI = BB->begin(), E = BB->end(); BI != E; ) {
          Instruction *Inst = BI++;
          if (Inst->getType() != Type::VoidTy) {
            LatticeVal &IV = Values[Inst];
            if (IV.isConstant() || IV.isUndefined() &&
                !isa<TerminatorInst>(Inst)) {
              Constant *Const = IV.isConstant()
                ? IV.getConstant() : UndefValue::get(Inst->getType());
              DOUT << "  Constant: " << *Const << " = " << *Inst;

              // Replaces all of the uses of a variable with uses of the
              // constant.
              Inst->replaceAllUsesWith(Const);

              // Delete the instruction.
              if (!isa<TerminatorInst>(Inst) && !isa<CallInst>(Inst))
                BB->getInstList().erase(Inst);

              // Hey, we just changed something!
              MadeChanges = true;
              ++IPNumInstRemoved;
            }
          }
        }
      }

    // Now that all instructions in the function are constant folded, erase dead
    // blocks, because we can now use ConstantFoldTerminator to get rid of
    // in-edges.
    for (unsigned i = 0, e = BlocksToErase.size(); i != e; ++i) {
      // If there are any PHI nodes in this successor, drop entries for BB now.
      BasicBlock *DeadBB = BlocksToErase[i];
      while (!DeadBB->use_empty()) {
        Instruction *I = cast<Instruction>(DeadBB->use_back());
        bool Folded = ConstantFoldTerminator(I->getParent());
        if (!Folded) {
          // The constant folder may not have been able to fold the terminator
          // if this is a branch or switch on undef.  Fold it manually as a
          // branch to the first successor.
          if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
            assert(BI->isConditional() && isa<UndefValue>(BI->getCondition()) &&
                   "Branch should be foldable!");
          } else if (SwitchInst *SI = dyn_cast<SwitchInst>(I)) {
            assert(isa<UndefValue>(SI->getCondition()) && "Switch should fold");
          } else {
            assert(0 && "Didn't fold away reference to block!");
          }
          
          // Make this an uncond branch to the first successor.
          TerminatorInst *TI = I->getParent()->getTerminator();
          new BranchInst(TI->getSuccessor(0), TI);
          
          // Remove entries in successor phi nodes to remove edges.
          for (unsigned i = 1, e = TI->getNumSuccessors(); i != e; ++i)
            TI->getSuccessor(i)->removePredecessor(TI->getParent());
          
          // Remove the old terminator.
          TI->eraseFromParent();
        }
      }

      // Finally, delete the basic block.
      F->getBasicBlockList().erase(DeadBB);
    }
    BlocksToErase.clear();
  }

  // If we inferred constant or undef return values for a function, we replaced
  // all call uses with the inferred value.  This means we don't need to bother
  // actually returning anything from the function.  Replace all return
  // instructions with return undef.
  const DenseMap<Function*, LatticeVal> &RV =Solver.getTrackedFunctionRetVals();
  for (DenseMap<Function*, LatticeVal>::const_iterator I = RV.begin(),
         E = RV.end(); I != E; ++I)
    if (!I->second.isOverdefined() &&
        I->first->getReturnType() != Type::VoidTy) {
      Function *F = I->first;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
          if (!isa<UndefValue>(RI->getOperand(0)))
            RI->setOperand(0, UndefValue::get(F->getReturnType()));
    }

  // If we infered constant or undef values for globals variables, we can delete
  // the global and any stores that remain to it.
  const DenseMap<GlobalVariable*, LatticeVal> &TG = Solver.getTrackedGlobals();
  for (DenseMap<GlobalVariable*, LatticeVal>::const_iterator I = TG.begin(),
         E = TG.end(); I != E; ++I) {
    GlobalVariable *GV = I->first;
    assert(!I->second.isOverdefined() &&
           "Overdefined values should have been taken out of the map!");
    DOUT << "Found that GV '" << GV->getName()<< "' is constant!\n";
    while (!GV->use_empty()) {
      StoreInst *SI = cast<StoreInst>(GV->use_back());
      SI->eraseFromParent();
    }
    M.getGlobalList().erase(GV);
    ++IPNumGlobalConst;
  }

  return MadeChanges;
}

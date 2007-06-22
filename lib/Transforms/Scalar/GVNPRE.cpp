//===- GVNPRE.cpp - Eliminate redundant values and expressions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a hybrid of global value numbering and partial redundancy
// elimination, known as GVN-PRE.  It performs partial redundancy elimination on
// values, rather than lexical expressions, allowing a more comprehensive view 
// the optimization.  It replaces redundant values with uses of earlier 
// occurences of the same value.  While this is beneficial in that it eliminates
// unneeded computation, it also increases register pressure by creating large
// live ranges, and should be used with caution on platforms that are very 
// sensitive to register pressure.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "gvnpre"
#include "llvm/Value.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <deque>
#include <map>
#include <vector>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                         ValueTable Class
//===----------------------------------------------------------------------===//

/// This class holds the mapping between values and value numbers.  It is used
/// as an efficient mechanism to determine the expression-wise equivalence of
/// two values.

namespace {
  class VISIBILITY_HIDDEN ValueTable {
    public:
      struct Expression {
        enum ExpressionOpcode { ADD, SUB, MUL, UDIV, SDIV, FDIV, UREM, SREM, 
                              FREM, SHL, LSHR, ASHR, AND, OR, XOR, ICMPEQ, 
                              ICMPNE, ICMPUGT, ICMPUGE, ICMPULT, ICMPULE, 
                              ICMPSGT, ICMPSGE, ICMPSLT, ICMPSLE, FCMPOEQ, 
                              FCMPOGT, FCMPOGE, FCMPOLT, FCMPOLE, FCMPONE, 
                              FCMPORD, FCMPUNO, FCMPUEQ, FCMPUGT, FCMPUGE, 
                              FCMPULT, FCMPULE, FCMPUNE };
    
        ExpressionOpcode opcode;
        uint32_t leftVN;
        uint32_t rightVN;
      
        bool operator< (const Expression& other) const {
          if (opcode < other.opcode)
            return true;
          else if (opcode > other.opcode)
            return false;
          else if (leftVN < other.leftVN)
            return true;
          else if (leftVN > other.leftVN)
            return false;
          else if (rightVN < other.rightVN)
            return true;
          else if (rightVN > other.rightVN)
            return false;
          else
            return false;
        }
      };
    
    private:
      DenseMap<Value*, uint32_t> valueNumbering;
      std::map<Expression, uint32_t> expressionNumbering;
  
      std::set<Expression> maximalExpressions;
      SmallPtrSet<Value*, 32> maximalValues;
  
      uint32_t nextValueNumber;
    
      Expression::ExpressionOpcode getOpcode(BinaryOperator* BO);
      Expression::ExpressionOpcode getOpcode(CmpInst* C);
      Expression create_expression(BinaryOperator* BO);
      Expression create_expression(CmpInst* C);
    public:
      ValueTable() { nextValueNumber = 1; }
      uint32_t lookup_or_add(Value* V);
      uint32_t lookup(Value* V);
      void add(Value* V, uint32_t num);
      void clear();
      std::set<Expression>& getMaximalExpressions() {
        return maximalExpressions;
      
      }
      SmallPtrSet<Value*, 32>& getMaximalValues() { return maximalValues; }
      void erase(Value* v);
      unsigned size();
  };
}

//===----------------------------------------------------------------------===//
//                     ValueTable Internal Functions
//===----------------------------------------------------------------------===//
ValueTable::Expression::ExpressionOpcode 
                             ValueTable::getOpcode(BinaryOperator* BO) {
  switch(BO->getOpcode()) {
    case Instruction::Add:
      return Expression::ADD;
    case Instruction::Sub:
      return Expression::SUB;
    case Instruction::Mul:
      return Expression::MUL;
    case Instruction::UDiv:
      return Expression::UDIV;
    case Instruction::SDiv:
      return Expression::SDIV;
    case Instruction::FDiv:
      return Expression::FDIV;
    case Instruction::URem:
      return Expression::UREM;
    case Instruction::SRem:
      return Expression::SREM;
    case Instruction::FRem:
      return Expression::FREM;
    case Instruction::Shl:
      return Expression::SHL;
    case Instruction::LShr:
      return Expression::LSHR;
    case Instruction::AShr:
      return Expression::ASHR;
    case Instruction::And:
      return Expression::AND;
    case Instruction::Or:
      return Expression::OR;
    case Instruction::Xor:
      return Expression::XOR;
    
    // THIS SHOULD NEVER HAPPEN
    default:
      assert(0 && "Binary operator with unknown opcode?");
      return Expression::ADD;
  }
}

ValueTable::Expression::ExpressionOpcode ValueTable::getOpcode(CmpInst* C) {
  if (C->getOpcode() == Instruction::ICmp) {
    switch (C->getPredicate()) {
      case ICmpInst::ICMP_EQ:
        return Expression::ICMPEQ;
      case ICmpInst::ICMP_NE:
        return Expression::ICMPNE;
      case ICmpInst::ICMP_UGT:
        return Expression::ICMPUGT;
      case ICmpInst::ICMP_UGE:
        return Expression::ICMPUGE;
      case ICmpInst::ICMP_ULT:
        return Expression::ICMPULT;
      case ICmpInst::ICMP_ULE:
        return Expression::ICMPULE;
      case ICmpInst::ICMP_SGT:
        return Expression::ICMPSGT;
      case ICmpInst::ICMP_SGE:
        return Expression::ICMPSGE;
      case ICmpInst::ICMP_SLT:
        return Expression::ICMPSLT;
      case ICmpInst::ICMP_SLE:
        return Expression::ICMPSLE;
      
      // THIS SHOULD NEVER HAPPEN
      default:
        assert(0 && "Comparison with unknown predicate?");
        return Expression::ICMPEQ;
    }
  } else {
    switch (C->getPredicate()) {
      case FCmpInst::FCMP_OEQ:
        return Expression::FCMPOEQ;
      case FCmpInst::FCMP_OGT:
        return Expression::FCMPOGT;
      case FCmpInst::FCMP_OGE:
        return Expression::FCMPOGE;
      case FCmpInst::FCMP_OLT:
        return Expression::FCMPOLT;
      case FCmpInst::FCMP_OLE:
        return Expression::FCMPOLE;
      case FCmpInst::FCMP_ONE:
        return Expression::FCMPONE;
      case FCmpInst::FCMP_ORD:
        return Expression::FCMPORD;
      case FCmpInst::FCMP_UNO:
        return Expression::FCMPUNO;
      case FCmpInst::FCMP_UEQ:
        return Expression::FCMPUEQ;
      case FCmpInst::FCMP_UGT:
        return Expression::FCMPUGT;
      case FCmpInst::FCMP_UGE:
        return Expression::FCMPUGE;
      case FCmpInst::FCMP_ULT:
        return Expression::FCMPULT;
      case FCmpInst::FCMP_ULE:
        return Expression::FCMPULE;
      case FCmpInst::FCMP_UNE:
        return Expression::FCMPUNE;
      
      // THIS SHOULD NEVER HAPPEN
      default:
        assert(0 && "Comparison with unknown predicate?");
        return Expression::FCMPOEQ;
    }
  }
}

ValueTable::Expression ValueTable::create_expression(BinaryOperator* BO) {
  Expression e;
    
  e.leftVN = lookup_or_add(BO->getOperand(0));
  e.rightVN = lookup_or_add(BO->getOperand(1));
  e.opcode = getOpcode(BO);
  
  maximalExpressions.insert(e);
  
  return e;
}

ValueTable::Expression ValueTable::create_expression(CmpInst* C) {
  Expression e;
    
  e.leftVN = lookup_or_add(C->getOperand(0));
  e.rightVN = lookup_or_add(C->getOperand(1));
  e.opcode = getOpcode(C);
  
  maximalExpressions.insert(e);
  
  return e;
}

//===----------------------------------------------------------------------===//
//                     ValueTable External Functions
//===----------------------------------------------------------------------===//

/// lookup_or_add - Returns the value number for the specified value, assigning
/// it a new number if it did not have one before.
uint32_t ValueTable::lookup_or_add(Value* V) {
  maximalValues.insert(V);

  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  
  
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Expression e = create_expression(BO);
    
    std::map<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (CmpInst* C = dyn_cast<CmpInst>(V)) {
    Expression e = create_expression(C);
    
    std::map<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else {
    valueNumbering.insert(std::make_pair(V, nextValueNumber));
    return nextValueNumber++;
  }
}

/// lookup - Returns the value number of the specified value. Fails if
/// the value has not yet been numbered.
uint32_t ValueTable::lookup(Value* V) {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  else
    assert(0 && "Value not numbered?");
  
  return 0;
}

/// add - Add the specified value with the given value number, removing
/// its old number, if any
void ValueTable::add(Value* V, uint32_t num) {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    valueNumbering.erase(VI);
  valueNumbering.insert(std::make_pair(V, num));
}

/// clear - Remove all entries from the ValueTable and the maximal sets
void ValueTable::clear() {
  valueNumbering.clear();
  expressionNumbering.clear();
  maximalExpressions.clear();
  maximalValues.clear();
  nextValueNumber = 1;
}

/// erase - Remove a value from the value numbering and maximal sets
void ValueTable::erase(Value* V) {
  maximalValues.erase(V);
  valueNumbering.erase(V);
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V))
    maximalExpressions.erase(create_expression(BO));
  else if (CmpInst* C = dyn_cast<CmpInst>(V))
    maximalExpressions.erase(create_expression(C));
}

/// size - Return the number of assigned value numbers
unsigned ValueTable::size() {
  // NOTE: zero is never assigned
  return nextValueNumber;
}

//===----------------------------------------------------------------------===//
//                         GVNPRE Pass
//===----------------------------------------------------------------------===//

namespace {

  class VISIBILITY_HIDDEN GVNPRE : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVNPRE() : FunctionPass((intptr_t)&ID) { }

  private:
    ValueTable VN;
    std::vector<Instruction*> createdExpressions;
    
    std::map<BasicBlock*, SmallPtrSet<Value*, 32> > availableOut;
    std::map<BasicBlock*, SmallPtrSet<Value*, 32> > anticipatedIn;
    
    // This transformation requires dominator postdominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<PostDominatorTree>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    void dump(const SmallPtrSet<Value*, 32>& s) const;
    void clean(SmallPtrSet<Value*, 32>& set);
    Value* find_leader(SmallPtrSet<Value*, 32>& vals,
                       uint32_t v);
    Value* phi_translate(Value* V, BasicBlock* pred, BasicBlock* succ);
    void phi_translate_set(SmallPtrSet<Value*, 32>& anticIn, BasicBlock* pred,
                           BasicBlock* succ, SmallPtrSet<Value*, 32>& out);
    
    void topo_sort(SmallPtrSet<Value*, 32>& set,
                   std::vector<Value*>& vec);
    
    void cleanup();
    bool elimination();
    
    void val_insert(SmallPtrSet<Value*, 32>& s, Value* v);
    void val_replace(SmallPtrSet<Value*, 32>& s, Value* v);
    bool dependsOnInvoke(Value* V);
    void buildsets_availout(BasicBlock::iterator I,
                            SmallPtrSet<Value*, 32>& currAvail,
                            SmallPtrSet<PHINode*, 32>& currPhis,
                            SmallPtrSet<Value*, 32>& currExps,
                            SmallPtrSet<Value*, 32>& currTemps,
                            BitVector& availNumbers,
                            BitVector& expNumbers);
    bool buildsets_anticout(BasicBlock* BB,
                            SmallPtrSet<Value*, 32>& anticOut,
                            std::set<BasicBlock*>& visited);
    unsigned buildsets_anticin(BasicBlock* BB,
                           SmallPtrSet<Value*, 32>& anticOut,
                           SmallPtrSet<Value*, 32>& currExps,
                           SmallPtrSet<Value*, 32>& currTemps,
                           std::set<BasicBlock*>& visited);
    unsigned buildsets(Function& F);
    
    void insertion_pre(Value* e, BasicBlock* BB,
                       std::map<BasicBlock*, Value*>& avail,
                       SmallPtrSet<Value*, 32>& new_set);
    unsigned insertion_mergepoint(std::vector<Value*>& workList,
                                  df_iterator<DomTreeNode*>& D,
                                  SmallPtrSet<Value*, 32>& new_set);
    bool insertion(Function& F);
  
  };
  
  char GVNPRE::ID = 0;
  
}

// createGVNPREPass - The public interface to this file...
FunctionPass *llvm::createGVNPREPass() { return new GVNPRE(); }

RegisterPass<GVNPRE> X("gvnpre",
                       "Global Value Numbering/Partial Redundancy Elimination");


STATISTIC(NumInsertedVals, "Number of values inserted");
STATISTIC(NumInsertedPhis, "Number of PHI nodes inserted");
STATISTIC(NumEliminated, "Number of redundant instructions eliminated");

/// find_leader - Given a set and a value number, return the first
/// element of the set with that value number, or 0 if no such element
/// is present
Value* GVNPRE::find_leader(SmallPtrSet<Value*, 32>& vals, uint32_t v) {
  for (SmallPtrSet<Value*, 32>::iterator I = vals.begin(), E = vals.end();
       I != E; ++I)
    if (v == VN.lookup(*I))
      return *I;
  
  return 0;
}

/// val_insert - Insert a value into a set only if there is not a value
/// with the same value number already in the set
void GVNPRE::val_insert(SmallPtrSet<Value*, 32>& s, Value* v) {
  uint32_t num = VN.lookup(v);
  Value* leader = find_leader(s, num);
  if (leader == 0)
    s.insert(v);
}

/// val_replace - Insert a value into a set, replacing any values already in
/// the set that have the same value number
void GVNPRE::val_replace(SmallPtrSet<Value*, 32>& s, Value* v) {
  uint32_t num = VN.lookup(v);
  Value* leader = find_leader(s, num);
  while (leader != 0) {
    s.erase(leader);
    leader = find_leader(s, num);
  }
  s.insert(v);
}

/// phi_translate - Given a value, its parent block, and a predecessor of its
/// parent, translate the value into legal for the predecessor block.  This 
/// means translating its operands (and recursively, their operands) through
/// any phi nodes in the parent into values available in the predecessor
Value* GVNPRE::phi_translate(Value* V, BasicBlock* pred, BasicBlock* succ) {
  if (V == 0)
    return 0;
  
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Value* newOp1 = 0;
    if (isa<Instruction>(BO->getOperand(0)))
      newOp1 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(BO->getOperand(0))),
                             pred, succ);
    else
      newOp1 = BO->getOperand(0);
    
    if (newOp1 == 0)
      return 0;
    
    Value* newOp2 = 0;
    if (isa<Instruction>(BO->getOperand(1)))
      newOp2 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(BO->getOperand(1))),
                             pred, succ);
    else
      newOp2 = BO->getOperand(1);
    
    if (newOp2 == 0)
      return 0;
    
    if (newOp1 != BO->getOperand(0) || newOp2 != BO->getOperand(1)) {
      Instruction* newVal = BinaryOperator::create(BO->getOpcode(),
                                             newOp1, newOp2,
                                             BO->getName()+".expr");
      
      uint32_t v = VN.lookup_or_add(newVal);
      
      Value* leader = find_leader(availableOut[pred], v);
      if (leader == 0) {
        createdExpressions.push_back(newVal);
        return newVal;
      } else {
        VN.erase(newVal);
        delete newVal;
        return leader;
      }
    }
  } else if (PHINode* P = dyn_cast<PHINode>(V)) {
    if (P->getParent() == succ)
      return P->getIncomingValueForBlock(pred);
  } else if (CmpInst* C = dyn_cast<CmpInst>(V)) {
    Value* newOp1 = 0;
    if (isa<Instruction>(C->getOperand(0)))
      newOp1 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(C->getOperand(0))),
                             pred, succ);
    else
      newOp1 = C->getOperand(0);
    
    if (newOp1 == 0)
      return 0;
    
    Value* newOp2 = 0;
    if (isa<Instruction>(C->getOperand(1)))
      newOp2 = phi_translate(find_leader(anticipatedIn[succ],         
                                         VN.lookup(C->getOperand(1))),
                             pred, succ);
    else
      newOp2 = C->getOperand(1);
      
    if (newOp2 == 0)
      return 0;
    
    if (newOp1 != C->getOperand(0) || newOp2 != C->getOperand(1)) {
      Instruction* newVal = CmpInst::create(C->getOpcode(),
                                            C->getPredicate(),
                                             newOp1, newOp2,
                                             C->getName()+".expr");
      
      uint32_t v = VN.lookup_or_add(newVal);
        
      Value* leader = find_leader(availableOut[pred], v);
      if (leader == 0) {
        createdExpressions.push_back(newVal);
        return newVal;
      } else {
        VN.erase(newVal);
        delete newVal;
        return leader;
      }
    }
  }
  
  return V;
}

/// phi_translate_set - Perform phi translation on every element of a set
void GVNPRE::phi_translate_set(SmallPtrSet<Value*, 32>& anticIn,
                              BasicBlock* pred, BasicBlock* succ,
                              SmallPtrSet<Value*, 32>& out) {
  for (SmallPtrSet<Value*, 32>::iterator I = anticIn.begin(),
       E = anticIn.end(); I != E; ++I) {
    Value* V = phi_translate(*I, pred, succ);
    if (V != 0)
      out.insert(V);
  }
}

/// dependsOnInvoke - Test if a value has an phi node as an operand, any of 
/// whose inputs is an invoke instruction.  If this is true, we cannot safely
/// PRE the instruction or anything that depends on it.
bool GVNPRE::dependsOnInvoke(Value* V) {
  if (PHINode* p = dyn_cast<PHINode>(V)) {
    for (PHINode::op_iterator I = p->op_begin(), E = p->op_end(); I != E; ++I)
      if (isa<InvokeInst>(*I))
        return true;
    return false;
  } else {
    return false;
  }
}

/// clean - Remove all non-opaque values from the set whose operands are not
/// themselves in the set, as well as all values that depend on invokes (see 
/// above)
void GVNPRE::clean(SmallPtrSet<Value*, 32>& set) {
  std::vector<Value*> worklist;
  worklist.reserve(set.size());
  topo_sort(set, worklist);
  
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value* v = worklist[i];
    
    if (BinaryOperator* BO = dyn_cast<BinaryOperator>(v)) {   
      bool lhsValid = !isa<Instruction>(BO->getOperand(0));
      if (!lhsValid)
        for (SmallPtrSet<Value*, 32>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN.lookup(*I) == VN.lookup(BO->getOperand(0))) {
            lhsValid = true;
            break;
          }
      if (lhsValid)
        lhsValid = !dependsOnInvoke(BO->getOperand(0));
    
      bool rhsValid = !isa<Instruction>(BO->getOperand(1));
      if (!rhsValid)
        for (SmallPtrSet<Value*, 32>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN.lookup(*I) == VN.lookup(BO->getOperand(1))) {
            rhsValid = true;
            break;
          }
      if (rhsValid)
        rhsValid = !dependsOnInvoke(BO->getOperand(1));
      
      if (!lhsValid || !rhsValid)
        set.erase(BO);
    } else if (CmpInst* C = dyn_cast<CmpInst>(v)) {
      bool lhsValid = !isa<Instruction>(C->getOperand(0));
      if (!lhsValid)
        for (SmallPtrSet<Value*, 32>::iterator I = set.begin(), E = set.end();
             I != E; ++I)
          if (VN.lookup(*I) == VN.lookup(C->getOperand(0))) {
            lhsValid = true;
            break;
          }
      if (lhsValid)
        lhsValid = !dependsOnInvoke(C->getOperand(0));
      
      bool rhsValid = !isa<Instruction>(C->getOperand(1));
      if (!rhsValid)
      for (SmallPtrSet<Value*, 32>::iterator I = set.begin(), E = set.end();
           I != E; ++I)
        if (VN.lookup(*I) == VN.lookup(C->getOperand(1))) {
          rhsValid = true;
          break;
        }
      if (rhsValid)
        rhsValid = !dependsOnInvoke(C->getOperand(1));
    
      if (!lhsValid || !rhsValid)
        set.erase(C);
    }
  }
}

/// topo_sort - Given a set of values, sort them by topological
/// order into the provided vector.
void GVNPRE::topo_sort(SmallPtrSet<Value*, 32>& set, std::vector<Value*>& vec) {
  SmallPtrSet<Value*, 32> visited;
  std::vector<Value*> stack;
  for (SmallPtrSet<Value*, 32>::iterator I = set.begin(), E = set.end();
       I != E; ++I) {
    if (visited.count(*I) == 0)
      stack.push_back(*I);
    
    while (!stack.empty()) {
      Value* e = stack.back();
  
      if (BinaryOperator* BO = dyn_cast<BinaryOperator>(e)) {
        Value* l = find_leader(set, VN.lookup(BO->getOperand(0)));
        Value* r = find_leader(set, VN.lookup(BO->getOperand(1)));
    
        if (l != 0 && isa<Instruction>(l) &&
            visited.count(l) == 0)
          stack.push_back(l);
        else if (r != 0 && isa<Instruction>(r) &&
                 visited.count(r) == 0)
          stack.push_back(r);
        else {
          vec.push_back(e);
          visited.insert(e);
          stack.pop_back();
        }
      } else if (CmpInst* C = dyn_cast<CmpInst>(e)) {
        Value* l = find_leader(set, VN.lookup(C->getOperand(0)));
        Value* r = find_leader(set, VN.lookup(C->getOperand(1)));
      
        if (l != 0 && isa<Instruction>(l) &&
            visited.count(l) == 0)
          stack.push_back(l);
        else if (r != 0 && isa<Instruction>(r) &&
                 visited.count(r) == 0)
          stack.push_back(r);
        else {
          vec.push_back(e);
          visited.insert(e);
          stack.pop_back();
        }
      } else {
        visited.insert(e);
        vec.push_back(e);
        stack.pop_back();
      }
    }
    
    stack.clear();
  }
}

/// dump - Dump a set of values to standard error
void GVNPRE::dump(const SmallPtrSet<Value*, 32>& s) const {
  DOUT << "{ ";
  for (SmallPtrSet<Value*, 32>::iterator I = s.begin(), E = s.end();
       I != E; ++I) {
    DEBUG((*I)->dump());
  }
  DOUT << "}\n\n";
}

/// elimination - Phase 3 of the main algorithm.  Perform full redundancy 
/// elimination by walking the dominator tree and removing any instruction that 
/// is dominated by another instruction with the same value number.
bool GVNPRE::elimination() {
  DOUT << "\n\nPhase 3: Elimination\n\n";
  
  bool changed_function = false;
  
  std::vector<std::pair<Instruction*, Value*> > replace;
  std::vector<Instruction*> erase;
  
  DominatorTree& DT = getAnalysis<DominatorTree>();
  
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    BasicBlock* BB = DI->getBlock();
    
    DOUT << "Block: " << BB->getName() << "\n";
    dump(availableOut[BB]);
    DOUT << "\n\n";
    
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ++BI) {

      if (isa<BinaryOperator>(BI) || isa<CmpInst>(BI)) {
         Value *leader = find_leader(availableOut[BB], VN.lookup(BI));
  
        if (leader != 0)
          if (Instruction* Instr = dyn_cast<Instruction>(leader))
            if (Instr->getParent() != 0 && Instr != BI) {
              replace.push_back(std::make_pair(BI, leader));
              erase.push_back(BI);
              ++NumEliminated;
            }
      }
    }
  }
  
  while (!replace.empty()) {
    std::pair<Instruction*, Value*> rep = replace.back();
    replace.pop_back();
    rep.first->replaceAllUsesWith(rep.second);
    changed_function = true;
  }
    
  for (std::vector<Instruction*>::iterator I = erase.begin(), E = erase.end();
       I != E; ++I)
     (*I)->eraseFromParent();
  
  return changed_function;
}

/// cleanup - Delete any extraneous values that were created to represent
/// expressions without leaders.
void GVNPRE::cleanup() {
  while (!createdExpressions.empty()) {
    Instruction* I = createdExpressions.back();
    createdExpressions.pop_back();
    
    delete I;
  }
}

/// buildsets_availout - When calculating availability, handle an instruction
/// by inserting it into the appropriate sets
void GVNPRE::buildsets_availout(BasicBlock::iterator I,
                                SmallPtrSet<Value*, 32>& currAvail,
                                SmallPtrSet<PHINode*, 32>& currPhis,
                                SmallPtrSet<Value*, 32>& currExps,
                                SmallPtrSet<Value*, 32>& currTemps,
                                BitVector& availNumbers,
                                BitVector& expNumbers) {
  // Handle PHI nodes...
  if (PHINode* p = dyn_cast<PHINode>(I)) {
    VN.lookup_or_add(p);
    expNumbers.resize(VN.size());
    availNumbers.resize(VN.size());
    
    currPhis.insert(p);
    
  // Handle binary ops...
  } else if (BinaryOperator* BO = dyn_cast<BinaryOperator>(I)) {
    Value* leftValue = BO->getOperand(0);
    Value* rightValue = BO->getOperand(1);
    
    unsigned num = VN.lookup_or_add(BO);
    expNumbers.resize(VN.size());
    availNumbers.resize(VN.size());
      
    if (isa<Instruction>(leftValue))
      if (!expNumbers.test(VN.lookup(leftValue))) {
        currExps.insert(leftValue);
        expNumbers.set(VN.lookup(leftValue));
      }
    
    if (isa<Instruction>(rightValue))
      if (!expNumbers.test(VN.lookup(rightValue))) {
        currExps.insert(rightValue);
        expNumbers.set(VN.lookup(rightValue));
      }
    
    if (!expNumbers.test(VN.lookup(BO))) {
      currExps.insert(BO);
      expNumbers.set(num);
    }
    
  // Handle cmp ops...
  } else if (CmpInst* C = dyn_cast<CmpInst>(I)) {
    Value* leftValue = C->getOperand(0);
    Value* rightValue = C->getOperand(1);
      
    VN.lookup_or_add(C);
    
    unsigned num = VN.lookup_or_add(C);
    expNumbers.resize(VN.size());
    availNumbers.resize(VN.size());
    
    if (isa<Instruction>(leftValue))
      if (!expNumbers.test(VN.lookup(leftValue))) {
        currExps.insert(leftValue);
        expNumbers.set(VN.lookup(leftValue));
      }
    if (isa<Instruction>(rightValue))
      if (!expNumbers.test(VN.lookup(rightValue))) {
        currExps.insert(rightValue);
        expNumbers.set(VN.lookup(rightValue));
      }
    
    if (!expNumbers.test(VN.lookup(C))) {
      currExps.insert(C);
      expNumbers.set(num);
    }
    
  // Handle unsupported ops
  } else if (!I->isTerminator()){
    VN.lookup_or_add(I);
    expNumbers.resize(VN.size());
    availNumbers.resize(VN.size());
    
    currTemps.insert(I);
  }
    
  if (!I->isTerminator())
    if (!availNumbers.test(VN.lookup(I))) {
      currAvail.insert(I);
      availNumbers.set(VN.lookup(I));
    }
}

/// buildsets_anticout - When walking the postdom tree, calculate the ANTIC_OUT
/// set as a function of the ANTIC_IN set of the block's predecessors
bool GVNPRE::buildsets_anticout(BasicBlock* BB,
                                SmallPtrSet<Value*, 32>& anticOut,
                                std::set<BasicBlock*>& visited) {
  if (BB->getTerminator()->getNumSuccessors() == 1) {
    if (visited.count(BB->getTerminator()->getSuccessor(0)) == 0)
      return true;
    else
      phi_translate_set(anticipatedIn[BB->getTerminator()->getSuccessor(0)],
                        BB,  BB->getTerminator()->getSuccessor(0), anticOut);
  } else if (BB->getTerminator()->getNumSuccessors() > 1) {
    BasicBlock* first = BB->getTerminator()->getSuccessor(0);
    anticOut.insert(anticipatedIn[first].begin(), anticipatedIn[first].end());
    
    for (unsigned i = 1; i < BB->getTerminator()->getNumSuccessors(); ++i) {
      BasicBlock* currSucc = BB->getTerminator()->getSuccessor(i);
      SmallPtrSet<Value*, 32>& succAnticIn = anticipatedIn[currSucc];
      
      std::vector<Value*> temp;
      
      for (SmallPtrSet<Value*, 32>::iterator I = anticOut.begin(),
           E = anticOut.end(); I != E; ++I)
        if (succAnticIn.count(*I) == 0)
          temp.push_back(*I);

      for (std::vector<Value*>::iterator I = temp.begin(), E = temp.end();
           I != E; ++I)
        anticOut.erase(*I);
    }
  }
  
  return false;
}

/// buildsets_anticin - Walk the postdom tree, calculating ANTIC_OUT for
/// each block.  ANTIC_IN is then a function of ANTIC_OUT and the GEN
/// sets populated in buildsets_availout
unsigned GVNPRE::buildsets_anticin(BasicBlock* BB,
                               SmallPtrSet<Value*, 32>& anticOut,
                               SmallPtrSet<Value*, 32>& currExps,
                               SmallPtrSet<Value*, 32>& currTemps,
                               std::set<BasicBlock*>& visited) {
  SmallPtrSet<Value*, 32>& anticIn = anticipatedIn[BB];
  unsigned old = anticIn.size();
      
  bool defer = buildsets_anticout(BB, anticOut, visited);
  if (defer)
    return 0;
  
  anticIn.clear();
  
  BitVector numbers(VN.size());
  for (SmallPtrSet<Value*, 32>::iterator I = anticOut.begin(),
       E = anticOut.end(); I != E; ++I) {
    anticIn.insert(*I);
    numbers.set(VN.lookup_or_add(*I));
  }
  for (SmallPtrSet<Value*, 32>::iterator I = currExps.begin(),
       E = currExps.end(); I != E; ++I) {
    if (!numbers.test(VN.lookup_or_add(*I))) {
      anticIn.insert(*I);
      numbers.set(VN.lookup(*I));
    }
  } 
  
  for (SmallPtrSet<Value*, 32>::iterator I = currTemps.begin(),
       E = currTemps.end(); I != E; ++I)
    anticIn.erase(*I);
  
  clean(anticIn);
  anticOut.clear();
  
  if (old != anticIn.size())
    return 2;
  else
    return 1;
}

/// buildsets - Phase 1 of the main algorithm.  Construct the AVAIL_OUT
/// and the ANTIC_IN sets.
unsigned GVNPRE::buildsets(Function& F) {
  std::map<BasicBlock*, SmallPtrSet<Value*, 32> > generatedExpressions;
  std::map<BasicBlock*, SmallPtrSet<PHINode*, 32> > generatedPhis;
  std::map<BasicBlock*, SmallPtrSet<Value*, 32> > generatedTemporaries;

  DominatorTree &DT = getAnalysis<DominatorTree>();   
  
  // Phase 1, Part 1: calculate AVAIL_OUT
  
  // Top-down walk of the dominator tree
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    
    // Get the sets to update for this block
    SmallPtrSet<Value*, 32>& currExps = generatedExpressions[DI->getBlock()];
    SmallPtrSet<PHINode*, 32>& currPhis = generatedPhis[DI->getBlock()];
    SmallPtrSet<Value*, 32>& currTemps = generatedTemporaries[DI->getBlock()];
    SmallPtrSet<Value*, 32>& currAvail = availableOut[DI->getBlock()];     
    
    BasicBlock* BB = DI->getBlock();
  
    // A block inherits AVAIL_OUT from its dominator
    if (DI->getIDom() != 0)
    currAvail.insert(availableOut[DI->getIDom()->getBlock()].begin(),
                     availableOut[DI->getIDom()->getBlock()].end());
    
    BitVector availNumbers(VN.size());
    for (SmallPtrSet<Value*, 32>::iterator I = currAvail.begin(),
        E = currAvail.end(); I != E; ++I)
      availNumbers.set(VN.lookup(*I));
    
    BitVector expNumbers(VN.size());
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ++BI)
      buildsets_availout(BI, currAvail, currPhis, currExps,
                         currTemps, availNumbers, expNumbers);
      
  }
  
  // If function has no exit blocks, only perform GVN
  PostDominatorTree &PDT = getAnalysis<PostDominatorTree>();
  if (PDT[&F.getEntryBlock()] == 0) {
    bool changed_function = elimination();
    cleanup();
    
    if (changed_function)
      return 2;  // Bailed early, made changes
    else
      return 1;  // Bailed early, no changes
  }
  
  
  // Phase 1, Part 2: calculate ANTIC_IN
  
  std::set<BasicBlock*> visited;
  
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;
    SmallPtrSet<Value*, 32> anticOut;
    
    // Top-down walk of the postdominator tree
    for (df_iterator<DomTreeNode*> PDI = 
         df_begin(PDT.getRootNode()), E = df_end(PDT.getRootNode());
         PDI != E; ++PDI) {
      BasicBlock* BB = PDI->getBlock();
      if (BB == 0)
        continue;
      
      
      
      unsigned ret = buildsets_anticin(BB, anticOut, generatedTemporaries[BB], 
                                   generatedExpressions[BB], visited);
      
      if (ret == 0) {
        changed = true;
        break;
      } else {
        visited.insert(BB);
        changed |= (ret == 2);
      }
    }
    
    iterations++;
  }
  
  return 0; // No bail, no changes
}

/// insertion_pre - When a partial redundancy has been identified, eliminate it
/// by inserting appropriate values into the predecessors and a phi node in
/// the main block
void GVNPRE::insertion_pre(Value* e, BasicBlock* BB,
                           std::map<BasicBlock*, Value*>& avail,
                           SmallPtrSet<Value*, 32>& new_set) {
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
    Value* e2 = avail[*PI];
    if (!find_leader(availableOut[*PI], VN.lookup(e2))) {
      User* U = cast<User>(e2);
      
      Value* s1 = 0;
      if (isa<BinaryOperator>(U->getOperand(0)) || 
          isa<CmpInst>(U->getOperand(0)))
        s1 = find_leader(availableOut[*PI], VN.lookup(U->getOperand(0)));
      else
        s1 = U->getOperand(0);
      
      Value* s2 = 0;
      if (isa<BinaryOperator>(U->getOperand(1)) ||
          isa<CmpInst>(U->getOperand(1)))
        s2 = find_leader(availableOut[*PI], VN.lookup(U->getOperand(1)));
      else
        s2 = U->getOperand(1);
      
      Value* newVal = 0;
      if (BinaryOperator* BO = dyn_cast<BinaryOperator>(U))
        newVal = BinaryOperator::create(BO->getOpcode(), s1, s2,
                                        BO->getName()+".gvnpre",
                                        (*PI)->getTerminator());
      else if (CmpInst* C = dyn_cast<CmpInst>(U))
        newVal = CmpInst::create(C->getOpcode(), C->getPredicate(), s1, s2,
                                 C->getName()+".gvnpre", 
                                 (*PI)->getTerminator());
                  
      VN.add(newVal, VN.lookup(U));
                  
      SmallPtrSet<Value*, 32>& predAvail = availableOut[*PI];
      val_replace(predAvail, newVal);
            
      std::map<BasicBlock*, Value*>::iterator av = avail.find(*PI);
      if (av != avail.end())
        avail.erase(av);
      avail.insert(std::make_pair(*PI, newVal));
                  
      ++NumInsertedVals;
    }
  }
              
  PHINode* p = 0;
              
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
    if (p == 0)
      p = new PHINode(avail[*PI]->getType(), "gvnpre-join", BB->begin());
                
    p->addIncoming(avail[*PI], *PI);
  }

  VN.add(p, VN.lookup(e));
  val_replace(availableOut[BB], p);
  new_set.insert(p);
              
  ++NumInsertedPhis;
}

/// insertion_mergepoint - When walking the dom tree, check at each merge
/// block for the possibility of a partial redundancy.  If present, eliminate it
unsigned GVNPRE::insertion_mergepoint(std::vector<Value*>& workList,
                                      df_iterator<DomTreeNode*>& D,
                                      SmallPtrSet<Value*, 32>& new_set) {
  bool changed_function = false;
  bool new_stuff = false;
  
  BasicBlock* BB = D->getBlock();
  for (unsigned i = 0; i < workList.size(); ++i) {
    Value* e = workList[i];
          
    if (isa<BinaryOperator>(e) || isa<CmpInst>(e)) {
      if (find_leader(availableOut[D->getIDom()->getBlock()],
                      VN.lookup(e)) != 0)
        continue;
            
      std::map<BasicBlock*, Value*> avail;
      bool by_some = false;
      int num_avail = 0;
            
      for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE;
           ++PI) {
        Value *e2 = phi_translate(e, *PI, BB);
        Value *e3 = find_leader(availableOut[*PI], VN.lookup(e2));
              
        if (e3 == 0) {
          std::map<BasicBlock*, Value*>::iterator av = avail.find(*PI);
          if (av != avail.end())
            avail.erase(av);
          avail.insert(std::make_pair(*PI, e2));
        } else {
          std::map<BasicBlock*, Value*>::iterator av = avail.find(*PI);
          if (av != avail.end())
            avail.erase(av);
          avail.insert(std::make_pair(*PI, e3));
                
          by_some = true;
          num_avail++;
        }
      }
            
      if (by_some && num_avail < std::distance(pred_begin(BB), pred_end(BB))) {
        insertion_pre(e, BB, avail, new_set);
              
        changed_function = true;
        new_stuff = true;
      }
    }
  }
  
  unsigned retval = 0;
  if (changed_function)
    retval += 1;
  if (new_stuff)
    retval += 2;
  
  return retval;
}

/// insert - Phase 2 of the main algorithm.  Walk the dominator tree looking for
/// merge points.  When one is found, check for a partial redundancy.  If one is
/// present, eliminate it.  Repeat this walk until no changes are made.
bool GVNPRE::insertion(Function& F) {
  bool changed_function = false;

  DominatorTree &DT = getAnalysis<DominatorTree>();  
  
  std::map<BasicBlock*, SmallPtrSet<Value*, 32> > new_sets;
  bool new_stuff = true;
  while (new_stuff) {
    new_stuff = false;
    for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
      BasicBlock* BB = DI->getBlock();
      
      if (BB == 0)
        continue;
      
      SmallPtrSet<Value*, 32>& new_set = new_sets[BB];
      SmallPtrSet<Value*, 32>& availOut = availableOut[BB];
      SmallPtrSet<Value*, 32>& anticIn = anticipatedIn[BB];
      
      new_set.clear();
      
      // Replace leaders with leaders inherited from dominator
      if (DI->getIDom() != 0) {
        SmallPtrSet<Value*, 32>& dom_set = new_sets[DI->getIDom()->getBlock()];
        for (SmallPtrSet<Value*, 32>::iterator I = dom_set.begin(),
             E = dom_set.end(); I != E; ++I) {
          new_set.insert(*I);
          val_replace(availOut, *I);
        }
      }
      
      // If there is more than one predecessor...
      if (pred_begin(BB) != pred_end(BB) && ++pred_begin(BB) != pred_end(BB)) {
        std::vector<Value*> workList;
        workList.reserve(anticIn.size());
        topo_sort(anticIn, workList);
        
        DOUT << "Merge Block: " << BB->getName() << "\n";
        DOUT << "ANTIC_IN: ";
        dump(anticIn);
        DOUT << "\n";
        
        unsigned result = insertion_mergepoint(workList, DI, new_set);
        if (result & 1)
          changed_function = true;
        if (result & 2)
          new_stuff = true;
      }
    }
  }
  
  return changed_function;
}

// GVNPRE::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GVNPRE::runOnFunction(Function &F) {
  // Clean out global sets from any previous functions
  VN.clear();
  createdExpressions.clear();
  availableOut.clear();
  anticipatedIn.clear();
  
  bool changed_function = false;
  
  // Phase 1: BuildSets
  // This phase calculates the AVAIL_OUT and ANTIC_IN sets
  // NOTE: If full postdom information is no available, this will bail
  // early, performing GVN but not PRE
  unsigned bail = buildsets(F);
  //If a bail occurred, terminate early
  if (bail != 0)
    return (bail == 2);
  
  // Phase 2: Insert
  // This phase inserts values to make partially redundant values
  // fully redundant
  changed_function |= insertion(F);
  
  // Phase 3: Eliminate
  // This phase performs trivial full redundancy elimination
  changed_function |= elimination();
  
  // Phase 4: Cleanup
  // This phase cleans up values that were created solely
  // as leaders for expressions
  cleanup();
  
  return changed_function;
}

//===- GVN.cpp - Eliminate redundant values and loads ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs global value numbering to eliminate fully redundant
// instructions.  It also performs simple dead load elimination.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "gvn"

#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                         ValueTable Class
//===----------------------------------------------------------------------===//

/// This class holds the mapping between values and value numbers.  It is used
/// as an efficient mechanism to determine the expression-wise equivalence of
/// two values.
namespace {
  struct VISIBILITY_HIDDEN Expression {
    enum ExpressionOpcode { ADD, SUB, MUL, UDIV, SDIV, FDIV, UREM, SREM, 
                            FREM, SHL, LSHR, ASHR, AND, OR, XOR, ICMPEQ, 
                            ICMPNE, ICMPUGT, ICMPUGE, ICMPULT, ICMPULE, 
                            ICMPSGT, ICMPSGE, ICMPSLT, ICMPSLE, FCMPOEQ, 
                            FCMPOGT, FCMPOGE, FCMPOLT, FCMPOLE, FCMPONE, 
                            FCMPORD, FCMPUNO, FCMPUEQ, FCMPUGT, FCMPUGE, 
                            FCMPULT, FCMPULE, FCMPUNE, EXTRACT, INSERT,
                            SHUFFLE, SELECT, TRUNC, ZEXT, SEXT, FPTOUI,
                            FPTOSI, UITOFP, SITOFP, FPTRUNC, FPEXT, 
                            PTRTOINT, INTTOPTR, BITCAST, GEP, CALL, EMPTY,
                            TOMBSTONE };

    ExpressionOpcode opcode;
    const Type* type;
    uint32_t firstVN;
    uint32_t secondVN;
    uint32_t thirdVN;
    SmallVector<uint32_t, 4> varargs;
    Value* function;
  
    Expression() { }
    Expression(ExpressionOpcode o) : opcode(o) { }
  
    bool operator==(const Expression &other) const {
      if (opcode != other.opcode)
        return false;
      else if (opcode == EMPTY || opcode == TOMBSTONE)
        return true;
      else if (type != other.type)
        return false;
      else if (function != other.function)
        return false;
      else if (firstVN != other.firstVN)
        return false;
      else if (secondVN != other.secondVN)
        return false;
      else if (thirdVN != other.thirdVN)
        return false;
      else {
        if (varargs.size() != other.varargs.size())
          return false;
      
        for (size_t i = 0; i < varargs.size(); ++i)
          if (varargs[i] != other.varargs[i])
            return false;
    
        return true;
      }
    }
  
    bool operator!=(const Expression &other) const {
      if (opcode != other.opcode)
        return true;
      else if (opcode == EMPTY || opcode == TOMBSTONE)
        return false;
      else if (type != other.type)
        return true;
      else if (function != other.function)
        return true;
      else if (firstVN != other.firstVN)
        return true;
      else if (secondVN != other.secondVN)
        return true;
      else if (thirdVN != other.thirdVN)
        return true;
      else {
        if (varargs.size() != other.varargs.size())
          return true;
      
        for (size_t i = 0; i < varargs.size(); ++i)
          if (varargs[i] != other.varargs[i])
            return true;
    
          return false;
      }
    }
  };
  
  class VISIBILITY_HIDDEN ValueTable {
    private:
      DenseMap<Value*, uint32_t> valueNumbering;
      DenseMap<Expression, uint32_t> expressionNumbering;
      AliasAnalysis* AA;
  
      uint32_t nextValueNumber;
    
      Expression::ExpressionOpcode getOpcode(BinaryOperator* BO);
      Expression::ExpressionOpcode getOpcode(CmpInst* C);
      Expression::ExpressionOpcode getOpcode(CastInst* C);
      Expression create_expression(BinaryOperator* BO);
      Expression create_expression(CmpInst* C);
      Expression create_expression(ShuffleVectorInst* V);
      Expression create_expression(ExtractElementInst* C);
      Expression create_expression(InsertElementInst* V);
      Expression create_expression(SelectInst* V);
      Expression create_expression(CastInst* C);
      Expression create_expression(GetElementPtrInst* G);
      Expression create_expression(CallInst* C);
    public:
      ValueTable() : nextValueNumber(1) { }
      uint32_t lookup_or_add(Value* V);
      uint32_t lookup(Value* V) const;
      void add(Value* V, uint32_t num);
      void clear();
      void erase(Value* v);
      unsigned size();
      void setAliasAnalysis(AliasAnalysis* A) { AA = A; }
      uint32_t hash_operand(Value* v);
  };
}

namespace llvm {
template <> struct DenseMapInfo<Expression> {
  static inline Expression getEmptyKey() {
    return Expression(Expression::EMPTY);
  }
  
  static inline Expression getTombstoneKey() {
    return Expression(Expression::TOMBSTONE);
  }
  
  static unsigned getHashValue(const Expression e) {
    unsigned hash = e.opcode;
    
    hash = e.firstVN + hash * 37;
    hash = e.secondVN + hash * 37;
    hash = e.thirdVN + hash * 37;
    
    hash = (unsigned)((uintptr_t)e.type >> 4) ^
            (unsigned)((uintptr_t)e.type >> 9) +
            hash * 37;
    
    for (SmallVector<uint32_t, 4>::const_iterator I = e.varargs.begin(),
         E = e.varargs.end(); I != E; ++I)
      hash = *I + hash * 37;
    
    hash = (unsigned)((uintptr_t)e.function >> 4) ^
            (unsigned)((uintptr_t)e.function >> 9) +
            hash * 37;
    
    return hash;
  }
  static bool isEqual(const Expression &LHS, const Expression &RHS) {
    return LHS == RHS;
  }
  static bool isPod() { return true; }
};
}

//===----------------------------------------------------------------------===//
//                     ValueTable Internal Functions
//===----------------------------------------------------------------------===//
Expression::ExpressionOpcode 
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

Expression::ExpressionOpcode ValueTable::getOpcode(CmpInst* C) {
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

Expression::ExpressionOpcode 
                             ValueTable::getOpcode(CastInst* C) {
  switch(C->getOpcode()) {
    case Instruction::Trunc:
      return Expression::TRUNC;
    case Instruction::ZExt:
      return Expression::ZEXT;
    case Instruction::SExt:
      return Expression::SEXT;
    case Instruction::FPToUI:
      return Expression::FPTOUI;
    case Instruction::FPToSI:
      return Expression::FPTOSI;
    case Instruction::UIToFP:
      return Expression::UITOFP;
    case Instruction::SIToFP:
      return Expression::SITOFP;
    case Instruction::FPTrunc:
      return Expression::FPTRUNC;
    case Instruction::FPExt:
      return Expression::FPEXT;
    case Instruction::PtrToInt:
      return Expression::PTRTOINT;
    case Instruction::IntToPtr:
      return Expression::INTTOPTR;
    case Instruction::BitCast:
      return Expression::BITCAST;
    
    // THIS SHOULD NEVER HAPPEN
    default:
      assert(0 && "Cast operator with unknown opcode?");
      return Expression::BITCAST;
  }
}

uint32_t ValueTable::hash_operand(Value* v) {
  if (CallInst* CI = dyn_cast<CallInst>(v))
    if (!AA->doesNotAccessMemory(CI))
      return nextValueNumber++;
  
  return lookup_or_add(v);
}

Expression ValueTable::create_expression(CallInst* C) {
  Expression e;
  
  e.type = C->getType();
  e.firstVN = 0;
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = C->getCalledFunction();
  e.opcode = Expression::CALL;
  
  for (CallInst::op_iterator I = C->op_begin()+1, E = C->op_end();
       I != E; ++I)
    e.varargs.push_back(hash_operand(*I));
  
  return e;
}

Expression ValueTable::create_expression(BinaryOperator* BO) {
  Expression e;
    
  e.firstVN = hash_operand(BO->getOperand(0));
  e.secondVN = hash_operand(BO->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = BO->getType();
  e.opcode = getOpcode(BO);
  
  return e;
}

Expression ValueTable::create_expression(CmpInst* C) {
  Expression e;
    
  e.firstVN = hash_operand(C->getOperand(0));
  e.secondVN = hash_operand(C->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = C->getType();
  e.opcode = getOpcode(C);
  
  return e;
}

Expression ValueTable::create_expression(CastInst* C) {
  Expression e;
    
  e.firstVN = hash_operand(C->getOperand(0));
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = 0;
  e.type = C->getType();
  e.opcode = getOpcode(C);
  
  return e;
}

Expression ValueTable::create_expression(ShuffleVectorInst* S) {
  Expression e;
    
  e.firstVN = hash_operand(S->getOperand(0));
  e.secondVN = hash_operand(S->getOperand(1));
  e.thirdVN = hash_operand(S->getOperand(2));
  e.function = 0;
  e.type = S->getType();
  e.opcode = Expression::SHUFFLE;
  
  return e;
}

Expression ValueTable::create_expression(ExtractElementInst* E) {
  Expression e;
    
  e.firstVN = hash_operand(E->getOperand(0));
  e.secondVN = hash_operand(E->getOperand(1));
  e.thirdVN = 0;
  e.function = 0;
  e.type = E->getType();
  e.opcode = Expression::EXTRACT;
  
  return e;
}

Expression ValueTable::create_expression(InsertElementInst* I) {
  Expression e;
    
  e.firstVN = hash_operand(I->getOperand(0));
  e.secondVN = hash_operand(I->getOperand(1));
  e.thirdVN = hash_operand(I->getOperand(2));
  e.function = 0;
  e.type = I->getType();
  e.opcode = Expression::INSERT;
  
  return e;
}

Expression ValueTable::create_expression(SelectInst* I) {
  Expression e;
    
  e.firstVN = hash_operand(I->getCondition());
  e.secondVN = hash_operand(I->getTrueValue());
  e.thirdVN = hash_operand(I->getFalseValue());
  e.function = 0;
  e.type = I->getType();
  e.opcode = Expression::SELECT;
  
  return e;
}

Expression ValueTable::create_expression(GetElementPtrInst* G) {
  Expression e;
    
  e.firstVN = hash_operand(G->getPointerOperand());
  e.secondVN = 0;
  e.thirdVN = 0;
  e.function = 0;
  e.type = G->getType();
  e.opcode = Expression::GEP;
  
  for (GetElementPtrInst::op_iterator I = G->idx_begin(), E = G->idx_end();
       I != E; ++I)
    e.varargs.push_back(hash_operand(*I));
  
  return e;
}

//===----------------------------------------------------------------------===//
//                     ValueTable External Functions
//===----------------------------------------------------------------------===//

/// lookup_or_add - Returns the value number for the specified value, assigning
/// it a new number if it did not have one before.
uint32_t ValueTable::lookup_or_add(Value* V) {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  
  if (CallInst* C = dyn_cast<CallInst>(V)) {
    if (AA->onlyReadsMemory(C)) { // includes doesNotAccessMemory
      Expression e = create_expression(C);
    
      DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
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
  } else if (BinaryOperator* BO = dyn_cast<BinaryOperator>(V)) {
    Expression e = create_expression(BO);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
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
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (ShuffleVectorInst* U = dyn_cast<ShuffleVectorInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (ExtractElementInst* U = dyn_cast<ExtractElementInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (InsertElementInst* U = dyn_cast<InsertElementInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (SelectInst* U = dyn_cast<SelectInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (CastInst* U = dyn_cast<CastInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
    if (EI != expressionNumbering.end()) {
      valueNumbering.insert(std::make_pair(V, EI->second));
      return EI->second;
    } else {
      expressionNumbering.insert(std::make_pair(e, nextValueNumber));
      valueNumbering.insert(std::make_pair(V, nextValueNumber));
      
      return nextValueNumber++;
    }
  } else if (GetElementPtrInst* U = dyn_cast<GetElementPtrInst>(V)) {
    Expression e = create_expression(U);
    
    DenseMap<Expression, uint32_t>::iterator EI = expressionNumbering.find(e);
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
uint32_t ValueTable::lookup(Value* V) const {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;
  else
    assert(0 && "Value not numbered?");
  
  return 0;
}

/// clear - Remove all entries from the ValueTable
void ValueTable::clear() {
  valueNumbering.clear();
  expressionNumbering.clear();
  nextValueNumber = 1;
}

/// erase - Remove a value from the value numbering
void ValueTable::erase(Value* V) {
  valueNumbering.erase(V);
}

//===----------------------------------------------------------------------===//
//                       ValueNumberedSet Class
//===----------------------------------------------------------------------===//
namespace {
class ValueNumberedSet {
  private:
    SmallPtrSet<Value*, 8> contents;
    BitVector numbers;
  public:
    ValueNumberedSet() { numbers.resize(1); }
    ValueNumberedSet(const ValueNumberedSet& other) {
      numbers = other.numbers;
      contents = other.contents;
    }
    
    typedef SmallPtrSet<Value*, 8>::iterator iterator;
    
    iterator begin() { return contents.begin(); }
    iterator end() { return contents.end(); }
    
    bool insert(Value* v) { return contents.insert(v); }
    void insert(iterator I, iterator E) { contents.insert(I, E); }
    void erase(Value* v) { contents.erase(v); }
    unsigned count(Value* v) { return contents.count(v); }
    size_t size() { return contents.size(); }
    
    void set(unsigned i)  {
      if (i >= numbers.size())
        numbers.resize(i+1);
      
      numbers.set(i);
    }
    
    void operator=(const ValueNumberedSet& other) {
      contents = other.contents;
      numbers = other.numbers;
    }
    
    void reset(unsigned i)  {
      if (i < numbers.size())
        numbers.reset(i);
    }
    
    bool test(unsigned i)  {
      if (i >= numbers.size())
        return false;
      
      return numbers.test(i);
    }
    
    void clear() {
      contents.clear();
      numbers.clear();
    }
};
}

//===----------------------------------------------------------------------===//
//                         GVN Pass
//===----------------------------------------------------------------------===//

namespace {

  class VISIBILITY_HIDDEN GVN : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    GVN() : FunctionPass((intptr_t)&ID) { }

  private:
    ValueTable VN;
    
    DenseMap<BasicBlock*, ValueNumberedSet> availableOut;
    
    typedef DenseMap<Value*, SmallPtrSet<Instruction*, 4> > PhiMapType;
    PhiMapType phiMap;
    
    
    // This transformation requires dominator postdominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<TargetData>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<MemoryDependenceAnalysis>();
      AU.addPreserved<TargetData>();
    }
  
    // Helper fuctions
    // FIXME: eliminate or document these better
    Value* find_leader(ValueNumberedSet& vals, uint32_t v) ;
    void val_insert(ValueNumberedSet& s, Value* v);
    bool processLoad(LoadInst* L,
                     DenseMap<Value*, LoadInst*>& lastLoad,
                     SmallVector<Instruction*, 4>& toErase);
    bool processInstruction(Instruction* I,
                            ValueNumberedSet& currAvail,
                            DenseMap<Value*, LoadInst*>& lastSeenLoad,
                            SmallVector<Instruction*, 4>& toErase);
    bool processNonLocalLoad(LoadInst* L,
                             SmallVector<Instruction*, 4>& toErase);
    bool processMemCpy(MemCpyInst* M, MemCpyInst* MDep,
                       SmallVector<Instruction*, 4>& toErase);
    bool performReturnSlotOptzn(MemCpyInst* cpy, CallInst* C,
                                SmallVector<Instruction*, 4>& toErase);
    Value *GetValueForBlock(BasicBlock *BB, LoadInst* orig,
                            DenseMap<BasicBlock*, Value*> &Phis,
                            bool top_level = false);
    void dump(DenseMap<BasicBlock*, Value*>& d);
    bool iterateOnFunction(Function &F);
    Value* CollapsePhi(PHINode* p);
    bool isSafeReplacement(PHINode* p, Instruction* inst);
  };
  
  char GVN::ID = 0;
  
}

// createGVNPass - The public interface to this file...
FunctionPass *llvm::createGVNPass() { return new GVN(); }

static RegisterPass<GVN> X("gvn",
                           "Global Value Numbering");

STATISTIC(NumGVNInstr, "Number of instructions deleted");
STATISTIC(NumGVNLoad, "Number of loads deleted");

/// find_leader - Given a set and a value number, return the first
/// element of the set with that value number, or 0 if no such element
/// is present
Value* GVN::find_leader(ValueNumberedSet& vals, uint32_t v) {
  if (!vals.test(v))
    return 0;
  
  for (ValueNumberedSet::iterator I = vals.begin(), E = vals.end();
       I != E; ++I)
    if (v == VN.lookup(*I))
      return *I;
  
  assert(0 && "No leader found, but present bit is set?");
  return 0;
}

/// val_insert - Insert a value into a set only if there is not a value
/// with the same value number already in the set
void GVN::val_insert(ValueNumberedSet& s, Value* v) {
  uint32_t num = VN.lookup(v);
  if (!s.test(num))
    s.insert(v);
}

void GVN::dump(DenseMap<BasicBlock*, Value*>& d) {
  printf("{\n");
  for (DenseMap<BasicBlock*, Value*>::iterator I = d.begin(),
       E = d.end(); I != E; ++I) {
    if (I->second == MemoryDependenceAnalysis::None)
      printf("None\n");
    else
      I->second->dump();
  }
  printf("}\n");
}

Value* GVN::CollapsePhi(PHINode* p) {
  DominatorTree &DT = getAnalysis<DominatorTree>();
  Value* constVal = p->hasConstantValue();
  
  if (constVal) {
    if (Instruction* inst = dyn_cast<Instruction>(constVal)) {
      if (DT.dominates(inst, p))
        if (isSafeReplacement(p, inst))
          return inst;
    } else {
      return constVal;
    }
  }
  
  return 0;
}

bool GVN::isSafeReplacement(PHINode* p, Instruction* inst) {
  if (!isa<PHINode>(inst))
    return true;
  
  for (Instruction::use_iterator UI = p->use_begin(), E = p->use_end();
       UI != E; ++UI)
    if (PHINode* use_phi = dyn_cast<PHINode>(UI))
      if (use_phi->getParent() == inst->getParent())
        return false;
  
  return true;
}

/// GetValueForBlock - Get the value to use within the specified basic block.
/// available values are in Phis.
Value *GVN::GetValueForBlock(BasicBlock *BB, LoadInst* orig,
                               DenseMap<BasicBlock*, Value*> &Phis,
                               bool top_level) { 
                                 
  // If we have already computed this value, return the previously computed val.
  DenseMap<BasicBlock*, Value*>::iterator V = Phis.find(BB);
  if (V != Phis.end() && !top_level) return V->second;
  
  BasicBlock* singlePred = BB->getSinglePredecessor();
  if (singlePred) {
    Value *ret = GetValueForBlock(singlePred, orig, Phis);
    Phis[BB] = ret;
    return ret;
  }
  // Otherwise, the idom is the loop, so we need to insert a PHI node.  Do so
  // now, then get values to fill in the incoming values for the PHI.
  PHINode *PN = new PHINode(orig->getType(), orig->getName()+".rle",
                            BB->begin());
  PN->reserveOperandSpace(std::distance(pred_begin(BB), pred_end(BB)));
  
  if (Phis.count(BB) == 0)
    Phis.insert(std::make_pair(BB, PN));
  
  // Fill in the incoming values for the block.
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    Value* val = GetValueForBlock(*PI, orig, Phis);
    
    PN->addIncoming(val, *PI);
  }
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  AA.copyValue(orig, PN);
  
  // Attempt to collapse PHI nodes that are trivially redundant
  Value* v = CollapsePhi(PN);
  if (v) {
    MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();

    MD.removeInstruction(PN);
    PN->replaceAllUsesWith(v);

    for (DenseMap<BasicBlock*, Value*>::iterator I = Phis.begin(),
         E = Phis.end(); I != E; ++I)
      if (I->second == PN)
        I->second = v;

    PN->eraseFromParent();

    Phis[BB] = v;

    return v;
  }

  // Cache our phi construction results
  phiMap[orig->getPointerOperand()].insert(PN);
  return PN;
}

/// processNonLocalLoad - Attempt to eliminate a load whose dependencies are
/// non-local by performing PHI construction.
bool GVN::processNonLocalLoad(LoadInst* L,
                              SmallVector<Instruction*, 4>& toErase) {
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  
  // Find the non-local dependencies of the load
  DenseMap<BasicBlock*, Value*> deps;
  MD.getNonLocalDependency(L, deps);
  
  DenseMap<BasicBlock*, Value*> repl;
  
  // Filter out useless results (non-locals, etc)
  for (DenseMap<BasicBlock*, Value*>::iterator I = deps.begin(), E = deps.end();
       I != E; ++I)
    if (I->second == MemoryDependenceAnalysis::None) {
      return false;
    } else if (I->second == MemoryDependenceAnalysis::NonLocal) {
      continue;
    } else if (StoreInst* S = dyn_cast<StoreInst>(I->second)) {
      if (S->getPointerOperand() == L->getPointerOperand())
        repl[I->first] = S->getOperand(0);
      else
        return false;
    } else if (LoadInst* LD = dyn_cast<LoadInst>(I->second)) {
      if (LD->getPointerOperand() == L->getPointerOperand())
        repl[I->first] = LD;
      else
        return false;
    } else {
      return false;
    }
  
  // Use cached PHI construction information from previous runs
  SmallPtrSet<Instruction*, 4>& p = phiMap[L->getPointerOperand()];
  for (SmallPtrSet<Instruction*, 4>::iterator I = p.begin(), E = p.end();
       I != E; ++I) {
    if ((*I)->getParent() == L->getParent()) {
      MD.removeInstruction(L);
      L->replaceAllUsesWith(*I);
      toErase.push_back(L);
      NumGVNLoad++;
      
      return true;
    } else {
      repl.insert(std::make_pair((*I)->getParent(), *I));
    }
  }
  
  // Perform PHI construction
  SmallPtrSet<BasicBlock*, 4> visited;
  Value* v = GetValueForBlock(L->getParent(), L, repl, true);
  
  MD.removeInstruction(L);
  L->replaceAllUsesWith(v);
  toErase.push_back(L);
  NumGVNLoad++;

  return true;
}

/// processLoad - Attempt to eliminate a load, first by eliminating it
/// locally, and then attempting non-local elimination if that fails.
bool GVN::processLoad(LoadInst* L,
                         DenseMap<Value*, LoadInst*>& lastLoad,
                         SmallVector<Instruction*, 4>& toErase) {
  if (L->isVolatile()) {
    lastLoad[L->getPointerOperand()] = L;
    return false;
  }
  
  Value* pointer = L->getPointerOperand();
  LoadInst*& last = lastLoad[pointer];
  
  // ... to a pointer that has been loaded from before...
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  bool removedNonLocal = false;
  Instruction* dep = MD.getDependency(L);
  if (dep == MemoryDependenceAnalysis::NonLocal &&
      L->getParent() != &L->getParent()->getParent()->getEntryBlock()) {
    removedNonLocal = processNonLocalLoad(L, toErase);
    
    if (!removedNonLocal)
      last = L;
    
    return removedNonLocal;
  }
  
  
  bool deletedLoad = false;
  
  // Walk up the dependency chain until we either find
  // a dependency we can use, or we can't walk any further
  while (dep != MemoryDependenceAnalysis::None &&
         dep != MemoryDependenceAnalysis::NonLocal &&
         (isa<LoadInst>(dep) || isa<StoreInst>(dep))) {
    // ... that depends on a store ...
    if (StoreInst* S = dyn_cast<StoreInst>(dep)) {
      if (S->getPointerOperand() == pointer) {
        // Remove it!
        MD.removeInstruction(L);
        
        L->replaceAllUsesWith(S->getOperand(0));
        toErase.push_back(L);
        deletedLoad = true;
        NumGVNLoad++;
      }
      
      // Whether we removed it or not, we can't
      // go any further
      break;
    } else if (!last) {
      // If we don't depend on a store, and we haven't
      // been loaded before, bail.
      break;
    } else if (dep == last) {
      // Remove it!
      MD.removeInstruction(L);
      
      L->replaceAllUsesWith(last);
      toErase.push_back(L);
      deletedLoad = true;
      NumGVNLoad++;
        
      break;
    } else {
      dep = MD.getDependency(L, dep);
    }
  }

  if (dep != MemoryDependenceAnalysis::None &&
      dep != MemoryDependenceAnalysis::NonLocal &&
      isa<AllocationInst>(dep)) {
    // Check that this load is actually from the
    // allocation we found
    Value* v = L->getOperand(0);
    while (true) {
      if (BitCastInst *BC = dyn_cast<BitCastInst>(v))
        v = BC->getOperand(0);
      else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(v))
        v = GEP->getOperand(0);
      else
        break;
    }
    if (v == dep) {
      // If this load depends directly on an allocation, there isn't
      // anything stored there; therefore, we can optimize this load
      // to undef.
      MD.removeInstruction(L);

      L->replaceAllUsesWith(UndefValue::get(L->getType()));
      toErase.push_back(L);
      deletedLoad = true;
      NumGVNLoad++;
    }
  }

  if (!deletedLoad)
    last = L;
  
  return deletedLoad;
}

/// isReturnSlotOptznProfitable - Determine if performing a return slot 
/// fusion with the slot dest is profitable
static bool isReturnSlotOptznProfitable(Value* dest, MemCpyInst* cpy) {
  // We currently consider it profitable if dest is otherwise dead.
  SmallVector<User*, 8> useList(dest->use_begin(), dest->use_end());
  while (!useList.empty()) {
    User* UI = useList.back();
    
    if (isa<GetElementPtrInst>(UI) || isa<BitCastInst>(UI)) {
      useList.pop_back();
      for (User::use_iterator I = UI->use_begin(), E = UI->use_end();
           I != E; ++I)
        useList.push_back(*I);
    } else if (UI == cpy)
      useList.pop_back();
    else
      return false;
  }
  
  return true;
}

/// performReturnSlotOptzn - takes a memcpy and a call that it depends on,
/// and checks for the possibility of a return slot optimization by having
/// the call write its result directly into the callees return parameter
/// rather than using memcpy
bool GVN::performReturnSlotOptzn(MemCpyInst* cpy, CallInst* C,
                                 SmallVector<Instruction*, 4>& toErase) {
  Value* cpyDest = cpy->getDest();
  Value* cpySrc = cpy->getSource();
  CallSite CS = CallSite::get(C);
  
  // Since this is a return slot optimization, we need to make sure that
  // the value being copied is, in fact, in a return slot.  We also need to
  // check that the return slot parameter is marked noalias, so that we can
  // be sure that changing it will not cause unexpected behavior changes due
  // to it being accessed through a global or another parameter.
  if (CS.arg_size() == 0 ||
      cpySrc != CS.getArgument(0) ||
      !CS.paramHasAttr(1, ParamAttr::NoAlias | ParamAttr::StructRet))
    return false;
  
  // We only perform the transformation if it will be profitable. 
  if (!isReturnSlotOptznProfitable(cpyDest, cpy))
    return false;
  
  // Check that something sneaky is not happening involving casting
  // return slot types around.
  if (CS.getArgument(0)->getType() != cpyDest->getType())
    return false;
  
  // We can only perform the transformation if the size of the memcpy
  // is constant and equal to the size of the structure.
  if (!isa<ConstantInt>(cpy->getLength()))
    return false;
  
  ConstantInt* cpyLength = cast<ConstantInt>(cpy->getLength());
  TargetData& TD = getAnalysis<TargetData>();
  if (TD.getTypeStoreSize(cpyDest->getType()) == cpyLength->getZExtValue())
    return false;
  
  // In addition to knowing that the call does not access the return slot
  // in some unexpected manner, which we derive from the noalias attribute,
  // we also need to know that it does not sneakily modify the destination
  // slot in the caller.  We don't have parameter attributes to go by
  // for this one, so we just rely on AA to figure it out for us.
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  if (AA.getModRefInfo(C, cpy->getRawDest(), cpyLength->getZExtValue()) !=
      AliasAnalysis::NoModRef)
    return false;
  
  // If all the checks have passed, then we're alright to do the transformation.
  CS.setArgument(0, cpyDest);
  
  // Drop any cached information about the call, because we may have changed
  // its dependence information by changing its parameter.
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  MD.dropInstruction(C);
  
  // Remove the memcpy
  toErase.push_back(cpy);
  
  return true;
}

/// processMemCpy - perform simplication of memcpy's.  If we have memcpy A which
/// copies X to Y, and memcpy B which copies Y to Z, then we can rewrite B to be
/// a memcpy from X to Z (or potentially a memmove, depending on circumstances).
///  This allows later passes to remove the first memcpy altogether.
bool GVN::processMemCpy(MemCpyInst* M, MemCpyInst* MDep,
                        SmallVector<Instruction*, 4>& toErase) {
  // We can only transforms memcpy's where the dest of one is the source of the
  // other
  if (M->getSource() != MDep->getDest())
    return false;
  
  // Second, the length of the memcpy's must be the same, or the preceeding one
  // must be larger than the following one.
  ConstantInt* C1 = dyn_cast<ConstantInt>(MDep->getLength());
  ConstantInt* C2 = dyn_cast<ConstantInt>(M->getLength());
  if (!C1 || !C2)
    return false;
  
  uint64_t CpySize = C1->getValue().getZExtValue();
  uint64_t DepSize = C2->getValue().getZExtValue();
  
  if (DepSize < CpySize)
    return false;
  
  // Finally, we have to make sure that the dest of the second does not
  // alias the source of the first
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  if (AA.alias(M->getRawDest(), CpySize, MDep->getRawSource(), DepSize) !=
      AliasAnalysis::NoAlias)
    return false;
  else if (AA.alias(M->getRawDest(), CpySize, M->getRawSource(), CpySize) !=
           AliasAnalysis::NoAlias)
    return false;
  else if (AA.alias(MDep->getRawDest(), DepSize, MDep->getRawSource(), DepSize)
           != AliasAnalysis::NoAlias)
    return false;
  
  // If all checks passed, then we can transform these memcpy's
  Function* MemCpyFun = Intrinsic::getDeclaration(
                                 M->getParent()->getParent()->getParent(),
                                 M->getIntrinsicID());
    
  std::vector<Value*> args;
  args.push_back(M->getRawDest());
  args.push_back(MDep->getRawSource());
  args.push_back(M->getLength());
  args.push_back(M->getAlignment());
  
  CallInst* C = new CallInst(MemCpyFun, args.begin(), args.end(), "", M);
  
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  if (MD.getDependency(C) == MDep) {
    MD.dropInstruction(M);
    toErase.push_back(M);
    return true;
  } else {
    MD.removeInstruction(C);
    toErase.push_back(C);
    return false;
  }
}

/// processInstruction - When calculating availability, handle an instruction
/// by inserting it into the appropriate sets
bool GVN::processInstruction(Instruction* I,
                                ValueNumberedSet& currAvail,
                                DenseMap<Value*, LoadInst*>& lastSeenLoad,
                                SmallVector<Instruction*, 4>& toErase) {
  if (LoadInst* L = dyn_cast<LoadInst>(I)) {
    return processLoad(L, lastSeenLoad, toErase);
  } else if (MemCpyInst* M = dyn_cast<MemCpyInst>(I)) {
    MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();

    // The are two possible optimizations we can do for memcpy:
    //   a) memcpy-memcpy xform which exposes redundance for DSE
    //   b) call-memcpy xform for sret return slot optimization
    Instruction* dep = MD.getDependency(M);
    if (dep == MemoryDependenceAnalysis::None ||
        dep == MemoryDependenceAnalysis::NonLocal)
      return false;
    if (isa<MemCpyInst>(dep))
      return processMemCpy(M, cast<MemCpyInst>(dep), toErase);
    if (CallInst* C = dyn_cast<CallInst>(dep))
      return performReturnSlotOptzn(M, C, toErase);
    return false;
  }
  
  unsigned num = VN.lookup_or_add(I);
  
  // Collapse PHI nodes
  if (PHINode* p = dyn_cast<PHINode>(I)) {
    Value* constVal = CollapsePhi(p);
    
    if (constVal) {
      for (PhiMapType::iterator PI = phiMap.begin(), PE = phiMap.end();
           PI != PE; ++PI)
        if (PI->second.count(p))
          PI->second.erase(p);
        
      p->replaceAllUsesWith(constVal);
      toErase.push_back(p);
    }
  // Perform value-number based elimination
  } else if (currAvail.test(num)) {
    Value* repl = find_leader(currAvail, num);
    
    if (CallInst* CI = dyn_cast<CallInst>(I)) {
      AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
      if (!AA.doesNotAccessMemory(CI)) {
        MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
        if (cast<Instruction>(repl)->getParent() != CI->getParent() ||
            MD.getDependency(CI) != MD.getDependency(cast<CallInst>(repl))) {
          // There must be an intervening may-alias store, so nothing from
          // this point on will be able to be replaced with the preceding call
          currAvail.erase(repl);
          currAvail.insert(I);
          
          return false;
        }
      }
    }
    
    // Remove it!
    MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
    MD.removeInstruction(I);
    
    VN.erase(I);
    I->replaceAllUsesWith(repl);
    toErase.push_back(I);
    return true;
  } else if (!I->isTerminator()) {
    currAvail.set(num);
    currAvail.insert(I);
  }
  
  return false;
}

// GVN::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GVN::runOnFunction(Function& F) {
  VN.setAliasAnalysis(&getAnalysis<AliasAnalysis>());
  
  bool changed = false;
  bool shouldContinue = true;
  
  while (shouldContinue) {
    shouldContinue = iterateOnFunction(F);
    changed |= shouldContinue;
  }
  
  return changed;
}


// GVN::iterateOnFunction - Executes one iteration of GVN
bool GVN::iterateOnFunction(Function &F) {
  // Clean out global sets from any previous functions
  VN.clear();
  availableOut.clear();
  phiMap.clear();
 
  bool changed_function = false;
  
  DominatorTree &DT = getAnalysis<DominatorTree>();   
  
  SmallVector<Instruction*, 4> toErase;
  
  // Top-down walk of the dominator tree
  for (df_iterator<DomTreeNode*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    
    // Get the set to update for this block
    ValueNumberedSet& currAvail = availableOut[DI->getBlock()];     
    DenseMap<Value*, LoadInst*> lastSeenLoad;
    
    BasicBlock* BB = DI->getBlock();
  
    // A block inherits AVAIL_OUT from its dominator
    if (DI->getIDom() != 0)
      currAvail = availableOut[DI->getIDom()->getBlock()];

    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ) {
      changed_function |= processInstruction(BI, currAvail,
                                             lastSeenLoad, toErase);
      
      NumGVNInstr += toErase.size();
      
      // Avoid iterator invalidation
      ++BI;

      for (SmallVector<Instruction*, 4>::iterator I = toErase.begin(),
           E = toErase.end(); I != E; ++I) {
        (*I)->eraseFromParent();
      }

      toErase.clear();
    }
  }
  
  return changed_function;
}

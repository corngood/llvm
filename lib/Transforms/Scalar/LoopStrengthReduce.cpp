//===- LoopStrengthReduce.cpp - Strength Reduce GEPs in Loops -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a strength reduction on array references inside loops that
// have as one or more of their components the loop induction variable.  This is
// accomplished by creating a new Value to hold the initial value of the array
// access for the first iteration, and then creating a new GEP instruction in
// the loop to increment the value by the appropriate amount.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumReduced ("loop-reduce", "Number of GEPs strength reduced");

  class GEPCache {
  public:
    GEPCache() : CachedPHINode(0), Map() {}

    GEPCache *get(Value *v) {
      std::map<Value *, GEPCache>::iterator I = Map.find(v);
      if (I == Map.end())
        I = Map.insert(std::pair<Value *, GEPCache>(v, GEPCache())).first;
      return &I->second;
    }

    PHINode *CachedPHINode;
    std::map<Value *, GEPCache> Map;
  };

  struct IVUse {
    /// Users - Keep track of all of the users of this stride as well as the
    /// initial value.
    std::vector<std::pair<SCEVHandle, Instruction*> > Users;
    std::vector<Instruction *> UserOperands;

    void addUser(SCEVHandle &SH, Instruction *U, Instruction *V) {
      Users.push_back(std::make_pair(SH, U));
      UserOperands.push_back(V);
    }
  };


  class LoopStrengthReduce : public FunctionPass {
    LoopInfo *LI;
    DominatorSet *DS;
    ScalarEvolution *SE;
    const TargetData *TD;
    const Type *UIntPtrTy;
    bool Changed;

    /// MaxTargetAMSize - This is the maximum power-of-two scale value that the
    /// target can handle for free with its addressing modes.
    unsigned MaxTargetAMSize;

    /// IVUsesByStride - Keep track of all uses of induction variables that we
    /// are interested in.  The key of the map is the stride of the access.
    std::map<Value*, IVUse> IVUsesByStride;

    /// CastedBasePointers - As we need to lower getelementptr instructions, we
    /// cast the pointer input to uintptr_t.  This keeps track of the casted
    /// values for the pointers we have processed so far.
    std::map<Value*, Value*> CastedBasePointers;

    /// DeadInsts - Keep track of instructions we may have made dead, so that
    /// we can remove them after we are done working.
    std::set<Instruction*> DeadInsts;
  public:
    LoopStrengthReduce(unsigned MTAMS = 1)
      : MaxTargetAMSize(MTAMS) {
    }

    virtual bool runOnFunction(Function &) {
      LI = &getAnalysis<LoopInfo>();
      DS = &getAnalysis<DominatorSet>();
      SE = &getAnalysis<ScalarEvolution>();
      TD = &getAnalysis<TargetData>();
      UIntPtrTy = TD->getIntPtrType();
      Changed = false;

      for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
        runOnLoop(*I);
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorSet>();
      AU.addRequired<TargetData>();
      AU.addRequired<ScalarEvolution>();
    }
  private:
    void runOnLoop(Loop *L);
    bool AddUsersIfInteresting(Instruction *I, Loop *L);
    void AnalyzeGetElementPtrUsers(GetElementPtrInst *GEP, Instruction *I,
                                   Loop *L);

    void StrengthReduceStridedIVUsers(Value *Stride, IVUse &Uses, Loop *L,
                                      bool isOnlyStride);

    void strengthReduceGEP(GetElementPtrInst *GEPI, Loop *L,
                           GEPCache* GEPCache,
                           Instruction *InsertBefore,
                           std::set<Instruction*> &DeadInsts);
    void DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts);
  };
  RegisterOpt<LoopStrengthReduce> X("loop-reduce",
                                    "Strength Reduce GEP Uses of Ind. Vars");
}

FunctionPass *llvm::createLoopStrengthReducePass(unsigned MaxTargetAMSize) {
  return new LoopStrengthReduce(MaxTargetAMSize);
}

/// DeleteTriviallyDeadInstructions - If any of the instructions is the
/// specified set are trivially dead, delete them and see if this makes any of
/// their operands subsequently dead.
void LoopStrengthReduce::
DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts) {
  while (!Insts.empty()) {
    Instruction *I = *Insts.begin();
    Insts.erase(Insts.begin());
    if (isInstructionTriviallyDead(I)) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *U = dyn_cast<Instruction>(I->getOperand(i)))
          Insts.insert(U);
      SE->deleteInstructionFromRecords(I);
      I->eraseFromParent();
      Changed = true;
    }
  }
}


/// CanReduceSCEV - Return true if we can strength reduce this scalar evolution
/// in the specified loop.
static bool CanReduceSCEV(const SCEVHandle &SH, Loop *L) {
  SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SH);
  if (!AddRec || AddRec->getLoop() != L) return false;

  // FIXME: Generalize to non-affine IV's.
  if (!AddRec->isAffine()) return false;

  // FIXME: generalize to IV's with more complex strides (must emit stride
  // expression outside of loop!)
  if (isa<SCEVConstant>(AddRec->getOperand(1)))
    return true;

  // We handle steps by unsigned values, because we know we won't have to insert
  // a cast for them.
  if (SCEVUnknown *SU = dyn_cast<SCEVUnknown>(AddRec->getOperand(1)))
    if (SU->getValue()->getType()->isUnsigned())
      return true;

  // Otherwise, no, we can't handle it yet.
  return false;
}


/// GetAdjustedIndex - Adjust the specified GEP sequential type index to match
/// the size of the pointer type, and scale it by the type size.
static SCEVHandle GetAdjustedIndex(const SCEVHandle &Idx, uint64_t TySize,
                                   const Type *UIntPtrTy) {
  SCEVHandle Result = Idx;
  if (Result->getType()->getUnsignedVersion() != UIntPtrTy) {
    if (UIntPtrTy->getPrimitiveSize() < Result->getType()->getPrimitiveSize())
      Result = SCEVTruncateExpr::get(Result, UIntPtrTy);
    else
      Result = SCEVZeroExtendExpr::get(Result, UIntPtrTy);
  }

  // This index is scaled by the type size being indexed.
  if (TySize != 1)
    Result = SCEVMulExpr::get(Result,
                              SCEVConstant::get(ConstantUInt::get(UIntPtrTy,
                                                                  TySize)));
  return Result;
}

/// AnalyzeGetElementPtrUsers - Analyze all of the users of the specified
/// getelementptr instruction, adding them to the IVUsesByStride table.  Note
/// that we only want to analyze a getelementptr instruction once, and it can
/// have multiple operands that are uses of the indvar (e.g. A[i][i]).  Because
/// of this, we only process a GEP instruction if its first recurrent operand is
/// "op", otherwise we will either have already processed it or we will sometime
/// later.
void LoopStrengthReduce::AnalyzeGetElementPtrUsers(GetElementPtrInst *GEP,
                                                   Instruction *Op, Loop *L) {
  // Analyze all of the subscripts of this getelementptr instruction, looking
  // for uses that are determined by the trip count of L.  First, skip all
  // operands the are not dependent on the IV.

  // Build up the base expression.  Insert an LLVM cast of the pointer to
  // uintptr_t first.
  Value *BasePtr;
  if (Constant *CB = dyn_cast<Constant>(GEP->getOperand(0)))
    BasePtr = ConstantExpr::getCast(CB, UIntPtrTy);
  else {
    Value *&BP = CastedBasePointers[GEP->getOperand(0)];
    if (BP == 0) {
      BasicBlock::iterator InsertPt;
      if (isa<Argument>(GEP->getOperand(0))) {
        InsertPt = GEP->getParent()->getParent()->begin()->begin();
      } else {
        InsertPt = cast<Instruction>(GEP->getOperand(0));
        if (InvokeInst *II = dyn_cast<InvokeInst>(GEP->getOperand(0)))
          InsertPt = II->getNormalDest()->begin();
        else
          ++InsertPt;
      }
      
      // Do not insert casts into the middle of PHI node blocks.
      while (isa<PHINode>(InsertPt)) ++InsertPt;
      
      BP = new CastInst(GEP->getOperand(0), UIntPtrTy,
                        GEP->getOperand(0)->getName(), InsertPt);
    }
    BasePtr = BP;
  }

  SCEVHandle Base = SCEVUnknown::get(BasePtr);

  gep_type_iterator GTI = gep_type_begin(GEP);
  unsigned i = 1;
  for (; GEP->getOperand(i) != Op; ++i, ++GTI) {
    // If this is a use of a recurrence that we can analyze, and it comes before
    // Op does in the GEP operand list, we will handle this when we process this
    // operand.
    if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
      const StructLayout *SL = TD->getStructLayout(STy);
      unsigned Idx = cast<ConstantUInt>(GEP->getOperand(i))->getValue();
      uint64_t Offset = SL->MemberOffsets[Idx];
      Base = SCEVAddExpr::get(Base, SCEVUnknown::getIntegerSCEV(Offset,
                                                                UIntPtrTy));
    } else {
      SCEVHandle Idx = SE->getSCEV(GEP->getOperand(i));

      // If this operand is reducible, and it's not the one we are looking at
      // currently, do not process the GEP at this time.
      if (CanReduceSCEV(Idx, L))
        return;
      Base = SCEVAddExpr::get(Base, GetAdjustedIndex(Idx,
                             TD->getTypeSize(GTI.getIndexedType()), UIntPtrTy));
    }
  }

  // Get the index, convert it to intptr_t.
  SCEVHandle GEPIndexExpr =
    GetAdjustedIndex(SE->getSCEV(Op), TD->getTypeSize(GTI.getIndexedType()),
                     UIntPtrTy);

  // Process all remaining subscripts in the GEP instruction.
  for (++i, ++GTI; i != GEP->getNumOperands(); ++i, ++GTI)
    if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
      const StructLayout *SL = TD->getStructLayout(STy);
      unsigned Idx = cast<ConstantUInt>(GEP->getOperand(i))->getValue();
      uint64_t Offset = SL->MemberOffsets[Idx];
      Base = SCEVAddExpr::get(Base, SCEVUnknown::getIntegerSCEV(Offset,
                                                                UIntPtrTy));
    } else {
      SCEVHandle Idx = SE->getSCEV(GEP->getOperand(i));
      if (CanReduceSCEV(Idx, L)) {   // Another IV subscript
        GEPIndexExpr = SCEVAddExpr::get(GEPIndexExpr,
                    GetAdjustedIndex(Idx, TD->getTypeSize(GTI.getIndexedType()),
                                   UIntPtrTy));
        assert(CanReduceSCEV(GEPIndexExpr, L) &&
               "Cannot reduce the sum of two reducible SCEV's??");
      } else {
        Base = SCEVAddExpr::get(Base, GetAdjustedIndex(Idx,
                             TD->getTypeSize(GTI.getIndexedType()), UIntPtrTy));
      }
    }

  assert(CanReduceSCEV(GEPIndexExpr, L) && "Non reducible idx??");

  // FIXME: If the base is not loop invariant, we currently cannot emit this.
  if (!Base->isLoopInvariant(L)) {
    DEBUG(std::cerr << "IGNORING GEP due to non-invaiant base: "
                    << *Base << "\n");
    return;
  }
  
  Base = SCEVAddExpr::get(Base, cast<SCEVAddRecExpr>(GEPIndexExpr)->getStart());
  SCEVHandle Stride = cast<SCEVAddRecExpr>(GEPIndexExpr)->getOperand(1);

  DEBUG(std::cerr << "GEP BASE  : " << *Base << "\n");
  DEBUG(std::cerr << "GEP STRIDE: " << *Stride << "\n");

  Value *Step = 0;   // Step of ISE.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(Stride))
    /// Always get the step value as an unsigned value.
    Step = ConstantExpr::getCast(SC->getValue(),
                               SC->getValue()->getType()->getUnsignedVersion());
  else
    Step = cast<SCEVUnknown>(Stride)->getValue();
  assert(Step->getType()->isUnsigned() && "Bad step value!");


  // Now that we know the base and stride contributed by the GEP instruction,
  // process all users.
  for (Value::use_iterator UI = GEP->use_begin(), E = GEP->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);

    // Do not infinitely recurse on PHI nodes.
    if (isa<PHINode>(User) && User->getParent() == L->getHeader())
      continue;

    // If this is an instruction defined in a nested loop, or outside this loop,
    // don't mess with it.
    if (LI->getLoopFor(User->getParent()) != L)
      continue;

    DEBUG(std::cerr << "FOUND USER: " << *User
          << "   OF STRIDE: " << *Step << " BASE = " << *Base << "\n");


    // Okay, we found a user that we cannot reduce.  Analyze the instruction
    // and decide what to do with it.
    IVUsesByStride[Step].addUser(Base, User, GEP);
  }
}

/// AddUsersIfInteresting - Inspect the specified instruction.  If it is a
/// reducible SCEV, recursively add its users to the IVUsesByStride set and
/// return true.  Otherwise, return false.
bool LoopStrengthReduce::AddUsersIfInteresting(Instruction *I, Loop *L) {
  if (I->getType() == Type::VoidTy) return false;
  SCEVHandle ISE = SE->getSCEV(I);
  if (!CanReduceSCEV(ISE, L)) return false;

  SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(ISE);
  SCEVHandle Start = AR->getStart();

  // Get the step value, canonicalizing to an unsigned integer type so that
  // lookups in the map will match.
  Value *Step = 0;   // Step of ISE.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(AR->getOperand(1)))
    /// Always get the step value as an unsigned value.
    Step = ConstantExpr::getCast(SC->getValue(),
                               SC->getValue()->getType()->getUnsignedVersion());
  else
    Step = cast<SCEVUnknown>(AR->getOperand(1))->getValue();
  assert(Step->getType()->isUnsigned() && "Bad step value!");

  std::set<GetElementPtrInst*> AnalyzedGEPs;

  for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;++UI){
    Instruction *User = cast<Instruction>(*UI);

    // Do not infinitely recurse on PHI nodes.
    if (isa<PHINode>(User) && User->getParent() == L->getHeader())
      continue;

    // If this is an instruction defined in a nested loop, or outside this loop,
    // don't mess with it.
    if (LI->getLoopFor(User->getParent()) != L)
      continue;

    // Next, see if this user is analyzable itself!
    if (!AddUsersIfInteresting(User, L)) {
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
        // If this is a getelementptr instruction, figure out what linear
        // expression of induction variable is actually being used.
        //
        if (AnalyzedGEPs.insert(GEP).second)   // Not already analyzed?
          AnalyzeGetElementPtrUsers(GEP, I, L);
      } else {
        DEBUG(std::cerr << "FOUND USER: " << *User
              << "   OF SCEV: " << *ISE << "\n");

        // Okay, we found a user that we cannot reduce.  Analyze the instruction
        // and decide what to do with it.
        IVUsesByStride[Step].addUser(Start, User, I);
      }
    }
  }
  return true;
}

namespace {
  /// BasedUser - For a particular base value, keep information about how we've
  /// partitioned the expression so far.
  struct BasedUser {
    /// Inst - The instruction using the induction variable.
    Instruction *Inst;

    /// Op - The value to replace with the EmittedBase.
    Value *Op;

    /// Imm - The immediate value that should be added to the base immediately
    /// before Inst, because it will be folded into the imm field of the
    /// instruction.
    SCEVHandle Imm;

    /// EmittedBase - The actual value* to use for the base value of this
    /// operation.  This is null if we should just use zero so far.
    Value *EmittedBase;

    BasedUser(Instruction *I, Value *V, const SCEVHandle &IMM)
      : Inst(I), Op(V), Imm(IMM), EmittedBase(0) {}


    // No need to compare these.
    bool operator<(const BasedUser &BU) const { return 0; }

    void dump() const;
  };
}

void BasedUser::dump() const {
  std::cerr << " Imm=" << *Imm;
  if (EmittedBase)
    std::cerr << "  EB=" << *EmittedBase;

  std::cerr << "   Inst: " << *Inst;
}

/// isTargetConstant - Return true if the following can be referenced by the
/// immediate field of a target instruction.
static bool isTargetConstant(const SCEVHandle &V) {

  // FIXME: Look at the target to decide if &GV is a legal constant immediate.
  if (isa<SCEVConstant>(V)) return true;

  return false;     // ENABLE this for x86

  if (SCEVUnknown *SU = dyn_cast<SCEVUnknown>(V))
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(SU->getValue()))
      if (CE->getOpcode() == Instruction::Cast)
        if (isa<GlobalValue>(CE->getOperand(0)))
          // FIXME: should check to see that the dest is uintptr_t!
          return true;
  return false;
}

/// GetImmediateValues - Look at Val, and pull out any additions of constants
/// that can fit into the immediate field of instructions in the target.
static SCEVHandle GetImmediateValues(SCEVHandle Val, bool isAddress) {
  if (!isAddress)
    return SCEVUnknown::getIntegerSCEV(0, Val->getType());
  if (isTargetConstant(Val))
    return Val;

  SCEVAddExpr *SAE = dyn_cast<SCEVAddExpr>(Val);
  if (SAE) {
    unsigned i = 0;
    for (; i != SAE->getNumOperands(); ++i)
      if (isTargetConstant(SAE->getOperand(i))) {
        SCEVHandle ImmVal = SAE->getOperand(i);

        // If there are any other immediates that we can handle here, pull them
        // out too.
        for (++i; i != SAE->getNumOperands(); ++i)
          if (isTargetConstant(SAE->getOperand(i)))
            ImmVal = SCEVAddExpr::get(ImmVal, SAE->getOperand(i));
        return ImmVal;
      }
  }

  return SCEVUnknown::getIntegerSCEV(0, Val->getType());
}

/// StrengthReduceStridedIVUsers - Strength reduce all of the users of a single
/// stride of IV.  All of the users may have different starting values, and this
/// may not be the only stride (we know it is if isOnlyStride is true).
void LoopStrengthReduce::StrengthReduceStridedIVUsers(Value *Stride,
                                                      IVUse &Uses, Loop *L,
                                                      bool isOnlyStride) {
  // Transform our list of users and offsets to a bit more complex table.  In
  // this new vector, the first entry for each element is the base of the
  // strided access, and the second is the BasedUser object for the use.  We
  // progressively move information from the first to the second entry, until we
  // eventually emit the object.
  std::vector<std::pair<SCEVHandle, BasedUser> > UsersToProcess;
  UsersToProcess.reserve(Uses.Users.size());

  SCEVHandle ZeroBase = SCEVUnknown::getIntegerSCEV(0,
                                              Uses.Users[0].first->getType());

  for (unsigned i = 0, e = Uses.Users.size(); i != e; ++i)
    UsersToProcess.push_back(std::make_pair(Uses.Users[i].first,
                                            BasedUser(Uses.Users[i].second,
                                                      Uses.UserOperands[i],
                                                      ZeroBase)));

  // First pass, figure out what we can represent in the immediate fields of
  // instructions.  If we can represent anything there, move it to the imm
  // fields of the BasedUsers.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i) {
    bool isAddress = isa<LoadInst>(UsersToProcess[i].second.Inst) ||
                     isa<StoreInst>(UsersToProcess[i].second.Inst);
    UsersToProcess[i].second.Imm = GetImmediateValues(UsersToProcess[i].first,
                                                      isAddress);
    UsersToProcess[i].first = SCEV::getMinusSCEV(UsersToProcess[i].first,
                                                 UsersToProcess[i].second.Imm);

    DEBUG(std::cerr << "BASE: " << *UsersToProcess[i].first);
    DEBUG(UsersToProcess[i].second.dump());
  }

  SCEVExpander Rewriter(*SE, *LI);
  BasicBlock  *Preheader = L->getLoopPreheader();
  Instruction *PreInsertPt = Preheader->getTerminator();
  Instruction *PhiInsertBefore = L->getHeader()->begin();

  assert(isa<PHINode>(PhiInsertBefore) &&
         "How could this loop have IV's without any phis?");
  PHINode *SomeLoopPHI = cast<PHINode>(PhiInsertBefore);
  assert(SomeLoopPHI->getNumIncomingValues() == 2 &&
         "This loop isn't canonicalized right");
  BasicBlock *LatchBlock =
   SomeLoopPHI->getIncomingBlock(SomeLoopPHI->getIncomingBlock(0) == Preheader);

  // FIXME: This loop needs increasing levels of intelligence.
  // STAGE 0: just emit everything as its own base.  <-- We are here
  // STAGE 1: factor out common vars from bases, and try and push resulting
  //          constants into Imm field.
  // STAGE 2: factor out large constants to try and make more constants
  //          acceptable for target loads and stores.
  std::sort(UsersToProcess.begin(), UsersToProcess.end());

  while (!UsersToProcess.empty()) {
    // Create a new Phi for this base, and stick it in the loop header.
    Value *Replaced = UsersToProcess.front().second.Op;
    const Type *ReplacedTy = Replaced->getType();
    PHINode *NewPHI = new PHINode(ReplacedTy, Replaced->getName()+".str",
                                  PhiInsertBefore);

    // Emit the initial base value into the loop preheader, and add it to the
    // Phi node.
    Value *BaseV = Rewriter.expandCodeFor(UsersToProcess.front().first,
                                          PreInsertPt, ReplacedTy);
    NewPHI->addIncoming(BaseV, Preheader);

    // Emit the increment of the base value before the terminator of the loop
    // latch block, and add it to the Phi node.
    SCEVHandle Inc = SCEVAddExpr::get(SCEVUnknown::get(NewPHI),
                                      SCEVUnknown::get(Stride));

    Value *IncV = Rewriter.expandCodeFor(Inc, LatchBlock->getTerminator(),
                                         ReplacedTy);
    IncV->setName(NewPHI->getName()+".inc");
    NewPHI->addIncoming(IncV, LatchBlock);

    // Emit the code to add the immediate offset to the Phi value, just before
    // the instruction that we identified as using this stride and base.
    // First, empty the SCEVExpander's expression map  so that we are guaranteed
    // to have the code emitted where we expect it.
    Rewriter.clear();
    SCEVHandle NewValSCEV = SCEVAddExpr::get(SCEVUnknown::get(NewPHI),
                                             UsersToProcess.front().second.Imm);
    Value *newVal = Rewriter.expandCodeFor(NewValSCEV,
                                           UsersToProcess.front().second.Inst,
                                           ReplacedTy);

    // Replace the use of the operand Value with the new Phi we just created.
    DEBUG(std::cerr << "REPLACING: " << *Replaced << "IN: " <<
          *UsersToProcess.front().second.Inst << "WITH: "<< *newVal << '\n');
    UsersToProcess.front().second.Inst->replaceUsesOfWith(Replaced, newVal);

    // Mark old value we replaced as possibly dead, so that it is elminated
    // if we just replaced the last use of that value.
    DeadInsts.insert(cast<Instruction>(Replaced));

    UsersToProcess.erase(UsersToProcess.begin());
    ++NumReduced;

    // TODO: Next, find out which base index is the most common, pull it out.
  }

  // IMPORTANT TODO: Figure out how to partition the IV's with this stride, but
  // different starting values, into different PHIs.

  // BEFORE writing this, it's probably useful to handle GEP's.

  // NOTE: pull all constants together, for REG+IMM addressing, include &GV in
  // 'IMM' if the target supports it.
}


void LoopStrengthReduce::runOnLoop(Loop *L) {
  // First step, transform all loops nesting inside of this loop.
  for (LoopInfo::iterator I = L->begin(), E = L->end(); I != E; ++I)
    runOnLoop(*I);

  // Next, find all uses of induction variables in this loop, and catagorize
  // them by stride.  Start by finding all of the PHI nodes in the header for
  // this loop.  If they are induction variables, inspect their uses.
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I)
    AddUsersIfInteresting(I, L);

  // If we have nothing to do, return.
  //if (IVUsesByStride.empty()) return;

  // FIXME: We can widen subreg IV's here for RISC targets.  e.g. instead of
  // doing computation in byte values, promote to 32-bit values if safe.

  // FIXME: Attempt to reuse values across multiple IV's.  In particular, we
  // could have something like "for(i) { foo(i*8); bar(i*16) }", which should be
  // codegened as "for (j = 0;; j+=8) { foo(j); bar(j+j); }" on X86/PPC.  Need
  // to be careful that IV's are all the same type.  Only works for intptr_t
  // indvars.

  // If we only have one stride, we can more aggressively eliminate some things.
  bool HasOneStride = IVUsesByStride.size() == 1;

  for (std::map<Value*, IVUse>::iterator SI = IVUsesByStride.begin(),
         E = IVUsesByStride.end(); SI != E; ++SI)
    StrengthReduceStridedIVUsers(SI->first, SI->second, L, HasOneStride);

  // Clean up after ourselves
  if (!DeadInsts.empty()) {
    DeleteTriviallyDeadInstructions(DeadInsts);

    BasicBlock::iterator I = L->getHeader()->begin();
    PHINode *PN;
    while ((PN = dyn_cast<PHINode>(I))) {
      ++I;  // Preincrement iterator to avoid invalidating it when deleting PN.
      
      // At this point, we know that we have killed one or more GEP instructions.
      // It is worth checking to see if the cann indvar is also dead, so that we
      // can remove it as well.  The requirements for the cann indvar to be
      // considered dead are:
      // 1. the cann indvar has one use
      // 2. the use is an add instruction
      // 3. the add has one use
      // 4. the add is used by the cann indvar
      // If all four cases above are true, then we can remove both the add and
      // the cann indvar.
      // FIXME: this needs to eliminate an induction variable even if it's being
      // compared against some value to decide loop termination.
      if (PN->hasOneUse()) {
        BinaryOperator *BO = dyn_cast<BinaryOperator>(*(PN->use_begin()));
        if (BO && BO->hasOneUse()) {
          if (PN == *(BO->use_begin())) {
            DeadInsts.insert(BO);
            // Break the cycle, then delete the PHI.
            PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
            SE->deleteInstructionFromRecords(PN);
            PN->eraseFromParent();
          }
        }
      }
    }
    DeleteTriviallyDeadInstructions(DeadInsts);
  }

  IVUsesByStride.clear();
  CastedBasePointers.clear();
  return;
}

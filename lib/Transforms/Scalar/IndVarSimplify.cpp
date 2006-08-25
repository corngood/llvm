//===- IndVarSimplify.cpp - Induction Variable Elimination ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into simpler forms suitable for subsequent
// analysis and transformation.
//
// This transformation makes the following changes to each loop with an
// identifiable induction variable:
//   1. All loops are transformed to have a SINGLE canonical induction variable
//      which starts at zero and steps by one.
//   2. The canonical induction variable is guaranteed to be the first PHI node
//      in the loop header block.
//   3. Any pointer arithmetic recurrences are raised to use array subscripts.
//
// If the trip count of a loop is computable, this pass also makes the following
// changes:
//   1. The exit condition for the loop is canonicalized to compare the
//      induction value against the exit value.  This turns loops like:
//        'for (i = 7; i*i < 1000; ++i)' into 'for (i = 0; i != 25; ++i)'
//   2. Any use outside of the loop of an expression derived from the indvar
//      is changed to compute the derived value outside of the loop, eliminating
//      the dependence on the exit value of the induction variable.  If the only
//      purpose of the loop is to compute the exit value of some derived
//      expression, this transformation will make the loop dead.
//
// This transformation should be followed by strength reduction after all of the
// desired loop transformations have been performed.  Additionally, on targets
// where it is profitable, the loop could be transformed to count down to zero
// (the "do loop" optimization).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumRemoved ("indvars", "Number of aux indvars removed");
  Statistic<> NumPointer ("indvars", "Number of pointer indvars promoted");
  Statistic<> NumInserted("indvars", "Number of canonical indvars added");
  Statistic<> NumReplaced("indvars", "Number of exit values replaced");
  Statistic<> NumLFTR    ("indvars", "Number of loop exit tests replaced");

  class IndVarSimplify : public FunctionPass {
    LoopInfo        *LI;
    ScalarEvolution *SE;
    bool Changed;
  public:
    virtual bool runOnFunction(Function &) {
      LI = &getAnalysis<LoopInfo>();
      SE = &getAnalysis<ScalarEvolution>();
      Changed = false;

      // Induction Variables live in the header nodes of loops
      for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
        runOnLoop(*I);
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<ScalarEvolution>();
      AU.addRequired<LoopInfo>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
      AU.setPreservesCFG();
    }
  private:
    void runOnLoop(Loop *L);
    void EliminatePointerRecurrence(PHINode *PN, BasicBlock *Preheader,
                                    std::set<Instruction*> &DeadInsts);
    void LinearFunctionTestReplace(Loop *L, SCEV *IterationCount,
                                   SCEVExpander &RW);
    void RewriteLoopExitValues(Loop *L);

    void DeleteTriviallyDeadInstructions(std::set<Instruction*> &Insts);
  };
  RegisterOpt<IndVarSimplify> X("indvars", "Canonicalize Induction Variables");
}

FunctionPass *llvm::createIndVarSimplifyPass() {
  return new IndVarSimplify();
}

/// DeleteTriviallyDeadInstructions - If any of the instructions is the
/// specified set are trivially dead, delete them and see if this makes any of
/// their operands subsequently dead.
void IndVarSimplify::
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


/// EliminatePointerRecurrence - Check to see if this is a trivial GEP pointer
/// recurrence.  If so, change it into an integer recurrence, permitting
/// analysis by the SCEV routines.
void IndVarSimplify::EliminatePointerRecurrence(PHINode *PN,
                                                BasicBlock *Preheader,
                                            std::set<Instruction*> &DeadInsts) {
  assert(PN->getNumIncomingValues() == 2 && "Noncanonicalized loop!");
  unsigned PreheaderIdx = PN->getBasicBlockIndex(Preheader);
  unsigned BackedgeIdx = PreheaderIdx^1;
  if (GetElementPtrInst *GEPI =
          dyn_cast<GetElementPtrInst>(PN->getIncomingValue(BackedgeIdx)))
    if (GEPI->getOperand(0) == PN) {
      assert(GEPI->getNumOperands() == 2 && "GEP types must match!");

      // Okay, we found a pointer recurrence.  Transform this pointer
      // recurrence into an integer recurrence.  Compute the value that gets
      // added to the pointer at every iteration.
      Value *AddedVal = GEPI->getOperand(1);

      // Insert a new integer PHI node into the top of the block.
      PHINode *NewPhi = new PHINode(AddedVal->getType(),
                                    PN->getName()+".rec", PN);
      NewPhi->addIncoming(Constant::getNullValue(NewPhi->getType()), Preheader);

      // Create the new add instruction.
      Value *NewAdd = BinaryOperator::createAdd(NewPhi, AddedVal,
                                                GEPI->getName()+".rec", GEPI);
      NewPhi->addIncoming(NewAdd, PN->getIncomingBlock(BackedgeIdx));

      // Update the existing GEP to use the recurrence.
      GEPI->setOperand(0, PN->getIncomingValue(PreheaderIdx));

      // Update the GEP to use the new recurrence we just inserted.
      GEPI->setOperand(1, NewAdd);

      // If the incoming value is a constant expr GEP, try peeling out the array
      // 0 index if possible to make things simpler.
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(GEPI->getOperand(0)))
        if (CE->getOpcode() == Instruction::GetElementPtr) {
          unsigned NumOps = CE->getNumOperands();
          assert(NumOps > 1 && "CE folding didn't work!");
          if (CE->getOperand(NumOps-1)->isNullValue()) {
            // Check to make sure the last index really is an array index.
            gep_type_iterator GTI = gep_type_begin(CE);
            for (unsigned i = 1, e = CE->getNumOperands()-1;
                 i != e; ++i, ++GTI)
              /*empty*/;
            if (isa<SequentialType>(*GTI)) {
              // Pull the last index out of the constant expr GEP.
              std::vector<Value*> CEIdxs(CE->op_begin()+1, CE->op_end()-1);
              Constant *NCE = ConstantExpr::getGetElementPtr(CE->getOperand(0),
                                                             CEIdxs);
              GetElementPtrInst *NGEPI =
                new GetElementPtrInst(NCE, Constant::getNullValue(Type::IntTy),
                                      NewAdd, GEPI->getName(), GEPI);
              GEPI->replaceAllUsesWith(NGEPI);
              GEPI->eraseFromParent();
              GEPI = NGEPI;
            }
          }
        }


      // Finally, if there are any other users of the PHI node, we must
      // insert a new GEP instruction that uses the pre-incremented version
      // of the induction amount.
      if (!PN->use_empty()) {
        BasicBlock::iterator InsertPos = PN; ++InsertPos;
        while (isa<PHINode>(InsertPos)) ++InsertPos;
        std::string Name = PN->getName(); PN->setName("");
        Value *PreInc =
          new GetElementPtrInst(PN->getIncomingValue(PreheaderIdx),
                                std::vector<Value*>(1, NewPhi), Name,
                                InsertPos);
        PN->replaceAllUsesWith(PreInc);
      }

      // Delete the old PHI for sure, and the GEP if its otherwise unused.
      DeadInsts.insert(PN);

      ++NumPointer;
      Changed = true;
    }
}

/// LinearFunctionTestReplace - This method rewrites the exit condition of the
/// loop to be a canonical != comparison against the incremented loop induction
/// variable.  This pass is able to rewrite the exit tests of any loop where the
/// SCEV analysis can determine a loop-invariant trip count of the loop, which
/// is actually a much broader range than just linear tests.
void IndVarSimplify::LinearFunctionTestReplace(Loop *L, SCEV *IterationCount,
                                               SCEVExpander &RW) {
  // Find the exit block for the loop.  We can currently only handle loops with
  // a single exit.
  std::vector<BasicBlock*> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() != 1) return;
  BasicBlock *ExitBlock = ExitBlocks[0];

  // Make sure there is only one predecessor block in the loop.
  BasicBlock *ExitingBlock = 0;
  for (pred_iterator PI = pred_begin(ExitBlock), PE = pred_end(ExitBlock);
       PI != PE; ++PI)
    if (L->contains(*PI)) {
      if (ExitingBlock == 0)
        ExitingBlock = *PI;
      else
        return;  // Multiple exits from loop to this block.
    }
  assert(ExitingBlock && "Loop info is broken");

  if (!isa<BranchInst>(ExitingBlock->getTerminator()))
    return;  // Can't rewrite non-branch yet
  BranchInst *BI = cast<BranchInst>(ExitingBlock->getTerminator());
  assert(BI->isConditional() && "Must be conditional to be part of loop!");

  std::set<Instruction*> InstructionsToDelete;
  if (Instruction *Cond = dyn_cast<Instruction>(BI->getCondition()))
    InstructionsToDelete.insert(Cond);

  // If the exiting block is not the same as the backedge block, we must compare
  // against the preincremented value, otherwise we prefer to compare against
  // the post-incremented value.
  BasicBlock *Header = L->getHeader();
  pred_iterator HPI = pred_begin(Header);
  assert(HPI != pred_end(Header) && "Loop with zero preds???");
  if (!L->contains(*HPI)) ++HPI;
  assert(HPI != pred_end(Header) && L->contains(*HPI) &&
         "No backedge in loop?");

  SCEVHandle TripCount = IterationCount;
  Value *IndVar;
  if (*HPI == ExitingBlock) {
    // The IterationCount expression contains the number of times that the
    // backedge actually branches to the loop header.  This is one less than the
    // number of times the loop executes, so add one to it.
    Constant *OneC = ConstantInt::get(IterationCount->getType(), 1);
    TripCount = SCEVAddExpr::get(IterationCount, SCEVUnknown::get(OneC));
    IndVar = L->getCanonicalInductionVariableIncrement();
  } else {
    // We have to use the preincremented value...
    IndVar = L->getCanonicalInductionVariable();
  }

  // Expand the code for the iteration count into the preheader of the loop.
  BasicBlock *Preheader = L->getLoopPreheader();
  Value *ExitCnt = RW.expandCodeFor(TripCount, Preheader->getTerminator(),
                                    IndVar->getType());

  // Insert a new setne or seteq instruction before the branch.
  Instruction::BinaryOps Opcode;
  if (L->contains(BI->getSuccessor(0)))
    Opcode = Instruction::SetNE;
  else
    Opcode = Instruction::SetEQ;

  Value *Cond = new SetCondInst(Opcode, IndVar, ExitCnt, "exitcond", BI);
  BI->setCondition(Cond);
  ++NumLFTR;
  Changed = true;

  DeleteTriviallyDeadInstructions(InstructionsToDelete);
}


/// RewriteLoopExitValues - Check to see if this loop has a computable
/// loop-invariant execution count.  If so, this means that we can compute the
/// final value of any expressions that are recurrent in the loop, and
/// substitute the exit values from the loop into any instructions outside of
/// the loop that use the final values of the current expressions.
void IndVarSimplify::RewriteLoopExitValues(Loop *L) {
  BasicBlock *Preheader = L->getLoopPreheader();

  // Scan all of the instructions in the loop, looking at those that have
  // extra-loop users and which are recurrences.
  SCEVExpander Rewriter(*SE, *LI);

  // We insert the code into the preheader of the loop if the loop contains
  // multiple exit blocks, or in the exit block if there is exactly one.
  BasicBlock *BlockToInsertInto;
  std::vector<BasicBlock*> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() == 1)
    BlockToInsertInto = ExitBlocks[0];
  else
    BlockToInsertInto = Preheader;
  BasicBlock::iterator InsertPt = BlockToInsertInto->begin();
  while (isa<PHINode>(InsertPt)) ++InsertPt;

  bool HasConstantItCount = isa<SCEVConstant>(SE->getIterationCount(L));

  std::set<Instruction*> InstructionsToDelete;

  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i)
    if (LI->getLoopFor(L->getBlocks()[i]) == L) {  // Not in a subloop...
      BasicBlock *BB = L->getBlocks()[i];
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
        if (I->getType()->isInteger()) {      // Is an integer instruction
          SCEVHandle SH = SE->getSCEV(I);
          if (SH->hasComputableLoopEvolution(L) ||    // Varies predictably
              HasConstantItCount) {
            // Find out if this predictably varying value is actually used
            // outside of the loop.  "extra" as opposed to "intra".
            std::vector<Instruction*> ExtraLoopUsers;
            for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
                 UI != E; ++UI) {
              Instruction *User = cast<Instruction>(*UI);
              if (!L->contains(User->getParent())) {
                // If this is a PHI node in the exit block and we're inserting,
                // into the exit block, it must have a single entry.  In this
                // case, we can't insert the code after the PHI and have the PHI
                // still use it.  Instead, don't insert the the PHI.
                if (PHINode *PN = dyn_cast<PHINode>(User)) {
                  // FIXME: This is a case where LCSSA pessimizes code, this
                  // should be fixed better.
                  if (PN->getNumOperands() == 2 && 
                      PN->getParent() == BlockToInsertInto)
                    continue;
                }
                ExtraLoopUsers.push_back(User);
              }
            }
            
            if (!ExtraLoopUsers.empty()) {
              // Okay, this instruction has a user outside of the current loop
              // and varies predictably in this loop.  Evaluate the value it
              // contains when the loop exits, and insert code for it.
              SCEVHandle ExitValue = SE->getSCEVAtScope(I, L->getParentLoop());
              if (!isa<SCEVCouldNotCompute>(ExitValue)) {
                Changed = true;
                ++NumReplaced;
                // Remember the next instruction.  The rewriter can move code
                // around in some cases.
                BasicBlock::iterator NextI = I; ++NextI;

                Value *NewVal = Rewriter.expandCodeFor(ExitValue, InsertPt,
                                                       I->getType());

                // Rewrite any users of the computed value outside of the loop
                // with the newly computed value.
                for (unsigned i = 0, e = ExtraLoopUsers.size(); i != e; ++i) {
                  PHINode* PN = dyn_cast<PHINode>(ExtraLoopUsers[i]);
                  if (PN && PN->getNumOperands() == 2 &&
                      !L->contains(PN->getParent())) {
                    // We're dealing with an LCSSA Phi.  Handle it specially.
                    Instruction* LCSSAInsertPt = BlockToInsertInto->begin();
                    
                    Instruction* NewInstr = dyn_cast<Instruction>(NewVal);
                    if (NewInstr && !isa<PHINode>(NewInstr) &&
                        !L->contains(NewInstr->getParent()))
                      for (unsigned j = 0; j < NewInstr->getNumOperands(); ++j){
                        Instruction* PredI = 
                                 dyn_cast<Instruction>(NewInstr->getOperand(j));
                        if (PredI && L->contains(PredI->getParent())) {
                          PHINode* NewLCSSA = new PHINode(PredI->getType(),
                                                    PredI->getName() + ".lcssa",
                                                    LCSSAInsertPt);
                          NewLCSSA->addIncoming(PredI, 
                                     BlockToInsertInto->getSinglePredecessor());
                        
                          NewInstr->replaceUsesOfWith(PredI, NewLCSSA);
                        }
                      }
                    
                    PN->replaceAllUsesWith(NewVal);
                    PN->eraseFromParent();
                  } else {
                    ExtraLoopUsers[i]->replaceUsesOfWith(I, NewVal);
                  }
                }

                // If this instruction is dead now, schedule it to be removed.
                if (I->use_empty())
                  InstructionsToDelete.insert(I);
                I = NextI;
                continue;  // Skip the ++I
              }
            }
          }
        }

        // Next instruction.  Continue instruction skips this.
        ++I;
      }
    }

  DeleteTriviallyDeadInstructions(InstructionsToDelete);
}


void IndVarSimplify::runOnLoop(Loop *L) {
  // First step.  Check to see if there are any trivial GEP pointer recurrences.
  // If there are, change them into integer recurrences, permitting analysis by
  // the SCEV routines.
  //
  BasicBlock *Header    = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();

  std::set<Instruction*> DeadInsts;
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    if (isa<PointerType>(PN->getType()))
      EliminatePointerRecurrence(PN, Preheader, DeadInsts);
  }

  if (!DeadInsts.empty())
    DeleteTriviallyDeadInstructions(DeadInsts);


  // Next, transform all loops nesting inside of this loop.
  for (LoopInfo::iterator I = L->begin(), E = L->end(); I != E; ++I)
    runOnLoop(*I);

  // Check to see if this loop has a computable loop-invariant execution count.
  // If so, this means that we can compute the final value of any expressions
  // that are recurrent in the loop, and substitute the exit values from the
  // loop into any instructions outside of the loop that use the final values of
  // the current expressions.
  //
  SCEVHandle IterationCount = SE->getIterationCount(L);
  if (!isa<SCEVCouldNotCompute>(IterationCount))
    RewriteLoopExitValues(L);

  // Next, analyze all of the induction variables in the loop, canonicalizing
  // auxillary induction variables.
  std::vector<std::pair<PHINode*, SCEVHandle> > IndVars;

  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    if (PN->getType()->isInteger()) {  // FIXME: when we have fast-math, enable!
      SCEVHandle SCEV = SE->getSCEV(PN);
      if (SCEV->hasComputableLoopEvolution(L))
        // FIXME: It is an extremely bad idea to indvar substitute anything more
        // complex than affine induction variables.  Doing so will put expensive
        // polynomial evaluations inside of the loop, and the str reduction pass
        // currently can only reduce affine polynomials.  For now just disable
        // indvar subst on anything more complex than an affine addrec.
        if (SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(SCEV))
          if (AR->isAffine())
            IndVars.push_back(std::make_pair(PN, SCEV));
    }
  }

  // If there are no induction variables in the loop, there is nothing more to
  // do.
  if (IndVars.empty()) {
    // Actually, if we know how many times the loop iterates, lets insert a
    // canonical induction variable to help subsequent passes.
    if (!isa<SCEVCouldNotCompute>(IterationCount)) {
      SCEVExpander Rewriter(*SE, *LI);
      Rewriter.getOrInsertCanonicalInductionVariable(L,
                                                     IterationCount->getType());
      LinearFunctionTestReplace(L, IterationCount, Rewriter);
    }
    return;
  }

  // Compute the type of the largest recurrence expression.
  //
  const Type *LargestType = IndVars[0].first->getType();
  bool DifferingSizes = false;
  for (unsigned i = 1, e = IndVars.size(); i != e; ++i) {
    const Type *Ty = IndVars[i].first->getType();
    DifferingSizes |= Ty->getPrimitiveSize() != LargestType->getPrimitiveSize();
    if (Ty->getPrimitiveSize() > LargestType->getPrimitiveSize())
      LargestType = Ty;
  }

  // Create a rewriter object which we'll use to transform the code with.
  SCEVExpander Rewriter(*SE, *LI);

  // Now that we know the largest of of the induction variables in this loop,
  // insert a canonical induction variable of the largest size.
  LargestType = LargestType->getUnsignedVersion();
  Value *IndVar = Rewriter.getOrInsertCanonicalInductionVariable(L,LargestType);
  ++NumInserted;
  Changed = true;

  if (!isa<SCEVCouldNotCompute>(IterationCount))
    LinearFunctionTestReplace(L, IterationCount, Rewriter);

  // Now that we have a canonical induction variable, we can rewrite any
  // recurrences in terms of the induction variable.  Start with the auxillary
  // induction variables, and recursively rewrite any of their uses.
  BasicBlock::iterator InsertPt = Header->begin();
  while (isa<PHINode>(InsertPt)) ++InsertPt;

  // If there were induction variables of other sizes, cast the primary
  // induction variable to the right size for them, avoiding the need for the
  // code evaluation methods to insert induction variables of different sizes.
  if (DifferingSizes) {
    bool InsertedSizes[17] = { false };
    InsertedSizes[LargestType->getPrimitiveSize()] = true;
    for (unsigned i = 0, e = IndVars.size(); i != e; ++i)
      if (!InsertedSizes[IndVars[i].first->getType()->getPrimitiveSize()]) {
        PHINode *PN = IndVars[i].first;
        InsertedSizes[PN->getType()->getPrimitiveSize()] = true;
        Instruction *New = new CastInst(IndVar,
                                        PN->getType()->getUnsignedVersion(),
                                        "indvar", InsertPt);
        Rewriter.addInsertedValue(New, SE->getSCEV(New));
      }
  }

  // If there were induction variables of other sizes, cast the primary
  // induction variable to the right size for them, avoiding the need for the
  // code evaluation methods to insert induction variables of different sizes.
  std::map<unsigned, Value*> InsertedSizes;
  while (!IndVars.empty()) {
    PHINode *PN = IndVars.back().first;
    Value *NewVal = Rewriter.expandCodeFor(IndVars.back().second, InsertPt,
                                           PN->getType());
    std::string Name = PN->getName();
    PN->setName("");
    NewVal->setName(Name);

    // Replace the old PHI Node with the inserted computation.
    PN->replaceAllUsesWith(NewVal);
    DeadInsts.insert(PN);
    IndVars.pop_back();
    ++NumRemoved;
    Changed = true;
  }

#if 0
  // Now replace all derived expressions in the loop body with simpler
  // expressions.
  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i)
    if (LI->getLoopFor(L->getBlocks()[i]) == L) {  // Not in a subloop...
      BasicBlock *BB = L->getBlocks()[i];
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        if (I->getType()->isInteger() &&      // Is an integer instruction
            !I->use_empty() &&
            !Rewriter.isInsertedInstruction(I)) {
          SCEVHandle SH = SE->getSCEV(I);
          Value *V = Rewriter.expandCodeFor(SH, I, I->getType());
          if (V != I) {
            if (isa<Instruction>(V)) {
              std::string Name = I->getName();
              I->setName("");
              V->setName(Name);
            }
            I->replaceAllUsesWith(V);
            DeadInsts.insert(I);
            ++NumRemoved;
            Changed = true;
          }
        }
    }
#endif

  DeleteTriviallyDeadInstructions(DeadInsts);
  
  assert(L->isLCSSAForm());
}

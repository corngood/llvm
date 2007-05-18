//===-- IfConversion.cpp - Machine code if conversion pass. ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine instruction level if-conversion pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ifconversion"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumIfConvBBs, "Number of if-converted blocks");

namespace {
  class IfConverter : public MachineFunctionPass {
    enum BBICKind {
      ICInvalid,       // BB data invalid.
      ICNotClassfied,  // BB data valid, but not classified.
      ICTriangle,      // BB is part of a triangle sub-CFG.
      ICDiamond,       // BB is part of a diamond sub-CFG.
      ICTriangleEntry, // BB is entry of a triangle sub-CFG.
      ICDiamondEntry   // BB is entry of a diamond sub-CFG.
    };

    /// BBInfo - One per MachineBasicBlock, this is used to cache the result
    /// if-conversion feasibility analysis. This includes results from
    /// TargetInstrInfo::AnalyzeBranch() (i.e. TBB, FBB, and Cond), and its
    /// classification, and common tail block of its successors (if it's a
    /// diamond shape), its size, whether it's predicable, and whether any
    /// instruction can clobber the 'would-be' predicate.
    struct BBInfo {
      BBICKind Kind;
      unsigned Size;
      bool isPredicable;
      bool ClobbersPred;
      MachineBasicBlock *BB;
      MachineBasicBlock *TrueBB;
      MachineBasicBlock *FalseBB;
      MachineBasicBlock *TailBB;
      std::vector<MachineOperand> Cond;
      BBInfo() : Kind(ICInvalid), Size(0), isPredicable(false),
                 ClobbersPred(false), BB(0), TrueBB(0), FalseBB(0), TailBB(0) {}
    };

    /// BBAnalysis - Results of if-conversion feasibility analysis indexed by
    /// basic block number.
    std::vector<BBInfo> BBAnalysis;

    const TargetLowering *TLI;
    const TargetInstrInfo *TII;
    bool MadeChange;
  public:
    static char ID;
    IfConverter() : MachineFunctionPass((intptr_t)&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "If converter"; }

  private:
    void StructuralAnalysis(MachineBasicBlock *BB);
    void FeasibilityAnalysis(BBInfo &BBI);
    void InitialFunctionAnalysis(MachineFunction &MF,
                                 std::vector<BBInfo*> &Candidates);
    bool IfConvertTriangle(BBInfo &BBI);
    bool IfConvertDiamond(BBInfo &BBI);
    void PredicateBlock(MachineBasicBlock *BB,
                        std::vector<MachineOperand> &Cond,
                        bool IgnoreTerm = false);
    void MergeBlocks(BBInfo &TrueBBI, BBInfo &FalseBBI);
  };
  char IfConverter::ID = 0;
}

FunctionPass *llvm::createIfConverterPass() { return new IfConverter(); }

bool IfConverter::runOnMachineFunction(MachineFunction &MF) {
  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  MF.RenumberBlocks();
  unsigned NumBBs = MF.getNumBlockIDs();
  BBAnalysis.resize(NumBBs);

  std::vector<BBInfo*> Candidates;
  // Do an intial analysis for each basic block and finding all the potential
  // candidates to perform if-convesion.
  InitialFunctionAnalysis(MF, Candidates);

  MadeChange = false;
  for (unsigned i = 0, e = Candidates.size(); i != e; ++i) {
    BBInfo &BBI = *Candidates[i];
    switch (BBI.Kind) {
    default: assert(false && "Unexpected!");
      break;
    case ICTriangleEntry:
      MadeChange |= IfConvertTriangle(BBI);
      break;
    case ICDiamondEntry:
      MadeChange |= IfConvertDiamond(BBI);
      break;
    }
  }

  BBAnalysis.clear();

  return MadeChange;
}

static MachineBasicBlock *findFalseBlock(MachineBasicBlock *BB,
                                         MachineBasicBlock *TrueBB) {
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         E = BB->succ_end(); SI != E; ++SI) {
    MachineBasicBlock *SuccBB = *SI;
    if (SuccBB != TrueBB)
      return SuccBB;
  }
  return NULL;
}

/// StructuralAnalysis - Analyze the structure of the sub-CFG starting from
/// the specified block. Record its successors and whether it looks like an
/// if-conversion candidate.
void IfConverter::StructuralAnalysis(MachineBasicBlock *BB) {
  BBInfo &BBI = BBAnalysis[BB->getNumber()];

  if (BBI.Kind != ICInvalid)
    return;  // Always analyzed.
  BBI.BB = BB;
  BBI.Size = std::distance(BB->begin(), BB->end());

  // Look for 'root' of a simple (non-nested) triangle or diamond.
  BBI.Kind = ICNotClassfied;
  if (TII->AnalyzeBranch(*BB, BBI.TrueBB, BBI.FalseBB, BBI.Cond)
      || !BBI.TrueBB || BBI.Cond.size() == 0)
    return;

  // Not a candidate if 'true' block has another predecessor.
  // FIXME: Use or'd predicate or predicated cmp.
  if (BBI.TrueBB->pred_size() > 1)
    return;

  // Not a candidate if 'true' block is going to be if-converted.
  StructuralAnalysis(BBI.TrueBB);
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  if (TrueBBI.Kind != ICNotClassfied)
    return;

  // TODO: Only handle very simple cases for now.
  if (TrueBBI.FalseBB || TrueBBI.Cond.size())
    return;

  // No false branch. This BB must end with a conditional branch and a
  // fallthrough.
  if (!BBI.FalseBB)
    BBI.FalseBB = findFalseBlock(BB, BBI.TrueBB);  
  assert(BBI.FalseBB && "Expected to find the fallthrough block!");

  // Not a candidate if 'false' block has another predecessor.
  // FIXME: Invert condition and swap 'true' / 'false' blocks?
  if (BBI.FalseBB->pred_size() > 1)
    return;

  // Not a candidate if 'false' block is going to be if-converted.
  StructuralAnalysis(BBI.FalseBB);
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  if (FalseBBI.Kind != ICNotClassfied)
    return;

  // TODO: Only handle very simple cases for now.
  if (FalseBBI.FalseBB || FalseBBI.Cond.size())
    return;

  if (TrueBBI.TrueBB && TrueBBI.TrueBB == BBI.FalseBB) {
    // Triangle:
    //   EBB
    //   | \_
    //   |  |
    //   | TBB
    //   |  /
    //   FBB
    BBI.Kind = ICTriangleEntry;
    TrueBBI.Kind = FalseBBI.Kind = ICTriangle;
  } else if (TrueBBI.TrueBB == FalseBBI.TrueBB) {
    // Diamond:
    //   EBB
    //   / \_
    //  |   |
    // TBB FBB
    //   \ /
    //  TailBB
    // Note MBB can be empty in case both TBB and FBB are return blocks.
    BBI.Kind = ICDiamondEntry;
    TrueBBI.Kind = FalseBBI.Kind = ICDiamond;
    BBI.TailBB = TrueBBI.TrueBB;
  }
  return;
}

/// FeasibilityAnalysis - Determine if the block is predicable. In most
/// cases, that means all the instructions in the block has M_PREDICABLE flag.
/// Also checks if the block contains any instruction which can clobber a
/// predicate (e.g. condition code register). If so, the block is not
/// predicable unless it's the last instruction. Note, this function assumes
/// all the terminator instructions can be converted or deleted so it ignore
/// them.
void IfConverter::FeasibilityAnalysis(BBInfo &BBI) {
  if (BBI.Size == 0 || BBI.Size > TLI->getIfCvtBlockSizeLimit())
    return;

  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    // TODO: check if instruction clobbers predicate.
    if (TII->isTerminatorInstr(I->getOpcode()))
      break;
    if (!I->isPredicable())
      return;
  }

  BBI.isPredicable = true;
}

/// InitialFunctionAnalysis - Analyze all blocks and find entries for all
/// if-conversion candidates.
void IfConverter::InitialFunctionAnalysis(MachineFunction &MF,
                                          std::vector<BBInfo*> &Candidates) {
  std::set<MachineBasicBlock*> Visited;
  MachineBasicBlock *Entry = MF.begin();
  for (df_ext_iterator<MachineBasicBlock*> DFI = df_ext_begin(Entry, Visited),
         E = df_ext_end(Entry, Visited); DFI != E; ++DFI) {
    MachineBasicBlock *BB = *DFI;
    StructuralAnalysis(BB);
    BBInfo &BBI = BBAnalysis[BB->getNumber()];
    if (BBI.Kind == ICTriangleEntry || BBI.Kind == ICDiamondEntry)
      Candidates.push_back(&BBI);
  }
}

/// TransferPreds - Transfer all the predecessors of FromBB to ToBB.
///
static void TransferPreds(MachineBasicBlock *ToBB, MachineBasicBlock *FromBB) {
   std::vector<MachineBasicBlock*> Preds(FromBB->pred_begin(),
                                         FromBB->pred_end());
    for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
      MachineBasicBlock *Pred = Preds[i];
      Pred->removeSuccessor(FromBB);
      if (!Pred->isSuccessor(ToBB))
        Pred->addSuccessor(ToBB);
    }
}

/// TransferSuccs - Transfer all the successors of FromBB to ToBB.
///
static void TransferSuccs(MachineBasicBlock *ToBB, MachineBasicBlock *FromBB) {
   std::vector<MachineBasicBlock*> Succs(FromBB->succ_begin(),
                                         FromBB->succ_end());
    for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
      MachineBasicBlock *Succ = Succs[i];
      FromBB->removeSuccessor(Succ);
      if (!ToBB->isSuccessor(Succ))
        ToBB->addSuccessor(Succ);
    }
}

/// IfConvertTriangle - If convert a triangle sub-CFG.
///
bool IfConverter::IfConvertTriangle(BBInfo &BBI) {
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  FeasibilityAnalysis(TrueBBI);

  if (TrueBBI.isPredicable) {
    BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];

    // Predicate the 'true' block after removing its branch.
    TrueBBI.Size -= TII->RemoveBranch(*BBI.TrueBB);
    PredicateBlock(BBI.TrueBB, BBI.Cond);

    // Join the 'true' and 'false' blocks by copying the instructions
    // from the 'false' block to the 'true' block.
    BBI.TrueBB->removeSuccessor(BBI.FalseBB);
    MergeBlocks(TrueBBI, FalseBBI);

    // Now merge the entry of the triangle with the true block.
    BBI.Size -= TII->RemoveBranch(*BBI.BB);
    MergeBlocks(BBI, TrueBBI);

    // Update block info.
    TrueBBI.Kind = ICInvalid;
    FalseBBI.Kind = ICInvalid;

    // FIXME: Must maintain LiveIns.
    NumIfConvBBs++;
    return true;
  }
  return false;
}

/// IfConvertDiamond - If convert a diamond sub-CFG.
///
bool IfConverter::IfConvertDiamond(BBInfo &BBI) {
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  FeasibilityAnalysis(TrueBBI);
  FeasibilityAnalysis(FalseBBI);

  if (TrueBBI.isPredicable && FalseBBI.isPredicable) {
    // Check the 'true' and 'false' blocks if either isn't ended with a branch.
    // Either the block fallthrough to another block or it ends with a
    // return. If it's the former, add a conditional branch to its successor.
    bool Proceed = true;
    bool TrueNeedCBr  = !TrueBBI.TrueBB && BBI.TrueBB->succ_size();
    bool FalseNeedCBr = !FalseBBI.TrueBB && BBI.FalseBB->succ_size();
    if (TrueNeedCBr && TrueBBI.ClobbersPred) {
      TrueBBI.isPredicable = false;
      Proceed = false;
    }
    if (FalseNeedCBr && FalseBBI.ClobbersPred) {
      FalseBBI.isPredicable = false;
      Proceed = false;
    }
    if (!Proceed)
      return false;

    std::vector<MachineInstr*> Dups;
    if (!BBI.TailBB) {
      // No common merge block. Check if the terminators (e.g. return) are
      // the same or predicable.
      MachineBasicBlock::iterator TT = BBI.TrueBB->getFirstTerminator();
      MachineBasicBlock::iterator FT = BBI.FalseBB->getFirstTerminator();
      while (TT != BBI.TrueBB->end() && FT != BBI.FalseBB->end()) {
        if (TT->isIdenticalTo(FT))
          Dups.push_back(TT);  // Will erase these later.
        else if (!TT->isPredicable() && !FT->isPredicable())
          return false; // Can't if-convert. Abort!
        ++TT;
        ++FT;
      }
      // One of the two pathes have more terminators, make sure they are all
      // predicable.
      while (TT != BBI.TrueBB->end())
        if (!TT->isPredicable())
          return false; // Can't if-convert. Abort!
      while (FT != BBI.FalseBB->end())
        if (!FT->isPredicable())
          return false; // Can't if-convert. Abort!
    }

    // Remove the duplicated instructions from the 'true' block.
    for (unsigned i = 0, e = Dups.size(); i != e; ++i) {
      Dups[i]->eraseFromParent();
      --TrueBBI.Size;
    }
    
    // Predicate the 'true' block after removing its branch.
    TrueBBI.Size -= TII->RemoveBranch(*BBI.TrueBB);
    PredicateBlock(BBI.TrueBB, BBI.Cond);

    // Add a conditional branch to 'true' successor if needed.
    if (TrueNeedCBr)
      TII->InsertBranch(*BBI.TrueBB, *BBI.TrueBB->succ_begin(), NULL, BBI.Cond);

    // Predicate the 'false' block.
    std::vector<MachineOperand> NewCond(BBI.Cond);
    TII->ReverseBranchCondition(NewCond);
    PredicateBlock(BBI.FalseBB, NewCond, true);

    // Add a conditional branch to 'false' successor if needed.
    if (FalseNeedCBr)
      TII->InsertBranch(*BBI.FalseBB, *BBI.FalseBB->succ_begin(), NULL,NewCond);

    // Merge the 'true' and 'false' blocks by copying the instructions
    // from the 'false' block to the 'true' block. That is, unless the true
    // block would clobber the predicate, in that case, do the opposite.
    BBInfo *CvtBBI;
    if (!TrueBBI.ClobbersPred) {
      MergeBlocks(TrueBBI, FalseBBI);
      CvtBBI = &TrueBBI;
    } else {
      MergeBlocks(FalseBBI, TrueBBI);
      CvtBBI = &FalseBBI;
    }

    // Remove the conditional branch from entry to the blocks.
    BBI.Size -= TII->RemoveBranch(*BBI.BB);

    // Merge the combined block into the entry of the diamond if the entry
    // block is its only predecessor. Otherwise, insert an unconditional
    // branch from entry to the if-converted block.
    if (CvtBBI->BB->pred_size() == 1) {
      BBI.BB->removeSuccessor(CvtBBI->BB);
      MergeBlocks(BBI, *CvtBBI);
      CvtBBI = &BBI;
    } else {
      std::vector<MachineOperand> NoCond;
      TII->InsertBranch(*BBI.BB, CvtBBI->BB, NULL, NoCond);
    }

    // If the if-converted block fallthrough into the tail block, then
    // fold the tail block in as well.
    if (BBI.TailBB && CvtBBI->BB->succ_size() == 1) {
      CvtBBI->Size -= TII->RemoveBranch(*CvtBBI->BB);
      CvtBBI->BB->removeSuccessor(BBI.TailBB);
      BBInfo TailBBI = BBAnalysis[BBI.TailBB->getNumber()];
      MergeBlocks(*CvtBBI, TailBBI);
      TailBBI.Kind = ICInvalid;
    }

    // Update block info.
    TrueBBI.Kind = ICInvalid;
    FalseBBI.Kind = ICInvalid;

    // FIXME: Must maintain LiveIns.
    NumIfConvBBs += 2;
    return true;
  }
  return false;
}

/// PredicateBlock - Predicate every instruction in the block with the specified
/// condition. If IgnoreTerm is true, skip over all terminator instructions.
void IfConverter::PredicateBlock(MachineBasicBlock *BB,
                                 std::vector<MachineOperand> &Cond,
                                 bool IgnoreTerm) {
  for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
       I != E; ++I) {
    if (IgnoreTerm && TII->isTerminatorInstr(I->getOpcode()))
      continue;
    if (!TII->PredicateInstruction(&*I, Cond)) {
      cerr << "Unable to predication " << *I << "!\n";
      abort();
    }
  }
}

/// MergeBlocks - Move all instructions from FromBB to the end of ToBB.
///
void IfConverter::MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI) {
  ToBBI.BB->splice(ToBBI.BB->end(),
                   FromBBI.BB, FromBBI.BB->begin(), FromBBI.BB->end());
  TransferPreds(ToBBI.BB, FromBBI.BB);
  TransferSuccs(ToBBI.BB, FromBBI.BB);
  ToBBI.Size += FromBBI.Size;
  FromBBI.Size = 0;
}

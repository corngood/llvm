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

#define DEBUG_TYPE "ifcvt"
#include "llvm/Function.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

namespace {
  // Hidden options for help debugging.
  cl::opt<int> IfCvtFnStart("ifcvt-fn-start", cl::init(-1), cl::Hidden);
  cl::opt<int> IfCvtFnStop("ifcvt-fn-stop", cl::init(-1), cl::Hidden);
  cl::opt<int> IfCvtLimit("ifcvt-limit", cl::init(-1), cl::Hidden);
  cl::opt<bool> DisableSimple("disable-ifcvt-simple", 
                              cl::init(false), cl::Hidden);
  cl::opt<bool> DisableSimpleF("disable-ifcvt-simple-false", 
                               cl::init(false), cl::Hidden);
  cl::opt<bool> DisableTriangle("disable-ifcvt-triangle", 
                                cl::init(false), cl::Hidden);
  cl::opt<bool> DisableTriangleR("disable-ifcvt-triangle-rev", 
                                 cl::init(false), cl::Hidden);
  cl::opt<bool> DisableTriangleF("disable-ifcvt-triangle-false", 
                                 cl::init(false), cl::Hidden);
  cl::opt<bool> DisableTriangleFR("disable-ifcvt-triangle-false-rev", 
                                  cl::init(false), cl::Hidden);
  cl::opt<bool> DisableDiamond("disable-ifcvt-diamond", 
                               cl::init(false), cl::Hidden);
}

STATISTIC(NumSimple,       "Number of simple if-conversions performed");
STATISTIC(NumSimpleFalse,  "Number of simple (F) if-conversions performed");
STATISTIC(NumTriangle,     "Number of triangle if-conversions performed");
STATISTIC(NumTriangleRev,  "Number of triangle (R) if-conversions performed");
STATISTIC(NumTriangleFalse,"Number of triangle (F) if-conversions performed");
STATISTIC(NumTriangleFRev, "Number of triangle (F/R) if-conversions performed");
STATISTIC(NumDiamonds,     "Number of diamond if-conversions performed");
STATISTIC(NumIfConvBBs,    "Number of if-converted blocks");
STATISTIC(NumDupBBs,       "Number of duplicated blocks");

namespace {
  class IfConverter : public MachineFunctionPass {
    enum IfcvtKind {
      ICNotClassfied,  // BB data valid, but not classified.
      ICSimpleFalse,   // Same as ICSimple, but on the false path.
      ICSimple,        // BB is entry of an one split, no rejoin sub-CFG.
      ICTriangleFRev,  // Same as ICTriangleFalse, but false path rev condition.
      ICTriangleRev,   // Same as ICTriangle, but true path rev condition.
      ICTriangleFalse, // Same as ICTriangle, but on the false path.
      ICTriangle,      // BB is entry of a triangle sub-CFG.
      ICDiamond        // BB is entry of a diamond sub-CFG.
    };

    /// BBInfo - One per MachineBasicBlock, this is used to cache the result
    /// if-conversion feasibility analysis. This includes results from
    /// TargetInstrInfo::AnalyzeBranch() (i.e. TBB, FBB, and Cond), and its
    /// classification, and common tail block of its successors (if it's a
    /// diamond shape), its size, whether it's predicable, and whether any
    /// instruction can clobber the 'would-be' predicate.
    ///
    /// IsDone          - True if BB is not to be considered for ifcvt.
    /// IsBeingAnalyzed - True if BB is currently being analyzed.
    /// IsAnalyzed      - True if BB has been analyzed (info is still valid).
    /// IsEnqueued      - True if BB has been enqueued to be ifcvt'ed.
    /// IsBrAnalyzable  - True if AnalyzeBranch() returns false.
    /// HasFallThrough  - True if BB may fallthrough to the following BB.
    /// IsUnpredicable  - True if BB is known to be unpredicable.
    /// ClobbersPredicate- True if BB would modify the predicate (e.g. has
    ///                   cmp, call, etc.)
    /// NonPredSize     - Number of non-predicated instructions.
    /// BB              - Corresponding MachineBasicBlock.
    /// TrueBB / FalseBB- See AnalyzeBranch().
    /// BrCond          - Conditions for end of block conditional branches.
    /// Predicate       - Predicate used in the BB.
    struct BBInfo {
      bool IsDone          : 1;
      bool IsBeingAnalyzed : 1;
      bool IsAnalyzed      : 1;
      bool IsEnqueued      : 1;
      bool IsBrAnalyzable  : 1;
      bool HasFallThrough  : 1;
      bool IsUnpredicable  : 1;
      bool CannotBeCopied  : 1;
      bool ClobbersPred    : 1;
      unsigned NonPredSize;
      MachineBasicBlock *BB;
      MachineBasicBlock *TrueBB;
      MachineBasicBlock *FalseBB;
      std::vector<MachineOperand> BrCond;
      std::vector<MachineOperand> Predicate;
      BBInfo() : IsDone(false), IsBeingAnalyzed(false),
                 IsAnalyzed(false), IsEnqueued(false), IsBrAnalyzable(false),
                 HasFallThrough(false), IsUnpredicable(false),
                 CannotBeCopied(false), ClobbersPred(false), NonPredSize(0),
                 BB(0), TrueBB(0), FalseBB(0) {}
    };

    /// IfcvtToken - Record information about pending if-conversions to attemp:
    /// BBI             - Corresponding BBInfo.
    /// Kind            - Type of block. See IfcvtKind.
    /// NeedSubsumsion  - True if the to be predicated BB has already been
    ///                   predicated.
    /// Duplicates      - Number of instructions that would be duplicated due
    ///                   to this if-conversion.
    struct IfcvtToken {
      BBInfo &BBI;
      IfcvtKind Kind;
      bool NeedSubsumsion;
      unsigned Duplicates;
      IfcvtToken(BBInfo &b, IfcvtKind k, bool s, unsigned d)
        : BBI(b), Kind(k), NeedSubsumsion(s), Duplicates(d) {}
    };

    /// Roots - Basic blocks that do not have successors. These are the starting
    /// points of Graph traversal.
    std::vector<MachineBasicBlock*> Roots;

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
    bool ReverseBranchCondition(BBInfo &BBI);
    bool ValidSimple(BBInfo &TrueBBI, unsigned &Dups) const;
    bool ValidTriangle(BBInfo &TrueBBI, BBInfo &FalseBBI,
                       bool FalseBranch, unsigned &Dups) const;
    bool ValidDiamond(BBInfo &TrueBBI, BBInfo &FalseBBI) const;
    void ScanInstructions(BBInfo &BBI);
    BBInfo &AnalyzeBlock(MachineBasicBlock *BB,
                         std::vector<IfcvtToken*> &Tokens);
    bool FeasibilityAnalysis(BBInfo &BBI, std::vector<MachineOperand> &Cond,
                             bool isTriangle = false, bool RevBranch = false);
    bool AnalyzeBlocks(MachineFunction &MF,
                       std::vector<IfcvtToken*> &Tokens);
    void ReTryPreds(MachineBasicBlock *BB);
    void RemoveExtraEdges(BBInfo &BBI);
    bool IfConvertSimple(BBInfo &BBI, IfcvtKind Kind);
    bool IfConvertTriangle(BBInfo &BBI, IfcvtKind Kind);
    bool IfConvertDiamond(BBInfo &BBI, IfcvtKind Kind);
    void PredicateBlock(BBInfo &BBI,
                        std::vector<MachineOperand> &Cond,
                        bool IgnoreTerm = false);
    void CopyAndPredicateBlock(BBInfo &ToBBI, BBInfo &FromBBI,
                               std::vector<MachineOperand> &Cond,
                               bool IgnoreBr = false);
    void MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI);

    // blockAlwaysFallThrough - Block ends without a terminator.
    bool blockAlwaysFallThrough(BBInfo &BBI) const {
      return BBI.IsBrAnalyzable && BBI.TrueBB == NULL;
    }

    // IfcvtTokenCmp - Used to sort if-conversion candidates.
    static bool IfcvtTokenCmp(IfcvtToken *C1, IfcvtToken *C2) {
      // Favors subsumsion.
      if (C1->NeedSubsumsion == false && C2->NeedSubsumsion == true)
        return true;
      else if (C1->NeedSubsumsion == C2->NeedSubsumsion) {
        if (C1->Duplicates > C2->Duplicates)
          return true;
        else if (C1->Duplicates == C2->Duplicates) {
          // Favors diamond over triangle, etc.
          if ((unsigned)C1->Kind < (unsigned)C2->Kind)
            return true;
          else if (C1->Kind == C2->Kind)
            return C1->BBI.BB->getNumber() < C2->BBI.BB->getNumber();
        }
      }
      return false;
    }
  };

  char IfConverter::ID = 0;
}

FunctionPass *llvm::createIfConverterPass() { return new IfConverter(); }

bool IfConverter::runOnMachineFunction(MachineFunction &MF) {
  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  static int FnNum = -1;
  DOUT << "\nIfcvt: function (" << ++FnNum <<  ") \'"
       << MF.getFunction()->getName() << "\'";

  if (FnNum < IfCvtFnStart || (IfCvtFnStop != -1 && FnNum > IfCvtFnStop)) {
    DOUT << " skipped\n";
    return false;
  }
  DOUT << "\n";

  MF.RenumberBlocks();
  BBAnalysis.resize(MF.getNumBlockIDs());

  // Look for root nodes, i.e. blocks without successors.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    if (I->succ_size() == 0)
      Roots.push_back(I);

  std::vector<IfcvtToken*> Tokens;
  MadeChange = false;
  unsigned NumIfCvts = NumSimple + NumSimpleFalse + NumTriangle +
    NumTriangleRev + NumTriangleFalse + NumTriangleFRev + NumDiamonds;
  while (IfCvtLimit == -1 || (int)NumIfCvts < IfCvtLimit) {
    // Do an intial analysis for each basic block and finding all the potential
    // candidates to perform if-convesion.
    bool Change = AnalyzeBlocks(MF, Tokens);
    while (!Tokens.empty()) {
      IfcvtToken *Token = Tokens.back();
      Tokens.pop_back();
      BBInfo &BBI = Token->BBI;
      IfcvtKind Kind = Token->Kind;

      // If the block has been evicted out of the queue or it has already been
      // marked dead (due to it being predicated), then skip it.
      if (BBI.IsDone)
        BBI.IsEnqueued = false;
      if (!BBI.IsEnqueued)
        continue;

      BBI.IsEnqueued = false;

      bool RetVal = false;
      switch (Kind) {
      default: assert(false && "Unexpected!");
        break;
      case ICSimple:
      case ICSimpleFalse: {
        bool isFalse = Kind == ICSimpleFalse;
        if ((isFalse && DisableSimpleF) || (!isFalse && DisableSimple)) break;
        DOUT << "Ifcvt (Simple" << (Kind == ICSimpleFalse ? " false" :"")
             << "): BB#" << BBI.BB->getNumber() << " ("
             << ((Kind == ICSimpleFalse)
                 ? BBI.FalseBB->getNumber()
                 : BBI.TrueBB->getNumber()) << ") ";
        RetVal = IfConvertSimple(BBI, Kind);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal)
          if (isFalse) NumSimpleFalse++;
          else         NumSimple++;
       break;
      }
      case ICTriangle:
      case ICTriangleRev:
      case ICTriangleFalse:
      case ICTriangleFRev: {
        bool isFalse = Kind == ICTriangleFalse;
        bool isRev   = (Kind == ICTriangleRev || Kind == ICTriangleFRev);
        if (DisableTriangle && !isFalse && !isRev) break;
        if (DisableTriangleR && !isFalse && isRev) break;
        if (DisableTriangleF && isFalse && !isRev) break;
        if (DisableTriangleFR && isFalse && isRev) break;
        DOUT << "Ifcvt (Triangle";
        if (isFalse)
          DOUT << " false";
        if (isRev)
          DOUT << " rev";
        DOUT << "): BB#" << BBI.BB->getNumber() << " (T:"
             << BBI.TrueBB->getNumber() << ",F:"
             << BBI.FalseBB->getNumber() << ") ";
        RetVal = IfConvertTriangle(BBI, Kind);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal) {
          if (isFalse) {
            if (isRev) NumTriangleFRev++;
            else       NumTriangleFalse++;
          } else {
            if (isRev) NumTriangleRev++;
            else       NumTriangle++;
          }
        }
        break;
      }
      case ICDiamond: {
        if (DisableDiamond) break;
        DOUT << "Ifcvt (Diamond): BB#" << BBI.BB->getNumber() << " (T:"
             << BBI.TrueBB->getNumber() << ",F:"
             << BBI.FalseBB->getNumber() << ") ";
        RetVal = IfConvertDiamond(BBI, Kind);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal) NumDiamonds++;
        break;
      }
      }

      Change |= RetVal;

      NumIfCvts = NumSimple + NumSimpleFalse + NumTriangle + NumTriangleRev +
        NumTriangleFalse + NumTriangleFRev + NumDiamonds;
      if (IfCvtLimit != -1 && (int)NumIfCvts >= IfCvtLimit)
        break;
    }

    if (!Change)
      break;
    MadeChange |= Change;
  }

  // Delete tokens in case of early exit.
  while (!Tokens.empty()) {
    IfcvtToken *Token = Tokens.back();
    Tokens.pop_back();
    delete Token;
  }

  Tokens.clear();
  Roots.clear();
  BBAnalysis.clear();

  return MadeChange;
}

/// findFalseBlock - BB has a fallthrough. Find its 'false' successor given
/// its 'true' successor.
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

/// ReverseBranchCondition - Reverse the condition of the end of the block
/// branchs. Swap block's 'true' and 'false' successors.
bool IfConverter::ReverseBranchCondition(BBInfo &BBI) {
  if (!TII->ReverseBranchCondition(BBI.BrCond)) {
    TII->RemoveBranch(*BBI.BB);
    TII->InsertBranch(*BBI.BB, BBI.FalseBB, BBI.TrueBB, BBI.BrCond);
    std::swap(BBI.TrueBB, BBI.FalseBB);
    return true;
  }
  return false;
}

/// getNextBlock - Returns the next block in the function blocks ordering. If
/// it is the end, returns NULL.
static inline MachineBasicBlock *getNextBlock(MachineBasicBlock *BB) {
  MachineFunction::iterator I = BB;
  MachineFunction::iterator E = BB->getParent()->end();
  if (++I == E)
    return NULL;
  return I;
}

/// ValidSimple - Returns true if the 'true' block (along with its
/// predecessor) forms a valid simple shape for ifcvt. It also returns the
/// number of instructions that the ifcvt would need to duplicate if performed
/// in Dups.
bool IfConverter::ValidSimple(BBInfo &TrueBBI, unsigned &Dups) const {
  Dups = 0;
  if (TrueBBI.IsBeingAnalyzed)
    return false;

  if (TrueBBI.BB->pred_size() > 1) {
    if (TrueBBI.CannotBeCopied ||
        TrueBBI.NonPredSize > TLI->getIfCvtDupBlockSizeLimit())
      return false;
    Dups = TrueBBI.NonPredSize;
  }

  return !blockAlwaysFallThrough(TrueBBI) && TrueBBI.BrCond.size() == 0;
}

/// ValidTriangle - Returns true if the 'true' and 'false' blocks (along
/// with their common predecessor) forms a valid triangle shape for ifcvt.
/// If 'FalseBranch' is true, it checks if 'true' block's false branch
/// branches to the false branch rather than the other way around. It also
/// returns the number of instructions that the ifcvt would need to duplicate
/// if performed in 'Dups'.
bool IfConverter::ValidTriangle(BBInfo &TrueBBI, BBInfo &FalseBBI,
                                bool FalseBranch, unsigned &Dups) const {
  Dups = 0;
  if (TrueBBI.IsBeingAnalyzed)
    return false;

  if (TrueBBI.BB->pred_size() > 1) {
    if (TrueBBI.CannotBeCopied)
      return false;

    unsigned Size = TrueBBI.NonPredSize;
    if (TrueBBI.IsBrAnalyzable) {
      if (TrueBBI.TrueBB && TrueBBI.BrCond.size() == 0)
        // End with an unconditional branch. It will be removed.
        --Size;
      else {
        MachineBasicBlock *FExit = FalseBranch
          ? TrueBBI.TrueBB : TrueBBI.FalseBB;
        if (FExit)
          // Require a conditional branch
          ++Size;
      }
    }
    if (Size > TLI->getIfCvtDupBlockSizeLimit())
      return false;
    Dups = Size;
  }

  MachineBasicBlock *TExit = FalseBranch ? TrueBBI.FalseBB : TrueBBI.TrueBB;
  if (!TExit && blockAlwaysFallThrough(TrueBBI)) {
    MachineFunction::iterator I = TrueBBI.BB;
    if (++I == TrueBBI.BB->getParent()->end())
      return false;
    TExit = I;
  }
  return TExit && TExit == FalseBBI.BB;
}

/// ValidDiamond - Returns true if the 'true' and 'false' blocks (along
/// with their common predecessor) forms a valid diamond shape for ifcvt.
bool IfConverter::ValidDiamond(BBInfo &TrueBBI, BBInfo &FalseBBI) const {
  if (TrueBBI.IsBeingAnalyzed || FalseBBI.IsBeingAnalyzed)
    return false;

  MachineBasicBlock *TT = TrueBBI.TrueBB;
  MachineBasicBlock *FT = FalseBBI.TrueBB;

  if (!TT && blockAlwaysFallThrough(TrueBBI))
    TT = getNextBlock(TrueBBI.BB);
  if (!FT && blockAlwaysFallThrough(FalseBBI))
    FT = getNextBlock(FalseBBI.BB);
  if (TT != FT)
    return false;
  if (TT == NULL && (TrueBBI.IsBrAnalyzable || FalseBBI.IsBrAnalyzable))
    return false;
  // FIXME: Allow false block to have an early exit?
  return (TrueBBI.BB->pred_size() == 1 &&
          FalseBBI.BB->pred_size() == 1 &&
          !TrueBBI.FalseBB && !FalseBBI.FalseBB);
}

/// ScanInstructions - Scan all the instructions in the block to determine if
/// the block is predicable. In most cases, that means all the instructions
/// in the block has M_PREDICABLE flag. Also checks if the block contains any
/// instruction which can clobber a predicate (e.g. condition code register).
/// If so, the block is not predicable unless it's the last instruction.
void IfConverter::ScanInstructions(BBInfo &BBI) {
  if (BBI.IsDone)
    return;

  // First analyze the end of BB branches.
  BBI.TrueBB = BBI.FalseBB = NULL;
  BBI.BrCond.clear();
  BBI.IsBrAnalyzable =
    !TII->AnalyzeBranch(*BBI.BB, BBI.TrueBB, BBI.FalseBB, BBI.BrCond);
  BBI.HasFallThrough = BBI.IsBrAnalyzable && BBI.FalseBB == NULL;

  if (BBI.BrCond.size()) {
    // No false branch. This BB must end with a conditional branch and a
    // fallthrough.
    if (!BBI.FalseBB)
      BBI.FalseBB = findFalseBlock(BBI.BB, BBI.TrueBB);  
    assert(BBI.FalseBB && "Expected to find the fallthrough block!");
  }

  // Then scan all the instructions.
  BBI.NonPredSize = 0;
  BBI.ClobbersPred = false;
  bool SeenCondBr = false;
  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    if (!BBI.CannotBeCopied && !TII->CanBeDuplicated(I))
      BBI.CannotBeCopied = true;

    const TargetInstrDescriptor *TID = I->getInstrDescriptor();
    bool isPredicated = TII->isPredicated(I);
    bool isCondBr = BBI.IsBrAnalyzable &&
      (TID->Flags & M_BRANCH_FLAG) != 0 && (TID->Flags & M_BARRIER_FLAG) == 0;

    if (!isPredicated && !isCondBr)
      BBI.NonPredSize++;

    if (BBI.ClobbersPred && !isPredicated) {
      // Predicate modification instruction should end the block (except for
      // already predicated instructions and end of block branches).
      if (isCondBr) {
        SeenCondBr = true;

        // Conditional branches is not predicable. But it may be eliminated.
        continue;
      }

      // Predicate may have been modified, the subsequent (currently)
      // unpredocated instructions cannot be correctly predicated.
      BBI.IsUnpredicable = true;
      return;
    }

    if (TID->Flags & M_CLOBBERS_PRED)
      BBI.ClobbersPred = true;

    if ((TID->Flags & M_PREDICABLE) == 0) {
      BBI.IsUnpredicable = true;
      return;
    }
  }
}

/// FeasibilityAnalysis - Determine if the block is a suitable candidate to be
/// predicated by the specified predicate.
bool IfConverter::FeasibilityAnalysis(BBInfo &BBI,
                                      std::vector<MachineOperand> &Pred,
                                      bool isTriangle, bool RevBranch) {
  // If the block is dead or unpredicable, then it cannot be predicated.
  if (BBI.IsDone || BBI.IsUnpredicable)
    return false;

  // Check predication threshold.
  if (BBI.NonPredSize == 0 || BBI.NonPredSize > TLI->getIfCvtBlockSizeLimit())
    return false;

  // If it is already predicated, check if its predicate subsumes the new
  // predicate.
  if (BBI.Predicate.size() && !TII->SubsumesPredicate(BBI.Predicate, Pred))
    return false;

  if (BBI.BrCond.size()) {
    if (!isTriangle)
      return false;

    // Test predicate subsumsion.
    std::vector<MachineOperand> RevPred(Pred);
    std::vector<MachineOperand> Cond(BBI.BrCond);
    if (RevBranch) {
      if (TII->ReverseBranchCondition(Cond))
        return false;
    }
    if (TII->ReverseBranchCondition(RevPred) ||
        !TII->SubsumesPredicate(Cond, RevPred))
      return false;
  }

  return true;
}

/// AnalyzeBlock - Analyze the structure of the sub-CFG starting from
/// the specified block. Record its successors and whether it looks like an
/// if-conversion candidate.
IfConverter::BBInfo &IfConverter::AnalyzeBlock(MachineBasicBlock *BB,
                                             std::vector<IfcvtToken*> &Tokens) {
  BBInfo &BBI = BBAnalysis[BB->getNumber()];

  if (BBI.IsAnalyzed || BBI.IsBeingAnalyzed)
    return BBI;

  BBI.BB = BB;
  BBI.IsBeingAnalyzed = true;

  ScanInstructions(BBI);

  // Unanalyable or ends with fallthrough or unconditional branch.
  if (!BBI.IsBrAnalyzable || BBI.BrCond.size() == 0) {
    BBI.IsBeingAnalyzed = false;
    BBI.IsAnalyzed = true;
    return BBI;
  }

  // Do not ifcvt if either path is a back edge to the entry block.
  if (BBI.TrueBB == BB || BBI.FalseBB == BB) {
    BBI.IsBeingAnalyzed = false;
    BBI.IsAnalyzed = true;
    return BBI;
  }

  BBInfo &TrueBBI  = AnalyzeBlock(BBI.TrueBB, Tokens);
  BBInfo &FalseBBI = AnalyzeBlock(BBI.FalseBB, Tokens);

  if (TrueBBI.IsDone && FalseBBI.IsDone) {
    BBI.IsBeingAnalyzed = false;
    BBI.IsAnalyzed = true;
    return BBI;
  }

  std::vector<MachineOperand> RevCond(BBI.BrCond);
  bool CanRevCond = !TII->ReverseBranchCondition(RevCond);

  unsigned Dups = 0;
  bool TNeedSub = TrueBBI.Predicate.size() > 0;
  bool FNeedSub = FalseBBI.Predicate.size() > 0;
  bool Enqueued = false;
  if (CanRevCond && ValidDiamond(TrueBBI, FalseBBI) &&
      !(TrueBBI.ClobbersPred && FalseBBI.ClobbersPred) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond) &&
      FeasibilityAnalysis(FalseBBI, RevCond)) {
    // Diamond:
    //   EBB
    //   / \_
    //  |   |
    // TBB FBB
    //   \ /
    //  TailBB
    // Note TailBB can be empty.
    Tokens.push_back(new IfcvtToken(BBI, ICDiamond, TNeedSub|FNeedSub, Dups));
    Enqueued = true;
  }

  if (ValidTriangle(TrueBBI, FalseBBI, false, Dups) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond, true)) {
    // Triangle:
    //   EBB
    //   | \_
    //   |  |
    //   | TBB
    //   |  /
    //   FBB
    Tokens.push_back(new IfcvtToken(BBI, ICTriangle, TNeedSub, Dups));
    Enqueued = true;
  }
  
  if (ValidTriangle(TrueBBI, FalseBBI, true, Dups) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond, true, true)) {
    Tokens.push_back(new IfcvtToken(BBI, ICTriangleRev, TNeedSub, Dups));
    Enqueued = true;
  }

  if (ValidSimple(TrueBBI, Dups) &&
      FeasibilityAnalysis(TrueBBI, BBI.BrCond)) {
    // Simple (split, no rejoin):
    //   EBB
    //   | \_
    //   |  |
    //   | TBB---> exit
    //   |    
    //   FBB
    Tokens.push_back(new IfcvtToken(BBI, ICSimple, TNeedSub, Dups));
    Enqueued = true;
  }

  if (CanRevCond) {
    // Try the other path...
    if (ValidTriangle(FalseBBI, TrueBBI, false, Dups) &&
        FeasibilityAnalysis(FalseBBI, RevCond, true)) {
      Tokens.push_back(new IfcvtToken(BBI, ICTriangleFalse, FNeedSub, Dups));
      Enqueued = true;
    }

    if (ValidTriangle(FalseBBI, TrueBBI, true, Dups) &&
        FeasibilityAnalysis(FalseBBI, RevCond, true, true)) {
      Tokens.push_back(new IfcvtToken(BBI, ICTriangleFRev, FNeedSub, Dups));
      Enqueued = true;
    }

    if (ValidSimple(FalseBBI, Dups) &&
        FeasibilityAnalysis(FalseBBI, RevCond)) {
      Tokens.push_back(new IfcvtToken(BBI, ICSimpleFalse, FNeedSub, Dups));
      Enqueued = true;
    }
  }

  BBI.IsEnqueued = Enqueued;
  BBI.IsBeingAnalyzed = false;
  BBI.IsAnalyzed = true;
  return BBI;
}

/// AnalyzeBlocks - Analyze all blocks and find entries for all if-conversion
/// candidates. It returns true if any CFG restructuring is done to expose more
/// if-conversion opportunities.
bool IfConverter::AnalyzeBlocks(MachineFunction &MF,
                                std::vector<IfcvtToken*> &Tokens) {
  bool Change = false;
  std::set<MachineBasicBlock*> Visited;
  for (unsigned i = 0, e = Roots.size(); i != e; ++i) {
    for (idf_ext_iterator<MachineBasicBlock*> I=idf_ext_begin(Roots[i],Visited),
           E = idf_ext_end(Roots[i], Visited); I != E; ++I) {
      MachineBasicBlock *BB = *I;
      AnalyzeBlock(BB, Tokens);
    }
  }

  // Sort to favor more complex ifcvt scheme.
  std::stable_sort(Tokens.begin(), Tokens.end(), IfcvtTokenCmp);

  return Change;
}

/// canFallThroughTo - Returns true either if ToBB is the next block after BB or
/// that all the intervening blocks are empty (given BB can fall through to its
/// next block).
static bool canFallThroughTo(MachineBasicBlock *BB, MachineBasicBlock *ToBB) {
  MachineFunction::iterator I = BB;
  MachineFunction::iterator TI = ToBB;
  MachineFunction::iterator E = BB->getParent()->end();
  while (++I != TI)
    if (I == E || !I->empty())
      return false;
  return true;
}

/// ReTryPreds - Invalidate predecessor BB info so it would be re-analyzed
/// to determine if it can be if-converted. If predecessor is already
/// enqueud, dequeue it!
void IfConverter::ReTryPreds(MachineBasicBlock *BB) {
  for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
         E = BB->pred_end(); PI != E; ++PI) {
    BBInfo &PBBI = BBAnalysis[(*PI)->getNumber()];
    if (PBBI.IsDone || PBBI.BB == BB)
      continue;
    PBBI.IsAnalyzed = false;
    PBBI.IsEnqueued = false;
  }
}

/// InsertUncondBranch - Inserts an unconditional branch from BB to ToBB.
///
static void InsertUncondBranch(MachineBasicBlock *BB, MachineBasicBlock *ToBB,
                               const TargetInstrInfo *TII) {
  std::vector<MachineOperand> NoCond;
  TII->InsertBranch(*BB, ToBB, NULL, NoCond);
}

/// RemoveExtraEdges - Remove true / false edges if either / both are no longer
/// successors.
void IfConverter::RemoveExtraEdges(BBInfo &BBI) {
  MachineBasicBlock *TBB = NULL, *FBB = NULL;
  std::vector<MachineOperand> Cond;
  bool isAnalyzable = !TII->AnalyzeBranch(*BBI.BB, TBB, FBB, Cond);
  bool CanFallthrough = isAnalyzable && (TBB == NULL || FBB == NULL);
  if (BBI.TrueBB && BBI.BB->isSuccessor(BBI.TrueBB))
    if (!(BBI.TrueBB == TBB || BBI.TrueBB == FBB ||
          (CanFallthrough && getNextBlock(BBI.BB) == BBI.TrueBB)))
      BBI.BB->removeSuccessor(BBI.TrueBB);
  if (BBI.FalseBB && BBI.BB->isSuccessor(BBI.FalseBB))
    if (!(BBI.FalseBB == TBB || BBI.FalseBB == FBB ||
          (CanFallthrough && getNextBlock(BBI.BB) == BBI.FalseBB)))
      BBI.BB->removeSuccessor(BBI.FalseBB);
}

/// IfConvertSimple - If convert a simple (split, no rejoin) sub-CFG.
///
bool IfConverter::IfConvertSimple(BBInfo &BBI, IfcvtKind Kind) {
  BBInfo &TrueBBI  = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  BBInfo *CvtBBI = &TrueBBI;
  BBInfo *NextBBI = &FalseBBI;

  std::vector<MachineOperand> Cond(BBI.BrCond);
  if (Kind == ICSimpleFalse)
    std::swap(CvtBBI, NextBBI);

  if (CvtBBI->CannotBeCopied && CvtBBI->BB->pred_size() > 1) {
    // Something has changed. It's no longer safe to predicate this block.
    BBI.IsAnalyzed = false;
    CvtBBI->IsAnalyzed = false;
    return false;
  }

  if (Kind == ICSimpleFalse)
    TII->ReverseBranchCondition(Cond);

  if (CvtBBI->BB->pred_size() > 1) {
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    // Copy instructions in the true block, predicate them add them to
    // the entry block.
    CopyAndPredicateBlock(BBI, *CvtBBI, Cond);
  } else {
    PredicateBlock(*CvtBBI, Cond);

    // Merge converted block into entry block.
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    MergeBlocks(BBI, *CvtBBI);
  }

  bool IterIfcvt = true;
  if (!canFallThroughTo(BBI.BB, NextBBI->BB)) {
    InsertUncondBranch(BBI.BB, NextBBI->BB, TII);
    BBI.HasFallThrough = false;
    // Now ifcvt'd block will look like this:
    // BB:
    // ...
    // t, f = cmp
    // if t op
    // b BBf
    //
    // We cannot further ifcvt this block because the unconditional branch
    // will have to be predicated on the new condition, that will not be
    // available if cmp executes.
    IterIfcvt = false;
  }

  RemoveExtraEdges(BBI);

  // Update block info. BB can be iteratively if-converted.
  if (!IterIfcvt)
    BBI.IsDone = true;
  ReTryPreds(BBI.BB);
  CvtBBI->IsDone = true;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// IfConvertTriangle - If convert a triangle sub-CFG.
///
bool IfConverter::IfConvertTriangle(BBInfo &BBI, IfcvtKind Kind) {
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  BBInfo *CvtBBI = &TrueBBI;
  BBInfo *NextBBI = &FalseBBI;

  std::vector<MachineOperand> Cond(BBI.BrCond);
  if (Kind == ICTriangleFalse || Kind == ICTriangleFRev)
    std::swap(CvtBBI, NextBBI);

  if (CvtBBI->CannotBeCopied && CvtBBI->BB->pred_size() > 1) {
    // Something has changed. It's no longer safe to predicate this block.
    BBI.IsAnalyzed = false;
    CvtBBI->IsAnalyzed = false;
    return false;
  }

  if (Kind == ICTriangleFalse || Kind == ICTriangleFRev)
    TII->ReverseBranchCondition(Cond);

  if (Kind == ICTriangleRev || Kind == ICTriangleFRev) {
    ReverseBranchCondition(*CvtBBI);
    // BB has been changed, modify its predecessors (except for this
    // one) so they don't get ifcvt'ed based on bad intel.
    for (MachineBasicBlock::pred_iterator PI = CvtBBI->BB->pred_begin(),
           E = CvtBBI->BB->pred_end(); PI != E; ++PI) {
      MachineBasicBlock *PBB = *PI;
      if (PBB == BBI.BB)
        continue;
      BBInfo &PBBI = BBAnalysis[PBB->getNumber()];
      if (PBBI.IsEnqueued) {
        PBBI.IsAnalyzed = false;
        PBBI.IsEnqueued = false;
      }
    }
  }

  bool HasEarlyExit = CvtBBI->FalseBB != NULL;
  bool DupBB = CvtBBI->BB->pred_size() > 1;
  if (DupBB) {
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    // Copy instructions in the true block, predicate them add them to
    // the entry block.
    CopyAndPredicateBlock(BBI, *CvtBBI, Cond, true);
  } else {
    // Predicate the 'true' block after removing its branch.
    CvtBBI->NonPredSize -= TII->RemoveBranch(*CvtBBI->BB);
    PredicateBlock(*CvtBBI, Cond);
  }

  // If 'true' block has a 'false' successor, add an exit branch to it.
  if (HasEarlyExit) {
    std::vector<MachineOperand> RevCond(CvtBBI->BrCond);
    if (TII->ReverseBranchCondition(RevCond))
      assert(false && "Unable to reverse branch condition!");
    if (DupBB) {
      TII->InsertBranch(*BBI.BB, CvtBBI->FalseBB, NULL, RevCond);
      BBI.BB->addSuccessor(CvtBBI->FalseBB);
    } else {
      TII->InsertBranch(*CvtBBI->BB, CvtBBI->FalseBB, NULL, RevCond);
    }
  }

  if (!DupBB) {
    // Now merge the entry of the triangle with the true block.
    BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
    MergeBlocks(BBI, *CvtBBI);
  }

  // Merge in the 'false' block if the 'false' block has no other
  // predecessors. Otherwise, add a unconditional branch from to 'false'.
  bool FalseBBDead = false;
  bool IterIfcvt = true;
  bool isFallThrough = canFallThroughTo(BBI.BB, NextBBI->BB);
  if (!isFallThrough) {
    // Only merge them if the true block does not fallthrough to the false
    // block. By not merging them, we make it possible to iteratively
    // ifcvt the blocks.
    if (!HasEarlyExit && NextBBI->BB->pred_size() == 1) {
      MergeBlocks(BBI, *NextBBI);
      FalseBBDead = true;
    } else {
      InsertUncondBranch(BBI.BB, NextBBI->BB, TII);
      BBI.HasFallThrough = false;
    }
    // Mixed predicated and unpredicated code. This cannot be iteratively
    // predicated.
    IterIfcvt = false;
  }

  RemoveExtraEdges(BBI);

  // Update block info. BB can be iteratively if-converted.
  if (!IterIfcvt) 
    BBI.IsDone = true;
  ReTryPreds(BBI.BB);
  CvtBBI->IsDone = true;
  if (FalseBBDead)
    NextBBI->IsDone = true;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// IfConvertDiamond - If convert a diamond sub-CFG.
///
bool IfConverter::IfConvertDiamond(BBInfo &BBI, IfcvtKind Kind) {
  BBInfo &TrueBBI  = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  MachineBasicBlock *TailBB = TrueBBI.TrueBB;

  SmallVector<MachineInstr*, 2> Dups;
  if (!TailBB) {
    // No common merge block. Check if the terminators (e.g. return) are
    // the same or predicable.
    MachineBasicBlock::iterator TT = BBI.TrueBB->getFirstTerminator();
    MachineBasicBlock::iterator FT = BBI.FalseBB->getFirstTerminator();
    while (TT != BBI.TrueBB->end() && FT != BBI.FalseBB->end()) {
      if (TT->isIdenticalTo(FT))
        Dups.push_back(TT);  // Will erase these later.
      else if ((TT->getInstrDescriptor()->Flags & M_PREDICABLE) == 0 ||
               (FT->getInstrDescriptor()->Flags & M_PREDICABLE) == 0)
        return false; // Can't if-convert. Abort!
      ++TT;
      ++FT;
    }

    // One of the two pathes have more terminators, make sure they are
    // all predicable.
    while (TT != BBI.TrueBB->end()) {
      if ((TT->getInstrDescriptor()->Flags & M_PREDICABLE) == 0)
        return false; // Can't if-convert. Abort!
      ++TT;
    }
    while (FT != BBI.FalseBB->end()) {
      if ((FT->getInstrDescriptor()->Flags & M_PREDICABLE) == 0)
        return false; // Can't if-convert. Abort!
      ++FT;
    }
  }

  // Remove the duplicated instructions from the 'true' block.
  for (unsigned i = 0, e = Dups.size(); i != e; ++i) {
    Dups[i]->eraseFromParent();
    --TrueBBI.NonPredSize;
  }
    
  // Merge the 'true' and 'false' blocks by copying the instructions
  // from the 'false' block to the 'true' block. That is, unless the true
  // block would clobber the predicate, in that case, do the opposite.
  BBInfo *BBI1 = &TrueBBI;
  BBInfo *BBI2 = &FalseBBI;
  std::vector<MachineOperand> RevCond(BBI.BrCond);
  TII->ReverseBranchCondition(RevCond);
  std::vector<MachineOperand> *Cond1 = &BBI.BrCond;
  std::vector<MachineOperand> *Cond2 = &RevCond;
  // Check the 'true' and 'false' blocks if either isn't ended with a branch.
  // Either the block fallthrough to another block or it ends with a
  // return. If it's the former, add a branch to its successor.
  bool NeedBr1 = !BBI1->TrueBB && BBI1->BB->succ_size() > 0;
  bool NeedBr2 = !BBI2->TrueBB && BBI2->BB->succ_size() > 0; 

  // Figure out the more profitable ordering.
  bool DoSwap = false;
  if (TrueBBI.ClobbersPred && !FalseBBI.ClobbersPred)
    DoSwap = true;
  else if (TrueBBI.ClobbersPred == FalseBBI.ClobbersPred) {
    if (!NeedBr1 && NeedBr2)
      DoSwap = true;
    else if (NeedBr1 == NeedBr2) {
      if (TrueBBI.NonPredSize > FalseBBI.NonPredSize)
        DoSwap = true;
    }
  }
  if (DoSwap) {
    std::swap(BBI1, BBI2);
    std::swap(Cond1, Cond2);
    std::swap(NeedBr1, NeedBr2);
  }

  // Predicate the 'true' block after removing its branch.
  BBI1->NonPredSize -= TII->RemoveBranch(*BBI1->BB);
  PredicateBlock(*BBI1, *Cond1);

  // Add an early exit branch if needed.
  if (NeedBr1)
    TII->InsertBranch(*BBI1->BB, *BBI1->BB->succ_begin(), NULL, *Cond1);

  // Predicate the 'false' block.
  PredicateBlock(*BBI2, *Cond2, true);

  // Add an unconditional branch from 'false' to to 'false' successor if it
  // will not be the fallthrough block.
  if (NeedBr2 && !NeedBr1) {
    // If BBI2 isn't going to be merged in, then the existing fallthrough
    // or branch is fine.
    if (!canFallThroughTo(BBI.BB, *BBI2->BB->succ_begin())) {
      InsertUncondBranch(BBI2->BB, *BBI2->BB->succ_begin(), TII);
      BBI2->HasFallThrough = false;
    }
  }

  // Keep them as two separate blocks if there is an early exit.
  if (!NeedBr1)
    MergeBlocks(*BBI1, *BBI2);

  // Remove the conditional branch from entry to the blocks.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);

  // Merge the combined block into the entry of the diamond.
  MergeBlocks(BBI, *BBI1);

  // 'True' and 'false' aren't combined, see if we need to add a unconditional
  // branch to the 'false' block.
  if (NeedBr1 && !canFallThroughTo(BBI.BB, BBI2->BB)) {
    InsertUncondBranch(BBI.BB, BBI2->BB, TII);
    BBI1->HasFallThrough = false;
  }

  // If the if-converted block fallthrough or unconditionally branch into the
  // tail block, and the tail block does not have other predecessors, then
  // fold the tail block in as well.
  BBInfo *CvtBBI = NeedBr1 ? BBI2 : &BBI;
  if (TailBB &&
      TailBB->pred_size() == 1 && CvtBBI->BB->succ_size() == 1) {
    CvtBBI->NonPredSize -= TII->RemoveBranch(*CvtBBI->BB);
    BBInfo TailBBI = BBAnalysis[TailBB->getNumber()];
    MergeBlocks(*CvtBBI, TailBBI);
    TailBBI.IsDone = true;
  }

  RemoveExtraEdges(BBI);

  // Update block info.
  BBI.IsDone = TrueBBI.IsDone = FalseBBI.IsDone = true;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// PredicateBlock - Predicate every instruction in the block with the specified
/// condition. If IgnoreTerm is true, skip over all terminator instructions.
void IfConverter::PredicateBlock(BBInfo &BBI,
                                 std::vector<MachineOperand> &Cond,
                                 bool IgnoreTerm) {
  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    if (IgnoreTerm && TII->isTerminatorInstr(I->getOpcode()))
      continue;
    if (TII->isPredicated(I))
      continue;
    if (!TII->PredicateInstruction(I, Cond)) {
      cerr << "Unable to predicate " << *I << "!\n";
      abort();
    }
  }

  std::copy(Cond.begin(), Cond.end(), std::back_inserter(BBI.Predicate));

  BBI.IsAnalyzed = false;
  BBI.NonPredSize = 0;

  NumIfConvBBs++;
}

/// CopyAndPredicateBlock - Copy and predicate instructions from source BB to
/// the destination block. Skip end of block branches if IgnoreBr is true.
void IfConverter::CopyAndPredicateBlock(BBInfo &ToBBI, BBInfo &FromBBI,
                                        std::vector<MachineOperand> &Cond,
                                        bool IgnoreBr) {
  for (MachineBasicBlock::iterator I = FromBBI.BB->begin(),
         E = FromBBI.BB->end(); I != E; ++I) {
    const TargetInstrDescriptor *TID = I->getInstrDescriptor();
    bool isPredicated = TII->isPredicated(I);
    // Do not copy the end of the block branches.
    if (IgnoreBr && !isPredicated && (TID->Flags & M_BRANCH_FLAG) != 0)
      break;

    MachineInstr *MI = I->clone();
    ToBBI.BB->insert(ToBBI.BB->end(), MI);
    ToBBI.NonPredSize++;

    if (!isPredicated)
      if (!TII->PredicateInstruction(MI, Cond)) {
        cerr << "Unable to predicate " << *MI << "!\n";
        abort();
      }
  }

  std::copy(FromBBI.Predicate.begin(), FromBBI.Predicate.end(),
            std::back_inserter(ToBBI.Predicate));
  std::copy(Cond.begin(), Cond.end(), std::back_inserter(ToBBI.Predicate));

  ToBBI.ClobbersPred |= FromBBI.ClobbersPred;
  ToBBI.IsAnalyzed = false;

  NumDupBBs++;
}

/// MergeBlocks - Move all instructions from FromBB to the end of ToBB.
///
void IfConverter::MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI) {
  ToBBI.BB->splice(ToBBI.BB->end(),
                   FromBBI.BB, FromBBI.BB->begin(), FromBBI.BB->end());

  // Redirect all branches to FromBB to ToBB.
  std::vector<MachineBasicBlock *> Preds(FromBBI.BB->pred_begin(),
                                         FromBBI.BB->pred_end());
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    MachineBasicBlock *Pred = Preds[i];
    if (Pred == ToBBI.BB)
      continue;
    Pred->ReplaceUsesOfBlockWith(FromBBI.BB, ToBBI.BB);
  }
 
  std::vector<MachineBasicBlock *> Succs(FromBBI.BB->succ_begin(),
                                         FromBBI.BB->succ_end());
  MachineBasicBlock *NBB = getNextBlock(FromBBI.BB);
  MachineBasicBlock *FallThrough = FromBBI.HasFallThrough ? NBB : NULL;

  for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
    MachineBasicBlock *Succ = Succs[i];
    // Fallthrough edge can't be transferred.
    if (Succ == FallThrough)
      continue;
    FromBBI.BB->removeSuccessor(Succ);
    if (!ToBBI.BB->isSuccessor(Succ))
      ToBBI.BB->addSuccessor(Succ);
  }

  // Now FromBBI always fall through to the next block!
  if (NBB)
    FromBBI.BB->addSuccessor(NBB);

  std::copy(FromBBI.Predicate.begin(), FromBBI.Predicate.end(),
            std::back_inserter(ToBBI.Predicate));
  FromBBI.Predicate.clear();

  ToBBI.NonPredSize += FromBBI.NonPredSize;
  FromBBI.NonPredSize = 0;

  ToBBI.ClobbersPred |= FromBBI.ClobbersPred;
  ToBBI.HasFallThrough = FromBBI.HasFallThrough;
  ToBBI.IsAnalyzed = false;
  FromBBI.IsAnalyzed = false;
}

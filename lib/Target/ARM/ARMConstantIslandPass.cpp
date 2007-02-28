//===-- ARMConstantIslandPass.cpp - ARM constant islands --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that splits the constant pool up into 'islands'
// which are scattered through-out the function.  This is required due to the
// limited pc-relative displacements that ARM has.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-cp-islands"
#include "ARM.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMInstrInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumCPEs,     "Number of constpool entries");
STATISTIC(NumSplit,    "Number of uncond branches inserted");
STATISTIC(NumCBrFixed, "Number of cond branches fixed");
STATISTIC(NumUBrFixed, "Number of uncond branches fixed");

namespace {
  /// ARMConstantIslands - Due to limited PC-relative displacements, ARM
  /// requires constant pool entries to be scattered among the instructions
  /// inside a function.  To do this, it completely ignores the normal LLVM
  /// constant pool; instead, it places constants wherever it feels like with
  /// special instructions.
  ///
  /// The terminology used in this pass includes:
  ///   Islands - Clumps of constants placed in the function.
  ///   Water   - Potential places where an island could be formed.
  ///   CPE     - A constant pool entry that has been placed somewhere, which
  ///             tracks a list of users.
  class VISIBILITY_HIDDEN ARMConstantIslands : public MachineFunctionPass {
    /// NextUID - Assign unique ID's to CPE's.
    unsigned NextUID;

    /// BBSizes - The size of each MachineBasicBlock in bytes of code, indexed
    /// by MBB Number.
    std::vector<unsigned> BBSizes;
    
    /// BBOffsets - the offset of each MBB in bytes, starting from 0.
    std::vector<unsigned> BBOffsets;

    /// WaterList - A sorted list of basic blocks where islands could be placed
    /// (i.e. blocks that don't fall through to the following block, due
    /// to a return, unreachable, or unconditional branch).
    std::vector<MachineBasicBlock*> WaterList;

    /// CPUser - One user of a constant pool, keeping the machine instruction
    /// pointer, the constant pool being referenced, and the max displacement
    /// allowed from the instruction to the CP.
    struct CPUser {
      MachineInstr *MI;
      MachineInstr *CPEMI;
      unsigned MaxDisp;
      CPUser(MachineInstr *mi, MachineInstr *cpemi, unsigned maxdisp)
        : MI(mi), CPEMI(cpemi), MaxDisp(maxdisp) {}
    };
    
    /// CPUsers - Keep track of all of the machine instructions that use various
    /// constant pools and their max displacement.
    std::vector<CPUser> CPUsers;
    
    /// CPEntry - One per constant pool entry, keeping the machine instruction
    /// pointer, the constpool index, and the number of CPUser's which
    /// reference this entry.
    struct CPEntry {
      MachineInstr *CPEMI;
      unsigned CPI;
      unsigned RefCount;
      CPEntry(MachineInstr *cpemi, unsigned cpi, unsigned rc = 0)
        : CPEMI(cpemi), CPI(cpi), RefCount(rc) {}
    };

    /// CPEntries - Keep track of all of the constant pool entry machine
    /// instructions. For each original constpool index (i.e. those that
    /// existed upon entry to this pass), it keeps a vector of entries.
    /// Original elements are cloned as we go along; the clones are
    /// put in the vector of the original element, but have distinct CPIs.
    std::vector<std::vector<CPEntry> > CPEntries;
    
    /// ImmBranch - One per immediate branch, keeping the machine instruction
    /// pointer, conditional or unconditional, the max displacement,
    /// and (if isCond is true) the corresponding unconditional branch
    /// opcode.
    struct ImmBranch {
      MachineInstr *MI;
      unsigned MaxDisp : 31;
      bool isCond : 1;
      int UncondBr;
      ImmBranch(MachineInstr *mi, unsigned maxdisp, bool cond, int ubr)
        : MI(mi), MaxDisp(maxdisp), isCond(cond), UncondBr(ubr) {}
    };

    /// Branches - Keep track of all the immediate branch instructions.
    ///
    std::vector<ImmBranch> ImmBranches;

    /// PushPopMIs - Keep track of all the Thumb push / pop instructions.
    ///
    SmallVector<MachineInstr*, 4> PushPopMIs;

    /// HasFarJump - True if any far jump instruction has been emitted during
    /// the branch fix up pass.
    bool HasFarJump;

    const TargetInstrInfo *TII;
    const ARMFunctionInfo *AFI;
  public:
    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM constant island placement and branch shortening pass";
    }
    
  private:
    void DoInitialPlacement(MachineFunction &Fn,
                            std::vector<MachineInstr*> &CPEMIs);
    CPEntry *findConstPoolEntry(unsigned CPI, const MachineInstr *CPEMI);
    void InitialFunctionScan(MachineFunction &Fn,
                             const std::vector<MachineInstr*> &CPEMIs);
    MachineBasicBlock *SplitBlockBeforeInstr(MachineInstr *MI);
    void UpdateForInsertedWaterBlock(MachineBasicBlock *NewBB);
    void AdjustBBOffsetsAfter(MachineBasicBlock *BB, int delta);
    bool DecrementOldEntry(unsigned CPI, MachineInstr* CPEMI, unsigned Size);
    int LookForExistingCPEntry(CPUser& U, unsigned UserOffset);
    bool HandleConstantPoolUser(MachineFunction &Fn, unsigned CPUserIndex);
    bool CPEIsInRange(MachineInstr *MI, unsigned UserOffset, 
                      MachineInstr *CPEMI, unsigned Disp,
                      bool DoDump);
    bool WaterIsInRange(unsigned UserOffset, MachineBasicBlock *Water,
                        unsigned Disp);
    bool OffsetIsInRange(unsigned UserOffset, unsigned TrialOffset,
                        unsigned Disp, bool NegativeOK);
    bool BBIsInRange(MachineInstr *MI, MachineBasicBlock *BB, unsigned Disp);
    bool FixUpImmediateBr(MachineFunction &Fn, ImmBranch &Br);
    bool FixUpConditionalBr(MachineFunction &Fn, ImmBranch &Br);
    bool FixUpUnconditionalBr(MachineFunction &Fn, ImmBranch &Br);
    bool UndoLRSpillRestore();

    unsigned GetOffsetOf(MachineInstr *MI) const;
  };
}

/// createARMConstantIslandPass - returns an instance of the constpool
/// island pass.
FunctionPass *llvm::createARMConstantIslandPass() {
  return new ARMConstantIslands();
}

bool ARMConstantIslands::runOnMachineFunction(MachineFunction &Fn) {
  MachineConstantPool &MCP = *Fn.getConstantPool();
  
  TII = Fn.getTarget().getInstrInfo();
  AFI = Fn.getInfo<ARMFunctionInfo>();

  HasFarJump = false;

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  Fn.RenumberBlocks();

  // Perform the initial placement of the constant pool entries.  To start with,
  // we put them all at the end of the function.
  std::vector<MachineInstr*> CPEMIs;
  if (!MCP.isEmpty())
    DoInitialPlacement(Fn, CPEMIs);
  
  /// The next UID to take is the first unused one.
  NextUID = CPEMIs.size();
  
  // Do the initial scan of the function, building up information about the
  // sizes of each block, the location of all the water, and finding all of the
  // constant pool users.
  InitialFunctionScan(Fn, CPEMIs);
  CPEMIs.clear();
  
  // Iteratively place constant pool entries and fix up branches until there
  // is no change.
  bool MadeChange = false;
  while (true) {
    bool Change = false;
    for (unsigned i = 0, e = CPUsers.size(); i != e; ++i)
      Change |= HandleConstantPoolUser(Fn, i);
    for (unsigned i = 0, e = ImmBranches.size(); i != e; ++i)
      Change |= FixUpImmediateBr(Fn, ImmBranches[i]);
    if (!Change)
      break;
    MadeChange = true;
  }
  
  // If LR has been forced spilled and no far jumps (i.e. BL) has been issued.
  // Undo the spill / restore of LR if possible.
  if (!HasFarJump && AFI->isLRForceSpilled() && AFI->isThumbFunction())
    MadeChange |= UndoLRSpillRestore();

  BBSizes.clear();
  BBOffsets.clear();
  WaterList.clear();
  CPUsers.clear();
  CPEntries.clear();
  ImmBranches.clear();
  PushPopMIs.clear();

  return MadeChange;
}

/// DoInitialPlacement - Perform the initial placement of the constant pool
/// entries.  To start with, we put them all at the end of the function.
void ARMConstantIslands::DoInitialPlacement(MachineFunction &Fn,
                                        std::vector<MachineInstr*> &CPEMIs){
  // Create the basic block to hold the CPE's.
  MachineBasicBlock *BB = new MachineBasicBlock();
  Fn.getBasicBlockList().push_back(BB);
  
  // Add all of the constants from the constant pool to the end block, use an
  // identity mapping of CPI's to CPE's.
  const std::vector<MachineConstantPoolEntry> &CPs =
    Fn.getConstantPool()->getConstants();
  
  const TargetData &TD = *Fn.getTarget().getTargetData();
  for (unsigned i = 0, e = CPs.size(); i != e; ++i) {
    unsigned Size = TD.getTypeSize(CPs[i].getType());
    // Verify that all constant pool entries are a multiple of 4 bytes.  If not,
    // we would have to pad them out or something so that instructions stay
    // aligned.
    assert((Size & 3) == 0 && "CP Entry not multiple of 4 bytes!");
    MachineInstr *CPEMI =
      BuildMI(BB, TII->get(ARM::CONSTPOOL_ENTRY))
                           .addImm(i).addConstantPoolIndex(i).addImm(Size);
    CPEMIs.push_back(CPEMI);

    // Add a new CPEntry, but no corresponding CPUser yet.
    std::vector<CPEntry> CPEs;
    CPEs.push_back(CPEntry(CPEMI, i));
    CPEntries.push_back(CPEs);
    NumCPEs++;
    DOUT << "Moved CPI#" << i << " to end of function as #" << i << "\n";
  }
}

/// BBHasFallthrough - Return true if the specified basic block can fallthrough
/// into the block immediately after it.
static bool BBHasFallthrough(MachineBasicBlock *MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::iterator MBBI = MBB;
  if (next(MBBI) == MBB->getParent()->end())  // Can't fall off end of function.
    return false;
  
  MachineBasicBlock *NextBB = next(MBBI);
  for (MachineBasicBlock::succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I)
    if (*I == NextBB)
      return true;
  
  return false;
}

/// findConstPoolEntry - Given the constpool index and CONSTPOOL_ENTRY MI,
/// look up the corresponding CPEntry.
ARMConstantIslands::CPEntry
*ARMConstantIslands::findConstPoolEntry(unsigned CPI,
                                        const MachineInstr *CPEMI) {
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  // Number of entries per constpool index should be small, just do a
  // linear search.
  for (unsigned i = 0, e = CPEs.size(); i != e; ++i) {
    if (CPEs[i].CPEMI == CPEMI)
      return &CPEs[i];
  }
  return NULL;
}

/// InitialFunctionScan - Do the initial scan of the function, building up
/// information about the sizes of each block, the location of all the water,
/// and finding all of the constant pool users.
void ARMConstantIslands::InitialFunctionScan(MachineFunction &Fn,
                                 const std::vector<MachineInstr*> &CPEMIs) {
  unsigned Offset = 0;
  for (MachineFunction::iterator MBBI = Fn.begin(), E = Fn.end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock &MBB = *MBBI;
    
    // If this block doesn't fall through into the next MBB, then this is
    // 'water' that a constant pool island could be placed.
    if (!BBHasFallthrough(&MBB))
      WaterList.push_back(&MBB);
    
    unsigned MBBSize = 0;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      // Add instruction size to MBBSize.
      MBBSize += ARM::GetInstSize(I);

      int Opc = I->getOpcode();
      if (TII->isBranch(Opc)) {
        bool isCond = false;
        unsigned Bits = 0;
        unsigned Scale = 1;
        int UOpc = Opc;
        switch (Opc) {
        default:
          continue;  // Ignore JT branches
        case ARM::Bcc:
          isCond = true;
          UOpc = ARM::B;
          // Fallthrough
        case ARM::B:
          Bits = 24;
          Scale = 4;
          break;
        case ARM::tBcc:
          isCond = true;
          UOpc = ARM::tB;
          Bits = 8;
          Scale = 2;
          break;
        case ARM::tB:
          Bits = 11;
          Scale = 2;
          break;
        }

        // Record this immediate branch.
        unsigned MaxOffs = ((1 << (Bits-1))-1) * Scale;
        ImmBranches.push_back(ImmBranch(I, MaxOffs, isCond, UOpc));
      }

      if (Opc == ARM::tPUSH || Opc == ARM::tPOP_RET)
        PushPopMIs.push_back(I);

      // Scan the instructions for constant pool operands.
      for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op)
        if (I->getOperand(op).isConstantPoolIndex()) {
          // We found one.  The addressing mode tells us the max displacement
          // from the PC that this instruction permits.
          
          // Basic size info comes from the TSFlags field.
          unsigned Bits = 0;
          unsigned Scale = 1;
          unsigned TSFlags = I->getInstrDescriptor()->TSFlags;
          switch (TSFlags & ARMII::AddrModeMask) {
          default: 
            // Constant pool entries can reach anything.
            if (I->getOpcode() == ARM::CONSTPOOL_ENTRY)
              continue;
            assert(0 && "Unknown addressing mode for CP reference!");
          case ARMII::AddrMode1: // AM1: 8 bits << 2
            Bits = 8;
            Scale = 4;  // Taking the address of a CP entry.
            break;
          case ARMII::AddrMode2:
            Bits = 12;  // +-offset_12
            break;
          case ARMII::AddrMode3:
            Bits = 8;   // +-offset_8
            break;
            // addrmode4 has no immediate offset.
          case ARMII::AddrMode5:
            Bits = 8;
            Scale = 4;  // +-(offset_8*4)
            break;
          case ARMII::AddrModeT1:
            Bits = 5;  // +offset_5
            break;
          case ARMII::AddrModeT2:
            Bits = 5;
            Scale = 2;  // +(offset_5*2)
            break;
          case ARMII::AddrModeT4:
            Bits = 5;
            Scale = 4;  // +(offset_5*4)
            break;
          case ARMII::AddrModeTs:
            Bits = 8;
            Scale = 4;  // +(offset_8*4)
            break;
          }

          // Remember that this is a user of a CP entry.
          unsigned CPI = I->getOperand(op).getConstantPoolIndex();
          MachineInstr *CPEMI = CPEMIs[CPI];
          unsigned MaxOffs = ((1 << Bits)-1) * Scale;          
          CPUsers.push_back(CPUser(I, CPEMI, MaxOffs));

          // Increment corresponding CPEntry reference count.
          CPEntry *CPE = findConstPoolEntry(CPI, CPEMI);
          assert(CPE && "Cannot find a corresponding CPEntry!");
          CPE->RefCount++;
          
          // Instructions can only use one CP entry, don't bother scanning the
          // rest of the operands.
          break;
        }
    }

    // In thumb mode, if this block is a constpool island, pessimistically 
    // assume it needs to be padded by two byte so it's aligned on 4 byte 
    // boundary.
    if (AFI->isThumbFunction() &&
        !MBB.empty() &&
        MBB.begin()->getOpcode() == ARM::CONSTPOOL_ENTRY)
      MBBSize += 2;

    BBSizes.push_back(MBBSize);
    BBOffsets.push_back(Offset);
    Offset += MBBSize;
  }
}

/// GetOffsetOf - Return the current offset of the specified machine instruction
/// from the start of the function.  This offset changes as stuff is moved
/// around inside the function.
unsigned ARMConstantIslands::GetOffsetOf(MachineInstr *MI) const {
  MachineBasicBlock *MBB = MI->getParent();
  
  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BBOffsets[MBB->getNumber()];

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::iterator I = MBB->begin(); ; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    if (&*I == MI) return Offset;
    Offset += ARM::GetInstSize(I);
  }
}

/// CompareMBBNumbers - Little predicate function to sort the WaterList by MBB
/// ID.
static bool CompareMBBNumbers(const MachineBasicBlock *LHS,
                              const MachineBasicBlock *RHS) {
  return LHS->getNumber() < RHS->getNumber();
}

/// UpdateForInsertedWaterBlock - When a block is newly inserted into the
/// machine function, it upsets all of the block numbers.  Renumber the blocks
/// and update the arrays that parallel this numbering.
void ARMConstantIslands::UpdateForInsertedWaterBlock(MachineBasicBlock *NewBB) {
  // Renumber the MBB's to keep them consequtive.
  NewBB->getParent()->RenumberBlocks(NewBB);
  
  // Insert a size into BBSizes to align it properly with the (newly
  // renumbered) block numbers.
  BBSizes.insert(BBSizes.begin()+NewBB->getNumber(), 0);

  // Likewise for BBOffsets.
  BBOffsets.insert(BBOffsets.begin()+NewBB->getNumber(), 0);
  
  // Next, update WaterList.  Specifically, we need to add NewMBB as having 
  // available water after it.
  std::vector<MachineBasicBlock*>::iterator IP =
    std::lower_bound(WaterList.begin(), WaterList.end(), NewBB,
                     CompareMBBNumbers);
  WaterList.insert(IP, NewBB);
}


/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update datastructures and renumber blocks to
/// account for this change and returns the newly created block.
MachineBasicBlock *ARMConstantIslands::SplitBlockBeforeInstr(MachineInstr *MI) {
  MachineBasicBlock *OrigBB = MI->getParent();
  bool isThumb = AFI->isThumbFunction();

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB = new MachineBasicBlock(OrigBB->getBasicBlock());
  MachineFunction::iterator MBBI = OrigBB; ++MBBI;
  OrigBB->getParent()->getBasicBlockList().insert(MBBI, NewBB);
  
  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());
  
  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  BuildMI(OrigBB, TII->get(isThumb ? ARM::tB : ARM::B)).addMBB(NewBB);
  NumSplit++;
  
  // Update the CFG.  All succs of OrigBB are now succs of NewBB.
  while (!OrigBB->succ_empty()) {
    MachineBasicBlock *Succ = *OrigBB->succ_begin();
    OrigBB->removeSuccessor(Succ);
    NewBB->addSuccessor(Succ);
    
    // This pass should be run after register allocation, so there should be no
    // PHI nodes to update.
    assert((Succ->empty() || Succ->begin()->getOpcode() != TargetInstrInfo::PHI)
           && "PHI nodes should be eliminated by now!");
  }
  
  // OrigBB branches to NewBB.
  OrigBB->addSuccessor(NewBB);
  
  // Update internal data structures to account for the newly inserted MBB.
  // This is almost the same as UpdateForInsertedWaterBlock, except that
  // the Water goes after OrigBB, not NewBB.
  NewBB->getParent()->RenumberBlocks(NewBB);
  
  // Insert a size into BBSizes to align it properly with the (newly
  // renumbered) block numbers.
  BBSizes.insert(BBSizes.begin()+NewBB->getNumber(), 0);
  
  // Likewise for BBOffsets.
  BBOffsets.insert(BBOffsets.begin()+NewBB->getNumber(), 0);

  // Next, update WaterList.  Specifically, we need to add OrigMBB as having 
  // available water after it (but not if it's already there, which happens
  // when splitting before a conditional branch that is followed by an
  // unconditional branch - in that case we want to insert NewBB).
  std::vector<MachineBasicBlock*>::iterator IP =
    std::lower_bound(WaterList.begin(), WaterList.end(), OrigBB,
                     CompareMBBNumbers);
  MachineBasicBlock* WaterBB = *IP;
  if (WaterBB == OrigBB)
    WaterList.insert(next(IP), NewBB);
  else
    WaterList.insert(IP, OrigBB);

  // Figure out how large the first NewMBB is.
  unsigned NewBBSize = 0;
  for (MachineBasicBlock::iterator I = NewBB->begin(), E = NewBB->end();
       I != E; ++I)
    NewBBSize += ARM::GetInstSize(I);
  
  unsigned OrigBBI = OrigBB->getNumber();
  unsigned NewBBI = NewBB->getNumber();
  // Set the size of NewBB in BBSizes.
  BBSizes[NewBBI] = NewBBSize;
  
  // We removed instructions from UserMBB, subtract that off from its size.
  // Add 2 or 4 to the block to count the unconditional branch we added to it.
  unsigned delta = isThumb ? 2 : 4;
  BBSizes[OrigBBI] -= NewBBSize - delta;

  // ...and adjust BBOffsets for NewBB accordingly.
  BBOffsets[NewBBI] = BBOffsets[OrigBBI] + BBSizes[OrigBBI];

  // All BBOffsets following these blocks must be modified.
  AdjustBBOffsetsAfter(NewBB, delta);

  return NewBB;
}

/// OffsetIsInRange - Checks whether UserOffset is within MaxDisp of
/// TrialOffset.
bool ARMConstantIslands::OffsetIsInRange(unsigned UserOffset, 
                      unsigned TrialOffset, unsigned MaxDisp, bool NegativeOK) {
  if (UserOffset <= TrialOffset) {
    // User before the Trial.
    if (TrialOffset-UserOffset <= MaxDisp)
      return true;
  } else if (NegativeOK) {
    if (UserOffset-TrialOffset <= MaxDisp)
      return true;
  }
  return false;
}

/// WaterIsInRange - Returns true if a CPE placed after the specified
/// Water (a basic block) will be in range for the specific MI.

bool ARMConstantIslands::WaterIsInRange(unsigned UserOffset,
                         MachineBasicBlock* Water, unsigned MaxDisp)
{
  bool isThumb = AFI->isThumbFunction();
  unsigned CPEOffset = BBOffsets[Water->getNumber()] + 
                       BBSizes[Water->getNumber()];
  // If the Water is a constpool island, it has already been aligned.
  // If not, align it.
  if (isThumb &&
      (Water->empty() ||
       Water->begin()->getOpcode() != ARM::CONSTPOOL_ENTRY))
    CPEOffset += 2;

  return OffsetIsInRange (UserOffset, CPEOffset, MaxDisp, !isThumb);
}

/// CPEIsInRange - Returns true if the distance between specific MI and
/// specific ConstPool entry instruction can fit in MI's displacement field.
bool ARMConstantIslands::CPEIsInRange(MachineInstr *MI, unsigned UserOffset,
                                      MachineInstr *CPEMI,
                                      unsigned MaxDisp, bool DoDump) {
  // In thumb mode, pessimistically assumes the .align 2 before the first CPE
  // in the island adds two byte padding.
  bool isThumb = AFI->isThumbFunction();
  unsigned AlignAdj   = isThumb ? 2 : 0;
  unsigned CPEOffset  = GetOffsetOf(CPEMI) + AlignAdj;

  if (DoDump) {
    DOUT << "User of CPE#" << CPEMI->getOperand(0).getImm()
         << " max delta=" << MaxDisp
         << " insn address=" << UserOffset
         << " CPE address=" << CPEOffset
         << " offset=" << int(CPEOffset-UserOffset) << "\t" << *MI;
  }

  return OffsetIsInRange(UserOffset, CPEOffset, MaxDisp, !isThumb);
}

/// BBIsJumpedOver - Return true of the specified basic block's only predecessor
/// unconditionally branches to its only successor.
static bool BBIsJumpedOver(MachineBasicBlock *MBB) {
  if (MBB->pred_size() != 1 || MBB->succ_size() != 1)
    return false;

  MachineBasicBlock *Succ = *MBB->succ_begin();
  MachineBasicBlock *Pred = *MBB->pred_begin();
  MachineInstr *PredMI = &Pred->back();
  if (PredMI->getOpcode() == ARM::B || PredMI->getOpcode() == ARM::tB)
    return PredMI->getOperand(0).getMBB() == Succ;
  return false;
}

void ARMConstantIslands::AdjustBBOffsetsAfter(MachineBasicBlock *BB, int delta)
{
  MachineFunction::iterator MBBI = BB->getParent()->end();
  for(unsigned i=BB->getNumber()+1; i<BB->getParent()->getNumBlockIDs(); i++)
    BBOffsets[i] += delta;
}

/// DecrementOldEntry - find the constant pool entry with index CPI
/// and instruction CPEMI, and decrement its refcount.  If the refcount
/// becomes 0 remove the entry and instruction.  Returns true if we removed 
/// the entry, false if we didn't.

bool ARMConstantIslands::DecrementOldEntry(unsigned CPI, MachineInstr *CPEMI, 
                              unsigned Size) {
  // Find the old entry. Eliminate it if it is no longer used.
  CPEntry *OldCPE = findConstPoolEntry(CPI, CPEMI);
  assert(OldCPE && "Unexpected!");
  if (--OldCPE->RefCount == 0) {
    MachineBasicBlock *OldCPEBB = OldCPE->CPEMI->getParent();
    if (OldCPEBB->empty()) {
      // In thumb mode, the size of island is padded by two to compensate for
      // the alignment requirement.  Thus it will now be 2 when the block is
      // empty, so fix this.
      // All succeeding offsets have the current size value added in, fix this.
      if (BBSizes[OldCPEBB->getNumber()] != 0) {
        AdjustBBOffsetsAfter(OldCPEBB, -BBSizes[OldCPEBB->getNumber()]);
        BBSizes[OldCPEBB->getNumber()] = 0;
      }
      // An island has only one predecessor BB and one successor BB. Check if
      // this BB's predecessor jumps directly to this BB's successor. This
      // shouldn't happen currently.
      assert(!BBIsJumpedOver(OldCPEBB) && "How did this happen?");
      // FIXME: remove the empty blocks after all the work is done?
    } else {
      BBSizes[OldCPEBB->getNumber()] -= Size;
      // All succeeding offsets have the current size value added in, fix this.
      AdjustBBOffsetsAfter(OldCPEBB, -Size);
    }
    OldCPE->CPEMI->eraseFromParent();
    OldCPE->CPEMI = NULL;
    NumCPEs--;
    return true;
  }
  return false;
}

/// LookForCPEntryInRange - see if the currently referenced CPE is in range;
/// if not, see if an in-range clone of the CPE is in range, and if so,
/// change the data structures so the user references the clone.  Returns:
/// 0 = no existing entry found
/// 1 = entry found, and there were no code insertions or deletions
/// 2 = entry found, and there were code insertions or deletions
int ARMConstantIslands::LookForExistingCPEntry(CPUser& U, unsigned UserOffset)
{
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;

  // Check to see if the CPE is already in-range.
  if (CPEIsInRange(UserMI, UserOffset, CPEMI, U.MaxDisp, true)) {
    DOUT << "In range\n";
    return 1;
  }

  // No.  Look for previously created clones of the CPE that are in range.
  unsigned CPI = CPEMI->getOperand(1).getConstantPoolIndex();
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  for (unsigned i = 0, e = CPEs.size(); i != e; ++i) {
    // We already tried this one
    if (CPEs[i].CPEMI == CPEMI)
      continue;
    // Removing CPEs can leave empty entries, skip
    if (CPEs[i].CPEMI == NULL)
      continue;
    if (CPEIsInRange(UserMI, UserOffset, CPEs[i].CPEMI, U.MaxDisp, false)) {
      DOUT << "Replacing CPE#" << CPI << " with CPE#" << CPEs[i].CPI << "\n";
      // Point the CPUser node to the replacement
      U.CPEMI = CPEs[i].CPEMI;
      // Change the CPI in the instruction operand to refer to the clone.
      for (unsigned j = 0, e = UserMI->getNumOperands(); j != e; ++j)
        if (UserMI->getOperand(j).isConstantPoolIndex()) {
          UserMI->getOperand(j).setConstantPoolIndex(CPEs[i].CPI);
          break;
        }
      // Adjust the refcount of the clone...
      CPEs[i].RefCount++;
      // ...and the original.  If we didn't remove the old entry, none of the
      // addresses changed, so we don't need another pass.
      unsigned Size = CPEMI->getOperand(2).getImm();
      return DecrementOldEntry(CPI, CPEMI, Size) ? 2 : 1;
    }
  }
  return 0;
}

/// getUnconditionalBrDisp - Returns the maximum displacement that can fit in
/// the specific unconditional branch instruction.
static inline unsigned getUnconditionalBrDisp(int Opc) {
  return (Opc == ARM::tB) ? ((1<<10)-1)*2 : ((1<<23)-1)*4;
}

/// HandleConstantPoolUser - Analyze the specified user, checking to see if it
/// is out-of-range.  If so, pick it up the constant pool value and move it some
/// place in-range.  Return true if we changed any addresses (thus must run
/// another pass of branch lengthening), false otherwise.
bool ARMConstantIslands::HandleConstantPoolUser(MachineFunction &Fn, 
                                                unsigned CPUserIndex){
  CPUser &U = CPUsers[CPUserIndex];
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;
  unsigned CPI = CPEMI->getOperand(1).getConstantPoolIndex();
  unsigned Size = CPEMI->getOperand(2).getImm();
  bool isThumb = AFI->isThumbFunction();
  MachineBasicBlock *NewMBB;
  // Compute this only once, it's expensive
  unsigned UserOffset = GetOffsetOf(UserMI) + (isThumb ? 4 : 8);
 
  // See if the current entry is within range, or there is a clone of it
  // in range.
  int result = LookForExistingCPEntry(U, UserOffset);
  if (result==1) return false;
  else if (result==2) return true;

  // No existing clone of this CPE is within range.
  // We will be generating a new clone.  Get a UID for it.
  unsigned ID  = NextUID++;

  // Look for water where we can place this CPE.  We look for the farthest one
  // away that will work.  Forward references only for now (although later
  // we might find some that are backwards).
  bool WaterFound = false;
  bool PadNewWater = true;
  if (!WaterList.empty()) {
    for (std::vector<MachineBasicBlock*>::iterator IP = prior(WaterList.end()),
        B = WaterList.begin();; --IP) {
      MachineBasicBlock* WaterBB = *IP;
      if (WaterIsInRange(UserOffset, WaterBB, U.MaxDisp)) {
        WaterFound = true;
        DOUT << "found water in range\n";
        // CPE goes before following block (NewMBB).
        NewMBB = next(MachineFunction::iterator(WaterBB));
        // If WaterBB is an island, don't pad the new island.
        // If WaterBB is empty, go backwards until we find something that
        // isn't.  WaterBB may become empty if it's an island whose
        // contents were moved farther back.
        if (isThumb) {
          MachineBasicBlock* BB = WaterBB;
          while (BB->empty())
            BB = BB->Prev;
          if (BB->begin()->getOpcode() == ARM::CONSTPOOL_ENTRY)
            PadNewWater = false;
        }
        // Remove the original WaterList entry; we want subsequent
        // insertions in this vicinity to go after the one we're
        // about to insert.  This considerably reduces the number
        // of times we have to move the same CPE more than once.
        WaterList.erase(IP);
        break;
      }
      if (IP == B)
        break;
    }
  }

  if (!WaterFound) {
    // No water found.

    DOUT << "No water found\n";
    MachineBasicBlock *UserMBB = UserMI->getParent();
    unsigned OffsetOfNextBlock = BBOffsets[UserMBB->getNumber()] + 
                                 BBSizes[UserMBB->getNumber()];
    assert(OffsetOfNextBlock = BBOffsets[UserMBB->getNumber()+1]);

    // If the use is at the end of the block, or the end of the block
    // is within range, make new water there.  (The +2 or 4 below is
    // for the unconditional branch we will be adding.  If the block ends in
    // an unconditional branch already, it is water, and is known to
    // be out of range, so we'll always be adding one.)
    if (&UserMBB->back() == UserMI ||
        OffsetIsInRange(UserOffset, OffsetOfNextBlock + (isThumb ? 2 : 4),
                        U.MaxDisp, !isThumb)) {
      DOUT << "Split at end of block\n";
      if (&UserMBB->back() == UserMI)
        assert(BBHasFallthrough(UserMBB) && "Expected a fallthrough BB!");
      NewMBB = next(MachineFunction::iterator(UserMBB));
      // Add an unconditional branch from UserMBB to fallthrough block.
      // Record it for branch lengthening; this new branch will not get out of
      // range, but if the preceding conditional branch is out of range, the
      // targets will be exchanged, and the altered branch may be out of
      // range, so the machinery has to know about it.
      int UncondBr = isThumb ? ARM::tB : ARM::B;
      BuildMI(UserMBB, TII->get(UncondBr)).addMBB(NewMBB);
      unsigned MaxDisp = getUnconditionalBrDisp(UncondBr);
      ImmBranches.push_back(ImmBranch(&UserMBB->back(), 
                            MaxDisp, false, UncondBr));
      int delta = isThumb ? 2 : 4;
      BBSizes[UserMBB->getNumber()] += delta;
      AdjustBBOffsetsAfter(UserMBB, delta);
    } else {
      // What a big block.  Find a place within the block to split it.
      // This is a little tricky on Thumb since instructions are 2 bytes
      // and constant pool entries are 4 bytes: if instruction I references
      // island CPE, and instruction I+1 references CPE', it will
      // not work well to put CPE as far forward as possible, since then
      // CPE' cannot immediately follow it (that location is 2 bytes
      // farther away from I+1 than CPE was from I) and we'd need to create
      // a new island.
      // The 4 in the following is for the unconditional branch we'll be
      // inserting (allows for long branch on Thumb).  The 2 or 0 is for
      // alignment of the island.
      unsigned BaseInsertOffset = UserOffset + U.MaxDisp -4 + (isThumb ? 2 : 0);
      // This could point off the end of the block if we've already got
      // constant pool entries following this block; only the last one is
      // in the water list.  Back past any possible branches.
      if (BaseInsertOffset >= BBOffsets[UserMBB->getNumber()+1])
        BaseInsertOffset = BBOffsets[UserMBB->getNumber()+1] - 6;
      unsigned EndInsertOffset = BaseInsertOffset +
             CPEMI->getOperand(2).getImm();
      MachineBasicBlock::iterator MI = UserMI;  ++MI;
      unsigned CPUIndex = CPUserIndex+1;
      for (unsigned Offset = UserOffset+ARM::GetInstSize(UserMI);
           Offset < BaseInsertOffset;
           Offset += ARM::GetInstSize(MI),
              MI = next(MI)) {
        if (CPUIndex < CPUsers.size() && CPUsers[CPUIndex].MI == MI) {
          if (!OffsetIsInRange(Offset, EndInsertOffset, 
                CPUsers[CPUIndex].MaxDisp, !isThumb)) {
            BaseInsertOffset -= (isThumb ? 2 : 4);
            EndInsertOffset -= (isThumb ? 2 : 4);
          }
          // This is overly conservative, as we don't account for CPEMIs
          // being reused within the block, but it doesn't matter much.
          EndInsertOffset += CPUsers[CPUIndex].CPEMI->getOperand(2).getImm();
          CPUIndex++;
        }
      }
      DOUT << "Split in middle of big block\n";
      NewMBB = SplitBlockBeforeInstr(prior(MI));
    }
  }

  // Okay, we know we can put an island before NewMBB now, do it!
  MachineBasicBlock *NewIsland = new MachineBasicBlock();
  Fn.getBasicBlockList().insert(NewMBB, NewIsland);

  // Update internal data structures to account for the newly inserted MBB.
  UpdateForInsertedWaterBlock(NewIsland);

  // Decrement the old entry, and remove it if refcount becomes 0.
  DecrementOldEntry(CPI, CPEMI, Size);

  // Now that we have an island to add the CPE to, clone the original CPE and
  // add it to the island.
  U.CPEMI = BuildMI(NewIsland, TII->get(ARM::CONSTPOOL_ENTRY))
                .addImm(ID).addConstantPoolIndex(CPI).addImm(Size);
  CPEntries[CPI].push_back(CPEntry(U.CPEMI, ID, 1));
  NumCPEs++;

  // Compensate for .align 2 in thumb mode.
  if (isThumb && PadNewWater) Size += 2;
  // Increase the size of the island block to account for the new entry.
  BBSizes[NewIsland->getNumber()] += Size;
  BBOffsets[NewIsland->getNumber()] = BBOffsets[NewMBB->getNumber()];
  AdjustBBOffsetsAfter(NewIsland, Size);
  
  // Finally, change the CPI in the instruction operand to be ID.
  for (unsigned i = 0, e = UserMI->getNumOperands(); i != e; ++i)
    if (UserMI->getOperand(i).isConstantPoolIndex()) {
      UserMI->getOperand(i).setConstantPoolIndex(ID);
      break;
    }
      
  DOUT << "  Moved CPE to #" << ID << " CPI=" << CPI << "\t" << *UserMI;
      
  return true;
}

/// BBIsInRange - Returns true if the distance between specific MI and
/// specific BB can fit in MI's displacement field.
bool ARMConstantIslands::BBIsInRange(MachineInstr *MI,MachineBasicBlock *DestBB,
                                     unsigned MaxDisp) {
  unsigned PCAdj      = AFI->isThumbFunction() ? 4 : 8;
  unsigned BrOffset   = GetOffsetOf(MI) + PCAdj;
  unsigned DestOffset = BBOffsets[DestBB->getNumber()];

  DOUT << "Branch of destination BB#" << DestBB->getNumber()
       << " from BB#" << MI->getParent()->getNumber()
       << " max delta=" << MaxDisp
       << " at offset " << int(DestOffset-BrOffset) << "\t" << *MI;

  return OffsetIsInRange(BrOffset, DestOffset, MaxDisp, true);
}

/// FixUpImmediateBr - Fix up an immediate branch whose destination is too far
/// away to fit in its displacement field.
bool ARMConstantIslands::FixUpImmediateBr(MachineFunction &Fn, ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = MI->getOperand(0).getMachineBasicBlock();

  // Check to see if the DestBB is already in-range.
  if (BBIsInRange(MI, DestBB, Br.MaxDisp))
    return false;

  if (!Br.isCond)
    return FixUpUnconditionalBr(Fn, Br);
  return FixUpConditionalBr(Fn, Br);
}

/// FixUpUnconditionalBr - Fix up an unconditional branch whose destination is
/// too far away to fit in its displacement field. If the LR register has been
/// spilled in the epilogue, then we can use BL to implement a far jump.
/// Otherwise, add an intermediate branch instruction to to a branch.
bool
ARMConstantIslands::FixUpUnconditionalBr(MachineFunction &Fn, ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *MBB = MI->getParent();
  assert(AFI->isThumbFunction() && "Expected a Thumb function!");

  // Use BL to implement far jump.
  Br.MaxDisp = (1 << 21) * 2;
  MI->setInstrDescriptor(TII->get(ARM::tBfar));
  BBSizes[MBB->getNumber()] += 2;
  AdjustBBOffsetsAfter(MBB, 2);
  HasFarJump = true;
  NumUBrFixed++;

  DOUT << "  Changed B to long jump " << *MI;

  return true;
}

/// FixUpConditionalBr - Fix up a conditional branch whose destination is too
/// far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool
ARMConstantIslands::FixUpConditionalBr(MachineFunction &Fn, ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = MI->getOperand(0).getMachineBasicBlock();

  // Add a unconditional branch to the destination and invert the branch
  // condition to jump over it:
  // blt L1
  // =>
  // bge L2
  // b   L1
  // L2:
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(1).getImmedValue();
  CC = ARMCC::getOppositeCondition(CC);

  // If the branch is at the end of its MBB and that has a fall-through block,
  // direct the updated conditional branch to the fall-through block. Otherwise,
  // split the MBB before the next instruction.
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstr *BMI = &MBB->back();
  bool NeedSplit = (BMI != MI) || !BBHasFallthrough(MBB);

  NumCBrFixed++;
  if (BMI != MI) {
    if (next(MachineBasicBlock::iterator(MI)) == MBB->back() &&
        BMI->getOpcode() == Br.UncondBr) {
      // Last MI in the BB is a unconditional branch. Can we simply invert the
      // condition and swap destinations:
      // beq L1
      // b   L2
      // =>
      // bne L2
      // b   L1
      MachineBasicBlock *NewDest = BMI->getOperand(0).getMachineBasicBlock();
      if (BBIsInRange(MI, NewDest, Br.MaxDisp)) {
        DOUT << "  Invert Bcc condition and swap its destination with " << *BMI;
        BMI->getOperand(0).setMachineBasicBlock(DestBB);
        MI->getOperand(0).setMachineBasicBlock(NewDest);
        MI->getOperand(1).setImm(CC);
        return true;
      }
    }
  }

  if (NeedSplit) {
    SplitBlockBeforeInstr(MI);
    // No need for the branch to the next block. We're adding a unconditional
    // branch to the destination.
    MBB->back().eraseFromParent();
  }
  MachineBasicBlock *NextBB = next(MachineFunction::iterator(MBB));
 
  DOUT << "  Insert B to BB#" << DestBB->getNumber()
       << " also invert condition and change dest. to BB#"
       << NextBB->getNumber() << "\n";

  // Insert a unconditional branch and replace the conditional branch.
  // Also update the ImmBranch as well as adding a new entry for the new branch.
  BuildMI(MBB, TII->get(MI->getOpcode())).addMBB(NextBB).addImm(CC);
  Br.MI = &MBB->back();
  BuildMI(MBB, TII->get(Br.UncondBr)).addMBB(DestBB);
  unsigned MaxDisp = getUnconditionalBrDisp(Br.UncondBr);
  ImmBranches.push_back(ImmBranch(&MBB->back(), MaxDisp, false, Br.UncondBr));
  MI->eraseFromParent();

  // Increase the size of MBB to account for the new unconditional branch.
  int delta = ARM::GetInstSize(&MBB->back());
  BBSizes[MBB->getNumber()] += delta;
  AdjustBBOffsetsAfter(MBB, delta);
  return true;
}

/// UndoLRSpillRestore - Remove Thumb push / pop instructions that only spills
/// LR / restores LR to pc.
bool ARMConstantIslands::UndoLRSpillRestore() {
  bool MadeChange = false;
  for (unsigned i = 0, e = PushPopMIs.size(); i != e; ++i) {
    MachineInstr *MI = PushPopMIs[i];
    if (MI->getNumOperands() == 1) {
        if (MI->getOpcode() == ARM::tPOP_RET &&
            MI->getOperand(0).getReg() == ARM::PC)
          BuildMI(MI->getParent(), TII->get(ARM::tBX_RET));
        MI->eraseFromParent();
        MadeChange = true;
    }
  }
  return MadeChange;
}

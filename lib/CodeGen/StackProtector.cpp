//===-- StackProtector.cpp - Stack Protector Insertion --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass inserts stack protectors into functions which need them. A variable
// with a random value in it is stored onto the stack before the local variables
// are allocated. Upon exiting the block, the stored value is checked. If it's
// changed, then there was some sort of violation and the program aborts.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "stack-protector"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

// Enable stack protectors.
static cl::opt<unsigned>
SSPBufferSize("stack-protector-buffer-size", cl::init(8),
              cl::desc("The lower bound for a buffer to be considered for "
                       "stack smashing protection."));

namespace {
  class VISIBILITY_HIDDEN StackProtector : public FunctionPass {
    /// Level - The level of stack protection.
    SSP::StackProtectorLevel Level;

    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// target type sizes.
    const TargetLowering *TLI;

    Function *F;
    Module *M;

    /// InsertStackProtectors - Insert code into the prologue and epilogue of
    /// the function.
    ///
    ///  - The prologue code loads and stores the stack guard onto the stack.
    ///  - The epilogue checks the value stored in the prologue against the
    ///    original value. It calls __stack_chk_fail if they differ.
    bool InsertStackProtectors();

    /// CreateFailBB - Create a basic block to jump to when the stack protector
    /// check fails.
    BasicBlock *CreateFailBB();

    /// RequiresStackProtector - Check whether or not this function needs a
    /// stack protector based upon the stack protector level.
    bool RequiresStackProtector() const;
  public:
    static char ID;             // Pass identification, replacement for typeid.
    StackProtector() : FunctionPass(&ID), Level(SSP::OFF), TLI(0) {}
    StackProtector(SSP::StackProtectorLevel lvl, const TargetLowering *tli)
      : FunctionPass(&ID), Level(lvl), TLI(tli) {}

    virtual bool runOnFunction(Function &Fn);
  };
} // end anonymous namespace

char StackProtector::ID = 0;
static RegisterPass<StackProtector>
X("stack-protector", "Insert stack protectors");

FunctionPass *llvm::createStackProtectorPass(SSP::StackProtectorLevel lvl,
                                             const TargetLowering *tli) {
  return new StackProtector(lvl, tli);
}

bool StackProtector::runOnFunction(Function &Fn) {
  F = &Fn;
  M = F->getParent();

  if (!RequiresStackProtector()) return false;
  
  return InsertStackProtectors();
}

/// InsertStackProtectors - Insert code into the prologue and epilogue of the
/// function.
///
///  - The prologue code loads and stores the stack guard onto the stack.
///  - The epilogue checks the value stored in the prologue against the original
///    value. It calls __stack_chk_fail if they differ.
bool StackProtector::InsertStackProtectors() {
  std::vector<BasicBlock*> ReturnBBs;

  for (Function::iterator I = F->begin(); I != F->end(); ++I)
    if (isa<ReturnInst>(I->getTerminator()))
      ReturnBBs.push_back(I);

  // If this function doesn't return, don't bother with stack protectors.
  if (ReturnBBs.empty()) return false;

  // Insert code into the entry block that stores the __stack_chk_guard variable
  // onto the stack.
  BasicBlock &Entry = F->getEntryBlock();
  Instruction *InsertPt = &Entry.front();
  const PointerType *GuardTy = PointerType::getUnqual(Type::Int8Ty);

  // The global variable for the stack guard.
  Constant *StackGuardVar = M->getOrInsertGlobal("__stack_chk_guard", GuardTy);

  // The place on the stack that the stack protector guard is kept.
  AllocaInst *StackProtFrameSlot =
    new AllocaInst(GuardTy, "StackProt_Frame", InsertPt);
  LoadInst *LI = new LoadInst(StackGuardVar, "StackGuard", false, InsertPt);
  new StoreInst(LI, StackProtFrameSlot, false, InsertPt);

  // Create the basic block to jump to when the guard check fails.
  BasicBlock *FailBB = CreateFailBB();

  // Loop through the basic blocks that have return instructions. Convert this:
  //
  //   return:
  //     ...
  //     ret ...
  //
  // into this:
  //
  //   return:
  //     ...
  //     %1 = load __stack_chk_guard
  //     %2 = load <stored stack guard>
  //     %3 = cmp i1 %1, %2
  //     br i1 %3, label %SPRet, label %CallStackCheckFailBlk
  //
  //   SP_return:
  //     ret ...
  //
  //   CallStackCheckFailBlk:
  //     call void @__stack_chk_fail()
  //     unreachable
  //
  for (std::vector<BasicBlock*>::iterator
         I = ReturnBBs.begin(), E = ReturnBBs.end(); I != E; ++I) {
    BasicBlock *BB = *I;
    ReturnInst *RI = cast<ReturnInst>(BB->getTerminator());
    Function::iterator InsPt = BB; ++InsPt; // Insertion point for new BB.

    // Split the basic block before the return instruction.
    BasicBlock *NewBB = BB->splitBasicBlock(RI, "SP_return");

    // Move the newly created basic block to the point right after the old basic
    // block so that it's in the "fall through" position.
    NewBB->removeFromParent();
    F->getBasicBlockList().insert(InsPt, NewBB);

    // Generate the stack protector instructions in the old basic block.
    LoadInst *LI2 = new LoadInst(StackGuardVar, "", false, BB);
    LoadInst *LI1 = new LoadInst(StackProtFrameSlot, "", true, BB);
    ICmpInst *Cmp = new ICmpInst(CmpInst::ICMP_EQ, LI1, LI2, "", BB);
    BranchInst::Create(NewBB, FailBB, Cmp, BB);
  }

  return true;
}

/// CreateFailBB - Create a basic block to jump to when the stack protector
/// check fails.
BasicBlock *StackProtector::CreateFailBB() {
  BasicBlock *FailBB = BasicBlock::Create("CallStackCheckFailBlk", F);
  std::vector<const Type*> Params;
  Constant *StackChkFail =
    M->getOrInsertFunction("__stack_chk_fail", Type::VoidTy, NULL);
  CallInst::Create(StackChkFail, "", FailBB);
  new UnreachableInst(FailBB);
  return FailBB;
}

/// RequiresStackProtector - Check whether or not this function needs a stack
/// protector based upon the stack protector level.
bool StackProtector::RequiresStackProtector() const {
  switch (Level) {
  default: return false;
  case SSP::ALL: return true;
  case SSP::SOME: {
    // If the size of the local variables allocated on the stack is greater than
    // SSPBufferSize, then we require a stack protector.
    uint64_t StackSize = 0;
    const TargetData *TD = TLI->getTargetData();

    for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
      BasicBlock *BB = I;

      for (BasicBlock::iterator
             II = BB->begin(), IE = BB->end(); II != IE; ++II)
        if (AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
          if (ConstantInt *CI = dyn_cast<ConstantInt>(AI->getArraySize())) {
            uint64_t Bytes = TD->getTypeSizeInBits(AI->getAllocatedType()) / 8;
            const APInt &Size = CI->getValue();
            StackSize += Bytes * Size.getZExtValue();

            if (SSPBufferSize <= StackSize)
              return true;
          }
        }
    }

    return false;
  }
  }
}

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
#include "llvm/Attributes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

// SSPBufferSize - The lower bound for a buffer to be considered for stack
// smashing protection.
static cl::opt<unsigned>
SSPBufferSize("stack-protector-buffer-size", cl::init(8),
              cl::desc("The lower bound for a buffer to be considered for "
                       "stack smashing protection."));

namespace {
  class VISIBILITY_HIDDEN StackProtector : public FunctionPass {
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
    StackProtector() : FunctionPass(&ID), TLI(0) {}
    StackProtector(const TargetLowering *tli)
      : FunctionPass(&ID), TLI(tli) {}

    virtual bool runOnFunction(Function &Fn);
  };
} // end anonymous namespace

char StackProtector::ID = 0;
static RegisterPass<StackProtector>
X("stack-protector", "Insert stack protectors");

FunctionPass *llvm::createStackProtectorPass(const TargetLowering *tli) {
  return new StackProtector(tli);
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
  //     br i1 %3, label %SP_return, label %CallStackCheckFailBlk
  //
  //   SP_return:
  //     ret ...
  //
  //   CallStackCheckFailBlk:
  //     call void @__stack_chk_fail()
  //     unreachable
  //
  BasicBlock *FailBB = 0;       // The basic block to jump to if check fails.
  AllocaInst *AI = 0;           // Place on stack that stores the stack guard.
  Constant *StackGuardVar = 0;  // The stack guard variable.

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ) {
    BasicBlock *BB = I;

    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
      if (!FailBB) {
        // Insert code into the entry block that stores the __stack_chk_guard
        // variable onto the stack.
        PointerType *PtrTy = PointerType::getUnqual(Type::Int8Ty);
        StackGuardVar = M->getOrInsertGlobal("__stack_chk_guard", PtrTy);

        BasicBlock &Entry = F->getEntryBlock();
        Instruction *InsPt = &Entry.front();

        AI = new AllocaInst(PtrTy, "StackGuardSlot", InsPt);
        LoadInst *LI = new LoadInst(StackGuardVar, "StackGuard", false, InsPt);

        Value *Args[] = { LI, AI };
        CallInst::
          Create(Intrinsic::getDeclaration(M, Intrinsic::stackprotector_create),
                 &Args[0], array_endof(Args), "", InsPt);

        // Create the basic block to jump to when the guard check fails.
        FailBB = CreateFailBB();
      }

      ++I; // Skip to the next block so that we don't resplit the return block.

      // Split the basic block before the return instruction.
      BasicBlock *NewBB = BB->splitBasicBlock(RI, "SP_return");

      // Move the newly created basic block to the point right after the old
      // basic block so that it's in the "fall through" position.
      NewBB->removeFromParent();
      F->getBasicBlockList().insert(I, NewBB);

      // Generate the stack protector instructions in the old basic block.
      LoadInst *LI1 = new LoadInst(StackGuardVar, "", false, BB);
      CallInst *CI = CallInst::
        Create(Intrinsic::getDeclaration(M, Intrinsic::stackprotector_check),
               AI, "", BB);
      ICmpInst *Cmp = new ICmpInst(CmpInst::ICMP_EQ, CI, LI1, "", BB);
      BranchInst::Create(NewBB, FailBB, Cmp, BB);
    } else {
      ++I;
    }
  }

  // Return if we didn't modify any basic blocks. I.e., there are no return
  // statements in the function.
  if (!FailBB) return false;

  return true;
}

/// CreateFailBB - Create a basic block to jump to when the stack protector
/// check fails.
BasicBlock *StackProtector::CreateFailBB() {
  BasicBlock *FailBB = BasicBlock::Create("CallStackCheckFailBlk", F);
  Constant *StackChkFail =
    M->getOrInsertFunction("__stack_chk_fail", Type::VoidTy, NULL);
  CallInst::Create(StackChkFail, "", FailBB);
  new UnreachableInst(FailBB);
  return FailBB;
}

/// RequiresStackProtector - Check whether or not this function needs a stack
/// protector based upon the stack protector level. The heuristic we use is to
/// add a guard variable to functions that call alloca, and functions with
/// buffers larger than 8 bytes.
bool StackProtector::RequiresStackProtector() const {
  if (F->hasFnAttr(Attribute::StackProtectReq))
      return true;

  if (F->hasFnAttr(Attribute::StackProtect)) {
    const TargetData *TD = TLI->getTargetData();

    for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
      BasicBlock *BB = I;

      for (BasicBlock::iterator
             II = BB->begin(), IE = BB->end(); II != IE; ++II)
        if (AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
          if (AI->isArrayAllocation())
            // This is a call to alloca with a variable size. Emit stack
            // protectors.
            return true;

          if (const ArrayType *AT = dyn_cast<ArrayType>(AI->getAllocatedType()))
            // If an array has more than 8 bytes of allocated space, then we
            // emit stack protectors.
            if (SSPBufferSize <= TD->getABITypeSize(AT))
              return true;
        }
    }

    return false;
  }

  return false;
}

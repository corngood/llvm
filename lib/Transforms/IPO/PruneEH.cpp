//===- PruneEH.cpp - Pass which deletes unused exception handlers ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple interprocedural pass which walks the
// call-graph, turning invoke instructions into calls, iff the callee cannot
// throw an exception.  It implements this as a bottom-up traversal of the
// call-graph.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/Statistic.h"
#include <set>
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumRemoved("prune-eh", "Number of invokes removed");

  struct PruneEH : public CallGraphSCCPass {
    /// DoesNotUnwind - This set contains all of the functions which we have
    /// determined cannot throw exceptions.
    std::set<CallGraphNode*> DoesNotUnwind;

    // runOnSCC - Analyze the SCC, performing the transformation if possible.
    bool runOnSCC(const std::vector<CallGraphNode *> &SCC);
  };
  RegisterOpt<PruneEH> X("prune-eh", "Remove unused exception handling info");
}

ModulePass *llvm::createPruneEHPass() { return new PruneEH(); }


bool PruneEH::runOnSCC(const std::vector<CallGraphNode *> &SCC) {
  CallGraph &CG = getAnalysis<CallGraph>();

  // First, check to see if any callees might throw or if there are any external
  // functions in this SCC: if so, we cannot prune any functions in this SCC.
  // If this SCC includes the unwind instruction, we KNOW it throws, so
  // obviously the SCC might throw.
  //
  bool SCCMightUnwind = false;
  for (unsigned i = 0, e = SCC.size(); !SCCMightUnwind && i != e; ++i) {
    Function *F = SCC[i]->getFunction();
    if (F == 0 || (F->isExternal() && !F->getIntrinsicID())) {
      SCCMightUnwind = true;
    } else {
      // Check to see if this function performs an unwind or calls an
      // unwinding function.
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
        if (isa<UnwindInst>(BB->getTerminator())) {  // Uses unwind!
          SCCMightUnwind = true;
          break;
        }

        // Invoke instructions don't allow unwinding to continue, so we are
        // only interested in call instructions.
        for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
          if (CallInst *CI = dyn_cast<CallInst>(I)) {
            if (Function *Callee = CI->getCalledFunction()) {
              CallGraphNode *CalleeNode = CG[Callee];
              // If the callee is outside our current SCC, or if it is not
              // known to throw, then we might throw also.
              if (std::find(SCC.begin(), SCC.end(), CalleeNode) == SCC.end()&&
                  !DoesNotUnwind.count(CalleeNode)) {
                SCCMightUnwind = true;
                break;
              }

            } else {
              // Indirect call, it might throw.
              SCCMightUnwind = true;
              break;
            }
          }
        if (SCCMightUnwind) break;
      }
    }
  }
  bool MadeChange = false;

  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    // If the SCC can't throw, remember this for callers...
    if (!SCCMightUnwind)
      DoesNotUnwind.insert(SCC[i]);

    // Convert any invoke instructions to non-throwing functions in this node
    // into call instructions with a branch.  This makes the exception blocks
    // dead.
    if (Function *F = SCC[i]->getFunction())
      for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
        if (InvokeInst *II = dyn_cast<InvokeInst>(I->getTerminator()))
          if (Function *F = II->getCalledFunction())
            if (DoesNotUnwind.count(CG[F])) {
              // Insert a call instruction before the invoke...
              std::string Name = II->getName();  II->setName("");
              Value *Call = new CallInst(II->getCalledValue(),
                                         std::vector<Value*>(II->op_begin()+3,
                                                             II->op_end()),
                                         Name, II);

              // Anything that used the value produced by the invoke instruction
              // now uses the value produced by the call instruction.
              II->replaceAllUsesWith(Call);
              II->getUnwindDest()->removePredecessor(II->getParent());

              // Insert a branch to the normal destination right before the
              // invoke.
              new BranchInst(II->getNormalDest(), II);

              // Finally, delete the invoke instruction!
              I->getInstList().pop_back();

              ++NumRemoved;
              MadeChange = true;
            }
  }

  return MadeChange;
}


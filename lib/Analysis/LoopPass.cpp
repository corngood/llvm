//===- LoopPass.cpp - Loop Pass and Loop Pass Manager ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements LoopPass and LPPassManager. All loop optimization
// and transformation passes are derived from LoopPass. LPPassManager is
// responsible for managing LoopPasses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPass.h"
#include <queue>
using namespace llvm;

//===----------------------------------------------------------------------===//
// LoopQueue

namespace llvm {

// Compare Two loops based on their depth in loop nest.
class LoopCompare {
public:
  bool operator()( Loop *L1, Loop *L2) const {
    return L1->getLoopDepth() > L2->getLoopDepth();
  }
};

// Loop queue used by Loop Pass Manager. This is a wrapper class
// that hides implemenation detail (use of priority_queue) inside .cpp file.
class LoopQueue {
public:
  inline void push(Loop *L) { LPQ.push(L); }
  inline void pop() { LPQ.pop(); }
  inline Loop *top() { return LPQ.top(); }
  inline bool empty() { return LPQ.empty(); }
private:
  std::priority_queue<Loop *, std::vector<Loop *>, LoopCompare> LPQ;
};

} // End of LLVM namespace

//===----------------------------------------------------------------------===//
// LPPassManager
//
/// LPPassManager manages FPPassManagers and CalLGraphSCCPasses.

LPPassManager::LPPassManager(int Depth) : PMDataManager(Depth) { 
  LQ = new LoopQueue(); 
}

LPPassManager::~LPPassManager() {
  delete LQ;
}

/// Delete loop from the loop queue. This is used by Loop pass to inform
/// Loop Pass Manager that it should skip rest of the passes for this loop.
void LPPassManager::deleteLoopFromQueue(Loop *L) {
  // Do not pop loop from LQ here. It will be done by runOnFunction while loop.
  skipThisLoop = true;
}

// Recurse through all subloops and all loops  into LQ.
static void addLoopIntoQueue(Loop *L, LoopQueue *LQ) {
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);
  LQ->push(L);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the function, and if so, return true.
bool LPPassManager::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfo>();
  bool Changed = false;

  // Populate Loop Queue
  for (LoopInfo::iterator I = LI.begin(), E = LI.end(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);

  std::string Msg1 = "Executing Pass '";
  std::string Msg3 = "' Made Modification '";

  // Walk Loops
  while (!LQ->empty()) {
      
    Loop *L  = LQ->top();
    skipThisLoop = false;

    // Run all passes on current SCC
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {  
        
      Pass *P = getContainedPass(Index);
      AnalysisUsage AnUsage;
      P->getAnalysisUsage(AnUsage);

      std::string Msg2 = "' on Loop ...\n'";
      dumpPassInfo(P, Msg1, Msg2);
      dumpAnalysisSetInfo("Required", P, AnUsage.getRequiredSet());

      initializeAnalysisImpl(P);

      StartPassTimer(P);
      LoopPass *LP = dynamic_cast<LoopPass *>(P);
      assert (LP && "Invalid LPPassManager member");
      LP->runOnLoop(*L, *this);
      StopPassTimer(P);

      if (Changed)
	dumpPassInfo(P, Msg3, Msg2);
      dumpAnalysisSetInfo("Preserved", P, AnUsage.getPreservedSet());
      
      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, Msg2);

      if (skipThisLoop)
        // Do not run other passes on this loop.
        break;
    }
    
    // Pop the loop from queue after running all passes.
    LQ->pop();
  }

  return Changed;
}



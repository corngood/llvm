//===- StrongPhiElimination.cpp - Eliminate PHI nodes by inserting copies -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates machine instruction PHI nodes by inserting copy
// instructions, using an intelligent copy-folding technique based on
// dominator information.  This is technique is derived from:
// 
//    Budimlic, et al. Fast copy coalescing and live-range identification.
//    In Proceedings of the ACM SIGPLAN 2002 Conference on Programming Language
//    Design and Implementation (Berlin, Germany, June 17 - 19, 2002).
//    PLDI '02. ACM, New York, NY, 25-32.
//    DOI= http://doi.acm.org/10.1145/512529.512534
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strongphielim"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;


namespace {
  struct VISIBILITY_HIDDEN StrongPHIElimination : public MachineFunctionPass {
    static char ID; // Pass identification, replacement for typeid
    StrongPHIElimination() : MachineFunctionPass((intptr_t)&ID) {}

    bool runOnMachineFunction(MachineFunction &Fn);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<LiveVariables>();
      AU.addPreservedID(PHIEliminationID);
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<LiveVariables>();
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    
    virtual void releaseMemory() {
      preorder.clear();
      maxpreorder.clear();
      
      waiting.clear();
    }

  private:
    struct DomForestNode {
    private:
      std::vector<DomForestNode*> children;
      unsigned reg;
      
      void addChild(DomForestNode* DFN) { children.push_back(DFN); }
      
    public:
      typedef std::vector<DomForestNode*>::iterator iterator;
      
      DomForestNode(unsigned r, DomForestNode* parent) : reg(r) {
        if (parent)
          parent->addChild(this);
      }
      
      ~DomForestNode() {
        for (iterator I = begin(), E = end(); I != E; ++I)
          delete *I;
      }
      
      inline unsigned getReg() { return reg; }
      
      inline DomForestNode::iterator begin() { return children.begin(); }
      inline DomForestNode::iterator end() { return children.end(); }
    };
    
    DenseMap<MachineBasicBlock*, unsigned> preorder;
    DenseMap<MachineBasicBlock*, unsigned> maxpreorder;
    
    DenseMap<MachineBasicBlock*, std::vector<MachineInstr*> > waiting;
    
    
    void computeDFS(MachineFunction& MF);
    void processPHI(MachineInstr* P);
    
    std::vector<DomForestNode*> computeDomForest(std::set<unsigned>& instrs);
    
  };

  char StrongPHIElimination::ID = 0;
  RegisterPass<StrongPHIElimination> X("strong-phi-node-elimination",
                  "Eliminate PHI nodes for register allocation, intelligently");
}

const PassInfo *llvm::StrongPHIEliminationID = X.getPassInfo();

/// computeDFS - Computes the DFS-in and DFS-out numbers of the dominator tree
/// of the given MachineFunction.  These numbers are then used in other parts
/// of the PHI elimination process.
void StrongPHIElimination::computeDFS(MachineFunction& MF) {
  SmallPtrSet<MachineDomTreeNode*, 8> frontier;
  SmallPtrSet<MachineDomTreeNode*, 8> visited;
  
  unsigned time = 0;
  
  MachineDominatorTree& DT = getAnalysis<MachineDominatorTree>();
  
  MachineDomTreeNode* node = DT.getRootNode();
  
  std::vector<MachineDomTreeNode*> worklist;
  worklist.push_back(node);
  
  while (!worklist.empty()) {
    MachineDomTreeNode* currNode = worklist.back();
    
    if (!frontier.count(currNode)) {
      frontier.insert(currNode);
      ++time;
      preorder.insert(std::make_pair(currNode->getBlock(), time));
    }
    
    bool inserted = false;
    for (MachineDomTreeNode::iterator I = node->begin(), E = node->end();
         I != E; ++I)
      if (!frontier.count(*I) && !visited.count(*I)) {
        worklist.push_back(*I);
        inserted = true;
        break;
      }
    
    if (!inserted) {
      frontier.erase(currNode);
      visited.insert(currNode);
      maxpreorder.insert(std::make_pair(currNode->getBlock(), time));
      
      worklist.pop_back();
    }
  }
}

/// PreorderSorter - a helper class that is used to sort registers
/// according to the preorder number of their defining blocks
class PreorderSorter {
private:
  DenseMap<MachineBasicBlock*, unsigned>& preorder;
  LiveVariables& LV;
  
public:
  PreorderSorter(DenseMap<MachineBasicBlock*, unsigned>& p,
                LiveVariables& L) : preorder(p), LV(L) { }
  
  bool operator()(unsigned A, unsigned B) {
    if (A == B)
      return false;
    
    MachineBasicBlock* ABlock = LV.getVarInfo(A).DefInst->getParent();
    MachineBasicBlock* BBlock = LV.getVarInfo(A).DefInst->getParent();
    
    if (preorder[ABlock] < preorder[BBlock])
      return true;
    else if (preorder[ABlock] > preorder[BBlock])
      return false;
    
    assert(0 && "Error sorting by dominance!");
    return false;
  }
};

/// computeDomForest - compute the subforest of the DomTree corresponding
/// to the defining blocks of the registers in question
std::vector<StrongPHIElimination::DomForestNode*>
StrongPHIElimination::computeDomForest(std::set<unsigned>& regs) {
  LiveVariables& LV = getAnalysis<LiveVariables>();
  
  DomForestNode* VirtualRoot = new DomForestNode(0, 0);
  maxpreorder.insert(std::make_pair((MachineBasicBlock*)0, ~0UL));
  
  std::vector<unsigned> worklist;
  worklist.reserve(regs.size());
  for (std::set<unsigned>::iterator I = regs.begin(), E = regs.end();
       I != E; ++I)
    worklist.push_back(*I);
  
  PreorderSorter PS(preorder, LV);
  std::sort(worklist.begin(), worklist.end(), PS);
  
  DomForestNode* CurrentParent = VirtualRoot;
  std::vector<DomForestNode*> stack;
  stack.push_back(VirtualRoot);
  
  for (std::vector<unsigned>::iterator I = worklist.begin(), E = worklist.end();
       I != E; ++I) {
    unsigned pre = preorder[LV.getVarInfo(*I).DefInst->getParent()];
    MachineBasicBlock* parentBlock =
      LV.getVarInfo(CurrentParent->getReg()).DefInst->getParent();
    
    while (pre > maxpreorder[parentBlock]) {
      stack.pop_back();
      CurrentParent = stack.back();
      
      parentBlock = LV.getVarInfo(CurrentParent->getReg()).DefInst->getParent();
    }
    
    DomForestNode* child = new DomForestNode(*I, CurrentParent);
    stack.push_back(child);
    CurrentParent = child;
  }
  
  std::vector<DomForestNode*> ret;
  ret.insert(ret.end(), VirtualRoot->begin(), VirtualRoot->end());
  return ret;
}

/// processPHI - Eliminate the given PHI node
void StrongPHIElimination::processPHI(MachineInstr* P) {
  
}

bool StrongPHIElimination::runOnMachineFunction(MachineFunction &Fn) {
  computeDFS(Fn);
  
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    for (MachineBasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;
         ++BI) {
      if (BI->getOpcode() == TargetInstrInfo::PHI)
        processPHI(BI);
      else
        break;
    }
  }
  
  return false;
}

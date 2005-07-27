//===-- MSScheduleSB.h - Schedule ------- -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The schedule generated by a scheduling algorithm
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MSSCHEDULESB_H
#define LLVM_MSSCHEDULESB_H

#include "MSchedGraphSB.h"
#include <vector>
#include <set>

namespace llvm {

  class MSScheduleSB {
    std::map<int, std::vector<MSchedGraphSBNode*> > schedule;
    unsigned numIssue;

    //Internal map to keep track of explicit resources
    std::map<int, std::map<int, int> > resourceNumPerCycle;

    //Check if all resources are free
    bool resourcesFree(MSchedGraphSBNode*, int, int II);
    bool resourceAvailable(int resourceNum, int cycle);
    void useResource(int resourceNum, int cycle);

    //Resulting kernel
    std::vector<std::pair<MachineInstr*, int> > kernel;

    //Max stage count
    int maxStage;

    //add at the right spot in the schedule
    void addToSchedule(int, MSchedGraphSBNode*);

  public:
    MSScheduleSB(int num) : numIssue(num) {}
    MSScheduleSB() : numIssue(4) {}
    bool insert(MSchedGraphSBNode *node, int cycle, int II);
    int getStartCycle(MSchedGraphSBNode *node);
    void clear() { schedule.clear(); resourceNumPerCycle.clear(); kernel.clear(); }
    std::vector<std::pair<MachineInstr*, int> >* getKernel() { return &kernel; }
    bool constructKernel(int II, std::vector<MSchedGraphSBNode*> &branches, std::map<const MachineInstr*, unsigned> &indVar);
    int getMaxStage() { return maxStage; }
    bool defPreviousStage(Value *def, int stage);

    //iterators
    typedef std::map<int, std::vector<MSchedGraphSBNode*> >::iterator schedule_iterator;
    typedef std::map<int, std::vector<MSchedGraphSBNode*> >::const_iterator schedule_const_iterator;
    schedule_iterator begin() { return schedule.begin(); };
    schedule_iterator end() { return schedule.end(); };
    void print(std::ostream &os) const;
    void printSchedule(std::ostream &os) const;

    typedef std::vector<std::pair<MachineInstr*, int> >::iterator kernel_iterator;
    typedef std::vector<std::pair<MachineInstr*, int> >::const_iterator kernel_const_iterator;
    kernel_iterator kernel_begin() { return kernel.begin(); }
    kernel_iterator kernel_end() { return kernel.end(); }

  };

}


#endif

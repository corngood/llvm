//===-- PrologEpilogInserter.cpp - Insert Prolog/Epilog code in function --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Compiler.h"
#include <climits>
using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN PEI : public MachineFunctionPass {
    const char *getPassName() const {
      return "Prolog/Epilog Insertion & Frame Finalization";
    }

    /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
    /// frame indexes with appropriate references.
    ///
    bool runOnMachineFunction(MachineFunction &Fn) {
      // Get MachineDebugInfo so that we can track the construction of the
      // frame.
      if (MachineDebugInfo *DI = getAnalysisToUpdate<MachineDebugInfo>()) {
        Fn.getFrameInfo()->setMachineDebugInfo(DI);
      }

      // Allow the target machine to make some adjustments to the function
      // e.g. UsedPhysRegs before calculateCalleeSavedRegisters.
      Fn.getTarget().getRegisterInfo()->processFunctionBeforeCalleeSaveScan(Fn);

      // Scan the function for modified callee saved registers and insert spill
      // code for any callee saved registers that are modified.  Also calculate
      // the MaxCallFrameSize and HasCalls variables for the function's frame
      // information and eliminates call frame pseudo instructions.
      calculateCalleeSavedRegisters(Fn);

      // Add the code to save and restore the callee saved registers
      saveCalleeSavedRegisters(Fn);

      // Allow the target machine to make final modifications to the function
      // before the frame layout is finalized.
      Fn.getTarget().getRegisterInfo()->processFunctionBeforeFrameFinalized(Fn);

      // Calculate actual frame offsets for all of the abstract stack objects...
      calculateFrameObjectOffsets(Fn);

      // Add prolog and epilog code to the function.  This function is required
      // to align the stack frame as necessary for any stack variables or
      // called functions.  Because of this, calculateCalleeSavedRegisters
      // must be called before this function in order to set the HasCalls
      // and MaxCallFrameSize variables.
      insertPrologEpilogCode(Fn);

      // Replace all MO_FrameIndex operands with physical register references
      // and actual offsets.
      //
      replaceFrameIndices(Fn);

      return true;
    }
  
  private:
    // MinCSFrameIndex, MaxCSFrameIndex - Keeps the range of callee save
    // stack frame indexes.
    unsigned MinCSFrameIndex, MaxCSFrameIndex;

    void calculateCalleeSavedRegisters(MachineFunction &Fn);
    void saveCalleeSavedRegisters(MachineFunction &Fn);
    void calculateFrameObjectOffsets(MachineFunction &Fn);
    void replaceFrameIndices(MachineFunction &Fn);
    void insertPrologEpilogCode(MachineFunction &Fn);
  };
}


/// createPrologEpilogCodeInserter - This function returns a pass that inserts
/// prolog and epilog code, and eliminates abstract frame references.
///
FunctionPass *llvm::createPrologEpilogCodeInserter() { return new PEI(); }


/// calculateCalleeSavedRegisters - Scan the function for modified callee saved
/// registers.  Also calculate the MaxCallFrameSize and HasCalls variables for
/// the function's frame information and eliminates call frame pseudo
/// instructions.
///
void PEI::calculateCalleeSavedRegisters(MachineFunction &Fn) {
  const MRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  const TargetFrameInfo *TFI = Fn.getTarget().getFrameInfo();

  // Get the callee saved register list...
  const unsigned *CSRegs = RegInfo->getCalleeSaveRegs();

  // Get the function call frame set-up and tear-down instruction opcode
  int FrameSetupOpcode   = RegInfo->getCallFrameSetupOpcode();
  int FrameDestroyOpcode = RegInfo->getCallFrameDestroyOpcode();

  // These are used to keep track the callee-save area. Initialize them.
  MinCSFrameIndex = INT_MAX;
  MaxCSFrameIndex = 0;

  // Early exit for targets which have no callee saved registers and no call
  // frame setup/destroy pseudo instructions.
  if ((CSRegs == 0 || CSRegs[0] == 0) &&
      FrameSetupOpcode == -1 && FrameDestroyOpcode == -1)
    return;

  unsigned MaxCallFrameSize = 0;
  bool HasCalls = false;

  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); )
      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        assert(I->getNumOperands() >= 1 && "Call Frame Setup/Destroy Pseudo"
               " instructions should have a single immediate argument!");
        unsigned Size = I->getOperand(0).getImmedValue();
        if (Size > MaxCallFrameSize) MaxCallFrameSize = Size;
        HasCalls = true;
        RegInfo->eliminateCallFramePseudoInstr(Fn, *BB, I++);
      } else {
        ++I;
      }

  MachineFrameInfo *FFI = Fn.getFrameInfo();
  FFI->setHasCalls(HasCalls);
  FFI->setMaxCallFrameSize(MaxCallFrameSize);

  // Now figure out which *callee saved* registers are modified by the current
  // function, thus needing to be saved and restored in the prolog/epilog.
  //
  const bool *PhysRegsUsed = Fn.getUsedPhysregs();
  const TargetRegisterClass* const *CSRegClasses =
    RegInfo->getCalleeSaveRegClasses();
  std::vector<CalleeSavedInfo> CSI;
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (PhysRegsUsed[Reg]) {
        // If the reg is modified, save it!
      CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
    } else {
      for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
           *AliasSet; ++AliasSet) {  // Check alias registers too.
        if (PhysRegsUsed[*AliasSet]) {
          CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
          break;
        }
      }
    }
  }

  if (CSI.empty())
    return;   // Early exit if no callee saved registers are modified!

  unsigned NumFixedSpillSlots;
  const std::pair<unsigned,int> *FixedSpillSlots =
    TFI->getCalleeSaveSpillSlots(NumFixedSpillSlots);

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    const TargetRegisterClass *RC = CSI[i].getRegClass();

    // Check to see if this physreg must be spilled to a particular stack slot
    // on this target.
    const std::pair<unsigned,int> *FixedSlot = FixedSpillSlots;
    while (FixedSlot != FixedSpillSlots+NumFixedSpillSlots &&
           FixedSlot->first != Reg)
      ++FixedSlot;

    int FrameIdx;
    if (FixedSlot == FixedSpillSlots+NumFixedSpillSlots) {
      // Nope, just spill it anywhere convenient.
      unsigned Align = RC->getAlignment();
      unsigned StackAlign = TFI->getStackAlignment();
      // We may not be able to sastify the desired alignment specification of
      // the TargetRegisterClass if the stack alignment is smaller. Use the min.
      Align = std::min(Align, StackAlign);
      FrameIdx = FFI->CreateStackObject(RC->getSize(), Align);
      if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
      if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;
    } else {
      // Spill it to the stack where we must.
      FrameIdx = FFI->CreateFixedObject(RC->getSize(), FixedSlot->second);
    }
    CSI[i].setFrameIdx(FrameIdx);
  }

  FFI->setCalleeSavedInfo(CSI);
}

/// saveCalleeSavedRegisters -  Insert spill code for any callee saved registers
/// that are modified in the function.
///
void PEI::saveCalleeSavedRegisters(MachineFunction &Fn) {
  // Get callee saved register information.
  MachineFrameInfo *FFI = Fn.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = FFI->getCalleeSavedInfo();
  
  // Early exit if no callee saved registers are modified!
  if (CSI.empty())
    return;

  const MRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();

  // Now that we have a stack slot for each register to be saved, insert spill
  // code into the entry block.
  MachineBasicBlock *MBB = Fn.begin();
  MachineBasicBlock::iterator I = MBB->begin();
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    // Insert the spill to the stack frame.
    RegInfo->storeRegToStackSlot(*MBB, I, CSI[i].getReg(), CSI[i].getFrameIdx(),
                                 CSI[i].getRegClass());
  }

  // Add code to restore the callee-save registers in each exiting block.
  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  for (MachineFunction::iterator FI = Fn.begin(), E = Fn.end(); FI != E; ++FI)
    // If last instruction is a return instruction, add an epilogue.
    if (!FI->empty() && TII.isReturn(FI->back().getOpcode())) {
      MBB = FI;
      I = MBB->end(); --I;

      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = I;
      while (I2 != MBB->begin() && TII.isTerminatorInstr((--I2)->getOpcode()))
        I = I2;

      bool AtStart = I == MBB->begin();
      MachineBasicBlock::iterator BeforeI = I;
      if (!AtStart)
        --BeforeI;
      
      // Restore all registers immediately before the return and any terminators
      // that preceed it.
      for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
        RegInfo->loadRegFromStackSlot(*MBB, I, CSI[i].getReg(),
                                      CSI[i].getFrameIdx(),
                                      CSI[i].getRegClass());
        assert(I != MBB->begin() &&
               "loadRegFromStackSlot didn't insert any code!");
        // Insert in reverse order.  loadRegFromStackSlot can insert multiple
        // instructions.
        if (AtStart)
          I = MBB->begin();
        else {
          I = BeforeI;
          ++I;
        }
      }
    }
}


/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void PEI::calculateFrameObjectOffsets(MachineFunction &Fn) {
  const TargetFrameInfo &TFI = *Fn.getTarget().getFrameInfo();

  bool StackGrowsDown =
    TFI.getStackGrowthDirection() == TargetFrameInfo::StackGrowsDown;

  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo *FFI = Fn.getFrameInfo();

  unsigned StackAlignment = TFI.getStackAlignment();
  unsigned MaxAlign = 0;

  // Start at the beginning of the local area.
  // The Offset is the distance from the stack top in the direction
  // of stack growth -- so it's always positive.
  int Offset = TFI.getOffsetOfLocalArea();
  if (StackGrowsDown)
    Offset = -Offset;
  assert(Offset >= 0
         && "Local area offset should be in direction of stack growth");

  // If there are fixed sized objects that are preallocated in the local area,
  // non-fixed objects can't be allocated right at the start of local area.
  // We currently don't support filling in holes in between fixed sized objects,
  // so we adjust 'Offset' to point to the end of last fixed sized
  // preallocated object.
  for (int i = FFI->getObjectIndexBegin(); i != 0; ++i) {
    int FixedOff;
    if (StackGrowsDown) {
      // The maximum distance from the stack pointer is at lower address of
      // the object -- which is given by offset. For down growing stack
      // the offset is negative, so we negate the offset to get the distance.
      FixedOff = -FFI->getObjectOffset(i);
    } else {
      // The maximum distance from the start pointer is at the upper
      // address of the object.
      FixedOff = FFI->getObjectOffset(i) + FFI->getObjectSize(i);
    }
    if (FixedOff > Offset) Offset = FixedOff;
  }

  // First assign frame offsets to stack objects that are used to spill
  // callee save registers.
  if (StackGrowsDown) {
    for (unsigned i = 0, e = FFI->getObjectIndexEnd(); i != e; ++i) {
      if (i < MinCSFrameIndex || i > MaxCSFrameIndex)
        continue;

      // If stack grows down, we need to add size of find the lowest
      // address of the object.
      Offset += FFI->getObjectSize(i);

      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack, then
      // increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, -Offset);        // Set the computed offset
    }
  } else {
    for (int i = FFI->getObjectIndexEnd()-1; i >= 0; --i) {
      if ((unsigned)i < MinCSFrameIndex || (unsigned)i > MaxCSFrameIndex)
        continue;

      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack, then
      // increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, Offset);
      Offset += FFI->getObjectSize(i);
    }
  }

  // Then assign frame offsets to stack objects that are not used to spill
  // callee save registers.
  for (unsigned i = 0, e = FFI->getObjectIndexEnd(); i != e; ++i) {
    if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
      continue;

    // If stack grows down, we need to add size of find the lowest
    // address of the object.
    if (StackGrowsDown)
      Offset += FFI->getObjectSize(i);

    unsigned Align = FFI->getObjectAlignment(i);
    // If the alignment of this object is greater than that of the stack, then
    // increase the stack alignment to match.
    MaxAlign = std::max(MaxAlign, Align);
    // Adjust to alignment boundary
    Offset = (Offset+Align-1)/Align*Align;

    if (StackGrowsDown) {
      FFI->setObjectOffset(i, -Offset);        // Set the computed offset
    } else {
      FFI->setObjectOffset(i, Offset);
      Offset += FFI->getObjectSize(i);
    }
  }


  // Align the final stack pointer offset, but only if there are calls in the
  // function.  This ensures that any calls to subroutines have their stack
  // frames suitable aligned.
  if (FFI->hasCalls())
    Offset = (Offset+StackAlignment-1)/StackAlignment*StackAlignment;

  // Set the final value of the stack pointer...
  FFI->setStackSize(Offset+TFI.getOffsetOfLocalArea());

  // Remember the required stack alignment in case targets need it to perform
  // dynamic stack alignment.
  assert(FFI->getMaxAlignment() == MaxAlign &&
         "Stack alignment calculation broken!");
}


/// insertPrologEpilogCode - Scan the function for modified callee saved
/// registers, insert spill code for these callee saved registers, then add
/// prolog and epilog code to the function.
///
void PEI::insertPrologEpilogCode(MachineFunction &Fn) {
  // Add prologue to the function...
  Fn.getTarget().getRegisterInfo()->emitPrologue(Fn);

  // Add epilogue to restore the callee-save registers in each exiting block
  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    // If last instruction is a return instruction, add an epilogue
    if (!I->empty() && TII.isReturn(I->back().getOpcode()))
      Fn.getTarget().getRegisterInfo()->emitEpilogue(Fn, *I);
  }
}


/// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
/// register references and actual offsets.
///
void PEI::replaceFrameIndices(MachineFunction &Fn) {
  if (!Fn.getFrameInfo()->hasStackObjects()) return; // Nothing to do?

  const TargetMachine &TM = Fn.getTarget();
  assert(TM.getRegisterInfo() && "TM::getRegisterInfo() must be implemented!");
  const MRegisterInfo &MRI = *TM.getRegisterInfo();

  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (I->getOperand(i).isFrameIndex()) {
          // If this instruction has a FrameIndex operand, we need to use that
          // target machine register info object to eliminate it.
          MRI.eliminateFrameIndex(I);
          break;
        }
}

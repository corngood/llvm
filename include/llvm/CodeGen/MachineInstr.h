//===-- llvm/CodeGen/MachineInstr.h - MachineInstr class --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MachineInstr class, which is the
// basic representation for all target dependent machine instructions used by
// the back end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTR_H
#define LLVM_CODEGEN_MACHINEINSTR_H

#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include <vector>

namespace llvm {

class TargetInstrDesc;
class TargetInstrInfo;
class TargetRegisterInfo;

template <typename T> struct ilist_traits;
template <typename T> struct ilist;

//===----------------------------------------------------------------------===//
/// MachineInstr - Representation of each machine instruction.
///
class MachineInstr {
  const TargetInstrDesc *TID;           // Instruction descriptor.
  unsigned short NumImplicitOps;        // Number of implicit operands (which
                                        // are determined at construction time).

  std::vector<MachineOperand> Operands; // the operands
  std::vector<MachineMemOperand> MemOperands;// information on memory references
  MachineInstr *Prev, *Next;            // Links for MBB's intrusive list.
  MachineBasicBlock *Parent;            // Pointer to the owning basic block.

  // OperandComplete - Return true if it's illegal to add a new operand
  bool OperandsComplete() const;

  MachineInstr(const MachineInstr&);
  void operator=(const MachineInstr&); // DO NOT IMPLEMENT

  // Intrusive list support
  friend struct ilist_traits<MachineInstr>;
  friend struct ilist_traits<MachineBasicBlock>;
  void setParent(MachineBasicBlock *P) { Parent = P; }
public:
  /// MachineInstr ctor - This constructor creates a dummy MachineInstr with
  /// TID NULL and no operands.
  MachineInstr();

  /// MachineInstr ctor - This constructor create a MachineInstr and add the
  /// implicit operands.  It reserves space for number of operands specified by
  /// TargetInstrDesc.
  explicit MachineInstr(const TargetInstrDesc &TID, bool NoImp = false);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.
  ///
  MachineInstr(MachineBasicBlock *MBB, const TargetInstrDesc &TID);

  ~MachineInstr();

  const MachineBasicBlock* getParent() const { return Parent; }
  MachineBasicBlock* getParent() { return Parent; }
  
  /// getDesc - Returns the target instruction descriptor of this
  /// MachineInstr.
  const TargetInstrDesc &getDesc() const { return *TID; }

  /// getOpcode - Returns the opcode of this MachineInstr.
  ///
  int getOpcode() const;

  /// Access to explicit operands of the instruction.
  ///
  unsigned getNumOperands() const { return (unsigned)Operands.size(); }

  const MachineOperand& getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }
  MachineOperand& getOperand(unsigned i) {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }

  /// getNumExplicitOperands - Returns the number of non-implicit operands.
  ///
  unsigned getNumExplicitOperands() const;
  
  /// Access to memory operands of the instruction
  unsigned getNumMemOperands() const { return (unsigned)MemOperands.size(); }

  const MachineMemOperand& getMemOperand(unsigned i) const {
    assert(i < getNumMemOperands() && "getMemOperand() out of range!");
    return MemOperands[i];
  }
  MachineMemOperand& getMemOperand(unsigned i) {
    assert(i < getNumMemOperands() && "getMemOperand() out of range!");
    return MemOperands[i];
  }

  /// isIdenticalTo - Return true if this instruction is identical to (same
  /// opcode and same operands as) the specified instruction.
  bool isIdenticalTo(const MachineInstr *Other) const {
    if (Other->getOpcode() != getOpcode() ||
        Other->getNumOperands() != getNumOperands())
      return false;
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (!getOperand(i).isIdenticalTo(Other->getOperand(i)))
        return false;
    return true;
  }

  /// clone - Create a copy of 'this' instruction that is identical in
  /// all ways except the the instruction has no parent, prev, or next.
  MachineInstr* clone() const { return new MachineInstr(*this); }
  
  /// removeFromParent - This method unlinks 'this' from the containing basic
  /// block, and returns it, but does not delete it.
  MachineInstr *removeFromParent();
  
  /// eraseFromParent - This method unlinks 'this' from the containing basic
  /// block and deletes it.
  void eraseFromParent() {
    delete removeFromParent();
  }

  /// isLabel - Returns true if the MachineInstr represents a label.
  ///
  bool isLabel() const;

  /// isDebugLabel - Returns true if the MachineInstr represents a debug label.
  ///
  bool isDebugLabel() const;

  /// readsRegister - Return true if the MachineInstr reads the specified
  /// register. If TargetRegisterInfo is passed, then it also checks if there
  /// is a read of a super-register.
  bool readsRegister(unsigned Reg, const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterUseOperandIdx(Reg, false, TRI) != -1;
  }

  /// killsRegister - Return true if the MachineInstr kills the specified
  /// register. If TargetRegisterInfo is passed, then it also checks if there is
  /// a kill of a super-register.
  bool killsRegister(unsigned Reg, const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterUseOperandIdx(Reg, true, TRI) != -1;
  }

  /// modifiesRegister - Return true if the MachineInstr modifies the
  /// specified register. If TargetRegisterInfo is passed, then it also checks
  /// if there is a def of a super-register.
  bool modifiesRegister(unsigned Reg,
                        const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterDefOperandIdx(Reg, false, TRI) != -1;
  }

  /// registerDefIsDead - Returns true if the register is dead in this machine
  /// instruction. If TargetRegisterInfo is passed, then it also checks
  /// if there is a dead def of a super-register.
  bool registerDefIsDead(unsigned Reg,
                         const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterDefOperandIdx(Reg, true, TRI) != -1;
  }

  /// findRegisterUseOperandIdx() - Returns the operand index that is a use of
  /// the specific register or -1 if it is not found. It further tightening
  /// the search criteria to a use that kills the register if isKill is true.
  int findRegisterUseOperandIdx(unsigned Reg, bool isKill = false,
                                const TargetRegisterInfo *TRI = NULL) const;

  /// findRegisterUseOperand - Wrapper for findRegisterUseOperandIdx, it returns
  /// a pointer to the MachineOperand rather than an index.
  MachineOperand *findRegisterUseOperand(unsigned Reg, bool isKill = false,
                                         const TargetRegisterInfo *TRI = NULL) {
    int Idx = findRegisterUseOperandIdx(Reg, isKill, TRI);
    return (Idx == -1) ? NULL : &getOperand(Idx);
  }
  
  /// findRegisterDefOperandIdx() - Returns the operand index that is a def of
  /// the specified register or -1 if it is not found. If isDead is true, defs
  /// that are not dead are skipped. If TargetRegisterInfo is non-null, then it
  /// also checks if there is a def of a super-register.
  int findRegisterDefOperandIdx(unsigned Reg, bool isDead = false,
                                const TargetRegisterInfo *TRI = NULL) const;

  /// findRegisterDefOperand - Wrapper for findRegisterDefOperandIdx, it returns
  /// a pointer to the MachineOperand rather than an index.
  MachineOperand *findRegisterDefOperand(unsigned Reg,bool isDead = false,
                                         const TargetRegisterInfo *TRI = NULL) {
    int Idx = findRegisterDefOperandIdx(Reg, isDead, TRI);
    return (Idx == -1) ? NULL : &getOperand(Idx);
  }

  /// findFirstPredOperandIdx() - Find the index of the first operand in the
  /// operand list that is used to represent the predicate. It returns -1 if
  /// none is found.
  int findFirstPredOperandIdx() const;
  
  /// isRegReDefinedByTwoAddr - Returns true if the Reg re-definition is due
  /// to two addr elimination.
  bool isRegReDefinedByTwoAddr(unsigned Reg) const;

  /// copyKillDeadInfo - Copies kill / dead operand properties from MI.
  ///
  void copyKillDeadInfo(const MachineInstr *MI);

  /// copyPredicates - Copies predicate operand(s) from MI.
  void copyPredicates(const MachineInstr *MI);

  /// addRegisterKilled - We have determined MI kills a register. Look for the
  /// operand that uses it and mark it as IsKill. If AddIfNotFound is true,
  /// add a implicit operand if it's not found. Returns true if the operand
  /// exists / is added.
  bool addRegisterKilled(unsigned IncomingReg,
                         const TargetRegisterInfo *RegInfo,
                         bool AddIfNotFound = false);
  
  /// addRegisterDead - We have determined MI defined a register without a use.
  /// Look for the operand that defines it and mark it as IsDead. If
  /// AddIfNotFound is true, add a implicit operand if it's not found. Returns
  /// true if the operand exists / is added.
  bool addRegisterDead(unsigned IncomingReg, const TargetRegisterInfo *RegInfo,
                       bool AddIfNotFound = false);

  /// copyKillDeadInfo - Copies killed/dead information from one instr to another
  void copyKillDeadInfo(MachineInstr *OldMI,
                        const TargetRegisterInfo *RegInfo);

  /// isSafeToMove - Return true if it is safe to this instruction. If SawStore
  /// true, it means there is a store (or call) between the instruction the
  /// localtion and its intended destination.
  bool isSafeToMove(const TargetInstrInfo *TII, bool &SawStore);

  //
  // Debugging support
  //
  void print(std::ostream *OS, const TargetMachine *TM) const {
    if (OS) print(*OS, TM);
  }
  void print(std::ostream &OS, const TargetMachine *TM = 0) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Accessors used to build up machine instructions.

  /// addOperand - Add the specified operand to the instruction.  If it is an
  /// implicit operand, it is added to the end of the operand list.  If it is
  /// an explicit operand it is added at the end of the explicit operand list
  /// (before the first implicit operand). 
  void addOperand(const MachineOperand &Op);
  
  /// setDesc - Replace the instruction descriptor (thus opcode) of
  /// the current instruction with a new one.
  ///
  void setDesc(const TargetInstrDesc &tid) { TID = &tid; }

  /// RemoveOperand - Erase an operand  from an instruction, leaving it with one
  /// fewer operand than it started with.
  ///
  void RemoveOperand(unsigned i);

  /// addMemOperand - Add a MachineMemOperand to the machine instruction,
  /// referencing arbitrary storage.
  void addMemOperand(const MachineMemOperand &MO) {
    MemOperands.push_back(MO);
  }

private:
  /// getRegInfo - If this instruction is embedded into a MachineFunction,
  /// return the MachineRegisterInfo object for the current function, otherwise
  /// return null.
  MachineRegisterInfo *getRegInfo();

  /// addImplicitDefUseOperands - Add all implicit def and use operands to
  /// this instruction.
  void addImplicitDefUseOperands();
  
  /// RemoveRegOperandsFromUseLists - Unlink all of the register operands in
  /// this instruction from their respective use lists.  This requires that the
  /// operands already be on their use lists.
  void RemoveRegOperandsFromUseLists();
  
  /// AddRegOperandsToUseLists - Add all of the register operands in
  /// this instruction from their respective use lists.  This requires that the
  /// operands not be on their use lists yet.
  void AddRegOperandsToUseLists(MachineRegisterInfo &RegInfo);
};

//===----------------------------------------------------------------------===//
// Debugging Support

inline std::ostream& operator<<(std::ostream &OS, const MachineInstr &MI) {
  MI.print(OS);
  return OS;
}

} // End llvm namespace

#endif

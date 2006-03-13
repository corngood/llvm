//===- X86ISelDAGToDAG.cpp - A DAG pattern matching inst selector for X86 -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DAG pattern matching instruction selector for X86,
// converting from a legalized dag to a X86 dag.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86ISelLowering.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CFG.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                      Pattern Matcher Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// X86ISelAddressMode - This corresponds to X86AddressMode, but uses
  /// SDOperand's instead of register numbers for the leaves of the matched
  /// tree.
  struct X86ISelAddressMode {
    enum {
      RegBase,
      FrameIndexBase,
    } BaseType;

    struct {            // This is really a union, discriminated by BaseType!
      SDOperand Reg;
      int FrameIndex;
    } Base;

    unsigned Scale;
    SDOperand IndexReg; 
    unsigned Disp;
    GlobalValue *GV;
    Constant *CP;
    unsigned Align;    // CP alignment.

    X86ISelAddressMode()
      : BaseType(RegBase), Scale(1), IndexReg(), Disp(0), GV(0),
        CP(0), Align(0) {
    }
  };
}

namespace {
  Statistic<>
  NumFPKill("x86-codegen", "Number of FP_REG_KILL instructions added");

  //===--------------------------------------------------------------------===//
  /// ISel - X86 specific code to select X86 machine instructions for
  /// SelectionDAG operations.
  ///
  class X86DAGToDAGISel : public SelectionDAGISel {
    /// ContainsFPCode - Every instruction we select that uses or defines a FP
    /// register should set this to true.
    bool ContainsFPCode;

    /// X86Lowering - This object fully describes how to lower LLVM code to an
    /// X86-specific SelectionDAG.
    X86TargetLowering X86Lowering;

    /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
    /// make the right decision when generating code for different targets.
    const X86Subtarget *Subtarget;

    unsigned GlobalBaseReg;
  public:
    X86DAGToDAGISel(X86TargetMachine &TM)
      : SelectionDAGISel(X86Lowering),
        X86Lowering(*TM.getTargetLowering()) {
      Subtarget = &TM.getSubtarget<X86Subtarget>();
    }

    virtual bool runOnFunction(Function &Fn) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseReg = 0;
      return SelectionDAGISel::runOnFunction(Fn);
    }
   
    virtual const char *getPassName() const {
      return "X86 DAG->DAG Instruction Selection";
    }

    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);

    virtual void EmitFunctionEntryCode(Function &Fn, MachineFunction &MF);

// Include the pieces autogenerated from the target description.
#include "X86GenDAGISel.inc"

  private:
    void Select(SDOperand &Result, SDOperand N);

    bool MatchAddress(SDOperand N, X86ISelAddressMode &AM, bool isRoot = true);
    bool SelectAddr(SDOperand N, SDOperand &Base, SDOperand &Scale,
                    SDOperand &Index, SDOperand &Disp);
    bool SelectLEAAddr(SDOperand N, SDOperand &Base, SDOperand &Scale,
                       SDOperand &Index, SDOperand &Disp);
    bool TryFoldLoad(SDOperand P, SDOperand N,
                     SDOperand &Base, SDOperand &Scale,
                     SDOperand &Index, SDOperand &Disp);

    inline void getAddressOperands(X86ISelAddressMode &AM, SDOperand &Base, 
                                   SDOperand &Scale, SDOperand &Index,
                                   SDOperand &Disp) {
      Base  = (AM.BaseType == X86ISelAddressMode::FrameIndexBase) ?
        CurDAG->getTargetFrameIndex(AM.Base.FrameIndex, MVT::i32) : AM.Base.Reg;
      Scale = getI8Imm(AM.Scale);
      Index = AM.IndexReg;
      Disp  = AM.GV ? CurDAG->getTargetGlobalAddress(AM.GV, MVT::i32, AM.Disp)
        : (AM.CP ?
           CurDAG->getTargetConstantPool(AM.CP, MVT::i32, AM.Align, AM.Disp)
           : getI32Imm(AM.Disp));
    }

    /// getI8Imm - Return a target constant with the specified value, of type
    /// i8.
    inline SDOperand getI8Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i8);
    }

    /// getI16Imm - Return a target constant with the specified value, of type
    /// i16.
    inline SDOperand getI16Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i16);
    }

    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDOperand getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }

    /// getGlobalBaseReg - insert code into the entry mbb to materialize the PIC
    /// base register.  Return the virtual register that holds this value.
    SDOperand getGlobalBaseReg();

#ifndef NDEBUG
    unsigned Indent;
#endif
  };
}

/// InstructionSelectBasicBlock - This callback is invoked by SelectionDAGISel
/// when it has created a SelectionDAG for us to codegen.
void X86DAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());
  MachineFunction::iterator FirstMBB = BB;

  // Codegen the basic block.
#ifndef NDEBUG
  DEBUG(std::cerr << "===== Instruction selection begins:\n");
  Indent = 0;
#endif
  DAG.setRoot(SelectRoot(DAG.getRoot()));
#ifndef NDEBUG
  DEBUG(std::cerr << "===== Instruction selection ends:\n");
#endif
  CodeGenMap.clear();
  DAG.RemoveDeadNodes();

  // Emit machine code to BB. 
  ScheduleAndEmitDAG(DAG);
  
  // If we are emitting FP stack code, scan the basic block to determine if this
  // block defines any FP values.  If so, put an FP_REG_KILL instruction before
  // the terminator of the block.
  if (!Subtarget->hasSSE2()) {
    // Note that FP stack instructions *are* used in SSE code when returning
    // values, but these are not live out of the basic block, so we don't need
    // an FP_REG_KILL in this case either.
    bool ContainsFPCode = false;
    
    // Scan all of the machine instructions in these MBBs, checking for FP
    // stores.
    MachineFunction::iterator MBBI = FirstMBB;
    do {
      for (MachineBasicBlock::iterator I = MBBI->begin(), E = MBBI->end();
           !ContainsFPCode && I != E; ++I) {
        for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
          if (I->getOperand(op).isRegister() && I->getOperand(op).isDef() &&
              MRegisterInfo::isVirtualRegister(I->getOperand(op).getReg()) &&
              RegMap->getRegClass(I->getOperand(0).getReg()) == 
                X86::RFPRegisterClass) {
            ContainsFPCode = true;
            break;
          }
        }
      }
    } while (!ContainsFPCode && &*(MBBI++) != BB);
    
    // Check PHI nodes in successor blocks.  These PHI's will be lowered to have
    // a copy of the input value in this block.
    if (!ContainsFPCode) {
      // Final check, check LLVM BB's that are successors to the LLVM BB
      // corresponding to BB for FP PHI nodes.
      const BasicBlock *LLVMBB = BB->getBasicBlock();
      const PHINode *PN;
      for (succ_const_iterator SI = succ_begin(LLVMBB), E = succ_end(LLVMBB);
           !ContainsFPCode && SI != E; ++SI) {
        for (BasicBlock::const_iterator II = SI->begin();
             (PN = dyn_cast<PHINode>(II)); ++II) {
          if (PN->getType()->isFloatingPoint()) {
            ContainsFPCode = true;
            break;
          }
        }
      }
    }

    // Finally, if we found any FP code, emit the FP_REG_KILL instruction.
    if (ContainsFPCode) {
      BuildMI(*BB, BB->getFirstTerminator(), X86::FP_REG_KILL, 0);
      ++NumFPKill;
    }
  }
}

/// EmitSpecialCodeForMain - Emit any code that needs to be executed only in
/// the main function.
static void EmitSpecialCodeForMain(MachineBasicBlock *BB,
                                   MachineFrameInfo *MFI) {
  // Switch the FPU to 64-bit precision mode for better compatibility and speed.
  int CWFrameIdx = MFI->CreateStackObject(2, 2);
  addFrameReference(BuildMI(BB, X86::FNSTCW16m, 4), CWFrameIdx);

  // Set the high part to be 64-bit precision.
  addFrameReference(BuildMI(BB, X86::MOV8mi, 5),
                    CWFrameIdx, 1).addImm(2);

  // Reload the modified control word now.
  addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);
}

void X86DAGToDAGISel::EmitFunctionEntryCode(Function &Fn, MachineFunction &MF) {
  // If this is main, emit special code for main.
  MachineBasicBlock *BB = MF.begin();
  if (Fn.hasExternalLinkage() && Fn.getName() == "main")
    EmitSpecialCodeForMain(BB, MF.getFrameInfo());
}

/// MatchAddress - Add the specified node to the specified addressing mode,
/// returning true if it cannot be done.  This just pattern matches for the
/// addressing mode
bool X86DAGToDAGISel::MatchAddress(SDOperand N, X86ISelAddressMode &AM,
                                   bool isRoot) {
  bool Available = false;
  // If N has already been selected, reuse the result unless in some very
  // specific cases.
  std::map<SDOperand, SDOperand>::iterator CGMI= CodeGenMap.find(N.getValue(0));
  if (CGMI != CodeGenMap.end()) {
    Available = true;
  }

  switch (N.getOpcode()) {
  default: break;
  case ISD::Constant:
    AM.Disp += cast<ConstantSDNode>(N)->getValue();
    return false;

  case X86ISD::Wrapper:
    // If both base and index components have been picked, we can't fit
    // the result available in the register in the addressing mode. Duplicate
    // GlobalAddress or ConstantPool as displacement.
    if (!Available || (AM.Base.Reg.Val && AM.IndexReg.Val)) {
      if (ConstantPoolSDNode *CP =
          dyn_cast<ConstantPoolSDNode>(N.getOperand(0))) {
        if (AM.CP == 0) {
          AM.CP = CP->get();
          AM.Align = CP->getAlignment();
          AM.Disp += CP->getOffset();
          return false;
        }
      } else if (GlobalAddressSDNode *G =
                 dyn_cast<GlobalAddressSDNode>(N.getOperand(0))) {
        if (AM.GV == 0) {
          AM.GV = G->getGlobal();
          AM.Disp += G->getOffset();
          return false;
        }
      }
    }
    break;

  case ISD::FrameIndex:
    if (AM.BaseType == X86ISelAddressMode::RegBase && AM.Base.Reg.Val == 0) {
      AM.BaseType = X86ISelAddressMode::FrameIndexBase;
      AM.Base.FrameIndex = cast<FrameIndexSDNode>(N)->getIndex();
      return false;
    }
    break;

  case ISD::SHL:
    if (!Available && AM.IndexReg.Val == 0 && AM.Scale == 1)
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.Val->getOperand(1))) {
        unsigned Val = CN->getValue();
        if (Val == 1 || Val == 2 || Val == 3) {
          AM.Scale = 1 << Val;
          SDOperand ShVal = N.Val->getOperand(0);

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (ShVal.Val->getOpcode() == ISD::ADD && ShVal.hasOneUse() &&
              isa<ConstantSDNode>(ShVal.Val->getOperand(1))) {
            AM.IndexReg = ShVal.Val->getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(ShVal.Val->getOperand(1));
            AM.Disp += AddVal->getValue() << Val;
          } else {
            AM.IndexReg = ShVal;
          }
          return false;
        }
      }
    break;

  case ISD::MUL:
    // X*[3,5,9] -> X+X*[2,4,8]
    if (!Available &&
        AM.BaseType == X86ISelAddressMode::RegBase &&
        AM.Base.Reg.Val == 0 &&
        AM.IndexReg.Val == 0)
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.Val->getOperand(1)))
        if (CN->getValue() == 3 || CN->getValue() == 5 || CN->getValue() == 9) {
          AM.Scale = unsigned(CN->getValue())-1;

          SDOperand MulVal = N.Val->getOperand(0);
          SDOperand Reg;

          // Okay, we know that we have a scale by now.  However, if the scaled
          // value is an add of something and a constant, we can fold the
          // constant into the disp field here.
          if (MulVal.Val->getOpcode() == ISD::ADD && MulVal.hasOneUse() &&
              isa<ConstantSDNode>(MulVal.Val->getOperand(1))) {
            Reg = MulVal.Val->getOperand(0);
            ConstantSDNode *AddVal =
              cast<ConstantSDNode>(MulVal.Val->getOperand(1));
            AM.Disp += AddVal->getValue() * CN->getValue();
          } else {
            Reg = N.Val->getOperand(0);
          }

          AM.IndexReg = AM.Base.Reg = Reg;
          return false;
        }
    break;

  case ISD::ADD: {
    if (!Available) {
      X86ISelAddressMode Backup = AM;
      if (!MatchAddress(N.Val->getOperand(0), AM, false) &&
          !MatchAddress(N.Val->getOperand(1), AM, false))
        return false;
      AM = Backup;
      if (!MatchAddress(N.Val->getOperand(1), AM, false) &&
          !MatchAddress(N.Val->getOperand(0), AM, false))
        return false;
      AM = Backup;
    }
    break;
  }
  }

  // Is the base register already occupied?
  if (AM.BaseType != X86ISelAddressMode::RegBase || AM.Base.Reg.Val) {
    // If so, check to see if the scale index register is set.
    if (AM.IndexReg.Val == 0) {
      AM.IndexReg = N;
      AM.Scale = 1;
      return false;
    }

    // Otherwise, we cannot select it.
    return true;
  }

  // Default, generate it as a register.
  AM.BaseType = X86ISelAddressMode::RegBase;
  AM.Base.Reg = N;
  return false;
}

/// SelectAddr - returns true if it is able pattern match an addressing mode.
/// It returns the operands which make up the maximal addressing mode it can
/// match by reference.
bool X86DAGToDAGISel::SelectAddr(SDOperand N, SDOperand &Base, SDOperand &Scale,
                                 SDOperand &Index, SDOperand &Disp) {
  X86ISelAddressMode AM;
  if (MatchAddress(N, AM))
    return false;

  if (AM.BaseType == X86ISelAddressMode::RegBase) {
    if (!AM.Base.Reg.Val)
      AM.Base.Reg = CurDAG->getRegister(0, MVT::i32);
  }

  if (!AM.IndexReg.Val)
    AM.IndexReg = CurDAG->getRegister(0, MVT::i32);

  getAddressOperands(AM, Base, Scale, Index, Disp);

  return true;
}

/// SelectLEAAddr - it calls SelectAddr and determines if the maximal addressing
/// mode it matches can be cost effectively emitted as an LEA instruction.
/// For X86, it always is unless it's just a (Reg + const).
bool X86DAGToDAGISel::SelectLEAAddr(SDOperand N, SDOperand &Base,
                                    SDOperand &Scale,
                                    SDOperand &Index, SDOperand &Disp) {
  X86ISelAddressMode AM;
  if (MatchAddress(N, AM))
    return false;

  unsigned Complexity = 0;
  if (AM.BaseType == X86ISelAddressMode::RegBase)
    if (AM.Base.Reg.Val)
      Complexity = 1;
    else
      AM.Base.Reg = CurDAG->getRegister(0, MVT::i32);
  else if (AM.BaseType == X86ISelAddressMode::FrameIndexBase)
    Complexity = 4;

  if (AM.IndexReg.Val)
    Complexity++;
  else
    AM.IndexReg = CurDAG->getRegister(0, MVT::i32);

  if (AM.Scale > 2) 
    Complexity += 2;
  // Don't match just leal(,%reg,2). It's cheaper to do addl %reg, %reg
  else if (AM.Scale > 1)
    Complexity++;

  // FIXME: We are artificially lowering the criteria to turn ADD %reg, $GA
  // to a LEA. This is determined with some expermentation but is by no means
  // optimal (especially for code size consideration). LEA is nice because of
  // its three-address nature. Tweak the cost function again when we can run
  // convertToThreeAddress() at register allocation time.
  if (AM.GV || AM.CP)
    Complexity += 2;

  if (AM.Disp && (AM.Base.Reg.Val || AM.IndexReg.Val))
    Complexity++;

  if (Complexity > 2) {
    getAddressOperands(AM, Base, Scale, Index, Disp);
    return true;
  }

  return false;
}

bool X86DAGToDAGISel::TryFoldLoad(SDOperand P, SDOperand N,
                                  SDOperand &Base, SDOperand &Scale,
                                  SDOperand &Index, SDOperand &Disp) {
  if (N.getOpcode() == ISD::LOAD &&
      N.hasOneUse() &&
      !CodeGenMap.count(N.getValue(0)) &&
      (P.getNumOperands() == 1 || !isNonImmUse(P.Val, N.Val)))
    return SelectAddr(N.getOperand(1), Base, Scale, Index, Disp);
  return false;
}

static bool isRegister0(SDOperand Op) {
  if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(Op))
    return (R->getReg() == 0);
  return false;
}

/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
SDOperand X86DAGToDAGISel::getGlobalBaseReg() {
  if (!GlobalBaseReg) {
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = BB->getParent()->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    SSARegMap *RegMap = BB->getParent()->getSSARegMap();
    // FIXME: when we get to LP64, we will need to create the appropriate
    // type of register here.
    GlobalBaseReg = RegMap->createVirtualRegister(X86::R32RegisterClass);
    BuildMI(FirstMBB, MBBI, X86::MovePCtoStack, 0);
    BuildMI(FirstMBB, MBBI, X86::POP32r, 1, GlobalBaseReg);
  }
  return CurDAG->getRegister(GlobalBaseReg, MVT::i32);
}

void X86DAGToDAGISel::Select(SDOperand &Result, SDOperand N) {
  SDNode *Node = N.Val;
  MVT::ValueType NVT = Node->getValueType(0);
  unsigned Opc, MOpc;
  unsigned Opcode = Node->getOpcode();

#ifndef NDEBUG
  DEBUG(std::cerr << std::string(Indent, ' '));
  DEBUG(std::cerr << "Selecting: ");
  DEBUG(Node->dump(CurDAG));
  DEBUG(std::cerr << "\n");
  Indent += 2;
#endif

  if (Opcode >= ISD::BUILTIN_OP_END && Opcode < X86ISD::FIRST_NUMBER) {
    Result = N;
#ifndef NDEBUG
    DEBUG(std::cerr << std::string(Indent-2, ' '));
    DEBUG(std::cerr << "== ");
    DEBUG(Node->dump(CurDAG));
    DEBUG(std::cerr << "\n");
    Indent -= 2;
#endif
    return;   // Already selected.
  }

  std::map<SDOperand, SDOperand>::iterator CGMI = CodeGenMap.find(N);
  if (CGMI != CodeGenMap.end()) {
    Result = CGMI->second;
#ifndef NDEBUG
    DEBUG(std::cerr << std::string(Indent-2, ' '));
    DEBUG(std::cerr << "== ");
    DEBUG(Result.Val->dump(CurDAG));
    DEBUG(std::cerr << "\n");
    Indent -= 2;
#endif
    return;
  }
  
  switch (Opcode) {
    default: break;
    case X86ISD::GlobalBaseReg: 
      Result = getGlobalBaseReg();
      return;

    case ISD::ADD: {
      // Turn ADD X, c to MOV32ri X+c. This cannot be done with tblgen'd
      // code and is matched first so to prevent it from being turned into
      // LEA32r X+c.
      SDOperand N0 = N.getOperand(0);
      SDOperand N1 = N.getOperand(1);
      if (N.Val->getValueType(0) == MVT::i32 &&
          N0.getOpcode() == X86ISD::Wrapper &&
          N1.getOpcode() == ISD::Constant) {
        unsigned Offset = (unsigned)cast<ConstantSDNode>(N1)->getValue();
        SDOperand C(0, 0);
        // TODO: handle ExternalSymbolSDNode.
        if (GlobalAddressSDNode *G =
            dyn_cast<GlobalAddressSDNode>(N0.getOperand(0))) {
          C = CurDAG->getTargetGlobalAddress(G->getGlobal(), MVT::i32,
                                             G->getOffset() + Offset);
        } else if (ConstantPoolSDNode *CP =
                   dyn_cast<ConstantPoolSDNode>(N0.getOperand(0))) {
          C = CurDAG->getTargetConstantPool(CP->get(), MVT::i32,
                                            CP->getAlignment(),
                                            CP->getOffset()+Offset);
        }

        if (C.Val) {
          if (N.Val->hasOneUse()) {
            Result = CurDAG->SelectNodeTo(N.Val, X86::MOV32ri, MVT::i32, C);
          } else {
            SDNode *ResNode = CurDAG->getTargetNode(X86::MOV32ri, MVT::i32, C);
            Result = CodeGenMap[N] = SDOperand(ResNode, 0);
          }
          return;
        }
      }

      // Other cases are handled by auto-generated code.
      break;
    }

    case ISD::MULHU:
    case ISD::MULHS: {
      if (Opcode == ISD::MULHU)
        switch (NVT) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::MUL8r;  MOpc = X86::MUL8m;  break;
        case MVT::i16: Opc = X86::MUL16r; MOpc = X86::MUL16m; break;
        case MVT::i32: Opc = X86::MUL32r; MOpc = X86::MUL32m; break;
        }
      else
        switch (NVT) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::IMUL8r;  MOpc = X86::IMUL8m;  break;
        case MVT::i16: Opc = X86::IMUL16r; MOpc = X86::IMUL16m; break;
        case MVT::i32: Opc = X86::IMUL32r; MOpc = X86::IMUL32m; break;
        }

      unsigned LoReg, HiReg;
      switch (NVT) {
      default: assert(0 && "Unsupported VT!");
      case MVT::i8:  LoReg = X86::AL;  HiReg = X86::AH;  break;
      case MVT::i16: LoReg = X86::AX;  HiReg = X86::DX;  break;
      case MVT::i32: LoReg = X86::EAX; HiReg = X86::EDX; break;
      }

      SDOperand N0 = Node->getOperand(0);
      SDOperand N1 = Node->getOperand(1);

      bool foldedLoad = false;
      SDOperand Tmp0, Tmp1, Tmp2, Tmp3;
      foldedLoad = TryFoldLoad(N, N1, Tmp0, Tmp1, Tmp2, Tmp3);
      // MULHU and MULHS are commmutative
      if (!foldedLoad) {
        foldedLoad = TryFoldLoad(N, N0, Tmp0, Tmp1, Tmp2, Tmp3);
        if (foldedLoad) {
          N0 = Node->getOperand(1);
          N1 = Node->getOperand(0);
        }
      }

      SDOperand Chain;
      if (foldedLoad)
        Select(Chain, N1.getOperand(0));
      else
        Chain = CurDAG->getEntryNode();

      SDOperand InFlag(0, 0);
      Select(N0, N0);
      Chain  = CurDAG->getCopyToReg(Chain, CurDAG->getRegister(LoReg, NVT),
                                    N0, InFlag);
      InFlag = Chain.getValue(1);

      if (foldedLoad) {
        Select(Tmp0, Tmp0);
        Select(Tmp1, Tmp1);
        Select(Tmp2, Tmp2);
        Select(Tmp3, Tmp3);
        SDNode *CNode =
          CurDAG->getTargetNode(MOpc, MVT::Other, MVT::Flag, Tmp0, Tmp1,
                                Tmp2, Tmp3, Chain, InFlag);
        Chain  = SDOperand(CNode, 0);
        InFlag = SDOperand(CNode, 1);
      } else {
        Select(N1, N1);
        InFlag =
          SDOperand(CurDAG->getTargetNode(Opc, MVT::Flag, N1, InFlag), 0);
      }

      Result = CurDAG->getCopyFromReg(Chain, HiReg, NVT, InFlag);
      CodeGenMap[N.getValue(0)] = Result;
      if (foldedLoad) {
        CodeGenMap[N1.getValue(1)] = Result.getValue(1);
        AddHandleReplacement(N1.Val, 1, Result.Val, 1);
      }

#ifndef NDEBUG
      DEBUG(std::cerr << std::string(Indent-2, ' '));
      DEBUG(std::cerr << "== ");
      DEBUG(Result.Val->dump(CurDAG));
      DEBUG(std::cerr << "\n");
      Indent -= 2;
#endif
      return;
    }
      
    case ISD::SDIV:
    case ISD::UDIV:
    case ISD::SREM:
    case ISD::UREM: {
      bool isSigned = Opcode == ISD::SDIV || Opcode == ISD::SREM;
      bool isDiv    = Opcode == ISD::SDIV || Opcode == ISD::UDIV;
      if (!isSigned)
        switch (NVT) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::DIV8r;  MOpc = X86::DIV8m;  break;
        case MVT::i16: Opc = X86::DIV16r; MOpc = X86::DIV16m; break;
        case MVT::i32: Opc = X86::DIV32r; MOpc = X86::DIV32m; break;
        }
      else
        switch (NVT) {
        default: assert(0 && "Unsupported VT!");
        case MVT::i8:  Opc = X86::IDIV8r;  MOpc = X86::IDIV8m;  break;
        case MVT::i16: Opc = X86::IDIV16r; MOpc = X86::IDIV16m; break;
        case MVT::i32: Opc = X86::IDIV32r; MOpc = X86::IDIV32m; break;
        }

      unsigned LoReg, HiReg;
      unsigned ClrOpcode, SExtOpcode;
      switch (NVT) {
      default: assert(0 && "Unsupported VT!");
      case MVT::i8:
        LoReg = X86::AL;  HiReg = X86::AH;
        ClrOpcode  = X86::MOV8ri;
        SExtOpcode = X86::CBW;
        break;
      case MVT::i16:
        LoReg = X86::AX;  HiReg = X86::DX;
        ClrOpcode  = X86::MOV16ri;
        SExtOpcode = X86::CWD;
        break;
      case MVT::i32:
        LoReg = X86::EAX; HiReg = X86::EDX;
        ClrOpcode  = X86::MOV32ri;
        SExtOpcode = X86::CDQ;
        break;
      }

      SDOperand N0 = Node->getOperand(0);
      SDOperand N1 = Node->getOperand(1);

      bool foldedLoad = false;
      SDOperand Tmp0, Tmp1, Tmp2, Tmp3;
      foldedLoad = TryFoldLoad(N, N1, Tmp0, Tmp1, Tmp2, Tmp3);
      SDOperand Chain;
      if (foldedLoad)
        Select(Chain, N1.getOperand(0));
      else
        Chain = CurDAG->getEntryNode();

      SDOperand InFlag(0, 0);
      Select(N0, N0);
      Chain  = CurDAG->getCopyToReg(Chain, CurDAG->getRegister(LoReg, NVT),
                                    N0, InFlag);
      InFlag = Chain.getValue(1);

      if (isSigned) {
        // Sign extend the low part into the high part.
        InFlag =
          SDOperand(CurDAG->getTargetNode(SExtOpcode, MVT::Flag, InFlag), 0);
      } else {
        // Zero out the high part, effectively zero extending the input.
        SDOperand ClrNode =
          SDOperand(CurDAG->getTargetNode(ClrOpcode, NVT,
                                         CurDAG->getTargetConstant(0, NVT)), 0);
        Chain  = CurDAG->getCopyToReg(Chain, CurDAG->getRegister(HiReg, NVT),
                                      ClrNode, InFlag);
        InFlag = Chain.getValue(1);
      }

      if (foldedLoad) {
        Select(Tmp0, Tmp0);
        Select(Tmp1, Tmp1);
        Select(Tmp2, Tmp2);
        Select(Tmp3, Tmp3);
        SDNode *CNode =
          CurDAG->getTargetNode(MOpc, MVT::Other, MVT::Flag, Tmp0, Tmp1,
                                Tmp2, Tmp3, Chain, InFlag);
        Chain  = SDOperand(CNode, 0);
        InFlag = SDOperand(CNode, 1);
      } else {
        Select(N1, N1);
        InFlag =
          SDOperand(CurDAG->getTargetNode(Opc, MVT::Flag, N1, InFlag), 0);
      }

      Result = CurDAG->getCopyFromReg(Chain, isDiv ? LoReg : HiReg,
                                      NVT, InFlag);
      CodeGenMap[N.getValue(0)] = Result;
      if (foldedLoad) {
        CodeGenMap[N1.getValue(1)] = Result.getValue(1);
        AddHandleReplacement(N1.Val, 1, Result.Val, 1);
      }

#ifndef NDEBUG
      DEBUG(std::cerr << std::string(Indent-2, ' '));
      DEBUG(std::cerr << "== ");
      DEBUG(Result.Val->dump(CurDAG));
      DEBUG(std::cerr << "\n");
      Indent -= 2;
#endif
      return;
    }

    case ISD::TRUNCATE: {
      unsigned Reg;
      MVT::ValueType VT;
      switch (Node->getOperand(0).getValueType()) {
        default: assert(0 && "Unknown truncate!");
        case MVT::i16: Reg = X86::AX;  Opc = X86::MOV16rr; VT = MVT::i16; break;
        case MVT::i32: Reg = X86::EAX; Opc = X86::MOV32rr; VT = MVT::i32; break;
      }
      SDOperand Tmp0, Tmp1;
      Select(Tmp0, Node->getOperand(0));
      Select(Tmp1, SDOperand(CurDAG->getTargetNode(Opc, VT, Tmp0), 0));
      SDOperand InFlag = SDOperand(0,0);
      Result = CurDAG->getCopyToReg(CurDAG->getEntryNode(), Reg, Tmp1, InFlag);
      SDOperand Chain = Result.getValue(0);
      InFlag = Result.getValue(1);

      switch (NVT) {
        default: assert(0 && "Unknown truncate!");
        case MVT::i8:  Reg = X86::AL;  Opc = X86::MOV8rr;  VT = MVT::i8;  break;
        case MVT::i16: Reg = X86::AX;  Opc = X86::MOV16rr; VT = MVT::i16; break;
      }

      Result = CurDAG->getCopyFromReg(Chain, Reg, VT, InFlag);
      if (N.Val->hasOneUse())
        Result = CurDAG->SelectNodeTo(N.Val, Opc, VT, Result);
      else
        Result = CodeGenMap[N] =
          SDOperand(CurDAG->getTargetNode(Opc, VT, Result), 0);

#ifndef NDEBUG
      DEBUG(std::cerr << std::string(Indent-2, ' '));
      DEBUG(std::cerr << "== ");
      DEBUG(Result.Val->dump(CurDAG));
      DEBUG(std::cerr << "\n");
      Indent -= 2;
#endif
      return;
    }
  }

  SelectCode(Result, N);
#ifndef NDEBUG
  DEBUG(std::cerr << std::string(Indent-2, ' '));
  DEBUG(std::cerr << "=> ");
  DEBUG(Result.Val->dump(CurDAG));
  DEBUG(std::cerr << "\n");
  Indent -= 2;
#endif
}

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createX86ISelDag(X86TargetMachine &TM) {
  return new X86DAGToDAGISel(TM);
}

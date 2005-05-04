//===-- AlphaAsmPrinter.cpp - Alpha LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format Alpha assembly language.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaInstrInfo.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/AsmPrinter.h"

#include "llvm/Target/TargetMachine.h"

#include "llvm/Support/Mangler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace llvm {
  extern cl::opt<bool> EnableAlphaFTOI;
  extern cl::opt<bool> EnableAlphaCT;
}

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

  struct AlphaAsmPrinter : public AsmPrinter {

    /// Unique incrementer for label values for referencing Global values.
    ///
    unsigned LabelNumber;

     AlphaAsmPrinter(std::ostream &o, TargetMachine &tm)
       : AsmPrinter(o, tm), LabelNumber(0)
    {
      AlignmentIsInBytes = false;
    }

    /// We name each basic block in a Function with a unique number, so
    /// that we can consistently refer to them later. This is cleared
    /// at the beginning of each call to runOnMachineFunction().
    ///
    typedef std::map<const Value *, unsigned> ValueMapTy;
    ValueMapTy NumberForBB;
    std::string CurSection;

    virtual const char *getPassName() const {
      return "Alpha Assembly Printer";
    }
    bool printInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool IsCallOp = false);
    void printConstantPool(MachineConstantPool *MCP);
    void printOperand(const MachineInstr *MI, int opNum, MVT::ValueType VT);
    void printBaseOffsetPair (const MachineInstr *MI, int i, bool brackets=true);
    void printMachineInstruction(const MachineInstr *MI);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
    void SwitchSection(std::ostream &OS, const char *NewSection);
  };
} // end of anonymous namespace

/// createAlphaCodePrinterPass - Returns a pass that prints the Alpha
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *llvm::createAlphaCodePrinterPass (std::ostream &o,
                                                  TargetMachine &tm) {
  return new AlphaAsmPrinter(o, tm);
}

#include "AlphaGenAsmWriter.inc"

void AlphaAsmPrinter::printOperand(const MachineInstr *MI, int opNum, MVT::ValueType VT)
{
  const MachineOperand &MO = MI->getOperand(opNum);
  if (MO.getType() == MachineOperand::MO_MachineRegister) {
    assert(MRegisterInfo::isPhysicalRegister(MO.getReg())&&"Not physreg??");
    O << TM.getRegisterInfo()->get(MO.getReg()).Name;
  } else if (MO.isImmediate()) {
    O << MO.getImmedValue();
  } else {
    printOp(MO);
  }
}


void AlphaAsmPrinter::printOp(const MachineOperand &MO, bool IsCallOp) {
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  int new_symbol;

  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      return;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
  case MachineOperand::MO_CCRegister:
    O << RI.get(MO.getReg()).Name;
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    std::cerr << "printOp() does not handle immediate values\n";
    abort();
    return;

  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building Alpha MachineInstrs";
    abort();
    return;

  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << "LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber() << "\t" << CommentString << " "
      << MBBOp->getBasicBlock()->getName();
    return;
  }

  case MachineOperand::MO_ConstantPoolIndex:
    O << "CPI" << CurrentFnName << "_" << MO.getConstantPoolIndex();
    return;

  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    return;

  case MachineOperand::MO_GlobalAddress:
    //Abuse PCrel to specify pcrel calls
    //calls are the only thing that use this flag
    if (MO.isPCRelative())
      O << "$" << Mang->getValueName(MO.getGlobal()) << "..ng";
    else
      O << Mang->getValueName(MO.getGlobal());
    return;

  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

/// printMachineInstruction -- Print out a single Alpha MI to
/// the current output stream.
///
void AlphaAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;
  if (printInstruction(MI))
    return; // Printer was automatically generated

  assert(0 && "Unhandled instruction in asm writer!");
  abort();
  return;
}


/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool AlphaAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  setupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  SwitchSection(O, "text");
  emitAlignment(4);
  O << "\t.globl\t" << CurrentFnName << "\n";
  O << "\t.ent\t" << CurrentFnName << "\n";

  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << "LBB" << CurrentFnName << "_" << I->getNumber() << ":\t"
      << CommentString << " " << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }
  ++LabelNumber;

  O << "\t.end " << CurrentFnName << "\n";

  // We didn't modify anything.
  return false;
}


/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void AlphaAsmPrinter::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();

  if (CP.empty()) return;

  SwitchSection(O, "section .rodata");
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    //    SwitchSection(O, "section .rodata, \"dr\"");
    emitAlignment(TD.getTypeAlignmentShift(CP[i]->getType()));
    O << "CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t" << CommentString
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

bool AlphaAsmPrinter::doInitialization(Module &M)
{
  AsmPrinter::doInitialization(M);
  if(EnableAlphaFTOI || EnableAlphaCT)
    O << "\t.arch ev6\n";
  else
    O << "\t.arch ev56\n";
  return false;
}


// SwitchSection - Switch to the specified section of the executable if we are
// not already in it!
//
void AlphaAsmPrinter::SwitchSection(std::ostream &OS, const char *NewSection)
{
  if (CurSection != NewSection) {
    CurSection = NewSection;
    if (!CurSection.empty())
      OS << "\t." << NewSection << "\n";
  }
}

bool AlphaAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();

  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
    if (I->hasInitializer()) {   // External global require no code
      O << "\n\n";
      std::string name = Mang->getValueName(I);
      Constant *C = I->getInitializer();
      unsigned Size = TD.getTypeSize(C->getType());
      unsigned Align = TD.getTypeAlignmentShift(C->getType());

      if (C->isNullValue() &&
          (I->hasLinkOnceLinkage() || I->hasInternalLinkage() ||
           I->hasWeakLinkage() /* FIXME: Verify correct */)) {
        SwitchSection(O, "data");
        if (I->hasInternalLinkage())
          O << "\t.local " << name << "\n";

        O << "\t.comm " << name << "," << TD.getTypeSize(C->getType())
          << "," << (1 << Align);
        O << "\t\t# ";
        WriteAsOperand(O, I, true, true, &M);
        O << "\n";
      } else {
        switch (I->getLinkage()) {
        case GlobalValue::LinkOnceLinkage:
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          SwitchSection(O, "");
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n";
          break;
        case GlobalValue::AppendingLinkage:
          // FIXME: appending linkage variables should go into a section of
          // their name or something.  For now, just emit them as external.
        case GlobalValue::ExternalLinkage:
          // If external or appending, declare as a global symbol
          O << "\t.globl " << name << "\n";
          // FALL THROUGH
        case GlobalValue::InternalLinkage:
          if (C->isNullValue())
            SwitchSection(O, "bss"); //was .bss
          else
            SwitchSection(O, "data");
          break;
        case GlobalValue::GhostLinkage:
          std::cerr << "GhostLinkage cannot appear in AlphaAsmPrinter!\n";
          abort();
        }

        emitAlignment(Align);
        O << "\t.type " << name << ",@object\n";
        O << "\t.size " << name << "," << Size << "\n";
        O << name << ":\t\t\t\t# ";
        WriteAsOperand(O, I, true, true, &M);
        O << " = ";
        WriteAsOperand(O, C, false, false, &M);
        O << "\n";
        emitGlobalConstant(C);
      }
    }

  AsmPrinter::doFinalization(M);
  return false;
}

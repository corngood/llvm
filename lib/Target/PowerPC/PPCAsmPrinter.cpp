//===-- PowerPCAsmPrinter.cpp - Print machine instrs to PowerPC assembly --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PowerPC assembly language. This printer is
// the output mechanism used by `llc'.
//
// Documentation at http://developer.apple.com/documentation/DeveloperTools/
// Reference/Assembler/ASMIntroduction/chapter_1_section_1.html
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asmprinter"
#include "PowerPC.h"
#include "PowerPCTargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

  struct PowerPCAsmPrinter : public AsmPrinter {
    std::set<std::string> FnStubs, GVStubs, LinkOnceStubs;

    PowerPCAsmPrinter(std::ostream &O, TargetMachine &TM)
      : AsmPrinter(O, TM), LabelNumber(0) {}

    /// Unique incrementer for label values for referencing Global values.
    ///
    unsigned LabelNumber;

    virtual const char *getPassName() const {
      return "PowerPC Assembly Printer";
    }

    PowerPCTargetMachine &getTM() {
      return static_cast<PowerPCTargetMachine&>(TM);
    }

    unsigned enumRegToMachineReg(unsigned enumReg) {
      switch (enumReg) {
      default: assert(0 && "Unhandled register!"); break;
      case PPC::CR0:  return  0;
      case PPC::CR1:  return  1;
      case PPC::CR2:  return  2;
      case PPC::CR3:  return  3;
      case PPC::CR4:  return  4;
      case PPC::CR5:  return  5;
      case PPC::CR6:  return  6;
      case PPC::CR7:  return  7;
      }
      abort();
    }

    /// printInstruction - This method is automatically generated by tablegen
    /// from the instruction set description.  This method returns true if the
    /// machine instruction was sufficiently described to print it, otherwise it
    /// returns false.
    bool printInstruction(const MachineInstr *MI);

    void printMachineInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO, bool IsCallOp = false);

    void printOperand(const MachineInstr *MI, unsigned OpNo, MVT::ValueType VT){
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (MO.getType() == MachineOperand::MO_MachineRegister) {
        assert(MRegisterInfo::isPhysicalRegister(MO.getReg())&&"Not physreg??");
        O << LowercaseString(TM.getRegisterInfo()->get(MO.getReg()).Name);
      } else if (MO.isImmediate()) {
        O << MO.getImmedValue();
      } else {
        printOp(MO);
      }
    }

    void printU5ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      unsigned char value = MI->getOperand(OpNo).getImmedValue();
      assert(value <= 31 && "Invalid u5imm argument!");
      O << (unsigned int)value;
    }
    void printU6ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      unsigned char value = MI->getOperand(OpNo).getImmedValue();
      assert(value <= 63 && "Invalid u6imm argument!");
      O << (unsigned int)value;
    }
    void printS16ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      O << (short)MI->getOperand(OpNo).getImmedValue();
    }
    void printU16ImmOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      O << (unsigned short)MI->getOperand(OpNo).getImmedValue();
    }
    void printBranchOperand(const MachineInstr *MI, unsigned OpNo,
                            MVT::ValueType VT) {
      // Branches can take an immediate operand.  This is used by the branch
      // selection pass to print $+8, an eight byte displacement from the PC.
      if (MI->getOperand(OpNo).isImmediate()) {
        O << "$+" << MI->getOperand(OpNo).getImmedValue();
      } else {
        printOp(MI->getOperand(OpNo),
                TM.getInstrInfo()->isCall(MI->getOpcode()));
      }
    }
    void printPICLabel(const MachineInstr *MI, unsigned OpNo,
                       MVT::ValueType VT) {
      // FIXME: should probably be converted to cout.width and cout.fill
      O << "\"L0000" << LabelNumber << "$pb\"\n";
      O << "\"L0000" << LabelNumber << "$pb\":";
    }
    void printSymbolHi(const MachineInstr *MI, unsigned OpNo,
                       MVT::ValueType VT) {
      O << "ha16(";
      printOp(MI->getOperand(OpNo));
      O << "-\"L0000" << LabelNumber << "$pb\")";
    }
    void printSymbolLo(const MachineInstr *MI, unsigned OpNo,
                       MVT::ValueType VT) {
      // FIXME: Because LFS, LFD, and LWZ can be used either with a s16imm or
      // a lo16 of a global or constant pool operand, we must handle both here.
      // this isn't a great design, but it works for now.
      if (MI->getOperand(OpNo).isImmediate()) {
        O << (short)MI->getOperand(OpNo).getImmedValue();
      } else {
        O << "lo16(";
        printOp(MI->getOperand(OpNo));
        O << "-\"L0000" << LabelNumber << "$pb\")";
      }
    }
    void printcrbit(const MachineInstr *MI, unsigned OpNo,
                       MVT::ValueType VT) {
      unsigned char value = MI->getOperand(OpNo).getImmedValue();
      assert(value <= 3 && "Invalid crbit argument!");
      unsigned CCReg = MI->getOperand(OpNo-1).getReg();
      unsigned RegNo = enumRegToMachineReg(CCReg);
      O << 4 * RegNo + value;
    }
    void printcrbitm(const MachineInstr *MI, unsigned OpNo,
                       MVT::ValueType VT) {
      unsigned CCReg = MI->getOperand(OpNo).getReg();
      unsigned RegNo = enumRegToMachineReg(CCReg);
      O << (0x80 >> RegNo);
    }

    virtual void printConstantPool(MachineConstantPool *MCP) = 0;
    virtual bool runOnMachineFunction(MachineFunction &F) = 0;
    virtual bool doFinalization(Module &M) = 0;
  };

  /// DarwinAsmPrinter - PowerPC assembly printer, customized for Darwin/Mac OS
  /// X
  ///
  struct DarwinAsmPrinter : public PowerPCAsmPrinter {

    DarwinAsmPrinter(std::ostream &O, TargetMachine &TM)
      : PowerPCAsmPrinter(O, TM) {
      CommentString = ";";
      GlobalPrefix = "_";
      ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
      Data64bitsDirective = 0;       // we can't emit a 64-bit unit
      AlignmentIsInBytes = false;    // Alignment is by power of 2.
    }

    virtual const char *getPassName() const {
      return "Darwin PPC Assembly Printer";
    }

    void printConstantPool(MachineConstantPool *MCP);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
  };

  /// AIXAsmPrinter - PowerPC assembly printer, customized for AIX
  ///
  struct AIXAsmPrinter : public PowerPCAsmPrinter {
    /// Map for labels corresponding to global variables
    ///
    std::map<const GlobalVariable*,std::string> GVToLabelMap;

    AIXAsmPrinter(std::ostream &O, TargetMachine &TM)
      : PowerPCAsmPrinter(O, TM) {
      CommentString = "#";
      GlobalPrefix = "_";
      ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
      Data64bitsDirective = 0;       // we can't emit a 64-bit unit
      AlignmentIsInBytes = false;    // Alignment is by power of 2.
    }

    virtual const char *getPassName() const {
      return "AIX PPC Assembly Printer";
    }

    void printConstantPool(MachineConstantPool *MCP);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
  };
} // end of anonymous namespace

// SwitchSection - Switch to the specified section of the executable if we are
// not already in it!
//
static void SwitchSection(std::ostream &OS, std::string &CurSection,
                          const char *NewSection) {
  if (CurSection != NewSection) {
    CurSection = NewSection;
    if (!CurSection.empty())
      OS << "\t" << NewSection << "\n";
  }
}

/// createDarwinAsmPrinterPass - Returns a pass that prints the PPC assembly
/// code for a MachineFunction to the given output stream, in a format that the
/// Darwin assembler can deal with.
///
FunctionPass *llvm::createDarwinAsmPrinter(std::ostream &o, TargetMachine &tm) {
  return new DarwinAsmPrinter(o, tm);
}

/// createAIXAsmPrinterPass - Returns a pass that prints the PPC assembly code
/// for a MachineFunction to the given output stream, in a format that the
/// AIX 5L assembler can deal with.
///
FunctionPass *llvm::createAIXAsmPrinter(std::ostream &o, TargetMachine &tm) {
  return new AIXAsmPrinter(o, tm);
}

// Include the auto-generated portion of the assembly writer
#include "PowerPCGenAsmWriter.inc"

void PowerPCAsmPrinter::printOp(const MachineOperand &MO, bool IsCallOp) {
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
    O << LowercaseString(RI.get(MO.getReg()).Name);
    return;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    std::cerr << "printOp() does not handle immediate values\n";
    abort();
    return;

  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building PPC MachineInstrs";
    abort();
    return;

  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber() << "\t; "
      << MBBOp->getBasicBlock()->getName();
    return;
  }

  case MachineOperand::MO_ConstantPoolIndex:
    O << ".CPI" << CurrentFnName << "_" << MO.getConstantPoolIndex();
    return;

  case MachineOperand::MO_ExternalSymbol:
    if (IsCallOp) {
      std::string Name(GlobalPrefix); Name += MO.getSymbolName();
      FnStubs.insert(Name);
      O << "L" << Name << "$stub";
      return;
    }
    O << GlobalPrefix << MO.getSymbolName();
    return;

  case MachineOperand::MO_GlobalAddress: {
    GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);

    // Dynamically-resolved functions need a stub for the function.  Be
    // wary however not to output $stub for external functions whose addresses
    // are taken.  Those should be emitted as $non_lazy_ptr below.
    Function *F = dyn_cast<Function>(GV);
    if (F && IsCallOp && F->isExternal()) {
      FnStubs.insert(Name);
      O << "L" << Name << "$stub";
      return;
    }

    // External or weakly linked global variables need non-lazily-resolved stubs
    if ((GV->isExternal() || GV->hasWeakLinkage() || GV->hasLinkOnceLinkage())){
      if (GV->hasLinkOnceLinkage())
        LinkOnceStubs.insert(Name);
      else
        GVStubs.insert(Name);
      O << "L" << Name << "$non_lazy_ptr";
      return;
    }

    O << Mang->getValueName(GV);
    return;
  }

  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

/// printMachineInstruction -- Print out a single PowerPC MI in Darwin syntax to
/// the current output stream.
///
void PowerPCAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;
  // Check for slwi/srwi mnemonics.
  if (MI->getOpcode() == PPC::RLWINM) {
    bool FoundMnemonic = false;
    unsigned char SH = MI->getOperand(2).getImmedValue();
    unsigned char MB = MI->getOperand(3).getImmedValue();
    unsigned char ME = MI->getOperand(4).getImmedValue();
    if (SH <= 31 && MB == 0 && ME == (31-SH)) {
      O << "slwi "; FoundMnemonic = true;
    }
    if (SH <= 31 && MB == (32-SH) && ME == 31) {
      O << "srwi "; FoundMnemonic = true;
      SH = 32-SH;
    }
    if (FoundMnemonic) {
      printOperand(MI, 0, MVT::i64);
      O << ", ";
      printOperand(MI, 1, MVT::i64);
      O << ", " << (unsigned int)SH << "\n";
      return;
    }
  }

  if (printInstruction(MI))
    return; // Printer was automatically generated

  assert(0 && "Unhandled instruction in asm writer!");
  abort();
  return;
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool DarwinAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  setupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\t.text\n";
  emitAlignment(2);
  O << "\t.globl\t" << CurrentFnName << "\n";
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << ".LBB" << CurrentFnName << "_" << I->getNumber() << ":\t"
      << CommentString << " " << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }
  ++LabelNumber;

  // We didn't modify anything.
  return false;
}

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void DarwinAsmPrinter::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();

  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.const\n";
    emitAlignment(TD.getTypeAlignmentShift(CP[i]->getType()));
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t" << CommentString
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

bool DarwinAsmPrinter::doInitialization(Module &M) {
  // FIXME: implment subtargets for PowerPC and pick this up from there.
  O << "\t.machine ppc970\n";

  AsmPrinter::doInitialization(M);
  return false;
}

bool DarwinAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
    if (I->hasInitializer()) {   // External global require no code
      O << '\n';
      std::string name = Mang->getValueName(I);
      Constant *C = I->getInitializer();
      unsigned Size = TD.getTypeSize(C->getType());
      unsigned Align = TD.getTypeAlignmentShift(C->getType());

      if (C->isNullValue() && /* FIXME: Verify correct */
          (I->hasInternalLinkage() || I->hasWeakLinkage() ||
           I->hasLinkOnceLinkage())) {
        SwitchSection(O, CurSection, ".data");
        if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
        if (I->hasInternalLinkage())
          O << ".lcomm " << name << "," << Size << "," << Align;
        else
          O << ".comm " << name << "," << Size;
        O << "\t\t; ";
        WriteAsOperand(O, I, true, true, &M);
        O << '\n';
      } else {
        switch (I->getLinkage()) {
        case GlobalValue::LinkOnceLinkage:
          O << ".section __TEXT,__textcoal_nt,coalesced,no_toc\n"
            << ".weak_definition " << name << '\n'
            << ".private_extern " << name << '\n'
            << ".section __DATA,__datacoal_nt,coalesced,no_toc\n";
          LinkOnceStubs.insert(name);
          break;
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          SwitchSection(O, CurSection, "");
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
          SwitchSection(O, CurSection, ".data");
          break;
        case GlobalValue::GhostLinkage:
          std::cerr << "Error: unmaterialized (GhostLinkage) function in asm!";
          abort();
        }

        emitAlignment(Align);
        O << name << ":\t\t\t\t; ";
        WriteAsOperand(O, I, true, true, &M);
        O << " = ";
        WriteAsOperand(O, C, false, false, &M);
        O << "\n";
        emitGlobalConstant(C);
      }
    }

  // Output stubs for dynamically-linked functions
  for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
       i != e; ++i)
  {
    O << ".data\n";
    O << ".section __TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32\n";
    emitAlignment(2);
    O << "L" << *i << "$stub:\n";
    O << "\t.indirect_symbol " << *i << "\n";
    O << "\tmflr r0\n";
    O << "\tbcl 20,31,L0$" << *i << "\n";
    O << "L0$" << *i << ":\n";
    O << "\tmflr r11\n";
    O << "\taddis r11,r11,ha16(L" << *i << "$lazy_ptr-L0$" << *i << ")\n";
    O << "\tmtlr r0\n";
    O << "\tlwzu r12,lo16(L" << *i << "$lazy_ptr-L0$" << *i << ")(r11)\n";
    O << "\tmtctr r12\n";
    O << "\tbctr\n";
    O << ".data\n";
    O << ".lazy_symbol_pointer\n";
    O << "L" << *i << "$lazy_ptr:\n";
    O << "\t.indirect_symbol " << *i << "\n";
    O << "\t.long dyld_stub_binding_helper\n";
  }

  O << "\n";

  // Output stubs for external global variables
  if (GVStubs.begin() != GVStubs.end())
    O << ".data\n.non_lazy_symbol_pointer\n";
  for (std::set<std::string>::iterator i = GVStubs.begin(), e = GVStubs.end();
       i != e; ++i) {
    O << "L" << *i << "$non_lazy_ptr:\n";
    O << "\t.indirect_symbol " << *i << "\n";
    O << "\t.long\t0\n";
  }

  // Output stubs for link-once variables
  if (LinkOnceStubs.begin() != LinkOnceStubs.end())
    O << ".data\n.align 2\n";
  for (std::set<std::string>::iterator i = LinkOnceStubs.begin(),
         e = LinkOnceStubs.end(); i != e; ++i) {
    O << "L" << *i << "$non_lazy_ptr:\n"
      << "\t.long\t" << *i << '\n';
  }

  AsmPrinter::doFinalization(M);
  return false; // success
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool AIXAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  CurrentFnName = MF.getFunction()->getName();

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out header for the function.
  O << "\t.csect .text[PR]\n"
    << "\t.align 2\n"
    << "\t.globl "  << CurrentFnName << '\n'
    << "\t.globl ." << CurrentFnName << '\n'
    << "\t.csect "  << CurrentFnName << "[DS],3\n"
    << CurrentFnName << ":\n"
    << "\t.llong ." << CurrentFnName << ", TOC[tc0], 0\n"
    << "\t.csect .text[PR]\n"
    << '.' << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << "LBB" << CurrentFnName << "_" << I->getNumber() << ":\t# "
      << I->getBasicBlock()->getName() << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
      II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }
  ++LabelNumber;

  O << "LT.." << CurrentFnName << ":\n"
    << "\t.long 0\n"
    << "\t.byte 0,0,32,65,128,0,0,0\n"
    << "\t.long LT.." << CurrentFnName << "-." << CurrentFnName << '\n'
    << "\t.short 3\n"
    << "\t.byte \"" << CurrentFnName << "\"\n"
    << "\t.align 2\n";

  // We didn't modify anything.
  return false;
}

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void AIXAsmPrinter::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();

  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.const\n";
    O << "\t.align " << (unsigned)TD.getTypeAlignment(CP[i]->getType())
      << "\n";
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t;"
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

bool AIXAsmPrinter::doInitialization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  O << "\t.machine \"ppc64\"\n"
    << "\t.toc\n"
    << "\t.csect .text[PR]\n";

  // Print out module-level global variables
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I) {
    if (!I->hasInitializer())
      continue;

    std::string Name = I->getName();
    Constant *C = I->getInitializer();
    // N.B.: We are defaulting to writable strings
    if (I->hasExternalLinkage()) {
      O << "\t.globl " << Name << '\n'
        << "\t.csect .data[RW],3\n";
    } else {
      O << "\t.csect _global.rw_c[RW],3\n";
    }
    O << Name << ":\n";
    emitGlobalConstant(C);
  }

  // Output labels for globals
  if (M.global_begin() != M.global_end()) O << "\t.toc\n";
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I) {
    const GlobalVariable *GV = I;
    // Do not output labels for unused variables
    if (GV->isExternal() && GV->use_begin() == GV->use_end())
      continue;

    std::string Name = GV->getName();
    std::string Label = "LC.." + utostr(LabelNumber++);
    GVToLabelMap[GV] = Label;
    O << Label << ":\n"
      << "\t.tc " << Name << "[TC]," << Name;
    if (GV->isExternal()) O << "[RW]";
    O << '\n';
  }

  Mang = new Mangler(M, ".");
  return false; // success
}

bool AIXAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  // Print out module-level global variables
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I) {
    if (I->hasInitializer() || I->hasExternalLinkage())
      continue;

    std::string Name = I->getName();
    if (I->hasInternalLinkage()) {
      O << "\t.lcomm " << Name << ",16,_global.bss_c";
    } else {
      O << "\t.comm " << Name << "," << TD.getTypeSize(I->getType())
        << "," << log2((unsigned)TD.getTypeAlignment(I->getType()));
    }
    O << "\t\t# ";
    WriteAsOperand(O, I, true, true, &M);
    O << "\n";
  }

  O << "_section_.text:\n"
    << "\t.csect .data[RW],3\n"
    << "\t.llong _section_.text\n";

  delete Mang;
  return false; // success
}

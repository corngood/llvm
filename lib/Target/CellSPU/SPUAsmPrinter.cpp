//===-- SPUAsmPrinter.cpp - Print machine instrs to Cell SPU assembly -------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to Cell SPU assembly language. This printer
// is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asmprinter"
#include "SPU.h"
#include "SPUTargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include <set>
using namespace llvm;

namespace {
  STATISTIC(EmittedInsts, "Number of machine instrs printed");

  const std::string bss_section(".bss");

  struct VISIBILITY_HIDDEN SPUAsmPrinter : public AsmPrinter {
    std::set<std::string> FnStubs, GVStubs;

    SPUAsmPrinter(std::ostream &O, TargetMachine &TM, const TargetAsmInfo *T) :
      AsmPrinter(O, TM, T)
    {
    }

    virtual const char *getPassName() const {
      return "STI CBEA SPU Assembly Printer";
    }

    SPUTargetMachine &getTM() {
      return static_cast<SPUTargetMachine&>(TM);
    }

    /// printInstruction - This method is automatically generated by tablegen
    /// from the instruction set description.  This method returns true if the
    /// machine instruction was sufficiently described to print it, otherwise it
    /// returns false.
    bool printInstruction(const MachineInstr *MI);

    void printMachineInstruction(const MachineInstr *MI);
    void printOp(const MachineOperand &MO);

    /// printRegister - Print register according to target requirements.
    ///
    void printRegister(const MachineOperand &MO, bool R0AsZero) {
      unsigned RegNo = MO.getReg();
      assert(MRegisterInfo::isPhysicalRegister(RegNo) && "Not physreg??");
      O << TM.getRegisterInfo()->get(RegNo).Name;
    }

    void printOperand(const MachineInstr *MI, unsigned OpNo) {
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (MO.isRegister()) {
        assert(MRegisterInfo::isPhysicalRegister(MO.getReg())&&"Not physreg??");
        O << TM.getRegisterInfo()->get(MO.getReg()).Name;
      } else if (MO.isImmediate()) {
        O << MO.getImm();
      } else {
        printOp(MO);
      }
    }
    
    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode);
    bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode);
   
   
    void
    printS7ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      int value = MI->getOperand(OpNo).getImm();
      value = (value << (32 - 7)) >> (32 - 7);

      assert((value >= -(1 << 8) && value <= (1 << 7) - 1)
	     && "Invalid s7 argument");
      O << value;
    }

    void
    printU7ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      unsigned int value = MI->getOperand(OpNo).getImm();
      assert(value < (1 << 8) && "Invalid u7 argument");
      O << value;
    }
 
    void
    printMemRegImmS7(const MachineInstr *MI, unsigned OpNo)
    {
      char value = MI->getOperand(OpNo).getImm();
      O << (int) value;
      O << "(";
      printOperand(MI, OpNo+1);
      O << ")";
    }

    void
    printS16ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      O << (short) MI->getOperand(OpNo).getImm();
    }

    void
    printU16ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      O << (unsigned short)MI->getOperand(OpNo).getImm();
    }

    void
    printU32ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      O << (unsigned)MI->getOperand(OpNo).getImm();
    }
    
    void
    printMemRegReg(const MachineInstr *MI, unsigned OpNo) {
      // When used as the base register, r0 reads constant zero rather than
      // the value contained in the register.  For this reason, the darwin
      // assembler requires that we print r0 as 0 (no r) when used as the base.
      const MachineOperand &MO = MI->getOperand(OpNo);
      O << TM.getRegisterInfo()->get(MO.getReg()).Name;
      O << ", ";
      printOperand(MI, OpNo+1);
    }

    void
    printU18ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      unsigned int value = MI->getOperand(OpNo).getImm();
      assert(value <= (1 << 19) - 1 && "Invalid u18 argument");
      O << value;
    }

    void
    printS10ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      short value = (short) (((int) MI->getOperand(OpNo).getImm() << 16)
                             >> 16);
      assert((value >= -(1 << 9) && value <= (1 << 9) - 1)
             && "Invalid s10 argument");
      O << value;
    }

    void
    printU10ImmOperand(const MachineInstr *MI, unsigned OpNo)
    {
      short value = (short) (((int) MI->getOperand(OpNo).getImm() << 16)
                             >> 16);
      assert((value <= (1 << 10) - 1) && "Invalid u10 argument");
      O << value;
    }

    void
    printMemRegImmS10(const MachineInstr *MI, unsigned OpNo)
    {
      const MachineOperand &MO = MI->getOperand(OpNo);
      assert(MO.isImmediate()
	     && "printMemRegImmS10 first operand is not immedate");
      printS10ImmOperand(MI, OpNo);
      O << "(";
      printOperand(MI, OpNo+1);
      O << ")";
    }

    void
    printAddr256K(const MachineInstr *MI, unsigned OpNo)
    {
      /* Note: operand 1 is an offset or symbol name. */
      if (MI->getOperand(OpNo).isImmediate()) {
        printS16ImmOperand(MI, OpNo);
      } else {
        printOp(MI->getOperand(OpNo));
        if (MI->getOperand(OpNo+1).isImmediate()) {
          int displ = int(MI->getOperand(OpNo+1).getImm());
          if (displ > 0)
            O << "+" << displ;
          else if (displ < 0)
            O << displ;
        }
      }
    }

    void printCallOperand(const MachineInstr *MI, unsigned OpNo) {
      printOp(MI->getOperand(OpNo));
    }

    void printPCRelativeOperand(const MachineInstr *MI, unsigned OpNo) {
      printOp(MI->getOperand(OpNo));
      O << "-.";
    }

    void printSymbolHi(const MachineInstr *MI, unsigned OpNo) {
      if (MI->getOperand(OpNo).isImmediate()) {
        printS16ImmOperand(MI, OpNo);
      } else {
        printOp(MI->getOperand(OpNo));
        O << "@h";
      }
    }

    void printSymbolLo(const MachineInstr *MI, unsigned OpNo) {
      if (MI->getOperand(OpNo).isImmediate()) {
        printS16ImmOperand(MI, OpNo);
      } else {
        printOp(MI->getOperand(OpNo));
        O << "@l";
      }
    }

    /// Print local store address
    void printSymbolLSA(const MachineInstr *MI, unsigned OpNo) {
      printOp(MI->getOperand(OpNo));
    }

    void printROTHNeg7Imm(const MachineInstr *MI, unsigned OpNo) {
      if (MI->getOperand(OpNo).isImmediate()) {
        int value = (int) MI->getOperand(OpNo).getImm();
        assert((value >= 0 && value < 16)
	       && "Invalid negated immediate rotate 7-bit argument");
        O << -value;
      } else {
        assert(0 &&"Invalid/non-immediate rotate amount in printRotateNeg7Imm");
      }
    }

    void printROTNeg7Imm(const MachineInstr *MI, unsigned OpNo) {
      if (MI->getOperand(OpNo).isImmediate()) {
        int value = (int) MI->getOperand(OpNo).getImm();
        assert((value >= 0 && value < 32)
	       && "Invalid negated immediate rotate 7-bit argument");
        O << -value;
      } else {
        assert(0 &&"Invalid/non-immediate rotate amount in printRotateNeg7Imm");
      }
    }

    virtual bool runOnMachineFunction(MachineFunction &F) = 0;
    virtual bool doFinalization(Module &M) = 0;
  };

  /// LinuxAsmPrinter - SPU assembly printer, customized for Linux
  struct VISIBILITY_HIDDEN LinuxAsmPrinter : public SPUAsmPrinter {
  
    DwarfWriter DW;

    LinuxAsmPrinter(std::ostream &O, SPUTargetMachine &TM,
                    const TargetAsmInfo *T) :
      SPUAsmPrinter(O, TM, T),
      DW(O, this, T)
    { }

    virtual const char *getPassName() const {
      return "STI CBEA SPU Assembly Printer";
    }
    
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
    
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<MachineModuleInfo>();
      SPUAsmPrinter::getAnalysisUsage(AU);
    }

    /// getSectionForFunction - Return the section that we should emit the
    /// specified function body into.
    virtual std::string getSectionForFunction(const Function &F) const;
  };
} // end of anonymous namespace

// Include the auto-generated portion of the assembly writer
#include "SPUGenAsmWriter.inc"

void SPUAsmPrinter::printOp(const MachineOperand &MO) {
  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    cerr << "printOp() does not handle immediate values\n";
    abort();
    return;

  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB());
    return;
  case MachineOperand::MO_JumpTableIndex:
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << '_' << MO.getIndex();
    return;
  case MachineOperand::MO_ConstantPoolIndex:
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber()
      << '_' << MO.getIndex();
    return;
  case MachineOperand::MO_ExternalSymbol:
    // Computing the address of an external symbol, not calling it.
    if (TM.getRelocationModel() != Reloc::Static) {
      std::string Name(TAI->getGlobalPrefix()); Name += MO.getSymbolName();
      GVStubs.insert(Name);
      O << "L" << Name << "$non_lazy_ptr";
      return;
    }
    O << TAI->getGlobalPrefix() << MO.getSymbolName();
    return;
  case MachineOperand::MO_GlobalAddress: {
    // Computing the address of a global symbol, not calling it.
    GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);

    // External or weakly linked global variables need non-lazily-resolved
    // stubs
    if (TM.getRelocationModel() != Reloc::Static) {
      if (((GV->isDeclaration() || GV->hasWeakLinkage() ||
            GV->hasLinkOnceLinkage()))) {
        GVStubs.insert(Name);
        O << "L" << Name << "$non_lazy_ptr";
        return;
      }
    }
    O << Name;
    
    if (GV->hasExternalWeakLinkage())
      ExtWeakSymbols.insert(GV);
    return;
  }

  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool SPUAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    unsigned AsmVariant, 
                                    const char *ExtraCode) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.
    
    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'L': // Write second word of DImode reference.  
      // Verify that this operand has two consecutive registers.
      if (!MI->getOperand(OpNo).isRegister() ||
          OpNo+1 == MI->getNumOperands() ||
          !MI->getOperand(OpNo+1).isRegister())
        return true;
      ++OpNo;   // Return the high-part.
      break;
    }
  }
  
  printOperand(MI, OpNo);
  return false;
}

bool SPUAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
    				          unsigned OpNo,
                                          unsigned AsmVariant, 
                                          const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.
  printMemRegReg(MI, OpNo);
  return false;
}

/// printMachineInstruction -- Print out a single PowerPC MI in Darwin syntax
/// to the current output stream.
///
void SPUAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;
  printInstruction(MI);
}



std::string LinuxAsmPrinter::getSectionForFunction(const Function &F) const {
  switch (F.getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::ExternalLinkage:
  case Function::InternalLinkage: return TAI->getTextSection();
  case Function::WeakLinkage:
  case Function::LinkOnceLinkage:
    return ""; // Print nothing for the time being...
  }
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool
LinuxAsmPrinter::runOnMachineFunction(MachineFunction &MF)
{
  DW.SetModuleInfo(&getAnalysis<MachineModuleInfo>());

  SetupMachineFunction(MF);
  O << "\n\n";
  
  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  const Function *F = MF.getFunction();

  SwitchToTextSection(getSectionForFunction(*F).c_str(), F);
  EmitAlignment(3, F);

  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
    break;
  case Function::ExternalLinkage:
    O << "\t.global\t" << CurrentFnName << "\n"
      << "\t.type\t" << CurrentFnName << ", @function\n";
    break;
  case Function::WeakLinkage:
  case Function::LinkOnceLinkage:
    O << "\t.global\t" << CurrentFnName << "\n";
    O << "\t.weak_definition\t" << CurrentFnName << "\n";
    break;
  }
  O << CurrentFnName << ":\n";

  // Emit pre-function debug information.
  DW.BeginFunction(&MF);

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I != MF.begin()) {
      printBasicBlockLabel(I, true);
      O << '\n';
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  O << "\t.size\t" << CurrentFnName << ",.-" << CurrentFnName << "\n";

  // Print out jump tables referenced by the function.
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);
  
  // Emit post-function debug information.
  DW.EndFunction();
  
  // We didn't modify anything.
  return false;
}


bool LinuxAsmPrinter::doInitialization(Module &M) {
  bool Result = AsmPrinter::doInitialization(M);
  SwitchToTextSection(TAI->getTextSection());
  // Emit initial debug information.
  DW.BeginModule(&M);
  return Result;
}

bool LinuxAsmPrinter::doFinalization(Module &M) {
  const TargetData *TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer()) continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I))
      continue;
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD->getTypeStoreSize(C->getType());
    unsigned Align = TD->getPreferredAlignmentLog(I);

    if (C->isNullValue() && /* FIXME: Verify correct */
        (I->hasInternalLinkage() || I->hasWeakLinkage() ||
         I->hasLinkOnceLinkage() ||
         (I->hasExternalLinkage() && !I->hasSection()))) {
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
      if (I->hasExternalLinkage()) {
        // External linkage globals -> .bss section
        // FIXME: Want to set the global variable's section so that
        // SwitchToDataSection emits the ".section" directive
        SwitchToDataSection("\t.section\t.bss", I);
        O << "\t.global\t" << name << '\n';
        O << "\t.align\t" << Align << '\n';
        O << "\t.type\t" << name << ", @object\n";
        O << "\t.size\t" << name << ", " << Size << '\n';
        O << name << ":\n";
        O << "\t.zero\t" << Size;
      } else if (I->hasInternalLinkage()) {
        SwitchToDataSection("\t.data", I);
        O << ".local " << name << "\n";
        O << TAI->getCOMMDirective() << name << "," << Size << "," << Align << "\n";
      } else {
        SwitchToDataSection("\t.data", I);
        O << ".comm " << name << "," << Size;
      }
      O << "\t\t# '" << I->getName() << "'\n";
    } else {
      switch (I->getLinkage()) {
      case GlobalValue::LinkOnceLinkage:
      case GlobalValue::WeakLinkage:
        O << "\t.global " << name << '\n'
          << "\t.weak_definition " << name << '\n';
        SwitchToDataSection(".section __DATA,__datacoal_nt,coalesced", I);
        break;
      case GlobalValue::AppendingLinkage:
        // FIXME: appending linkage variables should go into a section of
        // their name or something.  For now, just emit them as external.
      case GlobalValue::ExternalLinkage:
        // If external or appending, declare as a global symbol
        O << "\t.global " << name << "\n";
        // FALL THROUGH
      case GlobalValue::InternalLinkage:
        if (I->isConstant()) {
          const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
          if (TAI->getCStringSection() && CVA && CVA->isCString()) {
            SwitchToDataSection(TAI->getCStringSection(), I);
            break;
          }
        }

        SwitchToDataSection("\t.data", I);
        break;
      default:
        cerr << "Unknown linkage type!";
        abort();
      }

      EmitAlignment(Align, I);
      O << name << ":\t\t\t\t# '" << I->getName() << "'\n";

      // If the initializer is a extern weak symbol, remember to emit the weak
      // reference!
      if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
        if (GV->hasExternalWeakLinkage())
          ExtWeakSymbols.insert(GV);

      EmitGlobalConstant(C);
      O << '\n';
    }
  }

  // Output stubs for dynamically-linked functions
  if (TM.getRelocationModel() == Reloc::PIC_) {
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i) {
      SwitchToTextSection(".section __TEXT,__picsymbolstub1,symbol_stubs,"
                          "pure_instructions,32");
      EmitAlignment(4);
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
      SwitchToDataSection(".lazy_symbol_pointer");
      O << "L" << *i << "$lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long dyld_stub_binding_helper\n";
    }
  } else {
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i) {
      SwitchToTextSection(".section __TEXT,__symbol_stub1,symbol_stubs,"
                          "pure_instructions,16");
      EmitAlignment(4);
      O << "L" << *i << "$stub:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\tlis r11,ha16(L" << *i << "$lazy_ptr)\n";
      O << "\tlwzu r12,lo16(L" << *i << "$lazy_ptr)(r11)\n";
      O << "\tmtctr r12\n";
      O << "\tbctr\n";
      SwitchToDataSection(".lazy_symbol_pointer");
      O << "L" << *i << "$lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long dyld_stub_binding_helper\n";
    }
  }

  O << "\n";

  // Output stubs for external and common global variables.
  if (GVStubs.begin() != GVStubs.end()) {
    SwitchToDataSection(".non_lazy_symbol_pointer");
    for (std::set<std::string>::iterator I = GVStubs.begin(),
         E = GVStubs.end(); I != E; ++I) {
      O << "L" << *I << "$non_lazy_ptr:\n";
      O << "\t.indirect_symbol " << *I << "\n";
      O << "\t.long\t0\n";
    }
  }

  // Emit initial debug information.
  DW.EndModule();

  // Emit ident information
  O << "\t.ident\t\"(llvm 2.2+) STI CBEA Cell SPU backend\"\n";

  return AsmPrinter::doFinalization(M);
}



/// createSPUCodePrinterPass - Returns a pass that prints the Cell SPU
/// assembly code for a MachineFunction to the given output stream, in a format
/// that the Linux SPU assembler can deal with.
///
FunctionPass *llvm::createSPUAsmPrinterPass(std::ostream &o,
                                            SPUTargetMachine &tm) {
  return new LinuxAsmPrinter(o, tm, tm.getTargetAsmInfo());
}


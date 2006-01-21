//===-- PPCAsmPrinter.cpp - Print machine instrs to PowerPC assembly --------=//
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
#include "PPC.h"
#include "PPCTargetMachine.h"
#include "PPCSubtarget.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
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

  class PPCAsmPrinter : public AsmPrinter {
  public:
    std::set<std::string> FnStubs, GVStubs;
    
    PPCAsmPrinter(std::ostream &O, TargetMachine &TM)
      : AsmPrinter(O, TM) {}

    virtual const char *getPassName() const {
      return "PowerPC Assembly Printer";
    }

    PPCTargetMachine &getTM() {
      return static_cast<PPCTargetMachine&>(TM);
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
    void printOp(const MachineOperand &MO);

    void printOperand(const MachineInstr *MI, unsigned OpNo){
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (MO.getType() == MachineOperand::MO_MachineRegister) {
        assert(MRegisterInfo::isPhysicalRegister(MO.getReg())&&"Not physreg??");
        O << TM.getRegisterInfo()->get(MO.getReg()).Name;
      } else if (MO.isImmediate()) {
        O << MO.getImmedValue();
      } else {
        printOp(MO);
      }
    }

    void printU5ImmOperand(const MachineInstr *MI, unsigned OpNo) {
      unsigned char value = MI->getOperand(OpNo).getImmedValue();
      assert(value <= 31 && "Invalid u5imm argument!");
      O << (unsigned int)value;
    }
    void printU6ImmOperand(const MachineInstr *MI, unsigned OpNo) {
      unsigned char value = MI->getOperand(OpNo).getImmedValue();
      assert(value <= 63 && "Invalid u6imm argument!");
      O << (unsigned int)value;
    }
    void printS16ImmOperand(const MachineInstr *MI, unsigned OpNo) {
      O << (short)MI->getOperand(OpNo).getImmedValue();
    }
    void printU16ImmOperand(const MachineInstr *MI, unsigned OpNo) {
      O << (unsigned short)MI->getOperand(OpNo).getImmedValue();
    }
    void printS16X4ImmOperand(const MachineInstr *MI, unsigned OpNo) {
      O << (short)MI->getOperand(OpNo).getImmedValue()*4;
    }
    void printBranchOperand(const MachineInstr *MI, unsigned OpNo) {
      // Branches can take an immediate operand.  This is used by the branch
      // selection pass to print $+8, an eight byte displacement from the PC.
      if (MI->getOperand(OpNo).isImmediate()) {
        O << "$+" << MI->getOperand(OpNo).getImmedValue();
      } else {
        printOp(MI->getOperand(OpNo));
      }
    }
    void printCallOperand(const MachineInstr *MI, unsigned OpNo) {
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (!PPCGenerateStaticCode) {
        if (MO.getType() == MachineOperand::MO_GlobalAddress) {
          GlobalValue *GV = MO.getGlobal();
          if (((GV->isExternal() || GV->hasWeakLinkage() ||
                GV->hasLinkOnceLinkage()))) {
            // Dynamically-resolved functions need a stub for the function.
            std::string Name = Mang->getValueName(GV);
            FnStubs.insert(Name);
            O << "L" << Name << "$stub";
            return;
          }
        }
        if (MO.getType() == MachineOperand::MO_ExternalSymbol) {
          std::string Name(GlobalPrefix); Name += MO.getSymbolName();
          FnStubs.insert(Name);
          O << "L" << Name << "$stub";
          return;
        }
      }
      
      printOp(MI->getOperand(OpNo));
    }
    void printAbsAddrOperand(const MachineInstr *MI, unsigned OpNo) {
     O << (int)MI->getOperand(OpNo).getImmedValue()*4;
    }
    void printPICLabel(const MachineInstr *MI, unsigned OpNo) {
      O << "\"L" << getFunctionNumber() << "$pb\"\n";
      O << "\"L" << getFunctionNumber() << "$pb\":";
    }
    void printSymbolHi(const MachineInstr *MI, unsigned OpNo) {
      if (MI->getOperand(OpNo).isImmediate()) {
        printS16ImmOperand(MI, OpNo);
      } else {
        O << "ha16(";
        printOp(MI->getOperand(OpNo));
        if (PICEnabled)
          O << "-\"L" << getFunctionNumber() << "$pb\")";
        else
          O << ')';
      }
    }
    void printSymbolLo(const MachineInstr *MI, unsigned OpNo) {
      if (MI->getOperand(OpNo).isImmediate()) {
        printS16ImmOperand(MI, OpNo);
      } else {
        O << "lo16(";
        printOp(MI->getOperand(OpNo));
        if (PICEnabled)
          O << "-\"L" << getFunctionNumber() << "$pb\")";
        else
          O << ')';
      }
    }
    void printcrbitm(const MachineInstr *MI, unsigned OpNo) {
      unsigned CCReg = MI->getOperand(OpNo).getReg();
      unsigned RegNo = enumRegToMachineReg(CCReg);
      O << (0x80 >> RegNo);
    }
    // The new addressing mode printers, currently empty
    void printMemRegImm(const MachineInstr *MI, unsigned OpNo) {
      printSymbolLo(MI, OpNo);
      O << '(';
      printOperand(MI, OpNo+1);
      O << ')';
    }
    void printMemRegReg(const MachineInstr *MI, unsigned OpNo) {
      // When used as the base register, r0 reads constant zero rather than
      // the value contained in the register.  For this reason, the darwin
      // assembler requires that we print r0 as 0 (no r) when used as the base.
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (MO.getReg() == PPC::R0)
        O << '0';
      else
        O << TM.getRegisterInfo()->get(MO.getReg()).Name;
      O << ", ";
      printOperand(MI, OpNo+1);
    }
    
    virtual bool runOnMachineFunction(MachineFunction &F) = 0;
    virtual bool doFinalization(Module &M) = 0;
    
  };

  /// DarwinDwarfWriter - Dwarf debug info writer customized for Darwin/Mac OS X
  ///
  struct DarwinDwarfWriter : public DwarfWriter {
    // Ctor.
    DarwinDwarfWriter(std::ostream &o, AsmPrinter *ap)
    : DwarfWriter(o, ap)
    {
      needsSet = true;
      DwarfAbbrevSection = ".section __DWARFA,__debug_abbrev";
      DwarfInfoSection = ".section __DWARFA,__debug_info";
      DwarfLineSection = ".section __DWARFA,__debug_line";
      DwarfFrameSection =
          ".section __DWARFA,__debug_frame,,coalesced,no_toc+strip_static_syms";
      DwarfPubNamesSection = ".section __DWARFA,__debug_pubnames";
      DwarfPubTypesSection = ".section __DWARFA,__debug_pubtypes";
      DwarfStrSection = ".section __DWARFA,__debug_str";
      DwarfLocSection = ".section __DWARFA,__debug_loc";
      DwarfARangesSection = ".section __DWARFA,__debug_aranges";
      DwarfRangesSection = ".section __DWARFA,__debug_ranges";
      DwarfMacInfoSection = ".section __DWARFA,__debug_macinfo";
      TextSection = ".text";
      DataSection = ".data";
    }
  };

  /// DarwinAsmPrinter - PowerPC assembly printer, customized for Darwin/Mac OS
  /// X
  struct DarwinAsmPrinter : public PPCAsmPrinter {
  
    DarwinDwarfWriter DW;

    DarwinAsmPrinter(std::ostream &O, TargetMachine &TM)
      : PPCAsmPrinter(O, TM), DW(O, this)      
      {
      CommentString = ";";
      GlobalPrefix = "_";
      PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
      ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
      Data64bitsDirective = 0;       // we can't emit a 64-bit unit
      AlignmentIsInBytes = false;    // Alignment is by power of 2.
      ConstantPoolSection = "\t.const\t";
      LCOMMDirective = "\t.lcomm\t";
      StaticCtorsSection = ".mod_init_func";
      StaticDtorsSection = ".mod_term_func";
    }

    virtual const char *getPassName() const {
      return "Darwin PPC Assembly Printer";
    }
    
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
    
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<MachineDebugInfo>();
      PPCAsmPrinter::getAnalysisUsage(AU);
    }

  };

  /// AIXAsmPrinter - PowerPC assembly printer, customized for AIX
  ///
  struct AIXAsmPrinter : public PPCAsmPrinter {
    /// Map for labels corresponding to global variables
    ///
    std::map<const GlobalVariable*,std::string> GVToLabelMap;

    AIXAsmPrinter(std::ostream &O, TargetMachine &TM)
      : PPCAsmPrinter(O, TM) {
      CommentString = "#";
      GlobalPrefix = ".";
      ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
      Data64bitsDirective = 0;       // we can't emit a 64-bit unit
      AlignmentIsInBytes = false;    // Alignment is by power of 2.
      ConstantPoolSection = "\t.const\t";
    }

    virtual const char *getPassName() const {
      return "AIX PPC Assembly Printer";
    }

    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
  };
} // end of anonymous namespace

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
#include "PPCGenAsmWriter.inc"

void PPCAsmPrinter::printOp(const MachineOperand &MO) {
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
    std::cerr << "Shouldn't use addPCDisp() when building PPC MachineInstrs";
    abort();
    return;

  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << PrivateGlobalPrefix << "BB" << getFunctionNumber() << "_"
      << MBBOp->getNumber() << "\t; " << MBBOp->getBasicBlock()->getName();
    return;
  }

  case MachineOperand::MO_ConstantPoolIndex:
    O << PrivateGlobalPrefix << "CPI" << getFunctionNumber()
      << '_' << MO.getConstantPoolIndex();
    return;
  case MachineOperand::MO_ExternalSymbol:
    // Computing the address of an external symbol, not calling it.
    if (!PPCGenerateStaticCode) {
      std::string Name(GlobalPrefix); Name += MO.getSymbolName();
      GVStubs.insert(Name);
      O << "L" << Name << "$non_lazy_ptr";
      return;
    }
    O << GlobalPrefix << MO.getSymbolName();
    return;
  case MachineOperand::MO_GlobalAddress: {
    // Computing the address of a global symbol, not calling it.
    GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);
    int offset = MO.getOffset();

    // External or weakly linked global variables need non-lazily-resolved stubs
    if (!PPCGenerateStaticCode) {
      if (((GV->isExternal() || GV->hasWeakLinkage() ||
            GV->hasLinkOnceLinkage()))) {
        GVStubs.insert(Name);
        O << "L" << Name << "$non_lazy_ptr";
        return;
      }
    }

    O << Name;
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
void PPCAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
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
      printOperand(MI, 0);
      O << ", ";
      printOperand(MI, 1);
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
  // FIXME - is this the earliest this can be set.
  DW.SetDebugInfo(&getAnalysis<MachineDebugInfo>());

  SetupMachineFunction(MF);
  O << "\n\n";
  
  // Emit pre-function debug information.
  DW.BeginFunction();

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  const Function *F = MF.getFunction();
  switch (F->getLinkage()) {
  default: assert(0 && "Unknown linkage type!");
  case Function::InternalLinkage:  // Symbols default to internal.
    SwitchSection(".text", F);
    EmitAlignment(4, F);
    break;
  case Function::ExternalLinkage:
    SwitchSection(".text", F);
    EmitAlignment(4, F);
    O << "\t.globl\t" << CurrentFnName << "\n";
    break;
  case Function::WeakLinkage:
  case Function::LinkOnceLinkage:
    SwitchSection(".section __TEXT,__textcoal_nt,coalesced,pure_instructions",
                  F);
    O << "\t.globl\t" << CurrentFnName << "\n";
    O << "\t.weak_definition\t" << CurrentFnName << "\n";
    break;
  }
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I != MF.begin()) {
      O << PrivateGlobalPrefix << "BB" << getFunctionNumber() << '_'
        << I->getNumber() << ":\t";
      if (!I->getBasicBlock()->getName().empty())
        O << CommentString << " " << I->getBasicBlock()->getName();
      O << "\n";
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  // Emit post-function debug information.
  DW.EndFunction();

  // We didn't modify anything.
  return false;
}


bool DarwinAsmPrinter::doInitialization(Module &M) {
  if (TM.getSubtarget<PPCSubtarget>().isGigaProcessor())
    O << "\t.machine ppc970\n";
  AsmPrinter::doInitialization(M);
  
  // Darwin wants symbols to be quoted if they have complex names.
  Mang->setUseQuotes(true);
  
  // Emit initial debug information.
  DW.BeginModule();
  return false;
}

bool DarwinAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer()) continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (I->hasAppendingLinkage() && EmitSpecialLLVMGlobal(I))
      continue;
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD.getTypeSize(C->getType());
    unsigned Align = TD.getTypeAlignmentShift(C->getType());

    if (C->isNullValue() && /* FIXME: Verify correct */
        (I->hasInternalLinkage() || I->hasWeakLinkage() ||
         I->hasLinkOnceLinkage())) {
      SwitchSection(".data", I);
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
      if (I->hasInternalLinkage())
        O << LCOMMDirective << name << "," << Size << "," << Align;
      else
        O << ".comm " << name << "," << Size;
      O << "\t\t; '" << I->getName() << "'\n";
    } else {
      switch (I->getLinkage()) {
      case GlobalValue::LinkOnceLinkage:
      case GlobalValue::WeakLinkage:
        O << "\t.globl " << name << '\n'
          << "\t.weak_definition " << name << '\n';
        SwitchSection(".section __DATA,__datacoal_nt,coalesced", I);
        break;
      case GlobalValue::AppendingLinkage:
        // FIXME: appending linkage variables should go into a section of
        // their name or something.  For now, just emit them as external.
      case GlobalValue::ExternalLinkage:
        // If external or appending, declare as a global symbol
        O << "\t.globl " << name << "\n";
        // FALL THROUGH
      case GlobalValue::InternalLinkage:
        SwitchSection(".data", I);
        break;
      default:
        std::cerr << "Unknown linkage type!";
        abort();
      }

      EmitAlignment(Align, I);
      O << name << ":\t\t\t\t; '" << I->getName() << "'\n";
      EmitGlobalConstant(C);
      O << '\n';
    }
  }

  // Output stubs for dynamically-linked functions
  if (PICEnabled) {
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i) {
      SwitchSection(".section __TEXT,__picsymbolstub1,symbol_stubs,"
                    "pure_instructions,32", 0);
      EmitAlignment(2);
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
      SwitchSection(".lazy_symbol_pointer", 0);
      O << "L" << *i << "$lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long dyld_stub_binding_helper\n";
    }
  } else {
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i) {
      SwitchSection(".section __TEXT,__symbol_stub1,symbol_stubs,"
                    "pure_instructions,16", 0);
      EmitAlignment(4);
      O << "L" << *i << "$stub:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\tlis r11,ha16(L" << *i << "$lazy_ptr)\n";
      O << "\tlwzu r12,lo16(L" << *i << "$lazy_ptr)(r11)\n";
      O << "\tmtctr r12\n";
      O << "\tbctr\n";
      SwitchSection(".lazy_symbol_pointer", 0);
      O << "L" << *i << "$lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long dyld_stub_binding_helper\n";
    }
  }

  O << "\n";

  // Output stubs for external and common global variables.
  if (GVStubs.begin() != GVStubs.end()) {
    SwitchSection(".non_lazy_symbol_pointer", 0);
    for (std::set<std::string>::iterator I = GVStubs.begin(),
         E = GVStubs.end(); I != E; ++I) {
      O << "L" << *I << "$non_lazy_ptr:\n";
      O << "\t.indirect_symbol " << *I << "\n";
      O << "\t.long\t0\n";
    }
  }

  // Emit initial debug information.
  DW.EndModule();

  // Funny Darwin hack: This flag tells the linker that no global symbols
  // contain code that falls through to other global symbols (e.g. the obvious
  // implementation of multiple entry points).  If this doesn't occur, the
  // linker can safely perform dead code stripping.  Since LLVM never generates
  // code that does this, it is always safe to set.
  O << "\t.subsections_via_symbols\n";

  AsmPrinter::doFinalization(M);
  return false; // success
}

/// runOnMachineFunction - This uses the e()
/// method to print assembly for each instruction.
///
bool AIXAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  
  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

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
    O << PrivateGlobalPrefix << "BB" << getFunctionNumber() << '_'
      << I->getNumber()
      << ":\t" << CommentString << I->getBasicBlock()->getName() << '\n';
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
      II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

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

bool AIXAsmPrinter::doInitialization(Module &M) {
  SwitchSection("", 0);
  const TargetData &TD = TM.getTargetData();

  O << "\t.machine \"ppc64\"\n"
    << "\t.toc\n"
    << "\t.csect .text[PR]\n";

  // Print out module-level global variables
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
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
    EmitGlobalConstant(C);
  }

  // Output labels for globals
  if (M.global_begin() != M.global_end()) O << "\t.toc\n";
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    const GlobalVariable *GV = I;
    // Do not output labels for unused variables
    if (GV->isExternal() && GV->use_begin() == GV->use_end())
      continue;

    IncrementFunctionNumber();
    std::string Name = GV->getName();
    std::string Label = "LC.." + utostr(getFunctionNumber());
    GVToLabelMap[GV] = Label;
    O << Label << ":\n"
      << "\t.tc " << Name << "[TC]," << Name;
    if (GV->isExternal()) O << "[RW]";
    O << '\n';
   }

  AsmPrinter::doInitialization(M);
  return false; // success
}

bool AIXAsmPrinter::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  // Print out module-level global variables
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->hasInitializer() || I->hasExternalLinkage())
      continue;

    std::string Name = I->getName();
    if (I->hasInternalLinkage()) {
      O << "\t.lcomm " << Name << ",16,_global.bss_c";
    } else {
      O << "\t.comm " << Name << "," << TD.getTypeSize(I->getType())
        << "," << Log2_32((unsigned)TD.getTypeAlignment(I->getType()));
    }
    O << "\t\t" << CommentString << " ";
    WriteAsOperand(O, I, false, true, &M);
    O << "\n";
  }

  O << "_section_.text:\n"
    << "\t.csect .data[RW],3\n"
    << "\t.llong _section_.text\n";
  AsmPrinter::doFinalization(M);
  return false; // success
}

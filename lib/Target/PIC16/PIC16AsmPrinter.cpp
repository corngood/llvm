//===-- PIC16AsmPrinter.cpp - PIC16 LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PIC16 assembly language.
//
//===----------------------------------------------------------------------===//

#include "PIC16AsmPrinter.h"
#include "PIC16TargetAsmInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Mangler.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

#include "PIC16GenAsmWriter.inc"

bool PIC16AsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  printInstruction(MI);
  return true;
}

/// runOnMachineFunction - This uses the printInstruction()
/// method to print assembly for each instruction.
///
bool PIC16AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;

  // This calls the base class function required to be called at beginning
  // of runOnMachineFunction.
  SetupMachineFunction(MF);

  // Get the mangled name.
  const Function *F = MF.getFunction();
  CurrentFnName = Mang->getValueName(F);

  // Emit the function variables.
  emitFunctionData(MF);
  const char *codeSection = PAN::getCodeSectionName(CurrentFnName).c_str();
 
  const Section *fCodeSection = TAI->getNamedSection(codeSection,
                                                     SectionFlags::Code);
  O <<  "\n";
  // Start the Code Section.
  SwitchToSection (fCodeSection);

  // Emit the frame address of the function at the beginning of code.
  O << "\tretlw  low(" << PAN::getFrameLabel(CurrentFnName) << ")\n";
  O << "\tretlw  high(" << PAN::getFrameLabel(CurrentFnName) << ")\n";

  // Emit function start label.
  O << CurrentFnName << ":\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    if (I != MF.begin()) {
      printBasicBlockLabel(I, true);
      O << '\n';
    }
    
    // For emitting line directives, we need to keep track of the current
    // source line. When it changes then only emit the line directive.
    unsigned CurLine = 0;
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Emit the line directive if source line changed.
      const DebugLoc DL = II->getDebugLoc();
      if (!DL.isUnknown()) {
        unsigned line = MF.getDebugLocTuple(DL).Line;
        if (line != CurLine) {
          O << "\t.line " << line << "\n";
          CurLine = line;
        }
      }
        
      // Print the assembly for the instruction.
      printMachineInstruction(II);
    }
  }
  return false;  // we didn't modify anything.
}

/// createPIC16CodePrinterPass - Returns a pass that prints the PIC16
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *llvm::createPIC16CodePrinterPass(raw_ostream &o,
                                               PIC16TargetMachine &tm,
                                               CodeGenOpt::Level OptLevel,
                                               bool verbose) {
  return new PIC16AsmPrinter(o, tm, tm.getTargetAsmInfo(), OptLevel, verbose);
}


// printOperand - print operand of insn.
void PIC16AsmPrinter::printOperand(const MachineInstr *MI, int opNum) {
  const MachineOperand &MO = MI->getOperand(opNum);

  switch (MO.getType()) {
    case MachineOperand::MO_Register:
      if (TargetRegisterInfo::isPhysicalRegister(MO.getReg()))
        O << TM.getRegisterInfo()->get(MO.getReg()).AsmName;
      else
        assert(0 && "not implemented");
        return;

    case MachineOperand::MO_Immediate:
      O << (int)MO.getImm();
      return;

    case MachineOperand::MO_GlobalAddress: {
      O << Mang->getValueName(MO.getGlobal());
      break;
    }
    case MachineOperand::MO_ExternalSymbol: {
       const char *Sname = MO.getSymbolName();

      // If its a libcall name, record it to decls section.
      if (PAN::getSymbolTag(Sname) == PAN::LIBCALL) {
        LibcallDecls.push_back(Sname);
      }

      O  << Sname;
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      printBasicBlockLabel(MO.getMBB());
      return;

    default:
      assert(0 && " Operand type not supported.");
  }
}

void PIC16AsmPrinter::printCCOperand(const MachineInstr *MI, int opNum) {
  int CC = (int)MI->getOperand(opNum).getImm();
  O << PIC16CondCodeToString((PIC16CC::CondCodes)CC);
}

void PIC16AsmPrinter::printLibcallDecls(void) {
  // If no libcalls used, return.
  if (LibcallDecls.empty()) return;

  O << TAI->getCommentString() << "External decls for libcalls - BEGIN." <<"\n";
  // Remove duplicate entries.
  LibcallDecls.sort();
  LibcallDecls.unique();
  for (std::list<const char*>::const_iterator I = LibcallDecls.begin(); 
       I != LibcallDecls.end(); I++) {
    O << TAI->getExternDirective() << *I << "\n";
    O << TAI->getExternDirective() << PAN::getArgsLabel(*I) << "\n";
    O << TAI->getExternDirective() << PAN::getRetvalLabel(*I) << "\n";
  }
  O << TAI->getCommentString() << "External decls for libcalls - END." <<"\n";
}

bool PIC16AsmPrinter::doInitialization (Module &M) {
  bool Result = AsmPrinter::doInitialization(M);
  // FIXME:: This is temporary solution to generate the include file.
  // The processor should be passed to llc as in input and the header file
  // should be generated accordingly.
  O << "\t#include P16F1937.INC\n";
  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  assert(MMI);
  DwarfWriter *DW = getAnalysisIfAvailable<DwarfWriter>();
  assert(DW && "Dwarf Writer is not available");
  DW->BeginModule(&M, MMI, O, this, TAI);

  // Set the section names for all globals.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    I->setSection(TAI->SectionForGlobal(I)->getName());
  }

  EmitFunctionDecls(M);
  EmitUndefinedVars(M);
  EmitDefinedVars(M);
  EmitIData(M);
  EmitUData(M);
  EmitRomData(M);
  EmitAutos(M);
  return Result;
}

// Emit extern decls for functions imported from other modules, and emit
// global declarations for function defined in this module and which are
// available to other modules.
void PIC16AsmPrinter::EmitFunctionDecls (Module &M) {
 // Emit declarations for external functions.
  O << TAI->getCommentString() << "Function Declarations - BEGIN." <<"\n";
  for (Module::iterator I = M.begin(), E = M.end(); I != E; I++) {
    std::string Name = Mang->getValueName(I);
    if (Name.compare("@abort") == 0)
      continue;
    
    // If it is llvm intrinsic call then don't emit
    if (Name.find("llvm.") != std::string::npos)
      continue;

    if (! (I->isDeclaration() || I->hasExternalLinkage()))
      continue;

    const char *directive = I->isDeclaration() ? TAI->getExternDirective() :
                                                 TAI->getGlobalDirective();
      
    O << directive << Name << "\n";
    O << directive << PAN::getRetvalLabel(Name) << "\n";
    O << directive << PAN::getArgsLabel(Name) << "\n";
  }

  O << TAI->getCommentString() << "Function Declarations - END." <<"\n";
}

// Emit variables imported from other Modules.
void PIC16AsmPrinter::EmitUndefinedVars (Module &M)
{
  std::vector<const GlobalVariable*> Items = PTAI->ExternalVarDecls->Items;
  if (! Items.size()) return;

  O << "\n" << TAI->getCommentString() << "Imported Variables - BEGIN" << "\n";
  for (unsigned j = 0; j < Items.size(); j++) {
    O << TAI->getExternDirective() << Mang->getValueName(Items[j]) << "\n";
  }
  O << TAI->getCommentString() << "Imported Variables - END" << "\n";
}

// Emit variables defined in this module and are available to other modules.
void PIC16AsmPrinter::EmitDefinedVars (Module &M)
{
  std::vector<const GlobalVariable*> Items = PTAI->ExternalVarDefs->Items;
  if (! Items.size()) return;

  O << "\n" <<  TAI->getCommentString() << "Exported Variables - BEGIN" << "\n";
  for (unsigned j = 0; j < Items.size(); j++) {
    O << TAI->getGlobalDirective() << Mang->getValueName(Items[j]) << "\n";
  }
  O <<  TAI->getCommentString() << "Exported Variables - END" << "\n";
}

// Emit initialized data placed in ROM.
void PIC16AsmPrinter::EmitRomData (Module &M)
{

  std::vector<const GlobalVariable*> Items = PTAI->ROSection->Items;
  if (! Items.size()) return;

  // Print ROData ection.
  O << "\n";
  SwitchToSection(PTAI->ROSection->S_);
  for (unsigned j = 0; j < Items.size(); j++) {
    O << Mang->getValueName(Items[j]);
    Constant *C = Items[j]->getInitializer();
    int AddrSpace = Items[j]->getType()->getAddressSpace();
    EmitGlobalConstant(C, AddrSpace);
  }
}

bool PIC16AsmPrinter::doFinalization(Module &M) {
  printLibcallDecls();
  EmitVarDebugInfo(M);
  O << "\t" << ".EOF\n";
  O << "\t" << "END\n";
  bool Result = AsmPrinter::doFinalization(M);
  return Result;
}

void PIC16AsmPrinter::emitFunctionData(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  std::string FuncName = Mang->getValueName(F);
  const TargetData *TD = TM.getTargetData();
  // Emit the data section name.
  O << "\n"; 
  const char *SectionName = PAN::getFrameSectionName(CurrentFnName).c_str();

  const Section *fPDataSection = TAI->getNamedSection(SectionName,
                                                      SectionFlags::Writeable);
  SwitchToSection(fPDataSection);
  
  // Emit function frame label
  O << PAN::getFrameLabel(CurrentFnName) << ":\n";

  const Type *RetType = F->getReturnType();
  unsigned RetSize = 0; 
  if (RetType->getTypeID() != Type::VoidTyID) 
    RetSize = TD->getTypeAllocSize(RetType);
  
  //Emit function return value space
  // FIXME: Do not emit RetvalLable when retsize is zero. To do this
  // we will need to avoid printing a global directive for Retval label
  // in emitExternandGloblas.
  if(RetSize > 0)
     O << PAN::getRetvalLabel(CurrentFnName) << " RES " << RetSize << "\n";
  else
     O << PAN::getRetvalLabel(CurrentFnName) << ": \n";
   
  // Emit variable to hold the space for function arguments 
  unsigned ArgSize = 0;
  for (Function::const_arg_iterator argi = F->arg_begin(),
           arge = F->arg_end(); argi != arge ; ++argi) {
    const Type *Ty = argi->getType();
    ArgSize += TD->getTypeAllocSize(Ty);
   }

  O << PAN::getArgsLabel(CurrentFnName) << " RES " << ArgSize << "\n";

  // Emit temporary space
  int TempSize = PTLI->GetTmpSize();
  if (TempSize > 0 )
    O << PAN::getTempdataLabel(CurrentFnName) << " RES  " << TempSize <<"\n";
}

void PIC16AsmPrinter::EmitIData (Module &M) {

  // Print all IDATA sections.
  std::vector <PIC16Section *>IDATASections = PTAI->IDATASections;
  for (unsigned i = 0; i < IDATASections.size(); i++) {
    O << "\n";
    SwitchToSection(IDATASections[i]->S_);
    std::vector<const GlobalVariable*> Items = IDATASections[i]->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      std::string Name = Mang->getValueName(Items[j]);
      Constant *C = Items[j]->getInitializer();
      int AddrSpace = Items[j]->getType()->getAddressSpace();
      O << Name;
      EmitGlobalConstant(C, AddrSpace);
    }
  }
}

void PIC16AsmPrinter::EmitUData (Module &M) {
  const TargetData *TD = TM.getTargetData();

  // Print all BSS sections.
  std::vector <PIC16Section *>BSSSections = PTAI->BSSSections;
  for (unsigned i = 0; i < BSSSections.size(); i++) {
    O << "\n";
    SwitchToSection(BSSSections[i]->S_);
    std::vector<const GlobalVariable*> Items = BSSSections[i]->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      std::string Name = Mang->getValueName(Items[j]);
      Constant *C = Items[j]->getInitializer();
      const Type *Ty = C->getType();
      unsigned Size = TD->getTypeAllocSize(Ty);

      O << Name << " " <<"RES"<< " " << Size ;
      O << "\n";
    }
  }
}

void PIC16AsmPrinter::EmitAutos (Module &M)
{
  // Section names for all globals are already set.

  const TargetData *TD = TM.getTargetData();

  // Now print all Autos sections.
  std::vector <PIC16Section *>AutosSections = PTAI->AutosSections;
  for (unsigned i = 0; i < AutosSections.size(); i++) {
    O << "\n";
    SwitchToSection(AutosSections[i]->S_);
    std::vector<const GlobalVariable*> Items = AutosSections[i]->Items;
    for (unsigned j = 0; j < Items.size(); j++) {
      std::string VarName = Mang->getValueName(Items[j]);
      Constant *C = Items[j]->getInitializer();
      const Type *Ty = C->getType();
      unsigned Size = TD->getTypeAllocSize(Ty);
      // Emit memory reserve directive.
      O << VarName << "  RES  " << Size << "\n";
    }
  }
}

void PIC16AsmPrinter::EmitVarDebugInfo(Module &M) {
  GlobalVariable *Root = M.getGlobalVariable("llvm.dbg.global_variables");
  if (!Root)
    return;

  O << TAI->getCommentString() << " Debug Information:";
  Constant *RootC = cast<Constant>(*Root->use_begin());
  for (Value::use_iterator UI = RootC->use_begin(), UE = Root->use_end();
       UI != UE; ++UI) {
    for (Value::use_iterator UUI = UI->use_begin(), UUE = UI->use_end();
         UUI != UUE; ++UUI) {
      DIGlobalVariable DIGV(cast<GlobalVariable>(*UUI));
      DIType Ty = DIGV.getType();
      unsigned short TypeNo = 0;
      bool HasAux = false;
      int Aux[20] = { 0 };
      std::string TypeName = "";
      std::string VarName = TAI->getGlobalPrefix()+DIGV.getGlobal()->getName();
      DbgInfo.PopulateDebugInfo(Ty, TypeNo, HasAux, Aux, TypeName);
      // Emit debug info only if type information is availaible.
      if (TypeNo != PIC16Dbg::T_NULL) {
        O << "\n\t.type " << VarName << ", " << TypeNo;
        short ClassNo = DbgInfo.getClass(DIGV);
        O << "\n\t.class " << VarName << ", " << ClassNo;
        if (HasAux) {
          if (TypeName != "") {
           // Emit debug info for structure and union objects after
           // .dim directive supports structure/union tag name in aux entry.
           /* O << "\n\t.dim " << VarName << ", 1," << TypeName;
            for (int i = 0; i<20; i++)
              O << "," << Aux[i];*/
          }
          else {
            O << "\n\t.dim " << VarName << ", 1" ;
            for (int i = 0; i<20; i++)
              O << "," << Aux[i];
          }
        }
      }
    }
  }
  O << "\n";
}


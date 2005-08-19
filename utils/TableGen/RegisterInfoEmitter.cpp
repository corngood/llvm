//===- RegisterInfoEmitter.cpp - Generate a Register File Desc. -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of a target
// register file for a code generator.  It uses instances of the Register,
// RegisterAliases, and RegisterClass classes to gather this information.
//
//===----------------------------------------------------------------------===//

#include "RegisterInfoEmitter.h"
#include "CodeGenTarget.h"
#include "CodeGenRegisters.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
using namespace llvm;

// runEnums - Print out enum values for all of the registers.
void RegisterInfoEmitter::runEnums(std::ostream &OS) {
  CodeGenTarget Target;
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();

  std::string Namespace = Registers[0].TheDef->getValueAsString("Namespace");

  EmitSourceFileHeader("Target Register Enum Values", OS);
  OS << "namespace llvm {\n\n";

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n    NoRegister,\n";

  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    OS << "    " << Registers[i].getName() << ", \t// " << i+1 << "\n";

  OS << "  };\n";
  if (!Namespace.empty())
    OS << "}\n";
  OS << "} // End llvm namespace \n";
}

void RegisterInfoEmitter::runHeader(std::ostream &OS) {
  EmitSourceFileHeader("Register Information Header Fragment", OS);
  CodeGenTarget Target;
  const std::string &TargetName = Target.getName();
  std::string ClassName = TargetName + "GenRegisterInfo";

  OS << "#include \"llvm/Target/MRegisterInfo.h\"\n\n";

  OS << "namespace llvm {\n\n";

  OS << "struct " << ClassName << " : public MRegisterInfo {\n"
     << "  " << ClassName
     << "(int CallFrameSetupOpcode = -1, int CallFrameDestroyOpcode = -1);\n"
     << "  const unsigned* getCalleeSaveRegs() const;\n"
     << "};\n\n";

  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " { // Register classes\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      const std::string &Name = RegisterClasses[i].getName();
      OS << "  extern TargetRegisterClass * const "<< Name <<"RegisterClass;\n";
    }
    OS << "} // end of namespace " << TargetName << "\n\n";
  }
  OS << "} // End llvm namespace \n";
}

// RegisterInfoEmitter::run - Main register file description emitter.
//
void RegisterInfoEmitter::run(std::ostream &OS) {
  CodeGenTarget Target;
  EmitSourceFileHeader("Register Information Source Fragment", OS);

  OS << "namespace llvm {\n\n";

  // Start out by emitting each of the register classes... to do this, we build
  // a set of registers which belong to a register class, this is to ensure that
  // each register is only in a single register class.
  //
  const std::vector<CodeGenRegisterClass> &RegisterClasses =
    Target.getRegisterClasses();

  std::set<Record*> RegistersFound;

  // Loop over all of the register classes... emitting each one.
  OS << "namespace {     // Register classes...\n";

  // RegClassesBelongedTo - Keep track of which register classes each reg
  // belongs to.
  std::multimap<Record*, const CodeGenRegisterClass*> RegClassesBelongedTo;

  for (unsigned rc = 0, e = RegisterClasses.size(); rc != e; ++rc) {
    const CodeGenRegisterClass &RC = RegisterClasses[rc];

    // Give the register class a legal C name if it's anonymous.
    std::string Name = RC.TheDef->getName();
  
    // Emit the register list now.
    OS << "  // " << Name << " Register Class...\n  const unsigned " << Name
       << "[] = {\n    ";
    for (unsigned i = 0, e = RC.Elements.size(); i != e; ++i) {
      Record *Reg = RC.Elements[i];
      if (RegistersFound.count(Reg))
        throw "Register '" + Reg->getName() +
              "' included in multiple register classes!";
      RegistersFound.insert(Reg);
      OS << getQualifiedName(Reg) << ", ";

      // Keep track of which regclasses this register is in.
      RegClassesBelongedTo.insert(std::make_pair(Reg, &RC));
    }
    OS << "\n  };\n\n";

    OS << "  struct " << Name << "Class : public TargetRegisterClass {\n"
       << "    " << Name << "Class() : TargetRegisterClass("
       << RC.SpillSize/8 << ", " << RC.SpillAlignment/8 << ", " << Name << ", "
       << Name << " + " << RC.Elements.size() << ") {}\n"
       << RC.MethodProtos << "  };\n";
    OS << RC.MethodBodies << "\n";
  }
  OS << "}  // end anonymous namespace\n\n";
  
  // Now that all of the structs have been emitted, emit the instances.
  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " {   // Register class instances\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
      OS << "  " << RegisterClasses[i].getName()  << "Class\t"
         << RegisterClasses[i].getName() << "RegClassInstance;\n";
    OS << "}\n";
  }

  OS << "\nnamespace {\n";
  OS << "  const TargetRegisterClass* const RegisterClasses[] = {\n";
  for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i)
    OS << "    &" << getQualifiedName(RegisterClasses[i].TheDef)
       << "RegClassInstance,\n";
  OS << "  };\n";

  // Emit register class aliases...
  std::map<Record*, std::set<Record*> > RegisterAliases;
  const std::vector<CodeGenRegister> &Regs = Target.getRegisters();

  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    Record *R = Regs[i].TheDef;
    ListInit *LI = Regs[i].TheDef->getValueAsListInit("Aliases");
    // Add information that R aliases all of the elements in the list... and
    // that everything in the list aliases R.
    for (unsigned j = 0, e = LI->getSize(); j != e; ++j) {
      DefInit *Reg = dynamic_cast<DefInit*>(LI->getElement(j));
      if (!Reg) throw "ERROR: Alias list element is not a def!";
      if (RegisterAliases[R].count(Reg->getDef()))
        std::cerr << "Warning: register alias between " << getQualifiedName(R)
                  << " and " << getQualifiedName(Reg->getDef())
                  << " specified multiple times!\n";
      RegisterAliases[R].insert(Reg->getDef());

      if (RegisterAliases[Reg->getDef()].count(R))
        std::cerr << "Warning: register alias between " << getQualifiedName(R)
                  << " and " << getQualifiedName(Reg->getDef())
                  << " specified multiple times!\n";
      RegisterAliases[Reg->getDef()].insert(R);
    }
  }

  if (!RegisterAliases.empty())
    OS << "\n\n  // Register Alias Sets...\n";

  // Emit the empty alias list
  OS << "  const unsigned Empty_AliasSet[] = { 0 };\n";
  // Loop over all of the registers which have aliases, emitting the alias list
  // to memory.
  for (std::map<Record*, std::set<Record*> >::iterator
         I = RegisterAliases.begin(), E = RegisterAliases.end(); I != E; ++I) {
    OS << "  const unsigned " << I->first->getName() << "_AliasSet[] = { ";
    for (std::set<Record*>::iterator ASI = I->second.begin(),
           E = I->second.end(); ASI != E; ++ASI)
      OS << getQualifiedName(*ASI) << ", ";
    OS << "0 };\n";
  }

  OS << "\n  const MRegisterDesc RegisterDescriptors[] = { // Descriptors\n";
  OS << "    { \"NOREG\",\t0,\t\t0,\t0 },\n";


  // Now that register alias sets have been emitted, emit the register
  // descriptors now.
  const std::vector<CodeGenRegister> &Registers = Target.getRegisters();
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    const CodeGenRegister &Reg = Registers[i];
    OS << "    { \"";
    if (!Reg.TheDef->getValueAsString("Name").empty())
      OS << Reg.TheDef->getValueAsString("Name");
    else
      OS << Reg.getName();
    OS << "\",\t";
    if (RegisterAliases.count(Reg.TheDef))
      OS << Reg.getName() << "_AliasSet,\t";
    else
      OS << "Empty_AliasSet,\t";

    // Figure out what the size and alignment of the spill slots are for this
    // reg.  This may be explicitly declared in the register, or it may be
    // inferred from the register classes it is part of.
    std::multimap<Record*, const CodeGenRegisterClass*>::iterator I, E;
    tie(I, E) = RegClassesBelongedTo.equal_range(Reg.TheDef);
    unsigned SpillSize = Reg.DeclaredSpillSize;
    unsigned SpillAlign = Reg.DeclaredSpillAlignment;
    for (; I != E; ++I) {   // For each reg class this belongs to.
      const CodeGenRegisterClass *RC = I->second;
      if (SpillSize == 0)
        SpillSize = RC->SpillSize;
      else if (SpillSize != RC->SpillSize)
        throw "Spill size for regclass '" + RC->getName() +
              "' doesn't match spill sized already inferred for register '" +
              Reg.getName() + "'!";
      if (SpillAlign == 0)
        SpillAlign = RC->SpillAlignment;
      else if (SpillAlign != RC->SpillAlignment)
        throw "Spill alignment for regclass '" + RC->getName() +
              "' doesn't match spill sized already inferred for register '" +
              Reg.getName() + "'!";
    }

    OS << SpillSize << ", " << SpillAlign << " },\n";
  }
  OS << "  };\n";      // End of register descriptors...
  OS << "}\n\n";       // End of anonymous namespace...

  if (!RegisterClasses.empty()) {
    OS << "namespace " << RegisterClasses[0].Namespace
       << " { // Register classes\n";
    for (unsigned i = 0, e = RegisterClasses.size(); i != e; ++i) {
      OS << "  TargetRegisterClass * const " << RegisterClasses[i].getName()
         << "RegisterClass = &" << getQualifiedName(RegisterClasses[i].TheDef)
         << "RegClassInstance;\n";
    }
    OS << "} // end of namespace " << RegisterClasses[0].Namespace << "\n\n";
  }


  std::string ClassName = Target.getName() + "GenRegisterInfo";

  // Emit the constructor of the class...
  OS << ClassName << "::" << ClassName
     << "(int CallFrameSetupOpcode, int CallFrameDestroyOpcode)\n"
     << "  : MRegisterInfo(RegisterDescriptors, " << Registers.size()+1
     << ", RegisterClasses, RegisterClasses+" << RegisterClasses.size() <<",\n "
     << "                 CallFrameSetupOpcode, CallFrameDestroyOpcode) {}\n\n";

  // Emit the getCalleeSaveRegs method...
  OS << "const unsigned* " << ClassName << "::getCalleeSaveRegs() const {\n"
     << "  static const unsigned CalleeSaveRegs[] = {\n    ";

  const std::vector<Record*> &CSR = Target.getCalleeSavedRegisters();
  for (unsigned i = 0, e = CSR.size(); i != e; ++i)
    OS << getQualifiedName(CSR[i]) << ", ";
  OS << " 0\n  };\n  return CalleeSaveRegs;\n}\n\n";
  OS << "} // End llvm namespace \n";
}

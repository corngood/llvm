//===- IntrinsicEmitter.cpp - Generate intrinsic information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits information about intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "IntrinsicEmitter.h"
#include "Record.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// CodeGenIntrinsic Implementation
//===----------------------------------------------------------------------===//

std::vector<CodeGenIntrinsic> llvm::LoadIntrinsics(const RecordKeeper &RC) {
  std::vector<Record*> I = RC.getAllDerivedDefinitions("Intrinsic");
  return std::vector<CodeGenIntrinsic>(I.begin(), I.end());
}

CodeGenIntrinsic::CodeGenIntrinsic(Record *R) {
  std::string DefName = R->getName();
  
  if (DefName.size() <= 4 || 
      std::string(DefName.begin(), DefName.begin()+4) != "int_")
    throw "Intrinsic '" + DefName + "' does not start with 'int_'!";
  EnumName = std::string(DefName.begin()+4, DefName.end());
  
  Name = R->getValueAsString("LLVMName");
  if (Name == "") {
    // If an explicit name isn't specified, derive one from the DefName.
    Name = "llvm.";
    for (unsigned i = 0, e = EnumName.size(); i != e; ++i)
      if (EnumName[i] == '_')
        Name += '.';
      else
        Name += EnumName[i];
  }
  
  // Parse the list of argument types.
  ListInit *TypeList = R->getValueAsListInit("Types");
  for (unsigned i = 0, e = TypeList->getSize(); i != e; ++i) {
    DefInit *DI = dynamic_cast<DefInit*>(TypeList->getElement(i));
    assert(DI && "Invalid list type!");
    Record *TyEl = DI->getDef();
    assert(TyEl->isSubClassOf("LLVMType") && "Expected a type!");
    ArgTypes.push_back(TyEl->getValueAsString("TypeVal"));
  }
  if (ArgTypes.size() == 0)
    throw "Intrinsic '"+DefName+"' needs at least a type for the ret value!";
}

//===----------------------------------------------------------------------===//
// IntrinsicEmitter Implementation
//===----------------------------------------------------------------------===//

void IntrinsicEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("Intrinsic Function Source Fragment", OS);
  
  std::vector<CodeGenIntrinsic> Ints = LoadIntrinsics(Records);

  // Emit the enum information.
  EmitEnumInfo(Ints, OS);
  
  // Emit the function name recognizer.
  EmitFnNameRecognizer(Ints, OS);

  // Emit the intrinsic verifier.
  EmitVerifier(Ints, OS);
}

void IntrinsicEmitter::EmitEnumInfo(const std::vector<CodeGenIntrinsic> &Ints,
                                    std::ostream &OS) {
  OS << "// Enum values for Intrinsics.h\n";
  OS << "#ifdef GET_INTRINSIC_ENUM_VALUES\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "    " << Ints[i].EnumName;
    OS << ((i != e-1) ? ", " : "  ");
    OS << std::string(40-Ints[i].EnumName.size(), ' ') 
      << "// " << Ints[i].Name << "\n";
  }
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitFnNameRecognizer(const std::vector<CodeGenIntrinsic> &Ints, 
                     std::ostream &OS) {
  // Build a function name -> intrinsic name mapping.
  std::map<std::string, std::string> IntMapping;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    IntMapping[Ints[i].Name] = Ints[i].EnumName;
    
  OS << "// Function name -> enum value recognizer code.\n";
  OS << "#ifdef GET_FUNCTION_RECOGNIZER\n";
  OS << "  switch (Name[5]) {\n";
  OS << "  default: break;\n";
  // Emit the intrinsics in sorted order.
  char LastChar = 0;
  for (std::map<std::string, std::string>::iterator I = IntMapping.begin(),
       E = IntMapping.end(); I != E; ++I) {
    assert(I->first.size() > 5 && std::string(I->first.begin(),
                                              I->first.begin()+5) == "llvm." &&
           "Invalid intrinsic name!");
    if (I->first[5] != LastChar) {
      LastChar = I->first[5];
      OS << "  case '" << LastChar << "':\n";
    }
    
    OS << "    if (Name == \"" << I->first << "\") return Intrinsic::"
       << I->second << ";\n";
  }
  OS << "  }\n";
  OS << "  // The 'llvm.' namespace is reserved!\n";
  OS << "  assert(0 && \"Unknown LLVM intrinsic function!\");\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::EmitVerifier(const std::vector<CodeGenIntrinsic> &Ints, 
                                    std::ostream &OS) {
  OS << "// Verifier::visitIntrinsicFunctionCall code.\n";
  OS << "#ifdef GET_INTRINSIC_VERIFIER\n";
  OS << "  switch (ID) {\n";
  OS << "  default: assert(0 && \"Invalid intrinsic!\");\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "  case Intrinsic::" << Ints[i].EnumName << ":\t\t// "
       << Ints[i].Name << "\n";
    OS << "    Assert1(FTy->getNumParams() == " << Ints[i].ArgTypes.size()-1
       << ",\n"
       << "            \"Illegal # arguments for intrinsic function!\", IF);\n";
    OS << "    Assert1(FTy->getReturnType()->getTypeID() == "
       << Ints[i].ArgTypes[0] << ",\n"
       << "            \"Illegal result type!\", IF);\n";
    for (unsigned j = 1; j != Ints[i].ArgTypes.size(); ++j)
      OS << "    Assert1(FTy->getParamType(" << j-1 << ")->getTypeID() == "
         << Ints[i].ArgTypes[j] << ",\n"
         << "            \"Illegal result type!\", IF);\n";
    OS << "    break;\n";
  }
  OS << "  }\n";
  OS << "#endif\n\n";
}


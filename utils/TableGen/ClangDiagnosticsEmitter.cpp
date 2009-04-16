//=- ClangDiagnosticsEmitter.cpp - Generate Clang diagnostics tables -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang diagnostics tables.
//
//===----------------------------------------------------------------------===//

#include "ClangDiagnosticsEmitter.h"
#include "Record.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/VectorExtras.h"
#include <set>
#include <map>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Warning Tables (.inc file) generation.
//===----------------------------------------------------------------------===//

void ClangDiagsDefsEmitter::run(std::ostream &OS) {
  // Write the #if guard
  if (!Component.empty()) {
    std::string ComponentName = UppercaseString(Component);
    OS << "#ifdef " << ComponentName << "START\n";
    OS << "__" << ComponentName << "START = DIAG_START_" << ComponentName
       << ",\n";
    OS << "#undef " << ComponentName << "START\n";
    OS << "#endif\n";
  }

  const std::vector<Record*> &Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    const Record &R = *Diags[i];
    // Filter by component.
    if (!Component.empty() && Component != R.getValueAsString("Component"))
      continue;
    
    OS << "DIAG(" << R.getName() << ", ";
    OS << R.getValueAsDef("Class")->getName();
    OS << ", diag::" << R.getValueAsDef("DefaultMapping")->getName();
    OS << ", \"";
    std::string S = R.getValueAsString("Text");
    EscapeString(S);
    OS << S << "\")\n";
  }
}

//===----------------------------------------------------------------------===//
// Warning Group Tables generation
//===----------------------------------------------------------------------===//

struct GroupInfo {
  std::vector<const Record*> DiagsInGroup;
  std::vector<std::string> SubGroups;
};

void ClangDiagGroupsEmitter::run(std::ostream &OS) {
  // Invert the 1-[0/1] mapping of diags to group into a one to many mapping of
  // groups to diags in the group.
  std::map<std::string, GroupInfo> DiagsInGroup;
  
  std::vector<Record*> Diags =
    Records.getAllDerivedDefinitions("Diagnostic");
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    const Record *R = Diags[i];
    DefInit *DI = dynamic_cast<DefInit*>(R->getValueInit("Group"));
    if (DI == 0) continue;
    std::string GroupName = DI->getDef()->getValueAsString("GroupName");
    DiagsInGroup[GroupName].DiagsInGroup.push_back(R);
  }
  
  // Add all DiagGroup's to the DiagsInGroup list to make sure we pick up empty
  // groups (these are warnings that GCC supports that clang never produces).
  Diags = Records.getAllDerivedDefinitions("DiagGroup");
  for (unsigned i = 0, e = Diags.size(); i != e; ++i) {
    Record *Group = Diags[i];
    GroupInfo &GI = DiagsInGroup[Group->getValueAsString("GroupName")];
    
    std::vector<Record*> SubGroups = Group->getValueAsListOfDefs("SubGroups");
    for (unsigned j = 0, e = SubGroups.size(); j != e; ++j)
      GI.SubGroups.push_back(SubGroups[j]->getValueAsString("GroupName"));
  }
  
  // Walk through the groups emitting an array for each diagnostic of the diags
  // that are mapped to.
  OS << "\n#ifdef GET_DIAG_ARRAYS\n";
  unsigned IDNo = 0;
  unsigned MaxLen = 0;
  for (std::map<std::string, GroupInfo>::iterator
       I = DiagsInGroup.begin(), E = DiagsInGroup.end(); I != E; ++I) {
    MaxLen = std::max(MaxLen, (unsigned)I->first.size());
    
    std::vector<const Record*> &V = I->second.DiagsInGroup;
    if (V.empty()) continue;
    
    OS << "static const short DiagArray" << IDNo++
       << "[] = { ";
    for (unsigned i = 0, e = V.size(); i != e; ++i)
      OS << "diag::" << V[i]->getName() << ", ";
    OS << "-1 };\n";
  }
  OS << "#endif // GET_DIAG_ARRAYS\n\n";
  
  // Emit the table now.
  OS << "\n#ifdef GET_DIAG_TABLE\n";
  IDNo = 0;
  for (std::map<std::string, GroupInfo>::iterator
       I = DiagsInGroup.begin(), E = DiagsInGroup.end(); I != E; ++I) {
    std::string S = I->first;
    EscapeString(S);
    // Group option string.
    OS << "  { \"" << S << "\","
      << std::string(MaxLen-I->first.size()+1, ' ');
    
    // Diagnostics in the group.
    if (I->second.DiagsInGroup.empty())
      OS << "0, ";
    else
      OS << "DiagArray" << IDNo++ << ", ";
    
    // FIXME: Subgroups.
    OS << 0;
    OS << " },\n";
  }
  OS << "#endif // GET_DIAG_TABLE\n\n";
}

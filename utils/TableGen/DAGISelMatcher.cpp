//===- DAGISelMatcher.cpp - Representation of DAG pattern matcher ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DAGISelMatcher.h"
#include "CodeGenDAGPatterns.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

void Matcher::dump() const {
  print(errs(), 0);
}

void Matcher::print(raw_ostream &OS, unsigned indent) const {
  printImpl(OS, indent);
  if (Next)
    return Next->print(OS, indent);
}

void ScopeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Scope\n";
  Check->print(OS, indent+2);
}

void RecordMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "Record\n";
}

void RecordChildMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordChild: " << ChildNo << '\n';
}

void RecordMemRefMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "RecordMemRef\n";
}

void CaptureFlagInputMatcher::printImpl(raw_ostream &OS, unsigned indent) const{
  OS.indent(indent) << "CaptureFlagInput\n";
}

void MoveChildMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveChild " << ChildNo << '\n';
}

void MoveParentMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MoveParent\n";
}

void CheckSameMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckSame " << MatchNumber << '\n';
}

void CheckPatternPredicateMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPatternPredicate " << Predicate << '\n';
}

void CheckPredicateMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckPredicate " << PredName << '\n';
}

void CheckOpcodeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOpcode " << OpcodeName << '\n';
}

void CheckMultiOpcodeMatcher::printImpl(raw_ostream &OS, unsigned indent) const{
  OS.indent(indent) << "CheckMultiOpcode <todo args>\n";
}

void CheckTypeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckType " << getEnumName(Type) << '\n';
}

void CheckChildTypeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckChildType " << ChildNo << " "
    << getEnumName(Type) << '\n';
}


void CheckIntegerMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckInteger " << Value << '\n';
}

void CheckCondCodeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckCondCode ISD::" << CondCodeName << '\n';
}

void CheckValueTypeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckValueType MVT::" << TypeName << '\n';
}

void CheckComplexPatMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckComplexPat " << Pattern.getSelectFunc() << '\n';
}

void CheckAndImmMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckAndImm " << Value << '\n';
}

void CheckOrImmMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CheckOrImm " << Value << '\n';
}

void CheckFoldableChainNodeMatcher::printImpl(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckFoldableChainNode\n";
}

void CheckChainCompatibleMatcher::printImpl(raw_ostream &OS,
                                              unsigned indent) const {
  OS.indent(indent) << "CheckChainCompatible " << PreviousOp << "\n";
}

void EmitIntegerMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitInteger " << Val << " VT=" << VT << '\n';
}

void EmitStringIntegerMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitStringInteger " << Val << " VT=" << VT << '\n';
}

void EmitRegisterMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitRegister ";
  if (Reg)
    OS << Reg->getName();
  else
    OS << "zero_reg";
  OS << " VT=" << VT << '\n';
}

void EmitConvertToTargetMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitConvertToTarget " << Slot << '\n';
}

void EmitMergeInputChainsMatcher::
printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitMergeInputChains <todo: args>\n";
}

void EmitCopyToRegMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitCopyToReg <todo: args>\n";
}

void EmitNodeXFormMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNodeXForm " << NodeXForm->getName()
     << " Slot=" << Slot << '\n';
}


void EmitNodeMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "EmitNode: " << OpcodeName << ": <todo flags> ";

  for (unsigned i = 0, e = VTs.size(); i != e; ++i)
    OS << ' ' << getEnumName(VTs[i]);
  OS << '(';
  for (unsigned i = 0, e = Operands.size(); i != e; ++i)
    OS << Operands[i] << ' ';
  OS << ")\n";
}

void MarkFlagResultsMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "MarkFlagResults <todo: args>\n";
}

void CompleteMatchMatcher::printImpl(raw_ostream &OS, unsigned indent) const {
  OS.indent(indent) << "CompleteMatch <todo args>\n";
  OS.indent(indent) << "Src = " << *Pattern.getSrcPattern() << "\n";
  OS.indent(indent) << "Dst = " << *Pattern.getDstPattern() << "\n";
}

// getHashImpl Implementation.

unsigned CheckPatternPredicateMatcher::getHashImpl() const {
  return HashString(Predicate);
}

unsigned CheckPredicateMatcher::getHashImpl() const {
  return HashString(PredName);
}

unsigned CheckOpcodeMatcher::getHashImpl() const {
  return HashString(OpcodeName);
}

unsigned CheckMultiOpcodeMatcher::getHashImpl() const {
  unsigned Result = 0;
  for (unsigned i = 0, e = OpcodeNames.size(); i != e; ++i)
    Result |= HashString(OpcodeNames[i]);
  return Result;
}

unsigned CheckCondCodeMatcher::getHashImpl() const {
  return HashString(CondCodeName);
}

unsigned CheckValueTypeMatcher::getHashImpl() const {
  return HashString(TypeName);
}

unsigned EmitStringIntegerMatcher::getHashImpl() const {
  return HashString(Val) ^ VT;
}

template<typename It>
static unsigned HashUnsigneds(It I, It E) {
  unsigned Result = 0;
  for (; I != E; ++I)
    Result = (Result<<3) ^ *I;
  return Result;
}

unsigned EmitMergeInputChainsMatcher::getHashImpl() const {
  return HashUnsigneds(ChainNodes.begin(), ChainNodes.end());
}

bool EmitNodeMatcher::isEqualImpl(const Matcher *m) const {
  const EmitNodeMatcher *M = cast<EmitNodeMatcher>(m);
  return M->OpcodeName == OpcodeName && M->VTs == VTs &&
         M->Operands == Operands && M->HasChain == HasChain &&
         M->HasFlag == HasFlag && M->HasMemRefs == HasMemRefs &&
         M->NumFixedArityOperands == NumFixedArityOperands;
}

unsigned EmitNodeMatcher::getHashImpl() const {
  return (HashString(OpcodeName) << 4) | Operands.size();
}


unsigned MarkFlagResultsMatcher::getHashImpl() const {
  return HashUnsigneds(FlagResultNodes.begin(), FlagResultNodes.end());
}

unsigned CompleteMatchMatcher::getHashImpl() const {
  return HashUnsigneds(Results.begin(), Results.end()) ^ 
          ((unsigned)(intptr_t)&Pattern << 8);
}

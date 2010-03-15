//===- CodeGenDAGPatterns.cpp - Read DAG patterns from .td file -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CodeGenDAGPatterns class, which is used to read and
// represent the patterns present in a .td file for instructions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenDAGPatterns.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <set>
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  EEVT::TypeSet Implementation
//===----------------------------------------------------------------------===//

// FIXME: Remove EEVT::isUnknown!

static inline bool isInteger(MVT::SimpleValueType VT) {
  return EVT(VT).isInteger();
}

static inline bool isFloatingPoint(MVT::SimpleValueType VT) {
  return EVT(VT).isFloatingPoint();
}

static inline bool isVector(MVT::SimpleValueType VT) {
  return EVT(VT).isVector();
}

EEVT::TypeSet::TypeSet(MVT::SimpleValueType VT, TreePattern &TP) {
  if (VT == MVT::iAny)
    EnforceInteger(TP);
  else if (VT == MVT::fAny)
    EnforceFloatingPoint(TP);
  else if (VT == MVT::vAny)
    EnforceVector(TP);
  else {
    assert((VT < MVT::LAST_VALUETYPE || VT == MVT::iPTR ||
            VT == MVT::iPTRAny) && "Not a concrete type!");
    TypeVec.push_back(VT);
  }
}


EEVT::TypeSet::TypeSet(const std::vector<MVT::SimpleValueType> &VTList) {
  assert(!VTList.empty() && "empty list?");
  TypeVec.append(VTList.begin(), VTList.end());
  
  if (!VTList.empty())
    assert(VTList[0] != MVT::iAny && VTList[0] != MVT::vAny &&
           VTList[0] != MVT::fAny);
  
  // Remove duplicates.
  array_pod_sort(TypeVec.begin(), TypeVec.end());
  TypeVec.erase(std::unique(TypeVec.begin(), TypeVec.end()), TypeVec.end());
}


/// hasIntegerTypes - Return true if this TypeSet contains iAny or an
/// integer value type.
bool EEVT::TypeSet::hasIntegerTypes() const {
  for (unsigned i = 0, e = TypeVec.size(); i != e; ++i)
    if (isInteger(TypeVec[i]))
      return true;
  return false;
}  

/// hasFloatingPointTypes - Return true if this TypeSet contains an fAny or
/// a floating point value type.
bool EEVT::TypeSet::hasFloatingPointTypes() const {
  for (unsigned i = 0, e = TypeVec.size(); i != e; ++i)
    if (isFloatingPoint(TypeVec[i]))
      return true;
  return false;
}  

/// hasVectorTypes - Return true if this TypeSet contains a vAny or a vector
/// value type.
bool EEVT::TypeSet::hasVectorTypes() const {
  for (unsigned i = 0, e = TypeVec.size(); i != e; ++i)
    if (isVector(TypeVec[i]))
      return true;
  return false;
}


std::string EEVT::TypeSet::getName() const {
  if (TypeVec.empty()) return "isUnknown";
  
  std::string Result;
    
  for (unsigned i = 0, e = TypeVec.size(); i != e; ++i) {
    std::string VTName = llvm::getEnumName(TypeVec[i]);
    // Strip off MVT:: prefix if present.
    if (VTName.substr(0,5) == "MVT::")
      VTName = VTName.substr(5);
    if (i) Result += ':';
    Result += VTName;
  }
  
  if (TypeVec.size() == 1)
    return Result;
  return "{" + Result + "}";
}

/// MergeInTypeInfo - This merges in type information from the specified
/// argument.  If 'this' changes, it returns true.  If the two types are
/// contradictory (e.g. merge f32 into i32) then this throws an exception.
bool EEVT::TypeSet::MergeInTypeInfo(const EEVT::TypeSet &InVT, TreePattern &TP){
  if (InVT.isCompletelyUnknown() || *this == InVT)
    return false;
  
  if (isCompletelyUnknown()) {
    *this = InVT;
    return true;
  }
  
  assert(TypeVec.size() >= 1 && InVT.TypeVec.size() >= 1 && "No unknowns");
  
  // Handle the abstract cases, seeing if we can resolve them better.
  switch (TypeVec[0]) {
  default: break;
  case MVT::iPTR:
  case MVT::iPTRAny:
    if (InVT.hasIntegerTypes()) {
      EEVT::TypeSet InCopy(InVT);
      InCopy.EnforceInteger(TP);
      InCopy.EnforceScalar(TP);
      
      if (InCopy.isConcrete()) {
        // If the RHS has one integer type, upgrade iPTR to i32.
        TypeVec[0] = InVT.TypeVec[0];
        return true;
      }
      
      // If the input has multiple scalar integers, this doesn't add any info.
      if (!InCopy.isCompletelyUnknown())
        return false;
    }
    break;
  }
  
  // If the input constraint is iAny/iPTR and this is an integer type list,
  // remove non-integer types from the list.
  if ((InVT.TypeVec[0] == MVT::iPTR || InVT.TypeVec[0] == MVT::iPTRAny) &&
      hasIntegerTypes()) {
    bool MadeChange = EnforceInteger(TP);
    
    // If we're merging in iPTR/iPTRAny and the node currently has a list of
    // multiple different integer types, replace them with a single iPTR.
    if ((InVT.TypeVec[0] == MVT::iPTR || InVT.TypeVec[0] == MVT::iPTRAny) &&
        TypeVec.size() != 1) {
      TypeVec.resize(1);
      TypeVec[0] = InVT.TypeVec[0];
      MadeChange = true;
    }
    
    return MadeChange;
  }
  
  // If this is a type list and the RHS is a typelist as well, eliminate entries
  // from this list that aren't in the other one.
  bool MadeChange = false;
  TypeSet InputSet(*this);

  for (unsigned i = 0; i != TypeVec.size(); ++i) {
    bool InInVT = false;
    for (unsigned j = 0, e = InVT.TypeVec.size(); j != e; ++j)
      if (TypeVec[i] == InVT.TypeVec[j]) {
        InInVT = true;
        break;
      }
    
    if (InInVT) continue;
    TypeVec.erase(TypeVec.begin()+i--);
    MadeChange = true;
  }
  
  // If we removed all of our types, we have a type contradiction.
  if (!TypeVec.empty())
    return MadeChange;
  
  // FIXME: Really want an SMLoc here!
  TP.error("Type inference contradiction found, merging '" +
           InVT.getName() + "' into '" + InputSet.getName() + "'");
  return true; // unreachable
}

/// EnforceInteger - Remove all non-integer types from this set.
bool EEVT::TypeSet::EnforceInteger(TreePattern &TP) {
  TypeSet InputSet(*this);
  bool MadeChange = false;
  
  // If we know nothing, then get the full set.
  if (TypeVec.empty()) {
    *this = TP.getDAGPatterns().getTargetInfo().getLegalValueTypes();
    MadeChange = true;
  }
  
  if (!hasFloatingPointTypes())
    return MadeChange;
  
  // Filter out all the fp types.
  for (unsigned i = 0; i != TypeVec.size(); ++i)
    if (isFloatingPoint(TypeVec[i]))
      TypeVec.erase(TypeVec.begin()+i--);
  
  if (TypeVec.empty())
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be integer");
  return MadeChange;
}

/// EnforceFloatingPoint - Remove all integer types from this set.
bool EEVT::TypeSet::EnforceFloatingPoint(TreePattern &TP) {
  TypeSet InputSet(*this);
  bool MadeChange = false;
  
  // If we know nothing, then get the full set.
  if (TypeVec.empty()) {
    *this = TP.getDAGPatterns().getTargetInfo().getLegalValueTypes();
    MadeChange = true;
  }
  
  if (!hasIntegerTypes())
    return MadeChange;
  
  // Filter out all the fp types.
  for (unsigned i = 0; i != TypeVec.size(); ++i)
    if (isInteger(TypeVec[i]))
      TypeVec.erase(TypeVec.begin()+i--);
  
  if (TypeVec.empty())
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be floating point");
  return MadeChange;
}

/// EnforceScalar - Remove all vector types from this.
bool EEVT::TypeSet::EnforceScalar(TreePattern &TP) {
  TypeSet InputSet(*this);
  bool MadeChange = false;
  
  // If we know nothing, then get the full set.
  if (TypeVec.empty()) {
    *this = TP.getDAGPatterns().getTargetInfo().getLegalValueTypes();
    MadeChange = true;
  }
  
  if (!hasVectorTypes())
    return MadeChange;
  
  // Filter out all the vector types.
  for (unsigned i = 0; i != TypeVec.size(); ++i)
    if (isVector(TypeVec[i]))
      TypeVec.erase(TypeVec.begin()+i--);
  
  if (TypeVec.empty())
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be scalar");
  return MadeChange;
}

/// EnforceVector - Remove all vector types from this.
bool EEVT::TypeSet::EnforceVector(TreePattern &TP) {
  TypeSet InputSet(*this);
  bool MadeChange = false;
  
  // If we know nothing, then get the full set.
  if (TypeVec.empty()) {
    *this = TP.getDAGPatterns().getTargetInfo().getLegalValueTypes();
    MadeChange = true;
  }
  
  // Filter out all the scalar types.
  for (unsigned i = 0; i != TypeVec.size(); ++i)
    if (!isVector(TypeVec[i]))
      TypeVec.erase(TypeVec.begin()+i--);
  
  if (TypeVec.empty())
    TP.error("Type inference contradiction found, '" +
             InputSet.getName() + "' needs to be a vector");
  return MadeChange;
}


/// EnforceSmallerThan - 'this' must be a smaller VT than Other.  Update
/// this an other based on this information.
bool EEVT::TypeSet::EnforceSmallerThan(EEVT::TypeSet &Other, TreePattern &TP) {
  // Both operands must be integer or FP, but we don't care which.
  bool MadeChange = false;
  
  // This code does not currently handle nodes which have multiple types,
  // where some types are integer, and some are fp.  Assert that this is not
  // the case.
  assert(!(hasIntegerTypes() && hasFloatingPointTypes()) &&
         !(Other.hasIntegerTypes() && Other.hasFloatingPointTypes()) &&
         "SDTCisOpSmallerThanOp does not handle mixed int/fp types!");
  // If one side is known to be integer or known to be FP but the other side has
  // no information, get at least the type integrality info in there.
  if (hasIntegerTypes())
    MadeChange |= Other.EnforceInteger(TP);
  else if (hasFloatingPointTypes())
    MadeChange |= Other.EnforceFloatingPoint(TP);
  if (Other.hasIntegerTypes())
    MadeChange |= EnforceInteger(TP);
  else if (Other.hasFloatingPointTypes())
    MadeChange |= EnforceFloatingPoint(TP);
  
  assert(!isCompletelyUnknown() && !Other.isCompletelyUnknown() &&
         "Should have a type list now");
  
  // If one contains vectors but the other doesn't pull vectors out.
  if (!hasVectorTypes() && Other.hasVectorTypes())
    MadeChange |= Other.EnforceScalar(TP);
  if (hasVectorTypes() && !Other.hasVectorTypes())
    MadeChange |= EnforceScalar(TP);
  
  // FIXME: This is a bone-headed way to do this.
  
  // Get the set of legal VTs and filter it based on the known integrality.
  const CodeGenTarget &CGT = TP.getDAGPatterns().getTargetInfo();
  TypeSet LegalVTs = CGT.getLegalValueTypes();

  // TODO: If one or the other side is known to be a specific VT, we could prune
  // LegalVTs.
  if (hasIntegerTypes())
    LegalVTs.EnforceInteger(TP);
  else if (hasFloatingPointTypes())
    LegalVTs.EnforceFloatingPoint(TP);
  else
    return MadeChange;
  
  switch (LegalVTs.TypeVec.size()) {
  case 0: assert(0 && "No legal VTs?");
  default:         // Too many VT's to pick from.
    // TODO: If the biggest type in LegalVTs is in this set, we could remove it.
    // If one or the other side is known to be a specific VT, we could prune
    // LegalVTs.
    return MadeChange;
  case 1: 
    // Only one VT of this flavor.  Cannot ever satisfy the constraints.
    return MergeInTypeInfo(MVT::Other, TP);  // throw
  case 2:
    // If we have exactly two possible types, the little operand must be the
    // small one, the big operand should be the big one.  This is common with 
    // float/double for example.
    assert(LegalVTs.TypeVec[0] < LegalVTs.TypeVec[1] && "Should be sorted!");
    MadeChange |= MergeInTypeInfo(LegalVTs.TypeVec[0], TP);
    MadeChange |= Other.MergeInTypeInfo(LegalVTs.TypeVec[1], TP);
    return MadeChange;
  }    
}

/// EnforceVectorEltTypeIs - 'this' is now constrainted to be a vector type
/// whose element is VT.
bool EEVT::TypeSet::EnforceVectorEltTypeIs(MVT::SimpleValueType VT,
                                           TreePattern &TP) {
  TypeSet InputSet(*this);
  bool MadeChange = false;
  
  // If we know nothing, then get the full set.
  if (TypeVec.empty()) {
    *this = TP.getDAGPatterns().getTargetInfo().getLegalValueTypes();
    MadeChange = true;
  }
  
  // Filter out all the non-vector types and types which don't have the right
  // element type.
  for (unsigned i = 0; i != TypeVec.size(); ++i)
    if (!isVector(TypeVec[i]) ||
        EVT(TypeVec[i]).getVectorElementType().getSimpleVT().SimpleTy != VT) {
      TypeVec.erase(TypeVec.begin()+i--);
      MadeChange = true;
    }
  
  if (TypeVec.empty())  // FIXME: Really want an SMLoc here!
    TP.error("Type inference contradiction found, forcing '" +
             InputSet.getName() + "' to have a vector element");
  return MadeChange;
}

//===----------------------------------------------------------------------===//
// Helpers for working with extended types.

bool RecordPtrCmp::operator()(const Record *LHS, const Record *RHS) const {
  return LHS->getID() < RHS->getID();
}

/// Dependent variable map for CodeGenDAGPattern variant generation
typedef std::map<std::string, int> DepVarMap;

/// Const iterator shorthand for DepVarMap
typedef DepVarMap::const_iterator DepVarMap_citer;

namespace {
void FindDepVarsOf(TreePatternNode *N, DepVarMap &DepMap) {
  if (N->isLeaf()) {
    if (dynamic_cast<DefInit*>(N->getLeafValue()) != NULL) {
      DepMap[N->getName()]++;
    }
  } else {
    for (size_t i = 0, e = N->getNumChildren(); i != e; ++i)
      FindDepVarsOf(N->getChild(i), DepMap);
  }
}

//! Find dependent variables within child patterns
/*!
 */
void FindDepVars(TreePatternNode *N, MultipleUseVarSet &DepVars) {
  DepVarMap depcounts;
  FindDepVarsOf(N, depcounts);
  for (DepVarMap_citer i = depcounts.begin(); i != depcounts.end(); ++i) {
    if (i->second > 1) {            // std::pair<std::string, int>
      DepVars.insert(i->first);
    }
  }
}

//! Dump the dependent variable set:
void DumpDepVars(MultipleUseVarSet &DepVars) {
  if (DepVars.empty()) {
    DEBUG(errs() << "<empty set>");
  } else {
    DEBUG(errs() << "[ ");
    for (MultipleUseVarSet::const_iterator i = DepVars.begin(), e = DepVars.end();
         i != e; ++i) {
      DEBUG(errs() << (*i) << " ");
    }
    DEBUG(errs() << "]");
  }
}
}

//===----------------------------------------------------------------------===//
// PatternToMatch implementation
//

/// getPredicateCheck - Return a single string containing all of this
/// pattern's predicates concatenated with "&&" operators.
///
std::string PatternToMatch::getPredicateCheck() const {
  std::string PredicateCheck;
  for (unsigned i = 0, e = Predicates->getSize(); i != e; ++i) {
    if (DefInit *Pred = dynamic_cast<DefInit*>(Predicates->getElement(i))) {
      Record *Def = Pred->getDef();
      if (!Def->isSubClassOf("Predicate")) {
#ifndef NDEBUG
        Def->dump();
#endif
        assert(0 && "Unknown predicate type!");
      }
      if (!PredicateCheck.empty())
        PredicateCheck += " && ";
      PredicateCheck += "(" + Def->getValueAsString("CondString") + ")";
    }
  }

  return PredicateCheck;
}

//===----------------------------------------------------------------------===//
// SDTypeConstraint implementation
//

SDTypeConstraint::SDTypeConstraint(Record *R) {
  OperandNo = R->getValueAsInt("OperandNum");
  
  if (R->isSubClassOf("SDTCisVT")) {
    ConstraintType = SDTCisVT;
    x.SDTCisVT_Info.VT = getValueType(R->getValueAsDef("VT"));
  } else if (R->isSubClassOf("SDTCisPtrTy")) {
    ConstraintType = SDTCisPtrTy;
  } else if (R->isSubClassOf("SDTCisInt")) {
    ConstraintType = SDTCisInt;
  } else if (R->isSubClassOf("SDTCisFP")) {
    ConstraintType = SDTCisFP;
  } else if (R->isSubClassOf("SDTCisVec")) {
    ConstraintType = SDTCisVec;
  } else if (R->isSubClassOf("SDTCisSameAs")) {
    ConstraintType = SDTCisSameAs;
    x.SDTCisSameAs_Info.OtherOperandNum = R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisVTSmallerThanOp")) {
    ConstraintType = SDTCisVTSmallerThanOp;
    x.SDTCisVTSmallerThanOp_Info.OtherOperandNum = 
      R->getValueAsInt("OtherOperandNum");
  } else if (R->isSubClassOf("SDTCisOpSmallerThanOp")) {
    ConstraintType = SDTCisOpSmallerThanOp;
    x.SDTCisOpSmallerThanOp_Info.BigOperandNum = 
      R->getValueAsInt("BigOperandNum");
  } else if (R->isSubClassOf("SDTCisEltOfVec")) {
    ConstraintType = SDTCisEltOfVec;
    x.SDTCisEltOfVec_Info.OtherOperandNum = R->getValueAsInt("OtherOpNum");
  } else {
    errs() << "Unrecognized SDTypeConstraint '" << R->getName() << "'!\n";
    exit(1);
  }
}

/// getOperandNum - Return the node corresponding to operand #OpNo in tree
/// N, which has NumResults results.
TreePatternNode *SDTypeConstraint::getOperandNum(unsigned OpNo,
                                                 TreePatternNode *N,
                                                 unsigned NumResults) const {
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  
  if (OpNo >= (NumResults + N->getNumChildren())) {
    errs() << "Invalid operand number " << OpNo << " ";
    N->dump();
    errs() << '\n';
    exit(1);
  }

  if (OpNo < NumResults)
    return N;  // FIXME: need value #
  else
    return N->getChild(OpNo-NumResults);
}

/// ApplyTypeConstraint - Given a node in a pattern, apply this type
/// constraint to the nodes operands.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool SDTypeConstraint::ApplyTypeConstraint(TreePatternNode *N,
                                           const SDNodeInfo &NodeInfo,
                                           TreePattern &TP) const {
  unsigned NumResults = NodeInfo.getNumResults();
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  
  // Check that the number of operands is sane.  Negative operands -> varargs.
  if (NodeInfo.getNumOperands() >= 0) {
    if (N->getNumChildren() != (unsigned)NodeInfo.getNumOperands())
      TP.error(N->getOperator()->getName() + " node requires exactly " +
               itostr(NodeInfo.getNumOperands()) + " operands!");
  }

  TreePatternNode *NodeToApply = getOperandNum(OperandNo, N, NumResults);
  
  switch (ConstraintType) {
  default: assert(0 && "Unknown constraint type!");
  case SDTCisVT:
    // Operand must be a particular type.
    return NodeToApply->UpdateNodeType(x.SDTCisVT_Info.VT, TP);
  case SDTCisPtrTy:
    // Operand must be same as target pointer type.
    return NodeToApply->UpdateNodeType(MVT::iPTR, TP);
  case SDTCisInt:
    // Require it to be one of the legal integer VTs.
    return NodeToApply->getExtType().EnforceInteger(TP);
  case SDTCisFP:
    // Require it to be one of the legal fp VTs.
    return NodeToApply->getExtType().EnforceFloatingPoint(TP);
  case SDTCisVec:
    // Require it to be one of the legal vector VTs.
    return NodeToApply->getExtType().EnforceVector(TP);
  case SDTCisSameAs: {
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisSameAs_Info.OtherOperandNum, N, NumResults);
    return NodeToApply->UpdateNodeType(OtherNode->getExtType(), TP) |
           OtherNode->UpdateNodeType(NodeToApply->getExtType(), TP);
  }
  case SDTCisVTSmallerThanOp: {
    // The NodeToApply must be a leaf node that is a VT.  OtherOperandNum must
    // have an integer type that is smaller than the VT.
    if (!NodeToApply->isLeaf() ||
        !dynamic_cast<DefInit*>(NodeToApply->getLeafValue()) ||
        !static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef()
               ->isSubClassOf("ValueType"))
      TP.error(N->getOperator()->getName() + " expects a VT operand!");
    MVT::SimpleValueType VT =
     getValueType(static_cast<DefInit*>(NodeToApply->getLeafValue())->getDef());
    if (!isInteger(VT))
      TP.error(N->getOperator()->getName() + " VT operand must be integer!");
    
    TreePatternNode *OtherNode =
      getOperandNum(x.SDTCisVTSmallerThanOp_Info.OtherOperandNum, N,NumResults);
    
    // It must be integer.
    bool MadeChange = OtherNode->getExtType().EnforceInteger(TP);

    // This doesn't try to enforce any information on the OtherNode, it just
    // validates it when information is determined.
    if (OtherNode->hasTypeSet() && OtherNode->getType() <= VT)
      OtherNode->UpdateNodeType(MVT::Other, TP);  // Throw an error.
    return MadeChange;
  }
  case SDTCisOpSmallerThanOp: {
    TreePatternNode *BigOperand =
      getOperandNum(x.SDTCisOpSmallerThanOp_Info.BigOperandNum, N, NumResults);
    return NodeToApply->getExtType().
                  EnforceSmallerThan(BigOperand->getExtType(), TP);
  }
  case SDTCisEltOfVec: {
    TreePatternNode *VecOperand =
      getOperandNum(x.SDTCisEltOfVec_Info.OtherOperandNum, N, NumResults);
    if (VecOperand->hasTypeSet()) {
      if (!isVector(VecOperand->getType()))
        TP.error(N->getOperator()->getName() + " VT operand must be a vector!");
      EVT IVT = VecOperand->getType();
      IVT = IVT.getVectorElementType();
      return NodeToApply->UpdateNodeType(IVT.getSimpleVT().SimpleTy, TP);
    }
    
    if (NodeToApply->hasTypeSet() && VecOperand->getExtType().hasVectorTypes()){
      // Filter vector types out of VecOperand that don't have the right element
      // type.
      return VecOperand->getExtType().
        EnforceVectorEltTypeIs(NodeToApply->getType(), TP);
    }
    return false;
  }
  }  
  return false;
}

//===----------------------------------------------------------------------===//
// SDNodeInfo implementation
//
SDNodeInfo::SDNodeInfo(Record *R) : Def(R) {
  EnumName    = R->getValueAsString("Opcode");
  SDClassName = R->getValueAsString("SDClass");
  Record *TypeProfile = R->getValueAsDef("TypeProfile");
  NumResults = TypeProfile->getValueAsInt("NumResults");
  NumOperands = TypeProfile->getValueAsInt("NumOperands");
  
  // Parse the properties.
  Properties = 0;
  std::vector<Record*> PropList = R->getValueAsListOfDefs("Properties");
  for (unsigned i = 0, e = PropList.size(); i != e; ++i) {
    if (PropList[i]->getName() == "SDNPCommutative") {
      Properties |= 1 << SDNPCommutative;
    } else if (PropList[i]->getName() == "SDNPAssociative") {
      Properties |= 1 << SDNPAssociative;
    } else if (PropList[i]->getName() == "SDNPHasChain") {
      Properties |= 1 << SDNPHasChain;
    } else if (PropList[i]->getName() == "SDNPOutFlag") {
      Properties |= 1 << SDNPOutFlag;
    } else if (PropList[i]->getName() == "SDNPInFlag") {
      Properties |= 1 << SDNPInFlag;
    } else if (PropList[i]->getName() == "SDNPOptInFlag") {
      Properties |= 1 << SDNPOptInFlag;
    } else if (PropList[i]->getName() == "SDNPMayStore") {
      Properties |= 1 << SDNPMayStore;
    } else if (PropList[i]->getName() == "SDNPMayLoad") {
      Properties |= 1 << SDNPMayLoad;
    } else if (PropList[i]->getName() == "SDNPSideEffect") {
      Properties |= 1 << SDNPSideEffect;
    } else if (PropList[i]->getName() == "SDNPMemOperand") {
      Properties |= 1 << SDNPMemOperand;
    } else {
      errs() << "Unknown SD Node property '" << PropList[i]->getName()
             << "' on node '" << R->getName() << "'!\n";
      exit(1);
    }
  }
  
  
  // Parse the type constraints.
  std::vector<Record*> ConstraintList =
    TypeProfile->getValueAsListOfDefs("Constraints");
  TypeConstraints.assign(ConstraintList.begin(), ConstraintList.end());
}

/// getKnownType - If the type constraints on this node imply a fixed type
/// (e.g. all stores return void, etc), then return it as an
/// MVT::SimpleValueType.  Otherwise, return EEVT::isUnknown.
unsigned SDNodeInfo::getKnownType() const {
  unsigned NumResults = getNumResults();
  assert(NumResults <= 1 &&
         "We only work with nodes with zero or one result so far!");
  
  for (unsigned i = 0, e = TypeConstraints.size(); i != e; ++i) {
    // Make sure that this applies to the correct node result.
    if (TypeConstraints[i].OperandNo >= NumResults)  // FIXME: need value #
      continue;
    
    switch (TypeConstraints[i].ConstraintType) {
    default: break;
    case SDTypeConstraint::SDTCisVT:
      return TypeConstraints[i].x.SDTCisVT_Info.VT;
    case SDTypeConstraint::SDTCisPtrTy:
      return MVT::iPTR;
    }
  }
  return EEVT::isUnknown;
}

//===----------------------------------------------------------------------===//
// TreePatternNode implementation
//

TreePatternNode::~TreePatternNode() {
#if 0 // FIXME: implement refcounted tree nodes!
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    delete getChild(i);
#endif
}



void TreePatternNode::print(raw_ostream &OS) const {
  if (isLeaf()) {
    OS << *getLeafValue();
  } else {
    OS << '(' << getOperator()->getName();
  }
  
  if (!isTypeCompletelyUnknown())
    OS << ':' << getExtType().getName();

  if (!isLeaf()) {
    if (getNumChildren() != 0) {
      OS << " ";
      getChild(0)->print(OS);
      for (unsigned i = 1, e = getNumChildren(); i != e; ++i) {
        OS << ", ";
        getChild(i)->print(OS);
      }
    }
    OS << ")";
  }
  
  for (unsigned i = 0, e = PredicateFns.size(); i != e; ++i)
    OS << "<<P:" << PredicateFns[i] << ">>";
  if (TransformFn)
    OS << "<<X:" << TransformFn->getName() << ">>";
  if (!getName().empty())
    OS << ":$" << getName();

}
void TreePatternNode::dump() const {
  print(errs());
}

/// isIsomorphicTo - Return true if this node is recursively
/// isomorphic to the specified node.  For this comparison, the node's
/// entire state is considered. The assigned name is ignored, since
/// nodes with differing names are considered isomorphic. However, if
/// the assigned name is present in the dependent variable set, then
/// the assigned name is considered significant and the node is
/// isomorphic if the names match.
bool TreePatternNode::isIsomorphicTo(const TreePatternNode *N,
                                     const MultipleUseVarSet &DepVars) const {
  if (N == this) return true;
  if (N->isLeaf() != isLeaf() || getExtType() != N->getExtType() ||
      getPredicateFns() != N->getPredicateFns() ||
      getTransformFn() != N->getTransformFn())
    return false;

  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue())) {
      if (DefInit *NDI = dynamic_cast<DefInit*>(N->getLeafValue())) {
        return ((DI->getDef() == NDI->getDef())
                && (DepVars.find(getName()) == DepVars.end()
                    || getName() == N->getName()));
      }
    }
    return getLeafValue() == N->getLeafValue();
  }
  
  if (N->getOperator() != getOperator() ||
      N->getNumChildren() != getNumChildren()) return false;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->isIsomorphicTo(N->getChild(i), DepVars))
      return false;
  return true;
}

/// clone - Make a copy of this tree and all of its children.
///
TreePatternNode *TreePatternNode::clone() const {
  TreePatternNode *New;
  if (isLeaf()) {
    New = new TreePatternNode(getLeafValue());
  } else {
    std::vector<TreePatternNode*> CChildren;
    CChildren.reserve(Children.size());
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      CChildren.push_back(getChild(i)->clone());
    New = new TreePatternNode(getOperator(), CChildren);
  }
  New->setName(getName());
  New->setType(getExtType());
  New->setPredicateFns(getPredicateFns());
  New->setTransformFn(getTransformFn());
  return New;
}

/// RemoveAllTypes - Recursively strip all the types of this tree.
void TreePatternNode::RemoveAllTypes() {
  setType(EEVT::TypeSet());  // Reset to unknown type.
  if (isLeaf()) return;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    getChild(i)->RemoveAllTypes();
}


/// SubstituteFormalArguments - Replace the formal arguments in this tree
/// with actual values specified by ArgMap.
void TreePatternNode::
SubstituteFormalArguments(std::map<std::string, TreePatternNode*> &ArgMap) {
  if (isLeaf()) return;
  
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
    TreePatternNode *Child = getChild(i);
    if (Child->isLeaf()) {
      Init *Val = Child->getLeafValue();
      if (dynamic_cast<DefInit*>(Val) &&
          static_cast<DefInit*>(Val)->getDef()->getName() == "node") {
        // We found a use of a formal argument, replace it with its value.
        TreePatternNode *NewChild = ArgMap[Child->getName()];
        assert(NewChild && "Couldn't find formal argument!");
        assert((Child->getPredicateFns().empty() ||
                NewChild->getPredicateFns() == Child->getPredicateFns()) &&
               "Non-empty child predicate clobbered!");
        setChild(i, NewChild);
      }
    } else {
      getChild(i)->SubstituteFormalArguments(ArgMap);
    }
  }
}


/// InlinePatternFragments - If this pattern refers to any pattern
/// fragments, inline them into place, giving us a pattern without any
/// PatFrag references.
TreePatternNode *TreePatternNode::InlinePatternFragments(TreePattern &TP) {
  if (isLeaf()) return this;  // nothing to do.
  Record *Op = getOperator();
  
  if (!Op->isSubClassOf("PatFrag")) {
    // Just recursively inline children nodes.
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i) {
      TreePatternNode *Child = getChild(i);
      TreePatternNode *NewChild = Child->InlinePatternFragments(TP);

      assert((Child->getPredicateFns().empty() ||
              NewChild->getPredicateFns() == Child->getPredicateFns()) &&
             "Non-empty child predicate clobbered!");

      setChild(i, NewChild);
    }
    return this;
  }

  // Otherwise, we found a reference to a fragment.  First, look up its
  // TreePattern record.
  TreePattern *Frag = TP.getDAGPatterns().getPatternFragment(Op);
  
  // Verify that we are passing the right number of operands.
  if (Frag->getNumArgs() != Children.size())
    TP.error("'" + Op->getName() + "' fragment requires " +
             utostr(Frag->getNumArgs()) + " operands!");

  TreePatternNode *FragTree = Frag->getOnlyTree()->clone();

  std::string Code = Op->getValueAsCode("Predicate");
  if (!Code.empty())
    FragTree->addPredicateFn("Predicate_"+Op->getName());

  // Resolve formal arguments to their actual value.
  if (Frag->getNumArgs()) {
    // Compute the map of formal to actual arguments.
    std::map<std::string, TreePatternNode*> ArgMap;
    for (unsigned i = 0, e = Frag->getNumArgs(); i != e; ++i)
      ArgMap[Frag->getArgName(i)] = getChild(i)->InlinePatternFragments(TP);
  
    FragTree->SubstituteFormalArguments(ArgMap);
  }
  
  FragTree->setName(getName());
  FragTree->UpdateNodeType(getExtType(), TP);

  // Transfer in the old predicates.
  for (unsigned i = 0, e = getPredicateFns().size(); i != e; ++i)
    FragTree->addPredicateFn(getPredicateFns()[i]);

  // Get a new copy of this fragment to stitch into here.
  //delete this;    // FIXME: implement refcounting!
  
  // The fragment we inlined could have recursive inlining that is needed.  See
  // if there are any pattern fragments in it and inline them as needed.
  return FragTree->InlinePatternFragments(TP);
}

/// getImplicitType - Check to see if the specified record has an implicit
/// type which should be applied to it.  This will infer the type of register
/// references from the register file information, for example.
///
static EEVT::TypeSet getImplicitType(Record *R, bool NotRegisters,
                                     TreePattern &TP) {
  // Check to see if this is a register or a register class.
  if (R->isSubClassOf("RegisterClass")) {
    if (NotRegisters) 
      return EEVT::TypeSet(); // Unknown.
    const CodeGenTarget &T = TP.getDAGPatterns().getTargetInfo();
    return EEVT::TypeSet(T.getRegisterClass(R).getValueTypes());
  } else if (R->isSubClassOf("PatFrag")) {
    // Pattern fragment types will be resolved when they are inlined.
    return EEVT::TypeSet(); // Unknown.
  } else if (R->isSubClassOf("Register")) {
    if (NotRegisters) 
      return EEVT::TypeSet(); // Unknown.
    const CodeGenTarget &T = TP.getDAGPatterns().getTargetInfo();
    return EEVT::TypeSet(T.getRegisterVTs(R));
  } else if (R->isSubClassOf("ValueType") || R->isSubClassOf("CondCode")) {
    // Using a VTSDNode or CondCodeSDNode.
    return EEVT::TypeSet(MVT::Other, TP);
  } else if (R->isSubClassOf("ComplexPattern")) {
    if (NotRegisters) 
      return EEVT::TypeSet(); // Unknown.
   return EEVT::TypeSet(TP.getDAGPatterns().getComplexPattern(R).getValueType(),
                         TP);
  } else if (R->isSubClassOf("PointerLikeRegClass")) {
    return EEVT::TypeSet(MVT::iPTR, TP);
  } else if (R->getName() == "node" || R->getName() == "srcvalue" ||
             R->getName() == "zero_reg") {
    // Placeholder.
    return EEVT::TypeSet(); // Unknown.
  }
  
  TP.error("Unknown node flavor used in pattern: " + R->getName());
  return EEVT::TypeSet(MVT::Other, TP);
}


/// getIntrinsicInfo - If this node corresponds to an intrinsic, return the
/// CodeGenIntrinsic information for it, otherwise return a null pointer.
const CodeGenIntrinsic *TreePatternNode::
getIntrinsicInfo(const CodeGenDAGPatterns &CDP) const {
  if (getOperator() != CDP.get_intrinsic_void_sdnode() &&
      getOperator() != CDP.get_intrinsic_w_chain_sdnode() &&
      getOperator() != CDP.get_intrinsic_wo_chain_sdnode())
    return 0;
    
  unsigned IID = 
    dynamic_cast<IntInit*>(getChild(0)->getLeafValue())->getValue();
  return &CDP.getIntrinsicInfo(IID);
}

/// getComplexPatternInfo - If this node corresponds to a ComplexPattern,
/// return the ComplexPattern information, otherwise return null.
const ComplexPattern *
TreePatternNode::getComplexPatternInfo(const CodeGenDAGPatterns &CGP) const {
  if (!isLeaf()) return 0;
  
  DefInit *DI = dynamic_cast<DefInit*>(getLeafValue());
  if (DI && DI->getDef()->isSubClassOf("ComplexPattern"))
    return &CGP.getComplexPattern(DI->getDef());
  return 0;
}

/// NodeHasProperty - Return true if this node has the specified property.
bool TreePatternNode::NodeHasProperty(SDNP Property,
                                      const CodeGenDAGPatterns &CGP) const {
  if (isLeaf()) {
    if (const ComplexPattern *CP = getComplexPatternInfo(CGP))
      return CP->hasProperty(Property);
    return false;
  }
  
  Record *Operator = getOperator();
  if (!Operator->isSubClassOf("SDNode")) return false;
  
  return CGP.getSDNodeInfo(Operator).hasProperty(Property);
}




/// TreeHasProperty - Return true if any node in this tree has the specified
/// property.
bool TreePatternNode::TreeHasProperty(SDNP Property,
                                      const CodeGenDAGPatterns &CGP) const {
  if (NodeHasProperty(Property, CGP))
    return true;
  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (getChild(i)->TreeHasProperty(Property, CGP))
      return true;
  return false;
}  

/// isCommutativeIntrinsic - Return true if the node corresponds to a
/// commutative intrinsic.
bool
TreePatternNode::isCommutativeIntrinsic(const CodeGenDAGPatterns &CDP) const {
  if (const CodeGenIntrinsic *Int = getIntrinsicInfo(CDP))
    return Int->isCommutative;
  return false;
}


/// ApplyTypeConstraints - Apply all of the type constraints relevant to
/// this node and its children in the tree.  This returns true if it makes a
/// change, false otherwise.  If a type contradiction is found, throw an
/// exception.
bool TreePatternNode::ApplyTypeConstraints(TreePattern &TP, bool NotRegisters) {
  CodeGenDAGPatterns &CDP = TP.getDAGPatterns();
  if (isLeaf()) {
    if (DefInit *DI = dynamic_cast<DefInit*>(getLeafValue())) {
      // If it's a regclass or something else known, include the type.
      return UpdateNodeType(getImplicitType(DI->getDef(), NotRegisters, TP),TP);
    }
    
    if (IntInit *II = dynamic_cast<IntInit*>(getLeafValue())) {
      // Int inits are always integers. :)
      bool MadeChange = Type.EnforceInteger(TP);
      
      if (!hasTypeSet())
        return MadeChange;
      
      MVT::SimpleValueType VT = getType();
      if (VT == MVT::iPTR || VT == MVT::iPTRAny)
        return MadeChange;
      
      unsigned Size = EVT(VT).getSizeInBits();
      // Make sure that the value is representable for this type.
      if (Size >= 32) return MadeChange;
      
      int Val = (II->getValue() << (32-Size)) >> (32-Size);
      if (Val == II->getValue()) return MadeChange;
      
      // If sign-extended doesn't fit, does it fit as unsigned?
      unsigned ValueMask;
      unsigned UnsignedVal;
      ValueMask = unsigned(~uint32_t(0UL) >> (32-Size));
      UnsignedVal = unsigned(II->getValue());

      if ((ValueMask & UnsignedVal) == UnsignedVal)
        return MadeChange;
      
      TP.error("Integer value '" + itostr(II->getValue())+
               "' is out of range for type '" + getEnumName(getType()) + "'!");
      return MadeChange;
    }
    return false;
  }
  
  // special handling for set, which isn't really an SDNode.
  if (getOperator()->getName() == "set") {
    assert (getNumChildren() >= 2 && "Missing RHS of a set?");
    unsigned NC = getNumChildren();
    bool MadeChange = false;
    for (unsigned i = 0; i < NC-1; ++i) {
      MadeChange = getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
      MadeChange |= getChild(NC-1)->ApplyTypeConstraints(TP, NotRegisters);
    
      // Types of operands must match.
      MadeChange |=getChild(i)->UpdateNodeType(getChild(NC-1)->getExtType(),TP);
      MadeChange |=getChild(NC-1)->UpdateNodeType(getChild(i)->getExtType(),TP);
      MadeChange |=UpdateNodeType(MVT::isVoid, TP);
    }
    return MadeChange;
  }
  
  if (getOperator()->getName() == "implicit" ||
      getOperator()->getName() == "parallel") {
    bool MadeChange = false;
    for (unsigned i = 0; i < getNumChildren(); ++i)
      MadeChange = getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    return MadeChange;
  }
  
  if (getOperator()->getName() == "COPY_TO_REGCLASS") {
    bool MadeChange = false;
    MadeChange |= getChild(0)->ApplyTypeConstraints(TP, NotRegisters);
    MadeChange |= getChild(1)->ApplyTypeConstraints(TP, NotRegisters);
    
    // child #1 of COPY_TO_REGCLASS should be a register class.  We don't care
    // what type it gets, so if it didn't get a concrete type just give it the
    // first viable type from the reg class.
    if (!getChild(1)->hasTypeSet() &&
        !getChild(1)->getExtType().isCompletelyUnknown()) {
      MVT::SimpleValueType RCVT = getChild(1)->getExtType().getTypeList()[0];
      MadeChange |= getChild(1)->UpdateNodeType(RCVT, TP);
    }
    return MadeChange;
  }
  
  if (const CodeGenIntrinsic *Int = getIntrinsicInfo(CDP)) {
    bool MadeChange = false;

    // Apply the result type to the node.
    unsigned NumRetVTs = Int->IS.RetVTs.size();
    unsigned NumParamVTs = Int->IS.ParamVTs.size();

    for (unsigned i = 0, e = NumRetVTs; i != e; ++i)
      MadeChange |= UpdateNodeType(Int->IS.RetVTs[i], TP);

    if (getNumChildren() != NumParamVTs + NumRetVTs)
      TP.error("Intrinsic '" + Int->Name + "' expects " +
               utostr(NumParamVTs + NumRetVTs - 1) + " operands, not " +
               utostr(getNumChildren() - 1) + " operands!");

    // Apply type info to the intrinsic ID.
    MadeChange |= getChild(0)->UpdateNodeType(MVT::iPTR, TP);
    
    for (unsigned i = NumRetVTs, e = getNumChildren(); i != e; ++i) {
      MVT::SimpleValueType OpVT = Int->IS.ParamVTs[i - NumRetVTs];
      MadeChange |= getChild(i)->UpdateNodeType(OpVT, TP);
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    }
    return MadeChange;
  }
  
  if (getOperator()->isSubClassOf("SDNode")) {
    const SDNodeInfo &NI = CDP.getSDNodeInfo(getOperator());
    
    bool MadeChange = NI.ApplyTypeConstraints(this, TP);
    for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
      MadeChange |= getChild(i)->ApplyTypeConstraints(TP, NotRegisters);
    // Branch, etc. do not produce results and top-level forms in instr pattern
    // must have void types.
    if (NI.getNumResults() == 0)
      MadeChange |= UpdateNodeType(MVT::isVoid, TP);
    
    return MadeChange;  
  }
  
  if (getOperator()->isSubClassOf("Instruction")) {
    const DAGInstruction &Inst = CDP.getInstruction(getOperator());
    bool MadeChange = false;
    unsigned NumResults = Inst.getNumResults();
    
    assert(NumResults <= 1 &&
           "Only supports zero or one result instrs!");

    CodeGenInstruction &InstInfo =
      CDP.getTargetInfo().getInstruction(getOperator()->getName());
    // Apply the result type to the node
    if (NumResults == 0 || InstInfo.NumDefs == 0) {
      MadeChange = UpdateNodeType(MVT::isVoid, TP);
    } else {
      Record *ResultNode = Inst.getResult(0);
      
      if (ResultNode->isSubClassOf("PointerLikeRegClass")) {
        MadeChange = UpdateNodeType(MVT::iPTR, TP);
      } else if (ResultNode->getName() == "unknown") {
        // Nothing to do.
      } else {
        assert(ResultNode->isSubClassOf("RegisterClass") &&
               "Operands should be register classes!");

        const CodeGenRegisterClass &RC = 
          CDP.getTargetInfo().getRegisterClass(ResultNode);
        MadeChange = UpdateNodeType(RC.getValueTypes(), TP);
      }
    }
    
    // If this is an INSERT_SUBREG, constrain the source and destination VTs to
    // be the same.
    if (getOperator()->getName() == "INSERT_SUBREG") {
      MadeChange |= UpdateNodeType(getChild(0)->getExtType(), TP);
      MadeChange |= getChild(0)->UpdateNodeType(getExtType(), TP);
    }
    

    unsigned ChildNo = 0;
    for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i) {
      Record *OperandNode = Inst.getOperand(i);
      
      // If the instruction expects a predicate or optional def operand, we
      // codegen this by setting the operand to it's default value if it has a
      // non-empty DefaultOps field.
      if ((OperandNode->isSubClassOf("PredicateOperand") ||
           OperandNode->isSubClassOf("OptionalDefOperand")) &&
          !CDP.getDefaultOperand(OperandNode).DefaultOps.empty())
        continue;
       
      // Verify that we didn't run out of provided operands.
      if (ChildNo >= getNumChildren())
        TP.error("Instruction '" + getOperator()->getName() +
                 "' expects more operands than were provided.");
      
      MVT::SimpleValueType VT;
      TreePatternNode *Child = getChild(ChildNo++);
      if (OperandNode->isSubClassOf("RegisterClass")) {
        const CodeGenRegisterClass &RC = 
          CDP.getTargetInfo().getRegisterClass(OperandNode);
        MadeChange |= Child->UpdateNodeType(RC.getValueTypes(), TP);
      } else if (OperandNode->isSubClassOf("Operand")) {
        VT = getValueType(OperandNode->getValueAsDef("Type"));
        MadeChange |= Child->UpdateNodeType(VT, TP);
      } else if (OperandNode->isSubClassOf("PointerLikeRegClass")) {
        MadeChange |= Child->UpdateNodeType(MVT::iPTR, TP);
      } else if (OperandNode->getName() == "unknown") {
        // Nothing to do.
      } else {
        assert(0 && "Unknown operand type!");
        abort();
      }
      MadeChange |= Child->ApplyTypeConstraints(TP, NotRegisters);
    }

    if (ChildNo != getNumChildren())
      TP.error("Instruction '" + getOperator()->getName() +
               "' was provided too many operands!");
    
    return MadeChange;
  }
  
  assert(getOperator()->isSubClassOf("SDNodeXForm") && "Unknown node type!");
  
  // Node transforms always take one operand.
  if (getNumChildren() != 1)
    TP.error("Node transform '" + getOperator()->getName() +
             "' requires one operand!");

  bool MadeChange = getChild(0)->ApplyTypeConstraints(TP, NotRegisters);

  
  // If either the output or input of the xform does not have exact
  // type info. We assume they must be the same. Otherwise, it is perfectly
  // legal to transform from one type to a completely different type.
#if 0
  if (!hasTypeSet() || !getChild(0)->hasTypeSet()) {
    bool MadeChange = UpdateNodeType(getChild(0)->getExtType(), TP);
    MadeChange |= getChild(0)->UpdateNodeType(getExtType(), TP);
    return MadeChange;
  }
#endif
  return MadeChange;
}

/// OnlyOnRHSOfCommutative - Return true if this value is only allowed on the
/// RHS of a commutative operation, not the on LHS.
static bool OnlyOnRHSOfCommutative(TreePatternNode *N) {
  if (!N->isLeaf() && N->getOperator()->getName() == "imm")
    return true;
  if (N->isLeaf() && dynamic_cast<IntInit*>(N->getLeafValue()))
    return true;
  return false;
}


/// canPatternMatch - If it is impossible for this pattern to match on this
/// target, fill in Reason and return false.  Otherwise, return true.  This is
/// used as a sanity check for .td files (to prevent people from writing stuff
/// that can never possibly work), and to prevent the pattern permuter from
/// generating stuff that is useless.
bool TreePatternNode::canPatternMatch(std::string &Reason, 
                                      const CodeGenDAGPatterns &CDP) {
  if (isLeaf()) return true;

  for (unsigned i = 0, e = getNumChildren(); i != e; ++i)
    if (!getChild(i)->canPatternMatch(Reason, CDP))
      return false;

  // If this is an intrinsic, handle cases that would make it not match.  For
  // example, if an operand is required to be an immediate.
  if (getOperator()->isSubClassOf("Intrinsic")) {
    // TODO:
    return true;
  }
  
  // If this node is a commutative operator, check that the LHS isn't an
  // immediate.
  const SDNodeInfo &NodeInfo = CDP.getSDNodeInfo(getOperator());
  bool isCommIntrinsic = isCommutativeIntrinsic(CDP);
  if (NodeInfo.hasProperty(SDNPCommutative) || isCommIntrinsic) {
    // Scan all of the operands of the node and make sure that only the last one
    // is a constant node, unless the RHS also is.
    if (!OnlyOnRHSOfCommutative(getChild(getNumChildren()-1))) {
      bool Skip = isCommIntrinsic ? 1 : 0; // First operand is intrinsic id.
      for (unsigned i = Skip, e = getNumChildren()-1; i != e; ++i)
        if (OnlyOnRHSOfCommutative(getChild(i))) {
          Reason="Immediate value must be on the RHS of commutative operators!";
          return false;
        }
    }
  }
  
  return true;
}

//===----------------------------------------------------------------------===//
// TreePattern implementation
//

TreePattern::TreePattern(Record *TheRec, ListInit *RawPat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp){
  isInputPattern = isInput;
  for (unsigned i = 0, e = RawPat->getSize(); i != e; ++i)
    Trees.push_back(ParseTreePattern((DagInit*)RawPat->getElement(i)));
}

TreePattern::TreePattern(Record *TheRec, DagInit *Pat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp){
  isInputPattern = isInput;
  Trees.push_back(ParseTreePattern(Pat));
}

TreePattern::TreePattern(Record *TheRec, TreePatternNode *Pat, bool isInput,
                         CodeGenDAGPatterns &cdp) : TheRecord(TheRec), CDP(cdp){
  isInputPattern = isInput;
  Trees.push_back(Pat);
}

void TreePattern::error(const std::string &Msg) const {
  dump();
  throw TGError(TheRecord->getLoc(), "In " + TheRecord->getName() + ": " + Msg);
}

void TreePattern::ComputeNamedNodes() {
  for (unsigned i = 0, e = Trees.size(); i != e; ++i)
    ComputeNamedNodes(Trees[i]);
}

void TreePattern::ComputeNamedNodes(TreePatternNode *N) {
  if (!N->getName().empty())
    NamedNodes[N->getName()].push_back(N);
  
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    ComputeNamedNodes(N->getChild(i));
}

TreePatternNode *TreePattern::ParseTreePattern(DagInit *Dag) {
  DefInit *OpDef = dynamic_cast<DefInit*>(Dag->getOperator());
  if (!OpDef) error("Pattern has unexpected operator type!");
  Record *Operator = OpDef->getDef();
  
  if (Operator->isSubClassOf("ValueType")) {
    // If the operator is a ValueType, then this must be "type cast" of a leaf
    // node.
    if (Dag->getNumArgs() != 1)
      error("Type cast only takes one operand!");
    
    Init *Arg = Dag->getArg(0);
    TreePatternNode *New;
    if (DefInit *DI = dynamic_cast<DefInit*>(Arg)) {
      Record *R = DI->getDef();
      if (R->isSubClassOf("SDNode") || R->isSubClassOf("PatFrag")) {
        Dag->setArg(0, new DagInit(DI, "",
                                std::vector<std::pair<Init*, std::string> >()));
        return ParseTreePattern(Dag);
      }
      
      // Input argument?
      if (R->getName() == "node") {
        if (Dag->getArgName(0).empty())
          error("'node' argument requires a name to match with operand list");
        Args.push_back(Dag->getArgName(0));
      }
      
      New = new TreePatternNode(DI);
    } else if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      New = ParseTreePattern(DI);
    } else if (IntInit *II = dynamic_cast<IntInit*>(Arg)) {
      New = new TreePatternNode(II);
      if (!Dag->getArgName(0).empty())
        error("Constant int argument should not have a name!");
    } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Arg)) {
      // Turn this into an IntInit.
      Init *II = BI->convertInitializerTo(new IntRecTy());
      if (II == 0 || !dynamic_cast<IntInit*>(II))
        error("Bits value must be constants!");
      
      New = new TreePatternNode(dynamic_cast<IntInit*>(II));
      if (!Dag->getArgName(0).empty())
        error("Constant int argument should not have a name!");
    } else {
      Arg->dump();
      error("Unknown leaf value for tree pattern!");
      return 0;
    }
    
    // Apply the type cast.
    New->UpdateNodeType(getValueType(Operator), *this);
    if (New->getNumChildren() == 0)
      New->setName(Dag->getArgName(0));
    return New;
  }
  
  // Verify that this is something that makes sense for an operator.
  if (!Operator->isSubClassOf("PatFrag") && 
      !Operator->isSubClassOf("SDNode") &&
      !Operator->isSubClassOf("Instruction") && 
      !Operator->isSubClassOf("SDNodeXForm") &&
      !Operator->isSubClassOf("Intrinsic") &&
      Operator->getName() != "set" &&
      Operator->getName() != "implicit" &&
      Operator->getName() != "parallel")
    error("Unrecognized node '" + Operator->getName() + "'!");
  
  //  Check to see if this is something that is illegal in an input pattern.
  if (isInputPattern && (Operator->isSubClassOf("Instruction") ||
                         Operator->isSubClassOf("SDNodeXForm")))
    error("Cannot use '" + Operator->getName() + "' in an input pattern!");
  
  std::vector<TreePatternNode*> Children;
  
  for (unsigned i = 0, e = Dag->getNumArgs(); i != e; ++i) {
    Init *Arg = Dag->getArg(i);
    if (DagInit *DI = dynamic_cast<DagInit*>(Arg)) {
      Children.push_back(ParseTreePattern(DI));
      if (Children.back()->getName().empty())
        Children.back()->setName(Dag->getArgName(i));
    } else if (DefInit *DefI = dynamic_cast<DefInit*>(Arg)) {
      Record *R = DefI->getDef();
      // Direct reference to a leaf DagNode or PatFrag?  Turn it into a
      // TreePatternNode if its own.
      if (R->isSubClassOf("SDNode") || R->isSubClassOf("PatFrag")) {
        Dag->setArg(i, new DagInit(DefI, "",
                              std::vector<std::pair<Init*, std::string> >()));
        --i;  // Revisit this node...
      } else {
        TreePatternNode *Node = new TreePatternNode(DefI);
        Node->setName(Dag->getArgName(i));
        Children.push_back(Node);
        
        // Input argument?
        if (R->getName() == "node") {
          if (Dag->getArgName(i).empty())
            error("'node' argument requires a name to match with operand list");
          Args.push_back(Dag->getArgName(i));
        }
      }
    } else if (IntInit *II = dynamic_cast<IntInit*>(Arg)) {
      TreePatternNode *Node = new TreePatternNode(II);
      if (!Dag->getArgName(i).empty())
        error("Constant int argument should not have a name!");
      Children.push_back(Node);
    } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Arg)) {
      // Turn this into an IntInit.
      Init *II = BI->convertInitializerTo(new IntRecTy());
      if (II == 0 || !dynamic_cast<IntInit*>(II))
        error("Bits value must be constants!");
      
      TreePatternNode *Node = new TreePatternNode(dynamic_cast<IntInit*>(II));
      if (!Dag->getArgName(i).empty())
        error("Constant int argument should not have a name!");
      Children.push_back(Node);
    } else {
      errs() << '"';
      Arg->dump();
      errs() << "\": ";
      error("Unknown leaf value for tree pattern!");
    }
  }
  
  // If the operator is an intrinsic, then this is just syntactic sugar for for
  // (intrinsic_* <number>, ..children..).  Pick the right intrinsic node, and 
  // convert the intrinsic name to a number.
  if (Operator->isSubClassOf("Intrinsic")) {
    const CodeGenIntrinsic &Int = getDAGPatterns().getIntrinsic(Operator);
    unsigned IID = getDAGPatterns().getIntrinsicID(Operator)+1;

    // If this intrinsic returns void, it must have side-effects and thus a
    // chain.
    if (Int.IS.RetVTs[0] == MVT::isVoid) {
      Operator = getDAGPatterns().get_intrinsic_void_sdnode();
    } else if (Int.ModRef != CodeGenIntrinsic::NoMem) {
      // Has side-effects, requires chain.
      Operator = getDAGPatterns().get_intrinsic_w_chain_sdnode();
    } else {
      // Otherwise, no chain.
      Operator = getDAGPatterns().get_intrinsic_wo_chain_sdnode();
    }
    
    TreePatternNode *IIDNode = new TreePatternNode(new IntInit(IID));
    Children.insert(Children.begin(), IIDNode);
  }
  
  TreePatternNode *Result = new TreePatternNode(Operator, Children);
  Result->setName(Dag->getName());
  return Result;
}

/// InferAllTypes - Infer/propagate as many types throughout the expression
/// patterns as possible.  Return true if all types are inferred, false
/// otherwise.  Throw an exception if a type contradiction is found.
bool TreePattern::
InferAllTypes(const StringMap<SmallVector<TreePatternNode*,1> > *InNamedTypes) {
  if (NamedNodes.empty())
    ComputeNamedNodes();

  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (unsigned i = 0, e = Trees.size(); i != e; ++i)
      MadeChange |= Trees[i]->ApplyTypeConstraints(*this, false);

    // If there are constraints on our named nodes, apply them.
    for (StringMap<SmallVector<TreePatternNode*,1> >::iterator 
         I = NamedNodes.begin(), E = NamedNodes.end(); I != E; ++I) {
      SmallVectorImpl<TreePatternNode*> &Nodes = I->second;
      
      // If we have input named node types, propagate their types to the named
      // values here.
      if (InNamedTypes) {
        // FIXME: Should be error?
        assert(InNamedTypes->count(I->getKey()) &&
               "Named node in output pattern but not input pattern?");

        const SmallVectorImpl<TreePatternNode*> &InNodes =
          InNamedTypes->find(I->getKey())->second;

        // The input types should be fully resolved by now.
        for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
          // If this node is a register class, and it is the root of the pattern
          // then we're mapping something onto an input register.  We allow
          // changing the type of the input register in this case.  This allows
          // us to match things like:
          //  def : Pat<(v1i64 (bitconvert(v2i32 DPR:$src))), (v1i64 DPR:$src)>;
          if (Nodes[i] == Trees[0] && Nodes[i]->isLeaf()) {
            DefInit *DI = dynamic_cast<DefInit*>(Nodes[i]->getLeafValue());
            if (DI && DI->getDef()->isSubClassOf("RegisterClass"))
              continue;
          }
          
          MadeChange |=Nodes[i]->UpdateNodeType(InNodes[0]->getExtType(),*this);
        }
      }
      
      // If there are multiple nodes with the same name, they must all have the
      // same type.
      if (I->second.size() > 1) {
        for (unsigned i = 0, e = Nodes.size()-1; i != e; ++i) {
          MadeChange |=Nodes[i]->UpdateNodeType(Nodes[i+1]->getExtType(),*this);
          MadeChange |=Nodes[i+1]->UpdateNodeType(Nodes[i]->getExtType(),*this);
        }
      }
    }
  }
  
  bool HasUnresolvedTypes = false;
  for (unsigned i = 0, e = Trees.size(); i != e; ++i)
    HasUnresolvedTypes |= Trees[i]->ContainsUnresolvedType();
  return !HasUnresolvedTypes;
}

void TreePattern::print(raw_ostream &OS) const {
  OS << getRecord()->getName();
  if (!Args.empty()) {
    OS << "(" << Args[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i)
      OS << ", " << Args[i];
    OS << ")";
  }
  OS << ": ";
  
  if (Trees.size() > 1)
    OS << "[\n";
  for (unsigned i = 0, e = Trees.size(); i != e; ++i) {
    OS << "\t";
    Trees[i]->print(OS);
    OS << "\n";
  }

  if (Trees.size() > 1)
    OS << "]\n";
}

void TreePattern::dump() const { print(errs()); }

//===----------------------------------------------------------------------===//
// CodeGenDAGPatterns implementation
//

CodeGenDAGPatterns::CodeGenDAGPatterns(RecordKeeper &R) : Records(R) {
  Intrinsics = LoadIntrinsics(Records, false);
  TgtIntrinsics = LoadIntrinsics(Records, true);
  ParseNodeInfo();
  ParseNodeTransforms();
  ParseComplexPatterns();
  ParsePatternFragments();
  ParseDefaultOperands();
  ParseInstructions();
  ParsePatterns();
  
  // Generate variants.  For example, commutative patterns can match
  // multiple ways.  Add them to PatternsToMatch as well.
  GenerateVariants();

  // Infer instruction flags.  For example, we can detect loads,
  // stores, and side effects in many cases by examining an
  // instruction's pattern.
  InferInstructionFlags();
}

CodeGenDAGPatterns::~CodeGenDAGPatterns() {
  for (pf_iterator I = PatternFragments.begin(),
       E = PatternFragments.end(); I != E; ++I)
    delete I->second;
}


Record *CodeGenDAGPatterns::getSDNodeNamed(const std::string &Name) const {
  Record *N = Records.getDef(Name);
  if (!N || !N->isSubClassOf("SDNode")) {
    errs() << "Error getting SDNode '" << Name << "'!\n";
    exit(1);
  }
  return N;
}

// Parse all of the SDNode definitions for the target, populating SDNodes.
void CodeGenDAGPatterns::ParseNodeInfo() {
  std::vector<Record*> Nodes = Records.getAllDerivedDefinitions("SDNode");
  while (!Nodes.empty()) {
    SDNodes.insert(std::make_pair(Nodes.back(), Nodes.back()));
    Nodes.pop_back();
  }

  // Get the builtin intrinsic nodes.
  intrinsic_void_sdnode     = getSDNodeNamed("intrinsic_void");
  intrinsic_w_chain_sdnode  = getSDNodeNamed("intrinsic_w_chain");
  intrinsic_wo_chain_sdnode = getSDNodeNamed("intrinsic_wo_chain");
}

/// ParseNodeTransforms - Parse all SDNodeXForm instances into the SDNodeXForms
/// map, and emit them to the file as functions.
void CodeGenDAGPatterns::ParseNodeTransforms() {
  std::vector<Record*> Xforms = Records.getAllDerivedDefinitions("SDNodeXForm");
  while (!Xforms.empty()) {
    Record *XFormNode = Xforms.back();
    Record *SDNode = XFormNode->getValueAsDef("Opcode");
    std::string Code = XFormNode->getValueAsCode("XFormFunction");
    SDNodeXForms.insert(std::make_pair(XFormNode, NodeXForm(SDNode, Code)));

    Xforms.pop_back();
  }
}

void CodeGenDAGPatterns::ParseComplexPatterns() {
  std::vector<Record*> AMs = Records.getAllDerivedDefinitions("ComplexPattern");
  while (!AMs.empty()) {
    ComplexPatterns.insert(std::make_pair(AMs.back(), AMs.back()));
    AMs.pop_back();
  }
}


/// ParsePatternFragments - Parse all of the PatFrag definitions in the .td
/// file, building up the PatternFragments map.  After we've collected them all,
/// inline fragments together as necessary, so that there are no references left
/// inside a pattern fragment to a pattern fragment.
///
void CodeGenDAGPatterns::ParsePatternFragments() {
  std::vector<Record*> Fragments = Records.getAllDerivedDefinitions("PatFrag");
  
  // First step, parse all of the fragments.
  for (unsigned i = 0, e = Fragments.size(); i != e; ++i) {
    DagInit *Tree = Fragments[i]->getValueAsDag("Fragment");
    TreePattern *P = new TreePattern(Fragments[i], Tree, true, *this);
    PatternFragments[Fragments[i]] = P;
    
    // Validate the argument list, converting it to set, to discard duplicates.
    std::vector<std::string> &Args = P->getArgList();
    std::set<std::string> OperandsSet(Args.begin(), Args.end());
    
    if (OperandsSet.count(""))
      P->error("Cannot have unnamed 'node' values in pattern fragment!");
    
    // Parse the operands list.
    DagInit *OpsList = Fragments[i]->getValueAsDag("Operands");
    DefInit *OpsOp = dynamic_cast<DefInit*>(OpsList->getOperator());
    // Special cases: ops == outs == ins. Different names are used to
    // improve readability.
    if (!OpsOp ||
        (OpsOp->getDef()->getName() != "ops" &&
         OpsOp->getDef()->getName() != "outs" &&
         OpsOp->getDef()->getName() != "ins"))
      P->error("Operands list should start with '(ops ... '!");
    
    // Copy over the arguments.       
    Args.clear();
    for (unsigned j = 0, e = OpsList->getNumArgs(); j != e; ++j) {
      if (!dynamic_cast<DefInit*>(OpsList->getArg(j)) ||
          static_cast<DefInit*>(OpsList->getArg(j))->
          getDef()->getName() != "node")
        P->error("Operands list should all be 'node' values.");
      if (OpsList->getArgName(j).empty())
        P->error("Operands list should have names for each operand!");
      if (!OperandsSet.count(OpsList->getArgName(j)))
        P->error("'" + OpsList->getArgName(j) +
                 "' does not occur in pattern or was multiply specified!");
      OperandsSet.erase(OpsList->getArgName(j));
      Args.push_back(OpsList->getArgName(j));
    }
    
    if (!OperandsSet.empty())
      P->error("Operands list does not contain an entry for operand '" +
               *OperandsSet.begin() + "'!");

    // If there is a code init for this fragment, keep track of the fact that
    // this fragment uses it.
    std::string Code = Fragments[i]->getValueAsCode("Predicate");
    if (!Code.empty())
      P->getOnlyTree()->addPredicateFn("Predicate_"+Fragments[i]->getName());
    
    // If there is a node transformation corresponding to this, keep track of
    // it.
    Record *Transform = Fragments[i]->getValueAsDef("OperandTransform");
    if (!getSDNodeTransform(Transform).second.empty())    // not noop xform?
      P->getOnlyTree()->setTransformFn(Transform);
  }
  
  // Now that we've parsed all of the tree fragments, do a closure on them so
  // that there are not references to PatFrags left inside of them.
  for (unsigned i = 0, e = Fragments.size(); i != e; ++i) {
    TreePattern *ThePat = PatternFragments[Fragments[i]];
    ThePat->InlinePatternFragments();
        
    // Infer as many types as possible.  Don't worry about it if we don't infer
    // all of them, some may depend on the inputs of the pattern.
    try {
      ThePat->InferAllTypes();
    } catch (...) {
      // If this pattern fragment is not supported by this target (no types can
      // satisfy its constraints), just ignore it.  If the bogus pattern is
      // actually used by instructions, the type consistency error will be
      // reported there.
    }
    
    // If debugging, print out the pattern fragment result.
    DEBUG(ThePat->dump());
  }
}

void CodeGenDAGPatterns::ParseDefaultOperands() {
  std::vector<Record*> DefaultOps[2];
  DefaultOps[0] = Records.getAllDerivedDefinitions("PredicateOperand");
  DefaultOps[1] = Records.getAllDerivedDefinitions("OptionalDefOperand");

  // Find some SDNode.
  assert(!SDNodes.empty() && "No SDNodes parsed?");
  Init *SomeSDNode = new DefInit(SDNodes.begin()->first);
  
  for (unsigned iter = 0; iter != 2; ++iter) {
    for (unsigned i = 0, e = DefaultOps[iter].size(); i != e; ++i) {
      DagInit *DefaultInfo = DefaultOps[iter][i]->getValueAsDag("DefaultOps");
    
      // Clone the DefaultInfo dag node, changing the operator from 'ops' to
      // SomeSDnode so that we can parse this.
      std::vector<std::pair<Init*, std::string> > Ops;
      for (unsigned op = 0, e = DefaultInfo->getNumArgs(); op != e; ++op)
        Ops.push_back(std::make_pair(DefaultInfo->getArg(op),
                                     DefaultInfo->getArgName(op)));
      DagInit *DI = new DagInit(SomeSDNode, "", Ops);
    
      // Create a TreePattern to parse this.
      TreePattern P(DefaultOps[iter][i], DI, false, *this);
      assert(P.getNumTrees() == 1 && "This ctor can only produce one tree!");

      // Copy the operands over into a DAGDefaultOperand.
      DAGDefaultOperand DefaultOpInfo;
    
      TreePatternNode *T = P.getTree(0);
      for (unsigned op = 0, e = T->getNumChildren(); op != e; ++op) {
        TreePatternNode *TPN = T->getChild(op);
        while (TPN->ApplyTypeConstraints(P, false))
          /* Resolve all types */;
      
        if (TPN->ContainsUnresolvedType()) {
          if (iter == 0)
            throw "Value #" + utostr(i) + " of PredicateOperand '" +
              DefaultOps[iter][i]->getName() +"' doesn't have a concrete type!";
          else
            throw "Value #" + utostr(i) + " of OptionalDefOperand '" +
              DefaultOps[iter][i]->getName() +"' doesn't have a concrete type!";
        }
        DefaultOpInfo.DefaultOps.push_back(TPN);
      }

      // Insert it into the DefaultOperands map so we can find it later.
      DefaultOperands[DefaultOps[iter][i]] = DefaultOpInfo;
    }
  }
}

/// HandleUse - Given "Pat" a leaf in the pattern, check to see if it is an
/// instruction input.  Return true if this is a real use.
static bool HandleUse(TreePattern *I, TreePatternNode *Pat,
                      std::map<std::string, TreePatternNode*> &InstInputs,
                      std::vector<Record*> &InstImpInputs) {
  // No name -> not interesting.
  if (Pat->getName().empty()) {
    if (Pat->isLeaf()) {
      DefInit *DI = dynamic_cast<DefInit*>(Pat->getLeafValue());
      if (DI && DI->getDef()->isSubClassOf("RegisterClass"))
        I->error("Input " + DI->getDef()->getName() + " must be named!");
      else if (DI && DI->getDef()->isSubClassOf("Register")) 
        InstImpInputs.push_back(DI->getDef());
    }
    return false;
  }

  Record *Rec;
  if (Pat->isLeaf()) {
    DefInit *DI = dynamic_cast<DefInit*>(Pat->getLeafValue());
    if (!DI) I->error("Input $" + Pat->getName() + " must be an identifier!");
    Rec = DI->getDef();
  } else {
    Rec = Pat->getOperator();
  }

  // SRCVALUE nodes are ignored.
  if (Rec->getName() == "srcvalue")
    return false;

  TreePatternNode *&Slot = InstInputs[Pat->getName()];
  if (!Slot) {
    Slot = Pat;
    return true;
  }
  Record *SlotRec;
  if (Slot->isLeaf()) {
    SlotRec = dynamic_cast<DefInit*>(Slot->getLeafValue())->getDef();
  } else {
    assert(Slot->getNumChildren() == 0 && "can't be a use with children!");
    SlotRec = Slot->getOperator();
  }
  
  // Ensure that the inputs agree if we've already seen this input.
  if (Rec != SlotRec)
    I->error("All $" + Pat->getName() + " inputs must agree with each other");
  if (Slot->getExtType() != Pat->getExtType())
    I->error("All $" + Pat->getName() + " inputs must agree with each other");
  return true;
}

/// FindPatternInputsAndOutputs - Scan the specified TreePatternNode (which is
/// part of "I", the instruction), computing the set of inputs and outputs of
/// the pattern.  Report errors if we see anything naughty.
void CodeGenDAGPatterns::
FindPatternInputsAndOutputs(TreePattern *I, TreePatternNode *Pat,
                            std::map<std::string, TreePatternNode*> &InstInputs,
                            std::map<std::string, TreePatternNode*>&InstResults,
                            std::vector<Record*> &InstImpInputs,
                            std::vector<Record*> &InstImpResults) {
  if (Pat->isLeaf()) {
    bool isUse = HandleUse(I, Pat, InstInputs, InstImpInputs);
    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  }
  
  if (Pat->getOperator()->getName() == "implicit") {
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i) {
      TreePatternNode *Dest = Pat->getChild(i);
      if (!Dest->isLeaf())
        I->error("implicitly defined value should be a register!");
    
      DefInit *Val = dynamic_cast<DefInit*>(Dest->getLeafValue());
      if (!Val || !Val->getDef()->isSubClassOf("Register"))
        I->error("implicitly defined value should be a register!");
      InstImpResults.push_back(Val->getDef());
    }
    return;
  }
  
  if (Pat->getOperator()->getName() != "set") {
    // If this is not a set, verify that the children nodes are not void typed,
    // and recurse.
    for (unsigned i = 0, e = Pat->getNumChildren(); i != e; ++i) {
      if (Pat->getChild(i)->getType() == MVT::isVoid)
        I->error("Cannot have void nodes inside of patterns!");
      FindPatternInputsAndOutputs(I, Pat->getChild(i), InstInputs, InstResults,
                                  InstImpInputs, InstImpResults);
    }
    
    // If this is a non-leaf node with no children, treat it basically as if
    // it were a leaf.  This handles nodes like (imm).
    bool isUse = HandleUse(I, Pat, InstInputs, InstImpInputs);
    
    if (!isUse && Pat->getTransformFn())
      I->error("Cannot specify a transform function for a non-input value!");
    return;
  }
  
  // Otherwise, this is a set, validate and collect instruction results.
  if (Pat->getNumChildren() == 0)
    I->error("set requires operands!");
  
  if (Pat->getTransformFn())
    I->error("Cannot specify a transform function on a set node!");
  
  // Check the set destinations.
  unsigned NumDests = Pat->getNumChildren()-1;
  for (unsigned i = 0; i != NumDests; ++i) {
    TreePatternNode *Dest = Pat->getChild(i);
    if (!Dest->isLeaf())
      I->error("set destination should be a register!");
    
    DefInit *Val = dynamic_cast<DefInit*>(Dest->getLeafValue());
    if (!Val)
      I->error("set destination should be a register!");

    if (Val->getDef()->isSubClassOf("RegisterClass") ||
        Val->getDef()->isSubClassOf("PointerLikeRegClass")) {
      if (Dest->getName().empty())
        I->error("set destination must have a name!");
      if (InstResults.count(Dest->getName()))
        I->error("cannot set '" + Dest->getName() +"' multiple times");
      InstResults[Dest->getName()] = Dest;
    } else if (Val->getDef()->isSubClassOf("Register")) {
      InstImpResults.push_back(Val->getDef());
    } else {
      I->error("set destination should be a register!");
    }
  }
    
  // Verify and collect info from the computation.
  FindPatternInputsAndOutputs(I, Pat->getChild(NumDests),
                              InstInputs, InstResults,
                              InstImpInputs, InstImpResults);
}

//===----------------------------------------------------------------------===//
// Instruction Analysis
//===----------------------------------------------------------------------===//

class InstAnalyzer {
  const CodeGenDAGPatterns &CDP;
  bool &mayStore;
  bool &mayLoad;
  bool &HasSideEffects;
public:
  InstAnalyzer(const CodeGenDAGPatterns &cdp,
               bool &maystore, bool &mayload, bool &hse)
    : CDP(cdp), mayStore(maystore), mayLoad(mayload), HasSideEffects(hse){
  }

  /// Analyze - Analyze the specified instruction, returning true if the
  /// instruction had a pattern.
  bool Analyze(Record *InstRecord) {
    const TreePattern *Pattern = CDP.getInstruction(InstRecord).getPattern();
    if (Pattern == 0) {
      HasSideEffects = 1;
      return false;  // No pattern.
    }

    // FIXME: Assume only the first tree is the pattern. The others are clobber
    // nodes.
    AnalyzeNode(Pattern->getTree(0));
    return true;
  }

private:
  void AnalyzeNode(const TreePatternNode *N) {
    if (N->isLeaf()) {
      if (DefInit *DI = dynamic_cast<DefInit*>(N->getLeafValue())) {
        Record *LeafRec = DI->getDef();
        // Handle ComplexPattern leaves.
        if (LeafRec->isSubClassOf("ComplexPattern")) {
          const ComplexPattern &CP = CDP.getComplexPattern(LeafRec);
          if (CP.hasProperty(SDNPMayStore)) mayStore = true;
          if (CP.hasProperty(SDNPMayLoad)) mayLoad = true;
          if (CP.hasProperty(SDNPSideEffect)) HasSideEffects = true;
        }
      }
      return;
    }

    // Analyze children.
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
      AnalyzeNode(N->getChild(i));

    // Ignore set nodes, which are not SDNodes.
    if (N->getOperator()->getName() == "set")
      return;

    // Get information about the SDNode for the operator.
    const SDNodeInfo &OpInfo = CDP.getSDNodeInfo(N->getOperator());

    // Notice properties of the node.
    if (OpInfo.hasProperty(SDNPMayStore)) mayStore = true;
    if (OpInfo.hasProperty(SDNPMayLoad)) mayLoad = true;
    if (OpInfo.hasProperty(SDNPSideEffect)) HasSideEffects = true;

    if (const CodeGenIntrinsic *IntInfo = N->getIntrinsicInfo(CDP)) {
      // If this is an intrinsic, analyze it.
      if (IntInfo->ModRef >= CodeGenIntrinsic::ReadArgMem)
        mayLoad = true;// These may load memory.

      if (IntInfo->ModRef >= CodeGenIntrinsic::WriteArgMem)
        mayStore = true;// Intrinsics that can write to memory are 'mayStore'.

      if (IntInfo->ModRef >= CodeGenIntrinsic::WriteMem)
        // WriteMem intrinsics can have other strange effects.
        HasSideEffects = true;
    }
  }

};

static void InferFromPattern(const CodeGenInstruction &Inst,
                             bool &MayStore, bool &MayLoad,
                             bool &HasSideEffects,
                             const CodeGenDAGPatterns &CDP) {
  MayStore = MayLoad = HasSideEffects = false;

  bool HadPattern =
    InstAnalyzer(CDP, MayStore, MayLoad, HasSideEffects).Analyze(Inst.TheDef);

  // InstAnalyzer only correctly analyzes mayStore/mayLoad so far.
  if (Inst.mayStore) {  // If the .td file explicitly sets mayStore, use it.
    // If we decided that this is a store from the pattern, then the .td file
    // entry is redundant.
    if (MayStore)
      fprintf(stderr,
              "Warning: mayStore flag explicitly set on instruction '%s'"
              " but flag already inferred from pattern.\n",
              Inst.TheDef->getName().c_str());
    MayStore = true;
  }

  if (Inst.mayLoad) {  // If the .td file explicitly sets mayLoad, use it.
    // If we decided that this is a load from the pattern, then the .td file
    // entry is redundant.
    if (MayLoad)
      fprintf(stderr,
              "Warning: mayLoad flag explicitly set on instruction '%s'"
              " but flag already inferred from pattern.\n",
              Inst.TheDef->getName().c_str());
    MayLoad = true;
  }

  if (Inst.neverHasSideEffects) {
    if (HadPattern)
      fprintf(stderr, "Warning: neverHasSideEffects set on instruction '%s' "
              "which already has a pattern\n", Inst.TheDef->getName().c_str());
    HasSideEffects = false;
  }

  if (Inst.hasSideEffects) {
    if (HasSideEffects)
      fprintf(stderr, "Warning: hasSideEffects set on instruction '%s' "
              "which already inferred this.\n", Inst.TheDef->getName().c_str());
    HasSideEffects = true;
  }
}

/// ParseInstructions - Parse all of the instructions, inlining and resolving
/// any fragments involved.  This populates the Instructions list with fully
/// resolved instructions.
void CodeGenDAGPatterns::ParseInstructions() {
  std::vector<Record*> Instrs = Records.getAllDerivedDefinitions("Instruction");
  
  for (unsigned i = 0, e = Instrs.size(); i != e; ++i) {
    ListInit *LI = 0;
    
    if (dynamic_cast<ListInit*>(Instrs[i]->getValueInit("Pattern")))
      LI = Instrs[i]->getValueAsListInit("Pattern");
    
    // If there is no pattern, only collect minimal information about the
    // instruction for its operand list.  We have to assume that there is one
    // result, as we have no detailed info.
    if (!LI || LI->getSize() == 0) {
      std::vector<Record*> Results;
      std::vector<Record*> Operands;
      
      CodeGenInstruction &InstInfo =Target.getInstruction(Instrs[i]->getName());

      if (InstInfo.OperandList.size() != 0) {
        if (InstInfo.NumDefs == 0) {
          // These produce no results
          for (unsigned j = 0, e = InstInfo.OperandList.size(); j < e; ++j)
            Operands.push_back(InstInfo.OperandList[j].Rec);
        } else {
          // Assume the first operand is the result.
          Results.push_back(InstInfo.OperandList[0].Rec);
      
          // The rest are inputs.
          for (unsigned j = 1, e = InstInfo.OperandList.size(); j < e; ++j)
            Operands.push_back(InstInfo.OperandList[j].Rec);
        }
      }
      
      // Create and insert the instruction.
      std::vector<Record*> ImpResults;
      std::vector<Record*> ImpOperands;
      Instructions.insert(std::make_pair(Instrs[i], 
                          DAGInstruction(0, Results, Operands, ImpResults,
                                         ImpOperands)));
      continue;  // no pattern.
    }
    
    // Parse the instruction.
    TreePattern *I = new TreePattern(Instrs[i], LI, true, *this);
    // Inline pattern fragments into it.
    I->InlinePatternFragments();
    
    // Infer as many types as possible.  If we cannot infer all of them, we can
    // never do anything with this instruction pattern: report it to the user.
    if (!I->InferAllTypes())
      I->error("Could not infer all types in pattern!");
    
    // InstInputs - Keep track of all of the inputs of the instruction, along 
    // with the record they are declared as.
    std::map<std::string, TreePatternNode*> InstInputs;
    
    // InstResults - Keep track of all the virtual registers that are 'set'
    // in the instruction, including what reg class they are.
    std::map<std::string, TreePatternNode*> InstResults;

    std::vector<Record*> InstImpInputs;
    std::vector<Record*> InstImpResults;
    
    // Verify that the top-level forms in the instruction are of void type, and
    // fill in the InstResults map.
    for (unsigned j = 0, e = I->getNumTrees(); j != e; ++j) {
      TreePatternNode *Pat = I->getTree(j);
      if (!Pat->hasTypeSet() || Pat->getType() != MVT::isVoid)
        I->error("Top-level forms in instruction pattern should have"
                 " void types");

      // Find inputs and outputs, and verify the structure of the uses/defs.
      FindPatternInputsAndOutputs(I, Pat, InstInputs, InstResults,
                                  InstImpInputs, InstImpResults);
    }

    // Now that we have inputs and outputs of the pattern, inspect the operands
    // list for the instruction.  This determines the order that operands are
    // added to the machine instruction the node corresponds to.
    unsigned NumResults = InstResults.size();

    // Parse the operands list from the (ops) list, validating it.
    assert(I->getArgList().empty() && "Args list should still be empty here!");
    CodeGenInstruction &CGI = Target.getInstruction(Instrs[i]->getName());

    // Check that all of the results occur first in the list.
    std::vector<Record*> Results;
    TreePatternNode *Res0Node = NULL;
    for (unsigned i = 0; i != NumResults; ++i) {
      if (i == CGI.OperandList.size())
        I->error("'" + InstResults.begin()->first +
                 "' set but does not appear in operand list!");
      const std::string &OpName = CGI.OperandList[i].Name;
      
      // Check that it exists in InstResults.
      TreePatternNode *RNode = InstResults[OpName];
      if (RNode == 0)
        I->error("Operand $" + OpName + " does not exist in operand list!");
        
      if (i == 0)
        Res0Node = RNode;
      Record *R = dynamic_cast<DefInit*>(RNode->getLeafValue())->getDef();
      if (R == 0)
        I->error("Operand $" + OpName + " should be a set destination: all "
                 "outputs must occur before inputs in operand list!");
      
      if (CGI.OperandList[i].Rec != R)
        I->error("Operand $" + OpName + " class mismatch!");
      
      // Remember the return type.
      Results.push_back(CGI.OperandList[i].Rec);
      
      // Okay, this one checks out.
      InstResults.erase(OpName);
    }

    // Loop over the inputs next.  Make a copy of InstInputs so we can destroy
    // the copy while we're checking the inputs.
    std::map<std::string, TreePatternNode*> InstInputsCheck(InstInputs);

    std::vector<TreePatternNode*> ResultNodeOperands;
    std::vector<Record*> Operands;
    for (unsigned i = NumResults, e = CGI.OperandList.size(); i != e; ++i) {
      CodeGenInstruction::OperandInfo &Op = CGI.OperandList[i];
      const std::string &OpName = Op.Name;
      if (OpName.empty())
        I->error("Operand #" + utostr(i) + " in operands list has no name!");

      if (!InstInputsCheck.count(OpName)) {
        // If this is an predicate operand or optional def operand with an
        // DefaultOps set filled in, we can ignore this.  When we codegen it,
        // we will do so as always executed.
        if (Op.Rec->isSubClassOf("PredicateOperand") ||
            Op.Rec->isSubClassOf("OptionalDefOperand")) {
          // Does it have a non-empty DefaultOps field?  If so, ignore this
          // operand.
          if (!getDefaultOperand(Op.Rec).DefaultOps.empty())
            continue;
        }
        I->error("Operand $" + OpName +
                 " does not appear in the instruction pattern");
      }
      TreePatternNode *InVal = InstInputsCheck[OpName];
      InstInputsCheck.erase(OpName);   // It occurred, remove from map.
      
      if (InVal->isLeaf() &&
          dynamic_cast<DefInit*>(InVal->getLeafValue())) {
        Record *InRec = static_cast<DefInit*>(InVal->getLeafValue())->getDef();
        if (Op.Rec != InRec && !InRec->isSubClassOf("ComplexPattern"))
          I->error("Operand $" + OpName + "'s register class disagrees"
                   " between the operand and pattern");
      }
      Operands.push_back(Op.Rec);
      
      // Construct the result for the dest-pattern operand list.
      TreePatternNode *OpNode = InVal->clone();
      
      // No predicate is useful on the result.
      OpNode->clearPredicateFns();
      
      // Promote the xform function to be an explicit node if set.
      if (Record *Xform = OpNode->getTransformFn()) {
        OpNode->setTransformFn(0);
        std::vector<TreePatternNode*> Children;
        Children.push_back(OpNode);
        OpNode = new TreePatternNode(Xform, Children);
      }
      
      ResultNodeOperands.push_back(OpNode);
    }
    
    if (!InstInputsCheck.empty())
      I->error("Input operand $" + InstInputsCheck.begin()->first +
               " occurs in pattern but not in operands list!");

    TreePatternNode *ResultPattern =
      new TreePatternNode(I->getRecord(), ResultNodeOperands);
    // Copy fully inferred output node type to instruction result pattern.
    if (NumResults > 0)
      ResultPattern->setType(Res0Node->getExtType());

    // Create and insert the instruction.
    // FIXME: InstImpResults and InstImpInputs should not be part of
    // DAGInstruction.
    DAGInstruction TheInst(I, Results, Operands, InstImpResults, InstImpInputs);
    Instructions.insert(std::make_pair(I->getRecord(), TheInst));

    // Use a temporary tree pattern to infer all types and make sure that the
    // constructed result is correct.  This depends on the instruction already
    // being inserted into the Instructions map.
    TreePattern Temp(I->getRecord(), ResultPattern, false, *this);
    Temp.InferAllTypes(&I->getNamedNodesMap());

    DAGInstruction &TheInsertedInst = Instructions.find(I->getRecord())->second;
    TheInsertedInst.setResultPattern(Temp.getOnlyTree());
    
    DEBUG(I->dump());
  }
   
  // If we can, convert the instructions to be patterns that are matched!
  for (std::map<Record*, DAGInstruction, RecordPtrCmp>::iterator II =
        Instructions.begin(),
       E = Instructions.end(); II != E; ++II) {
    DAGInstruction &TheInst = II->second;
    const TreePattern *I = TheInst.getPattern();
    if (I == 0) continue;  // No pattern.

    // FIXME: Assume only the first tree is the pattern. The others are clobber
    // nodes.
    TreePatternNode *Pattern = I->getTree(0);
    TreePatternNode *SrcPattern;
    if (Pattern->getOperator()->getName() == "set") {
      SrcPattern = Pattern->getChild(Pattern->getNumChildren()-1)->clone();
    } else{
      // Not a set (store or something?)
      SrcPattern = Pattern;
    }
    
    Record *Instr = II->first;
    AddPatternToMatch(I,
                      PatternToMatch(Instr->getValueAsListInit("Predicates"),
                                     SrcPattern,
                                     TheInst.getResultPattern(),
                                     TheInst.getImpResults(),
                                     Instr->getValueAsInt("AddedComplexity"),
                                     Instr->getID()));
  }
}


typedef std::pair<const TreePatternNode*, unsigned> NameRecord;

static void FindNames(const TreePatternNode *P, 
                      std::map<std::string, NameRecord> &Names,
                      const TreePattern *PatternTop) {
  if (!P->getName().empty()) {
    NameRecord &Rec = Names[P->getName()];
    // If this is the first instance of the name, remember the node.
    if (Rec.second++ == 0)
      Rec.first = P;
    else if (Rec.first->getType() != P->getType())
      PatternTop->error("repetition of value: $" + P->getName() +
                        " where different uses have different types!");
  }
  
  if (!P->isLeaf()) {
    for (unsigned i = 0, e = P->getNumChildren(); i != e; ++i)
      FindNames(P->getChild(i), Names, PatternTop);
  }
}

void CodeGenDAGPatterns::AddPatternToMatch(const TreePattern *Pattern,
                                           const PatternToMatch &PTM) {
  // Do some sanity checking on the pattern we're about to match.
  std::string Reason;
  if (!PTM.getSrcPattern()->canPatternMatch(Reason, *this))
    Pattern->error("Pattern can never match: " + Reason);
  
  // If the source pattern's root is a complex pattern, that complex pattern
  // must specify the nodes it can potentially match.
  if (const ComplexPattern *CP =
        PTM.getSrcPattern()->getComplexPatternInfo(*this))
    if (CP->getRootNodes().empty())
      Pattern->error("ComplexPattern at root must specify list of opcodes it"
                     " could match");
  
  
  // Find all of the named values in the input and output, ensure they have the
  // same type.
  std::map<std::string, NameRecord> SrcNames, DstNames;
  FindNames(PTM.getSrcPattern(), SrcNames, Pattern);
  FindNames(PTM.getDstPattern(), DstNames, Pattern);

  // Scan all of the named values in the destination pattern, rejecting them if
  // they don't exist in the input pattern.
  for (std::map<std::string, NameRecord>::iterator
       I = DstNames.begin(), E = DstNames.end(); I != E; ++I) {
    if (SrcNames[I->first].first == 0)
      Pattern->error("Pattern has input without matching name in output: $" +
                     I->first);
    
#if 0
    const std::vector<unsigned char> &SrcTypeVec =
      SrcNames[I->first].first->getExtTypes();
    const std::vector<unsigned char> &DstTypeVec =
      I->second.first->getExtTypes();
    if (SrcTypeVec == DstTypeVec) continue;
    
    std::string SrcType, DstType;
    for (unsigned i = 0, e = SrcTypeVec.size(); i != e; ++i)
      SrcType += ":" + GetTypeName(SrcTypeVec[i]);
    for (unsigned i = 0, e = DstTypeVec.size(); i != e; ++i)
      DstType += ":" + GetTypeName(DstTypeVec[i]);
    
    Pattern->error("Variable $" + I->first +
                   " has different types in source (" + SrcType +
                   ") and dest (" + DstType + ") pattern!");
#endif
  }
  
  // Scan all of the named values in the source pattern, rejecting them if the
  // name isn't used in the dest, and isn't used to tie two values together.
  for (std::map<std::string, NameRecord>::iterator
       I = SrcNames.begin(), E = SrcNames.end(); I != E; ++I)
    if (DstNames[I->first].first == 0 && SrcNames[I->first].second == 1)
      Pattern->error("Pattern has dead named input: $" + I->first);
  
  PatternsToMatch.push_back(PTM);
}



void CodeGenDAGPatterns::InferInstructionFlags() {
  std::map<std::string, CodeGenInstruction> &InstrDescs =
    Target.getInstructions();
  for (std::map<std::string, CodeGenInstruction>::iterator
         II = InstrDescs.begin(), E = InstrDescs.end(); II != E; ++II) {
    CodeGenInstruction &InstInfo = II->second;
    // Determine properties of the instruction from its pattern.
    bool MayStore, MayLoad, HasSideEffects;
    InferFromPattern(InstInfo, MayStore, MayLoad, HasSideEffects, *this);
    InstInfo.mayStore = MayStore;
    InstInfo.mayLoad = MayLoad;
    InstInfo.hasSideEffects = HasSideEffects;
  }
}

/// Given a pattern result with an unresolved type, see if we can find one
/// instruction with an unresolved result type.  Force this result type to an
/// arbitrary element if it's possible types to converge results.
static bool ForceArbitraryInstResultType(TreePatternNode *N, TreePattern &TP) {
  if (N->isLeaf())
    return false;
  
  // Analyze children.
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    if (ForceArbitraryInstResultType(N->getChild(i), TP))
      return true;

  if (!N->getOperator()->isSubClassOf("Instruction"))
    return false;

  // If this type is already concrete or completely unknown we can't do
  // anything.
  if (N->getExtType().isCompletelyUnknown() || N->getExtType().isConcrete())
    return false;
  
  // Otherwise, force its type to the first possibility (an arbitrary choice).
  return N->getExtType().MergeInTypeInfo(N->getExtType().getTypeList()[0], TP);
}

void CodeGenDAGPatterns::ParsePatterns() {
  std::vector<Record*> Patterns = Records.getAllDerivedDefinitions("Pattern");

  for (unsigned i = 0, e = Patterns.size(); i != e; ++i) {
    DagInit *Tree = Patterns[i]->getValueAsDag("PatternToMatch");
    DefInit *OpDef = dynamic_cast<DefInit*>(Tree->getOperator());
    Record *Operator = OpDef->getDef();
    TreePattern *Pattern;
    if (Operator->getName() != "parallel")
      Pattern = new TreePattern(Patterns[i], Tree, true, *this);
    else {
      std::vector<Init*> Values;
      RecTy *ListTy = 0;
      for (unsigned j = 0, ee = Tree->getNumArgs(); j != ee; ++j) {
        Values.push_back(Tree->getArg(j));
        TypedInit *TArg = dynamic_cast<TypedInit*>(Tree->getArg(j));
        if (TArg == 0) {
          errs() << "In dag: " << Tree->getAsString();
          errs() << " --  Untyped argument in pattern\n";
          assert(0 && "Untyped argument in pattern");
        }
        if (ListTy != 0) {
          ListTy = resolveTypes(ListTy, TArg->getType());
          if (ListTy == 0) {
            errs() << "In dag: " << Tree->getAsString();
            errs() << " --  Incompatible types in pattern arguments\n";
            assert(0 && "Incompatible types in pattern arguments");
          }
        }
        else {
          ListTy = TArg->getType();
        }
      }
      ListInit *LI = new ListInit(Values, new ListRecTy(ListTy));
      Pattern = new TreePattern(Patterns[i], LI, true, *this);
    }

    // Inline pattern fragments into it.
    Pattern->InlinePatternFragments();
    
    ListInit *LI = Patterns[i]->getValueAsListInit("ResultInstrs");
    if (LI->getSize() == 0) continue;  // no pattern.
    
    // Parse the instruction.
    TreePattern *Result = new TreePattern(Patterns[i], LI, false, *this);
    
    // Inline pattern fragments into it.
    Result->InlinePatternFragments();

    if (Result->getNumTrees() != 1)
      Result->error("Cannot handle instructions producing instructions "
                    "with temporaries yet!");
    
    bool IterateInference;
    bool InferredAllPatternTypes, InferredAllResultTypes;
    do {
      // Infer as many types as possible.  If we cannot infer all of them, we
      // can never do anything with this pattern: report it to the user.
      InferredAllPatternTypes =
        Pattern->InferAllTypes(&Pattern->getNamedNodesMap());
      
      // Infer as many types as possible.  If we cannot infer all of them, we
      // can never do anything with this pattern: report it to the user.
      InferredAllResultTypes =
        Result->InferAllTypes(&Pattern->getNamedNodesMap());

      // Apply the type of the result to the source pattern.  This helps us
      // resolve cases where the input type is known to be a pointer type (which
      // is considered resolved), but the result knows it needs to be 32- or
      // 64-bits.  Infer the other way for good measure.
      IterateInference = Pattern->getTree(0)->
        UpdateNodeType(Result->getTree(0)->getExtType(), *Result);
      IterateInference |= Result->getTree(0)->
        UpdateNodeType(Pattern->getTree(0)->getExtType(), *Result);
      
      // If our iteration has converged and the input pattern's types are fully
      // resolved but the result pattern is not fully resolved, we may have a
      // situation where we have two instructions in the result pattern and
      // the instructions require a common register class, but don't care about
      // what actual MVT is used.  This is actually a bug in our modelling:
      // output patterns should have register classes, not MVTs.
      //
      // In any case, to handle this, we just go through and disambiguate some
      // arbitrary types to the result pattern's nodes.
      if (!IterateInference && InferredAllPatternTypes &&
          !InferredAllResultTypes)
        IterateInference = ForceArbitraryInstResultType(Result->getTree(0),
                                                        *Result);
      
    } while (IterateInference);
    
    // Verify that we inferred enough types that we can do something with the
    // pattern and result.  If these fire the user has to add type casts.
    if (!InferredAllPatternTypes)
      Pattern->error("Could not infer all types in pattern!");
    if (!InferredAllResultTypes) {
      Pattern->dump();
      Result->error("Could not infer all types in pattern result!");
    }
    
    // Validate that the input pattern is correct.
    std::map<std::string, TreePatternNode*> InstInputs;
    std::map<std::string, TreePatternNode*> InstResults;
    std::vector<Record*> InstImpInputs;
    std::vector<Record*> InstImpResults;
    for (unsigned j = 0, ee = Pattern->getNumTrees(); j != ee; ++j)
      FindPatternInputsAndOutputs(Pattern, Pattern->getTree(j),
                                  InstInputs, InstResults,
                                  InstImpInputs, InstImpResults);

    // Promote the xform function to be an explicit node if set.
    TreePatternNode *DstPattern = Result->getOnlyTree();
    std::vector<TreePatternNode*> ResultNodeOperands;
    for (unsigned ii = 0, ee = DstPattern->getNumChildren(); ii != ee; ++ii) {
      TreePatternNode *OpNode = DstPattern->getChild(ii);
      if (Record *Xform = OpNode->getTransformFn()) {
        OpNode->setTransformFn(0);
        std::vector<TreePatternNode*> Children;
        Children.push_back(OpNode);
        OpNode = new TreePatternNode(Xform, Children);
      }
      ResultNodeOperands.push_back(OpNode);
    }
    DstPattern = Result->getOnlyTree();
    if (!DstPattern->isLeaf())
      DstPattern = new TreePatternNode(DstPattern->getOperator(),
                                       ResultNodeOperands);
    DstPattern->setType(Result->getOnlyTree()->getExtType());
    TreePattern Temp(Result->getRecord(), DstPattern, false, *this);
    Temp.InferAllTypes();

    
    AddPatternToMatch(Pattern,
                 PatternToMatch(Patterns[i]->getValueAsListInit("Predicates"),
                                Pattern->getTree(0),
                                Temp.getOnlyTree(), InstImpResults,
                                Patterns[i]->getValueAsInt("AddedComplexity"),
                                Patterns[i]->getID()));
  }
}

/// CombineChildVariants - Given a bunch of permutations of each child of the
/// 'operator' node, put them together in all possible ways.
static void CombineChildVariants(TreePatternNode *Orig, 
               const std::vector<std::vector<TreePatternNode*> > &ChildVariants,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 CodeGenDAGPatterns &CDP,
                                 const MultipleUseVarSet &DepVars) {
  // Make sure that each operand has at least one variant to choose from.
  for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
    if (ChildVariants[i].empty())
      return;
        
  // The end result is an all-pairs construction of the resultant pattern.
  std::vector<unsigned> Idxs;
  Idxs.resize(ChildVariants.size());
  bool NotDone;
  do {
#ifndef NDEBUG
    DEBUG(if (!Idxs.empty()) {
            errs() << Orig->getOperator()->getName() << ": Idxs = [ ";
              for (unsigned i = 0; i < Idxs.size(); ++i) {
                errs() << Idxs[i] << " ";
            }
            errs() << "]\n";
          });
#endif
    // Create the variant and add it to the output list.
    std::vector<TreePatternNode*> NewChildren;
    for (unsigned i = 0, e = ChildVariants.size(); i != e; ++i)
      NewChildren.push_back(ChildVariants[i][Idxs[i]]);
    TreePatternNode *R = new TreePatternNode(Orig->getOperator(), NewChildren);
    
    // Copy over properties.
    R->setName(Orig->getName());
    R->setPredicateFns(Orig->getPredicateFns());
    R->setTransformFn(Orig->getTransformFn());
    R->setType(Orig->getExtType());
    
    // If this pattern cannot match, do not include it as a variant.
    std::string ErrString;
    if (!R->canPatternMatch(ErrString, CDP)) {
      delete R;
    } else {
      bool AlreadyExists = false;
      
      // Scan to see if this pattern has already been emitted.  We can get
      // duplication due to things like commuting:
      //   (and GPRC:$a, GPRC:$b) -> (and GPRC:$b, GPRC:$a)
      // which are the same pattern.  Ignore the dups.
      for (unsigned i = 0, e = OutVariants.size(); i != e; ++i)
        if (R->isIsomorphicTo(OutVariants[i], DepVars)) {
          AlreadyExists = true;
          break;
        }
      
      if (AlreadyExists)
        delete R;
      else
        OutVariants.push_back(R);
    }
    
    // Increment indices to the next permutation by incrementing the
    // indicies from last index backward, e.g., generate the sequence
    // [0, 0], [0, 1], [1, 0], [1, 1].
    int IdxsIdx;
    for (IdxsIdx = Idxs.size() - 1; IdxsIdx >= 0; --IdxsIdx) {
      if (++Idxs[IdxsIdx] == ChildVariants[IdxsIdx].size())
        Idxs[IdxsIdx] = 0;
      else
        break;
    }
    NotDone = (IdxsIdx >= 0);
  } while (NotDone);
}

/// CombineChildVariants - A helper function for binary operators.
///
static void CombineChildVariants(TreePatternNode *Orig, 
                                 const std::vector<TreePatternNode*> &LHS,
                                 const std::vector<TreePatternNode*> &RHS,
                                 std::vector<TreePatternNode*> &OutVariants,
                                 CodeGenDAGPatterns &CDP,
                                 const MultipleUseVarSet &DepVars) {
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.push_back(LHS);
  ChildVariants.push_back(RHS);
  CombineChildVariants(Orig, ChildVariants, OutVariants, CDP, DepVars);
}  


static void GatherChildrenOfAssociativeOpcode(TreePatternNode *N,
                                     std::vector<TreePatternNode *> &Children) {
  assert(N->getNumChildren()==2 &&"Associative but doesn't have 2 children!");
  Record *Operator = N->getOperator();
  
  // Only permit raw nodes.
  if (!N->getName().empty() || !N->getPredicateFns().empty() ||
      N->getTransformFn()) {
    Children.push_back(N);
    return;
  }

  if (N->getChild(0)->isLeaf() || N->getChild(0)->getOperator() != Operator)
    Children.push_back(N->getChild(0));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(0), Children);

  if (N->getChild(1)->isLeaf() || N->getChild(1)->getOperator() != Operator)
    Children.push_back(N->getChild(1));
  else
    GatherChildrenOfAssociativeOpcode(N->getChild(1), Children);
}

/// GenerateVariantsOf - Given a pattern N, generate all permutations we can of
/// the (potentially recursive) pattern by using algebraic laws.
///
static void GenerateVariantsOf(TreePatternNode *N,
                               std::vector<TreePatternNode*> &OutVariants,
                               CodeGenDAGPatterns &CDP,
                               const MultipleUseVarSet &DepVars) {
  // We cannot permute leaves.
  if (N->isLeaf()) {
    OutVariants.push_back(N);
    return;
  }

  // Look up interesting info about the node.
  const SDNodeInfo &NodeInfo = CDP.getSDNodeInfo(N->getOperator());

  // If this node is associative, re-associate.
  if (NodeInfo.hasProperty(SDNPAssociative)) {
    // Re-associate by pulling together all of the linked operators 
    std::vector<TreePatternNode*> MaximalChildren;
    GatherChildrenOfAssociativeOpcode(N, MaximalChildren);

    // Only handle child sizes of 3.  Otherwise we'll end up trying too many
    // permutations.
    if (MaximalChildren.size() == 3) {
      // Find the variants of all of our maximal children.
      std::vector<TreePatternNode*> AVariants, BVariants, CVariants;
      GenerateVariantsOf(MaximalChildren[0], AVariants, CDP, DepVars);
      GenerateVariantsOf(MaximalChildren[1], BVariants, CDP, DepVars);
      GenerateVariantsOf(MaximalChildren[2], CVariants, CDP, DepVars);
      
      // There are only two ways we can permute the tree:
      //   (A op B) op C    and    A op (B op C)
      // Within these forms, we can also permute A/B/C.
      
      // Generate legal pair permutations of A/B/C.
      std::vector<TreePatternNode*> ABVariants;
      std::vector<TreePatternNode*> BAVariants;
      std::vector<TreePatternNode*> ACVariants;
      std::vector<TreePatternNode*> CAVariants;
      std::vector<TreePatternNode*> BCVariants;
      std::vector<TreePatternNode*> CBVariants;
      CombineChildVariants(N, AVariants, BVariants, ABVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, AVariants, BAVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, CVariants, ACVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, AVariants, CAVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, CVariants, BCVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, BVariants, CBVariants, CDP, DepVars);

      // Combine those into the result: (x op x) op x
      CombineChildVariants(N, ABVariants, CVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BAVariants, CVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, ACVariants, BVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CAVariants, BVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BCVariants, AVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CBVariants, AVariants, OutVariants, CDP, DepVars);

      // Combine those into the result: x op (x op x)
      CombineChildVariants(N, CVariants, ABVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, CVariants, BAVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, ACVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, BVariants, CAVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, BCVariants, OutVariants, CDP, DepVars);
      CombineChildVariants(N, AVariants, CBVariants, OutVariants, CDP, DepVars);
      return;
    }
  }
  
  // Compute permutations of all children.
  std::vector<std::vector<TreePatternNode*> > ChildVariants;
  ChildVariants.resize(N->getNumChildren());
  for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i)
    GenerateVariantsOf(N->getChild(i), ChildVariants[i], CDP, DepVars);

  // Build all permutations based on how the children were formed.
  CombineChildVariants(N, ChildVariants, OutVariants, CDP, DepVars);

  // If this node is commutative, consider the commuted order.
  bool isCommIntrinsic = N->isCommutativeIntrinsic(CDP);
  if (NodeInfo.hasProperty(SDNPCommutative) || isCommIntrinsic) {
    assert((N->getNumChildren()==2 || isCommIntrinsic) &&
           "Commutative but doesn't have 2 children!");
    // Don't count children which are actually register references.
    unsigned NC = 0;
    for (unsigned i = 0, e = N->getNumChildren(); i != e; ++i) {
      TreePatternNode *Child = N->getChild(i);
      if (Child->isLeaf())
        if (DefInit *DI = dynamic_cast<DefInit*>(Child->getLeafValue())) {
          Record *RR = DI->getDef();
          if (RR->isSubClassOf("Register"))
            continue;
        }
      NC++;
    }
    // Consider the commuted order.
    if (isCommIntrinsic) {
      // Commutative intrinsic. First operand is the intrinsic id, 2nd and 3rd
      // operands are the commutative operands, and there might be more operands
      // after those.
      assert(NC >= 3 &&
             "Commutative intrinsic should have at least 3 childrean!");
      std::vector<std::vector<TreePatternNode*> > Variants;
      Variants.push_back(ChildVariants[0]); // Intrinsic id.
      Variants.push_back(ChildVariants[2]);
      Variants.push_back(ChildVariants[1]);
      for (unsigned i = 3; i != NC; ++i)
        Variants.push_back(ChildVariants[i]);
      CombineChildVariants(N, Variants, OutVariants, CDP, DepVars);
    } else if (NC == 2)
      CombineChildVariants(N, ChildVariants[1], ChildVariants[0],
                           OutVariants, CDP, DepVars);
  }
}


// GenerateVariants - Generate variants.  For example, commutative patterns can
// match multiple ways.  Add them to PatternsToMatch as well.
void CodeGenDAGPatterns::GenerateVariants() {
  DEBUG(errs() << "Generating instruction variants.\n");
  
  // Loop over all of the patterns we've collected, checking to see if we can
  // generate variants of the instruction, through the exploitation of
  // identities.  This permits the target to provide aggressive matching without
  // the .td file having to contain tons of variants of instructions.
  //
  // Note that this loop adds new patterns to the PatternsToMatch list, but we
  // intentionally do not reconsider these.  Any variants of added patterns have
  // already been added.
  //
  for (unsigned i = 0, e = PatternsToMatch.size(); i != e; ++i) {
    MultipleUseVarSet             DepVars;
    std::vector<TreePatternNode*> Variants;
    FindDepVars(PatternsToMatch[i].getSrcPattern(), DepVars);
    DEBUG(errs() << "Dependent/multiply used variables: ");
    DEBUG(DumpDepVars(DepVars));
    DEBUG(errs() << "\n");
    GenerateVariantsOf(PatternsToMatch[i].getSrcPattern(), Variants, *this, DepVars);

    assert(!Variants.empty() && "Must create at least original variant!");
    Variants.erase(Variants.begin());  // Remove the original pattern.

    if (Variants.empty())  // No variants for this pattern.
      continue;

    DEBUG(errs() << "FOUND VARIANTS OF: ";
          PatternsToMatch[i].getSrcPattern()->dump();
          errs() << "\n");

    for (unsigned v = 0, e = Variants.size(); v != e; ++v) {
      TreePatternNode *Variant = Variants[v];

      DEBUG(errs() << "  VAR#" << v <<  ": ";
            Variant->dump();
            errs() << "\n");
      
      // Scan to see if an instruction or explicit pattern already matches this.
      bool AlreadyExists = false;
      for (unsigned p = 0, e = PatternsToMatch.size(); p != e; ++p) {
        // Skip if the top level predicates do not match.
        if (PatternsToMatch[i].getPredicates() !=
            PatternsToMatch[p].getPredicates())
          continue;
        // Check to see if this variant already exists.
        if (Variant->isIsomorphicTo(PatternsToMatch[p].getSrcPattern(), DepVars)) {
          DEBUG(errs() << "  *** ALREADY EXISTS, ignoring variant.\n");
          AlreadyExists = true;
          break;
        }
      }
      // If we already have it, ignore the variant.
      if (AlreadyExists) continue;

      // Otherwise, add it to the list of patterns we have.
      PatternsToMatch.
        push_back(PatternToMatch(PatternsToMatch[i].getPredicates(),
                                 Variant, PatternsToMatch[i].getDstPattern(),
                                 PatternsToMatch[i].getDstRegs(),
                                 PatternsToMatch[i].getAddedComplexity(),
                                 Record::getNewUID()));
    }

    DEBUG(errs() << "\n");
  }
}


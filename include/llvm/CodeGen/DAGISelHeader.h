//==-llvm/CodeGen/DAGISelHeader.h - Common DAG ISel definitions  -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides definitions of the common, target-independent methods and 
// data, which is used by SelectionDAG-based instruction selectors.
//
// *** NOTE: This file is #included into the middle of the target
// instruction selector class.  These functions are really methods.
// This is a little awkward, but it allows this code to be shared
// by all the targets while still being able to call into
// target-specific code without using a virtual function call.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DAGISEL_HEADER_H
#define LLVM_CODEGEN_DAGISEL_HEADER_H

/// ISelPosition - Node iterator marking the current position of
/// instruction selection as it procedes through the topologically-sorted
/// node list.
SelectionDAG::allnodes_iterator ISelPosition;

/// ChainNotReachable - Returns true if Chain does not reach Op.
static bool ChainNotReachable(SDNode *Chain, SDNode *Op) {
  if (Chain->getOpcode() == ISD::EntryToken)
    return true;
  if (Chain->getOpcode() == ISD::TokenFactor)
    return false;
  if (Chain->getNumOperands() > 0) {
    SDValue C0 = Chain->getOperand(0);
    if (C0.getValueType() == MVT::Other)
      return C0.getNode() != Op && ChainNotReachable(C0.getNode(), Op);
  }
  return true;
}

/// IsChainCompatible - Returns true if Chain is Op or Chain does not reach Op.
/// This is used to ensure that there are no nodes trapped between Chain, which
/// is the first chain node discovered in a pattern and Op, a later node, that
/// will not be selected into the pattern.
static bool IsChainCompatible(SDNode *Chain, SDNode *Op) {
  return Chain == Op || ChainNotReachable(Chain, Op);
}


/// ISelUpdater - helper class to handle updates of the 
/// instruciton selection graph.
class VISIBILITY_HIDDEN ISelUpdater : public SelectionDAG::DAGUpdateListener {
  SelectionDAG::allnodes_iterator &ISelPosition;
public:
  explicit ISelUpdater(SelectionDAG::allnodes_iterator &isp)
    : ISelPosition(isp) {}
  
  /// NodeDeleted - Handle nodes deleted from the graph. If the
  /// node being deleted is the current ISelPosition node, update
  /// ISelPosition.
  ///
  virtual void NodeDeleted(SDNode *N, SDNode *E) {
    if (ISelPosition == SelectionDAG::allnodes_iterator(N))
      ++ISelPosition;
  }

  /// NodeUpdated - Ignore updates for now.
  virtual void NodeUpdated(SDNode *N) {}
};

/// ReplaceUses - replace all uses of the old node F with the use
/// of the new node T.
DISABLE_INLINE void ReplaceUses(SDValue F, SDValue T) {
  ISelUpdater ISU(ISelPosition);
  CurDAG->ReplaceAllUsesOfValueWith(F, T, &ISU);
}

/// ReplaceUses - replace all uses of the old nodes F with the use
/// of the new nodes T.
DISABLE_INLINE void ReplaceUses(const SDValue *F, const SDValue *T,
                                unsigned Num) {
  ISelUpdater ISU(ISelPosition);
  CurDAG->ReplaceAllUsesOfValuesWith(F, T, Num, &ISU);
}

/// ReplaceUses - replace all uses of the old node F with the use
/// of the new node T.
DISABLE_INLINE void ReplaceUses(SDNode *F, SDNode *T) {
  ISelUpdater ISU(ISelPosition);
  CurDAG->ReplaceAllUsesWith(F, T, &ISU);
}

/// SelectRoot - Top level entry to DAG instruction selector.
/// Selects instructions starting at the root of the current DAG.
void SelectRoot(SelectionDAG &DAG) {
  SelectRootInit();

  // Create a dummy node (which is not added to allnodes), that adds
  // a reference to the root node, preventing it from being deleted,
  // and tracking any changes of the root.
  HandleSDNode Dummy(CurDAG->getRoot());
  ISelPosition = SelectionDAG::allnodes_iterator(CurDAG->getRoot().getNode());
  ++ISelPosition;

  // The AllNodes list is now topological-sorted. Visit the
  // nodes by starting at the end of the list (the root of the
  // graph) and preceding back toward the beginning (the entry
  // node).
  while (ISelPosition != CurDAG->allnodes_begin()) {
    SDNode *Node = --ISelPosition;
    // Skip dead nodes. DAGCombiner is expected to eliminate all dead nodes,
    // but there are currently some corner cases that it misses. Also, this
    // makes it theoretically possible to disable the DAGCombiner.
    if (Node->use_empty())
      continue;

    SDNode *ResNode = Select(Node);
    // If node should not be replaced, continue with the next one.
    if (ResNode == Node)
      continue;
    // Replace node.
    if (ResNode)
      ReplaceUses(Node, ResNode);

    // If after the replacement this node is not used any more,
    // remove this dead node.
    if (Node->use_empty()) { // Don't delete EntryToken, etc.
      ISelUpdater ISU(ISelPosition);
      CurDAG->RemoveDeadNode(Node, &ISU);
    }
  }

  CurDAG->setRoot(Dummy.getValue());
}


/// CheckInteger - Return true if the specified node is not a ConstantSDNode or
/// if it doesn't have the specified value.
static bool CheckInteger(SDValue V, int64_t Val) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(V);
  return C == 0 || C->getSExtValue() != Val;
}

/// CheckAndImmediate - Check to see if the specified node is an and with an
/// immediate returning true on failure.
///
/// FIXME: Inline this gunk into CheckAndMask.
bool CheckAndImmediate(SDValue V, int64_t Val) {
  if (V->getOpcode() == ISD::AND)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(V->getOperand(1)))
      if (CheckAndMask(V.getOperand(0), C, Val))
        return false;
  return true;
}

/// CheckOrImmediate - Check to see if the specified node is an or with an
/// immediate returning true on failure.
///
/// FIXME: Inline this gunk into CheckOrMask.
bool CheckOrImmediate(SDValue V, int64_t Val) {
  if (V->getOpcode() == ISD::OR)
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(V->getOperand(1)))
      if (CheckOrMask(V.getOperand(0), C, Val))
        return false;
  return true;
}

// These functions are marked always inline so that Idx doesn't get pinned to
// the stack.
ALWAYS_INLINE static int8_t
GetInt1(const unsigned char *MatcherTable, unsigned &Idx) {
  return MatcherTable[Idx++];
}

ALWAYS_INLINE static int16_t
GetInt2(const unsigned char *MatcherTable, unsigned &Idx) {
  int16_t Val = (uint8_t)GetInt1(MatcherTable, Idx);
  Val |= int16_t(GetInt1(MatcherTable, Idx)) << 8;
  return Val;
}

/// GetVBR - decode a vbr encoding whose top bit is set.
ALWAYS_INLINE static uint64_t
GetVBR(uint64_t Val, const unsigned char *MatcherTable, unsigned &Idx) {
  assert(Val >= 128 && "Not a VBR");
  Val &= 127;  // Remove first vbr bit.
  
  unsigned Shift = 7;
  uint64_t NextBits;
  do {
    NextBits = GetInt1(MatcherTable, Idx);
    Val |= (NextBits&127) << Shift;
    Shift += 7;
  } while (NextBits & 128);
  
  return Val;
}

/// UpdateChainsAndFlags - When a match is complete, this method updates uses of
/// interior flag and chain results to use the new flag and chain results.
void UpdateChainsAndFlags(SDNode *NodeToMatch, SDValue InputChain,
                          const SmallVectorImpl<SDNode*> &ChainNodesMatched,
                          SDValue InputFlag,
                          const SmallVectorImpl<SDNode*>&FlagResultNodesMatched,
                          bool isMorphNodeTo) {
  // Now that all the normal results are replaced, we replace the chain and
  // flag results if present.
  if (!ChainNodesMatched.empty()) {
    assert(InputChain.getNode() != 0 &&
           "Matched input chains but didn't produce a chain");
    // Loop over all of the nodes we matched that produced a chain result.
    // Replace all the chain results with the final chain we ended up with.
    for (unsigned i = 0, e = ChainNodesMatched.size(); i != e; ++i) {
      SDNode *ChainNode = ChainNodesMatched[i];
      
      // Don't replace the results of the root node if we're doing a
      // MorphNodeTo.
      if (ChainNode == NodeToMatch && isMorphNodeTo)
        continue;
      
      SDValue ChainVal = SDValue(ChainNode, ChainNode->getNumValues()-1);
      if (ChainVal.getValueType() == MVT::Flag)
        ChainVal = ChainVal.getValue(ChainVal->getNumValues()-2);
      assert(ChainVal.getValueType() == MVT::Other && "Not a chain?");
      ReplaceUses(ChainVal, InputChain);
    }
  }
  
  // If the result produces a flag, update any flag results in the matched
  // pattern with the flag result.
  if (InputFlag.getNode() != 0) {
    // Handle any interior nodes explicitly marked.
    for (unsigned i = 0, e = FlagResultNodesMatched.size(); i != e; ++i) {
      SDNode *FRN = FlagResultNodesMatched[i];
      assert(FRN->getValueType(FRN->getNumValues()-1) == MVT::Flag &&
             "Doesn't have a flag result");
      ReplaceUses(SDValue(FRN, FRN->getNumValues()-1), InputFlag);
    }
  }
  
  DEBUG(errs() << "ISEL: Match complete!\n");
}



/// getNumFixedFromVariadicInfo - Transform an EmitNode flags word into the
/// number of fixed arity values that should be skipped when copying from the
/// root.
static inline int getNumFixedFromVariadicInfo(unsigned Flags) {
  return ((Flags&OPFL_VariadicInfo) >> 4)-1;
}

struct MatchScope {
  /// FailIndex - If this match fails, this is the index to continue with.
  unsigned FailIndex;
  
  /// NodeStack - The node stack when the scope was formed.
  SmallVector<SDValue, 4> NodeStack;
  
  /// NumRecordedNodes - The number of recorded nodes when the scope was formed.
  unsigned NumRecordedNodes;
  
  /// NumMatchedMemRefs - The number of matched memref entries.
  unsigned NumMatchedMemRefs;
  
  /// InputChain/InputFlag - The current chain/flag 
  SDValue InputChain, InputFlag;

  /// HasChainNodesMatched - True if the ChainNodesMatched list is non-empty.
  bool HasChainNodesMatched, HasFlagResultNodesMatched;
};

SDNode *SelectCodeCommon(SDNode *NodeToMatch, const unsigned char *MatcherTable,
                         unsigned TableSize) {
  // FIXME: Should these even be selected?  Handle these cases in the caller?
  switch (NodeToMatch->getOpcode()) {
  default:
    break;
  case ISD::EntryToken:       // These nodes remain the same.
  case ISD::BasicBlock:
  case ISD::Register:
  case ISD::HANDLENODE:
  case ISD::TargetConstant:
  case ISD::TargetConstantFP:
  case ISD::TargetConstantPool:
  case ISD::TargetFrameIndex:
  case ISD::TargetExternalSymbol:
  case ISD::TargetBlockAddress:
  case ISD::TargetJumpTable:
  case ISD::TargetGlobalTLSAddress:
  case ISD::TargetGlobalAddress:
  case ISD::TokenFactor:
  case ISD::CopyFromReg:
  case ISD::CopyToReg:
    return 0;
  case ISD::AssertSext:
  case ISD::AssertZext:
    ReplaceUses(SDValue(NodeToMatch, 0), NodeToMatch->getOperand(0));
    return 0;
  case ISD::INLINEASM: return Select_INLINEASM(NodeToMatch);
  case ISD::EH_LABEL:  return Select_EH_LABEL(NodeToMatch);
  case ISD::UNDEF:     return Select_UNDEF(NodeToMatch);
  }
  
  assert(!NodeToMatch->isMachineOpcode() && "Node already selected!");

  // Set up the node stack with NodeToMatch as the only node on the stack.
  SmallVector<SDValue, 8> NodeStack;
  SDValue N = SDValue(NodeToMatch, 0);
  NodeStack.push_back(N);

  // MatchScopes - Scopes used when matching, if a match failure happens, this
  // indicates where to continue checking.
  SmallVector<MatchScope, 8> MatchScopes;
  
  // RecordedNodes - This is the set of nodes that have been recorded by the
  // state machine.
  SmallVector<SDValue, 8> RecordedNodes;
  
  // MatchedMemRefs - This is the set of MemRef's we've seen in the input
  // pattern.
  SmallVector<MachineMemOperand*, 2> MatchedMemRefs;
  
  // These are the current input chain and flag for use when generating nodes.
  // Various Emit operations change these.  For example, emitting a copytoreg
  // uses and updates these.
  SDValue InputChain, InputFlag;
  
  // ChainNodesMatched - If a pattern matches nodes that have input/output
  // chains, the OPC_EmitMergeInputChains operation is emitted which indicates
  // which ones they are.  The result is captured into this list so that we can
  // update the chain results when the pattern is complete.
  SmallVector<SDNode*, 3> ChainNodesMatched;
  SmallVector<SDNode*, 3> FlagResultNodesMatched;
  
  DEBUG(errs() << "ISEL: Starting pattern match on root node: ";
        NodeToMatch->dump(CurDAG);
        errs() << '\n');
  
  // Interpreter starts at opcode #0.
  unsigned MatcherIndex = 0;
  while (1) {
    assert(MatcherIndex < TableSize && "Invalid index");
    BuiltinOpcodes Opcode = (BuiltinOpcodes)MatcherTable[MatcherIndex++];
    switch (Opcode) {
    case OPC_Scope: {
      unsigned NumToSkip = MatcherTable[MatcherIndex++];
      if (NumToSkip & 128)
        NumToSkip = GetVBR(NumToSkip, MatcherTable, MatcherIndex);
      assert(NumToSkip != 0 &&
             "First entry of OPC_Scope shouldn't be 0, scope has no children?");

      // Push a MatchScope which indicates where to go if the first child fails
      // to match.
      MatchScope NewEntry;
      NewEntry.FailIndex = MatcherIndex+NumToSkip;
      NewEntry.NodeStack.append(NodeStack.begin(), NodeStack.end());
      NewEntry.NumRecordedNodes = RecordedNodes.size();
      NewEntry.NumMatchedMemRefs = MatchedMemRefs.size();
      NewEntry.InputChain = InputChain;
      NewEntry.InputFlag = InputFlag;
      NewEntry.HasChainNodesMatched = !ChainNodesMatched.empty();
      NewEntry.HasFlagResultNodesMatched = !FlagResultNodesMatched.empty();
      MatchScopes.push_back(NewEntry);
      continue;
    }
    case OPC_RecordNode:
      // Remember this node, it may end up being an operand in the pattern.
      RecordedNodes.push_back(N);
      continue;
        
    case OPC_RecordChild0: case OPC_RecordChild1:
    case OPC_RecordChild2: case OPC_RecordChild3:
    case OPC_RecordChild4: case OPC_RecordChild5:
    case OPC_RecordChild6: case OPC_RecordChild7: {
      unsigned ChildNo = Opcode-OPC_RecordChild0;
      if (ChildNo >= N.getNumOperands())
        break;  // Match fails if out of range child #.

      RecordedNodes.push_back(N->getOperand(ChildNo));
      continue;
    }
    case OPC_RecordMemRef:
      MatchedMemRefs.push_back(cast<MemSDNode>(N)->getMemOperand());
      continue;
        
    case OPC_CaptureFlagInput:
      // If the current node has an input flag, capture it in InputFlag.
      if (N->getNumOperands() != 0 &&
          N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Flag)
        InputFlag = N->getOperand(N->getNumOperands()-1);
      continue;
        
    case OPC_MoveChild: {
      unsigned ChildNo = MatcherTable[MatcherIndex++];
      if (ChildNo >= N.getNumOperands())
        break;  // Match fails if out of range child #.
      N = N.getOperand(ChildNo);
      NodeStack.push_back(N);
      continue;
    }
        
    case OPC_MoveParent:
      // Pop the current node off the NodeStack.
      NodeStack.pop_back();
      assert(!NodeStack.empty() && "Node stack imbalance!");
      N = NodeStack.back();  
      continue;
     
    case OPC_CheckSame: {
      // Accept if it is exactly the same as a previously recorded node.
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      if (N != RecordedNodes[RecNo]) break;
      continue;
    }
    case OPC_CheckPatternPredicate:
      if (!CheckPatternPredicate(MatcherTable[MatcherIndex++])) break;
      continue;
    case OPC_CheckPredicate:
      if (!CheckNodePredicate(N.getNode(), MatcherTable[MatcherIndex++])) break;
      continue;
    case OPC_CheckComplexPat:
      if (!CheckComplexPattern(NodeToMatch, N, 
                               MatcherTable[MatcherIndex++], RecordedNodes))
        break;
      continue;
    case OPC_CheckOpcode:
      if (N->getOpcode() != MatcherTable[MatcherIndex++]) break;
      continue;
        
    case OPC_CheckMultiOpcode: {
      unsigned NumOps = MatcherTable[MatcherIndex++];
      bool OpcodeEquals = false;
      for (unsigned i = 0; i != NumOps; ++i)
        OpcodeEquals |= N->getOpcode() == MatcherTable[MatcherIndex++];
      if (!OpcodeEquals) break;
      continue;
    }
        
    case OPC_CheckType: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      if (N.getValueType() != VT) {
        // Handle the case when VT is iPTR.
        if (VT != MVT::iPTR || N.getValueType() != TLI.getPointerTy())
          break;
      }
      continue;
    }
    case OPC_CheckChild0Type: case OPC_CheckChild1Type:
    case OPC_CheckChild2Type: case OPC_CheckChild3Type:
    case OPC_CheckChild4Type: case OPC_CheckChild5Type:
    case OPC_CheckChild6Type: case OPC_CheckChild7Type: {
      unsigned ChildNo = Opcode-OPC_CheckChild0Type;
      if (ChildNo >= N.getNumOperands())
        break;  // Match fails if out of range child #.
      
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      EVT ChildVT = N.getOperand(ChildNo).getValueType();
      if (ChildVT != VT) {
        // Handle the case when VT is iPTR.
        if (VT != MVT::iPTR || ChildVT != TLI.getPointerTy())
          break;
      }
      continue;
    }
    case OPC_CheckCondCode:
      if (cast<CondCodeSDNode>(N)->get() !=
          (ISD::CondCode)MatcherTable[MatcherIndex++]) break;
      continue;
    case OPC_CheckValueType: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      if (cast<VTSDNode>(N)->getVT() != VT) {
        // Handle the case when VT is iPTR.
        if (VT != MVT::iPTR || cast<VTSDNode>(N)->getVT() != TLI.getPointerTy())
          break;
      }
      continue;
    }
    case OPC_CheckInteger: {
      int64_t Val = MatcherTable[MatcherIndex++];
      if (Val & 128)
        Val = GetVBR(Val, MatcherTable, MatcherIndex);
      if (CheckInteger(N, Val)) break;
      continue;
    }        
    case OPC_CheckAndImm: {
      int64_t Val = MatcherTable[MatcherIndex++];
      if (Val & 128)
        Val = GetVBR(Val, MatcherTable, MatcherIndex);
      if (CheckAndImmediate(N, Val)) break;
      continue;
    }
    case OPC_CheckOrImm: {
      int64_t Val = MatcherTable[MatcherIndex++];
      if (Val & 128)
        Val = GetVBR(Val, MatcherTable, MatcherIndex);
      if (CheckOrImmediate(N, Val)) break;
      continue;
    }
        
    case OPC_CheckFoldableChainNode: {
      assert(NodeStack.size() != 1 && "No parent node");
      // Verify that all intermediate nodes between the root and this one have
      // a single use.
      bool HasMultipleUses = false;
      for (unsigned i = 1, e = NodeStack.size()-1; i != e; ++i)
        if (!NodeStack[i].hasOneUse()) {
          HasMultipleUses = true;
          break;
        }
      if (HasMultipleUses) break;

      // Check to see that the target thinks this is profitable to fold and that
      // we can fold it without inducing cycles in the graph.
      if (!IsProfitableToFold(N, NodeStack[NodeStack.size()-2].getNode(),
                              NodeToMatch) ||
          !IsLegalToFold(N, NodeStack[NodeStack.size()-2].getNode(),
                         NodeToMatch))
        break;
      
      continue;
    }
    case OPC_CheckChainCompatible: {
      unsigned PrevNode = MatcherTable[MatcherIndex++];
      assert(PrevNode < RecordedNodes.size() && "Invalid CheckChainCompatible");
      SDValue PrevChainedNode = RecordedNodes[PrevNode];
      SDValue ThisChainedNode = RecordedNodes.back();
      
      // We have two nodes with chains, verify that their input chains are good.
      assert(PrevChainedNode.getOperand(0).getValueType() == MVT::Other &&
             ThisChainedNode.getOperand(0).getValueType() == MVT::Other &&
             "Invalid chained nodes");
      
      if (!IsChainCompatible(// Input chain of the previous node.
                             PrevChainedNode.getOperand(0).getNode(),
                             // Node with chain.
                             ThisChainedNode.getNode()))
        break;
      continue;
    }
        
    case OPC_EmitInteger: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      int64_t Val = MatcherTable[MatcherIndex++];
      if (Val & 128)
        Val = GetVBR(Val, MatcherTable, MatcherIndex);
      RecordedNodes.push_back(CurDAG->getTargetConstant(Val, VT));
      continue;
    }
    case OPC_EmitRegister: {
      MVT::SimpleValueType VT =
        (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
      unsigned RegNo = MatcherTable[MatcherIndex++];
      RecordedNodes.push_back(CurDAG->getRegister(RegNo, VT));
      continue;
    }
        
    case OPC_EmitConvertToTarget:  {
      // Convert from IMM/FPIMM to target version.
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      SDValue Imm = RecordedNodes[RecNo];

      if (Imm->getOpcode() == ISD::Constant) {
        int64_t Val = cast<ConstantSDNode>(Imm)->getZExtValue();
        Imm = CurDAG->getTargetConstant(Val, Imm.getValueType());
      } else if (Imm->getOpcode() == ISD::ConstantFP) {
        const ConstantFP *Val=cast<ConstantFPSDNode>(Imm)->getConstantFPValue();
        Imm = CurDAG->getTargetConstantFP(*Val, Imm.getValueType());
      }
      
      RecordedNodes.push_back(Imm);
      continue;
    }
        
    case OPC_EmitMergeInputChains: {
      assert(InputChain.getNode() == 0 &&
             "EmitMergeInputChains should be the first chain producing node");
      // This node gets a list of nodes we matched in the input that have
      // chains.  We want to token factor all of the input chains to these nodes
      // together.  However, if any of the input chains is actually one of the
      // nodes matched in this pattern, then we have an intra-match reference.
      // Ignore these because the newly token factored chain should not refer to
      // the old nodes.
      unsigned NumChains = MatcherTable[MatcherIndex++];
      assert(NumChains != 0 && "Can't TF zero chains");

      assert(ChainNodesMatched.empty() &&
             "Should only have one EmitMergeInputChains per match");

      // Handle the first chain.
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      ChainNodesMatched.push_back(RecordedNodes[RecNo].getNode());
      
      // If the chained node is not the root, we can't fold it if it has
      // multiple uses.
      // FIXME: What if other value results of the node have uses not matched by
      // this pattern?
      if (ChainNodesMatched.back() != NodeToMatch &&
          !RecordedNodes[RecNo].hasOneUse()) {
        ChainNodesMatched.clear();
        break;
      }
      
      // The common case here is that we have exactly one chain, which is really
      // cheap to handle, just do it.
      if (NumChains == 1) {
        InputChain = RecordedNodes[RecNo].getOperand(0);
        assert(InputChain.getValueType() == MVT::Other && "Not a chain");
        continue;
      }
      
      // Read all of the chained nodes.
      for (unsigned i = 1; i != NumChains; ++i) {
        RecNo = MatcherTable[MatcherIndex++];
        assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
        ChainNodesMatched.push_back(RecordedNodes[RecNo].getNode());
        
        // FIXME: What if other value results of the node have uses not matched by
        // this pattern?
        if (ChainNodesMatched.back() != NodeToMatch &&
            !RecordedNodes[RecNo].hasOneUse()) {
          ChainNodesMatched.clear();
          break;
        }
      }

      // Walk all the chained nodes, adding the input chains if they are not in
      // ChainedNodes (and this, not in the matched pattern).  This is an N^2
      // algorithm, but # chains is usually 2 here, at most 3 for MSP430.
      SmallVector<SDValue, 3> InputChains;
      for (unsigned i = 0, e = ChainNodesMatched.size(); i != e; ++i) {
        SDValue InChain = ChainNodesMatched[i]->getOperand(0);
        assert(InChain.getValueType() == MVT::Other && "Not a chain");
        bool Invalid = false;
        for (unsigned j = 0; j != e; ++j)
          Invalid |= ChainNodesMatched[j] == InChain.getNode();
        if (!Invalid)
          InputChains.push_back(InChain);
      }

      SDValue Res;
      if (InputChains.size() == 1)
        InputChain = InputChains[0];
      else
        InputChain = CurDAG->getNode(ISD::TokenFactor,
                                     NodeToMatch->getDebugLoc(), MVT::Other,
                                     &InputChains[0], InputChains.size());
      continue;
    }
        
    case OPC_EmitCopyToReg: {
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      unsigned DestPhysReg = MatcherTable[MatcherIndex++];
      
      if (InputChain.getNode() == 0)
        InputChain = CurDAG->getEntryNode();
      
      InputChain = CurDAG->getCopyToReg(InputChain, NodeToMatch->getDebugLoc(),
                                        DestPhysReg, RecordedNodes[RecNo],
                                        InputFlag);
      
      InputFlag = InputChain.getValue(1);
      continue;
    }
        
    case OPC_EmitNodeXForm: {
      unsigned XFormNo = MatcherTable[MatcherIndex++];
      unsigned RecNo = MatcherTable[MatcherIndex++];
      assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
      RecordedNodes.push_back(RunSDNodeXForm(RecordedNodes[RecNo], XFormNo));
      continue;
    }
        
    case OPC_EmitNode:
    case OPC_MorphNodeTo: {
      uint16_t TargetOpc = GetInt2(MatcherTable, MatcherIndex);
      unsigned EmitNodeInfo = MatcherTable[MatcherIndex++];
      // Get the result VT list.
      unsigned NumVTs = MatcherTable[MatcherIndex++];
      SmallVector<EVT, 4> VTs;
      for (unsigned i = 0; i != NumVTs; ++i) {
        MVT::SimpleValueType VT =
          (MVT::SimpleValueType)MatcherTable[MatcherIndex++];
        if (VT == MVT::iPTR) VT = TLI.getPointerTy().SimpleTy;
        VTs.push_back(VT);
      }
      
      if (EmitNodeInfo & OPFL_Chain)
        VTs.push_back(MVT::Other);
      if (EmitNodeInfo & OPFL_FlagOutput)
        VTs.push_back(MVT::Flag);
      
      // FIXME: Use faster version for the common 'one VT' case?
      SDVTList VTList = CurDAG->getVTList(VTs.data(), VTs.size());

      // Get the operand list.
      unsigned NumOps = MatcherTable[MatcherIndex++];
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0; i != NumOps; ++i) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        if (RecNo & 128)
          RecNo = GetVBR(RecNo, MatcherTable, MatcherIndex);
        
        assert(RecNo < RecordedNodes.size() && "Invalid EmitNode");
        Ops.push_back(RecordedNodes[RecNo]);
      }
      
      // If there are variadic operands to add, handle them now.
      if (EmitNodeInfo & OPFL_VariadicInfo) {
        // Determine the start index to copy from.
        unsigned FirstOpToCopy = getNumFixedFromVariadicInfo(EmitNodeInfo);
        FirstOpToCopy += (EmitNodeInfo & OPFL_Chain) ? 1 : 0;
        assert(NodeToMatch->getNumOperands() >= FirstOpToCopy &&
               "Invalid variadic node");
        // Copy all of the variadic operands, not including a potential flag
        // input.
        for (unsigned i = FirstOpToCopy, e = NodeToMatch->getNumOperands();
             i != e; ++i) {
          SDValue V = NodeToMatch->getOperand(i);
          if (V.getValueType() == MVT::Flag) break;
          Ops.push_back(V);
        }
      }
      
      // If this has chain/flag inputs, add them.
      if (EmitNodeInfo & OPFL_Chain)
        Ops.push_back(InputChain);
      if ((EmitNodeInfo & OPFL_FlagInput) && InputFlag.getNode() != 0)
        Ops.push_back(InputFlag);
      
      // Create the node.
      SDNode *Res = 0;
      if (Opcode != OPC_MorphNodeTo) {
        // If this is a normal EmitNode command, just create the new node and
        // add the results to the RecordedNodes list.
        Res = CurDAG->getMachineNode(TargetOpc, NodeToMatch->getDebugLoc(),
                                     VTList, Ops.data(), Ops.size());
        
        // Add all the non-flag/non-chain results to the RecordedNodes list.
        for (unsigned i = 0, e = VTs.size(); i != e; ++i) {
          if (VTs[i] == MVT::Other || VTs[i] == MVT::Flag) break;
          RecordedNodes.push_back(SDValue(Res, i));
        }
        
      } else {
        // It is possible we're using MorphNodeTo to replace a node with no
        // normal results with one that has a normal result (or we could be
        // adding a chain) and the input could have flags and chains as well.
        // In this case we need to shifting the operands down.
        // FIXME: This is a horrible hack and broken in obscure cases, no worse
        // than the old isel though.  We should sink this into MorphNodeTo.
        int OldFlagResultNo = -1, OldChainResultNo = -1;
        
        unsigned NTMNumResults = NodeToMatch->getNumValues();
        if (NodeToMatch->getValueType(NTMNumResults-1) == MVT::Flag) {
          OldFlagResultNo = NTMNumResults-1;
          if (NTMNumResults != 1 &&
              NodeToMatch->getValueType(NTMNumResults-2) == MVT::Other)
            OldChainResultNo = NTMNumResults-2;
        } else if (NodeToMatch->getValueType(NTMNumResults-1) == MVT::Other)
          OldChainResultNo = NTMNumResults-1;
        
        Res = CurDAG->MorphNodeTo(NodeToMatch, ~TargetOpc, VTList,
                                  Ops.data(), Ops.size());
        
        // MorphNodeTo can operate in two ways: if an existing node with the
        // specified operands exists, it can just return it.  Otherwise, it
        // updates the node in place to have the requested operands.
        if (Res == NodeToMatch) {
          // If we updated the node in place, reset the node ID.  To the isel,
          // this should be just like a newly allocated machine node.
          Res->setNodeId(-1);
        }
        
        unsigned ResNumResults = Res->getNumValues();
        // Move the flag if needed.
        if ((EmitNodeInfo & OPFL_FlagOutput) && OldFlagResultNo != -1 &&
            (unsigned)OldFlagResultNo != ResNumResults-1)
          ReplaceUses(SDValue(NodeToMatch, OldFlagResultNo), 
                      SDValue(Res, ResNumResults-1));
        
        if ((EmitNodeInfo & OPFL_FlagOutput) != 0)
          --ResNumResults;

        // Move the chain reference if needed.
        if ((EmitNodeInfo & OPFL_Chain) && OldChainResultNo != -1 &&
            (unsigned)OldChainResultNo != ResNumResults-1)
          ReplaceUses(SDValue(NodeToMatch, OldChainResultNo), 
                      SDValue(Res, ResNumResults-1));

        if (Res != NodeToMatch) {
          // Otherwise, no replacement happened because the node already exists.
          ReplaceUses(NodeToMatch, Res);
        }
      }
      
      // If the node had chain/flag results, update our notion of the current
      // chain and flag.
      if (VTs.back() == MVT::Flag) {
        InputFlag = SDValue(Res, VTs.size()-1);
        if (EmitNodeInfo & OPFL_Chain)
          InputChain = SDValue(Res, VTs.size()-2);
      } else if (EmitNodeInfo & OPFL_Chain)
        InputChain = SDValue(Res, VTs.size()-1);

      // If the OPFL_MemRefs flag is set on this node, slap all of the
      // accumulated memrefs onto it.
      //
      // FIXME: This is vastly incorrect for patterns with multiple outputs
      // instructions that access memory and for ComplexPatterns that match
      // loads.
      if (EmitNodeInfo & OPFL_MemRefs) {
        MachineSDNode::mmo_iterator MemRefs =
          MF->allocateMemRefsArray(MatchedMemRefs.size());
        std::copy(MatchedMemRefs.begin(), MatchedMemRefs.end(), MemRefs);
        cast<MachineSDNode>(Res)
          ->setMemRefs(MemRefs, MemRefs + MatchedMemRefs.size());
      }
      
      DEBUG(errs() << "  "
                   << (Opcode == OPC_MorphNodeTo ? "Morphed" : "Created")
                   << " node: "; Res->dump(CurDAG); errs() << "\n");
      
      // If this was a MorphNodeTo then we're completely done!
      if (Opcode == OPC_MorphNodeTo) {
        // Update chain and flag uses.
        UpdateChainsAndFlags(NodeToMatch, InputChain, ChainNodesMatched,
                             InputFlag, FlagResultNodesMatched, true);
        return 0;
      }
      
      continue;
    }
        
    case OPC_MarkFlagResults: {
      unsigned NumNodes = MatcherTable[MatcherIndex++];
      
      // Read and remember all the flag-result nodes.
      for (unsigned i = 0; i != NumNodes; ++i) {
        unsigned RecNo = MatcherTable[MatcherIndex++];
        if (RecNo & 128)
          RecNo = GetVBR(RecNo, MatcherTable, MatcherIndex);

        assert(RecNo < RecordedNodes.size() && "Invalid CheckSame");
        FlagResultNodesMatched.push_back(RecordedNodes[RecNo].getNode());
      }
      continue;
    }
      
    case OPC_CompleteMatch: {
      // The match has been completed, and any new nodes (if any) have been
      // created.  Patch up references to the matched dag to use the newly
      // created nodes.
      unsigned NumResults = MatcherTable[MatcherIndex++];

      for (unsigned i = 0; i != NumResults; ++i) {
        unsigned ResSlot = MatcherTable[MatcherIndex++];
        if (ResSlot & 128)
          ResSlot = GetVBR(ResSlot, MatcherTable, MatcherIndex);
        
        assert(ResSlot < RecordedNodes.size() && "Invalid CheckSame");
        SDValue Res = RecordedNodes[ResSlot];
        
        // FIXME2: Eliminate this horrible hack by fixing the 'Gen' program
        // after (parallel) on input patterns are removed.  This would also
        // allow us to stop encoding #results in OPC_CompleteMatch's table
        // entry.
        if (NodeToMatch->getNumValues() <= i ||
            NodeToMatch->getValueType(i) == MVT::Other ||
            NodeToMatch->getValueType(i) == MVT::Flag)
          break;
        assert((NodeToMatch->getValueType(i) == Res.getValueType() ||
                NodeToMatch->getValueType(i) == MVT::iPTR ||
                Res.getValueType() == MVT::iPTR ||
                NodeToMatch->getValueType(i).getSizeInBits() ==
                    Res.getValueType().getSizeInBits()) &&
               "invalid replacement");
        ReplaceUses(SDValue(NodeToMatch, i), Res);
      }

      // If the root node defines a flag, add it to the flag nodes to update
      // list.
      if (NodeToMatch->getValueType(NodeToMatch->getNumValues()-1) == MVT::Flag)
        FlagResultNodesMatched.push_back(NodeToMatch);
      
      // Update chain and flag uses.
      UpdateChainsAndFlags(NodeToMatch, InputChain, ChainNodesMatched,
                           InputFlag, FlagResultNodesMatched, false);
      
      assert(NodeToMatch->use_empty() &&
             "Didn't replace all uses of the node?");
      
      // FIXME: We just return here, which interacts correctly with SelectRoot
      // above.  We should fix this to not return an SDNode* anymore.
      return 0;
    }
    }
    
    // If the code reached this point, then the match failed.  See if there is
    // another child to try in the current 'Scope', otherwise pop it until we
    // find a case to check.
    while (1) {
      if (MatchScopes.empty()) {
        CannotYetSelect(NodeToMatch);
        return 0;
      }

      // Restore the interpreter state back to the point where the scope was
      // formed.
      MatchScope &LastScope = MatchScopes.back();
      RecordedNodes.resize(LastScope.NumRecordedNodes);
      NodeStack.clear();
      NodeStack.append(LastScope.NodeStack.begin(), LastScope.NodeStack.end());
      N = NodeStack.back();

      DEBUG(errs() << "  Match failed at index " << MatcherIndex
                   << " continuing at " << LastScope.FailIndex << "\n");
    
      if (LastScope.NumMatchedMemRefs != MatchedMemRefs.size())
        MatchedMemRefs.resize(LastScope.NumMatchedMemRefs);
      MatcherIndex = LastScope.FailIndex;
      
      InputChain = LastScope.InputChain;
      InputFlag = LastScope.InputFlag;
      if (!LastScope.HasChainNodesMatched)
        ChainNodesMatched.clear();
      if (!LastScope.HasFlagResultNodesMatched)
        FlagResultNodesMatched.clear();

      // Check to see what the offset is at the new MatcherIndex.  If it is zero
      // we have reached the end of this scope, otherwise we have another child
      // in the current scope to try.
      unsigned NumToSkip = MatcherTable[MatcherIndex++];
      if (NumToSkip & 128)
        NumToSkip = GetVBR(NumToSkip, MatcherTable, MatcherIndex);

      // If we have another child in this scope to match, update FailIndex and
      // try it.
      if (NumToSkip != 0) {
        LastScope.FailIndex = MatcherIndex+NumToSkip;
        break;
      }
      
      // End of this scope, pop it and try the next child in the containing
      // scope.
      MatchScopes.pop_back();
    }
  }
}
    

#endif /* LLVM_CODEGEN_DAGISEL_HEADER_H */

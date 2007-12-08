//===-- LegalizeTypes.cpp - Common code for DAG type legalizer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SelectionDAG::LegalizeTypes method.  It transforms
// an arbitrary well-formed SelectionDAG to only consist of legal types.  This
// is common code shared among the LegalizeTypes*.cpp files.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// run - This is the main entry point for the type legalizer.  This does a
/// top-down traversal of the dag, legalizing types as it goes.
void DAGTypeLegalizer::run() {
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());

  // The root of the dag may dangle to deleted nodes until the type legalizer is
  // done.  Set it to null to avoid confusion.
  DAG.setRoot(SDOperand());
  
  // Walk all nodes in the graph, assigning them a NodeID of 'ReadyToProcess'
  // (and remembering them) if they are leaves and assigning 'NewNode' if
  // non-leaves.
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    if (I->getNumOperands() == 0) {
      I->setNodeId(ReadyToProcess);
      Worklist.push_back(I);
    } else {
      I->setNodeId(NewNode);
    }
  }
  
  // Now that we have a set of nodes to process, handle them all.
  while (!Worklist.empty()) {
    SDNode *N = Worklist.back();
    Worklist.pop_back();
    assert(N->getNodeId() == ReadyToProcess &&
           "Node should be ready if on worklist!");
    
    // Scan the values produced by the node, checking to see if any result
    // types are illegal.
    unsigned i = 0;
    unsigned NumResults = N->getNumValues();
    do {
      MVT::ValueType ResultVT = N->getValueType(i);
      LegalizeAction Action = getTypeAction(ResultVT);
      if (Action == Promote) {
        PromoteResult(N, i);
        goto NodeDone;
      } else if (Action == Expand) {
        // Expand can mean 1) split integer in half 2) scalarize single-element
        // vector 3) split vector in half.
        if (!MVT::isVector(ResultVT))
          ExpandResult(N, i);
        else if (MVT::getVectorNumElements(ResultVT) == 1)
          ScalarizeResult(N, i);     // Scalarize the single-element vector.
        else         // Split the vector in half.
          assert(0 && "Vector splitting not implemented");
        goto NodeDone;
      } else {
        assert(Action == Legal && "Unknown action!");
      }
    } while (++i < NumResults);
    
    // Scan the operand list for the node, handling any nodes with operands that
    // are illegal.
    {
    unsigned NumOperands = N->getNumOperands();
    bool NeedsRevisit = false;
    for (i = 0; i != NumOperands; ++i) {
      MVT::ValueType OpVT = N->getOperand(i).getValueType();
      LegalizeAction Action = getTypeAction(OpVT);
      if (Action == Promote) {
        NeedsRevisit = PromoteOperand(N, i);
        break;
      } else if (Action == Expand) {
        // Expand can mean 1) split integer in half 2) scalarize single-element
        // vector 3) split vector in half.
        if (!MVT::isVector(OpVT)) {
          NeedsRevisit = ExpandOperand(N, i);
        } else if (MVT::getVectorNumElements(OpVT) == 1) {
          // Scalarize the single-element vector.
          NeedsRevisit = ScalarizeOperand(N, i);
        } else {
          // Split the vector in half.
          assert(0 && "Vector splitting not implemented");
        }
        break;
      } else {
        assert(Action == Legal && "Unknown action!");
      }
    }

    // If the node needs revisiting, don't add all users to the worklist etc.
    if (NeedsRevisit)
      continue;
    
    if (i == NumOperands)
      DEBUG(cerr << "Legally typed node: "; N->dump(&DAG); cerr << "\n");
    }
NodeDone:

    // If we reach here, the node was processed, potentially creating new nodes.
    // Mark it as processed and add its users to the worklist as appropriate.
    N->setNodeId(Processed);
    
    for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end();
         UI != E; ++UI) {
      SDNode *User = *UI;
      int NodeID = User->getNodeId();
      assert(NodeID != ReadyToProcess && NodeID != Processed &&
             "Invalid node id for user of unprocessed node!");
      
      // This node has two options: it can either be a new node or its Node ID
      // may be a count of the number of operands it has that are not ready.
      if (NodeID > 0) {
        User->setNodeId(NodeID-1);
        
        // If this was the last use it was waiting on, add it to the ready list.
        if (NodeID-1 == ReadyToProcess)
          Worklist.push_back(User);
        continue;
      }
      
      // Otherwise, this node is new: this is the first operand of it that
      // became ready.  Its new NodeID is the number of operands it has minus 1
      // (as this node is now processed).
      assert(NodeID == NewNode && "Unknown node ID!");
      User->setNodeId(User->getNumOperands()-1);
      
      // If the node only has a single operand, it is now ready.
      if (User->getNumOperands() == 1)
        Worklist.push_back(User);
    }
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  DAG.setRoot(Dummy.getValue());

  //DAG.viewGraph();

  // Remove dead nodes.  This is important to do for cleanliness but also before
  // the checking loop below.  Implicit folding by the DAG.getNode operators can
  // cause unreachable nodes to be around with their flags set to new.
  DAG.RemoveDeadNodes();

  // In a debug build, scan all the nodes to make sure we found them all.  This
  // ensures that there are no cycles and that everything got processed.
#ifndef NDEBUG
  for (SelectionDAG::allnodes_iterator I = DAG.allnodes_begin(),
       E = DAG.allnodes_end(); I != E; ++I) {
    if (I->getNodeId() == Processed)
      continue;
    cerr << "Unprocessed node: ";
    I->dump(&DAG); cerr << "\n";

    if (I->getNodeId() == NewNode)
      cerr << "New node not 'noticed'?\n";
    else if (I->getNodeId() > 0)
      cerr << "Operand not processed?\n";
    else if (I->getNodeId() == ReadyToProcess)
      cerr << "Not added to worklist?\n";
    abort();
  }
#endif
}

/// MarkNewNodes - The specified node is the root of a subtree of potentially
/// new nodes.  Add the correct NodeId to mark it.
void DAGTypeLegalizer::MarkNewNodes(SDNode *N) {
  // If this was an existing node that is already done, we're done.
  if (N->getNodeId() != NewNode)
    return;

  // Okay, we know that this node is new.  Recursively walk all of its operands
  // to see if they are new also.  The depth of this walk is bounded by the size
  // of the new tree that was constructed (usually 2-3 nodes), so we don't worry
  // about revisiting of nodes.
  //
  // As we walk the operands, keep track of the number of nodes that are
  // processed.  If non-zero, this will become the new nodeid of this node.
  unsigned NumProcessed = 0;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    int OpId = N->getOperand(i).Val->getNodeId();
    if (OpId == NewNode)
      MarkNewNodes(N->getOperand(i).Val);
    else if (OpId == Processed)
      ++NumProcessed;
  }
  
  N->setNodeId(N->getNumOperands()-NumProcessed);
  if (N->getNodeId() == ReadyToProcess)
    Worklist.push_back(N);
}

/// ReplaceValueWith - The specified value was legalized to the specified other
/// value.  If they are different, update the DAG and NodeIDs replacing any uses
/// of From to use To instead.
void DAGTypeLegalizer::ReplaceValueWith(SDOperand From, SDOperand To) {
  if (From == To) return;
  
  // If expansion produced new nodes, make sure they are properly marked.
  if (To.Val->getNodeId() == NewNode)
    MarkNewNodes(To.Val);
  
  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  DAG.ReplaceAllUsesOfValueWith(From, To);

  // The old node may still be present in ExpandedNodes or PromotedNodes.
  // Inform them about the replacement.
  ReplacedNodes[From] = To;

  // Since we just made an unstructured update to the DAG, which could wreak
  // general havoc on anything that once used From and now uses To, walk all
  // users of the result, updating their flags.
  for (SDNode::use_iterator I = To.Val->use_begin(), E = To.Val->use_end();
       I != E; ++I) {
    SDNode *User = *I;
    // If the node isn't already processed or in the worklist, mark it as new,
    // then use MarkNewNodes to recompute its ID.
    int NodeId = User->getNodeId();
    if (NodeId != ReadyToProcess && NodeId != Processed) {
      User->setNodeId(NewNode);
      MarkNewNodes(User);
    }
  }
}

/// ReplaceNodeWith - Replace uses of the 'from' node's results with the 'to'
/// node's results.  The from and to node must define identical result types.
void DAGTypeLegalizer::ReplaceNodeWith(SDNode *From, SDNode *To) {
  if (From == To) return;
  assert(From->getNumValues() == To->getNumValues() &&
         "Node results don't match");
  
  // If expansion produced new nodes, make sure they are properly marked.
  if (To->getNodeId() == NewNode)
    MarkNewNodes(To);
  
  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  DAG.ReplaceAllUsesWith(From, To);
  
  // The old node may still be present in ExpandedNodes or PromotedNodes.
  // Inform them about the replacement.
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i) {
    assert(From->getValueType(i) == To->getValueType(i) &&
           "Node results don't match");
    ReplacedNodes[SDOperand(From, i)] = SDOperand(To, i);
  }
  
  // Since we just made an unstructured update to the DAG, which could wreak
  // general havoc on anything that once used From and now uses To, walk all
  // users of the result, updating their flags.
  for (SDNode::use_iterator I = To->use_begin(), E = To->use_end();I != E; ++I){
    SDNode *User = *I;
    // If the node isn't already processed or in the worklist, mark it as new,
    // then use MarkNewNodes to recompute its ID.
    int NodeId = User->getNodeId();
    if (NodeId != ReadyToProcess && NodeId != Processed) {
      User->setNodeId(NewNode);
      MarkNewNodes(User);
    }
  }
}


/// RemapNode - If the specified value was already legalized to another value,
/// replace it by that value.
void DAGTypeLegalizer::RemapNode(SDOperand &N) {
  DenseMap<SDOperand, SDOperand>::iterator I = ReplacedNodes.find(N);
  if (I != ReplacedNodes.end()) {
    // Use path compression to speed up future lookups if values get multiply
    // replaced with other values.
    RemapNode(I->second);
    N = I->second;
  }
}

void DAGTypeLegalizer::SetPromotedOp(SDOperand Op, SDOperand Result) {
  if (Result.Val->getNodeId() == NewNode) 
    MarkNewNodes(Result.Val);

  SDOperand &OpEntry = PromotedNodes[Op];
  assert(OpEntry.Val == 0 && "Node is already promoted!");
  OpEntry = Result;
}

void DAGTypeLegalizer::SetScalarizedOp(SDOperand Op, SDOperand Result) {
  if (Result.Val->getNodeId() == NewNode) 
    MarkNewNodes(Result.Val);
  
  SDOperand &OpEntry = ScalarizedNodes[Op];
  assert(OpEntry.Val == 0 && "Node is already scalarized!");
  OpEntry = Result;
}


void DAGTypeLegalizer::GetExpandedOp(SDOperand Op, SDOperand &Lo, 
                                     SDOperand &Hi) {
  std::pair<SDOperand, SDOperand> &Entry = ExpandedNodes[Op];
  RemapNode(Entry.first);
  RemapNode(Entry.second);
  assert(Entry.first.Val && "Operand isn't expanded");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetExpandedOp(SDOperand Op, SDOperand Lo, 
                                     SDOperand Hi) {
  // Remember that this is the result of the node.
  std::pair<SDOperand, SDOperand> &Entry = ExpandedNodes[Op];
  assert(Entry.first.Val == 0 && "Node already expanded");
  Entry.first = Lo;
  Entry.second = Hi;
  
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  if (Lo.Val->getNodeId() == NewNode) 
    MarkNewNodes(Lo.Val);
  if (Hi.Val->getNodeId() == NewNode) 
    MarkNewNodes(Hi.Val);
}

SDOperand DAGTypeLegalizer::CreateStackStoreLoad(SDOperand Op, 
                                                 MVT::ValueType DestVT) {
  // Create the stack frame object.
  SDOperand FIPtr = DAG.CreateStackTemporary(DestVT);
  
  // Emit a store to the stack slot.
  SDOperand Store = DAG.getStore(DAG.getEntryNode(), Op, FIPtr, NULL, 0);
  // Result is a load from the stack slot.
  return DAG.getLoad(DestVT, Store, FIPtr, NULL, 0);
}

/// HandleMemIntrinsic - This handles memcpy/memset/memmove with invalid
/// operands.  This promotes or expands the operands as required.
SDOperand DAGTypeLegalizer::HandleMemIntrinsic(SDNode *N) {
  // The chain and pointer [operands #0 and #1] are always valid types.
  SDOperand Chain = N->getOperand(0);
  SDOperand Ptr   = N->getOperand(1);
  SDOperand Op2   = N->getOperand(2);
  
  // Op #2 is either a value (memset) or a pointer.  Promote it if required.
  switch (getTypeAction(Op2.getValueType())) {
  default: assert(0 && "Unknown action for pointer/value operand");
  case Legal: break;
  case Promote: Op2 = GetPromotedOp(Op2); break;
  }
  
  // The length could have any action required.
  SDOperand Length = N->getOperand(3);
  switch (getTypeAction(Length.getValueType())) {
  default: assert(0 && "Unknown action for memop operand");
  case Legal: break;
  case Promote: Length = GetPromotedZExtOp(Length); break;
  case Expand:
    SDOperand Dummy;  // discard the high part.
    GetExpandedOp(Length, Length, Dummy);
    break;
  }
  
  SDOperand Align = N->getOperand(4);
  switch (getTypeAction(Align.getValueType())) {
  default: assert(0 && "Unknown action for memop operand");
  case Legal: break;
  case Promote: Align = GetPromotedZExtOp(Align); break;
  }
  
  SDOperand AlwaysInline = N->getOperand(5);
  switch (getTypeAction(AlwaysInline.getValueType())) {
  default: assert(0 && "Unknown action for memop operand");
  case Legal: break;
  case Promote: AlwaysInline = GetPromotedZExtOp(AlwaysInline); break;
  }
  
  SDOperand Ops[] = { Chain, Ptr, Op2, Length, Align, AlwaysInline };
  return DAG.UpdateNodeOperands(SDOperand(N, 0), Ops, 6);
}

/// SplitOp - Return the lower and upper halves of Op's bits in a value type
/// half the size of Op's.
void DAGTypeLegalizer::SplitOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi) {
  unsigned NVTBits = MVT::getSizeInBits(Op.getValueType())/2;
  assert(MVT::getSizeInBits(Op.getValueType()) == 2*NVTBits &&
         "Cannot split odd sized integer type");
  MVT::ValueType NVT = MVT::getIntegerType(NVTBits);
  Lo = DAG.getNode(ISD::TRUNCATE, NVT, Op);
  Hi = DAG.getNode(ISD::SRL, Op.getValueType(), Op,
                   DAG.getConstant(NVTBits, TLI.getShiftAmountTy()));
  Hi = DAG.getNode(ISD::TRUNCATE, NVT, Hi);
}


//===----------------------------------------------------------------------===//
//  Result Vector Scalarization: <1 x ty> -> ty.
//===----------------------------------------------------------------------===//


void DAGTypeLegalizer::ScalarizeResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Scalarize node result " << ResNo << ": "; N->dump(&DAG); 
        cerr << "\n");
  SDOperand R = SDOperand();
  
  // FIXME: Custom lowering for scalarization?
#if 0
  // See if the target wants to custom expand this node.
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) == 
      TargetLowering::Custom) {
    // If the target wants to, allow it to lower this itself.
    if (SDNode *P = TLI.ExpandOperationResult(N, DAG)) {
      // Everything that once used N now uses P.  We are guaranteed that the
      // result value types of N and the result value types of P match.
      ReplaceNodeWith(N, P);
      return;
    }
  }
#endif
  
  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ScalarizeResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to scalarize the result of this operator!");
    abort();
    
  case ISD::UNDEF:       R = ScalarizeRes_UNDEF(N); break;
  case ISD::LOAD:        R = ScalarizeRes_LOAD(cast<LoadSDNode>(N)); break;
  case ISD::ADD:
  case ISD::FADD:
  case ISD::SUB:
  case ISD::FSUB:
  case ISD::MUL:
  case ISD::FMUL:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::FDIV:
  case ISD::SREM:
  case ISD::UREM:
  case ISD::FREM:
  case ISD::FPOW:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:         R = ScalarizeRes_BinOp(N); break;
  case ISD::FNEG:
  case ISD::FABS:
  case ISD::FSQRT:
  case ISD::FSIN:
  case ISD::FCOS:              R = ScalarizeRes_UnaryOp(N); break;
  case ISD::FPOWI:             R = ScalarizeRes_FPOWI(N); break;
  case ISD::BUILD_VECTOR:      R = N->getOperand(0); break;
  case ISD::INSERT_VECTOR_ELT: R = N->getOperand(1); break;
  case ISD::VECTOR_SHUFFLE:    R = ScalarizeRes_VECTOR_SHUFFLE(N); break;
  case ISD::BIT_CONVERT:       R = ScalarizeRes_BIT_CONVERT(N); break;
  case ISD::SELECT:            R = ScalarizeRes_SELECT(N); break;
  }
  
  // If R is null, the sub-method took care of registering the resul.
  if (R.Val)
    SetScalarizedOp(SDOperand(N, ResNo), R);
}

SDOperand DAGTypeLegalizer::ScalarizeRes_UNDEF(SDNode *N) {
  return DAG.getNode(ISD::UNDEF, MVT::getVectorElementType(N->getValueType(0)));
}

SDOperand DAGTypeLegalizer::ScalarizeRes_LOAD(LoadSDNode *N) {
  SDOperand Result = DAG.getLoad(MVT::getVectorElementType(N->getValueType(0)),
                                 N->getChain(), N->getBasePtr(), 
                                 N->getSrcValue(), N->getSrcValueOffset(),
                                 N->isVolatile(), N->getAlignment());
  
  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDOperand(N, 1), Result.getValue(1));
  return Result;
}

SDOperand DAGTypeLegalizer::ScalarizeRes_BinOp(SDNode *N) {
  SDOperand LHS = GetScalarizedOp(N->getOperand(0));
  SDOperand RHS = GetScalarizedOp(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), LHS.getValueType(), LHS, RHS);
}

SDOperand DAGTypeLegalizer::ScalarizeRes_UnaryOp(SDNode *N) {
  SDOperand Op = GetScalarizedOp(N->getOperand(0));
  return DAG.getNode(N->getOpcode(), Op.getValueType(), Op);
}

SDOperand DAGTypeLegalizer::ScalarizeRes_FPOWI(SDNode *N) {
  SDOperand Op = GetScalarizedOp(N->getOperand(0));
  return DAG.getNode(ISD::FPOWI, Op.getValueType(), Op, N->getOperand(1));
}

SDOperand DAGTypeLegalizer::ScalarizeRes_VECTOR_SHUFFLE(SDNode *N) {
  // Figure out if the scalar is the LHS or RHS and return it.
  SDOperand EltNum = N->getOperand(2).getOperand(0);
  unsigned Op = cast<ConstantSDNode>(EltNum)->getValue() != 0;
  return GetScalarizedOp(N->getOperand(Op));
}

SDOperand DAGTypeLegalizer::ScalarizeRes_BIT_CONVERT(SDNode *N) {
  MVT::ValueType NewVT = MVT::getVectorElementType(N->getValueType(0));
  return DAG.getNode(ISD::BIT_CONVERT, NewVT, N->getOperand(0));
}

SDOperand DAGTypeLegalizer::ScalarizeRes_SELECT(SDNode *N) {
  SDOperand LHS = GetScalarizedOp(N->getOperand(1));
  return DAG.getNode(ISD::SELECT, LHS.getValueType(), N->getOperand(0), LHS,
                     GetScalarizedOp(N->getOperand(2)));
}


//===----------------------------------------------------------------------===//
//  Operand Vector Scalarization <1 x ty> -> ty.
//===----------------------------------------------------------------------===//

bool DAGTypeLegalizer::ScalarizeOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Scalarize node operand " << OpNo << ": "; N->dump(&DAG); 
        cerr << "\n");
  SDOperand Res(0, 0);
  
  // FIXME: Should we support custom lowering for scalarization?
#if 0
  if (TLI.getOperationAction(N->getOpcode(), N->getValueType(0)) == 
      TargetLowering::Custom)
    Res = TLI.LowerOperation(SDOperand(N, 0), DAG);
#endif
  
  if (Res.Val == 0) {
    switch (N->getOpcode()) {
    default:
#ifndef NDEBUG
      cerr << "ScalarizeOperand Op #" << OpNo << ": ";
      N->dump(&DAG); cerr << "\n";
#endif
      assert(0 && "Do not know how to scalarize this operator's operand!");
      abort();
      
    case ISD::EXTRACT_VECTOR_ELT:
      Res = ScalarizeOp_EXTRACT_VECTOR_ELT(N, OpNo);
      break;
    }
  }
  
  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.Val) return false;
  
  // If the result is N, the sub-method updated N in place.  Check to see if any
  // operands are new, and if so, mark them.
  if (Res.Val == N) {
    // Mark N as new and remark N and its operands.  This allows us to correctly
    // revisit N if it needs another step of promotion and allows us to visit
    // any new operands to N.
    N->setNodeId(NewNode);
    MarkNewNodes(N);
    return true;
  }
  
  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");
  
  ReplaceValueWith(SDOperand(N, 0), Res);
  return false;
}

/// ScalarizeOp_EXTRACT_VECTOR_ELT - If the input is a vector that needs to be
/// scalarized, it must be <1 x ty>, just return the operand, ignoring the
/// index.
SDOperand DAGTypeLegalizer::ScalarizeOp_EXTRACT_VECTOR_ELT(SDNode *N, 
                                                           unsigned OpNo) {
  return GetScalarizedOp(N->getOperand(0));
}


//===----------------------------------------------------------------------===//
//  Entry Point
//===----------------------------------------------------------------------===//

/// LegalizeTypes - This transforms the SelectionDAG into a SelectionDAG that
/// only uses types natively supported by the target.
///
/// Note that this is an involved process that may invalidate pointers into
/// the graph.
void SelectionDAG::LegalizeTypes() {
  DAGTypeLegalizer(*this).run();
}

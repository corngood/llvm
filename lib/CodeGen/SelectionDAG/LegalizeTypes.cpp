//===-- LegalizeTypes.cpp - Common code for DAG type legalizer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

#ifndef NDEBUG
static cl::opt<bool>
ViewLegalizeTypesDAGs("view-legalize-types-dags", cl::Hidden,
                cl::desc("Pop up a window to show dags before legalize types"));
#else
static const bool ViewLegalizeTypesDAGs = 0;
#endif



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
      switch (getTypeAction(ResultVT)) {
      default:
        assert(false && "Unknown action!");
      case Legal:
        break;
      case Promote:
        PromoteResult(N, i);
        goto NodeDone;
      case Expand:
        ExpandResult(N, i);
        goto NodeDone;
      case Scalarize:
        ScalarizeResult(N, i);
        goto NodeDone;
      case Split:
        SplitResult(N, i);
        goto NodeDone;
      }
    } while (++i < NumResults);

    // Scan the operand list for the node, handling any nodes with operands that
    // are illegal.
    {
    unsigned NumOperands = N->getNumOperands();
    bool NeedsRevisit = false;
    for (i = 0; i != NumOperands; ++i) {
      MVT::ValueType OpVT = N->getOperand(i).getValueType();
      switch (getTypeAction(OpVT)) {
      default:
        assert(false && "Unknown action!");
      case Legal:
        continue;
      case Promote:
        NeedsRevisit = PromoteOperand(N, i);
        break;
      case Expand:
        NeedsRevisit = ExpandOperand(N, i);
        break;
      case Scalarize:
        NeedsRevisit = ScalarizeOperand(N, i);
        break;
      case Split:
        NeedsRevisit = SplitOperand(N, i);
        break;
      }
      break;
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
    bool Failed = false;

    // Check that all result types are legal.
    for (unsigned i = 0, NumVals = I->getNumValues(); i < NumVals; ++i)
      if (!isTypeLegal(I->getValueType(i))) {
        cerr << "Result type " << i << " illegal!\n";
        Failed = true;
      }

    // Check that all operand types are legal.
    for (unsigned i = 0, NumOps = I->getNumOperands(); i < NumOps; ++i)
      if (!isTypeLegal(I->getOperand(i).getValueType())) {
        cerr << "Operand type " << i << " illegal!\n";
        Failed = true;
      }

    if (I->getNodeId() != Processed) {
       if (I->getNodeId() == NewNode)
         cerr << "New node not 'noticed'?\n";
       else if (I->getNodeId() > 0)
         cerr << "Operand not processed?\n";
       else if (I->getNodeId() == ReadyToProcess)
         cerr << "Not added to worklist?\n";
       Failed = true;
    }

    if (Failed) {
      I->dump(&DAG); cerr << "\n";
      abort();
    }
  }
#endif
}

/// AnalyzeNewNode - The specified node is the root of a subtree of potentially
/// new nodes.  Correct any processed operands (this may change the node) and
/// calculate the NodeId.
void DAGTypeLegalizer::AnalyzeNewNode(SDNode *&N) {
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
  // Already processed operands may need to be remapped to the node that
  // replaced them, which can result in our node changing.  Since remapping
  // is rare, the code tries to minimize overhead in the non-remapping case.

  SmallVector<SDOperand, 8> NewOps;
  unsigned NumProcessed = 0;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDOperand OrigOp = N->getOperand(i);
    SDOperand Op = OrigOp;

    if (Op.Val->getNodeId() == Processed)
      RemapNode(Op);

    if (Op.Val->getNodeId() == NewNode)
      AnalyzeNewNode(Op.Val);
    else if (Op.Val->getNodeId() == Processed)
      ++NumProcessed;

    if (!NewOps.empty()) {
      // Some previous operand changed.  Add this one to the list.
      NewOps.push_back(Op);
    } else if (Op != OrigOp) {
      // This is the first operand to change - add all operands so far.
      for (unsigned j = 0; j < i; ++j)
        NewOps.push_back(N->getOperand(j));
      NewOps.push_back(Op);
    }
  }

  // Some operands changed - update the node.
  if (!NewOps.empty())
    N = DAG.UpdateNodeOperands(SDOperand(N, 0), &NewOps[0], NewOps.size()).Val;

  N->setNodeId(N->getNumOperands()-NumProcessed);
  if (N->getNodeId() == ReadyToProcess)
    Worklist.push_back(N);
}

namespace {
  /// NodeUpdateListener - This class is a DAGUpdateListener that listens for
  /// updates to nodes and recomputes their ready state.
  class VISIBILITY_HIDDEN NodeUpdateListener :
    public SelectionDAG::DAGUpdateListener {
    DAGTypeLegalizer &DTL;
  public:
    NodeUpdateListener(DAGTypeLegalizer &dtl) : DTL(dtl) {}

    virtual void NodeDeleted(SDNode *N) {
      // Ignore deletes.
      assert(N->getNodeId() != DAGTypeLegalizer::Processed &&
             N->getNodeId() != DAGTypeLegalizer::ReadyToProcess &&
             "RAUW deleted processed node!");
    }

    virtual void NodeUpdated(SDNode *N) {
      // Node updates can mean pretty much anything.  It is possible that an
      // operand was set to something already processed (f.e.) in which case
      // this node could become ready.  Recompute its flags.
      assert(N->getNodeId() != DAGTypeLegalizer::Processed &&
             N->getNodeId() != DAGTypeLegalizer::ReadyToProcess &&
             "RAUW updated processed node!");
      DTL.ReanalyzeNode(N);
    }
  };
}


/// ReplaceValueWith - The specified value was legalized to the specified other
/// value.  If they are different, update the DAG and NodeIDs replacing any uses
/// of From to use To instead.
void DAGTypeLegalizer::ReplaceValueWith(SDOperand From, SDOperand To) {
  if (From == To) return;

  // If expansion produced new nodes, make sure they are properly marked.
  AnalyzeNewNode(To.Val);

  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  NodeUpdateListener NUL(*this);
  DAG.ReplaceAllUsesOfValueWith(From, To, &NUL);

  // The old node may still be present in ExpandedNodes or PromotedNodes.
  // Inform them about the replacement.
  ReplacedNodes[From] = To;
}

/// ReplaceNodeWith - Replace uses of the 'from' node's results with the 'to'
/// node's results.  The from and to node must define identical result types.
void DAGTypeLegalizer::ReplaceNodeWith(SDNode *From, SDNode *To) {
  if (From == To) return;

  // If expansion produced new nodes, make sure they are properly marked.
  AnalyzeNewNode(To);

  assert(From->getNumValues() == To->getNumValues() &&
         "Node results don't match");

  // Anything that used the old node should now use the new one.  Note that this
  // can potentially cause recursive merging.
  NodeUpdateListener NUL(*this);
  DAG.ReplaceAllUsesWith(From, To, &NUL);
  
  // The old node may still be present in ExpandedNodes or PromotedNodes.
  // Inform them about the replacement.
  for (unsigned i = 0, e = From->getNumValues(); i != e; ++i) {
    assert(From->getValueType(i) == To->getValueType(i) &&
           "Node results don't match");
    ReplacedNodes[SDOperand(From, i)] = SDOperand(To, i);
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
  AnalyzeNewNode(Result.Val);

  SDOperand &OpEntry = PromotedNodes[Op];
  assert(OpEntry.Val == 0 && "Node is already promoted!");
  OpEntry = Result;
}

void DAGTypeLegalizer::SetScalarizedOp(SDOperand Op, SDOperand Result) {
  AnalyzeNewNode(Result.Val);

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

void DAGTypeLegalizer::SetExpandedOp(SDOperand Op, SDOperand Lo, SDOperand Hi) {
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewNode(Lo.Val);
  AnalyzeNewNode(Hi.Val);

  // Remember that this is the result of the node.
  std::pair<SDOperand, SDOperand> &Entry = ExpandedNodes[Op];
  assert(Entry.first.Val == 0 && "Node already expanded");
  Entry.first = Lo;
  Entry.second = Hi;
}

void DAGTypeLegalizer::GetSplitOp(SDOperand Op, SDOperand &Lo, SDOperand &Hi) {
  std::pair<SDOperand, SDOperand> &Entry = SplitNodes[Op];
  RemapNode(Entry.first);
  RemapNode(Entry.second);
  assert(Entry.first.Val && "Operand isn't split");
  Lo = Entry.first;
  Hi = Entry.second;
}

void DAGTypeLegalizer::SetSplitOp(SDOperand Op, SDOperand Lo, SDOperand Hi) {
  // Lo/Hi may have been newly allocated, if so, add nodeid's as relevant.
  AnalyzeNewNode(Lo.Val);
  AnalyzeNewNode(Hi.Val);

  // Remember that this is the result of the node.
  std::pair<SDOperand, SDOperand> &Entry = SplitNodes[Op];
  assert(Entry.first.Val == 0 && "Node already split");
  Entry.first = Lo;
  Entry.second = Hi;
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

/// JoinIntegers - Build an integer with low bits Lo and high bits Hi.
SDOperand DAGTypeLegalizer::JoinIntegers(SDOperand Lo, SDOperand Hi) {
  MVT::ValueType LVT = Lo.getValueType();
  MVT::ValueType HVT = Hi.getValueType();
  MVT::ValueType NVT = MVT::getIntegerType(MVT::getSizeInBits(LVT) +
                                           MVT::getSizeInBits(HVT));

  Lo = DAG.getNode(ISD::ZERO_EXTEND, NVT, Lo);
  Hi = DAG.getNode(ISD::ANY_EXTEND, NVT, Hi);
  Hi = DAG.getNode(ISD::SHL, NVT, Hi, DAG.getConstant(MVT::getSizeInBits(LVT),
                                                      TLI.getShiftAmountTy()));
  return DAG.getNode(ISD::OR, NVT, Lo, Hi);
}

/// SplitInteger - Return the lower LoVT bits of Op in Lo and the upper HiVT
/// bits in Hi.
void DAGTypeLegalizer::SplitInteger(SDOperand Op,
                                    MVT::ValueType LoVT, MVT::ValueType HiVT,
                                    SDOperand &Lo, SDOperand &Hi) {
  assert(MVT::getSizeInBits(LoVT) + MVT::getSizeInBits(HiVT) ==
         MVT::getSizeInBits(Op.getValueType()) && "Invalid integer splitting!");
  Lo = DAG.getNode(ISD::TRUNCATE, LoVT, Op);
  Hi = DAG.getNode(ISD::SRL, Op.getValueType(), Op,
                   DAG.getConstant(MVT::getSizeInBits(LoVT),
                                   TLI.getShiftAmountTy()));
  Hi = DAG.getNode(ISD::TRUNCATE, HiVT, Hi);
}

/// SplitInteger - Return the lower and upper halves of Op's bits in a value type
/// half the size of Op's.
void DAGTypeLegalizer::SplitInteger(SDOperand Op,
                                    SDOperand &Lo, SDOperand &Hi) {
  MVT::ValueType HalfVT =
    MVT::getIntegerType(MVT::getSizeInBits(Op.getValueType())/2);
  SplitInteger(Op, HalfVT, HalfVT, Lo, Hi);
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
  if (ViewLegalizeTypesDAGs) viewGraph();
  
  DAGTypeLegalizer(*this).run();
}

//===- Local.cpp - Compute a local data structure graph for a function ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Compute the local version of the data structure graph for a function.  The
// external interface to this file is the DSGraph constructor.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include <iostream>

// FIXME: This should eventually be a FunctionPass that is automatically
// aggregated into a Pass.
//
#include "llvm/Module.h"

using namespace llvm;

static RegisterAnalysis<LocalDataStructures>
X("datastructure", "Local Data Structure Analysis");

static cl::opt<bool>
TrackIntegersAsPointers("dsa-track-integers", cl::Hidden,
         cl::desc("If this is set, track integers as potential pointers"));

static cl::list<std::string>
AllocList("dsa-alloc-list",
          cl::value_desc("list"),
          cl::desc("List of functions that allocate memory from the heap"),
          cl::CommaSeparated, cl::Hidden);

static cl::list<std::string>
FreeList("dsa-free-list",
          cl::value_desc("list"),
          cl::desc("List of functions that free memory from the heap"),
          cl::CommaSeparated, cl::Hidden);

namespace llvm {
namespace DS {
  // isPointerType - Return true if this type is big enough to hold a pointer.
  bool isPointerType(const Type *Ty) {
    if (isa<PointerType>(Ty))
      return true;
    else if (TrackIntegersAsPointers && Ty->isPrimitiveType() &&Ty->isInteger())
      return Ty->getPrimitiveSize() >= PointerSize;
    return false;
  }
}}

using namespace DS;

namespace {
  cl::opt<bool>
  DisableDirectCallOpt("disable-direct-call-dsopt", cl::Hidden,
                       cl::desc("Disable direct call optimization in "
                                "DSGraph construction"));
  cl::opt<bool>
  DisableFieldSensitivity("disable-ds-field-sensitivity", cl::Hidden,
                          cl::desc("Disable field sensitivity in DSGraphs"));

  //===--------------------------------------------------------------------===//
  //  GraphBuilder Class
  //===--------------------------------------------------------------------===//
  //
  /// This class is the builder class that constructs the local data structure
  /// graph by performing a single pass over the function in question.
  ///
  class GraphBuilder : InstVisitor<GraphBuilder> {
    DSGraph &G;
    DSNodeHandle *RetNode;               // Node that gets returned...
    DSScalarMap &ScalarMap;
    std::list<DSCallSite> *FunctionCalls;

  public:
    GraphBuilder(Function &f, DSGraph &g, DSNodeHandle &retNode,
                 std::list<DSCallSite> &fc)
      : G(g), RetNode(&retNode), ScalarMap(G.getScalarMap()),
        FunctionCalls(&fc) {

      // Create scalar nodes for all pointer arguments...
      for (Function::arg_iterator I = f.arg_begin(), E = f.arg_end();
           I != E; ++I)
        if (isPointerType(I->getType()))
          getValueDest(*I);

      visit(f);  // Single pass over the function
    }

    // GraphBuilder ctor for working on the globals graph
    GraphBuilder(DSGraph &g)
      : G(g), RetNode(0), ScalarMap(G.getScalarMap()), FunctionCalls(0) {
    }

    void mergeInGlobalInitializer(GlobalVariable *GV);

  private:
    // Visitor functions, used to handle each instruction type we encounter...
    friend class InstVisitor<GraphBuilder>;
    void visitMallocInst(MallocInst &MI) { handleAlloc(MI, true); }
    void visitAllocaInst(AllocaInst &AI) { handleAlloc(AI, false); }
    void handleAlloc(AllocationInst &AI, bool isHeap);

    void visitPHINode(PHINode &PN);
    void visitSelectInst(SelectInst &SI);

    void visitGetElementPtrInst(User &GEP);
    void visitReturnInst(ReturnInst &RI);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitCallInst(CallInst &CI);
    void visitInvokeInst(InvokeInst &II);
    void visitSetCondInst(SetCondInst &SCI);
    void visitFreeInst(FreeInst &FI);
    void visitCastInst(CastInst &CI);
    void visitInstruction(Instruction &I);

    void visitCallSite(CallSite CS);
    void visitVAArgInst(VAArgInst   &I);

    void MergeConstantInitIntoNode(DSNodeHandle &NH, Constant *C);
  private:
    // Helper functions used to implement the visitation functions...

    /// createNode - Create a new DSNode, ensuring that it is properly added to
    /// the graph.
    ///
    DSNode *createNode(const Type *Ty = 0) {
      DSNode *N = new DSNode(Ty, &G);   // Create the node
      if (DisableFieldSensitivity) {
        // Create node handle referring to the old node so that it is
        // immediately removed from the graph when the node handle is destroyed.
        DSNodeHandle OldNNH = N;
        N->foldNodeCompletely();
        if (DSNode *FN = N->getForwardNode())
          N = FN;
      }
      return N;
    }

    /// setDestTo - Set the ScalarMap entry for the specified value to point to
    /// the specified destination.  If the Value already points to a node, make
    /// sure to merge the two destinations together.
    ///
    void setDestTo(Value &V, const DSNodeHandle &NH);

    /// getValueDest - Return the DSNode that the actual value points to.
    ///
    DSNodeHandle getValueDest(Value &V);

    /// getLink - This method is used to return the specified link in the
    /// specified node if one exists.  If a link does not already exist (it's
    /// null), then we create a new node, link it, then return it.
    ///
    DSNodeHandle &getLink(const DSNodeHandle &Node, unsigned Link = 0);
  };
}

using namespace DS;

//===----------------------------------------------------------------------===//
// DSGraph constructor - Simply use the GraphBuilder to construct the local
// graph.
DSGraph::DSGraph(EquivalenceClasses<GlobalValue*> &ECs, const TargetData &td,
                 Function &F, DSGraph *GG)
  : GlobalsGraph(GG), ScalarMap(ECs), TD(td) {
  PrintAuxCalls = false;

  DEBUG(std::cerr << "  [Loc] Calculating graph for: " << F.getName() << "\n");

  // Use the graph builder to construct the local version of the graph
  GraphBuilder B(F, *this, ReturnNodes[&F], FunctionCalls);
#ifndef NDEBUG
  Timer::addPeakMemoryMeasurement();
#endif

  // If there are any constant globals referenced in this function, merge their
  // initializers into the local graph from the globals graph.
  if (ScalarMap.global_begin() != ScalarMap.global_end()) {
    ReachabilityCloner RC(*this, *GG, 0);

    for (DSScalarMap::global_iterator I = ScalarMap.global_begin();
         I != ScalarMap.global_end(); ++I)
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(*I))
        if (!GV->isExternal() && GV->isConstant())
          RC.merge(ScalarMap[GV], GG->ScalarMap[GV]);
  }

  markIncompleteNodes(DSGraph::MarkFormalArgs);

  // Remove any nodes made dead due to merging...
  removeDeadNodes(DSGraph::KeepUnreachableGlobals);
}


//===----------------------------------------------------------------------===//
// Helper method implementations...
//

/// getValueDest - Return the DSNode that the actual value points to.
///
DSNodeHandle GraphBuilder::getValueDest(Value &Val) {
  Value *V = &Val;
  if (isa<Constant>(V) && cast<Constant>(V)->isNullValue())
    return 0;  // Null doesn't point to anything, don't add to ScalarMap!

  DSNodeHandle &NH = ScalarMap[V];
  if (!NH.isNull())
    return NH;     // Already have a node?  Just return it...

  // Otherwise we need to create a new node to point to.
  // Check first for constant expressions that must be traversed to
  // extract the actual value.
  DSNode* N;
  if (GlobalValue* GV = dyn_cast<GlobalValue>(V)) {
    // Create a new global node for this global variable.
    N = createNode(GV->getType()->getElementType());
    N->addGlobal(GV);
  } else if (Constant *C = dyn_cast<Constant>(V)) {
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::Cast) {
        if (isa<PointerType>(CE->getOperand(0)->getType()))
          NH = getValueDest(*CE->getOperand(0));
        else
          NH = createNode()->setUnknownNodeMarker();
      } else if (CE->getOpcode() == Instruction::GetElementPtr) {
        visitGetElementPtrInst(*CE);
        DSScalarMap::iterator I = ScalarMap.find(CE);
        assert(I != ScalarMap.end() && "GEP didn't get processed right?");
        NH = I->second;
      } else {
        // This returns a conservative unknown node for any unhandled ConstExpr
        return NH = createNode()->setUnknownNodeMarker();
      }
      if (NH.isNull()) {  // (getelementptr null, X) returns null
        ScalarMap.erase(V);
        return 0;
      }
      return NH;
    } else if (isa<UndefValue>(C)) {
      ScalarMap.erase(V);
      return 0;
    } else {
      assert(0 && "Unknown constant type!");
    }
    N = createNode(); // just create a shadow node
  } else {
    // Otherwise just create a shadow node
    N = createNode();
  }

  NH.setTo(N, 0);      // Remember that we are pointing to it...
  return NH;
}


/// getLink - This method is used to return the specified link in the
/// specified node if one exists.  If a link does not already exist (it's
/// null), then we create a new node, link it, then return it.  We must
/// specify the type of the Node field we are accessing so that we know what
/// type should be linked to if we need to create a new node.
///
DSNodeHandle &GraphBuilder::getLink(const DSNodeHandle &node, unsigned LinkNo) {
  DSNodeHandle &Node = const_cast<DSNodeHandle&>(node);
  DSNodeHandle &Link = Node.getLink(LinkNo);
  if (Link.isNull()) {
    // If the link hasn't been created yet, make and return a new shadow node
    Link = createNode();
  }
  return Link;
}


/// setDestTo - Set the ScalarMap entry for the specified value to point to the
/// specified destination.  If the Value already points to a node, make sure to
/// merge the two destinations together.
///
void GraphBuilder::setDestTo(Value &V, const DSNodeHandle &NH) {
  ScalarMap[&V].mergeWith(NH);
}


//===----------------------------------------------------------------------===//
// Specific instruction type handler implementations...
//

/// Alloca & Malloc instruction implementation - Simply create a new memory
/// object, pointing the scalar to it.
///
void GraphBuilder::handleAlloc(AllocationInst &AI, bool isHeap) {
  DSNode *N = createNode();
  if (isHeap)
    N->setHeapNodeMarker();
  else
    N->setAllocaNodeMarker();
  setDestTo(AI, N);
}

// PHINode - Make the scalar for the PHI node point to all of the things the
// incoming values point to... which effectively causes them to be merged.
//
void GraphBuilder::visitPHINode(PHINode &PN) {
  if (!isPointerType(PN.getType())) return; // Only pointer PHIs

  DSNodeHandle &PNDest = ScalarMap[&PN];
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    PNDest.mergeWith(getValueDest(*PN.getIncomingValue(i)));
}

void GraphBuilder::visitSelectInst(SelectInst &SI) {
  if (!isPointerType(SI.getType())) return; // Only pointer Selects

  DSNodeHandle &Dest = ScalarMap[&SI];
  Dest.mergeWith(getValueDest(*SI.getOperand(1)));
  Dest.mergeWith(getValueDest(*SI.getOperand(2)));
}

void GraphBuilder::visitSetCondInst(SetCondInst &SCI) {
  if (!isPointerType(SCI.getOperand(0)->getType()) ||
      isa<ConstantPointerNull>(SCI.getOperand(1))) return; // Only pointers
  ScalarMap[SCI.getOperand(0)].mergeWith(getValueDest(*SCI.getOperand(1)));
}


void GraphBuilder::visitGetElementPtrInst(User &GEP) {
  DSNodeHandle Value = getValueDest(*GEP.getOperand(0));
  if (Value.isNull())
    Value = createNode();

  // As a special case, if all of the index operands of GEP are constant zeros,
  // handle this just like we handle casts (ie, don't do much).
  bool AllZeros = true;
  for (unsigned i = 1, e = GEP.getNumOperands(); i != e; ++i)
    if (GEP.getOperand(i) !=
           Constant::getNullValue(GEP.getOperand(i)->getType())) {
      AllZeros = false;
      break;
    }

  // If all of the indices are zero, the result points to the operand without
  // applying the type.
  if (AllZeros || (!Value.isNull() &&
                   Value.getNode()->isNodeCompletelyFolded())) {
    setDestTo(GEP, Value);
    return;
  }


  const PointerType *PTy = cast<PointerType>(GEP.getOperand(0)->getType());
  const Type *CurTy = PTy->getElementType();

  if (Value.getNode()->mergeTypeInfo(CurTy, Value.getOffset())) {
    // If the node had to be folded... exit quickly
    setDestTo(GEP, Value);  // GEP result points to folded node
    return;
  }

  const TargetData &TD = Value.getNode()->getTargetData();

#if 0
  // Handle the pointer index specially...
  if (GEP.getNumOperands() > 1 &&
      (!isa<Constant>(GEP.getOperand(1)) ||
       !cast<Constant>(GEP.getOperand(1))->isNullValue())) {

    // If we already know this is an array being accessed, don't do anything...
    if (!TopTypeRec.isArray) {
      TopTypeRec.isArray = true;

      // If we are treating some inner field pointer as an array, fold the node
      // up because we cannot handle it right.  This can come because of
      // something like this:  &((&Pt->X)[1]) == &Pt->Y
      //
      if (Value.getOffset()) {
        // Value is now the pointer we want to GEP to be...
        Value.getNode()->foldNodeCompletely();
        setDestTo(GEP, Value);  // GEP result points to folded node
        return;
      } else {
        // This is a pointer to the first byte of the node.  Make sure that we
        // are pointing to the outter most type in the node.
        // FIXME: We need to check one more case here...
      }
    }
  }
#endif

  // All of these subscripts are indexing INTO the elements we have...
  unsigned Offset = 0;
  for (gep_type_iterator I = gep_type_begin(GEP), E = gep_type_end(GEP);
       I != E; ++I)
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      unsigned FieldNo =
           (unsigned)cast<ConstantUInt>(I.getOperand())->getValue();
      Offset += (unsigned)TD.getStructLayout(STy)->MemberOffsets[FieldNo];
    } else if (const PointerType *PTy = dyn_cast<PointerType>(*I)) {
      if (!isa<Constant>(I.getOperand()) ||
          !cast<Constant>(I.getOperand())->isNullValue())
        Value.getNode()->setArrayMarker();
    }


#if 0
    if (const SequentialType *STy = cast<SequentialType>(*I)) {
      CurTy = STy->getElementType();
      if (ConstantSInt *CS = dyn_cast<ConstantSInt>(GEP.getOperand(i))) {
        Offset += CS->getValue()*TD.getTypeSize(CurTy);
      } else {
        // Variable index into a node.  We must merge all of the elements of the
        // sequential type here.
        if (isa<PointerType>(STy))
          std::cerr << "Pointer indexing not handled yet!\n";
        else {
          const ArrayType *ATy = cast<ArrayType>(STy);
          unsigned ElSize = TD.getTypeSize(CurTy);
          DSNode *N = Value.getNode();
          assert(N && "Value must have a node!");
          unsigned RawOffset = Offset+Value.getOffset();

          // Loop over all of the elements of the array, merging them into the
          // zeroth element.
          for (unsigned i = 1, e = ATy->getNumElements(); i != e; ++i)
            // Merge all of the byte components of this array element
            for (unsigned j = 0; j != ElSize; ++j)
              N->mergeIndexes(RawOffset+j, RawOffset+i*ElSize+j);
        }
      }
    }
#endif

  // Add in the offset calculated...
  Value.setOffset(Value.getOffset()+Offset);

  // Check the offset
  DSNode *N = Value.getNode();
  if (N &&
      !N->isNodeCompletelyFolded() &&
      (N->getSize() != 0 || Offset != 0) &&
      !N->isForwarding()) {
    if ((Offset >= N->getSize()) || int(Offset) < 0) {
      // Accessing offsets out of node size range
      // This is seen in the "magic" struct in named (from bind), where the
      // fourth field is an array of length 0, presumably used to create struct
      // instances of different sizes

      // Collapse the node since its size is now variable
      N->foldNodeCompletely();
    }
  }
  
  // Value is now the pointer we want to GEP to be...  
  setDestTo(GEP, Value);
}

void GraphBuilder::visitLoadInst(LoadInst &LI) {
  DSNodeHandle Ptr = getValueDest(*LI.getOperand(0));
  if (Ptr.isNull())
    Ptr = createNode();

  // Make that the node is read from...
  Ptr.getNode()->setReadMarker();

  // Ensure a typerecord exists...
  Ptr.getNode()->mergeTypeInfo(LI.getType(), Ptr.getOffset(), false);

  if (isPointerType(LI.getType()))
    setDestTo(LI, getLink(Ptr));
}

void GraphBuilder::visitStoreInst(StoreInst &SI) {
  const Type *StoredTy = SI.getOperand(0)->getType();
  DSNodeHandle Dest = getValueDest(*SI.getOperand(1));
  if (Dest.isNull()) return;

  // Mark that the node is written to...
  Dest.getNode()->setModifiedMarker();

  // Ensure a type-record exists...
  Dest.getNode()->mergeTypeInfo(StoredTy, Dest.getOffset());

  // Avoid adding edges from null, or processing non-"pointer" stores
  if (isPointerType(StoredTy))
    Dest.addEdgeTo(getValueDest(*SI.getOperand(0)));
}

void GraphBuilder::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() && isPointerType(RI.getOperand(0)->getType()))
    RetNode->mergeWith(getValueDest(*RI.getOperand(0)));
}

void GraphBuilder::visitVAArgInst(VAArgInst &I) {
  //FIXME: also updates the argument
  DSNodeHandle Ptr = getValueDest(*I.getOperand(0));
  if (Ptr.isNull()) return;

  // Make that the node is read from.
  Ptr.getNode()->setReadMarker();

  // Ensure a type record exists.
  DSNode *PtrN = Ptr.getNode();
  PtrN->mergeTypeInfo(I.getType(), Ptr.getOffset(), false);

  if (isPointerType(I.getType()))
    setDestTo(I, getLink(Ptr));
}


void GraphBuilder::visitCallInst(CallInst &CI) {
  visitCallSite(&CI);
}

void GraphBuilder::visitInvokeInst(InvokeInst &II) {
  visitCallSite(&II);
}

void GraphBuilder::visitCallSite(CallSite CS) {
  Value *Callee = CS.getCalledValue();

  // Special case handling of certain libc allocation functions here.
  if (Function *F = dyn_cast<Function>(Callee))
    if (F->isExternal())
      switch (F->getIntrinsicID()) {
      case Intrinsic::vastart:
        getValueDest(*CS.getInstruction()).getNode()->setAllocaNodeMarker();
        return;
      case Intrinsic::vacopy:
        getValueDest(*CS.getInstruction()).
          mergeWith(getValueDest(**(CS.arg_begin())));
        return;
      case Intrinsic::vaend:
        return;  // noop
      case Intrinsic::memcpy_i32: 
      case Intrinsic::memcpy_i64: {
        //write first location
        if (DSNode *N = getValueDest(**CS.arg_begin()).getNode())
          N->setHeapNodeMarker()->setModifiedMarker();
        //and read second pointer
        if (DSNode *N = getValueDest(**(CS.arg_begin() + 1)).getNode())
          N->setReadMarker();
        return;
      }
      case Intrinsic::memmove_i32:
      case Intrinsic::memmove_i64: {
        // Merge the first & second arguments, and mark the memory read and
        // modified.
        DSNodeHandle RetNH = getValueDest(**CS.arg_begin());
        RetNH.mergeWith(getValueDest(**(CS.arg_begin()+1)));
        if (DSNode *N = RetNH.getNode())
          N->setModifiedMarker()->setReadMarker();
        return;
      }
      case Intrinsic::memset_i32:
      case Intrinsic::memset_i64:
        // Mark the memory modified.
        if (DSNode *N = getValueDest(**CS.arg_begin()).getNode())
          N->setModifiedMarker();
        return;
      default:
        // Determine if the called function is one of the specified heap
        // allocation functions
        for (cl::list<std::string>::iterator AllocFunc = AllocList.begin(),
             LastAllocFunc = AllocList.end();
             AllocFunc != LastAllocFunc;
             ++AllocFunc) {
          if (F->getName() == *(AllocFunc)) {
            setDestTo(*CS.getInstruction(),
                      createNode()->setHeapNodeMarker()->setModifiedMarker());
            return;
          }
        }

        // Determine if the called function is one of the specified heap
        // free functions
        for (cl::list<std::string>::iterator FreeFunc = FreeList.begin(),
             LastFreeFunc = FreeList.end();
             FreeFunc != LastFreeFunc;
             ++FreeFunc) {
          if (F->getName() == *(FreeFunc)) {
            // Mark that the node is written to...
            if (DSNode *N = getValueDest(*(CS.getArgument(0))).getNode())
              N->setModifiedMarker()->setHeapNodeMarker();
            return;
          }
        }

        if (F->getName() == "calloc" || F->getName() == "posix_memalign" ||
            F->getName() == "memalign" || F->getName() == "valloc") {
          setDestTo(*CS.getInstruction(),
                    createNode()->setHeapNodeMarker()->setModifiedMarker());
          return;
        } else if (F->getName() == "realloc") {
          DSNodeHandle RetNH = getValueDest(*CS.getInstruction());
          if (CS.arg_begin() != CS.arg_end())
            RetNH.mergeWith(getValueDest(**CS.arg_begin()));
          if (DSNode *N = RetNH.getNode())
            N->setHeapNodeMarker()->setModifiedMarker()->setReadMarker();
          return;
        } else if (F->getName() == "memmove") {
          // Merge the first & second arguments, and mark the memory read and
          // modified.
          DSNodeHandle RetNH = getValueDest(**CS.arg_begin());
          RetNH.mergeWith(getValueDest(**(CS.arg_begin()+1)));
          if (DSNode *N = RetNH.getNode())
            N->setModifiedMarker()->setReadMarker();
          return;

        } else if (F->getName() == "atoi" || F->getName() == "atof" ||
                   F->getName() == "atol" || F->getName() == "atoll" ||
                   F->getName() == "remove" || F->getName() == "unlink" ||
                   F->getName() == "rename" || F->getName() == "memcmp" ||
                   F->getName() == "strcmp" || F->getName() == "strncmp" ||
                   F->getName() == "execl" || F->getName() == "execlp" ||
                   F->getName() == "execle" || F->getName() == "execv" ||
                   F->getName() == "execvp" || F->getName() == "chmod" ||
                   F->getName() == "puts" || F->getName() == "write" ||
                   F->getName() == "open" || F->getName() == "create" ||
                   F->getName() == "truncate" || F->getName() == "chdir" ||
                   F->getName() == "mkdir" || F->getName() == "rmdir") {
          // These functions read all of their pointer operands.
          for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
               AI != E; ++AI) {
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker();
          }
          return;
        } else if (F->getName() == "read" || F->getName() == "pipe" ||
                   F->getName() == "wait" || F->getName() == "time") {
          // These functions write all of their pointer operands.
          for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
               AI != E; ++AI) {
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setModifiedMarker();
          }
          return;
        } else if (F->getName() == "stat" || F->getName() == "fstat" ||
                   F->getName() == "lstat") {
          // These functions read their first operand if its a pointer.
          CallSite::arg_iterator AI = CS.arg_begin();
          if (isPointerType((*AI)->getType())) {
            DSNodeHandle Path = getValueDest(**AI);
            if (DSNode *N = Path.getNode()) N->setReadMarker();
          }

          // Then they write into the stat buffer.
          DSNodeHandle StatBuf = getValueDest(**++AI);
          if (DSNode *N = StatBuf.getNode()) {
            N->setModifiedMarker();
            const Type *StatTy = F->getFunctionType()->getParamType(1);
            if (const PointerType *PTy = dyn_cast<PointerType>(StatTy))
              N->mergeTypeInfo(PTy->getElementType(), StatBuf.getOffset());
          }
          return;
        } else if (F->getName() == "strtod" || F->getName() == "strtof" ||
                   F->getName() == "strtold") {
          // These functions read the first pointer
          if (DSNode *Str = getValueDest(**CS.arg_begin()).getNode()) {
            Str->setReadMarker();
            // If the second parameter is passed, it will point to the first
            // argument node.
            const DSNodeHandle &EndPtrNH = getValueDest(**(CS.arg_begin()+1));
            if (DSNode *End = EndPtrNH.getNode()) {
              End->mergeTypeInfo(PointerType::get(Type::SByteTy),
                                 EndPtrNH.getOffset(), false);
              End->setModifiedMarker();
              DSNodeHandle &Link = getLink(EndPtrNH);
              Link.mergeWith(getValueDest(**CS.arg_begin()));
            }
          }
          return;
        } else if (F->getName() == "fopen" || F->getName() == "fdopen" ||
                   F->getName() == "freopen") {
          // These functions read all of their pointer operands.
          for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
               AI != E; ++AI)
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker();

          // fopen allocates in an unknown way and writes to the file
          // descriptor.  Also, merge the allocated type into the node.
          DSNodeHandle Result = getValueDest(*CS.getInstruction());
          if (DSNode *N = Result.getNode()) {
            N->setModifiedMarker()->setUnknownNodeMarker();
            const Type *RetTy = F->getFunctionType()->getReturnType();
            if (const PointerType *PTy = dyn_cast<PointerType>(RetTy))
              N->mergeTypeInfo(PTy->getElementType(), Result.getOffset());
          }

          // If this is freopen, merge the file descriptor passed in with the
          // result.
          if (F->getName() == "freopen") {
            // ICC doesn't handle getting the iterator, decrementing and
            // dereferencing it in one operation without error. Do it in 2 steps
            CallSite::arg_iterator compit = CS.arg_end();
            Result.mergeWith(getValueDest(**--compit));
          }
          return;
        } else if (F->getName() == "fclose" && CS.arg_end()-CS.arg_begin() ==1){
          // fclose reads and deallocates the memory in an unknown way for the
          // file descriptor.  It merges the FILE type into the descriptor.
          DSNodeHandle H = getValueDest(**CS.arg_begin());
          if (DSNode *N = H.getNode()) {
            N->setReadMarker()->setUnknownNodeMarker();
            const Type *ArgTy = F->getFunctionType()->getParamType(0);
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }
          return;
        } else if (CS.arg_end()-CS.arg_begin() == 1 &&
                   (F->getName() == "fflush" || F->getName() == "feof" ||
                    F->getName() == "fileno" || F->getName() == "clearerr" ||
                    F->getName() == "rewind" || F->getName() == "ftell" ||
                    F->getName() == "ferror" || F->getName() == "fgetc" ||
                    F->getName() == "fgetc" || F->getName() == "_IO_getc")) {
          // fflush reads and writes the memory for the file descriptor.  It
          // merges the FILE type into the descriptor.
          DSNodeHandle H = getValueDest(**CS.arg_begin());
          if (DSNode *N = H.getNode()) {
            N->setReadMarker()->setModifiedMarker();

            const Type *ArgTy = F->getFunctionType()->getParamType(0);
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }
          return;
        } else if (CS.arg_end()-CS.arg_begin() == 4 &&
                   (F->getName() == "fwrite" || F->getName() == "fread")) {
          // fread writes the first operand, fwrite reads it.  They both
          // read/write the FILE descriptor, and merges the FILE type.
          CallSite::arg_iterator compit = CS.arg_end();
          DSNodeHandle H = getValueDest(**--compit);
          if (DSNode *N = H.getNode()) {
            N->setReadMarker()->setModifiedMarker();
            const Type *ArgTy = F->getFunctionType()->getParamType(3);
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }

          H = getValueDest(**CS.arg_begin());
          if (DSNode *N = H.getNode())
            if (F->getName() == "fwrite")
              N->setReadMarker();
            else
              N->setModifiedMarker();
          return;
        } else if (F->getName() == "fgets" && CS.arg_end()-CS.arg_begin() == 3){
          // fgets reads and writes the memory for the file descriptor.  It
          // merges the FILE type into the descriptor, and writes to the
          // argument.  It returns the argument as well.
          CallSite::arg_iterator AI = CS.arg_begin();
          DSNodeHandle H = getValueDest(**AI);
          if (DSNode *N = H.getNode())
            N->setModifiedMarker();                        // Writes buffer
          H.mergeWith(getValueDest(*CS.getInstruction())); // Returns buffer
          ++AI; ++AI;

          // Reads and writes file descriptor, merge in FILE type.
          H = getValueDest(**AI);
          if (DSNode *N = H.getNode()) {
            N->setReadMarker()->setModifiedMarker();
            const Type *ArgTy = F->getFunctionType()->getParamType(2);
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }
          return;
        } else if (F->getName() == "ungetc" || F->getName() == "fputc" ||
                   F->getName() == "fputs" || F->getName() == "putc" ||
                   F->getName() == "ftell" || F->getName() == "rewind" ||
                   F->getName() == "_IO_putc") {
          // These functions read and write the memory for the file descriptor,
          // which is passes as the last argument.
          CallSite::arg_iterator compit = CS.arg_end();
          DSNodeHandle H = getValueDest(**--compit);
          if (DSNode *N = H.getNode()) {
            N->setReadMarker()->setModifiedMarker();
            FunctionType::param_iterator compit2 = F->getFunctionType()->param_end();
            const Type *ArgTy = *--compit2;
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }

          // Any pointer arguments are read.
          for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
               AI != E; ++AI)
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker();
          return;
        } else if (F->getName() == "fseek" || F->getName() == "fgetpos" ||
                   F->getName() == "fsetpos") {
          // These functions read and write the memory for the file descriptor,
          // and read/write all other arguments.
          DSNodeHandle H = getValueDest(**CS.arg_begin());
          if (DSNode *N = H.getNode()) {
            FunctionType::param_iterator compit2 = F->getFunctionType()->param_end();
            const Type *ArgTy = *--compit2;
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }

          // Any pointer arguments are read.
          for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
               AI != E; ++AI)
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker()->setModifiedMarker();
          return;
        } else if (F->getName() == "printf" || F->getName() == "fprintf" ||
                   F->getName() == "sprintf") {
          CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();

          if (F->getName() == "fprintf") {
            // fprintf reads and writes the FILE argument, and applies the type
            // to it.
            DSNodeHandle H = getValueDest(**AI);
            if (DSNode *N = H.getNode()) {
              N->setModifiedMarker();
              const Type *ArgTy = (*AI)->getType();
              if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
                N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
            }
          } else if (F->getName() == "sprintf") {
            // sprintf writes the first string argument.
            DSNodeHandle H = getValueDest(**AI++);
            if (DSNode *N = H.getNode()) {
              N->setModifiedMarker();
              const Type *ArgTy = (*AI)->getType();
              if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
                N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
            }
          }

          for (; AI != E; ++AI) {
            // printf reads all pointer arguments.
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker();
          }
          return;
        } else if (F->getName() == "vprintf" || F->getName() == "vfprintf" ||
                   F->getName() == "vsprintf") {
          CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();

          if (F->getName() == "vfprintf") {
            // ffprintf reads and writes the FILE argument, and applies the type
            // to it.
            DSNodeHandle H = getValueDest(**AI);
            if (DSNode *N = H.getNode()) {
              N->setModifiedMarker()->setReadMarker();
              const Type *ArgTy = (*AI)->getType();
              if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
                N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
            }
            ++AI;
          } else if (F->getName() == "vsprintf") {
            // vsprintf writes the first string argument.
            DSNodeHandle H = getValueDest(**AI++);
            if (DSNode *N = H.getNode()) {
              N->setModifiedMarker();
              const Type *ArgTy = (*AI)->getType();
              if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
                N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
            }
          }

          // Read the format
          if (AI != E) {
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker();
            ++AI;
          }

          // Read the valist, and the pointed-to objects.
          if (AI != E && isPointerType((*AI)->getType())) {
            const DSNodeHandle &VAList = getValueDest(**AI);
            if (DSNode *N = VAList.getNode()) {
              N->setReadMarker();
              N->mergeTypeInfo(PointerType::get(Type::SByteTy),
                               VAList.getOffset(), false);

              DSNodeHandle &VAListObjs = getLink(VAList);
              VAListObjs.getNode()->setReadMarker();
            }
          }

          return;
        } else if (F->getName() == "scanf" || F->getName() == "fscanf" ||
                   F->getName() == "sscanf") {
          CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();

          if (F->getName() == "fscanf") {
            // fscanf reads and writes the FILE argument, and applies the type
            // to it.
            DSNodeHandle H = getValueDest(**AI);
            if (DSNode *N = H.getNode()) {
              N->setReadMarker();
              const Type *ArgTy = (*AI)->getType();
              if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
                N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
            }
          } else if (F->getName() == "sscanf") {
            // sscanf reads the first string argument.
            DSNodeHandle H = getValueDest(**AI++);
            if (DSNode *N = H.getNode()) {
              N->setReadMarker();
              const Type *ArgTy = (*AI)->getType();
              if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
                N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
            }
          }

          for (; AI != E; ++AI) {
            // scanf writes all pointer arguments.
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setModifiedMarker();
          }
          return;
        } else if (F->getName() == "strtok") {
          // strtok reads and writes the first argument, returning it.  It reads
          // its second arg.  FIXME: strtok also modifies some hidden static
          // data.  Someday this might matter.
          CallSite::arg_iterator AI = CS.arg_begin();
          DSNodeHandle H = getValueDest(**AI++);
          if (DSNode *N = H.getNode()) {
            N->setReadMarker()->setModifiedMarker();      // Reads/Writes buffer
            const Type *ArgTy = F->getFunctionType()->getParamType(0);
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }
          H.mergeWith(getValueDest(*CS.getInstruction())); // Returns buffer

          H = getValueDest(**AI);       // Reads delimiter
          if (DSNode *N = H.getNode()) {
            N->setReadMarker();
            const Type *ArgTy = F->getFunctionType()->getParamType(1);
            if (const PointerType *PTy = dyn_cast<PointerType>(ArgTy))
              N->mergeTypeInfo(PTy->getElementType(), H.getOffset());
          }
          return;
        } else if (F->getName() == "strchr" || F->getName() == "strrchr" ||
                   F->getName() == "strstr") {
          // These read their arguments, and return the first one
          DSNodeHandle H = getValueDest(**CS.arg_begin());
          H.mergeWith(getValueDest(*CS.getInstruction())); // Returns buffer

          for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
               AI != E; ++AI)
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker();

          if (DSNode *N = H.getNode())
            N->setReadMarker();
          return;
        } else if (F->getName() == "__assert_fail") {
          for (CallSite::arg_iterator AI = CS.arg_begin(), E = CS.arg_end();
               AI != E; ++AI)
            if (isPointerType((*AI)->getType()))
              if (DSNode *N = getValueDest(**AI).getNode())
                N->setReadMarker();
          return;
        } else if (F->getName() == "modf" && CS.arg_end()-CS.arg_begin() == 2) {
          // This writes its second argument, and forces it to double.
          CallSite::arg_iterator compit = CS.arg_end();
          DSNodeHandle H = getValueDest(**--compit);
          if (DSNode *N = H.getNode()) {
            N->setModifiedMarker();
            N->mergeTypeInfo(Type::DoubleTy, H.getOffset());
          }
          return;
        } else if (F->getName() == "strcat" || F->getName() == "strncat") {
          //This might be making unsafe assumptions about usage
          //Merge return and first arg
          DSNodeHandle RetNH = getValueDest(*CS.getInstruction());
          RetNH.mergeWith(getValueDest(**CS.arg_begin()));
          if (DSNode *N = RetNH.getNode())
            N->setHeapNodeMarker()->setModifiedMarker()->setReadMarker();
          //and read second pointer
          if (DSNode *N = getValueDest(**(CS.arg_begin() + 1)).getNode())
            N->setReadMarker();
          return;
        } else {
          // Unknown function, warn if it returns a pointer type or takes a
          // pointer argument.
          bool Warn = isPointerType(CS.getInstruction()->getType());
          if (!Warn)
            for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
                 I != E; ++I)
              if (isPointerType((*I)->getType())) {
                Warn = true;
                break;
              }
          if (Warn)
            std::cerr << "WARNING: Call to unknown external function '"
                      << F->getName() << "' will cause pessimistic results!\n";
        }
      }


  // Set up the return value...
  DSNodeHandle RetVal;
  Instruction *I = CS.getInstruction();
  if (isPointerType(I->getType()))
    RetVal = getValueDest(*I);

  DSNode *CalleeNode = 0;
  if (DisableDirectCallOpt || !isa<Function>(Callee)) {
    CalleeNode = getValueDest(*Callee).getNode();
    if (CalleeNode == 0) {
      std::cerr << "WARNING: Program is calling through a null pointer?\n"<< *I;
      return;  // Calling a null pointer?
    }
  }

  std::vector<DSNodeHandle> Args;
  Args.reserve(CS.arg_end()-CS.arg_begin());

  // Calculate the arguments vector...
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end(); I != E; ++I)
    if (isPointerType((*I)->getType()))
      Args.push_back(getValueDest(**I));

  // Add a new function call entry...
  if (CalleeNode)
    FunctionCalls->push_back(DSCallSite(CS, RetVal, CalleeNode, Args));
  else
    FunctionCalls->push_back(DSCallSite(CS, RetVal, cast<Function>(Callee),
                                        Args));
}

void GraphBuilder::visitFreeInst(FreeInst &FI) {
  // Mark that the node is written to...
  if (DSNode *N = getValueDest(*FI.getOperand(0)).getNode())
    N->setModifiedMarker()->setHeapNodeMarker();
}

/// Handle casts...
void GraphBuilder::visitCastInst(CastInst &CI) {
  if (isPointerType(CI.getType()))
    if (isPointerType(CI.getOperand(0)->getType())) {
      DSNodeHandle Ptr = getValueDest(*CI.getOperand(0));
      if (Ptr.getNode() == 0) return;

      // Cast one pointer to the other, just act like a copy instruction
      setDestTo(CI, Ptr);
    } else {
      // Cast something (floating point, small integer) to a pointer.  We need
      // to track the fact that the node points to SOMETHING, just something we
      // don't know about.  Make an "Unknown" node.
      //
      setDestTo(CI, createNode()->setUnknownNodeMarker());
    }
}


// visitInstruction - For all other instruction types, if we have any arguments
// that are of pointer type, make them have unknown composition bits, and merge
// the nodes together.
void GraphBuilder::visitInstruction(Instruction &Inst) {
  DSNodeHandle CurNode;
  if (isPointerType(Inst.getType()))
    CurNode = getValueDest(Inst);
  for (User::op_iterator I = Inst.op_begin(), E = Inst.op_end(); I != E; ++I)
    if (isPointerType((*I)->getType()))
      CurNode.mergeWith(getValueDest(**I));

  if (DSNode *N = CurNode.getNode())
    N->setUnknownNodeMarker();
}



//===----------------------------------------------------------------------===//
// LocalDataStructures Implementation
//===----------------------------------------------------------------------===//

// MergeConstantInitIntoNode - Merge the specified constant into the node
// pointed to by NH.
void GraphBuilder::MergeConstantInitIntoNode(DSNodeHandle &NH, Constant *C) {
  // Ensure a type-record exists...
  DSNode *NHN = NH.getNode();
  NHN->mergeTypeInfo(C->getType(), NH.getOffset());

  if (C->getType()->isFirstClassType()) {
    if (isPointerType(C->getType()))
      // Avoid adding edges from null, or processing non-"pointer" stores
      NH.addEdgeTo(getValueDest(*C));
    return;
  }

  const TargetData &TD = NH.getNode()->getTargetData();

  if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
    for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
      // We don't currently do any indexing for arrays...
      MergeConstantInitIntoNode(NH, cast<Constant>(CA->getOperand(i)));
  } else if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
    const StructLayout *SL = TD.getStructLayout(CS->getType());
    for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i) {
      DSNode *NHN = NH.getNode();
      DSNodeHandle NewNH(NHN, NH.getOffset()+(unsigned)SL->MemberOffsets[i]);
      MergeConstantInitIntoNode(NewNH, cast<Constant>(CS->getOperand(i)));
    }
  } else if (isa<ConstantAggregateZero>(C) || isa<UndefValue>(C)) {
    // Noop
  } else {
    assert(0 && "Unknown constant type!");
  }
}

void GraphBuilder::mergeInGlobalInitializer(GlobalVariable *GV) {
  assert(!GV->isExternal() && "Cannot merge in external global!");
  // Get a node handle to the global node and merge the initializer into it.
  DSNodeHandle NH = getValueDest(*GV);
  MergeConstantInitIntoNode(NH, GV->getInitializer());
}


/// BuildGlobalECs - Look at all of the nodes in the globals graph.  If any node
/// contains multiple globals, DSA will never, ever, be able to tell the globals
/// apart.  Instead of maintaining this information in all of the graphs
/// throughout the entire program, store only a single global (the "leader") in
/// the graphs, and build equivalence classes for the rest of the globals.
static void BuildGlobalECs(DSGraph &GG, std::set<GlobalValue*> &ECGlobals) {
  DSScalarMap &SM = GG.getScalarMap();
  EquivalenceClasses<GlobalValue*> &GlobalECs = SM.getGlobalECs();
  for (DSGraph::node_iterator I = GG.node_begin(), E = GG.node_end();
       I != E; ++I) {
    if (I->getGlobalsList().size() <= 1) continue;

    // First, build up the equivalence set for this block of globals.
    const std::vector<GlobalValue*> &GVs = I->getGlobalsList();
    GlobalValue *First = GVs[0];
    for (unsigned i = 1, e = GVs.size(); i != e; ++i)
      GlobalECs.unionSets(First, GVs[i]);

    // Next, get the leader element.
    assert(First == GlobalECs.getLeaderValue(First) &&
           "First did not end up being the leader?");

    // Next, remove all globals from the scalar map that are not the leader.
    assert(GVs[0] == First && "First had to be at the front!");
    for (unsigned i = 1, e = GVs.size(); i != e; ++i) {
      ECGlobals.insert(GVs[i]);
      SM.erase(SM.find(GVs[i]));
    }

    // Finally, change the global node to only contain the leader.
    I->clearGlobals();
    I->addGlobal(First);
  }

  DEBUG(GG.AssertGraphOK());
}

/// EliminateUsesOfECGlobals - Once we have determined that some globals are in
/// really just equivalent to some other globals, remove the globals from the
/// specified DSGraph (if present), and merge any nodes with their leader nodes.
static void EliminateUsesOfECGlobals(DSGraph &G,
                                     const std::set<GlobalValue*> &ECGlobals) {
  DSScalarMap &SM = G.getScalarMap();
  EquivalenceClasses<GlobalValue*> &GlobalECs = SM.getGlobalECs();

  bool MadeChange = false;
  for (DSScalarMap::global_iterator GI = SM.global_begin(), E = SM.global_end();
       GI != E; ) {
    GlobalValue *GV = *GI++;
    if (!ECGlobals.count(GV)) continue;

    const DSNodeHandle &GVNH = SM[GV];
    assert(!GVNH.isNull() && "Global has null NH!?");

    // Okay, this global is in some equivalence class.  Start by finding the
    // leader of the class.
    GlobalValue *Leader = GlobalECs.getLeaderValue(GV);

    // If the leader isn't already in the graph, insert it into the node
    // corresponding to GV.
    if (!SM.global_count(Leader)) {
      GVNH.getNode()->addGlobal(Leader);
      SM[Leader] = GVNH;
    } else {
      // Otherwise, the leader is in the graph, make sure the nodes are the
      // merged in the specified graph.
      const DSNodeHandle &LNH = SM[Leader];
      if (LNH.getNode() != GVNH.getNode())
        LNH.mergeWith(GVNH);
    }

    // Next step, remove the global from the DSNode.
    GVNH.getNode()->removeGlobal(GV);

    // Finally, remove the global from the ScalarMap.
    SM.erase(GV);
    MadeChange = true;
  }

  DEBUG(if(MadeChange) G.AssertGraphOK());
}

bool LocalDataStructures::runOnModule(Module &M) {
  const TargetData &TD = getAnalysis<TargetData>();

  // First step, build the globals graph.
  GlobalsGraph = new DSGraph(GlobalECs, TD);
  {
    GraphBuilder GGB(*GlobalsGraph);

    // Add initializers for all of the globals to the globals graph.
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I)
      if (!I->isExternal())
        GGB.mergeInGlobalInitializer(I);
  }

  // Next step, iterate through the nodes in the globals graph, unioning
  // together the globals into equivalence classes.
  std::set<GlobalValue*> ECGlobals;
  BuildGlobalECs(*GlobalsGraph, ECGlobals);
  DEBUG(std::cerr << "Eliminating " << ECGlobals.size() << " EC Globals!\n");
  ECGlobals.clear();

  // Calculate all of the graphs...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      DSInfo.insert(std::make_pair(I, new DSGraph(GlobalECs, TD, *I,
                                                  GlobalsGraph)));

  GlobalsGraph->removeTriviallyDeadNodes();
  GlobalsGraph->markIncompleteNodes(DSGraph::MarkFormalArgs);

  // Now that we've computed all of the graphs, and merged all of the info into
  // the globals graph, see if we have further constrained the globals in the
  // program if so, update GlobalECs and remove the extraneous globals from the
  // program.
  BuildGlobalECs(*GlobalsGraph, ECGlobals);
  if (!ECGlobals.empty()) {
    DEBUG(std::cerr << "Eliminating " << ECGlobals.size() << " EC Globals!\n");
    for (hash_map<Function*, DSGraph*>::iterator I = DSInfo.begin(),
           E = DSInfo.end(); I != E; ++I)
      EliminateUsesOfECGlobals(*I->second, ECGlobals);
  }

  return false;
}

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void LocalDataStructures::releaseMemory() {
  for (hash_map<Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I) {
    I->second->getReturnNodes().erase(I->first);
    if (I->second->getReturnNodes().empty())
      delete I->second;
  }

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
  delete GlobalsGraph;
  GlobalsGraph = 0;
}


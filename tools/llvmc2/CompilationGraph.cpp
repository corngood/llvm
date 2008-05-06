//===--- CompilationGraph.cpp - The LLVM Compiler Driver --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Compilation graph - implementation.
//
//===----------------------------------------------------------------------===//

#include "CompilationGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <queue>
#include <stdexcept>

using namespace llvm;
using namespace llvmc;

extern cl::list<std::string> InputFilenames;
extern cl::opt<std::string> OutputFilename;
extern cl::list<std::string> Languages;

namespace {

  // Return the edge with the maximum weight.
  template <class C>
  const Edge* ChooseEdge(const C& EdgesContainer,
                         const std::string& NodeName = "root") {
    const Edge* MaxEdge = 0;
    unsigned MaxWeight = 0;
    bool SingleMax = true;

    for (typename C::const_iterator B = EdgesContainer.begin(),
           E = EdgesContainer.end(); B != E; ++B) {
      const Edge* E = B->getPtr();
      unsigned EW = E->Weight();
      if (EW > MaxWeight) {
        MaxEdge = E;
        MaxWeight = EW;
        SingleMax = true;
      }
      else if (EW == MaxWeight) {
        SingleMax = false;
      }
    }

    if (!SingleMax)
      throw std::runtime_error("Node " + NodeName +
                               ": multiple maximal outward edges found!"
                               " Most probably a specification error.");
    if (!MaxEdge)
      throw std::runtime_error("Node " + NodeName +
                               ": no maximal outward edge found!"
                               " Most probably a specification error.");
    return MaxEdge;
  }

}

CompilationGraph::CompilationGraph() {
  NodesMap["root"] = Node(this);
}

Node& CompilationGraph::getNode(const std::string& ToolName) {
  nodes_map_type::iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in the graph");
  return I->second;
}

const Node& CompilationGraph::getNode(const std::string& ToolName) const {
  nodes_map_type::const_iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in the graph!");
  return I->second;
}

const std::string& CompilationGraph::getLanguage(const sys::Path& File) const {
  LanguageMap::const_iterator Lang = ExtsToLangs.find(File.getSuffix());
  if (Lang == ExtsToLangs.end())
    throw std::runtime_error("Unknown suffix: " + File.getSuffix() + '!');
  return Lang->second;
}

const CompilationGraph::tools_vector_type&
CompilationGraph::getToolsVector(const std::string& LangName) const
{
  tools_map_type::const_iterator I = ToolsMap.find(LangName);
  if (I == ToolsMap.end())
    throw std::runtime_error("No tool corresponding to the language "
                             + LangName + "found!");
  return I->second;
}

void CompilationGraph::insertNode(Tool* V) {
  if (NodesMap.count(V->Name()) == 0) {
    Node N;
    N.OwningGraph = this;
    N.ToolPtr = V;
    NodesMap[V->Name()] = N;
  }
}

void CompilationGraph::insertEdge(const std::string& A, Edge* E) {
  Node& B = getNode(E->ToolName());
  if (A == "root") {
    const std::string& InputLanguage = B.ToolPtr->InputLanguage();
    ToolsMap[InputLanguage].push_back(IntrusiveRefCntPtr<Edge>(E));
    NodesMap["root"].AddEdge(E);
  }
  else {
    Node& N = getNode(A);
    N.AddEdge(E);
  }
  // Increase the inward edge counter.
  B.IncrInEdges();
}

namespace {
  sys::Path MakeTempFile(const sys::Path& TempDir, const std::string& BaseName,
                         const std::string& Suffix) {
    sys::Path Out = TempDir;
    Out.appendComponent(BaseName);
    Out.appendSuffix(Suffix);
    Out.makeUnique(true, NULL);
    return Out;
  }
}

// Pass input file through the chain until we bump into a Join node or
// a node that says that it is the last.
void CompilationGraph::PassThroughGraph (const sys::Path& InFile,
                                         const Node* StartNode,
                                         const sys::Path& TempDir) const {
  bool Last = false;
  sys::Path In = InFile;
  const Node* CurNode = StartNode;

  while(!Last) {
    sys::Path Out;
    Tool* CurTool = CurNode->ToolPtr.getPtr();

    if (CurTool->IsJoin()) {
      JoinTool& JT = dynamic_cast<JoinTool&>(*CurTool);
      JT.AddToJoinList(In);
      break;
    }

    // Since toolchains do not have to end with a Join node, we should
    // check if this Node is the last.
    if (!CurNode->HasChildren() || CurTool->IsLast()) {
      if (!OutputFilename.empty()) {
        Out.set(OutputFilename);
      }
      else {
        Out.set(In.getBasename());
        Out.appendSuffix(CurTool->OutputSuffix());
      }
      Last = true;
    }
    else {
      Out = MakeTempFile(TempDir, In.getBasename(), CurTool->OutputSuffix());
    }

    if (CurTool->GenerateAction(In, Out).Execute() != 0)
      throw std::runtime_error("Tool returned error code!");

    if (Last)
      return;

    CurNode = &getNode(ChooseEdge(CurNode->OutEdges,
                                  CurNode->Name())->ToolName());
    In = Out; Out.clear();
  }
}

// Sort the nodes in topological order.
void CompilationGraph::TopologicalSort(std::vector<const Node*>& Out) {
  std::queue<const Node*> Q;
  Q.push(&getNode("root"));

  while (!Q.empty()) {
    const Node* A = Q.front();
    Q.pop();
    Out.push_back(A);
    for (Node::const_iterator EB = A->EdgesBegin(), EE = A->EdgesEnd();
         EB != EE; ++EB) {
      Node* B = &getNode((*EB)->ToolName());
      B->DecrInEdges();
      if (B->HasNoInEdges())
        Q.push(B);
    }
  }
}

namespace {
  bool NotJoinNode(const Node* N) {
    return N->ToolPtr ? !N->ToolPtr->IsJoin() : true;
  }
}

// Call TopologicalSort and filter the resulting list to include
// only Join nodes.
void CompilationGraph::
TopologicalSortFilterJoinNodes(std::vector<const Node*>& Out) {
  std::vector<const Node*> TopSorted;
  TopologicalSort(TopSorted);
  std::remove_copy_if(TopSorted.begin(), TopSorted.end(),
                      std::back_inserter(Out), NotJoinNode);
}

// Find head of the toolchain corresponding to the given file.
const Node* CompilationGraph::
FindToolChain(const sys::Path& In, const std::string* forceLanguage) const {
  const std::string& InLanguage =
    forceLanguage ? *forceLanguage : getLanguage(In);
  const tools_vector_type& TV = getToolsVector(InLanguage);
  if (TV.empty())
    throw std::runtime_error("No toolchain corresponding to language"
                             + InLanguage + " found!");
  return &getNode(ChooseEdge(TV)->ToolName());
}

// Build the targets. Command-line options are passed through
// temporary variables.
int CompilationGraph::Build (const sys::Path& TempDir) {

  // This is related to -x option handling.
  cl::list<std::string>::const_iterator xIter = Languages.begin(),
    xBegin = xIter, xEnd = Languages.end();
  bool xEmpty = true;
  const std::string* xLanguage = 0;
  unsigned xPos = 0, xPosNext = 0, filePos = 0;

  if (xIter != xEnd) {
    xEmpty = false;
    xPos = Languages.getPosition(xIter - xBegin);
    cl::list<std::string>::const_iterator xNext = llvm::next(xIter);
    xPosNext = (xNext == xEnd) ? std::numeric_limits<unsigned>::max()
      : Languages.getPosition(xNext - xBegin);
    xLanguage = (*xIter == "none") ? 0 : &(*xIter);
  }

  // For each input file:
  for (cl::list<std::string>::const_iterator B = InputFilenames.begin(),
         CB = B, E = InputFilenames.end(); B != E; ++B) {
    sys::Path In = sys::Path(*B);

    // Code for handling the -x option.
    // Output: std::string* xLanguage (can be NULL).
    if (!xEmpty) {
      filePos = InputFilenames.getPosition(B - CB);

      if (xPos < filePos) {
        if (filePos < xPosNext) {
          xLanguage = (*xIter == "none") ? 0 : &(*xIter);
        }
        else { // filePos >= xPosNext
          // Skip xIters while filePos > xPosNext
          while (filePos > xPosNext) {
            ++xIter;
            xPos = xPosNext;

            cl::list<std::string>::const_iterator xNext = llvm::next(xIter);
            if (xNext == xEnd)
              xPosNext = std::numeric_limits<unsigned>::max();
            else
              xPosNext = Languages.getPosition(xNext - xBegin);
            xLanguage = (*xIter == "none") ? 0 : &(*xIter);
          }
        }
      }
    }

    // Find the toolchain corresponding to this file.
    const Node* N = FindToolChain(In, xLanguage);
    // Pass file through the chain starting at head.
    PassThroughGraph(In, N, TempDir);
  }

  std::vector<const Node*> JTV;
  TopologicalSortFilterJoinNodes(JTV);

  // For all join nodes in topological order:
  for (std::vector<const Node*>::iterator B = JTV.begin(), E = JTV.end();
       B != E; ++B) {

    sys::Path Out;
    const Node* CurNode = *B;
    JoinTool* JT = &dynamic_cast<JoinTool&>(*CurNode->ToolPtr.getPtr());
    bool IsLast = false;

    // Are there any files to be joined?
    if (JT->JoinListEmpty())
      continue;

    // Is this the last tool in the chain?
    // NOTE: we can process several chains in parallel.
    if (!CurNode->HasChildren() || JT->IsLast()) {
      if (OutputFilename.empty()) {
        Out.set("a");
        Out.appendSuffix(JT->OutputSuffix());
      }
      else
        Out.set(OutputFilename);
      IsLast = true;
    }
    else {
      Out = MakeTempFile(TempDir, "tmp", JT->OutputSuffix());
    }

    if (JT->GenerateAction(Out).Execute() != 0)
      throw std::runtime_error("Tool returned error code!");

    if (!IsLast) {
      const Node* NextNode = &getNode(ChooseEdge(CurNode->OutEdges,
                                                 CurNode->Name())->ToolName());
      PassThroughGraph(Out, NextNode, TempDir);
    }
  }

  return 0;
}

// Code related to graph visualization.

namespace llvm {
  template <>
  struct DOTGraphTraits<llvmc::CompilationGraph*>
    : public DefaultDOTGraphTraits
  {

    template<typename GraphType>
    static std::string getNodeLabel(const Node* N, const GraphType&)
    {
      if (N->ToolPtr)
        if (N->ToolPtr->IsJoin())
          return N->Name() + "\n (join" +
            (N->HasChildren() ? ")"
             : std::string(": ") + N->ToolPtr->OutputLanguage() + ')');
        else
          return N->Name();
      else
        return "root";
    }

    template<typename EdgeIter>
    static std::string getEdgeSourceLabel(const Node* N, EdgeIter I) {
      if (N->ToolPtr)
        return N->ToolPtr->OutputLanguage();
      else
        return I->ToolPtr->InputLanguage();
    }
  };

}

void CompilationGraph::writeGraph() {
  std::ofstream O("CompilationGraph.dot");

  if (O.good()) {
    llvm::WriteGraph(this, "compilation-graph");
    O.close();
  }
  else {
    throw std::runtime_error("");
  }
}

void CompilationGraph::viewGraph() {
  llvm::ViewGraph(this, "compilation-graph");
}

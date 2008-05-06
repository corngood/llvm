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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include <stdexcept>

using namespace llvm;
using namespace llvmcc;

extern cl::list<std::string> InputFilenames;
extern cl::opt<std::string> OutputFilename;

// Choose one of the edges based on command-line options.
const Edge* Node::ChooseEdge() const {
  const Edge* DefaultEdge = 0;
  for (const_iterator B = EdgesBegin(), E = EdgesEnd();
       B != E; ++B) {
    const Edge* E = (*B).getPtr();
    if (E->isDefault())
      if (!DefaultEdge)
        DefaultEdge = E;
      else
        throw std::runtime_error("Node " + Name() +
                                 ": multiple default edges found!"
                                 "Most probably a specification error.");
    if (E->isEnabled())
      return E;
  }

  if (DefaultEdge)
    return DefaultEdge;
  else
    throw std::runtime_error("Node " + Name() +
                             ": no suitable edge found! "
                             "Most probably a specification error.");
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
    throw std::runtime_error("No tools corresponding to " + LangName
                             + " found!");
  return I->second;
}

void CompilationGraph::insertNode(Tool* V) {
  if (!NodesMap.count(V->Name())) {
    Node N;
    N.OwningGraph = this;
    N.ToolPtr = V;
    NodesMap[V->Name()] = N;
  }
}

void CompilationGraph::insertEdge(const std::string& A, Edge* E) {
  if (A == "root") {
    const Node& N = getNode(E->ToolName());
    const std::string& InputLanguage = N.ToolPtr->InputLanguage();
    ToolsMap[InputLanguage].push_back(E->ToolName());

    // Needed to support iteration via GraphTraits.
    NodesMap["root"].AddEdge(E);
  }
  else {
    Node& N = getNode(A);
    N.AddEdge(E);
  }
}

// Pass input file through the chain until we bump into a Join node or
// a node that says that it is the last.
const Tool* CompilationGraph::PassThroughGraph (sys::Path& In,
                                                sys::Path Out,
                                                const sys::Path& TempDir,
                                                PathVector& JoinList) const {
  bool Last = false;
  const Tool* ret = 0;

  // Get to the head of the toolchain.
  const tools_vector_type& TV = getToolsVector(getLanguage(In));
  if (TV.empty())
    throw std::runtime_error("Tool names vector is empty!");
  const Node* N = &getNode(*TV.begin());

  while(!Last) {
    const Tool* CurTool = N->ToolPtr.getPtr();

    if (CurTool->IsJoin()) {
      JoinList.push_back(In);
      ret = CurTool;
      break;
    }

    // Is this the last tool?
    if (!N->HasChildren() || CurTool->IsLast()) {
      // Check if the first tool is also the last
      if (Out.empty())
        Out.set(In.getBasename());
      else
        Out.appendComponent(In.getBasename());
      Out.appendSuffix(CurTool->OutputSuffix());
      Last = true;
    }
    else {
      Out = TempDir;
      Out.appendComponent(In.getBasename());
      Out.appendSuffix(CurTool->OutputSuffix());
      Out.makeUnique(true, NULL);
      Out.eraseFromDisk();
    }

    if (CurTool->GenerateAction(In, Out).Execute() != 0)
      throw std::runtime_error("Tool returned error code!");

    N = &getNode(N->ChooseEdge()->ToolName());
    In = Out; Out.clear();
  }

  return ret;
}

// TOFIX: support more interesting graph topologies. We will need to
// do topological sorting to process multiple Join nodes correctly.
int CompilationGraph::Build (const sys::Path& TempDir) const {
  PathVector JoinList;
  const Tool* JoinTool = 0;
  sys::Path In, Out;

  // For each input file
  for (cl::list<std::string>::const_iterator B = InputFilenames.begin(),
        E = InputFilenames.end(); B != E; ++B) {
    In = sys::Path(*B);

    const Tool* NewJoin = PassThroughGraph(In, Out, TempDir, JoinList);
    if (JoinTool && NewJoin && JoinTool != NewJoin)
      throw std::runtime_error("Graphs with multiple Join nodes"
                               "are not yet supported!");
    else if (NewJoin)
      JoinTool = NewJoin;
  }

  if (JoinTool) {
    // If the final output name is empty, set it to "a.out"
    if (!OutputFilename.empty()) {
      Out = sys::Path(OutputFilename);
    }
    else {
      Out = sys::Path("a");
      Out.appendSuffix(JoinTool->OutputSuffix());
    }

    if (JoinTool->GenerateAction(JoinList, Out).Execute() != 0)
      throw std::runtime_error("Tool returned error code!");
  }

  return 0;
}

namespace llvm {
  template <>
  struct DOTGraphTraits<llvmcc::CompilationGraph*>
    : public DefaultDOTGraphTraits
  {

  template<typename GraphType>
  static std::string getNodeLabel(const Node* N, const GraphType&) {
    if (N->ToolPtr)
      return N->Name();
    else
      return "root";
  }

  };
}

void CompilationGraph::writeGraph() {
  std::ofstream O("CompilationGraph.dot");

  if (O.good()) {
    llvm::WriteGraph(this, "CompilationGraph");
    O.close();
  }
  else {
    throw std::runtime_error("");
  }
}

void CompilationGraph::viewGraph() {
  llvm::ViewGraph(this, "compilation-graph");
}

//===-- EmitFunctions.cpp - interface to insert instrumentation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This inserts into the input module three new global constants containing
// mapping information pertinent to the Reoptimizer's runtime library:
// 1) a structure containing a pointer to each function;
// 2) an array containing a boolean which is true iff the corresponding
//    function in 1) contains a back-edge branch suitable for the Reoptimizer's
//    first-level instrumentation;
// 3) an integer containing the number of entries in 1) and 2).
//
// NOTE: This pass is used by the reoptimizer only.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Instrumentation.h"
using namespace llvm;

namespace llvm {

namespace {
  enum Color{
    WHITE,
    GREY,
    BLACK
  };

  struct VISIBILITY_HIDDEN EmitFunctionTable : public ModulePass {
    bool runOnModule(Module &M);
  };

  RegisterPass<EmitFunctionTable>
  X("emitfuncs", "Emit a function table for the reoptimizer");
}

static char doDFS(BasicBlock * node,std::map<BasicBlock *, Color > &color){
  color[node] = GREY;

  for(succ_iterator vl = succ_begin(node), ve = succ_end(node); vl != ve; ++vl){

    BasicBlock *BB = *vl;

    if(color[BB]!=GREY && color[BB]!=BLACK){
      if(!doDFS(BB, color)){
        return 0;
      }
    }

    //if has backedge
    else if(color[BB]==GREY)
      return 0;

  }

  color[node] = BLACK;
  return 1;
}

static char hasBackEdge(Function *F){
  std::map<BasicBlock *, Color > color;
  return doDFS(F->begin(), color);
}

// Per Module pass for inserting function table
bool EmitFunctionTable::runOnModule(Module &M){
  std::vector<const Type*> vType;

  std::vector<Constant *> vConsts;
  std::vector<Constant *> sBCons;

  unsigned int counter = 0;
  for(Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
    if (!MI->isDeclaration()) {
      vType.push_back(MI->getType());

      //std::cerr<<MI;

      vConsts.push_back(MI);
      sBCons.push_back(ConstantInt::get(Type::Int8Ty, hasBackEdge(MI)));

      counter++;
    }

  StructType *sttype = StructType::get(vType);
  Constant *cstruct = ConstantStruct::get(sttype, vConsts);

  GlobalVariable *gb = new GlobalVariable(cstruct->getType(), true,
                                          GlobalValue::ExternalLinkage,
                                          cstruct, "llvmFunctionTable");
  M.getGlobalList().push_back(gb);

  Constant *constArray = ConstantArray::get(ArrayType::get(Type::Int8Ty,
                                                                sBCons.size()),
                                                 sBCons);

  GlobalVariable *funcArray = new GlobalVariable(constArray->getType(), true,
                                              GlobalValue::ExternalLinkage,
                                              constArray, "llvmSimpleFunction");

  M.getGlobalList().push_back(funcArray);

  ConstantInt *cnst = ConstantInt::get(Type::Int32Ty, counter);
  GlobalVariable *fnCount = new GlobalVariable(Type::Int32Ty, true,
                                               GlobalValue::ExternalLinkage,
                                               cnst, "llvmFunctionCount");
  M.getGlobalList().push_back(fnCount);
  return true;  // Always modifies program
}

ModulePass *createEmitFunctionTablePass () {
  return new EmitFunctionTable();
}

} // end namespace llvm

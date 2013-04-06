   //===-- Vectorize.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMVectorizeOpts.a, which
// implements several vectorization transformations over the LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/AMPToOpenCL.h"

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "../../Target/NVPTX/MCTargetDesc/NVPTXBaseInfo.h"
#include <set>

using namespace llvm;

namespace {
  void filterAMPFunctions(Module &M, bool const amp) {
    std::set<Function*> AMPFunctions;

    NamedMDNode * const Metadata = M.getNamedMetadata("amp.restrict");
    if(!Metadata) return;

    unsigned const Count = Metadata->getNumOperands();
    for(unsigned i = 0; i < Count; ++i) {
      MDNode * const Node = Metadata->getOperand(i);
      assert(Node->getNumOperands() == 1);
      if(!Node->getOperand(0)) continue;

      Function * const F = dyn_cast<Function>(Node->getOperand(0));
      if(!F) continue;

      AMPFunctions.insert(F);
    }

    if(amp) {
      Module::FunctionListType &FunctionList = M.getFunctionList();

      std::set<Function*> CPUFunctions;

      for(Module::FunctionListType::iterator FI = FunctionList.begin(), FE = FunctionList.end(); FI != FE; ++FI) {
        if(AMPFunctions.find(&*FI) == AMPFunctions.end()) {
          CPUFunctions.insert(&*FI);
        }
      }

      for(std::set<Function*>::const_iterator FI = CPUFunctions.begin(), FE = CPUFunctions.end(); FI != FE; ++FI) {
        if(!(*FI)->isDeclaration())
          (*FI)->setLinkage(GlobalValue::InternalLinkage);
      }

      Module::GlobalListType &GlobalList = M.getGlobalList();

      for(Module::GlobalListType::iterator GI = GlobalList.begin(), GE = GlobalList.end(); GI != GE; ++GI) {
        if(!(*GI).isDeclaration())
          (*GI).setLinkage(GlobalValue::InternalLinkage);
      }
    } else {
      for(std::set<Function*>::const_iterator FI = AMPFunctions.begin(), FE = AMPFunctions.end(); FI != FE; ++FI) {
        (*FI)->setLinkage(GlobalValue::InternalLinkage);
      }
    }

    Metadata->eraseFromParent();
  }

#if 1
  Constant *getFunctionConstant(Function * const TargetFunction) {
    if(!TargetFunction) return NULL;

    Module &M = *TargetFunction->getParent();

    GlobalVariable * const CodeVariable = new GlobalVariable(M,
      PointerType::get(IntegerType::get(M.getContext(), 8), 0),
      true, GlobalValue::ExternalLinkage, 0, TargetFunction->getName().str() + "_CODE");

    return CodeVariable;
  }
#else
  Constant *getFunctionConstant(Function * const TargetFunction) {
    if(!TargetFunction) return NULL;
    if(TargetFunction->arg_size() != 2) return NULL;

    Module &M = *TargetFunction->getParent();

    Function::arg_iterator AI = TargetFunction->arg_begin();
    Argument & DataArgument = *AI++;

    Type * const DataPtrType = DataArgument.getType();
    if(!DataPtrType->isPointerTy()) return NULL;

    Type * const DataType = DataPtrType->getElementType();
    if(!DataType->isStructTy()) return NULL;

    std::string codeString;
    raw_string_ostream codeStream(codeString);

    codeStream << "void " << TargetFunction->getName().str() <<  "(";

    for(unsigned i = 0; i < DataType->getNumContainedTypes(); ++i) {
      if(i > 0) codeStream << ", ";
      Type * const ParameterType = DataType->getContainedType(i);
      ParameterType->print(codeStream);
    }

    codeStream << ") {\n";

    Function::BasicBlockListType &BlockList = TargetFunction->getBasicBlockList();

    for(Function::BasicBlockListType::const_iterator BI = BlockList.begin(), BE = BlockList.end(); BI != BE; ++BI) {
      BasicBlock const &Block = *BI;

      codeStream << Block.getName() << ":\n";

      BasicBlock::InstListType const &InstList = Block.getInstList();

      for(BasicBlock::InstListType::const_iterator II = InstList.begin(), IE = InstList.end(); II != IE; ++II) {
        Instruction const &I = *II;

        I.print(codeStream);
        codeStream << "\n";
      }
    }

    codeStream << "}\n";

    Constant * const CodeConstant = ConstantDataArray::getString(M.getContext(), codeStream.str());
    GlobalVariable * const CodeVariable = new GlobalVariable(M, CodeConstant->getType(), true, GlobalValue::PrivateLinkage, CodeConstant);
    CodeVariable->setUnnamedAddr(true);

    IntegerType *IdxTy = IntegerType::get(M.getContext(), 32);
    Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
    Constant * const IdxList[] = {IdxZero, IdxZero};

    return ConstantExpr::getInBoundsGetElementPtr(CodeVariable, IdxList);
  }
#endif

  class AMPToOpenCL : public ModulePass {

    /*template<typename ListType> void findKernelArguments(Type * const T, ListType &ArgumentTypeList) {
      if(StructType * const ST = dyn_cast<StructType>(T)) {
        for(unsigned i = 0; i < ST->getNumContainedTypes(); ++i) {
          findKernelArguments(ST->getContainedType(i), ArgumentTypeList);
        }
      } else if(ArrayType * const AT = dyn_cast<ArrayType>(T)) {
        for(unsigned i = 0; i < AT->getNumElements(); ++i) {
          findKernelArguments(AT->getElementType(), ArgumentTypeList);
        }
      } else if(PointerType * const PT = dyn_cast<PointerType>(T)) {
        ArgumentTypeList.push_back(PointerType::get(cast<PointerType>(T)->getElementType(), ADDRESS_SPACE_GENERIC));
      } else if(T->isSingleValueType()) {
        ArgumentTypeList.push_back(T);
      } else {
        assert(false && "unsupported type in AMP kernel data");
      }
    }*/

    template<typename ListType> static void findKernelArguments(Type * const T, ListType &ArgumentTypeList) {
      if(StructType * const ST = dyn_cast<StructType>(T)) {
        for(unsigned i = 0; i < ST->getNumContainedTypes(); ++i) {
          findKernelArguments(ST->getContainedType(i), ArgumentTypeList);
        }
      } else if(ArrayType * const AT = dyn_cast<ArrayType>(T)) {
        for(unsigned i = 0; i < AT->getNumElements(); ++i) {
          findKernelArguments(AT->getElementType(), ArgumentTypeList);
        }
      } else if(PointerType * const PT = dyn_cast<PointerType>(T)) {
		ArgumentTypeList.push_back(PT);
		//ArgumentTypeList.push_back(PointerType::get(cast<PointerType>(T)->getElementType(), ADDRESS_SPACE_GLOBAL));
      } else if(T->isSingleValueType()) {
        ArgumentTypeList.push_back(T);
      } else {
        assert(false && "unsupported type in AMP kernel data");
      }
    }

    static void loadKernelArguments(Value * const P, Function::ArgumentListType::iterator &AI, BasicBlock * const BB) {
      Type * const T = cast<PointerType>(P->getType())->getElementType();

      if(StructType * const ST = dyn_cast<StructType>(T)) {
        IntegerType *IdxTy = IntegerType::get(P->getContext(), 32);
        Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
        Value * IdxList[] = {IdxZero, IdxZero};

        for(unsigned i = 0; i < ST->getNumContainedTypes(); ++i) {
          IdxList[1] = ConstantInt::get(IdxTy, i, false);
          loadKernelArguments(GetElementPtrInst::Create(P, IdxList, "", BB), AI, BB);
        }
      } else if(ArrayType * const AT = dyn_cast<ArrayType>(T)) {
        IntegerType *IdxTy = IntegerType::get(P->getContext(), 32);
        Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
        Value * IdxList[] = {IdxZero, IdxZero};

        for(unsigned i = 0; i < AT->getNumElements(); ++i) {
          IdxList[1] = ConstantInt::get(IdxTy, i, false);
          loadKernelArguments(GetElementPtrInst::Create(P, IdxList, "", BB), AI, BB);
        }
      } else if(T->isPointerTy() || T->isSingleValueType()) {
        new StoreInst(&*AI++, P, BB);
      } else {
        assert(false && "unsupported type in AMP kernel data");
      }
    }

    /*template<typename ListType> void findIndices(Type * const T, ListType &IndexList) {
      if(T->isSingleValueType()) {
        IndexList.push_back(T);
      } else if(StructType * const ST = dyn_cast<StructType>(T)) {
        for(unsigned i = 0; i < ST->getNumContainedTypes(); ++i) {
          findIndices(ST->getContainedType(i), IndexList);
        }
      } else if(ArrayType * const AT = dyn_cast<ArrayType>(T)) {
        for(unsigned i = 0; i < AT->getNumElements(); ++i) {
          findIndices(AT->getElementType(), IndexList);
        }
      } else {
        assert(false && "unsupported type in AMP kernel indices");
      }
    }*/

    static void loadIndices(Value * const P, BasicBlock * const BB, Module * const M) {
      unsigned index = 0;
      loadIndices(P, BB, M, index);
    }

    static CallInst *intrinsic(Intrinsic::ID ID, BasicBlock * const BB, Module * const M) {
      return CallInst::Create(Intrinsic::getDeclaration(M, ID), "", BB);
    }

    static void loadIndices(Value * const P, BasicBlock * const BB, Module * const M, unsigned &index) {
      Type * const T = cast<PointerType>(P->getType())->getElementType();

      if(T->isSingleValueType()) {
        Intrinsic::ID Itid, Intid, Ictaid;

        switch(index++) {
          default: assert(false && "too many indices in AMP kernel");
          case 0: Itid = Intrinsic::ptx_read_tid_x; Intid = Intrinsic::ptx_read_ntid_x; Ictaid = Intrinsic::ptx_read_ctaid_x; break;
          case 1: Itid = Intrinsic::ptx_read_tid_y; Intid = Intrinsic::ptx_read_ntid_y; Ictaid = Intrinsic::ptx_read_ctaid_y; break;
          case 2: Itid = Intrinsic::ptx_read_tid_z; Intid = Intrinsic::ptx_read_ntid_z; Ictaid = Intrinsic::ptx_read_ctaid_z; break;
          case 3: Itid = Intrinsic::ptx_read_tid_w; Intid = Intrinsic::ptx_read_ntid_w; Ictaid = Intrinsic::ptx_read_ctaid_w; break;
        }

        new StoreInst(BinaryOperator::Create(Instruction::Add,
          BinaryOperator::Create(Instruction::Mul, intrinsic(Ictaid, BB, M), intrinsic(Intid, BB, M), "", BB),
          intrinsic(Itid, BB, M), "", BB), P, BB);
      } else if(StructType * const ST = dyn_cast<StructType>(T)) {
        IntegerType *IdxTy = IntegerType::get(P->getContext(), 32);
        Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
        Value * IdxList[] = {IdxZero, IdxZero};

        for(unsigned i = 0; i < ST->getNumContainedTypes(); ++i) {
          IdxList[1] = ConstantInt::get(IdxTy, i, false);
          loadIndices(GetElementPtrInst::Create(P, IdxList, "", BB), BB, M, index);
        }
      } else if(ArrayType * const AT = dyn_cast<ArrayType>(T)) {
        IntegerType *IdxTy = IntegerType::get(P->getContext(), 32);
        Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
        Value * IdxList[] = {IdxZero, IdxZero};

        for(unsigned i = 0; i < AT->getNumElements(); ++i) {
          IdxList[1] = ConstantInt::get(IdxTy, i, false);
          loadIndices(GetElementPtrInst::Create(P, IdxList, "", BB), BB, M, index);
        }
      } else {
        assert(false && "unsupported type in AMP kernel indices");
      }
    }

  public:
    static char ID;
    AMPToOpenCL() : ModulePass(ID) {}

    bool runOnModule(Module &M) {
      LLVMContext &C = M.getContext();

      bool modified = false;

#if 1
      NamedMDNode * const Metadata = M.getNamedMetadata("amp.kernel");
      if(!Metadata) return modified;

      unsigned const Count = Metadata->getNumOperands();
      for(unsigned i = 0; i < Count; ++i) {
        MDNode * const Node = Metadata->getOperand(i);
        assert(Node->getNumOperands() == 1);

        if(Node->getNumOperands() < 1) continue;

        Function * const TargetFunction = llvm::dyn_cast<Function>(Node->getOperand(0));
        TargetFunction->setLinkage(GlobalValue::InternalLinkage);

        PointerType * const KernelPointerType = dyn_cast<PointerType>(TargetFunction->getFunctionType()->getParamType(0));
        if(!KernelPointerType) continue;

        PointerType * const IndexPointerType = dyn_cast<PointerType>(TargetFunction->getFunctionType()->getParamType(1));
        if(!IndexPointerType) continue;

        Type * const KernelType = KernelPointerType->getElementType();
        Type * const IndexType = IndexPointerType->getElementType();

        SmallVector<Type*, 8> Params;
        findKernelArguments(KernelType, Params);

        Function * const KernelFunction = Function::Create(FunctionType::get(Type::getVoidTy(C), Params, false), GlobalValue::ExternalLinkage, TargetFunction->getName().str() + "_Kernel", &M);

        BasicBlock * const BB = BasicBlock::Create(C, "entry", KernelFunction);

        AllocaInst * const DataObject = new AllocaInst(KernelType, "data", BB);
        Function::ArgumentListType::iterator AI = KernelFunction->arg_begin();
        loadKernelArguments(DataObject, AI, BB);
        assert(AI == KernelFunction->arg_end());

        AllocaInst * const IndexObject = new AllocaInst(IndexType, "index", BB);
        loadIndices(IndexObject, BB, &M);

        SmallVector<Value*, 2> Args;
        Args.push_back(DataObject);
        Args.push_back(IndexObject);
        CallInst::Create(TargetFunction, Args, "", BB);
        ReturnInst::Create(C, BB);

        NamedMDNode * const NA = M.getOrInsertNamedMetadata("nvvm.annotations");

        SmallVector<Value*, 3> MD;
        MD.push_back(KernelFunction);
        MD.push_back(MDString::get(C, "kernel"));
        MD.push_back(ConstantInt::get(IntegerType::get(C, 32), 1));

        NA->addOperand(MDNode::get(C, MD));

        modified = true;
      }
#else // old method using kernel_info
      if(M.getNamedMetadata("amp.restrict")) {
        filterAMPFunctions(M, true);
        modified = true;
      }

      NamedMDNode * const Metadata = M.getNamedMetadata("amp.kernel_info");
      if(!Metadata) return modified;

      unsigned const Count = Metadata->getNumOperands();
      for(unsigned i = 0; i < Count; ++i) {
        MDNode * const Node = Metadata->getOperand(i);
        assert(Node->getNumOperands() == 1);

        if(Node->getNumOperands() < 1) continue;

        GlobalVariable * const OriginalVariable = llvm::dyn_cast<GlobalVariable>(Node->getOperand(0));
        if(!OriginalVariable) continue;
        if(!OriginalVariable->hasUniqueInitializer()) continue;

        Constant * const OriginalInitializer = OriginalVariable->getInitializer();
        if(!OriginalInitializer) continue;
        if(OriginalInitializer->getNumOperands() != 2) continue;

        Function * const TargetFunction = dyn_cast<Function>(OriginalInitializer->getOperand(0));
        //TargetFunction->setLinkage(GlobalValue::ExternalLinkage);

        PointerType * const KernelPointerType = dyn_cast<PointerType>(TargetFunction->getFunctionType()->getParamType(0));
        if(!KernelPointerType) continue;

        PointerType * const IndexPointerType = dyn_cast<PointerType>(TargetFunction->getFunctionType()->getParamType(1));
        if(!IndexPointerType) continue;

        Type * const KernelType = KernelPointerType->getElementType();
        Type * const IndexType = IndexPointerType->getElementType();

        SmallVector<Type*, 8> Params;
        findKernelArguments(KernelType, Params);

        Function * const KernelFunction = Function::Create(FunctionType::get(Type::getVoidTy(C), Params, false), GlobalValue::ExternalLinkage, TargetFunction->getName().str() + "_Kernel", &M);

        BasicBlock * const BB = BasicBlock::Create(C, "entry", KernelFunction);

        AllocaInst * const DataObject = new AllocaInst(KernelType, "data", BB);
        Function::ArgumentListType::iterator AI = KernelFunction->arg_begin();
        loadKernelArguments(DataObject, AI, BB);
        assert(AI == KernelFunction->arg_end());

        AllocaInst * const IndexObject = new AllocaInst(IndexType, "index", BB);
        loadIndices(IndexObject, BB, &M);

        SmallVector<Value*, 2> Args;
        Args.push_back(DataObject);
        Args.push_back(IndexObject);
        CallInst::Create(TargetFunction, Args, "", BB);
        ReturnInst::Create(C, BB);

        NamedMDNode * const NA = M.getOrInsertNamedMetadata("nvvm.annotations");

        SmallVector<Value*, 3> MD;
        MD.push_back(KernelFunction);
        MD.push_back(MDString::get(C, "kernel"));
        MD.push_back(ConstantInt::get(IntegerType::get(C, 32), 1));

        NA->addOperand(MDNode::get(C, MD));

        modified = true;
#if 0
        Constant * const InfoConstant = getFunctionConstant(TargetFunction);
        if(!InfoConstant) continue;

        Constant * const KernelInfoConstant = ConstantStruct::get(
          cast<StructType>(OriginalInitializer->getType()),
          TargetFunction,
          InfoConstant,
          NULL);
        GlobalVariable * const KernelInfoVariable = new GlobalVariable(M, KernelInfoConstant->getType(), true, GlobalValue::PrivateLinkage, KernelInfoConstant);

        OriginalVariable->replaceAllUsesWith(KernelInfoVariable);

        modified = true;
#endif
      }
#endif

      Metadata->eraseFromParent();

      // Dead code elimination needed to ensure consistency for AMP module
      modified = createGlobalDCEPass()->runOnModule(M) || modified;

      return modified;
    }
  };

  class StripAMP : public ModulePass {

  public:
    static char ID;
    StripAMP() : ModulePass(ID) {}

    bool runOnModule(Module &M) {
      bool modified = false;

      if(M.getNamedMetadata("amp.restrict")) {
        filterAMPFunctions(M, false);
        modified = true;
      }

      NamedMDNode * const Metadata = M.getNamedMetadata("amp.kernel_info");
      if(!Metadata) return modified;

      unsigned const Count = Metadata->getNumOperands();
      for(unsigned i = 0; i < Count; ++i) {
        MDNode * const Node = Metadata->getOperand(i);
        assert(Node->getNumOperands() == 1);

        if(Node->getNumOperands() < 1) continue;

        GlobalVariable * const OriginalVariable = llvm::dyn_cast<GlobalVariable>(Node->getOperand(0));
        if(!OriginalVariable) continue;
        if(!OriginalVariable->hasUniqueInitializer()) continue;

        Constant * const OriginalInitializer = OriginalVariable->getInitializer();
        if(!OriginalInitializer) continue;
        if(OriginalInitializer->getNumOperands() != 2) continue;

        Function * const TargetFunction = dyn_cast<Function>(OriginalInitializer->getOperand(0));
        if(!TargetFunction) continue;

        Constant * const InfoConstant = getFunctionConstant(TargetFunction);
        if(!InfoConstant) continue;

        Constant * const KernelInfoConstant = ConstantStruct::get(
          cast<StructType>(OriginalInitializer->getType()),
          TargetFunction,
          //ConstantPointerNull::get(TargetFunction->getType()),
          InfoConstant,
          NULL);
        GlobalVariable * const KernelInfoVariable = new GlobalVariable(M, KernelInfoConstant->getType(), true, GlobalValue::PrivateLinkage, KernelInfoConstant);

        OriginalVariable->replaceAllUsesWith(KernelInfoVariable);

        modified = true;
      }

      Metadata->eraseFromParent();

      return modified;
    }
  };
}

namespace llvm {
  void initializeAMPToOpenCLPass(PassRegistry&);
  void initializeStripAMPPass(PassRegistry&);

  void initializeAMP(PassRegistry& R) {
    initializeAMPToOpenCLPass(R);
    initializeStripAMPPass(R);
  }
}

char AMPToOpenCL::ID = 0;
char StripAMP::ID = 0;
INITIALIZE_PASS(AMPToOpenCL, "amp-to-opencl", "Generate OpenCL kernels for AMP", false, false)
INITIALIZE_PASS(StripAMP, "strip-amp", "Strip AMP functions", false, false)

Pass *llvm::createAMPToOpenCLPass() {
  return new AMPToOpenCL();
}

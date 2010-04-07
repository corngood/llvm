//===-- ARMGlobalMerge.cpp - Internal globals merging  --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-global-merge"
#include "ARM.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Attributes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN ARMGlobalMerge : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// target type sizes.
    const TargetLowering *TLI;

    std::vector<GlobalVariable*> InternalGlobals;

  public:
    static char ID;             // Pass identification, replacement for typeid.
    explicit ARMGlobalMerge(const TargetLowering *tli)
      : FunctionPass(&ID), TLI(tli) {}

    virtual bool doInitialization(Module &M);
    virtual bool runOnFunction(Function& F);

    const char *getPassName() const {
      return "Merge internal globals";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      FunctionPass::getAnalysisUsage(AU);
    }

    struct GlobalCmp {
      const TargetData *TD;

      GlobalCmp(const TargetData *td):
        TD(td) { };

      bool operator() (const GlobalVariable* GV1,
                       const GlobalVariable* GV2) {
        const Type* Ty1 = cast<PointerType>(GV1->getType())->getElementType();
        const Type* Ty2 = cast<PointerType>(GV2->getType())->getElementType();

        return (TD->getTypeAllocSize(Ty1) <
                TD->getTypeAllocSize(Ty2));
      }
    };
  };
} // end anonymous namespace

char ARMGlobalMerge::ID = 0;

#define MAX_OFFSET 4095

bool ARMGlobalMerge::doInitialization(Module& M) {
  const TargetData *TD = TLI->getTargetData();

  for (Module::global_iterator I = M.global_begin(),
         E = M.global_end(); I != E; ++I) {
    // FIXME: Can we just grab all 'local' vars here?
    // Won't we break some semantics?
    if (I->hasInternalLinkage() &&
        TD->getTypeAllocSize(I->getType()) < MAX_OFFSET)
      InternalGlobals.push_back(I);
  }

  // FIXME: Find better heuristics
  std::stable_sort(InternalGlobals.begin(), InternalGlobals.end(),
                   GlobalCmp(TD));

  const Type *Int32Ty = Type::getInt32Ty(M.getContext());

  for (size_t i = 0, e = InternalGlobals.size(); i != e; ) {
    size_t j = 0;
    uint64_t MergedSize = 0;
    std::vector<const Type*> Tys;
    std::vector<Constant*> Inits;
    for (j = i; MergedSize < MAX_OFFSET && j != e; ++j) {
      const Type* Ty =
        cast<PointerType>(InternalGlobals[j]->getType())->getElementType();
      Tys.push_back(Ty);
      Inits.push_back(InternalGlobals[j]->getInitializer());
      MergedSize += TD->getTypeAllocSize(Ty);
    }

    StructType* MergedTy = StructType::get(M.getContext(), Tys);
    Constant* MergedInit = ConstantStruct::get(MergedTy, Inits);
    // FIXME: Should we handle constants and 'normal' globals separately?
    GlobalVariable* MergedGV = new GlobalVariable(M, MergedTy, false,
                                                  GlobalValue::InternalLinkage,
                                                  MergedInit, "merged");
    for (size_t k = i; k < j; ++k) {
      SmallVector<Constant*, 2> Idx;
      Idx.push_back(ConstantInt::get(Int32Ty, 0));
      Idx.push_back(ConstantInt::get(Int32Ty, k-i));

      Constant* GEP =
        ConstantExpr::getInBoundsGetElementPtr(MergedGV,
                                               &Idx[0], Idx.size());

      InternalGlobals[k]->replaceAllUsesWith(GEP);
    }
    i = j;
  }

  return true;
}

bool ARMGlobalMerge::runOnFunction(Function& F) {
  return false;
}

FunctionPass *llvm::createARMGlobalMergePass(const TargetLowering *tli) {
  return new ARMGlobalMerge(tli);
}

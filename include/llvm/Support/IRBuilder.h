//===---- llvm/Support/IRBuilder.h - Builder for LLVM Instrs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IRBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_IRBUILDER_H
#define LLVM_SUPPORT_IRBUILDER_H

#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"

namespace llvm {

/// IRBuilder - This provides a uniform API for creating instructions and
/// inserting them into a basic block: either at the end of a BasicBlock, or 
/// at a specific iterator location in a block.
///
/// Note that the builder does not expose the full generality of LLVM
/// instructions.  For example, it cannot be used to create instructions with
/// arbitrary names (specifically, names with nul characters in them) - It only
/// supports nul-terminated C strings.  For fully generic names, use
/// I->setName().  For access to extra instruction properties, use the mutators
/// (e.g. setVolatile) on the instructions after they have been created.
class IRBuilder {
  BasicBlock *BB;
  BasicBlock::iterator InsertPt;
public:
  IRBuilder() { ClearInsertionPoint(); }
  explicit IRBuilder(BasicBlock *TheBB) { SetInsertPoint(TheBB); }
  IRBuilder(BasicBlock *TheBB, BasicBlock::iterator IP) {
    SetInsertPoint(TheBB, IP);
  }

  //===--------------------------------------------------------------------===//
  // Builder configuration methods
  //===--------------------------------------------------------------------===//

  /// ClearInsertionPoint - Clear the insertion point: created instructions will
  /// not be inserted into a block.
  void ClearInsertionPoint() {
    BB = 0;
  }
  
  BasicBlock *GetInsertBlock() const { return BB; }
  
  /// SetInsertPoint - This specifies that created instructions should be
  /// appended to the end of the specified block.
  void SetInsertPoint(BasicBlock *TheBB) {
    BB = TheBB;
    InsertPt = BB->end();
  }
  
  /// SetInsertPoint - This specifies that created instructions should be
  /// inserted at the specified point.
  void SetInsertPoint(BasicBlock *TheBB, BasicBlock::iterator IP) {
    BB = TheBB;
    InsertPt = IP;
  }
  
  /// Insert - Insert and return the specified instruction.
  template<typename InstTy>
  InstTy *Insert(InstTy *I) const {
    InsertHelper(I);
    return I;
  }
  
  /// InsertHelper - Insert the specified instruction at the specified insertion
  /// point.  This is split out of Insert so that it isn't duplicated for every
  /// template instantiation.
  void InsertHelper(Instruction *I) const {
    if (BB) BB->getInstList().insert(InsertPt, I);
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Terminators
  //===--------------------------------------------------------------------===//

  /// CreateRetVoid - Create a 'ret void' instruction.
  ReturnInst *CreateRetVoid() {
    return Insert(ReturnInst::Create());
  }

  /// @verbatim 
  /// CreateRet - Create a 'ret <val>' instruction. 
  /// @endverbatim
  ReturnInst *CreateRet(Value *V) {
    return Insert(ReturnInst::Create(V));
  }

  ReturnInst *CreateRet(Value * const* retVals, unsigned N) {
    return Insert(ReturnInst::Create(retVals,  N));
  }
  
  GetResultInst *CreateGetResult(Value *V, unsigned Index, 
                                 const char *Name = "") {
    return Insert(new GetResultInst(V, Index, Name));
  }
    
  /// CreateBr - Create an unconditional 'br label X' instruction.
  BranchInst *CreateBr(BasicBlock *Dest) {
    return Insert(BranchInst::Create(Dest));
  }

  /// CreateCondBr - Create a conditional 'br Cond, TrueDest, FalseDest'
  /// instruction.
  BranchInst *CreateCondBr(Value *Cond, BasicBlock *True, BasicBlock *False) {
    return Insert(BranchInst::Create(True, False, Cond));
  }
  
  /// CreateSwitch - Create a switch instruction with the specified value,
  /// default dest, and with a hint for the number of cases that will be added
  /// (for efficient allocation).
  SwitchInst *CreateSwitch(Value *V, BasicBlock *Dest, unsigned NumCases = 10) {
    return Insert(SwitchInst::Create(V, Dest, NumCases));
  }
  
  /// CreateInvoke - Create an invoke instruction.
  template<typename InputIterator>
  InvokeInst *CreateInvoke(Value *Callee, BasicBlock *NormalDest, 
                           BasicBlock *UnwindDest, InputIterator ArgBegin, 
                           InputIterator ArgEnd, const char *Name = "") {
    return Insert(InvokeInst::Create(Callee, NormalDest, UnwindDest,
                                     ArgBegin, ArgEnd, Name));
  }
  
  UnwindInst *CreateUnwind() {
    return Insert(new UnwindInst());
  }

  UnreachableInst *CreateUnreachable() {
    return Insert(new UnreachableInst());
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Binary Operators
  //===--------------------------------------------------------------------===//

  Value *CreateAdd(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getAdd(LC, RC);      
    return Insert(BinaryOperator::createAdd(LHS, RHS, Name));
  }
  Value *CreateSub(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getSub(LC, RC);
    return Insert(BinaryOperator::createSub(LHS, RHS, Name));
  }
  Value *CreateMul(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getMul(LC, RC);
    return Insert(BinaryOperator::createMul(LHS, RHS, Name));
  }
  Value *CreateUDiv(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getUDiv(LC, RC);
    return Insert(BinaryOperator::createUDiv(LHS, RHS, Name));
  }
  Value *CreateSDiv(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getSDiv(LC, RC);      
    return Insert(BinaryOperator::createSDiv(LHS, RHS, Name));
  }
  Value *CreateFDiv(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getFDiv(LC, RC);      
    return Insert(BinaryOperator::createFDiv(LHS, RHS, Name));
  }
  Value *CreateURem(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getURem(LC, RC);
    return Insert(BinaryOperator::createURem(LHS, RHS, Name));
  }
  Value *CreateSRem(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getSRem(LC, RC);
    return Insert(BinaryOperator::createSRem(LHS, RHS, Name));
  }
  Value *CreateFRem(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getFRem(LC, RC);
    return Insert(BinaryOperator::createFRem(LHS, RHS, Name));
  }
  Value *CreateShl(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getShl(LC, RC);
    return Insert(BinaryOperator::createShl(LHS, RHS, Name));
  }
  Value *CreateLShr(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getLShr(LC, RC);
    return Insert(BinaryOperator::createLShr(LHS, RHS, Name));
  }
  Value *CreateAShr(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getAShr(LC, RC);
    return Insert(BinaryOperator::createAShr(LHS, RHS, Name));
  }
  Value *CreateAnd(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getAnd(LC, RC);
    return Insert(BinaryOperator::createAnd(LHS, RHS, Name));
  }
  Value *CreateOr(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getOr(LC, RC);
    return Insert(BinaryOperator::createOr(LHS, RHS, Name));
  }
  Value *CreateXor(Value *LHS, Value *RHS, const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getXor(LC, RC);
    return Insert(BinaryOperator::createXor(LHS, RHS, Name));
  }

  BinaryOperator *CreateBinOp(Instruction::BinaryOps Opc,
                              Value *LHS, Value *RHS, const char *Name = "") {
    return Insert(BinaryOperator::create(Opc, LHS, RHS, Name));
  }
  
  BinaryOperator *CreateNeg(Value *V, const char *Name = "") {
    return Insert(BinaryOperator::createNeg(V, Name));
  }
  BinaryOperator *CreateNot(Value *V, const char *Name = "") {
    return Insert(BinaryOperator::createNot(V, Name));
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Memory Instructions
  //===--------------------------------------------------------------------===//
  
  MallocInst *CreateMalloc(const Type *Ty, Value *ArraySize = 0,
                           const char *Name = "") {
    return Insert(new MallocInst(Ty, ArraySize, Name));
  }
  AllocaInst *CreateAlloca(const Type *Ty, Value *ArraySize = 0,
                           const char *Name = "") {
    return Insert(new AllocaInst(Ty, ArraySize, Name));
  }
  FreeInst *CreateFree(Value *Ptr) {
    return Insert(new FreeInst(Ptr));
  }
  LoadInst *CreateLoad(Value *Ptr, const char *Name = 0) {
    return Insert(new LoadInst(Ptr, Name));
  }
  LoadInst *CreateLoad(Value *Ptr, bool isVolatile, const char *Name = 0) {
    return Insert(new LoadInst(Ptr, Name, isVolatile));
  }
  StoreInst *CreateStore(Value *Val, Value *Ptr, bool isVolatile = false) {
    return Insert(new StoreInst(Val, Ptr, isVolatile));
  }
  template<typename InputIterator>
  Value *CreateGEP(Value *Ptr, InputIterator IdxBegin, 
                               InputIterator IdxEnd, const char *Name = "") {
      
    if (Constant *PC = dyn_cast<Constant>(Ptr)) {
      // Every index must be constant.
      InputIterator i;
      for (i = IdxBegin; i < IdxEnd; ++i) {
        if (!dyn_cast<Constant>(*i))
          break;
      }
      if (i == IdxEnd)
        return ConstantExpr::getGetElementPtr(PC, &IdxBegin[0], IdxEnd - IdxBegin);
    }      
    return(Insert(GetElementPtrInst::Create(Ptr, IdxBegin, IdxEnd, Name)));
  }
  Value *CreateGEP(Value *Ptr, Value *Idx, const char *Name = "") {
    if (Constant *PC = dyn_cast<Constant>(Ptr))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return ConstantExpr::getGetElementPtr(PC, &IC, 1);
    return Insert(GetElementPtrInst::Create(Ptr, Idx, Name));
  }
  Value *CreateStructGEP(Value *Ptr, unsigned Idx, const char *Name = "") {
    llvm::Value *Idxs[] = {
      ConstantInt::get(llvm::Type::Int32Ty, 0),
      ConstantInt::get(llvm::Type::Int32Ty, Idx)
    };
    
    if (Constant *PC = dyn_cast<Constant>(Ptr))
      return ConstantExpr::getGetElementPtr(PC, Idxs, 2);
    
    return Insert(GetElementPtrInst::Create(Ptr, Idxs, Idxs+2, Name));
  }
  
  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Cast/Conversion Operators
  //===--------------------------------------------------------------------===//
    
  Value *CreateTrunc(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::Trunc, V, DestTy, Name);
  }
  Value *CreateZExt(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::ZExt, V, DestTy, Name);
  }
  Value *CreateSExt(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::SExt, V, DestTy, Name);
  }
  Value *CreateFPToUI(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::FPToUI, V, DestTy, Name);
  }
  Value *CreateFPToSI(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::FPToSI, V, DestTy, Name);
  }
  Value *CreateUIToFP(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::UIToFP, V, DestTy, Name);
  }
  Value *CreateSIToFP(Value *V, const Type *DestTy, const char *Name = ""){
    return CreateCast(Instruction::SIToFP, V, DestTy, Name);
  }
  Value *CreateFPTrunc(Value *V, const Type *DestTy,
                       const char *Name = "") {
    return CreateCast(Instruction::FPTrunc, V, DestTy, Name);
  }
  Value *CreateFPExt(Value *V, const Type *DestTy, const char *Name = "") {
    return CreateCast(Instruction::FPExt, V, DestTy, Name);
  }
  Value *CreatePtrToInt(Value *V, const Type *DestTy,
                        const char *Name = "") {
    return CreateCast(Instruction::PtrToInt, V, DestTy, Name);
  }
  Value *CreateIntToPtr(Value *V, const Type *DestTy,
                        const char *Name = "") {
    return CreateCast(Instruction::IntToPtr, V, DestTy, Name);
  }
  Value *CreateBitCast(Value *V, const Type *DestTy,
                       const char *Name = "") {
    return CreateCast(Instruction::BitCast, V, DestTy, Name);
  }

  Value *CreateCast(Instruction::CastOps Op, Value *V, const Type *DestTy,
                     const char *Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return ConstantExpr::getCast(Op, VC, DestTy);      
    return Insert(CastInst::create(Op, V, DestTy, Name));
  }
  Value *CreateIntCast(Value *V, const Type *DestTy, bool isSigned,
                        const char *Name = "") {
    if (V->getType() == DestTy)
      return V;
    if (Constant *VC = dyn_cast<Constant>(V))
      return ConstantExpr::getIntegerCast(VC, DestTy, isSigned);
    return Insert(CastInst::createIntegerCast(V, DestTy, isSigned, Name));
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Compare Instructions
  //===--------------------------------------------------------------------===//
  
  Value *CreateICmpEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_EQ, LHS, RHS, Name);
  }
  Value *CreateICmpNE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_NE, LHS, RHS, Name);
  }
  Value *CreateICmpUGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGT, LHS, RHS, Name);
  }
  Value *CreateICmpUGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_UGE, LHS, RHS, Name);
  }
  Value *CreateICmpULT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULT, LHS, RHS, Name);
  }
  Value *CreateICmpULE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_ULE, LHS, RHS, Name);
  }
  Value *CreateICmpSGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGT, LHS, RHS, Name);
  }
  Value *CreateICmpSGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SGE, LHS, RHS, Name);
  }
  Value *CreateICmpSLT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLT, LHS, RHS, Name);
  }
  Value *CreateICmpSLE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateICmp(ICmpInst::ICMP_SLE, LHS, RHS, Name);
  }
  
  Value *CreateFCmpOEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OEQ, LHS, RHS, Name);
  }
  Value *CreateFCmpOGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OGT, LHS, RHS, Name);
  }
  Value *CreateFCmpOGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OGE, LHS, RHS, Name);
  }
  Value *CreateFCmpOLT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OLT, LHS, RHS, Name);
  }
  Value *CreateFCmpOLE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_OLE, LHS, RHS, Name);
  }
  Value *CreateFCmpONE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ONE, LHS, RHS, Name);
  }
  Value *CreateFCmpORD(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ORD, LHS, RHS, Name);
  }
  Value *CreateFCmpUNO(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UNO, LHS, RHS, Name);
  }
  Value *CreateFCmpUEQ(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UEQ, LHS, RHS, Name);
  }
  Value *CreateFCmpUGT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UGT, LHS, RHS, Name);
  }
  Value *CreateFCmpUGE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UGE, LHS, RHS, Name);
  }
  Value *CreateFCmpULT(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ULT, LHS, RHS, Name);
  }
  Value *CreateFCmpULE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_ULE, LHS, RHS, Name);
  }
  Value *CreateFCmpUNE(Value *LHS, Value *RHS, const char *Name = "") {
    return CreateFCmp(FCmpInst::FCMP_UNE, LHS, RHS, Name);
  }

  Value *CreateICmp(ICmpInst::Predicate P, Value *LHS, Value *RHS, 
                     const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getCompare(P, LC, RC);      
    return Insert(new ICmpInst(P, LHS, RHS, Name));
  }
  Value *CreateFCmp(FCmpInst::Predicate P, Value *LHS, Value *RHS, 
                     const char *Name = "") {
    if (Constant *LC = dyn_cast<Constant>(LHS))
      if (Constant *RC = dyn_cast<Constant>(RHS))
        return ConstantExpr::getCompare(P, LC, RC);
    return Insert(new FCmpInst(P, LHS, RHS, Name));
  }

  //===--------------------------------------------------------------------===//
  // Instruction creation methods: Other Instructions
  //===--------------------------------------------------------------------===//

  PHINode *CreatePHI(const Type *Ty, const char *Name = "") {
    return Insert(PHINode::Create(Ty, Name));
  }

  CallInst *CreateCall(Value *Callee, const char *Name = "") {
    return Insert(CallInst::Create(Callee, Name));
  }
  CallInst *CreateCall(Value *Callee, Value *Arg, const char *Name = "") {
    return Insert(CallInst::Create(Callee, Arg, Name));
  }

  template<typename InputIterator>
  CallInst *CreateCall(Value *Callee, InputIterator ArgBegin, 
                     InputIterator ArgEnd, const char *Name = "") {
    return Insert(CallInst::Create(Callee, ArgBegin, ArgEnd, Name));
  }

  Value *CreateSelect(Value *C, Value *True, Value *False,
                         const char *Name = "") {
    if (Constant *CC = dyn_cast<Constant>(C))
      if (Constant *TC = dyn_cast<Constant>(True))
        if (Constant *FC = dyn_cast<Constant>(False))
          return ConstantExpr::getSelect(CC, TC, FC);      
    return Insert(SelectInst::Create(C, True, False, Name));
  }

  VAArgInst *CreateVAArg(Value *List, const Type *Ty, const char *Name = "") {
    return Insert(new VAArgInst(List, Ty, Name));
  }

  Value *CreateExtractElement(Value *Vec, Value *Idx,
                                         const char *Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *IC = dyn_cast<Constant>(Idx))
        return ConstantExpr::getExtractElement(VC, IC);
    return Insert(new ExtractElementInst(Vec, Idx, Name));
  }

  Value *CreateInsertElement(Value *Vec, Value *NewElt, Value *Idx,
                             const char *Name = "") {
    if (Constant *VC = dyn_cast<Constant>(Vec))
      if (Constant *NC = dyn_cast<Constant>(NewElt))
        if (Constant *IC = dyn_cast<Constant>(Idx))
          return ConstantExpr::getInsertElement(VC, NC, IC);
    return Insert(InsertElementInst::Create(Vec, NewElt, Idx, Name));
  }

  Value *CreateShuffleVector(Value *V1, Value *V2, Value *Mask,
                                       const char *Name = "") {
    if (Constant *V1C = dyn_cast<Constant>(V1))
      if (Constant *V2C = dyn_cast<Constant>(V2))
        if (Constant *MC = dyn_cast<Constant>(Mask))
          return ConstantExpr::getShuffleVector(V1C, V2C, MC);      
    return Insert(new ShuffleVectorInst(V1, V2, Mask, Name));
  }
};
  
}

#endif

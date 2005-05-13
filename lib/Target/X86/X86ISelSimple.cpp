//===-- X86ISelSimple.cpp - A simple instruction selector for x86 ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple peephole instruction selector for the x86 target
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

namespace {
  Statistic<>
  NumFPKill("x86-codegen", "Number of FP_REG_KILL instructions added");

  /// TypeClass - Used by the X86 backend to group LLVM types by their basic X86
  /// Representation.
  ///
  enum TypeClass {
    cByte, cShort, cInt, cFP, cLong
  };
}

/// getClass - Turn a primitive type into a "class" number which is based on the
/// size of the type, and whether or not it is floating point.
///
static inline TypeClass getClass(const Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::SByteTyID:
  case Type::UByteTyID:   return cByte;      // Byte operands are class #0
  case Type::ShortTyID:
  case Type::UShortTyID:  return cShort;     // Short operands are class #1
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return cInt;       // Int's and pointers are class #2

  case Type::FloatTyID:
  case Type::DoubleTyID:  return cFP;        // Floating Point is #3

  case Type::LongTyID:
  case Type::ULongTyID:   return cLong;      // Longs are class #4
  default:
    assert(0 && "Invalid type to getClass!");
    return cByte;  // not reached
  }
}

// getClassB - Just like getClass, but treat boolean values as bytes.
static inline TypeClass getClassB(const Type *Ty) {
  if (Ty == Type::BoolTy) return cByte;
  return getClass(Ty);
}

namespace {
  struct X86ISel : public FunctionPass, InstVisitor<X86ISel> {
    TargetMachine &TM;
    MachineFunction *F;                 // The function we are compiling into
    MachineBasicBlock *BB;              // The current MBB we are compiling
    int VarArgsFrameIndex;              // FrameIndex for start of varargs area
    int ReturnAddressIndex;             // FrameIndex for the return address

    std::map<Value*, unsigned> RegMap;  // Mapping between Val's and SSA Regs

    // MBBMap - Mapping between LLVM BB -> Machine BB
    std::map<const BasicBlock*, MachineBasicBlock*> MBBMap;

    // AllocaMap - Mapping from fixed sized alloca instructions to the
    // FrameIndex for the alloca.
    std::map<AllocaInst*, unsigned> AllocaMap;

    X86ISel(TargetMachine &tm) : TM(tm), F(0), BB(0) {}

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnFunction(Function &Fn) {
      // Lazily create a stack slot for the return address if needed.
      ReturnAddressIndex = 0;

      // First pass over the function, lower any unknown intrinsic functions
      // with the IntrinsicLowering class.
      LowerUnknownIntrinsicFunctionCalls(Fn);

      F = &MachineFunction::construct(&Fn, TM);

      // Create all of the machine basic blocks for the function...
      for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
        F->getBasicBlockList().push_back(MBBMap[I] = new MachineBasicBlock(I));

      BB = &F->front();

      // Copy incoming arguments off of the stack...
      LoadArgumentsToVirtualRegs(Fn);

      // If this is main, emit special code.
      if (Fn.hasExternalLinkage() && Fn.getName() == "main")
        EmitSpecialCodeForMain();

      // Instruction select everything except PHI nodes
      visit(Fn);

      // Select the PHI nodes
      SelectPHINodes();

      // Insert the FP_REG_KILL instructions into blocks that need them.
      InsertFPRegKills();

      RegMap.clear();
      MBBMap.clear();
      AllocaMap.clear();
      F = 0;
      // We always build a machine code representation for the function
      return true;
    }

    virtual const char *getPassName() const {
      return "X86 Simple Instruction Selection";
    }

    /// EmitSpecialCodeForMain - Emit any code that needs to be executed only in
    /// the main function.
    void EmitSpecialCodeForMain();

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = MBBMap[&LLVM_BB];
    }

    /// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
    /// function, lowering any calls to unknown intrinsic functions into the
    /// equivalent LLVM code.
    ///
    void LowerUnknownIntrinsicFunctionCalls(Function &F);

    /// LoadArgumentsToVirtualRegs - Load all of the arguments to this function
    /// from the stack into virtual registers.
    ///
    void LoadArgumentsToVirtualRegs(Function &F);

    /// SelectPHINodes - Insert machine code to generate phis.  This is tricky
    /// because we have to generate our sources into the source basic blocks,
    /// not the current one.
    ///
    void SelectPHINodes();

    /// InsertFPRegKills - Insert FP_REG_KILL instructions into basic blocks
    /// that need them.  This only occurs due to the floating point stackifier
    /// not being aggressive enough to handle arbitrary global stackification.
    ///
    void InsertFPRegKills();

    // Visitation methods for various instructions.  These methods simply emit
    // fixed X86 code for each instruction.
    //

    // Control flow operators
    void visitReturnInst(ReturnInst &RI);
    void visitBranchInst(BranchInst &BI);
    void visitUnreachableInst(UnreachableInst &UI) {}

    struct ValueRecord {
      Value *Val;
      unsigned Reg;
      const Type *Ty;
      ValueRecord(unsigned R, const Type *T) : Val(0), Reg(R), Ty(T) {}
      ValueRecord(Value *V) : Val(V), Reg(0), Ty(V->getType()) {}
    };
    void doCall(const ValueRecord &Ret, MachineInstr *CallMI,
                const std::vector<ValueRecord> &Args);
    void visitCallInst(CallInst &I);
    void visitIntrinsicCall(Intrinsic::ID ID, CallInst &I);

    // Arithmetic operators
    void visitSimpleBinary(BinaryOperator &B, unsigned OpcodeClass);
    void visitAdd(BinaryOperator &B) { visitSimpleBinary(B, 0); }
    void visitSub(BinaryOperator &B) { visitSimpleBinary(B, 1); }
    void visitMul(BinaryOperator &B);

    void visitDiv(BinaryOperator &B) { visitDivRem(B); }
    void visitRem(BinaryOperator &B) { visitDivRem(B); }
    void visitDivRem(BinaryOperator &B);

    // Bitwise operators
    void visitAnd(BinaryOperator &B) { visitSimpleBinary(B, 2); }
    void visitOr (BinaryOperator &B) { visitSimpleBinary(B, 3); }
    void visitXor(BinaryOperator &B) { visitSimpleBinary(B, 4); }

    // Comparison operators...
    void visitSetCondInst(SetCondInst &I);
    unsigned EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                            MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator MBBI);
    void visitSelectInst(SelectInst &SI);


    // Memory Instructions
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitAllocaInst(AllocaInst &I);
    void visitMallocInst(MallocInst &I);
    void visitFreeInst(FreeInst &I);

    // Other operators
    void visitShiftInst(ShiftInst &I);
    void visitPHINode(PHINode &I) {}      // PHI nodes handled by second pass
    void visitCastInst(CastInst &I);
    void visitVANextInst(VANextInst &I);
    void visitVAArgInst(VAArgInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "Cannot instruction select: " << I;
      abort();
    }

    /// promote32 - Make a value 32-bits wide, and put it somewhere.
    ///
    void promote32(unsigned targetReg, const ValueRecord &VR);

    /// getAddressingMode - Get the addressing mode to use to address the
    /// specified value.  The returned value should be used with addFullAddress.
    void getAddressingMode(Value *Addr, X86AddressMode &AM);


    /// getGEPIndex - This is used to fold GEP instructions into X86 addressing
    /// expressions.
    void getGEPIndex(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                     std::vector<Value*> &GEPOps,
                     std::vector<const Type*> &GEPTypes,
                     X86AddressMode &AM);

    /// isGEPFoldable - Return true if the specified GEP can be completely
    /// folded into the addressing mode of a load/store or lea instruction.
    bool isGEPFoldable(MachineBasicBlock *MBB,
                       Value *Src, User::op_iterator IdxBegin,
                       User::op_iterator IdxEnd, X86AddressMode &AM);

    /// emitGEPOperation - Common code shared between visitGetElementPtrInst and
    /// constant expression GEP support.
    ///
    void emitGEPOperation(MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                          Value *Src, User::op_iterator IdxBegin,
                          User::op_iterator IdxEnd, unsigned TargetReg);

    /// emitCastOperation - Common code shared between visitCastInst and
    /// constant expression cast support.
    ///
    void emitCastOperation(MachineBasicBlock *BB,MachineBasicBlock::iterator IP,
                           Value *Src, const Type *DestTy, unsigned TargetReg);

    /// emitSimpleBinaryOperation - Common code shared between visitSimpleBinary
    /// and constant expression support.
    ///
    void emitSimpleBinaryOperation(MachineBasicBlock *BB,
                                   MachineBasicBlock::iterator IP,
                                   Value *Op0, Value *Op1,
                                   unsigned OperatorClass, unsigned TargetReg);

    /// emitBinaryFPOperation - This method handles emission of floating point
    /// Add (0), Sub (1), Mul (2), and Div (3) operations.
    void emitBinaryFPOperation(MachineBasicBlock *BB,
                               MachineBasicBlock::iterator IP,
                               Value *Op0, Value *Op1,
                               unsigned OperatorClass, unsigned TargetReg);

    void emitMultiply(MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                      Value *Op0, Value *Op1, unsigned TargetReg);

    void doMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, const Type *DestTy,
                    unsigned Op0Reg, unsigned Op1Reg);
    void doMultiplyConst(MachineBasicBlock *MBB,
                         MachineBasicBlock::iterator MBBI,
                         unsigned DestReg, const Type *DestTy,
                         unsigned Op0Reg, unsigned Op1Val);

    void emitDivRemOperation(MachineBasicBlock *BB,
                             MachineBasicBlock::iterator IP,
                             Value *Op0, Value *Op1, bool isDiv,
                             unsigned TargetReg);

    /// emitSetCCOperation - Common code shared between visitSetCondInst and
    /// constant expression support.
    ///
    void emitSetCCOperation(MachineBasicBlock *BB,
                            MachineBasicBlock::iterator IP,
                            Value *Op0, Value *Op1, unsigned Opcode,
                            unsigned TargetReg);

    /// emitShiftOperation - Common code shared between visitShiftInst and
    /// constant expression support.
    ///
    void emitShiftOperation(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator IP,
                            Value *Op, Value *ShiftAmount, bool isLeftShift,
                            const Type *ResultTy, unsigned DestReg);

    // Emit code for a 'SHLD DestReg, Op0, Op1, Amt' operation, where Amt is a
    // constant.
    void doSHLDConst(MachineBasicBlock *MBB,
                     MachineBasicBlock::iterator MBBI,
                     unsigned DestReg, unsigned Op0Reg, unsigned Op1Reg,
                     unsigned Op1Val);

    /// emitSelectOperation - Common code shared between visitSelectInst and the
    /// constant expression support.
    void emitSelectOperation(MachineBasicBlock *MBB,
                             MachineBasicBlock::iterator IP,
                             Value *Cond, Value *TrueVal, Value *FalseVal,
                             unsigned DestReg);

    /// copyConstantToRegister - Output the instructions required to put the
    /// specified constant into the specified register.
    ///
    void copyConstantToRegister(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator MBBI,
                                Constant *C, unsigned Reg);

    void emitUCOMr(MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
                   unsigned LHS, unsigned RHS);

    /// makeAnotherReg - This method returns the next register number we haven't
    /// yet used.
    ///
    /// Long values are handled somewhat specially.  They are always allocated
    /// as pairs of 32 bit integer values.  The register number returned is the
    /// lower 32 bits of the long value, and the regNum+1 is the upper 32 bits
    /// of the long value.
    ///
    unsigned makeAnotherReg(const Type *Ty) {
      assert(dynamic_cast<const X86RegisterInfo*>(TM.getRegisterInfo()) &&
             "Current target doesn't have X86 reg info??");
      const X86RegisterInfo *MRI =
        static_cast<const X86RegisterInfo*>(TM.getRegisterInfo());
      if (Ty == Type::LongTy || Ty == Type::ULongTy) {
        const TargetRegisterClass *RC = MRI->getRegClassForType(Type::IntTy);
        // Create the lower part
        F->getSSARegMap()->createVirtualRegister(RC);
        // Create the upper part.
        return F->getSSARegMap()->createVirtualRegister(RC)-1;
      }

      // Add the mapping of regnumber => reg class to MachineFunction
      const TargetRegisterClass *RC = MRI->getRegClassForType(Ty);
      return F->getSSARegMap()->createVirtualRegister(RC);
    }

    /// getReg - This method turns an LLVM value into a register number.
    ///
    unsigned getReg(Value &V) { return getReg(&V); }  // Allow references
    unsigned getReg(Value *V) {
      // Just append to the end of the current bb.
      MachineBasicBlock::iterator It = BB->end();
      return getReg(V, BB, It);
    }
    unsigned getReg(Value *V, MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator IPt);

    /// getFixedSizedAllocaFI - Return the frame index for a fixed sized alloca
    /// that is to be statically allocated with the initial stack frame
    /// adjustment.
    unsigned getFixedSizedAllocaFI(AllocaInst *AI);
  };
}

/// dyn_castFixedAlloca - If the specified value is a fixed size alloca
/// instruction in the entry block, return it.  Otherwise, return a null
/// pointer.
static AllocaInst *dyn_castFixedAlloca(Value *V) {
  if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    BasicBlock *BB = AI->getParent();
    if (isa<ConstantUInt>(AI->getArraySize()) && BB ==&BB->getParent()->front())
      return AI;
  }
  return 0;
}

/// getReg - This method turns an LLVM value into a register number.
///
unsigned X86ISel::getReg(Value *V, MachineBasicBlock *MBB,
                         MachineBasicBlock::iterator IPt) {
  // If this operand is a constant, emit the code to copy the constant into
  // the register here...
  if (Constant *C = dyn_cast<Constant>(V)) {
    unsigned Reg = makeAnotherReg(V->getType());
    copyConstantToRegister(MBB, IPt, C, Reg);
    return Reg;
  } else if (CastInst *CI = dyn_cast<CastInst>(V)) {
    // Do not emit noop casts at all, unless it's a double -> float cast.
    if (getClassB(CI->getType()) == getClassB(CI->getOperand(0)->getType()) &&
        (CI->getType() != Type::FloatTy ||
         CI->getOperand(0)->getType() != Type::DoubleTy))
      return getReg(CI->getOperand(0), MBB, IPt);
  } else if (AllocaInst *AI = dyn_castFixedAlloca(V)) {
    // If the alloca address couldn't be folded into the instruction addressing,
    // emit an explicit LEA as appropriate.
    unsigned Reg = makeAnotherReg(V->getType());
    unsigned FI = getFixedSizedAllocaFI(AI);
    addFrameReference(BuildMI(*MBB, IPt, X86::LEA32r, 4, Reg), FI);
    return Reg;
  }

  unsigned &Reg = RegMap[V];
  if (Reg == 0) {
    Reg = makeAnotherReg(V->getType());
    RegMap[V] = Reg;
  }

  return Reg;
}

/// getFixedSizedAllocaFI - Return the frame index for a fixed sized alloca
/// that is to be statically allocated with the initial stack frame
/// adjustment.
unsigned X86ISel::getFixedSizedAllocaFI(AllocaInst *AI) {
  // Already computed this?
  std::map<AllocaInst*, unsigned>::iterator I = AllocaMap.lower_bound(AI);
  if (I != AllocaMap.end() && I->first == AI) return I->second;

  const Type *Ty = AI->getAllocatedType();
  ConstantUInt *CUI = cast<ConstantUInt>(AI->getArraySize());
  unsigned TySize = TM.getTargetData().getTypeSize(Ty);
  TySize *= CUI->getValue();   // Get total allocated size...
  unsigned Alignment = TM.getTargetData().getTypeAlignment(Ty);

  // Create a new stack object using the frame manager...
  int FrameIdx = F->getFrameInfo()->CreateStackObject(TySize, Alignment);
  AllocaMap.insert(I, std::make_pair(AI, FrameIdx));
  return FrameIdx;
}


/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void X86ISel::copyConstantToRegister(MachineBasicBlock *MBB,
                                     MachineBasicBlock::iterator IP,
                                     Constant *C, unsigned R) {
  if (isa<UndefValue>(C)) {
    switch (getClassB(C->getType())) {
    case cFP:
      // FIXME: SHOULD TEACH STACKIFIER ABOUT UNDEF VALUES!
      BuildMI(*MBB, IP, X86::FLD0, 0, R);
      return;
    case cLong:
      BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, R+1);
      // FALL THROUGH
    default:
      BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, R);
      return;
    }
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    unsigned Class = 0;
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      emitGEPOperation(MBB, IP, CE->getOperand(0),
                       CE->op_begin()+1, CE->op_end(), R);
      return;
    case Instruction::Cast:
      emitCastOperation(MBB, IP, CE->getOperand(0), CE->getType(), R);
      return;

    case Instruction::Xor: ++Class; // FALL THROUGH
    case Instruction::Or:  ++Class; // FALL THROUGH
    case Instruction::And: ++Class; // FALL THROUGH
    case Instruction::Sub: ++Class; // FALL THROUGH
    case Instruction::Add:
      emitSimpleBinaryOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                                Class, R);
      return;

    case Instruction::Mul:
      emitMultiply(MBB, IP, CE->getOperand(0), CE->getOperand(1), R);
      return;

    case Instruction::Div:
    case Instruction::Rem:
      emitDivRemOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                          CE->getOpcode() == Instruction::Div, R);
      return;

    case Instruction::SetNE:
    case Instruction::SetEQ:
    case Instruction::SetLT:
    case Instruction::SetGT:
    case Instruction::SetLE:
    case Instruction::SetGE:
      emitSetCCOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                         CE->getOpcode(), R);
      return;

    case Instruction::Shl:
    case Instruction::Shr:
      emitShiftOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                         CE->getOpcode() == Instruction::Shl, CE->getType(), R);
      return;

    case Instruction::Select:
      emitSelectOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                          CE->getOperand(2), R);
      return;

    default:
      std::cerr << "Offending expr: " << *C << "\n";
      assert(0 && "Constant expression not yet handled!\n");
    }
  }

  if (C->getType()->isIntegral()) {
    unsigned Class = getClassB(C->getType());

    if (Class == cLong) {
      // Copy the value into the register pair.
      uint64_t Val = cast<ConstantInt>(C)->getRawValue();
      BuildMI(*MBB, IP, X86::MOV32ri, 1, R).addImm(Val & 0xFFFFFFFF);
      BuildMI(*MBB, IP, X86::MOV32ri, 1, R+1).addImm(Val >> 32);
      return;
    }

    assert(Class <= cInt && "Type not handled yet!");

    static const unsigned IntegralOpcodeTab[] = {
      X86::MOV8ri, X86::MOV16ri, X86::MOV32ri
    };

    if (C->getType() == Type::BoolTy) {
      BuildMI(*MBB, IP, X86::MOV8ri, 1, R).addImm(C == ConstantBool::True);
    } else {
      ConstantInt *CI = cast<ConstantInt>(C);
      BuildMI(*MBB, IP, IntegralOpcodeTab[Class],1,R).addImm(CI->getRawValue());
    }
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    if (CFP->isExactlyValue(+0.0))
      BuildMI(*MBB, IP, X86::FLD0, 0, R);
    else if (CFP->isExactlyValue(+1.0))
      BuildMI(*MBB, IP, X86::FLD1, 0, R);
    else if (CFP->isExactlyValue(-0.0)) {
      unsigned Tmp = makeAnotherReg(Type::DoubleTy);
      BuildMI(*MBB, IP, X86::FLD0, 0, Tmp);
      BuildMI(*MBB, IP, X86::FCHS, 1, R).addReg(Tmp);
    } else if (CFP->isExactlyValue(-1.0)) {
      unsigned Tmp = makeAnotherReg(Type::DoubleTy);
      BuildMI(*MBB, IP, X86::FLD1, 0, Tmp);
      BuildMI(*MBB, IP, X86::FCHS, 1, R).addReg(Tmp);
    } else {  // FIXME: PI, other native values
      // FIXME: 2*PI -> LDPI + FADD

      // Otherwise we need to spill the constant to memory.
      MachineConstantPool *CP = F->getConstantPool();

      const Type *Ty = CFP->getType();

      // If a FP immediate is precise when represented as a float, we put it
      // into the constant pool as a float, even if it's is statically typed as
      // a double.
      if (Ty == Type::DoubleTy)
        if (CFP->isExactlyValue((float)CFP->getValue())) {
          Ty = Type::FloatTy;
          CFP = cast<ConstantFP>(ConstantExpr::getCast(CFP, Ty));
        }

      unsigned CPI = CP->getConstantPoolIndex(CFP);

      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
      unsigned LoadOpcode = Ty == Type::FloatTy ? X86::FLD32m : X86::FLD64m;
      addConstantPoolReference(BuildMI(*MBB, IP, LoadOpcode, 4, R), CPI);
    }

  } else if (isa<ConstantPointerNull>(C)) {
    // Copy zero (null pointer) to the register.
    BuildMI(*MBB, IP, X86::MOV32ri, 1, R).addImm(0);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
    BuildMI(*MBB, IP, X86::MOV32ri, 1, R).addGlobalAddress(GV);
  } else {
    std::cerr << "Offending constant: " << *C << "\n";
    assert(0 && "Type not handled yet!");
  }
}

/// LoadArgumentsToVirtualRegs - Load all of the arguments to this function from
/// the stack into virtual registers.
///
void X86ISel::LoadArgumentsToVirtualRegs(Function &Fn) {
  // Emit instructions to load the arguments...  On entry to a function on the
  // X86, the stack frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first argument (leftmost lexically)
  // [ESP + 8] -- second argument, if first argument is four bytes in size
  //    ...
  //
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  MachineFrameInfo *MFI = F->getFrameInfo();

  for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end();
       I != E; ++I) {
    bool ArgLive = !I->use_empty();
    unsigned Reg = ArgLive ? getReg(*I) : 0;
    int FI;          // Frame object index

    switch (getClassB(I->getType())) {
    case cByte:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(1, ArgOffset);
        addFrameReference(BuildMI(BB, X86::MOV8rm, 4, Reg), FI);
      }
      break;
    case cShort:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(2, ArgOffset);
        addFrameReference(BuildMI(BB, X86::MOV16rm, 4, Reg), FI);
      }
      break;
    case cInt:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(4, ArgOffset);
        addFrameReference(BuildMI(BB, X86::MOV32rm, 4, Reg), FI);
      }
      break;
    case cLong:
      if (ArgLive) {
        FI = MFI->CreateFixedObject(8, ArgOffset);
        addFrameReference(BuildMI(BB, X86::MOV32rm, 4, Reg), FI);
        addFrameReference(BuildMI(BB, X86::MOV32rm, 4, Reg+1), FI, 4);
      }
      ArgOffset += 4;   // longs require 4 additional bytes
      break;
    case cFP:
      if (ArgLive) {
        unsigned Opcode;
        if (I->getType() == Type::FloatTy) {
          Opcode = X86::FLD32m;
          FI = MFI->CreateFixedObject(4, ArgOffset);
        } else {
          Opcode = X86::FLD64m;
          FI = MFI->CreateFixedObject(8, ArgOffset);
        }
        addFrameReference(BuildMI(BB, Opcode, 4, Reg), FI);
      }
      if (I->getType() == Type::DoubleTy)
        ArgOffset += 4;   // doubles require 4 additional bytes
      break;
    default:
      assert(0 && "Unhandled argument type!");
    }
    ArgOffset += 4;  // Each argument takes at least 4 bytes on the stack...
  }

  // If the function takes variable number of arguments, add a frame offset for
  // the start of the first vararg value... this is used to expand
  // llvm.va_start.
  if (Fn.getFunctionType()->isVarArg())
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);

  // Finally, inform the compiler what our live-outs will be, aka, what we will
  // be returning in registers.
  if (Fn.getReturnType() != Type::VoidTy)
    switch (getClassB(Fn.getReturnType())) {
    default: assert(0 && "Unknown type!");
    case cByte:
    case cShort:
    case cInt:
      F->addLiveOut(X86::EAX);
      break;
    case cLong:
      F->addLiveOut(X86::EAX);
      F->addLiveOut(X86::EDX);
      break;
    case cFP:
      F->addLiveOut(X86::ST0);
      break;
    }
}

/// EmitSpecialCodeForMain - Emit any code that needs to be executed only in
/// the main function.
void X86ISel::EmitSpecialCodeForMain() {
  // Switch the FPU to 64-bit precision mode for better compatibility and speed.
  int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2);
  addFrameReference(BuildMI(BB, X86::FNSTCW16m, 4), CWFrameIdx);

  // Set the high part to be 64-bit precision.
  addFrameReference(BuildMI(BB, X86::MOV8mi, 5),
                    CWFrameIdx, 1).addImm(2);

  // Reload the modified control word now.
  addFrameReference(BuildMI(BB, X86::FLDCW16m, 4), CWFrameIdx);
}

/// SelectPHINodes - Insert machine code to generate phis.  This is tricky
/// because we have to generate our sources into the source basic blocks, not
/// the current one.
///
void X86ISel::SelectPHINodes() {
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const Function &LF = *F->getFunction();  // The LLVM function...
  for (Function::const_iterator I = LF.begin(), E = LF.end(); I != E; ++I) {
    const BasicBlock *BB = I;
    MachineBasicBlock &MBB = *MBBMap[I];

    // Loop over all of the PHI nodes in the LLVM basic block...
    MachineBasicBlock::iterator PHIInsertPoint = MBB.begin();
    for (BasicBlock::const_iterator I = BB->begin(); isa<PHINode>(I); ++I) {
      PHINode *PN = const_cast<PHINode*>(dyn_cast<PHINode>(I));

      // Create a new machine instr PHI node, and insert it.
      unsigned PHIReg = getReg(*PN);
      MachineInstr *PhiMI = BuildMI(MBB, PHIInsertPoint,
                                    X86::PHI, PN->getNumOperands(), PHIReg);

      MachineInstr *LongPhiMI = 0;
      if (PN->getType() == Type::LongTy || PN->getType() == Type::ULongTy)
        LongPhiMI = BuildMI(MBB, PHIInsertPoint,
                            X86::PHI, PN->getNumOperands(), PHIReg+1);

      // PHIValues - Map of blocks to incoming virtual registers.  We use this
      // so that we only initialize one incoming value for a particular block,
      // even if the block has multiple entries in the PHI node.
      //
      std::map<MachineBasicBlock*, unsigned> PHIValues;

      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        MachineBasicBlock *PredMBB = MBBMap[PN->getIncomingBlock(i)];
        unsigned ValReg;
        std::map<MachineBasicBlock*, unsigned>::iterator EntryIt =
          PHIValues.lower_bound(PredMBB);

        if (EntryIt != PHIValues.end() && EntryIt->first == PredMBB) {
          // We already inserted an initialization of the register for this
          // predecessor.  Recycle it.
          ValReg = EntryIt->second;

        } else {
          // Get the incoming value into a virtual register.
          //
          Value *Val = PN->getIncomingValue(i);

          // If this is a constant or GlobalValue, we may have to insert code
          // into the basic block to compute it into a virtual register.
          if ((isa<Constant>(Val) && !isa<ConstantExpr>(Val))) {
            // Simple constants get emitted at the end of the basic block,
            // before any terminator instructions.  We "know" that the code to
            // move a constant into a register will never clobber any flags.
            ValReg = getReg(Val, PredMBB, PredMBB->getFirstTerminator());
          } else {
            // Because we don't want to clobber any values which might be in
            // physical registers with the computation of this constant (which
            // might be arbitrarily complex if it is a constant expression),
            // just insert the computation at the top of the basic block.
            MachineBasicBlock::iterator PI = PredMBB->begin();

            // Skip over any PHI nodes though!
            while (PI != PredMBB->end() && PI->getOpcode() == X86::PHI)
              ++PI;

            ValReg = getReg(Val, PredMBB, PI);
          }

          // Remember that we inserted a value for this PHI for this predecessor
          PHIValues.insert(EntryIt, std::make_pair(PredMBB, ValReg));
        }

        PhiMI->addRegOperand(ValReg);
        PhiMI->addMachineBasicBlockOperand(PredMBB);
        if (LongPhiMI) {
          LongPhiMI->addRegOperand(ValReg+1);
          LongPhiMI->addMachineBasicBlockOperand(PredMBB);
        }
      }

      // Now that we emitted all of the incoming values for the PHI node, make
      // sure to reposition the InsertPoint after the PHI that we just added.
      // This is needed because we might have inserted a constant into this
      // block, right after the PHI's which is before the old insert point!
      PHIInsertPoint = LongPhiMI ? LongPhiMI : PhiMI;
      ++PHIInsertPoint;
    }
  }
}

/// RequiresFPRegKill - The floating point stackifier pass cannot insert
/// compensation code on critical edges.  As such, it requires that we kill all
/// FP registers on the exit from any blocks that either ARE critical edges, or
/// branch to a block that has incoming critical edges.
///
/// Note that this kill instruction will eventually be eliminated when
/// restrictions in the stackifier are relaxed.
///
static bool RequiresFPRegKill(const MachineBasicBlock *MBB) {
#if 0
  const BasicBlock *BB = MBB->getBasicBlock ();
  for (succ_const_iterator SI = succ_begin(BB), E = succ_end(BB); SI!=E; ++SI) {
    const BasicBlock *Succ = *SI;
    pred_const_iterator PI = pred_begin(Succ), PE = pred_end(Succ);
    ++PI;  // Block have at least one predecessory
    if (PI != PE) {             // If it has exactly one, this isn't crit edge
      // If this block has more than one predecessor, check all of the
      // predecessors to see if they have multiple successors.  If so, then the
      // block we are analyzing needs an FPRegKill.
      for (PI = pred_begin(Succ); PI != PE; ++PI) {
        const BasicBlock *Pred = *PI;
        succ_const_iterator SI2 = succ_begin(Pred);
        ++SI2;  // There must be at least one successor of this block.
        if (SI2 != succ_end(Pred))
          return true;   // Yes, we must insert the kill on this edge.
      }
    }
  }
  // If we got this far, there is no need to insert the kill instruction.
  return false;
#else
  return true;
#endif
}

// InsertFPRegKills - Insert FP_REG_KILL instructions into basic blocks that
// need them.  This only occurs due to the floating point stackifier not being
// aggressive enough to handle arbitrary global stackification.
//
// Currently we insert an FP_REG_KILL instruction into each block that uses or
// defines a floating point virtual register.
//
// When the global register allocators (like linear scan) finally update live
// variable analysis, we can keep floating point values in registers across
// portions of the CFG that do not involve critical edges.  This will be a big
// win, but we are waiting on the global allocators before we can do this.
//
// With a bit of work, the floating point stackifier pass can be enhanced to
// break critical edges as needed (to make a place to put compensation code),
// but this will require some infrastructure improvements as well.
//
void X86ISel::InsertFPRegKills() {
  SSARegMap &RegMap = *F->getSSARegMap();

  for (MachineFunction::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end(); I!=E; ++I)
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = I->getOperand(i);
        if (MO.isRegister() && MO.getReg()) {
          unsigned Reg = MO.getReg();
          if (MRegisterInfo::isVirtualRegister(Reg)) {
            unsigned RegSize = RegMap.getRegClass(Reg)->getSize();
            if (RegSize == 10 || RegSize == 8)
              goto UsesFPReg;
          }
        }
      }
    // If we haven't found an FP register use or def in this basic block, check
    // to see if any of our successors has an FP PHI node, which will cause a
    // copy to be inserted into this block.
    for (MachineBasicBlock::const_succ_iterator SI = BB->succ_begin(),
         SE = BB->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock *SBB = *SI;
      for (MachineBasicBlock::iterator I = SBB->begin();
           I != SBB->end() && I->getOpcode() == X86::PHI; ++I) {
        const TargetRegisterClass *RC =
          RegMap.getRegClass(I->getOperand(0).getReg());
        if (RC->getSize() == 10 || RC->getSize() == 8)
          goto UsesFPReg;
      }
    }
    continue;
  UsesFPReg:
    // Okay, this block uses an FP register.  If the block has successors (ie,
    // it's not an unwind/return), insert the FP_REG_KILL instruction.
    if (BB->succ_size() && RequiresFPRegKill(BB)) {
      BuildMI(*BB, BB->getFirstTerminator(), X86::FP_REG_KILL, 0);
      ++NumFPKill;
    }
  }
}


void X86ISel::getAddressingMode(Value *Addr, X86AddressMode &AM) {
  AM.BaseType = X86AddressMode::RegBase;
  AM.Base.Reg = 0; AM.Scale = 1; AM.IndexReg = 0; AM.Disp = 0;
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr)) {
    if (isGEPFoldable(BB, GEP->getOperand(0), GEP->op_begin()+1, GEP->op_end(),
                       AM))
      return;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Addr)) {
    if (CE->getOpcode() == Instruction::GetElementPtr)
      if (isGEPFoldable(BB, CE->getOperand(0), CE->op_begin()+1, CE->op_end(),
                        AM))
        return;
  } else if (AllocaInst *AI = dyn_castFixedAlloca(Addr)) {
    AM.BaseType = X86AddressMode::FrameIndexBase;
    AM.Base.FrameIndex = getFixedSizedAllocaFI(AI);
    return;
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(Addr)) {
    AM.GV = GV;
    return;
  }

  // If it's not foldable, reset addr mode.
  AM.BaseType = X86AddressMode::RegBase;
  AM.Base.Reg = getReg(Addr);
  AM.Scale = 1; AM.IndexReg = 0; AM.Disp = 0;
}

// canFoldSetCCIntoBranchOrSelect - Return the setcc instruction if we can fold
// it into the conditional branch or select instruction which is the only user
// of the cc instruction.  This is the case if the conditional branch is the
// only user of the setcc.  We also don't handle long arguments below, so we
// reject them here as well.
//
static SetCondInst *canFoldSetCCIntoBranchOrSelect(Value *V) {
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(V))
    if (SCI->hasOneUse()) {
      Instruction *User = cast<Instruction>(SCI->use_back());
      if ((isa<BranchInst>(User) || isa<SelectInst>(User)) &&
          (getClassB(SCI->getOperand(0)->getType()) != cLong ||
           SCI->getOpcode() == Instruction::SetEQ ||
           SCI->getOpcode() == Instruction::SetNE) &&
          (isa<BranchInst>(User) || User->getOperand(0) == V))
        return SCI;
    }
  return 0;
}

// Return a fixed numbering for setcc instructions which does not depend on the
// order of the opcodes.
//
static unsigned getSetCCNumber(unsigned Opcode) {
  switch(Opcode) {
  default: assert(0 && "Unknown setcc instruction!");
  case Instruction::SetEQ: return 0;
  case Instruction::SetNE: return 1;
  case Instruction::SetLT: return 2;
  case Instruction::SetGE: return 3;
  case Instruction::SetGT: return 4;
  case Instruction::SetLE: return 5;
  }
}

// LLVM  -> X86 signed  X86 unsigned
// -----    ----------  ------------
// seteq -> sete        sete
// setne -> setne       setne
// setlt -> setl        setb
// setge -> setge       setae
// setgt -> setg        seta
// setle -> setle       setbe
// ----
//          sets                       // Used by comparison with 0 optimization
//          setns
static const unsigned SetCCOpcodeTab[2][8] = {
  { X86::SETEr, X86::SETNEr, X86::SETBr, X86::SETAEr, X86::SETAr, X86::SETBEr,
    0, 0 },
  { X86::SETEr, X86::SETNEr, X86::SETLr, X86::SETGEr, X86::SETGr, X86::SETLEr,
    X86::SETSr, X86::SETNSr },
};

/// emitUCOMr - In the future when we support processors before the P6, this
/// wraps the logic for emitting an FUCOMr vs FUCOMIr.
void X86ISel::emitUCOMr(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                        unsigned LHS, unsigned RHS) {
  if (0) { // for processors prior to the P6
    BuildMI(*MBB, IP, X86::FUCOMr, 2).addReg(LHS).addReg(RHS);
    BuildMI(*MBB, IP, X86::FNSTSW8r, 0);
    BuildMI(*MBB, IP, X86::SAHF, 1);
  } else {
    BuildMI(*MBB, IP, X86::FUCOMIr, 2).addReg(LHS).addReg(RHS);
  }
}

// EmitComparison - This function emits a comparison of the two operands,
// returning the extended setcc code to use.
unsigned X86ISel::EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                                 MachineBasicBlock *MBB,
                                 MachineBasicBlock::iterator IP) {
  // The arguments are already supposed to be of the same type.
  const Type *CompTy = Op0->getType();
  unsigned Class = getClassB(CompTy);

  // Special case handling of: cmp R, i
  if (isa<ConstantPointerNull>(Op1)) {
    unsigned Op0r = getReg(Op0, MBB, IP);
    if (OpNum < 2)    // seteq/setne -> test
      BuildMI(*MBB, IP, X86::TEST32rr, 2).addReg(Op0r).addReg(Op0r);
    else
      BuildMI(*MBB, IP, X86::CMP32ri, 2).addReg(Op0r).addImm(0);
    return OpNum;

  } else if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    if (Class == cByte || Class == cShort || Class == cInt) {
      unsigned Op1v = CI->getRawValue();

      // Mask off any upper bits of the constant, if there are any...
      Op1v &= (1ULL << (8 << Class)) - 1;

      // If this is a comparison against zero, emit more efficient code.  We
      // can't handle unsigned comparisons against zero unless they are == or
      // !=.  These should have been strength reduced already anyway.
      if (Op1v == 0 && (CompTy->isSigned() || OpNum < 2)) {

        // If this is a comparison against zero and the LHS is an and of a
        // register with a constant, use the test to do the and.
        if (Instruction *Op0I = dyn_cast<Instruction>(Op0))
          if (Op0I->getOpcode() == Instruction::And && Op0->hasOneUse() &&
              isa<ConstantInt>(Op0I->getOperand(1))) {
            static const unsigned TESTTab[] = {
              X86::TEST8ri, X86::TEST16ri, X86::TEST32ri
            };

            // Emit test X, i
            unsigned LHS = getReg(Op0I->getOperand(0), MBB, IP);
            unsigned Imm =
              cast<ConstantInt>(Op0I->getOperand(1))->getRawValue();
            BuildMI(*MBB, IP, TESTTab[Class], 2).addReg(LHS).addImm(Imm);

            if (OpNum == 2) return 6;   // Map jl -> js
            if (OpNum == 3) return 7;   // Map jg -> jns
            return OpNum;
          }

        unsigned Op0r = getReg(Op0, MBB, IP);
        static const unsigned TESTTab[] = {
          X86::TEST8rr, X86::TEST16rr, X86::TEST32rr
        };
        BuildMI(*MBB, IP, TESTTab[Class], 2).addReg(Op0r).addReg(Op0r);

        if (OpNum == 2) return 6;   // Map jl -> js
        if (OpNum == 3) return 7;   // Map jg -> jns
        return OpNum;
      }

      static const unsigned CMPTab[] = {
        X86::CMP8ri, X86::CMP16ri, X86::CMP32ri
      };

      unsigned Op0r = getReg(Op0, MBB, IP);
      BuildMI(*MBB, IP, CMPTab[Class], 2).addReg(Op0r).addImm(Op1v);
      return OpNum;
    } else {
      unsigned Op0r = getReg(Op0, MBB, IP);
      assert(Class == cLong && "Unknown integer class!");
      unsigned LowCst = CI->getRawValue();
      unsigned HiCst = CI->getRawValue() >> 32;
      if (OpNum < 2) {    // seteq, setne
        unsigned LoTmp = Op0r;
        if (LowCst != 0) {
          LoTmp = makeAnotherReg(Type::IntTy);
          BuildMI(*MBB, IP, X86::XOR32ri, 2, LoTmp).addReg(Op0r).addImm(LowCst);
        }
        unsigned HiTmp = Op0r+1;
        if (HiCst != 0) {
          HiTmp = makeAnotherReg(Type::IntTy);
          BuildMI(*MBB, IP, X86::XOR32ri, 2,HiTmp).addReg(Op0r+1).addImm(HiCst);
        }
        unsigned FinalTmp = makeAnotherReg(Type::IntTy);
        BuildMI(*MBB, IP, X86::OR32rr, 2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
        return OpNum;
      } else {
        // Emit a sequence of code which compares the high and low parts once
        // each, then uses a conditional move to handle the overflow case.  For
        // example, a setlt for long would generate code like this:
        //
        // AL = lo(op1) < lo(op2)   // Always unsigned comparison
        // BL = hi(op1) < hi(op2)   // Signedness depends on operands
        // dest = hi(op1) == hi(op2) ? BL : AL;
        //

        // FIXME: This would be much better if we had hierarchical register
        // classes!  Until then, hardcode registers so that we can deal with
        // their aliases (because we don't have conditional byte moves).
        //
        BuildMI(*MBB, IP, X86::CMP32ri, 2).addReg(Op0r).addImm(LowCst);
        BuildMI(*MBB, IP, SetCCOpcodeTab[0][OpNum], 0, X86::AL);
        BuildMI(*MBB, IP, X86::CMP32ri, 2).addReg(Op0r+1).addImm(HiCst);
        BuildMI(*MBB, IP, SetCCOpcodeTab[CompTy->isSigned()][OpNum], 0,X86::BL);
        BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, X86::BH);
        BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, X86::AH);
        BuildMI(*MBB, IP, X86::CMOVE16rr, 2, X86::BX).addReg(X86::BX)
          .addReg(X86::AX);
        // NOTE: visitSetCondInst knows that the value is dumped into the BL
        // register at this point for long values...
        return OpNum;
      }
    }
  }

  unsigned Op0r = getReg(Op0, MBB, IP);

  // Special case handling of comparison against +/- 0.0
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(Op1))
    if (CFP->isExactlyValue(+0.0) || CFP->isExactlyValue(-0.0)) {
      BuildMI(*MBB, IP, X86::FTST, 1).addReg(Op0r);
      BuildMI(*MBB, IP, X86::FNSTSW8r, 0);
      BuildMI(*MBB, IP, X86::SAHF, 1);
      return OpNum;
    }

  unsigned Op1r = getReg(Op1, MBB, IP);
  switch (Class) {
  default: assert(0 && "Unknown type class!");
    // Emit: cmp <var1>, <var2> (do the comparison).  We can
    // compare 8-bit with 8-bit, 16-bit with 16-bit, 32-bit with
    // 32-bit.
  case cByte:
    BuildMI(*MBB, IP, X86::CMP8rr, 2).addReg(Op0r).addReg(Op1r);
    break;
  case cShort:
    BuildMI(*MBB, IP, X86::CMP16rr, 2).addReg(Op0r).addReg(Op1r);
    break;
  case cInt:
    BuildMI(*MBB, IP, X86::CMP32rr, 2).addReg(Op0r).addReg(Op1r);
    break;
  case cFP:
    emitUCOMr(MBB, IP, Op0r, Op1r);
    break;

  case cLong:
    if (OpNum < 2) {    // seteq, setne
      unsigned LoTmp = makeAnotherReg(Type::IntTy);
      unsigned HiTmp = makeAnotherReg(Type::IntTy);
      unsigned FinalTmp = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, X86::XOR32rr, 2, LoTmp).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, X86::XOR32rr, 2, HiTmp).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, X86::OR32rr,  2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
      break;  // Allow the sete or setne to be generated from flags set by OR
    } else {
      // Emit a sequence of code which compares the high and low parts once
      // each, then uses a conditional move to handle the overflow case.  For
      // example, a setlt for long would generate code like this:
      //
      // AL = lo(op1) < lo(op2)   // Signedness depends on operands
      // BL = hi(op1) < hi(op2)   // Always unsigned comparison
      // dest = hi(op1) == hi(op2) ? BL : AL;
      //

      // FIXME: This would be much better if we had hierarchical register
      // classes!  Until then, hardcode registers so that we can deal with their
      // aliases (because we don't have conditional byte moves).
      //
      BuildMI(*MBB, IP, X86::CMP32rr, 2).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, SetCCOpcodeTab[0][OpNum], 0, X86::AL);
      BuildMI(*MBB, IP, X86::CMP32rr, 2).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, SetCCOpcodeTab[CompTy->isSigned()][OpNum], 0, X86::BL);
      BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, X86::BH);
      BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, X86::AH);
      BuildMI(*MBB, IP, X86::CMOVE16rr, 2, X86::BX).addReg(X86::BX)
                                                   .addReg(X86::AX);
      // NOTE: visitSetCondInst knows that the value is dumped into the BL
      // register at this point for long values...
      return OpNum;
    }
  }
  return OpNum;
}

/// SetCC instructions - Here we just emit boilerplate code to set a byte-sized
/// register, then move it to wherever the result should be.
///
void X86ISel::visitSetCondInst(SetCondInst &I) {
  if (canFoldSetCCIntoBranchOrSelect(&I))
    return;  // Fold this into a branch or select.

  unsigned DestReg = getReg(I);
  MachineBasicBlock::iterator MII = BB->end();
  emitSetCCOperation(BB, MII, I.getOperand(0), I.getOperand(1), I.getOpcode(),
                     DestReg);
}

/// emitSetCCOperation - Common code shared between visitSetCondInst and
/// constant expression support.
///
void X86ISel::emitSetCCOperation(MachineBasicBlock *MBB,
                                 MachineBasicBlock::iterator IP,
                                 Value *Op0, Value *Op1, unsigned Opcode,
                                 unsigned TargetReg) {
  unsigned OpNum = getSetCCNumber(Opcode);
  OpNum = EmitComparison(OpNum, Op0, Op1, MBB, IP);

  const Type *CompTy = Op0->getType();
  unsigned CompClass = getClassB(CompTy);
  bool isSigned = CompTy->isSigned() && CompClass != cFP;

  if (CompClass != cLong || OpNum < 2) {
    // Handle normal comparisons with a setcc instruction...
    BuildMI(*MBB, IP, SetCCOpcodeTab[isSigned][OpNum], 0, TargetReg);
  } else {
    // Handle long comparisons by copying the value which is already in BL into
    // the register we want...
    BuildMI(*MBB, IP, X86::MOV8rr, 1, TargetReg).addReg(X86::BL);
  }
}

void X86ISel::visitSelectInst(SelectInst &SI) {
  unsigned DestReg = getReg(SI);
  MachineBasicBlock::iterator MII = BB->end();
  emitSelectOperation(BB, MII, SI.getCondition(), SI.getTrueValue(),
                      SI.getFalseValue(), DestReg);
}

/// emitSelect - Common code shared between visitSelectInst and the constant
/// expression support.
void X86ISel::emitSelectOperation(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator IP,
                                  Value *Cond, Value *TrueVal, Value *FalseVal,
                                  unsigned DestReg) {
  unsigned SelectClass = getClassB(TrueVal->getType());

  // We don't support 8-bit conditional moves.  If we have incoming constants,
  // transform them into 16-bit constants to avoid having a run-time conversion.
  if (SelectClass == cByte) {
    if (Constant *T = dyn_cast<Constant>(TrueVal))
      TrueVal = ConstantExpr::getCast(T, Type::ShortTy);
    if (Constant *F = dyn_cast<Constant>(FalseVal))
      FalseVal = ConstantExpr::getCast(F, Type::ShortTy);
  }

  unsigned TrueReg  = getReg(TrueVal, MBB, IP);
  unsigned FalseReg = getReg(FalseVal, MBB, IP);
  if (TrueReg == FalseReg) {
    static const unsigned Opcode[] = {
      X86::MOV8rr, X86::MOV16rr, X86::MOV32rr, X86::FpMOV, X86::MOV32rr
    };
    BuildMI(*MBB, IP, Opcode[SelectClass], 1, DestReg).addReg(TrueReg);
    if (SelectClass == cLong)
      BuildMI(*MBB, IP, X86::MOV32rr, 1, DestReg+1).addReg(TrueReg+1);
    return;
  }

  unsigned Opcode;
  if (SetCondInst *SCI = canFoldSetCCIntoBranchOrSelect(Cond)) {
    // We successfully folded the setcc into the select instruction.

    unsigned OpNum = getSetCCNumber(SCI->getOpcode());
    OpNum = EmitComparison(OpNum, SCI->getOperand(0), SCI->getOperand(1), MBB,
                           IP);

    const Type *CompTy = SCI->getOperand(0)->getType();
    bool isSigned = CompTy->isSigned() && getClassB(CompTy) != cFP;

    // LLVM  -> X86 signed  X86 unsigned
    // -----    ----------  ------------
    // seteq -> cmovNE      cmovNE
    // setne -> cmovE       cmovE
    // setlt -> cmovGE      cmovAE
    // setge -> cmovL       cmovB
    // setgt -> cmovLE      cmovBE
    // setle -> cmovG       cmovA
    // ----
    //          cmovNS              // Used by comparison with 0 optimization
    //          cmovS

    switch (SelectClass) {
    default: assert(0 && "Unknown value class!");
    case cFP: {
      // Annoyingly, we don't have a full set of floating point conditional
      // moves.  :(
      static const unsigned OpcodeTab[2][8] = {
        { X86::FCMOVNE, X86::FCMOVE, X86::FCMOVAE, X86::FCMOVB,
          X86::FCMOVBE, X86::FCMOVA, 0, 0 },
        { X86::FCMOVNE, X86::FCMOVE, 0, 0, 0, 0, 0, 0 },
      };
      Opcode = OpcodeTab[isSigned][OpNum];

      // If opcode == 0, we hit a case that we don't support.  Output a setcc
      // and compare the result against zero.
      if (Opcode == 0) {
        unsigned CompClass = getClassB(CompTy);
        unsigned CondReg;
        if (CompClass != cLong || OpNum < 2) {
          CondReg = makeAnotherReg(Type::BoolTy);
          // Handle normal comparisons with a setcc instruction...
          BuildMI(*MBB, IP, SetCCOpcodeTab[isSigned][OpNum], 0, CondReg);
        } else {
          // Long comparisons end up in the BL register.
          CondReg = X86::BL;
        }

        BuildMI(*MBB, IP, X86::TEST8rr, 2).addReg(CondReg).addReg(CondReg);
        Opcode = X86::FCMOVE;
      }
      break;
    }
    case cByte:
    case cShort: {
      static const unsigned OpcodeTab[2][8] = {
        { X86::CMOVNE16rr, X86::CMOVE16rr, X86::CMOVAE16rr, X86::CMOVB16rr,
          X86::CMOVBE16rr, X86::CMOVA16rr, 0, 0 },
        { X86::CMOVNE16rr, X86::CMOVE16rr, X86::CMOVGE16rr, X86::CMOVL16rr,
          X86::CMOVLE16rr, X86::CMOVG16rr, X86::CMOVNS16rr, X86::CMOVS16rr },
      };
      Opcode = OpcodeTab[isSigned][OpNum];
      break;
    }
    case cInt:
    case cLong: {
      static const unsigned OpcodeTab[2][8] = {
        { X86::CMOVNE32rr, X86::CMOVE32rr, X86::CMOVAE32rr, X86::CMOVB32rr,
          X86::CMOVBE32rr, X86::CMOVA32rr, 0, 0 },
        { X86::CMOVNE32rr, X86::CMOVE32rr, X86::CMOVGE32rr, X86::CMOVL32rr,
          X86::CMOVLE32rr, X86::CMOVG32rr, X86::CMOVNS32rr, X86::CMOVS32rr },
      };
      Opcode = OpcodeTab[isSigned][OpNum];
      break;
    }
    }
  } else {
    // Get the value being branched on, and use it to set the condition codes.
    unsigned CondReg = getReg(Cond, MBB, IP);
    BuildMI(*MBB, IP, X86::TEST8rr, 2).addReg(CondReg).addReg(CondReg);
    switch (SelectClass) {
    default: assert(0 && "Unknown value class!");
    case cFP:    Opcode = X86::FCMOVE; break;
    case cByte:
    case cShort: Opcode = X86::CMOVE16rr; break;
    case cInt:
    case cLong:  Opcode = X86::CMOVE32rr; break;
    }
  }

  unsigned RealDestReg = DestReg;


  // Annoyingly enough, X86 doesn't HAVE 8-bit conditional moves.  Because of
  // this, we have to promote the incoming values to 16 bits, perform a 16-bit
  // cmove, then truncate the result.
  if (SelectClass == cByte) {
    DestReg = makeAnotherReg(Type::ShortTy);
    if (getClassB(TrueVal->getType()) == cByte) {
      // Promote the true value, by storing it into AL, and reading from AX.
      BuildMI(*MBB, IP, X86::MOV8rr, 1, X86::AL).addReg(TrueReg);
      BuildMI(*MBB, IP, X86::MOV8ri, 1, X86::AH).addImm(0);
      TrueReg = makeAnotherReg(Type::ShortTy);
      BuildMI(*MBB, IP, X86::MOV16rr, 1, TrueReg).addReg(X86::AX);
    }
    if (getClassB(FalseVal->getType()) == cByte) {
      // Promote the true value, by storing it into CL, and reading from CX.
      BuildMI(*MBB, IP, X86::MOV8rr, 1, X86::CL).addReg(FalseReg);
      BuildMI(*MBB, IP, X86::MOV8ri, 1, X86::CH).addImm(0);
      FalseReg = makeAnotherReg(Type::ShortTy);
      BuildMI(*MBB, IP, X86::MOV16rr, 1, FalseReg).addReg(X86::CX);
    }
  }

  BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(TrueReg).addReg(FalseReg);

  switch (SelectClass) {
  case cByte:
    // We did the computation with 16-bit registers.  Truncate back to our
    // result by copying into AX then copying out AL.
    BuildMI(*MBB, IP, X86::MOV16rr, 1, X86::AX).addReg(DestReg);
    BuildMI(*MBB, IP, X86::MOV8rr, 1, RealDestReg).addReg(X86::AL);
    break;
  case cLong:
    // Move the upper half of the value as well.
    BuildMI(*MBB, IP, Opcode, 2,DestReg+1).addReg(TrueReg+1).addReg(FalseReg+1);
    break;
  }
}



/// promote32 - Emit instructions to turn a narrow operand into a 32-bit-wide
/// operand, in the specified target register.
///
void X86ISel::promote32(unsigned targetReg, const ValueRecord &VR) {
  bool isUnsigned = VR.Ty->isUnsigned() || VR.Ty == Type::BoolTy;

  Value *Val = VR.Val;
  const Type *Ty = VR.Ty;
  if (Val) {
    if (Constant *C = dyn_cast<Constant>(Val)) {
      Val = ConstantExpr::getCast(C, Type::IntTy);
      Ty = Type::IntTy;
    }

    // If this is a simple constant, just emit a MOVri directly to avoid the
    // copy.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      int TheVal = CI->getRawValue() & 0xFFFFFFFF;
      BuildMI(BB, X86::MOV32ri, 1, targetReg).addImm(TheVal);
      return;
    }
  }

  // Make sure we have the register number for this value...
  unsigned Reg = Val ? getReg(Val) : VR.Reg;

  switch (getClassB(Ty)) {
  case cByte:
    // Extend value into target register (8->32)
    if (isUnsigned)
      BuildMI(BB, X86::MOVZX32rr8, 1, targetReg).addReg(Reg);
    else
      BuildMI(BB, X86::MOVSX32rr8, 1, targetReg).addReg(Reg);
    break;
  case cShort:
    // Extend value into target register (16->32)
    if (isUnsigned)
      BuildMI(BB, X86::MOVZX32rr16, 1, targetReg).addReg(Reg);
    else
      BuildMI(BB, X86::MOVSX32rr16, 1, targetReg).addReg(Reg);
    break;
  case cInt:
    // Move value into target register (32->32)
    BuildMI(BB, X86::MOV32rr, 1, targetReg).addReg(Reg);
    break;
  default:
    assert(0 && "Unpromotable operand class in promote32");
  }
}

/// 'ret' instruction - Here we are interested in meeting the x86 ABI.  As such,
/// we have the following possibilities:
///
///   ret void: No return value, simply emit a 'ret' instruction
///   ret sbyte, ubyte : Extend value into EAX and return
///   ret short, ushort: Extend value into EAX and return
///   ret int, uint    : Move value into EAX and return
///   ret pointer      : Move value into EAX and return
///   ret long, ulong  : Move value into EAX/EDX and return
///   ret float/double : Top of FP stack
///
void X86ISel::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() == 0) {
    BuildMI(BB, X86::RET, 0); // Just emit a 'ret' instruction
    return;
  }

  Value *RetVal = I.getOperand(0);
  switch (getClassB(RetVal->getType())) {
  case cByte:   // integral return values: extend or move into EAX and return
  case cShort:
  case cInt:
    promote32(X86::EAX, ValueRecord(RetVal));
    break;
  case cFP: {                  // Floats & Doubles: Return in ST(0)
    unsigned RetReg = getReg(RetVal);
    BuildMI(BB, X86::FpSETRESULT, 1).addReg(RetReg);
    break;
  }
  case cLong: {
    unsigned RetReg = getReg(RetVal);
    BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(RetReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDX).addReg(RetReg+1);
    break;
  }
  default:
    visitInstruction(I);
  }
  // Emit a 'ret' instruction
  BuildMI(BB, X86::RET, 0);
}

// getBlockAfter - Return the basic block which occurs lexically after the
// specified one.
static inline BasicBlock *getBlockAfter(BasicBlock *BB) {
  Function::iterator I = BB; ++I;  // Get iterator to next block
  return I != BB->getParent()->end() ? &*I : 0;
}

/// visitBranchInst - Handle conditional and unconditional branches here.  Note
/// that since code layout is frozen at this point, that if we are trying to
/// jump to a block that is the immediate successor of the current block, we can
/// just make a fall-through (but we don't currently).
///
void X86ISel::visitBranchInst(BranchInst &BI) {
  // Update machine-CFG edges
  BB->addSuccessor (MBBMap[BI.getSuccessor(0)]);
  if (BI.isConditional())
    BB->addSuccessor (MBBMap[BI.getSuccessor(1)]);

  BasicBlock *NextBB = getBlockAfter(BI.getParent());  // BB after current one

  if (!BI.isConditional()) {  // Unconditional branch?
    if (BI.getSuccessor(0) != NextBB)
      BuildMI(BB, X86::JMP, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
    return;
  }

  // See if we can fold the setcc into the branch itself...
  SetCondInst *SCI = canFoldSetCCIntoBranchOrSelect(BI.getCondition());
  if (SCI == 0) {
    // Nope, cannot fold setcc into this branch.  Emit a branch on a condition
    // computed some other way...
    unsigned condReg = getReg(BI.getCondition());
    BuildMI(BB, X86::TEST8rr, 2).addReg(condReg).addReg(condReg);
    if (BI.getSuccessor(1) == NextBB) {
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, X86::JNE, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
    } else {
      BuildMI(BB, X86::JE, 1).addMBB(MBBMap[BI.getSuccessor(1)]);

      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, X86::JMP, 1).addMBB(MBBMap[BI.getSuccessor(0)]);
    }
    return;
  }

  unsigned OpNum = getSetCCNumber(SCI->getOpcode());
  MachineBasicBlock::iterator MII = BB->end();
  OpNum = EmitComparison(OpNum, SCI->getOperand(0), SCI->getOperand(1), BB,MII);

  const Type *CompTy = SCI->getOperand(0)->getType();
  bool isSigned = CompTy->isSigned() && getClassB(CompTy) != cFP;


  // LLVM  -> X86 signed  X86 unsigned
  // -----    ----------  ------------
  // seteq -> je          je
  // setne -> jne         jne
  // setlt -> jl          jb
  // setge -> jge         jae
  // setgt -> jg          ja
  // setle -> jle         jbe
  // ----
  //          js                  // Used by comparison with 0 optimization
  //          jns

  static const unsigned OpcodeTab[2][8] = {
    { X86::JE, X86::JNE, X86::JB, X86::JAE, X86::JA, X86::JBE, 0, 0 },
    { X86::JE, X86::JNE, X86::JL, X86::JGE, X86::JG, X86::JLE,
      X86::JS, X86::JNS },
  };

  if (BI.getSuccessor(0) != NextBB) {
    BuildMI(BB, OpcodeTab[isSigned][OpNum], 1)
      .addMBB(MBBMap[BI.getSuccessor(0)]);
    if (BI.getSuccessor(1) != NextBB)
      BuildMI(BB, X86::JMP, 1).addMBB(MBBMap[BI.getSuccessor(1)]);
  } else {
    // Change to the inverse condition...
    if (BI.getSuccessor(1) != NextBB) {
      OpNum ^= 1;
      BuildMI(BB, OpcodeTab[isSigned][OpNum], 1)
        .addMBB(MBBMap[BI.getSuccessor(1)]);
    }
  }
}


/// doCall - This emits an abstract call instruction, setting up the arguments
/// and the return value as appropriate.  For the actual function call itself,
/// it inserts the specified CallMI instruction into the stream.
///
void X86ISel::doCall(const ValueRecord &Ret, MachineInstr *CallMI,
                     const std::vector<ValueRecord> &Args) {
  // Count how many bytes are to be pushed on the stack...
  unsigned NumBytes = 0;

  if (!Args.empty()) {
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
      switch (getClassB(Args[i].Ty)) {
      case cByte: case cShort: case cInt:
        NumBytes += 4; break;
      case cLong:
        NumBytes += 8; break;
      case cFP:
        NumBytes += Args[i].Ty == Type::FloatTy ? 4 : 8;
        break;
      default: assert(0 && "Unknown class!");
      }

    // Adjust the stack pointer for the new arguments...
    BuildMI(BB, X86::ADJCALLSTACKDOWN, 1).addImm(NumBytes);

    // Arguments go on the stack in reverse order, as specified by the ABI.
    unsigned ArgOffset = 0;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      unsigned ArgReg;
      switch (getClassB(Args[i].Ty)) {
      case cByte:
        if (Args[i].Val && isa<ConstantBool>(Args[i].Val)) {
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5), X86::ESP, ArgOffset)
            .addImm(Args[i].Val == ConstantBool::True);
          break;
        }
        // FALL THROUGH
      case cShort:
        if (Args[i].Val && isa<ConstantInt>(Args[i].Val)) {
          // Zero/Sign extend constant, then stuff into memory.
          ConstantInt *Val = cast<ConstantInt>(Args[i].Val);
          Val = cast<ConstantInt>(ConstantExpr::getCast(Val, Type::IntTy));
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5), X86::ESP, ArgOffset)
            .addImm(Val->getRawValue() & 0xFFFFFFFF);
        } else {
          // Promote arg to 32 bits wide into a temporary register...
          ArgReg = makeAnotherReg(Type::UIntTy);
          promote32(ArgReg, Args[i]);
          addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                       X86::ESP, ArgOffset).addReg(ArgReg);
        }
        break;
      case cInt:
        if (Args[i].Val && isa<ConstantInt>(Args[i].Val)) {
          unsigned Val = cast<ConstantInt>(Args[i].Val)->getRawValue();
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                       X86::ESP, ArgOffset).addImm(Val);
        } else if (Args[i].Val && isa<ConstantPointerNull>(Args[i].Val)) {
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                       X86::ESP, ArgOffset).addImm(0);
        } else if (Args[i].Val && isa<GlobalValue>(Args[i].Val)) {
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5), X86::ESP, ArgOffset)
            .addGlobalAddress(cast<GlobalValue>(Args[i].Val));
        } else {
          ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
          addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                       X86::ESP, ArgOffset).addReg(ArgReg);
        }
        break;
      case cLong:
        if (Args[i].Val && isa<ConstantInt>(Args[i].Val)) {
          uint64_t Val = cast<ConstantInt>(Args[i].Val)->getRawValue();
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                       X86::ESP, ArgOffset).addImm(Val & ~0U);
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                       X86::ESP, ArgOffset+4).addImm(Val >> 32ULL);
        } else {
          ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
          addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                       X86::ESP, ArgOffset).addReg(ArgReg);
          addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                       X86::ESP, ArgOffset+4).addReg(ArgReg+1);
        }
        ArgOffset += 4;        // 8 byte entry, not 4.
        break;

      case cFP:
        if (ConstantFP *CFP = dyn_cast_or_null<ConstantFP>(Args[i].Val)) {
          // Store constant FP values with integer instructions to avoid having
          // to load the constants from the constant pool then do a store.
          if (CFP->getType() == Type::FloatTy) {
            union {
              unsigned I;
              float    F;
            } V;
            V.F = CFP->getValue();
            addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                         X86::ESP, ArgOffset).addImm(V.I);
          } else {
            union {
              uint64_t I;
              double   F;
            } V;
            V.F = CFP->getValue();
            addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                          X86::ESP, ArgOffset).addImm((unsigned)V.I);
            addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                         X86::ESP, ArgOffset+4).addImm(unsigned(V.I >> 32));
            ArgOffset += 4;       // 8 byte entry, not 4.
          }
        } else {
          ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
          if (Args[i].Ty == Type::FloatTy) {
            addRegOffset(BuildMI(BB, X86::FST32m, 5),
                         X86::ESP, ArgOffset).addReg(ArgReg);
          } else {
            assert(Args[i].Ty == Type::DoubleTy && "Unknown FP type!");
            addRegOffset(BuildMI(BB, X86::FST64m, 5),
                         X86::ESP, ArgOffset).addReg(ArgReg);
            ArgOffset += 4;       // 8 byte entry, not 4.
          }
        }
        break;

      default: assert(0 && "Unknown class!");
      }
      ArgOffset += 4;
    }
  } else {
    BuildMI(BB, X86::ADJCALLSTACKDOWN, 1).addImm(0);
  }

  BB->push_back(CallMI);

  BuildMI(BB, X86::ADJCALLSTACKUP, 2).addImm(NumBytes).addImm(0);

  // If there is a return value, scavenge the result from the location the call
  // leaves it in...
  //
  if (Ret.Ty != Type::VoidTy) {
    unsigned DestClass = getClassB(Ret.Ty);
    switch (DestClass) {
    case cByte:
    case cShort:
    case cInt: {
      // Integral results are in %eax, or the appropriate portion
      // thereof.
      static const unsigned regRegMove[] = {
        X86::MOV8rr, X86::MOV16rr, X86::MOV32rr
      };
      static const unsigned AReg[] = { X86::AL, X86::AX, X86::EAX };
      BuildMI(BB, regRegMove[DestClass], 1, Ret.Reg).addReg(AReg[DestClass]);
      break;
    }
    case cFP:     // Floating-point return values live in %ST(0)
      BuildMI(BB, X86::FpGETRESULT, 1, Ret.Reg);
      break;
    case cLong:   // Long values are left in EDX:EAX
      BuildMI(BB, X86::MOV32rr, 1, Ret.Reg).addReg(X86::EAX);
      BuildMI(BB, X86::MOV32rr, 1, Ret.Reg+1).addReg(X86::EDX);
      break;
    default: assert(0 && "Unknown class!");
    }
  }
}


/// visitCallInst - Push args on stack and do a procedure call instruction.
void X86ISel::visitCallInst(CallInst &CI) {
  MachineInstr *TheCall;
  if (Function *F = CI.getCalledFunction()) {
    // Is it an intrinsic function call?
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {
      visitIntrinsicCall(ID, CI);   // Special intrinsics are not handled here
      return;
    } else if (F->getName() == "fabs" || F->getName() == "fabsf") {
      if (CI.getNumOperands() == 2 &&   // Basic sanity checks.
          CI.getOperand(1)->getType()->isFloatingPoint() &&
          CI.getType() == CI.getOperand(1)->getType()) {
        unsigned op1Reg = getReg(CI.getOperand(1));
        unsigned DestReg = getReg(CI);
        BuildMI(BB, X86::FABS, 1, DestReg).addReg(op1Reg);
        return;
      }
    } else if (F->getName() == "sin" && UnsafeFPMath || F->getName() == "sinf") {
      if (CI.getNumOperands() == 2 &&   // Basic sanity checks.
          CI.getOperand(1)->getType()->isFloatingPoint() &&
          CI.getType() == CI.getOperand(1)->getType()) {
        unsigned op1Reg = getReg(CI.getOperand(1));
        unsigned DestReg = getReg(CI);
        BuildMI(BB, X86::FSIN, 1, DestReg).addReg(op1Reg);
        return;
      }
    }
    else if (F->getName() == "cos" && UnsafeFPMath || F->getName() == "cosf") {
      if (CI.getNumOperands() == 2 &&   // Basic sanity checks.
          CI.getOperand(1)->getType()->isFloatingPoint() &&
          CI.getType() == CI.getOperand(1)->getType()) {
        unsigned op1Reg = getReg(CI.getOperand(1));
        unsigned DestReg = getReg(CI);
        BuildMI(BB, X86::FCOS, 1, DestReg).addReg(op1Reg);
        return;
      }
    }

    // Emit a CALL instruction with PC-relative displacement.
    TheCall = BuildMI(X86::CALLpcrel32, 1).addGlobalAddress(F, true);
  } else {  // Emit an indirect call...
    unsigned Reg = getReg(CI.getCalledValue());
    TheCall = BuildMI(X86::CALL32r, 1).addReg(Reg);
  }

  std::vector<ValueRecord> Args;
  for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i)
    Args.push_back(ValueRecord(CI.getOperand(i)));

  unsigned DestReg = CI.getType() != Type::VoidTy ? getReg(CI) : 0;
  doCall(ValueRecord(DestReg, CI.getType()), TheCall, Args);
}

/// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
/// function, lowering any calls to unknown intrinsic functions into the
/// equivalent LLVM code.
///
void X86ISel::LowerUnknownIntrinsicFunctionCalls(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
          case Intrinsic::returnaddress:
          case Intrinsic::frameaddress:
          case Intrinsic::memcpy:
          case Intrinsic::memset:
          case Intrinsic::isunordered:
          case Intrinsic::sqrt:
          case Intrinsic::readport:
          case Intrinsic::writeport:
            // We directly implement these intrinsics
            break;
          case Intrinsic::readio: {
            // On X86, memory operations are in-order.  Lower this intrinsic
            // into a volatile load.
            LoadInst * LI = new LoadInst(CI->getOperand(1), "", true, CI);
            CI->replaceAllUsesWith(LI);
            BB->getInstList().erase(CI);
            break;
          }
          case Intrinsic::writeio: {
            // On X86, memory operations are in-order.  Lower this intrinsic
            // into a volatile store.
            StoreInst *LI = new StoreInst(CI->getOperand(1),
                                          CI->getOperand(2), true, CI);
            CI->replaceAllUsesWith(LI);
            BB->getInstList().erase(CI);
            break;
          }
          default:
            // All other intrinsic calls we must lower.
            Instruction *Before = CI->getPrev();
            TM.getIntrinsicLowering().LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before; ++I;
            } else {
              I = BB->begin();
            }
          }
}

void X86ISel::visitIntrinsicCall(Intrinsic::ID ID, CallInst &CI) {
  unsigned TmpReg1, TmpReg2;
  switch (ID) {
  case Intrinsic::vastart:
    // Get the address of the first vararg value...
    TmpReg1 = getReg(CI);
    addFrameReference(BuildMI(BB, X86::LEA32r, 5, TmpReg1), VarArgsFrameIndex);
    return;

  case Intrinsic::vacopy:
    TmpReg1 = getReg(CI);
    TmpReg2 = getReg(CI.getOperand(1));
    BuildMI(BB, X86::MOV32rr, 1, TmpReg1).addReg(TmpReg2);
    return;
  case Intrinsic::vaend: return;   // Noop on X86

  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    TmpReg1 = getReg(CI);
    if (cast<Constant>(CI.getOperand(1))->isNullValue()) {
      if (ReturnAddressIndex == 0) {
        // Set up a frame object for the return address.
        ReturnAddressIndex = F->getFrameInfo()->CreateFixedObject(4, -4);
      }

      if (ID == Intrinsic::returnaddress) {
        // Just load the return address
        addFrameReference(BuildMI(BB, X86::MOV32rm, 4, TmpReg1),
                          ReturnAddressIndex);
      } else {
        addFrameReference(BuildMI(BB, X86::LEA32r, 4, TmpReg1),
                          ReturnAddressIndex, -4);
      }
    } else {
      // Values other than zero are not implemented yet.
      BuildMI(BB, X86::MOV32ri, 1, TmpReg1).addImm(0);
    }
    return;

  case Intrinsic::isunordered:
    TmpReg1 = getReg(CI.getOperand(1));
    TmpReg2 = getReg(CI.getOperand(2));
    emitUCOMr(BB, BB->end(), TmpReg2, TmpReg1);
    TmpReg2 = getReg(CI);
    BuildMI(BB, X86::SETPr, 0, TmpReg2);
    return;

  case Intrinsic::sqrt:
    TmpReg1 = getReg(CI.getOperand(1));
    TmpReg2 = getReg(CI);
    BuildMI(BB, X86::FSQRT, 1, TmpReg2).addReg(TmpReg1);
    return;

  case Intrinsic::memcpy: {
    assert(CI.getNumOperands() == 5 && "Illegal llvm.memcpy call!");
    unsigned Align = 1;
    if (ConstantInt *AlignC = dyn_cast<ConstantInt>(CI.getOperand(4))) {
      Align = AlignC->getRawValue();
      if (Align == 0) Align = 1;
    }

    // Turn the byte code into # iterations
    unsigned CountReg;
    unsigned Opcode;
    switch (Align & 3) {
    case 2:   // WORD aligned
      if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
        CountReg = getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/2));
      } else {
        CountReg = makeAnotherReg(Type::IntTy);
        unsigned ByteReg = getReg(CI.getOperand(3));
        BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(1);
      }
      Opcode = X86::REP_MOVSW;
      break;
    case 0:   // DWORD aligned
      if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
        CountReg = getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/4));
      } else {
        CountReg = makeAnotherReg(Type::IntTy);
        unsigned ByteReg = getReg(CI.getOperand(3));
        BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(2);
      }
      Opcode = X86::REP_MOVSD;
      break;
    default:  // BYTE aligned
      CountReg = getReg(CI.getOperand(3));
      Opcode = X86::REP_MOVSB;
      break;
    }

    // No matter what the alignment is, we put the source in ESI, the
    // destination in EDI, and the count in ECX.
    TmpReg1 = getReg(CI.getOperand(1));
    TmpReg2 = getReg(CI.getOperand(2));
    BuildMI(BB, X86::MOV32rr, 1, X86::ECX).addReg(CountReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDI).addReg(TmpReg1);
    BuildMI(BB, X86::MOV32rr, 1, X86::ESI).addReg(TmpReg2);
    BuildMI(BB, Opcode, 0);
    return;
  }
  case Intrinsic::memset: {
    assert(CI.getNumOperands() == 5 && "Illegal llvm.memset call!");
    unsigned Align = 1;
    if (ConstantInt *AlignC = dyn_cast<ConstantInt>(CI.getOperand(4))) {
      Align = AlignC->getRawValue();
      if (Align == 0) Align = 1;
    }

    // Turn the byte code into # iterations
    unsigned CountReg;
    unsigned Opcode;
    if (ConstantInt *ValC = dyn_cast<ConstantInt>(CI.getOperand(2))) {
      unsigned Val = ValC->getRawValue() & 255;

      // If the value is a constant, then we can potentially use larger copies.
      switch (Align & 3) {
      case 2:   // WORD aligned
        if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
          CountReg =getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/2));
        } else {
          CountReg = makeAnotherReg(Type::IntTy);
          unsigned ByteReg = getReg(CI.getOperand(3));
          BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(1);
        }
        BuildMI(BB, X86::MOV16ri, 1, X86::AX).addImm((Val << 8) | Val);
        Opcode = X86::REP_STOSW;
        break;
      case 0:   // DWORD aligned
        if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
          CountReg =getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/4));
        } else {
          CountReg = makeAnotherReg(Type::IntTy);
          unsigned ByteReg = getReg(CI.getOperand(3));
          BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(2);
        }
        Val = (Val << 8) | Val;
        BuildMI(BB, X86::MOV32ri, 1, X86::EAX).addImm((Val << 16) | Val);
        Opcode = X86::REP_STOSD;
        break;
      default:  // BYTE aligned
        CountReg = getReg(CI.getOperand(3));
        BuildMI(BB, X86::MOV8ri, 1, X86::AL).addImm(Val);
        Opcode = X86::REP_STOSB;
        break;
      }
    } else {
      // If it's not a constant value we are storing, just fall back.  We could
      // try to be clever to form 16 bit and 32 bit values, but we don't yet.
      unsigned ValReg = getReg(CI.getOperand(2));
      BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(ValReg);
      CountReg = getReg(CI.getOperand(3));
      Opcode = X86::REP_STOSB;
    }

    // No matter what the alignment is, we put the source in ESI, the
    // destination in EDI, and the count in ECX.
    TmpReg1 = getReg(CI.getOperand(1));
    //TmpReg2 = getReg(CI.getOperand(2));
    BuildMI(BB, X86::MOV32rr, 1, X86::ECX).addReg(CountReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDI).addReg(TmpReg1);
    BuildMI(BB, Opcode, 0);
    return;
  }

  case Intrinsic::readport: {
    // First, determine that the size of the operand falls within the acceptable
    // range for this architecture.
    //
    if (getClassB(CI.getOperand(1)->getType()) != cShort) {
      std::cerr << "llvm.readport: Address size is not 16 bits\n";
      exit(1);
    }

    // Now, move the I/O port address into the DX register and use the IN
    // instruction to get the input data.
    //
    unsigned Class = getClass(CI.getCalledFunction()->getReturnType());
    unsigned DestReg = getReg(CI);

    // If the port is a single-byte constant, use the immediate form.
    if (ConstantInt *C = dyn_cast<ConstantInt>(CI.getOperand(1)))
      if ((C->getRawValue() & 255) == C->getRawValue()) {
        switch (Class) {
        case cByte:
          BuildMI(BB, X86::IN8ri, 1).addImm((unsigned char)C->getRawValue());
          BuildMI(BB, X86::MOV8rr, 1, DestReg).addReg(X86::AL);
          return;
        case cShort:
          BuildMI(BB, X86::IN16ri, 1).addImm((unsigned char)C->getRawValue());
          BuildMI(BB, X86::MOV8rr, 1, DestReg).addReg(X86::AX);
          return;
        case cInt:
          BuildMI(BB, X86::IN32ri, 1).addImm((unsigned char)C->getRawValue());
          BuildMI(BB, X86::MOV8rr, 1, DestReg).addReg(X86::EAX);
          return;
        }
      }

    unsigned Reg = getReg(CI.getOperand(1));
    BuildMI(BB, X86::MOV16rr, 1, X86::DX).addReg(Reg);
    switch (Class) {
    case cByte:
      BuildMI(BB, X86::IN8rr, 0);
      BuildMI(BB, X86::MOV8rr, 1, DestReg).addReg(X86::AL);
      break;
    case cShort:
      BuildMI(BB, X86::IN16rr, 0);
      BuildMI(BB, X86::MOV16rr, 1, DestReg).addReg(X86::AX);
      break;
    case cInt:
      BuildMI(BB, X86::IN32rr, 0);
      BuildMI(BB, X86::MOV32rr, 1, DestReg).addReg(X86::EAX);
      break;
    default:
      std::cerr << "Cannot do input on this data type";
      exit (1);
    }
    return;
  }

  case Intrinsic::writeport: {
    // First, determine that the size of the operand falls within the
    // acceptable range for this architecture.
    if (getClass(CI.getOperand(2)->getType()) != cShort) {
      std::cerr << "llvm.writeport: Address size is not 16 bits\n";
      exit(1);
    }

    unsigned Class = getClassB(CI.getOperand(1)->getType());
    unsigned ValReg = getReg(CI.getOperand(1));
    switch (Class) {
    case cByte:
      BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(ValReg);
      break;
    case cShort:
      BuildMI(BB, X86::MOV16rr, 1, X86::AX).addReg(ValReg);
      break;
    case cInt:
      BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(ValReg);
      break;
    default:
      std::cerr << "llvm.writeport: invalid data type for X86 target";
      exit(1);
    }


    // If the port is a single-byte constant, use the immediate form.
    if (ConstantInt *C = dyn_cast<ConstantInt>(CI.getOperand(2)))
      if ((C->getRawValue() & 255) == C->getRawValue()) {
        static const unsigned O[] = { X86::OUT8ir, X86::OUT16ir, X86::OUT32ir };
        BuildMI(BB, O[Class], 1).addImm((unsigned char)C->getRawValue());
        return;
      }

    // Otherwise, move the I/O port address into the DX register and the value
    // to write into the AL/AX/EAX register.
    static const unsigned Opc[] = { X86::OUT8rr, X86::OUT16rr, X86::OUT32rr };
    unsigned Reg = getReg(CI.getOperand(2));
    BuildMI(BB, X86::MOV16rr, 1, X86::DX).addReg(Reg);
    BuildMI(BB, Opc[Class], 0);
    return;
  }

  default: assert(0 && "Error: unknown intrinsics should have been lowered!");
  }
}

static bool isSafeToFoldLoadIntoInstruction(LoadInst &LI, Instruction &User) {
  if (LI.getParent() != User.getParent())
    return false;
  BasicBlock::iterator It = &LI;
  // Check all of the instructions between the load and the user.  We should
  // really use alias analysis here, but for now we just do something simple.
  for (++It; It != BasicBlock::iterator(&User); ++It) {
    switch (It->getOpcode()) {
    case Instruction::Free:
    case Instruction::Store:
    case Instruction::Call:
    case Instruction::Invoke:
      return false;
    case Instruction::Load:
      if (cast<LoadInst>(It)->isVolatile() && LI.isVolatile())
        return false;
      break;
    }
  }
  return true;
}

/// visitSimpleBinary - Implement simple binary operators for integral types...
/// OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for Or, 4 for
/// Xor.
///
void X86ISel::visitSimpleBinary(BinaryOperator &B, unsigned OperatorClass) {
  unsigned DestReg = getReg(B);
  MachineBasicBlock::iterator MI = BB->end();
  Value *Op0 = B.getOperand(0), *Op1 = B.getOperand(1);
  unsigned Class = getClassB(B.getType());

  // If this is AND X, C, and it is only used by a setcc instruction, it will
  // be folded.  There is no need to emit this instruction.
  if (B.hasOneUse() && OperatorClass == 2 && isa<ConstantInt>(Op1))
    if (Class == cByte || Class == cShort || Class == cInt) {
      Instruction *Use = cast<Instruction>(B.use_back());
      if (isa<SetCondInst>(Use) &&
          Use->getOperand(1) == Constant::getNullValue(B.getType())) {
        switch (getSetCCNumber(Use->getOpcode())) {
        case 0:
        case 1:
          return;
        default:
          if (B.getType()->isSigned()) return;
        }
      }
    }

  // Special case: op Reg, load [mem]
  if (isa<LoadInst>(Op0) && !isa<LoadInst>(Op1) && Class != cLong &&
      Op0->hasOneUse() &&
      isSafeToFoldLoadIntoInstruction(*cast<LoadInst>(Op0), B))
    if (!B.swapOperands())
      std::swap(Op0, Op1);  // Make sure any loads are in the RHS.

  if (isa<LoadInst>(Op1) && Class != cLong && Op1->hasOneUse() &&
      isSafeToFoldLoadIntoInstruction(*cast<LoadInst>(Op1), B)) {

    unsigned Opcode;
    if (Class != cFP) {
      static const unsigned OpcodeTab[][3] = {
        // Arithmetic operators
        { X86::ADD8rm, X86::ADD16rm, X86::ADD32rm },  // ADD
        { X86::SUB8rm, X86::SUB16rm, X86::SUB32rm },  // SUB

        // Bitwise operators
        { X86::AND8rm, X86::AND16rm, X86::AND32rm },  // AND
        { X86:: OR8rm, X86:: OR16rm, X86:: OR32rm },  // OR
        { X86::XOR8rm, X86::XOR16rm, X86::XOR32rm },  // XOR
      };
      Opcode = OpcodeTab[OperatorClass][Class];
    } else {
      static const unsigned OpcodeTab[][2] = {
        { X86::FADD32m, X86::FADD64m },  // ADD
        { X86::FSUB32m, X86::FSUB64m },  // SUB
      };
      const Type *Ty = Op0->getType();
      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
      Opcode = OpcodeTab[OperatorClass][Ty == Type::DoubleTy];
    }

    unsigned Op0r = getReg(Op0);
    if (AllocaInst *AI =
        dyn_castFixedAlloca(cast<LoadInst>(Op1)->getOperand(0))) {
      unsigned FI = getFixedSizedAllocaFI(AI);
      addFrameReference(BuildMI(BB, Opcode, 5, DestReg).addReg(Op0r), FI);

    } else {
      X86AddressMode AM;
      getAddressingMode(cast<LoadInst>(Op1)->getOperand(0), AM);

      addFullAddress(BuildMI(BB, Opcode, 5, DestReg).addReg(Op0r), AM);
    }
    return;
  }

  // If this is a floating point subtract, check to see if we can fold the first
  // operand in.
  if (Class == cFP && OperatorClass == 1 &&
      isa<LoadInst>(Op0) &&
      isSafeToFoldLoadIntoInstruction(*cast<LoadInst>(Op0), B)) {
    const Type *Ty = Op0->getType();
    assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
    unsigned Opcode = Ty == Type::FloatTy ? X86::FSUBR32m : X86::FSUBR64m;

    unsigned Op1r = getReg(Op1);
    if (AllocaInst *AI =
        dyn_castFixedAlloca(cast<LoadInst>(Op0)->getOperand(0))) {
      unsigned FI = getFixedSizedAllocaFI(AI);
      addFrameReference(BuildMI(BB, Opcode, 5, DestReg).addReg(Op1r), FI);
    } else {
      X86AddressMode AM;
      getAddressingMode(cast<LoadInst>(Op0)->getOperand(0), AM);

      addFullAddress(BuildMI(BB, Opcode, 5, DestReg).addReg(Op1r), AM);
    }
    return;
  }

  emitSimpleBinaryOperation(BB, MI, Op0, Op1, OperatorClass, DestReg);
}


/// emitBinaryFPOperation - This method handles emission of floating point
/// Add (0), Sub (1), Mul (2), and Div (3) operations.
void X86ISel::emitBinaryFPOperation(MachineBasicBlock *BB,
                                    MachineBasicBlock::iterator IP,
                                    Value *Op0, Value *Op1,
                                    unsigned OperatorClass, unsigned DestReg) {
  // Special case: op Reg, <const fp>
  if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1))
    if (!Op1C->isExactlyValue(+0.0) && !Op1C->isExactlyValue(+1.0)) {
      // Create a constant pool entry for this constant.
      MachineConstantPool *CP = F->getConstantPool();
      unsigned CPI = CP->getConstantPoolIndex(Op1C);
      const Type *Ty = Op1->getType();

      static const unsigned OpcodeTab[][4] = {
        { X86::FADD32m, X86::FSUB32m, X86::FMUL32m, X86::FDIV32m },   // Float
        { X86::FADD64m, X86::FSUB64m, X86::FMUL64m, X86::FDIV64m },   // Double
      };

      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
      unsigned Opcode = OpcodeTab[Ty != Type::FloatTy][OperatorClass];
      unsigned Op0r = getReg(Op0, BB, IP);
      addConstantPoolReference(BuildMI(*BB, IP, Opcode, 5,
                                       DestReg).addReg(Op0r), CPI);
      return;
    }

  // Special case: R1 = op <const fp>, R2
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(Op0))
    if (CFP->isExactlyValue(-0.0) && OperatorClass == 1) {
      // -0.0 - X === -X
      unsigned op1Reg = getReg(Op1, BB, IP);
      BuildMI(*BB, IP, X86::FCHS, 1, DestReg).addReg(op1Reg);
      return;
    } else if (!CFP->isExactlyValue(+0.0) && !CFP->isExactlyValue(+1.0)) {
      // R1 = op CST, R2  -->  R1 = opr R2, CST

      // Create a constant pool entry for this constant.
      MachineConstantPool *CP = F->getConstantPool();
      unsigned CPI = CP->getConstantPoolIndex(CFP);
      const Type *Ty = CFP->getType();

      static const unsigned OpcodeTab[][4] = {
        { X86::FADD32m, X86::FSUBR32m, X86::FMUL32m, X86::FDIVR32m }, // Float
        { X86::FADD64m, X86::FSUBR64m, X86::FMUL64m, X86::FDIVR64m }, // Double
      };

      assert(Ty == Type::FloatTy||Ty == Type::DoubleTy && "Unknown FP type!");
      unsigned Opcode = OpcodeTab[Ty != Type::FloatTy][OperatorClass];
      unsigned Op1r = getReg(Op1, BB, IP);
      addConstantPoolReference(BuildMI(*BB, IP, Opcode, 5,
                                       DestReg).addReg(Op1r), CPI);
      return;
    }

  // General case.
  static const unsigned OpcodeTab[4] = {
    X86::FpADD, X86::FpSUB, X86::FpMUL, X86::FpDIV
  };

  unsigned Opcode = OpcodeTab[OperatorClass];
  unsigned Op0r = getReg(Op0, BB, IP);
  unsigned Op1r = getReg(Op1, BB, IP);
  BuildMI(*BB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
}

/// emitSimpleBinaryOperation - Implement simple binary operators for integral
/// types...  OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for
/// Or, 4 for Xor.
///
/// emitSimpleBinaryOperation - Common code shared between visitSimpleBinary
/// and constant expression support.
///
void X86ISel::emitSimpleBinaryOperation(MachineBasicBlock *MBB,
                                        MachineBasicBlock::iterator IP,
                                        Value *Op0, Value *Op1,
                                        unsigned OperatorClass,
                                        unsigned DestReg) {
  unsigned Class = getClassB(Op0->getType());

  if (Class == cFP) {
    assert(OperatorClass < 2 && "No logical ops for FP!");
    emitBinaryFPOperation(MBB, IP, Op0, Op1, OperatorClass, DestReg);
    return;
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0))
    if (OperatorClass == 1) {
      static unsigned const NEGTab[] = {
        X86::NEG8r, X86::NEG16r, X86::NEG32r, 0, X86::NEG32r
      };

      // sub 0, X -> neg X
      if (CI->isNullValue()) {
        unsigned op1Reg = getReg(Op1, MBB, IP);
        BuildMI(*MBB, IP, NEGTab[Class], 1, DestReg).addReg(op1Reg);

        if (Class == cLong) {
          // We just emitted: Dl = neg Sl
          // Now emit       : T  = addc Sh, 0
          //                : Dh = neg T
          unsigned T = makeAnotherReg(Type::IntTy);
          BuildMI(*MBB, IP, X86::ADC32ri, 2, T).addReg(op1Reg+1).addImm(0);
          BuildMI(*MBB, IP, X86::NEG32r, 1, DestReg+1).addReg(T);
        }
        return;
      } else if (Op1->hasOneUse() && Class != cLong) {
        // sub C, X -> tmp = neg X; DestReg = add tmp, C.  This is better
        // than copying C into a temporary register, because of register
        // pressure (tmp and destreg can share a register.
        static unsigned const ADDRITab[] = {
          X86::ADD8ri, X86::ADD16ri, X86::ADD32ri, 0, X86::ADD32ri
        };
        unsigned op1Reg = getReg(Op1, MBB, IP);
        unsigned Tmp = makeAnotherReg(Op0->getType());
        BuildMI(*MBB, IP, NEGTab[Class], 1, Tmp).addReg(op1Reg);
        BuildMI(*MBB, IP, ADDRITab[Class], 2,
                DestReg).addReg(Tmp).addImm(CI->getRawValue());
        return;
      }
    }

  // Special case: op Reg, <const int>
  if (ConstantInt *Op1C = dyn_cast<ConstantInt>(Op1)) {
    unsigned Op0r = getReg(Op0, MBB, IP);

    // xor X, -1 -> not X
    if (OperatorClass == 4 && Op1C->isAllOnesValue()) {
      static unsigned const NOTTab[] = {
        X86::NOT8r, X86::NOT16r, X86::NOT32r, 0, X86::NOT32r
      };
      BuildMI(*MBB, IP, NOTTab[Class], 1, DestReg).addReg(Op0r);
      if (Class == cLong)  // Invert the top part too
        BuildMI(*MBB, IP, X86::NOT32r, 1, DestReg+1).addReg(Op0r+1);
      return;
    }

    // add X, -1 -> dec X
    if (OperatorClass == 0 && Op1C->isAllOnesValue() && Class != cLong) {
      // Note that we can't use dec for 64-bit decrements, because it does not
      // set the carry flag!
      static unsigned const DECTab[] = { X86::DEC8r, X86::DEC16r, X86::DEC32r };
      BuildMI(*MBB, IP, DECTab[Class], 1, DestReg).addReg(Op0r);
      return;
    }

    // add X, 1 -> inc X
    if (OperatorClass == 0 && Op1C->equalsInt(1) && Class != cLong) {
      // Note that we can't use inc for 64-bit increments, because it does not
      // set the carry flag!
      static unsigned const INCTab[] = { X86::INC8r, X86::INC16r, X86::INC32r };
      BuildMI(*MBB, IP, INCTab[Class], 1, DestReg).addReg(Op0r);
      return;
    }

    static const unsigned OpcodeTab[][5] = {
      // Arithmetic operators
      { X86::ADD8ri, X86::ADD16ri, X86::ADD32ri, 0, X86::ADD32ri },  // ADD
      { X86::SUB8ri, X86::SUB16ri, X86::SUB32ri, 0, X86::SUB32ri },  // SUB

      // Bitwise operators
      { X86::AND8ri, X86::AND16ri, X86::AND32ri, 0, X86::AND32ri },  // AND
      { X86:: OR8ri, X86:: OR16ri, X86:: OR32ri, 0, X86::OR32ri  },  // OR
      { X86::XOR8ri, X86::XOR16ri, X86::XOR32ri, 0, X86::XOR32ri },  // XOR
    };

    unsigned Opcode = OpcodeTab[OperatorClass][Class];
    unsigned Op1l = cast<ConstantInt>(Op1C)->getRawValue();

    if (Class != cLong) {
      BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addImm(Op1l);
      return;
    }

    // If this is a long value and the high or low bits have a special
    // property, emit some special cases.
    unsigned Op1h = cast<ConstantInt>(Op1C)->getRawValue() >> 32LL;

    // If the constant is zero in the low 32-bits, just copy the low part
    // across and apply the normal 32-bit operation to the high parts.  There
    // will be no carry or borrow into the top.
    if (Op1l == 0) {
      if (OperatorClass != 2) // All but and...
        BuildMI(*MBB, IP, X86::MOV32rr, 1, DestReg).addReg(Op0r);
      else
        BuildMI(*MBB, IP, X86::MOV32ri, 1, DestReg).addImm(0);
      BuildMI(*MBB, IP, OpcodeTab[OperatorClass][cLong], 2, DestReg+1)
        .addReg(Op0r+1).addImm(Op1h);
      return;
    }

    // If this is a logical operation and the top 32-bits are zero, just
    // operate on the lower 32.
    if (Op1h == 0 && OperatorClass > 1) {
      BuildMI(*MBB, IP, OpcodeTab[OperatorClass][cLong], 2, DestReg)
        .addReg(Op0r).addImm(Op1l);
      if (OperatorClass != 2)  // All but and
        BuildMI(*MBB, IP, X86::MOV32rr, 1, DestReg+1).addReg(Op0r+1);
      else
        BuildMI(*MBB, IP, X86::MOV32ri, 1, DestReg+1).addImm(0);
      return;
    }

    // TODO: We could handle lots of other special cases here, such as AND'ing
    // with 0xFFFFFFFF00000000 -> noop, etc.

    // Otherwise, code generate the full operation with a constant.
    static const unsigned TopTab[] = {
      X86::ADC32ri, X86::SBB32ri, X86::AND32ri, X86::OR32ri, X86::XOR32ri
    };

    BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addImm(Op1l);
    BuildMI(*MBB, IP, TopTab[OperatorClass], 2, DestReg+1)
      .addReg(Op0r+1).addImm(Op1h);
    return;
  }

  // Finally, handle the general case now.
  static const unsigned OpcodeTab[][5] = {
    // Arithmetic operators
    { X86::ADD8rr, X86::ADD16rr, X86::ADD32rr, 0, X86::ADD32rr },  // ADD
    { X86::SUB8rr, X86::SUB16rr, X86::SUB32rr, 0, X86::SUB32rr },  // SUB

    // Bitwise operators
    { X86::AND8rr, X86::AND16rr, X86::AND32rr, 0, X86::AND32rr },  // AND
    { X86:: OR8rr, X86:: OR16rr, X86:: OR32rr, 0, X86:: OR32rr },  // OR
    { X86::XOR8rr, X86::XOR16rr, X86::XOR32rr, 0, X86::XOR32rr },  // XOR
  };

  unsigned Opcode = OpcodeTab[OperatorClass][Class];
  unsigned Op0r = getReg(Op0, MBB, IP);
  unsigned Op1r = getReg(Op1, MBB, IP);
  BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);

  if (Class == cLong) {        // Handle the upper 32 bits of long values...
    static const unsigned TopTab[] = {
      X86::ADC32rr, X86::SBB32rr, X86::AND32rr, X86::OR32rr, X86::XOR32rr
    };
    BuildMI(*MBB, IP, TopTab[OperatorClass], 2,
            DestReg+1).addReg(Op0r+1).addReg(Op1r+1);
  }
}

/// doMultiply - Emit appropriate instructions to multiply together the
/// registers op0Reg and op1Reg, and put the result in DestReg.  The type of the
/// result should be given as DestTy.
///
void X86ISel::doMultiply(MachineBasicBlock *MBB,
                         MachineBasicBlock::iterator MBBI,
                         unsigned DestReg, const Type *DestTy,
                         unsigned op0Reg, unsigned op1Reg) {
  unsigned Class = getClass(DestTy);
  switch (Class) {
  case cInt:
  case cShort:
    BuildMI(*MBB, MBBI, Class == cInt ? X86::IMUL32rr:X86::IMUL16rr, 2, DestReg)
      .addReg(op0Reg).addReg(op1Reg);
    return;
  case cByte:
    // Must use the MUL instruction, which forces use of AL...
    BuildMI(*MBB, MBBI, X86::MOV8rr, 1, X86::AL).addReg(op0Reg);
    BuildMI(*MBB, MBBI, X86::MUL8r, 1).addReg(op1Reg);
    BuildMI(*MBB, MBBI, X86::MOV8rr, 1, DestReg).addReg(X86::AL);
    return;
  default:
  case cLong: assert(0 && "doMultiply cannot operate on LONG values!");
  }
}

// ExactLog2 - This function solves for (Val == 1 << (N-1)) and returns N.  It
// returns zero when the input is not exactly a power of two.
static unsigned ExactLog2(unsigned Val) {
  if (Val == 0 || (Val & (Val-1))) return 0;
  unsigned Count = 0;
  while (Val != 1) {
    Val >>= 1;
    ++Count;
  }
  return Count+1;
}


/// doMultiplyConst - This function is specialized to efficiently codegen an 8,
/// 16, or 32-bit integer multiply by a constant.
void X86ISel::doMultiplyConst(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP,
                              unsigned DestReg, const Type *DestTy,
                              unsigned op0Reg, unsigned ConstRHS) {
  static const unsigned MOVrrTab[] = {X86::MOV8rr, X86::MOV16rr, X86::MOV32rr};
  static const unsigned MOVriTab[] = {X86::MOV8ri, X86::MOV16ri, X86::MOV32ri};
  static const unsigned ADDrrTab[] = {X86::ADD8rr, X86::ADD16rr, X86::ADD32rr};
  static const unsigned NEGrTab[]  = {X86::NEG8r , X86::NEG16r , X86::NEG32r };

  unsigned Class = getClass(DestTy);
  unsigned TmpReg;

  // Handle special cases here.
  switch (ConstRHS) {
  case -2:
    TmpReg = makeAnotherReg(DestTy);
    BuildMI(*MBB, IP, NEGrTab[Class], 1, TmpReg).addReg(op0Reg);
    BuildMI(*MBB, IP, ADDrrTab[Class], 1,DestReg).addReg(TmpReg).addReg(TmpReg);
    return;
  case -1:
    BuildMI(*MBB, IP, NEGrTab[Class], 1, DestReg).addReg(op0Reg);
    return;
  case 0:
    BuildMI(*MBB, IP, MOVriTab[Class], 1, DestReg).addImm(0);
    return;
  case 1:
    BuildMI(*MBB, IP, MOVrrTab[Class], 1, DestReg).addReg(op0Reg);
    return;
  case 2:
    BuildMI(*MBB, IP, ADDrrTab[Class], 1,DestReg).addReg(op0Reg).addReg(op0Reg);
    return;
  case 3:
  case 5:
  case 9:
    if (Class == cInt) {
      X86AddressMode AM;
      AM.BaseType = X86AddressMode::RegBase;
      AM.Base.Reg = op0Reg;
      AM.Scale = ConstRHS-1;
      AM.IndexReg = op0Reg;
      AM.Disp = 0;
      addFullAddress(BuildMI(*MBB, IP, X86::LEA32r, 5, DestReg), AM);
      return;
    }
  case -3:
  case -5:
  case -9:
    if (Class == cInt) {
      TmpReg = makeAnotherReg(DestTy);
      X86AddressMode AM;
      AM.BaseType = X86AddressMode::RegBase;
      AM.Base.Reg = op0Reg;
      AM.Scale = -ConstRHS-1;
      AM.IndexReg = op0Reg;
      AM.Disp = 0;
      addFullAddress(BuildMI(*MBB, IP, X86::LEA32r, 5, TmpReg), AM);
      BuildMI(*MBB, IP, NEGrTab[Class], 1, DestReg).addReg(TmpReg);
      return;
    }
  }

  // If the element size is exactly a power of 2, use a shift to get it.
  if (unsigned Shift = ExactLog2(ConstRHS)) {
    switch (Class) {
    default: assert(0 && "Unknown class for this function!");
    case cByte:
      BuildMI(*MBB, IP, X86::SHL8ri,2, DestReg).addReg(op0Reg).addImm(Shift-1);
      return;
    case cShort:
      BuildMI(*MBB, IP, X86::SHL16ri,2, DestReg).addReg(op0Reg).addImm(Shift-1);
      return;
    case cInt:
      BuildMI(*MBB, IP, X86::SHL32ri,2, DestReg).addReg(op0Reg).addImm(Shift-1);
      return;
    }
  }

  // If the element size is a negative power of 2, use a shift/neg to get it.
  if (unsigned Shift = ExactLog2(-ConstRHS)) {
    TmpReg = makeAnotherReg(DestTy);
    BuildMI(*MBB, IP, NEGrTab[Class], 1, TmpReg).addReg(op0Reg);
    switch (Class) {
    default: assert(0 && "Unknown class for this function!");
    case cByte:
      BuildMI(*MBB, IP, X86::SHL8ri,2, DestReg).addReg(TmpReg).addImm(Shift-1);
      return;
    case cShort:
      BuildMI(*MBB, IP, X86::SHL16ri,2, DestReg).addReg(TmpReg).addImm(Shift-1);
      return;
    case cInt:
      BuildMI(*MBB, IP, X86::SHL32ri,2, DestReg).addReg(TmpReg).addImm(Shift-1);
      return;
    }
  }

  if (Class == cShort) {
    BuildMI(*MBB, IP, X86::IMUL16rri,2,DestReg).addReg(op0Reg).addImm(ConstRHS);
    return;
  } else if (Class == cInt) {
    BuildMI(*MBB, IP, X86::IMUL32rri,2,DestReg).addReg(op0Reg).addImm(ConstRHS);
    return;
  }

  // Most general case, emit a normal multiply...
  TmpReg = makeAnotherReg(DestTy);
  BuildMI(*MBB, IP, MOVriTab[Class], 1, TmpReg).addImm(ConstRHS);

  // Emit a MUL to multiply the register holding the index by
  // elementSize, putting the result in OffsetReg.
  doMultiply(MBB, IP, DestReg, DestTy, op0Reg, TmpReg);
}

/// visitMul - Multiplies are not simple binary operators because they must deal
/// with the EAX register explicitly.
///
void X86ISel::visitMul(BinaryOperator &I) {
  unsigned ResultReg = getReg(I);

  Value *Op0 = I.getOperand(0);
  Value *Op1 = I.getOperand(1);

  // Fold loads into floating point multiplies.
  if (getClass(Op0->getType()) == cFP) {
    if (isa<LoadInst>(Op0) && !isa<LoadInst>(Op1))
      if (!I.swapOperands())
        std::swap(Op0, Op1);  // Make sure any loads are in the RHS.
    if (LoadInst *LI = dyn_cast<LoadInst>(Op1))
      if (isSafeToFoldLoadIntoInstruction(*LI, I)) {
        const Type *Ty = Op0->getType();
        assert(Ty == Type::FloatTy||Ty == Type::DoubleTy && "Unknown FP type!");
        unsigned Opcode = Ty == Type::FloatTy ? X86::FMUL32m : X86::FMUL64m;

        unsigned Op0r = getReg(Op0);
        if (AllocaInst *AI = dyn_castFixedAlloca(LI->getOperand(0))) {
          unsigned FI = getFixedSizedAllocaFI(AI);
          addFrameReference(BuildMI(BB, Opcode, 5, ResultReg).addReg(Op0r), FI);
        } else {
          X86AddressMode AM;
          getAddressingMode(LI->getOperand(0), AM);

          addFullAddress(BuildMI(BB, Opcode, 5, ResultReg).addReg(Op0r), AM);
        }
        return;
      }
  }

  MachineBasicBlock::iterator IP = BB->end();
  emitMultiply(BB, IP, Op0, Op1, ResultReg);
}

void X86ISel::emitMultiply(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator IP,
                           Value *Op0, Value *Op1, unsigned DestReg) {
  MachineBasicBlock &BB = *MBB;
  TypeClass Class = getClass(Op0->getType());

  // Simple scalar multiply?
  unsigned Op0Reg  = getReg(Op0, &BB, IP);
  switch (Class) {
  case cByte:
  case cShort:
  case cInt:
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
      unsigned Val = (unsigned)CI->getRawValue(); // Isn't a 64-bit constant
      doMultiplyConst(&BB, IP, DestReg, Op0->getType(), Op0Reg, Val);
    } else {
      unsigned Op1Reg  = getReg(Op1, &BB, IP);
      doMultiply(&BB, IP, DestReg, Op1->getType(), Op0Reg, Op1Reg);
    }
    return;
  case cFP:
    emitBinaryFPOperation(MBB, IP, Op0, Op1, 2, DestReg);
    return;
  case cLong:
    break;
  }

  // Long value.  We have to do things the hard way...
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    unsigned CLow = CI->getRawValue();
    unsigned CHi  = CI->getRawValue() >> 32;

    if (CLow == 0) {
      // If the low part of the constant is all zeros, things are simple.
      BuildMI(BB, IP, X86::MOV32ri, 1, DestReg).addImm(0);
      doMultiplyConst(&BB, IP, DestReg+1, Type::UIntTy, Op0Reg, CHi);
      return;
    }

    // Multiply the two low parts... capturing carry into EDX
    unsigned OverflowReg = 0;
    if (CLow == 1) {
      BuildMI(BB, IP, X86::MOV32rr, 1, DestReg).addReg(Op0Reg);
    } else {
      unsigned Op1RegL = makeAnotherReg(Type::UIntTy);
      OverflowReg = makeAnotherReg(Type::UIntTy);
      BuildMI(BB, IP, X86::MOV32ri, 1, Op1RegL).addImm(CLow);
      BuildMI(BB, IP, X86::MOV32rr, 1, X86::EAX).addReg(Op0Reg);
      BuildMI(BB, IP, X86::MUL32r, 1).addReg(Op1RegL);  // AL*BL

      BuildMI(BB, IP, X86::MOV32rr, 1, DestReg).addReg(X86::EAX);   // AL*BL
      BuildMI(BB, IP, X86::MOV32rr, 1,
              OverflowReg).addReg(X86::EDX);                    // AL*BL >> 32
    }

    unsigned AHBLReg = makeAnotherReg(Type::UIntTy);   // AH*BL
    doMultiplyConst(&BB, IP, AHBLReg, Type::UIntTy, Op0Reg+1, CLow);

    unsigned AHBLplusOverflowReg;
    if (OverflowReg) {
      AHBLplusOverflowReg = makeAnotherReg(Type::UIntTy);
      BuildMI(BB, IP, X86::ADD32rr, 2,                // AH*BL+(AL*BL >> 32)
              AHBLplusOverflowReg).addReg(AHBLReg).addReg(OverflowReg);
    } else {
      AHBLplusOverflowReg = AHBLReg;
    }

    if (CHi == 0) {
      BuildMI(BB, IP, X86::MOV32rr, 1, DestReg+1).addReg(AHBLplusOverflowReg);
    } else {
      unsigned ALBHReg = makeAnotherReg(Type::UIntTy); // AL*BH
      doMultiplyConst(&BB, IP, ALBHReg, Type::UIntTy, Op0Reg, CHi);

      BuildMI(BB, IP, X86::ADD32rr, 2,      // AL*BH + AH*BL + (AL*BL >> 32)
              DestReg+1).addReg(AHBLplusOverflowReg).addReg(ALBHReg);
    }
    return;
  }

  // General 64x64 multiply

  unsigned Op1Reg  = getReg(Op1, &BB, IP);
  // Multiply the two low parts... capturing carry into EDX
  BuildMI(BB, IP, X86::MOV32rr, 1, X86::EAX).addReg(Op0Reg);
  BuildMI(BB, IP, X86::MUL32r, 1).addReg(Op1Reg);  // AL*BL

  unsigned OverflowReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, IP, X86::MOV32rr, 1, DestReg).addReg(X86::EAX);     // AL*BL
  BuildMI(BB, IP, X86::MOV32rr, 1,
          OverflowReg).addReg(X86::EDX); // AL*BL >> 32

  unsigned AHBLReg = makeAnotherReg(Type::UIntTy);   // AH*BL
  BuildMI(BB, IP, X86::IMUL32rr, 2,
          AHBLReg).addReg(Op0Reg+1).addReg(Op1Reg);

  unsigned AHBLplusOverflowReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, IP, X86::ADD32rr, 2,                // AH*BL+(AL*BL >> 32)
          AHBLplusOverflowReg).addReg(AHBLReg).addReg(OverflowReg);

  unsigned ALBHReg = makeAnotherReg(Type::UIntTy); // AL*BH
  BuildMI(BB, IP, X86::IMUL32rr, 2,
          ALBHReg).addReg(Op0Reg).addReg(Op1Reg+1);

  BuildMI(BB, IP, X86::ADD32rr, 2,      // AL*BH + AH*BL + (AL*BL >> 32)
          DestReg+1).addReg(AHBLplusOverflowReg).addReg(ALBHReg);
}


/// visitDivRem - Handle division and remainder instructions... these
/// instruction both require the same instructions to be generated, they just
/// select the result from a different register.  Note that both of these
/// instructions work differently for signed and unsigned operands.
///
void X86ISel::visitDivRem(BinaryOperator &I) {
  unsigned ResultReg = getReg(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Fold loads into floating point divides.
  if (getClass(Op0->getType()) == cFP) {
    if (LoadInst *LI = dyn_cast<LoadInst>(Op1))
      if (isSafeToFoldLoadIntoInstruction(*LI, I)) {
        const Type *Ty = Op0->getType();
        assert(Ty == Type::FloatTy||Ty == Type::DoubleTy && "Unknown FP type!");
        unsigned Opcode = Ty == Type::FloatTy ? X86::FDIV32m : X86::FDIV64m;

        unsigned Op0r = getReg(Op0);
        if (AllocaInst *AI = dyn_castFixedAlloca(LI->getOperand(0))) {
          unsigned FI = getFixedSizedAllocaFI(AI);
          addFrameReference(BuildMI(BB, Opcode, 5, ResultReg).addReg(Op0r), FI);
        } else {
          X86AddressMode AM;
          getAddressingMode(LI->getOperand(0), AM);

          addFullAddress(BuildMI(BB, Opcode, 5, ResultReg).addReg(Op0r), AM);
        }
        return;
      }

    if (LoadInst *LI = dyn_cast<LoadInst>(Op0))
      if (isSafeToFoldLoadIntoInstruction(*LI, I)) {
        const Type *Ty = Op0->getType();
        assert(Ty == Type::FloatTy||Ty == Type::DoubleTy && "Unknown FP type!");
        unsigned Opcode = Ty == Type::FloatTy ? X86::FDIVR32m : X86::FDIVR64m;

        unsigned Op1r = getReg(Op1);
        if (AllocaInst *AI = dyn_castFixedAlloca(LI->getOperand(0))) {
          unsigned FI = getFixedSizedAllocaFI(AI);
          addFrameReference(BuildMI(BB, Opcode, 5, ResultReg).addReg(Op1r), FI);
        } else {
          X86AddressMode AM;
          getAddressingMode(LI->getOperand(0), AM);
          addFullAddress(BuildMI(BB, Opcode, 5, ResultReg).addReg(Op1r), AM);
        }
        return;
      }
  }


  MachineBasicBlock::iterator IP = BB->end();
  emitDivRemOperation(BB, IP, Op0, Op1,
                      I.getOpcode() == Instruction::Div, ResultReg);
}

void X86ISel::emitDivRemOperation(MachineBasicBlock *BB,
                                  MachineBasicBlock::iterator IP,
                                  Value *Op0, Value *Op1, bool isDiv,
                                  unsigned ResultReg) {
  const Type *Ty = Op0->getType();
  unsigned Class = getClass(Ty);
  switch (Class) {
  case cFP:              // Floating point divide
    if (isDiv) {
      emitBinaryFPOperation(BB, IP, Op0, Op1, 3, ResultReg);
      return;
    } else {               // Floating point remainder...
      unsigned Op0Reg = getReg(Op0, BB, IP);
      unsigned Op1Reg = getReg(Op1, BB, IP);
      MachineInstr *TheCall =
        BuildMI(X86::CALLpcrel32, 1).addExternalSymbol("fmod", true);
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(Op0Reg, Type::DoubleTy));
      Args.push_back(ValueRecord(Op1Reg, Type::DoubleTy));
      doCall(ValueRecord(ResultReg, Type::DoubleTy), TheCall, Args);
    }
    return;
  case cLong: {
    static const char *FnName[] =
      { "__moddi3", "__divdi3", "__umoddi3", "__udivdi3" };
    unsigned Op0Reg = getReg(Op0, BB, IP);
    unsigned Op1Reg = getReg(Op1, BB, IP);
    unsigned NameIdx = Ty->isUnsigned()*2 + isDiv;
    MachineInstr *TheCall =
      BuildMI(X86::CALLpcrel32, 1).addExternalSymbol(FnName[NameIdx], true);

    std::vector<ValueRecord> Args;
    Args.push_back(ValueRecord(Op0Reg, Type::LongTy));
    Args.push_back(ValueRecord(Op1Reg, Type::LongTy));
    doCall(ValueRecord(ResultReg, Type::LongTy), TheCall, Args);
    return;
  }
  case cByte: case cShort: case cInt:
    break;          // Small integrals, handled below...
  default: assert(0 && "Unknown class!");
  }

  static const unsigned MovOpcode[]={ X86::MOV8rr, X86::MOV16rr, X86::MOV32rr };
  static const unsigned NEGOpcode[]={ X86::NEG8r,  X86::NEG16r,  X86::NEG32r };
  static const unsigned SAROpcode[]={ X86::SAR8ri, X86::SAR16ri, X86::SAR32ri };
  static const unsigned SHROpcode[]={ X86::SHR8ri, X86::SHR16ri, X86::SHR32ri };
  static const unsigned ADDOpcode[]={ X86::ADD8rr, X86::ADD16rr, X86::ADD32rr };

  // Special case signed division by power of 2.
  if (ConstantSInt *CI = dyn_cast<ConstantSInt>(Op1))
    if (isDiv) {
      assert(Class != cLong && "This doesn't handle 64-bit divides!");
      int V = CI->getValue();

      if (V == 1) {       // X /s 1 => X
        unsigned Op0Reg = getReg(Op0, BB, IP);
        BuildMI(*BB, IP, MovOpcode[Class], 1, ResultReg).addReg(Op0Reg);
        return;
      }

      if (V == -1) {      // X /s -1 => -X
        unsigned Op0Reg = getReg(Op0, BB, IP);
        BuildMI(*BB, IP, NEGOpcode[Class], 1, ResultReg).addReg(Op0Reg);
        return;
      }

      if (V == 2 || V == -2) {      // X /s 2
        static const unsigned CMPOpcode[] = {
          X86::CMP8ri, X86::CMP16ri, X86::CMP32ri
        };
        static const unsigned SBBOpcode[] = {
          X86::SBB8ri, X86::SBB16ri, X86::SBB32ri
        };
        unsigned Op0Reg = getReg(Op0, BB, IP);
        unsigned SignBit = 1 << (CI->getType()->getPrimitiveSize()*8-1);
        BuildMI(*BB, IP, CMPOpcode[Class], 2).addReg(Op0Reg).addImm(SignBit);

        unsigned TmpReg = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, SBBOpcode[Class], 2, TmpReg).addReg(Op0Reg).addImm(-1);

        unsigned TmpReg2 = V == 2 ? ResultReg : makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, SAROpcode[Class], 2, TmpReg2).addReg(TmpReg).addImm(1);
        if (V == -2) {
          BuildMI(*BB, IP, NEGOpcode[Class], 1, ResultReg).addReg(TmpReg2);
        }
        return;
      }

      bool isNeg = false;
      if (V < 0) {         // Not a positive power of 2?
        V = -V;
        isNeg = true;      // Maybe it's a negative power of 2.
      }
      if (unsigned Log = ExactLog2(V)) {
        --Log;
        unsigned Op0Reg = getReg(Op0, BB, IP);
        unsigned TmpReg = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, SAROpcode[Class], 2, TmpReg)
          .addReg(Op0Reg).addImm(Log-1);
        unsigned TmpReg2 = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, SHROpcode[Class], 2, TmpReg2)
          .addReg(TmpReg).addImm(CI->getType()->getPrimitiveSizeInBits()-Log);
        unsigned TmpReg3 = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, ADDOpcode[Class], 2, TmpReg3)
          .addReg(Op0Reg).addReg(TmpReg2);

        unsigned TmpReg4 = isNeg ? makeAnotherReg(Op0->getType()) : ResultReg;
        BuildMI(*BB, IP, SAROpcode[Class], 2, TmpReg4)
          .addReg(TmpReg3).addImm(Log);
        if (isNeg)
          BuildMI(*BB, IP, NEGOpcode[Class], 1, ResultReg).addReg(TmpReg4);
        return;
      }
    } else {    // X % C
      assert(Class != cLong && "This doesn't handle 64-bit remainder!");
      int V = CI->getValue();

      if (V == 2 || V == -2) {       // X % 2, X % -2
        static const unsigned SExtOpcode[] = { X86::CBW, X86::CWD, X86::CDQ };
        static const unsigned BaseReg[]    = { X86::AL , X86::AX , X86::EAX };
        static const unsigned SExtReg[]    = { X86::AH , X86::DX , X86::EDX };
        static const unsigned ANDOpcode[]  = {
          X86::AND8ri, X86::AND16ri, X86::AND32ri
        };
        static const unsigned XOROpcode[]  = {
          X86::XOR8rr, X86::XOR16rr, X86::XOR32rr
        };
        static const unsigned SUBOpcode[]  = {
          X86::SUB8rr, X86::SUB16rr, X86::SUB32rr
        };

        // Sign extend result into reg of -1 or 0.
        unsigned Op0Reg = getReg(Op0, BB, IP);
        BuildMI(*BB, IP, MovOpcode[Class], 1, BaseReg[Class]).addReg(Op0Reg);
        BuildMI(*BB, IP, SExtOpcode[Class], 0);
        unsigned TmpReg0 = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, MovOpcode[Class], 1, TmpReg0).addReg(SExtReg[Class]);

        unsigned TmpReg1 = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, ANDOpcode[Class], 2, TmpReg1).addReg(Op0Reg).addImm(1);

        unsigned TmpReg2 = makeAnotherReg(Op0->getType());
        BuildMI(*BB, IP, XOROpcode[Class], 2,
                TmpReg2).addReg(TmpReg1).addReg(TmpReg0);
        BuildMI(*BB, IP, SUBOpcode[Class], 2,
                ResultReg).addReg(TmpReg2).addReg(TmpReg0);
        return;
      }
    }

  static const unsigned Regs[]     ={ X86::AL    , X86::AX     , X86::EAX     };
  static const unsigned ClrOpcode[]={ X86::MOV8ri, X86::MOV16ri, X86::MOV32ri };
  static const unsigned ExtRegs[]  ={ X86::AH    , X86::DX     , X86::EDX     };
  static const unsigned SExOpcode[]={ X86::CBW   , X86::CWD    , X86::CDQ     };

  static const unsigned DivOpcode[][4] = {
    { X86::DIV8r , X86::DIV16r , X86::DIV32r , 0 },  // Unsigned division
    { X86::IDIV8r, X86::IDIV16r, X86::IDIV32r, 0 },  // Signed division
  };

  unsigned Reg    = Regs[Class];
  unsigned ExtReg = ExtRegs[Class];

  // Put the first operand into one of the A registers...
  unsigned Op0Reg = getReg(Op0, BB, IP);
  unsigned Op1Reg = getReg(Op1, BB, IP);
  BuildMI(*BB, IP, MovOpcode[Class], 1, Reg).addReg(Op0Reg);

  if (Ty->isSigned()) {
    // Emit a sign extension instruction.
    BuildMI(*BB, IP, SExOpcode[Class], 0);

    // Emit the appropriate divide or remainder instruction...
    BuildMI(*BB, IP, DivOpcode[1][Class], 1).addReg(Op1Reg);
  } else {
    // If unsigned, emit a zeroing instruction... (reg = 0)
    BuildMI(*BB, IP, ClrOpcode[Class], 2, ExtReg).addImm(0);

    // Emit the appropriate divide or remainder instruction...
    BuildMI(*BB, IP, DivOpcode[0][Class], 1).addReg(Op1Reg);
  }

  // Figure out which register we want to pick the result out of...
  unsigned DestReg = isDiv ? Reg : ExtReg;

  // Put the result into the destination register...
  BuildMI(*BB, IP, MovOpcode[Class], 1, ResultReg).addReg(DestReg);
}


/// Shift instructions: 'shl', 'sar', 'shr' - Some special cases here
/// for constant immediate shift values, and for constant immediate
/// shift values equal to 1. Even the general case is sort of special,
/// because the shift amount has to be in CL, not just any old register.
///
void X86ISel::visitShiftInst(ShiftInst &I) {
  MachineBasicBlock::iterator IP = BB->end ();
  emitShiftOperation (BB, IP, I.getOperand (0), I.getOperand (1),
                      I.getOpcode () == Instruction::Shl, I.getType (),
                      getReg (I));
}

/// Emit code for a 'SHLD DestReg, Op0, Op1, Amt' operation, where Amt is a
/// constant.
void X86ISel::doSHLDConst(MachineBasicBlock *MBB,
                          MachineBasicBlock::iterator IP,
                          unsigned DestReg, unsigned Op0Reg, unsigned Op1Reg,
                          unsigned Amt) {
  // SHLD is a very inefficient operation on every processor, try to do
  // somethign simpler for common values of 'Amt'.
  if (Amt == 0) {
    BuildMI(*MBB, IP, X86::MOV32rr, 1, DestReg).addReg(Op0Reg);
  } else if (Amt == 1) {
    unsigned Tmp = makeAnotherReg(Type::UIntTy);
    BuildMI(*MBB, IP, X86::ADD32rr, 2, Tmp).addReg(Op1Reg).addReg(Op1Reg);
    BuildMI(*MBB, IP, X86::ADC32rr, 2, DestReg).addReg(Op0Reg).addReg(Op0Reg);
  } else if (Amt == 2 || Amt == 3) {
    // On the P4 and Athlon it is cheaper to replace shld ..., 2|3 with a
    // shift/lea pair.  NOTE: This should not be done on the P6 family!
    unsigned Tmp = makeAnotherReg(Type::UIntTy);
    BuildMI(*MBB, IP, X86::SHR32ri, 2, Tmp).addReg(Op1Reg).addImm(32-Amt);
    X86AddressMode AM;
    AM.BaseType = X86AddressMode::RegBase;
    AM.Base.Reg = Tmp;
    AM.Scale = 1 << Amt;
    AM.IndexReg = Op0Reg;
    AM.Disp = 0;
    addFullAddress(BuildMI(*MBB, IP, X86::LEA32r, 4, DestReg), AM);
  } else {
    // NOTE: It is always cheaper on the P4 to emit SHLD as two shifts and an OR
    // than it is to emit a real SHLD.

    BuildMI(*MBB, IP, X86::SHLD32rri8, 3,
            DestReg).addReg(Op0Reg).addReg(Op1Reg).addImm(Amt);
  }
}

/// emitShiftOperation - Common code shared between visitShiftInst and
/// constant expression support.
void X86ISel::emitShiftOperation(MachineBasicBlock *MBB,
                                 MachineBasicBlock::iterator IP,
                                 Value *Op, Value *ShiftAmount,
                                 bool isLeftShift, const Type *ResultTy,
                                 unsigned DestReg) {
  unsigned SrcReg = getReg (Op, MBB, IP);
  bool isSigned = ResultTy->isSigned ();
  unsigned Class = getClass (ResultTy);

  static const unsigned ConstantOperand[][3] = {
    { X86::SHR8ri, X86::SHR16ri, X86::SHR32ri },  // SHR
    { X86::SAR8ri, X86::SAR16ri, X86::SAR32ri },  // SAR
    { X86::SHL8ri, X86::SHL16ri, X86::SHL32ri },  // SHL
    { X86::SHL8ri, X86::SHL16ri, X86::SHL32ri },  // SAL = SHL
  };

  static const unsigned NonConstantOperand[][3] = {
    { X86::SHR8rCL, X86::SHR16rCL, X86::SHR32rCL },  // SHR
    { X86::SAR8rCL, X86::SAR16rCL, X86::SAR32rCL },  // SAR
    { X86::SHL8rCL, X86::SHL16rCL, X86::SHL32rCL },  // SHL
    { X86::SHL8rCL, X86::SHL16rCL, X86::SHL32rCL },  // SAL = SHL
  };

  // Longs, as usual, are handled specially.
  if (Class == cLong) {
    if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(ShiftAmount)) {
      unsigned Amount = CUI->getValue();
      if (Amount == 1 && isLeftShift) {   // X << 1 == X+X
        BuildMI(*MBB, IP, X86::ADD32rr, 2,
                DestReg).addReg(SrcReg).addReg(SrcReg);
        BuildMI(*MBB, IP, X86::ADC32rr, 2,
                DestReg+1).addReg(SrcReg+1).addReg(SrcReg+1);
      } else if (Amount < 32) {
        const unsigned *Opc = ConstantOperand[isLeftShift*2+isSigned];
        if (isLeftShift) {
          doSHLDConst(MBB, IP, DestReg+1, SrcReg+1, SrcReg, Amount);
          BuildMI(*MBB, IP, Opc[2], 2, DestReg).addReg(SrcReg).addImm(Amount);
        } else {
          BuildMI(*MBB, IP, X86::SHRD32rri8, 3,
                  DestReg).addReg(SrcReg  ).addReg(SrcReg+1).addImm(Amount);
          BuildMI(*MBB, IP, Opc[2],2,DestReg+1).addReg(SrcReg+1).addImm(Amount);
        }
      } else if (Amount == 32) {
        if (isLeftShift) {
          BuildMI(*MBB, IP, X86::MOV32rr, 1, DestReg+1).addReg(SrcReg);
          BuildMI(*MBB, IP, X86::MOV32ri, 1, DestReg).addImm(0);
        } else {
          BuildMI(*MBB, IP, X86::MOV32rr, 1, DestReg).addReg(SrcReg+1);
          if (!isSigned) {
            BuildMI(*MBB, IP, X86::MOV32ri, 1, DestReg+1).addImm(0);
          } else {
            BuildMI(*MBB, IP, X86::SAR32ri, 2,
                    DestReg+1).addReg(SrcReg).addImm(31);
          }
        }
      } else {                 // Shifting more than 32 bits
        Amount -= 32;
        if (isLeftShift) {
          BuildMI(*MBB, IP, X86::SHL32ri, 2,
                  DestReg + 1).addReg(SrcReg).addImm(Amount);
          BuildMI(*MBB, IP, X86::MOV32ri, 1, DestReg).addImm(0);
        } else {
          BuildMI(*MBB, IP, isSigned ? X86::SAR32ri : X86::SHR32ri, 2,
                  DestReg).addReg(SrcReg+1).addImm(Amount);
          if (isSigned)
            BuildMI(*MBB, IP, X86::SAR32ri, 2,
                    DestReg+1).addReg(SrcReg+1).addImm(31);
          else
            BuildMI(*MBB, IP, X86::MOV32ri, 1, DestReg+1).addImm(0);
        }
      }
    } else {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      if (!isLeftShift && isSigned) {
        // If this is a SHR of a Long, then we need to do funny sign extension
        // stuff.  TmpReg gets the value to use as the high-part if we are
        // shifting more than 32 bits.
        BuildMI(*MBB, IP, X86::SAR32ri, 2, TmpReg).addReg(SrcReg).addImm(31);
      } else {
        // Other shifts use a fixed zero value if the shift is more than 32
        // bits.
        BuildMI(*MBB, IP, X86::MOV32ri, 1, TmpReg).addImm(0);
      }

      // Initialize CL with the shift amount...
      unsigned ShiftAmountReg = getReg(ShiftAmount, MBB, IP);
      BuildMI(*MBB, IP, X86::MOV8rr, 1, X86::CL).addReg(ShiftAmountReg);

      unsigned TmpReg2 = makeAnotherReg(Type::IntTy);
      unsigned TmpReg3 = makeAnotherReg(Type::IntTy);
      if (isLeftShift) {
        // TmpReg2 = shld inHi, inLo
        BuildMI(*MBB, IP, X86::SHLD32rrCL,2,TmpReg2).addReg(SrcReg+1)
                                                    .addReg(SrcReg);
        // TmpReg3 = shl  inLo, CL
        BuildMI(*MBB, IP, X86::SHL32rCL, 1, TmpReg3).addReg(SrcReg);

        // Set the flags to indicate whether the shift was by more than 32 bits.
        BuildMI(*MBB, IP, X86::TEST8ri, 2).addReg(X86::CL).addImm(32);

        // DestHi = (>32) ? TmpReg3 : TmpReg2;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2,
                DestReg+1).addReg(TmpReg2).addReg(TmpReg3);
        // DestLo = (>32) ? TmpReg : TmpReg3;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2,
            DestReg).addReg(TmpReg3).addReg(TmpReg);
      } else {
        // TmpReg2 = shrd inLo, inHi
        BuildMI(*MBB, IP, X86::SHRD32rrCL,2,TmpReg2).addReg(SrcReg)
                                                    .addReg(SrcReg+1);
        // TmpReg3 = s[ah]r  inHi, CL
        BuildMI(*MBB, IP, isSigned ? X86::SAR32rCL : X86::SHR32rCL, 1, TmpReg3)
                       .addReg(SrcReg+1);

        // Set the flags to indicate whether the shift was by more than 32 bits.
        BuildMI(*MBB, IP, X86::TEST8ri, 2).addReg(X86::CL).addImm(32);

        // DestLo = (>32) ? TmpReg3 : TmpReg2;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2,
                DestReg).addReg(TmpReg2).addReg(TmpReg3);

        // DestHi = (>32) ? TmpReg : TmpReg3;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2,
                DestReg+1).addReg(TmpReg3).addReg(TmpReg);
      }
    }
    return;
  }

  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(ShiftAmount)) {
    // The shift amount is constant, guaranteed to be a ubyte. Get its value.
    assert(CUI->getType() == Type::UByteTy && "Shift amount not a ubyte?");

    if (CUI->getValue() == 1 && isLeftShift) {    // X << 1 -> X+X
      static const int AddOpC[] = { X86::ADD8rr, X86::ADD16rr, X86::ADD32rr };
      BuildMI(*MBB, IP, AddOpC[Class], 2,DestReg).addReg(SrcReg).addReg(SrcReg);
    } else {
      const unsigned *Opc = ConstantOperand[isLeftShift*2+isSigned];
      BuildMI(*MBB, IP, Opc[Class], 2,
              DestReg).addReg(SrcReg).addImm(CUI->getValue());
    }
  } else {                  // The shift amount is non-constant.
    unsigned ShiftAmountReg = getReg (ShiftAmount, MBB, IP);
    BuildMI(*MBB, IP, X86::MOV8rr, 1, X86::CL).addReg(ShiftAmountReg);

    const unsigned *Opc = NonConstantOperand[isLeftShift*2+isSigned];
    BuildMI(*MBB, IP, Opc[Class], 1, DestReg).addReg(SrcReg);
  }
}


/// visitLoadInst - Implement LLVM load instructions in terms of the x86 'mov'
/// instruction.  The load and store instructions are the only place where we
/// need to worry about the memory layout of the target machine.
///
void X86ISel::visitLoadInst(LoadInst &I) {
  // Check to see if this load instruction is going to be folded into a binary
  // instruction, like add.  If so, we don't want to emit it.  Wouldn't a real
  // pattern matching instruction selector be nice?
  unsigned Class = getClassB(I.getType());
  if (I.hasOneUse()) {
    Instruction *User = cast<Instruction>(I.use_back());
    switch (User->getOpcode()) {
    case Instruction::Cast:
      // If this is a cast from a signed-integer type to a floating point type,
      // fold the cast here.
      if (getClassB(User->getType()) == cFP &&
          (I.getType() == Type::ShortTy || I.getType() == Type::IntTy ||
           I.getType() == Type::LongTy)) {
        unsigned DestReg = getReg(User);
        static const unsigned Opcode[] = {
          0/*BYTE*/, X86::FILD16m, X86::FILD32m, 0/*FP*/, X86::FILD64m
        };

        if (AllocaInst *AI = dyn_castFixedAlloca(I.getOperand(0))) {
          unsigned FI = getFixedSizedAllocaFI(AI);
          addFrameReference(BuildMI(BB, Opcode[Class], 4, DestReg), FI);
        } else {
          X86AddressMode AM;
          getAddressingMode(I.getOperand(0), AM);
          addFullAddress(BuildMI(BB, Opcode[Class], 4, DestReg), AM);
        }
        return;
      } else {
        User = 0;
      }
      break;

    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      if (Class == cLong) User = 0;
      break;
    case Instruction::Mul:
    case Instruction::Div:
      if (Class != cFP) User = 0;
      break;  // Folding only implemented for floating point.
    default: User = 0; break;
    }

    if (User) {
      // Okay, we found a user.  If the load is the first operand and there is
      // no second operand load, reverse the operand ordering.  Note that this
      // can fail for a subtract (ie, no change will be made).
      bool Swapped = false;
      if (!isa<LoadInst>(User->getOperand(1)))
        Swapped = !cast<BinaryOperator>(User)->swapOperands();

      // Okay, now that everything is set up, if this load is used by the second
      // operand, and if there are no instructions that invalidate the load
      // before the binary operator, eliminate the load.
      if (User->getOperand(1) == &I &&
          isSafeToFoldLoadIntoInstruction(I, *User))
        return;   // Eliminate the load!

      // If this is a floating point sub or div, we won't be able to swap the
      // operands, but we will still be able to eliminate the load.
      if (Class == cFP && User->getOperand(0) == &I &&
          !isa<LoadInst>(User->getOperand(1)) &&
          (User->getOpcode() == Instruction::Sub ||
           User->getOpcode() == Instruction::Div) &&
          isSafeToFoldLoadIntoInstruction(I, *User))
        return;  // Eliminate the load!

      // If we swapped the operands to the instruction, but couldn't fold the
      // load anyway, swap them back.  We don't want to break add X, int
      // folding.
      if (Swapped) cast<BinaryOperator>(User)->swapOperands();
    }
  }

  static const unsigned Opcodes[] = {
    X86::MOV8rm, X86::MOV16rm, X86::MOV32rm, X86::FLD32m, X86::MOV32rm
  };
  unsigned Opcode = Opcodes[Class];
  if (I.getType() == Type::DoubleTy) Opcode = X86::FLD64m;

  unsigned DestReg = getReg(I);

  if (AllocaInst *AI = dyn_castFixedAlloca(I.getOperand(0))) {
    unsigned FI = getFixedSizedAllocaFI(AI);
    if (Class == cLong) {
      addFrameReference(BuildMI(BB, X86::MOV32rm, 4, DestReg), FI);
      addFrameReference(BuildMI(BB, X86::MOV32rm, 4, DestReg+1), FI, 4);
    } else {
      addFrameReference(BuildMI(BB, Opcode, 4, DestReg), FI);
    }
  } else {
    X86AddressMode AM;
    getAddressingMode(I.getOperand(0), AM);

    if (Class == cLong) {
      addFullAddress(BuildMI(BB, X86::MOV32rm, 4, DestReg), AM);
      AM.Disp += 4;
      addFullAddress(BuildMI(BB, X86::MOV32rm, 4, DestReg+1), AM);
    } else {
      addFullAddress(BuildMI(BB, Opcode, 4, DestReg), AM);
    }
  }
}

/// visitStoreInst - Implement LLVM store instructions in terms of the x86 'mov'
/// instruction.
///
void X86ISel::visitStoreInst(StoreInst &I) {
  X86AddressMode AM;
  getAddressingMode(I.getOperand(1), AM);

  const Type *ValTy = I.getOperand(0)->getType();
  unsigned Class = getClassB(ValTy);

  if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(0))) {
    uint64_t Val = CI->getRawValue();
    if (Class == cLong) {
      addFullAddress(BuildMI(BB, X86::MOV32mi, 5), AM).addImm(Val & ~0U);
      AM.Disp += 4;
      addFullAddress(BuildMI(BB, X86::MOV32mi, 5), AM).addImm(Val>>32);
    } else {
      static const unsigned Opcodes[] = {
        X86::MOV8mi, X86::MOV16mi, X86::MOV32mi
      };
      unsigned Opcode = Opcodes[Class];
      addFullAddress(BuildMI(BB, Opcode, 5), AM).addImm(Val);
    }
  } else if (isa<ConstantPointerNull>(I.getOperand(0))) {
    addFullAddress(BuildMI(BB, X86::MOV32mi, 5), AM).addImm(0);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(I.getOperand(0))) {
    addFullAddress(BuildMI(BB, X86::MOV32mi, 5), AM).addGlobalAddress(GV);
  } else if (ConstantBool *CB = dyn_cast<ConstantBool>(I.getOperand(0))) {
    addFullAddress(BuildMI(BB, X86::MOV8mi, 5), AM).addImm(CB->getValue());
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(I.getOperand(0))) {
    // Store constant FP values with integer instructions to avoid having to
    // load the constants from the constant pool then do a store.
    if (CFP->getType() == Type::FloatTy) {
      union {
        unsigned I;
        float    F;
      } V;
      V.F = CFP->getValue();
      addFullAddress(BuildMI(BB, X86::MOV32mi, 5), AM).addImm(V.I);
    } else {
      union {
        uint64_t I;
        double   F;
      } V;
      V.F = CFP->getValue();
      addFullAddress(BuildMI(BB, X86::MOV32mi, 5), AM).addImm((unsigned)V.I);
      AM.Disp += 4;
      addFullAddress(BuildMI(BB, X86::MOV32mi, 5), AM).addImm(
                                                          unsigned(V.I >> 32));
    }

  } else if (Class == cLong) {
    unsigned ValReg = getReg(I.getOperand(0));
    addFullAddress(BuildMI(BB, X86::MOV32mr, 5), AM).addReg(ValReg);
    AM.Disp += 4;
    addFullAddress(BuildMI(BB, X86::MOV32mr, 5), AM).addReg(ValReg+1);
  } else {
    // FIXME: stop emitting these two instructions:
    //    movl $global,%eax
    //    movl %eax,(%ebx)
    // when one instruction will suffice.  That includes when the global
    // has an offset applied to it.
    unsigned ValReg = getReg(I.getOperand(0));
    static const unsigned Opcodes[] = {
      X86::MOV8mr, X86::MOV16mr, X86::MOV32mr, X86::FST32m
    };
    unsigned Opcode = Opcodes[Class];
    if (ValTy == Type::DoubleTy) Opcode = X86::FST64m;

    addFullAddress(BuildMI(BB, Opcode, 1+4), AM).addReg(ValReg);
  }
}


/// visitCastInst - Here we have various kinds of copying with or without sign
/// extension going on.
///
void X86ISel::visitCastInst(CastInst &CI) {
  Value *Op = CI.getOperand(0);

  unsigned SrcClass = getClassB(Op->getType());
  unsigned DestClass = getClassB(CI.getType());
  // Noop casts are not emitted: getReg will return the source operand as the
  // register to use for any uses of the noop cast.
  if (DestClass == SrcClass) {
    // The only detail in this plan is that casts from double -> float are
    // truncating operations that we have to codegen through memory (despite
    // the fact that the source/dest registers are the same class).
    if (CI.getType() != Type::FloatTy || Op->getType() != Type::DoubleTy)
      return;
  }

  // If this is a cast from a 32-bit integer to a Long type, and the only uses
  // of the case are GEP instructions, then the cast does not need to be
  // generated explicitly, it will be folded into the GEP.
  if (DestClass == cLong && SrcClass == cInt) {
    bool AllUsesAreGEPs = true;
    for (Value::use_iterator I = CI.use_begin(), E = CI.use_end(); I != E; ++I)
      if (!isa<GetElementPtrInst>(*I)) {
        AllUsesAreGEPs = false;
        break;
      }

    // No need to codegen this cast if all users are getelementptr instrs...
    if (AllUsesAreGEPs) return;
  }

  // If this cast converts a load from a short,int, or long integer to a FP
  // value, we will have folded this cast away.
  if (DestClass == cFP && isa<LoadInst>(Op) && Op->hasOneUse() &&
      (Op->getType() == Type::ShortTy || Op->getType() == Type::IntTy ||
       Op->getType() == Type::LongTy))
    return;


  unsigned DestReg = getReg(CI);
  MachineBasicBlock::iterator MI = BB->end();
  emitCastOperation(BB, MI, Op, CI.getType(), DestReg);
}

/// emitCastOperation - Common code shared between visitCastInst and constant
/// expression cast support.
///
void X86ISel::emitCastOperation(MachineBasicBlock *BB,
                                MachineBasicBlock::iterator IP,
                                Value *Src, const Type *DestTy,
                                unsigned DestReg) {
  const Type *SrcTy = Src->getType();
  unsigned SrcClass = getClassB(SrcTy);
  unsigned DestClass = getClassB(DestTy);
  unsigned SrcReg = getReg(Src, BB, IP);

  // Implement casts to bool by using compare on the operand followed by set if
  // not zero on the result.
  if (DestTy == Type::BoolTy) {
    switch (SrcClass) {
    case cByte:
      BuildMI(*BB, IP, X86::TEST8rr, 2).addReg(SrcReg).addReg(SrcReg);
      break;
    case cShort:
      BuildMI(*BB, IP, X86::TEST16rr, 2).addReg(SrcReg).addReg(SrcReg);
      break;
    case cInt:
      BuildMI(*BB, IP, X86::TEST32rr, 2).addReg(SrcReg).addReg(SrcReg);
      break;
    case cLong: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      BuildMI(*BB, IP, X86::OR32rr, 2, TmpReg).addReg(SrcReg).addReg(SrcReg+1);
      break;
    }
    case cFP:
      BuildMI(*BB, IP, X86::FTST, 1).addReg(SrcReg);
      BuildMI(*BB, IP, X86::FNSTSW8r, 0);
      BuildMI(*BB, IP, X86::SAHF, 1);
      break;
    }

    // If the zero flag is not set, then the value is true, set the byte to
    // true.
    BuildMI(*BB, IP, X86::SETNEr, 1, DestReg);
    return;
  }

  static const unsigned RegRegMove[] = {
    X86::MOV8rr, X86::MOV16rr, X86::MOV32rr, X86::FpMOV, X86::MOV32rr
  };

  // Implement casts between values of the same type class (as determined by
  // getClass) by using a register-to-register move.
  if (SrcClass == DestClass) {
    if (SrcClass <= cInt || (SrcClass == cFP && SrcTy == DestTy)) {
      BuildMI(*BB, IP, RegRegMove[SrcClass], 1, DestReg).addReg(SrcReg);
    } else if (SrcClass == cFP) {
      if (SrcTy == Type::FloatTy) {  // double -> float
        assert(DestTy == Type::DoubleTy && "Unknown cFP member!");
        BuildMI(*BB, IP, X86::FpMOV, 1, DestReg).addReg(SrcReg);
      } else {                       // float -> double
        assert(SrcTy == Type::DoubleTy && DestTy == Type::FloatTy &&
               "Unknown cFP member!");
        // Truncate from double to float by storing to memory as short, then
        // reading it back.
        unsigned FltAlign = TM.getTargetData().getFloatAlignment();
        int FrameIdx = F->getFrameInfo()->CreateStackObject(4, FltAlign);
        addFrameReference(BuildMI(*BB, IP, X86::FST32m, 5),
			  FrameIdx).addReg(SrcReg);
        addFrameReference(BuildMI(*BB, IP, X86::FLD32m, 5, DestReg), FrameIdx);
      }
    } else if (SrcClass == cLong) {
      BuildMI(*BB, IP, X86::MOV32rr, 1, DestReg).addReg(SrcReg);
      BuildMI(*BB, IP, X86::MOV32rr, 1, DestReg+1).addReg(SrcReg+1);
    } else {
      assert(0 && "Cannot handle this type of cast instruction!");
      abort();
    }
    return;
  }

  // Handle cast of SMALLER int to LARGER int using a move with sign extension
  // or zero extension, depending on whether the source type was signed.
  if (SrcClass <= cInt && (DestClass <= cInt || DestClass == cLong) &&
      SrcClass < DestClass) {
    bool isLong = DestClass == cLong;
    if (isLong) DestClass = cInt;

    static const unsigned Opc[][4] = {
      { X86::MOVSX16rr8, X86::MOVSX32rr8, X86::MOVSX32rr16, X86::MOV32rr }, // s
      { X86::MOVZX16rr8, X86::MOVZX32rr8, X86::MOVZX32rr16, X86::MOV32rr }  // u
    };

    bool isUnsigned = SrcTy->isUnsigned() || SrcTy == Type::BoolTy;
    BuildMI(*BB, IP, Opc[isUnsigned][SrcClass + DestClass - 1], 1,
        DestReg).addReg(SrcReg);

    if (isLong) {  // Handle upper 32 bits as appropriate...
      if (isUnsigned)     // Zero out top bits...
        BuildMI(*BB, IP, X86::MOV32ri, 1, DestReg+1).addImm(0);
      else                // Sign extend bottom half...
        BuildMI(*BB, IP, X86::SAR32ri, 2, DestReg+1).addReg(DestReg).addImm(31);
    }
    return;
  }

  // Special case long -> int ...
  if (SrcClass == cLong && DestClass == cInt) {
    BuildMI(*BB, IP, X86::MOV32rr, 1, DestReg).addReg(SrcReg);
    return;
  }

  // Handle cast of LARGER int to SMALLER int using a move to EAX followed by a
  // move out of AX or AL.
  if ((SrcClass <= cInt || SrcClass == cLong) && DestClass <= cInt
      && SrcClass > DestClass) {
    static const unsigned AReg[] = { X86::AL, X86::AX, X86::EAX, 0, X86::EAX };
    BuildMI(*BB, IP, RegRegMove[SrcClass], 1, AReg[SrcClass]).addReg(SrcReg);
    BuildMI(*BB, IP, RegRegMove[DestClass], 1, DestReg).addReg(AReg[DestClass]);
    return;
  }

  // Handle casts from integer to floating point now...
  if (DestClass == cFP) {
    // Promote the integer to a type supported by FLD.  We do this because there
    // are no unsigned FLD instructions, so we must promote an unsigned value to
    // a larger signed value, then use FLD on the larger value.
    //
    const Type *PromoteType = 0;
    unsigned PromoteOpcode = 0;
    unsigned RealDestReg = DestReg;
    switch (SrcTy->getTypeID()) {
    case Type::BoolTyID:
    case Type::SByteTyID:
      // We don't have the facilities for directly loading byte sized data from
      // memory (even signed).  Promote it to 16 bits.
      PromoteType = Type::ShortTy;
      PromoteOpcode = X86::MOVSX16rr8;
      break;
    case Type::UByteTyID:
      PromoteType = Type::ShortTy;
      PromoteOpcode = X86::MOVZX16rr8;
      break;
    case Type::UShortTyID:
      PromoteType = Type::IntTy;
      PromoteOpcode = X86::MOVZX32rr16;
      break;
    case Type::ULongTyID:
    case Type::UIntTyID:
      // Don't fild into the read destination.
      DestReg = makeAnotherReg(Type::DoubleTy);
      break;
    default:  // No promotion needed...
      break;
    }

    if (PromoteType) {
      unsigned TmpReg = makeAnotherReg(PromoteType);
      BuildMI(*BB, IP, PromoteOpcode, 1, TmpReg).addReg(SrcReg);
      SrcTy = PromoteType;
      SrcClass = getClass(PromoteType);
      SrcReg = TmpReg;
    }

    // Spill the integer to memory and reload it from there...
    int FrameIdx =
      F->getFrameInfo()->CreateStackObject(SrcTy, TM.getTargetData());

    if (SrcClass == cLong) {
      addFrameReference(BuildMI(*BB, IP, X86::MOV32mr, 5),
                        FrameIdx).addReg(SrcReg);
      addFrameReference(BuildMI(*BB, IP, X86::MOV32mr, 5),
                        FrameIdx, 4).addReg(SrcReg+1);
    } else {
      static const unsigned Op1[] = { X86::MOV8mr, X86::MOV16mr, X86::MOV32mr };
      addFrameReference(BuildMI(*BB, IP, Op1[SrcClass], 5),
                        FrameIdx).addReg(SrcReg);
    }

    static const unsigned Op2[] =
      { 0/*byte*/, X86::FILD16m, X86::FILD32m, 0/*FP*/, X86::FILD64m };
    addFrameReference(BuildMI(*BB, IP, Op2[SrcClass], 5, DestReg), FrameIdx);

    if (SrcTy == Type::UIntTy) {
      // If this is a cast from uint -> double, we need to be careful about if
      // the "sign" bit is set.  If so, we don't want to make a negative number,
      // we want to make a positive number.  Emit code to add an offset if the
      // sign bit is set.

      // Compute whether the sign bit is set by shifting the reg right 31 bits.
      unsigned IsNeg = makeAnotherReg(Type::IntTy);
      BuildMI(*BB, IP, X86::SHR32ri, 2, IsNeg).addReg(SrcReg).addImm(31);

      // Create a CP value that has the offset in one word and 0 in the other.
      static ConstantInt *TheOffset = ConstantUInt::get(Type::ULongTy,
                                                        0x4f80000000000000ULL);
      unsigned CPI = F->getConstantPool()->getConstantPoolIndex(TheOffset);
      BuildMI(*BB, IP, X86::FADD32m, 5, RealDestReg).addReg(DestReg)
        .addConstantPoolIndex(CPI).addZImm(4).addReg(IsNeg).addSImm(0);

    } else if (SrcTy == Type::ULongTy) {
      // We need special handling for unsigned 64-bit integer sources.  If the
      // input number has the "sign bit" set, then we loaded it incorrectly as a
      // negative 64-bit number.  In this case, add an offset value.

      // Emit a test instruction to see if the dynamic input value was signed.
      BuildMI(*BB, IP, X86::TEST32rr, 2).addReg(SrcReg+1).addReg(SrcReg+1);

      // If the sign bit is set, get a pointer to an offset, otherwise get a
      // pointer to a zero.
      MachineConstantPool *CP = F->getConstantPool();
      unsigned Zero = makeAnotherReg(Type::IntTy);
      Constant *Null = Constant::getNullValue(Type::UIntTy);
      addConstantPoolReference(BuildMI(*BB, IP, X86::LEA32r, 5, Zero),
                               CP->getConstantPoolIndex(Null));
      unsigned Offset = makeAnotherReg(Type::IntTy);
      Constant *OffsetCst = ConstantUInt::get(Type::UIntTy, 0x5f800000);

      addConstantPoolReference(BuildMI(*BB, IP, X86::LEA32r, 5, Offset),
                               CP->getConstantPoolIndex(OffsetCst));
      unsigned Addr = makeAnotherReg(Type::IntTy);
      BuildMI(*BB, IP, X86::CMOVS32rr, 2, Addr).addReg(Zero).addReg(Offset);

      // Load the constant for an add.  FIXME: this could make an 'fadd' that
      // reads directly from memory, but we don't support these yet.
      unsigned ConstReg = makeAnotherReg(Type::DoubleTy);
      addDirectMem(BuildMI(*BB, IP, X86::FLD32m, 4, ConstReg), Addr);

      BuildMI(*BB, IP, X86::FpADD, 2, RealDestReg)
                .addReg(ConstReg).addReg(DestReg);
    }

    return;
  }

  // Handle casts from floating point to integer now...
  if (SrcClass == cFP) {
    // Change the floating point control register to use "round towards zero"
    // mode when truncating to an integer value.
    //
    int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2);
    addFrameReference(BuildMI(*BB, IP, X86::FNSTCW16m, 4), CWFrameIdx);

    // Load the old value of the high byte of the control word...
    unsigned HighPartOfCW = makeAnotherReg(Type::UByteTy);
    addFrameReference(BuildMI(*BB, IP, X86::MOV8rm, 4, HighPartOfCW),
                      CWFrameIdx, 1);

    // Set the high part to be round to zero...
    addFrameReference(BuildMI(*BB, IP, X86::MOV8mi, 5),
                      CWFrameIdx, 1).addImm(12);

    // Reload the modified control word now...
    addFrameReference(BuildMI(*BB, IP, X86::FLDCW16m, 4), CWFrameIdx);

    // Restore the memory image of control word to original value
    addFrameReference(BuildMI(*BB, IP, X86::MOV8mr, 5),
                      CWFrameIdx, 1).addReg(HighPartOfCW);

    // We don't have the facilities for directly storing byte sized data to
    // memory.  Promote it to 16 bits.  We also must promote unsigned values to
    // larger classes because we only have signed FP stores.
    unsigned StoreClass  = DestClass;
    const Type *StoreTy  = DestTy;
    if (StoreClass == cByte || DestTy->isUnsigned())
      switch (StoreClass) {
      case cByte:  StoreTy = Type::ShortTy; StoreClass = cShort; break;
      case cShort: StoreTy = Type::IntTy;   StoreClass = cInt;   break;
      case cInt:   StoreTy = Type::LongTy;  StoreClass = cLong;  break;
      // The following treatment of cLong may not be perfectly right,
      // but it survives chains of casts of the form
      // double->ulong->double.
      case cLong:  StoreTy = Type::LongTy;  StoreClass = cLong;  break;
      default: assert(0 && "Unknown store class!");
      }

    // Spill the integer to memory and reload it from there...
    int FrameIdx =
      F->getFrameInfo()->CreateStackObject(StoreTy, TM.getTargetData());

    static const unsigned Op1[] =
      { 0, X86::FIST16m, X86::FIST32m, 0, X86::FISTP64m };
    addFrameReference(BuildMI(*BB, IP, Op1[StoreClass], 5),
                      FrameIdx).addReg(SrcReg);

    if (DestClass == cLong) {
      addFrameReference(BuildMI(*BB, IP, X86::MOV32rm, 4, DestReg), FrameIdx);
      addFrameReference(BuildMI(*BB, IP, X86::MOV32rm, 4, DestReg+1),
                        FrameIdx, 4);
    } else {
      static const unsigned Op2[] = { X86::MOV8rm, X86::MOV16rm, X86::MOV32rm };
      addFrameReference(BuildMI(*BB, IP, Op2[DestClass], 4, DestReg), FrameIdx);
    }

    // Reload the original control word now...
    addFrameReference(BuildMI(*BB, IP, X86::FLDCW16m, 4), CWFrameIdx);
    return;
  }

  // Anything we haven't handled already, we can't (yet) handle at all.
  assert(0 && "Unhandled cast instruction!");
  abort();
}

/// visitVANextInst - Implement the va_next instruction...
///
void X86ISel::visitVANextInst(VANextInst &I) {
  unsigned VAList = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  unsigned Size;
  switch (I.getArgType()->getTypeID()) {
  default:
    std::cerr << I;
    assert(0 && "Error: bad type for va_next instruction!");
    return;
  case Type::PointerTyID:
  case Type::UIntTyID:
  case Type::IntTyID:
    Size = 4;
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
  case Type::DoubleTyID:
    Size = 8;
    break;
  }

  // Increment the VAList pointer...
  BuildMI(BB, X86::ADD32ri, 2, DestReg).addReg(VAList).addImm(Size);
}

void X86ISel::visitVAArgInst(VAArgInst &I) {
  unsigned VAList = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  switch (I.getType()->getTypeID()) {
  default:
    std::cerr << I;
    assert(0 && "Error: bad type for va_next instruction!");
    return;
  case Type::PointerTyID:
  case Type::UIntTyID:
  case Type::IntTyID:
    addDirectMem(BuildMI(BB, X86::MOV32rm, 4, DestReg), VAList);
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
    addDirectMem(BuildMI(BB, X86::MOV32rm, 4, DestReg), VAList);
    addRegOffset(BuildMI(BB, X86::MOV32rm, 4, DestReg+1), VAList, 4);
    break;
  case Type::DoubleTyID:
    addDirectMem(BuildMI(BB, X86::FLD64m, 4, DestReg), VAList);
    break;
  }
}

/// visitGetElementPtrInst - instruction-select GEP instructions
///
void X86ISel::visitGetElementPtrInst(GetElementPtrInst &I) {
  // If this GEP instruction will be folded into all of its users, we don't need
  // to explicitly calculate it!
  X86AddressMode AM;
  if (isGEPFoldable(0, I.getOperand(0), I.op_begin()+1, I.op_end(), AM)) {
    // Check all of the users of the instruction to see if they are loads and
    // stores.
    bool AllWillFold = true;
    for (Value::use_iterator UI = I.use_begin(), E = I.use_end(); UI != E; ++UI)
      if (cast<Instruction>(*UI)->getOpcode() != Instruction::Load)
        if (cast<Instruction>(*UI)->getOpcode() != Instruction::Store ||
            cast<Instruction>(*UI)->getOperand(0) == &I) {
          AllWillFold = false;
          break;
        }

    // If the instruction is foldable, and will be folded into all users, don't
    // emit it!
    if (AllWillFold) return;
  }

  unsigned outputReg = getReg(I);
  emitGEPOperation(BB, BB->end(), I.getOperand(0),
                   I.op_begin()+1, I.op_end(), outputReg);
}

/// getGEPIndex - Inspect the getelementptr operands specified with GEPOps and
/// GEPTypes (the derived types being stepped through at each level).  On return
/// from this function, if some indexes of the instruction are representable as
/// an X86 lea instruction, the machine operands are put into the Ops
/// instruction and the consumed indexes are poped from the GEPOps/GEPTypes
/// lists.  Otherwise, GEPOps.size() is returned.  If this returns a an
/// addressing mode that only partially consumes the input, the BaseReg input of
/// the addressing mode must be left free.
///
/// Note that there is one fewer entry in GEPTypes than there is in GEPOps.
///
void X86ISel::getGEPIndex(MachineBasicBlock *MBB,
                          MachineBasicBlock::iterator IP,
                          std::vector<Value*> &GEPOps,
                          std::vector<const Type*> &GEPTypes,
                          X86AddressMode &AM) {
  const TargetData &TD = TM.getTargetData();

  // Clear out the state we are working with...
  AM.BaseType = X86AddressMode::RegBase;
  AM.Base.Reg = 0;   // No base register
  AM.Scale = 1;      // Unit scale
  AM.IndexReg = 0;   // No index register
  AM.Disp = 0;       // No displacement

  // While there are GEP indexes that can be folded into the current address,
  // keep processing them.
  while (!GEPTypes.empty()) {
    if (const StructType *StTy = dyn_cast<StructType>(GEPTypes.back())) {
      // It's a struct access.  CUI is the index into the structure,
      // which names the field. This index must have unsigned type.
      const ConstantUInt *CUI = cast<ConstantUInt>(GEPOps.back());

      // Use the TargetData structure to pick out what the layout of the
      // structure is in memory.  Since the structure index must be constant, we
      // can get its value and use it to find the right byte offset from the
      // StructLayout class's list of structure member offsets.
      AM.Disp += TD.getStructLayout(StTy)->MemberOffsets[CUI->getValue()];
      GEPOps.pop_back();        // Consume a GEP operand
      GEPTypes.pop_back();
    } else {
      // It's an array or pointer access: [ArraySize x ElementType].
      const SequentialType *SqTy = cast<SequentialType>(GEPTypes.back());
      Value *idx = GEPOps.back();

      // idx is the index into the array.  Unlike with structure
      // indices, we may not know its actual value at code-generation
      // time.

      // If idx is a constant, fold it into the offset.
      unsigned TypeSize = TD.getTypeSize(SqTy->getElementType());
      if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(idx)) {
        AM.Disp += TypeSize*CSI->getValue();
      } else if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(idx)) {
        AM.Disp += TypeSize*CUI->getValue();
      } else {
        // If the index reg is already taken, we can't handle this index.
        if (AM.IndexReg) return;

        // If this is a size that we can handle, then add the index as
        switch (TypeSize) {
        case 1: case 2: case 4: case 8:
          // These are all acceptable scales on X86.
          AM.Scale = TypeSize;
          break;
        default:
          // Otherwise, we can't handle this scale
          return;
        }

        if (CastInst *CI = dyn_cast<CastInst>(idx))
          if (CI->getOperand(0)->getType() == Type::IntTy ||
              CI->getOperand(0)->getType() == Type::UIntTy)
            idx = CI->getOperand(0);

        AM.IndexReg = MBB ? getReg(idx, MBB, IP) : 1;
      }

      GEPOps.pop_back();        // Consume a GEP operand
      GEPTypes.pop_back();
    }
  }

  // GEPTypes is empty, which means we have a single operand left.  Set it as
  // the base register.
  //
  assert(AM.Base.Reg == 0);

  if (AllocaInst *AI = dyn_castFixedAlloca(GEPOps.back())) {
    AM.BaseType = X86AddressMode::FrameIndexBase;
    AM.Base.FrameIndex = getFixedSizedAllocaFI(AI);
    GEPOps.pop_back();
    return;
  }

  if (GlobalValue *GV = dyn_cast<GlobalValue>(GEPOps.back())) {
    AM.GV = GV;
    GEPOps.pop_back();
    return;
  }

  AM.Base.Reg = MBB ? getReg(GEPOps[0], MBB, IP) : 1;
  GEPOps.pop_back();        // Consume the last GEP operand
}


/// isGEPFoldable - Return true if the specified GEP can be completely
/// folded into the addressing mode of a load/store or lea instruction.
bool X86ISel::isGEPFoldable(MachineBasicBlock *MBB,
                            Value *Src, User::op_iterator IdxBegin,
                            User::op_iterator IdxEnd, X86AddressMode &AM) {

  std::vector<Value*> GEPOps;
  GEPOps.resize(IdxEnd-IdxBegin+1);
  GEPOps[0] = Src;
  std::copy(IdxBegin, IdxEnd, GEPOps.begin()+1);

  std::vector<const Type*>
    GEPTypes(gep_type_begin(Src->getType(), IdxBegin, IdxEnd),
             gep_type_end(Src->getType(), IdxBegin, IdxEnd));

  MachineBasicBlock::iterator IP;
  if (MBB) IP = MBB->end();
  getGEPIndex(MBB, IP, GEPOps, GEPTypes, AM);

  // We can fold it away iff the getGEPIndex call eliminated all operands.
  return GEPOps.empty();
}

void X86ISel::emitGEPOperation(MachineBasicBlock *MBB,
                               MachineBasicBlock::iterator IP,
                               Value *Src, User::op_iterator IdxBegin,
                               User::op_iterator IdxEnd, unsigned TargetReg) {
  const TargetData &TD = TM.getTargetData();

  // If this is a getelementptr null, with all constant integer indices, just
  // replace it with TargetReg = 42.
  if (isa<ConstantPointerNull>(Src)) {
    User::op_iterator I = IdxBegin;
    for (; I != IdxEnd; ++I)
      if (!isa<ConstantInt>(*I))
        break;
    if (I == IdxEnd) {   // All constant indices
      unsigned Offset = TD.getIndexedOffset(Src->getType(),
                                         std::vector<Value*>(IdxBegin, IdxEnd));
      BuildMI(*MBB, IP, X86::MOV32ri, 1, TargetReg).addImm(Offset);
      return;
    }
  }

  std::vector<Value*> GEPOps;
  GEPOps.resize(IdxEnd-IdxBegin+1);
  GEPOps[0] = Src;
  std::copy(IdxBegin, IdxEnd, GEPOps.begin()+1);

  std::vector<const Type*> GEPTypes;
  GEPTypes.assign(gep_type_begin(Src->getType(), IdxBegin, IdxEnd),
                  gep_type_end(Src->getType(), IdxBegin, IdxEnd));

  // Keep emitting instructions until we consume the entire GEP instruction.
  while (!GEPOps.empty()) {
    unsigned OldSize = GEPOps.size();
    X86AddressMode AM;
    getGEPIndex(MBB, IP, GEPOps, GEPTypes, AM);

    if (GEPOps.size() != OldSize) {
      // getGEPIndex consumed some of the input.  Build an LEA instruction here.
      unsigned NextTarget = 0;
      if (!GEPOps.empty()) {
        assert(AM.Base.Reg == 0 &&
           "getGEPIndex should have left the base register open for chaining!");
        NextTarget = AM.Base.Reg = makeAnotherReg(Type::UIntTy);
      }

      if (AM.BaseType == X86AddressMode::RegBase &&
          AM.IndexReg == 0 && AM.Disp == 0 && !AM.GV)
        BuildMI(*MBB, IP, X86::MOV32rr, 1, TargetReg).addReg(AM.Base.Reg);
      else if (AM.BaseType == X86AddressMode::RegBase && AM.Base.Reg == 0 &&
               AM.IndexReg == 0 && AM.Disp == 0)
        BuildMI(*MBB, IP, X86::MOV32ri, 1, TargetReg).addGlobalAddress(AM.GV);
      else
        addFullAddress(BuildMI(*MBB, IP, X86::LEA32r, 5, TargetReg), AM);
      --IP;
      TargetReg = NextTarget;
    } else if (GEPTypes.empty()) {
      // The getGEPIndex operation didn't want to build an LEA.  Check to see if
      // all operands are consumed but the base pointer.  If so, just load it
      // into the register.
      if (GlobalValue *GV = dyn_cast<GlobalValue>(GEPOps[0])) {
        BuildMI(*MBB, IP, X86::MOV32ri, 1, TargetReg).addGlobalAddress(GV);
      } else {
        unsigned BaseReg = getReg(GEPOps[0], MBB, IP);
        BuildMI(*MBB, IP, X86::MOV32rr, 1, TargetReg).addReg(BaseReg);
      }
      break;                // we are now done

    } else {
      // It's an array or pointer access: [ArraySize x ElementType].
      const SequentialType *SqTy = cast<SequentialType>(GEPTypes.back());
      Value *idx = GEPOps.back();
      GEPOps.pop_back();        // Consume a GEP operand
      GEPTypes.pop_back();

      // Many GEP instructions use a [cast (int/uint) to LongTy] as their
      // operand on X86.  Handle this case directly now...
      if (CastInst *CI = dyn_cast<CastInst>(idx))
        if (CI->getOperand(0)->getType() == Type::IntTy ||
            CI->getOperand(0)->getType() == Type::UIntTy)
          idx = CI->getOperand(0);

      // We want to add BaseReg to(idxReg * sizeof ElementType). First, we
      // must find the size of the pointed-to type (Not coincidentally, the next
      // type is the type of the elements in the array).
      const Type *ElTy = SqTy->getElementType();
      unsigned elementSize = TD.getTypeSize(ElTy);

      // If idxReg is a constant, we don't need to perform the multiply!
      if (ConstantInt *CSI = dyn_cast<ConstantInt>(idx)) {
        if (!CSI->isNullValue()) {
          unsigned Offset = elementSize*CSI->getRawValue();
          unsigned Reg = makeAnotherReg(Type::UIntTy);
          BuildMI(*MBB, IP, X86::ADD32ri, 2, TargetReg)
                                .addReg(Reg).addImm(Offset);
          --IP;            // Insert the next instruction before this one.
          TargetReg = Reg; // Codegen the rest of the GEP into this
        }
      } else if (elementSize == 1) {
        // If the element size is 1, we don't have to multiply, just add
        unsigned idxReg = getReg(idx, MBB, IP);
        unsigned Reg = makeAnotherReg(Type::UIntTy);
        BuildMI(*MBB, IP, X86::ADD32rr, 2,TargetReg).addReg(Reg).addReg(idxReg);
        --IP;            // Insert the next instruction before this one.
        TargetReg = Reg; // Codegen the rest of the GEP into this
      } else {
        unsigned idxReg = getReg(idx, MBB, IP);
        unsigned OffsetReg = makeAnotherReg(Type::UIntTy);

        // Make sure we can back the iterator up to point to the first
        // instruction emitted.
        MachineBasicBlock::iterator BeforeIt = IP;
        if (IP == MBB->begin())
          BeforeIt = MBB->end();
        else
          --BeforeIt;
        doMultiplyConst(MBB, IP, OffsetReg, Type::IntTy, idxReg, elementSize);

        // Emit an ADD to add OffsetReg to the basePtr.
        unsigned Reg = makeAnotherReg(Type::UIntTy);
        BuildMI(*MBB, IP, X86::ADD32rr, 2, TargetReg)
                          .addReg(Reg).addReg(OffsetReg);

        // Step to the first instruction of the multiply.
        if (BeforeIt == MBB->end())
          IP = MBB->begin();
        else
          IP = ++BeforeIt;

        TargetReg = Reg; // Codegen the rest of the GEP into this
      }
    }
  }
}

/// visitAllocaInst - If this is a fixed size alloca, allocate space from the
/// frame manager, otherwise do it the hard way.
///
void X86ISel::visitAllocaInst(AllocaInst &I) {
  // If this is a fixed size alloca in the entry block for the function, we
  // statically stack allocate the space, so we don't need to do anything here.
  //
  if (dyn_castFixedAlloca(&I)) return;

  // Find the data size of the alloca inst's getAllocatedType.
  const Type *Ty = I.getAllocatedType();
  unsigned TySize = TM.getTargetData().getTypeSize(Ty);

  // Create a register to hold the temporary result of multiplying the type size
  // constant by the variable amount.
  unsigned TotalSizeReg = makeAnotherReg(Type::UIntTy);
  unsigned SrcReg1 = getReg(I.getArraySize());

  // TotalSizeReg = mul <numelements>, <TypeSize>
  MachineBasicBlock::iterator MBBI = BB->end();
  doMultiplyConst(BB, MBBI, TotalSizeReg, Type::UIntTy, SrcReg1, TySize);

  // AddedSize = add <TotalSizeReg>, 15
  unsigned AddedSizeReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, X86::ADD32ri, 2, AddedSizeReg).addReg(TotalSizeReg).addImm(15);

  // AlignedSize = and <AddedSize>, ~15
  unsigned AlignedSize = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, X86::AND32ri, 2, AlignedSize).addReg(AddedSizeReg).addImm(~15);

  // Subtract size from stack pointer, thereby allocating some space.
  BuildMI(BB, X86::SUB32rr, 2, X86::ESP).addReg(X86::ESP).addReg(AlignedSize);

  // Put a pointer to the space into the result register, by copying
  // the stack pointer.
  BuildMI(BB, X86::MOV32rr, 1, getReg(I)).addReg(X86::ESP);

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  F->getFrameInfo()->CreateVariableSizedObject();
}

/// visitMallocInst - Malloc instructions are code generated into direct calls
/// to the library malloc.
///
void X86ISel::visitMallocInst(MallocInst &I) {
  unsigned AllocSize = TM.getTargetData().getTypeSize(I.getAllocatedType());
  unsigned Arg;

  if (ConstantUInt *C = dyn_cast<ConstantUInt>(I.getOperand(0))) {
    Arg = getReg(ConstantUInt::get(Type::UIntTy, C->getValue() * AllocSize));
  } else {
    Arg = makeAnotherReg(Type::UIntTy);
    unsigned Op0Reg = getReg(I.getOperand(0));
    MachineBasicBlock::iterator MBBI = BB->end();
    doMultiplyConst(BB, MBBI, Arg, Type::UIntTy, Op0Reg, AllocSize);
  }

  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(Arg, Type::UIntTy));
  MachineInstr *TheCall = BuildMI(X86::CALLpcrel32,
                                  1).addExternalSymbol("malloc", true);
  doCall(ValueRecord(getReg(I), I.getType()), TheCall, Args);
}


/// visitFreeInst - Free instructions are code gen'd to call the free libc
/// function.
///
void X86ISel::visitFreeInst(FreeInst &I) {
  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(I.getOperand(0)));
  MachineInstr *TheCall = BuildMI(X86::CALLpcrel32,
                                  1).addExternalSymbol("free", true);
  doCall(ValueRecord(0, Type::VoidTy), TheCall, Args);
}

/// createX86SimpleInstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
FunctionPass *llvm::createX86SimpleInstructionSelector(TargetMachine &TM) {
  return new X86ISel(TM);
}

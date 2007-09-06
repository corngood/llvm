//===-- ExecutionEngine.cpp - Common Implementation shared by EEs ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the common interface used by the various execution engine
// subclasses.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/Target/TargetData.h"
#include <math.h>
using namespace llvm;

STATISTIC(NumInitBytes, "Number of bytes of global vars initialized");
STATISTIC(NumGlobals  , "Number of global vars initialized");

ExecutionEngine::EECtorFn ExecutionEngine::JITCtor = 0;
ExecutionEngine::EECtorFn ExecutionEngine::InterpCtor = 0;

ExecutionEngine::ExecutionEngine(ModuleProvider *P) {
  LazyCompilationDisabled = false;
  Modules.push_back(P);
  assert(P && "ModuleProvider is null?");
}

ExecutionEngine::ExecutionEngine(Module *M) {
  LazyCompilationDisabled = false;
  assert(M && "Module is null?");
  Modules.push_back(new ExistingModuleProvider(M));
}

ExecutionEngine::~ExecutionEngine() {
  clearAllGlobalMappings();
  for (unsigned i = 0, e = Modules.size(); i != e; ++i)
    delete Modules[i];
}

/// FindFunctionNamed - Search all of the active modules to find the one that
/// defines FnName.  This is very slow operation and shouldn't be used for
/// general code.
Function *ExecutionEngine::FindFunctionNamed(const char *FnName) {
  for (unsigned i = 0, e = Modules.size(); i != e; ++i) {
    if (Function *F = Modules[i]->getModule()->getFunction(FnName))
      return F;
  }
  return 0;
}


/// addGlobalMapping - Tell the execution engine that the specified global is
/// at the specified location.  This is used internally as functions are JIT'd
/// and as global variables are laid out in memory.  It can and should also be
/// used by clients of the EE that want to have an LLVM global overlay
/// existing data in memory.
void ExecutionEngine::addGlobalMapping(const GlobalValue *GV, void *Addr) {
  MutexGuard locked(lock);
  
  void *&CurVal = state.getGlobalAddressMap(locked)[GV];
  assert((CurVal == 0 || Addr == 0) && "GlobalMapping already established!");
  CurVal = Addr;
  
  // If we are using the reverse mapping, add it too
  if (!state.getGlobalAddressReverseMap(locked).empty()) {
    const GlobalValue *&V = state.getGlobalAddressReverseMap(locked)[Addr];
    assert((V == 0 || GV == 0) && "GlobalMapping already established!");
    V = GV;
  }
}

/// clearAllGlobalMappings - Clear all global mappings and start over again
/// use in dynamic compilation scenarios when you want to move globals
void ExecutionEngine::clearAllGlobalMappings() {
  MutexGuard locked(lock);
  
  state.getGlobalAddressMap(locked).clear();
  state.getGlobalAddressReverseMap(locked).clear();
}

/// updateGlobalMapping - Replace an existing mapping for GV with a new
/// address.  This updates both maps as required.  If "Addr" is null, the
/// entry for the global is removed from the mappings.
void ExecutionEngine::updateGlobalMapping(const GlobalValue *GV, void *Addr) {
  MutexGuard locked(lock);
  
  // Deleting from the mapping?
  if (Addr == 0) {
    state.getGlobalAddressMap(locked).erase(GV);
    if (!state.getGlobalAddressReverseMap(locked).empty())
      state.getGlobalAddressReverseMap(locked).erase(Addr);
    return;
  }
  
  void *&CurVal = state.getGlobalAddressMap(locked)[GV];
  if (CurVal && !state.getGlobalAddressReverseMap(locked).empty())
    state.getGlobalAddressReverseMap(locked).erase(CurVal);
  CurVal = Addr;
  
  // If we are using the reverse mapping, add it too
  if (!state.getGlobalAddressReverseMap(locked).empty()) {
    const GlobalValue *&V = state.getGlobalAddressReverseMap(locked)[Addr];
    assert((V == 0 || GV == 0) && "GlobalMapping already established!");
    V = GV;
  }
}

/// getPointerToGlobalIfAvailable - This returns the address of the specified
/// global value if it is has already been codegen'd, otherwise it returns null.
///
void *ExecutionEngine::getPointerToGlobalIfAvailable(const GlobalValue *GV) {
  MutexGuard locked(lock);
  
  std::map<const GlobalValue*, void*>::iterator I =
  state.getGlobalAddressMap(locked).find(GV);
  return I != state.getGlobalAddressMap(locked).end() ? I->second : 0;
}

/// getGlobalValueAtAddress - Return the LLVM global value object that starts
/// at the specified address.
///
const GlobalValue *ExecutionEngine::getGlobalValueAtAddress(void *Addr) {
  MutexGuard locked(lock);

  // If we haven't computed the reverse mapping yet, do so first.
  if (state.getGlobalAddressReverseMap(locked).empty()) {
    for (std::map<const GlobalValue*, void *>::iterator
         I = state.getGlobalAddressMap(locked).begin(),
         E = state.getGlobalAddressMap(locked).end(); I != E; ++I)
      state.getGlobalAddressReverseMap(locked).insert(std::make_pair(I->second,
                                                                     I->first));
  }

  std::map<void *, const GlobalValue*>::iterator I =
    state.getGlobalAddressReverseMap(locked).find(Addr);
  return I != state.getGlobalAddressReverseMap(locked).end() ? I->second : 0;
}

// CreateArgv - Turn a vector of strings into a nice argv style array of
// pointers to null terminated strings.
//
static void *CreateArgv(ExecutionEngine *EE,
                        const std::vector<std::string> &InputArgv) {
  unsigned PtrSize = EE->getTargetData()->getPointerSize();
  char *Result = new char[(InputArgv.size()+1)*PtrSize];

  DOUT << "ARGV = " << (void*)Result << "\n";
  const Type *SBytePtr = PointerType::get(Type::Int8Ty);

  for (unsigned i = 0; i != InputArgv.size(); ++i) {
    unsigned Size = InputArgv[i].size()+1;
    char *Dest = new char[Size];
    DOUT << "ARGV[" << i << "] = " << (void*)Dest << "\n";

    std::copy(InputArgv[i].begin(), InputArgv[i].end(), Dest);
    Dest[Size-1] = 0;

    // Endian safe: Result[i] = (PointerTy)Dest;
    EE->StoreValueToMemory(PTOGV(Dest), (GenericValue*)(Result+i*PtrSize),
                           SBytePtr);
  }

  // Null terminate it
  EE->StoreValueToMemory(PTOGV(0),
                         (GenericValue*)(Result+InputArgv.size()*PtrSize),
                         SBytePtr);
  return Result;
}


/// runStaticConstructorsDestructors - This method is used to execute all of
/// the static constructors or destructors for a program, depending on the
/// value of isDtors.
void ExecutionEngine::runStaticConstructorsDestructors(bool isDtors) {
  const char *Name = isDtors ? "llvm.global_dtors" : "llvm.global_ctors";
  
  // Execute global ctors/dtors for each module in the program.
  for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
    GlobalVariable *GV = Modules[m]->getModule()->getNamedGlobal(Name);

    // If this global has internal linkage, or if it has a use, then it must be
    // an old-style (llvmgcc3) static ctor with __main linked in and in use.  If
    // this is the case, don't execute any of the global ctors, __main will do
    // it.
    if (!GV || GV->isDeclaration() || GV->hasInternalLinkage()) continue;
  
    // Should be an array of '{ int, void ()* }' structs.  The first value is
    // the init priority, which we ignore.
    ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
    if (!InitList) continue;
    for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
      if (ConstantStruct *CS = 
          dyn_cast<ConstantStruct>(InitList->getOperand(i))) {
        if (CS->getNumOperands() != 2) break; // Not array of 2-element structs.
      
        Constant *FP = CS->getOperand(1);
        if (FP->isNullValue())
          break;  // Found a null terminator, exit.
      
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
          if (CE->isCast())
            FP = CE->getOperand(0);
        if (Function *F = dyn_cast<Function>(FP)) {
          // Execute the ctor/dtor function!
          runFunction(F, std::vector<GenericValue>());
        }
      }
  }
}

/// runFunctionAsMain - This is a helper function which wraps runFunction to
/// handle the common task of starting up main with the specified argc, argv,
/// and envp parameters.
int ExecutionEngine::runFunctionAsMain(Function *Fn,
                                       const std::vector<std::string> &argv,
                                       const char * const * envp) {
  std::vector<GenericValue> GVArgs;
  GenericValue GVArgc;
  GVArgc.IntVal = APInt(32, argv.size());

  // Check main() type
  unsigned NumArgs = Fn->getFunctionType()->getNumParams();
  const FunctionType *FTy = Fn->getFunctionType();
  const Type* PPInt8Ty = PointerType::get(PointerType::get(Type::Int8Ty));
  switch (NumArgs) {
  case 3:
   if (FTy->getParamType(2) != PPInt8Ty) {
     cerr << "Invalid type for third argument of main() supplied\n";
     abort();
   }
   // FALLS THROUGH
  case 2:
   if (FTy->getParamType(1) != PPInt8Ty) {
     cerr << "Invalid type for second argument of main() supplied\n";
     abort();
   }
   // FALLS THROUGH
  case 1:
   if (FTy->getParamType(0) != Type::Int32Ty) {
     cerr << "Invalid type for first argument of main() supplied\n";
     abort();
   }
   // FALLS THROUGH
  case 0:
   if (FTy->getReturnType() != Type::Int32Ty &&
       FTy->getReturnType() != Type::VoidTy) {
     cerr << "Invalid return type of main() supplied\n";
     abort();
   }
   break;
  default:
   cerr << "Invalid number of arguments of main() supplied\n";
   abort();
  }
  
  if (NumArgs) {
    GVArgs.push_back(GVArgc); // Arg #0 = argc.
    if (NumArgs > 1) {
      GVArgs.push_back(PTOGV(CreateArgv(this, argv))); // Arg #1 = argv.
      assert(((char **)GVTOP(GVArgs[1]))[0] &&
             "argv[0] was null after CreateArgv");
      if (NumArgs > 2) {
        std::vector<std::string> EnvVars;
        for (unsigned i = 0; envp[i]; ++i)
          EnvVars.push_back(envp[i]);
        GVArgs.push_back(PTOGV(CreateArgv(this, EnvVars))); // Arg #2 = envp.
      }
    }
  }
  return runFunction(Fn, GVArgs).IntVal.getZExtValue();
}

/// If possible, create a JIT, unless the caller specifically requests an
/// Interpreter or there's an error. If even an Interpreter cannot be created,
/// NULL is returned.
///
ExecutionEngine *ExecutionEngine::create(ModuleProvider *MP,
                                         bool ForceInterpreter,
                                         std::string *ErrorStr) {
  ExecutionEngine *EE = 0;

  // Unless the interpreter was explicitly selected, try making a JIT.
  if (!ForceInterpreter && JITCtor)
    EE = JITCtor(MP, ErrorStr);

  // If we can't make a JIT, make an interpreter instead.
  if (EE == 0 && InterpCtor)
    EE = InterpCtor(MP, ErrorStr);

  if (EE) {
    // Make sure we can resolve symbols in the program as well. The zero arg
    // to the function tells DynamicLibrary to load the program, not a library.
    try {
      sys::DynamicLibrary::LoadLibraryPermanently(0);
    } catch (...) {
    }
  }

  return EE;
}

/// getPointerToGlobal - This returns the address of the specified global
/// value.  This may involve code generation if it's a function.
///
void *ExecutionEngine::getPointerToGlobal(const GlobalValue *GV) {
  if (Function *F = const_cast<Function*>(dyn_cast<Function>(GV)))
    return getPointerToFunction(F);

  MutexGuard locked(lock);
  void *p = state.getGlobalAddressMap(locked)[GV];
  if (p)
    return p;

  // Global variable might have been added since interpreter started.
  if (GlobalVariable *GVar =
          const_cast<GlobalVariable *>(dyn_cast<GlobalVariable>(GV)))
    EmitGlobalVariable(GVar);
  else
    assert(0 && "Global hasn't had an address allocated yet!");
  return state.getGlobalAddressMap(locked)[GV];
}

/// This function converts a Constant* into a GenericValue. The interesting 
/// part is if C is a ConstantExpr.
/// @brief Get a GenericValue for a Constant*
GenericValue ExecutionEngine::getConstantValue(const Constant *C) {
  // If its undefined, return the garbage.
  if (isa<UndefValue>(C)) 
    return GenericValue();

  // If the value is a ConstantExpr
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    Constant *Op0 = CE->getOperand(0);
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // Compute the index 
      GenericValue Result = getConstantValue(Op0);
      SmallVector<Value*, 8> Indices(CE->op_begin()+1, CE->op_end());
      uint64_t Offset =
        TD->getIndexedOffset(Op0->getType(), &Indices[0], Indices.size());

      char* tmp = (char*) Result.PointerVal;
      Result = PTOGV(tmp + Offset);
      return Result;
    }
    case Instruction::Trunc: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      GV.IntVal = GV.IntVal.trunc(BitWidth);
      return GV;
    }
    case Instruction::ZExt: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      GV.IntVal = GV.IntVal.zext(BitWidth);
      return GV;
    }
    case Instruction::SExt: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      GV.IntVal = GV.IntVal.sext(BitWidth);
      return GV;
    }
    case Instruction::FPTrunc: {
      GenericValue GV = getConstantValue(Op0);
      GV.FloatVal = float(GV.DoubleVal);
      return GV;
    }
    case Instruction::FPExt:{
      GenericValue GV = getConstantValue(Op0);
      GV.DoubleVal = double(GV.FloatVal);
      return GV;
    }
    case Instruction::UIToFP: {
      GenericValue GV = getConstantValue(Op0);
      if (CE->getType() == Type::FloatTy)
        GV.FloatVal = float(GV.IntVal.roundToDouble());
      else
        GV.DoubleVal = GV.IntVal.roundToDouble();
      return GV;
    }
    case Instruction::SIToFP: {
      GenericValue GV = getConstantValue(Op0);
      if (CE->getType() == Type::FloatTy)
        GV.FloatVal = float(GV.IntVal.signedRoundToDouble());
      else
        GV.DoubleVal = GV.IntVal.signedRoundToDouble();
      return GV;
    }
    case Instruction::FPToUI: // double->APInt conversion handles sign
    case Instruction::FPToSI: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t BitWidth = cast<IntegerType>(CE->getType())->getBitWidth();
      if (Op0->getType() == Type::FloatTy)
        GV.IntVal = APIntOps::RoundFloatToAPInt(GV.FloatVal, BitWidth);
      else
        GV.IntVal = APIntOps::RoundDoubleToAPInt(GV.DoubleVal, BitWidth);
      return GV;
    }
    case Instruction::PtrToInt: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t PtrWidth = TD->getPointerSizeInBits();
      GV.IntVal = APInt(PtrWidth, uintptr_t(GV.PointerVal));
      return GV;
    }
    case Instruction::IntToPtr: {
      GenericValue GV = getConstantValue(Op0);
      uint32_t PtrWidth = TD->getPointerSizeInBits();
      if (PtrWidth != GV.IntVal.getBitWidth())
        GV.IntVal = GV.IntVal.zextOrTrunc(PtrWidth);
      assert(GV.IntVal.getBitWidth() <= 64 && "Bad pointer width");
      GV.PointerVal = PointerTy(uintptr_t(GV.IntVal.getZExtValue()));
      return GV;
    }
    case Instruction::BitCast: {
      GenericValue GV = getConstantValue(Op0);
      const Type* DestTy = CE->getType();
      switch (Op0->getType()->getTypeID()) {
        default: assert(0 && "Invalid bitcast operand");
        case Type::IntegerTyID:
          assert(DestTy->isFloatingPoint() && "invalid bitcast");
          if (DestTy == Type::FloatTy)
            GV.FloatVal = GV.IntVal.bitsToFloat();
          else if (DestTy == Type::DoubleTy)
            GV.DoubleVal = GV.IntVal.bitsToDouble();
          break;
        case Type::FloatTyID: 
          assert(DestTy == Type::Int32Ty && "Invalid bitcast");
          GV.IntVal.floatToBits(GV.FloatVal);
          break;
        case Type::DoubleTyID:
          assert(DestTy == Type::Int64Ty && "Invalid bitcast");
          GV.IntVal.doubleToBits(GV.DoubleVal);
          break;
        case Type::PointerTyID:
          assert(isa<PointerType>(DestTy) && "Invalid bitcast");
          break; // getConstantValue(Op0)  above already converted it
      }
      return GV;
    }
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      GenericValue LHS = getConstantValue(Op0);
      GenericValue RHS = getConstantValue(CE->getOperand(1));
      GenericValue GV;
      switch (CE->getOperand(0)->getType()->getTypeID()) {
      default: assert(0 && "Bad add type!"); abort();
      case Type::IntegerTyID:
        switch (CE->getOpcode()) {
          default: assert(0 && "Invalid integer opcode");
          case Instruction::Add: GV.IntVal = LHS.IntVal + RHS.IntVal; break;
          case Instruction::Sub: GV.IntVal = LHS.IntVal - RHS.IntVal; break;
          case Instruction::Mul: GV.IntVal = LHS.IntVal * RHS.IntVal; break;
          case Instruction::UDiv:GV.IntVal = LHS.IntVal.udiv(RHS.IntVal); break;
          case Instruction::SDiv:GV.IntVal = LHS.IntVal.sdiv(RHS.IntVal); break;
          case Instruction::URem:GV.IntVal = LHS.IntVal.urem(RHS.IntVal); break;
          case Instruction::SRem:GV.IntVal = LHS.IntVal.srem(RHS.IntVal); break;
          case Instruction::And: GV.IntVal = LHS.IntVal & RHS.IntVal; break;
          case Instruction::Or:  GV.IntVal = LHS.IntVal | RHS.IntVal; break;
          case Instruction::Xor: GV.IntVal = LHS.IntVal ^ RHS.IntVal; break;
        }
        break;
      case Type::FloatTyID:
        switch (CE->getOpcode()) {
          default: assert(0 && "Invalid float opcode"); abort();
          case Instruction::Add:  
            GV.FloatVal = LHS.FloatVal + RHS.FloatVal; break;
          case Instruction::Sub:  
            GV.FloatVal = LHS.FloatVal - RHS.FloatVal; break;
          case Instruction::Mul:  
            GV.FloatVal = LHS.FloatVal * RHS.FloatVal; break;
          case Instruction::FDiv: 
            GV.FloatVal = LHS.FloatVal / RHS.FloatVal; break;
          case Instruction::FRem: 
            GV.FloatVal = ::fmodf(LHS.FloatVal,RHS.FloatVal); break;
        }
        break;
      case Type::DoubleTyID:
        switch (CE->getOpcode()) {
          default: assert(0 && "Invalid double opcode"); abort();
          case Instruction::Add:  
            GV.DoubleVal = LHS.DoubleVal + RHS.DoubleVal; break;
          case Instruction::Sub:  
            GV.DoubleVal = LHS.DoubleVal - RHS.DoubleVal; break;
          case Instruction::Mul:  
            GV.DoubleVal = LHS.DoubleVal * RHS.DoubleVal; break;
          case Instruction::FDiv: 
            GV.DoubleVal = LHS.DoubleVal / RHS.DoubleVal; break;
          case Instruction::FRem: 
            GV.DoubleVal = ::fmod(LHS.DoubleVal,RHS.DoubleVal); break;
        }
        break;
      }
      return GV;
    }
    default:
      break;
    }
    cerr << "ConstantExpr not handled: " << *CE << "\n";
    abort();
  }

  GenericValue Result;
  switch (C->getType()->getTypeID()) {
  case Type::FloatTyID: 
    Result.FloatVal = cast<ConstantFP>(C)->getValueAPF().convertToFloat(); 
    break;
  case Type::DoubleTyID:
    Result.DoubleVal = cast<ConstantFP>(C)->getValueAPF().convertToDouble();
    break;
  case Type::IntegerTyID:
    Result.IntVal = cast<ConstantInt>(C)->getValue();
    break;
  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(C))
      Result.PointerVal = 0;
    else if (const Function *F = dyn_cast<Function>(C))
      Result = PTOGV(getPointerToFunctionOrStub(const_cast<Function*>(F)));
    else if (const GlobalVariable* GV = dyn_cast<GlobalVariable>(C))
      Result = PTOGV(getOrEmitGlobalVariable(const_cast<GlobalVariable*>(GV)));
    else
      assert(0 && "Unknown constant pointer type!");
    break;
  default:
    cerr << "ERROR: Constant unimplemented for type: " << *C->getType() << "\n";
    abort();
  }
  return Result;
}

/// StoreValueToMemory - Stores the data in Val of type Ty at address Ptr.  Ptr
/// is the address of the memory at which to store Val, cast to GenericValue *.
/// It is not a pointer to a GenericValue containing the address at which to
/// store Val.
///
void ExecutionEngine::StoreValueToMemory(const GenericValue &Val, GenericValue *Ptr,
                                         const Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    unsigned BitWidth = cast<IntegerType>(Ty)->getBitWidth();
    GenericValue TmpVal = Val;
    if (BitWidth <= 8)
      *((uint8_t*)Ptr) = uint8_t(Val.IntVal.getZExtValue());
    else if (BitWidth <= 16) {
      *((uint16_t*)Ptr) = uint16_t(Val.IntVal.getZExtValue());
    } else if (BitWidth <= 32) {
      *((uint32_t*)Ptr) = uint32_t(Val.IntVal.getZExtValue());
    } else if (BitWidth <= 64) {
      *((uint64_t*)Ptr) = uint64_t(Val.IntVal.getZExtValue());
    } else {
      uint64_t *Dest = (uint64_t*)Ptr;
      const uint64_t *Src = Val.IntVal.getRawData();
      for (uint32_t i = 0; i < Val.IntVal.getNumWords(); ++i)
        Dest[i] = Src[i];
    }
    break;
  }
  case Type::FloatTyID:
    *((float*)Ptr) = Val.FloatVal;
    break;
  case Type::DoubleTyID:
    *((double*)Ptr) = Val.DoubleVal;
    break;
  case Type::PointerTyID: 
    *((PointerTy*)Ptr) = Val.PointerVal;
    break;
  default:
    cerr << "Cannot store value of type " << *Ty << "!\n";
  }
}

/// FIXME: document
///
void ExecutionEngine::LoadValueFromMemory(GenericValue &Result, 
                                                  GenericValue *Ptr,
                                                  const Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    unsigned BitWidth = cast<IntegerType>(Ty)->getBitWidth();
    if (BitWidth <= 8)
      Result.IntVal = APInt(BitWidth, *((uint8_t*)Ptr));
    else if (BitWidth <= 16) {
      Result.IntVal = APInt(BitWidth, *((uint16_t*)Ptr));
    } else if (BitWidth <= 32) {
      Result.IntVal = APInt(BitWidth, *((uint32_t*)Ptr));
    } else if (BitWidth <= 64) {
      Result.IntVal = APInt(BitWidth, *((uint64_t*)Ptr));
    } else
      Result.IntVal = APInt(BitWidth, (BitWidth+63)/64, (uint64_t*)Ptr);
    break;
  }
  case Type::FloatTyID:
    Result.FloatVal = *((float*)Ptr);
    break;
  case Type::DoubleTyID:
    Result.DoubleVal = *((double*)Ptr); 
    break;
  case Type::PointerTyID: 
    Result.PointerVal = *((PointerTy*)Ptr);
    break;
  default:
    cerr << "Cannot load value of type " << *Ty << "!\n";
    abort();
  }
}

// InitializeMemory - Recursive function to apply a Constant value into the
// specified memory location...
//
void ExecutionEngine::InitializeMemory(const Constant *Init, void *Addr) {
  if (isa<UndefValue>(Init)) {
    return;
  } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(Init)) {
    unsigned ElementSize =
      getTargetData()->getTypeSize(CP->getType()->getElementType());
    for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
      InitializeMemory(CP->getOperand(i), (char*)Addr+i*ElementSize);
    return;
  } else if (Init->getType()->isFirstClassType()) {
    GenericValue Val = getConstantValue(Init);
    StoreValueToMemory(Val, (GenericValue*)Addr, Init->getType());
    return;
  } else if (isa<ConstantAggregateZero>(Init)) {
    memset(Addr, 0, (size_t)getTargetData()->getTypeSize(Init->getType()));
    return;
  }

  switch (Init->getType()->getTypeID()) {
  case Type::ArrayTyID: {
    const ConstantArray *CPA = cast<ConstantArray>(Init);
    unsigned ElementSize =
      getTargetData()->getTypeSize(CPA->getType()->getElementType());
    for (unsigned i = 0, e = CPA->getNumOperands(); i != e; ++i)
      InitializeMemory(CPA->getOperand(i), (char*)Addr+i*ElementSize);
    return;
  }

  case Type::StructTyID: {
    const ConstantStruct *CPS = cast<ConstantStruct>(Init);
    const StructLayout *SL =
      getTargetData()->getStructLayout(cast<StructType>(CPS->getType()));
    for (unsigned i = 0, e = CPS->getNumOperands(); i != e; ++i)
      InitializeMemory(CPS->getOperand(i), (char*)Addr+SL->getElementOffset(i));
    return;
  }

  default:
    cerr << "Bad Type: " << *Init->getType() << "\n";
    assert(0 && "Unknown constant type to initialize memory with!");
  }
}

/// EmitGlobals - Emit all of the global variables to memory, storing their
/// addresses into GlobalAddress.  This must make sure to copy the contents of
/// their initializers into the memory.
///
void ExecutionEngine::emitGlobals() {
  const TargetData *TD = getTargetData();

  // Loop over all of the global variables in the program, allocating the memory
  // to hold them.  If there is more than one module, do a prepass over globals
  // to figure out how the different modules should link together.
  //
  std::map<std::pair<std::string, const Type*>,
           const GlobalValue*> LinkedGlobalsMap;

  if (Modules.size() != 1) {
    for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
      Module &M = *Modules[m]->getModule();
      for (Module::const_global_iterator I = M.global_begin(),
           E = M.global_end(); I != E; ++I) {
        const GlobalValue *GV = I;
        if (GV->hasInternalLinkage() || GV->isDeclaration() ||
            GV->hasAppendingLinkage() || !GV->hasName())
          continue;// Ignore external globals and globals with internal linkage.
          
        const GlobalValue *&GVEntry = 
          LinkedGlobalsMap[std::make_pair(GV->getName(), GV->getType())];

        // If this is the first time we've seen this global, it is the canonical
        // version.
        if (!GVEntry) {
          GVEntry = GV;
          continue;
        }
        
        // If the existing global is strong, never replace it.
        if (GVEntry->hasExternalLinkage() ||
            GVEntry->hasDLLImportLinkage() ||
            GVEntry->hasDLLExportLinkage())
          continue;
        
        // Otherwise, we know it's linkonce/weak, replace it if this is a strong
        // symbol.
        if (GV->hasExternalLinkage() || GVEntry->hasExternalWeakLinkage())
          GVEntry = GV;
      }
    }
  }
  
  std::vector<const GlobalValue*> NonCanonicalGlobals;
  for (unsigned m = 0, e = Modules.size(); m != e; ++m) {
    Module &M = *Modules[m]->getModule();
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      // In the multi-module case, see what this global maps to.
      if (!LinkedGlobalsMap.empty()) {
        if (const GlobalValue *GVEntry = 
              LinkedGlobalsMap[std::make_pair(I->getName(), I->getType())]) {
          // If something else is the canonical global, ignore this one.
          if (GVEntry != &*I) {
            NonCanonicalGlobals.push_back(I);
            continue;
          }
        }
      }
      
      if (!I->isDeclaration()) {
        // Get the type of the global.
        const Type *Ty = I->getType()->getElementType();

        // Allocate some memory for it!
        unsigned Size = TD->getTypeSize(Ty);
        addGlobalMapping(I, new char[Size]);
      } else {
        // External variable reference. Try to use the dynamic loader to
        // get a pointer to it.
        if (void *SymAddr =
            sys::DynamicLibrary::SearchForAddressOfSymbol(I->getName().c_str()))
          addGlobalMapping(I, SymAddr);
        else {
          cerr << "Could not resolve external global address: "
               << I->getName() << "\n";
          abort();
        }
      }
    }
    
    // If there are multiple modules, map the non-canonical globals to their
    // canonical location.
    if (!NonCanonicalGlobals.empty()) {
      for (unsigned i = 0, e = NonCanonicalGlobals.size(); i != e; ++i) {
        const GlobalValue *GV = NonCanonicalGlobals[i];
        const GlobalValue *CGV =
          LinkedGlobalsMap[std::make_pair(GV->getName(), GV->getType())];
        void *Ptr = getPointerToGlobalIfAvailable(CGV);
        assert(Ptr && "Canonical global wasn't codegen'd!");
        addGlobalMapping(GV, getPointerToGlobalIfAvailable(CGV));
      }
    }
    
    // Now that all of the globals are set up in memory, loop through them all 
    // and initialize their contents.
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      if (!I->isDeclaration()) {
        if (!LinkedGlobalsMap.empty()) {
          if (const GlobalValue *GVEntry = 
                LinkedGlobalsMap[std::make_pair(I->getName(), I->getType())])
            if (GVEntry != &*I)  // Not the canonical variable.
              continue;
        }
        EmitGlobalVariable(I);
      }
    }
  }
}

// EmitGlobalVariable - This method emits the specified global variable to the
// address specified in GlobalAddresses, or allocates new memory if it's not
// already in the map.
void ExecutionEngine::EmitGlobalVariable(const GlobalVariable *GV) {
  void *GA = getPointerToGlobalIfAvailable(GV);
  DOUT << "Global '" << GV->getName() << "' -> " << GA << "\n";

  const Type *ElTy = GV->getType()->getElementType();
  size_t GVSize = (size_t)getTargetData()->getTypeSize(ElTy);
  if (GA == 0) {
    // If it's not already specified, allocate memory for the global.
    GA = new char[GVSize];
    addGlobalMapping(GV, GA);
  }

  InitializeMemory(GV->getInitializer(), GA);
  NumInitBytes += (unsigned)GVSize;
  ++NumGlobals;
}

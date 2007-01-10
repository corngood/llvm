//===-- AsmWriter.cpp - Printing LLVM as an assembly file -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/Assembly/Writer.h
//
// Note that these routines must be extremely tolerant of various errors in the
// LLVM code, because it can be used for debugging transformations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Writer.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Assembly/AsmAnnotationWriter.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instruction.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Streams.h"
#include <algorithm>
using namespace llvm;

namespace llvm {

// Make virtual table appear in this compilation unit.
AssemblyAnnotationWriter::~AssemblyAnnotationWriter() {}

/// This class provides computation of slot numbers for LLVM Assembly writing.
/// @brief LLVM Assembly Writing Slot Computation.
class SlotMachine {

/// @name Types
/// @{
public:

  /// @brief A mapping of Values to slot numbers
  typedef std::map<const Value*, unsigned> ValueMap;

  /// @brief A plane with next slot number and ValueMap
  struct ValuePlane {
    unsigned next_slot;        ///< The next slot number to use
    ValueMap map;              ///< The map of Value* -> unsigned
    ValuePlane() { next_slot = 0; } ///< Make sure we start at 0
  };

  /// @brief The map of planes by Type
  typedef std::map<const Type*, ValuePlane> TypedPlanes;

/// @}
/// @name Constructors
/// @{
public:
  /// @brief Construct from a module
  SlotMachine(const Module *M);

  /// @brief Construct from a function, starting out in incorp state.
  SlotMachine(const Function *F);

/// @}
/// @name Accessors
/// @{
public:
  /// Return the slot number of the specified value in it's type
  /// plane.  Its an error to ask for something not in the SlotMachine.
  /// Its an error to ask for a Type*
  int getSlot(const Value *V);

/// @}
/// @name Mutators
/// @{
public:
  /// If you'd like to deal with a function instead of just a module, use
  /// this method to get its data into the SlotMachine.
  void incorporateFunction(const Function *F) {
    TheFunction = F;
    FunctionProcessed = false;
  }

  /// After calling incorporateFunction, use this method to remove the
  /// most recently incorporated function from the SlotMachine. This
  /// will reset the state of the machine back to just the module contents.
  void purgeFunction();

/// @}
/// @name Implementation Details
/// @{
private:
  /// This function does the actual initialization.
  inline void initialize();

  /// CreateModuleSlot - Insert the specified GlobalValue* into the slot table.
  void CreateModuleSlot(const GlobalValue *V);
  
  /// CreateFunctionSlot - Insert the specified Value* into the slot table.
  void CreateFunctionSlot(const Value *V);

  /// Add all of the module level global variables (and their initializers)
  /// and function declarations, but not the contents of those functions.
  void processModule();

  /// Add all of the functions arguments, basic blocks, and instructions
  void processFunction();

  SlotMachine(const SlotMachine &);  // DO NOT IMPLEMENT
  void operator=(const SlotMachine &);  // DO NOT IMPLEMENT

/// @}
/// @name Data
/// @{
public:

  /// @brief The module for which we are holding slot numbers
  const Module* TheModule;

  /// @brief The function for which we are holding slot numbers
  const Function* TheFunction;
  bool FunctionProcessed;

  /// @brief The TypePlanes map for the module level data
  TypedPlanes mMap;

  /// @brief The TypePlanes map for the function level data
  TypedPlanes fMap;

/// @}

};

}  // end namespace llvm

static RegisterPass<PrintModulePass>
X("printm", "Print module to stderr");
static RegisterPass<PrintFunctionPass>
Y("print","Print function to stderr");

static void WriteAsOperandInternal(std::ostream &Out, const Value *V,
                               std::map<const Type *, std::string> &TypeTable,
                                   SlotMachine *Machine);

static const Module *getModuleFromVal(const Value *V) {
  if (const Argument *MA = dyn_cast<Argument>(V))
    return MA->getParent() ? MA->getParent()->getParent() : 0;
  else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent() ? BB->getParent()->getParent() : 0;
  else if (const Instruction *I = dyn_cast<Instruction>(V)) {
    const Function *M = I->getParent() ? I->getParent()->getParent() : 0;
    return M ? M->getParent() : 0;
  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return GV->getParent();
  return 0;
}

static SlotMachine *createSlotMachine(const Value *V) {
  if (const Argument *FA = dyn_cast<Argument>(V)) {
    return new SlotMachine(FA->getParent());
  } else if (const Instruction *I = dyn_cast<Instruction>(V)) {
    return new SlotMachine(I->getParent()->getParent());
  } else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    return new SlotMachine(BB->getParent());
  } else if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V)){
    return new SlotMachine(GV->getParent());
  } else if (const Function *Func = dyn_cast<Function>(V)) {
    return new SlotMachine(Func);
  }
  return 0;
}

// getLLVMName - Turn the specified string into an 'LLVM name', which is either
// prefixed with % (if the string only contains simple characters) or is
// surrounded with ""'s (if it has special chars in it).
static std::string getLLVMName(const std::string &Name,
                               bool prefixName = true) {
  assert(!Name.empty() && "Cannot get empty name!");

  // First character cannot start with a number...
  if (Name[0] >= '0' && Name[0] <= '9')
    return "\"" + Name + "\"";

  // Scan to see if we have any characters that are not on the "white list"
  for (unsigned i = 0, e = Name.size(); i != e; ++i) {
    char C = Name[i];
    assert(C != '"' && "Illegal character in LLVM value name!");
    if ((C < 'a' || C > 'z') && (C < 'A' || C > 'Z') && (C < '0' || C > '9') &&
        C != '-' && C != '.' && C != '_')
      return "\"" + Name + "\"";
  }

  // If we get here, then the identifier is legal to use as a "VarID".
  if (prefixName)
    return "%"+Name;
  else
    return Name;
}


/// fillTypeNameTable - If the module has a symbol table, take all global types
/// and stuff their names into the TypeNames map.
///
static void fillTypeNameTable(const Module *M,
                              std::map<const Type *, std::string> &TypeNames) {
  if (!M) return;
  const TypeSymbolTable &ST = M->getTypeSymbolTable();
  TypeSymbolTable::const_iterator TI = ST.begin();
  for (; TI != ST.end(); ++TI) {
    // As a heuristic, don't insert pointer to primitive types, because
    // they are used too often to have a single useful name.
    //
    const Type *Ty = cast<Type>(TI->second);
    if (!isa<PointerType>(Ty) ||
        !cast<PointerType>(Ty)->getElementType()->isPrimitiveType() ||
        isa<OpaqueType>(cast<PointerType>(Ty)->getElementType()))
      TypeNames.insert(std::make_pair(Ty, getLLVMName(TI->first)));
  }
}



static void calcTypeName(const Type *Ty,
                         std::vector<const Type *> &TypeStack,
                         std::map<const Type *, std::string> &TypeNames,
                         std::string & Result){
  if (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty)) {
    Result += Ty->getDescription();  // Base case
    return;
  }

  // Check to see if the type is named.
  std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) {
    Result += I->second;
    return;
  }

  if (isa<OpaqueType>(Ty)) {
    Result += "opaque";
    return;
  }

  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type

  // This is another base case for the recursion.  In this case, we know
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  if (Slot < CurSize) {
    Result += "\\" + utostr(CurSize-Slot);     // Here's the upreference
    return;
  }

  TypeStack.push_back(Ty);    // Recursive case: Add us to the stack..

  switch (Ty->getTypeID()) {
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    calcTypeName(FTy->getReturnType(), TypeStack, TypeNames, Result);
    Result += " (";
    unsigned Idx = 1;
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Result += ", ";
      calcTypeName(*I, TypeStack, TypeNames, Result);
      if (FTy->getParamAttrs(Idx)) {
        Result += + " ";
        Result += FunctionType::getParamAttrsText(FTy->getParamAttrs(Idx));
      }
      Idx++;
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) Result += ", ";
      Result += "...";
    }
    Result += ")";
    if (FTy->getParamAttrs(0)) {
      Result += " ";
      Result += FunctionType::getParamAttrsText(FTy->getParamAttrs(0));
    }
    break;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    if (STy->isPacked())
      Result += '<';
    Result += "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        Result += ", ";
      calcTypeName(*I, TypeStack, TypeNames, Result);
    }
    Result += " }";
    if (STy->isPacked())
      Result += '>';
    break;
  }
  case Type::PointerTyID:
    calcTypeName(cast<PointerType>(Ty)->getElementType(),
                          TypeStack, TypeNames, Result);
    Result += "*";
    break;
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    Result += "[" + utostr(ATy->getNumElements()) + " x ";
    calcTypeName(ATy->getElementType(), TypeStack, TypeNames, Result);
    Result += "]";
    break;
  }
  case Type::PackedTyID: {
    const PackedType *PTy = cast<PackedType>(Ty);
    Result += "<" + utostr(PTy->getNumElements()) + " x ";
    calcTypeName(PTy->getElementType(), TypeStack, TypeNames, Result);
    Result += ">";
    break;
  }
  case Type::OpaqueTyID:
    Result += "opaque";
    break;
  default:
    Result += "<unrecognized-type>";
    break;
  }

  TypeStack.pop_back();       // Remove self from stack...
}


/// printTypeInt - The internal guts of printing out a type that has a
/// potentially named portion.
///
static std::ostream &printTypeInt(std::ostream &Out, const Type *Ty,
                              std::map<const Type *, std::string> &TypeNames) {
  // Primitive types always print out their description, regardless of whether
  // they have been named or not.
  //
  if (Ty->isPrimitiveType() && !isa<OpaqueType>(Ty))
    return Out << Ty->getDescription();

  // Check to see if the type is named.
  std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
  if (I != TypeNames.end()) return Out << I->second;

  // Otherwise we have a type that has not been named but is a derived type.
  // Carefully recurse the type hierarchy to print out any contained symbolic
  // names.
  //
  std::vector<const Type *> TypeStack;
  std::string TypeName;
  calcTypeName(Ty, TypeStack, TypeNames, TypeName);
  TypeNames.insert(std::make_pair(Ty, TypeName));//Cache type name for later use
  return (Out << TypeName);
}


/// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
/// type, iff there is an entry in the modules symbol table for the specified
/// type or one of it's component types. This is slower than a simple x << Type
///
std::ostream &llvm::WriteTypeSymbolic(std::ostream &Out, const Type *Ty,
                                      const Module *M) {
  Out << ' ';

  // If they want us to print out a type, but there is no context, we can't
  // print it symbolically.
  if (!M)
    return Out << Ty->getDescription();
    
  std::map<const Type *, std::string> TypeNames;
  fillTypeNameTable(M, TypeNames);
  return printTypeInt(Out, Ty, TypeNames);
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const std::string &Str, std::ostream &Out) {
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    unsigned char C = Str[i];
    if (isprint(C) && C != '"' && C != '\\') {
      Out << C;
    } else {
      Out << '\\'
          << (char) ((C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A'))
          << (char)(((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A'));
    }
  }
}

static const char *getPredicateText(unsigned predicate) {
  const char * pred = "unknown";
  switch (predicate) {
    case FCmpInst::FCMP_FALSE: pred = "false"; break;
    case FCmpInst::FCMP_OEQ:   pred = "oeq"; break;
    case FCmpInst::FCMP_OGT:   pred = "ogt"; break;
    case FCmpInst::FCMP_OGE:   pred = "oge"; break;
    case FCmpInst::FCMP_OLT:   pred = "olt"; break;
    case FCmpInst::FCMP_OLE:   pred = "ole"; break;
    case FCmpInst::FCMP_ONE:   pred = "one"; break;
    case FCmpInst::FCMP_ORD:   pred = "ord"; break;
    case FCmpInst::FCMP_UNO:   pred = "uno"; break;
    case FCmpInst::FCMP_UEQ:   pred = "ueq"; break;
    case FCmpInst::FCMP_UGT:   pred = "ugt"; break;
    case FCmpInst::FCMP_UGE:   pred = "uge"; break;
    case FCmpInst::FCMP_ULT:   pred = "ult"; break;
    case FCmpInst::FCMP_ULE:   pred = "ule"; break;
    case FCmpInst::FCMP_UNE:   pred = "une"; break;
    case FCmpInst::FCMP_TRUE:  pred = "true"; break;
    case ICmpInst::ICMP_EQ:    pred = "eq"; break;
    case ICmpInst::ICMP_NE:    pred = "ne"; break;
    case ICmpInst::ICMP_SGT:   pred = "sgt"; break;
    case ICmpInst::ICMP_SGE:   pred = "sge"; break;
    case ICmpInst::ICMP_SLT:   pred = "slt"; break;
    case ICmpInst::ICMP_SLE:   pred = "sle"; break;
    case ICmpInst::ICMP_UGT:   pred = "ugt"; break;
    case ICmpInst::ICMP_UGE:   pred = "uge"; break;
    case ICmpInst::ICMP_ULT:   pred = "ult"; break;
    case ICmpInst::ICMP_ULE:   pred = "ule"; break;
  }
  return pred;
}

/// @brief Internal constant writer.
static void WriteConstantInt(std::ostream &Out, const Constant *CV,
                             std::map<const Type *, std::string> &TypeTable,
                             SlotMachine *Machine) {
  const int IndentSize = 4;
  static std::string Indent = "\n";
  if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    Out << (CB->getValue() ? "true" : "false");
  } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    Out << CI->getSExtValue();
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    // We would like to output the FP constant value in exponential notation,
    // but we cannot do this if doing so will lose precision.  Check here to
    // make sure that we only output it in exponential format if we can parse
    // the value back and get the same value.
    //
    std::string StrVal = ftostr(CFP->getValue());

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check that
    // the string matches the "[-+]?[0-9]" regex.
    //
    if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
        ((StrVal[0] == '-' || StrVal[0] == '+') &&
         (StrVal[1] >= '0' && StrVal[1] <= '9')))
      // Reparse stringized version!
      if (atof(StrVal.c_str()) == CFP->getValue()) {
        Out << StrVal;
        return;
      }

    // Otherwise we could not reparse it to exactly the same value, so we must
    // output the string in hexadecimal format!
    assert(sizeof(double) == sizeof(uint64_t) &&
           "assuming that double is 64 bits!");
    Out << "0x" << utohexstr(DoubleToBits(CFP->getValue()));

  } else if (isa<ConstantAggregateZero>(CV)) {
    Out << "zeroinitializer";
  } else if (const ConstantArray *CA = dyn_cast<ConstantArray>(CV)) {
    // As a special case, print the array as a string if it is an array of
    // ubytes or an array of sbytes with positive values.
    //
    const Type *ETy = CA->getType()->getElementType();
    if (CA->isString()) {
      Out << "c\"";
      PrintEscapedString(CA->getAsString(), Out);
      Out << "\"";

    } else {                // Cannot output in string format...
      Out << '[';
      if (CA->getNumOperands()) {
        Out << ' ';
        printTypeInt(Out, ETy, TypeTable);
        WriteAsOperandInternal(Out, CA->getOperand(0),
                               TypeTable, Machine);
        for (unsigned i = 1, e = CA->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printTypeInt(Out, ETy, TypeTable);
          WriteAsOperandInternal(Out, CA->getOperand(i), TypeTable, Machine);
        }
      }
      Out << " ]";
    }
  } else if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(CV)) {
    if (CS->getType()->isPacked())
      Out << '<';
    Out << '{';
    unsigned N = CS->getNumOperands();
    if (N) {
      if (N > 2) {
        Indent += std::string(IndentSize, ' ');
        Out << Indent;
      } else {
        Out << ' ';
      }
      printTypeInt(Out, CS->getOperand(0)->getType(), TypeTable);

      WriteAsOperandInternal(Out, CS->getOperand(0), TypeTable, Machine);

      for (unsigned i = 1; i < N; i++) {
        Out << ", ";
        if (N > 2) Out << Indent;
        printTypeInt(Out, CS->getOperand(i)->getType(), TypeTable);

        WriteAsOperandInternal(Out, CS->getOperand(i), TypeTable, Machine);
      }
      if (N > 2) Indent.resize(Indent.size() - IndentSize);
    }
 
    Out << " }";
    if (CS->getType()->isPacked())
      Out << '>';
  } else if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(CV)) {
      const Type *ETy = CP->getType()->getElementType();
      assert(CP->getNumOperands() > 0 &&
             "Number of operands for a PackedConst must be > 0");
      Out << '<';
      Out << ' ';
      printTypeInt(Out, ETy, TypeTable);
      WriteAsOperandInternal(Out, CP->getOperand(0), TypeTable, Machine);
      for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
          Out << ", ";
          printTypeInt(Out, ETy, TypeTable);
          WriteAsOperandInternal(Out, CP->getOperand(i), TypeTable, Machine);
      }
      Out << " >";
  } else if (isa<ConstantPointerNull>(CV)) {
    Out << "null";

  } else if (isa<UndefValue>(CV)) {
    Out << "undef";

  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    Out << CE->getOpcodeName();
    if (CE->isCompare())
      Out << " " << getPredicateText(CE->getPredicate());
    Out << " (";

    for (User::const_op_iterator OI=CE->op_begin(); OI != CE->op_end(); ++OI) {
      printTypeInt(Out, (*OI)->getType(), TypeTable);
      WriteAsOperandInternal(Out, *OI, TypeTable, Machine);
      if (OI+1 != CE->op_end())
        Out << ", ";
    }

    if (CE->isCast()) {
      Out << " to ";
      printTypeInt(Out, CE->getType(), TypeTable);
    }

    Out << ')';

  } else {
    Out << "<placeholder or erroneous Constant>";
  }
}


/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
static void WriteAsOperandInternal(std::ostream &Out, const Value *V,
                                  std::map<const Type*, std::string> &TypeTable,
                                   SlotMachine *Machine) {
  Out << ' ';
  if (V->hasName())
    Out << getLLVMName(V->getName());
  else {
    const Constant *CV = dyn_cast<Constant>(V);
    if (CV && !isa<GlobalValue>(CV)) {
      WriteConstantInt(Out, CV, TypeTable, Machine);
    } else if (const InlineAsm *IA = dyn_cast<InlineAsm>(V)) {
      Out << "asm ";
      if (IA->hasSideEffects())
        Out << "sideeffect ";
      Out << '"';
      PrintEscapedString(IA->getAsmString(), Out);
      Out << "\", \"";
      PrintEscapedString(IA->getConstraintString(), Out);
      Out << '"';
    } else {
      int Slot;
      if (Machine) {
        Slot = Machine->getSlot(V);
      } else {
        Machine = createSlotMachine(V);
        if (Machine)
          Slot = Machine->getSlot(V);
        else
          Slot = -1;
        delete Machine;
      }
      if (Slot != -1)
        Out << '%' << Slot;
      else
        Out << "<badref>";
    }
  }
}

/// WriteAsOperand - Write the name of the specified value out to the specified
/// ostream.  This can be useful when you just want to print int %reg126, not
/// the whole instruction that generated it.
///
std::ostream &llvm::WriteAsOperand(std::ostream &Out, const Value *V,
                                   bool PrintType, const Module *Context) {
  std::map<const Type *, std::string> TypeNames;
  if (Context == 0) Context = getModuleFromVal(V);

  if (Context)
    fillTypeNameTable(Context, TypeNames);

  if (PrintType)
    printTypeInt(Out, V->getType(), TypeNames);

  WriteAsOperandInternal(Out, V, TypeNames, 0);
  return Out;
}


namespace llvm {

class AssemblyWriter {
  std::ostream &Out;
  SlotMachine &Machine;
  const Module *TheModule;
  std::map<const Type *, std::string> TypeNames;
  AssemblyAnnotationWriter *AnnotationWriter;
public:
  inline AssemblyWriter(std::ostream &o, SlotMachine &Mac, const Module *M,
                        AssemblyAnnotationWriter *AAW)
    : Out(o), Machine(Mac), TheModule(M), AnnotationWriter(AAW) {

    // If the module has a symbol table, take all global types and stuff their
    // names into the TypeNames map.
    //
    fillTypeNameTable(M, TypeNames);
  }

  inline void write(const Module *M)         { printModule(M);      }
  inline void write(const GlobalVariable *G) { printGlobal(G);      }
  inline void write(const Function *F)       { printFunction(F);    }
  inline void write(const BasicBlock *BB)    { printBasicBlock(BB); }
  inline void write(const Instruction *I)    { printInstruction(*I); }
  inline void write(const Constant *CPV)     { printConstant(CPV);  }
  inline void write(const Type *Ty)          { printType(Ty);       }

  void writeOperand(const Value *Op, bool PrintType);

  const Module* getModule() { return TheModule; }

private:
  void printModule(const Module *M);
  void printTypeSymbolTable(const TypeSymbolTable &ST);
  void printValueSymbolTable(const SymbolTable &ST);
  void printConstant(const Constant *CPV);
  void printGlobal(const GlobalVariable *GV);
  void printFunction(const Function *F);
  void printArgument(const Argument *FA, FunctionType::ParameterAttributes A);
  void printBasicBlock(const BasicBlock *BB);
  void printInstruction(const Instruction &I);

  // printType - Go to extreme measures to attempt to print out a short,
  // symbolic version of a type name.
  //
  std::ostream &printType(const Type *Ty) {
    return printTypeInt(Out, Ty, TypeNames);
  }

  // printTypeAtLeastOneLevel - Print out one level of the possibly complex type
  // without considering any symbolic types that we may have equal to it.
  //
  std::ostream &printTypeAtLeastOneLevel(const Type *Ty);

  // printInfoComment - Print a little comment after the instruction indicating
  // which slot it occupies.
  void printInfoComment(const Value &V);
};
}  // end of llvm namespace

/// printTypeAtLeastOneLevel - Print out one level of the possibly complex type
/// without considering any symbolic types that we may have equal to it.
///
std::ostream &AssemblyWriter::printTypeAtLeastOneLevel(const Type *Ty) {
  if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    printType(FTy->getReturnType());
    Out << " (";
    unsigned Idx = 1;
    for (FunctionType::param_iterator I = FTy->param_begin(),
           E = FTy->param_end(); I != E; ++I) {
      if (I != FTy->param_begin())
        Out << ", ";
      printType(*I);
      if (FTy->getParamAttrs(Idx)) {
        Out << " " << FunctionType::getParamAttrsText(FTy->getParamAttrs(Idx));
      }
      Idx++;
    }
    if (FTy->isVarArg()) {
      if (FTy->getNumParams()) Out << ", ";
      Out << "...";
    }
    Out << ')';
    if (FTy->getParamAttrs(0))
      Out << ' ' << FunctionType::getParamAttrsText(FTy->getParamAttrs(0));
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    if (STy->isPacked())
      Out << '<';
    Out << "{ ";
    for (StructType::element_iterator I = STy->element_begin(),
           E = STy->element_end(); I != E; ++I) {
      if (I != STy->element_begin())
        Out << ", ";
      printType(*I);
    }
    Out << " }";
    if (STy->isPacked())
      Out << '>';
  } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    printType(PTy->getElementType()) << '*';
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Out << '[' << ATy->getNumElements() << " x ";
    printType(ATy->getElementType()) << ']';
  } else if (const PackedType *PTy = dyn_cast<PackedType>(Ty)) {
    Out << '<' << PTy->getNumElements() << " x ";
    printType(PTy->getElementType()) << '>';
  }
  else if (isa<OpaqueType>(Ty)) {
    Out << "opaque";
  } else {
    if (!Ty->isPrimitiveType())
      Out << "<unknown derived type>";
    printType(Ty);
  }
  return Out;
}


void AssemblyWriter::writeOperand(const Value *Operand, bool PrintType) {
  if (Operand == 0) {
    Out << "<null operand!>";
  } else {
    if (PrintType) { Out << ' '; printType(Operand->getType()); }
    WriteAsOperandInternal(Out, Operand, TypeNames, &Machine);
  }
}


void AssemblyWriter::printModule(const Module *M) {
  if (!M->getModuleIdentifier().empty() &&
      // Don't print the ID if it will start a new line (which would
      // require a comment char before it).
      M->getModuleIdentifier().find('\n') == std::string::npos)
    Out << "; ModuleID = '" << M->getModuleIdentifier() << "'\n";

  if (!M->getDataLayout().empty())
    Out << "target datalayout = \"" << M->getDataLayout() << "\"\n";

  switch (M->getEndianness()) {
  case Module::LittleEndian: Out << "target endian = little\n"; break;
  case Module::BigEndian:    Out << "target endian = big\n";    break;
  case Module::AnyEndianness: break;
  }
  switch (M->getPointerSize()) {
  case Module::Pointer32:    Out << "target pointersize = 32\n"; break;
  case Module::Pointer64:    Out << "target pointersize = 64\n"; break;
  case Module::AnyPointerSize: break;
  }
  if (!M->getTargetTriple().empty())
    Out << "target triple = \"" << M->getTargetTriple() << "\"\n";

  if (!M->getModuleInlineAsm().empty()) {
    // Split the string into lines, to make it easier to read the .ll file.
    std::string Asm = M->getModuleInlineAsm();
    size_t CurPos = 0;
    size_t NewLine = Asm.find_first_of('\n', CurPos);
    while (NewLine != std::string::npos) {
      // We found a newline, print the portion of the asm string from the
      // last newline up to this newline.
      Out << "module asm \"";
      PrintEscapedString(std::string(Asm.begin()+CurPos, Asm.begin()+NewLine),
                         Out);
      Out << "\"\n";
      CurPos = NewLine+1;
      NewLine = Asm.find_first_of('\n', CurPos);
    }
    Out << "module asm \"";
    PrintEscapedString(std::string(Asm.begin()+CurPos, Asm.end()), Out);
    Out << "\"\n";
  }
  
  // Loop over the dependent libraries and emit them.
  Module::lib_iterator LI = M->lib_begin();
  Module::lib_iterator LE = M->lib_end();
  if (LI != LE) {
    Out << "deplibs = [ ";
    while (LI != LE) {
      Out << '"' << *LI << '"';
      ++LI;
      if (LI != LE)
        Out << ", ";
    }
    Out << " ]\n";
  }

  // Loop over the symbol table, emitting all named constants.
  printTypeSymbolTable(M->getTypeSymbolTable());
  printValueSymbolTable(M->getValueSymbolTable());

  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    printGlobal(I);

  Out << "\nimplementation   ; Functions:\n";

  // Output all of the functions.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    printFunction(I);
}

void AssemblyWriter::printGlobal(const GlobalVariable *GV) {
  if (GV->hasName()) Out << getLLVMName(GV->getName()) << " = ";

  if (!GV->hasInitializer())
    switch (GV->getLinkage()) {
     case GlobalValue::DLLImportLinkage:   Out << "dllimport "; break;
     case GlobalValue::ExternalWeakLinkage: Out << "extern_weak "; break;
     default: Out << "external "; break;
    }
  else
    switch (GV->getLinkage()) {
    case GlobalValue::InternalLinkage:     Out << "internal "; break;
    case GlobalValue::LinkOnceLinkage:     Out << "linkonce "; break;
    case GlobalValue::WeakLinkage:         Out << "weak "; break;
    case GlobalValue::AppendingLinkage:    Out << "appending "; break;
    case GlobalValue::DLLImportLinkage:    Out << "dllimport "; break;
    case GlobalValue::DLLExportLinkage:    Out << "dllexport "; break;     
    case GlobalValue::ExternalWeakLinkage: Out << "extern_weak "; break;
    case GlobalValue::ExternalLinkage:     break;
    case GlobalValue::GhostLinkage:
      cerr << "GhostLinkage not allowed in AsmWriter!\n";
      abort();
    }

  Out << (GV->isConstant() ? "constant " : "global ");
  printType(GV->getType()->getElementType());

  if (GV->hasInitializer()) {
    Constant* C = cast<Constant>(GV->getInitializer());
    assert(C &&  "GlobalVar initializer isn't constant?");
    writeOperand(GV->getInitializer(), false);
  }
  
  if (GV->hasSection())
    Out << ", section \"" << GV->getSection() << '"';
  if (GV->getAlignment())
    Out << ", align " << GV->getAlignment();
  
  printInfoComment(*GV);
  Out << "\n";
}

void AssemblyWriter::printTypeSymbolTable(const TypeSymbolTable &ST) {
  // Print the types.
  for (TypeSymbolTable::const_iterator TI = ST.begin(), TE = ST.end();
       TI != TE; ++TI) {
    Out << "\t" << getLLVMName(TI->first) << " = type ";

    // Make sure we print out at least one level of the type structure, so
    // that we do not get %FILE = type %FILE
    //
    printTypeAtLeastOneLevel(TI->second) << "\n";
  }
}

// printSymbolTable - Run through symbol table looking for constants
// and types. Emit their declarations.
void AssemblyWriter::printValueSymbolTable(const SymbolTable &ST) {

  // Print the constants, in type plane order.
  for (SymbolTable::plane_const_iterator PI = ST.plane_begin();
       PI != ST.plane_end(); ++PI) {
    SymbolTable::value_const_iterator VI = ST.value_begin(PI->first);
    SymbolTable::value_const_iterator VE = ST.value_end(PI->first);

    for (; VI != VE; ++VI) {
      const Value* V = VI->second;
      const Constant *CPV = dyn_cast<Constant>(V) ;
      if (CPV && !isa<GlobalValue>(V)) {
        printConstant(CPV);
      }
    }
  }
}


/// printConstant - Print out a constant pool entry...
///
void AssemblyWriter::printConstant(const Constant *CPV) {
  // Don't print out unnamed constants, they will be inlined
  if (!CPV->hasName()) return;

  // Print out name...
  Out << "\t" << getLLVMName(CPV->getName()) << " =";

  // Write the value out now.
  writeOperand(CPV, true);

  printInfoComment(*CPV);
  Out << "\n";
}

/// printFunction - Print all aspects of a function.
///
void AssemblyWriter::printFunction(const Function *F) {
  // Print out the return type and name...
  Out << "\n";

  // Ensure that no local symbols conflict with global symbols.
  const_cast<Function*>(F)->renameLocalSymbols();

  if (AnnotationWriter) AnnotationWriter->emitFunctionAnnot(F, Out);

  if (F->isExternal())
    switch (F->getLinkage()) {
    case GlobalValue::DLLImportLinkage:    Out << "declare dllimport "; break;
    case GlobalValue::ExternalWeakLinkage: Out << "declare extern_weak "; break;
    default: Out << "declare ";
    }
  else {
    Out << "define ";
    switch (F->getLinkage()) {
    case GlobalValue::InternalLinkage:     Out << "internal "; break;
    case GlobalValue::LinkOnceLinkage:     Out << "linkonce "; break;
    case GlobalValue::WeakLinkage:         Out << "weak "; break;
    case GlobalValue::AppendingLinkage:    Out << "appending "; break;
    case GlobalValue::DLLImportLinkage:    Out << "dllimport "; break;
    case GlobalValue::DLLExportLinkage:    Out << "dllexport "; break;
    case GlobalValue::ExternalWeakLinkage: Out << "extern_weak "; break;      
    case GlobalValue::ExternalLinkage: break;
    case GlobalValue::GhostLinkage:
      cerr << "GhostLinkage not allowed in AsmWriter!\n";
      abort();
    }
  }

  // Print the calling convention.
  switch (F->getCallingConv()) {
  case CallingConv::C: break;   // default
  case CallingConv::CSRet:        Out << "csretcc "; break;
  case CallingConv::Fast:         Out << "fastcc "; break;
  case CallingConv::Cold:         Out << "coldcc "; break;
  case CallingConv::X86_StdCall:  Out << "x86_stdcallcc "; break;
  case CallingConv::X86_FastCall: Out << "x86_fastcallcc "; break; 
  default: Out << "cc" << F->getCallingConv() << " "; break;
  }

  const FunctionType *FT = F->getFunctionType();
  printType(F->getReturnType()) << ' ';
  if (!F->getName().empty())
    Out << getLLVMName(F->getName());
  else
    Out << "\"\"";
  Out << '(';
  Machine.incorporateFunction(F);

  // Loop over the arguments, printing them...

  unsigned Idx = 1;
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I) {
    // Insert commas as we go... the first arg doesn't get a comma
    if (I != F->arg_begin()) Out << ", ";
    printArgument(I, FT->getParamAttrs(Idx));
    Idx++;
  }

  // Finish printing arguments...
  if (FT->isVarArg()) {
    if (FT->getNumParams()) Out << ", ";
    Out << "...";  // Output varargs portion of signature!
  }
  Out << ')';
  if (FT->getParamAttrs(0))
    Out << ' ' << FunctionType::getParamAttrsText(FT->getParamAttrs(0));
  if (F->hasSection())
    Out << " section \"" << F->getSection() << '"';
  if (F->getAlignment())
    Out << " align " << F->getAlignment();

  if (F->isExternal()) {
    Out << "\n";
  } else {
    Out << " {";

    // Output all of its basic blocks... for the function
    for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
      printBasicBlock(I);

    Out << "}\n";
  }

  Machine.purgeFunction();
}

/// printArgument - This member is called for every argument that is passed into
/// the function.  Simply print it out
///
void AssemblyWriter::printArgument(const Argument *Arg, 
                                   FunctionType::ParameterAttributes attrs) {
  // Output type...
  printType(Arg->getType());

  if (attrs != FunctionType::NoAttributeSet)
    Out << ' ' << FunctionType::getParamAttrsText(attrs);

  // Output name, if available...
  if (Arg->hasName())
    Out << ' ' << getLLVMName(Arg->getName());
}

/// printBasicBlock - This member is called for each basic block in a method.
///
void AssemblyWriter::printBasicBlock(const BasicBlock *BB) {
  if (BB->hasName()) {              // Print out the label if it exists...
    Out << "\n" << getLLVMName(BB->getName(), false) << ':';
  } else if (!BB->use_empty()) {      // Don't print block # of no uses...
    Out << "\n; <label>:";
    int Slot = Machine.getSlot(BB);
    if (Slot != -1)
      Out << Slot;
    else
      Out << "<badref>";
  }

  if (BB->getParent() == 0)
    Out << "\t\t; Error: Block without parent!";
  else {
    if (BB != &BB->getParent()->front()) {  // Not the entry block?
      // Output predecessors for the block...
      Out << "\t\t;";
      pred_const_iterator PI = pred_begin(BB), PE = pred_end(BB);

      if (PI == PE) {
        Out << " No predecessors!";
      } else {
        Out << " preds =";
        writeOperand(*PI, false);
        for (++PI; PI != PE; ++PI) {
          Out << ',';
          writeOperand(*PI, false);
        }
      }
    }
  }

  Out << "\n";

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockStartAnnot(BB, Out);

  // Output all of the instructions in the basic block...
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    printInstruction(*I);

  if (AnnotationWriter) AnnotationWriter->emitBasicBlockEndAnnot(BB, Out);
}


/// printInfoComment - Print a little comment after the instruction indicating
/// which slot it occupies.
///
void AssemblyWriter::printInfoComment(const Value &V) {
  if (V.getType() != Type::VoidTy) {
    Out << "\t\t; <";
    printType(V.getType()) << '>';

    if (!V.hasName()) {
      int SlotNum = Machine.getSlot(&V);
      if (SlotNum == -1)
        Out << ":<badref>";
      else
        Out << ':' << SlotNum; // Print out the def slot taken.
    }
    Out << " [#uses=" << V.getNumUses() << ']';  // Output # uses
  }
}

// This member is called for each Instruction in a function..
void AssemblyWriter::printInstruction(const Instruction &I) {
  if (AnnotationWriter) AnnotationWriter->emitInstructionAnnot(&I, Out);

  Out << "\t";

  // Print out name if it exists...
  if (I.hasName())
    Out << getLLVMName(I.getName()) << " = ";

  // If this is a volatile load or store, print out the volatile marker.
  if ((isa<LoadInst>(I)  && cast<LoadInst>(I).isVolatile()) ||
      (isa<StoreInst>(I) && cast<StoreInst>(I).isVolatile())) {
      Out << "volatile ";
  } else if (isa<CallInst>(I) && cast<CallInst>(I).isTailCall()) {
    // If this is a call, check if it's a tail call.
    Out << "tail ";
  }

  // Print out the opcode...
  Out << I.getOpcodeName();

  // Print out the compare instruction predicates
  if (const FCmpInst *FCI = dyn_cast<FCmpInst>(&I)) {
    Out << " " << getPredicateText(FCI->getPredicate());
  } else if (const ICmpInst *ICI = dyn_cast<ICmpInst>(&I)) {
    Out << " " << getPredicateText(ICI->getPredicate());
  }

  // Print out the type of the operands...
  const Value *Operand = I.getNumOperands() ? I.getOperand(0) : 0;

  // Special case conditional branches to swizzle the condition out to the front
  if (isa<BranchInst>(I) && I.getNumOperands() > 1) {
    writeOperand(I.getOperand(2), true);
    Out << ',';
    writeOperand(Operand, true);
    Out << ',';
    writeOperand(I.getOperand(1), true);

  } else if (isa<SwitchInst>(I)) {
    // Special case switch statement to get formatting nice and correct...
    writeOperand(Operand        , true); Out << ',';
    writeOperand(I.getOperand(1), true); Out << " [";

    for (unsigned op = 2, Eop = I.getNumOperands(); op < Eop; op += 2) {
      Out << "\n\t\t";
      writeOperand(I.getOperand(op  ), true); Out << ',';
      writeOperand(I.getOperand(op+1), true);
    }
    Out << "\n\t]";
  } else if (isa<PHINode>(I)) {
    Out << ' ';
    printType(I.getType());
    Out << ' ';

    for (unsigned op = 0, Eop = I.getNumOperands(); op < Eop; op += 2) {
      if (op) Out << ", ";
      Out << '[';
      writeOperand(I.getOperand(op  ), false); Out << ',';
      writeOperand(I.getOperand(op+1), false); Out << " ]";
    }
  } else if (isa<ReturnInst>(I) && !Operand) {
    Out << " void";
  } else if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
    // Print the calling convention being used.
    switch (CI->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::CSRet: Out << " csretcc"; break;
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    case CallingConv::X86_StdCall:  Out << "x86_stdcallcc "; break;
    case CallingConv::X86_FastCall: Out << "x86_fastcallcc "; break; 
    default: Out << " cc" << CI->getCallingConv(); break;
    }

    const PointerType  *PTy = cast<PointerType>(Operand->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
    const Type       *RetTy = FTy->getReturnType();

    // If possible, print out the short form of the call instruction.  We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    if (!FTy->isVarArg() &&
        (!isa<PointerType>(RetTy) ||
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      Out << ' '; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }
    Out << '(';
    for (unsigned op = 1, Eop = I.getNumOperands(); op < Eop; ++op) {
      if (op > 1)
        Out << ',';
      writeOperand(I.getOperand(op), true);
      if (FTy->getParamAttrs(op) != FunctionType::NoAttributeSet)
        Out << " " << FTy->getParamAttrsText(FTy->getParamAttrs(op));
    }
    Out << " )";
    if (FTy->getParamAttrs(0) != FunctionType::NoAttributeSet)
      Out << ' ' << FTy->getParamAttrsText(FTy->getParamAttrs(0));
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    const PointerType  *PTy = cast<PointerType>(Operand->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
    const Type       *RetTy = FTy->getReturnType();

    // Print the calling convention being used.
    switch (II->getCallingConv()) {
    case CallingConv::C: break;   // default
    case CallingConv::CSRet: Out << " csretcc"; break;
    case CallingConv::Fast:  Out << " fastcc"; break;
    case CallingConv::Cold:  Out << " coldcc"; break;
    case CallingConv::X86_StdCall:  Out << "x86_stdcallcc "; break;
    case CallingConv::X86_FastCall: Out << "x86_fastcallcc "; break;
    default: Out << " cc" << II->getCallingConv(); break;
    }

    // If possible, print out the short form of the invoke instruction. We can
    // only do this if the first argument is a pointer to a nonvararg function,
    // and if the return type is not a pointer to a function.
    //
    if (!FTy->isVarArg() &&
        (!isa<PointerType>(RetTy) ||
         !isa<FunctionType>(cast<PointerType>(RetTy)->getElementType()))) {
      Out << ' '; printType(RetTy);
      writeOperand(Operand, false);
    } else {
      writeOperand(Operand, true);
    }

    Out << '(';
    for (unsigned op = 3, Eop = I.getNumOperands(); op < Eop; ++op) {
      if (op > 3)
        Out << ',';
      writeOperand(I.getOperand(op), true);
      if (FTy->getParamAttrs(op-2) != FunctionType::NoAttributeSet)
        Out << " " << FTy->getParamAttrsText(FTy->getParamAttrs(op-2));
    }

    Out << " )";
    if (FTy->getParamAttrs(0) != FunctionType::NoAttributeSet)
      Out << " " << FTy->getParamAttrsText(FTy->getParamAttrs(0));
    Out << "\n\t\t\tto";
    writeOperand(II->getNormalDest(), true);
    Out << " unwind";
    writeOperand(II->getUnwindDest(), true);

  } else if (const AllocationInst *AI = dyn_cast<AllocationInst>(&I)) {
    Out << ' ';
    printType(AI->getType()->getElementType());
    if (AI->isArrayAllocation()) {
      Out << ',';
      writeOperand(AI->getArraySize(), true);
    }
    if (AI->getAlignment()) {
      Out << ", align " << AI->getAlignment();
    }
  } else if (isa<CastInst>(I)) {
    if (Operand) writeOperand(Operand, true);   // Work with broken code
    Out << " to ";
    printType(I.getType());
  } else if (isa<VAArgInst>(I)) {
    if (Operand) writeOperand(Operand, true);   // Work with broken code
    Out << ", ";
    printType(I.getType());
  } else if (Operand) {   // Print the normal way...

    // PrintAllTypes - Instructions who have operands of all the same type
    // omit the type from all but the first operand.  If the instruction has
    // different type operands (for example br), then they are all printed.
    bool PrintAllTypes = false;
    const Type *TheType = Operand->getType();

    // Shift Left & Right print both types even for Ubyte LHS, and select prints
    // types even if all operands are bools.
    if (isa<ShiftInst>(I) || isa<SelectInst>(I) || isa<StoreInst>(I) ||
        isa<ShuffleVectorInst>(I)) {
      PrintAllTypes = true;
    } else {
      for (unsigned i = 1, E = I.getNumOperands(); i != E; ++i) {
        Operand = I.getOperand(i);
        if (Operand->getType() != TheType) {
          PrintAllTypes = true;    // We have differing types!  Print them all!
          break;
        }
      }
    }

    if (!PrintAllTypes) {
      Out << ' ';
      printType(TheType);
    }

    for (unsigned i = 0, E = I.getNumOperands(); i != E; ++i) {
      if (i) Out << ',';
      writeOperand(I.getOperand(i), PrintAllTypes);
    }
  }

  printInfoComment(I);
  Out << "\n";
}


//===----------------------------------------------------------------------===//
//                       External Interface declarations
//===----------------------------------------------------------------------===//

void Module::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  SlotMachine SlotTable(this);
  AssemblyWriter W(o, SlotTable, this, AAW);
  W.write(this);
}

void GlobalVariable::print(std::ostream &o) const {
  SlotMachine SlotTable(getParent());
  AssemblyWriter W(o, SlotTable, getParent(), 0);
  W.write(this);
}

void Function::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  SlotMachine SlotTable(getParent());
  AssemblyWriter W(o, SlotTable, getParent(), AAW);

  W.write(this);
}

void InlineAsm::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  WriteAsOperand(o, this, true, 0);
}

void BasicBlock::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  SlotMachine SlotTable(getParent());
  AssemblyWriter W(o, SlotTable,
                   getParent() ? getParent()->getParent() : 0, AAW);
  W.write(this);
}

void Instruction::print(std::ostream &o, AssemblyAnnotationWriter *AAW) const {
  const Function *F = getParent() ? getParent()->getParent() : 0;
  SlotMachine SlotTable(F);
  AssemblyWriter W(o, SlotTable, F ? F->getParent() : 0, AAW);

  W.write(this);
}

void Constant::print(std::ostream &o) const {
  if (this == 0) { o << "<null> constant value\n"; return; }

  o << ' ' << getType()->getDescription() << ' ';

  std::map<const Type *, std::string> TypeTable;
  WriteConstantInt(o, this, TypeTable, 0);
}

void Type::print(std::ostream &o) const {
  if (this == 0)
    o << "<null Type>";
  else
    o << getDescription();
}

void Argument::print(std::ostream &o) const {
  WriteAsOperand(o, this, true, getParent() ? getParent()->getParent() : 0);
}

// Value::dump - allow easy printing of  Values from the debugger.
// Located here because so much of the needed functionality is here.
void Value::dump() const { print(*cerr.stream()); cerr << '\n'; }

// Type::dump - allow easy printing of  Values from the debugger.
// Located here because so much of the needed functionality is here.
void Type::dump() const { print(*cerr.stream()); cerr << '\n'; }

//===----------------------------------------------------------------------===//
//                         SlotMachine Implementation
//===----------------------------------------------------------------------===//

#if 0
#define SC_DEBUG(X) cerr << X
#else
#define SC_DEBUG(X)
#endif

// Module level constructor. Causes the contents of the Module (sans functions)
// to be added to the slot table.
SlotMachine::SlotMachine(const Module *M)
  : TheModule(M)    ///< Saved for lazy initialization.
  , TheFunction(0)
  , FunctionProcessed(false)
{
}

// Function level constructor. Causes the contents of the Module and the one
// function provided to be added to the slot table.
SlotMachine::SlotMachine(const Function *F)
  : TheModule(F ? F->getParent() : 0) ///< Saved for lazy initialization
  , TheFunction(F) ///< Saved for lazy initialization
  , FunctionProcessed(false)
{
}

inline void SlotMachine::initialize(void) {
  if (TheModule) {
    processModule();
    TheModule = 0; ///< Prevent re-processing next time we're called.
  }
  if (TheFunction && !FunctionProcessed)
    processFunction();
}

// Iterate through all the global variables, functions, and global
// variable initializers and create slots for them.
void SlotMachine::processModule() {
  SC_DEBUG("begin processModule!\n");

  // Add all of the unnamed global variables to the value table.
  for (Module::const_global_iterator I = TheModule->global_begin(),
       E = TheModule->global_end(); I != E; ++I)
    if (!I->hasName()) 
      CreateModuleSlot(I);

  // Add all the unnamed functions to the table.
  for (Module::const_iterator I = TheModule->begin(), E = TheModule->end();
       I != E; ++I)
    if (!I->hasName())
      CreateModuleSlot(I);

  SC_DEBUG("end processModule!\n");
}


// Process the arguments, basic blocks, and instructions  of a function.
void SlotMachine::processFunction() {
  SC_DEBUG("begin processFunction!\n");

  // Add all the function arguments with no names.
  for(Function::const_arg_iterator AI = TheFunction->arg_begin(),
      AE = TheFunction->arg_end(); AI != AE; ++AI)
    if (!AI->hasName())
      CreateFunctionSlot(AI);

  SC_DEBUG("Inserting Instructions:\n");

  // Add all of the basic blocks and instructions with no names.
  for (Function::const_iterator BB = TheFunction->begin(),
       E = TheFunction->end(); BB != E; ++BB) {
    if (!BB->hasName())
      CreateFunctionSlot(BB);
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (I->getType() != Type::VoidTy && !I->hasName())
        CreateFunctionSlot(I);
  }

  FunctionProcessed = true;

  SC_DEBUG("end processFunction!\n");
}

/// Clean up after incorporating a function. This is the only way to get out of
/// the function incorporation state that affects getSlot/Create*Slot. Function
/// incorporation state is indicated by TheFunction != 0.
void SlotMachine::purgeFunction() {
  SC_DEBUG("begin purgeFunction!\n");
  fMap.clear(); // Simply discard the function level map
  TheFunction = 0;
  FunctionProcessed = false;
  SC_DEBUG("end purgeFunction!\n");
}

/// Get the slot number for a value. This function will assert if you
/// ask for a Value that hasn't previously been inserted with Create*Slot.
int SlotMachine::getSlot(const Value *V) {
  assert(V && "Can't get slot for null Value");
  assert(!isa<Constant>(V) || isa<GlobalValue>(V) &&
    "Can't insert a non-GlobalValue Constant into SlotMachine");

  // Check for uninitialized state and do lazy initialization
  this->initialize();

  // Get the type of the value
  const Type* VTy = V->getType();

  // Find the type plane in the module map
  TypedPlanes::const_iterator MI = mMap.find(VTy);

  if (TheFunction) {
    // Lookup the type in the function map too
    TypedPlanes::const_iterator FI = fMap.find(VTy);
    // If there is a corresponding type plane in the function map
    if (FI != fMap.end()) {
      // Lookup the Value in the function map
      ValueMap::const_iterator FVI = FI->second.map.find(V);
      // If the value doesn't exist in the function map
      if (FVI == FI->second.map.end()) {
        // Look up the value in the module map.
        if (MI == mMap.end()) return -1;
        ValueMap::const_iterator MVI = MI->second.map.find(V);
        // If we didn't find it, it wasn't inserted
        if (MVI == MI->second.map.end()) return -1;
        assert(MVI != MI->second.map.end() && "Value not found");
        // We found it only at the module level
        return MVI->second;

      // else the value exists in the function map
      } else {
        // Return the slot number as the module's contribution to
        // the type plane plus the index in the function's contribution
        // to the type plane.
        if (MI != mMap.end())
          return MI->second.next_slot + FVI->second;
        else
          return FVI->second;
      }
    }
  }

  // N.B. Can get here only if either !TheFunction or the function doesn't
  // have a corresponding type plane for the Value

  // Make sure the type plane exists
  if (MI == mMap.end()) return -1;
  // Lookup the value in the module's map
  ValueMap::const_iterator MVI = MI->second.map.find(V);
  // Make sure we found it.
  if (MVI == MI->second.map.end()) return -1;
  // Return it.
  return MVI->second;
}


/// CreateModuleSlot - Insert the specified GlobalValue* into the slot table.
void SlotMachine::CreateModuleSlot(const GlobalValue *V) {
  assert(V && "Can't insert a null Value into SlotMachine!");
  
  unsigned DestSlot = 0;
  const Type *VTy = V->getType();
  
  ValuePlane &PlaneMap = mMap[VTy];
  DestSlot = PlaneMap.map[V] = PlaneMap.next_slot++;
  
  SC_DEBUG("  Inserting value [" << VTy << "] = " << V << " slot=" <<
           DestSlot << " [");
  // G = Global, F = Function, o = other
  SC_DEBUG((isa<GlobalVariable>(V) ? 'G' : 'F') << "]\n");
}


/// CreateSlot - Create a new slot for the specified value if it has no name.
void SlotMachine::CreateFunctionSlot(const Value *V) {
  const Type *VTy = V->getType();
  assert(VTy != Type::VoidTy && !V->hasName() && "Doesn't need a slot!");
  
  unsigned DestSlot = 0;
  
  ValuePlane &PlaneMap = fMap[VTy];
  DestSlot = PlaneMap.map[V] = PlaneMap.next_slot++;
  
  // G = Global, F = Function, o = other
  SC_DEBUG("  Inserting value [" << VTy << "] = " << V << " slot=" <<
           DestSlot << " [o]\n");
}  

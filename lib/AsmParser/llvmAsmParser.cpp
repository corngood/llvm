/* A Bison parser, made by GNU Bison 1.875c.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Written by Richard Stallman by simplifying the original so called
   ``semantic'' parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0

/* If NAME_PREFIX is specified substitute the variables and functions
   names.  */
#define yyparse llvmAsmparse
#define yylex   llvmAsmlex
#define yyerror llvmAsmerror
#define yylval  llvmAsmlval
#define yychar  llvmAsmchar
#define yydebug llvmAsmdebug
#define yynerrs llvmAsmnerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     ESINT64VAL = 258,
     EUINT64VAL = 259,
     SINTVAL = 260,
     UINTVAL = 261,
     FPVAL = 262,
     VOID = 263,
     BOOL = 264,
     SBYTE = 265,
     UBYTE = 266,
     SHORT = 267,
     USHORT = 268,
     INT = 269,
     UINT = 270,
     LONG = 271,
     ULONG = 272,
     FLOAT = 273,
     DOUBLE = 274,
     TYPE = 275,
     LABEL = 276,
     VAR_ID = 277,
     LABELSTR = 278,
     STRINGCONSTANT = 279,
     IMPLEMENTATION = 280,
     ZEROINITIALIZER = 281,
     TRUETOK = 282,
     FALSETOK = 283,
     BEGINTOK = 284,
     ENDTOK = 285,
     DECLARE = 286,
     GLOBAL = 287,
     CONSTANT = 288,
     SECTION = 289,
     VOLATILE = 290,
     TO = 291,
     DOTDOTDOT = 292,
     NULL_TOK = 293,
     UNDEF = 294,
     CONST = 295,
     INTERNAL = 296,
     LINKONCE = 297,
     WEAK = 298,
     APPENDING = 299,
     OPAQUE = 300,
     NOT = 301,
     EXTERNAL = 302,
     TARGET = 303,
     TRIPLE = 304,
     ENDIAN = 305,
     POINTERSIZE = 306,
     LITTLE = 307,
     BIG = 308,
     ALIGN = 309,
     DEPLIBS = 310,
     CALL = 311,
     TAIL = 312,
     CC_TOK = 313,
     CCC_TOK = 314,
     FASTCC_TOK = 315,
     COLDCC_TOK = 316,
     RET = 317,
     BR = 318,
     SWITCH = 319,
     INVOKE = 320,
     UNWIND = 321,
     UNREACHABLE = 322,
     ADD = 323,
     SUB = 324,
     MUL = 325,
     DIV = 326,
     REM = 327,
     AND = 328,
     OR = 329,
     XOR = 330,
     SETLE = 331,
     SETGE = 332,
     SETLT = 333,
     SETGT = 334,
     SETEQ = 335,
     SETNE = 336,
     MALLOC = 337,
     ALLOCA = 338,
     FREE = 339,
     LOAD = 340,
     STORE = 341,
     GETELEMENTPTR = 342,
     PHI_TOK = 343,
     CAST = 344,
     SELECT = 345,
     SHL = 346,
     SHR = 347,
     VAARG = 348,
     EXTRACTELEMENT = 349,
     VAARG_old = 350,
     VANEXT_old = 351
   };
#endif
#define ESINT64VAL 258
#define EUINT64VAL 259
#define SINTVAL 260
#define UINTVAL 261
#define FPVAL 262
#define VOID 263
#define BOOL 264
#define SBYTE 265
#define UBYTE 266
#define SHORT 267
#define USHORT 268
#define INT 269
#define UINT 270
#define LONG 271
#define ULONG 272
#define FLOAT 273
#define DOUBLE 274
#define TYPE 275
#define LABEL 276
#define VAR_ID 277
#define LABELSTR 278
#define STRINGCONSTANT 279
#define IMPLEMENTATION 280
#define ZEROINITIALIZER 281
#define TRUETOK 282
#define FALSETOK 283
#define BEGINTOK 284
#define ENDTOK 285
#define DECLARE 286
#define GLOBAL 287
#define CONSTANT 288
#define SECTION 289
#define VOLATILE 290
#define TO 291
#define DOTDOTDOT 292
#define NULL_TOK 293
#define UNDEF 294
#define CONST 295
#define INTERNAL 296
#define LINKONCE 297
#define WEAK 298
#define APPENDING 299
#define OPAQUE 300
#define NOT 301
#define EXTERNAL 302
#define TARGET 303
#define TRIPLE 304
#define ENDIAN 305
#define POINTERSIZE 306
#define LITTLE 307
#define BIG 308
#define ALIGN 309
#define DEPLIBS 310
#define CALL 311
#define TAIL 312
#define CC_TOK 313
#define CCC_TOK 314
#define FASTCC_TOK 315
#define COLDCC_TOK 316
#define RET 317
#define BR 318
#define SWITCH 319
#define INVOKE 320
#define UNWIND 321
#define UNREACHABLE 322
#define ADD 323
#define SUB 324
#define MUL 325
#define DIV 326
#define REM 327
#define AND 328
#define OR 329
#define XOR 330
#define SETLE 331
#define SETGE 332
#define SETLT 333
#define SETGT 334
#define SETEQ 335
#define SETNE 336
#define MALLOC 337
#define ALLOCA 338
#define FREE 339
#define LOAD 340
#define STORE 341
#define GETELEMENTPTR 342
#define PHI_TOK 343
#define CAST 344
#define SELECT 345
#define SHL 346
#define SHR 347
#define VAARG 348
#define EXTRACTELEMENT 349
#define VAARG_old 350
#define VANEXT_old 351




/* Copy the first part of user declarations.  */
#line 14 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"

#include "ParserInternals.h"
#include "llvm/CallingConv.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Assembly/AutoUpgrade.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <utility>

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit
int yylex();                       // declaration" of xxx warnings.
int yyparse();

namespace llvm {
  std::string CurFilename;
}
using namespace llvm;

static Module *ParserResult;

// DEBUG_UPREFS - Define this symbol if you want to enable debugging output
// relating to upreferences in the input stream.
//
//#define DEBUG_UPREFS 1
#ifdef DEBUG_UPREFS
#define UR_OUT(X) std::cerr << X
#else
#define UR_OUT(X)
#endif

#define YYERROR_VERBOSE 1

static bool ObsoleteVarArgs;
static bool NewVarArgs;
static BasicBlock *CurBB;
static GlobalVariable *CurGV;


// This contains info used when building the body of a function.  It is
// destroyed when the function is completed.
//
typedef std::vector<Value *> ValueList;           // Numbered defs
static void 
ResolveDefinitions(std::map<const Type *,ValueList> &LateResolvers,
                   std::map<const Type *,ValueList> *FutureLateResolvers = 0);

static struct PerModuleInfo {
  Module *CurrentModule;
  std::map<const Type *, ValueList> Values; // Module level numbered definitions
  std::map<const Type *,ValueList> LateResolveValues;
  std::vector<PATypeHolder>    Types;
  std::map<ValID, PATypeHolder> LateResolveTypes;

  /// PlaceHolderInfo - When temporary placeholder objects are created, remember
  /// how they were referenced and one which line of the input they came from so
  /// that we can resolve them later and print error messages as appropriate.
  std::map<Value*, std::pair<ValID, int> > PlaceHolderInfo;

  // GlobalRefs - This maintains a mapping between <Type, ValID>'s and forward
  // references to global values.  Global values may be referenced before they
  // are defined, and if so, the temporary object that they represent is held
  // here.  This is used for forward references of GlobalValues.
  //
  typedef std::map<std::pair<const PointerType *,
                             ValID>, GlobalValue*> GlobalRefsType;
  GlobalRefsType GlobalRefs;

  void ModuleDone() {
    // If we could not resolve some functions at function compilation time
    // (calls to functions before they are defined), resolve them now...  Types
    // are resolved when the constant pool has been completely parsed.
    //
    ResolveDefinitions(LateResolveValues);

    // Check to make sure that all global value forward references have been
    // resolved!
    //
    if (!GlobalRefs.empty()) {
      std::string UndefinedReferences = "Unresolved global references exist:\n";

      for (GlobalRefsType::iterator I = GlobalRefs.begin(), E =GlobalRefs.end();
           I != E; ++I) {
        UndefinedReferences += "  " + I->first.first->getDescription() + " " +
                               I->first.second.getName() + "\n";
      }
      ThrowException(UndefinedReferences);
    }

    // Rename any overloaded intrinsic functions.
    for (Module::iterator FI = CurrentModule->begin(), FE =
         CurrentModule->end(); FI != FE; ++FI)
      UpgradeIntrinsicFunction(&(*FI));

    Values.clear();         // Clear out function local definitions
    Types.clear();
    CurrentModule = 0;
  }

  // GetForwardRefForGlobal - Check to see if there is a forward reference
  // for this global.  If so, remove it from the GlobalRefs map and return it.
  // If not, just return null.
  GlobalValue *GetForwardRefForGlobal(const PointerType *PTy, ValID ID) {
    // Check to see if there is a forward reference to this global variable...
    // if there is, eliminate it and patch the reference to use the new def'n.
    GlobalRefsType::iterator I = GlobalRefs.find(std::make_pair(PTy, ID));
    GlobalValue *Ret = 0;
    if (I != GlobalRefs.end()) {
      Ret = I->second;
      GlobalRefs.erase(I);
    }
    return Ret;
  }
} CurModule;

static struct PerFunctionInfo {
  Function *CurrentFunction;     // Pointer to current function being created

  std::map<const Type*, ValueList> Values;   // Keep track of #'d definitions
  std::map<const Type*, ValueList> LateResolveValues;
  bool isDeclare;                // Is this function a forward declararation?

  /// BBForwardRefs - When we see forward references to basic blocks, keep
  /// track of them here.
  std::map<BasicBlock*, std::pair<ValID, int> > BBForwardRefs;
  std::vector<BasicBlock*> NumberedBlocks;
  unsigned NextBBNum;

  inline PerFunctionInfo() {
    CurrentFunction = 0;
    isDeclare = false;
  }

  inline void FunctionStart(Function *M) {
    CurrentFunction = M;
    NextBBNum = 0;
  }

  void FunctionDone() {
    NumberedBlocks.clear();

    // Any forward referenced blocks left?
    if (!BBForwardRefs.empty())
      ThrowException("Undefined reference to label " +
                     BBForwardRefs.begin()->first->getName());

    // Resolve all forward references now.
    ResolveDefinitions(LateResolveValues, &CurModule.LateResolveValues);

    Values.clear();         // Clear out function local definitions
    CurrentFunction = 0;
    isDeclare = false;
  }
} CurFun;  // Info for the current function...

static bool inFunctionScope() { return CurFun.CurrentFunction != 0; }


//===----------------------------------------------------------------------===//
//               Code to handle definitions of all the types
//===----------------------------------------------------------------------===//

static int InsertValue(Value *V,
                  std::map<const Type*,ValueList> &ValueTab = CurFun.Values) {
  if (V->hasName()) return -1;           // Is this a numbered definition?

  // Yes, insert the value into the value table...
  ValueList &List = ValueTab[V->getType()];
  List.push_back(V);
  return List.size()-1;
}

static const Type *getTypeVal(const ValID &D, bool DoNotImprovise = false) {
  switch (D.Type) {
  case ValID::NumberVal:               // Is it a numbered definition?
    // Module constants occupy the lowest numbered slots...
    if ((unsigned)D.Num < CurModule.Types.size())
      return CurModule.Types[(unsigned)D.Num];
    break;
  case ValID::NameVal:                 // Is it a named definition?
    if (const Type *N = CurModule.CurrentModule->getTypeByName(D.Name)) {
      D.destroy();  // Free old strdup'd memory...
      return N;
    }
    break;
  default:
    ThrowException("Internal parser error: Invalid symbol type reference!");
  }

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  if (DoNotImprovise) return 0;  // Do we just want a null to be returned?


  if (inFunctionScope()) {
    if (D.Type == ValID::NameVal)
      ThrowException("Reference to an undefined type: '" + D.getName() + "'");
    else
      ThrowException("Reference to an undefined type: #" + itostr(D.Num));
  }

  std::map<ValID, PATypeHolder>::iterator I =CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end())
    return I->second;

  Type *Typ = OpaqueType::get();
  CurModule.LateResolveTypes.insert(std::make_pair(D, Typ));
  return Typ;
 }

static Value *lookupInSymbolTable(const Type *Ty, const std::string &Name) {
  SymbolTable &SymTab =
    inFunctionScope() ? CurFun.CurrentFunction->getSymbolTable() :
                        CurModule.CurrentModule->getSymbolTable();
  return SymTab.lookup(Ty, Name);
}

// getValNonImprovising - Look up the value specified by the provided type and
// the provided ValID.  If the value exists and has already been defined, return
// it.  Otherwise return null.
//
static Value *getValNonImprovising(const Type *Ty, const ValID &D) {
  if (isa<FunctionType>(Ty))
    ThrowException("Functions are not values and "
                   "must be referenced as pointers");

  switch (D.Type) {
  case ValID::NumberVal: {                 // Is it a numbered definition?
    unsigned Num = (unsigned)D.Num;

    // Module constants occupy the lowest numbered slots...
    std::map<const Type*,ValueList>::iterator VI = CurModule.Values.find(Ty);
    if (VI != CurModule.Values.end()) {
      if (Num < VI->second.size())
        return VI->second[Num];
      Num -= VI->second.size();
    }

    // Make sure that our type is within bounds
    VI = CurFun.Values.find(Ty);
    if (VI == CurFun.Values.end()) return 0;

    // Check that the number is within bounds...
    if (VI->second.size() <= Num) return 0;

    return VI->second[Num];
  }

  case ValID::NameVal: {                // Is it a named definition?
    Value *N = lookupInSymbolTable(Ty, std::string(D.Name));
    if (N == 0) return 0;

    D.destroy();  // Free old strdup'd memory...
    return N;
  }

  // Check to make sure that "Ty" is an integral type, and that our
  // value will fit into the specified type...
  case ValID::ConstSIntVal:    // Is it a constant pool reference??
    if (!ConstantSInt::isValueValidForType(Ty, D.ConstPool64))
      ThrowException("Signed integral constant '" +
                     itostr(D.ConstPool64) + "' is invalid for type '" +
                     Ty->getDescription() + "'!");
    return ConstantSInt::get(Ty, D.ConstPool64);

  case ValID::ConstUIntVal:     // Is it an unsigned const pool reference?
    if (!ConstantUInt::isValueValidForType(Ty, D.UConstPool64)) {
      if (!ConstantSInt::isValueValidForType(Ty, D.ConstPool64)) {
        ThrowException("Integral constant '" + utostr(D.UConstPool64) +
                       "' is invalid or out of range!");
      } else {     // This is really a signed reference.  Transmogrify.
        return ConstantSInt::get(Ty, D.ConstPool64);
      }
    } else {
      return ConstantUInt::get(Ty, D.UConstPool64);
    }

  case ValID::ConstFPVal:        // Is it a floating point const pool reference?
    if (!ConstantFP::isValueValidForType(Ty, D.ConstPoolFP))
      ThrowException("FP constant invalid for type!!");
    return ConstantFP::get(Ty, D.ConstPoolFP);

  case ValID::ConstNullVal:      // Is it a null value?
    if (!isa<PointerType>(Ty))
      ThrowException("Cannot create a a non pointer null!");
    return ConstantPointerNull::get(cast<PointerType>(Ty));

  case ValID::ConstUndefVal:      // Is it an undef value?
    return UndefValue::get(Ty);

  case ValID::ConstZeroVal:      // Is it a zero value?
    return Constant::getNullValue(Ty);
    
  case ValID::ConstantVal:       // Fully resolved constant?
    if (D.ConstantValue->getType() != Ty)
      ThrowException("Constant expression type different from required type!");
    return D.ConstantValue;

  default:
    assert(0 && "Unhandled case!");
    return 0;
  }   // End of switch

  assert(0 && "Unhandled case!");
  return 0;
}

// getVal - This function is identical to getValNonImprovising, except that if a
// value is not already defined, it "improvises" by creating a placeholder var
// that looks and acts just like the requested variable.  When the value is
// defined later, all uses of the placeholder variable are replaced with the
// real thing.
//
static Value *getVal(const Type *Ty, const ValID &ID) {
  if (Ty == Type::LabelTy)
    ThrowException("Cannot use a basic block here");

  // See if the value has already been defined.
  Value *V = getValNonImprovising(Ty, ID);
  if (V) return V;

  if (!Ty->isFirstClassType() && !isa<OpaqueType>(Ty))
    ThrowException("Invalid use of a composite type!");

  // If we reached here, we referenced either a symbol that we don't know about
  // or an id number that hasn't been read yet.  We may be referencing something
  // forward, so just create an entry to be resolved later and get to it...
  //
  V = new Argument(Ty);

  // Remember where this forward reference came from.  FIXME, shouldn't we try
  // to recycle these things??
  CurModule.PlaceHolderInfo.insert(std::make_pair(V, std::make_pair(ID,
                                                               llvmAsmlineno)));

  if (inFunctionScope())
    InsertValue(V, CurFun.LateResolveValues);
  else
    InsertValue(V, CurModule.LateResolveValues);
  return V;
}

/// getBBVal - This is used for two purposes:
///  * If isDefinition is true, a new basic block with the specified ID is being
///    defined.
///  * If isDefinition is true, this is a reference to a basic block, which may
///    or may not be a forward reference.
///
static BasicBlock *getBBVal(const ValID &ID, bool isDefinition = false) {
  assert(inFunctionScope() && "Can't get basic block at global scope!");

  std::string Name;
  BasicBlock *BB = 0;
  switch (ID.Type) {
  default: ThrowException("Illegal label reference " + ID.getName());
  case ValID::NumberVal:                // Is it a numbered definition?
    if (unsigned(ID.Num) >= CurFun.NumberedBlocks.size())
      CurFun.NumberedBlocks.resize(ID.Num+1);
    BB = CurFun.NumberedBlocks[ID.Num];
    break;
  case ValID::NameVal:                  // Is it a named definition?
    Name = ID.Name;
    if (Value *N = CurFun.CurrentFunction->
                   getSymbolTable().lookup(Type::LabelTy, Name))
      BB = cast<BasicBlock>(N);
    break;
  }

  // See if the block has already been defined.
  if (BB) {
    // If this is the definition of the block, make sure the existing value was
    // just a forward reference.  If it was a forward reference, there will be
    // an entry for it in the PlaceHolderInfo map.
    if (isDefinition && !CurFun.BBForwardRefs.erase(BB))
      // The existing value was a definition, not a forward reference.
      ThrowException("Redefinition of label " + ID.getName());

    ID.destroy();                       // Free strdup'd memory.
    return BB;
  }

  // Otherwise this block has not been seen before.
  BB = new BasicBlock("", CurFun.CurrentFunction);
  if (ID.Type == ValID::NameVal) {
    BB->setName(ID.Name);
  } else {
    CurFun.NumberedBlocks[ID.Num] = BB;
  }

  // If this is not a definition, keep track of it so we can use it as a forward
  // reference.
  if (!isDefinition) {
    // Remember where this forward reference came from.
    CurFun.BBForwardRefs[BB] = std::make_pair(ID, llvmAsmlineno);
  } else {
    // The forward declaration could have been inserted anywhere in the
    // function: insert it into the correct place now.
    CurFun.CurrentFunction->getBasicBlockList().remove(BB);
    CurFun.CurrentFunction->getBasicBlockList().push_back(BB);
  }
  ID.destroy();
  return BB;
}


//===----------------------------------------------------------------------===//
//              Code to handle forward references in instructions
//===----------------------------------------------------------------------===//
//
// This code handles the late binding needed with statements that reference
// values not defined yet... for example, a forward branch, or the PHI node for
// a loop body.
//
// This keeps a table (CurFun.LateResolveValues) of all such forward references
// and back patchs after we are done.
//

// ResolveDefinitions - If we could not resolve some defs at parsing
// time (forward branches, phi functions for loops, etc...) resolve the
// defs now...
//
static void 
ResolveDefinitions(std::map<const Type*,ValueList> &LateResolvers,
                   std::map<const Type*,ValueList> *FutureLateResolvers) {
  // Loop over LateResolveDefs fixing up stuff that couldn't be resolved
  for (std::map<const Type*,ValueList>::iterator LRI = LateResolvers.begin(),
         E = LateResolvers.end(); LRI != E; ++LRI) {
    ValueList &List = LRI->second;
    while (!List.empty()) {
      Value *V = List.back();
      List.pop_back();

      std::map<Value*, std::pair<ValID, int> >::iterator PHI =
        CurModule.PlaceHolderInfo.find(V);
      assert(PHI != CurModule.PlaceHolderInfo.end() && "Placeholder error!");

      ValID &DID = PHI->second.first;

      Value *TheRealValue = getValNonImprovising(LRI->first, DID);
      if (TheRealValue) {
        V->replaceAllUsesWith(TheRealValue);
        delete V;
        CurModule.PlaceHolderInfo.erase(PHI);
      } else if (FutureLateResolvers) {
        // Functions have their unresolved items forwarded to the module late
        // resolver table
        InsertValue(V, *FutureLateResolvers);
      } else {
        if (DID.Type == ValID::NameVal)
          ThrowException("Reference to an invalid definition: '" +DID.getName()+
                         "' of type '" + V->getType()->getDescription() + "'",
                         PHI->second.second);
        else
          ThrowException("Reference to an invalid definition: #" +
                         itostr(DID.Num) + " of type '" +
                         V->getType()->getDescription() + "'",
                         PHI->second.second);
      }
    }
  }

  LateResolvers.clear();
}

// ResolveTypeTo - A brand new type was just declared.  This means that (if
// name is not null) things referencing Name can be resolved.  Otherwise, things
// refering to the number can be resolved.  Do this now.
//
static void ResolveTypeTo(char *Name, const Type *ToTy) {
  ValID D;
  if (Name) D = ValID::create(Name);
  else      D = ValID::create((int)CurModule.Types.size());

  std::map<ValID, PATypeHolder>::iterator I =
    CurModule.LateResolveTypes.find(D);
  if (I != CurModule.LateResolveTypes.end()) {
    ((DerivedType*)I->second.get())->refineAbstractTypeTo(ToTy);
    CurModule.LateResolveTypes.erase(I);
  }
}

// setValueName - Set the specified value to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is free'd by this function.
//
static void setValueName(Value *V, char *NameStr) {
  if (NameStr) {
    std::string Name(NameStr);      // Copy string
    free(NameStr);                  // Free old string

    if (V->getType() == Type::VoidTy)
      ThrowException("Can't assign name '" + Name+"' to value with void type!");

    assert(inFunctionScope() && "Must be in function scope!");
    SymbolTable &ST = CurFun.CurrentFunction->getSymbolTable();
    if (ST.lookup(V->getType(), Name))
      ThrowException("Redefinition of value named '" + Name + "' in the '" +
                     V->getType()->getDescription() + "' type plane!");

    // Set the name.
    V->setName(Name);
  }
}

/// ParseGlobalVariable - Handle parsing of a global.  If Initializer is null,
/// this is a declaration, otherwise it is a definition.
static GlobalVariable *
ParseGlobalVariable(char *NameStr,GlobalValue::LinkageTypes Linkage,
                    bool isConstantGlobal, const Type *Ty,
                    Constant *Initializer) {
  if (isa<FunctionType>(Ty))
    ThrowException("Cannot declare global vars of function type!");

  const PointerType *PTy = PointerType::get(Ty);

  std::string Name;
  if (NameStr) {
    Name = NameStr;      // Copy string
    free(NameStr);       // Free old string
  }

  // See if this global value was forward referenced.  If so, recycle the
  // object.
  ValID ID;
  if (!Name.empty()) {
    ID = ValID::create((char*)Name.c_str());
  } else {
    ID = ValID::create((int)CurModule.Values[PTy].size());
  }

  if (GlobalValue *FWGV = CurModule.GetForwardRefForGlobal(PTy, ID)) {
    // Move the global to the end of the list, from whereever it was
    // previously inserted.
    GlobalVariable *GV = cast<GlobalVariable>(FWGV);
    CurModule.CurrentModule->getGlobalList().remove(GV);
    CurModule.CurrentModule->getGlobalList().push_back(GV);
    GV->setInitializer(Initializer);
    GV->setLinkage(Linkage);
    GV->setConstant(isConstantGlobal);
    InsertValue(GV, CurModule.Values);
    return GV;
  }

  // If this global has a name, check to see if there is already a definition
  // of this global in the module.  If so, merge as appropriate.  Note that
  // this is really just a hack around problems in the CFE.  :(
  if (!Name.empty()) {
    // We are a simple redefinition of a value, check to see if it is defined
    // the same as the old one.
    if (GlobalVariable *EGV =
                CurModule.CurrentModule->getGlobalVariable(Name, Ty)) {
      // We are allowed to redefine a global variable in two circumstances:
      // 1. If at least one of the globals is uninitialized or
      // 2. If both initializers have the same value.
      //
      if (!EGV->hasInitializer() || !Initializer ||
          EGV->getInitializer() == Initializer) {

        // Make sure the existing global version gets the initializer!  Make
        // sure that it also gets marked const if the new version is.
        if (Initializer && !EGV->hasInitializer())
          EGV->setInitializer(Initializer);
        if (isConstantGlobal)
          EGV->setConstant(true);
        EGV->setLinkage(Linkage);
        return EGV;
      }

      ThrowException("Redefinition of global variable named '" + Name +
                     "' in the '" + Ty->getDescription() + "' type plane!");
    }
  }

  // Otherwise there is no existing GV to use, create one now.
  GlobalVariable *GV =
    new GlobalVariable(Ty, isConstantGlobal, Linkage, Initializer, Name,
                       CurModule.CurrentModule);
  InsertValue(GV, CurModule.Values);
  return GV;
}

// setTypeName - Set the specified type to the name given.  The name may be
// null potentially, in which case this is a noop.  The string passed in is
// assumed to be a malloc'd string buffer, and is freed by this function.
//
// This function returns true if the type has already been defined, but is
// allowed to be redefined in the specified context.  If the name is a new name
// for the type plane, it is inserted and false is returned.
static bool setTypeName(const Type *T, char *NameStr) {
  assert(!inFunctionScope() && "Can't give types function-local names!");
  if (NameStr == 0) return false;
 
  std::string Name(NameStr);      // Copy string
  free(NameStr);                  // Free old string

  // We don't allow assigning names to void type
  if (T == Type::VoidTy)
    ThrowException("Can't assign name '" + Name + "' to the void type!");

  // Set the type name, checking for conflicts as we do so.
  bool AlreadyExists = CurModule.CurrentModule->addTypeName(Name, T);

  if (AlreadyExists) {   // Inserting a name that is already defined???
    const Type *Existing = CurModule.CurrentModule->getTypeByName(Name);
    assert(Existing && "Conflict but no matching type?");

    // There is only one case where this is allowed: when we are refining an
    // opaque type.  In this case, Existing will be an opaque type.
    if (const OpaqueType *OpTy = dyn_cast<OpaqueType>(Existing)) {
      // We ARE replacing an opaque type!
      const_cast<OpaqueType*>(OpTy)->refineAbstractTypeTo(T);
      return true;
    }

    // Otherwise, this is an attempt to redefine a type. That's okay if
    // the redefinition is identical to the original. This will be so if
    // Existing and T point to the same Type object. In this one case we
    // allow the equivalent redefinition.
    if (Existing == T) return true;  // Yes, it's equal.

    // Any other kind of (non-equivalent) redefinition is an error.
    ThrowException("Redefinition of type named '" + Name + "' in the '" +
                   T->getDescription() + "' type plane!");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Code for handling upreferences in type names...
//

// TypeContains - Returns true if Ty directly contains E in it.
//
static bool TypeContains(const Type *Ty, const Type *E) {
  return std::find(Ty->subtype_begin(), Ty->subtype_end(),
                   E) != Ty->subtype_end();
}

namespace {
  struct UpRefRecord {
    // NestingLevel - The number of nesting levels that need to be popped before
    // this type is resolved.
    unsigned NestingLevel;

    // LastContainedTy - This is the type at the current binding level for the
    // type.  Every time we reduce the nesting level, this gets updated.
    const Type *LastContainedTy;

    // UpRefTy - This is the actual opaque type that the upreference is
    // represented with.
    OpaqueType *UpRefTy;

    UpRefRecord(unsigned NL, OpaqueType *URTy)
      : NestingLevel(NL), LastContainedTy(URTy), UpRefTy(URTy) {}
  };
}

// UpRefs - A list of the outstanding upreferences that need to be resolved.
static std::vector<UpRefRecord> UpRefs;

/// HandleUpRefs - Every time we finish a new layer of types, this function is
/// called.  It loops through the UpRefs vector, which is a list of the
/// currently active types.  For each type, if the up reference is contained in
/// the newly completed type, we decrement the level count.  When the level
/// count reaches zero, the upreferenced type is the type that is passed in:
/// thus we can complete the cycle.
///
static PATypeHolder HandleUpRefs(const Type *ty) {
  if (!ty->isAbstract()) return ty;
  PATypeHolder Ty(ty);
  UR_OUT("Type '" << Ty->getDescription() <<
         "' newly formed.  Resolving upreferences.\n" <<
         UpRefs.size() << " upreferences active!\n");

  // If we find any resolvable upreferences (i.e., those whose NestingLevel goes
  // to zero), we resolve them all together before we resolve them to Ty.  At
  // the end of the loop, if there is anything to resolve to Ty, it will be in
  // this variable.
  OpaqueType *TypeToResolve = 0;

  for (unsigned i = 0; i != UpRefs.size(); ++i) {
    UR_OUT("  UR#" << i << " - TypeContains(" << Ty->getDescription() << ", "
           << UpRefs[i].second->getDescription() << ") = "
           << (TypeContains(Ty, UpRefs[i].second) ? "true" : "false") << "\n");
    if (TypeContains(Ty, UpRefs[i].LastContainedTy)) {
      // Decrement level of upreference
      unsigned Level = --UpRefs[i].NestingLevel;
      UpRefs[i].LastContainedTy = Ty;
      UR_OUT("  Uplevel Ref Level = " << Level << "\n");
      if (Level == 0) {                     // Upreference should be resolved!
        if (!TypeToResolve) {
          TypeToResolve = UpRefs[i].UpRefTy;
        } else {
          UR_OUT("  * Resolving upreference for "
                 << UpRefs[i].second->getDescription() << "\n";
                 std::string OldName = UpRefs[i].UpRefTy->getDescription());
          UpRefs[i].UpRefTy->refineAbstractTypeTo(TypeToResolve);
          UR_OUT("  * Type '" << OldName << "' refined upreference to: "
                 << (const void*)Ty << ", " << Ty->getDescription() << "\n");
        }
        UpRefs.erase(UpRefs.begin()+i);     // Remove from upreference list...
        --i;                                // Do not skip the next element...
      }
    }
  }

  if (TypeToResolve) {
    UR_OUT("  * Resolving upreference for "
           << UpRefs[i].second->getDescription() << "\n";
           std::string OldName = TypeToResolve->getDescription());
    TypeToResolve->refineAbstractTypeTo(Ty);
  }

  return Ty;
}


// common code from the two 'RunVMAsmParser' functions
 static Module * RunParser(Module * M) {

  llvmAsmlineno = 1;      // Reset the current line number...
  ObsoleteVarArgs = false;
  NewVarArgs = false;

  CurModule.CurrentModule = M;
  yyparse();       // Parse the file, potentially throwing exception

  Module *Result = ParserResult;
  ParserResult = 0;

  //Not all functions use vaarg, so make a second check for ObsoleteVarArgs
  {
    Function* F;
    if ((F = Result->getNamedFunction("llvm.va_start"))
        && F->getFunctionType()->getNumParams() == 0)
      ObsoleteVarArgs = true;
    if((F = Result->getNamedFunction("llvm.va_copy"))
       && F->getFunctionType()->getNumParams() == 1)
      ObsoleteVarArgs = true;
  }

  if (ObsoleteVarArgs && NewVarArgs)
    ThrowException("This file is corrupt: it uses both new and old style varargs");

  if(ObsoleteVarArgs) {
    if(Function* F = Result->getNamedFunction("llvm.va_start")) {
      if (F->arg_size() != 0)
        ThrowException("Obsolete va_start takes 0 argument!");
      
      //foo = va_start()
      // ->
      //bar = alloca typeof(foo)
      //va_start(bar)
      //foo = load bar

      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getReturnType();
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_start", 
                                                 RetTy, ArgTyPtr, (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vastart.fix.1", CI);
        new CallInst(NF, bar, "", CI);
        Value* foo = new LoadInst(bar, "vastart.fix.2", CI);
        CI->replaceAllUsesWith(foo);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }
    
    if(Function* F = Result->getNamedFunction("llvm.va_end")) {
      if(F->arg_size() != 1)
        ThrowException("Obsolete va_end takes 1 argument!");

      //vaend foo
      // ->
      //bar = alloca 1 of typeof(foo)
      //vaend bar
      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getParamType(0);
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_end", 
                                                 RetTy, ArgTyPtr, (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* bar = new AllocaInst(ArgTy, 0, "vaend.fix.1", CI);
        new StoreInst(CI->getOperand(1), bar, CI);
        new CallInst(NF, bar, "", CI);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }

    if(Function* F = Result->getNamedFunction("llvm.va_copy")) {
      if(F->arg_size() != 1)
        ThrowException("Obsolete va_copy takes 1 argument!");
      //foo = vacopy(bar)
      // ->
      //a = alloca 1 of typeof(foo)
      //b = alloca 1 of typeof(foo)
      //store bar -> b
      //vacopy(a, b)
      //foo = load a
      
      const Type* RetTy = Type::getPrimitiveType(Type::VoidTyID);
      const Type* ArgTy = F->getFunctionType()->getReturnType();
      const Type* ArgTyPtr = PointerType::get(ArgTy);
      Function* NF = Result->getOrInsertFunction("llvm.va_copy", 
                                                 RetTy, ArgTyPtr, ArgTyPtr,
                                                 (Type *)0);

      while (!F->use_empty()) {
        CallInst* CI = cast<CallInst>(F->use_back());
        AllocaInst* a = new AllocaInst(ArgTy, 0, "vacopy.fix.1", CI);
        AllocaInst* b = new AllocaInst(ArgTy, 0, "vacopy.fix.2", CI);
        new StoreInst(CI->getOperand(1), b, CI);
        new CallInst(NF, a, b, "", CI);
        Value* foo = new LoadInst(a, "vacopy.fix.3", CI);
        CI->replaceAllUsesWith(foo);
        CI->getParent()->getInstList().erase(CI);
      }
      Result->getFunctionList().erase(F);
    }
  }

  return Result;

 }

//===----------------------------------------------------------------------===//
//            RunVMAsmParser - Define an interface to this parser
//===----------------------------------------------------------------------===//
//
Module *llvm::RunVMAsmParser(const std::string &Filename, FILE *F) {
  set_scan_file(F);

  CurFilename = Filename;
  return RunParser(new Module(CurFilename));
}

Module *llvm::RunVMAsmParser(const char * AsmString, Module * M) {
  set_scan_string(AsmString);

  CurFilename = "from_memory";
  if (M == NULL) {
    return RunParser(new Module (CurFilename));
  } else {
    return RunParser(M);
  }
}



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 878 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
typedef union YYSTYPE {
  llvm::Module                           *ModuleVal;
  llvm::Function                         *FunctionVal;
  std::pair<llvm::PATypeHolder*, char*>  *ArgVal;
  llvm::BasicBlock                       *BasicBlockVal;
  llvm::TerminatorInst                   *TermInstVal;
  llvm::Instruction                      *InstVal;
  llvm::Constant                         *ConstVal;

  const llvm::Type                       *PrimType;
  llvm::PATypeHolder                     *TypeVal;
  llvm::Value                            *ValueVal;

  std::vector<std::pair<llvm::PATypeHolder*,char*> > *ArgList;
  std::vector<llvm::Value*>              *ValueList;
  std::list<llvm::PATypeHolder>          *TypeList;
  // Represent the RHS of PHI node
  std::list<std::pair<llvm::Value*,
                      llvm::BasicBlock*> > *PHIList;
  std::vector<std::pair<llvm::Constant*, llvm::BasicBlock*> > *JumpTable;
  std::vector<llvm::Constant*>           *ConstVector;

  llvm::GlobalValue::LinkageTypes         Linkage;
  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  double                            FPVal;
  bool                              BoolVal;

  char                             *StrVal;   // This memory is strdup'd!
  llvm::ValID                             ValIDVal; // strdup'd memory maybe!

  llvm::Instruction::BinaryOps            BinaryOpVal;
  llvm::Instruction::TermOps              TermOpVal;
  llvm::Instruction::MemoryOps            MemOpVal;
  llvm::Instruction::OtherOps             OtherOpVal;
  llvm::Module::Endianness                Endianness;
} YYSTYPE;
/* Line 191 of yacc.c.  */
#line 1181 "llvmAsmParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 1193 "llvmAsmParser.tab.c"

#if ! defined (yyoverflow) || YYERROR_VERBOSE

# ifndef YYFREE
#  define YYFREE free
# endif
# ifndef YYMALLOC
#  define YYMALLOC malloc
# endif

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   define YYSTACK_ALLOC alloca
#  endif
# else
#  if defined (alloca) || defined (_ALLOCA_H)
#   define YYSTACK_ALLOC alloca
#  else
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
# else
#  if defined (__STDC__) || defined (__cplusplus)
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   define YYSIZE_T size_t
#  endif
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (defined (YYSTYPE_IS_TRIVIAL) && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short) + sizeof (YYSTYPE))				\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined (__GNUC__) && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  register YYSIZE_T yyi;		\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (0)
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (0)

#endif

#if defined (__STDC__) || defined (__cplusplus)
   typedef signed char yysigned_char;
#else
   typedef short yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1236

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  111
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  70
/* YYNRULES -- Number of rules. */
#define YYNRULES  228
/* YYNRULES -- Number of states. */
#define YYNSTATES  454

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   351

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     100,   101,   109,     2,    98,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     105,    97,   106,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   102,    99,   104,     2,     2,     2,     2,     2,   110,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     103,     2,     2,   107,     2,   108,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned short yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    11,    13,    15,    17,
      19,    21,    23,    25,    27,    29,    31,    33,    35,    37,
      39,    41,    43,    45,    47,    49,    51,    53,    55,    57,
      59,    61,    63,    65,    67,    70,    71,    73,    75,    77,
      79,    80,    81,    83,    85,    87,    90,    91,    94,    95,
      99,   102,   103,   105,   106,   110,   112,   115,   117,   119,
     121,   123,   125,   127,   129,   131,   133,   135,   137,   139,
     141,   143,   145,   147,   149,   151,   153,   155,   157,   160,
     165,   171,   177,   181,   184,   187,   189,   193,   195,   199,
     201,   202,   207,   211,   215,   220,   225,   229,   232,   235,
     238,   241,   244,   247,   250,   253,   256,   259,   266,   272,
     281,   288,   295,   302,   309,   316,   320,   322,   324,   326,
     328,   331,   334,   337,   339,   344,   347,   348,   356,   357,
     365,   369,   374,   375,   377,   379,   383,   387,   391,   395,
     399,   401,   402,   404,   406,   408,   409,   412,   416,   418,
     420,   424,   426,   427,   436,   438,   440,   444,   446,   448,
     451,   452,   456,   458,   460,   462,   464,   466,   468,   470,
     472,   476,   478,   480,   482,   484,   486,   489,   492,   495,
     499,   502,   503,   505,   508,   511,   515,   525,   535,   544,
     558,   560,   562,   569,   575,   578,   585,   593,   595,   599,
     601,   602,   605,   607,   613,   619,   625,   628,   633,   638,
     645,   650,   655,   660,   665,   668,   676,   678,   681,   682,
     684,   685,   689,   696,   700,   707,   710,   715,   722
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
     142,     0,    -1,     5,    -1,     6,    -1,     3,    -1,     4,
      -1,    68,    -1,    69,    -1,    70,    -1,    71,    -1,    72,
      -1,    73,    -1,    74,    -1,    75,    -1,    76,    -1,    77,
      -1,    78,    -1,    79,    -1,    80,    -1,    81,    -1,    91,
      -1,    92,    -1,    16,    -1,    14,    -1,    12,    -1,    10,
      -1,    17,    -1,    15,    -1,    13,    -1,    11,    -1,   118,
      -1,   119,    -1,    18,    -1,    19,    -1,   151,    97,    -1,
      -1,    41,    -1,    42,    -1,    43,    -1,    44,    -1,    -1,
      -1,    59,    -1,    60,    -1,    61,    -1,    58,     4,    -1,
      -1,    54,     4,    -1,    -1,    98,    54,     4,    -1,    34,
      24,    -1,    -1,   127,    -1,    -1,    98,   130,   129,    -1,
     127,    -1,    54,     4,    -1,   133,    -1,     8,    -1,   135,
      -1,     8,    -1,   135,    -1,     9,    -1,    10,    -1,    11,
      -1,    12,    -1,    13,    -1,    14,    -1,    15,    -1,    16,
      -1,    17,    -1,    18,    -1,    19,    -1,    20,    -1,    21,
      -1,    45,    -1,   134,    -1,   164,    -1,    99,     4,    -1,
     132,   100,   137,   101,    -1,   102,     4,   103,   135,   104,
      -1,   105,     4,   103,   135,   106,    -1,   107,   136,   108,
      -1,   107,   108,    -1,   135,   109,    -1,   135,    -1,   136,
      98,   135,    -1,   136,    -1,   136,    98,    37,    -1,    37,
      -1,    -1,   133,   102,   140,   104,    -1,   133,   102,   104,
      -1,   133,   110,    24,    -1,   133,   105,   140,   106,    -1,
     133,   107,   140,   108,    -1,   133,   107,   108,    -1,   133,
      38,    -1,   133,    39,    -1,   133,   164,    -1,   133,   139,
      -1,   133,    26,    -1,   118,   113,    -1,   119,     4,    -1,
       9,    27,    -1,     9,    28,    -1,   121,     7,    -1,    89,
     100,   138,    36,   133,   101,    -1,    87,   100,   138,   178,
     101,    -1,    90,   100,   138,    98,   138,    98,   138,   101,
      -1,   114,   100,   138,    98,   138,   101,    -1,   115,   100,
     138,    98,   138,   101,    -1,   116,   100,   138,    98,   138,
     101,    -1,   117,   100,   138,    98,   138,   101,    -1,    94,
     100,   138,    98,   138,   101,    -1,   140,    98,   138,    -1,
     138,    -1,    32,    -1,    33,    -1,   143,    -1,   143,   160,
      -1,   143,   161,    -1,   143,    25,    -1,   144,    -1,   144,
     122,    20,   131,    -1,   144,   161,    -1,    -1,   144,   122,
     123,   141,   138,   145,   129,    -1,    -1,   144,   122,    47,
     141,   133,   146,   129,    -1,   144,    48,   148,    -1,   144,
      55,    97,   149,    -1,    -1,    53,    -1,    52,    -1,    50,
      97,   147,    -1,    51,    97,     4,    -1,    49,    97,    24,
      -1,   102,   150,   104,    -1,   150,    98,    24,    -1,    24,
      -1,    -1,    22,    -1,    24,    -1,   151,    -1,    -1,   133,
     152,    -1,   154,    98,   153,    -1,   153,    -1,   154,    -1,
     154,    98,    37,    -1,    37,    -1,    -1,   124,   131,   151,
     100,   155,   101,   128,   125,    -1,    29,    -1,   107,    -1,
     123,   156,   157,    -1,    30,    -1,   108,    -1,   167,   159,
      -1,    -1,    31,   162,   156,    -1,     3,    -1,     4,    -1,
       7,    -1,    27,    -1,    28,    -1,    38,    -1,    39,    -1,
      26,    -1,   105,   140,   106,    -1,   139,    -1,   112,    -1,
     151,    -1,   164,    -1,   163,    -1,   133,   165,    -1,   167,
     168,    -1,   158,   168,    -1,   169,   122,   170,    -1,   169,
     172,    -1,    -1,    23,    -1,    62,   166,    -1,    62,     8,
      -1,    63,    21,   165,    -1,    63,     9,   165,    98,    21,
     165,    98,    21,   165,    -1,    64,   120,   165,    98,    21,
     165,   102,   171,   104,    -1,    64,   120,   165,    98,    21,
     165,   102,   104,    -1,    65,   124,   131,   165,   100,   175,
     101,    36,    21,   165,    66,    21,   165,    -1,    66,    -1,
      67,    -1,   171,   120,   163,    98,    21,   165,    -1,   120,
     163,    98,    21,   165,    -1,   122,   177,    -1,   133,   102,
     165,    98,   165,   104,    -1,   173,    98,   102,   165,    98,
     165,   104,    -1,   166,    -1,   174,    98,   166,    -1,   174,
      -1,    -1,    57,    56,    -1,    56,    -1,   114,   133,   165,
      98,   165,    -1,   115,   133,   165,    98,   165,    -1,   116,
     133,   165,    98,   165,    -1,    46,   166,    -1,   117,   166,
      98,   166,    -1,    89,   166,    36,   133,    -1,    90,   166,
      98,   166,    98,   166,    -1,    93,   166,    98,   133,    -1,
      95,   166,    98,   133,    -1,    96,   166,    98,   133,    -1,
      94,   166,    98,   166,    -1,    88,   173,    -1,   176,   124,
     131,   165,   100,   175,   101,    -1,   180,    -1,    98,   174,
      -1,    -1,    35,    -1,    -1,    82,   133,   126,    -1,    82,
     133,    98,    15,   165,   126,    -1,    83,   133,   126,    -1,
      83,   133,    98,    15,   165,   126,    -1,    84,   166,    -1,
     179,    85,   133,   165,    -1,   179,    86,   166,    98,   133,
     165,    -1,    87,   133,   165,   178,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   996,   996,   997,  1004,  1005,  1014,  1014,  1014,  1014,
    1014,  1015,  1015,  1015,  1016,  1016,  1016,  1016,  1016,  1016,
    1018,  1018,  1022,  1022,  1022,  1022,  1023,  1023,  1023,  1023,
    1024,  1024,  1025,  1025,  1028,  1031,  1035,  1036,  1037,  1038,
    1039,  1041,  1042,  1043,  1044,  1045,  1053,  1054,  1059,  1060,
    1067,  1074,  1075,  1080,  1081,  1082,  1086,  1099,  1099,  1100,
    1100,  1102,  1111,  1111,  1111,  1111,  1111,  1111,  1111,  1112,
    1112,  1112,  1112,  1112,  1112,  1113,  1116,  1119,  1125,  1132,
    1144,  1148,  1159,  1168,  1171,  1179,  1183,  1188,  1189,  1192,
    1195,  1205,  1230,  1243,  1271,  1296,  1316,  1328,  1337,  1341,
    1400,  1406,  1414,  1419,  1424,  1427,  1430,  1437,  1447,  1478,
    1485,  1506,  1516,  1521,  1528,  1538,  1541,  1548,  1548,  1558,
    1565,  1569,  1572,  1575,  1588,  1608,  1610,  1610,  1616,  1616,
    1623,  1625,  1627,  1632,  1633,  1635,  1638,  1646,  1651,  1653,
    1657,  1661,  1669,  1669,  1670,  1670,  1672,  1678,  1683,  1689,
    1692,  1697,  1701,  1705,  1791,  1791,  1793,  1801,  1801,  1803,
    1807,  1807,  1816,  1819,  1822,  1825,  1828,  1831,  1834,  1837,
    1840,  1864,  1871,  1874,  1879,  1879,  1885,  1889,  1892,  1900,
    1909,  1913,  1923,  1934,  1937,  1940,  1943,  1946,  1960,  1964,
    2017,  2020,  2026,  2034,  2044,  2051,  2056,  2063,  2067,  2073,
    2073,  2075,  2078,  2084,  2096,  2107,  2117,  2129,  2136,  2143,
    2150,  2155,  2174,  2196,  2204,  2218,  2275,  2281,  2283,  2287,
    2290,  2296,  2300,  2304,  2308,  2312,  2319,  2329,  2342
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "ESINT64VAL", "EUINT64VAL", "SINTVAL",
  "UINTVAL", "FPVAL", "VOID", "BOOL", "SBYTE", "UBYTE", "SHORT", "USHORT",
  "INT", "UINT", "LONG", "ULONG", "FLOAT", "DOUBLE", "TYPE", "LABEL",
  "VAR_ID", "LABELSTR", "STRINGCONSTANT", "IMPLEMENTATION",
  "ZEROINITIALIZER", "TRUETOK", "FALSETOK", "BEGINTOK", "ENDTOK",
  "DECLARE", "GLOBAL", "CONSTANT", "SECTION", "VOLATILE", "TO",
  "DOTDOTDOT", "NULL_TOK", "UNDEF", "CONST", "INTERNAL", "LINKONCE",
  "WEAK", "APPENDING", "OPAQUE", "NOT", "EXTERNAL", "TARGET", "TRIPLE",
  "ENDIAN", "POINTERSIZE", "LITTLE", "BIG", "ALIGN", "DEPLIBS", "CALL",
  "TAIL", "CC_TOK", "CCC_TOK", "FASTCC_TOK", "COLDCC_TOK", "RET", "BR",
  "SWITCH", "INVOKE", "UNWIND", "UNREACHABLE", "ADD", "SUB", "MUL", "DIV",
  "REM", "AND", "OR", "XOR", "SETLE", "SETGE", "SETLT", "SETGT", "SETEQ",
  "SETNE", "MALLOC", "ALLOCA", "FREE", "LOAD", "STORE", "GETELEMENTPTR",
  "PHI_TOK", "CAST", "SELECT", "SHL", "SHR", "VAARG", "EXTRACTELEMENT",
  "VAARG_old", "VANEXT_old", "'='", "','", "'\\\\'", "'('", "')'", "'['",
  "'x'", "']'", "'<'", "'>'", "'{'", "'}'", "'*'", "'c'", "$accept",
  "INTVAL", "EINT64VAL", "ArithmeticOps", "LogicalOps", "SetCondOps",
  "ShiftOps", "SIntType", "UIntType", "IntType", "FPType", "OptAssign",
  "OptLinkage", "OptCallingConv", "OptAlign", "OptCAlign", "SectionString",
  "OptSection", "GlobalVarAttributes", "GlobalVarAttribute", "TypesV",
  "UpRTypesV", "Types", "PrimType", "UpRTypes", "TypeListI",
  "ArgTypeListI", "ConstVal", "ConstExpr", "ConstVector", "GlobalType",
  "Module", "FunctionList", "ConstPool", "@1", "@2", "BigOrLittle",
  "TargetDefinition", "LibrariesDefinition", "LibList", "Name", "OptName",
  "ArgVal", "ArgListH", "ArgList", "FunctionHeaderH", "BEGIN",
  "FunctionHeader", "END", "Function", "FunctionProto", "@3",
  "ConstValueRef", "SymbolicValueRef", "ValueRef", "ResolvedVal",
  "BasicBlockList", "BasicBlock", "InstructionList", "BBTerminatorInst",
  "JumpTable", "Inst", "PHIList", "ValueRefList", "ValueRefListE",
  "OptTailCall", "InstVal", "IndexList", "OptVolatile", "MemoryInst", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,    61,    44,    92,
      40,    41,    91,   120,    93,    60,    62,   123,   125,    42,
      99
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   111,   112,   112,   113,   113,   114,   114,   114,   114,
     114,   115,   115,   115,   116,   116,   116,   116,   116,   116,
     117,   117,   118,   118,   118,   118,   119,   119,   119,   119,
     120,   120,   121,   121,   122,   122,   123,   123,   123,   123,
     123,   124,   124,   124,   124,   124,   125,   125,   126,   126,
     127,   128,   128,   129,   129,   130,   130,   131,   131,   132,
     132,   133,   134,   134,   134,   134,   134,   134,   134,   134,
     134,   134,   134,   134,   134,   135,   135,   135,   135,   135,
     135,   135,   135,   135,   135,   136,   136,   137,   137,   137,
     137,   138,   138,   138,   138,   138,   138,   138,   138,   138,
     138,   138,   138,   138,   138,   138,   138,   139,   139,   139,
     139,   139,   139,   139,   139,   140,   140,   141,   141,   142,
     143,   143,   143,   143,   144,   144,   145,   144,   146,   144,
     144,   144,   144,   147,   147,   148,   148,   148,   149,   150,
     150,   150,   151,   151,   152,   152,   153,   154,   154,   155,
     155,   155,   155,   156,   157,   157,   158,   159,   159,   160,
     162,   161,   163,   163,   163,   163,   163,   163,   163,   163,
     163,   163,   164,   164,   165,   165,   166,   167,   167,   168,
     169,   169,   169,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   171,   171,   172,   173,   173,   174,   174,   175,
     175,   176,   176,   177,   177,   177,   177,   177,   177,   177,
     177,   177,   177,   177,   177,   177,   177,   178,   178,   179,
     179,   180,   180,   180,   180,   180,   180,   180,   180
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     0,     1,     1,     1,     1,
       0,     0,     1,     1,     1,     2,     0,     2,     0,     3,
       2,     0,     1,     0,     3,     1,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     4,
       5,     5,     3,     2,     2,     1,     3,     1,     3,     1,
       0,     4,     3,     3,     4,     4,     3,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     6,     5,     8,
       6,     6,     6,     6,     6,     3,     1,     1,     1,     1,
       2,     2,     2,     1,     4,     2,     0,     7,     0,     7,
       3,     4,     0,     1,     1,     3,     3,     3,     3,     3,
       1,     0,     1,     1,     1,     0,     2,     3,     1,     1,
       3,     1,     0,     8,     1,     1,     3,     1,     1,     2,
       0,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     1,     1,     1,     1,     1,     2,     2,     2,     3,
       2,     0,     1,     2,     2,     3,     9,     9,     8,    13,
       1,     1,     6,     5,     2,     6,     7,     1,     3,     1,
       0,     2,     1,     5,     5,     5,     2,     4,     4,     6,
       4,     4,     4,     4,     2,     7,     1,     2,     0,     1,
       0,     3,     6,     3,     6,     2,     4,     6,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
     132,     0,    40,   123,     1,   122,   160,    36,    37,    38,
      39,    41,   181,   120,   121,   181,   142,   143,     0,     0,
      40,     0,   125,    41,     0,    42,    43,    44,     0,     0,
     182,   178,    35,   157,   158,   159,   177,     0,     0,     0,
     130,     0,     0,     0,     0,    34,   161,    45,     2,     3,
      58,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,     0,     0,     0,     0,   172,
       0,     0,    57,    76,    61,   173,    77,   154,   155,   156,
     220,   180,     0,     0,     0,   141,   131,   124,   117,   118,
       0,     0,    78,     0,     0,    60,    83,    85,     0,     0,
      90,    84,   219,     0,   202,     0,     0,     0,     0,    41,
     190,   191,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,     0,     0,     0,     0,
       0,     0,     0,    20,    21,     0,     0,     0,     0,     0,
       0,     0,     0,   179,    41,   194,     0,   216,   137,   134,
     133,   135,   136,   140,     0,   128,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,     0,     0,     0,
       0,   126,     0,     0,     0,    82,   152,    89,    87,     0,
       0,   206,   201,   184,   183,     0,     0,    25,    29,    24,
      28,    23,    27,    22,    26,    30,    31,     0,     0,    48,
      48,   225,     0,     0,   214,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   138,
      53,   104,   105,     4,     5,   102,   103,   106,   101,    97,
      98,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   100,    99,    53,    59,    59,    86,   151,
     145,   148,   149,     0,     0,    79,   162,   163,   164,   169,
     165,   166,   167,   168,     0,   171,   175,   174,   176,     0,
     185,     0,     0,     0,   221,     0,   223,   218,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   139,     0,   129,     0,     0,     0,     0,
      92,   116,     0,     0,    96,     0,    93,     0,     0,     0,
       0,   127,    80,    81,   144,   146,     0,    51,    88,     0,
       0,     0,     0,     0,     0,     0,     0,   228,     0,     0,
     208,     0,   210,   213,   211,   212,     0,     0,     0,   207,
       0,   226,     0,     0,     0,    55,    53,   218,     0,     0,
       0,     0,    91,    94,    95,     0,     0,     0,     0,   150,
     147,    52,    46,   170,     0,     0,   200,    48,    49,    48,
     197,   217,     0,     0,     0,   203,   204,   205,   200,     0,
      50,    56,    54,     0,     0,     0,     0,   115,     0,     0,
       0,     0,     0,   153,     0,     0,   199,     0,     0,   222,
     224,     0,     0,     0,   209,     0,   227,   108,     0,     0,
       0,     0,     0,     0,     0,    47,     0,     0,     0,   198,
     195,     0,   215,   107,     0,   114,   110,   111,   112,   113,
       0,   188,     0,     0,     0,   196,     0,   186,     0,   187,
       0,     0,   109,     0,     0,     0,     0,     0,     0,   193,
       0,     0,   192,   189
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,    69,   225,   239,   240,   241,   242,   167,   168,   197,
     169,    20,    11,    28,   393,   274,   345,   362,   295,   346,
      70,    71,   170,    73,    74,    98,   179,   301,   265,   302,
      90,     1,     2,     3,   245,   220,   151,    40,    86,   154,
      75,   315,   251,   252,   253,    29,    79,    12,    35,    13,
      14,    23,   266,    76,   268,   370,    15,    31,    32,   143,
     433,    81,   204,   396,   397,   144,   145,   327,   146,   147
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -408
static const short yypact[] =
{
    -408,    21,   120,   181,  -408,  -408,  -408,  -408,  -408,  -408,
    -408,    63,     1,  -408,  -408,   -14,  -408,  -408,    87,   -66,
      48,   -23,  -408,    63,    79,  -408,  -408,  -408,   979,   -22,
    -408,  -408,    62,  -408,  -408,  -408,  -408,   -10,    37,    47,
    -408,    44,   979,    64,    64,  -408,  -408,  -408,  -408,  -408,
      65,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,  -408,   165,   170,   172,   491,  -408,
      62,    80,  -408,  -408,   -82,  -408,  -408,  -408,  -408,  -408,
    1140,  -408,   157,    53,   178,   160,  -408,  -408,  -408,  -408,
    1017,  1055,  -408,    83,    89,  -408,  -408,   -82,   -85,    90,
     789,  -408,  -408,  1017,  -408,   137,  1093,    11,   116,    63,
    -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,  -408,  -408,  1017,  1017,  1017,  1017,
    1017,  1017,  1017,  -408,  -408,  1017,  1017,  1017,  1017,  1017,
    1017,  1017,  1017,  -408,    63,  -408,    57,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,   -62,  -408,   122,   150,   190,   152,
     191,   155,   192,   168,   194,   193,   195,   175,   200,   199,
     360,  -408,  1017,  1017,  1017,  -408,   827,  -408,   101,   110,
     610,  -408,  -408,    65,  -408,   610,   610,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,  -408,  -408,  -408,   610,   979,   117,
     118,  -408,   610,   115,   128,   196,   132,   133,   136,   139,
     140,   610,   610,   610,   141,   979,  1017,  1017,   211,  -408,
     142,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,
    -408,   143,   144,   147,   148,   865,  1055,   567,   217,   149,
     151,   158,   163,  -408,  -408,   142,   -33,   -39,   -82,  -408,
      62,  -408,   162,   164,   903,  -408,  -408,  -408,  -408,  -408,
    -408,  -408,  -408,  -408,  1055,  -408,  -408,  -408,  -408,   166,
    -408,   169,   610,     7,  -408,    10,  -408,   171,   610,   173,
    1017,  1017,  1017,  1017,  1017,  1017,   174,   176,   179,  1017,
     610,   610,   183,  -408,   -19,  -408,  1055,  1055,  1055,  1055,
    -408,  -408,   -16,   -29,  -408,   -79,  -408,  1055,  1055,  1055,
    1055,  -408,  -408,  -408,  -408,  -408,   941,   208,  -408,   -26,
     229,   245,   182,   610,   264,   610,  1017,  -408,   185,   610,
    -408,   187,  -408,  -408,  -408,  -408,   610,   610,   610,  -408,
     186,  -408,  1017,   246,   267,  -408,   142,   171,   237,   189,
     197,  1055,  -408,  -408,  -408,   209,   210,   213,   215,  -408,
    -408,  -408,   225,  -408,   610,   610,  1017,   219,  -408,   219,
    -408,   220,   610,   224,  1017,  -408,  -408,  -408,  1017,   610,
    -408,  -408,  -408,   188,  1017,  1055,  1055,  -408,  1055,  1055,
    1055,  1055,   286,  -408,   226,   204,   220,   214,   240,  -408,
    -408,  1017,   205,   610,  -408,   222,  -408,  -408,   228,   227,
     232,   235,   236,   238,   239,  -408,   320,    35,   306,  -408,
    -408,   241,  -408,  -408,  1055,  -408,  -408,  -408,  -408,  -408,
     610,  -408,   691,    43,   322,  -408,   243,  -408,   249,  -408,
     691,   610,  -408,   327,   251,   285,   610,   331,   332,  -408,
     610,   610,  -408,  -408
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -408,  -408,  -408,   274,   275,   277,   279,  -107,  -105,  -407,
    -408,   328,   341,   -81,  -408,  -194,    45,  -408,  -228,  -408,
     -38,  -408,   -28,  -408,   -57,   263,  -408,   -89,   198,  -171,
     323,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,  -408,
       9,  -408,    54,  -408,  -408,   346,  -408,  -408,  -408,  -408,
     361,  -408,  -359,   -45,   119,   -98,  -408,   356,  -408,  -408,
    -408,  -408,  -408,    46,    -4,  -408,  -408,    28,  -408,  -408
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -120
static const short yytable[] =
{
      72,   195,   171,   196,    87,   181,   276,    77,   184,    30,
     432,    97,    21,   174,    72,   343,    33,   311,   -59,   351,
     185,     4,   323,   175,    30,   325,   440,   101,   198,   354,
     201,    41,   186,   205,   206,   344,   218,   207,   208,   209,
     210,    21,   219,    97,   214,   187,   188,   189,   190,   191,
     192,   193,   194,   187,   188,   189,   190,   191,   192,   193,
     194,   324,   155,   215,   324,   303,   305,   313,    42,   351,
     101,   312,   351,   438,    45,   180,   101,   353,   180,    99,
     363,   444,   351,    47,    16,    78,    17,    82,   352,     7,
       8,     9,    10,   319,    34,    43,    88,    89,   199,   200,
     180,   202,   203,   180,   180,   149,   150,   180,   180,   180,
     180,   211,   212,   213,   180,   246,   247,   248,   382,   292,
    -119,    24,    25,    26,    27,   244,   187,   188,   189,   190,
     191,   192,   193,   194,    83,   267,    37,    38,    39,   431,
     267,   267,   216,   217,    84,     5,    85,   439,   250,   221,
     222,     6,   267,   -25,   -25,   -24,   -24,   267,   -23,   -23,
     272,     7,     8,     9,    10,   -60,   267,   267,   267,    92,
      72,   -22,   -22,   399,    93,   400,    94,   290,   223,   224,
     100,   148,   152,   331,   153,   333,   172,    72,   291,   180,
     176,   339,   173,   182,   -29,   -28,   -27,   248,   -26,   254,
     -32,   -35,   -33,    16,   226,    17,   227,   347,   348,   349,
     350,   255,     6,   -35,   -35,   273,   275,   278,   355,   356,
     357,   358,   -35,   -35,   -35,   -35,   279,   267,   -35,    18,
     281,   282,   280,   267,   283,   293,    19,   284,   285,   289,
     294,   306,   343,   296,   297,   267,   267,   298,   299,   307,
     364,   308,   330,   180,   332,   180,   334,   335,   309,   314,
     316,   180,   387,   310,   320,   317,   365,   321,   368,   326,
     380,   381,   336,   384,   337,   329,   404,   338,   267,   392,
     267,   342,   366,   372,   267,   374,   378,   385,   250,   407,
     415,   267,   267,   267,   324,   386,   409,   410,   180,   411,
     412,   413,   414,   419,   269,   270,   417,   388,   389,   420,
     195,   390,   196,   391,   379,   418,   271,   398,   401,   267,
     267,   277,   403,   422,   416,   424,   195,   267,   196,   423,
     286,   287,   288,   425,   267,   436,   426,   427,   180,   428,
     429,   430,   434,   441,   442,   435,   180,   443,   446,   447,
     180,   448,   450,   451,   139,   140,   408,   141,   267,   142,
      80,    44,   361,   178,    22,    48,    49,    91,   243,    46,
     360,    36,   371,   180,   405,   383,     0,     0,     0,     0,
       0,     0,    16,     0,    17,   267,   228,     0,     0,     0,
       0,   322,     0,     0,     0,     0,   267,   328,   229,   230,
       0,   267,     0,     0,     0,   267,   267,     0,     0,   340,
     341,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   367,     0,   369,     0,     0,   231,   373,   232,
     233,   133,   134,     0,   234,   375,   376,   377,     0,     0,
       0,     0,   235,     0,     0,   236,     0,   237,     0,     0,
     238,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   394,   395,     0,     0,     0,     0,     0,
       0,   402,     0,     0,     0,     0,    48,    49,   406,    95,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    16,     0,    17,     0,     0,     0,     0,
       0,     0,   421,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    64,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   437,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     445,     0,     0,     0,     0,   449,     0,     0,     0,   452,
     453,     0,    48,    49,     0,    95,   156,   157,   158,   159,
     160,   161,   162,   163,   164,   165,   166,    62,    63,    16,
      65,    17,     0,    66,     0,     0,    67,     0,    68,    96,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    64,   256,   257,    48,    49,   258,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    16,     0,    17,     0,   259,   260,   261,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   262,   263,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    65,     0,     0,    66,
       0,     0,    67,     0,    68,   304,     0,     0,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,     0,     0,   256,   257,     0,   231,   258,   232,
     233,   133,   134,     0,   234,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   264,     0,   259,   260,   261,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   262,
     263,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,     0,     0,     0,     0,     0,   231,     0,
     232,   233,   133,   134,     0,   234,     0,     0,     0,     0,
       0,     0,     0,     0,    48,    49,   264,    95,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    16,     0,    17,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   177,     0,     0,     0,
       0,     0,    48,    49,    64,    95,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    16,
       0,    17,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   249,     0,     0,     0,     0,     0,
      48,    49,    64,    95,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,    62,    63,    16,    65,    17,
       0,    66,     0,     0,    67,     0,    68,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
      64,    95,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    16,    65,    17,     0,    66,
       0,     0,    67,     0,    68,     0,     0,     0,     0,     0,
     318,     0,     0,     0,     0,     0,    48,    49,    64,    95,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    16,    65,    17,     0,    66,     0,   300,
      67,     0,    68,     0,     0,     0,     0,     0,   359,     0,
       0,     0,     0,     0,    48,    49,    64,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    16,    65,    17,     0,    66,     0,     0,    67,     0,
      68,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,    49,    64,    95,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    16,
      65,    17,     0,    66,     0,     0,    67,     0,    68,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,    49,    64,    95,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,    62,    63,    16,    65,    17,
       0,    66,     0,     0,    67,     0,    68,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,    49,
      64,   183,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    16,    65,    17,     0,    66,
       0,     0,    67,     0,    68,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    64,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    65,     0,     0,    66,     0,     0,
      67,     0,    68,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   102,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   103,     0,     0,     0,
       0,     0,    65,     0,     0,    66,   104,   105,    67,     0,
      68,     0,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,     0,     0,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138
};

static const short yycheck[] =
{
      28,   108,    91,   108,    42,   103,   200,    29,   106,    23,
     417,    68,     3,    98,    42,    34,    30,   245,   100,    98,
       9,     0,    15,   108,    23,    15,   433,   109,   109,   108,
     128,    97,    21,   131,   132,    54,    98,   135,   136,   137,
     138,    32,   104,   100,   142,    10,    11,    12,    13,    14,
      15,    16,    17,    10,    11,    12,    13,    14,    15,    16,
      17,    54,    90,   144,    54,   236,   237,   106,    20,    98,
     109,   104,    98,   432,    97,   103,   109,   106,   106,    70,
     106,   440,    98,     4,    22,   107,    24,    97,   104,    41,
      42,    43,    44,   264,   108,    47,    32,    33,   126,   127,
     128,   129,   130,   131,   132,    52,    53,   135,   136,   137,
     138,   139,   140,   141,   142,   172,   173,   174,   346,   217,
       0,    58,    59,    60,    61,   170,    10,    11,    12,    13,
      14,    15,    16,    17,    97,   180,    49,    50,    51,   104,
     185,   186,    85,    86,    97,    25,   102,   104,   176,    27,
      28,    31,   197,     3,     4,     3,     4,   202,     3,     4,
     198,    41,    42,    43,    44,   100,   211,   212,   213,     4,
     198,     3,     4,   367,     4,   369,     4,   215,     3,     4,
     100,    24,     4,   281,    24,   283,   103,   215,   216,   217,
     100,   289,   103,    56,     4,     4,     4,   254,     4,    98,
       7,    20,     7,    22,     4,    24,     7,   296,   297,   298,
     299,   101,    31,    32,    33,    98,    98,   102,   307,   308,
     309,   310,    41,    42,    43,    44,    98,   272,    47,    48,
      98,    98,    36,   278,    98,    24,    55,    98,    98,    98,
      98,    24,    34,   100,   100,   290,   291,   100,   100,   100,
      21,   100,   280,   281,   282,   283,   284,   285,   100,   250,
      98,   289,   351,   100,    98,   101,    21,    98,     4,    98,
      24,     4,    98,    36,    98,   102,   374,    98,   323,    54,
     325,    98,   100,    98,   329,    98,   100,    98,   316,   101,
       4,   336,   337,   338,    54,    98,   385,   386,   326,   388,
     389,   390,   391,   401,   185,   186,   102,    98,    98,   104,
     417,    98,   417,    98,   342,   101,   197,    98,    98,   364,
     365,   202,    98,   101,    98,    98,   433,   372,   433,   101,
     211,   212,   213,   101,   379,   424,   101,   101,   366,   101,
     101,    21,    36,    21,   101,   104,   374,    98,    21,    98,
     378,    66,    21,    21,    80,    80,   384,    80,   403,    80,
      32,    20,   317,   100,     3,     5,     6,    44,   170,    23,
     316,    15,   326,   401,   378,   347,    -1,    -1,    -1,    -1,
      -1,    -1,    22,    -1,    24,   430,    26,    -1,    -1,    -1,
      -1,   272,    -1,    -1,    -1,    -1,   441,   278,    38,    39,
      -1,   446,    -1,    -1,    -1,   450,   451,    -1,    -1,   290,
     291,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,   323,    -1,   325,    -1,    -1,    87,   329,    89,
      90,    91,    92,    -1,    94,   336,   337,   338,    -1,    -1,
      -1,    -1,   102,    -1,    -1,   105,    -1,   107,    -1,    -1,
     110,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   364,   365,    -1,    -1,    -1,    -1,    -1,
      -1,   372,    -1,    -1,    -1,    -1,     5,     6,   379,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    -1,    24,    -1,    -1,    -1,    -1,
      -1,    -1,   403,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   430,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     441,    -1,    -1,    -1,    -1,   446,    -1,    -1,    -1,   450,
     451,    -1,     5,     6,    -1,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      99,    24,    -1,   102,    -1,    -1,   105,    -1,   107,   108,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    45,     3,     4,     5,     6,     7,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    22,    -1,    24,    -1,    26,    27,    28,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    39,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,   102,
      -1,    -1,   105,    -1,   107,   108,    -1,    -1,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    -1,    -1,     3,     4,    -1,    87,     7,    89,
      90,    91,    92,    -1,    94,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   105,    -1,    26,    27,    28,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,
      39,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    -1,    -1,    -1,    -1,    -1,    87,    -1,
      89,    90,    91,    92,    -1,    94,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     5,     6,   105,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    -1,    24,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,     5,     6,    45,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      -1,    24,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
       5,     6,    45,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    99,    24,
      -1,   102,    -1,    -1,   105,    -1,   107,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     5,     6,
      45,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    99,    24,    -1,   102,
      -1,    -1,   105,    -1,   107,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,    -1,    -1,    -1,     5,     6,    45,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    99,    24,    -1,   102,    -1,   104,
     105,    -1,   107,    -1,    -1,    -1,    -1,    -1,    37,    -1,
      -1,    -1,    -1,    -1,     5,     6,    45,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    99,    24,    -1,   102,    -1,    -1,   105,    -1,
     107,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     5,     6,    45,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      99,    24,    -1,   102,    -1,    -1,   105,    -1,   107,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       5,     6,    45,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    99,    24,
      -1,   102,    -1,    -1,   105,    -1,   107,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     5,     6,
      45,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    99,    24,    -1,   102,
      -1,    -1,   105,    -1,   107,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    99,    -1,    -1,   102,    -1,    -1,
     105,    -1,   107,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    35,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    46,    -1,    -1,    -1,
      -1,    -1,    99,    -1,    -1,   102,    56,    57,   105,    -1,
     107,    -1,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    -1,    -1,    87,    88,    89,
      90,    91,    92,    93,    94,    95,    96
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,   142,   143,   144,     0,    25,    31,    41,    42,    43,
      44,   123,   158,   160,   161,   167,    22,    24,    48,    55,
     122,   151,   161,   162,    58,    59,    60,    61,   124,   156,
      23,   168,   169,    30,   108,   159,   168,    49,    50,    51,
     148,    97,    20,    47,   123,    97,   156,     4,     5,     6,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    45,    99,   102,   105,   107,   112,
     131,   132,   133,   134,   135,   151,   164,    29,   107,   157,
     122,   172,    97,    97,    97,   102,   149,   131,    32,    33,
     141,   141,     4,     4,     4,     8,   108,   135,   136,   151,
     100,   109,    35,    46,    56,    57,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,   114,
     115,   116,   117,   170,   176,   177,   179,   180,    24,    52,
      53,   147,     4,    24,   150,   133,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,   118,   119,   121,
     133,   138,   103,   103,    98,   108,   100,    37,   136,   137,
     133,   166,    56,     8,   166,     9,    21,    10,    11,    12,
      13,    14,    15,    16,    17,   118,   119,   120,   124,   133,
     133,   166,   133,   133,   173,   166,   166,   166,   166,   166,
     166,   133,   133,   133,   166,   124,    85,    86,    98,   104,
     146,    27,    28,     3,     4,   113,     4,     7,    26,    38,
      39,    87,    89,    90,    94,   102,   105,   107,   110,   114,
     115,   116,   117,   139,   164,   145,   135,   135,   135,    37,
     133,   153,   154,   155,    98,   101,     3,     4,     7,    26,
      27,    28,    38,    39,   105,   139,   163,   164,   165,   165,
     165,   165,   131,    98,   126,    98,   126,   165,   102,    98,
      36,    98,    98,    98,    98,    98,   165,   165,   165,    98,
     131,   133,   166,    24,    98,   129,   100,   100,   100,   100,
     104,   138,   140,   140,   108,   140,    24,   100,   100,   100,
     100,   129,   104,   106,   151,   152,    98,   101,    37,   140,
      98,    98,   165,    15,    54,    15,    98,   178,   165,   102,
     133,   166,   133,   166,   133,   133,    98,    98,    98,   166,
     165,   165,    98,    34,    54,   127,   130,   138,   138,   138,
     138,    98,   104,   106,   108,   138,   138,   138,   138,    37,
     153,   127,   128,   106,    21,    21,   100,   165,     4,   165,
     166,   174,    98,   165,    98,   165,   165,   165,   100,   133,
      24,     4,   129,   178,    36,    98,    98,   138,    98,    98,
      98,    98,    54,   125,   165,   165,   174,   175,    98,   126,
     126,    98,   165,    98,   166,   175,   165,   101,   133,   138,
     138,   138,   138,   138,   138,     4,    98,   102,   101,   166,
     104,   165,   101,   101,    98,   101,   101,   101,   101,   101,
      21,   104,   120,   171,    36,   104,   138,   165,   163,   104,
     120,    21,   101,    98,   163,   165,    21,    98,    66,   165,
      21,    21,   165,   165
};

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# if defined (__STDC__) || defined (__cplusplus)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# endif
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { 								\
      yyerror ("syntax error: cannot back up");\
      YYERROR;							\
    }								\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

/* YYLLOC_DEFAULT -- Compute the default location (before the actions
   are run).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)		\
   ((Current).first_line   = (Rhs)[1].first_line,	\
    (Current).first_column = (Rhs)[1].first_column,	\
    (Current).last_line    = (Rhs)[N].last_line,	\
    (Current).last_column  = (Rhs)[N].last_column)
#endif

/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (0)

# define YYDSYMPRINT(Args)			\
do {						\
  if (yydebug)					\
    yysymprint Args;				\
} while (0)

# define YYDSYMPRINTF(Title, Token, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr, 					\
                  Token, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short *bottom, short *top)
#else
static void
yy_stack_print (bottom, top)
    short *bottom;
    short *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (/* Nothing. */; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_reduce_print (int yyrule)
#else
static void
yy_reduce_print (yyrule)
    int yyrule;
#endif
{
  int yyi;
  unsigned int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             yyrule - 1, yylno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname [yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname [yyr1[yyrule]]);
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (Rule);		\
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YYDSYMPRINT(Args)
# define YYDSYMPRINTF(Title, Token, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   SIZE_MAX < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#if defined (YYMAXDEPTH) && YYMAXDEPTH == 0
# undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined (__GLIBC__) && defined (_STRING_H)
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
#   if defined (__STDC__) || defined (__cplusplus)
yystrlen (const char *yystr)
#   else
yystrlen (yystr)
     const char *yystr;
#   endif
{
  register const char *yys = yystr;

  while (*yys++ != '\0')
    continue;

  return yys - yystr - 1;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined (__GLIBC__) && defined (_STRING_H) && defined (_GNU_SOURCE)
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
#   if defined (__STDC__) || defined (__cplusplus)
yystpcpy (char *yydest, const char *yysrc)
#   else
yystpcpy (yydest, yysrc)
     char *yydest;
     const char *yysrc;
#   endif
{
  register char *yyd = yydest;
  register const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

#endif /* !YYERROR_VERBOSE */



#if YYDEBUG
/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yysymprint (FILE *yyoutput, int yytype, YYSTYPE *yyvaluep)
#else
static void
yysymprint (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  if (yytype < YYNTOKENS)
    {
      YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
# ifdef YYPRINT
      YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
    }
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  switch (yytype)
    {
      default:
        break;
    }
  YYFPRINTF (yyoutput, ")");
}

#endif /* ! YYDEBUG */
/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yydestruct (int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yytype, yyvaluep)
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  switch (yytype)
    {

      default:
        break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM);
# else
int yyparse ();
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM)
# else
int yyparse (YYPARSE_PARAM)
  void *YYPARSE_PARAM;
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  register int yystate;
  register int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short	yyssa[YYINITDEPTH];
  short *yyss = yyssa;
  register short *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  register YYSTYPE *yyvsp;



#define YYPOPSTACK   (yyvsp--, yyssp--)

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* When reducing, the number of symbols on the RHS of the reduced
     rule.  */
  int yylen;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed. so pushing a state here evens the stacks.
     */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack. Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	short *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow ("parser stack overflow",
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyoverflowlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyoverflowlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	short *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyoverflowlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YYDSYMPRINTF ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */
  YYDPRINTF ((stderr, "Shifting token %s, ", yytname[yytoken]));

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;


  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  yystate = yyn;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 3:
#line 997 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  if (yyvsp[0].UIntVal > (uint32_t)INT32_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  yyval.SIntVal = (int32_t)yyvsp[0].UIntVal;
;}
    break;

  case 5:
#line 1005 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  if (yyvsp[0].UInt64Val > (uint64_t)INT64_MAX)     // Outside of my range!
    ThrowException("Value too large for type!");
  yyval.SInt64Val = (int64_t)yyvsp[0].UInt64Val;
;}
    break;

  case 34:
#line 1028 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.StrVal = yyvsp[-1].StrVal;
  ;}
    break;

  case 35:
#line 1031 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.StrVal = 0;
  ;}
    break;

  case 36:
#line 1035 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.Linkage = GlobalValue::InternalLinkage; ;}
    break;

  case 37:
#line 1036 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.Linkage = GlobalValue::LinkOnceLinkage; ;}
    break;

  case 38:
#line 1037 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.Linkage = GlobalValue::WeakLinkage; ;}
    break;

  case 39:
#line 1038 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.Linkage = GlobalValue::AppendingLinkage; ;}
    break;

  case 40:
#line 1039 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.Linkage = GlobalValue::ExternalLinkage; ;}
    break;

  case 41:
#line 1041 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.UIntVal = CallingConv::C; ;}
    break;

  case 42:
#line 1042 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.UIntVal = CallingConv::C; ;}
    break;

  case 43:
#line 1043 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.UIntVal = CallingConv::Fast; ;}
    break;

  case 44:
#line 1044 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.UIntVal = CallingConv::Cold; ;}
    break;

  case 45:
#line 1045 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
                   if ((unsigned)yyvsp[0].UInt64Val != yyvsp[0].UInt64Val)
                     ThrowException("Calling conv too large!");
                   yyval.UIntVal = yyvsp[0].UInt64Val;
                 ;}
    break;

  case 46:
#line 1053 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.UIntVal = 0; ;}
    break;

  case 47:
#line 1054 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  yyval.UIntVal = yyvsp[0].UInt64Val;
  if (yyval.UIntVal != 0 && !isPowerOf2_32(yyval.UIntVal))
    ThrowException("Alignment must be a power of two!");
;}
    break;

  case 48:
#line 1059 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.UIntVal = 0; ;}
    break;

  case 49:
#line 1060 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  yyval.UIntVal = yyvsp[0].UInt64Val;
  if (yyval.UIntVal != 0 && !isPowerOf2_32(yyval.UIntVal))
    ThrowException("Alignment must be a power of two!");
;}
    break;

  case 50:
#line 1067 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  for (unsigned i = 0, e = strlen(yyvsp[0].StrVal); i != e; ++i)
    if (yyvsp[0].StrVal[i] == '"' || yyvsp[0].StrVal[i] == '\\')
      ThrowException("Invalid character in section name!");
  yyval.StrVal = yyvsp[0].StrVal;
;}
    break;

  case 51:
#line 1074 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.StrVal = 0; ;}
    break;

  case 52:
#line 1075 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.StrVal = yyvsp[0].StrVal; ;}
    break;

  case 53:
#line 1080 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {;}
    break;

  case 54:
#line 1081 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {;}
    break;

  case 55:
#line 1082 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    CurGV->setSection(yyvsp[0].StrVal);
    free(yyvsp[0].StrVal);
  ;}
    break;

  case 56:
#line 1086 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[0].UInt64Val != 0 && !isPowerOf2_32(yyvsp[0].UInt64Val))
      ThrowException("Alignment must be a power of two!");
    CurGV->setAlignment(yyvsp[0].UInt64Val);
  ;}
    break;

  case 58:
#line 1099 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.TypeVal = new PATypeHolder(yyvsp[0].PrimType); ;}
    break;

  case 60:
#line 1100 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.TypeVal = new PATypeHolder(yyvsp[0].PrimType); ;}
    break;

  case 61:
#line 1102 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!UpRefs.empty())
      ThrowException("Invalid upreference in type: " + (*yyvsp[0].TypeVal)->getDescription());
    yyval.TypeVal = yyvsp[0].TypeVal;
  ;}
    break;

  case 75:
#line 1113 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.TypeVal = new PATypeHolder(OpaqueType::get());
  ;}
    break;

  case 76:
#line 1116 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.TypeVal = new PATypeHolder(yyvsp[0].PrimType);
  ;}
    break;

  case 77:
#line 1119 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {            // Named types are also simple types...
  yyval.TypeVal = new PATypeHolder(getTypeVal(yyvsp[0].ValIDVal));
;}
    break;

  case 78:
#line 1125 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                   // Type UpReference
    if (yyvsp[0].UInt64Val > (uint64_t)~0U) ThrowException("Value out of range!");
    OpaqueType *OT = OpaqueType::get();        // Use temporary placeholder
    UpRefs.push_back(UpRefRecord((unsigned)yyvsp[0].UInt64Val, OT));  // Add to vector...
    yyval.TypeVal = new PATypeHolder(OT);
    UR_OUT("New Upreference!\n");
  ;}
    break;

  case 79:
#line 1132 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {           // Function derived type?
    std::vector<const Type*> Params;
    for (std::list<llvm::PATypeHolder>::iterator I = yyvsp[-1].TypeList->begin(),
           E = yyvsp[-1].TypeList->end(); I != E; ++I)
      Params.push_back(*I);
    bool isVarArg = Params.size() && Params.back() == Type::VoidTy;
    if (isVarArg) Params.pop_back();

    yyval.TypeVal = new PATypeHolder(HandleUpRefs(FunctionType::get(*yyvsp[-3].TypeVal,Params,isVarArg)));
    delete yyvsp[-1].TypeList;      // Delete the argument list
    delete yyvsp[-3].TypeVal;      // Delete the return type handle
  ;}
    break;

  case 80:
#line 1144 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {          // Sized array type?
    yyval.TypeVal = new PATypeHolder(HandleUpRefs(ArrayType::get(*yyvsp[-1].TypeVal, (unsigned)yyvsp[-3].UInt64Val)));
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 81:
#line 1148 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {          // Packed array type?
     const llvm::Type* ElemTy = yyvsp[-1].TypeVal->get();
     if ((unsigned)yyvsp[-3].UInt64Val != yyvsp[-3].UInt64Val)
        ThrowException("Unsigned result not equal to signed result");
     if (!ElemTy->isPrimitiveType())
        ThrowException("Elemental type of a PackedType must be primitive");
     if (!isPowerOf2_32(yyvsp[-3].UInt64Val))
       ThrowException("Vector length should be a power of 2!");
     yyval.TypeVal = new PATypeHolder(HandleUpRefs(PackedType::get(*yyvsp[-1].TypeVal, (unsigned)yyvsp[-3].UInt64Val)));
     delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 82:
#line 1159 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                        // Structure type?
    std::vector<const Type*> Elements;
    for (std::list<llvm::PATypeHolder>::iterator I = yyvsp[-1].TypeList->begin(),
           E = yyvsp[-1].TypeList->end(); I != E; ++I)
      Elements.push_back(*I);

    yyval.TypeVal = new PATypeHolder(HandleUpRefs(StructType::get(Elements)));
    delete yyvsp[-1].TypeList;
  ;}
    break;

  case 83:
#line 1168 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                                  // Empty structure type?
    yyval.TypeVal = new PATypeHolder(StructType::get(std::vector<const Type*>()));
  ;}
    break;

  case 84:
#line 1171 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                             // Pointer type?
    yyval.TypeVal = new PATypeHolder(HandleUpRefs(PointerType::get(*yyvsp[-1].TypeVal)));
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 85:
#line 1179 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.TypeList = new std::list<PATypeHolder>();
    yyval.TypeList->push_back(*yyvsp[0].TypeVal); delete yyvsp[0].TypeVal;
  ;}
    break;

  case 86:
#line 1183 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    (yyval.TypeList=yyvsp[-2].TypeList)->push_back(*yyvsp[0].TypeVal); delete yyvsp[0].TypeVal;
  ;}
    break;

  case 88:
#line 1189 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    (yyval.TypeList=yyvsp[-2].TypeList)->push_back(Type::VoidTy);
  ;}
    break;

  case 89:
#line 1192 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    (yyval.TypeList = new std::list<PATypeHolder>())->push_back(Type::VoidTy);
  ;}
    break;

  case 90:
#line 1195 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.TypeList = new std::list<PATypeHolder>();
  ;}
    break;

  case 91:
#line 1205 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { // Nonempty unsized arr
    const ArrayType *ATy = dyn_cast<ArrayType>(yyvsp[-3].TypeVal->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*yyvsp[-3].TypeVal)->getDescription() + "'!");
    const Type *ETy = ATy->getElementType();
    int NumElements = ATy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)yyvsp[-1].ConstVector->size())
      ThrowException("Type mismatch: constant sized array initialized with " +
                     utostr(yyvsp[-1].ConstVector->size()) +  " arguments, but has size of " + 
                     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < yyvsp[-1].ConstVector->size(); i++) {
      if (ETy != (*yyvsp[-1].ConstVector)[i]->getType())
        ThrowException("Element #" + utostr(i) + " is not of type '" + 
                       ETy->getDescription() +"' as required!\nIt is of type '"+
                       (*yyvsp[-1].ConstVector)[i]->getType()->getDescription() + "'.");
    }

    yyval.ConstVal = ConstantArray::get(ATy, *yyvsp[-1].ConstVector);
    delete yyvsp[-3].TypeVal; delete yyvsp[-1].ConstVector;
  ;}
    break;

  case 92:
#line 1230 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const ArrayType *ATy = dyn_cast<ArrayType>(yyvsp[-2].TypeVal->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*yyvsp[-2].TypeVal)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    if (NumElements != -1 && NumElements != 0) 
      ThrowException("Type mismatch: constant sized array initialized with 0"
                     " arguments, but has size of " + itostr(NumElements) +"!");
    yyval.ConstVal = ConstantArray::get(ATy, std::vector<Constant*>());
    delete yyvsp[-2].TypeVal;
  ;}
    break;

  case 93:
#line 1243 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const ArrayType *ATy = dyn_cast<ArrayType>(yyvsp[-2].TypeVal->get());
    if (ATy == 0)
      ThrowException("Cannot make array constant with type: '" + 
                     (*yyvsp[-2].TypeVal)->getDescription() + "'!");

    int NumElements = ATy->getNumElements();
    const Type *ETy = ATy->getElementType();
    char *EndStr = UnEscapeLexed(yyvsp[0].StrVal, true);
    if (NumElements != -1 && NumElements != (EndStr-yyvsp[0].StrVal))
      ThrowException("Can't build string constant of size " + 
                     itostr((int)(EndStr-yyvsp[0].StrVal)) +
                     " when array has size " + itostr(NumElements) + "!");
    std::vector<Constant*> Vals;
    if (ETy == Type::SByteTy) {
      for (char *C = yyvsp[0].StrVal; C != EndStr; ++C)
        Vals.push_back(ConstantSInt::get(ETy, *C));
    } else if (ETy == Type::UByteTy) {
      for (char *C = yyvsp[0].StrVal; C != EndStr; ++C)
        Vals.push_back(ConstantUInt::get(ETy, (unsigned char)*C));
    } else {
      free(yyvsp[0].StrVal);
      ThrowException("Cannot build string arrays of non byte sized elements!");
    }
    free(yyvsp[0].StrVal);
    yyval.ConstVal = ConstantArray::get(ATy, Vals);
    delete yyvsp[-2].TypeVal;
  ;}
    break;

  case 94:
#line 1271 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { // Nonempty unsized arr
    const PackedType *PTy = dyn_cast<PackedType>(yyvsp[-3].TypeVal->get());
    if (PTy == 0)
      ThrowException("Cannot make packed constant with type: '" + 
                     (*yyvsp[-3].TypeVal)->getDescription() + "'!");
    const Type *ETy = PTy->getElementType();
    int NumElements = PTy->getNumElements();

    // Verify that we have the correct size...
    if (NumElements != -1 && NumElements != (int)yyvsp[-1].ConstVector->size())
      ThrowException("Type mismatch: constant sized packed initialized with " +
                     utostr(yyvsp[-1].ConstVector->size()) +  " arguments, but has size of " + 
                     itostr(NumElements) + "!");

    // Verify all elements are correct type!
    for (unsigned i = 0; i < yyvsp[-1].ConstVector->size(); i++) {
      if (ETy != (*yyvsp[-1].ConstVector)[i]->getType())
        ThrowException("Element #" + utostr(i) + " is not of type '" + 
           ETy->getDescription() +"' as required!\nIt is of type '"+
           (*yyvsp[-1].ConstVector)[i]->getType()->getDescription() + "'.");
    }

    yyval.ConstVal = ConstantPacked::get(PTy, *yyvsp[-1].ConstVector);
    delete yyvsp[-3].TypeVal; delete yyvsp[-1].ConstVector;
  ;}
    break;

  case 95:
#line 1296 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const StructType *STy = dyn_cast<StructType>(yyvsp[-3].TypeVal->get());
    if (STy == 0)
      ThrowException("Cannot make struct constant with type: '" + 
                     (*yyvsp[-3].TypeVal)->getDescription() + "'!");

    if (yyvsp[-1].ConstVector->size() != STy->getNumContainedTypes())
      ThrowException("Illegal number of initializers for structure type!");

    // Check to ensure that constants are compatible with the type initializer!
    for (unsigned i = 0, e = yyvsp[-1].ConstVector->size(); i != e; ++i)
      if ((*yyvsp[-1].ConstVector)[i]->getType() != STy->getElementType(i))
        ThrowException("Expected type '" +
                       STy->getElementType(i)->getDescription() +
                       "' for element #" + utostr(i) +
                       " of structure initializer!");

    yyval.ConstVal = ConstantStruct::get(STy, *yyvsp[-1].ConstVector);
    delete yyvsp[-3].TypeVal; delete yyvsp[-1].ConstVector;
  ;}
    break;

  case 96:
#line 1316 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const StructType *STy = dyn_cast<StructType>(yyvsp[-2].TypeVal->get());
    if (STy == 0)
      ThrowException("Cannot make struct constant with type: '" + 
                     (*yyvsp[-2].TypeVal)->getDescription() + "'!");

    if (STy->getNumContainedTypes() != 0)
      ThrowException("Illegal number of initializers for structure type!");

    yyval.ConstVal = ConstantStruct::get(STy, std::vector<Constant*>());
    delete yyvsp[-2].TypeVal;
  ;}
    break;

  case 97:
#line 1328 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const PointerType *PTy = dyn_cast<PointerType>(yyvsp[-1].TypeVal->get());
    if (PTy == 0)
      ThrowException("Cannot make null pointer constant with type: '" + 
                     (*yyvsp[-1].TypeVal)->getDescription() + "'!");

    yyval.ConstVal = ConstantPointerNull::get(PTy);
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 98:
#line 1337 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ConstVal = UndefValue::get(yyvsp[-1].TypeVal->get());
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 99:
#line 1341 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const PointerType *Ty = dyn_cast<PointerType>(yyvsp[-1].TypeVal->get());
    if (Ty == 0)
      ThrowException("Global const reference must be a pointer type!");

    // ConstExprs can exist in the body of a function, thus creating
    // GlobalValues whenever they refer to a variable.  Because we are in
    // the context of a function, getValNonImprovising will search the functions
    // symbol table instead of the module symbol table for the global symbol,
    // which throws things all off.  To get around this, we just tell
    // getValNonImprovising that we are at global scope here.
    //
    Function *SavedCurFn = CurFun.CurrentFunction;
    CurFun.CurrentFunction = 0;

    Value *V = getValNonImprovising(Ty, yyvsp[0].ValIDVal);

    CurFun.CurrentFunction = SavedCurFn;

    // If this is an initializer for a constant pointer, which is referencing a
    // (currently) undefined variable, create a stub now that shall be replaced
    // in the future with the right type of variable.
    //
    if (V == 0) {
      assert(isa<PointerType>(Ty) && "Globals may only be used as pointers!");
      const PointerType *PT = cast<PointerType>(Ty);

      // First check to see if the forward references value is already created!
      PerModuleInfo::GlobalRefsType::iterator I =
        CurModule.GlobalRefs.find(std::make_pair(PT, yyvsp[0].ValIDVal));
    
      if (I != CurModule.GlobalRefs.end()) {
        V = I->second;             // Placeholder already exists, use it...
        yyvsp[0].ValIDVal.destroy();
      } else {
        std::string Name;
        if (yyvsp[0].ValIDVal.Type == ValID::NameVal) Name = yyvsp[0].ValIDVal.Name;

        // Create the forward referenced global.
        GlobalValue *GV;
        if (const FunctionType *FTy = 
                 dyn_cast<FunctionType>(PT->getElementType())) {
          GV = new Function(FTy, GlobalValue::ExternalLinkage, Name,
                            CurModule.CurrentModule);
        } else {
          GV = new GlobalVariable(PT->getElementType(), false,
                                  GlobalValue::ExternalLinkage, 0,
                                  Name, CurModule.CurrentModule);
        }

        // Keep track of the fact that we have a forward ref to recycle it
        CurModule.GlobalRefs.insert(std::make_pair(std::make_pair(PT, yyvsp[0].ValIDVal), GV));
        V = GV;
      }
    }

    yyval.ConstVal = cast<GlobalValue>(V);
    delete yyvsp[-1].TypeVal;            // Free the type handle
  ;}
    break;

  case 100:
#line 1400 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[-1].TypeVal->get() != yyvsp[0].ConstVal->getType())
      ThrowException("Mismatched types for constant expression!");
    yyval.ConstVal = yyvsp[0].ConstVal;
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 101:
#line 1406 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const Type *Ty = yyvsp[-1].TypeVal->get();
    if (isa<FunctionType>(Ty) || Ty == Type::LabelTy || isa<OpaqueType>(Ty))
      ThrowException("Cannot create a null initialized value of this type!");
    yyval.ConstVal = Constant::getNullValue(Ty);
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 102:
#line 1414 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {      // integral constants
    if (!ConstantSInt::isValueValidForType(yyvsp[-1].PrimType, yyvsp[0].SInt64Val))
      ThrowException("Constant value doesn't fit in type!");
    yyval.ConstVal = ConstantSInt::get(yyvsp[-1].PrimType, yyvsp[0].SInt64Val);
  ;}
    break;

  case 103:
#line 1419 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {            // integral constants
    if (!ConstantUInt::isValueValidForType(yyvsp[-1].PrimType, yyvsp[0].UInt64Val))
      ThrowException("Constant value doesn't fit in type!");
    yyval.ConstVal = ConstantUInt::get(yyvsp[-1].PrimType, yyvsp[0].UInt64Val);
  ;}
    break;

  case 104:
#line 1424 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                      // Boolean constants
    yyval.ConstVal = ConstantBool::True;
  ;}
    break;

  case 105:
#line 1427 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                     // Boolean constants
    yyval.ConstVal = ConstantBool::False;
  ;}
    break;

  case 106:
#line 1430 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                   // Float & Double constants
    if (!ConstantFP::isValueValidForType(yyvsp[-1].PrimType, yyvsp[0].FPVal))
      ThrowException("Floating point constant invalid for type!!");
    yyval.ConstVal = ConstantFP::get(yyvsp[-1].PrimType, yyvsp[0].FPVal);
  ;}
    break;

  case 107:
#line 1437 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!yyvsp[-3].ConstVal->getType()->isFirstClassType())
      ThrowException("cast constant expression from a non-primitive type: '" +
                     yyvsp[-3].ConstVal->getType()->getDescription() + "'!");
    if (!yyvsp[-1].TypeVal->get()->isFirstClassType())
      ThrowException("cast constant expression to a non-primitive type: '" +
                     yyvsp[-1].TypeVal->get()->getDescription() + "'!");
    yyval.ConstVal = ConstantExpr::getCast(yyvsp[-3].ConstVal, yyvsp[-1].TypeVal->get());
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 108:
#line 1447 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!isa<PointerType>(yyvsp[-2].ConstVal->getType()))
      ThrowException("GetElementPtr requires a pointer operand!");

    // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte struct
    // indices to uint struct indices for compatibility.
    generic_gep_type_iterator<std::vector<Value*>::iterator>
      GTI = gep_type_begin(yyvsp[-2].ConstVal->getType(), yyvsp[-1].ValueList->begin(), yyvsp[-1].ValueList->end()),
      GTE = gep_type_end(yyvsp[-2].ConstVal->getType(), yyvsp[-1].ValueList->begin(), yyvsp[-1].ValueList->end());
    for (unsigned i = 0, e = yyvsp[-1].ValueList->size(); i != e && GTI != GTE; ++i, ++GTI)
      if (isa<StructType>(*GTI))        // Only change struct indices
        if (ConstantUInt *CUI = dyn_cast<ConstantUInt>((*yyvsp[-1].ValueList)[i]))
          if (CUI->getType() == Type::UByteTy)
            (*yyvsp[-1].ValueList)[i] = ConstantExpr::getCast(CUI, Type::UIntTy);

    const Type *IdxTy =
      GetElementPtrInst::getIndexedType(yyvsp[-2].ConstVal->getType(), *yyvsp[-1].ValueList, true);
    if (!IdxTy)
      ThrowException("Index list invalid for constant getelementptr!");

    std::vector<Constant*> IdxVec;
    for (unsigned i = 0, e = yyvsp[-1].ValueList->size(); i != e; ++i)
      if (Constant *C = dyn_cast<Constant>((*yyvsp[-1].ValueList)[i]))
        IdxVec.push_back(C);
      else
        ThrowException("Indices to constant getelementptr must be constants!");

    delete yyvsp[-1].ValueList;

    yyval.ConstVal = ConstantExpr::getGetElementPtr(yyvsp[-2].ConstVal, IdxVec);
  ;}
    break;

  case 109:
#line 1478 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[-5].ConstVal->getType() != Type::BoolTy)
      ThrowException("Select condition must be of boolean type!");
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("Select operand types must match!");
    yyval.ConstVal = ConstantExpr::getSelect(yyvsp[-5].ConstVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;}
    break;

  case 110:
#line 1485 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("Binary operator types must match!");
    // HACK: llvm 1.3 and earlier used to emit invalid pointer constant exprs.
    // To retain backward compatibility with these early compilers, we emit a
    // cast to the appropriate integer type automatically if we are in the
    // broken case.  See PR424 for more information.
    if (!isa<PointerType>(yyvsp[-3].ConstVal->getType())) {
      yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
    } else {
      const Type *IntPtrTy = 0;
      switch (CurModule.CurrentModule->getPointerSize()) {
      case Module::Pointer32: IntPtrTy = Type::IntTy; break;
      case Module::Pointer64: IntPtrTy = Type::LongTy; break;
      default: ThrowException("invalid pointer binary constant expr!");
      }
      yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, ConstantExpr::getCast(yyvsp[-3].ConstVal, IntPtrTy),
                             ConstantExpr::getCast(yyvsp[-1].ConstVal, IntPtrTy));
      yyval.ConstVal = ConstantExpr::getCast(yyval.ConstVal, yyvsp[-3].ConstVal->getType());
    }
  ;}
    break;

  case 111:
#line 1506 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("Logical operator types must match!");
    if (!yyvsp[-3].ConstVal->getType()->isIntegral()) {
      if (!isa<PackedType>(yyvsp[-3].ConstVal->getType()) || 
          !cast<PackedType>(yyvsp[-3].ConstVal->getType())->getElementType()->isIntegral())
        ThrowException("Logical operator requires integral operands!");
    }
    yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;}
    break;

  case 112:
#line 1516 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[-3].ConstVal->getType() != yyvsp[-1].ConstVal->getType())
      ThrowException("setcc operand types must match!");
    yyval.ConstVal = ConstantExpr::get(yyvsp[-5].BinaryOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;}
    break;

  case 113:
#line 1521 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[-1].ConstVal->getType() != Type::UByteTy)
      ThrowException("Shift count for shift constant must be unsigned byte!");
    if (!yyvsp[-3].ConstVal->getType()->isInteger())
      ThrowException("Shift constant expression requires integer operand!");
    yyval.ConstVal = ConstantExpr::get(yyvsp[-5].OtherOpVal, yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;}
    break;

  case 114:
#line 1528 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
        if (!isa<PackedType>(yyvsp[-3].ConstVal->getType()))
      ThrowException("First operand of extractelement must be "
                     "packed type!");
    if (yyvsp[-1].ConstVal->getType() != Type::UIntTy)
      ThrowException("Second operand of extractelement must be uint!");
    yyval.ConstVal = ConstantExpr::getExtractElement(yyvsp[-3].ConstVal, yyvsp[-1].ConstVal);
  ;}
    break;

  case 115:
#line 1538 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    (yyval.ConstVector = yyvsp[-2].ConstVector)->push_back(yyvsp[0].ConstVal);
  ;}
    break;

  case 116:
#line 1541 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ConstVector = new std::vector<Constant*>();
    yyval.ConstVector->push_back(yyvsp[0].ConstVal);
  ;}
    break;

  case 117:
#line 1548 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.BoolVal = false; ;}
    break;

  case 118:
#line 1548 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.BoolVal = true; ;}
    break;

  case 119:
#line 1558 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  yyval.ModuleVal = ParserResult = yyvsp[0].ModuleVal;
  CurModule.ModuleDone();
;}
    break;

  case 120:
#line 1565 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ModuleVal = yyvsp[-1].ModuleVal;
    CurFun.FunctionDone();
  ;}
    break;

  case 121:
#line 1569 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ModuleVal = yyvsp[-1].ModuleVal;
  ;}
    break;

  case 122:
#line 1572 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ModuleVal = yyvsp[-1].ModuleVal;
  ;}
    break;

  case 123:
#line 1575 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ModuleVal = CurModule.CurrentModule;
    // Emit an error if there are any unresolved types left.
    if (!CurModule.LateResolveTypes.empty()) {
      const ValID &DID = CurModule.LateResolveTypes.begin()->first;
      if (DID.Type == ValID::NameVal)
        ThrowException("Reference to an undefined type: '"+DID.getName() + "'");
      else
        ThrowException("Reference to an undefined type: #" + itostr(DID.Num));
    }
  ;}
    break;

  case 124:
#line 1588 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    // Eagerly resolve types.  This is not an optimization, this is a
    // requirement that is due to the fact that we could have this:
    //
    // %list = type { %list * }
    // %list = type { %list * }    ; repeated type decl
    //
    // If types are not resolved eagerly, then the two types will not be
    // determined to be the same type!
    //
    ResolveTypeTo(yyvsp[-2].StrVal, *yyvsp[0].TypeVal);

    if (!setTypeName(*yyvsp[0].TypeVal, yyvsp[-2].StrVal) && !yyvsp[-2].StrVal) {
      // If this is a named type that is not a redefinition, add it to the slot
      // table.
      CurModule.Types.push_back(*yyvsp[0].TypeVal);
    }

    delete yyvsp[0].TypeVal;
  ;}
    break;

  case 125:
#line 1608 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {       // Function prototypes can be in const pool
  ;}
    break;

  case 126:
#line 1610 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[0].ConstVal == 0) ThrowException("Global value initializer is not a constant!");
    CurGV = ParseGlobalVariable(yyvsp[-3].StrVal, yyvsp[-2].Linkage, yyvsp[-1].BoolVal, yyvsp[0].ConstVal->getType(), yyvsp[0].ConstVal);
                                                       ;}
    break;

  case 127:
#line 1613 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    CurGV = 0;
  ;}
    break;

  case 128:
#line 1616 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    CurGV = ParseGlobalVariable(yyvsp[-3].StrVal, GlobalValue::ExternalLinkage,
                                             yyvsp[-1].BoolVal, *yyvsp[0].TypeVal, 0);
    delete yyvsp[0].TypeVal;
                                                   ;}
    break;

  case 129:
#line 1620 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    CurGV = 0;
  ;}
    break;

  case 130:
#line 1623 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { 
  ;}
    break;

  case 131:
#line 1625 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  ;}
    break;

  case 132:
#line 1627 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { 
  ;}
    break;

  case 133:
#line 1632 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.Endianness = Module::BigEndian; ;}
    break;

  case 134:
#line 1633 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.Endianness = Module::LittleEndian; ;}
    break;

  case 135:
#line 1635 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    CurModule.CurrentModule->setEndianness(yyvsp[0].Endianness);
  ;}
    break;

  case 136:
#line 1638 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[0].UInt64Val == 32)
      CurModule.CurrentModule->setPointerSize(Module::Pointer32);
    else if (yyvsp[0].UInt64Val == 64)
      CurModule.CurrentModule->setPointerSize(Module::Pointer64);
    else
      ThrowException("Invalid pointer size: '" + utostr(yyvsp[0].UInt64Val) + "'!");
  ;}
    break;

  case 137:
#line 1646 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    CurModule.CurrentModule->setTargetTriple(yyvsp[0].StrVal);
    free(yyvsp[0].StrVal);
  ;}
    break;

  case 139:
#line 1653 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
          CurModule.CurrentModule->addLibrary(yyvsp[0].StrVal);
          free(yyvsp[0].StrVal);
        ;}
    break;

  case 140:
#line 1657 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
          CurModule.CurrentModule->addLibrary(yyvsp[0].StrVal);
          free(yyvsp[0].StrVal);
        ;}
    break;

  case 141:
#line 1661 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
        ;}
    break;

  case 145:
#line 1670 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.StrVal = 0; ;}
    break;

  case 146:
#line 1672 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  if (*yyvsp[-1].TypeVal == Type::VoidTy)
    ThrowException("void typed arguments are invalid!");
  yyval.ArgVal = new std::pair<PATypeHolder*, char*>(yyvsp[-1].TypeVal, yyvsp[0].StrVal);
;}
    break;

  case 147:
#line 1678 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ArgList = yyvsp[-2].ArgList;
    yyvsp[-2].ArgList->push_back(*yyvsp[0].ArgVal);
    delete yyvsp[0].ArgVal;
  ;}
    break;

  case 148:
#line 1683 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ArgList = new std::vector<std::pair<PATypeHolder*,char*> >();
    yyval.ArgList->push_back(*yyvsp[0].ArgVal);
    delete yyvsp[0].ArgVal;
  ;}
    break;

  case 149:
#line 1689 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ArgList = yyvsp[0].ArgList;
  ;}
    break;

  case 150:
#line 1692 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ArgList = yyvsp[-2].ArgList;
    yyval.ArgList->push_back(std::pair<PATypeHolder*,
                            char*>(new PATypeHolder(Type::VoidTy), 0));
  ;}
    break;

  case 151:
#line 1697 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ArgList = new std::vector<std::pair<PATypeHolder*,char*> >();
    yyval.ArgList->push_back(std::make_pair(new PATypeHolder(Type::VoidTy), (char*)0));
  ;}
    break;

  case 152:
#line 1701 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ArgList = 0;
  ;}
    break;

  case 153:
#line 1706 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  UnEscapeLexed(yyvsp[-5].StrVal);
  std::string FunctionName(yyvsp[-5].StrVal);
  free(yyvsp[-5].StrVal);  // Free strdup'd memory!
  
  if (!(*yyvsp[-6].TypeVal)->isFirstClassType() && *yyvsp[-6].TypeVal != Type::VoidTy)
    ThrowException("LLVM functions cannot return aggregate types!");

  std::vector<const Type*> ParamTypeList;
  if (yyvsp[-3].ArgList) {   // If there are arguments...
    for (std::vector<std::pair<PATypeHolder*,char*> >::iterator I = yyvsp[-3].ArgList->begin();
         I != yyvsp[-3].ArgList->end(); ++I)
      ParamTypeList.push_back(I->first->get());
  }

  bool isVarArg = ParamTypeList.size() && ParamTypeList.back() == Type::VoidTy;
  if (isVarArg) ParamTypeList.pop_back();

  const FunctionType *FT = FunctionType::get(*yyvsp[-6].TypeVal, ParamTypeList, isVarArg);
  const PointerType *PFT = PointerType::get(FT);
  delete yyvsp[-6].TypeVal;

  ValID ID;
  if (!FunctionName.empty()) {
    ID = ValID::create((char*)FunctionName.c_str());
  } else {
    ID = ValID::create((int)CurModule.Values[PFT].size());
  }

  Function *Fn = 0;
  // See if this function was forward referenced.  If so, recycle the object.
  if (GlobalValue *FWRef = CurModule.GetForwardRefForGlobal(PFT, ID)) {
    // Move the function to the end of the list, from whereever it was 
    // previously inserted.
    Fn = cast<Function>(FWRef);
    CurModule.CurrentModule->getFunctionList().remove(Fn);
    CurModule.CurrentModule->getFunctionList().push_back(Fn);
  } else if (!FunctionName.empty() &&     // Merge with an earlier prototype?
             (Fn = CurModule.CurrentModule->getFunction(FunctionName, FT))) {
    // If this is the case, either we need to be a forward decl, or it needs 
    // to be.
    if (!CurFun.isDeclare && !Fn->isExternal())
      ThrowException("Redefinition of function '" + FunctionName + "'!");
    
    // Make sure to strip off any argument names so we can't get conflicts.
    if (Fn->isExternal())
      for (Function::arg_iterator AI = Fn->arg_begin(), AE = Fn->arg_end();
           AI != AE; ++AI)
        AI->setName("");

  } else  {  // Not already defined?
    Fn = new Function(FT, GlobalValue::ExternalLinkage, FunctionName,
                      CurModule.CurrentModule);
    InsertValue(Fn, CurModule.Values);
  }

  CurFun.FunctionStart(Fn);
  Fn->setCallingConv(yyvsp[-7].UIntVal);
  Fn->setAlignment(yyvsp[0].UIntVal);
  if (yyvsp[-1].StrVal) {
    Fn->setSection(yyvsp[-1].StrVal);
    free(yyvsp[-1].StrVal);
  }

  // Add all of the arguments we parsed to the function...
  if (yyvsp[-3].ArgList) {                     // Is null if empty...
    if (isVarArg) {  // Nuke the last entry
      assert(yyvsp[-3].ArgList->back().first->get() == Type::VoidTy && yyvsp[-3].ArgList->back().second == 0&&
             "Not a varargs marker!");
      delete yyvsp[-3].ArgList->back().first;
      yyvsp[-3].ArgList->pop_back();  // Delete the last entry
    }
    Function::arg_iterator ArgIt = Fn->arg_begin();
    for (std::vector<std::pair<PATypeHolder*,char*> >::iterator I = yyvsp[-3].ArgList->begin();
         I != yyvsp[-3].ArgList->end(); ++I, ++ArgIt) {
      delete I->first;                          // Delete the typeholder...

      setValueName(ArgIt, I->second);           // Insert arg into symtab...
      InsertValue(ArgIt);
    }

    delete yyvsp[-3].ArgList;                     // We're now done with the argument list
  }
;}
    break;

  case 156:
#line 1793 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  yyval.FunctionVal = CurFun.CurrentFunction;

  // Make sure that we keep track of the linkage type even if there was a
  // previous "declare".
  yyval.FunctionVal->setLinkage(yyvsp[-2].Linkage);
;}
    break;

  case 159:
#line 1803 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  yyval.FunctionVal = yyvsp[-1].FunctionVal;
;}
    break;

  case 160:
#line 1807 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { CurFun.isDeclare = true; ;}
    break;

  case 161:
#line 1807 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  yyval.FunctionVal = CurFun.CurrentFunction;
  CurFun.FunctionDone();
;}
    break;

  case 162:
#line 1816 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {    // A reference to a direct constant
    yyval.ValIDVal = ValID::create(yyvsp[0].SInt64Val);
  ;}
    break;

  case 163:
#line 1819 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValIDVal = ValID::create(yyvsp[0].UInt64Val);
  ;}
    break;

  case 164:
#line 1822 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                     // Perhaps it's an FP constant?
    yyval.ValIDVal = ValID::create(yyvsp[0].FPVal);
  ;}
    break;

  case 165:
#line 1825 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValIDVal = ValID::create(ConstantBool::True);
  ;}
    break;

  case 166:
#line 1828 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValIDVal = ValID::create(ConstantBool::False);
  ;}
    break;

  case 167:
#line 1831 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValIDVal = ValID::createNull();
  ;}
    break;

  case 168:
#line 1834 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValIDVal = ValID::createUndef();
  ;}
    break;

  case 169:
#line 1837 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {     // A vector zero constant.
    yyval.ValIDVal = ValID::createZeroInit();
  ;}
    break;

  case 170:
#line 1840 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { // Nonempty unsized packed vector
    const Type *ETy = (*yyvsp[-1].ConstVector)[0]->getType();
    int NumElements = yyvsp[-1].ConstVector->size(); 
    
    PackedType* pt = PackedType::get(ETy, NumElements);
    PATypeHolder* PTy = new PATypeHolder(
                                         HandleUpRefs(
                                            PackedType::get(
                                                ETy, 
                                                NumElements)
                                            )
                                         );
    
    // Verify all elements are correct type!
    for (unsigned i = 0; i < yyvsp[-1].ConstVector->size(); i++) {
      if (ETy != (*yyvsp[-1].ConstVector)[i]->getType())
        ThrowException("Element #" + utostr(i) + " is not of type '" + 
                     ETy->getDescription() +"' as required!\nIt is of type '" +
                     (*yyvsp[-1].ConstVector)[i]->getType()->getDescription() + "'.");
    }

    yyval.ValIDVal = ValID::create(ConstantPacked::get(pt, *yyvsp[-1].ConstVector));
    delete PTy; delete yyvsp[-1].ConstVector;
  ;}
    break;

  case 171:
#line 1864 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValIDVal = ValID::create(yyvsp[0].ConstVal);
  ;}
    break;

  case 172:
#line 1871 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {  // Is it an integer reference...?
    yyval.ValIDVal = ValID::create(yyvsp[0].SIntVal);
  ;}
    break;

  case 173:
#line 1874 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                   // Is it a named reference...?
    yyval.ValIDVal = ValID::create(yyvsp[0].StrVal);
  ;}
    break;

  case 176:
#line 1885 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValueVal = getVal(*yyvsp[-1].TypeVal, yyvsp[0].ValIDVal); delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 177:
#line 1889 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.FunctionVal = yyvsp[-1].FunctionVal;
  ;}
    break;

  case 178:
#line 1892 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { // Do not allow functions with 0 basic blocks   
    yyval.FunctionVal = yyvsp[-1].FunctionVal;
  ;}
    break;

  case 179:
#line 1900 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    setValueName(yyvsp[0].TermInstVal, yyvsp[-1].StrVal);
    InsertValue(yyvsp[0].TermInstVal);

    yyvsp[-2].BasicBlockVal->getInstList().push_back(yyvsp[0].TermInstVal);
    InsertValue(yyvsp[-2].BasicBlockVal);
    yyval.BasicBlockVal = yyvsp[-2].BasicBlockVal;
  ;}
    break;

  case 180:
#line 1909 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyvsp[-1].BasicBlockVal->getInstList().push_back(yyvsp[0].InstVal);
    yyval.BasicBlockVal = yyvsp[-1].BasicBlockVal;
  ;}
    break;

  case 181:
#line 1913 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.BasicBlockVal = CurBB = getBBVal(ValID::create((int)CurFun.NextBBNum++), true);

    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, yyval.BasicBlockVal);
  ;}
    break;

  case 182:
#line 1923 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.BasicBlockVal = CurBB = getBBVal(ValID::create(yyvsp[0].StrVal), true);

    // Make sure to move the basic block to the correct location in the
    // function, instead of leaving it inserted wherever it was first
    // referenced.
    Function::BasicBlockListType &BBL = 
      CurFun.CurrentFunction->getBasicBlockList();
    BBL.splice(BBL.end(), BBL, yyval.BasicBlockVal);
  ;}
    break;

  case 183:
#line 1934 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {              // Return with a result...
    yyval.TermInstVal = new ReturnInst(yyvsp[0].ValueVal);
  ;}
    break;

  case 184:
#line 1937 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                                       // Return with no result...
    yyval.TermInstVal = new ReturnInst();
  ;}
    break;

  case 185:
#line 1940 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {                         // Unconditional Branch...
    yyval.TermInstVal = new BranchInst(getBBVal(yyvsp[0].ValIDVal));
  ;}
    break;

  case 186:
#line 1943 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {  
    yyval.TermInstVal = new BranchInst(getBBVal(yyvsp[-3].ValIDVal), getBBVal(yyvsp[0].ValIDVal), getVal(Type::BoolTy, yyvsp[-6].ValIDVal));
  ;}
    break;

  case 187:
#line 1946 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    SwitchInst *S = new SwitchInst(getVal(yyvsp[-7].PrimType, yyvsp[-6].ValIDVal), getBBVal(yyvsp[-3].ValIDVal), yyvsp[-1].JumpTable->size());
    yyval.TermInstVal = S;

    std::vector<std::pair<Constant*,BasicBlock*> >::iterator I = yyvsp[-1].JumpTable->begin(),
      E = yyvsp[-1].JumpTable->end();
    for (; I != E; ++I) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(I->first))
          S->addCase(CI, I->second);
      else
        ThrowException("Switch case is constant, but not a simple integer!");
    }
    delete yyvsp[-1].JumpTable;
  ;}
    break;

  case 188:
#line 1960 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    SwitchInst *S = new SwitchInst(getVal(yyvsp[-6].PrimType, yyvsp[-5].ValIDVal), getBBVal(yyvsp[-2].ValIDVal), 0);
    yyval.TermInstVal = S;
  ;}
    break;

  case 189:
#line 1965 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>(yyvsp[-10].TypeVal->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if (yyvsp[-7].ValueList) {
        for (std::vector<Value*>::iterator I = yyvsp[-7].ValueList->begin(), E = yyvsp[-7].ValueList->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      Ty = FunctionType::get(yyvsp[-10].TypeVal->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }

    Value *V = getVal(PFTy, yyvsp[-9].ValIDVal);   // Get the function we're calling...

    BasicBlock *Normal = getBBVal(yyvsp[-3].ValIDVal);
    BasicBlock *Except = getBBVal(yyvsp[0].ValIDVal);

    // Create the call node...
    if (!yyvsp[-7].ValueList) {                                   // Has no arguments?
      yyval.TermInstVal = new InvokeInst(V, Normal, Except, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      std::vector<Value*>::iterator ArgI = yyvsp[-7].ValueList->begin(), ArgE = yyvsp[-7].ValueList->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
        if ((*ArgI)->getType() != *I)
          ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
        ThrowException("Invalid number of parameters detected!");

      yyval.TermInstVal = new InvokeInst(V, Normal, Except, *yyvsp[-7].ValueList);
    }
    cast<InvokeInst>(yyval.TermInstVal)->setCallingConv(yyvsp[-11].UIntVal);
  
    delete yyvsp[-10].TypeVal;
    delete yyvsp[-7].ValueList;
  ;}
    break;

  case 190:
#line 2017 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.TermInstVal = new UnwindInst();
  ;}
    break;

  case 191:
#line 2020 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.TermInstVal = new UnreachableInst();
  ;}
    break;

  case 192:
#line 2026 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.JumpTable = yyvsp[-5].JumpTable;
    Constant *V = cast<Constant>(getValNonImprovising(yyvsp[-4].PrimType, yyvsp[-3].ValIDVal));
    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    yyval.JumpTable->push_back(std::make_pair(V, getBBVal(yyvsp[0].ValIDVal)));
  ;}
    break;

  case 193:
#line 2034 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.JumpTable = new std::vector<std::pair<Constant*, BasicBlock*> >();
    Constant *V = cast<Constant>(getValNonImprovising(yyvsp[-4].PrimType, yyvsp[-3].ValIDVal));

    if (V == 0)
      ThrowException("May only switch on a constant pool value!");

    yyval.JumpTable->push_back(std::make_pair(V, getBBVal(yyvsp[0].ValIDVal)));
  ;}
    break;

  case 194:
#line 2044 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
  // Is this definition named?? if so, assign the name...
  setValueName(yyvsp[0].InstVal, yyvsp[-1].StrVal);
  InsertValue(yyvsp[0].InstVal);
  yyval.InstVal = yyvsp[0].InstVal;
;}
    break;

  case 195:
#line 2051 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {    // Used for PHI nodes
    yyval.PHIList = new std::list<std::pair<Value*, BasicBlock*> >();
    yyval.PHIList->push_back(std::make_pair(getVal(*yyvsp[-5].TypeVal, yyvsp[-3].ValIDVal), getBBVal(yyvsp[-1].ValIDVal)));
    delete yyvsp[-5].TypeVal;
  ;}
    break;

  case 196:
#line 2056 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.PHIList = yyvsp[-6].PHIList;
    yyvsp[-6].PHIList->push_back(std::make_pair(getVal(yyvsp[-6].PHIList->front().first->getType(), yyvsp[-3].ValIDVal),
                                 getBBVal(yyvsp[-1].ValIDVal)));
  ;}
    break;

  case 197:
#line 2063 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {    // Used for call statements, and memory insts...
    yyval.ValueList = new std::vector<Value*>();
    yyval.ValueList->push_back(yyvsp[0].ValueVal);
  ;}
    break;

  case 198:
#line 2067 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.ValueList = yyvsp[-2].ValueList;
    yyvsp[-2].ValueList->push_back(yyvsp[0].ValueVal);
  ;}
    break;

  case 200:
#line 2073 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { yyval.ValueList = 0; ;}
    break;

  case 201:
#line 2075 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.BoolVal = true;
  ;}
    break;

  case 202:
#line 2078 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.BoolVal = false;
  ;}
    break;

  case 203:
#line 2084 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!(*yyvsp[-3].TypeVal)->isInteger() && !(*yyvsp[-3].TypeVal)->isFloatingPoint() && 
        !isa<PackedType>((*yyvsp[-3].TypeVal).get()))
      ThrowException(
        "Arithmetic operator requires integer, FP, or packed operands!");
    if (isa<PackedType>((*yyvsp[-3].TypeVal).get()) && yyvsp[-4].BinaryOpVal == Instruction::Rem)
      ThrowException("Rem not supported on packed types!");
    yyval.InstVal = BinaryOperator::create(yyvsp[-4].BinaryOpVal, getVal(*yyvsp[-3].TypeVal, yyvsp[-2].ValIDVal), getVal(*yyvsp[-3].TypeVal, yyvsp[0].ValIDVal));
    if (yyval.InstVal == 0)
      ThrowException("binary operator returned null!");
    delete yyvsp[-3].TypeVal;
  ;}
    break;

  case 204:
#line 2096 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!(*yyvsp[-3].TypeVal)->isIntegral()) {
      if (!isa<PackedType>(yyvsp[-3].TypeVal->get()) ||
          !cast<PackedType>(yyvsp[-3].TypeVal->get())->getElementType()->isIntegral())
        ThrowException("Logical operator requires integral operands!");
    }
    yyval.InstVal = BinaryOperator::create(yyvsp[-4].BinaryOpVal, getVal(*yyvsp[-3].TypeVal, yyvsp[-2].ValIDVal), getVal(*yyvsp[-3].TypeVal, yyvsp[0].ValIDVal));
    if (yyval.InstVal == 0)
      ThrowException("binary operator returned null!");
    delete yyvsp[-3].TypeVal;
  ;}
    break;

  case 205:
#line 2107 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if(isa<PackedType>((*yyvsp[-3].TypeVal).get())) {
      ThrowException(
        "PackedTypes currently not supported in setcc instructions!");
    }
    yyval.InstVal = new SetCondInst(yyvsp[-4].BinaryOpVal, getVal(*yyvsp[-3].TypeVal, yyvsp[-2].ValIDVal), getVal(*yyvsp[-3].TypeVal, yyvsp[0].ValIDVal));
    if (yyval.InstVal == 0)
      ThrowException("binary operator returned null!");
    delete yyvsp[-3].TypeVal;
  ;}
    break;

  case 206:
#line 2117 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    std::cerr << "WARNING: Use of eliminated 'not' instruction:"
              << " Replacing with 'xor'.\n";

    Value *Ones = ConstantIntegral::getAllOnesValue(yyvsp[0].ValueVal->getType());
    if (Ones == 0)
      ThrowException("Expected integral type for not instruction!");

    yyval.InstVal = BinaryOperator::create(Instruction::Xor, yyvsp[0].ValueVal, Ones);
    if (yyval.InstVal == 0)
      ThrowException("Could not create a xor instruction!");
  ;}
    break;

  case 207:
#line 2129 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[0].ValueVal->getType() != Type::UByteTy)
      ThrowException("Shift amount must be ubyte!");
    if (!yyvsp[-2].ValueVal->getType()->isInteger())
      ThrowException("Shift constant expression requires integer operand!");
    yyval.InstVal = new ShiftInst(yyvsp[-3].OtherOpVal, yyvsp[-2].ValueVal, yyvsp[0].ValueVal);
  ;}
    break;

  case 208:
#line 2136 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!yyvsp[0].TypeVal->get()->isFirstClassType())
      ThrowException("cast instruction to a non-primitive type: '" +
                     yyvsp[0].TypeVal->get()->getDescription() + "'!");
    yyval.InstVal = new CastInst(yyvsp[-2].ValueVal, *yyvsp[0].TypeVal);
    delete yyvsp[0].TypeVal;
  ;}
    break;

  case 209:
#line 2143 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (yyvsp[-4].ValueVal->getType() != Type::BoolTy)
      ThrowException("select condition must be boolean!");
    if (yyvsp[-2].ValueVal->getType() != yyvsp[0].ValueVal->getType())
      ThrowException("select value types should match!");
    yyval.InstVal = new SelectInst(yyvsp[-4].ValueVal, yyvsp[-2].ValueVal, yyvsp[0].ValueVal);
  ;}
    break;

  case 210:
#line 2150 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    NewVarArgs = true;
    yyval.InstVal = new VAArgInst(yyvsp[-2].ValueVal, *yyvsp[0].TypeVal);
    delete yyvsp[0].TypeVal;
  ;}
    break;

  case 211:
#line 2155 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    ObsoleteVarArgs = true;
    const Type* ArgTy = yyvsp[-2].ValueVal->getType();
    Function* NF = CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0);

    //b = vaarg a, t -> 
    //foo = alloca 1 of t
    //bar = vacopy a 
    //store bar -> foo
    //b = vaarg foo, t
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vaarg.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, yyvsp[-2].ValueVal);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    yyval.InstVal = new VAArgInst(foo, *yyvsp[0].TypeVal);
    delete yyvsp[0].TypeVal;
  ;}
    break;

  case 212:
#line 2174 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    ObsoleteVarArgs = true;
    const Type* ArgTy = yyvsp[-2].ValueVal->getType();
    Function* NF = CurModule.CurrentModule->
      getOrInsertFunction("llvm.va_copy", ArgTy, ArgTy, (Type *)0);

    //b = vanext a, t ->
    //foo = alloca 1 of t
    //bar = vacopy a
    //store bar -> foo
    //tmp = vaarg foo, t
    //b = load foo
    AllocaInst* foo = new AllocaInst(ArgTy, 0, "vanext.fix");
    CurBB->getInstList().push_back(foo);
    CallInst* bar = new CallInst(NF, yyvsp[-2].ValueVal);
    CurBB->getInstList().push_back(bar);
    CurBB->getInstList().push_back(new StoreInst(bar, foo));
    Instruction* tmp = new VAArgInst(foo, *yyvsp[0].TypeVal);
    CurBB->getInstList().push_back(tmp);
    yyval.InstVal = new LoadInst(foo);
    delete yyvsp[0].TypeVal;
  ;}
    break;

  case 213:
#line 2196 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!isa<PackedType>(yyvsp[-2].ValueVal->getType()))
      ThrowException("First operand of extractelement must be a "
                     "packed type val!");
    if (yyvsp[0].ValueVal->getType() != Type::UIntTy)
      ThrowException("Second operand of extractelement must be a uint!");
    yyval.InstVal = new ExtractElementInst(yyvsp[-2].ValueVal, yyvsp[0].ValueVal);
  ;}
    break;

  case 214:
#line 2204 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const Type *Ty = yyvsp[0].PHIList->front().first->getType();
    if (!Ty->isFirstClassType())
      ThrowException("PHI node operands must be of first class type!");
    yyval.InstVal = new PHINode(Ty);
    ((PHINode*)yyval.InstVal)->reserveOperandSpace(yyvsp[0].PHIList->size());
    while (yyvsp[0].PHIList->begin() != yyvsp[0].PHIList->end()) {
      if (yyvsp[0].PHIList->front().first->getType() != Ty) 
        ThrowException("All elements of a PHI node must be of the same type!");
      cast<PHINode>(yyval.InstVal)->addIncoming(yyvsp[0].PHIList->front().first, yyvsp[0].PHIList->front().second);
      yyvsp[0].PHIList->pop_front();
    }
    delete yyvsp[0].PHIList;  // Free the list...
  ;}
    break;

  case 215:
#line 2218 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const PointerType *PFTy;
    const FunctionType *Ty;

    if (!(PFTy = dyn_cast<PointerType>(yyvsp[-4].TypeVal->get())) ||
        !(Ty = dyn_cast<FunctionType>(PFTy->getElementType()))) {
      // Pull out the types of all of the arguments...
      std::vector<const Type*> ParamTypes;
      if (yyvsp[-1].ValueList) {
        for (std::vector<Value*>::iterator I = yyvsp[-1].ValueList->begin(), E = yyvsp[-1].ValueList->end();
             I != E; ++I)
          ParamTypes.push_back((*I)->getType());
      }

      bool isVarArg = ParamTypes.size() && ParamTypes.back() == Type::VoidTy;
      if (isVarArg) ParamTypes.pop_back();

      if (!(*yyvsp[-4].TypeVal)->isFirstClassType() && *yyvsp[-4].TypeVal != Type::VoidTy)
        ThrowException("LLVM functions cannot return aggregate types!");

      Ty = FunctionType::get(yyvsp[-4].TypeVal->get(), ParamTypes, isVarArg);
      PFTy = PointerType::get(Ty);
    }

    Value *V = getVal(PFTy, yyvsp[-3].ValIDVal);   // Get the function we're calling...

    // Create the call node...
    if (!yyvsp[-1].ValueList) {                                   // Has no arguments?
      // Make sure no arguments is a good thing!
      if (Ty->getNumParams() != 0)
        ThrowException("No arguments passed to a function that "
                       "expects arguments!");

      yyval.InstVal = new CallInst(V, std::vector<Value*>());
    } else {                                     // Has arguments?
      // Loop through FunctionType's arguments and ensure they are specified
      // correctly!
      //
      FunctionType::param_iterator I = Ty->param_begin();
      FunctionType::param_iterator E = Ty->param_end();
      std::vector<Value*>::iterator ArgI = yyvsp[-1].ValueList->begin(), ArgE = yyvsp[-1].ValueList->end();

      for (; ArgI != ArgE && I != E; ++ArgI, ++I)
        if ((*ArgI)->getType() != *I)
          ThrowException("Parameter " +(*ArgI)->getName()+ " is not of type '" +
                         (*I)->getDescription() + "'!");

      if (I != E || (ArgI != ArgE && !Ty->isVarArg()))
        ThrowException("Invalid number of parameters detected!");

      yyval.InstVal = new CallInst(V, *yyvsp[-1].ValueList);
    }
    cast<CallInst>(yyval.InstVal)->setTailCall(yyvsp[-6].BoolVal);
    cast<CallInst>(yyval.InstVal)->setCallingConv(yyvsp[-5].UIntVal);
    delete yyvsp[-4].TypeVal;
    delete yyvsp[-1].ValueList;
  ;}
    break;

  case 216:
#line 2275 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.InstVal = yyvsp[0].InstVal;
  ;}
    break;

  case 217:
#line 2281 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { 
    yyval.ValueList = yyvsp[0].ValueList; 
  ;}
    break;

  case 218:
#line 2283 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    { 
    yyval.ValueList = new std::vector<Value*>(); 
  ;}
    break;

  case 219:
#line 2287 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.BoolVal = true;
  ;}
    break;

  case 220:
#line 2290 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.BoolVal = false;
  ;}
    break;

  case 221:
#line 2296 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.InstVal = new MallocInst(*yyvsp[-1].TypeVal, 0, yyvsp[0].UIntVal);
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 222:
#line 2300 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.InstVal = new MallocInst(*yyvsp[-4].TypeVal, getVal(yyvsp[-2].PrimType, yyvsp[-1].ValIDVal), yyvsp[0].UIntVal);
    delete yyvsp[-4].TypeVal;
  ;}
    break;

  case 223:
#line 2304 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.InstVal = new AllocaInst(*yyvsp[-1].TypeVal, 0, yyvsp[0].UIntVal);
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 224:
#line 2308 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    yyval.InstVal = new AllocaInst(*yyvsp[-4].TypeVal, getVal(yyvsp[-2].PrimType, yyvsp[-1].ValIDVal), yyvsp[0].UIntVal);
    delete yyvsp[-4].TypeVal;
  ;}
    break;

  case 225:
#line 2312 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!isa<PointerType>(yyvsp[0].ValueVal->getType()))
      ThrowException("Trying to free nonpointer type " + 
                     yyvsp[0].ValueVal->getType()->getDescription() + "!");
    yyval.InstVal = new FreeInst(yyvsp[0].ValueVal);
  ;}
    break;

  case 226:
#line 2319 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!isa<PointerType>(yyvsp[-1].TypeVal->get()))
      ThrowException("Can't load from nonpointer type: " +
                     (*yyvsp[-1].TypeVal)->getDescription());
    if (!cast<PointerType>(yyvsp[-1].TypeVal->get())->getElementType()->isFirstClassType())
      ThrowException("Can't load from pointer of non-first-class type: " +
                     (*yyvsp[-1].TypeVal)->getDescription());
    yyval.InstVal = new LoadInst(getVal(*yyvsp[-1].TypeVal, yyvsp[0].ValIDVal), "", yyvsp[-3].BoolVal);
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 227:
#line 2329 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    const PointerType *PT = dyn_cast<PointerType>(yyvsp[-1].TypeVal->get());
    if (!PT)
      ThrowException("Can't store to a nonpointer type: " +
                     (*yyvsp[-1].TypeVal)->getDescription());
    const Type *ElTy = PT->getElementType();
    if (ElTy != yyvsp[-3].ValueVal->getType())
      ThrowException("Can't store '" + yyvsp[-3].ValueVal->getType()->getDescription() +
                     "' into space of type '" + ElTy->getDescription() + "'!");

    yyval.InstVal = new StoreInst(yyvsp[-3].ValueVal, getVal(*yyvsp[-1].TypeVal, yyvsp[0].ValIDVal), yyvsp[-5].BoolVal);
    delete yyvsp[-1].TypeVal;
  ;}
    break;

  case 228:
#line 2342 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
    {
    if (!isa<PointerType>(yyvsp[-2].TypeVal->get()))
      ThrowException("getelementptr insn requires pointer operand!");

    // LLVM 1.2 and earlier used ubyte struct indices.  Convert any ubyte struct
    // indices to uint struct indices for compatibility.
    generic_gep_type_iterator<std::vector<Value*>::iterator>
      GTI = gep_type_begin(yyvsp[-2].TypeVal->get(), yyvsp[0].ValueList->begin(), yyvsp[0].ValueList->end()),
      GTE = gep_type_end(yyvsp[-2].TypeVal->get(), yyvsp[0].ValueList->begin(), yyvsp[0].ValueList->end());
    for (unsigned i = 0, e = yyvsp[0].ValueList->size(); i != e && GTI != GTE; ++i, ++GTI)
      if (isa<StructType>(*GTI))        // Only change struct indices
        if (ConstantUInt *CUI = dyn_cast<ConstantUInt>((*yyvsp[0].ValueList)[i]))
          if (CUI->getType() == Type::UByteTy)
            (*yyvsp[0].ValueList)[i] = ConstantExpr::getCast(CUI, Type::UIntTy);

    if (!GetElementPtrInst::getIndexedType(*yyvsp[-2].TypeVal, *yyvsp[0].ValueList, true))
      ThrowException("Invalid getelementptr indices for type '" +
                     (*yyvsp[-2].TypeVal)->getDescription()+ "'!");
    yyval.InstVal = new GetElementPtrInst(getVal(*yyvsp[-2].TypeVal, yyvsp[-1].ValIDVal), *yyvsp[0].ValueList);
    delete yyvsp[-2].TypeVal; delete yyvsp[0].ValueList;
  ;}
    break;


    }

/* Line 1000 of yacc.c.  */
#line 4554 "llvmAsmParser.tab.c"

  yyvsp -= yylen;
  yyssp -= yylen;


  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (YYPACT_NINF < yyn && yyn < YYLAST)
	{
	  YYSIZE_T yysize = 0;
	  int yytype = YYTRANSLATE (yychar);
	  const char* yyprefix;
	  char *yymsg;
	  int yyx;

	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  int yyxbegin = yyn < 0 ? -yyn : 0;

	  /* Stay within bounds of both yycheck and yytname.  */
	  int yychecklim = YYLAST - yyn;
	  int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
	  int yycount = 0;

	  yyprefix = ", expecting ";
	  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      {
		yysize += yystrlen (yyprefix) + yystrlen (yytname [yyx]);
		yycount += 1;
		if (yycount == 5)
		  {
		    yysize = 0;
		    break;
		  }
	      }
	  yysize += (sizeof ("syntax error, unexpected ")
		     + yystrlen (yytname[yytype]));
	  yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg != 0)
	    {
	      char *yyp = yystpcpy (yymsg, "syntax error, unexpected ");
	      yyp = yystpcpy (yyp, yytname[yytype]);

	      if (yycount < 5)
		{
		  yyprefix = ", expecting ";
		  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
		    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
		      {
			yyp = yystpcpy (yyp, yyprefix);
			yyp = yystpcpy (yyp, yytname[yyx]);
			yyprefix = " or ";
		      }
		}
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    yyerror ("syntax error; also virtual memory exhausted");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror ("syntax error");
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* If at end of input, pop the error token,
	     then the rest of the stack, then return failure.  */
	  if (yychar == YYEOF)
	     for (;;)
	       {
		 YYPOPSTACK;
		 if (yyssp == yyss)
		   YYABORT;
		 YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
		 yydestruct (yystos[*yyssp], yyvsp);
	       }
        }
      else
	{
	  YYDSYMPRINTF ("Error: discarding", yytoken, &yylval, &yylloc);
	  yydestruct (yytoken, &yylval);
	  yychar = YYEMPTY;

	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

#ifdef __GNUC__
  /* Pacify GCC when the user code never invokes YYERROR and the label
     yyerrorlab therefore never appears in user code.  */
  if (0)
     goto yyerrorlab;
#endif

  yyvsp -= yylen;
  yyssp -= yylen;
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;

      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
      yydestruct (yystos[yystate], yyvsp);
      YYPOPSTACK;
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  YYDPRINTF ((stderr, "Shifting error token, "));

  *++yyvsp = yylval;


  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*----------------------------------------------.
| yyoverflowlab -- parser overflow comes here.  |
`----------------------------------------------*/
yyoverflowlab:
  yyerror ("parser stack overflow");
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 2365 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"

int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + utostr((unsigned) llvmAsmlineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(llvmAsmtext, llvmAsmleng) + "'";
  ThrowException(errMsg);
  return 0;
}


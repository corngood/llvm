//===-- UpgradeParser.y - Upgrade parser for llvm assmbly -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the bison parser for LLVM 1.9 assembly language.
//
//===----------------------------------------------------------------------===//

%{
#define YYERROR_VERBOSE 1
#define YYSTYPE std::string*

#include "ParserInternals.h"
#include <llvm/ADT/StringExtras.h>
#include <algorithm>
#include <list>
#include <utility>
#include <iostream>

#define YYINCLUDED_STDLIB_H

int yylex();                       // declaration" of xxx warnings.
int yyparse();

static std::string CurFilename;

static std::ostream *O = 0;

std::istream* LexInput = 0;

void UpgradeAssembly(const std::string &infile, std::istream& in, 
                     std::ostream &out)
{
  Upgradelineno = 1; 
  CurFilename = infile;
  LexInput = &in;
  O = &out;

  if (yyparse()) {
    std::cerr << "Parse failed.\n";
    exit(1);
  }
}

%}

%token ESINT64VAL
%token EUINT64VAL
%token SINTVAL   // Signed 32 bit ints...
%token UINTVAL   // Unsigned 32 bit ints...
%token FPVAL     // Float or Double constant
%token VOID BOOL SBYTE UBYTE SHORT USHORT INT UINT LONG ULONG
%token FLOAT DOUBLE TYPE LABEL
%token VAR_ID LABELSTR STRINGCONSTANT
%token IMPLEMENTATION ZEROINITIALIZER TRUETOK FALSETOK BEGINTOK ENDTOK
%token DECLARE GLOBAL CONSTANT SECTION VOLATILE
%token TO DOTDOTDOT NULL_TOK UNDEF CONST INTERNAL LINKONCE WEAK APPENDING
%token DLLIMPORT DLLEXPORT EXTERN_WEAK
%token OPAQUE NOT EXTERNAL TARGET TRIPLE ENDIAN POINTERSIZE LITTLE BIG ALIGN
%token DEPLIBS CALL TAIL ASM_TOK MODULE SIDEEFFECT
%token CC_TOK CCC_TOK CSRETCC_TOK FASTCC_TOK COLDCC_TOK
%token X86_STDCALLCC_TOK X86_FASTCALLCC_TOK
%token DATALAYOUT
%token RET BR SWITCH INVOKE UNWIND UNREACHABLE
%token ADD SUB MUL UDIV SDIV FDIV UREM SREM FREM AND OR XOR
%token SETLE SETGE SETLT SETGT SETEQ SETNE  // Binary Comparators
%token MALLOC ALLOCA FREE LOAD STORE GETELEMENTPTR
%token TRUNC ZEXT SEXT FPTRUNC FPEXT BITCAST
%token UITOFP SITOFP FPTOUI FPTOSI INTTOPTR PTRTOINT
%token PHI_TOK SELECT SHL LSHR ASHR VAARG
%token EXTRACTELEMENT INSERTELEMENT SHUFFLEVECTOR
%token CAST

%start Module

%%

// Handle constant integer size restriction and conversion...
INTVAL : SINTVAL | UINTVAL 
EINT64VAL : ESINT64VAL | EUINT64VAL;

// Operations that are notably excluded from this list include:
// RET, BR, & SWITCH because they end basic blocks and are treated specially.
ArithmeticOps: ADD | SUB | MUL | UDIV | SDIV | FDIV | UREM | SREM | FREM;
LogicalOps   : AND | OR | XOR;
SetCondOps   : SETLE | SETGE | SETLT | SETGT | SETEQ | SETNE;
CastOps      : CAST;
ShiftOps     : SHL | LSHR | ASHR;

// These are some types that allow classification if we only want a particular 
// thing... for example, only a signed, unsigned, or integral type.
SIntType :  LONG |  INT |  SHORT | SBYTE;
UIntType : ULONG | UINT | USHORT | UBYTE;
IntType  : SIntType | UIntType;
FPType   : FLOAT | DOUBLE;

// OptAssign - Value producing statements have an optional assignment component
OptAssign : Name '=' {
    $1->append(" = ");
    $$ = $1;
  }
  | /*empty*/ {
    $$ = new std::string(""); 
  };

OptLinkage 
  : INTERNAL | LINKONCE | WEAK | APPENDING | DLLIMPORT | DLLEXPORT 
  | EXTERN_WEAK 
  | /*empty*/   { $$ = new std::string(""); } ;

OptCallingConv 
  : CCC_TOK | CSRETCC_TOK | FASTCC_TOK | COLDCC_TOK | X86_STDCALLCC_TOK 
  | X86_FASTCALLCC_TOK | CC_TOK EUINT64VAL
  | /*empty*/ { $$ = new std::string(""); } ;

// OptAlign/OptCAlign - An optional alignment, and an optional alignment with
// a comma before it.
OptAlign 
  : /*empty*/        { $$ = new std::string(); }
  | ALIGN EUINT64VAL { *$1 += " " + *$2; delete $2; $$ = $1; };
         ;
OptCAlign 
  : /*empty*/            { $$ = new std::string(); } 
  | ',' ALIGN EUINT64VAL { 
    $2->insert(0, ", "); 
    *$2 += " " + *$3;
    delete $3;
    $$ = $2;
  };

SectionString 
  : SECTION STRINGCONSTANT { 
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  };

OptSection : /*empty*/     { $$ = new std::string(); } 
           | SectionString;

GlobalVarAttributes 
    : /* empty */ { $$ = new std::string(); } 
    | ',' GlobalVarAttribute GlobalVarAttributes  {
      $2->insert(0, ", ");
      if (!$3->empty())
        *$2 += " " + *$3;
      delete $3;
      $$ = $2;
    };

GlobalVarAttribute 
    : SectionString 
    | ALIGN EUINT64VAL {
      *$1 += " " + *$2;
      delete $2;
      $$ = $1;
    };

//===----------------------------------------------------------------------===//
// Types includes all predefined types... except void, because it can only be
// used in specific contexts (function returning void for example).  To have
// access to it, a user must explicitly use TypesV.
//

// TypesV includes all of 'Types', but it also includes the void type.
TypesV    : Types    | VOID ;
UpRTypesV : UpRTypes | VOID ; 
Types     : UpRTypes ;

// Derived types are added later...
//
PrimType : BOOL | SBYTE | UBYTE | SHORT  | USHORT | INT   | UINT ;
PrimType : LONG | ULONG | FLOAT | DOUBLE | TYPE   | LABEL;
UpRTypes : OPAQUE | PrimType | SymbolicValueRef ;

// Include derived types in the Types production.
//
UpRTypes : '\\' EUINT64VAL {                   // Type UpReference
    $2->insert(0, "\\");
    $$ = $2;
  }
  | UpRTypesV '(' ArgTypeListI ')' {           // Function derived type?
    *$1 += "( " + *$3 + " )";
    delete $3;
    $$ = $1;
  }
  | '[' EUINT64VAL 'x' UpRTypes ']' {          // Sized array type?
    $2->insert(0,"[ ");
    *$2 += " x " + *$4 + " ]";
    delete $4;
    $$ = $2;
  }
  | '<' EUINT64VAL 'x' UpRTypes '>' {          // Packed array type?
    $2->insert(0,"< ");
    *$2 += " x " + *$4 + " >";
    delete $4;
    $$ = $2;
  }
  | '{' TypeListI '}' {                        // Structure type?
    $2->insert(0, "{ ");
    *$2 += " }";
    $$ = $2;
  }
  | '{' '}' {                                  // Empty structure type?
    $$ = new std::string("{ }");
  }
  | UpRTypes '*' {                             // Pointer type?
    *$1 += '*';
    $$ = $1;
  };

// TypeList - Used for struct declarations and as a basis for function type 
// declaration type lists
//
TypeListI : UpRTypes | TypeListI ',' UpRTypes {
    *$1 += ", " + *$3;
    delete $3;
    $$ = $1;
  };

// ArgTypeList - List of types for a function type declaration...
ArgTypeListI : TypeListI
  | TypeListI ',' DOTDOTDOT {
    *$1 += ", ...";
    delete $3;
    $$ = $1;
  }
  | DOTDOTDOT {
    $$ = $1;
  }
  | /*empty*/ {
    $$ = new std::string();
  };

// ConstVal - The various declarations that go into the constant pool.  This
// production is used ONLY to represent constants that show up AFTER a 'const',
// 'constant' or 'global' token at global scope.  Constants that can be inlined
// into other expressions (such as integers and constexprs) are handled by the
// ResolvedVal, ValueRef and ConstValueRef productions.
//
ConstVal: Types '[' ConstVector ']' { // Nonempty unsized arr
    *$1 += " [ " + *$3 + " ]";
    delete $3;
    $$ = $1;
  }
  | Types '[' ']' {
    $$ = new std::string("[ ]");
  }
  | Types 'c' STRINGCONSTANT {
    *$1 += " c" + *$3;
    delete $3;
    $$ = $1;
  }
  | Types '<' ConstVector '>' { // Nonempty unsized arr
    *$1 += " < " + *$3 + " >";
    delete $3;
    $$ = $1;
  }
  | Types '{' ConstVector '}' {
    *$1 += " { " + *$3 + " }";
    delete $3;
    $$ = $1;
  }
  | Types '{' '}' {
    $$ = new std::string("[ ]");
  }
  | Types NULL_TOK {
    *$1 += " " + *$2; 
    delete $2;
    $$ = $1;
  }
  | Types UNDEF {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | Types SymbolicValueRef {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | Types ConstExpr {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | Types ZEROINITIALIZER {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  };

ConstVal : SIntType EINT64VAL {      // integral constants
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | UIntType EUINT64VAL {            // integral constants
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | BOOL TRUETOK {                      // Boolean constants
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | BOOL FALSETOK {                     // Boolean constants
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | FPType FPVAL {                   // Float & Double constants
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  };


ConstExpr: CastOps '(' ConstVal TO Types ')' {
    *$1 += " (" + *$3 + " " + *$4 + " " + *$5 + ")";
    delete $3; delete $4; delete $5;
    $$ = $1;
  }
  | GETELEMENTPTR '(' ConstVal IndexList ')' {
  }
  | SELECT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
  }
  | ArithmeticOps '(' ConstVal ',' ConstVal ')' {
  }
  | LogicalOps '(' ConstVal ',' ConstVal ')' {
  }
  | SetCondOps '(' ConstVal ',' ConstVal ')' {
  }
  | ShiftOps '(' ConstVal ',' ConstVal ')' {
  }
  | EXTRACTELEMENT '(' ConstVal ',' ConstVal ')' {
  }
  | INSERTELEMENT '(' ConstVal ',' ConstVal ',' ConstVal ')' {
  }
  | SHUFFLEVECTOR '(' ConstVal ',' ConstVal ',' ConstVal ')' {
  };


// ConstVector - A list of comma separated constants.
ConstVector : ConstVector ',' ConstVal {
  }
  | ConstVal {
  };


// GlobalType - Match either GLOBAL or CONSTANT for global declarations...
GlobalType : GLOBAL { } | CONSTANT { };


//===----------------------------------------------------------------------===//
//                             Rules to match Modules
//===----------------------------------------------------------------------===//

// Module rule: Capture the result of parsing the whole file into a result
// variable...
//
Module : DefinitionList {
};

// DefinitionList - Top level definitions
//
DefinitionList : DefinitionList Function {
    $$ = 0;
  } 
  | DefinitionList FunctionProto {
    *O << *$2 << "\n";
    delete $2;
    $$ = 0;
  }
  | DefinitionList MODULE ASM_TOK AsmBlock {
    *O << "module asm " << " " << *$4 << "\n";
  }  
  | DefinitionList IMPLEMENTATION {
    *O << "implementation\n";
  }
  | ConstPool {
  };

// ConstPool - Constants with optional names assigned to them.
ConstPool : ConstPool OptAssign TYPE TypesV {
    *O << *$2 << " " << *$3 << " " << *$4 << "\n";
    delete $2; delete $3; delete $4;
    $$ = 0;
  }
  | ConstPool FunctionProto {       // Function prototypes can be in const pool
    *O << *$2 << "\n";
    delete $2;
    $$ = 0;
  }
  | ConstPool MODULE ASM_TOK AsmBlock {  // Asm blocks can be in the const pool
    *O << *$2 << " " << *$3 << " " << *$4 << "\n";
    delete $2; delete $3; delete $4; 
    $$ = 0;
  }
  | ConstPool OptAssign OptLinkage GlobalType ConstVal  GlobalVarAttributes {
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5 << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; delete $5; delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign EXTERNAL GlobalType Types  GlobalVarAttributes {
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5 << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; delete $5; delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign DLLIMPORT GlobalType Types  GlobalVarAttributes {
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5 << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; delete $5; delete $6;
    $$ = 0;
  }
  | ConstPool OptAssign EXTERN_WEAK GlobalType Types  GlobalVarAttributes {
    *O << *$2 << " " << *$3 << " " << *$4 << " " << *$5 << " " << *$6 << "\n";
    delete $2; delete $3; delete $4; delete $5; delete $6;
    $$ = 0;
  }
  | ConstPool TARGET TargetDefinition { 
    *O << *$2 << " " << *$3 << "\n";
    delete $2; delete $3;
    $$ = 0;
  }
  | ConstPool DEPLIBS '=' LibrariesDefinition {
    *O << *$2 << " = " << *$4 << "\n";
    delete $2; delete $4;
    $$ = 0;
  }
  | /* empty: end of list */ { 
    $$ = 0;
  };


AsmBlock : STRINGCONSTANT ;

BigOrLittle : BIG | LITTLE 

TargetDefinition 
  : ENDIAN '=' BigOrLittle {
    *$1 += " = " + *$2;
    delete $2;
    $$ = $1;
  }
  | POINTERSIZE '=' EUINT64VAL {
    *$1 += " = " + *$2;
    delete $2;
    $$ = $1;
  }
  | TRIPLE '=' STRINGCONSTANT {
    *$1 += " = " + *$2;
    delete $2;
    $$ = $1;
  }
  | DATALAYOUT '=' STRINGCONSTANT {
    *$1 += " = " + *$2;
    delete $2;
    $$ = $1;
  };

LibrariesDefinition 
  : '[' LibList ']' {
    $2->insert(0, "[ ");
    *$2 += " ]";
    $$ = $2;
  };

LibList 
  : LibList ',' STRINGCONSTANT {
    *$1 += ", " + *$3;
    delete $3;
    $$ = $1;
  }
  | STRINGCONSTANT 
  | /* empty: end of list */ {
    $$ = new std::string();
  };

//===----------------------------------------------------------------------===//
//                       Rules to match Function Headers
//===----------------------------------------------------------------------===//

Name : VAR_ID | STRINGCONSTANT;
OptName : Name | /*empty*/ { $$ = new std::string(); };

ArgVal : Types OptName {
  $$ = $1;
  if (!$2->empty())
    *$$ += " " + *$2;
};

ArgListH : ArgListH ',' ArgVal {
    *$1 += ", " + *$3;
  }
  | ArgVal {
    $$ = $1;
  };

ArgList : ArgListH {
    $$ = $1;
  }
  | ArgListH ',' DOTDOTDOT {
    *$1 += ", ...";
    $$ = $1;
  }
  | DOTDOTDOT {
    $$ = $1;
  }
  | /* empty */ {
    $$ = new std::string();
  };

FunctionHeaderH : OptCallingConv TypesV Name '(' ArgList ')' 
                  OptSection OptAlign {
    if (!$1->empty()) {
      $2->insert(0, *$1 + " ");
    }
    *$2 += " " + *$3 + "( " + *$5 + " )";
    if (!$7->empty()) {
      *$2 += " " + *$7;
    }
    if (!$8->empty()) {
      *$2 += " " + *$8;
    }
    $$ = $2;
  };

BEGIN : BEGINTOK {
    $$ = new std::string("begin");
  }
  | '{' { 
    $$ = new std::string ("{");
  }

FunctionHeader : OptLinkage FunctionHeaderH BEGIN {
  if (!$1->empty()) {
    *O << *$1 << " ";
  }
  *O << *$2 << " " << *$3 << "\n";
  delete $1; delete $2; delete $3;
  $$ = 0;
};

END : ENDTOK { $$ = new std::string("end"); }
    | '}' { $$ = new std::string("}"); };

Function : FunctionHeader BasicBlockList END {
  if ($2)
    *O << *$2;
  *O << '\n' << *$3 << "\n";
};

FnDeclareLinkage: /*default*/ 
  | DLLIMPORT    
  | EXTERN_WEAK 
  ;
  
FunctionProto 
  : DECLARE FnDeclareLinkage FunctionHeaderH { 
    *$1 += " " + *$2 + " " + *$3;
    delete $2; delete $3;
    $$ = $1;
  };

//===----------------------------------------------------------------------===//
//                        Rules to match Basic Blocks
//===----------------------------------------------------------------------===//

OptSideEffect : /* empty */ {
  }
  | SIDEEFFECT {
  };

ConstValueRef : ESINT64VAL | EUINT64VAL | FPVAL | TRUETOK  | FALSETOK 
  | NULL_TOK | UNDEF | ZEROINITIALIZER 
  | '<' ConstVector '>' { 
    $2->insert(0, "<");
    *$2 += ">";
    $$ = $2;
  }
  | ConstExpr 
  | ASM_TOK OptSideEffect STRINGCONSTANT ',' STRINGCONSTANT {
    if (!$2->empty()) {
      *$1 += " " + *$2;
    }
    *$1 += " " + *$3 + ", " + *$4;
    delete $2; delete $3; delete $4;
    $$ = $1;
  };

SymbolicValueRef : INTVAL | Name ;

// ValueRef - A reference to a definition... either constant or symbolic
ValueRef : SymbolicValueRef | ConstValueRef;


// ResolvedVal - a <type> <value> pair.  This is used only in cases where the
// type immediately preceeds the value reference, and allows complex constant
// pool references (for things like: 'ret [2 x int] [ int 12, int 42]')
ResolvedVal : Types ValueRef {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  };

BasicBlockList : BasicBlockList BasicBlock {
  }
  | BasicBlock { // Do not allow functions with 0 basic blocks   
  };


// Basic blocks are terminated by branching instructions: 
// br, br/cc, switch, ret
//
BasicBlock : InstructionList OptAssign BBTerminatorInst  {
    *O << *$2 ;
  };

InstructionList : InstructionList Inst {
    *O << "    " << *$2 << "\n";
    delete $2;
    $$ = 0;
  }
  | /* empty */ {
    $$ = 0;
  }
  | LABELSTR {
    *O << *$1 << "\n";
    delete $1;
    $$ = 0;
  };

BBTerminatorInst : RET ResolvedVal {              // Return with a result...
    *O << "    " << *$1 << " " << *$2 << "\n";
    delete $1; delete $2;
    $$ = 0;
  }
  | RET VOID {                                       // Return with no result...
    *O << "    " << *$1 << " " << *$2 << "\n";
    delete $1; delete $2;
    $$ = 0;
  }
  | BR LABEL ValueRef {                         // Unconditional Branch...
    *O << "    " << *$1 << " " << *$2 << " " << *$3 << "\n";
    delete $1; delete $2; delete $3;
    $$ = 0;
  }                                                  // Conditional Branch...
  | BR BOOL ValueRef ',' LABEL ValueRef ',' LABEL ValueRef {  
    *O << "    " << *$1 << " " << *$2 << " " << *$3 << ", " << *$5 << " "
       << *$6 << ", " << *$8 << " " << *$9 << "\n";
    delete $1; delete $2; delete $3; delete $5; delete $6; delete $8; delete $9;
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' JumpTable ']' {
    *O << "    " << *$1 << " " << *$2 << " " << *$3 << ", " << *$5 << " " 
       << *$6 << " [" << *$8 << " ]\n";
    delete $1; delete $2; delete $3; delete $5; delete $6; delete $8;
    $$ = 0;
  }
  | SWITCH IntType ValueRef ',' LABEL ValueRef '[' ']' {
    *O << "    " << *$1 << " " << *$2 << " " << *$3 << ", " << *$5 << " " 
       << *$6 << "[]\n";
    delete $1; delete $2; delete $3; delete $5; delete $6;
    $$ = 0;
  }
  | INVOKE OptCallingConv TypesV ValueRef '(' ValueRefListE ')'
    TO LABEL ValueRef UNWIND LABEL ValueRef {
    *O << "    " << *$1 << " " << *$2 << " " << *$3 << " " << *$4 << " ("
       << *$6 << ") " << *$8 << " " << *$9 << " " << *$10 << " " << *$11 << " "
       << *$12 << " " << *$13 << "\n";
    delete $1; delete $2; delete $3; delete $4; delete $6; delete $8; delete $9;
    delete $10; delete $11; delete $12; delete $13; 
    $$ = 0;
  }
  | UNWIND {
    *O << "    " << *$1 << "\n";
    delete $1;
    $$ = 0;
  }
  | UNREACHABLE {
    *O << "    " << *$1 << "\n";
    delete $1;
    $$ = 0;
  };

JumpTable : JumpTable IntType ConstValueRef ',' LABEL ValueRef {
    *$1 += *$2 + " " + *$3 + ", " + *$5 + " " + *$6;
    delete $2; delete $3; delete $5; delete $6;
    $$ = $1;
  }
  | IntType ConstValueRef ',' LABEL ValueRef {
    *$1 += *$2 + ", " + *$4 + " " + *$5;
    delete $2; delete $4; delete $5;
    $$ = $1;
  };

Inst 
  : OptAssign InstVal {
    *$1 += *$2;
    delete $2;
    $$ = $1; 
  };

PHIList 
  : Types '[' ValueRef ',' ValueRef ']' {    // Used for PHI nodes
    *$1 += " [" + *$3 + "," + *$5 + "]";
    delete $3; delete $5;
    $$ = $1;
  }
  | PHIList ',' '[' ValueRef ',' ValueRef ']' {
    *$1 += ", [" + *$4 + "," + *$6 + "]";
    delete $4; delete $6;
    $$ = $1;
  };


ValueRefList 
  : ResolvedVal 
  | ValueRefList ',' ResolvedVal {
    *$1 += ", " + *$3;
    delete $3;
    $$ = $1;
  };

// ValueRefListE - Just like ValueRefList, except that it may also be empty!
ValueRefListE 
  : ValueRefList 
  | /*empty*/ { $$ = new std::string(); }
  ;

OptTailCall 
  : TAIL CALL {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | CALL 
  ;

InstVal : ArithmeticOps Types ValueRef ',' ValueRef {
    *$1 += " " + *$2 + " " + *$3 + ", " + *$5;
    delete $2; delete $3; delete $5;
    $$ = $1;
  }
  | LogicalOps Types ValueRef ',' ValueRef {
    *$1 += " " + *$2 + " " + *$3 + ", " + *$5;
    delete $2; delete $3; delete $5;
    $$ = $1;
  }
  | SetCondOps Types ValueRef ',' ValueRef {
    *$1 += " " + *$2 + " " + *$3 + ", " + *$5;
    delete $2; delete $3; delete $5;
    $$ = $1;
  }
  | NOT ResolvedVal {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | ShiftOps ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2 + ", " + *$4;
    delete $2; delete $4;
    $$ = $1;
  }
  | CastOps ResolvedVal TO Types {
    *$1 += " " + *$2 + " " + *$3 + ", " + *$4;
    delete $2; delete $3; delete $4;
    $$ = $1;
  }
  | SELECT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2 + ", " + *$4 + ", " + *$6;
    delete $2; delete $4; delete $6;
    $$ = $1;
  }
  | VAARG ResolvedVal ',' Types {
    *$1 += " " + *$2 + ", " + *$4;
    delete $2; delete $4;
    $$ = $1;
  }
  | EXTRACTELEMENT ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2 + ", " + *$4;
    delete $2; delete $4;
    $$ = $1;
  }
  | INSERTELEMENT ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2 + ", " + *$4 + ", " + *$6;
    delete $2; delete $4; delete $6;
    $$ = $1;
  }
  | SHUFFLEVECTOR ResolvedVal ',' ResolvedVal ',' ResolvedVal {
    *$1 += " " + *$2 + ", " + *$4 + ", " + *$6;
    delete $2; delete $4; delete $6;
    $$ = $1;
  }
  | PHI_TOK PHIList {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | OptTailCall OptCallingConv TypesV ValueRef '(' ValueRefListE ')'  {
    if (!$2->empty())
      *$1 += " " + *$2;
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$3 += " " + *$4 + "(" + *$5 + ")";
    delete $2; delete $3; delete $4; delete $6;
    $$ = $1;
  }
  | MemoryInst ;


// IndexList - List of indices for GEP based instructions...
IndexList 
  : ',' ValueRefList { 
    $2->insert(0, ", ");
    $$ = $2;
  } 
  | /* empty */ {  $$ = new std::string(); }
  ;

OptVolatile 
  : VOLATILE 
  | /* empty */ { $$ = new std::string(); }
  ;

MemoryInst : MALLOC Types OptCAlign {
    *$1 += " " + *$2;
    if (!$3->empty())
      *$1 += " " + *$3;
    delete $2; delete $3;
    $$ = $1;
  }
  | MALLOC Types ',' UINT ValueRef OptCAlign {
    *$1 += " " + *$2 + ", " + *$4 + " " + *$5;
    if (!$6->empty())
      *$1 += " " + *$6;
    delete $2; delete $4; delete $5; delete $6;
    $$ = $1;
  }
  | ALLOCA Types OptCAlign {
    *$1 += " " + *$2;
    if (!$3->empty())
      *$1 += " " + *$3;
    delete $2; delete $3;
    $$ = $1;
  }
  | ALLOCA Types ',' UINT ValueRef OptCAlign {
    *$1 += " " + *$2 + ", " + *$4 + " " + *$5;
    if (!$6->empty())
      *$1 += " " + *$6;
    delete $2; delete $4; delete $5; delete $6;
    $$ = $1;
  }
  | FREE ResolvedVal {
    *$1 += " " + *$2;
    delete $2;
    $$ = $1;
  }
  | OptVolatile LOAD Types ValueRef {
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$2 + " " + *$3 + " " + *$4;
    delete $2; delete $3; delete $4;
    $$ = $1;
  }
  | OptVolatile STORE ResolvedVal ',' Types ValueRef {
    if (!$1->empty())
      *$1 += " ";
    *$1 += *$2 + " " + *$3 + ", " + *$5 + " " + *$6;
    delete $2; delete $3; delete $5; delete $6;
    $$ = $1;
  }
  | GETELEMENTPTR Types ValueRef IndexList {
    *$1 += *$2 + " " + *$3 + " " + *$4;
    delete $2; delete $3; delete $4;
    $$ = $1;
  };

%%

int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << errMsg << '\n';
  exit(1);
}

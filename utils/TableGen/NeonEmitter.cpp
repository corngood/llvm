//===- NeonEmitter.cpp - Generate arm_neon.h for use with clang -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting arm_neon.h, which includes
// a declaration and definition of each function specified by the ARM NEON 
// compiler interface.  See ARM document DUI0348B.
//
//===----------------------------------------------------------------------===//

#include "NeonEmitter.h"
#include "Record.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include <string>

using namespace llvm;

enum OpKind {
  OpNone,
  OpAdd,
  OpSub,
  OpMul,
  OpMla,
  OpMls,
  OpEq,
  OpGe,
  OpLe,
  OpGt,
  OpLt,
  OpNeg,
  OpNot,
  OpAnd,
  OpOr,
  OpXor,
  OpAndNot,
  OpOrNot,
  OpCast
};

enum ClassKind {
  ClassNone,
  ClassI,
  ClassS,
  ClassW,
  ClassB
};

static void ParseTypes(Record *r, std::string &s,
                       SmallVectorImpl<StringRef> &TV) {
  const char *data = s.data();
  int len = 0;
  
  for (unsigned i = 0, e = s.size(); i != e; ++i, ++len) {
    if (data[len] == 'P' || data[len] == 'Q' || data[len] == 'U')
      continue;
    
    switch (data[len]) {
      case 'c':
      case 's':
      case 'i':
      case 'l':
      case 'h':
      case 'f':
        break;
      default:
        throw TGError(r->getLoc(),
                      "Unexpected letter: " + std::string(data + len, 1));
        break;
    }
    TV.push_back(StringRef(data, len + 1));
    data += len + 1;
    len = -1;
  }
}

static char Widen(const char t) {
  switch (t) {
    case 'c':
      return 's';
    case 's':
      return 'i';
    case 'i':
      return 'l';
    default: throw "unhandled type in widen!";
  }
  return '\0';
}

static char Narrow(const char t) {
  switch (t) {
    case 's':
      return 'c';
    case 'i':
      return 's';
    case 'l':
      return 'i';
    default: throw "unhandled type in widen!";
  }
  return '\0';
}

static char ClassifyType(StringRef ty, bool &quad, bool &poly, bool &usgn) {
  unsigned off = 0;
  
  // remember quad.
  if (ty[off] == 'Q') {
    quad = true;
    ++off;
  }
  
  // remember poly.
  if (ty[off] == 'P') {
    poly = true;
    ++off;
  }
  
  // remember unsigned.
  if (ty[off] == 'U') {
    usgn = true;
    ++off;
  }
  
  // base type to get the type string for.
  return ty[off];
}

static std::string TypeString(const char mod, StringRef typestr) {
  bool quad = false;
  bool poly = false;
  bool usgn = false;
  bool scal = false;
  bool cnst = false;
  bool pntr = false;
  
  // base type to get the type string for.
  char type = ClassifyType(typestr, quad, poly, usgn);
  
  // Based on the modifying character, change the type and width if necessary.
  switch (mod) {
    case 'v':
      return "void";
    case 'i':
      return "int";
    case 't':
      if (poly) {
        poly = false;
        usgn = true;
      }
      break;
    case 'x':
      usgn = true;
      poly = false;
      if (type == 'f')
        type = 'i';
      break;
    case 'f':
      type = 'f';
      usgn = false;
      break;
    case 'w':
      type = Widen(type);
      quad = true;
      break;
    case 'n':
      type = Widen(type);
      break;
    case 'l':
      type = 'l';
      scal = true;
      usgn = true;
      break;
    case 's':
      scal = true;
      break;
    case 'k':
      quad = true;
      break;
    case 'c':
      cnst = true;
    case 'p':
      pntr = true;
      scal = true;
      break;
    case 'h':
      type = Narrow(type);
      break;
    case 'e':
      type = Narrow(type);
      usgn = true;
      break;
    default:
      break;
  }
  
  SmallString<128> s;
  
  if (usgn)
    s.push_back('u');
  
  switch (type) {
    case 'c':
      s += poly ? "poly8" : "int8";
      if (scal)
        break;
      s += quad ? "x16" : "x8";
      break;
    case 's':
      s += poly ? "poly16" : "int16";
      if (scal)
        break;
      s += quad ? "x8" : "x4";
      break;
    case 'i':
      s += "int32";
      if (scal)
        break;
      s += quad ? "x4" : "x2";
      break;
    case 'l':
      s += "int64";
      if (scal)
        break;
      s += quad ? "x2" : "x1";
      break;
    case 'h':
      s += "float16";
      if (scal)
        break;
      s += quad ? "x8" : "x4";
      break;
    case 'f':
      s += "float32";
      if (scal)
        break;
      s += quad ? "x4" : "x2";
      break;
    default:
      throw "unhandled type!";
      break;
  }

  if (mod == '2')
    s += "x2";
  if (mod == '3')
    s += "x3";
  if (mod == '4')
    s += "x4";
  
  // Append _t, finishing the type string typedef type.
  s += "_t";
  
  if (cnst)
    s += " const";
  
  if (pntr)
    s += " *";
  
  return s.str();
}

// Turn "vst2_lane" into "vst2q_lane_f32", etc.
static std::string MangleName(const std::string &name, StringRef typestr,
                              ClassKind ck) {
  bool quad = false;
  bool poly = false;
  bool usgn = false;
  char type = ClassifyType(typestr, quad, poly, usgn);

  std::string s = name;
  
  switch (type) {
  case 'c':
    switch (ck) {
    case ClassS: s += poly ? "_p8" : usgn ? "_u8" : "_s8"; break;
    case ClassI: s += "_i8"; break;
    case ClassW: s += "_8"; break;
    default: break;
    }
    break;
  case 's':
    switch (ck) {
    case ClassS: s += poly ? "_p16" : usgn ? "_u16" : "_s16"; break;
    case ClassI: s += "_i16"; break;
    case ClassW: s += "_16"; break;
    default: break;
    }
    break;
  case 'i':
    switch (ck) {
    case ClassS: s += usgn ? "_u32" : "_s32"; break;
    case ClassI: s += "_i32"; break;
    case ClassW: s += "_32"; break;
    default: break;
    }
    break;
  case 'l':
    switch (ck) {
    case ClassS: s += usgn ? "_u64" : "_s64"; break;
    case ClassI: s += "_i64"; break;
    case ClassW: s += "_64"; break;
    default: break;
    }
    break;
  case 'h':
    switch (ck) {
    case ClassS:
    case ClassI: s += "_f16"; break;
    case ClassW: s += "_16"; break;
    default: break;
    }
    break;
  case 'f':
    switch (ck) {
    case ClassS:
    case ClassI: s += "_f32"; break;
    case ClassW: s += "_32"; break;
    default: break;
    }
    break;
  default:
    throw "unhandled type!";
    break;
  }
  if (ck == ClassB)
    return s += "_v";
    
  // Insert a 'q' before the first '_' character so that it ends up before 
  // _lane or _n on vector-scalar operations.
  if (quad) {
    size_t pos = s.find('_');
    s = s.insert(pos, "q");
  }
  return s;
}

// Generate the string "(argtype a, argtype b, ...)"
static std::string GenArgs(const std::string &proto, StringRef typestr) {
  char arg = 'a';
  
  std::string s;
  s += "(";
  
  for (unsigned i = 1, e = proto.size(); i != e; ++i, ++arg) {
    s += TypeString(proto[i], typestr);
    s.push_back(' ');
    s.push_back(arg);
    if ((i + 1) < e)
      s += ", ";
  }
  
  s += ")";
  return s;
}

// Generate the definition for this intrinsic, e.g. "a + b" for OpAdd.
// If structTypes is true, the NEON types are structs of vector types rather
// than vector types, and the call becomes "a.val + b.val"
static std::string GenOpString(OpKind op, const std::string &proto,
                               StringRef typestr, bool structTypes = true) {
  std::string s("return ");
  std::string ts = TypeString(proto[0], typestr);
  if (structTypes)
    s += "(" + ts + "){";
  
  std::string a, b, c;
  if (proto.size() > 1)
    a = (structTypes && proto[1] != 'l') ? "a.val" : "a";
  b = structTypes ? "b.val" : "b";
  c = structTypes ? "c.val" : "c";
  
  switch(op) {
  case OpAdd:
    s += a + " + " + b;
    break;
  case OpSub:
    s += a + " - " + b;
    break;
  case OpMul:
    s += a + " * " + b;
    break;
  case OpMla:
    s += a + " + ( " + b + " * " + c + " )";
    break;
  case OpMls:
    s += a + " - ( " + b + " * " + c + " )";
    break;
  case OpEq:
    s += "(__neon_" + ts + ")(" + a + " == " + b + ")";
    break;
  case OpGe:
    s += "(__neon_" + ts + ")(" + a + " >= " + b + ")";
    break;
  case OpLe:
    s += "(__neon_" + ts + ")(" + a + " <= " + b + ")";
    break;
  case OpGt:
    s += "(__neon_" + ts + ")(" + a + " > " + b + ")";
    break;
  case OpLt:
    s += "(__neon_" + ts + ")(" + a + " < " + b + ")";
    break;
  case OpNeg:
    s += " -" + a;
    break;
  case OpNot:
    s += " ~" + a;
    break;
  case OpAnd:
    s += a + " & " + b;
    break;
  case OpOr:
    s += a + " | " + b;
    break;
  case OpXor:
    s += a + " ^ " + b;
    break;
  case OpAndNot:
    s += a + " & ~" + b;
    break;
  case OpOrNot:
    s += a + " | ~" + b;
    break;
  case OpCast:
    s += "(__neon_" + ts + ")" + a;
    break;
  default:
    throw "unknown OpKind!";
    break;
  }
  
  if (structTypes)
    s += "}";
  s += ";";
  return s;
}

// Generate the definition for this intrinsic, e.g. __builtin_neon_cls(a)
// If structTypes is true, the NEON types are structs of vector types rather
// than vector types, and the call becomes __builtin_neon_cls(a.val)
static std::string GenBuiltin(const std::string &name, const std::string &proto,
                              StringRef typestr, ClassKind ck,
                              bool structTypes = true) {
  char arg = 'a';
  std::string s;
  
  if (proto[0] != 'v') {
    // FIXME: if return type is 2/3/4, emit unioning code.
    s += "return ";
    if (structTypes) {
      s += "(";
      s += TypeString(proto[0], typestr);
      s += "){";
    }
  }    
  
  s += "__builtin_neon_";
  s += MangleName(name, typestr, ck);
  s += "(";
  
  for (unsigned i = 1, e = proto.size(); i != e; ++i, ++arg) {
    s.push_back(arg);
    if (structTypes && proto[i] != 's' && proto[i] != 'i' && proto[i] != 'l' &&
        proto[i] != 'p' && proto[i] != 'c') {
      s += ".val";
    }
    if ((i + 1) < e)
      s += ", ";
  }
  
  s += ")";
  if (proto[0] != 'v' && structTypes)
    s += "}";
  s += ";";
  return s;
}

void NeonEmitter::run(raw_ostream &OS) {
  EmitSourceFileHeader("ARM NEON Header", OS);
  
  // FIXME: emit license into file?
  
  OS << "#ifndef __ARM_NEON_H\n";
  OS << "#define __ARM_NEON_H\n\n";
  
  OS << "#ifndef __ARM_NEON__\n";
  OS << "#error \"NEON support not enabled\"\n";
  OS << "#endif\n\n";

  OS << "#include <stdint.h>\n\n";

  // Emit NEON-specific scalar typedefs.
  // FIXME: probably need to do something better for polynomial types.
  // FIXME: is this the correct thing to do for float16?
  OS << "typedef float float32_t;\n";
  OS << "typedef uint8_t poly8_t;\n";
  OS << "typedef uint16_t poly16_t;\n";
  OS << "typedef uint16_t float16_t;\n";
  
  // Emit Neon vector typedefs.
  std::string TypedefTypes("cQcsQsiQilQlUcQUcUsQUsUiQUiUlQUlhQhfQfPcQPcPsQPs");
  SmallVector<StringRef, 24> TDTypeVec;
  ParseTypes(0, TypedefTypes, TDTypeVec);

  // Emit vector typedefs.
  for (unsigned i = 0, e = TDTypeVec.size(); i != e; ++i) {
    bool dummy, quad = false;
    (void) ClassifyType(TDTypeVec[i], quad, dummy, dummy);
    OS << "typedef __attribute__(( __vector_size__(";
    OS << (quad ? "16) )) " : "8) ))  ");
    OS << TypeString('s', TDTypeVec[i]);
    OS << " __neon_";
    OS << TypeString('d', TDTypeVec[i]) << ";\n";
  }
  OS << "\n";

  // Emit struct typedefs.
  for (unsigned vi = 1; vi != 5; ++vi) {
    for (unsigned i = 0, e = TDTypeVec.size(); i != e; ++i) {
      std::string ts = TypeString('d', TDTypeVec[i]);
      std::string vs = (vi > 1) ? TypeString('0' + vi, TDTypeVec[i]) : ts;
      OS << "typedef struct __" << vs << " {\n";
      OS << "  __neon_" << ts << " val";
      if (vi > 1)
        OS << "[" << utostr(vi) << "]";
      OS << ";\n} " << vs << ";\n\n";
    }
  }
  
  OS << "#define __ai static __attribute__((__always_inline__))\n\n";

  std::vector<Record*> RV = Records.getAllDerivedDefinitions("Inst");
  
  StringMap<OpKind> OpMap;
  OpMap["OP_NONE"] = OpNone;
  OpMap["OP_ADD"]  = OpAdd;
  OpMap["OP_SUB"]  = OpSub;
  OpMap["OP_MUL"]  = OpMul;
  OpMap["OP_MLA"]  = OpMla;
  OpMap["OP_MLS"]  = OpMls;
  OpMap["OP_EQ"]   = OpEq;
  OpMap["OP_GE"]   = OpGe;
  OpMap["OP_LE"]   = OpLe;
  OpMap["OP_GT"]   = OpGt;
  OpMap["OP_LT"]   = OpLt;
  OpMap["OP_NEG"]  = OpNeg;
  OpMap["OP_NOT"]  = OpNot;
  OpMap["OP_AND"]  = OpAnd;
  OpMap["OP_OR"]   = OpOr;
  OpMap["OP_XOR"]  = OpXor;
  OpMap["OP_ANDN"] = OpAndNot;
  OpMap["OP_ORN"]  = OpOrNot;
  OpMap["OP_CAST"] = OpCast;
  
  DenseMap<Record*, ClassKind> ClassMap;
  Record *SI = Records.getClass("SInst");
  Record *II = Records.getClass("IInst");
  Record *WI = Records.getClass("WInst");
  Record *BI = Records.getClass("BInst");
  ClassMap[SI] = ClassS;
  ClassMap[II] = ClassI;
  ClassMap[WI] = ClassW;
  ClassMap[BI] = ClassB;
  
  // Unique the return+pattern types, and assign them.
  for (unsigned i = 0, e = RV.size(); i != e; ++i) {
    Record *R = RV[i];
    std::string name = LowercaseString(R->getName());
    std::string Proto = R->getValueAsString("Prototype");
    std::string Types = R->getValueAsString("Types");
    
    SmallVector<StringRef, 16> TypeVec;
    ParseTypes(R, Types, TypeVec);
    
    OpKind k = OpMap[R->getValueAsDef("Operand")->getName()];
    
    for (unsigned ti = 0, te = TypeVec.size(); ti != te; ++ti) {
      assert(!Proto.empty() && "");
      
      // static always inline + return type
      OS << "__ai " << TypeString(Proto[0], TypeVec[ti]);
      
      // Function name with type suffix
      OS << " " << MangleName(name, TypeVec[ti], ClassS);
      
      // Function arguments
      OS << GenArgs(Proto, TypeVec[ti]);
      
      // Definition.
      OS << " { ";
      
      if (k != OpNone) {
        OS << GenOpString(k, Proto, TypeVec[ti]);
      } else {
        if (R->getSuperClasses().size() < 2)
          throw TGError(R->getLoc(), "Builtin has no class kind");
        
        ClassKind ck = ClassMap[R->getSuperClasses()[1]];

        if (ck == ClassNone)
          throw TGError(R->getLoc(), "Builtin has no class kind");
        OS << GenBuiltin(name, Proto, TypeVec[ti], ck);
      }

      OS << " }\n";
    }
    OS << "\n";
  }

  // TODO: 
  // Unique the return+pattern types, and assign them to each record
  // Emit a #define for each unique "type" of intrinsic declaring all variants.
  // Emit a #define for each intrinsic mapping it to a particular type.
  
  OS << "#endif /* __ARM_NEON_H */\n";
}

void NeonEmitter::runHeader(raw_ostream &OS) {
}

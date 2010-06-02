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
  OpOrNot
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
      if (type == 'f')
        type = 'i';
      break;
    case 'f':
      type = 'f';
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
static std::string MangleName(const std::string &name, StringRef typestr) {
  bool quad = false;
  bool poly = false;
  bool usgn = false;
  char type = ClassifyType(typestr, quad, poly, usgn);

  std::string s = name;
  
  switch (type) {
    case 'c':
      s += poly ? "_p8" : usgn ? "_u8" : "_s8";
      break;
    case 's':
      s += poly ? "_p16" : usgn ? "_u16" : "_s16";
      break;
    case 'i':
      s += usgn ? "_u32" : "_s32";
      break;
    case 'l':
      s += usgn ? "_u64" : "_s64";
      break;
    case 'h':
      s += "_f16";
      break;
    case 'f':
      s += "_f32";
      break;
    default:
      throw "unhandled type!";
      break;
  }

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

static OpKind ParseOp(Record *R) {
  return OpNone;
}

// Generate the definition for this intrinsic, e.g. "a + b" for OpAdd.
// If structTypes is true, the NEON types are structs of vector types rather
// than vector types, and the call becomes "a.val + b.val"
static std::string GenOpString(OpKind op, const std::string &proto,
                               bool structTypes = true) {
  return "";
}

// Generate the definition for this intrinsic, e.g. __builtin_neon_cls(a)
// If structTypes is true, the NEON types are structs of vector types rather
// than vector types, and the call becomes __builtin_neon_cls(a.val)
static std::string GenBuiltin(const std::string &name, const std::string &proto,
                              StringRef typestr, bool structTypes = true) {
  char arg = 'a';
  std::string s("return ");
  
  // FIXME: if return type is 2/3/4, emit unioning code.
  
  if (structTypes) {
    s += "(";
    s += TypeString(proto[0], typestr);
    s += "){";
  }
  
  s += "__builtin_neon_";
  s += name;
  s += "(";
  
  for (unsigned i = 1, e = proto.size(); i != e; ++i, ++arg) {
    s.push_back(arg);
    if (structTypes)
      s += ".val";
    if ((i + 1) < e)
      s += ", ";
  }
  
  s += ")";
  if (structTypes)
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
  OS << "typedef float float32_t;\n";
  OS << "typedef uint8_t poly8_t;\n";
  OS << "typedef uint16_t poly16_t;\n";
  
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
    OS << TypeString('d', TDTypeVec[i]) << "\n";
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
  
  // Unique the return+pattern types, and assign them.
  for (unsigned i = 0, e = RV.size(); i != e; ++i) {
    Record *R = RV[i];
    std::string name = LowercaseString(R->getName());
    std::string Proto = R->getValueAsString("Prototype");
    std::string Types = R->getValueAsString("Types");
    
    SmallVector<StringRef, 16> TypeVec;
    ParseTypes(R, Types, TypeVec);
    
    OpKind k = ParseOp(R);
    
    for (unsigned ti = 0, te = TypeVec.size(); ti != te; ++ti) {
      assert(!Proto.empty() && "");
      
      // static always inline + return type
      OS << "__ai " << TypeString(Proto[0], TypeVec[ti]);
      
      // Function name with type suffix
      OS << " " << MangleName(name, TypeVec[ti]);
      
      // Function arguments
      OS << GenArgs(Proto, TypeVec[ti]);
      
      // Definition.
      OS << " { ";
      
      if (k != OpNone)
        OS << GenOpString(k, Proto);
      else
        OS << GenBuiltin(name, Proto, TypeVec[ti]);

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

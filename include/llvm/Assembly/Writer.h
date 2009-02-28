//===-- llvm/Assembly/Writer.h - Printer for LLVM assembly files --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This functionality is implemented by lib/VMCore/AsmWriter.cpp.
// This library is used to print LLVM assembly language files to an iostream. It
// can print LLVM code at a variety of granularities, including Modules,
// BasicBlocks, and Instructions.  This makes it useful for debugging.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_WRITER_H
#define LLVM_ASSEMBLY_WRITER_H

#include <iosfwd>

namespace llvm {

class Type;
class Module;
class Value;
class raw_ostream;
template <typename T> class SmallVectorImpl;
  
/// TypePrinting - Type printing machinery.
class TypePrinting {
  void *TypeNames;
public:
  TypePrinting(const Module *M);
  ~TypePrinting();
  
  void clear();
  
  void print(const Type *Ty, raw_ostream &OS);
  void printAtLeastOneLevel(const Type *Ty, raw_ostream &OS);
  
private:
  void CalcTypeName(const Type *Ty, SmallVectorImpl<const Type *> &TypeStack,
                    raw_ostream &OS);
};

// WriteTypeSymbolic - This attempts to write the specified type as a symbolic
// type, if there is an entry in the Module's symbol table for the specified
// type or one of its component types.
//
void WriteTypeSymbolic(raw_ostream &, const Type *, const Module *M);

// WriteAsOperand - Write the name of the specified value out to the specified
// ostream.  This can be useful when you just want to print int %reg126, not the
// whole instruction that generated it.  If you specify a Module for context,
// then even constants get pretty-printed; for example, the type of a null
// pointer is printed symbolically.
//
void WriteAsOperand(std::ostream &, const Value *, bool PrintTy = true,
                    const Module *Context = 0);
void WriteAsOperand(raw_ostream &, const Value *, bool PrintTy = true,
                    const Module *Context = 0);

} // End llvm namespace

#endif

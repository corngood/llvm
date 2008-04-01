//===- Parser.cpp - Main dispatch module for the Parser library -------------===
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library implements the functionality defined in llvm/assembly/parser.h
//
//===------------------------------------------------------------------------===

#include "ParserInternals.h"
#include "llvm/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>
using namespace llvm;


ParseError* TheParseError = 0; /// FIXME: Not threading friendly

Module *llvm::ParseAssemblyFile(const std::string &Filename, ParseError* Err) {
  std::string ErrorStr;
  MemoryBuffer *F = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), &ErrorStr);
  if (F == 0) {
    if (Err)
      Err->setError(Filename, "Could not open input file '" + Filename + "'");
    return 0;
  }
  
  TheParseError = Err;
  Module *Result = RunVMAsmParser(F);
  delete F;
  return Result;
}

Module *llvm::ParseAssemblyString(const char *AsmString, Module *M, 
                                  ParseError *Err) {
  TheParseError = Err;
  MemoryBuffer *F = MemoryBuffer::getMemBuffer(AsmString, 
                                               AsmString+strlen(AsmString),
                                               "<string>");
  Module *Result = RunVMAsmParser(F);
  delete F;
  return Result;
}


//===------------------------------------------------------------------------===
//                              ParseError Class
//===------------------------------------------------------------------------===


void ParseError::setError(const std::string &filename,
                          const std::string &message,
                          int lineNo, int colNo) {
  Filename = filename;
  Message = message;
  LineNo = lineNo;
  colNo = colNo;
}

ParseError::ParseError(const ParseError &E)
  : Filename(E.Filename), Message(E.Message) {
  LineNo = E.LineNo;
  ColumnNo = E.ColumnNo;
}

// Includes info from options
const std::string ParseError::getMessage() const {
  std::string Result;
  char Buffer[10];

  if (Filename == "-")
    Result += "<stdin>";
  else
    Result += Filename;

  if (LineNo != -1) {
    sprintf(Buffer, "%d", LineNo);
    Result += std::string(":") + Buffer;
    if (ColumnNo != -1) {
      sprintf(Buffer, "%d", ColumnNo);
      Result += std::string(",") + Buffer;
    }
  }

  return Result + ": " + Message;
}

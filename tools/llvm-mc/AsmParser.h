//===- AsmParser.h - Parser for Assembly Files ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class declares the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef ASMPARSER_H
#define ASMPARSER_H

#include "AsmLexer.h"
#include "llvm/MC/MCAsmParser.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class AsmExpr;
class MCContext;
class MCInst;
class MCStreamer;
class MCValue;
class TargetAsmParser;

class AsmParser : MCAsmParser {
public:
  struct X86Operand;

private:  
  AsmLexer Lexer;
  MCContext &Ctx;
  MCStreamer &Out;
  TargetAsmParser &TargetParser;
  
public:
  AsmParser(SourceMgr &_SM, MCContext &_Ctx, MCStreamer &_Out, 
            TargetAsmParser &_TargetParser)
    : Lexer(_SM), Ctx(_Ctx), Out(_Out), TargetParser(_TargetParser) {}
  ~AsmParser() {}
  
  bool Run();
  
public:
  TargetAsmParser &getTargetParser() const { return TargetParser; }

  virtual MCAsmLexer &getLexer() { return Lexer; }

private:
  bool ParseStatement();

  void Warning(SMLoc L, const char *Msg);
  bool Error(SMLoc L, const char *Msg);
  bool TokError(const char *Msg);
  
  void EatToEndOfStatement();
  
  bool ParseAssignment(const StringRef &Name, bool IsDotSet);

  /// ParseExpression - Parse a general assembly expression.
  ///
  /// @param Res - The resulting expression. The pointer value is null on error.
  /// @result - False on success.
  bool ParseExpression(AsmExpr *&Res);

  /// ParseAbsoluteExpression - Parse an expression which must evaluate to an
  /// absolute value.
  ///
  /// @param Res - The value of the absolute expression. The result is undefined
  /// on error.
  /// @result - False on success.
  bool ParseAbsoluteExpression(int64_t &Res);

  /// ParseRelocatableExpression - Parse an expression which must be
  /// relocatable.
  ///
  /// @param Res - The relocatable expression value. The result is undefined on
  /// error.  
  /// @result - False on success.
  bool ParseRelocatableExpression(MCValue &Res);

  /// ParseParenRelocatableExpression - Parse an expression which must be
  /// relocatable, assuming that an initial '(' has already been consumed.
  ///
  /// @param Res - The relocatable expression value. The result is undefined on
  /// error.  
  /// @result - False on success.
  ///
  /// @see ParseRelocatableExpression, ParseParenExpr.
  bool ParseParenRelocatableExpression(MCValue &Res);

  bool ParsePrimaryExpr(AsmExpr *&Res);
  bool ParseBinOpRHS(unsigned Precedence, AsmExpr *&Res);
  bool ParseParenExpr(AsmExpr *&Res);
  
  // X86 specific.
  bool ParseX86InstOperands(const StringRef &InstName, MCInst &Inst);
  bool ParseX86Operand(X86Operand &Op);
  bool ParseX86MemOperand(X86Operand &Op);
  bool ParseX86Register(X86Operand &Op);
  
  // Directive Parsing.
  bool ParseDirectiveDarwinSection(); // Darwin specific ".section".
  bool ParseDirectiveSectionSwitch(const char *Section,
                                   const char *Directives = 0);
  bool ParseDirectiveAscii(bool ZeroTerminated); // ".ascii", ".asciiz"
  bool ParseDirectiveValue(unsigned Size); // ".byte", ".long", ...
  bool ParseDirectiveFill(); // ".fill"
  bool ParseDirectiveSpace(); // ".space"
  bool ParseDirectiveSet(); // ".set"
  bool ParseDirectiveOrg(); // ".org"
  // ".align{,32}", ".p2align{,w,l}"
  bool ParseDirectiveAlign(bool IsPow2, unsigned ValueSize);

  /// ParseDirectiveSymbolAttribute - Parse a directive like ".globl" which
  /// accepts a single symbol (which should be a label or an external).
  bool ParseDirectiveSymbolAttribute(MCStreamer::SymbolAttr Attr);
  bool ParseDirectiveDarwinSymbolDesc(); // Darwin specific ".desc"
  bool ParseDirectiveDarwinLsym(); // Darwin specific ".lsym"

  bool ParseDirectiveComm(bool IsLocal); // ".comm" and ".lcomm"
  bool ParseDirectiveDarwinZerofill(); // Darwin specific ".zerofill"

  // Darwin specific ".subsections_via_symbols"
  bool ParseDirectiveDarwinSubsectionsViaSymbols();
  // Darwin specific .dump and .load
  bool ParseDirectiveDarwinDumpOrLoad(SMLoc IDLoc, bool IsDump);

  bool ParseDirectiveAbort(); // ".abort"
  bool ParseDirectiveInclude(); // ".include"
};

} // end namespace llvm

#endif

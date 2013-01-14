//===-- llvm/MC/MCAsmParser.h - Abstract Asm Parser Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCPARSER_MCASMPARSER_H
#define LLVM_MC_MCPARSER_MCASMPARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {
class MCAsmInfo;
class MCAsmLexer;
class MCAsmParserExtension;
class MCContext;
class MCExpr;
class MCInstPrinter;
class MCInstrInfo;
class MCParsedAsmOperand;
class MCStreamer;
class MCTargetAsmParser;
class SMLoc;
class SMRange;
class SourceMgr;
class StringRef;
class Twine;

/// MCAsmParserSemaCallback - Generic Sema callback for assembly parser.
class MCAsmParserSemaCallback {
public:
  virtual ~MCAsmParserSemaCallback(); 
  virtual void *LookupInlineAsmIdentifier(StringRef Name, void *Loc,
                                          unsigned &Size, bool &IsVarDecl) = 0;
  virtual bool LookupInlineAsmField(StringRef Base, StringRef Member,
                                    unsigned &Offset) = 0;
};


/// \brief Helper types for tracking macro definitions.
typedef std::vector<AsmToken> MCAsmMacroArgument;
typedef std::vector<MCAsmMacroArgument> MCAsmMacroArguments;
typedef std::pair<StringRef, MCAsmMacroArgument> MCAsmMacroParameter;
typedef std::vector<MCAsmMacroParameter> MCAsmMacroParameters;

struct MCAsmMacro {
  StringRef Name;
  StringRef Body;
  MCAsmMacroParameters Parameters;

public:
  MCAsmMacro(StringRef N, StringRef B, const MCAsmMacroParameters &P) :
    Name(N), Body(B), Parameters(P) {}

  MCAsmMacro(const MCAsmMacro& Other)
    : Name(Other.Name), Body(Other.Body), Parameters(Other.Parameters) {}
};

/// MCAsmParser - Generic assembler parser interface, for use by target specific
/// assembly parsers.
class MCAsmParser {
public:
  typedef bool (*DirectiveHandler)(MCAsmParserExtension*, StringRef, SMLoc);

private:
  MCAsmParser(const MCAsmParser &) LLVM_DELETED_FUNCTION;
  void operator=(const MCAsmParser &) LLVM_DELETED_FUNCTION;

  MCTargetAsmParser *TargetParser;

  unsigned ShowParsedOperands : 1;

protected: // Can only create subclasses.
  MCAsmParser();

public:
  virtual ~MCAsmParser();

  virtual void AddDirectiveHandler(MCAsmParserExtension *Object,
                                   StringRef Directive,
                                   DirectiveHandler Handler) = 0;

  virtual SourceMgr &getSourceManager() = 0;

  virtual MCAsmLexer &getLexer() = 0;

  virtual MCContext &getContext() = 0;

  /// getStreamer - Return the output streamer for the assembler.
  virtual MCStreamer &getStreamer() = 0;

  MCTargetAsmParser &getTargetParser() const { return *TargetParser; }
  void setTargetParser(MCTargetAsmParser &P);

  virtual unsigned getAssemblerDialect() { return 0;}
  virtual void setAssemblerDialect(unsigned i) { }

  bool getShowParsedOperands() const { return ShowParsedOperands; }
  void setShowParsedOperands(bool Value) { ShowParsedOperands = Value; }

  /// Run - Run the parser on the input source buffer.
  virtual bool Run(bool NoInitialTextSection, bool NoFinalize = false) = 0;

  virtual void setParsingInlineAsm(bool V) = 0;
  virtual bool isParsingInlineAsm() = 0;

  /// ParseMSInlineAsm - Parse ms-style inline assembly.
  virtual bool ParseMSInlineAsm(void *AsmLoc, std::string &AsmString,
                                unsigned &NumOutputs, unsigned &NumInputs,
                                SmallVectorImpl<std::pair<void *, bool> > &OpDecls,
                                SmallVectorImpl<std::string> &Constraints,
                                SmallVectorImpl<std::string> &Clobbers,
                                const MCInstrInfo *MII,
                                const MCInstPrinter *IP,
                                MCAsmParserSemaCallback &SI) = 0;

  /// Warning - Emit a warning at the location \p L, with the message \p Msg.
  ///
  /// \return The return value is true, if warnings are fatal.
  virtual bool Warning(SMLoc L, const Twine &Msg,
                       ArrayRef<SMRange> Ranges = ArrayRef<SMRange>()) = 0;

  /// Error - Emit an error at the location \p L, with the message \p Msg.
  ///
  /// \return The return value is always true, as an idiomatic convenience to
  /// clients.
  virtual bool Error(SMLoc L, const Twine &Msg,
                     ArrayRef<SMRange> Ranges = ArrayRef<SMRange>()) = 0;

  /// Lex - Get the next AsmToken in the stream, possibly handling file
  /// inclusion first.
  virtual const AsmToken &Lex() = 0;

  /// getTok - Get the current AsmToken from the stream.
  const AsmToken &getTok();

  /// \brief Report an error at the current lexer location.
  bool TokError(const Twine &Msg,
                ArrayRef<SMRange> Ranges = ArrayRef<SMRange>());

  /// ParseIdentifier - Parse an identifier or string (as a quoted identifier)
  /// and set \p Res to the identifier contents.
  virtual bool ParseIdentifier(StringRef &Res) = 0;

  /// \brief Parse up to the end of statement and return the contents from the
  /// current token until the end of the statement; the current token on exit
  /// will be either the EndOfStatement or EOF.
  virtual StringRef ParseStringToEndOfStatement() = 0;

  /// EatToEndOfStatement - Skip to the end of the current statement, for error
  /// recovery.
  virtual void EatToEndOfStatement() = 0;

  /// \brief Are macros enabled in the parser?
  virtual bool MacrosEnabled() = 0;

  /// \brief Control a flag in the parser that enables or disables macros.
  virtual void SetMacrosEnabled(bool flag) = 0;

  /// \brief Lookup a previously defined macro.
  /// \param Name Macro name.
  /// \returns Pointer to macro. NULL if no such macro was defined.
  virtual const MCAsmMacro* LookupMacro(StringRef Name) = 0;

  /// \brief Define a new macro with the given name and information.
  virtual void DefineMacro(StringRef Name, const MCAsmMacro& Macro) = 0;

  /// \brief Undefine a macro. If no such macro was defined, it's a no-op.
  virtual void UndefineMacro(StringRef Name) = 0;

  /// \brief Are we inside a macro instantiation?
  virtual bool InsideMacroInstantiation() = 0;

  /// \brief Handle entry to macro instantiation. 
  ///
  /// \param M The macro.
  /// \param NameLoc Instantiation location.
  virtual bool HandleMacroEntry(const MCAsmMacro *M, SMLoc NameLoc) = 0;

  /// \brief Handle exit from macro instantiation.
  virtual void HandleMacroExit() = 0;

  /// ParseMacroArgument - Extract AsmTokens for a macro argument. If the
  /// argument delimiter is initially unknown, set it to AsmToken::Eof. It will
  /// be set to the correct delimiter by the method.
  virtual bool ParseMacroArgument(MCAsmMacroArgument &MA,
                                  AsmToken::TokenKind &ArgumentDelimiter) = 0;

  /// ParseExpression - Parse an arbitrary expression.
  ///
  /// @param Res - The value of the expression. The result is undefined
  /// on error.
  /// @result - False on success.
  virtual bool ParseExpression(const MCExpr *&Res, SMLoc &EndLoc) = 0;
  bool ParseExpression(const MCExpr *&Res);

  /// ParseParenExpression - Parse an arbitrary expression, assuming that an
  /// initial '(' has already been consumed.
  ///
  /// @param Res - The value of the expression. The result is undefined
  /// on error.
  /// @result - False on success.
  virtual bool ParseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) = 0;

  /// ParseAbsoluteExpression - Parse an expression which must evaluate to an
  /// absolute value.
  ///
  /// @param Res - The value of the absolute expression. The result is undefined
  /// on error.
  /// @result - False on success.
  virtual bool ParseAbsoluteExpression(int64_t &Res) = 0;

  /// CheckForValidSection - Ensure that we have a valid section set in the
  /// streamer. Otherwise, report and error and switch to .text.
  virtual void CheckForValidSection() = 0;
};

/// \brief Create an MCAsmParser instance.
MCAsmParser *createMCAsmParser(SourceMgr &, MCContext &,
                               MCStreamer &, const MCAsmInfo &);

} // End llvm namespace

#endif

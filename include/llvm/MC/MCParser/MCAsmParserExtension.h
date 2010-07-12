//===-- llvm/MC/MCAsmParserExtension.h - Asm Parser Hooks -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMPARSEREXTENSION_H
#define LLVM_MC_MCASMPARSEREXTENSION_H

#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {

/// \brief Generic interface for extending the MCAsmParser,
/// which is implemented by target and object file assembly parser
/// implementations.
class MCAsmParserExtension {
  MCAsmParserExtension(const MCAsmParserExtension &);   // DO NOT IMPLEMENT
  void operator=(const MCAsmParserExtension &);  // DO NOT IMPLEMENT

  MCAsmParser *Parser;

protected:
  MCAsmParserExtension();

public:
  virtual ~MCAsmParserExtension();

  /// \brief Initialize the extension for parsing using the given \arg
  /// Parser. The extension should use the AsmParser interfaces to register its
  /// parsing routines.
  virtual void Initialize(MCAsmParser &Parser);

  /// @name MCAsmParser Proxy Interfaces
  /// @{

  MCContext &getContext() { return getParser().getContext(); }
  MCAsmLexer &getLexer() { return getParser().getLexer(); }
  MCAsmParser &getParser() { return *Parser; }
  MCStreamer &getStreamer() { return getParser().getStreamer(); }
  void Warning(SMLoc L, const Twine &Msg) {
    return getParser().Warning(L, Msg);
  }
  bool Error(SMLoc L, const Twine &Msg) {
    return getParser().Error(L, Msg);
  }

  const AsmToken &Lex() { return getParser().Lex(); }

  const AsmToken &getTok() { return getParser().getTok(); }

  bool TokError(const char *Msg) {
    return getParser().TokError(Msg);
  }

  /// @}
};

} // End llvm namespace

#endif

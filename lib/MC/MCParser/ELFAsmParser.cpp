//===- ELFAsmParser.cpp - ELF Assembly Parser -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
using namespace llvm;

namespace {

class ELFAsmParser : public MCAsmParserExtension {
  bool ParseSectionSwitch(StringRef Section, unsigned Type,
                          unsigned Flags, SectionKind Kind);

public:
  ELFAsmParser() {}

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    Parser.AddDirectiveHandler(this, ".data", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseSectionDirectiveData));
    Parser.AddDirectiveHandler(this, ".text", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseSectionDirectiveText));
    Parser.AddDirectiveHandler(this, ".section", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseDirectiveSection));
    Parser.AddDirectiveHandler(this, ".size", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseDirectiveSize));
    Parser.AddDirectiveHandler(this, ".sleb128", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseDirectiveLEB128));
    Parser.AddDirectiveHandler(this, ".uleb128", MCAsmParser::DirectiveHandler(
                                 &ELFAsmParser::ParseDirectiveLEB128));
  }

  bool ParseSectionDirectiveData(StringRef, SMLoc) {
    return ParseSectionSwitch(".data", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_WRITE |MCSectionELF::SHF_ALLOC,
                              SectionKind::getDataRel());
  }
  bool ParseSectionDirectiveText(StringRef, SMLoc) {
    return ParseSectionSwitch(".text", MCSectionELF::SHT_PROGBITS,
                              MCSectionELF::SHF_EXECINSTR |
                              MCSectionELF::SHF_ALLOC, SectionKind::getText());
  }
  bool ParseDirectiveLEB128(StringRef, SMLoc);
  bool ParseDirectiveSection(StringRef, SMLoc);
  bool ParseDirectiveSize(StringRef, SMLoc);
};

}

bool ELFAsmParser::ParseSectionSwitch(StringRef Section, unsigned Type,
                                      unsigned Flags, SectionKind Kind) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lex();

  getStreamer().SwitchSection(getContext().getELFSection(
                                Section, Type, Flags, Kind));

  return false;
}

bool ELFAsmParser::ParseDirectiveSize(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  const MCExpr *Expr;
  if (getParser().ParseExpression(Expr))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  getStreamer().EmitELFSize(Sym, Expr);
  return false;
}

// FIXME: This is a work in progress.
bool ELFAsmParser::ParseDirectiveSection(StringRef, SMLoc) {
  StringRef SectionName;
  // FIXME: This doesn't parse section names like ".note.GNU-stack" correctly.
  if (getParser().ParseIdentifier(SectionName))
    return TokError("expected identifier in directive");

  std::string FlagsStr;
  StringRef TypeName;
  int64_t Size = 0;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (getLexer().isNot(AsmToken::String))
      return TokError("expected string in directive");

    FlagsStr = getTok().getStringContents();
    Lex();

    AsmToken::TokenKind TypeStartToken;
    if (getContext().getAsmInfo().getCommentString()[0] == '@')
      TypeStartToken = AsmToken::Percent;
    else
      TypeStartToken = AsmToken::At;

    if (getLexer().is(AsmToken::Comma)) {
      Lex();
      if (getLexer().is(TypeStartToken)) {
        Lex();
        if (getParser().ParseIdentifier(TypeName))
          return TokError("expected identifier in directive");

        if (getLexer().is(AsmToken::Comma)) {
          Lex();

          if (getParser().ParseAbsoluteExpression(Size))
            return true;

          if (Size <= 0)
            return TokError("section size must be positive");
        }
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  unsigned Flags = 0;
  for (unsigned i = 0; i < FlagsStr.size(); i++) {
    switch (FlagsStr[i]) {
    case 'a':
      Flags |= MCSectionELF::SHF_ALLOC;
      break;
    case 'x':
      Flags |= MCSectionELF::SHF_EXECINSTR;
      break;
    case 'w':
      Flags |= MCSectionELF::SHF_WRITE;
      break;
    case 'M':
      Flags |= MCSectionELF::SHF_MERGE;
      break;
    case 'S':
      Flags |= MCSectionELF::SHF_STRINGS;
      break;
    case 'T':
      Flags |= MCSectionELF::SHF_TLS;
      break;
    case 'c':
      Flags |= MCSectionELF::XCORE_SHF_CP_SECTION;
      break;
    case 'd':
      Flags |= MCSectionELF::XCORE_SHF_DP_SECTION;
      break;
    default:
      return TokError("unknown flag");
    }
  }

  unsigned Type = MCSectionELF::SHT_NULL;
  if (!TypeName.empty()) {
    if (TypeName == "init_array")
      Type = MCSectionELF::SHT_INIT_ARRAY;
    else if (TypeName == "fini_array")
      Type = MCSectionELF::SHT_FINI_ARRAY;
    else if (TypeName == "preinit_array")
      Type = MCSectionELF::SHT_PREINIT_ARRAY;
    else if (TypeName == "nobits")
      Type = MCSectionELF::SHT_NOBITS;
    else if (TypeName == "progbits")
      Type = MCSectionELF::SHT_PROGBITS;
    else
      return TokError("unknown section type");
  }

  SectionKind Kind = (Flags & MCSectionELF::SHF_EXECINSTR)
                     ? SectionKind::getText()
                     : SectionKind::getDataRel();
  getStreamer().SwitchSection(getContext().getELFSection(SectionName, Type,
                                                         Flags, Kind, false));
  return false;
}

bool ELFAsmParser::ParseDirectiveLEB128(StringRef DirName, SMLoc) {
  int64_t Value;
  if (getParser().ParseAbsoluteExpression(Value))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  // FIXME: Add proper MC support.
  if (getContext().getAsmInfo().hasLEB128()) {
    if (DirName[1] == 's')
      getStreamer().EmitRawText("\t.sleb128\t" + Twine(Value));
    else
      getStreamer().EmitRawText("\t.uleb128\t" + Twine(Value));
    return false;
  }
  // FIXME: This shouldn't be an error!
  return TokError("LEB128 not supported yet");
}

namespace llvm {

MCAsmParserExtension *createELFAsmParser() {
  return new ELFAsmParser;
}

}

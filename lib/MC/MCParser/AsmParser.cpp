//===- AsmParser.cpp - Parser for Assembly Files --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/AsmParser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmParser.h"
using namespace llvm;


namespace {

/// \brief Generic implementations of directive handling, etc. which is shared
/// (or the default, at least) for all assembler parser.
class GenericAsmParser : public MCAsmParserExtension {
public:
  GenericAsmParser() {}

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    // Debugging directives.
    Parser.AddDirectiveHandler(this, ".file", MCAsmParser::DirectiveHandler(
                                 &GenericAsmParser::ParseDirectiveFile));
    Parser.AddDirectiveHandler(this, ".line", MCAsmParser::DirectiveHandler(
                                 &GenericAsmParser::ParseDirectiveLine));
    Parser.AddDirectiveHandler(this, ".loc", MCAsmParser::DirectiveHandler(
                                 &GenericAsmParser::ParseDirectiveLoc));
  }

  bool ParseDirectiveFile(StringRef, SMLoc DirectiveLoc); // ".file"
  bool ParseDirectiveLine(StringRef, SMLoc DirectiveLoc); // ".line"
  bool ParseDirectiveLoc(StringRef, SMLoc DirectiveLoc); // ".loc"
};

/// \brief Implementation of directive handling which is shared across all
/// Darwin targets.
class DarwinAsmParser : public MCAsmParserExtension {
  bool ParseSectionSwitch(const char *Segment, const char *Section,
                          unsigned TAA = 0, unsigned ImplicitAlign = 0,
                          unsigned StubSize = 0);

public:
  DarwinAsmParser() {}

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    Parser.AddDirectiveHandler(this, ".desc", MCAsmParser::DirectiveHandler(
                                 &DarwinAsmParser::ParseDirectiveDesc));
    Parser.AddDirectiveHandler(this, ".lsym", MCAsmParser::DirectiveHandler(
                                 &DarwinAsmParser::ParseDirectiveLsym));
    Parser.AddDirectiveHandler(this, ".subsections_via_symbols",
                               MCAsmParser::DirectiveHandler(
                        &DarwinAsmParser::ParseDirectiveSubsectionsViaSymbols));
    Parser.AddDirectiveHandler(this, ".dump", MCAsmParser::DirectiveHandler(
                                 &DarwinAsmParser::ParseDirectiveDumpOrLoad));
    Parser.AddDirectiveHandler(this, ".load", MCAsmParser::DirectiveHandler(
                                 &DarwinAsmParser::ParseDirectiveDumpOrLoad));
    Parser.AddDirectiveHandler(this, ".secure_log_unique",
                               MCAsmParser::DirectiveHandler(
                             &DarwinAsmParser::ParseDirectiveSecureLogUnique));
    Parser.AddDirectiveHandler(this, ".secure_log_reset",
                               MCAsmParser::DirectiveHandler(
                             &DarwinAsmParser::ParseDirectiveSecureLogReset));
    Parser.AddDirectiveHandler(this, ".tbss",
                               MCAsmParser::DirectiveHandler(
                                 &DarwinAsmParser::ParseDirectiveTBSS));
    Parser.AddDirectiveHandler(this, ".zerofill",
                               MCAsmParser::DirectiveHandler(
                                 &DarwinAsmParser::ParseDirectiveZerofill));

    // Special section directives.
    Parser.AddDirectiveHandler(this, ".const",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveConst));
    Parser.AddDirectiveHandler(this, ".const_data",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveConstData));
    Parser.AddDirectiveHandler(this, ".constructor",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveConstructor));
    Parser.AddDirectiveHandler(this, ".cstring",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveCString));
    Parser.AddDirectiveHandler(this, ".data",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveData));
    Parser.AddDirectiveHandler(this, ".destructor",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveDestructor));
    Parser.AddDirectiveHandler(this, ".dyld",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveDyld));
    Parser.AddDirectiveHandler(this, ".fvmlib_init0",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveFVMLibInit0));
    Parser.AddDirectiveHandler(this, ".fvmlib_init1",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveFVMLibInit1));
    Parser.AddDirectiveHandler(this, ".lazy_symbol_pointer",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveLazySymbolPointers));
    Parser.AddDirectiveHandler(this, ".literal16",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveLiteral16));
    Parser.AddDirectiveHandler(this, ".literal4",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveLiteral4));
    Parser.AddDirectiveHandler(this, ".literal8",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveLiteral8));
    Parser.AddDirectiveHandler(this, ".mod_init_func",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveModInitFunc));
    Parser.AddDirectiveHandler(this, ".mod_term_func",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveModTermFunc));
    Parser.AddDirectiveHandler(this, ".non_lazy_symbol_pointer",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveNonLazySymbolPointers));
    Parser.AddDirectiveHandler(this, ".objc_cat_cls_meth",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCCatClsMeth));
    Parser.AddDirectiveHandler(this, ".objc_cat_inst_meth",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCCatInstMeth));
    Parser.AddDirectiveHandler(this, ".objc_category",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCCategory));
    Parser.AddDirectiveHandler(this, ".objc_class",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCClass));
    Parser.AddDirectiveHandler(this, ".objc_class_names",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCClassNames));
    Parser.AddDirectiveHandler(this, ".objc_class_vars",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCClassVars));
    Parser.AddDirectiveHandler(this, ".objc_cls_meth",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCClsMeth));
    Parser.AddDirectiveHandler(this, ".objc_cls_refs",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCClsRefs));
    Parser.AddDirectiveHandler(this, ".objc_inst_meth",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCInstMeth));
    Parser.AddDirectiveHandler(this, ".objc_instance_vars",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCInstanceVars));
    Parser.AddDirectiveHandler(this, ".objc_message_refs",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCMessageRefs));
    Parser.AddDirectiveHandler(this, ".objc_meta_class",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCMetaClass));
    Parser.AddDirectiveHandler(this, ".objc_meth_var_names",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCMethVarNames));
    Parser.AddDirectiveHandler(this, ".objc_meth_var_types",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCMethVarTypes));
    Parser.AddDirectiveHandler(this, ".objc_module_info",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCModuleInfo));
    Parser.AddDirectiveHandler(this, ".objc_protocol",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCProtocol));
    Parser.AddDirectiveHandler(this, ".objc_selector_strs",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCSelectorStrs));
    Parser.AddDirectiveHandler(this, ".objc_string_object",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCStringObject));
    Parser.AddDirectiveHandler(this, ".objc_symbols",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveObjCSymbols));
    Parser.AddDirectiveHandler(this, ".picsymbol_stub",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectivePICSymbolStub));
    Parser.AddDirectiveHandler(this, ".static_const",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveStaticConst));
    Parser.AddDirectiveHandler(this, ".static_data",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveStaticData));
    Parser.AddDirectiveHandler(this, ".symbol_stub",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveSymbolStub));
    Parser.AddDirectiveHandler(this, ".tdata",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveTData));
    Parser.AddDirectiveHandler(this, ".text",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveText));
    Parser.AddDirectiveHandler(this, ".thread_init_func",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveThreadInitFunc));
    Parser.AddDirectiveHandler(this, ".tlv",
                               MCAsmParser::DirectiveHandler(
                 &DarwinAsmParser::ParseSectionDirectiveTLV));
  }

  bool ParseDirectiveDesc(StringRef, SMLoc);
  bool ParseDirectiveDumpOrLoad(StringRef, SMLoc);
  bool ParseDirectiveLsym(StringRef, SMLoc);
  bool ParseDirectiveSecureLogReset(StringRef, SMLoc);
  bool ParseDirectiveSecureLogUnique(StringRef, SMLoc);
  bool ParseDirectiveSubsectionsViaSymbols(StringRef, SMLoc);
  bool ParseDirectiveTBSS(StringRef, SMLoc);
  bool ParseDirectiveZerofill(StringRef, SMLoc);

  // Named Section Directive
  bool ParseSectionDirectiveConst(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__const");
  }
  bool ParseSectionDirectiveStaticConst(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__static_const");
  }
  bool ParseSectionDirectiveCString(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveLiteral4(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__literal4",
                              MCSectionMachO::S_4BYTE_LITERALS, 4);
  }
  bool ParseSectionDirectiveLiteral8(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__literal8",
                              MCSectionMachO::S_8BYTE_LITERALS, 8);
  }
  bool ParseSectionDirectiveLiteral16(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__literal16",
                              MCSectionMachO::S_16BYTE_LITERALS, 16);
  }
  bool ParseSectionDirectiveConstructor(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__constructor");
  }
  bool ParseSectionDirectiveDestructor(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__destructor");
  }
  bool ParseSectionDirectiveFVMLibInit0(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__fvmlib_init0");
  }
  bool ParseSectionDirectiveFVMLibInit1(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__fvmlib_init1");
  }
  bool ParseSectionDirectiveSymbolStub(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__symbol_stub",
                              MCSectionMachO::S_SYMBOL_STUBS |
                              MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                              // FIXME: Different on PPC and ARM.
                              0, 16);
  }
  bool ParseSectionDirectivePICSymbolStub(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__picsymbol_stub",
                              MCSectionMachO::S_SYMBOL_STUBS |
                              MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS, 0, 26);
  }
  bool ParseSectionDirectiveData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__data");
  }
  bool ParseSectionDirectiveStaticData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__static_data");
  }
  bool ParseSectionDirectiveNonLazySymbolPointers(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__nl_symbol_ptr",
                              MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS, 4);
  }
  bool ParseSectionDirectiveLazySymbolPointers(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__la_symbol_ptr",
                              MCSectionMachO::S_LAZY_SYMBOL_POINTERS, 4);
  }
  bool ParseSectionDirectiveDyld(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__dyld");
  }
  bool ParseSectionDirectiveModInitFunc(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__mod_init_func",
                              MCSectionMachO::S_MOD_INIT_FUNC_POINTERS, 4);
  }
  bool ParseSectionDirectiveModTermFunc(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__mod_term_func",
                              MCSectionMachO::S_MOD_TERM_FUNC_POINTERS, 4);
  }
  bool ParseSectionDirectiveConstData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__const");
  }
  bool ParseSectionDirectiveObjCClass(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__class",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCMetaClass(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__meta_class",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCCatClsMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cat_cls_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCCatInstMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cat_inst_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCProtocol(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__protocol",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCStringObject(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__string_object",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClsMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cls_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCInstMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__inst_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClsRefs(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cls_refs",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP |
                              MCSectionMachO::S_LITERAL_POINTERS, 4);
  }
  bool ParseSectionDirectiveObjCMessageRefs(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__message_refs",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP |
                              MCSectionMachO::S_LITERAL_POINTERS, 4);
  }
  bool ParseSectionDirectiveObjCSymbols(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__symbols",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCCategory(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__category",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClassVars(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__class_vars",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCInstanceVars(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__instance_vars",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCModuleInfo(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__module_info",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClassNames(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveObjCMethVarTypes(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveObjCMethVarNames(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveObjCSelectorStrs(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__selector_strs",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveTData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__thread_data",
                              MCSectionMachO::S_THREAD_LOCAL_REGULAR);
  }
  bool ParseSectionDirectiveText(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__text",
                              MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS);
  }
  bool ParseSectionDirectiveTLV(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__thread_vars",
                              MCSectionMachO::S_THREAD_LOCAL_VARIABLES);
  }
  bool ParseSectionDirectiveThreadInitFunc(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__thread_init",
                         MCSectionMachO::S_THREAD_LOCAL_INIT_FUNCTION_POINTERS);
  }

};

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
};

}

enum { DEFAULT_ADDRSPACE = 0 };

AsmParser::AsmParser(const Target &T, SourceMgr &_SM, MCContext &_Ctx,
                     MCStreamer &_Out, const MCAsmInfo &_MAI)
  : Lexer(_MAI), Ctx(_Ctx), Out(_Out), SrcMgr(_SM),
    GenericParser(new GenericAsmParser), PlatformParser(0),
    TargetParser(0), CurBuffer(0) {
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer));

  // Initialize the generic parser.
  GenericParser->Initialize(*this);

  // Initialize the platform / file format parser.
  //
  // FIXME: This is a hack, we need to (majorly) cleanup how these objects are
  // created.
  if (_MAI.hasSubsectionsViaSymbols()) {
    PlatformParser = new DarwinAsmParser;
    PlatformParser->Initialize(*this);
  } else {
    PlatformParser = new ELFAsmParser;
    PlatformParser->Initialize(*this);
  }
}

AsmParser::~AsmParser() {
  delete PlatformParser;
  delete GenericParser;
}

void AsmParser::setTargetParser(TargetAsmParser &P) {
  assert(!TargetParser && "Target parser is already initialized!");
  TargetParser = &P;
  TargetParser->Initialize(*this);
}

void AsmParser::Warning(SMLoc L, const Twine &Msg) {
  PrintMessage(L, Msg.str(), "warning");
}

bool AsmParser::Error(SMLoc L, const Twine &Msg) {
  PrintMessage(L, Msg.str(), "error");
  return true;
}

void AsmParser::PrintMessage(SMLoc Loc, const std::string &Msg, 
                             const char *Type) const {
  SrcMgr.PrintMessage(Loc, Msg, Type);
}
                  
bool AsmParser::EnterIncludeFile(const std::string &Filename) {
  int NewBuf = SrcMgr.AddIncludeFile(Filename, Lexer.getLoc());
  if (NewBuf == -1)
    return true;
  
  CurBuffer = NewBuf;
  
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer));
  
  return false;
}
                  
const AsmToken &AsmParser::Lex() {
  const AsmToken *tok = &Lexer.Lex();
  
  if (tok->is(AsmToken::Eof)) {
    // If this is the end of an included file, pop the parent file off the
    // include stack.
    SMLoc ParentIncludeLoc = SrcMgr.getParentIncludeLoc(CurBuffer);
    if (ParentIncludeLoc != SMLoc()) {
      CurBuffer = SrcMgr.FindBufferContainingLoc(ParentIncludeLoc);
      Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer), 
                      ParentIncludeLoc.getPointer());
      tok = &Lexer.Lex();
    }
  }
    
  if (tok->is(AsmToken::Error))
    PrintMessage(Lexer.getErrLoc(), Lexer.getErr(), "error");
  
  return *tok;
}

bool AsmParser::Run(bool NoInitialTextSection, bool NoFinalize) {
  // Create the initial section, if requested.
  //
  // FIXME: Target hook & command line option for initial section.
  if (!NoInitialTextSection)
    Out.SwitchSection(Ctx.getMachOSection("__TEXT", "__text",
                                      MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                      0, SectionKind::getText()));

  // Prime the lexer.
  Lex();
  
  bool HadError = false;
  
  AsmCond StartingCondState = TheCondState;

  // While we have input, parse each statement.
  while (Lexer.isNot(AsmToken::Eof)) {
    if (!ParseStatement()) continue;
  
    // We had an error, remember it and recover by skipping to the next line.
    HadError = true;
    EatToEndOfStatement();
  }

  if (TheCondState.TheCond != StartingCondState.TheCond ||
      TheCondState.Ignore != StartingCondState.Ignore)
    return TokError("unmatched .ifs or .elses");
  
  // Finalize the output stream if there are no errors and if the client wants
  // us to.
  if (!HadError && !NoFinalize)  
    Out.Finish();

  return HadError;
}

/// EatToEndOfStatement - Throw away the rest of the line for testing purposes.
void AsmParser::EatToEndOfStatement() {
  while (Lexer.isNot(AsmToken::EndOfStatement) &&
         Lexer.isNot(AsmToken::Eof))
    Lex();
  
  // Eat EOL.
  if (Lexer.is(AsmToken::EndOfStatement))
    Lex();
}


/// ParseParenExpr - Parse a paren expression and return it.
/// NOTE: This assumes the leading '(' has already been consumed.
///
/// parenexpr ::= expr)
///
bool AsmParser::ParseParenExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (ParseExpression(Res)) return true;
  if (Lexer.isNot(AsmToken::RParen))
    return TokError("expected ')' in parentheses expression");
  EndLoc = Lexer.getLoc();
  Lex();
  return false;
}

/// ParsePrimaryExpr - Parse a primary expression and return it.
///  primaryexpr ::= (parenexpr
///  primaryexpr ::= symbol
///  primaryexpr ::= number
///  primaryexpr ::= '.'
///  primaryexpr ::= ~,+,- primaryexpr
bool AsmParser::ParsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token in expression");
  case AsmToken::Exclaim:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreateLNot(Res, getContext());
    return false;
  case AsmToken::String:
  case AsmToken::Identifier: {
    // This is a symbol reference.
    std::pair<StringRef, StringRef> Split = getTok().getIdentifier().split('@');
    MCSymbol *Sym = getContext().GetOrCreateSymbol(Split.first);

    // Mark the symbol as used in an expression.
    Sym->setUsedInExpr(true);

    // Lookup the symbol variant if used.
    MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
    if (Split.first.size() != getTok().getIdentifier().size())
      Variant = MCSymbolRefExpr::getVariantKindForName(Split.second);

    EndLoc = Lexer.getLoc();
    Lex(); // Eat identifier.

    // If this is an absolute variable reference, substitute it now to preserve
    // semantics in the face of reassignment.
    if (Sym->isVariable() && isa<MCConstantExpr>(Sym->getVariableValue())) {
      if (Variant)
        return Error(EndLoc, "unexpected modified on variable reference");

      Res = Sym->getVariableValue();
      return false;
    }

    // Otherwise create a symbol ref.
    Res = MCSymbolRefExpr::Create(Sym, Variant, getContext());
    return false;
  }
  case AsmToken::Integer: {
    SMLoc Loc = getTok().getLoc();
    int64_t IntVal = getTok().getIntVal();
    Res = MCConstantExpr::Create(IntVal, getContext());
    EndLoc = Lexer.getLoc();
    Lex(); // Eat token.
    // Look for 'b' or 'f' following an Integer as a directional label
    if (Lexer.getKind() == AsmToken::Identifier) {
      StringRef IDVal = getTok().getString();
      if (IDVal == "f" || IDVal == "b"){
        MCSymbol *Sym = Ctx.GetDirectionalLocalSymbol(IntVal,
                                                      IDVal == "f" ? 1 : 0);
        Res = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None,
                                      getContext());
        if(IDVal == "b" && Sym->isUndefined())
          return Error(Loc, "invalid reference to undefined symbol");
        EndLoc = Lexer.getLoc();
        Lex(); // Eat identifier.
      }
    }
    return false;
  }
  case AsmToken::Dot: {
    // This is a '.' reference, which references the current PC.  Emit a
    // temporary label to the streamer and refer to it.
    MCSymbol *Sym = Ctx.CreateTempSymbol();
    Out.EmitLabel(Sym);
    Res = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, getContext());
    EndLoc = Lexer.getLoc();
    Lex(); // Eat identifier.
    return false;
  }
      
  case AsmToken::LParen:
    Lex(); // Eat the '('.
    return ParseParenExpr(Res, EndLoc);
  case AsmToken::Minus:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreateMinus(Res, getContext());
    return false;
  case AsmToken::Plus:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreatePlus(Res, getContext());
    return false;
  case AsmToken::Tilde:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreateNot(Res, getContext());
    return false;
  }
}

bool AsmParser::ParseExpression(const MCExpr *&Res) {
  SMLoc EndLoc;
  return ParseExpression(Res, EndLoc);
}

/// ParseExpression - Parse an expression and return it.
/// 
///  expr ::= expr +,- expr          -> lowest.
///  expr ::= expr |,^,&,! expr      -> middle.
///  expr ::= expr *,/,%,<<,>> expr  -> highest.
///  expr ::= primaryexpr
///
bool AsmParser::ParseExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  // Parse the expression.
  Res = 0;
  if (ParsePrimaryExpr(Res, EndLoc) || ParseBinOpRHS(1, Res, EndLoc))
    return true;

  // Try to constant fold it up front, if possible.
  int64_t Value;
  if (Res->EvaluateAsAbsolute(Value))
    Res = MCConstantExpr::Create(Value, getContext());

  return false;
}

bool AsmParser::ParseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  Res = 0;
  return ParseParenExpr(Res, EndLoc) ||
         ParseBinOpRHS(1, Res, EndLoc);
}

bool AsmParser::ParseAbsoluteExpression(int64_t &Res) {
  const MCExpr *Expr;
  
  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Expr))
    return true;

  if (!Expr->EvaluateAsAbsolute(Res))
    return Error(StartLoc, "expected absolute expression");

  return false;
}

static unsigned getBinOpPrecedence(AsmToken::TokenKind K, 
                                   MCBinaryExpr::Opcode &Kind) {
  switch (K) {
  default:
    return 0;    // not a binop.

    // Lowest Precedence: &&, ||
  case AsmToken::AmpAmp:
    Kind = MCBinaryExpr::LAnd;
    return 1;
  case AsmToken::PipePipe:
    Kind = MCBinaryExpr::LOr;
    return 1;

    // Low Precedence: +, -, ==, !=, <>, <, <=, >, >=
  case AsmToken::Plus:
    Kind = MCBinaryExpr::Add;
    return 2;
  case AsmToken::Minus:
    Kind = MCBinaryExpr::Sub;
    return 2;
  case AsmToken::EqualEqual:
    Kind = MCBinaryExpr::EQ;
    return 2;
  case AsmToken::ExclaimEqual:
  case AsmToken::LessGreater:
    Kind = MCBinaryExpr::NE;
    return 2;
  case AsmToken::Less:
    Kind = MCBinaryExpr::LT;
    return 2;
  case AsmToken::LessEqual:
    Kind = MCBinaryExpr::LTE;
    return 2;
  case AsmToken::Greater:
    Kind = MCBinaryExpr::GT;
    return 2;
  case AsmToken::GreaterEqual:
    Kind = MCBinaryExpr::GTE;
    return 2;

    // Intermediate Precedence: |, &, ^
    //
    // FIXME: gas seems to support '!' as an infix operator?
  case AsmToken::Pipe:
    Kind = MCBinaryExpr::Or;
    return 3;
  case AsmToken::Caret:
    Kind = MCBinaryExpr::Xor;
    return 3;
  case AsmToken::Amp:
    Kind = MCBinaryExpr::And;
    return 3;

    // Highest Precedence: *, /, %, <<, >>
  case AsmToken::Star:
    Kind = MCBinaryExpr::Mul;
    return 4;
  case AsmToken::Slash:
    Kind = MCBinaryExpr::Div;
    return 4;
  case AsmToken::Percent:
    Kind = MCBinaryExpr::Mod;
    return 4;
  case AsmToken::LessLess:
    Kind = MCBinaryExpr::Shl;
    return 4;
  case AsmToken::GreaterGreater:
    Kind = MCBinaryExpr::Shr;
    return 4;
  }
}


/// ParseBinOpRHS - Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool AsmParser::ParseBinOpRHS(unsigned Precedence, const MCExpr *&Res,
                              SMLoc &EndLoc) {
  while (1) {
    MCBinaryExpr::Opcode Kind = MCBinaryExpr::Add;
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind(), Kind);
    
    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;
    
    Lex();
    
    // Eat the next primary expression.
    const MCExpr *RHS;
    if (ParsePrimaryExpr(RHS, EndLoc)) return true;
    
    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    MCBinaryExpr::Opcode Dummy;
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind(), Dummy);
    if (TokPrec < NextTokPrec) {
      if (ParseBinOpRHS(Precedence+1, RHS, EndLoc)) return true;
    }

    // Merge LHS and RHS according to operator.
    Res = MCBinaryExpr::Create(Kind, Res, RHS, getContext());
  }
}

  
  
  
/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool AsmParser::ParseStatement() {
  if (Lexer.is(AsmToken::EndOfStatement)) {
    Out.AddBlankLine();
    Lex();
    return false;
  }

  // Statements always start with an identifier.
  AsmToken ID = getTok();
  SMLoc IDLoc = ID.getLoc();
  StringRef IDVal;
  int64_t LocalLabelVal = -1;
  // GUESS allow an integer followed by a ':' as a directional local label
  if (Lexer.is(AsmToken::Integer)) {
    LocalLabelVal = getTok().getIntVal();
    if (LocalLabelVal < 0) {
      if (!TheCondState.Ignore)
        return TokError("unexpected token at start of statement");
      IDVal = "";
    }
    else {
      IDVal = getTok().getString();
      Lex(); // Consume the integer token to be used as an identifier token.
      if (Lexer.getKind() != AsmToken::Colon) {
        if (!TheCondState.Ignore)
          return TokError("unexpected token at start of statement");
      }
    }
  }
  else if (ParseIdentifier(IDVal)) {
    if (!TheCondState.Ignore)
      return TokError("unexpected token at start of statement");
    IDVal = "";
  }

  // Handle conditional assembly here before checking for skipping.  We
  // have to do this so that .endif isn't skipped in a ".if 0" block for
  // example.
  if (IDVal == ".if")
    return ParseDirectiveIf(IDLoc);
  if (IDVal == ".elseif")
    return ParseDirectiveElseIf(IDLoc);
  if (IDVal == ".else")
    return ParseDirectiveElse(IDLoc);
  if (IDVal == ".endif")
    return ParseDirectiveEndIf(IDLoc);
    
  // If we are in a ".if 0" block, ignore this statement.
  if (TheCondState.Ignore) {
    EatToEndOfStatement();
    return false;
  }
  
  // FIXME: Recurse on local labels?

  // See what kind of statement we have.
  switch (Lexer.getKind()) {
  case AsmToken::Colon: {
    // identifier ':'   -> Label.
    Lex();

    // Diagnose attempt to use a variable as a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: This doesn't diagnose assignment to a symbol which has been
    // implicitly marked as external.
    MCSymbol *Sym;
    if (LocalLabelVal == -1)
      Sym = getContext().GetOrCreateSymbol(IDVal);
    else
      Sym = Ctx.CreateDirectionalLocalSymbol(LocalLabelVal);
    if (!Sym->isUndefined() || Sym->isVariable())
      return Error(IDLoc, "invalid symbol redefinition");
    
    // Emit the label.
    Out.EmitLabel(Sym);
   
    // Consume any end of statement token, if present, to avoid spurious
    // AddBlankLine calls().
    if (Lexer.is(AsmToken::EndOfStatement)) {
      Lex();
      if (Lexer.is(AsmToken::Eof))
        return false;
    }

    return ParseStatement();
  }

  case AsmToken::Equal:
    // identifier '=' ... -> assignment statement
    Lex();

    return ParseAssignment(IDVal);

  default: // Normal instruction or directive.
    break;
  }
  
  // Otherwise, we have a normal instruction or directive.  
  if (IDVal[0] == '.') {
    // FIXME: This should be driven based on a hash lookup and callback.
    if (IDVal == ".section")
      return ParseDirectiveDarwinSection();

    // Assembler features
    if (IDVal == ".set")
      return ParseDirectiveSet();

    // Data directives

    if (IDVal == ".ascii")
      return ParseDirectiveAscii(false);
    if (IDVal == ".asciz")
      return ParseDirectiveAscii(true);

    if (IDVal == ".byte")
      return ParseDirectiveValue(1);
    if (IDVal == ".short")
      return ParseDirectiveValue(2);
    if (IDVal == ".long")
      return ParseDirectiveValue(4);
    if (IDVal == ".quad")
      return ParseDirectiveValue(8);

    // FIXME: Target hooks for IsPow2.
    if (IDVal == ".align")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    if (IDVal == ".align32")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);
    if (IDVal == ".balign")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/1);
    if (IDVal == ".balignw")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/2);
    if (IDVal == ".balignl")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/4);
    if (IDVal == ".p2align")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    if (IDVal == ".p2alignw")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/2);
    if (IDVal == ".p2alignl")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);

    if (IDVal == ".org")
      return ParseDirectiveOrg();

    if (IDVal == ".fill")
      return ParseDirectiveFill();
    if (IDVal == ".space")
      return ParseDirectiveSpace();

    // Symbol attribute directives

    if (IDVal == ".globl" || IDVal == ".global")
      return ParseDirectiveSymbolAttribute(MCSA_Global);
    if (IDVal == ".hidden")
      return ParseDirectiveSymbolAttribute(MCSA_Hidden);
    if (IDVal == ".indirect_symbol")
      return ParseDirectiveSymbolAttribute(MCSA_IndirectSymbol);
    if (IDVal == ".internal")
      return ParseDirectiveSymbolAttribute(MCSA_Internal);
    if (IDVal == ".lazy_reference")
      return ParseDirectiveSymbolAttribute(MCSA_LazyReference);
    if (IDVal == ".no_dead_strip")
      return ParseDirectiveSymbolAttribute(MCSA_NoDeadStrip);
    if (IDVal == ".private_extern")
      return ParseDirectiveSymbolAttribute(MCSA_PrivateExtern);
    if (IDVal == ".protected")
      return ParseDirectiveSymbolAttribute(MCSA_Protected);
    if (IDVal == ".reference")
      return ParseDirectiveSymbolAttribute(MCSA_Reference);
    if (IDVal == ".type")
      return ParseDirectiveELFType();
    if (IDVal == ".weak")
      return ParseDirectiveSymbolAttribute(MCSA_Weak);
    if (IDVal == ".weak_definition")
      return ParseDirectiveSymbolAttribute(MCSA_WeakDefinition);
    if (IDVal == ".weak_reference")
      return ParseDirectiveSymbolAttribute(MCSA_WeakReference);
    if (IDVal == ".weak_def_can_be_hidden")
      return ParseDirectiveSymbolAttribute(MCSA_WeakDefAutoPrivate);

    if (IDVal == ".comm")
      return ParseDirectiveComm(/*IsLocal=*/false);
    if (IDVal == ".lcomm")
      return ParseDirectiveComm(/*IsLocal=*/true);

    if (IDVal == ".abort")
      return ParseDirectiveAbort();
    if (IDVal == ".include")
      return ParseDirectiveInclude();

    // Look up the handler in the handler table.
    std::pair<MCAsmParserExtension*, DirectiveHandler> Handler =
      DirectiveMap.lookup(IDVal);
    if (Handler.first)
      return (Handler.first->*Handler.second)(IDVal, IDLoc);

    // Target hook for parsing target specific directives.
    if (!getTargetParser().ParseDirective(ID))
      return false;

    Warning(IDLoc, "ignoring directive for now");
    EatToEndOfStatement();
    return false;
  }

  // Canonicalize the opcode to lower case.
  SmallString<128> Opcode;
  for (unsigned i = 0, e = IDVal.size(); i != e; ++i)
    Opcode.push_back(tolower(IDVal[i]));
  
  SmallVector<MCParsedAsmOperand*, 8> ParsedOperands;
  bool HadError = getTargetParser().ParseInstruction(Opcode.str(), IDLoc,
                                                     ParsedOperands);
  if (!HadError && Lexer.isNot(AsmToken::EndOfStatement))
    HadError = TokError("unexpected token in argument list");

  // If parsing succeeded, match the instruction.
  if (!HadError) {
    MCInst Inst;
    if (!getTargetParser().MatchInstruction(ParsedOperands, Inst)) {
      // Emit the instruction on success.
      Out.EmitInstruction(Inst);
    } else {
      // Otherwise emit a diagnostic about the match failure and set the error
      // flag.
      //
      // FIXME: We should give nicer diagnostics about the exact failure.
      Error(IDLoc, "unrecognized instruction");
      HadError = true;
    }
  }

  // If there was no error, consume the end-of-statement token. Otherwise this
  // will be done by our caller.
  if (!HadError)
    Lex();

  // Free any parsed operands.
  for (unsigned i = 0, e = ParsedOperands.size(); i != e; ++i)
    delete ParsedOperands[i];

  return HadError;
}

bool AsmParser::ParseAssignment(const StringRef &Name) {
  // FIXME: Use better location, we should use proper tokens.
  SMLoc EqualLoc = Lexer.getLoc();

  const MCExpr *Value;
  if (ParseExpression(Value))
    return true;
  
  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in assignment");

  // Eat the end of statement marker.
  Lex();

  // Validate that the LHS is allowed to be a variable (either it has not been
  // used as a symbol, or it is an absolute symbol).
  MCSymbol *Sym = getContext().LookupSymbol(Name);
  if (Sym) {
    // Diagnose assignment to a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: Diagnose assignment to protected identifier (e.g., register name).
    if (Sym->isUndefined() && !Sym->isUsedInExpr())
      ; // Allow redefinitions of undefined symbols only used in directives.
    else if (!Sym->isUndefined() && !Sym->isAbsolute())
      return Error(EqualLoc, "redefinition of '" + Name + "'");
    else if (!Sym->isVariable())
      return Error(EqualLoc, "invalid assignment to '" + Name + "'");
    else if (!isa<MCConstantExpr>(Sym->getVariableValue()))
      return Error(EqualLoc, "invalid reassignment of non-absolute variable '" +
                   Name + "'");
  } else
    Sym = getContext().GetOrCreateSymbol(Name);

  // FIXME: Handle '.'.

  Sym->setUsedInExpr(true);

  // Do the assignment.
  Out.EmitAssignment(Sym, Value);

  return false;
}

/// ParseIdentifier:
///   ::= identifier
///   ::= string
bool AsmParser::ParseIdentifier(StringRef &Res) {
  if (Lexer.isNot(AsmToken::Identifier) &&
      Lexer.isNot(AsmToken::String))
    return true;

  Res = getTok().getIdentifier();

  Lex(); // Consume the identifier token.

  return false;
}

/// ParseDirectiveSet:
///   ::= .set identifier ',' expression
bool AsmParser::ParseDirectiveSet() {
  StringRef Name;

  if (ParseIdentifier(Name))
    return TokError("expected identifier after '.set' directive");
  
  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.set'");
  Lex();

  return ParseAssignment(Name);
}

/// ParseDirectiveSection:
///   ::= .section identifier (',' identifier)*
/// FIXME: This should actually parse out the segment, section, attributes and
/// sizeof_stub fields.
bool AsmParser::ParseDirectiveDarwinSection() {
  SMLoc Loc = getLexer().getLoc();

  StringRef SectionName;
  if (ParseIdentifier(SectionName))
    return Error(Loc, "expected identifier after '.section' directive");

  // Verify there is a following comma.
  if (!getLexer().is(AsmToken::Comma))
    return TokError("unexpected token in '.section' directive");

  std::string SectionSpec = SectionName;
  SectionSpec += ",";

  // Add all the tokens until the end of the line, ParseSectionSpecifier will
  // handle this.
  StringRef EOL = Lexer.LexUntilEndOfStatement();
  SectionSpec.append(EOL.begin(), EOL.end());

  Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.section' directive");
  Lex();


  StringRef Segment, Section;
  unsigned TAA, StubSize;
  std::string ErrorStr = 
    MCSectionMachO::ParseSectionSpecifier(SectionSpec, Segment, Section,
                                          TAA, StubSize);
  
  if (!ErrorStr.empty())
    return Error(Loc, ErrorStr.c_str());
  
  // FIXME: Arch specific.
  bool isText = Segment == "__TEXT";  // FIXME: Hack.
  getStreamer().SwitchSection(Ctx.getMachOSection(
                                Segment, Section, TAA, StubSize,
                                isText ? SectionKind::getText()
                                : SectionKind::getDataRel()));
  return false;
}

bool DarwinAsmParser::ParseSectionSwitch(const char *Segment,
                                         const char *Section,
                                         unsigned TAA, unsigned Align,
                                         unsigned StubSize) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lex();

  // FIXME: Arch specific.
  bool isText = StringRef(Segment) == "__TEXT";  // FIXME: Hack.
  getStreamer().SwitchSection(getContext().getMachOSection(
                                Segment, Section, TAA, StubSize,
                                isText ? SectionKind::getText()
                                       : SectionKind::getDataRel()));

  // Set the implicit alignment, if any.
  //
  // FIXME: This isn't really what 'as' does; I think it just uses the implicit
  // alignment on the section (e.g., if one manually inserts bytes into the
  // section, then just issueing the section switch directive will not realign
  // the section. However, this is arguably more reasonable behavior, and there
  // is no good reason for someone to intentionally emit incorrectly sized
  // values into the implicitly aligned sections.
  if (Align)
    getStreamer().EmitValueToAlignment(Align, 0, 1, 0);

  return false;
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

bool AsmParser::ParseEscapedString(std::string &Data) {
  assert(getLexer().is(AsmToken::String) && "Unexpected current token!");

  Data = "";
  StringRef Str = getTok().getStringContents();
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] != '\\') {
      Data += Str[i];
      continue;
    }

    // Recognize escaped characters. Note that this escape semantics currently
    // loosely follows Darwin 'as'. Notably, it doesn't support hex escapes.
    ++i;
    if (i == e)
      return TokError("unexpected backslash at end of string");

    // Recognize octal sequences.
    if ((unsigned) (Str[i] - '0') <= 7) {
      // Consume up to three octal characters.
      unsigned Value = Str[i] - '0';

      if (i + 1 != e && ((unsigned) (Str[i + 1] - '0')) <= 7) {
        ++i;
        Value = Value * 8 + (Str[i] - '0');

        if (i + 1 != e && ((unsigned) (Str[i + 1] - '0')) <= 7) {
          ++i;
          Value = Value * 8 + (Str[i] - '0');
        }
      }

      if (Value > 255)
        return TokError("invalid octal escape sequence (out of range)");

      Data += (unsigned char) Value;
      continue;
    }

    // Otherwise recognize individual escapes.
    switch (Str[i]) {
    default:
      // Just reject invalid escape sequences for now.
      return TokError("invalid escape sequence (unrecognized character)");

    case 'b': Data += '\b'; break;
    case 'f': Data += '\f'; break;
    case 'n': Data += '\n'; break;
    case 'r': Data += '\r'; break;
    case 't': Data += '\t'; break;
    case '"': Data += '"'; break;
    case '\\': Data += '\\'; break;
    }
  }

  return false;
}

/// ParseDirectiveAscii:
///   ::= ( .ascii | .asciz ) [ "string" ( , "string" )* ]
bool AsmParser::ParseDirectiveAscii(bool ZeroTerminated) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      if (getLexer().isNot(AsmToken::String))
        return TokError("expected string in '.ascii' or '.asciz' directive");

      std::string Data;
      if (ParseEscapedString(Data))
        return true;

      getStreamer().EmitBytes(Data, DEFAULT_ADDRSPACE);
      if (ZeroTerminated)
        getStreamer().EmitBytes(StringRef("\0", 1), DEFAULT_ADDRSPACE);

      Lex();

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in '.ascii' or '.asciz' directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// ParseDirectiveValue
///  ::= (.byte | .short | ... ) [ expression (, expression)* ]
bool AsmParser::ParseDirectiveValue(unsigned Size) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      SMLoc ATTRIBUTE_UNUSED StartLoc = getLexer().getLoc();
      if (ParseExpression(Value))
        return true;

      // Special case constant expressions to match code generator.
      if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value))
        getStreamer().EmitIntValue(MCE->getValue(), Size, DEFAULT_ADDRSPACE);
      else
        getStreamer().EmitValue(Value, Size, DEFAULT_ADDRSPACE);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;
      
      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// ParseDirectiveSpace
///  ::= .space expression [ , expression ]
bool AsmParser::ParseDirectiveSpace() {
  int64_t NumBytes;
  if (ParseAbsoluteExpression(NumBytes))
    return true;

  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.space' directive");
    Lex();
    
    if (ParseAbsoluteExpression(FillExpr))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.space' directive");
  }

  Lex();

  if (NumBytes <= 0)
    return TokError("invalid number of bytes in '.space' directive");

  // FIXME: Sometimes the fill expr is 'nop' if it isn't supplied, instead of 0.
  getStreamer().EmitFill(NumBytes, FillExpr, DEFAULT_ADDRSPACE);

  return false;
}

/// ParseDirectiveFill
///  ::= .fill expression , expression , expression
bool AsmParser::ParseDirectiveFill() {
  int64_t NumValues;
  if (ParseAbsoluteExpression(NumValues))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lex();
  
  int64_t FillSize;
  if (ParseAbsoluteExpression(FillSize))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lex();
  
  int64_t FillExpr;
  if (ParseAbsoluteExpression(FillExpr))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.fill' directive");
  
  Lex();

  if (FillSize != 1 && FillSize != 2 && FillSize != 4 && FillSize != 8)
    return TokError("invalid '.fill' size, expected 1, 2, 4, or 8");

  for (uint64_t i = 0, e = NumValues; i != e; ++i)
    getStreamer().EmitIntValue(FillExpr, FillSize, DEFAULT_ADDRSPACE);

  return false;
}

/// ParseDirectiveOrg
///  ::= .org expression [ , expression ]
bool AsmParser::ParseDirectiveOrg() {
  const MCExpr *Offset;
  if (ParseExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.org' directive");
    Lex();
    
    if (ParseAbsoluteExpression(FillExpr))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.org' directive");
  }

  Lex();

  // FIXME: Only limited forms of relocatable expressions are accepted here, it
  // has to be relative to the current section.
  getStreamer().EmitValueToOffset(Offset, FillExpr);

  return false;
}

/// ParseDirectiveAlign
///  ::= {.align, ...} expression [ , expression [ , expression ]]
bool AsmParser::ParseDirectiveAlign(bool IsPow2, unsigned ValueSize) {
  SMLoc AlignmentLoc = getLexer().getLoc();
  int64_t Alignment;
  if (ParseAbsoluteExpression(Alignment))
    return true;

  SMLoc MaxBytesLoc;
  bool HasFillExpr = false;
  int64_t FillExpr = 0;
  int64_t MaxBytesToFill = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in directive");
    Lex();

    // The fill expression can be omitted while specifying a maximum number of
    // alignment bytes, e.g:
    //  .align 3,,4
    if (getLexer().isNot(AsmToken::Comma)) {
      HasFillExpr = true;
      if (ParseAbsoluteExpression(FillExpr))
        return true;
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();

      MaxBytesLoc = getLexer().getLoc();
      if (ParseAbsoluteExpression(MaxBytesToFill))
        return true;
      
      if (getLexer().isNot(AsmToken::EndOfStatement))
        return TokError("unexpected token in directive");
    }
  }

  Lex();

  if (!HasFillExpr)
    FillExpr = 0;

  // Compute alignment in bytes.
  if (IsPow2) {
    // FIXME: Diagnose overflow.
    if (Alignment >= 32) {
      Error(AlignmentLoc, "invalid alignment value");
      Alignment = 31;
    }

    Alignment = 1ULL << Alignment;
  }

  // Diagnose non-sensical max bytes to align.
  if (MaxBytesLoc.isValid()) {
    if (MaxBytesToFill < 1) {
      Error(MaxBytesLoc, "alignment directive can never be satisfied in this "
            "many bytes, ignoring maximum bytes expression");
      MaxBytesToFill = 0;
    }

    if (MaxBytesToFill >= Alignment) {
      Warning(MaxBytesLoc, "maximum bytes expression exceeds alignment and "
              "has no effect");
      MaxBytesToFill = 0;
    }
  }

  // Check whether we should use optimal code alignment for this .align
  // directive.
  //
  // FIXME: This should be using a target hook.
  bool UseCodeAlign = false;
  if (const MCSectionMachO *S = dyn_cast<MCSectionMachO>(
        getStreamer().getCurrentSection()))
      UseCodeAlign = S->hasAttribute(MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS);
  if ((!HasFillExpr || Lexer.getMAI().getTextAlignFillValue() == FillExpr) &&
      ValueSize == 1 && UseCodeAlign) {
    getStreamer().EmitCodeAlignment(Alignment, MaxBytesToFill);
  } else {
    // FIXME: Target specific behavior about how the "extra" bytes are filled.
    getStreamer().EmitValueToAlignment(Alignment, FillExpr, ValueSize, MaxBytesToFill);
  }

  return false;
}

/// ParseDirectiveSymbolAttribute
///  ::= { ".globl", ".weak", ... } [ identifier ( , identifier )* ]
bool AsmParser::ParseDirectiveSymbolAttribute(MCSymbolAttr Attr) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      StringRef Name;

      if (ParseIdentifier(Name))
        return TokError("expected identifier in directive");
      
      MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

      getStreamer().EmitSymbolAttribute(Sym, Attr);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;  
}

/// ParseDirectiveELFType
///  ::= .type identifier , @attribute
bool AsmParser::ParseDirectiveELFType() {
  StringRef Name;
  if (ParseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.type' directive");
  Lex();

  if (getLexer().isNot(AsmToken::At))
    return TokError("expected '@' before type");
  Lex();

  StringRef Type;
  SMLoc TypeLoc;

  TypeLoc = getLexer().getLoc();
  if (ParseIdentifier(Type))
    return TokError("expected symbol type in directive");

  MCSymbolAttr Attr = StringSwitch<MCSymbolAttr>(Type)
    .Case("function", MCSA_ELF_TypeFunction)
    .Case("object", MCSA_ELF_TypeObject)
    .Case("tls_object", MCSA_ELF_TypeTLS)
    .Case("common", MCSA_ELF_TypeCommon)
    .Case("notype", MCSA_ELF_TypeNoType)
    .Default(MCSA_Invalid);

  if (Attr == MCSA_Invalid)
    return Error(TypeLoc, "unsupported attribute in '.type' directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.type' directive");

  Lex();

  getStreamer().EmitSymbolAttribute(Sym, Attr);

  return false;
}

/// ParseDirectiveDesc
///  ::= .desc identifier , expression
bool DarwinAsmParser::ParseDirectiveDesc(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.desc' directive");
  Lex();

  int64_t DescValue;
  if (getParser().ParseAbsoluteExpression(DescValue))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.desc' directive");
  
  Lex();

  // Set the n_desc field of this Symbol to this DescValue
  getStreamer().EmitSymbolDesc(Sym, DescValue);

  return false;
}

/// ParseDirectiveComm
///  ::= ( .comm | .lcomm ) identifier , size_expression [ , align_expression ]
bool AsmParser::ParseDirectiveComm(bool IsLocal) {
  SMLoc IDLoc = getLexer().getLoc();
  StringRef Name;
  if (ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (ParseAbsoluteExpression(Pow2Alignment))
      return true;
    
    // If this target takes alignments in bytes (not log) validate and convert.
    if (Lexer.getMAI().getAlignmentIsInBytes()) {
      if (!isPowerOf2_64(Pow2Alignment))
        return Error(Pow2AlignmentLoc, "alignment must be a power of 2");
      Pow2Alignment = Log2_64(Pow2Alignment);
    }
  }
  
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.comm' or '.lcomm' directive");
  
  Lex();

  // NOTE: a size of zero for a .comm should create a undefined symbol
  // but a size of .lcomm creates a bss symbol of size zero.
  if (Size < 0)
    return Error(SizeLoc, "invalid '.comm' or '.lcomm' directive size, can't "
                 "be less than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assembler
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.comm' or '.lcomm' directive "
                 "alignment, can't be less than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // '.lcomm' is equivalent to '.zerofill'.
  // Create the Symbol as a common or local common with Size and Pow2Alignment
  if (IsLocal) {
    getStreamer().EmitZerofill(Ctx.getMachOSection(
                                 "__DATA", "__bss", MCSectionMachO::S_ZEROFILL,
                                 0, SectionKind::getBSS()),
                               Sym, Size, 1 << Pow2Alignment);
    return false;
  }

  getStreamer().EmitCommonSymbol(Sym, Size, 1 << Pow2Alignment);
  return false;
}

/// ParseDirectiveZerofill
///  ::= .zerofill segname , sectname [, identifier , size_expression [
///      , align_expression ]]
bool DarwinAsmParser::ParseDirectiveZerofill(StringRef, SMLoc) {
  StringRef Segment;
  if (getParser().ParseIdentifier(Segment))
    return TokError("expected segment name after '.zerofill' directive");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  StringRef Section;
  if (getParser().ParseIdentifier(Section))
    return TokError("expected section name after comma in '.zerofill' "
                    "directive");

  // If this is the end of the line all that was wanted was to create the
  // the section but with no symbol.
  if (getLexer().is(AsmToken::EndOfStatement)) {
    // Create the zerofill section but no symbol
    getStreamer().EmitZerofill(getContext().getMachOSection(
                                 Segment, Section, MCSectionMachO::S_ZEROFILL,
                                 0, SectionKind::getBSS()));
    return false;
  }

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  SMLoc IDLoc = getLexer().getLoc();
  StringRef IDStr;
  if (getParser().ParseIdentifier(IDStr))
    return TokError("expected identifier in directive");
  
  // handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(IDStr);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (getParser().ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (getParser().ParseAbsoluteExpression(Pow2Alignment))
      return true;
  }
  
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.zerofill' directive");
  
  Lex();

  if (Size < 0)
    return Error(SizeLoc, "invalid '.zerofill' directive size, can't be less "
                 "than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assembler
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.zerofill' directive alignment, "
                 "can't be less than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // Create the zerofill Symbol with Size and Pow2Alignment
  //
  // FIXME: Arch specific.
  getStreamer().EmitZerofill(getContext().getMachOSection(
                               Segment, Section, MCSectionMachO::S_ZEROFILL,
                               0, SectionKind::getBSS()),
                             Sym, Size, 1 << Pow2Alignment);

  return false;
}

/// ParseDirectiveTBSS
///  ::= .tbss identifier, size, align
bool DarwinAsmParser::ParseDirectiveTBSS(StringRef, SMLoc) {
  SMLoc IDLoc = getLexer().getLoc();
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");
    
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (getParser().ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (getParser().ParseAbsoluteExpression(Pow2Alignment))
      return true;
  }
  
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.tbss' directive");
  
  Lex();

  if (Size < 0)
    return Error(SizeLoc, "invalid '.tbss' directive size, can't be less than"
                 "zero");

  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.tbss' alignment, can't be less"
                 "than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");
  
  getStreamer().EmitTBSSSymbol(getContext().getMachOSection(
                                 "__DATA", "__thread_bss",
                                 MCSectionMachO::S_THREAD_LOCAL_ZEROFILL,
                                 0, SectionKind::getThreadBSS()),
                               Sym, Size, 1 << Pow2Alignment);
  
  return false;
}

/// ParseDirectiveSubsectionsViaSymbols
///  ::= .subsections_via_symbols
bool DarwinAsmParser::ParseDirectiveSubsectionsViaSymbols(StringRef, SMLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.subsections_via_symbols' directive");
  
  Lex();

  getStreamer().EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);

  return false;
}

/// ParseDirectiveAbort
///  ::= .abort [ "abort_string" ]
bool AsmParser::ParseDirectiveAbort() {
  // FIXME: Use loc from directive.
  SMLoc Loc = getLexer().getLoc();

  StringRef Str = "";
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::String))
      return TokError("expected string in '.abort' directive");
    
    Str = getTok().getString();

    Lex();
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.abort' directive");
  
  Lex();

  // FIXME: Handle here.
  if (Str.empty())
    Error(Loc, ".abort detected. Assembly stopping.");
  else
    Error(Loc, ".abort '" + Str + "' detected. Assembly stopping.");

  return false;
}

/// ParseDirectiveLsym
///  ::= .lsym identifier , expression
bool DarwinAsmParser::ParseDirectiveLsym(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");
  
  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.lsym' directive");
  Lex();

  const MCExpr *Value;
  if (getParser().ParseExpression(Value))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.lsym' directive");
  
  Lex();

  // We don't currently support this directive.
  //
  // FIXME: Diagnostic location!
  (void) Sym;
  return TokError("directive '.lsym' is unsupported");
}

/// ParseDirectiveInclude
///  ::= .include "filename"
bool AsmParser::ParseDirectiveInclude() {
  if (getLexer().isNot(AsmToken::String))
    return TokError("expected string in '.include' directive");
  
  std::string Filename = getTok().getString();
  SMLoc IncludeLoc = getLexer().getLoc();
  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.include' directive");
  
  // Strip the quotes.
  Filename = Filename.substr(1, Filename.size()-2);
  
  // Attempt to switch the lexer to the included file before consuming the end
  // of statement to avoid losing it when we switch.
  if (EnterIncludeFile(Filename)) {
    PrintMessage(IncludeLoc,
                 "Could not find include file '" + Filename + "'",
                 "error");
    return true;
  }

  return false;
}

/// ParseDirectiveDumpOrLoad
///  ::= ( .dump | .load ) "filename"
bool DarwinAsmParser::ParseDirectiveDumpOrLoad(StringRef Directive,
                                               SMLoc IDLoc) {
  bool IsDump = Directive == ".dump";
  if (getLexer().isNot(AsmToken::String))
    return TokError("expected string in '.dump' or '.load' directive");
  
  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.dump' or '.load' directive");
  
  Lex();

  // FIXME: If/when .dump and .load are implemented they will be done in the
  // the assembly parser and not have any need for an MCStreamer API.
  if (IsDump)
    Warning(IDLoc, "ignoring directive .dump for now");
  else
    Warning(IDLoc, "ignoring directive .load for now");

  return false;
}

/// ParseDirectiveSecureLogUnique
///  ::= .secure_log_unique "log message"
bool DarwinAsmParser::ParseDirectiveSecureLogUnique(StringRef, SMLoc IDLoc) {
  std::string LogMessage;

  if (getLexer().isNot(AsmToken::String))
    LogMessage = "";
  else{
    LogMessage = getTok().getString();
    Lex();
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.secure_log_unique' directive");
  
  if (getContext().getSecureLogUsed() != false)
    return Error(IDLoc, ".secure_log_unique specified multiple times");

  char *SecureLogFile = getContext().getSecureLogFile();
  if (SecureLogFile == NULL)
    return Error(IDLoc, ".secure_log_unique used but AS_SECURE_LOG_FILE "
                 "environment variable unset.");

  raw_ostream *OS = getContext().getSecureLog();
  if (OS == NULL) {
    std::string Err;
    OS = new raw_fd_ostream(SecureLogFile, Err, raw_fd_ostream::F_Append);
    if (!Err.empty()) {
       delete OS;
       return Error(IDLoc, Twine("can't open secure log file: ") +
                    SecureLogFile + " (" + Err + ")");
    }
    getContext().setSecureLog(OS);
  }

  int CurBuf = getSourceManager().FindBufferContainingLoc(IDLoc);
  *OS << getSourceManager().getBufferInfo(CurBuf).Buffer->getBufferIdentifier()
      << ":" << getSourceManager().FindLineNumber(IDLoc, CurBuf) << ":"
      << LogMessage + "\n";

  getContext().setSecureLogUsed(true);

  return false;
}

/// ParseDirectiveSecureLogReset
///  ::= .secure_log_reset
bool DarwinAsmParser::ParseDirectiveSecureLogReset(StringRef, SMLoc IDLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.secure_log_reset' directive");
  
  Lex();

  getContext().setSecureLogUsed(false);

  return false;
}

/// ParseDirectiveIf
/// ::= .if expression
bool AsmParser::ParseDirectiveIf(SMLoc DirectiveLoc) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;
  if(TheCondState.Ignore) {
    EatToEndOfStatement();
  }
  else {
    int64_t ExprValue;
    if (ParseAbsoluteExpression(ExprValue))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.if' directive");
    
    Lex();

    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveElseIf
/// ::= .elseif expression
bool AsmParser::ParseDirectiveElseIf(SMLoc DirectiveLoc) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
      Error(DirectiveLoc, "Encountered a .elseif that doesn't follow a .if or "
                          " an .elseif");
  TheCondState.TheCond = AsmCond::ElseIfCond;

  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
      LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet) {
    TheCondState.Ignore = true;
    EatToEndOfStatement();
  }
  else {
    int64_t ExprValue;
    if (ParseAbsoluteExpression(ExprValue))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.elseif' directive");
    
    Lex();
    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveElse
/// ::= .else
bool AsmParser::ParseDirectiveElse(SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.else' directive");
  
  Lex();

  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
      Error(DirectiveLoc, "Encountered a .else that doesn't follow a .if or an "
                          ".elseif");
  TheCondState.TheCond = AsmCond::ElseCond;
  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet)
    TheCondState.Ignore = true;
  else
    TheCondState.Ignore = false;

  return false;
}

/// ParseDirectiveEndIf
/// ::= .endif
bool AsmParser::ParseDirectiveEndIf(SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.endif' directive");
  
  Lex();

  if ((TheCondState.TheCond == AsmCond::NoCond) ||
      TheCondStack.empty())
    Error(DirectiveLoc, "Encountered a .endif that doesn't follow a .if or "
                        ".else");
  if (!TheCondStack.empty()) {
    TheCondState = TheCondStack.back();
    TheCondStack.pop_back();
  }

  return false;
}

/// ParseDirectiveFile
/// ::= .file [number] string
bool GenericAsmParser::ParseDirectiveFile(StringRef, SMLoc DirectiveLoc) {
  // FIXME: I'm not sure what this is.
  int64_t FileNumber = -1;
  if (getLexer().is(AsmToken::Integer)) {
    FileNumber = getTok().getIntVal();
    Lex();

    if (FileNumber < 1)
      return TokError("file number less than one");
  }

  if (getLexer().isNot(AsmToken::String))
    return TokError("unexpected token in '.file' directive");

  StringRef Filename = getTok().getString();
  Filename = Filename.substr(1, Filename.size()-2);
  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.file' directive");

  if (FileNumber == -1)
    getStreamer().EmitFileDirective(Filename);
  else
    getStreamer().EmitDwarfFileDirective(FileNumber, Filename);

  return false;
}

/// ParseDirectiveLine
/// ::= .line [number]
bool GenericAsmParser::ParseDirectiveLine(StringRef, SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Integer))
      return TokError("unexpected token in '.line' directive");

    int64_t LineNumber = getTok().getIntVal();
    (void) LineNumber;
    Lex();

    // FIXME: Do something with the .line.
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.line' directive");

  return false;
}


/// ParseDirectiveLoc
/// ::= .loc number [number [number]]
bool GenericAsmParser::ParseDirectiveLoc(StringRef, SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::Integer))
    return TokError("unexpected token in '.loc' directive");

  // FIXME: What are these fields?
  int64_t FileNumber = getTok().getIntVal();
  (void) FileNumber;
  // FIXME: Validate file.

  Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Integer))
      return TokError("unexpected token in '.loc' directive");

    int64_t Param2 = getTok().getIntVal();
    (void) Param2;
    Lex();

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (getLexer().isNot(AsmToken::Integer))
        return TokError("unexpected token in '.loc' directive");

      int64_t Param3 = getTok().getIntVal();
      (void) Param3;
      Lex();

      // FIXME: Do something with the .loc.
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.file' directive");

  return false;
}


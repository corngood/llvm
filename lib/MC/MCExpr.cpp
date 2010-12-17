//===- MCExpr.cpp - Assembly Level Expression Implementation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mcexpr"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFormat.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"
using namespace llvm;

namespace {
namespace stats {
STATISTIC(MCExprEvaluate, "Number of MCExpr evaluations");
}
}

void MCExpr::print(raw_ostream &OS) const {
  switch (getKind()) {
  case MCExpr::Target:
    return cast<MCTargetExpr>(this)->PrintImpl(OS);
  case MCExpr::Constant:
    OS << cast<MCConstantExpr>(*this).getValue();
    return;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SRE = cast<MCSymbolRefExpr>(*this);
    const MCSymbol &Sym = SRE.getSymbol();
    // Parenthesize names that start with $ so that they don't look like
    // absolute names.
    bool UseParens = Sym.getName()[0] == '$';

    if (SRE.getKind() == MCSymbolRefExpr::VK_ARM_HI16 ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_LO16)
      OS << MCSymbolRefExpr::getVariantKindName(SRE.getKind());

    if (SRE.getKind() == MCSymbolRefExpr::VK_PPC_HA16 ||
        SRE.getKind() == MCSymbolRefExpr::VK_PPC_LO16) {
      OS << MCSymbolRefExpr::getVariantKindName(SRE.getKind());
      UseParens = true;
    }

    if (UseParens)
      OS << '(' << Sym << ')';
    else
      OS << Sym;

    if (SRE.getKind() == MCSymbolRefExpr::VK_ARM_PLT ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_TLSGD ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_GOT ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_GOTOFF ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_TPOFF ||
        SRE.getKind() == MCSymbolRefExpr::VK_ARM_GOTTPOFF)
      OS << MCSymbolRefExpr::getVariantKindName(SRE.getKind());
    else if (SRE.getKind() != MCSymbolRefExpr::VK_None &&
             SRE.getKind() != MCSymbolRefExpr::VK_ARM_HI16 &&
             SRE.getKind() != MCSymbolRefExpr::VK_ARM_LO16 &&
             SRE.getKind() != MCSymbolRefExpr::VK_PPC_HA16 &&
             SRE.getKind() != MCSymbolRefExpr::VK_PPC_LO16)
      OS << '@' << MCSymbolRefExpr::getVariantKindName(SRE.getKind());

    return;
  }

  case MCExpr::Unary: {
    const MCUnaryExpr &UE = cast<MCUnaryExpr>(*this);
    switch (UE.getOpcode()) {
    default: assert(0 && "Invalid opcode!");
    case MCUnaryExpr::LNot:  OS << '!'; break;
    case MCUnaryExpr::Minus: OS << '-'; break;
    case MCUnaryExpr::Not:   OS << '~'; break;
    case MCUnaryExpr::Plus:  OS << '+'; break;
    }
    OS << *UE.getSubExpr();
    return;
  }

  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(*this);

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getLHS()) || isa<MCSymbolRefExpr>(BE.getLHS())) {
      OS << *BE.getLHS();
    } else {
      OS << '(' << *BE.getLHS() << ')';
    }

    switch (BE.getOpcode()) {
    default: assert(0 && "Invalid opcode!");
    case MCBinaryExpr::Add:
      // Print "X-42" instead of "X+-42".
      if (const MCConstantExpr *RHSC = dyn_cast<MCConstantExpr>(BE.getRHS())) {
        if (RHSC->getValue() < 0) {
          OS << RHSC->getValue();
          return;
        }
      }

      OS <<  '+';
      break;
    case MCBinaryExpr::And:  OS <<  '&'; break;
    case MCBinaryExpr::Div:  OS <<  '/'; break;
    case MCBinaryExpr::EQ:   OS << "=="; break;
    case MCBinaryExpr::GT:   OS <<  '>'; break;
    case MCBinaryExpr::GTE:  OS << ">="; break;
    case MCBinaryExpr::LAnd: OS << "&&"; break;
    case MCBinaryExpr::LOr:  OS << "||"; break;
    case MCBinaryExpr::LT:   OS <<  '<'; break;
    case MCBinaryExpr::LTE:  OS << "<="; break;
    case MCBinaryExpr::Mod:  OS <<  '%'; break;
    case MCBinaryExpr::Mul:  OS <<  '*'; break;
    case MCBinaryExpr::NE:   OS << "!="; break;
    case MCBinaryExpr::Or:   OS <<  '|'; break;
    case MCBinaryExpr::Shl:  OS << "<<"; break;
    case MCBinaryExpr::Shr:  OS << ">>"; break;
    case MCBinaryExpr::Sub:  OS <<  '-'; break;
    case MCBinaryExpr::Xor:  OS <<  '^'; break;
    }

    // Only print parens around the LHS if it is non-trivial.
    if (isa<MCConstantExpr>(BE.getRHS()) || isa<MCSymbolRefExpr>(BE.getRHS())) {
      OS << *BE.getRHS();
    } else {
      OS << '(' << *BE.getRHS() << ')';
    }
    return;
  }
  }

  assert(0 && "Invalid expression kind!");
}

void MCExpr::dump() const {
  print(dbgs());
  dbgs() << '\n';
}

/* *** */

const MCBinaryExpr *MCBinaryExpr::Create(Opcode Opc, const MCExpr *LHS,
                                         const MCExpr *RHS, MCContext &Ctx) {
  return new (Ctx) MCBinaryExpr(Opc, LHS, RHS);
}

const MCUnaryExpr *MCUnaryExpr::Create(Opcode Opc, const MCExpr *Expr,
                                       MCContext &Ctx) {
  return new (Ctx) MCUnaryExpr(Opc, Expr);
}

const MCConstantExpr *MCConstantExpr::Create(int64_t Value, MCContext &Ctx) {
  return new (Ctx) MCConstantExpr(Value);
}

/* *** */

const MCSymbolRefExpr *MCSymbolRefExpr::Create(const MCSymbol *Sym,
                                               VariantKind Kind,
                                               MCContext &Ctx) {
  return new (Ctx) MCSymbolRefExpr(Sym, Kind);
}

const MCSymbolRefExpr *MCSymbolRefExpr::Create(StringRef Name, VariantKind Kind,
                                               MCContext &Ctx) {
  return Create(Ctx.GetOrCreateSymbol(Name), Kind, Ctx);
}

StringRef MCSymbolRefExpr::getVariantKindName(VariantKind Kind) {
  switch (Kind) {
  default:
  case VK_Invalid: return "<<invalid>>";
  case VK_None: return "<<none>>";

  case VK_GOT: return "GOT";
  case VK_GOTOFF: return "GOTOFF";
  case VK_GOTPCREL: return "GOTPCREL";
  case VK_GOTTPOFF: return "GOTTPOFF";
  case VK_INDNTPOFF: return "INDNTPOFF";
  case VK_NTPOFF: return "NTPOFF";
  case VK_GOTNTPOFF: return "GOTNTPOFF";
  case VK_PLT: return "PLT";
  case VK_TLSGD: return "TLSGD";
  case VK_TLSLD: return "TLSLD";
  case VK_TLSLDM: return "TLSLDM";
  case VK_TPOFF: return "TPOFF";
  case VK_DTPOFF: return "DTPOFF";
  case VK_TLVP: return "TLVP";
  case VK_ARM_HI16: return ":upper16:";
  case VK_ARM_LO16: return ":lower16:";
  case VK_ARM_PLT: return "(PLT)";
  case VK_ARM_GOT: return "(GOT)";
  case VK_ARM_GOTOFF: return "(GOTOFF)";
  case VK_ARM_TPOFF: return "(tpoff)";
  case VK_ARM_GOTTPOFF: return "(gottpoff)";
  case VK_ARM_TLSGD: return "(tlsgd)";
  case VK_PPC_TOC: return "toc";
  case VK_PPC_HA16: return "ha16";
  case VK_PPC_LO16: return "lo16";
  }
}

MCSymbolRefExpr::VariantKind
MCSymbolRefExpr::getVariantKindForName(StringRef Name) {
  return StringSwitch<VariantKind>(Name)
    .Case("GOT", VK_GOT)
    .Case("GOTOFF", VK_GOTOFF)
    .Case("GOTPCREL", VK_GOTPCREL)
    .Case("GOTTPOFF", VK_GOTTPOFF)
    .Case("INDNTPOFF", VK_INDNTPOFF)
    .Case("NTPOFF", VK_NTPOFF)
    .Case("GOTNTPOFF", VK_GOTNTPOFF)
    .Case("PLT", VK_PLT)
    .Case("TLSGD", VK_TLSGD)
    .Case("TLSLD", VK_TLSLD)
    .Case("TLSLDM", VK_TLSLDM)
    .Case("TPOFF", VK_TPOFF)
    .Case("DTPOFF", VK_DTPOFF)
    .Case("TLVP", VK_TLVP)
    .Default(VK_Invalid);
}

/* *** */

void MCTargetExpr::Anchor() {}

/* *** */

bool MCExpr::EvaluateAsAbsolute(int64_t &Res, const MCAsmLayout *Layout,
                                const SectionAddrMap *Addrs) const {
  MCValue Value;

  // Fast path constants.
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(this)) {
    Res = CE->getValue();
    return true;
  }

  // FIXME: This use of Addrs is wrong, right?
  if (!EvaluateAsRelocatableImpl(Value, Layout, Addrs, /*InSet=*/Addrs) ||
      !Value.isAbsolute()) {
    // EvaluateAsAbsolute is defined to return the "current value" of
    // the expression if we are given a Layout object, even in cases
    // when the value is not fixed.
    if (Layout) {
      Res = Value.getConstant();
      if (Value.getSymA()) {
       Res += Layout->getSymbolOffset(
          &Layout->getAssembler().getSymbolData(Value.getSymA()->getSymbol()));
      }
      if (Value.getSymB()) {
       Res -= Layout->getSymbolOffset(
          &Layout->getAssembler().getSymbolData(Value.getSymB()->getSymbol()));
      }
    }
    return false;
  }

  Res = Value.getConstant();
  return true;
}

/// \brief Helper method for \see EvaluateSymbolAdd().
static void AttemptToFoldSymbolOffsetDifference(const MCAsmLayout *Layout,
                                                const MCSymbolRefExpr *&A,
                                                const MCSymbolRefExpr *&B,
                                                int64_t &Addend) {
  const MCAssembler &Asm = Layout->getAssembler();

  if (A && B &&
      Asm.getWriter().IsSymbolRefDifferenceFullyResolved(Asm, A, B)) {
    // Eagerly evaluate.
    Addend += (Layout->getSymbolOffset(&Asm.getSymbolData(A->getSymbol())) -
               Layout->getSymbolOffset(&Asm.getSymbolData(B->getSymbol())));

    // Clear the symbol expr pointers to indicate we have folded these
    // operands.
    A = B = 0;
  }
}

/// \brief Evaluate the result of an add between (conceptually) two MCValues.
///
/// This routine conceptually attempts to construct an MCValue:
///   Result = (Result_A - Result_B + Result_Cst)
/// from two MCValue's LHS and RHS where
///   Result = LHS + RHS
/// and
///   Result = (LHS_A - LHS_B + LHS_Cst) + (RHS_A - RHS_B + RHS_Cst).
///
/// This routine attempts to aggresively fold the operands such that the result
/// is representable in an MCValue, but may not always succeed.
///
/// \returns True on success, false if the result is not representable in an
/// MCValue.
static bool EvaluateSymbolicAdd(const MCAsmLayout *Layout,
                                const SectionAddrMap *Addrs,
                                bool InSet,
                                const MCValue &LHS,const MCSymbolRefExpr *RHS_A,
                                const MCSymbolRefExpr *RHS_B, int64_t RHS_Cst,
                                MCValue &Res) {
  // FIXME: This routine (and other evaluation parts) are *incredibly* sloppy
  // about dealing with modifiers. This will ultimately bite us, one day.
  const MCSymbolRefExpr *LHS_A = LHS.getSymA();
  const MCSymbolRefExpr *LHS_B = LHS.getSymB();
  int64_t LHS_Cst = LHS.getConstant();

  // Fold the result constant immediately.
  int64_t Result_Cst = LHS_Cst + RHS_Cst;

  // If we have a layout, we can fold resolved differences.
  if (Layout) {
    // First, fold out any differences which are fully resolved. By
    // reassociating terms in
    //   Result = (LHS_A - LHS_B + LHS_Cst) + (RHS_A - RHS_B + RHS_Cst).
    // we have the four possible differences:
    //   (LHS_A - LHS_B),
    //   (LHS_A - RHS_B),
    //   (RHS_A - LHS_B),
    //   (RHS_A - RHS_B).
    // Since we are attempting to be as aggresive as possible about folding, we
    // attempt to evaluate each possible alternative.
    AttemptToFoldSymbolOffsetDifference(Layout, LHS_A, LHS_B, Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Layout, LHS_A, RHS_B, Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Layout, RHS_A, LHS_B, Result_Cst);
    AttemptToFoldSymbolOffsetDifference(Layout, RHS_A, RHS_B, Result_Cst);
  }

  // We can't represent the addition or subtraction of two symbols.
  if ((LHS_A && RHS_A) || (LHS_B && RHS_B))
    return false;

  // At this point, we have at most one additive symbol and one subtractive
  // symbol -- find them.
  const MCSymbolRefExpr *A = LHS_A ? LHS_A : RHS_A;
  const MCSymbolRefExpr *B = LHS_B ? LHS_B : RHS_B;

  // If we have a negated symbol, then we must have also have a non-negated
  // symbol in order to encode the expression.
  if (B && !A)
    return false;

  // Absolutize symbol differences between defined symbols when we have a
  // layout object and the target requests it.
  if (Layout && A && B) {
    const MCAssembler &Asm = Layout->getAssembler();
    const MCSymbol &SA = A->getSymbol();
    const MCSymbol &SB = B->getSymbol();
    const MCObjectFormat &F = Asm.getBackend().getObjectFormat();
    if (SA.isDefined() && SB.isDefined() && F.isAbsolute(InSet, SA, SB)) {
      MCSymbolData &AD = Asm.getSymbolData(A->getSymbol());
      MCSymbolData &BD = Asm.getSymbolData(B->getSymbol());

      if (AD.getFragment() == BD.getFragment()) {
        Res = MCValue::get(+ AD.getOffset()
                           - BD.getOffset()
                           + Result_Cst);
        return true;
      }

      if (Layout) {
        const MCSectionData &SecA = *AD.getFragment()->getParent();
        const MCSectionData &SecB = *BD.getFragment()->getParent();
        int64_t Val = + Layout->getSymbolOffset(&AD)
                      - Layout->getSymbolOffset(&BD)
                      + Result_Cst;
        if (&SecA != &SecB) {
          if (!Addrs)
            return false;
          Val += Addrs->lookup(&SecA);
          Val -= Addrs->lookup(&SecB);
        }
        Res = MCValue::get(Val);
        return true;
      }
    }
  }

  Res = MCValue::get(A, B, Result_Cst);
  return true;
}

bool MCExpr::EvaluateAsRelocatableImpl(MCValue &Res,
                                       const MCAsmLayout *Layout,
                                       const SectionAddrMap *Addrs,
                                       bool InSet) const {
  ++stats::MCExprEvaluate;

  switch (getKind()) {
  case Target:
    return cast<MCTargetExpr>(this)->EvaluateAsRelocatableImpl(Res, Layout);

  case Constant:
    Res = MCValue::get(cast<MCConstantExpr>(this)->getValue());
    return true;

  case SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(this);
    const MCSymbol &Sym = SRE->getSymbol();

    // Evaluate recursively if this is a variable.
    if (Sym.isVariable() && SRE->getKind() == MCSymbolRefExpr::VK_None) {
      bool Ret = Sym.getVariableValue()->EvaluateAsRelocatableImpl(Res, Layout,
                                                                   Addrs, true);
      // If we failed to simplify this to a constant, let the target
      // handle it.
      if (Ret && !Res.getSymA() && !Res.getSymB())
        return true;
    }

    Res = MCValue::get(SRE, 0, 0);
    return true;
  }

  case Unary: {
    const MCUnaryExpr *AUE = cast<MCUnaryExpr>(this);
    MCValue Value;

    if (!AUE->getSubExpr()->EvaluateAsRelocatableImpl(Value, Layout,
                                                      Addrs, InSet))
      return false;

    switch (AUE->getOpcode()) {
    case MCUnaryExpr::LNot:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(!Value.getConstant());
      break;
    case MCUnaryExpr::Minus:
      /// -(a - b + const) ==> (b - a - const)
      if (Value.getSymA() && !Value.getSymB())
        return false;
      Res = MCValue::get(Value.getSymB(), Value.getSymA(),
                         -Value.getConstant());
      break;
    case MCUnaryExpr::Not:
      if (!Value.isAbsolute())
        return false;
      Res = MCValue::get(~Value.getConstant());
      break;
    case MCUnaryExpr::Plus:
      Res = Value;
      break;
    }

    return true;
  }

  case Binary: {
    const MCBinaryExpr *ABE = cast<MCBinaryExpr>(this);
    MCValue LHSValue, RHSValue;

    if (!ABE->getLHS()->EvaluateAsRelocatableImpl(LHSValue, Layout,
                                                  Addrs, InSet) ||
        !ABE->getRHS()->EvaluateAsRelocatableImpl(RHSValue, Layout,
                                                  Addrs, InSet))
      return false;

    // We only support a few operations on non-constant expressions, handle
    // those first.
    if (!LHSValue.isAbsolute() || !RHSValue.isAbsolute()) {
      switch (ABE->getOpcode()) {
      default:
        return false;
      case MCBinaryExpr::Sub:
        // Negate RHS and add.
        return EvaluateSymbolicAdd(Layout, Addrs, InSet, LHSValue,
                                   RHSValue.getSymB(), RHSValue.getSymA(),
                                   -RHSValue.getConstant(),
                                   Res);

      case MCBinaryExpr::Add:
        return EvaluateSymbolicAdd(Layout, Addrs, InSet, LHSValue,
                                   RHSValue.getSymA(), RHSValue.getSymB(),
                                   RHSValue.getConstant(),
                                   Res);
      }
    }

    // FIXME: We need target hooks for the evaluation. It may be limited in
    // width, and gas defines the result of comparisons and right shifts
    // differently from Apple as.
    int64_t LHS = LHSValue.getConstant(), RHS = RHSValue.getConstant();
    int64_t Result = 0;
    switch (ABE->getOpcode()) {
    case MCBinaryExpr::Add:  Result = LHS + RHS; break;
    case MCBinaryExpr::And:  Result = LHS & RHS; break;
    case MCBinaryExpr::Div:  Result = LHS / RHS; break;
    case MCBinaryExpr::EQ:   Result = LHS == RHS; break;
    case MCBinaryExpr::GT:   Result = LHS > RHS; break;
    case MCBinaryExpr::GTE:  Result = LHS >= RHS; break;
    case MCBinaryExpr::LAnd: Result = LHS && RHS; break;
    case MCBinaryExpr::LOr:  Result = LHS || RHS; break;
    case MCBinaryExpr::LT:   Result = LHS < RHS; break;
    case MCBinaryExpr::LTE:  Result = LHS <= RHS; break;
    case MCBinaryExpr::Mod:  Result = LHS % RHS; break;
    case MCBinaryExpr::Mul:  Result = LHS * RHS; break;
    case MCBinaryExpr::NE:   Result = LHS != RHS; break;
    case MCBinaryExpr::Or:   Result = LHS | RHS; break;
    case MCBinaryExpr::Shl:  Result = LHS << RHS; break;
    case MCBinaryExpr::Shr:  Result = LHS >> RHS; break;
    case MCBinaryExpr::Sub:  Result = LHS - RHS; break;
    case MCBinaryExpr::Xor:  Result = LHS ^ RHS; break;
    }

    Res = MCValue::get(Result);
    return true;
  }
  }

  assert(0 && "Invalid assembly expression kind!");
  return false;
}

//===-- SparcV8AsmPrinter.cpp - SparcV8 LLVM assembly writer --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format Sparc V8 assembly language.
//
//===----------------------------------------------------------------------===//

#include "SparcV8.h"
#include "SparcV8InstrInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include <cctype>
using namespace llvm;

namespace {
  Statistic<> EmittedInsts("asm-printer", "Number of machine instrs printed");

  struct V8Printer : public MachineFunctionPass {
    /// Output stream on which we're printing assembly code.
    ///
    std::ostream &O;

    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    TargetMachine &TM;

    /// Name-mangler for global names.
    ///
    Mangler *Mang;

    V8Printer(std::ostream &o, TargetMachine &tm) : O(o), TM(tm) { }

    /// We name each basic block in a Function with a unique number, so
    /// that we can consistently refer to them later. This is cleared
    /// at the beginning of each call to runOnMachineFunction().
    ///
    typedef std::map<const Value *, unsigned> ValueMapTy;
    ValueMapTy NumberForBB;

    /// Cache of mangled name for current function. This is
    /// recalculated at the beginning of each call to
    /// runOnMachineFunction().
    ///
    std::string CurrentFnName;

    virtual const char *getPassName() const {
      return "SparcV8 Assembly Printer";
    }

    void emitConstantValueOnly(const Constant *CV);
    void emitGlobalConstant(const Constant *CV);
    void printConstantPool(MachineConstantPool *MCP);
    void printOperand(const MachineInstr *MI, int opNum);
    void printBaseOffsetPair (const MachineInstr *MI, int i, bool brackets=true);
    void printMachineInstruction(const MachineInstr *MI);
    bool runOnMachineFunction(MachineFunction &F);
    bool doInitialization(Module &M);
    bool doFinalization(Module &M);
  };
} // end of anonymous namespace

/// createSparcV8CodePrinterPass - Returns a pass that prints the SparcV8
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.  This should work
/// regardless of whether the function is in SSA form.
///
FunctionPass *llvm::createSparcV8CodePrinterPass (std::ostream &o,
                                                  TargetMachine &tm) {
  return new V8Printer(o, tm);
}

/// toOctal - Convert the low order bits of X into an octal digit.
///
static inline char toOctal(int X) {
  return (X&7)+'0';
}

/// getAsCString - Return the specified array as a C compatible
/// string, only if the predicate isStringCompatible is true.
///
static void printAsCString(std::ostream &O, const ConstantArray *CVA) {
  assert(CVA->isString() && "Array is not string compatible!");

  O << "\"";
  for (unsigned i = 0; i != CVA->getNumOperands(); ++i) {
    unsigned char C = cast<ConstantInt>(CVA->getOperand(i))->getRawValue();

    if (C == '"') {
      O << "\\\"";
    } else if (C == '\\') {
      O << "\\\\";
    } else if (isprint(C)) {
      O << C;
    } else {
      switch(C) {
      case '\b': O << "\\b"; break;
      case '\f': O << "\\f"; break;
      case '\n': O << "\\n"; break;
      case '\r': O << "\\r"; break;
      case '\t': O << "\\t"; break;
      default:
        O << '\\';
        O << toOctal(C >> 6);
        O << toOctal(C >> 3);
        O << toOctal(C >> 0);
        break;
      }
    }
  }
  O << "\"";
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void V8Printer::emitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue() || isa<UndefValue> (CV))
    O << "0";
  else if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    assert(CB == ConstantBool::True);
    O << "1";
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
    if (((CI->getValue() << 32) >> 32) == CI->getValue())
      O << CI->getValue();
    else
      O << (unsigned long long)CI->getValue();
  else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
    O << CI->getValue();
  else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV))
    // This is a constant address for a global variable or function.  Use the
    // name of the variable or function as the address value.
    O << Mang->getValueName(GV);
  else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData &TD = TM.getTargetData();
    switch(CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      if (unsigned Offset = TD.getIndexedOffset(ptrVal->getType(), idxVec)) {
        O << "(";
        emitConstantValueOnly(ptrVal);
        O << ") + " << Offset;
      } else {
        emitConstantValueOnly(ptrVal);
      }
      break;
    }
    case Instruction::Cast: {
      // Support only non-converting or widening casts for now, that is, ones
      // that do not involve a change in value.  This assertion is really gross,
      // and may not even be a complete check.
      Constant *Op = CE->getOperand(0);
      const Type *OpTy = Op->getType(), *Ty = CE->getType();

      // Pointers on ILP32 machines can be losslessly converted back and
      // forth into 32-bit or wider integers, regardless of signedness.
      assert(((isa<PointerType>(OpTy)
               && (Ty == Type::LongTy || Ty == Type::ULongTy
                   || Ty == Type::IntTy || Ty == Type::UIntTy))
              || (isa<PointerType>(Ty)
                  && (OpTy == Type::LongTy || OpTy == Type::ULongTy
                      || OpTy == Type::IntTy || OpTy == Type::UIntTy))
              || (((TD.getTypeSize(Ty) >= TD.getTypeSize(OpTy))
                   && OpTy->isLosslesslyConvertibleTo(Ty))))
             && "FIXME: Don't yet support this kind of constant cast expr");
      O << "(";
      emitConstantValueOnly(Op);
      O << ")";
      break;
    }
    case Instruction::Add:
      O << "(";
      emitConstantValueOnly(CE->getOperand(0));
      O << ") + (";
      emitConstantValueOnly(CE->getOperand(1));
      O << ")";
      break;
    default:
      assert(0 && "Unsupported operator!");
    }
  } else {
    assert(0 && "Unknown constant value!");
  }
}

// Print a constant value or values, with the appropriate storage class as a
// prefix.
void V8Printer::emitGlobalConstant(const Constant *CV) {
  const TargetData &TD = TM.getTargetData();

  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (CVA->isString()) {
      O << "\t.ascii\t";
      printAsCString(O, CVA);
      O << "\n";
    } else { // Not a string.  Print the values in successive locations
      for (unsigned i = 0, e = CVA->getNumOperands(); i != e; i++)
        emitGlobalConstant(CVA->getOperand(i));
    }
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    // Print the fields in successive locations. Pad to align if needed!
    const StructLayout *cvsLayout = TD.getStructLayout(CVS->getType());
    unsigned sizeSoFar = 0;
    for (unsigned i = 0, e = CVS->getNumOperands(); i != e; i++) {
      const Constant* field = CVS->getOperand(i);

      // Check if padding is needed and insert one or more 0s.
      unsigned fieldSize = TD.getTypeSize(field->getType());
      unsigned padSize = ((i == e-1? cvsLayout->StructSize
                           : cvsLayout->MemberOffsets[i+1])
                          - cvsLayout->MemberOffsets[i]) - fieldSize;
      sizeSoFar += fieldSize + padSize;

      // Now print the actual field value
      emitGlobalConstant(field);

      // Insert the field padding unless it's zero bytes...
      if (padSize)
        O << "\t.skip\t " << padSize << "\n";
    }
    assert(sizeSoFar == cvsLayout->StructSize &&
           "Layout of constant struct may be incorrect!");
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    // FP Constants are printed as integer constants to avoid losing
    // precision...
    double Val = CFP->getValue();
    switch (CFP->getType()->getTypeID()) {
    default: assert(0 && "Unknown floating point type!");
    case Type::FloatTyID: {
      O << ".long\t" << FloatToBits(Val) << "\t! float " << Val << "\n";
      return;
    }
    case Type::DoubleTyID: {
      O << ".word\t0x" << std::hex << (DoubleToBits(Val) >> 32) << std::dec << "\t! double " << Val << "\n";
      O << ".word\t0x" << std::hex << (DoubleToBits(Val) & 0xffffffffUL) << std::dec << "\t! double " << Val << "\n";
      return;
    }
    }
  } else if (isa<UndefValue> (CV)) {
    unsigned size = TD.getTypeSize (CV->getType ());
    O << "\t.skip\t " << size << "\n";
    return;
  } else if (isa<ConstantAggregateZero> (CV)) {
    unsigned size = TD.getTypeSize (CV->getType ());
    for (unsigned i = 0; i < size; ++i)
      O << "\t.byte 0\n";
    return;
  }

  const Type *type = CV->getType();
  O << "\t";
  switch (type->getTypeID()) {
  case Type::BoolTyID: case Type::UByteTyID: case Type::SByteTyID:
    O << ".byte";
    break;
  case Type::UShortTyID: case Type::ShortTyID:
    O << ".half";
    break;
  case Type::FloatTyID: case Type::PointerTyID:
  case Type::UIntTyID: case Type::IntTyID:
    O << ".word";
    break;
  case Type::DoubleTyID:
  case Type::ULongTyID: case Type::LongTyID:
    O << ".xword";
    break;
  default:
    assert (0 && "Can't handle printing this type of thing");
    break;
  }
  O << "\t";
  emitConstantValueOnly(CV);
  O << "\n";
}

/// printConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void V8Printer::printConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  const TargetData &TD = TM.getTargetData();

  if (CP.empty()) return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << "\t.section \".rodata\"\n";
    O << "\t.align " << (unsigned)TD.getTypeAlignment(CP[i]->getType())
      << "\n";
    O << ".CPI" << CurrentFnName << "_" << i << ":\t\t\t\t\t!"
      << *CP[i] << "\n";
    emitGlobalConstant(CP[i]);
  }
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool V8Printer::runOnMachineFunction(MachineFunction &MF) {
  // BBNumber is used here so that a given Printer will never give two
  // BBs the same name. (If you have a better way, please let me know!)
  static unsigned BBNumber = 0;

  O << "\n\n";
  // What's my mangled name?
  CurrentFnName = Mang->getValueName(MF.getFunction());

  // Print out constants referenced by the function
  printConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  O << "\t.text\n";
  O << "\t.align 16\n";
  O << "\t.globl\t" << CurrentFnName << "\n";
  O << "\t.type\t" << CurrentFnName << ", #function\n";
  O << CurrentFnName << ":\n";

  // Number each basic block so that we can consistently refer to them
  // in PC-relative references.
  NumberForBB.clear();
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    NumberForBB[I->getBasicBlock()] = BBNumber++;
  }

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    O << ".LBB" << Mang->getValueName(MF.getFunction ())
      << "_" << I->getNumber () << ":\t! "
      << I->getBasicBlock ()->getName () << "\n";
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      O << "\t";
      printMachineInstruction(II);
    }
  }

  // We didn't modify anything.
  return false;
}

void V8Printer::printOperand(const MachineInstr *MI, int opNum) {
  const MachineOperand &MO = MI->getOperand (opNum);
  const MRegisterInfo &RI = *TM.getRegisterInfo();
  bool CloseParen = false;
  if (MI->getOpcode() == V8::SETHIi && !MO.isRegister() && !MO.isImmediate()) {
    O << "%hi(";
    CloseParen = true;
  } else if (MI->getOpcode() ==V8::ORri &&!MO.isRegister() &&!MO.isImmediate())
  {
    O << "%lo(";
    CloseParen = true;
  }
  switch (MO.getType()) {
  case MachineOperand::MO_VirtualRegister:
    if (Value *V = MO.getVRegValueOrNull()) {
      O << "<" << V->getName() << ">";
      break;
    }
    // FALLTHROUGH
  case MachineOperand::MO_MachineRegister:
    if (MRegisterInfo::isPhysicalRegister(MO.getReg()))
      O << "%" << LowercaseString (RI.get(MO.getReg()).Name);
    else
      O << "%reg" << MO.getReg();
    break;

  case MachineOperand::MO_SignExtendedImmed:
  case MachineOperand::MO_UnextendedImmed:
    O << (int)MO.getImmedValue();
    break;
  case MachineOperand::MO_MachineBasicBlock: {
    MachineBasicBlock *MBBOp = MO.getMachineBasicBlock();
    O << ".LBB" << Mang->getValueName(MBBOp->getParent()->getFunction())
      << "_" << MBBOp->getNumber () << "\t! "
      << MBBOp->getBasicBlock ()->getName ();
    return;
  }
  case MachineOperand::MO_PCRelativeDisp:
    std::cerr << "Shouldn't use addPCDisp() when building SparcV8 MachineInstrs";
    abort ();
    return;
  case MachineOperand::MO_GlobalAddress:
    O << Mang->getValueName(MO.getGlobal());
    break;
  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    O << ".CPI" << CurrentFnName << "_" << MO.getConstantPoolIndex();
    break;
  default:
    O << "<unknown operand type>"; abort (); break;
  }
  if (CloseParen) O << ")";
}

static bool isLoadInstruction (const MachineInstr *MI) {
  switch (MI->getOpcode ()) {
  case V8::LDSB:
  case V8::LDSH:
  case V8::LDUB:
  case V8::LDUH:
  case V8::LD:
  case V8::LDD:
  case V8::LDFrr:
  case V8::LDFri:
  case V8::LDDFrr:
  case V8::LDDFri:
    return true;
  default:
    return false;
  }
}

static bool isStoreInstruction (const MachineInstr *MI) {
  switch (MI->getOpcode ()) {
  case V8::STB:
  case V8::STH:
  case V8::ST:
  case V8::STD:
  case V8::STFrr:
  case V8::STFri:
  case V8::STDFrr:
  case V8::STDFri:
    return true;
  default:
    return false;
  }
}

static bool isPseudoInstruction (const MachineInstr *MI) {
  switch (MI->getOpcode ()) {
  case V8::PHI:
  case V8::ADJCALLSTACKUP:
  case V8::ADJCALLSTACKDOWN:
  case V8::IMPLICIT_USE:
  case V8::IMPLICIT_DEF:
    return true;
  default:
    return false;
  }
}

/// printBaseOffsetPair - Print two consecutive operands of MI, starting at #i,
/// which form a base + offset pair (which may have brackets around it, if
/// brackets is true, or may be in the form base - constant, if offset is a
/// negative constant).
///
void V8Printer::printBaseOffsetPair (const MachineInstr *MI, int i,
                                     bool brackets) {
  if (brackets) O << "[";
  printOperand (MI, i);
  if (MI->getOperand (i + 1).isImmediate()) {
    int Val = (int) MI->getOperand (i + 1).getImmedValue ();
    if (Val != 0) {
      O << ((Val >= 0) ? " + " : " - ");
      O << ((Val >= 0) ? Val : -Val);
    }
  } else {
    O << " + ";
    printOperand (MI, i + 1);
  }
  if (brackets) O << "]";
}

/// printMachineInstruction -- Print out a single SparcV8 LLVM instruction
/// MI in GAS syntax to the current output stream.
///
void V8Printer::printMachineInstruction(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const TargetInstrDescriptor &Desc = TII.get(Opcode);

  // If it's a pseudo-instruction, comment it out.
  if (isPseudoInstruction (MI))
    O << "! ";

  O << Desc.Name << " ";

  // Printing memory instructions is a special case.
  // for loads:  %dest = op %base, offset --> op [%base + offset], %dest
  // for stores: op %base, offset, %src   --> op %src, [%base + offset]
  if (isLoadInstruction (MI)) {
    printBaseOffsetPair (MI, 1);
    O << ", ";
    printOperand (MI, 0);
    O << "\n";
    return;
  } else if (isStoreInstruction (MI)) {
    printOperand (MI, 2);
    O << ", ";
    printBaseOffsetPair (MI, 0);
    O << "\n";
    return;
  } else if (Opcode == V8::JMPLrr) {
    printBaseOffsetPair (MI, 1, false);
    O << ", ";
    printOperand (MI, 0);
    O << "\n";
    return;
  }

  // print non-immediate, non-register-def operands
  // then print immediate operands
  // then print register-def operands.
  std::vector<int> print_order;
  for (unsigned i = 0; i < MI->getNumOperands (); ++i)
    if (!(MI->getOperand (i).isImmediate ()
          || (MI->getOperand (i).isRegister ()
              && MI->getOperand (i).isDef ())))
      print_order.push_back (i);
  for (unsigned i = 0; i < MI->getNumOperands (); ++i)
    if (MI->getOperand (i).isImmediate ())
      print_order.push_back (i);
  for (unsigned i = 0; i < MI->getNumOperands (); ++i)
    if (MI->getOperand (i).isRegister () && MI->getOperand (i).isDef ())
      print_order.push_back (i);
  for (unsigned i = 0, e = print_order.size (); i != e; ++i) {
    printOperand (MI, print_order[i]);
    if (i != (print_order.size () - 1))
      O << ", ";
  }
  O << "\n";
}

bool V8Printer::doInitialization(Module &M) {
  Mang = new Mangler(M);
  return false; // success
}

// SwitchSection - Switch to the specified section of the executable if we are
// not already in it!
//
static void SwitchSection(std::ostream &OS, std::string &CurSection,
                          const char *NewSection) {
  if (CurSection != NewSection) {
    CurSection = NewSection;
    if (!CurSection.empty())
      OS << "\t.section \"" << NewSection << "\"\n";
  }
}

bool V8Printer::doFinalization(Module &M) {
  const TargetData &TD = TM.getTargetData();
  std::string CurSection;

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
    if (I->hasInitializer()) {   // External global require no code
      O << "\n\n";
      std::string name = Mang->getValueName(I);
      Constant *C = I->getInitializer();
      unsigned Size = TD.getTypeSize(C->getType());
      unsigned Align = TD.getTypeAlignment(C->getType());

      if (C->isNullValue() &&
          (I->hasLinkOnceLinkage() || I->hasInternalLinkage() ||
           I->hasWeakLinkage() /* FIXME: Verify correct */)) {
        SwitchSection(O, CurSection, ".data");
        if (I->hasInternalLinkage())
          O << "\t.local " << name << "\n";

        O << "\t.comm " << name << "," << TD.getTypeSize(C->getType())
          << "," << (unsigned)TD.getTypeAlignment(C->getType());
        O << "\t\t! ";
        WriteAsOperand(O, I, true, true, &M);
        O << "\n";
      } else {
        switch (I->getLinkage()) {
        case GlobalValue::LinkOnceLinkage:
        case GlobalValue::WeakLinkage:   // FIXME: Verify correct for weak.
          // Nonnull linkonce -> weak
          O << "\t.weak " << name << "\n";
          SwitchSection(O, CurSection, "");
          O << "\t.section\t\".llvm.linkonce.d." << name << "\",\"aw\",@progbits\n";
          break;

        case GlobalValue::AppendingLinkage:
          // FIXME: appending linkage variables should go into a section of
          // their name or something.  For now, just emit them as external.
        case GlobalValue::ExternalLinkage:
          // If external or appending, declare as a global symbol
          O << "\t.globl " << name << "\n";
          // FALL THROUGH
        case GlobalValue::InternalLinkage:
          if (C->isNullValue())
            SwitchSection(O, CurSection, ".bss");
          else
            SwitchSection(O, CurSection, ".data");
          break;
        case GlobalValue::GhostLinkage:
          std::cerr << "Should not have any unmaterialized functions!\n";
          abort();
        }

        O << "\t.align " << Align << "\n";
        O << "\t.type " << name << ",#object\n";
        O << "\t.size " << name << "," << Size << "\n";
        O << name << ":\t\t\t\t! ";
        WriteAsOperand(O, I, true, true, &M);
        O << " = ";
        WriteAsOperand(O, C, false, false, &M);
        O << "\n";
        emitGlobalConstant(C);
      }
    }

  delete Mang;
  return false; // success
}

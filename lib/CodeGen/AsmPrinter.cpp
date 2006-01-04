//===-- AsmPrinter.cpp - Common AsmPrinter code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AsmPrinter class.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <iostream>
using namespace llvm;

AsmPrinter::AsmPrinter(std::ostream &o, TargetMachine &tm)
: FunctionNumber(0), O(o), TM(tm),
  CommentString("#"),
  GlobalPrefix(""),
  PrivateGlobalPrefix("."),
  GlobalVarAddrPrefix(""),
  GlobalVarAddrSuffix(""),
  FunctionAddrPrefix(""),
  FunctionAddrSuffix(""),
  ZeroDirective("\t.zero\t"),
  AsciiDirective("\t.ascii\t"),
  AscizDirective("\t.asciz\t"),
  Data8bitsDirective("\t.byte\t"),
  Data16bitsDirective("\t.short\t"),
  Data32bitsDirective("\t.long\t"),
  Data64bitsDirective("\t.quad\t"),
  AlignDirective("\t.align\t"),
  AlignmentIsInBytes(true),
  SwitchToSectionDirective("\t.section\t"),
  ConstantPoolSection("\t.section .rodata\n"),
  StaticCtorsSection("\t.section .ctors,\"aw\",@progbits"),
  StaticDtorsSection("\t.section .dtors,\"aw\",@progbits"),
  LCOMMDirective(0),
  COMMDirective("\t.comm\t"),
  COMMDirectiveTakesAlignment(true),
  HasDotTypeDotSizeDirective(true) {
}


/// SwitchSection - Switch to the specified section of the executable if we
/// are not already in it!
///
void AsmPrinter::SwitchSection(const char *NewSection, const GlobalValue *GV) {
  std::string NS;
  
  if (GV && GV->hasSection())
    NS = SwitchToSectionDirective + GV->getSection();
  else
    NS = std::string("\t")+NewSection;
  
  if (CurrentSection != NS) {
    CurrentSection = NS;
    if (!CurrentSection.empty())
      O << CurrentSection << '\n';
  }
}

bool AsmPrinter::doInitialization(Module &M) {
  Mang = new Mangler(M, GlobalPrefix);
  SwitchSection("", 0);   // Reset back to no section.
  return false;
}

bool AsmPrinter::doFinalization(Module &M) {
  delete Mang; Mang = 0;
  return false;
}

void AsmPrinter::SetupMachineFunction(MachineFunction &MF) {
  // What's my mangled name?
  CurrentFnName = Mang->getValueName(MF.getFunction());
  IncrementFunctionNumber();
}

/// EmitConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void AsmPrinter::EmitConstantPool(MachineConstantPool *MCP) {
  const std::vector<Constant*> &CP = MCP->getConstants();
  if (CP.empty()) return;
  const TargetData &TD = TM.getTargetData();
  
  SwitchSection(ConstantPoolSection, 0);
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    // FIXME: force doubles to be naturally aligned.  We should handle this
    // more correctly in the future.
    unsigned Alignment = TD.getTypeAlignmentShift(CP[i]->getType());
    if (CP[i]->getType() == Type::DoubleTy && Alignment < 3) Alignment = 3;
    
    EmitAlignment(Alignment);
    O << PrivateGlobalPrefix << "CPI" << getFunctionNumber() << '_' << i
      << ":\t\t\t\t\t" << CommentString << *CP[i] << '\n';
    EmitGlobalConstant(CP[i]);
  }
}

/// EmitSpecialLLVMGlobal - Check to see if the specified global is a
/// special global used by LLVM.  If so, emit it and return true, otherwise
/// do nothing and return false.
bool AsmPrinter::EmitSpecialLLVMGlobal(const GlobalVariable *GV) {
  assert(GV->hasInitializer() && GV->hasAppendingLinkage() &&
         "Not a special LLVM global!");
  
  if (GV->getName() == "llvm.used")
    return true;  // No need to emit this at all.

  if (GV->getName() == "llvm.global_ctors") {
    SwitchSection(StaticCtorsSection, 0);
    EmitAlignment(2, 0);
    EmitXXStructorList(GV->getInitializer());
    return true;
  } 
  
  if (GV->getName() == "llvm.global_dtors") {
    SwitchSection(StaticDtorsSection, 0);
    EmitAlignment(2, 0);
    EmitXXStructorList(GV->getInitializer());
    return true;
  }
  
  return false;
}

/// EmitXXStructorList - Emit the ctor or dtor list.  This just prints out the 
/// function pointers, ignoring the init priority.
void AsmPrinter::EmitXXStructorList(Constant *List) {
  // Should be an array of '{ int, void ()* }' structs.  The first value is the
  // init priority, which we ignore.
  if (!isa<ConstantArray>(List)) return;
  ConstantArray *InitList = cast<ConstantArray>(List);
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))){
      if (CS->getNumOperands() != 2) return;  // Not array of 2-element structs.

      if (CS->getOperand(1)->isNullValue())
        return;  // Found a null terminator, exit printing.
      // Emit the function pointer.
      EmitGlobalConstant(CS->getOperand(1));
    }
}


// EmitAlignment - Emit an alignment directive to the specified power of two.
void AsmPrinter::EmitAlignment(unsigned NumBits, const GlobalValue *GV) const {
  if (GV && GV->getAlignment())
    NumBits = Log2_32(GV->getAlignment());
  if (NumBits == 0) return;   // No need to emit alignment.
  if (AlignmentIsInBytes) NumBits = 1 << NumBits;
  O << AlignDirective << NumBits << "\n";
}

/// EmitZeros - Emit a block of zeros.
///
void AsmPrinter::EmitZeros(uint64_t NumZeros) const {
  if (NumZeros) {
    if (ZeroDirective)
      O << ZeroDirective << NumZeros << "\n";
    else {
      for (; NumZeros; --NumZeros)
        O << Data8bitsDirective << "0\n";
    }
  }
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void AsmPrinter::EmitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue() || isa<UndefValue>(CV))
    O << "0";
  else if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    assert(CB == ConstantBool::True);
    O << "1";
  } else if (const ConstantSInt *CI = dyn_cast<ConstantSInt>(CV))
    if (((CI->getValue() << 32) >> 32) == CI->getValue())
      O << CI->getValue();
    else
      O << (uint64_t)CI->getValue();
  else if (const ConstantUInt *CI = dyn_cast<ConstantUInt>(CV))
    O << CI->getValue();
  else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV)) {
    // This is a constant address for a global variable or function. Use the
    // name of the variable or function as the address value, possibly
    // decorating it with GlobalVarAddrPrefix/Suffix or
    // FunctionAddrPrefix/Suffix (these all default to "" )
    if (isa<Function>(GV))
      O << FunctionAddrPrefix << Mang->getValueName(GV) << FunctionAddrSuffix;
    else
      O << GlobalVarAddrPrefix << Mang->getValueName(GV) << GlobalVarAddrSuffix;
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData &TD = TM.getTargetData();
    switch(CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      if (int64_t Offset = TD.getIndexedOffset(ptrVal->getType(), idxVec)) {
        if (Offset)
          O << "(";
        EmitConstantValueOnly(ptrVal);
        if (Offset > 0)
          O << ") + " << Offset;
        else if (Offset < 0)
          O << ") - " << -Offset;
      } else {
        EmitConstantValueOnly(ptrVal);
      }
      break;
    }
    case Instruction::Cast: {
      // Support only non-converting or widening casts for now, that is, ones
      // that do not involve a change in value.  This assertion is really gross,
      // and may not even be a complete check.
      Constant *Op = CE->getOperand(0);
      const Type *OpTy = Op->getType(), *Ty = CE->getType();

      // Remember, kids, pointers can be losslessly converted back and forth
      // into 32-bit or wider integers, regardless of signedness. :-P
      assert(((isa<PointerType>(OpTy)
               && (Ty == Type::LongTy || Ty == Type::ULongTy
                   || Ty == Type::IntTy || Ty == Type::UIntTy))
              || (isa<PointerType>(Ty)
                  && (OpTy == Type::LongTy || OpTy == Type::ULongTy
                      || OpTy == Type::IntTy || OpTy == Type::UIntTy))
              || (((TD.getTypeSize(Ty) >= TD.getTypeSize(OpTy))
                   && OpTy->isLosslesslyConvertibleTo(Ty))))
             && "FIXME: Don't yet support this kind of constant cast expr");
      EmitConstantValueOnly(Op);
      break;
    }
    case Instruction::Add:
      O << "(";
      EmitConstantValueOnly(CE->getOperand(0));
      O << ") + (";
      EmitConstantValueOnly(CE->getOperand(1));
      O << ")";
      break;
    default:
      assert(0 && "Unsupported operator!");
    }
  } else {
    assert(0 && "Unknown constant value!");
  }
}

/// toOctal - Convert the low order bits of X into an octal digit.
///
static inline char toOctal(int X) {
  return (X&7)+'0';
}

/// printAsCString - Print the specified array as a C compatible string, only if
/// the predicate isString is true.
///
static void printAsCString(std::ostream &O, const ConstantArray *CVA,
                           unsigned LastElt) {
  assert(CVA->isString() && "Array is not string compatible!");

  O << "\"";
  for (unsigned i = 0; i != LastElt; ++i) {
    unsigned char C =
        (unsigned char)cast<ConstantInt>(CVA->getOperand(i))->getRawValue();

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

/// EmitGlobalConstant - Print a general LLVM constant to the .s file.
///
void AsmPrinter::EmitGlobalConstant(const Constant *CV) {
  const TargetData &TD = TM.getTargetData();

  if (CV->isNullValue() || isa<UndefValue>(CV)) {
    EmitZeros(TD.getTypeSize(CV->getType()));
    return;
  } else if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (CVA->isString()) {
      unsigned NumElts = CVA->getNumOperands();
      if (AscizDirective && NumElts && 
          cast<ConstantInt>(CVA->getOperand(NumElts-1))->getRawValue() == 0) {
        O << AscizDirective;
        printAsCString(O, CVA, NumElts-1);
      } else {
        O << AsciiDirective;
        printAsCString(O, CVA, NumElts);
      }
      O << "\n";
    } else { // Not a string.  Print the values in successive locations
      for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
        EmitGlobalConstant(CVA->getOperand(i));
    }
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    // Print the fields in successive locations. Pad to align if needed!
    const StructLayout *cvsLayout = TD.getStructLayout(CVS->getType());
    uint64_t sizeSoFar = 0;
    for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
      const Constant* field = CVS->getOperand(i);

      // Check if padding is needed and insert one or more 0s.
      uint64_t fieldSize = TD.getTypeSize(field->getType());
      uint64_t padSize = ((i == e-1? cvsLayout->StructSize
                           : cvsLayout->MemberOffsets[i+1])
                          - cvsLayout->MemberOffsets[i]) - fieldSize;
      sizeSoFar += fieldSize + padSize;

      // Now print the actual field value
      EmitGlobalConstant(field);

      // Insert the field padding unless it's zero bytes...
      EmitZeros(padSize);
    }
    assert(sizeSoFar == cvsLayout->StructSize &&
           "Layout of constant struct may be incorrect!");
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    // FP Constants are printed as integer constants to avoid losing
    // precision...
    double Val = CFP->getValue();
    if (CFP->getType() == Type::DoubleTy) {
      if (Data64bitsDirective)
        O << Data64bitsDirective << DoubleToBits(Val) << "\t" << CommentString
          << " double value: " << Val << "\n";
      else if (TD.isBigEndian()) {
        O << Data32bitsDirective << unsigned(DoubleToBits(Val) >> 32)
          << "\t" << CommentString << " double most significant word "
          << Val << "\n";
        O << Data32bitsDirective << unsigned(DoubleToBits(Val))
          << "\t" << CommentString << " double least significant word "
          << Val << "\n";
      } else {
        O << Data32bitsDirective << unsigned(DoubleToBits(Val))
          << "\t" << CommentString << " double least significant word " << Val
          << "\n";
        O << Data32bitsDirective << unsigned(DoubleToBits(Val) >> 32)
          << "\t" << CommentString << " double most significant word " << Val
          << "\n";
      }
      return;
    } else {
      O << Data32bitsDirective << FloatToBits(Val) << "\t" << CommentString
        << " float " << Val << "\n";
      return;
    }
  } else if (CV->getType() == Type::ULongTy || CV->getType() == Type::LongTy) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
      uint64_t Val = CI->getRawValue();

      if (Data64bitsDirective)
        O << Data64bitsDirective << Val << "\n";
      else if (TD.isBigEndian()) {
        O << Data32bitsDirective << unsigned(Val >> 32)
          << "\t" << CommentString << " Double-word most significant word "
          << Val << "\n";
        O << Data32bitsDirective << unsigned(Val)
          << "\t" << CommentString << " Double-word least significant word "
          << Val << "\n";
      } else {
        O << Data32bitsDirective << unsigned(Val)
          << "\t" << CommentString << " Double-word least significant word "
          << Val << "\n";
        O << Data32bitsDirective << unsigned(Val >> 32)
          << "\t" << CommentString << " Double-word most significant word "
          << Val << "\n";
      }
      return;
    }
  } else if (const ConstantPacked *CP = dyn_cast<ConstantPacked>(CV)) {
    const PackedType *PTy = CP->getType();
    
    for (unsigned I = 0, E = PTy->getNumElements(); I < E; ++I)
      EmitGlobalConstant(CP->getOperand(I));
    
    return;
  }

  const Type *type = CV->getType();
  switch (type->getTypeID()) {
  case Type::BoolTyID:
  case Type::UByteTyID: case Type::SByteTyID:
    O << Data8bitsDirective;
    break;
  case Type::UShortTyID: case Type::ShortTyID:
    O << Data16bitsDirective;
    break;
  case Type::PointerTyID:
    if (TD.getPointerSize() == 8) {
      O << Data64bitsDirective;
      break;
    }
    //Fall through for pointer size == int size
  case Type::UIntTyID: case Type::IntTyID:
    O << Data32bitsDirective;
    break;
  case Type::ULongTyID: case Type::LongTyID:
    assert(Data64bitsDirective &&"Target cannot handle 64-bit constant exprs!");
    O << Data64bitsDirective;
    break;
  case Type::FloatTyID: case Type::DoubleTyID:
    assert (0 && "Should have already output floating point constant.");
  default:
    assert (0 && "Can't handle printing this type of thing");
    break;
  }
  EmitConstantValueOnly(CV);
  O << "\n";
}

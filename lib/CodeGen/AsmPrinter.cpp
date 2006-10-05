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

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include <iostream>
#include <cerrno>
using namespace llvm;

AsmPrinter::AsmPrinter(std::ostream &o, TargetMachine &tm,
                       const TargetAsmInfo *T)
: FunctionNumber(0), O(o), TM(tm), TAI(T)
{}

std::string AsmPrinter::getSectionForFunction(const Function &F) const {
  return TAI->getTextSection();
}


/// SwitchToTextSection - Switch to the specified text section of the executable
/// if we are not already in it!
///
void AsmPrinter::SwitchToTextSection(const char *NewSection,
                                     const GlobalValue *GV) {
  std::string NS;
  if (GV && GV->hasSection())
    NS = TAI->getSwitchToSectionDirective() + GV->getSection();
  else
    NS = NewSection;
  
  // If we're already in this section, we're done.
  if (CurrentSection == NS) return;

  // Close the current section, if applicable.
  if (TAI->getSectionEndDirectiveSuffix() && !CurrentSection.empty())
    O << CurrentSection << TAI->getSectionEndDirectiveSuffix() << "\n";

  CurrentSection = NS;

  if (!CurrentSection.empty())
    O << CurrentSection << TAI->getTextSectionStartSuffix() << '\n';
}

/// SwitchToDataSection - Switch to the specified data section of the executable
/// if we are not already in it!
///
void AsmPrinter::SwitchToDataSection(const char *NewSection,
                                     const GlobalValue *GV) {
  std::string NS;
  if (GV && GV->hasSection())
    NS = TAI->getSwitchToSectionDirective() + GV->getSection();
  else
    NS = NewSection;
  
  // If we're already in this section, we're done.
  if (CurrentSection == NS) return;

  // Close the current section, if applicable.
  if (TAI->getSectionEndDirectiveSuffix() && !CurrentSection.empty())
    O << CurrentSection << TAI->getSectionEndDirectiveSuffix() << "\n";

  CurrentSection = NS;
  
  if (!CurrentSection.empty())
    O << CurrentSection << TAI->getDataSectionStartSuffix() << '\n';
}


bool AsmPrinter::doInitialization(Module &M) {
  Mang = new Mangler(M, TAI->getGlobalPrefix());
  
  if (!M.getModuleInlineAsm().empty())
    O << TAI->getCommentString() << " Start of file scope inline assembly\n"
      << M.getModuleInlineAsm()
      << "\n" << TAI->getCommentString()
      << " End of file scope inline assembly\n";

  SwitchToDataSection("", 0);   // Reset back to no section.
  
  if (MachineDebugInfo *DebugInfo = getAnalysisToUpdate<MachineDebugInfo>()) {
    DebugInfo->AnalyzeModule(M);
  }
  
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
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty()) return;

  // Some targets require 4-, 8-, and 16- byte constant literals to be placed
  // in special sections.
  std::vector<std::pair<MachineConstantPoolEntry,unsigned> > FourByteCPs;
  std::vector<std::pair<MachineConstantPoolEntry,unsigned> > EightByteCPs;
  std::vector<std::pair<MachineConstantPoolEntry,unsigned> > SixteenByteCPs;
  std::vector<std::pair<MachineConstantPoolEntry,unsigned> > OtherCPs;
  std::vector<std::pair<MachineConstantPoolEntry,unsigned> > TargetCPs;
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = CP[i];
    const Type *Ty = CPE.getType();
    if (TAI->getFourByteConstantSection() &&
        TM.getTargetData()->getTypeSize(Ty) == 4)
      FourByteCPs.push_back(std::make_pair(CPE, i));
    else if (TAI->getEightByteConstantSection() &&
             TM.getTargetData()->getTypeSize(Ty) == 8)
      EightByteCPs.push_back(std::make_pair(CPE, i));
    else if (TAI->getSixteenByteConstantSection() &&
             TM.getTargetData()->getTypeSize(Ty) == 16)
      SixteenByteCPs.push_back(std::make_pair(CPE, i));
    else
      OtherCPs.push_back(std::make_pair(CPE, i));
  }

  unsigned Alignment = MCP->getConstantPoolAlignment();
  EmitConstantPool(Alignment, TAI->getFourByteConstantSection(), FourByteCPs);
  EmitConstantPool(Alignment, TAI->getEightByteConstantSection(), EightByteCPs);
  EmitConstantPool(Alignment, TAI->getSixteenByteConstantSection(),
                   SixteenByteCPs);
  EmitConstantPool(Alignment, TAI->getConstantPoolSection(), OtherCPs);
}

void AsmPrinter::EmitConstantPool(unsigned Alignment, const char *Section,
               std::vector<std::pair<MachineConstantPoolEntry,unsigned> > &CP) {
  if (CP.empty()) return;

  SwitchToDataSection(Section, 0);
  EmitAlignment(Alignment);
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
      << CP[i].second << ":\t\t\t\t\t" << TAI->getCommentString() << " ";
    WriteTypeSymbolic(O, CP[i].first.getType(), 0) << '\n';
    if (CP[i].first.isMachineConstantPoolEntry())
      EmitMachineConstantPoolValue(CP[i].first.Val.MachineCPVal);
     else
      EmitGlobalConstant(CP[i].first.Val.ConstVal);
    if (i != e-1) {
      const Type *Ty = CP[i].first.getType();
      unsigned EntSize =
        TM.getTargetData()->getTypeSize(Ty);
      unsigned ValEnd = CP[i].first.getOffset() + EntSize;
      // Emit inter-object padding for alignment.
      EmitZeros(CP[i+1].first.getOffset()-ValEnd);
    }
  }
}

/// EmitJumpTableInfo - Print assembly representations of the jump tables used
/// by the current function to the current output stream.  
///
void AsmPrinter::EmitJumpTableInfo(MachineJumpTableInfo *MJTI,
                                   MachineFunction &MF) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;
  const TargetData *TD = TM.getTargetData();
  
  // JTEntryDirective is a string to print sizeof(ptr) for non-PIC jump tables,
  // and 32 bits for PIC since PIC jump table entries are differences, not
  // pointers to blocks.
  // Use the architecture specific relocation directive, if it is set
  const char *JTEntryDirective = TAI->getJumpTableDirective();
  if (!JTEntryDirective)
    JTEntryDirective = TAI->getData32bitsDirective();
  
  // Pick the directive to use to print the jump table entries, and switch to 
  // the appropriate section.
  if (TM.getRelocationModel() == Reloc::PIC_) {
    // In PIC mode, we need to emit the jump table to the same section as the
    // function body itself, otherwise the label differences won't make sense.
    const Function *F = MF.getFunction();
    SwitchToTextSection(getSectionForFunction(*F).c_str(), F);
  } else {
    SwitchToDataSection(TAI->getJumpTableDataSection(), 0);
    if (TD->getPointerSize() == 8)
      JTEntryDirective = TAI->getData64bitsDirective();
  }
  EmitAlignment(Log2_32(TD->getPointerAlignment()));
  
  for (unsigned i = 0, e = JT.size(); i != e; ++i) {
    const std::vector<MachineBasicBlock*> &JTBBs = JT[i].MBBs;

    // For PIC codegen, if possible we want to use the SetDirective to reduce
    // the number of relocations the assembler will generate for the jump table.
    // Set directives are all printed before the jump table itself.
    std::set<MachineBasicBlock*> EmittedSets;
    if (TAI->getSetDirective() && TM.getRelocationModel() == Reloc::PIC_)
      for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii)
        if (EmittedSets.insert(JTBBs[ii]).second)
          printSetLabel(i, JTBBs[ii]);
    
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
      << '_' << i << ":\n";
    
    for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii) {
      O << JTEntryDirective << ' ';
      // If we have emitted set directives for the jump table entries, print 
      // them rather than the entries themselves.  If we're emitting PIC, then
      // emit the table entries as differences between two text section labels.
      // If we're emitting non-PIC code, then emit the entries as direct
      // references to the target basic blocks.
      if (!EmittedSets.empty()) {
        O << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
          << '_' << i << "_set_" << JTBBs[ii]->getNumber();
      } else if (TM.getRelocationModel() == Reloc::PIC_) {
        printBasicBlockLabel(JTBBs[ii], false, false);
	//If the arch uses custom Jump Table directives, don't calc relative to JT
	if (!TAI->getJumpTableDirective()) 
	  O << '-' << TAI->getPrivateGlobalPrefix() << "JTI"
	    << getFunctionNumber() << '_' << i;
      } else {
        printBasicBlockLabel(JTBBs[ii], false, false);
      }
      O << '\n';
    }
  }
}

/// EmitSpecialLLVMGlobal - Check to see if the specified global is a
/// special global used by LLVM.  If so, emit it and return true, otherwise
/// do nothing and return false.
bool AsmPrinter::EmitSpecialLLVMGlobal(const GlobalVariable *GV) {
  // Ignore debug and non-emitted data.
  if (GV->getSection() == "llvm.metadata") return true;
  
  if (!GV->hasAppendingLinkage()) return false;

  assert(GV->hasInitializer() && "Not a special LLVM global!");
  
  if (GV->getName() == "llvm.used") {
    if (TAI->getUsedDirective() != 0)    // No need to emit this at all.
      EmitLLVMUsedList(GV->getInitializer());
    return true;
  }

  if (GV->getName() == "llvm.global_ctors" && GV->use_empty()) {
    SwitchToDataSection(TAI->getStaticCtorsSection(), 0);
    EmitAlignment(2, 0);
    EmitXXStructorList(GV->getInitializer());
    return true;
  } 
  
  if (GV->getName() == "llvm.global_dtors" && GV->use_empty()) {
    SwitchToDataSection(TAI->getStaticDtorsSection(), 0);
    EmitAlignment(2, 0);
    EmitXXStructorList(GV->getInitializer());
    return true;
  }
  
  return false;
}

/// EmitLLVMUsedList - For targets that define a TAI::UsedDirective, mark each
/// global in the specified llvm.used list as being used with this directive.
void AsmPrinter::EmitLLVMUsedList(Constant *List) {
  const char *Directive = TAI->getUsedDirective();

  // Should be an array of 'sbyte*'.
  ConstantArray *InitList = dyn_cast<ConstantArray>(List);
  if (InitList == 0) return;
  
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    O << Directive;
    EmitConstantValueOnly(InitList->getOperand(i));
    O << "\n";
  }
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

/// getPreferredAlignmentLog - Return the preferred alignment of the
/// specified global, returned in log form.  This includes an explicitly
/// requested alignment (if the global has one).
unsigned AsmPrinter::getPreferredAlignmentLog(const GlobalVariable *GV) const {
  const Type *ElemType = GV->getType()->getElementType();
  unsigned Alignment = TM.getTargetData()->getTypeAlignmentShift(ElemType);
  if (GV->getAlignment() > (1U << Alignment))
    Alignment = Log2_32(GV->getAlignment());
  
  if (GV->hasInitializer()) {
    // Always round up alignment of global doubles to 8 bytes.
    if (GV->getType()->getElementType() == Type::DoubleTy && Alignment < 3)
      Alignment = 3;
    if (Alignment < 4) {
      // If the global is not external, see if it is large.  If so, give it a
      // larger alignment.
      if (TM.getTargetData()->getTypeSize(ElemType) > 128)
        Alignment = 4;    // 16-byte alignment.
    }
  }
  return Alignment;
}

// EmitAlignment - Emit an alignment directive to the specified power of two.
void AsmPrinter::EmitAlignment(unsigned NumBits, const GlobalValue *GV) const {
  if (GV && GV->getAlignment())
    NumBits = Log2_32(GV->getAlignment());
  if (NumBits == 0) return;   // No need to emit alignment.
  if (TAI->getAlignmentIsInBytes()) NumBits = 1 << NumBits;
  O << TAI->getAlignDirective() << NumBits << "\n";
}

/// EmitZeros - Emit a block of zeros.
///
void AsmPrinter::EmitZeros(uint64_t NumZeros) const {
  if (NumZeros) {
    if (TAI->getZeroDirective()) {
      O << TAI->getZeroDirective() << NumZeros;
      if (TAI->getZeroDirectiveSuffix())
        O << TAI->getZeroDirectiveSuffix();
      O << "\n";
    } else {
      for (; NumZeros; --NumZeros)
        O << TAI->getData8bitsDirective() << "0\n";
    }
  }
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void AsmPrinter::EmitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue() || isa<UndefValue>(CV))
    O << "0";
  else if (const ConstantBool *CB = dyn_cast<ConstantBool>(CV)) {
    assert(CB->getValue());
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
    if (isa<Function>(GV)) {
      O << TAI->getFunctionAddrPrefix()
        << Mang->getValueName(GV)
        << TAI->getFunctionAddrSuffix();
    } else {
      O << TAI->getGlobalVarAddrPrefix()
        << Mang->getValueName(GV)
        << TAI->getGlobalVarAddrSuffix();
    }
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData *TD = TM.getTargetData();
    switch(CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      std::vector<Value*> idxVec(CE->op_begin()+1, CE->op_end());
      if (int64_t Offset = TD->getIndexedOffset(ptrVal->getType(), idxVec)) {
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
      // Support only foldable casts to/from pointers that can be eliminated by
      // changing the pointer to the appropriately sized integer type.
      Constant *Op = CE->getOperand(0);
      const Type *OpTy = Op->getType(), *Ty = CE->getType();

      // Handle casts to pointers by changing them into casts to the appropriate
      // integer type.  This promotes constant folding and simplifies this code.
      if (isa<PointerType>(Ty)) {
        const Type *IntPtrTy = TD->getIntPtrType();
        Op = ConstantExpr::getCast(Op, IntPtrTy);
        return EmitConstantValueOnly(Op);
      }
      
      // We know the dest type is not a pointer.  Is the src value a pointer or
      // integral?
      if (isa<PointerType>(OpTy) || OpTy->isIntegral()) {
        // We can emit the pointer value into this slot if the slot is an
        // integer slot greater or equal to the size of the pointer.
        if (Ty->isIntegral() && TD->getTypeSize(Ty) >= TD->getTypeSize(OpTy))
          return EmitConstantValueOnly(Op);
      }
      
      assert(0 && "FIXME: Don't yet support this kind of constant cast expr");
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

/// EmitString - Emit a zero-byte-terminated string constant.
///
void AsmPrinter::EmitString(const ConstantArray *CVA) const {
  unsigned NumElts = CVA->getNumOperands();
  if (TAI->getAscizDirective() && NumElts && 
      cast<ConstantInt>(CVA->getOperand(NumElts-1))->getRawValue() == 0) {
    O << TAI->getAscizDirective();
    printAsCString(O, CVA, NumElts-1);
  } else {
    O << TAI->getAsciiDirective();
    printAsCString(O, CVA, NumElts);
  }
  O << "\n";
}

/// EmitGlobalConstant - Print a general LLVM constant to the .s file.
///
void AsmPrinter::EmitGlobalConstant(const Constant *CV) {
  const TargetData *TD = TM.getTargetData();

  if (CV->isNullValue() || isa<UndefValue>(CV)) {
    EmitZeros(TD->getTypeSize(CV->getType()));
    return;
  } else if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (CVA->isString()) {
      EmitString(CVA);
    } else { // Not a string.  Print the values in successive locations
      for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
        EmitGlobalConstant(CVA->getOperand(i));
    }
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    // Print the fields in successive locations. Pad to align if needed!
    const StructLayout *cvsLayout = TD->getStructLayout(CVS->getType());
    uint64_t sizeSoFar = 0;
    for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
      const Constant* field = CVS->getOperand(i);

      // Check if padding is needed and insert one or more 0s.
      uint64_t fieldSize = TD->getTypeSize(field->getType());
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
      if (TAI->getData64bitsDirective())
        O << TAI->getData64bitsDirective() << DoubleToBits(Val) << "\t"
          << TAI->getCommentString() << " double value: " << Val << "\n";
      else if (TD->isBigEndian()) {
        O << TAI->getData32bitsDirective() << unsigned(DoubleToBits(Val) >> 32)
          << "\t" << TAI->getCommentString()
          << " double most significant word " << Val << "\n";
        O << TAI->getData32bitsDirective() << unsigned(DoubleToBits(Val))
          << "\t" << TAI->getCommentString()
          << " double least significant word " << Val << "\n";
      } else {
        O << TAI->getData32bitsDirective() << unsigned(DoubleToBits(Val))
          << "\t" << TAI->getCommentString()
          << " double least significant word " << Val << "\n";
        O << TAI->getData32bitsDirective() << unsigned(DoubleToBits(Val) >> 32)
          << "\t" << TAI->getCommentString()
          << " double most significant word " << Val << "\n";
      }
      return;
    } else {
      O << TAI->getData32bitsDirective() << FloatToBits(Val)
        << "\t" << TAI->getCommentString() << " float " << Val << "\n";
      return;
    }
  } else if (CV->getType() == Type::ULongTy || CV->getType() == Type::LongTy) {
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
      uint64_t Val = CI->getRawValue();

      if (TAI->getData64bitsDirective())
        O << TAI->getData64bitsDirective() << Val << "\n";
      else if (TD->isBigEndian()) {
        O << TAI->getData32bitsDirective() << unsigned(Val >> 32)
          << "\t" << TAI->getCommentString()
          << " Double-word most significant word " << Val << "\n";
        O << TAI->getData32bitsDirective() << unsigned(Val)
          << "\t" << TAI->getCommentString()
          << " Double-word least significant word " << Val << "\n";
      } else {
        O << TAI->getData32bitsDirective() << unsigned(Val)
          << "\t" << TAI->getCommentString()
          << " Double-word least significant word " << Val << "\n";
        O << TAI->getData32bitsDirective() << unsigned(Val >> 32)
          << "\t" << TAI->getCommentString()
          << " Double-word most significant word " << Val << "\n";
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
  printDataDirective(type);
  EmitConstantValueOnly(CV);
  O << "\n";
}

void
AsmPrinter::EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  // Target doesn't support this yet!
  abort();
}

/// PrintSpecial - Print information related to the specified machine instr
/// that is independent of the operand, and may be independent of the instr
/// itself.  This can be useful for portably encoding the comment character
/// or other bits of target-specific knowledge into the asmstrings.  The
/// syntax used is ${:comment}.  Targets can override this to add support
/// for their own strange codes.
void AsmPrinter::PrintSpecial(const MachineInstr *MI, const char *Code) {
  if (!strcmp(Code, "private")) {
    O << TAI->getPrivateGlobalPrefix();
  } else if (!strcmp(Code, "comment")) {
    O << TAI->getCommentString();
  } else if (!strcmp(Code, "uid")) {
    // Assign a unique ID to this machine instruction.
    static const MachineInstr *LastMI = 0;
    static unsigned Counter = 0U-1;
    // If this is a new machine instruction, bump the counter.
    if (LastMI != MI) { ++Counter; LastMI = MI; }
    O << Counter;
  } else {
    std::cerr << "Unknown special formatter '" << Code
              << "' for machine instr: " << *MI;
    exit(1);
  }    
}


/// printInlineAsm - This method formats and prints the specified machine
/// instruction that is an inline asm.
void AsmPrinter::printInlineAsm(const MachineInstr *MI) const {
  unsigned NumOperands = MI->getNumOperands();
  
  // Count the number of register definitions.
  unsigned NumDefs = 0;
  for (; MI->getOperand(NumDefs).isReg() && MI->getOperand(NumDefs).isDef();
       ++NumDefs)
    assert(NumDefs != NumOperands-1 && "No asm string?");
  
  assert(MI->getOperand(NumDefs).isExternalSymbol() && "No asm string?");

  // Disassemble the AsmStr, printing out the literal pieces, the operands, etc.
  const char *AsmStr = MI->getOperand(NumDefs).getSymbolName();

  // If this asmstr is empty, don't bother printing the #APP/#NOAPP markers.
  if (AsmStr[0] == 0) {
    O << "\n";  // Tab already printed, avoid double indenting next instr.
    return;
  }
  
  O << TAI->getInlineAsmStart() << "\n\t";

  // The variant of the current asmprinter: FIXME: change.
  int AsmPrinterVariant = 0;
  
  int CurVariant = -1;            // The number of the {.|.|.} region we are in.
  const char *LastEmitted = AsmStr; // One past the last character emitted.
  
  while (*LastEmitted) {
    switch (*LastEmitted) {
    default: {
      // Not a special case, emit the string section literally.
      const char *LiteralEnd = LastEmitted+1;
      while (*LiteralEnd && *LiteralEnd != '{' && *LiteralEnd != '|' &&
             *LiteralEnd != '}' && *LiteralEnd != '$' && *LiteralEnd != '\n')
        ++LiteralEnd;
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
        O.write(LastEmitted, LiteralEnd-LastEmitted);
      LastEmitted = LiteralEnd;
      break;
    }
    case '\n':
      ++LastEmitted;   // Consume newline character.
      O << "\n\t";     // Indent code with newline.
      break;
    case '$': {
      ++LastEmitted;   // Consume '$' character.
      bool Done = true;

      // Handle escapes.
      switch (*LastEmitted) {
      default: Done = false; break;
      case '$':     // $$ -> $
        if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
          O << '$';
        ++LastEmitted;  // Consume second '$' character.
        break;
      case '(':             // $( -> same as GCC's { character.
        ++LastEmitted;      // Consume '(' character.
        if (CurVariant != -1) {
          std::cerr << "Nested variants found in inline asm string: '"
          << AsmStr << "'\n";
          exit(1);
        }
        CurVariant = 0;     // We're in the first variant now.
        break;
      case '|':
        ++LastEmitted;  // consume '|' character.
        if (CurVariant == -1) {
          std::cerr << "Found '|' character outside of variant in inline asm "
          << "string: '" << AsmStr << "'\n";
          exit(1);
        }
        ++CurVariant;   // We're in the next variant.
        break;
      case ')':         // $) -> same as GCC's } char.
        ++LastEmitted;  // consume ')' character.
        if (CurVariant == -1) {
          std::cerr << "Found '}' character outside of variant in inline asm "
                    << "string: '" << AsmStr << "'\n";
          exit(1);
        }
        CurVariant = -1;
        break;
      }
      if (Done) break;
      
      bool HasCurlyBraces = false;
      if (*LastEmitted == '{') {     // ${variable}
        ++LastEmitted;               // Consume '{' character.
        HasCurlyBraces = true;
      }
      
      const char *IDStart = LastEmitted;
      char *IDEnd;
      long Val = strtol(IDStart, &IDEnd, 10); // We only accept numbers for IDs.
      if (!isdigit(*IDStart) || (Val == 0 && errno == EINVAL)) {
        std::cerr << "Bad $ operand number in inline asm string: '" 
                  << AsmStr << "'\n";
        exit(1);
      }
      LastEmitted = IDEnd;
      
      char Modifier[2] = { 0, 0 };
      
      if (HasCurlyBraces) {
        // If we have curly braces, check for a modifier character.  This
        // supports syntax like ${0:u}, which correspond to "%u0" in GCC asm.
        if (*LastEmitted == ':') {
          ++LastEmitted;    // Consume ':' character.
          if (*LastEmitted == 0) {
            std::cerr << "Bad ${:} expression in inline asm string: '" 
                      << AsmStr << "'\n";
            exit(1);
          }
          
          Modifier[0] = *LastEmitted;
          ++LastEmitted;    // Consume modifier character.
        }
        
        if (*LastEmitted != '}') {
          std::cerr << "Bad ${} expression in inline asm string: '" 
                    << AsmStr << "'\n";
          exit(1);
        }
        ++LastEmitted;    // Consume '}' character.
      }
      
      if ((unsigned)Val >= NumOperands-1) {
        std::cerr << "Invalid $ operand number in inline asm string: '" 
                  << AsmStr << "'\n";
        exit(1);
      }
      
      // Okay, we finally have a value number.  Ask the target to print this
      // operand!
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant) {
        unsigned OpNo = 1;

        bool Error = false;

        // Scan to find the machine operand number for the operand.
        for (; Val; --Val) {
          if (OpNo >= MI->getNumOperands()) break;
          unsigned OpFlags = MI->getOperand(OpNo).getImmedValue();
          OpNo += (OpFlags >> 3) + 1;
        }

        if (OpNo >= MI->getNumOperands()) {
          Error = true;
        } else {
          unsigned OpFlags = MI->getOperand(OpNo).getImmedValue();
          ++OpNo;  // Skip over the ID number.

          AsmPrinter *AP = const_cast<AsmPrinter*>(this);
          if ((OpFlags & 7) == 4 /*ADDR MODE*/) {
            Error = AP->PrintAsmMemoryOperand(MI, OpNo, AsmPrinterVariant,
                                              Modifier[0] ? Modifier : 0);
          } else {
            Error = AP->PrintAsmOperand(MI, OpNo, AsmPrinterVariant,
                                        Modifier[0] ? Modifier : 0);
          }
        }
        if (Error) {
          std::cerr << "Invalid operand found in inline asm: '"
                    << AsmStr << "'\n";
          MI->dump();
          exit(1);
        }
      }
      break;
    }
    }
  }
  O << "\n\t" << TAI->getInlineAsmEnd() << "\n";
}

/// PrintAsmOperand - Print the specified operand of MI, an INLINEASM
/// instruction, using the specified assembler variant.  Targets should
/// overried this to format as appropriate.
bool AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode) {
  // Target doesn't support this yet!
  return true;
}

bool AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode) {
  // Target doesn't support this yet!
  return true;
}

/// printBasicBlockLabel - This method prints the label for the specified
/// MachineBasicBlock
void AsmPrinter::printBasicBlockLabel(const MachineBasicBlock *MBB,
                                      bool printColon,
                                      bool printComment) const {
  O << TAI->getPrivateGlobalPrefix() << "BB" << FunctionNumber << "_"
    << MBB->getNumber();
  if (printColon)
    O << ':';
  if (printComment && MBB->getBasicBlock())
    O << '\t' << TAI->getCommentString() << MBB->getBasicBlock()->getName();
}

/// printSetLabel - This method prints a set label for the specified
/// MachineBasicBlock
void AsmPrinter::printSetLabel(unsigned uid, 
                               const MachineBasicBlock *MBB) const {
  if (!TAI->getSetDirective())
    return;
  
  O << TAI->getSetDirective() << ' ' << TAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << "_set_" << MBB->getNumber() << ',';
  printBasicBlockLabel(MBB, false, false);
  O << '-' << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
    << '_' << uid << '\n';
}

/// printDataDirective - This method prints the asm directive for the
/// specified type.
void AsmPrinter::printDataDirective(const Type *type) {
  const TargetData *TD = TM.getTargetData();
  switch (type->getTypeID()) {
  case Type::BoolTyID:
  case Type::UByteTyID: case Type::SByteTyID:
    O << TAI->getData8bitsDirective();
    break;
  case Type::UShortTyID: case Type::ShortTyID:
    O << TAI->getData16bitsDirective();
    break;
  case Type::PointerTyID:
    if (TD->getPointerSize() == 8) {
      assert(TAI->getData64bitsDirective() &&
             "Target cannot handle 64-bit pointer exprs!");
      O << TAI->getData64bitsDirective();
      break;
    }
    //Fall through for pointer size == int size
  case Type::UIntTyID: case Type::IntTyID:
    O << TAI->getData32bitsDirective();
    break;
  case Type::ULongTyID: case Type::LongTyID:
    assert(TAI->getData64bitsDirective() &&
           "Target cannot handle 64-bit constant exprs!");
    O << TAI->getData64bitsDirective();
    break;
  case Type::FloatTyID: case Type::DoubleTyID:
    assert (0 && "Should have already output floating point constant.");
  default:
    assert (0 && "Can't handle printing this type of thing");
    break;
  }
}

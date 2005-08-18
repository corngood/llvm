//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "InstrInfoEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
using namespace llvm;

// runEnums - Print out enum values for all of the instructions.
void InstrInfoEmitter::runEnums(std::ostream &OS) {
  EmitSourceFileHeader("Target Instruction Enum Values", OS);
  OS << "namespace llvm {\n\n";

  CodeGenTarget Target;

  // We must emit the PHI opcode first...
  Record *InstrInfo = Target.getInstructionSet();

  std::string Namespace = Target.inst_begin()->second.Namespace;

  if (!Namespace.empty())
    OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n";

  std::vector<const CodeGenInstruction*> NumberedInstructions;
  Target.getInstructionsByEnumValue(NumberedInstructions);

  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    OS << "    " << NumberedInstructions[i]->TheDef->getName()
       << ", \t// " << i << "\n";
  }
  OS << "  };\n";
  if (!Namespace.empty())
    OS << "}\n";
  OS << "} // End llvm namespace \n";
}

void InstrInfoEmitter::printDefList(ListInit *LI, const std::string &Name,
                                    std::ostream &OS) const {
  OS << "static const unsigned " << Name << "[] = { ";
  for (unsigned j = 0, e = LI->getSize(); j != e; ++j)
    if (DefInit *DI = dynamic_cast<DefInit*>(LI->getElement(j)))
      OS << getQualifiedName(DI->getDef()) << ", ";
    else
      throw "Illegal value in '" + Name + "' list!";
  OS << "0 };\n";
}


// run - Emit the main instruction description records for the target...
void InstrInfoEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("Target Instruction Descriptors", OS);
  OS << "namespace llvm {\n\n";

  CodeGenTarget Target;
  const std::string &TargetName = Target.getName();
  Record *InstrInfo = Target.getInstructionSet();
  Record *PHI = InstrInfo->getValueAsDef("PHIInst");

  // Emit empty implicit uses and defs lists
  OS << "static const unsigned EmptyImpUses[] = { 0 };\n"
     << "static const unsigned EmptyImpDefs[] = { 0 };\n";

  // Emit all of the instruction's implicit uses and defs...
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
         E = Target.inst_end(); II != E; ++II) {
    Record *Inst = II->second.TheDef;
    ListInit *LI = Inst->getValueAsListInit("Uses");
    if (LI->getSize()) printDefList(LI, Inst->getName()+"ImpUses", OS);
    LI = Inst->getValueAsListInit("Defs");
    if (LI->getSize()) printDefList(LI, Inst->getName()+"ImpDefs", OS);
  }

  OS << "\nstatic const TargetInstrDescriptor " << TargetName
     << "Insts[] = {\n";
  emitRecord(Target.getPHIInstruction(), 0, InstrInfo, OS);

  unsigned i = 0;
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
         E = Target.inst_end(); II != E; ++II)
    if (II->second.TheDef != PHI)
      emitRecord(II->second, ++i, InstrInfo, OS);
  OS << "};\n";
  OS << "} // End llvm namespace \n";
}

void InstrInfoEmitter::emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                                  Record *InstrInfo, std::ostream &OS) {
  OS << "  { \"";
  if (Inst.Name.empty())
    OS << Inst.TheDef->getName();
  else
    OS << Inst.Name;
  OS << "\",\t" << Inst.OperandList.size() << ", -1, 0, false, 0, 0, 0, 0";

  // Emit all of the target indepedent flags...
  if (Inst.isReturn)     OS << "|M_RET_FLAG";
  if (Inst.isBranch)     OS << "|M_BRANCH_FLAG";
  if (Inst.isBarrier)    OS << "|M_BARRIER_FLAG";
  if (Inst.hasDelaySlot) OS << "|M_DELAY_SLOT_FLAG";
  if (Inst.isCall)       OS << "|M_CALL_FLAG";
  if (Inst.isLoad)       OS << "|M_LOAD_FLAG";
  if (Inst.isStore)      OS << "|M_STORE_FLAG";
  if (Inst.isTwoAddress) OS << "|M_2_ADDR_FLAG";
  if (Inst.isConvertibleToThreeAddress) OS << "|M_CONVERTIBLE_TO_3_ADDR";
  if (Inst.isCommutable) OS << "|M_COMMUTABLE";
  if (Inst.isTerminator) OS << "|M_TERMINATOR_FLAG";
  OS << ", 0";

  // Emit all of the target-specific flags...
  ListInit *LI    = InstrInfo->getValueAsListInit("TSFlagsFields");
  ListInit *Shift = InstrInfo->getValueAsListInit("TSFlagsShifts");
  if (LI->getSize() != Shift->getSize())
    throw "Lengths of " + InstrInfo->getName() +
          ":(TargetInfoFields, TargetInfoPositions) must be equal!";

  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    emitShiftedValue(Inst.TheDef, dynamic_cast<StringInit*>(LI->getElement(i)),
                     dynamic_cast<IntInit*>(Shift->getElement(i)), OS);

  OS << ", ";

  // Emit the implicit uses and defs lists...
  LI = Inst.TheDef->getValueAsListInit("Uses");
  if (!LI->getSize())
    OS << "EmptyImpUses, ";
  else
    OS << Inst.TheDef->getName() << "ImpUses, ";

  LI = Inst.TheDef->getValueAsListInit("Defs");
  if (!LI->getSize())
    OS << "EmptyImpDefs ";
  else
    OS << Inst.TheDef->getName() << "ImpDefs ";

  OS << " },  // Inst #" << Num << " = " << Inst.TheDef->getName() << "\n";
}

void InstrInfoEmitter::emitShiftedValue(Record *R, StringInit *Val,
                                        IntInit *ShiftInt, std::ostream &OS) {
  if (Val == 0 || ShiftInt == 0)
    throw std::string("Illegal value or shift amount in TargetInfo*!");
  RecordVal *RV = R->getValue(Val->getValue());
  int Shift = ShiftInt->getValue();

  if (RV == 0 || RV->getValue() == 0)
    throw R->getName() + " doesn't have a field named '" + Val->getValue()+"'!";

  Init *Value = RV->getValue();
  if (BitInit *BI = dynamic_cast<BitInit*>(Value)) {
    if (BI->getValue()) OS << "|(1<<" << Shift << ")";
    return;
  } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Value)) {
    // Convert the Bits to an integer to print...
    Init *I = BI->convertInitializerTo(new IntRecTy());
    if (I)
      if (IntInit *II = dynamic_cast<IntInit*>(I)) {
        if (II->getValue())
          OS << "|(" << II->getValue() << "<<" << Shift << ")";
        return;
      }

  } else if (IntInit *II = dynamic_cast<IntInit*>(Value)) {
    if (II->getValue()) OS << "|(" << II->getValue() << "<<" << Shift << ")";
    return;
  }

  std::cerr << "Unhandled initializer: " << *Val << "\n";
  throw "In record '" + R->getName() + "' for TSFlag emission.";
}


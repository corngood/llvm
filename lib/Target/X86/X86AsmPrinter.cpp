//===-- X86AsmPrinter.cpp - Convert X86 LLVM IR to X86 assembly -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file the shared super class printer that converts from our internal
// representation of machine-dependent LLVM code to Intel and AT&T format
// assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "X86AsmPrinter.h"
#include "X86ATTAsmPrinter.h"
#include "X86IntelAsmPrinter.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetAsmInfo.h"

using namespace llvm;

Statistic<> llvm::EmittedInsts("asm-printer",
                               "Number of machine instrs printed");

static X86FunctionInfo calculateFunctionInfo(const Function *F,
                                             const TargetData *TD) {
  X86FunctionInfo Info;
  uint64_t Size = 0;
  
  switch (F->getCallingConv()) {
  case CallingConv::X86_StdCall:
    Info.setDecorationStyle(StdCall);
    break;
  case CallingConv::X86_FastCall:
    Info.setDecorationStyle(FastCall);
    break;
  default:
    return Info;
  }

  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI)
    Size += TD->getTypeSize(AI->getType());

  // Size should be aligned to DWORD boundary
  Size = ((Size + 3)/4)*4;
  
  // We're not supporting tooooo huge arguments :)
  Info.setBytesToPopOnReturn((unsigned int)Size);
  return Info;
}


/// decorateName - Query FunctionInfoMap and use this information for various
/// name decoration.
void X86SharedAsmPrinter::decorateName(std::string &Name,
                                       const GlobalValue *GV) {
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;

  // We don't want to decorate non-stdcall or non-fastcall functions right now
  unsigned CC = F->getCallingConv();
  if (CC != CallingConv::X86_StdCall && CC != CallingConv::X86_FastCall)
    return;
    
  FMFInfoMap::const_iterator info_item = FunctionInfoMap.find(F);

  const X86FunctionInfo *Info;
  if (info_item == FunctionInfoMap.end()) {
    // Calculate apropriate function info and populate map
    FunctionInfoMap[F] = calculateFunctionInfo(F, TM.getTargetData());
    Info = &FunctionInfoMap[F];
  } else {
    Info = &info_item->second;
  }
        
  switch (Info->getDecorationStyle()) {
  case None:
    break;
  case StdCall:
    if (!F->isVarArg()) // Variadic functions do not receive @0 suffix.
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());
    break;
  case FastCall:
    if (!F->isVarArg()) // Variadic functions do not receive @0 suffix.
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());

    if (Name[0] == '_') {
      Name[0] = '@';
    } else {
      Name = '@' + Name;
    }    
    break;
  default:
    assert(0 && "Unsupported DecorationStyle");
  }
}

/// doInitialization
bool X86SharedAsmPrinter::doInitialization(Module &M) {
  if (Subtarget->isTargetDarwin()) {
    const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
    if (!Subtarget->is64Bit())
      X86PICStyle = PICStyle::Stub;

    // Emit initial debug information.
    DW.BeginModule(&M);
  }

  return AsmPrinter::doInitialization(M);
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  // Note: this code is not shared by the Intel printer as it is too different
  // from how MASM does things.  When making changes here don't forget to look
  // at X86IntelAsmPrinter::doFinalization().
  const TargetData *TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer()) continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I))
      continue;
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Size = TD->getTypeSize(C->getType());
    unsigned Align = getPreferredAlignmentLog(I);

    if (C->isNullValue() && /* FIXME: Verify correct */
        (I->hasInternalLinkage() || I->hasWeakLinkage() ||
         I->hasLinkOnceLinkage() ||
         (Subtarget->isTargetDarwin() && 
          I->hasExternalLinkage() && !I->hasSection()))) {
      if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
      if (I->hasExternalLinkage()) {
          O << "\t.globl\t" << name << "\n";
          O << "\t.zerofill __DATA__, __common, " << name << ", "
            << Size << ", " << Align;
      } else {
        SwitchToDataSection(TAI->getDataSection(), I);
        if (TAI->getLCOMMDirective() != NULL) {
          if (I->hasInternalLinkage()) {
            O << TAI->getLCOMMDirective() << name << "," << Size;
            if (Subtarget->isTargetDarwin())
              O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
          } else
            O << TAI->getCOMMDirective()  << name << "," << Size;
        } else {
          if (!Subtarget->isTargetCygwin()) {
            if (I->hasInternalLinkage())
              O << "\t.local\t" << name << "\n";
          }
          O << TAI->getCOMMDirective()  << name << "," << Size;
          if (TAI->getCOMMDirectiveTakesAlignment())
            O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
        }
      }
      O << "\t\t" << TAI->getCommentString() << " " << I->getName() << "\n";
    } else {
      switch (I->getLinkage()) {
      case GlobalValue::LinkOnceLinkage:
      case GlobalValue::WeakLinkage:
        if (Subtarget->isTargetDarwin()) {
          O << "\t.globl " << name << "\n"
            << "\t.weak_definition " << name << "\n";
          SwitchToDataSection(".section __DATA,__const_coal,coalesced", I);
        } else if (Subtarget->isTargetCygwin()) {
          O << "\t.section\t.data$linkonce." << name << ",\"aw\"\n"
            << "\t.globl " << name << "\n"
            << "\t.linkonce same_size\n";
        } else {
          O << "\t.section\t.llvm.linkonce.d." << name << ",\"aw\",@progbits\n"
            << "\t.weak " << name << "\n";
        }
        break;
      case GlobalValue::AppendingLinkage:
        // FIXME: appending linkage variables should go into a section of
        // their name or something.  For now, just emit them as external.
      case GlobalValue::DLLExportLinkage:
        DLLExportedGVs.insert(Mang->makeNameProper(I->getName(),""));
        // FALL THROUGH
      case GlobalValue::ExternalLinkage:
        // If external or appending, declare as a global symbol
        O << "\t.globl " << name << "\n";
        // FALL THROUGH
      case GlobalValue::InternalLinkage:
        SwitchToDataSection(TAI->getDataSection(), I);
        break;
      default:
        assert(0 && "Unknown linkage type!");
      }

      EmitAlignment(Align, I);
      O << name << ":\t\t\t\t" << TAI->getCommentString() << " " << I->getName()
        << "\n";
      if (TAI->hasDotTypeDotSizeDirective())
        O << "\t.size " << name << ", " << Size << "\n";

      EmitGlobalConstant(C);
      O << '\n';
    }
  }
  
  // Output linker support code for dllexported globals
  if (DLLExportedGVs.begin() != DLLExportedGVs.end()) {
    SwitchToDataSection(".section .drectve", 0);    
  }

  for (std::set<std::string>::iterator i = DLLExportedGVs.begin(),
         e = DLLExportedGVs.end();
         i != e; ++i) {
    O << "\t.ascii \" -export:" << *i << ",data\"\n";
  }    

  if (DLLExportedFns.begin() != DLLExportedFns.end()) {
    SwitchToDataSection(".section .drectve", 0);    
  }

  for (std::set<std::string>::iterator i = DLLExportedFns.begin(),
         e = DLLExportedFns.end();
         i != e; ++i) {
    O << "\t.ascii \" -export:" << *i << "\"\n";
  }    
 
  if (Subtarget->isTargetDarwin()) {
    SwitchToDataSection("", 0);

    // Output stubs for dynamically-linked functions
    unsigned j = 1;
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i, ++j) {
      SwitchToDataSection(".section __IMPORT,__jump_table,symbol_stubs,"
                          "self_modifying_code+pure_instructions,5", 0);
      O << "L" << *i << "$stub:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\thlt ; hlt ; hlt ; hlt ; hlt\n";
    }

    O << "\n";

    // Output stubs for external and common global variables.
    if (GVStubs.begin() != GVStubs.end())
      SwitchToDataSection(
                    ".section __IMPORT,__pointers,non_lazy_symbol_pointers", 0);
    for (std::set<std::string>::iterator i = GVStubs.begin(), e = GVStubs.end();
         i != e; ++i) {
      O << "L" << *i << "$non_lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long\t0\n";
    }

    // Emit initial debug information.
    DW.EndModule();

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    O << "\t.subsections_via_symbols\n";
  }

  AsmPrinter::doFinalization(M);
  return false; // success
}

/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
FunctionPass *llvm::createX86CodePrinterPass(std::ostream &o,
                                             X86TargetMachine &tm) {
  const X86Subtarget *Subtarget = &tm.getSubtarget<X86Subtarget>();

  if (Subtarget->isFlavorIntel()) {
    return new X86IntelAsmPrinter(o, tm, tm.getTargetAsmInfo());
  } else {
    return new X86ATTAsmPrinter(o, tm, tm.getTargetAsmInfo());
  }
}

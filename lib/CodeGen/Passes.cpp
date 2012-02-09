//===-- Passes.cpp - Target independent code generation passes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static cl::opt<bool> DisablePostRA("disable-post-ra", cl::Hidden,
    cl::desc("Disable Post Regalloc"));
static cl::opt<bool> DisableBranchFold("disable-branch-fold", cl::Hidden,
    cl::desc("Disable branch folding"));
static cl::opt<bool> DisableTailDuplicate("disable-tail-duplicate", cl::Hidden,
    cl::desc("Disable tail duplication"));
static cl::opt<bool> DisableEarlyTailDup("disable-early-taildup", cl::Hidden,
    cl::desc("Disable pre-register allocation tail duplication"));
static cl::opt<bool> EnableBlockPlacement("enable-block-placement",
    cl::Hidden, cl::desc("Enable probability-driven block placement"));
static cl::opt<bool> EnableBlockPlacementStats("enable-block-placement-stats",
    cl::Hidden, cl::desc("Collect probability-driven block placement stats"));
static cl::opt<bool> DisableCodePlace("disable-code-place", cl::Hidden,
    cl::desc("Disable code placement"));
static cl::opt<bool> DisableSSC("disable-ssc", cl::Hidden,
    cl::desc("Disable Stack Slot Coloring"));
static cl::opt<bool> DisableMachineDCE("disable-machine-dce", cl::Hidden,
    cl::desc("Disable Machine Dead Code Elimination"));
static cl::opt<bool> DisableMachineLICM("disable-machine-licm", cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineCSE("disable-machine-cse", cl::Hidden,
    cl::desc("Disable Machine Common Subexpression Elimination"));
static cl::opt<bool> DisablePostRAMachineLICM("disable-postra-machine-licm",
    cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineSink("disable-machine-sink", cl::Hidden,
    cl::desc("Disable Machine Sinking"));
static cl::opt<bool> DisableLSR("disable-lsr", cl::Hidden,
    cl::desc("Disable Loop Strength Reduction Pass"));
static cl::opt<bool> DisableCGP("disable-cgp", cl::Hidden,
    cl::desc("Disable Codegen Prepare"));
static cl::opt<bool> DisableCopyProp("disable-copyprop", cl::Hidden,
    cl::desc("Disable Copy Propagation pass"));
static cl::opt<bool> PrintLSR("print-lsr-output", cl::Hidden,
    cl::desc("Print LLVM IR produced by the loop-reduce pass"));
static cl::opt<bool> PrintISelInput("print-isel-input", cl::Hidden,
    cl::desc("Print LLVM IR input to isel pass"));
static cl::opt<bool> PrintGCInfo("print-gc", cl::Hidden,
    cl::desc("Dump garbage collector data"));
static cl::opt<bool> VerifyMachineCode("verify-machineinstrs", cl::Hidden,
    cl::desc("Verify generated machine code"),
    cl::init(getenv("LLVM_VERIFY_MACHINEINSTRS")!=NULL));

//===---------------------------------------------------------------------===//
/// TargetPassConfig
//===---------------------------------------------------------------------===//

INITIALIZE_PASS(TargetPassConfig, "targetpassconfig",
                "Target Pass Configuration", false, false)
char TargetPassConfig::ID = 0;

// Out of line virtual method.
TargetPassConfig::~TargetPassConfig() {}

// Out of line constructor provides default values for pass options and
// registers all common codegen passes.
TargetPassConfig::TargetPassConfig(TargetMachine *tm, PassManagerBase &pm)
  : ImmutablePass(ID), TM(tm), PM(pm), Initialized(false),
    DisableVerify(false),
    EnableTailMerge(true) {

  // Register all target independent codegen passes to activate their PassIDs,
  // including this pass itself.
  initializeCodeGen(*PassRegistry::getPassRegistry());
}

/// createPassConfig - Create a pass configuration object to be used by
/// addPassToEmitX methods for generating a pipeline of CodeGen passes.
///
/// Targets may override this to extend TargetPassConfig.
TargetPassConfig *LLVMTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new TargetPassConfig(this, PM);
}

TargetPassConfig::TargetPassConfig()
  : ImmutablePass(ID), PM(*(PassManagerBase*)0) {
  llvm_unreachable("TargetPassConfig should not be constructed on-the-fly");
}

// Helper to verify the analysis is really immutable.
void TargetPassConfig::setOpt(bool &Opt, bool Val) {
  assert(!Initialized && "PassConfig is immutable");
  Opt = Val;
}

void TargetPassConfig::addPass(char &ID) {
  // FIXME: check user overrides
  Pass *P = Pass::createPass(ID);
  if (!P)
    llvm_unreachable("Pass ID not registered");
  PM.add(P);
}

void TargetPassConfig::printNoVerify(const char *Banner) const {
  if (TM->shouldPrintMachineCode())
    PM.add(createMachineFunctionPrinterPass(dbgs(), Banner));
}

void TargetPassConfig::printAndVerify(const char *Banner) const {
  if (TM->shouldPrintMachineCode())
    PM.add(createMachineFunctionPrinterPass(dbgs(), Banner));

  if (VerifyMachineCode)
    PM.add(createMachineVerifierPass(Banner));
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
void TargetPassConfig::addIRPasses() {
  // Basic AliasAnalysis support.
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM.add(createTypeBasedAliasAnalysisPass());
  PM.add(createBasicAliasAnalysisPass());

  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!DisableVerify)
    PM.add(createVerifierPass());

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOpt::None && !DisableLSR) {
    PM.add(createLoopStrengthReducePass(getTargetLowering()));
    if (PrintLSR)
      PM.add(createPrintFunctionPass("\n\n*** Code after LSR ***\n", &dbgs()));
  }

  PM.add(createGCLoweringPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
void TargetPassConfig::addISelPrepare() {
  if (getOptLevel() != CodeGenOpt::None && !DisableCGP)
    PM.add(createCodeGenPreparePass(getTargetLowering()));

  PM.add(createStackProtectorPass(getTargetLowering()));

  addPreISel();

  if (PrintISelInput)
    PM.add(createPrintFunctionPass("\n\n"
                                   "*** Final LLVM Code input to ISel ***\n",
                                   &dbgs()));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!DisableVerify)
    PM.add(createVerifierPass());
}

/// Add the complete set of target-independent postISel code generator passes.
///
/// This can be read as the standard order of major LLVM CodeGen stages. Stages
/// with nontrivial configuration or multiple passes are broken out below in
/// add%Stage routines.
///
/// Any TargetPassConfig::addXX routine may be overriden by the Target. The
/// addPre/Post methods with empty header implementations allow injecting
/// target-specific fixups just before or after major stages. Additionally,
/// targets have the flexibility to change pass order within a stage by
/// overriding default implementation of add%Stage routines below. Each
/// technique has maintainability tradeoffs because alternate pass orders are
/// not well supported. addPre/Post works better if the target pass is easily
/// tied to a common pass. But if it has subtle dependencies on multiple passes,
/// overriding the stage instead.
///
/// TODO: We could use a single addPre/Post(ID) hook to allow pass injection
/// before/after any target-independent pass. But it's currently overkill.
void TargetPassConfig::addMachinePasses() {
  // Print the instruction selected machine code...
  printAndVerify("After Instruction Selection");

  // Expand pseudo-instructions emitted by ISel.
  addPass(ExpandISelPseudosID);

  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOpt::None) {
    addMachineSSAOptimization();
  }
  else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addPass(LocalStackSlotAllocationID);
  }

  // Run pre-ra passes.
  if (addPreRegAlloc())
    printAndVerify("After PreRegAlloc passes");

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  addRegAlloc();

  // Run post-ra passes.
  if (addPostRegAlloc())
    printAndVerify("After PostRegAlloc passes");

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  addPass(PrologEpilogCodeInserterID);
  printAndVerify("After PrologEpilogCodeInserter");

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOpt::None)
    addMachineLateOptimization();

  // Expand pseudo instructions before second scheduling pass.
  addPass(ExpandPostRAPseudosID);
  printNoVerify("After ExpandPostRAPseudos");

  // Run pre-sched2 passes.
  if (addPreSched2())
    printNoVerify("After PreSched2 passes");

  // Second pass scheduler.
  if (getOptLevel() != CodeGenOpt::None && !DisablePostRA) {
    addPass(PostRASchedulerID);
    printNoVerify("After PostRAScheduler");
  }

  // GC
  addPass(GCMachineCodeAnalysisID);
  if (PrintGCInfo)
    PM.add(createGCInfoPrinter(dbgs()));

  // Basic block placement.
  if (getOptLevel() != CodeGenOpt::None && !DisableCodePlace)
    addBlockPlacement();

  if (addPreEmitPass())
    printNoVerify("After PreEmit passes");
}

/// Add passes that optimize machine instructions in SSA form.
void TargetPassConfig::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  if (!DisableEarlyTailDup) {
    addPass(TailDuplicateID);
    printAndVerify("After Pre-RegAlloc TailDuplicate");
  }

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(OptimizePHIsID);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(LocalStackSlotAllocationID);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  if (!DisableMachineDCE)
    addPass(DeadMachineInstructionElimID);
  printAndVerify("After codegen DCE pass");

  if (!DisableMachineLICM)
    addPass(MachineLICMID);
  if (!DisableMachineCSE)
    addPass(MachineCSEID);
  if (!DisableMachineSink)
    addPass(MachineSinkingID);
  printAndVerify("After Machine LICM, CSE and Sinking passes");

  addPass(PeepholeOptimizerID);
  printAndVerify("After codegen peephole optimization pass");
}

//===---------------------------------------------------------------------===//
/// Register Allocation Pass Configuration
//===---------------------------------------------------------------------===//

/// RegisterRegAlloc's global Registry tracks allocator registration.
MachinePassRegistry RegisterRegAlloc::Registry;

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
static FunctionPass *createDefaultRegisterAllocator() { return 0; }
static RegisterRegAlloc
defaultRegAlloc("default",
                "pick register allocator based on -O option",
                createDefaultRegisterAllocator);

/// -regalloc=... command line option.
static cl::opt<RegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RegisterRegAlloc> >
RegAlloc("regalloc",
         cl::init(&createDefaultRegisterAllocator),
         cl::desc("Register allocator to use"));


/// createRegisterAllocator - choose the appropriate register allocator.
FunctionPass *llvm::createRegisterAllocator(CodeGenOpt::Level OptLevel) {
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = RegAlloc;
    RegisterRegAlloc::setDefault(RegAlloc);
  }

  if (Ctor != createDefaultRegisterAllocator)
    return Ctor();

  // When the 'default' allocator is requested, pick one based on OptLevel.
  switch (OptLevel) {
  case CodeGenOpt::None:
    return createFastRegisterAllocator();
  default:
    return createGreedyRegisterAllocator();
  }
}

/// Add standard target-independent passes that are tightly coupled with
/// register allocation, including coalescing, machine instruction scheduling,
/// and register allocation itself.
///
/// FIXME: This will become the register allocation "super pass" pipeline.
void TargetPassConfig::addRegAlloc() {
  // Perform register allocation.
  PM.add(createRegisterAllocator(getOptLevel()));
  printAndVerify("After Register Allocation");

  // Perform stack slot coloring and post-ra machine LICM.
  if (getOptLevel() != CodeGenOpt::None) {
    // FIXME: Re-enable coloring with register when it's capable of adding
    // kill markers.
    if (!DisableSSC)
      addPass(StackSlotColoringID);

    // Run post-ra machine LICM to hoist reloads / remats.
    //
    // FIXME: can this move into MachineLateOptimization?
    if (!DisablePostRAMachineLICM)
      addPass(MachineLICMID);

    printAndVerify("After StackSlotColoring and postra Machine LICM");
  }
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
void TargetPassConfig::addMachineLateOptimization() {
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (!DisableBranchFold) {
    addPass(BranchFolderPassID);
    printNoVerify("After BranchFolding");
  }

  // Tail duplication.
  if (!DisableTailDuplicate) {
    addPass(TailDuplicateID);
    printNoVerify("After TailDuplicate");
  }

  // Copy propagation.
  if (!DisableCopyProp) {
    addPass(MachineCopyPropagationID);
    printNoVerify("After copy propagation pass");
  }
}

/// Add standard basic block placement passes.
void TargetPassConfig::addBlockPlacement() {
  if (EnableBlockPlacement) {
    // MachineBlockPlacement is an experimental pass which is disabled by
    // default currently. Eventually it should subsume CodePlacementOpt, so
    // when enabled, the other is disabled.
    addPass(MachineBlockPlacementID);
    printNoVerify("After MachineBlockPlacement");
  } else {
    addPass(CodePlacementOptID);
    printNoVerify("After CodePlacementOpt");
  }

  // Run a separate pass to collect block placement statistics.
  if (EnableBlockPlacementStats) {
    addPass(MachineBlockPlacementStatsID);
    printNoVerify("After MachineBlockPlacementStats");
  }
}

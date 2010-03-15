//===-- llvm/CodeGen/MachineModuleInfo.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfo.h"

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;
using namespace llvm::dwarf;

// Handle the Pass registration stuff necessary to use TargetData's.
static RegisterPass<MachineModuleInfo>
X("machinemoduleinfo", "Machine Module Information");
char MachineModuleInfo::ID = 0;

// Out of line virtual method.
MachineModuleInfoImpl::~MachineModuleInfoImpl() {}

namespace llvm {
class MMIAddrLabelMapCallbackPtr : CallbackVH {
  MMIAddrLabelMap *Map;
public:
  MMIAddrLabelMapCallbackPtr() : Map(0) {}
  MMIAddrLabelMapCallbackPtr(Value *V) : CallbackVH(V), Map(0) {}
  
  void setMap(MMIAddrLabelMap *map) { Map = map; }
  
  virtual void deleted();
  virtual void allUsesReplacedWith(Value *V2);
};
  
class MMIAddrLabelMap {
  MCContext &Context;
  struct AddrLabelSymEntry {
    MCSymbol *Sym;
    unsigned Index;
  };
  
  DenseMap<AssertingVH<BasicBlock>, AddrLabelSymEntry> AddrLabelSymbols;
  
  std::vector<MMIAddrLabelMapCallbackPtr> BBCallbacks;
public:
  
  MMIAddrLabelMap(MCContext &context) : Context(context) {}
  
  MCSymbol *getAddrLabelSymbol(BasicBlock *BB);  
  void UpdateForDeletedBlock(BasicBlock *BB);
  void UpdateForRAUWBlock(BasicBlock *Old, BasicBlock *New);
};
}

MCSymbol *MMIAddrLabelMap::getAddrLabelSymbol(BasicBlock *BB) {
  assert(BB->hasAddressTaken() &&
         "Shouldn't get label for block without address taken");
  AddrLabelSymEntry &Entry = AddrLabelSymbols[BB];
  
  // If we already had an entry for this block, just return it.
  if (Entry.Sym) return Entry.Sym;
  
  // Otherwise, this is a new entry, create a new symbol for it and add an
  // entry to BBCallbacks so we can be notified if the BB is deleted or RAUWd.
  BBCallbacks.push_back(BB);
  BBCallbacks.back().setMap(this);
  Entry.Index = BBCallbacks.size()-1;
  return Entry.Sym = Context.CreateTempSymbol();
}

void MMIAddrLabelMap::UpdateForDeletedBlock(BasicBlock *BB) {
  // If the block got deleted, there is no need for the symbol.  If the symbol
  // was already emitted, we can just forget about it, otherwise we need to
  // queue it up for later emission when the function is output.
  AddrLabelSymEntry Entry = AddrLabelSymbols[BB];
  AddrLabelSymbols.erase(BB);
  assert(Entry.Sym && "Didn't have a symbol, why a callback?");
  BBCallbacks[Entry.Index] = 0;  // Clear the callback.

  if (Entry.Sym->isDefined())
    return;
  
  // If the block is not yet defined, we need to emit it at the end of the
  // function.
  assert(0 && "Case not handled yet!");
  abort();
}

void MMIAddrLabelMap::UpdateForRAUWBlock(BasicBlock *Old, BasicBlock *New) {
  // Get the entry for the RAUW'd block and remove it from our map.
  AddrLabelSymEntry OldEntry = AddrLabelSymbols[Old];
  AddrLabelSymbols.erase(Old);
  assert(OldEntry.Sym && "Didn't have a symbol, why a callback?");
  
  // If New is not address taken, just move our symbol over to it.
  if (!AddrLabelSymbols.count(New)) {
    BBCallbacks[OldEntry.Index] = New;    // Update the callback.
    AddrLabelSymbols[New] = OldEntry;     // Set New's entry.
  } else {
    assert(0 && "Case not handled yet!");
    abort();
  }
}


void MMIAddrLabelMapCallbackPtr::deleted() {
  Map->UpdateForDeletedBlock(cast<BasicBlock>(getValPtr()));
}

void MMIAddrLabelMapCallbackPtr::allUsesReplacedWith(Value *V2) {
  Map->UpdateForRAUWBlock(cast<BasicBlock>(getValPtr()), cast<BasicBlock>(V2));
}


//===----------------------------------------------------------------------===//

MachineModuleInfo::MachineModuleInfo(const MCAsmInfo &MAI)
: ImmutablePass(&ID), Context(MAI),
  ObjFileMMI(0),
  CurCallSite(0), CallsEHReturn(0), CallsUnwindInit(0), DbgInfoAvailable(false){
  // Always emit some info, by default "no personality" info.
  Personalities.push_back(NULL);
  AddrLabelSymbols = 0;
}

MachineModuleInfo::MachineModuleInfo()
: ImmutablePass(&ID), Context(*(MCAsmInfo*)0) {
  assert(0 && "This MachineModuleInfo constructor should never be called, MMI "
         "should always be explicitly constructed by LLVMTargetMachine");
  abort();
}

MachineModuleInfo::~MachineModuleInfo() {
  delete ObjFileMMI;
  
  // FIXME: Why isn't doFinalization being called??
  //assert(AddrLabelSymbols == 0 && "doFinalization not called");
  delete AddrLabelSymbols;
  AddrLabelSymbols = 0;
}

/// doInitialization - Initialize the state for a new module.
///
bool MachineModuleInfo::doInitialization() {
  assert(AddrLabelSymbols == 0 && "Improperly initialized");
  return false;
}

/// doFinalization - Tear down the state after completion of a module.
///
bool MachineModuleInfo::doFinalization() {
  delete AddrLabelSymbols;
  AddrLabelSymbols = 0;
  return false;
}

/// EndFunction - Discard function meta information.
///
void MachineModuleInfo::EndFunction() {
  // Clean up frame info.
  FrameMoves.clear();

  // Clean up exception info.
  LandingPads.clear();
  CallSiteMap.clear();
  TypeInfos.clear();
  FilterIds.clear();
  FilterEnds.clear();
  CallsEHReturn = 0;
  CallsUnwindInit = 0;
  VariableDbgInfo.clear();
}

/// AnalyzeModule - Scan the module for global debug information.
///
void MachineModuleInfo::AnalyzeModule(Module &M) {
  // Insert functions in the llvm.used array (but not llvm.compiler.used) into
  // UsedFunctions.
  GlobalVariable *GV = M.getGlobalVariable("llvm.used");
  if (!GV || !GV->hasInitializer()) return;

  // Should be an array of 'i8*'.
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (InitList == 0) return;

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (Function *F =
          dyn_cast<Function>(InitList->getOperand(i)->stripPointerCasts()))
      UsedFunctions.insert(F);
}

//===- Address of Block Management ----------------------------------------===//


/// getAddrLabelSymbol - Return the symbol to be used for the specified basic
/// block when its address is taken.  This cannot be its normal LBB label
/// because the block may be accessed outside its containing function.
MCSymbol *MachineModuleInfo::getAddrLabelSymbol(const BasicBlock *BB) {
  // Lazily create AddrLabelSymbols.
  if (AddrLabelSymbols == 0)
    AddrLabelSymbols = new MMIAddrLabelMap(Context);
  return AddrLabelSymbols->getAddrLabelSymbol(const_cast<BasicBlock*>(BB));
}


//===- EH -----------------------------------------------------------------===//

/// getOrCreateLandingPadInfo - Find or create an LandingPadInfo for the
/// specified MachineBasicBlock.
LandingPadInfo &MachineModuleInfo::getOrCreateLandingPadInfo
    (MachineBasicBlock *LandingPad) {
  unsigned N = LandingPads.size();
  for (unsigned i = 0; i < N; ++i) {
    LandingPadInfo &LP = LandingPads[i];
    if (LP.LandingPadBlock == LandingPad)
      return LP;
  }

  LandingPads.push_back(LandingPadInfo(LandingPad));
  return LandingPads[N];
}

/// addInvoke - Provide the begin and end labels of an invoke style call and
/// associate it with a try landing pad block.
void MachineModuleInfo::addInvoke(MachineBasicBlock *LandingPad,
                                  MCSymbol *BeginLabel, MCSymbol *EndLabel) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.BeginLabels.push_back(BeginLabel);
  LP.EndLabels.push_back(EndLabel);
}

/// addLandingPad - Provide the label of a try LandingPad block.
///
MCSymbol *MachineModuleInfo::addLandingPad(MachineBasicBlock *LandingPad) {
  MCSymbol *LandingPadLabel = Context.CreateTempSymbol();
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.LandingPadLabel = LandingPadLabel;
  return LandingPadLabel;
}

/// addPersonality - Provide the personality function for the exception
/// information.
void MachineModuleInfo::addPersonality(MachineBasicBlock *LandingPad,
                                       Function *Personality) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.Personality = Personality;

  for (unsigned i = 0; i < Personalities.size(); ++i)
    if (Personalities[i] == Personality)
      return;

  // If this is the first personality we're adding go
  // ahead and add it at the beginning.
  if (Personalities[0] == NULL)
    Personalities[0] = Personality;
  else
    Personalities.push_back(Personality);
}

/// addCatchTypeInfo - Provide the catch typeinfo for a landing pad.
///
void MachineModuleInfo::addCatchTypeInfo(MachineBasicBlock *LandingPad,
                                        std::vector<GlobalVariable *> &TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  for (unsigned N = TyInfo.size(); N; --N)
    LP.TypeIds.push_back(getTypeIDFor(TyInfo[N - 1]));
}

/// addFilterTypeInfo - Provide the filter typeinfo for a landing pad.
///
void MachineModuleInfo::addFilterTypeInfo(MachineBasicBlock *LandingPad,
                                        std::vector<GlobalVariable *> &TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  std::vector<unsigned> IdsInFilter(TyInfo.size());
  for (unsigned I = 0, E = TyInfo.size(); I != E; ++I)
    IdsInFilter[I] = getTypeIDFor(TyInfo[I]);
  LP.TypeIds.push_back(getFilterIDFor(IdsInFilter));
}

/// addCleanup - Add a cleanup action for a landing pad.
///
void MachineModuleInfo::addCleanup(MachineBasicBlock *LandingPad) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.TypeIds.push_back(0);
}

/// TidyLandingPads - Remap landing pad labels and remove any deleted landing
/// pads.
void MachineModuleInfo::TidyLandingPads() {
  for (unsigned i = 0; i != LandingPads.size(); ) {
    LandingPadInfo &LandingPad = LandingPads[i];
    if (LandingPad.LandingPadLabel && !LandingPad.LandingPadLabel->isDefined())
      LandingPad.LandingPadLabel = 0;

    // Special case: we *should* emit LPs with null LP MBB. This indicates
    // "nounwind" case.
    if (!LandingPad.LandingPadLabel && LandingPad.LandingPadBlock) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    for (unsigned j = 0, e = LandingPads[i].BeginLabels.size(); j != e; ++j) {
      MCSymbol *BeginLabel = LandingPad.BeginLabels[j];
      MCSymbol *EndLabel = LandingPad.EndLabels[j];
      if (BeginLabel->isDefined() && EndLabel->isDefined()) continue;
      
      LandingPad.BeginLabels.erase(LandingPad.BeginLabels.begin() + j);
      LandingPad.EndLabels.erase(LandingPad.EndLabels.begin() + j);
      --j, --e;
    }

    // Remove landing pads with no try-ranges.
    if (LandingPads[i].BeginLabels.empty()) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    // If there is no landing pad, ensure that the list of typeids is empty.
    // If the only typeid is a cleanup, this is the same as having no typeids.
    if (!LandingPad.LandingPadBlock ||
        (LandingPad.TypeIds.size() == 1 && !LandingPad.TypeIds[0]))
      LandingPad.TypeIds.clear();
    ++i;
  }
}

/// getTypeIDFor - Return the type id for the specified typeinfo.  This is
/// function wide.
unsigned MachineModuleInfo::getTypeIDFor(GlobalVariable *TI) {
  for (unsigned i = 0, N = TypeInfos.size(); i != N; ++i)
    if (TypeInfos[i] == TI) return i + 1;

  TypeInfos.push_back(TI);
  return TypeInfos.size();
}

/// getFilterIDFor - Return the filter id for the specified typeinfos.  This is
/// function wide.
int MachineModuleInfo::getFilterIDFor(std::vector<unsigned> &TyIds) {
  // If the new filter coincides with the tail of an existing filter, then
  // re-use the existing filter.  Folding filters more than this requires
  // re-ordering filters and/or their elements - probably not worth it.
  for (std::vector<unsigned>::iterator I = FilterEnds.begin(),
       E = FilterEnds.end(); I != E; ++I) {
    unsigned i = *I, j = TyIds.size();

    while (i && j)
      if (FilterIds[--i] != TyIds[--j])
        goto try_next;

    if (!j)
      // The new filter coincides with range [i, end) of the existing filter.
      return -(1 + i);

try_next:;
  }

  // Add the new filter.
  int FilterID = -(1 + FilterIds.size());
  FilterIds.reserve(FilterIds.size() + TyIds.size() + 1);
  for (unsigned I = 0, N = TyIds.size(); I != N; ++I)
    FilterIds.push_back(TyIds[I]);
  FilterEnds.push_back(FilterIds.size());
  FilterIds.push_back(0); // terminator
  return FilterID;
}

/// getPersonality - Return the personality function for the current function.
Function *MachineModuleInfo::getPersonality() const {
  // FIXME: Until PR1414 will be fixed, we're using 1 personality function per
  // function
  return !LandingPads.empty() ? LandingPads[0].Personality : NULL;
}

/// getPersonalityIndex - Return unique index for current personality
/// function. NULL/first personality function should always get zero index.
unsigned MachineModuleInfo::getPersonalityIndex() const {
  const Function* Personality = NULL;

  // Scan landing pads. If there is at least one non-NULL personality - use it.
  for (unsigned i = 0; i != LandingPads.size(); ++i)
    if (LandingPads[i].Personality) {
      Personality = LandingPads[i].Personality;
      break;
    }

  for (unsigned i = 0; i < Personalities.size(); ++i) {
    if (Personalities[i] == Personality)
      return i;
  }

  // This will happen if the current personality function is
  // in the zero index.
  return 0;
}


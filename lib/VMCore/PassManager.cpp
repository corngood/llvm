//===- PassManager.cpp - LLVM Pass Infrastructure Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass Manager infrastructure. 
//
//===----------------------------------------------------------------------===//


#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/ManagedStatic.h"
#include <vector>
#include <map>

using namespace llvm;
class llvm::PMDataManager;

//===----------------------------------------------------------------------===//
// Overview:
// The Pass Manager Infrastructure manages passes. It's responsibilities are:
// 
//   o Manage optimization pass execution order
//   o Make required Analysis information available before pass P is run
//   o Release memory occupied by dead passes
//   o If Analysis information is dirtied by a pass then regenerate Analysis 
//     information before it is consumed by another pass.
//
// Pass Manager Infrastructure uses multiple pass managers.  They are
// PassManager, FunctionPassManager, ModulePassManager, BasicBlockPassManager.
// This class hierarcy uses multiple inheritance but pass managers do not derive
// from another pass manager.
//
// PassManager and FunctionPassManager are two top-level pass manager that
// represents the external interface of this entire pass manager infrastucture.
//
// Important classes :
//
// [o] class PMTopLevelManager;
//
// Two top level managers, PassManager and FunctionPassManager, derive from 
// PMTopLevelManager. PMTopLevelManager manages information used by top level 
// managers such as last user info.
//
// [o] class PMDataManager;
//
// PMDataManager manages information, e.g. list of available analysis info, 
// used by a pass manager to manage execution order of passes. It also provides
// a place to implement common pass manager APIs. All pass managers derive from
// PMDataManager.
//
// [o] class BasicBlockPassManager : public FunctionPass, public PMDataManager;
//
// BasicBlockPassManager manages BasicBlockPasses.
//
// [o] class FunctionPassManager;
//
// This is a external interface used by JIT to manage FunctionPasses. This
// interface relies on FunctionPassManagerImpl to do all the tasks.
//
// [o] class FunctionPassManagerImpl : public ModulePass, PMDataManager,
//                                     public PMTopLevelManager;
//
// FunctionPassManagerImpl is a top level manager. It manages FunctionPasses
// and BasicBlockPassManagers.
//
// [o] class ModulePassManager : public Pass, public PMDataManager;
//
// ModulePassManager manages ModulePasses and FunctionPassManagerImpls.
//
// [o] class PassManager;
//
// This is a external interface used by various tools to manages passes. It
// relies on PassManagerImpl to do all the tasks.
//
// [o] class PassManagerImpl : public Pass, public PMDataManager,
//                             public PMDTopLevelManager
//
// PassManagerImpl is a top level pass manager responsible for managing
// ModulePassManagers.
//===----------------------------------------------------------------------===//

namespace llvm {

//===----------------------------------------------------------------------===//
// Pass debugging information.  Often it is useful to find out what pass is
// running when a crash occurs in a utility.  When this library is compiled with
// debugging on, a command line option (--debug-pass) is enabled that causes the
// pass name to be printed before it executes.
//

// Different debug levels that can be enabled...
enum PassDebugLevel {
  None, Arguments, Structure, Executions, Details
};

static cl::opt<enum PassDebugLevel>
PassDebugging_New("debug-pass", cl::Hidden,
                  cl::desc("Print PassManager debugging information"),
                  cl::values(
  clEnumVal(None      , "disable debug output"),
  clEnumVal(Arguments , "print pass arguments to pass to 'opt'"),
  clEnumVal(Structure , "print pass structure before run()"),
  clEnumVal(Executions, "print pass name before it is executed"),
  clEnumVal(Details   , "print pass details when it is executed"),
                             clEnumValEnd));
} // End of llvm namespace

#ifndef USE_OLD_PASSMANAGER
namespace {

//===----------------------------------------------------------------------===//
// PMTopLevelManager
//
/// PMTopLevelManager manages LastUser info and collects common APIs used by
/// top level pass managers.
class VISIBILITY_HIDDEN PMTopLevelManager {
public:

  inline std::vector<Pass *>::iterator passManagersBegin() { 
    return PassManagers.begin(); 
  }

  inline std::vector<Pass *>::iterator passManagersEnd() { 
    return PassManagers.end();
  }

  /// Schedule pass P for execution. Make sure that passes required by
  /// P are run before P is run. Update analysis info maintained by
  /// the manager. Remove dead passes. This is a recursive function.
  void schedulePass(Pass *P);

  /// This is implemented by top level pass manager and used by 
  /// schedulePass() to add analysis info passes that are not available.
  virtual void addTopLevelPass(Pass  *P) = 0;

  /// Set pass P as the last user of the given analysis passes.
  void setLastUser(std::vector<Pass *> &AnalysisPasses, Pass *P);

  /// Collect passes whose last user is P
  void collectLastUses(std::vector<Pass *> &LastUses, Pass *P);

  /// Find the pass that implements Analysis AID. Search immutable
  /// passes and all pass managers. If desired pass is not found
  /// then return NULL.
  Pass *findAnalysisPass(AnalysisID AID);

  inline void clearManagers() { 
    PassManagers.clear();
  }

  virtual ~PMTopLevelManager() {
    for (std::vector<Pass *>::iterator I = PassManagers.begin(),
           E = PassManagers.end(); I != E; ++I)
      delete *I;

    for (std::vector<ImmutablePass *>::iterator
           I = ImmutablePasses.begin(), E = ImmutablePasses.end(); I != E; ++I)
      delete *I;

    PassManagers.clear();
  }

  /// Add immutable pass and initialize it.
  inline void addImmutablePass(ImmutablePass *P) {
    P->initializePass();
    ImmutablePasses.push_back(P);
  }

  inline std::vector<ImmutablePass *>& getImmutablePasses() {
    return ImmutablePasses;
  }

  void addPassManager(Pass *Manager) {
    PassManagers.push_back(Manager);
  }

  // Add Manager into the list of managers that are not directly
  // maintained by this top level pass manager
  inline void addIndirectPassManager(PMDataManager *Manager) {
    IndirectPassManagers.push_back(Manager);
  }

  // Print passes managed by this top level manager.
  void dumpPasses();
  void dumpArguments();

private:
  
  /// Collection of pass managers
  std::vector<Pass *> PassManagers;

  /// Collection of pass managers that are not directly maintained
  /// by this pass manager
  std::vector<PMDataManager *> IndirectPassManagers;

  // Map to keep track of last user of the analysis pass.
  // LastUser->second is the last user of Lastuser->first.
  std::map<Pass *, Pass *> LastUser;

  /// Immutable passes are managed by top level manager.
  std::vector<ImmutablePass *> ImmutablePasses;
};

} // End of anon namespace
  
//===----------------------------------------------------------------------===//
// PMDataManager

namespace llvm {
/// PMDataManager provides the common place to manage the analysis data
/// used by pass managers.
class PMDataManager {
public:
  PMDataManager(int D) : TPM(NULL), Depth(D) {
    initializeAnalysisInfo();
  }

  virtual ~PMDataManager() {

    for (std::vector<Pass *>::iterator I = PassVector.begin(),
           E = PassVector.end(); I != E; ++I)
      delete *I;

    PassVector.clear();
  }

  /// Return true IFF pass P's required analysis set does not required new
  /// manager.
  bool manageablePass(Pass *P);

  /// Augment AvailableAnalysis by adding analysis made available by pass P.
  void recordAvailableAnalysis(Pass *P);

  /// Remove Analysis that is not preserved by the pass
  void removeNotPreservedAnalysis(Pass *P);
  
  /// Remove dead passes
  void removeDeadPasses(Pass *P, std::string &Msg);

  /// Add pass P into the PassVector. Update 
  /// AvailableAnalysis appropriately if ProcessAnalysis is true.
  void addPassToManager(Pass *P, bool ProcessAnalysis = true);

  /// Initialize available analysis information.
  void initializeAnalysisInfo() { 
    TransferLastUses.clear();
    AvailableAnalysis.clear();
  }

  /// Populate RequiredPasses with the analysis pass that are required by
  /// pass P.
  void collectRequiredAnalysisPasses(std::vector<Pass *> &RequiredPasses,
                                     Pass *P);

  /// All Required analyses should be available to the pass as it runs!  Here
  /// we fill in the AnalysisImpls member of the pass so that it can
  /// successfully use the getAnalysis() method to retrieve the
  /// implementations it needs.
  void initializeAnalysisImpl(Pass *P);

  /// Find the pass that implements Analysis AID. If desired pass is not found
  /// then return NULL.
  Pass *findAnalysisPass(AnalysisID AID, bool Direction);

  inline std::vector<Pass *>::iterator passVectorBegin() { 
    return PassVector.begin(); 
  }

  inline std::vector<Pass *>::iterator passVectorEnd() { 
    return PassVector.end();
  }

  // Access toplevel manager
  PMTopLevelManager *getTopLevelManager() { return TPM; }
  void setTopLevelManager(PMTopLevelManager *T) { TPM = T; }

  unsigned getDepth() { return Depth; }

  // Print list of passes that are last used by P.
  void dumpLastUses(Pass *P, unsigned Offset) {
    
    std::vector<Pass *> LUses;

    assert (TPM && "Top Level Manager is missing");
    TPM->collectLastUses(LUses, P);
    
    for (std::vector<Pass *>::iterator I = LUses.begin(),
           E = LUses.end(); I != E; ++I) {
      llvm::cerr << "--" << std::string(Offset*2, ' ');
      (*I)->dumpPassStructure(0);
    }
  }

  void dumpPassArguments() {
    for(std::vector<Pass *>::iterator I = PassVector.begin(),
          E = PassVector.end(); I != E; ++I) {
      if (PMDataManager *PMD = dynamic_cast<PMDataManager *>(*I))
        PMD->dumpPassArguments();
      else
        if (const PassInfo *PI = (*I)->getPassInfo())
          if (!PI->isAnalysisGroup())
            cerr << " -" << PI->getPassArgument();
    }
  }

  void dumpPassInfo(Pass *P,  std::string &Msg1, std::string &Msg2) {
    if (PassDebugging_New < Executions)
      return;
    cerr << (void*)this << std::string(getDepth()*2+1, ' ');
    cerr << Msg1;
    cerr << P->getPassName();
    cerr << Msg2;
  }

  void dumpAnalysisSetInfo(const char *Msg, Pass *P,
                           const std::vector<AnalysisID> &Set) {
    if (PassDebugging_New >= Details && !Set.empty()) {
      cerr << (void*)P << std::string(getDepth()*2+3, ' ') << Msg << " Analyses:";
      for (unsigned i = 0; i != Set.size(); ++i) {
        if (i) cerr << ",";
        cerr << " " << Set[i]->getPassName();
      }
      cerr << "\n";
    }
  }

  std::vector<Pass *>& getTransferredLastUses() {
    return TransferLastUses;
  }

protected:

  // If a FunctionPass F is the last user of ModulePass info M
  // then the F's manager, not F, records itself as a last user of M.
  // Current pass manage is requesting parent manager to record parent
  // manager as the last user of these TrransferLastUses passes.
  std::vector<Pass *> TransferLastUses;

  // Top level manager.
  PMTopLevelManager *TPM;

private:
  // Set of available Analysis. This information is used while scheduling 
  // pass. If a pass requires an analysis which is not not available then 
  // equired analysis pass is scheduled to run before the pass itself is 
  // scheduled to run.
  std::map<AnalysisID, Pass*> AvailableAnalysis;

  // Collection of pass that are managed by this manager
  std::vector<Pass *> PassVector;

  unsigned Depth;
};

//===----------------------------------------------------------------------===//
// BasicBlockPassManager
//
/// BasicBlockPassManager manages BasicBlockPass. It batches all the
/// pass together and sequence them to process one basic block before
/// processing next basic block.
class VISIBILITY_HIDDEN BasicBlockPassManager : public PMDataManager, 
                              public FunctionPass {

public:
  BasicBlockPassManager(int D) : PMDataManager(D) { }

  /// Add a pass into a passmanager queue. 
  bool addPass(Pass *p);
  
  /// Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the function, and if so, return true.
  bool runOnFunction(Function &F);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    Info.setPreservesAll();
  }

  bool doInitialization(Module &M);
  bool doInitialization(Function &F);
  bool doFinalization(Module &M);
  bool doFinalization(Function &F);

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) {
    llvm::cerr << std::string(Offset*2, ' ') << "BasicBlockPass Manager\n";
    for (std::vector<Pass *>::iterator I = passVectorBegin(),
           E = passVectorEnd(); I != E; ++I)  {
      (*I)->dumpPassStructure(Offset + 1);
      dumpLastUses(*I, Offset+1);
    }
  }
};

//===----------------------------------------------------------------------===//
// FunctionPassManagerImpl_New
//
/// FunctionPassManagerImpl_New manages FunctionPasses and
/// BasicBlockPassManagers.  It batches all function passes and basic block pass
/// managers together and sequence them to process one function at a time before
/// processing next function.
class FunctionPassManagerImpl_New : public ModulePass, 
                                    public PMDataManager,
                                    public PMTopLevelManager {
public:
  FunctionPassManagerImpl_New(int D) : PMDataManager(D) { 
    activeBBPassManager = NULL;
  }
  ~FunctionPassManagerImpl_New() { /* TODO */ };
 
  inline void addTopLevelPass(Pass *P) { 

    if (ImmutablePass *IP = dynamic_cast<ImmutablePass *> (P)) {

      // P is a immutable pass then it will be managed by this
      // top level manager. Set up analysis resolver to connect them.
      AnalysisResolver_New *AR = new AnalysisResolver_New(*this);
      P->setResolver(AR);
      initializeAnalysisImpl(P);
      addImmutablePass(IP);
      recordAvailableAnalysis(IP);
    } 
    else 
      addPass(P);
  }

  /// add - Add a pass to the queue of passes to run.  This passes
  /// ownership of the Pass to the PassManager.  When the
  /// PassManager_X is destroyed, the pass will be destroyed as well, so
  /// there is no need to delete the pass. (TODO delete passes.)
  /// This implies that all passes MUST be allocated with 'new'.
  void add(Pass *P) { 
    schedulePass(P);
  }

  /// Add pass into the pass manager queue.
  bool addPass(Pass *P);

  /// Execute all of the passes scheduled for execution.  Keep
  /// track of whether any of the passes modifies the function, and if
  /// so, return true.
  bool runOnModule(Module &M);
  bool runOnFunction(Function &F);
  bool run(Function &F);

  /// doInitialization - Run all of the initializers for the function passes.
  ///
  bool doInitialization(Module &M);
  
  /// doFinalization - Run all of the initializers for the function passes.
  ///
  bool doFinalization(Module &M);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    Info.setPreservesAll();
  }

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) {
    llvm::cerr << std::string(Offset*2, ' ') << "FunctionPass Manager\n";
    for (std::vector<Pass *>::iterator I = passVectorBegin(),
           E = passVectorEnd(); I != E; ++I)  {
      (*I)->dumpPassStructure(Offset + 1);
      dumpLastUses(*I, Offset+1);
    }
  }

private:
  // Active Pass Managers
  BasicBlockPassManager *activeBBPassManager;
};

//===----------------------------------------------------------------------===//
// ModulePassManager
//
/// ModulePassManager manages ModulePasses and function pass managers.
/// It batches all Module passes  passes and function pass managers together and
/// sequence them to process one module.
class ModulePassManager : public Pass, public PMDataManager {
 
public:
  ModulePassManager(int D) : PMDataManager(D) { 
    activeFunctionPassManager = NULL; 
  }
  
  /// Add a pass into a passmanager queue. 
  bool addPass(Pass *p);
  
  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnModule(Module &M);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    Info.setPreservesAll();
  }

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) {
    llvm::cerr << std::string(Offset*2, ' ') << "ModulePass Manager\n";
    for (std::vector<Pass *>::iterator I = passVectorBegin(),
           E = passVectorEnd(); I != E; ++I)  {
      (*I)->dumpPassStructure(Offset + 1);
      dumpLastUses(*I, Offset+1);
    }
  }

private:
  // Active Pass Manager
  FunctionPassManagerImpl_New *activeFunctionPassManager;
};

//===----------------------------------------------------------------------===//
// PassManagerImpl_New
//
/// PassManagerImpl_New manages ModulePassManagers
class PassManagerImpl_New : public Pass,
                            public PMDataManager,
                            public PMTopLevelManager {

public:

  PassManagerImpl_New(int D) : PMDataManager(D) {
    activeManager = NULL;
  }

  /// add - Add a pass to the queue of passes to run.  This passes ownership of
  /// the Pass to the PassManager.  When the PassManager is destroyed, the pass
  /// will be destroyed as well, so there is no need to delete the pass.  This
  /// implies that all passes MUST be allocated with 'new'.
  void add(Pass *P) {
    schedulePass(P);
  }
 
  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool run(Module &M);

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const {
    Info.setPreservesAll();
  }

  inline void addTopLevelPass(Pass *P) {

    if (ImmutablePass *IP = dynamic_cast<ImmutablePass *> (P)) {
      
      // P is a immutable pass and it will be managed by this
      // top level manager. Set up analysis resolver to connect them.
      AnalysisResolver_New *AR = new AnalysisResolver_New(*this);
      P->setResolver(AR);
      initializeAnalysisImpl(P);
      addImmutablePass(IP);
      recordAvailableAnalysis(IP);
    }
    else 
      addPass(P);
  }

private:

  /// Add a pass into a passmanager queue.
  bool addPass(Pass *p);

  // Active Pass Manager
  ModulePassManager *activeManager;
};

} // End of llvm namespace

namespace {

//===----------------------------------------------------------------------===//
// TimingInfo Class - This class is used to calculate information about the
// amount of time each pass takes to execute.  This only happens when
// -time-passes is enabled on the command line.
//

class VISIBILITY_HIDDEN TimingInfo {
  std::map<Pass*, Timer> TimingData;
  TimerGroup TG;

public:
  // Use 'create' member to get this.
  TimingInfo() : TG("... Pass execution timing report ...") {}
  
  // TimingDtor - Print out information about timing information
  ~TimingInfo() {
    // Delete all of the timers...
    TimingData.clear();
    // TimerGroup is deleted next, printing the report.
  }

  // createTheTimeInfo - This method either initializes the TheTimeInfo pointer
  // to a non null value (if the -time-passes option is enabled) or it leaves it
  // null.  It may be called multiple times.
  static void createTheTimeInfo();

  void passStarted(Pass *P) {

    if (dynamic_cast<PMDataManager *>(P)) 
      return;

    std::map<Pass*, Timer>::iterator I = TimingData.find(P);
    if (I == TimingData.end())
      I=TimingData.insert(std::make_pair(P, Timer(P->getPassName(), TG))).first;
    I->second.startTimer();
  }
  void passEnded(Pass *P) {

    if (dynamic_cast<PMDataManager *>(P)) 
      return;

    std::map<Pass*, Timer>::iterator I = TimingData.find(P);
    assert (I != TimingData.end() && "passStarted/passEnded not nested right!");
    I->second.stopTimer();
  }
};

static TimingInfo *TheTimeInfo;

} // End of anon namespace

//===----------------------------------------------------------------------===//
// PMTopLevelManager implementation

/// Set pass P as the last user of the given analysis passes.
void PMTopLevelManager::setLastUser(std::vector<Pass *> &AnalysisPasses, 
                                    Pass *P) {

  for (std::vector<Pass *>::iterator I = AnalysisPasses.begin(),
         E = AnalysisPasses.end(); I != E; ++I) {
    Pass *AP = *I;
    LastUser[AP] = P;
    // If AP is the last user of other passes then make P last user of
    // such passes.
    for (std::map<Pass *, Pass *>::iterator LUI = LastUser.begin(),
           LUE = LastUser.end(); LUI != LUE; ++LUI) {
      if (LUI->second == AP)
        LastUser[LUI->first] = P;
    }
  }
}

/// Collect passes whose last user is P
void PMTopLevelManager::collectLastUses(std::vector<Pass *> &LastUses,
                                            Pass *P) {
   for (std::map<Pass *, Pass *>::iterator LUI = LastUser.begin(),
          LUE = LastUser.end(); LUI != LUE; ++LUI)
      if (LUI->second == P)
        LastUses.push_back(LUI->first);
}

/// Schedule pass P for execution. Make sure that passes required by
/// P are run before P is run. Update analysis info maintained by
/// the manager. Remove dead passes. This is a recursive function.
void PMTopLevelManager::schedulePass(Pass *P) {

  // TODO : Allocate function manager for this pass, other wise required set
  // may be inserted into previous function manager

  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);
  const std::vector<AnalysisID> &RequiredSet = AnUsage.getRequiredSet();
  for (std::vector<AnalysisID>::const_iterator I = RequiredSet.begin(),
         E = RequiredSet.end(); I != E; ++I) {

    Pass *AnalysisPass = findAnalysisPass(*I);
    if (!AnalysisPass) {
      // Schedule this analysis run first.
      AnalysisPass = (*I)->createPass();
      schedulePass(AnalysisPass);
    }
  }

  // Now all required passes are available.
  addTopLevelPass(P);
}

/// Find the pass that implements Analysis AID. Search immutable
/// passes and all pass managers. If desired pass is not found
/// then return NULL.
Pass *PMTopLevelManager::findAnalysisPass(AnalysisID AID) {

  Pass *P = NULL;
  // Check pass managers
  for (std::vector<Pass *>::iterator I = PassManagers.begin(),
         E = PassManagers.end(); P == NULL && I != E; ++I) {
    PMDataManager *PMD = dynamic_cast<PMDataManager *>(*I);
    assert(PMD && "This is not a PassManager");
    P = PMD->findAnalysisPass(AID, false);
  }

  // Check other pass managers
  for (std::vector<PMDataManager *>::iterator I = IndirectPassManagers.begin(),
         E = IndirectPassManagers.end(); P == NULL && I != E; ++I)
    P = (*I)->findAnalysisPass(AID, false);

  for (std::vector<ImmutablePass *>::iterator I = ImmutablePasses.begin(),
         E = ImmutablePasses.end(); P == NULL && I != E; ++I) {
    const PassInfo *PI = (*I)->getPassInfo();
    if (PI == AID)
      P = *I;

    // If Pass not found then check the interfaces implemented by Immutable Pass
    if (!P) {
      const std::vector<const PassInfo*> &ImmPI = 
        PI->getInterfacesImplemented();
      for (unsigned Index = 0, End = ImmPI.size(); 
           P == NULL && Index != End; ++Index)
        if (ImmPI[Index] == AID)
          P = *I;
    }
  }

  return P;
}

// Print passes managed by this top level manager.
void PMTopLevelManager::dumpPasses() {

  // Print out the immutable passes
  for (unsigned i = 0, e = ImmutablePasses.size(); i != e; ++i) {
    ImmutablePasses[i]->dumpPassStructure(0);
  }
  
  for (std::vector<Pass *>::iterator I = PassManagers.begin(),
         E = PassManagers.end(); I != E; ++I)
    (*I)->dumpPassStructure(1);

}

void PMTopLevelManager::dumpArguments() {

  if (PassDebugging_New < Arguments)
    return;

  cerr << "Pass Arguments: ";
  for (std::vector<Pass *>::iterator I = PassManagers.begin(),
         E = PassManagers.end(); I != E; ++I) {
    PMDataManager *PMD = dynamic_cast<PMDataManager *>(*I);
    assert(PMD && "This is not a PassManager");
    PMD->dumpPassArguments();
  }
  cerr << "\n";
}

//===----------------------------------------------------------------------===//
// PMDataManager implementation

/// Return true IFF pass P's required analysis set does not required new
/// manager.
bool PMDataManager::manageablePass(Pass *P) {

  // TODO 
  // If this pass is not preserving information that is required by a
  // pass maintained by higher level pass manager then do not insert
  // this pass into current manager. Use new manager. For example,
  // For example, If FunctionPass F is not preserving ModulePass Info M1
  // that is used by another ModulePass M2 then do not insert F in
  // current function pass manager.
  return true;
}

/// Augement AvailableAnalysis by adding analysis made available by pass P.
void PMDataManager::recordAvailableAnalysis(Pass *P) {
                                                
  if (const PassInfo *PI = P->getPassInfo()) {
    AvailableAnalysis[PI] = P;

    //This pass is the current implementation of all of the interfaces it
    //implements as well.
    const std::vector<const PassInfo*> &II = PI->getInterfacesImplemented();
    for (unsigned i = 0, e = II.size(); i != e; ++i)
      AvailableAnalysis[II[i]] = P;
  }
}

/// Remove Analyss not preserved by Pass P
void PMDataManager::removeNotPreservedAnalysis(Pass *P) {
  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);

  if (AnUsage.getPreservesAll())
    return;

  const std::vector<AnalysisID> &PreservedSet = AnUsage.getPreservedSet();
  for (std::map<AnalysisID, Pass*>::iterator I = AvailableAnalysis.begin(),
         E = AvailableAnalysis.end(); I != E; ) {
    if (std::find(PreservedSet.begin(), PreservedSet.end(), I->first) == 
        PreservedSet.end()) {
      // Remove this analysis
      if (!dynamic_cast<ImmutablePass*>(I->second)) {
        std::map<AnalysisID, Pass*>::iterator J = I++;
        AvailableAnalysis.erase(J);
      } else
        ++I;
    } else
      ++I;
  }
}

/// Remove analysis passes that are not used any longer
void PMDataManager::removeDeadPasses(Pass *P, std::string &Msg) {

  std::vector<Pass *> DeadPasses;
  TPM->collectLastUses(DeadPasses, P);

  for (std::vector<Pass *>::iterator I = DeadPasses.begin(),
         E = DeadPasses.end(); I != E; ++I) {

    std::string Msg1 = "  Freeing Pass '";
    dumpPassInfo(*I, Msg1, Msg);

    if (TheTimeInfo) TheTimeInfo->passStarted(P);
    (*I)->releaseMemory();
    if (TheTimeInfo) TheTimeInfo->passEnded(P);

    std::map<AnalysisID, Pass*>::iterator Pos = 
      AvailableAnalysis.find((*I)->getPassInfo());
    
    // It is possible that pass is already removed from the AvailableAnalysis
    if (Pos != AvailableAnalysis.end())
      AvailableAnalysis.erase(Pos);
  }
}

/// Add pass P into the PassVector. Update 
/// AvailableAnalysis appropriately if ProcessAnalysis is true.
void PMDataManager::addPassToManager(Pass *P, 
                                     bool ProcessAnalysis) {

  // This manager is going to manage pass P. Set up analysis resolver
  // to connect them.
  AnalysisResolver_New *AR = new AnalysisResolver_New(*this);
  P->setResolver(AR);

  if (ProcessAnalysis) {

    // At the moment, this pass is the last user of all required passes.
    std::vector<Pass *> LastUses;
    std::vector<Pass *> RequiredPasses;
    unsigned PDepth = this->getDepth();

    collectRequiredAnalysisPasses(RequiredPasses, P);
    for (std::vector<Pass *>::iterator I = RequiredPasses.begin(),
           E = RequiredPasses.end(); I != E; ++I) {
      Pass *PRequired = *I;
      unsigned RDepth = 0;

      PMDataManager &DM = PRequired->getResolver()->getPMDataManager();
      RDepth = DM.getDepth();

      if (PDepth == RDepth)
        LastUses.push_back(PRequired);
      else if (PDepth >  RDepth) {
        // Let the parent claim responsibility of last use
        TransferLastUses.push_back(PRequired);
      } else {
        // Note : This feature is not yet implemented
        assert (0 && 
                "Unable to handle Pass that requires lower level Analysis pass");
      }
    }

    LastUses.push_back(P);
    TPM->setLastUser(LastUses, P);

    // Take a note of analysis required and made available by this pass.
    // Remove the analysis not preserved by this pass
    removeNotPreservedAnalysis(P);
    recordAvailableAnalysis(P);
  }

  // Add pass
  PassVector.push_back(P);
}

/// Populate RequiredPasses with the analysis pass that are required by
/// pass P.
void PMDataManager::collectRequiredAnalysisPasses(std::vector<Pass *> &RP,
                                                  Pass *P) {
  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);
  const std::vector<AnalysisID> &RequiredSet = AnUsage.getRequiredSet();
  for (std::vector<AnalysisID>::const_iterator 
         I = RequiredSet.begin(), E = RequiredSet.end();
       I != E; ++I) {
    Pass *AnalysisPass = findAnalysisPass(*I, true);
    assert (AnalysisPass && "Analysis pass is not available");
    RP.push_back(AnalysisPass);
  }

  const std::vector<AnalysisID> &IDs = AnUsage.getRequiredTransitiveSet();
  for (std::vector<AnalysisID>::const_iterator I = IDs.begin(),
         E = IDs.end(); I != E; ++I) {
    Pass *AnalysisPass = findAnalysisPass(*I, true);
    assert (AnalysisPass && "Analysis pass is not available");
    RP.push_back(AnalysisPass);
  }
}

// All Required analyses should be available to the pass as it runs!  Here
// we fill in the AnalysisImpls member of the pass so that it can
// successfully use the getAnalysis() method to retrieve the
// implementations it needs.
//
void PMDataManager::initializeAnalysisImpl(Pass *P) {
  AnalysisUsage AnUsage;
  P->getAnalysisUsage(AnUsage);
 
  for (std::vector<const PassInfo *>::const_iterator
         I = AnUsage.getRequiredSet().begin(),
         E = AnUsage.getRequiredSet().end(); I != E; ++I) {
    Pass *Impl = findAnalysisPass(*I, true);
    if (Impl == 0)
      assert(0 && "Analysis used but not available!");
    AnalysisResolver_New *AR = P->getResolver();
    AR->addAnalysisImplsPair(*I, Impl);
  }
}

/// Find the pass that implements Analysis AID. If desired pass is not found
/// then return NULL.
Pass *PMDataManager::findAnalysisPass(AnalysisID AID, bool SearchParent) {

  // Check if AvailableAnalysis map has one entry.
  std::map<AnalysisID, Pass*>::const_iterator I =  AvailableAnalysis.find(AID);

  if (I != AvailableAnalysis.end())
    return I->second;

  // Search Parents through TopLevelManager
  if (SearchParent)
    return TPM->findAnalysisPass(AID);
  
  return NULL;
}


//===----------------------------------------------------------------------===//
// NOTE: Is this the right place to define this method ?
// getAnalysisToUpdate - Return an analysis result or null if it doesn't exist
Pass *AnalysisResolver_New::getAnalysisToUpdate(AnalysisID ID, bool dir) const {
  return PM.findAnalysisPass(ID, dir);
}

//===----------------------------------------------------------------------===//
// BasicBlockPassManager implementation

/// Add pass P into PassVector and return true. If this pass is not
/// manageable by this manager then return false.
bool
BasicBlockPassManager::addPass(Pass *P) {

  BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
  if (!BP)
    return false;

  // If this pass does not preserve anlysis that is used by other passes
  // managed by this manager than it is not a suiable pass for this manager.
  if (!manageablePass(P))
    return false;

  addPassToManager(BP);

  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnBasicBlock method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool
BasicBlockPassManager::runOnFunction(Function &F) {

  if (F.isExternal())
    return false;

  bool Changed = doInitialization(F);
  initializeAnalysisInfo();

  std::string Msg1 = "Executing Pass '";
  std::string Msg3 = "' Made Modification '";

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    for (std::vector<Pass *>::iterator itr = passVectorBegin(),
           e = passVectorEnd(); itr != e; ++itr) {
      Pass *P = *itr;
      AnalysisUsage AnUsage;
      P->getAnalysisUsage(AnUsage);

      std::string Msg2 = "' on BasicBlock '" + (*I).getName() + "'...\n";
      dumpPassInfo(P, Msg1, Msg2);
      dumpAnalysisSetInfo("Required", P, AnUsage.getRequiredSet());

      initializeAnalysisImpl(P);

      BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);
      if (TheTimeInfo) TheTimeInfo->passStarted(P);
      Changed |= BP->runOnBasicBlock(*I);
      if (TheTimeInfo) TheTimeInfo->passEnded(P);

      if (Changed)
        dumpPassInfo(P, Msg3, Msg2);
      dumpAnalysisSetInfo("Preserved", P, AnUsage.getPreservedSet());

      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P, Msg2);
    }
  return Changed | doFinalization(F);
}

// Implement doInitialization and doFinalization
inline bool BasicBlockPassManager::doInitialization(Module &M) {
  bool Changed = false;

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);    
    Changed |= BP->doInitialization(M);
  }

  return Changed;
}

inline bool BasicBlockPassManager::doFinalization(Module &M) {
  bool Changed = false;

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);    
    Changed |= BP->doFinalization(M);
  }

  return Changed;
}

inline bool BasicBlockPassManager::doInitialization(Function &F) {
  bool Changed = false;

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);    
    Changed |= BP->doInitialization(F);
  }

  return Changed;
}

inline bool BasicBlockPassManager::doFinalization(Function &F) {
  bool Changed = false;

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P);    
    Changed |= BP->doFinalization(F);
  }

  return Changed;
}


//===----------------------------------------------------------------------===//
// FunctionPassManager implementation

/// Create new Function pass manager
FunctionPassManager::FunctionPassManager(ModuleProvider *P) {
  FPM = new FunctionPassManagerImpl_New(0);
  // FPM is the top level manager.
  FPM->setTopLevelManager(FPM);

  PMDataManager *PMD = dynamic_cast<PMDataManager *>(FPM);
  AnalysisResolver_New *AR = new AnalysisResolver_New(*PMD);
  FPM->setResolver(AR);
  
  FPM->addPassManager(FPM);
  MP = P;
}

FunctionPassManager::~FunctionPassManager() {
  // Note : FPM maintains one entry in PassManagers vector.
  // This one entry is FPM itself. This is not ideal. One
  // alternative is have one additional layer between
  // FunctionPassManager and FunctionPassManagerImpl.
  // Meanwhile, to avoid going into infinte loop, first
  // remove FPM from its PassMangers vector.
  FPM->clearManagers();
  delete FPM;
}

/// add - Add a pass to the queue of passes to run.  This passes
/// ownership of the Pass to the PassManager.  When the
/// PassManager_X is destroyed, the pass will be destroyed as well, so
/// there is no need to delete the pass. (TODO delete passes.)
/// This implies that all passes MUST be allocated with 'new'.
void FunctionPassManager::add(Pass *P) { 
  FPM->add(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep
/// track of whether any of the passes modifies the function, and if
/// so, return true.
///
bool FunctionPassManager::run(Function &F) {
  std::string errstr;
  if (MP->materializeFunction(&F, &errstr)) {
    cerr << "Error reading bytecode file: " << errstr << "\n";
    abort();
  }
  return FPM->run(F);
}


/// doInitialization - Run all of the initializers for the function passes.
///
bool FunctionPassManager::doInitialization() {
  return FPM->doInitialization(*MP->getModule());
}

/// doFinalization - Run all of the initializers for the function passes.
///
bool FunctionPassManager::doFinalization() {
  return FPM->doFinalization(*MP->getModule());
}

//===----------------------------------------------------------------------===//
// FunctionPassManagerImpl_New implementation

/// Add pass P into the pass manager queue. If P is a BasicBlockPass then
/// either use it into active basic block pass manager or create new basic
/// block pass manager to handle pass P.
bool
FunctionPassManagerImpl_New::addPass(Pass *P) {

  // If P is a BasicBlockPass then use BasicBlockPassManager.
  if (BasicBlockPass *BP = dynamic_cast<BasicBlockPass*>(P)) {

    if (!activeBBPassManager || !activeBBPassManager->addPass(BP)) {

      // If active manager exists then clear its analysis info.
      if (activeBBPassManager)
        activeBBPassManager->initializeAnalysisInfo();

      // Create and add new manager
      activeBBPassManager = 
        new BasicBlockPassManager(getDepth() + 1);
      // Inherit top level manager
      activeBBPassManager->setTopLevelManager(this->getTopLevelManager());

      // Add new manager into current manager's list.
      addPassToManager(activeBBPassManager, false);

      // Add new manager into top level manager's indirect passes list
      PMDataManager *PMD = dynamic_cast<PMDataManager *>(activeBBPassManager);
      assert (PMD && "Manager is not Pass Manager");
      TPM->addIndirectPassManager(PMD);

      // Add pass into new manager. This time it must succeed.
      if (!activeBBPassManager->addPass(BP))
        assert(0 && "Unable to add Pass");

      // If activeBBPassManager transfered any Last Uses then handle them here.
      std::vector<Pass *> &TLU = activeBBPassManager->getTransferredLastUses();
      if (!TLU.empty())
        TPM->setLastUser(TLU, this);

    }


    return true;
  }

  FunctionPass *FP = dynamic_cast<FunctionPass *>(P);
  if (!FP)
    return false;

  // If this pass does not preserve anlysis that is used by other passes
  // managed by this manager than it is not a suiable pass for this manager.
  if (!manageablePass(P))
    return false;

  addPassToManager (FP);

  // If active manager exists then clear its analysis info.
  if (activeBBPassManager) {
    activeBBPassManager->initializeAnalysisInfo();
    activeBBPassManager = NULL;
  }

  return true;
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnFunction method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool FunctionPassManagerImpl_New::runOnModule(Module &M) {

  bool Changed = doInitialization(M);
  initializeAnalysisInfo();

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    this->runOnFunction(*I);

  return Changed | doFinalization(M);
}

/// Execute all of the passes scheduled for execution by invoking 
/// runOnFunction method.  Keep track of whether any of the passes modifies 
/// the function, and if so, return true.
bool FunctionPassManagerImpl_New::runOnFunction(Function &F) {

  bool Changed = false;

  if (F.isExternal())
    return false;

  initializeAnalysisInfo();

  std::string Msg1 = "Executing Pass '";
  std::string Msg3 = "' Made Modification '";

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    AnalysisUsage AnUsage;
    P->getAnalysisUsage(AnUsage);

    std::string Msg2 = "' on Function '" + F.getName() + "'...\n";
    dumpPassInfo(P, Msg1, Msg2);
    dumpAnalysisSetInfo("Required", P, AnUsage.getRequiredSet());

    initializeAnalysisImpl(P);    
    FunctionPass *FP = dynamic_cast<FunctionPass*>(P);

    if (TheTimeInfo) TheTimeInfo->passStarted(P);
    Changed |= FP->runOnFunction(F);
    if (TheTimeInfo) TheTimeInfo->passEnded(P);

    if (Changed)
      dumpPassInfo(P, Msg3, Msg2);
    dumpAnalysisSetInfo("Preserved", P, AnUsage.getPreservedSet());

    removeNotPreservedAnalysis(P);
    recordAvailableAnalysis(P);
    removeDeadPasses(P, Msg2);
  }
  return Changed;
}


inline bool FunctionPassManagerImpl_New::doInitialization(Module &M) {
  bool Changed = false;

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    
    FunctionPass *FP = dynamic_cast<FunctionPass*>(P);
    Changed |= FP->doInitialization(M);
  }

  return Changed;
}

inline bool FunctionPassManagerImpl_New::doFinalization(Module &M) {
  bool Changed = false;

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    
    FunctionPass *FP = dynamic_cast<FunctionPass*>(P);
    Changed |= FP->doFinalization(M);
  }

  return Changed;
}

// Execute all the passes managed by this top level manager.
// Return true if any function is modified by a pass.
bool FunctionPassManagerImpl_New::run(Function &F) {

  bool Changed = false;
  for (std::vector<Pass *>::iterator I = passManagersBegin(),
         E = passManagersEnd(); I != E; ++I) {
    FunctionPassManagerImpl_New *FP = 
      dynamic_cast<FunctionPassManagerImpl_New *>(*I);
    Changed |= FP->runOnFunction(F);
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// ModulePassManager implementation

/// Add P into pass vector if it is manageble. If P is a FunctionPass
/// then use FunctionPassManagerImpl_New to manage it. Return false if P
/// is not manageable by this manager.
bool
ModulePassManager::addPass(Pass *P) {

  // If P is FunctionPass then use function pass maanager.
  if (FunctionPass *FP = dynamic_cast<FunctionPass*>(P)) {

    if (!activeFunctionPassManager || !activeFunctionPassManager->addPass(P)) {

      // If active manager exists then clear its analysis info.
      if (activeFunctionPassManager) 
        activeFunctionPassManager->initializeAnalysisInfo();

      // Create and add new manager
      activeFunctionPassManager = 
        new FunctionPassManagerImpl_New(getDepth() + 1);
      
      // Add new manager into current manager's list
      addPassToManager(activeFunctionPassManager, false);

      // Inherit top level manager
      activeFunctionPassManager->setTopLevelManager(this->getTopLevelManager());

      // Add new manager into top level manager's indirect passes list
      PMDataManager *PMD =
        dynamic_cast<PMDataManager *>(activeFunctionPassManager);
      assert(PMD && "Manager is not Pass Manager");
      TPM->addIndirectPassManager(PMD);
      
      // Add pass into new manager. This time it must succeed.
      if (!activeFunctionPassManager->addPass(FP))
        assert(0 && "Unable to add pass");

      // If activeFunctionPassManager transfered any Last Uses then 
      // handle them here.
      std::vector<Pass *> &TLU = 
        activeFunctionPassManager->getTransferredLastUses();
      if (!TLU.empty())
        TPM->setLastUser(TLU, this);
    }

    return true;
  }

  ModulePass *MP = dynamic_cast<ModulePass *>(P);
  if (!MP)
    return false;

  // If this pass does not preserve anlysis that is used by other passes
  // managed by this manager than it is not a suiable pass for this manager.
  if (!manageablePass(P))
    return false;

  addPassToManager(MP);
  // If active manager exists then clear its analysis info.
  if (activeFunctionPassManager) {
    activeFunctionPassManager->initializeAnalysisInfo();
    activeFunctionPassManager = NULL;
  }

  return true;
}


/// Execute all of the passes scheduled for execution by invoking 
/// runOnModule method.  Keep track of whether any of the passes modifies 
/// the module, and if so, return true.
bool
ModulePassManager::runOnModule(Module &M) {
  bool Changed = false;
  initializeAnalysisInfo();

  std::string Msg1 = "Executing Pass '";
  std::string Msg3 = "' Made Modification '";

  for (std::vector<Pass *>::iterator itr = passVectorBegin(),
         e = passVectorEnd(); itr != e; ++itr) {
    Pass *P = *itr;
    AnalysisUsage AnUsage;
    P->getAnalysisUsage(AnUsage);

    std::string Msg2 = "' on Module '" + M.getModuleIdentifier() + "'...\n";
    dumpPassInfo(P, Msg1, Msg2);
    dumpAnalysisSetInfo("Required", P, AnUsage.getRequiredSet());

    initializeAnalysisImpl(P);
    ModulePass *MP = dynamic_cast<ModulePass*>(P);

    if (TheTimeInfo) TheTimeInfo->passStarted(P);
    Changed |= MP->runOnModule(M);
    if (TheTimeInfo) TheTimeInfo->passEnded(P);

    if (Changed)
      dumpPassInfo(P, Msg3, Msg2);
    dumpAnalysisSetInfo("Preserved", P, AnUsage.getPreservedSet());
      
    removeNotPreservedAnalysis(P);
    recordAvailableAnalysis(P);
    removeDeadPasses(P, Msg2);
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// PassManagerImpl implementation
//
/// Add P into active pass manager or use new module pass manager to
/// manage it.
bool PassManagerImpl_New::addPass(Pass *P) {

  if (!activeManager || !activeManager->addPass(P)) {
    activeManager = new ModulePassManager(getDepth() + 1);
    // Inherit top level manager
    activeManager->setTopLevelManager(this->getTopLevelManager());

    // This top level manager is going to manage activeManager. 
    // Set up analysis resolver to connect them.
    AnalysisResolver_New *AR = new AnalysisResolver_New(*this);
    activeManager->setResolver(AR);

    addPassManager(activeManager);
    return activeManager->addPass(P);
  }
  return true;
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool PassManagerImpl_New::run(Module &M) {

  bool Changed = false;

  TimingInfo::createTheTimeInfo();

  dumpArguments();
  if (PassDebugging_New >= Structure)
    dumpPasses();

  for (std::vector<Pass *>::iterator I = passManagersBegin(),
         E = passManagersEnd(); I != E; ++I) {
    ModulePassManager *MP = dynamic_cast<ModulePassManager *>(*I);
    Changed |= MP->runOnModule(M);
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// PassManager implementation

/// Create new pass manager
PassManager::PassManager() {
  PM = new PassManagerImpl_New(0);
  // PM is the top level manager
  PM->setTopLevelManager(PM);
}

PassManager::~PassManager() {
  delete PM;
}

/// add - Add a pass to the queue of passes to run.  This passes ownership of
/// the Pass to the PassManager.  When the PassManager is destroyed, the pass
/// will be destroyed as well, so there is no need to delete the pass.  This
/// implies that all passes MUST be allocated with 'new'.
void 
PassManager::add(Pass *P) {
  PM->add(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool
PassManager::run(Module &M) {
  return PM->run(M);
}

//===----------------------------------------------------------------------===//
// TimingInfo Class - This class is used to calculate information about the
// amount of time each pass takes to execute.  This only happens with
// -time-passes is enabled on the command line.
//
bool llvm::TimePassesIsEnabled = false;
static cl::opt<bool,true>
EnableTiming("time-passes", cl::location(TimePassesIsEnabled),
            cl::desc("Time each pass, printing elapsed time for each on exit"));

// createTheTimeInfo - This method either initializes the TheTimeInfo pointer to
// a non null value (if the -time-passes option is enabled) or it leaves it
// null.  It may be called multiple times.
void TimingInfo::createTheTimeInfo() {
  if (!TimePassesIsEnabled || TheTimeInfo) return;

  // Constructed the first time this is called, iff -time-passes is enabled.
  // This guarantees that the object will be constructed before static globals,
  // thus it will be destroyed before them.
  static ManagedStatic<TimingInfo> TTI;
  TheTimeInfo = &*TTI;
}

#endif

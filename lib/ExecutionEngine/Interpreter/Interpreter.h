//===-- Interpreter.h ------------------------------------------*- C++ -*--===//
//
// This header file defines the interpreter structure
//
//===----------------------------------------------------------------------===//

#ifndef LLI_INTERPRETER_H
#define LLI_INTERPRETER_H

// Uncomment this line to enable profiling of structure field accesses.
//#define PROFILE_STRUCTURE_FIELDS 1

#include "../ExecutionEngine.h"
#include "Support/DataTypes.h"
#include "llvm/Assembly/CachedWriter.h"
#include "llvm/Target/TargetData.h"
#include "llvm/BasicBlock.h"
#include "../GenericValue.h"

extern CachedWriter CW;     // Object to accelerate printing of LLVM

struct MethodInfo;          // Defined in ExecutionAnnotations.h
class CallInst;
class ReturnInst;
class BranchInst;
class LoadInst;
class StoreInst;
class AllocationInst;

// AllocaHolder - Object to track all of the blocks of memory allocated by
// alloca.  When the function returns, this object is poped off the execution
// stack, which causes the dtor to be run, which frees all the alloca'd memory.
//
class AllocaHolder {
  friend class AllocaHolderHandle;
  std::vector<void*> Allocations;
  unsigned RefCnt;
public:
  AllocaHolder() : RefCnt(0) {}
  void add(void *mem) { Allocations.push_back(mem); }
  ~AllocaHolder() {
    for (unsigned i = 0; i < Allocations.size(); ++i)
      free(Allocations[i]);
  }
};

// AllocaHolderHandle gives AllocaHolder value semantics so we can stick it into
// a vector...
//
class AllocaHolderHandle {
  AllocaHolder *H;
public:
  AllocaHolderHandle() : H(new AllocaHolder()) { H->RefCnt++; }
  AllocaHolderHandle(const AllocaHolderHandle &AH) : H(AH.H) { H->RefCnt++; }
  ~AllocaHolderHandle() { if (--H->RefCnt == 0) delete H; }

  void add(void *mem) { H->add(mem); }
};

typedef std::vector<GenericValue> ValuePlaneTy;

// ExecutionContext struct - This struct represents one stack frame currently
// executing.
//
struct ExecutionContext {
  Function             *CurMethod;  // The currently executing function
  BasicBlock           *CurBB;      // The currently executing BB
  BasicBlock::iterator  CurInst;    // The next instruction to execute
  MethodInfo           *MethInfo;   // The MethInfo annotation for the function
  std::vector<ValuePlaneTy>  Values;// ValuePlanes for each type
  std::vector<GenericValue>  VarArgs; // Values passed through an ellipsis

  BasicBlock           *PrevBB;     // The previous BB or null if in first BB
  CallInst             *Caller;     // Holds the call that called subframes.
                                    // NULL if main func or debugger invoked fn
  AllocaHolderHandle    Allocas;    // Track memory allocated by alloca
};

// Interpreter - This class represents the entirety of the interpreter.
//
class Interpreter : public ExecutionEngine {
  int ExitCode;                // The exit code to be returned by the lli util
  bool Debug;                  // Debug mode enabled?
  bool Profile;                // Profiling enabled?
  bool Trace;                  // Tracing enabled?
  int CurFrame;                // The current stack frame being inspected
  TargetData TD;

  // The runtime stack of executing code.  The top of the stack is the current
  // function record.
  std::vector<ExecutionContext> ECStack;

public:
  Interpreter(Module *M, unsigned Config, bool DebugMode, bool TraceMode);
  inline ~Interpreter() { CW.setModule(0); }

  // getExitCode - return the code that should be the exit code for the lli
  // utility.
  inline int getExitCode() const { return ExitCode; }

  /// run - Start execution with the specified function and arguments.
  ///
  virtual int run(const std::string &FnName,
		  const std::vector<std::string> &Args);
 

  // enableProfiling() - Turn profiling on, clear stats?
  void enableProfiling() { Profile = true; }
  void enableTracing() { Trace = true; }

  void handleUserInput();

  // User Interation Methods...
  bool callMethod(const std::string &Name);      // return true on failure
  void setBreakpoint(const std::string &Name);
  void infoValue(const std::string &Name);
  void print(const std::string &Name);
  static void print(const Type *Ty, GenericValue V);
  static void printValue(const Type *Ty, GenericValue V);

  bool callMainMethod(const std::string &MainName,
                      const std::vector<std::string> &InputFilename);

  void list();             // Do the 'list' command
  void printStackTrace();  // Do the 'backtrace' command

  // Code execution methods...
  void callMethod(Function *F, const std::vector<GenericValue> &ArgVals);
  bool executeInstruction(); // Execute one instruction...

  void stepInstruction();  // Do the 'step' command
  void nextInstruction();  // Do the 'next' command
  void run();              // Do the 'run' command
  void finish();           // Do the 'finish' command

  // Opcode Implementations
  void executeCallInst(CallInst &I, ExecutionContext &SF);
  void executeRetInst(ReturnInst &I, ExecutionContext &SF);
  void executeBrInst(BranchInst &I, ExecutionContext &SF);
  void executeAllocInst(AllocationInst &I, ExecutionContext &SF);
  GenericValue callExternalMethod(Function *F, 
                                  const std::vector<GenericValue> &ArgVals);
  void exitCalled(GenericValue GV);

  // getCurrentMethod - Return the currently executing method
  inline Function *getCurrentMethod() const {
    return CurFrame < 0 ? 0 : ECStack[CurFrame].CurMethod;
  }

  // isStopped - Return true if a program is stopped.  Return false if no
  // program is running.
  //
  inline bool isStopped() const { return !ECStack.empty(); }

  //FIXME: private:
public:
  GenericValue executeGEPOperation(Value *Ptr, User::op_iterator I,
				   User::op_iterator E, ExecutionContext &SF);
  void executeLoadInst(LoadInst &I, ExecutionContext &SF);
  void executeStoreInst(StoreInst &I, ExecutionContext &SF);


private:  // Helper functions
  void *getPointerToFunction(const Function *F) { return (void*)F; }

  // getCurrentExecutablePath() - Return the directory that the lli executable
  // lives in.
  //
  std::string getCurrentExecutablePath() const;

  // printCurrentInstruction - Print out the instruction that the virtual PC is
  // at, or fail silently if no program is running.
  //
  void printCurrentInstruction();

  // printStackFrame - Print information about the specified stack frame, or -1
  // for the default one.
  //
  void printStackFrame(int FrameNo = -1);

  // LookupMatchingNames - Search the current function namespace, then the
  // global namespace looking for values that match the specified name.  Return
  // ALL matches to that name.  This is obviously slow, and should only be used
  // for user interaction.
  //
  std::vector<Value*> LookupMatchingNames(const std::string &Name);

  // ChooseOneOption - Prompt the user to choose among the specified options to
  // pick one value.  If no options are provided, emit an error.  If a single 
  // option is provided, just return that option.
  //
  Value *ChooseOneOption(const std::string &Name,
                         const std::vector<Value*> &Opts);


  void initializeExecutionEngine();
  void initializeExternalMethods();
};

#endif

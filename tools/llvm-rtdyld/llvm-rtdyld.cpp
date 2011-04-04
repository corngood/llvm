//===-- llvm-rtdyld.cpp - MCJIT Testing Tool ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testing tool for use with the MC-JIT LLVM components.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

enum ActionType {
  AC_Execute
};

static cl::opt<ActionType>
Action(cl::desc("Action to perform:"),
       cl::init(AC_Execute),
       cl::values(clEnumValN(AC_Execute, "execute",
                             "Load, link, and execute the inputs."),
                  clEnumValEnd));

/* *** */

// A trivial memory manager that doesn't do anything fancy, just uses the
// support library allocation routines directly.
class TrivialMemoryManager : public RTDyldMemoryManager {
public:
  uint64_t startFunctionBody(const char *Name, uintptr_t &Size);
  void endFunctionBody(const char *Name, uint64_t FunctionStart,
                       uint64_t FunctionEnd) {}
};

uint64_t TrivialMemoryManager::startFunctionBody(const char *Name,
                                                 uintptr_t &Size) {
  return (uint64_t)sys::Memory::AllocateRWX(Size, 0, 0).base();
}

static const char *ProgramName;

static void Message(const char *Type, const Twine &Msg) {
  errs() << ProgramName << ": " << Type << ": " << Msg << "\n";
}

static int Error(const Twine &Msg) {
  Message("error", Msg);
  return 1;
}

/* *** */

static int executeInput() {
  // Load the input memory buffer.
  OwningPtr<MemoryBuffer> InputBuffer;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFile, InputBuffer))
    return Error("unable to read input: '" + ec.message() + "'");

  // Instantiate a dynamic linker.
  RuntimeDyld Dyld(new TrivialMemoryManager);

  // Load the object file into it.
  if (Dyld.loadObject(InputBuffer.take())) {
    return Error(Dyld.getErrorString());
  }

  // Get the address of "_main".
  uint64_t MainAddress = Dyld.getSymbolAddress("_main");
  if (MainAddress == 0)
    return Error("no definition for '_main'");

  // Invalidate the instruction cache.
  sys::MemoryBlock Data = Dyld.getMemoryBlock();
  sys::Memory::InvalidateInstructionCache(Data.base(), Data.size());

  // Make sure the memory is executable.
  std::string ErrorStr;
  if (!sys::Memory::setExecutable(Data, &ErrorStr))
    return Error("unable to mark function executable: '" + ErrorStr + "'");

  // Dispatch to _main().
  errs() << "loaded '_main' at: " << (void*)MainAddress << "\n";

  int (*Main)(int, const char**) =
    (int(*)(int,const char**)) uintptr_t(MainAddress);
  const char **Argv = new const char*[2];
  Argv[0] = InputFile.c_str();
  Argv[1] = 0;
  return Main(1, Argv);
}

int main(int argc, char **argv) {
  ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm MC-JIT tool\n");

  switch (Action) {
  default:
  case AC_Execute:
    return executeInput();
  }

  return 0;
}

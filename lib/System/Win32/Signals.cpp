//===- Win32/Signals.cpp - Win32 Signals Implementation ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Signals class.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include <llvm/System/Signals.h>
#include <vector>

#include "dbghelp.h"
#include "psapi.h"

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "dbghelp.lib")

// Forward declare.
static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep);
static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType);

static std::vector<std::string> *FilesToRemove = NULL;
static std::vector<llvm::sys::Path> *DirectoriesToRemove = NULL;
static bool RegisteredUnhandledExceptionFilter = false;

// Windows creates a new thread to execute the console handler when an event
// (such as CTRL/C) occurs.  This causes concurrency issues with the above
// globals which this critical section addresses.
static CRITICAL_SECTION CriticalSection;

namespace llvm {

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//


static void RegisterHandler() { 
  if (RegisteredUnhandledExceptionFilter)
  {
    EnterCriticalSection(&CriticalSection);
    return;
  }

  // Now's the time to create the critical section.  This is the first time
  // through here, and there's only one thread.
  InitializeCriticalSection(&CriticalSection);

  // Enter it immediately.  Now if someone hits CTRL/C, the console handler
  // can't proceed until the globals are updated.
  EnterCriticalSection(&CriticalSection);

  RegisteredUnhandledExceptionFilter = true;
  SetUnhandledExceptionFilter(LLVMUnhandledExceptionFilter);
  SetConsoleCtrlHandler(LLVMConsoleCtrlHandler, TRUE);

  // IMPORTANT NOTE: Caller must call LeaveCriticalSection(&CriticalSection) or
  // else multi-threading problems will ensue.
}

// RemoveFileOnSignal - The public API
void sys::RemoveFileOnSignal(const std::string &Filename) {
  RegisterHandler();

  if (FilesToRemove == NULL)
    FilesToRemove = new std::vector<std::string>;

  FilesToRemove->push_back(Filename);

  LeaveCriticalSection(&CriticalSection);
}

// RemoveDirectoryOnSignal - The public API
void sys::RemoveDirectoryOnSignal(const sys::Path& path) {
  RegisterHandler();

  if (path.is_directory()) {
    if (DirectoriesToRemove == NULL)
      DirectoriesToRemove = new std::vector<sys::Path>;

    DirectoriesToRemove->push_back(path);
  }

  LeaveCriticalSection(&CriticalSection);
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void sys::PrintStackTraceOnErrorSignal() {
  RegisterHandler();
  LeaveCriticalSection(&CriticalSection);
}

}

static void Cleanup() {
  EnterCriticalSection(&CriticalSection);

  if (FilesToRemove != NULL)
    while (!FilesToRemove->empty()) {
      try {
        std::remove(FilesToRemove->back().c_str());
      } catch (...) {
      }
      FilesToRemove->pop_back();
    }

  if (DirectoriesToRemove != NULL)
    while (!DirectoriesToRemove->empty()) {
      try {
        DirectoriesToRemove->back().destroy_directory(true);
      } catch (...) {
      }
      DirectoriesToRemove->pop_back();
    }

  LeaveCriticalSection(&CriticalSection);
}

static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep) {
  try {
    Cleanup();

    // Initialize the STACKFRAME structure.
    STACKFRAME StackFrame;
    memset(&StackFrame, 0, sizeof(StackFrame));

    StackFrame.AddrPC.Offset = ep->ContextRecord->Eip;
    StackFrame.AddrPC.Mode = AddrModeFlat;
    StackFrame.AddrStack.Offset = ep->ContextRecord->Esp;
    StackFrame.AddrStack.Mode = AddrModeFlat;
    StackFrame.AddrFrame.Offset = ep->ContextRecord->Ebp;
    StackFrame.AddrFrame.Mode = AddrModeFlat;

    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();

    // Initialize the symbol handler.
    SymSetOptions(SYMOPT_DEFERRED_LOADS|SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), NULL, TRUE);

    while (true) {
      if (!StackWalk(IMAGE_FILE_MACHINE_I386, hProcess, hThread, &StackFrame,
                     ep->ContextRecord, NULL, SymFunctionTableAccess,
                     SymGetModuleBase, NULL)) {
        break;
      }

      if (StackFrame.AddrFrame.Offset == 0)
        break;

      // Print the PC in hexadecimal.
      DWORD PC = StackFrame.AddrPC.Offset;
      fprintf(stderr, "%04X:%08X", ep->ContextRecord->SegCs, PC);

      // Print the parameters.  Assume there are four.
      fprintf(stderr, " (0x%08X 0x%08X 0x%08X 0x%08X)", StackFrame.Params[0],
              StackFrame.Params[1], StackFrame.Params[2], StackFrame.Params[3]);

      // Verify the PC belongs to a module in this process.
      if (!SymGetModuleBase(hProcess, PC)) {
        fputc('\n', stderr);
        continue;
      }

      // Print the symbol name.
      char buffer[512];
      IMAGEHLP_SYMBOL *symbol = reinterpret_cast<IMAGEHLP_SYMBOL *>(buffer);
      memset(symbol, 0, sizeof(IMAGEHLP_SYMBOL));
      symbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
      symbol->MaxNameLength = 512 - sizeof(IMAGEHLP_SYMBOL);

      DWORD dwDisp;
      if (!SymGetSymFromAddr(hProcess, PC, &dwDisp, symbol)) {
        fputc('\n', stderr);
        continue;
      }

      buffer[511] = 0;
      if (dwDisp > 0)
        fprintf(stderr, ", %s()+%04d bytes(s)", symbol->Name, dwDisp);
      else
        fprintf(stderr, ", %s", symbol->Name);

      // Print the source file and line number information.
      IMAGEHLP_LINE line;
      memset(&line, 0, sizeof(line));
      line.SizeOfStruct = sizeof(line);
      if (SymGetLineFromAddr(hProcess, PC, &dwDisp, &line)) {
        fprintf(stderr, ", %s, line %d", line.FileName, line.LineNumber);
        if (dwDisp > 0)
          fprintf(stderr, "+%04d byte(s)", dwDisp);
      }

      fputc('\n', stderr);
    }
  }
  catch (...)
  {
      assert(!"Crashed in LLVMUnhandledExceptionFilter");
  }

  // Allow dialog box to pop up allowing choice to start debugger.
  return EXCEPTION_CONTINUE_SEARCH;
}

static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType) {
  // FIXME: This handler executes on a different thread.  The main thread
  // is still running, potentially creating new files to be cleaned up
  // in the tiny window between the call to Cleanup() and process termination.
  // Also, any files currently open cannot be deleted.
  Cleanup();

  // Allow normal processing to take place; i.e., the process dies.
  return FALSE;
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab

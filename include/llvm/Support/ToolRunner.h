//===-- llvm/Support/ToolRunner.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an abstraction around a platform C compiler, used to
// compile C and assembly code.  It also exposes an "AbstractIntepreter"
// interface, which is used to execute code using one of the LLVM execution
// engines.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TOOLRUNNER_H
#define LLVM_SUPPORT_TOOLRUNNER_H

#include "llvm/Support/SystemUtils.h"
#include <exception>
#include <vector>

namespace llvm {

class CBE;
class LLC;


/// ToolExecutionError - An instance of this class is thrown by the
/// AbstractInterpreter instances if there is an error running a tool (e.g., LLC
/// crashes) which prevents execution of the program.
///
class ToolExecutionError : std::exception {
  std::string Message;
public:
  explicit ToolExecutionError(const std::string &M) : Message(M) {}
  virtual ~ToolExecutionError() throw();
  virtual const char* what() const throw() { return Message.c_str(); }
};


//===---------------------------------------------------------------------===//
// GCC abstraction
//
class GCC {
  sys::Path GCCPath;          // The path to the gcc executable
  GCC(const sys::Path &gccPath) : GCCPath(gccPath) { }
public:
  enum FileType { AsmFile, CFile };

  static GCC* create(const std::string &ProgramPath, std::string &Message);

  /// ExecuteProgram - Execute the program specified by "ProgramFile" (which is
  /// either a .s file, or a .c file, specified by FileType), with the specified
  /// arguments.  Standard input is specified with InputFile, and standard
  /// Output is captured to the specified OutputFile location.  The SharedLibs
  /// option specifies optional native shared objects that can be loaded into
  /// the program for execution.
  ///
  int ExecuteProgram(const std::string &ProgramFile,
                     const std::vector<std::string> &Args,
                     FileType fileType,
                     const std::string &InputFile,
                     const std::string &OutputFile,
                     const std::vector<std::string> &GCCArgs =
                         std::vector<std::string>(), 
                     unsigned Timeout = 0);

  /// MakeSharedObject - This compiles the specified file (which is either a .c
  /// file or a .s file) into a shared object.
  ///
  int MakeSharedObject(const std::string &InputFile, FileType fileType,
                       std::string &OutputFile);
};


//===---------------------------------------------------------------------===//
/// AbstractInterpreter Class - Subclasses of this class are used to execute
/// LLVM bytecode in a variety of ways.  This abstract interface hides this
/// complexity behind a simple interface.
///
class AbstractInterpreter {
public:
  static CBE *createCBE(const std::string &ProgramPath, std::string &Message,
                        const std::vector<std::string> *Args = 0);
  static LLC *createLLC(const std::string &ProgramPath, std::string &Message,
                        const std::vector<std::string> *Args = 0);

  static AbstractInterpreter* createLLI(const std::string &ProgramPath,
                                        std::string &Message,
                                        const std::vector<std::string> *Args=0);

  static AbstractInterpreter* createJIT(const std::string &ProgramPath,
                                        std::string &Message,
                                        const std::vector<std::string> *Args=0);


  virtual ~AbstractInterpreter() {}

  /// compileProgram - Compile the specified program from bytecode to executable
  /// code.  This does not produce any output, it is only used when debugging
  /// the code generator.  If the code generator fails, an exception should be
  /// thrown, otherwise, this function will just return.
  virtual void compileProgram(const std::string &Bytecode) {}

  /// ExecuteProgram - Run the specified bytecode file, emitting output to the
  /// specified filename.  This returns the exit code of the program.
  ///
  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::vector<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::vector<std::string> &GCCArgs =
                               std::vector<std::string>(),
                             const std::vector<std::string> &SharedLibs =
                               std::vector<std::string>(),
                             unsigned Timeout = 0) = 0;
};

//===---------------------------------------------------------------------===//
// CBE Implementation of AbstractIntepreter interface
//
class CBE : public AbstractInterpreter {
  sys::Path LLCPath;          // The path to the `llc' executable
  std::vector<std::string> ToolArgs; // Extra args to pass to LLC
  GCC *gcc;
public:
  CBE(const sys::Path &llcPath, GCC *Gcc,
      const std::vector<std::string> *Args) : LLCPath(llcPath), gcc(Gcc) {
    ToolArgs.clear ();
    if (Args) { ToolArgs = *Args; }
  }
  ~CBE() { delete gcc; }

  /// compileProgram - Compile the specified program from bytecode to executable
  /// code.  This does not produce any output, it is only used when debugging
  /// the code generator.  If the code generator fails, an exception should be
  /// thrown, otherwise, this function will just return.
  virtual void compileProgram(const std::string &Bytecode);

  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::vector<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::vector<std::string> &GCCArgs =
                               std::vector<std::string>(),
                             const std::vector<std::string> &SharedLibs =
                               std::vector<std::string>(),
                             unsigned Timeout = 0);

  // Sometimes we just want to go half-way and only generate the .c file, not
  // necessarily compile it with GCC and run the program.  This throws an
  // exception if LLC crashes.
  //
  virtual void OutputC(const std::string &Bytecode, sys::Path& OutputCFile);
};


//===---------------------------------------------------------------------===//
// LLC Implementation of AbstractIntepreter interface
//
class LLC : public AbstractInterpreter {
  std::string LLCPath;          // The path to the LLC executable
  std::vector<std::string> ToolArgs; // Extra args to pass to LLC
  GCC *gcc;
public:
  LLC(const std::string &llcPath, GCC *Gcc,
    const std::vector<std::string> *Args) : LLCPath(llcPath), gcc(Gcc) {
    ToolArgs.clear ();
    if (Args) { ToolArgs = *Args; }
  }
  ~LLC() { delete gcc; }

  /// compileProgram - Compile the specified program from bytecode to executable
  /// code.  This does not produce any output, it is only used when debugging
  /// the code generator.  If the code generator fails, an exception should be
  /// thrown, otherwise, this function will just return.
  virtual void compileProgram(const std::string &Bytecode);

  virtual int ExecuteProgram(const std::string &Bytecode,
                             const std::vector<std::string> &Args,
                             const std::string &InputFile,
                             const std::string &OutputFile,
                             const std::vector<std::string> &GCCArgs =
                               std::vector<std::string>(),
                             const std::vector<std::string> &SharedLibs =
                                std::vector<std::string>(),
                             unsigned Timeout = 0);

  // Sometimes we just want to go half-way and only generate the .s file,
  // not necessarily compile it all the way and run the program.  This throws
  // an exception if execution of LLC fails.
  //
  void OutputAsm(const std::string &Bytecode, sys::Path &OutputAsmFile);
};

} // End llvm namespace

#endif

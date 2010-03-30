//===- llvm-ld.cpp - LLVM 'ld' compatible linker --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility is intended to be compatible with GCC, and follows standard
// system 'ld' conventions.  As such, the default output file is ./a.out.
// Additionally, this program outputs a shell script that is used to invoke LLI
// to execute the program.  In this manner, the generated executable (a.out for
// example), is directly executable, whereas the bitcode file actually lives in
// the a.out.bc file generated by this program.
//
// Note that if someone (or a script) deletes the executable program generated,
// the .bc file will be left around.  Considering that this is a temporary hack,
// I'm not too worried about this.
//
//===----------------------------------------------------------------------===//

#include "llvm/LinkAllVMCore.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/System/Program.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include "llvm/Config/config.h"
#include <memory>
#include <cstring>
using namespace llvm;

// Rightly this should go in a header file but it just seems such a waste.
namespace llvm {
extern void Optimize(Module*);
}

// Input/Output Options
static cl::list<std::string> InputFilenames(cl::Positional, cl::OneOrMore,
  cl::desc("<input bitcode files>"));

static cl::opt<std::string> OutputFilename("o", cl::init("a.out"),
  cl::desc("Override output filename"),
  cl::value_desc("filename"));

static cl::opt<std::string> BitcodeOutputFilename("b", cl::init(""),
  cl::desc("Override bitcode output filename"),
  cl::value_desc("filename"));

static cl::opt<bool> Verbose("v",
  cl::desc("Print information about actions taken"));

static cl::list<std::string> LibPaths("L", cl::Prefix,
  cl::desc("Specify a library search path"),
  cl::value_desc("directory"));

static cl::list<std::string> FrameworkPaths("F", cl::Prefix,
  cl::desc("Specify a framework search path"),
  cl::value_desc("directory"));

static cl::list<std::string> Libraries("l", cl::Prefix,
  cl::desc("Specify libraries to link to"),
  cl::value_desc("library prefix"));

static cl::list<std::string> Frameworks("framework",
  cl::desc("Specify frameworks to link to"),
  cl::value_desc("framework"));

// Options to control the linking, optimization, and code gen processes
static cl::opt<bool> LinkAsLibrary("link-as-library",
  cl::desc("Link the .bc files together as a library, not an executable"));

static cl::alias Relink("r", cl::aliasopt(LinkAsLibrary),
  cl::desc("Alias for -link-as-library"));

static cl::opt<bool> Native("native",
  cl::desc("Generate a native binary instead of a shell script"));

static cl::opt<bool>NativeCBE("native-cbe",
  cl::desc("Generate a native binary with the C backend and GCC"));

static cl::list<std::string> PostLinkOpts("post-link-opts",
  cl::value_desc("path"),
  cl::desc("Run one or more optimization programs after linking"));

static cl::list<std::string> XLinker("Xlinker", cl::value_desc("option"),
  cl::desc("Pass options to the system linker"));

// Compatibility options that llvm-ld ignores but are supported for 
// compatibility with LD
static cl::opt<std::string> CO3("soname", cl::Hidden,
  cl::desc("Compatibility option: ignored"));

static cl::opt<std::string> CO4("version-script", cl::Hidden,
  cl::desc("Compatibility option: ignored"));

static cl::opt<bool> CO5("eh-frame-hdr", cl::Hidden,
  cl::desc("Compatibility option: ignored"));

static  cl::opt<std::string> CO6("h", cl::Hidden,
  cl::desc("Compatibility option: ignored"));

static cl::opt<bool> CO7("start-group", cl::Hidden, 
  cl::desc("Compatibility option: ignored"));

static cl::opt<bool> CO8("end-group", cl::Hidden, 
  cl::desc("Compatibility option: ignored"));

static cl::opt<std::string> CO9("m", cl::Hidden, 
  cl::desc("Compatibility option: ignored"));

/// This is just for convenience so it doesn't have to be passed around
/// everywhere.
static std::string progname;

/// FileRemover objects to clean up output files in the event of an error.
static FileRemover OutputRemover;
static FileRemover BitcodeOutputRemover;

/// PrintAndExit - Prints a message to standard error and exits with error code
///
/// Inputs:
///  Message  - The message to print to standard error.
///
static void PrintAndExit(const std::string &Message, Module *M, int errcode = 1) {
  errs() << progname << ": " << Message << "\n";
  delete M;
  llvm_shutdown();
  exit(errcode);
}

static void PrintCommand(const std::vector<const char*> &args) {
  std::vector<const char*>::const_iterator I = args.begin(), E = args.end(); 
  for (; I != E; ++I)
    if (*I)
      outs() << "'" << *I << "'" << " ";
  outs() << "\n"; outs().flush();
}

/// CopyEnv - This function takes an array of environment variables and makes a
/// copy of it.  This copy can then be manipulated any way the caller likes
/// without affecting the process's real environment.
///
/// Inputs:
///  envp - An array of C strings containing an environment.
///
/// Return value:
///  NULL - An error occurred.
///
///  Otherwise, a pointer to a new array of C strings is returned.  Every string
///  in the array is a duplicate of the one in the original array (i.e. we do
///  not copy the char *'s from one array to another).
///
static char ** CopyEnv(char ** const envp) {
  // Count the number of entries in the old list;
  unsigned entries;   // The number of entries in the old environment list
  for (entries = 0; envp[entries] != NULL; entries++)
    /*empty*/;

  // Add one more entry for the NULL pointer that ends the list.
  ++entries;

  // If there are no entries at all, just return NULL.
  if (entries == 0)
    return NULL;

  // Allocate a new environment list.
  char **newenv = new char* [entries];
  if ((newenv = new char* [entries]) == NULL)
    return NULL;

  // Make a copy of the list.  Don't forget the NULL that ends the list.
  entries = 0;
  while (envp[entries] != NULL) {
    size_t len = strlen(envp[entries]) + 1;
    newenv[entries] = new char[len];
    memcpy(newenv[entries], envp[entries], len);
    ++entries;
  }
  newenv[entries] = NULL;

  return newenv;
}


/// RemoveEnv - Remove the specified environment variable from the environment
/// array.
///
/// Inputs:
///  name - The name of the variable to remove.  It cannot be NULL.
///  envp - The array of environment variables.  It cannot be NULL.
///
/// Notes:
///  This is mainly done because functions to remove items from the environment
///  are not available across all platforms.  In particular, Solaris does not
///  seem to have an unsetenv() function or a setenv() function (or they are
///  undocumented if they do exist).
///
static void RemoveEnv(const char * name, char ** const envp) {
  for (unsigned index=0; envp[index] != NULL; index++) {
    // Find the first equals sign in the array and make it an EOS character.
    char *p = strchr (envp[index], '=');
    if (p == NULL)
      continue;
    else
      *p = '\0';

    // Compare the two strings.  If they are equal, zap this string.
    // Otherwise, restore it.
    if (!strcmp(name, envp[index]))
      *envp[index] = '\0';
    else
      *p = '=';
  }

  return;
}

/// GenerateBitcode - generates a bitcode file from the module provided
void GenerateBitcode(Module* M, const std::string& FileName) {

  if (Verbose)
    outs() << "Generating Bitcode To " << FileName << '\n';

  // Create the output file.
  std::string ErrorInfo;
  raw_fd_ostream Out(FileName.c_str(), ErrorInfo,
                     raw_fd_ostream::F_Binary);
  if (!ErrorInfo.empty())
    PrintAndExit(ErrorInfo, M);

  // Write it out
  WriteBitcodeToFile(M, Out);
}

/// GenerateAssembly - generates a native assembly language source file from the
/// specified bitcode file.
///
/// Inputs:
///  InputFilename  - The name of the input bitcode file.
///  OutputFilename - The name of the file to generate.
///  llc            - The pathname to use for LLC.
///  envp           - The environment to use when running LLC.
///
/// Return non-zero value on error.
///
static int GenerateAssembly(const std::string &OutputFilename,
                            const std::string &InputFilename,
                            const sys::Path &llc,
                            std::string &ErrMsg ) {
  // Run LLC to convert the bitcode file into assembly code.
  std::vector<const char*> args;
  args.push_back(llc.c_str());
  // We will use GCC to assemble the program so set the assembly syntax to AT&T,
  // regardless of what the target in the bitcode file is.
  args.push_back("-x86-asm-syntax=att");
  args.push_back("-f");
  args.push_back("-o");
  args.push_back(OutputFilename.c_str());
  args.push_back(InputFilename.c_str());
  args.push_back(0);

  if (Verbose) {
    outs() << "Generating Assembly With: \n";
    PrintCommand(args);
  }

  return sys::Program::ExecuteAndWait(llc, &args[0], 0, 0, 0, 0, &ErrMsg);
}

/// GenerateCFile - generates a C source file from the specified bitcode file.
static int GenerateCFile(const std::string &OutputFile,
                         const std::string &InputFile,
                         const sys::Path &llc,
                         std::string& ErrMsg) {
  // Run LLC to convert the bitcode file into C.
  std::vector<const char*> args;
  args.push_back(llc.c_str());
  args.push_back("-march=c");
  args.push_back("-f");
  args.push_back("-o");
  args.push_back(OutputFile.c_str());
  args.push_back(InputFile.c_str());
  args.push_back(0);

  if (Verbose) {
    outs() << "Generating C Source With: \n";
    PrintCommand(args);
  }

  return sys::Program::ExecuteAndWait(llc, &args[0], 0, 0, 0, 0, &ErrMsg);
}

/// GenerateNative - generates a native object file from the
/// specified bitcode file.
///
/// Inputs:
///  InputFilename   - The name of the input bitcode file.
///  OutputFilename  - The name of the file to generate.
///  NativeLinkItems - The native libraries, files, code with which to link
///  LibPaths        - The list of directories in which to find libraries.
///  FrameworksPaths - The list of directories in which to find frameworks.
///  Frameworks      - The list of frameworks (dynamic libraries)
///  gcc             - The pathname to use for GGC.
///  envp            - A copy of the process's current environment.
///
/// Outputs:
///  None.
///
/// Returns non-zero value on error.
///
static int GenerateNative(const std::string &OutputFilename,
                          const std::string &InputFilename,
                          const Linker::ItemList &LinkItems,
                          const sys::Path &gcc, char ** const envp,
                          std::string& ErrMsg) {
  // Remove these environment variables from the environment of the
  // programs that we will execute.  It appears that GCC sets these
  // environment variables so that the programs it uses can configure
  // themselves identically.
  //
  // However, when we invoke GCC below, we want it to use its normal
  // configuration.  Hence, we must sanitize its environment.
  char ** clean_env = CopyEnv(envp);
  if (clean_env == NULL)
    return 1;
  RemoveEnv("LIBRARY_PATH", clean_env);
  RemoveEnv("COLLECT_GCC_OPTIONS", clean_env);
  RemoveEnv("GCC_EXEC_PREFIX", clean_env);
  RemoveEnv("COMPILER_PATH", clean_env);
  RemoveEnv("COLLECT_GCC", clean_env);


  // Run GCC to assemble and link the program into native code.
  //
  // Note:
  //  We can't just assemble and link the file with the system assembler
  //  and linker because we don't know where to put the _start symbol.
  //  GCC mysteriously knows how to do it.
  std::vector<std::string> args;
  args.push_back(gcc.c_str());
  args.push_back("-fno-strict-aliasing");
  args.push_back("-O3");
  args.push_back("-o");
  args.push_back(OutputFilename);
  args.push_back(InputFilename);

  // Add in the library and framework paths
  for (unsigned index = 0; index < LibPaths.size(); index++) {
    args.push_back("-L" + LibPaths[index]);
  }
  for (unsigned index = 0; index < FrameworkPaths.size(); index++) {
    args.push_back("-F" + FrameworkPaths[index]);
  }

  // Add the requested options
  for (unsigned index = 0; index < XLinker.size(); index++)
    args.push_back(XLinker[index]);

  // Add in the libraries to link.
  for (unsigned index = 0; index < LinkItems.size(); index++)
    if (LinkItems[index].first != "crtend") {
      if (LinkItems[index].second)
        args.push_back("-l" + LinkItems[index].first);
      else
        args.push_back(LinkItems[index].first);
    }

  // Add in frameworks to link.
  for (unsigned index = 0; index < Frameworks.size(); index++) {
    args.push_back("-framework");
    args.push_back(Frameworks[index]);
  }
      
  // Now that "args" owns all the std::strings for the arguments, call the c_str
  // method to get the underlying string array.  We do this game so that the
  // std::string array is guaranteed to outlive the const char* array.
  std::vector<const char *> Args;
  for (unsigned i = 0, e = args.size(); i != e; ++i)
    Args.push_back(args[i].c_str());
  Args.push_back(0);

  if (Verbose) {
    outs() << "Generating Native Executable With:\n";
    PrintCommand(Args);
  }

  // Run the compiler to assembly and link together the program.
  int R = sys::Program::ExecuteAndWait(
    gcc, &Args[0], (const char**)clean_env, 0, 0, 0, &ErrMsg);
  delete [] clean_env;
  return R;
}

/// EmitShellScript - Output the wrapper file that invokes the JIT on the LLVM
/// bitcode file for the program.
static void EmitShellScript(char **argv, Module *M) {
  if (Verbose)
    outs() << "Emitting Shell Script\n";
#if defined(_WIN32) || defined(__CYGWIN__)
  // Windows doesn't support #!/bin/sh style shell scripts in .exe files.  To
  // support windows systems, we copy the llvm-stub.exe executable from the
  // build tree to the destination file.
  std::string ErrMsg;  
  sys::Path llvmstub = FindExecutable("llvm-stub.exe", argv[0],
                                      (void *)(intptr_t)&Optimize);
  if (llvmstub.isEmpty())
    PrintAndExit("Could not find llvm-stub.exe executable!", M);

  if (0 != sys::CopyFile(sys::Path(OutputFilename), llvmstub, &ErrMsg))
    PrintAndExit(ErrMsg, M);

  return;
#endif

  // Output the script to start the program...
  std::string ErrorInfo;
  raw_fd_ostream Out2(OutputFilename.c_str(), ErrorInfo);
  if (!ErrorInfo.empty())
    PrintAndExit(ErrorInfo, M);

  Out2 << "#!/bin/sh\n";
  // Allow user to setenv LLVMINTERP if lli is not in their PATH.
  Out2 << "lli=${LLVMINTERP-lli}\n";
  Out2 << "exec $lli \\\n";
  // gcc accepts -l<lib> and implicitly searches /lib and /usr/lib.
  LibPaths.push_back("/lib");
  LibPaths.push_back("/usr/lib");
  LibPaths.push_back("/usr/X11R6/lib");
  // We don't need to link in libc! In fact, /usr/lib/libc.so may not be a
  // shared object at all! See RH 8: plain text.
  std::vector<std::string>::iterator libc =
    std::find(Libraries.begin(), Libraries.end(), "c");
  if (libc != Libraries.end()) Libraries.erase(libc);
  // List all the shared object (native) libraries this executable will need
  // on the command line, so that we don't have to do this manually!
  for (std::vector<std::string>::iterator i = Libraries.begin(),
         e = Libraries.end(); i != e; ++i) {
    // try explicit -L arguments first:
    sys::Path FullLibraryPath;
    for (cl::list<std::string>::const_iterator P = LibPaths.begin(),
           E = LibPaths.end(); P != E; ++P) {
      FullLibraryPath = *P;
      FullLibraryPath.appendComponent("lib" + *i);
      FullLibraryPath.appendSuffix(&(LTDL_SHLIB_EXT[1]));
      if (!FullLibraryPath.isEmpty()) {
        if (!FullLibraryPath.isDynamicLibrary()) {
          // Not a native shared library; mark as invalid
          FullLibraryPath = sys::Path();
        } else break;
      }
    }
    if (FullLibraryPath.isEmpty())
      FullLibraryPath = sys::Path::FindLibrary(*i);
    if (!FullLibraryPath.isEmpty())
      Out2 << "    -load=" << FullLibraryPath.str() << " \\\n";
  }
  Out2 << "    "  << BitcodeOutputFilename << " ${1+\"$@\"}\n";
}

// BuildLinkItems -- This function generates a LinkItemList for the LinkItems
// linker function by combining the Files and Libraries in the order they were
// declared on the command line.
static void BuildLinkItems(
  Linker::ItemList& Items,
  const cl::list<std::string>& Files,
  const cl::list<std::string>& Libraries) {

  // Build the list of linkage items for LinkItems.

  cl::list<std::string>::const_iterator fileIt = Files.begin();
  cl::list<std::string>::const_iterator libIt  = Libraries.begin();

  int libPos = -1, filePos = -1;
  while ( libIt != Libraries.end() || fileIt != Files.end() ) {
    if (libIt != Libraries.end())
      libPos = Libraries.getPosition(libIt - Libraries.begin());
    else
      libPos = -1;
    if (fileIt != Files.end())
      filePos = Files.getPosition(fileIt - Files.begin());
    else
      filePos = -1;

    if (filePos != -1 && (libPos == -1 || filePos < libPos)) {
      // Add a source file
      Items.push_back(std::make_pair(*fileIt++, false));
    } else if (libPos != -1 && (filePos == -1 || libPos < filePos)) {
      // Add a library
      Items.push_back(std::make_pair(*libIt++, true));
    }
  }
}

int main(int argc, char **argv, char **envp) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  
  // Initial global variable above for convenience printing of program name.
  progname = sys::Path(argv[0]).getBasename();

  // Parse the command line options
  cl::ParseCommandLineOptions(argc, argv, "llvm linker\n");

#if defined(_WIN32) || defined(__CYGWIN__)
  if (!LinkAsLibrary) {
    // Default to "a.exe" instead of "a.out".
    if (OutputFilename.getNumOccurrences() == 0)
      OutputFilename = "a.exe";

    // If there is no suffix add an "exe" one.
    sys::Path ExeFile( OutputFilename );
    if (ExeFile.getSuffix() == "") {
      ExeFile.appendSuffix("exe");
      OutputFilename = ExeFile.str();
    }
  }
#endif

  // Generate the bitcode for the optimized module.
  // If -b wasn't specified, use the name specified
  // with -o to construct BitcodeOutputFilename.
  if (BitcodeOutputFilename.empty()) {
    BitcodeOutputFilename = OutputFilename;
    if (!LinkAsLibrary) BitcodeOutputFilename += ".bc";
  }

  // Arrange for the bitcode output file to be deleted on any errors.
  BitcodeOutputRemover.setFile(sys::Path(BitcodeOutputFilename));
  sys::RemoveFileOnSignal(sys::Path(BitcodeOutputFilename));

  // Arrange for the output file to be deleted on any errors.
  if (!LinkAsLibrary) {
    OutputRemover.setFile(sys::Path(OutputFilename));
    sys::RemoveFileOnSignal(sys::Path(OutputFilename));
  }

  // Construct a Linker (now that Verbose is set)
  Linker TheLinker(progname, OutputFilename, Context, Verbose);

  // Keep track of the native link items (versus the bitcode items)
  Linker::ItemList NativeLinkItems;

  // Add library paths to the linker
  TheLinker.addPaths(LibPaths);
  TheLinker.addSystemPaths();

  // Remove any consecutive duplicates of the same library...
  Libraries.erase(std::unique(Libraries.begin(), Libraries.end()),
                  Libraries.end());

  if (LinkAsLibrary) {
    std::vector<sys::Path> Files;
    for (unsigned i = 0; i < InputFilenames.size(); ++i )
      Files.push_back(sys::Path(InputFilenames[i]));
    if (TheLinker.LinkInFiles(Files))
      return 1; // Error already printed

    // The libraries aren't linked in but are noted as "dependent" in the
    // module.
    for (cl::list<std::string>::const_iterator I = Libraries.begin(),
         E = Libraries.end(); I != E ; ++I) {
      TheLinker.getModule()->addLibrary(*I);
    }
  } else {
    // Build a list of the items from our command line
    Linker::ItemList Items;
    BuildLinkItems(Items, InputFilenames, Libraries);

    // Link all the items together
    if (TheLinker.LinkInItems(Items, NativeLinkItems) )
      return 1; // Error already printed
  }

  std::auto_ptr<Module> Composite(TheLinker.releaseModule());

  // Optimize the module
  Optimize(Composite.get());

  // Generate the bitcode output.
  GenerateBitcode(Composite.get(), BitcodeOutputFilename);

  // If we are not linking a library, generate either a native executable
  // or a JIT shell script, depending upon what the user wants.
  if (!LinkAsLibrary) {
    // If the user wants to run a post-link optimization, run it now.
    if (!PostLinkOpts.empty()) {
      std::vector<std::string> opts = PostLinkOpts;
      for (std::vector<std::string>::iterator I = opts.begin(),
           E = opts.end(); I != E; ++I) {
        sys::Path prog(*I);
        if (!prog.canExecute()) {
          prog = sys::Program::FindProgramByName(*I);
          if (prog.isEmpty())
            PrintAndExit(std::string("Optimization program '") + *I +
                         "' is not found or not executable.", Composite.get());
        }
        // Get the program arguments
        sys::Path tmp_output("opt_result");
        std::string ErrMsg;
        if (tmp_output.createTemporaryFileOnDisk(true, &ErrMsg))
          PrintAndExit(ErrMsg, Composite.get());

        const char* args[4];
        args[0] = I->c_str();
        args[1] = BitcodeOutputFilename.c_str();
        args[2] = tmp_output.c_str();
        args[3] = 0;
        if (0 == sys::Program::ExecuteAndWait(prog, args, 0,0,0,0, &ErrMsg)) {
          if (tmp_output.isBitcodeFile()) {
            sys::Path target(BitcodeOutputFilename);
            target.eraseFromDisk();
            if (tmp_output.renamePathOnDisk(target, &ErrMsg))
              PrintAndExit(ErrMsg, Composite.get(), 2);
          } else
            PrintAndExit("Post-link optimization output is not bitcode",
                         Composite.get());
        } else {
          PrintAndExit(ErrMsg, Composite.get());
        }
      }
    }

    // If the user wants to generate a native executable, compile it from the
    // bitcode file.
    //
    // Otherwise, create a script that will run the bitcode through the JIT.
    if (Native) {
      // Name of the Assembly Language output file
      sys::Path AssemblyFile ( OutputFilename);
      AssemblyFile.appendSuffix("s");

      // Mark the output files for removal.
      FileRemover AssemblyFileRemover(AssemblyFile);
      sys::RemoveFileOnSignal(AssemblyFile);

      // Determine the locations of the llc and gcc programs.
      sys::Path llc = FindExecutable("llc", argv[0],
                                     (void *)(intptr_t)&Optimize);
      if (llc.isEmpty())
        PrintAndExit("Failed to find llc", Composite.get());

      sys::Path gcc = sys::Program::FindProgramByName("gcc");
      if (gcc.isEmpty())
        PrintAndExit("Failed to find gcc", Composite.get());

      // Generate an assembly language file for the bitcode.
      std::string ErrMsg;
      if (0 != GenerateAssembly(AssemblyFile.str(), BitcodeOutputFilename,
          llc, ErrMsg))
        PrintAndExit(ErrMsg, Composite.get());

      if (0 != GenerateNative(OutputFilename, AssemblyFile.str(),
                              NativeLinkItems, gcc, envp, ErrMsg))
        PrintAndExit(ErrMsg, Composite.get());
    } else if (NativeCBE) {
      sys::Path CFile (OutputFilename);
      CFile.appendSuffix("cbe.c");

      // Mark the output files for removal.
      FileRemover CFileRemover(CFile);
      sys::RemoveFileOnSignal(CFile);

      // Determine the locations of the llc and gcc programs.
      sys::Path llc = FindExecutable("llc", argv[0],
                                     (void *)(intptr_t)&Optimize);
      if (llc.isEmpty())
        PrintAndExit("Failed to find llc", Composite.get());

      sys::Path gcc = sys::Program::FindProgramByName("gcc");
      if (gcc.isEmpty())
        PrintAndExit("Failed to find gcc", Composite.get());

      // Generate an assembly language file for the bitcode.
      std::string ErrMsg;
      if (GenerateCFile(CFile.str(), BitcodeOutputFilename, llc, ErrMsg))
        PrintAndExit(ErrMsg, Composite.get());

      if (GenerateNative(OutputFilename, CFile.str(), 
                         NativeLinkItems, gcc, envp, ErrMsg))
        PrintAndExit(ErrMsg, Composite.get());
    } else {
      EmitShellScript(argv, Composite.get());
    }

    // Make the script executable...
    std::string ErrMsg;
    if (sys::Path(OutputFilename).makeExecutableOnDisk(&ErrMsg))
      PrintAndExit(ErrMsg, Composite.get());

    // Make the bitcode file readable and directly executable in LLEE as well
    if (sys::Path(BitcodeOutputFilename).makeExecutableOnDisk(&ErrMsg))
      PrintAndExit(ErrMsg, Composite.get());

    if (sys::Path(BitcodeOutputFilename).makeReadableOnDisk(&ErrMsg))
      PrintAndExit(ErrMsg, Composite.get());
  }

  // Operations which may fail are now complete.
  BitcodeOutputRemover.releaseFile();
  if (!LinkAsLibrary)
    OutputRemover.releaseFile();

  // Graceful exit
  return 0;
}

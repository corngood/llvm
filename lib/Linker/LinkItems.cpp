//===- lib/Linker/LinkItems.cpp - Link LLVM objects and libraries ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines to handle linking together LLVM bytecode files,
// and to handle annoying things like static libraries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Module.h"

using namespace llvm;

// LinkItems - preserve link order for an arbitrary set of linkage items.
bool
Linker::LinkInItems(const ItemList& Items) {
  // For each linkage item ...
  for (ItemList::const_iterator I = Items.begin(), E = Items.end();
       I != E; ++I) {
    if (I->second) {
      // Link in the library suggested.
      if (LinkInLibrary(I->first))
        return true;
    } else {
      if (LinkInFile(sys::Path(I->first)))
        return true;
    }
  }

  // At this point we have processed all the link items provided to us. Since
  // we have an aggregated module at this point, the dependent libraries in
  // that module should also be aggregated with duplicates eliminated. This is
  // now the time to process the dependent libraries to resolve any remaining
  // symbols.
  for (Module::lib_iterator I = Composite->lib_begin(),
         E = Composite->lib_end(); I != E; ++I)
    if(LinkInLibrary(*I))
      return true;

  return false;
}


/// LinkInLibrary - links one library into the HeadModule.
///
bool Linker::LinkInLibrary(const std::string& Lib) {
  // Determine where this library lives.
  sys::Path Pathname = FindLib(Lib);
  if (Pathname.isEmpty())
    return warning("Cannot find library '" + Lib + "'");

  // If its an archive, try to link it in
  if (Pathname.isArchive()) {
    if (LinkInArchive(Pathname))
      return error("Cannot link archive '" + Pathname.toString() + "'");
  } else if (Pathname.isBytecodeFile()) {
    // LLVM ".so" file.
    if (LinkInFile(Pathname))
      return error("Cannot link file '" + Pathname.toString() + "'");

  } else if (Pathname.isDynamicLibrary()) {
    return warning("Library '" + Lib + "' is a native dynamic library.");
  } else {
    return warning("Supposed library '" + Lib + "' isn't a library.");
  }
  return false;
}

/// LinkLibraries - takes the specified library files and links them into the
/// main bytecode object file.
///
/// Inputs:
///  Libraries  - The list of libraries to link into the module.
///
/// Return value:
///  FALSE - No error.
///  TRUE  - Error.
///
bool Linker::LinkInLibraries(const std::vector<std::string> &Libraries) {

  // Process the set of libraries we've been provided.
  for (unsigned i = 0; i < Libraries.size(); ++i)
    if (LinkInLibrary(Libraries[i]))
      return true;

  // At this point we have processed all the libraries provided to us. Since
  // we have an aggregated module at this point, the dependent libraries in
  // that module should also be aggregated with duplicates eliminated. This is
  // now the time to process the dependent libraries to resolve any remaining
  // symbols.
  const Module::LibraryListType& DepLibs = Composite->getLibraries();
  for (Module::LibraryListType::const_iterator I = DepLibs.begin(),
         E = DepLibs.end(); I != E; ++I)
    if (LinkInLibrary(*I))
      return true;

  return false;
}

/// LinkInFile - opens a bytecode file and links in all objects which
/// provide symbols that are currently undefined.
///
/// Inputs:
///  File - The pathname of the bytecode file.
///
/// Outputs:
///  ErrorMessage - A C++ string detailing what error occurred, if any.
///
/// Return Value:
///  TRUE  - An error occurred.
///  FALSE - No errors.
///
bool Linker::LinkInFile(const sys::Path &File) {
  // Make sure we can at least read the file
  if (!File.canRead())
    return error("Cannot find linker input '" + File.toString() + "'");

  // A user may specify an ar archive without -l, perhaps because it
  // is not installed as a library. Detect that and link the library.
  if (File.isArchive()) {
    if (LinkInArchive(File))
      return error("Cannot link archive '" + File.toString() + "'");
  } else if (File.isBytecodeFile()) {
    verbose("Linking bytecode file '" + File.toString() + "'");

    std::auto_ptr<Module> M(LoadObject(File));
    if (M.get() == 0)
      return error("Cannot load file '" + File.toString() + "'" + Error);
    if (LinkInModule(M.get()))
      return error("Cannot link file '" + File.toString() + "'" + Error);

    verbose("Linked in file '" + File.toString() + "'");
  } else {
    return warning("File of unknown type '" + File.toString() + "' ignored.");
  }
  return false;
}

/// LinkFiles - takes a module and a list of files and links them all together.
/// It locates the file either in the current directory, as its absolute
/// or relative pathname, or as a file somewhere in LLVM_LIB_SEARCH_PATH.
///
/// Inputs:
///  Files      - A vector of sys::Path indicating the LLVM bytecode filenames
///               to be linked.  The names can refer to a mixture of pure LLVM
///               bytecode files and archive (ar) formatted files.
///
/// Return value:
///  FALSE - No errors.
///  TRUE  - Some error occurred.
///
bool Linker::LinkInFiles(const std::vector<sys::Path> &Files) {
  for (unsigned i = 0; i < Files.size(); ++i)
    if (LinkInFile(Files[i]))
      return true;
  return false;
}

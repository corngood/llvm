//===-- llvm/CodeGen/MachineDebugInfo.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect debug information for a module.  This information should be in a
// neutral form that can be used by different debugging schemes.
//
// The organization of information is primarily clustered around the source
// compile units.  The main exception is source line coorespondence where
// inlining may interleave code from various compile units.
//
// The following information can be retrieved from the MachineDebugInfo.
//
//  -- Source directories - Directories are uniqued based on their canonical
//     string and assigned a sequential numeric ID (base 1.)  A directory ID - 1
//     provides the index of directory information in a queried directory list.
//  -- Source files - Files are also uniqued based on their name and directory
//     ID.  A file ID is sequential number (base 1.)  A file ID - 1 provides the
//     index of source information in a queried file list.
//  -- Source line coorespondence - A vector of file ID, line#, column# triples.
//     A DEBUG_LOCATION instruction is generated  by the DAG Legalizer
//     corresponding to each entry in the source line list.  This allows a debug
//     information emitter to generate labels to map code addressed to debug
//     tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDEBUGINFO_H
#define LLVM_CODEGEN_MACHINEDEBUGINFO_H

#include "llvm/Pass.h"
#include "llvm/ADT/UniqueVector.h"
#include <string>

namespace llvm {

//===----------------------------------------------------------------------===//
/// SourceLineInfo - This class is used to record source line correspondence.
///
class SourceLineInfo {
private:
  unsigned Line;                        // Source line number.
  unsigned Column;                      // Source column.
  unsigned SourceID;                    // Source ID number.

public:
  SourceLineInfo(unsigned L, unsigned C, unsigned S)
  : Line(L), Column(C), SourceID(S) {}
  
  // Accessors
  unsigned getLine()     const { return Line; }
  unsigned getColumn()   const { return Column; }
  unsigned getSourceID() const { return SourceID; }
};

//===----------------------------------------------------------------------===//
/// SourceFileInfo - This class is used to track source information.
///
class SourceFileInfo {
private:
  unsigned DirectoryID;                 // Directory ID number.
  std::string Name;                     // File name (not including directory.)
  
public:
  SourceFileInfo(unsigned D, const std::string &N) : DirectoryID(D), Name(N) {}
            
  // Accessors
  unsigned getDirectoryID()    const { return DirectoryID; }
  const std::string &getName() const { return Name; }

  /// operator== - Used by UniqueVector to locate entry.
  ///
  bool operator==(const SourceFileInfo &SI) const {
    return getDirectoryID() == SI.getDirectoryID() && getName() == SI.getName();
  }

  /// operator< - Used by UniqueVector to locate entry.
  ///
  bool operator<(const SourceFileInfo &SI) const {
    return getDirectoryID() < SI.getDirectoryID() ||
          (getDirectoryID() == SI.getDirectoryID() && getName() < SI.getName());
  }
};

//===----------------------------------------------------------------------===//
/// MachineDebugInfo - This class contains debug information specific to a
/// module.  Queries can be made by different debugging schemes and reformated
/// for specific use.
///
class MachineDebugInfo : public ImmutablePass {
private:
  // DirectoryMap - UniqueVector for directories.
  UniqueVector<std::string> Directories;
                                         
  // SourceMap - UniqueVector for source files.
  UniqueVector<SourceFileInfo> SourceFiles;

  // Lines - List of of source line correspondence.
  std::vector<SourceLineInfo *> Lines;

public:
  MachineDebugInfo()
  : Directories()
  , SourceFiles()
  , Lines()
  {}
  ~MachineDebugInfo() { }
  
  /// doInitialization - Initialize the debug state for a new module.
  ///
  bool doInitialization();
  
  
  /// doFinalization - Tear down the debug state after completion of a module.
  ///
  bool doFinalization();
  
  /// hasInfo - Returns true if debug info is present.
  ///
  // FIXME - need proper scheme to suppress debug output.
  bool hasInfo() const { return !SourceFiles.empty(); }
  
  /// RecordLabel - Records location information and associates it with a
  /// debug label.  Returns a unique label ID used to generate a label and 
  /// provide correspondence to the source line list.
  unsigned RecordLabel(unsigned Line, unsigned Column, unsigned Source) {
    Lines.push_back(new SourceLineInfo(Line, Column, Source));
    return Lines.size();
  }
  
  /// RecordSource - Register a source file with debug info. Returns an source
  /// ID.
  unsigned RecordSource(const std::string &Directory,
                               const std::string &Source) {
    unsigned DirectoryID = Directories.insert(Directory);
    return SourceFiles.insert(SourceFileInfo(DirectoryID, Source));
  }
  
  /// getDirectories - Return the UniqueVector of std::string representing
  /// directories.
  const UniqueVector<std::string> &getDirectories() const {
    return Directories;
  }
  
  /// getSourceFiles - Return the UniqueVector of source files. 
  ///
  const UniqueVector<SourceFileInfo> &getSourceFiles() const {
    return SourceFiles;
  }
  
  /// getSourceLines - Return a vector of source lines.  Vector index + 1
  /// equals label ID.
  const std::vector<SourceLineInfo *> &getSourceLines() const {
    return Lines;
  }
  
}; // End class MachineDebugInfo

} // End llvm namespace

#endif

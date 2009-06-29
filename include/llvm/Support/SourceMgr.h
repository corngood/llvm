//===- SourceMgr.h - Manager for Source Buffers & Diagnostics ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SourceMgr class.  This class is used as a simple
// substrate for diagnostics, #include handling, and other low level things for
// simple parsers.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SOURCEMGR_H
#define SUPPORT_SOURCEMGR_H

#include <string>
#include <vector>
#include <cassert>

namespace llvm {
  class MemoryBuffer;
  class SourceMgr;
  
class SMLoc {
  const char *Ptr;
public:
  SMLoc() : Ptr(0) {}
  SMLoc(const SMLoc &RHS) : Ptr(RHS.Ptr) {}
  
  bool isValid() const { return Ptr != 0; }

  bool operator==(const SMLoc &RHS) const { return RHS.Ptr == Ptr; }
  bool operator!=(const SMLoc &RHS) const { return RHS.Ptr != Ptr; }

  const char *getPointer() const { return Ptr; }
  
  static SMLoc getFromPointer(const char *Ptr) {
    SMLoc L;
    L.Ptr = Ptr;
    return L;
  }
};

/// SourceMgr - This owns the files read by tblgen, handles include stacks,
/// and handles printing of diagnostics.
class SourceMgr {
  struct SrcBuffer {
    /// Buffer - The memory buffer for the file.
    MemoryBuffer *Buffer;
    
    /// IncludeLoc - This is the location of the parent include, or null if at
    /// the top level.
    SMLoc IncludeLoc;
  };
  
  /// Buffers - This is all of the buffers that we are reading from.
  std::vector<SrcBuffer> Buffers;
  
  // IncludeDirectories - This is the list of directories we should search for
  // include files in.
  std::vector<std::string> IncludeDirectories;
  
  SourceMgr(const SourceMgr&);    // DO NOT IMPLEMENT
  void operator=(const SourceMgr&); // DO NOT IMPLEMENT
public:
  SourceMgr() {}
  ~SourceMgr();
  
  void setIncludeDirs(const std::vector<std::string> &Dirs) {
    IncludeDirectories = Dirs;
  }
  
  const SrcBuffer &getBufferInfo(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i];
  }

  const MemoryBuffer *getMemoryBuffer(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i].Buffer;
  }
  
  SMLoc getParentIncludeLoc(unsigned i) const {
    assert(i < Buffers.size() && "Invalid Buffer ID!");
    return Buffers[i].IncludeLoc;
  }
  
  unsigned AddNewSourceBuffer(MemoryBuffer *F, SMLoc IncludeLoc) {
    SrcBuffer NB;
    NB.Buffer = F;
    NB.IncludeLoc = IncludeLoc;
    Buffers.push_back(NB);
    return Buffers.size()-1;
  }
  
  /// AddIncludeFile - Search for a file with the specified name in the current
  /// directory or in one of the IncludeDirs.  If no file is found, this returns
  /// ~0, otherwise it returns the buffer ID of the stacked file.
  unsigned AddIncludeFile(const std::string &Filename, SMLoc IncludeLoc);
  
  /// FindBufferContainingLoc - Return the ID of the buffer containing the
  /// specified location, returning -1 if not found.
  int FindBufferContainingLoc(SMLoc Loc) const;
  
  /// FindLineNumber - Find the line number for the specified location in the
  /// specified file.  This is not a fast method.
  unsigned FindLineNumber(SMLoc Loc, int BufferID = -1) const;
  
  /// PrintMessage - Emit a message about the specified location with the
  /// specified string.
  void PrintMessage(SMLoc Loc, const std::string &Msg) const;
  
private:
  void PrintIncludeStack(SMLoc IncludeLoc) const;
};
  
}  // end llvm namespace

#endif

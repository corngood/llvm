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
// compile units.  The main exception is source line correspondence where
// inlining may interleave code from various compile units.
//
// The following information can be retrieved from the MachineDebugInfo.
//
//  -- Source directories - Directories are uniqued based on their canonical
//     string and assigned a sequential numeric ID (base 1.)
//  -- Source files - Files are also uniqued based on their name and directory
//     ID.  A file ID is sequential number (base 1.)
//  -- Source line coorespondence - A vector of file ID, line#, column# triples.
//     A DEBUG_LOCATION instruction is generated  by the DAG Legalizer
//     corresponding to each entry in the source line list.  This allows a debug
//     emitter to generate labels referenced by debug information tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDEBUGINFO_H
#define LLVM_CODEGEN_MACHINEDEBUGINFO_H

#include "llvm/Support/Dwarf.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/GlobalValue.h"
#include "llvm/Pass.h"
#include "llvm/User.h"

#include <string>
#include <set>

namespace llvm {

//===----------------------------------------------------------------------===//
// Forward declarations.
class Constant;
class DebugInfoDesc;
class GlobalVariable;
class Module;
class PointerType;
class StructType;

//===----------------------------------------------------------------------===//
// Debug info constants.

enum {
  LLVMDebugVersion = 1,                 // Current version of debug information.
  DIInvalid = ~0U,                      // Invalid result indicator.
  
  // DebugInfoDesc type identifying tags.
  // FIXME - Change over with gcc4.
  DI_TAG_anchor = 0,
#if 1
  DI_TAG_compile_unit = DW_TAG_compile_unit,
  DI_TAG_global_variable = DW_TAG_variable,
  DI_TAG_subprogram = DW_TAG_subprogram
#else
  DI_TAG_compile_unit,
  DI_TAG_global_variable,
  DI_TAG_subprogram
#endif
};

//===----------------------------------------------------------------------===//
/// DIVisitor - Subclasses of this class apply steps to each of the fields in
/// the supplied DebugInfoDesc.
class DIVisitor {
public:
  DIVisitor() {}
  virtual ~DIVisitor() {}
  
  
  /// ApplyToFields - Target the visitor to each field of the debug information
  /// descriptor.
  void ApplyToFields(DebugInfoDesc *DD);
  
  /// Apply - Subclasses override each of these methods to perform the
  /// appropriate action for the type of field.
  virtual void Apply(int &Field) = 0;
  virtual void Apply(unsigned &Field) = 0;
  virtual void Apply(bool &Field) = 0;
  virtual void Apply(std::string &Field) = 0;
  virtual void Apply(DebugInfoDesc *&Field) = 0;
  virtual void Apply(GlobalVariable *&Field) = 0;
};

//===----------------------------------------------------------------------===//
/// DebugInfoDesc - This class is the base class for debug info descriptors.
///
class DebugInfoDesc {
private:
  unsigned Tag;                         // Content indicator.  Dwarf values are
                                        // used but that does not limit use to
                                        // Dwarf writers.
  
protected:
  DebugInfoDesc(unsigned T) : Tag(T) {}
  
public:
  virtual ~DebugInfoDesc() {}

  // Accessors
  unsigned getTag()          const { return Tag; }
  
  /// TagFromGlobal - Returns the Tag number from a debug info descriptor
  /// GlobalVariable.  Return DIIValid if operand is not an unsigned int.
  static unsigned TagFromGlobal(GlobalVariable *GV);

  /// DescFactory - Create an instance of debug info descriptor based on Tag.
  /// Return NULL if not a recognized Tag.
  static DebugInfoDesc *DescFactory(unsigned Tag);
  
  /// getLinkage - get linkage appropriate for this type of descriptor.
  ///
  virtual GlobalValue::LinkageTypes getLinkage() const;
    
  //===--------------------------------------------------------------------===//
  // Subclasses should supply the following static methods.
  
  // Implement isa/cast/dyncast.
  static bool classof(const DebugInfoDesc *)  { return true; }
  
  //===--------------------------------------------------------------------===//
  // Subclasses should supply the following virtual methods.
  
  /// ApplyToFields - Target the vistor to the fields of the descriptor.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const = 0;
  
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const = 0;
  
#ifndef NDEBUG
  virtual void dump() = 0;
#endif
};


//===----------------------------------------------------------------------===//
/// AnchorDesc - Descriptors of this class act as markers for identifying
/// descriptors of certain groups.
class AnchorDesc : public DebugInfoDesc {
private:  
  std::string Name;                     // Anchor type string.
  
public:
  AnchorDesc()
  : DebugInfoDesc(DI_TAG_anchor)
  , Name("")
  {}
  AnchorDesc(const std::string &N)
  : DebugInfoDesc(DI_TAG_anchor)
  , Name(N)
  {}
  
  // Accessors
  const std::string &getName() const { return Name; }

  // Implement isa/cast/dyncast.
  static bool classof(const AnchorDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D) {
    return D->getTag() == DI_TAG_anchor;
  }

  /// getLinkage - get linkage appropriate for this type of descriptor.
  ///
  virtual GlobalValue::LinkageTypes getLinkage() const;

  /// ApplyToFields - Target the visitor to the fields of the AnchorDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;
    
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// AnchoredDesc - This class manages anchors for a variety of top level
/// descriptors.
class AnchoredDesc : public DebugInfoDesc {
private:  
  AnchorDesc *Anchor;                   // Anchor for all descriptors of the
                                        // same type.

protected:

  AnchoredDesc(unsigned T);

public:  
  // Accessors.
  AnchorDesc *getAnchor() const { return Anchor; }
  void setAnchor(AnchorDesc *A) { Anchor = A; }

  //===--------------------------------------------------------------------===//
  // Subclasses should supply the following virtual methods.
  
  /// ApplyToFields - Target the visitor to the fields of the AnchoredDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);
};

//===----------------------------------------------------------------------===//
/// CompileUnitDesc - This class packages debug information associated with a 
/// source/header file.
class CompileUnitDesc : public AnchoredDesc {
private:  
  unsigned DebugVersion;                // LLVM debug version when produced.
  unsigned Language;                    // Language number (ex. DW_LANG_C89.)
  std::string FileName;                 // Source file name.
  std::string Directory;                // Source file directory.
  std::string Producer;                 // Compiler string.
  
public:
  CompileUnitDesc();
  
  // Accessors
  unsigned getDebugVersion()              const { return DebugVersion; }
  unsigned getLanguage()                  const { return Language; }
  const std::string &getFileName()        const { return FileName; }
  const std::string &getDirectory()       const { return Directory; }
  const std::string &getProducer()        const { return Producer; }
  void setLanguage(unsigned L)                  { Language = L; }
  void setFileName(const std::string &FN)       { FileName = FN; }
  void setDirectory(const std::string &D)       { Directory = D; }
  void setProducer(const std::string &P)        { Producer = P; }
  
  // FIXME - Need translation unit getter/setter.

  // Implement isa/cast/dyncast.
  static bool classof(const CompileUnitDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D) {
    return D->getTag() == DI_TAG_compile_unit;
  }
  
  /// DebugVersionFromGlobal - Returns the version number from a compile unit
  /// GlobalVariable.  Return DIIValid if operand is not an unsigned int.
  static unsigned DebugVersionFromGlobal(GlobalVariable *GV);
  
  /// ApplyToFields - Target the visitor to the fields of the CompileUnitDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;
    
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  virtual const char *getAnchorString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// GlobalDesc - This class is the base descriptor for global functions and
/// variables.
class GlobalDesc : public AnchoredDesc {
private:
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Global name.
  // FIXME - Use a descriptor.
  GlobalVariable *TyDesc;               // Type debug descriptor.
  bool IsStatic;                        // Is the global a static.
  bool IsDefinition;                    // Is the global defined in context.
  
protected:
  GlobalDesc(unsigned T);

public:
  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  const std::string &getName()               const { return Name; }
  bool isStatic()                            const { return IsStatic; }
  bool isDefinition()                        const { return IsDefinition; }
  void setContext(DebugInfoDesc *C)                { Context = C; }
  void setName(const std::string &N)               { Name = N; }
  void setIsStatic(bool IS)                        { IsStatic = IS; }
  void setIsDefinition(bool ID)                    { IsDefinition = ID; }

  /// ApplyToFields - Target the visitor to the fields of the GlobalDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);
};

//===----------------------------------------------------------------------===//
/// GlobalVariableDesc - This class packages debug information associated with a
/// GlobalVariable.
class GlobalVariableDesc : public GlobalDesc {
private:
  GlobalVariable *Global;               // llvm global.
  
public:
  GlobalVariableDesc();

  // Accessors.
  GlobalVariable *getGlobalVariable()        const { return Global; }
  void setGlobalVariable(GlobalVariable *GV)       { Global = GV; }
  
  // Implement isa/cast/dyncast.
  static bool classof(const GlobalVariableDesc *)  { return true; }
  static bool classof(const DebugInfoDesc *D) {
    return D->getTag() == DI_TAG_global_variable;
  }
  
  /// ApplyToFields - Target the visitor to the fields of the
  /// GlobalVariableDesc.
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  virtual const char *getAnchorString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// SubprogramDesc - This class packages debug information associated with a
/// subprogram/function.
class SubprogramDesc : public GlobalDesc {
private:
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Subprogram name.
  // FIXME - Use a descriptor.
  GlobalVariable *TyDesc;               // Type debug descriptor.
  bool IsStatic;                        // Is the subprogram a static.
  bool IsDefinition;                    // Is the subprogram defined in context.
  
public:
  SubprogramDesc();
  
  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  const std::string &getName()               const { return Name; }
  bool isStatic()                            const { return IsStatic; }
  bool isDefinition()                        const { return IsDefinition; }
  void setContext(DebugInfoDesc *C)                { Context = C; }
  void setName(const std::string &N)               { Name = N; }
  void setIsStatic(bool IS)                        { IsStatic = IS; }
  void setIsDefinition(bool ID)                    { IsDefinition = ID; }
  // FIXME - Other getters/setters.
  
  // Implement isa/cast/dyncast.
  static bool classof(const SubprogramDesc *)  { return true; }
  static bool classof(const DebugInfoDesc *D) {
    return D->getTag() == DI_TAG_subprogram;
  }
  
  /// ApplyToFields - Target the visitor to the fields of the  SubprogramDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  virtual const char *getAnchorString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// DIDeserializer - This class is responsible for casting GlobalVariables
/// into DebugInfoDesc objects.
class DIDeserializer {
private:
  unsigned DebugVersion;                // Version of debug information in use.
  std::map<GlobalVariable *, DebugInfoDesc *> GlobalDescs;
                                        // Previously defined gloabls.
  
public:
  DIDeserializer() : DebugVersion(LLVMDebugVersion) {}
  ~DIDeserializer() {}
  
  // Accessors
  unsigned getDebugVersion() const { return DebugVersion; }
  
  /// Deserialize - Reconstitute a GlobalVariable into it's component
  /// DebugInfoDesc objects.
  DebugInfoDesc *Deserialize(Value *V);
  DebugInfoDesc *Deserialize(GlobalVariable *GV);
};

//===----------------------------------------------------------------------===//
/// DISerializer - This class is responsible for casting DebugInfoDesc objects
/// into GlobalVariables.
class DISerializer {
private:
  Module *M;                            // Definition space module.
  PointerType *StrPtrTy;                // A "sbyte *" type.  Created lazily.
  PointerType *EmptyStructPtrTy;        // A "{ }*" type.  Created lazily.
  std::map<unsigned, StructType *> TagTypes;
                                        // Types per Tag.  Created lazily.
  std::map<DebugInfoDesc *, GlobalVariable *> DescGlobals;
                                        // Previously defined descriptors.
  std::map<const std::string, Constant *> StringCache;
                                        // Previously defined strings.
                                          
public:
  DISerializer()
  : M(NULL)
  , StrPtrTy(NULL)
  , EmptyStructPtrTy(NULL)
  , TagTypes()
  , DescGlobals()
  , StringCache()
  {}
  ~DISerializer() {}
  
  // Accessors
  Module *getModule()        const { return M; };
  void setModule(Module *module)  { M = module; }

  /// getStrPtrType - Return a "sbyte *" type.
  ///
  const PointerType *getStrPtrType();
  
  /// getEmptyStructPtrType - Return a "{ }*" type.
  ///
  const PointerType *getEmptyStructPtrType();
  
  /// getTagType - Return the type describing the specified descriptor (via
  /// tag.)
  const StructType *getTagType(DebugInfoDesc *DD);
  
  /// getString - Construct the string as constant string global.
  ///
  Constant *getString(const std::string &String);
  
  /// Serialize - Recursively cast the specified descriptor into a
  /// GlobalVariable so that it can be serialized to a .bc or .ll file.
  GlobalVariable *Serialize(DebugInfoDesc *DD);
};

//===----------------------------------------------------------------------===//
/// DIVerifier - This class is responsible for verifying the given network of
/// GlobalVariables are valid as DebugInfoDesc objects.
class DIVerifier {
private:
  unsigned DebugVersion;                // Version of debug information in use.
  std::set<GlobalVariable *> Visited;   // Tracks visits during recursion.
  std::map<unsigned, unsigned> Counts;  // Count of fields per Tag type.

  /// markVisited - Return true if the GlobalVariable hase been "seen" before.
  /// Mark markVisited otherwise.
  bool markVisited(GlobalVariable *GV);
  
public:
  DIVerifier() : DebugVersion(LLVMDebugVersion) {}
  ~DIVerifier() {}
  
  /// Verify - Return true if the GlobalVariable appears to be a valid
  /// serialization of a DebugInfoDesc.
  bool Verify(Value *V);
  bool Verify(GlobalVariable *GV);
};

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
  // Use the same serializer/deserializer/verifier for the module.
  DIDeserializer DR;

  // CompileUnits - Uniquing vector for compile units.
  UniqueVector<CompileUnitDesc *> CompileUnits;
  
  // Directories - Uniquing vector for directories.
  UniqueVector<std::string> Directories;
                                         
  // SourceFiles - Uniquing vector for source files.
  UniqueVector<SourceFileInfo> SourceFiles;

  // Lines - List of of source line correspondence.
  std::vector<SourceLineInfo *> Lines;

public:
  MachineDebugInfo();
  ~MachineDebugInfo();
  
  /// doInitialization - Initialize the debug state for a new module.
  ///
  bool doInitialization();
  
  /// doFinalization - Tear down the debug state after completion of a module.
  ///
  bool doFinalization();
  
  /// getDescFor - Convert a Value to a debug information descriptor.
  ///
  // FIXME - use new Value type when available.
  DebugInfoDesc *getDescFor(Value *V);
  
  /// Verify - Verify that a Value is debug information descriptor.
  ///
  bool Verify(Value *V);
  
  /// AnalyzeModule - Scan the module for global debug information.
  ///
  void AnalyzeModule(Module &M);
  
  /// hasInfo - Returns true if valid debug info is present.
  ///
  bool hasInfo() const { return !CompileUnits.empty(); }
  
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
  
  /// SetupCompileUnits - Set up the unique vector of compile units.
  ///
  void SetupCompileUnits(Module &M);

  /// getCompileUnits - Return a vector of debug compile units.
  ///
  const UniqueVector<CompileUnitDesc *> getCompileUnits() const;

  /// getGlobalVariables - Return a vector of debug GlobalVariables.
  ///
  std::vector<GlobalVariableDesc *> getGlobalVariables(Module &M);

}; // End class MachineDebugInfo

} // End llvm namespace

#endif

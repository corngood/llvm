//===-- llvm/CodeGen/DwarfDebug.h - Dwarf Debug Framework ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFDEBUG_H__
#define CODEGEN_ASMPRINTER_DWARFDEBUG_H__

#include "llvm/CodeGen/AsmPrinter.h"
#include "DIE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

class CompileUnit;
class DbgConcreteScope;
class DbgScope;
class DbgVariable;
class MachineFrameInfo;
class MachineLocation;
class MachineModuleInfo;
class MCAsmInfo;
class DIEAbbrev;
class DIE;
class DIEBlock;
class DIEEntry;

class DIEnumerator;
class DIDescriptor;
class DIVariable;
class DIGlobal;
class DIGlobalVariable;
class DISubprogram;
class DIBasicType;
class DIDerivedType;
class DIType;
class DINameSpace;
class DISubrange;
class DICompositeType;

//===----------------------------------------------------------------------===//
/// SrcLineInfo - This class is used to record source line correspondence.
///
class SrcLineInfo {
  unsigned Line;                     // Source line number.
  unsigned Column;                   // Source column.
  unsigned SourceID;                 // Source ID number.
  MCSymbol *Label;                   // Label in code ID number.
public:
  SrcLineInfo(unsigned L, unsigned C, unsigned S, MCSymbol *label)
    : Line(L), Column(C), SourceID(S), Label(label) {}

  // Accessors
  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }
  unsigned getSourceID() const { return SourceID; }
  MCSymbol *getLabel() const { return Label; }
};

class DwarfDebug {
  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  /// MMI - Collected machine module information.
  MachineModuleInfo *MMI;

  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //

  /// ModuleCU - All DIEs are inserted in ModuleCU.
  CompileUnit *ModuleCU;

  /// AbbreviationsSet - Used to uniquely define abbreviations.
  ///
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  /// Abbreviations - A list of all the unique abbreviations in use.
  ///
  std::vector<DIEAbbrev *> Abbreviations;

  /// DirectoryIdMap - Directory name to directory id map.
  ///
  StringMap<unsigned> DirectoryIdMap;

  /// DirectoryNames - A list of directory names.
  SmallVector<std::string, 8> DirectoryNames;

  /// SourceFileIdMap - Source file name to source file id map.
  ///
  StringMap<unsigned> SourceFileIdMap;

  /// SourceFileNames - A list of source file names.
  SmallVector<std::string, 8> SourceFileNames;

  /// SourceIdMap - Source id map, i.e. pair of directory id and source file
  /// id mapped to a unique id.
  DenseMap<std::pair<unsigned, unsigned>, unsigned> SourceIdMap;

  /// SourceIds - Reverse map from source id to directory id + file id pair.
  ///
  SmallVector<std::pair<unsigned, unsigned>, 8> SourceIds;

  /// Lines - List of source line correspondence.
  std::vector<SrcLineInfo> Lines;

  /// DIEBlocks - A list of all the DIEBlocks in use.
  std::vector<DIEBlock *> DIEBlocks;

  // DIEValueAllocator - All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  /// StringPool - A String->Symbol mapping of strings used by indirect
  /// references.
  StringMap<std::pair<MCSymbol*, unsigned> > StringPool;
  unsigned NextStringPoolNumber;
  
  MCSymbol *getStringPoolEntry(StringRef Str);

  /// SectionMap - Provides a unique id per text section.
  ///
  UniqueVector<const MCSection*> SectionMap;

  /// SectionSourceLines - Tracks line numbers per text section.
  ///
  std::vector<std::vector<SrcLineInfo> > SectionSourceLines;

  // CurrentFnDbgScope - Top level scope for the current function.
  //
  DbgScope *CurrentFnDbgScope;
  
  /// DbgScopeMap - Tracks the scopes in the current function.  Owns the
  /// contained DbgScope*s.
  ///
  DenseMap<MDNode *, DbgScope *> DbgScopeMap;

  /// ConcreteScopes - Tracks the concrete scopees in the current function.
  /// These scopes are also included in DbgScopeMap.
  DenseMap<MDNode *, DbgScope *> ConcreteScopes;

  /// AbstractScopes - Tracks the abstract scopes a module. These scopes are
  /// not included DbgScopeMap.  AbstractScopes owns its DbgScope*s.
  DenseMap<MDNode *, DbgScope *> AbstractScopes;

  /// AbstractScopesList - Tracks abstract scopes constructed while processing
  /// a function. This list is cleared during endFunction().
  SmallVector<DbgScope *, 4>AbstractScopesList;

  /// AbstractVariables - Collection on abstract variables.  Owned by the
  /// DbgScopes in AbstractScopes.
  DenseMap<MDNode *, DbgVariable *> AbstractVariables;

  /// DbgValueStartMap - Tracks starting scope of variable DIEs.
  /// If the scope of an object begins sometime after the low pc value for the 
  /// scope most closely enclosing the object, the object entry may have a 
  /// DW_AT_start_scope attribute.
  DenseMap<const MachineInstr *, DbgVariable *> DbgValueStartMap;

  /// InliendSubprogramDIEs - Collection of subprgram DIEs that are marked
  /// (at the end of the module) as DW_AT_inline.
  SmallPtrSet<DIE *, 4> InlinedSubprogramDIEs;

  /// ContainingTypeMap - This map is used to keep track of subprogram DIEs that
  /// need DW_AT_containing_type attribute. This attribute points to a DIE that
  /// corresponds to the MDNode mapped with the subprogram DIE.
  DenseMap<DIE *, MDNode *> ContainingTypeMap;

  typedef SmallVector<DbgScope *, 2> ScopeVector;
  SmallPtrSet<const MachineInstr *, 8> InsnsBeginScopeSet;
  SmallPtrSet<const MachineInstr *, 8> InsnsEndScopeSet;

  /// InlineInfo - Keep track of inlined functions and their location.  This
  /// information is used to populate debug_inlined section.
  typedef std::pair<MCSymbol*, DIE *> InlineInfoLabels;
  DenseMap<MDNode*, SmallVector<InlineInfoLabels, 4> > InlineInfo;
  SmallVector<MDNode *, 4> InlinedSPNodes;

  /// InsnBeforeLabelMap - Maps instruction with label emitted before 
  /// instruction.
  DenseMap<const MachineInstr *, MCSymbol *> InsnBeforeLabelMap;

  /// InsnAfterLabelMap - Maps instruction with label emitted after
  /// instruction.
  DenseMap<const MachineInstr *, MCSymbol *> InsnAfterLabelMap;

  SmallVector<const MCSymbol *, 8> DebugRangeSymbols;

  /// Previous instruction's location information. This is used to determine
  /// label location to indicate scope boundries in dwarf debug info.
  DebugLoc PrevInstLoc;
  MCSymbol *PrevLabel;

  struct FunctionDebugFrameInfo {
    unsigned Number;
    std::vector<MachineMove> Moves;

    FunctionDebugFrameInfo(unsigned Num, const std::vector<MachineMove> &M)
      : Number(Num), Moves(M) {}
  };

  std::vector<FunctionDebugFrameInfo> DebugFrames;

  // Section Symbols: these are assembler temporary labels that are emitted at
  // the beginning of each supported dwarf section.  These are used to form
  // section offsets and are created by EmitSectionLabels.
  MCSymbol *DwarfFrameSectionSym, *DwarfInfoSectionSym, *DwarfAbbrevSectionSym;
  MCSymbol *DwarfStrSectionSym, *TextSectionSym, *DwarfDebugRangeSectionSym;
  
private:
  
  /// getSourceDirectoryAndFileIds - Return the directory and file ids that
  /// maps to the source id. Source id starts at 1.
  std::pair<unsigned, unsigned>
  getSourceDirectoryAndFileIds(unsigned SId) const {
    return SourceIds[SId-1];
  }

  /// getNumSourceDirectories - Return the number of source directories in the
  /// debug info.
  unsigned getNumSourceDirectories() const {
    return DirectoryNames.size();
  }

  /// getSourceDirectoryName - Return the name of the directory corresponding
  /// to the id.
  const std::string &getSourceDirectoryName(unsigned Id) const {
    return DirectoryNames[Id - 1];
  }

  /// getSourceFileName - Return the name of the source file corresponding
  /// to the id.
  const std::string &getSourceFileName(unsigned Id) const {
    return SourceFileNames[Id - 1];
  }

  /// getNumSourceIds - Return the number of unique source ids.
  unsigned getNumSourceIds() const {
    return SourceIds.size();
  }

  /// assignAbbrevNumber - Define a unique number for the abbreviation.
  ///
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  /// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
  /// information entry.
  DIEEntry *createDIEEntry(DIE *Entry);

  /// addUInt - Add an unsigned integer attribute data and value.
  ///
  void addUInt(DIE *Die, unsigned Attribute, unsigned Form, uint64_t Integer);

  /// addSInt - Add an signed integer attribute data and value.
  ///
  void addSInt(DIE *Die, unsigned Attribute, unsigned Form, int64_t Integer);

  /// addString - Add a string attribute data and value.
  ///
  void addString(DIE *Die, unsigned Attribute, unsigned Form,
                 const StringRef Str);

  /// addLabel - Add a Dwarf label attribute data and value.
  ///
  void addLabel(DIE *Die, unsigned Attribute, unsigned Form,
                const MCSymbol *Label);

  /// addDelta - Add a label delta attribute data and value.
  ///
  void addDelta(DIE *Die, unsigned Attribute, unsigned Form,
                const MCSymbol *Hi, const MCSymbol *Lo);

  /// addDIEEntry - Add a DIE attribute data and value.
  ///
  void addDIEEntry(DIE *Die, unsigned Attribute, unsigned Form, DIE *Entry);
  
  /// addBlock - Add block data.
  ///
  void addBlock(DIE *Die, unsigned Attribute, unsigned Form, DIEBlock *Block);

  /// addSourceLine - Add location information to specified debug information
  /// entry.
  void addSourceLine(DIE *Die, const DIVariable *V);
  void addSourceLine(DIE *Die, const DIGlobal *G);
  void addSourceLine(DIE *Die, const DISubprogram *SP);
  void addSourceLine(DIE *Die, const DIType *Ty);
  void addSourceLine(DIE *Die, const DINameSpace *NS);

  /// addAddress - Add an address attribute to a die based on the location
  /// provided.
  void addAddress(DIE *Die, unsigned Attribute,
                  const MachineLocation &Location);

  /// addComplexAddress - Start with the address based on the location provided,
  /// and generate the DWARF information necessary to find the actual variable
  /// (navigating the extra location information encoded in the type) based on
  /// the starting location.  Add the DWARF information to the die.
  ///
  void addComplexAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                         const MachineLocation &Location);

  // FIXME: Should be reformulated in terms of addComplexAddress.
  /// addBlockByrefAddress - Start with the address based on the location
  /// provided, and generate the DWARF information necessary to find the
  /// actual Block variable (navigating the Block struct) based on the
  /// starting location.  Add the DWARF information to the die.  Obsolete,
  /// please use addComplexAddress instead.
  ///
  void addBlockByrefAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                            const MachineLocation &Location);

  /// addToContextOwner - Add Die into the list of its context owner's children.
  void addToContextOwner(DIE *Die, DIDescriptor Context);

  /// addType - Add a new type attribute to the specified entity.
  void addType(DIE *Entity, DIType Ty);

 
  /// getOrCreateNameSpace - Create a DIE for DINameSpace.
  DIE *getOrCreateNameSpace(DINameSpace NS);

  /// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
  /// given DIType.
  DIE *getOrCreateTypeDIE(DIType Ty);

  void addPubTypes(DISubprogram SP);

  /// constructTypeDIE - Construct basic type die from DIBasicType.
  void constructTypeDIE(DIE &Buffer,
                        DIBasicType BTy);

  /// constructTypeDIE - Construct derived type die from DIDerivedType.
  void constructTypeDIE(DIE &Buffer,
                        DIDerivedType DTy);

  /// constructTypeDIE - Construct type DIE from DICompositeType.
  void constructTypeDIE(DIE &Buffer,
                        DICompositeType CTy);

  /// constructSubrangeDIE - Construct subrange DIE from DISubrange.
  void constructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy);

  /// constructArrayTypeDIE - Construct array type DIE from DICompositeType.
  void constructArrayTypeDIE(DIE &Buffer, 
                             DICompositeType *CTy);

  /// constructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
  DIE *constructEnumTypeDIE(DIEnumerator ETy);

  /// createGlobalVariableDIE - Create new DIE using GV.
  DIE *createGlobalVariableDIE(const DIGlobalVariable &GV);

  /// createMemberDIE - Create new member DIE.
  DIE *createMemberDIE(const DIDerivedType &DT);

  /// createSubprogramDIE - Create new DIE using SP.
  DIE *createSubprogramDIE(const DISubprogram &SP, bool MakeDecl = false);

  /// getOrCreateDbgScope - Create DbgScope for the scope.
  DbgScope *getOrCreateDbgScope(MDNode *Scope, MDNode *InlinedAt);

  DbgScope *getOrCreateAbstractScope(MDNode *N);

  /// findAbstractVariable - Find abstract variable associated with Var.
  DbgVariable *findAbstractVariable(DIVariable &Var, unsigned FrameIdx, 
                                    DebugLoc Loc);
  DbgVariable *findAbstractVariable(DIVariable &Var, const MachineInstr *MI,
                                    DebugLoc Loc);

  /// updateSubprogramScopeDIE - Find DIE for the given subprogram and 
  /// attach appropriate DW_AT_low_pc and DW_AT_high_pc attributes.
  /// If there are global variables in this scope then create and insert
  /// DIEs for these variables.
  DIE *updateSubprogramScopeDIE(MDNode *SPNode);

  /// constructLexicalScope - Construct new DW_TAG_lexical_block 
  /// for this scope and attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(DbgScope *Scope);

  /// constructInlinedScopeDIE - This scope represents inlined body of
  /// a function. Construct DIE to represent this concrete inlined copy
  /// of the function.
  DIE *constructInlinedScopeDIE(DbgScope *Scope);

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  DIE *constructVariableDIE(DbgVariable *DV, DbgScope *S);

  /// constructScopeDIE - Construct a DIE for this scope.
  DIE *constructScopeDIE(DbgScope *Scope);

  /// EmitSectionLabels - Emit initial Dwarf sections with a label at
  /// the start of each one.
  void EmitSectionLabels();

  /// emitDIE - Recusively Emits a debug information entry.
  ///
  void emitDIE(DIE *Die);

  /// computeSizeAndOffset - Compute the size and offset of a DIE.
  ///
  unsigned computeSizeAndOffset(DIE *Die, unsigned Offset, bool Last);

  /// computeSizeAndOffsets - Compute the size and offset of all the DIEs.
  ///
  void computeSizeAndOffsets();

  /// EmitDebugInfo - Emit the debug info section.
  ///
  void emitDebugInfo();

  /// emitAbbreviations - Emit the abbreviation section.
  ///
  void emitAbbreviations() const;

  /// emitEndOfLineMatrix - Emit the last address of the section and the end of
  /// the line matrix.
  ///
  void emitEndOfLineMatrix(unsigned SectionEnd);

  /// emitDebugLines - Emit source line information.
  ///
  void emitDebugLines();

  /// emitCommonDebugFrame - Emit common frame info into a debug frame section.
  ///
  void emitCommonDebugFrame();

  /// emitFunctionDebugFrame - Emit per function frame info into a debug frame
  /// section.
  void emitFunctionDebugFrame(const FunctionDebugFrameInfo &DebugFrameInfo);

  /// emitDebugPubNames - Emit visible names into a debug pubnames section.
  ///
  void emitDebugPubNames();

  /// emitDebugPubTypes - Emit visible types into a debug pubtypes section.
  ///
  void emitDebugPubTypes();

  /// emitDebugStr - Emit visible names into a debug str section.
  ///
  void emitDebugStr();

  /// emitDebugLoc - Emit visible names into a debug loc section.
  ///
  void emitDebugLoc();

  /// EmitDebugARanges - Emit visible names into a debug aranges section.
  ///
  void EmitDebugARanges();

  /// emitDebugRanges - Emit visible names into a debug ranges section.
  ///
  void emitDebugRanges();

  /// emitDebugMacInfo - Emit visible names into a debug macinfo section.
  ///
  void emitDebugMacInfo();

  /// emitDebugInlineInfo - Emit inline info using following format.
  /// Section Header:
  /// 1. length of section
  /// 2. Dwarf version number
  /// 3. address size.
  ///
  /// Entries (one "entry" for each function that was inlined):
  ///
  /// 1. offset into __debug_str section for MIPS linkage name, if exists; 
  ///   otherwise offset into __debug_str for regular function name.
  /// 2. offset into __debug_str section for regular function name.
  /// 3. an unsigned LEB128 number indicating the number of distinct inlining 
  /// instances for the function.
  /// 
  /// The rest of the entry consists of a {die_offset, low_pc}  pair for each 
  /// inlined instance; the die_offset points to the inlined_subroutine die in
  /// the __debug_info section, and the low_pc is the starting address  for the
  ///  inlining instance.
  void emitDebugInlineInfo();

  /// GetOrCreateSourceID - Look up the source id with the given directory and
  /// source file names. If none currently exists, create a new id and insert it
  /// in the SourceIds map. This can update DirectoryNames and SourceFileNames
  /// maps as well.
  unsigned GetOrCreateSourceID(StringRef DirName, StringRef FileName);

  void constructCompileUnit(MDNode *N);

  void constructGlobalVariableDIE(MDNode *N);

  void constructSubprogramDIE(MDNode *N);

  // FIXME: This should go away in favor of complex addresses.
  /// Find the type the programmer originally declared the variable to be
  /// and return that type.  Obsolete, use GetComplexAddrType instead.
  ///
  DIType getBlockByrefType(DIType Ty, std::string Name);

  /// recordSourceLine - Register a source line with debug info. Returns the
  /// unique label that was emitted and which provides correspondence to
  /// the source line list.
  MCSymbol *recordSourceLine(unsigned Line, unsigned Col, MDNode *Scope);
  
  /// getSourceLineCount - Return the number of source lines in the debug
  /// info.
  unsigned getSourceLineCount() const {
    return Lines.size();
  }
  
  /// identifyScopeMarkers() - Indentify instructions that are marking
  /// beginning of or end of a scope.
  void identifyScopeMarkers();

  /// extractScopeInformation - Scan machine instructions in this function
  /// and collect DbgScopes. Return true, if atleast one scope was found.
  bool extractScopeInformation();
  
  /// collectVariableInfo - Populate DbgScope entries with variables' info.
  void collectVariableInfo();
  
public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(AsmPrinter *A, Module *M);
  ~DwarfDebug();

  /// beginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void beginModule(Module *M);

  /// endModule - Emit all Dwarf sections that should come after the content.
  ///
  void endModule();

  /// beginFunction - Gather pre-function debug information.  Assumes being
  /// emitted immediately after the function entry point.
  void beginFunction(const MachineFunction *MF);

  /// endFunction - Gather and emit post-function debug information.
  ///
  void endFunction(const MachineFunction *MF);

  /// beginScope - Process beginning of a scope.
  void beginScope(const MachineInstr *MI);

  /// endScope - Prcess end of a scope.
  void endScope(const MachineInstr *MI);
};
} // End of namespace llvm

#endif

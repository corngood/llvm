//===-- llvm/CodeGen/DwarfWriter.cpp - Dwarf Framework ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf info into asm files.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DwarfWriter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Mangler.h"
#include "llvm/System/Path.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <ostream>
#include <string>
using namespace llvm;
using namespace llvm::dwarf;

namespace llvm {
  
//===----------------------------------------------------------------------===//

/// Configuration values for initial hash set sizes (log2).
///
static const unsigned InitDiesSetSize          = 9; // 512
static const unsigned InitAbbreviationsSetSize = 9; // 512
static const unsigned InitValuesSetSize        = 9; // 512

//===----------------------------------------------------------------------===//
/// Forward declarations.
///
class DIE;
class DIEValue;

//===----------------------------------------------------------------------===//
/// DWLabel - Labels are used to track locations in the assembler file.
/// Labels appear in the form @verbatim <prefix><Tag><Number> @endverbatim, 
/// where the tag is a category of label (Ex. location) and number is a value 
/// unique in that category.
class DWLabel {
public:
  /// Tag - Label category tag. Should always be a staticly declared C string.
  ///
  const char *Tag;
  
  /// Number - Value to make label unique.
  ///
  unsigned    Number;

  DWLabel(const char *T, unsigned N) : Tag(T), Number(N) {}
  
  void Profile(FoldingSetNodeID &ID) const {
    ID.AddString(std::string(Tag));
    ID.AddInteger(Number);
  }
  
#ifndef NDEBUG
  void print(std::ostream *O) const {
    if (O) print(*O);
  }
  void print(std::ostream &O) const {
    O << "." << Tag;
    if (Number) O << Number;
  }
#endif
};

//===----------------------------------------------------------------------===//
/// DIEAbbrevData - Dwarf abbreviation data, describes the one attribute of a
/// Dwarf abbreviation.
class DIEAbbrevData {
private:
  /// Attribute - Dwarf attribute code.
  ///
  unsigned Attribute;
  
  /// Form - Dwarf form code.
  ///              
  unsigned Form;                      
  
public:
  DIEAbbrevData(unsigned A, unsigned F)
  : Attribute(A)
  , Form(F)
  {}
  
  // Accessors.
  unsigned getAttribute() const { return Attribute; }
  unsigned getForm()      const { return Form; }

  /// Profile - Used to gather unique data for the abbreviation folding set.
  ///
  void Profile(FoldingSetNodeID &ID)const  {
    ID.AddInteger(Attribute);
    ID.AddInteger(Form);
  }
};

//===----------------------------------------------------------------------===//
/// DIEAbbrev - Dwarf abbreviation, describes the organization of a debug
/// information object.
class DIEAbbrev : public FoldingSetNode {
private:
  /// Tag - Dwarf tag code.
  ///
  unsigned Tag;
  
  /// Unique number for node.
  ///
  unsigned Number;

  /// ChildrenFlag - Dwarf children flag.
  ///
  unsigned ChildrenFlag;

  /// Data - Raw data bytes for abbreviation.
  ///
  std::vector<DIEAbbrevData> Data;

public:

  DIEAbbrev(unsigned T, unsigned C)
  : Tag(T)
  , ChildrenFlag(C)
  , Data()
  {}
  ~DIEAbbrev() {}
  
  // Accessors.
  unsigned getTag()                           const { return Tag; }
  unsigned getNumber()                        const { return Number; }
  unsigned getChildrenFlag()                  const { return ChildrenFlag; }
  const std::vector<DIEAbbrevData> &getData() const { return Data; }
  void setTag(unsigned T)                           { Tag = T; }
  void setChildrenFlag(unsigned CF)                 { ChildrenFlag = CF; }
  void setNumber(unsigned N)                        { Number = N; }
  
  /// AddAttribute - Adds another set of attribute information to the
  /// abbreviation.
  void AddAttribute(unsigned Attribute, unsigned Form) {
    Data.push_back(DIEAbbrevData(Attribute, Form));
  }
  
  /// AddFirstAttribute - Adds a set of attribute information to the front
  /// of the abbreviation.
  void AddFirstAttribute(unsigned Attribute, unsigned Form) {
    Data.insert(Data.begin(), DIEAbbrevData(Attribute, Form));
  }
  
  /// Profile - Used to gather unique data for the abbreviation folding set.
  ///
  void Profile(FoldingSetNodeID &ID) {
    ID.AddInteger(Tag);
    ID.AddInteger(ChildrenFlag);
    
    // For each attribute description.
    for (unsigned i = 0, N = Data.size(); i < N; ++i)
      Data[i].Profile(ID);
  }
  
  /// Emit - Print the abbreviation using the specified Dwarf writer.
  ///
  void Emit(const DwarfDebug &DD) const; 
      
#ifndef NDEBUG
  void print(std::ostream *O) {
    if (O) print(*O);
  }
  void print(std::ostream &O);
  void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// DIE - A structured debug information entry.  Has an abbreviation which
/// describes it's organization.
class DIE : public FoldingSetNode {
protected:
  /// Abbrev - Buffer for constructing abbreviation.
  ///
  DIEAbbrev Abbrev;
  
  /// Offset - Offset in debug info section.
  ///
  unsigned Offset;
  
  /// Size - Size of instance + children.
  ///
  unsigned Size;
  
  /// Children DIEs.
  ///
  std::vector<DIE *> Children;
  
  /// Attributes values.
  ///
  std::vector<DIEValue *> Values;
  
public:
  explicit DIE(unsigned Tag)
  : Abbrev(Tag, DW_CHILDREN_no)
  , Offset(0)
  , Size(0)
  , Children()
  , Values()
  {}
  virtual ~DIE();
  
  // Accessors.
  DIEAbbrev &getAbbrev()                           { return Abbrev; }
  unsigned   getAbbrevNumber()               const {
    return Abbrev.getNumber();
  }
  unsigned getTag()                          const { return Abbrev.getTag(); }
  unsigned getOffset()                       const { return Offset; }
  unsigned getSize()                         const { return Size; }
  const std::vector<DIE *> &getChildren()    const { return Children; }
  std::vector<DIEValue *> &getValues()       { return Values; }
  void setTag(unsigned Tag)                  { Abbrev.setTag(Tag); }
  void setOffset(unsigned O)                 { Offset = O; }
  void setSize(unsigned S)                   { Size = S; }
  
  /// AddValue - Add a value and attributes to a DIE.
  ///
  void AddValue(unsigned Attribute, unsigned Form, DIEValue *Value) {
    Abbrev.AddAttribute(Attribute, Form);
    Values.push_back(Value);
  }
  
  /// SiblingOffset - Return the offset of the debug information entry's
  /// sibling.
  unsigned SiblingOffset() const { return Offset + Size; }
  
  /// AddSiblingOffset - Add a sibling offset field to the front of the DIE.
  ///
  void AddSiblingOffset();

  /// AddChild - Add a child to the DIE.
  ///
  void AddChild(DIE *Child) {
    Abbrev.setChildrenFlag(DW_CHILDREN_yes);
    Children.push_back(Child);
  }
  
  /// Detach - Detaches objects connected to it after copying.
  ///
  void Detach() {
    Children.clear();
  }
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  void Profile(FoldingSetNodeID &ID) ;
      
#ifndef NDEBUG
  void print(std::ostream *O, unsigned IncIndent = 0) {
    if (O) print(*O, IncIndent);
  }
  void print(std::ostream &O, unsigned IncIndent = 0);
  void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// DIEValue - A debug information entry value.
///
class DIEValue : public FoldingSetNode {
public:
  enum {
    isInteger,
    isString,
    isLabel,
    isAsIsLabel,
    isDelta,
    isEntry,
    isBlock
  };
  
  /// Type - Type of data stored in the value.
  ///
  unsigned Type;
  
  explicit DIEValue(unsigned T)
  : Type(T)
  {}
  virtual ~DIEValue() {}
  
  // Accessors
  unsigned getType()  const { return Type; }
  
  // Implement isa/cast/dyncast.
  static bool classof(const DIEValue *) { return true; }
  
  /// EmitValue - Emit value via the Dwarf writer.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form) = 0;
  
  /// SizeOf - Return the size of a value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const = 0;
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  virtual void Profile(FoldingSetNodeID &ID) = 0;
      
#ifndef NDEBUG
  void print(std::ostream *O) {
    if (O) print(*O);
  }
  virtual void print(std::ostream &O) = 0;
  void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// DWInteger - An integer value DIE.
/// 
class DIEInteger : public DIEValue {
private:
  uint64_t Integer;
  
public:
  explicit DIEInteger(uint64_t I) : DIEValue(isInteger), Integer(I) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEInteger *) { return true; }
  static bool classof(const DIEValue *I)  { return I->Type == isInteger; }
  
  /// BestForm - Choose the best form for integer.
  ///
  static unsigned BestForm(bool IsSigned, uint64_t Integer) {
    if (IsSigned) {
      if ((char)Integer == (signed)Integer)   return DW_FORM_data1;
      if ((short)Integer == (signed)Integer)  return DW_FORM_data2;
      if ((int)Integer == (signed)Integer)    return DW_FORM_data4;
    } else {
      if ((unsigned char)Integer == Integer)  return DW_FORM_data1;
      if ((unsigned short)Integer == Integer) return DW_FORM_data2;
      if ((unsigned int)Integer == Integer)   return DW_FORM_data4;
    }
    return DW_FORM_data8;
  }
    
  /// EmitValue - Emit integer of appropriate size.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form);
  
  /// SizeOf - Determine size of integer value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const;
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  static void Profile(FoldingSetNodeID &ID, unsigned Integer) {
    ID.AddInteger(isInteger);
    ID.AddInteger(Integer);
  }
  virtual void Profile(FoldingSetNodeID &ID) { Profile(ID, Integer); }
  
#ifndef NDEBUG
  virtual void print(std::ostream &O) {
    O << "Int: " << (int64_t)Integer
      << "  0x" << std::hex << Integer << std::dec;
  }
#endif
};

//===----------------------------------------------------------------------===//
/// DIEString - A string value DIE.
/// 
class DIEString : public DIEValue {
public:
  const std::string String;
  
  explicit DIEString(const std::string &S) : DIEValue(isString), String(S) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEString *) { return true; }
  static bool classof(const DIEValue *S) { return S->Type == isString; }
  
  /// EmitValue - Emit string value.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form);
  
  /// SizeOf - Determine size of string value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const {
    return String.size() + sizeof(char); // sizeof('\0');
  }
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  static void Profile(FoldingSetNodeID &ID, const std::string &String) {
    ID.AddInteger(isString);
    ID.AddString(String);
  }
  virtual void Profile(FoldingSetNodeID &ID) { Profile(ID, String); }
  
#ifndef NDEBUG
  virtual void print(std::ostream &O) {
    O << "Str: \"" << String << "\"";
  }
#endif
};

//===----------------------------------------------------------------------===//
/// DIEDwarfLabel - A Dwarf internal label expression DIE.
//
class DIEDwarfLabel : public DIEValue {
public:

  const DWLabel Label;
  
  explicit DIEDwarfLabel(const DWLabel &L) : DIEValue(isLabel), Label(L) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEDwarfLabel *)  { return true; }
  static bool classof(const DIEValue *L) { return L->Type == isLabel; }
  
  /// EmitValue - Emit label value.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form);
  
  /// SizeOf - Determine size of label value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const;
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  static void Profile(FoldingSetNodeID &ID, const DWLabel &Label) {
    ID.AddInteger(isLabel);
    Label.Profile(ID);
  }
  virtual void Profile(FoldingSetNodeID &ID) { Profile(ID, Label); }
  
#ifndef NDEBUG
  virtual void print(std::ostream &O) {
    O << "Lbl: ";
    Label.print(O);
  }
#endif
};


//===----------------------------------------------------------------------===//
/// DIEObjectLabel - A label to an object in code or data.
//
class DIEObjectLabel : public DIEValue {
public:
  const std::string Label;
  
  explicit DIEObjectLabel(const std::string &L)
  : DIEValue(isAsIsLabel), Label(L) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEObjectLabel *) { return true; }
  static bool classof(const DIEValue *L)    { return L->Type == isAsIsLabel; }
  
  /// EmitValue - Emit label value.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form);
  
  /// SizeOf - Determine size of label value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const;
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  static void Profile(FoldingSetNodeID &ID, const std::string &Label) {
    ID.AddInteger(isAsIsLabel);
    ID.AddString(Label);
  }
  virtual void Profile(FoldingSetNodeID &ID) { Profile(ID, Label); }

#ifndef NDEBUG
  virtual void print(std::ostream &O) {
    O << "Obj: " << Label;
  }
#endif
};

//===----------------------------------------------------------------------===//
/// DIEDelta - A simple label difference DIE.
/// 
class DIEDelta : public DIEValue {
public:
  const DWLabel LabelHi;
  const DWLabel LabelLo;
  
  DIEDelta(const DWLabel &Hi, const DWLabel &Lo)
  : DIEValue(isDelta), LabelHi(Hi), LabelLo(Lo) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEDelta *)  { return true; }
  static bool classof(const DIEValue *D) { return D->Type == isDelta; }
  
  /// EmitValue - Emit delta value.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form);
  
  /// SizeOf - Determine size of delta value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const;
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  static void Profile(FoldingSetNodeID &ID, const DWLabel &LabelHi,
                                            const DWLabel &LabelLo) {
    ID.AddInteger(isDelta);
    LabelHi.Profile(ID);
    LabelLo.Profile(ID);
  }
  virtual void Profile(FoldingSetNodeID &ID) { Profile(ID, LabelHi, LabelLo); }

#ifndef NDEBUG
  virtual void print(std::ostream &O) {
    O << "Del: ";
    LabelHi.print(O);
    O << "-";
    LabelLo.print(O);
  }
#endif
};

//===----------------------------------------------------------------------===//
/// DIEntry - A pointer to another debug information entry.  An instance of this
/// class can also be used as a proxy for a debug information entry not yet
/// defined (ie. types.)
class DIEntry : public DIEValue {
public:
  DIE *Entry;
  
  explicit DIEntry(DIE *E) : DIEValue(isEntry), Entry(E) {}
  
  // Implement isa/cast/dyncast.
  static bool classof(const DIEntry *)   { return true; }
  static bool classof(const DIEValue *E) { return E->Type == isEntry; }
  
  /// EmitValue - Emit debug information entry offset.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form);
  
  /// SizeOf - Determine size of debug information entry in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const {
    return sizeof(int32_t);
  }
  
  /// Profile - Used to gather unique data for the value folding set.
  ///
  static void Profile(FoldingSetNodeID &ID, DIE *Entry) {
    ID.AddInteger(isEntry);
    ID.AddPointer(Entry);
  }
  virtual void Profile(FoldingSetNodeID &ID) {
    ID.AddInteger(isEntry);
    
    if (Entry) {
      ID.AddPointer(Entry);
    } else {
      ID.AddPointer(this);
    }
  }
  
#ifndef NDEBUG
  virtual void print(std::ostream &O) {
    O << "Die: 0x" << std::hex << (intptr_t)Entry << std::dec;
  }
#endif
};

//===----------------------------------------------------------------------===//
/// DIEBlock - A block of values.  Primarily used for location expressions.
//
class DIEBlock : public DIEValue, public DIE {
public:
  unsigned Size;                        // Size in bytes excluding size header.
  
  DIEBlock()
  : DIEValue(isBlock)
  , DIE(0)
  , Size(0)
  {}
  ~DIEBlock()  {
  }
  
  // Implement isa/cast/dyncast.
  static bool classof(const DIEBlock *)  { return true; }
  static bool classof(const DIEValue *E) { return E->Type == isBlock; }
  
  /// ComputeSize - calculate the size of the block.
  ///
  unsigned ComputeSize(DwarfDebug &DD);
  
  /// BestForm - Choose the best form for data.
  ///
  unsigned BestForm() const {
    if ((unsigned char)Size == Size)  return DW_FORM_block1;
    if ((unsigned short)Size == Size) return DW_FORM_block2;
    if ((unsigned int)Size == Size)   return DW_FORM_block4;
    return DW_FORM_block;
  }

  /// EmitValue - Emit block data.
  ///
  virtual void EmitValue(DwarfDebug &DD, unsigned Form);
  
  /// SizeOf - Determine size of block data in bytes.
  ///
  virtual unsigned SizeOf(const DwarfDebug &DD, unsigned Form) const;
  

  /// Profile - Used to gather unique data for the value folding set.
  ///
  virtual void Profile(FoldingSetNodeID &ID) {
    ID.AddInteger(isBlock);
    DIE::Profile(ID);
  }
  
#ifndef NDEBUG
  virtual void print(std::ostream &O) {
    O << "Blk: ";
    DIE::print(O, 5);
  }
#endif
};

//===----------------------------------------------------------------------===//
/// CompileUnit - This dwarf writer support class manages information associate
/// with a source file.
class CompileUnit {
private:
  /// Desc - Compile unit debug descriptor.
  ///
  CompileUnitDesc *Desc;
  
  /// ID - File identifier for source.
  ///
  unsigned ID;
  
  /// Die - Compile unit debug information entry.
  ///
  DIE *Die;
  
  /// DescToDieMap - Tracks the mapping of unit level debug informaton
  /// descriptors to debug information entries.
  std::map<DebugInfoDesc *, DIE *> DescToDieMap;

  /// DescToDIEntryMap - Tracks the mapping of unit level debug informaton
  /// descriptors to debug information entries using a DIEntry proxy.
  std::map<DebugInfoDesc *, DIEntry *> DescToDIEntryMap;

  /// Globals - A map of globally visible named entities for this unit.
  ///
  std::map<std::string, DIE *> Globals;

  /// DiesSet - Used to uniquely define dies within the compile unit.
  ///
  FoldingSet<DIE> DiesSet;
  
  /// Dies - List of all dies in the compile unit.
  ///
  std::vector<DIE *> Dies;
  
public:
  CompileUnit(CompileUnitDesc *CUD, unsigned I, DIE *D)
  : Desc(CUD)
  , ID(I)
  , Die(D)
  , DescToDieMap()
  , DescToDIEntryMap()
  , Globals()
  , DiesSet(InitDiesSetSize)
  , Dies()
  {}
  
  ~CompileUnit() {
    delete Die;
    
    for (unsigned i = 0, N = Dies.size(); i < N; ++i)
      delete Dies[i];
  }
  
  // Accessors.
  CompileUnitDesc *getDesc() const { return Desc; }
  unsigned getID()           const { return ID; }
  DIE* getDie()              const { return Die; }
  std::map<std::string, DIE *> &getGlobals() { return Globals; }

  /// hasContent - Return true if this compile unit has something to write out.
  ///
  bool hasContent() const {
    return !Die->getChildren().empty();
  }

  /// AddGlobal - Add a new global entity to the compile unit.
  ///
  void AddGlobal(const std::string &Name, DIE *Die) {
    Globals[Name] = Die;
  }
  
  /// getDieMapSlotFor - Returns the debug information entry map slot for the
  /// specified debug descriptor.
  DIE *&getDieMapSlotFor(DebugInfoDesc *DID) {
    return DescToDieMap[DID];
  }
  
  /// getDIEntrySlotFor - Returns the debug information entry proxy slot for the
  /// specified debug descriptor.
  DIEntry *&getDIEntrySlotFor(DebugInfoDesc *DID) {
    return DescToDIEntryMap[DID];
  }
  
  /// AddDie - Adds or interns the DIE to the compile unit.
  ///
  DIE *AddDie(DIE &Buffer) {
    FoldingSetNodeID ID;
    Buffer.Profile(ID);
    void *Where;
    DIE *Die = DiesSet.FindNodeOrInsertPos(ID, Where);
    
    if (!Die) {
      Die = new DIE(Buffer);
      DiesSet.InsertNode(Die, Where);
      this->Die->AddChild(Die);
      Buffer.Detach();
    }
    
    return Die;
  }
};

//===----------------------------------------------------------------------===//
/// Dwarf - Emits general Dwarf directives. 
///
class Dwarf {

protected:

  //===--------------------------------------------------------------------===//
  // Core attributes used by the Dwarf writer.
  //
  
  //
  /// O - Stream to .s file.
  ///
  std::ostream &O;

  /// Asm - Target of Dwarf emission.
  ///
  AsmPrinter *Asm;
  
  /// TAI - Target Asm Printer.
  const TargetAsmInfo *TAI;
  
  /// TD - Target data.
  const TargetData *TD;
  
  /// RI - Register Information.
  const TargetRegisterInfo *RI;
  
  /// M - Current module.
  ///
  Module *M;
  
  /// MF - Current machine function.
  ///
  MachineFunction *MF;
  
  /// MMI - Collected machine module information.
  ///
  MachineModuleInfo *MMI;
  
  /// SubprogramCount - The running count of functions being compiled.
  ///
  unsigned SubprogramCount;
  
  /// Flavor - A unique string indicating what dwarf producer this is, used to
  /// unique labels.
  const char * const Flavor;

  unsigned SetCounter;
  Dwarf(std::ostream &OS, AsmPrinter *A, const TargetAsmInfo *T,
        const char *flavor)
  : O(OS)
  , Asm(A)
  , TAI(T)
  , TD(Asm->TM.getTargetData())
  , RI(Asm->TM.getRegisterInfo())
  , M(NULL)
  , MF(NULL)
  , MMI(NULL)
  , SubprogramCount(0)
  , Flavor(flavor)
  , SetCounter(1)
  {
  }

public:

  //===--------------------------------------------------------------------===//
  // Accessors.
  //
  AsmPrinter *getAsm() const { return Asm; }
  MachineModuleInfo *getMMI() const { return MMI; }
  const TargetAsmInfo *getTargetAsmInfo() const { return TAI; }
  const TargetData *getTargetData() const { return TD; }

  void PrintRelDirective(bool Force32Bit = false, bool isInSection = false)
                                                                         const {
    if (isInSection && TAI->getDwarfSectionOffsetDirective())
      O << TAI->getDwarfSectionOffsetDirective();
    else if (Force32Bit || TD->getPointerSize() == sizeof(int32_t))
      O << TAI->getData32bitsDirective();
    else
      O << TAI->getData64bitsDirective();
  }
  
  /// PrintLabelName - Print label name in form used by Dwarf writer.
  ///
  void PrintLabelName(DWLabel Label) const {
    PrintLabelName(Label.Tag, Label.Number);
  }
  void PrintLabelName(const char *Tag, unsigned Number) const {
    O << TAI->getPrivateGlobalPrefix() << Tag;
    if (Number) O << Number;
  }
  
  void PrintLabelName(const char *Tag, unsigned Number,
                      const char *Suffix) const {
    O << TAI->getPrivateGlobalPrefix() << Tag;
    if (Number) O << Number;
    O << Suffix;
  }
  
  /// EmitLabel - Emit location label for internal use by Dwarf.
  ///
  void EmitLabel(DWLabel Label) const {
    EmitLabel(Label.Tag, Label.Number);
  }
  void EmitLabel(const char *Tag, unsigned Number) const {
    PrintLabelName(Tag, Number);
    O << ":\n";
  }
  
  /// EmitReference - Emit a reference to a label.
  ///
  void EmitReference(DWLabel Label, bool IsPCRelative = false,
                     bool Force32Bit = false) const {
    EmitReference(Label.Tag, Label.Number, IsPCRelative, Force32Bit);
  }
  void EmitReference(const char *Tag, unsigned Number,
                     bool IsPCRelative = false, bool Force32Bit = false) const {
    PrintRelDirective(Force32Bit);
    PrintLabelName(Tag, Number);
    
    if (IsPCRelative) O << "-" << TAI->getPCSymbol();
  }
  void EmitReference(const std::string &Name, bool IsPCRelative = false,
                     bool Force32Bit = false) const {
    PrintRelDirective(Force32Bit);
    
    O << Name;
    
    if (IsPCRelative) O << "-" << TAI->getPCSymbol();
  }

  /// EmitDifference - Emit the difference between two labels.  Some
  /// assemblers do not behave with absolute expressions with data directives,
  /// so there is an option (needsSet) to use an intermediary set expression.
  void EmitDifference(DWLabel LabelHi, DWLabel LabelLo,
                      bool IsSmall = false) {
    EmitDifference(LabelHi.Tag, LabelHi.Number,
                   LabelLo.Tag, LabelLo.Number,
                   IsSmall);
  }
  void EmitDifference(const char *TagHi, unsigned NumberHi,
                      const char *TagLo, unsigned NumberLo,
                      bool IsSmall = false) {
    if (TAI->needsSet()) {
      O << "\t.set\t";
      PrintLabelName("set", SetCounter, Flavor);
      O << ",";
      PrintLabelName(TagHi, NumberHi);
      O << "-";
      PrintLabelName(TagLo, NumberLo);
      O << "\n";

      PrintRelDirective(IsSmall);
      PrintLabelName("set", SetCounter, Flavor);
      ++SetCounter;
    } else {
      PrintRelDirective(IsSmall);
        
      PrintLabelName(TagHi, NumberHi);
      O << "-";
      PrintLabelName(TagLo, NumberLo);
    }
  }

  void EmitSectionOffset(const char* Label, const char* Section,
                         unsigned LabelNumber, unsigned SectionNumber,
                         bool IsSmall = false, bool isEH = false) {
    bool printAbsolute = false;
    if (TAI->needsSet()) {
      O << "\t.set\t";
      PrintLabelName("set", SetCounter, Flavor);
      O << ",";
      PrintLabelName(Label, LabelNumber);

      if (isEH)
        printAbsolute = TAI->isAbsoluteEHSectionOffsets();
      else
        printAbsolute = TAI->isAbsoluteDebugSectionOffsets();
      
      if (!printAbsolute) {
        O << "-";
        PrintLabelName(Section, SectionNumber);
      }      
      O << "\n";

      PrintRelDirective(IsSmall);
        
      PrintLabelName("set", SetCounter, Flavor);
      ++SetCounter;
    } else {
      PrintRelDirective(IsSmall, true);
        
      PrintLabelName(Label, LabelNumber);

      if (isEH)
        printAbsolute = TAI->isAbsoluteEHSectionOffsets();
      else
        printAbsolute = TAI->isAbsoluteDebugSectionOffsets();

      if (!printAbsolute) {
        O << "-";
        PrintLabelName(Section, SectionNumber);
      }
    }    
  }
  
  /// EmitFrameMoves - Emit frame instructions to describe the layout of the
  /// frame.
  void EmitFrameMoves(const char *BaseLabel, unsigned BaseLabelID,
                      const std::vector<MachineMove> &Moves, bool isEH) {
    int stackGrowth =
        Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
          TargetFrameInfo::StackGrowsUp ?
            TD->getPointerSize() : -TD->getPointerSize();
    bool IsLocal = BaseLabel && strcmp(BaseLabel, "label") == 0;

    for (unsigned i = 0, N = Moves.size(); i < N; ++i) {
      const MachineMove &Move = Moves[i];
      unsigned LabelID = Move.getLabelID();
      
      if (LabelID) {
        LabelID = MMI->MappedLabel(LabelID);
      
        // Throw out move if the label is invalid.
        if (!LabelID) continue;
      }
      
      const MachineLocation &Dst = Move.getDestination();
      const MachineLocation &Src = Move.getSource();
      
      // Advance row if new location.
      if (BaseLabel && LabelID && (BaseLabelID != LabelID || !IsLocal)) {
        Asm->EmitInt8(DW_CFA_advance_loc4);
        Asm->EOL("DW_CFA_advance_loc4");
        EmitDifference("label", LabelID, BaseLabel, BaseLabelID, true);
        Asm->EOL();
        
        BaseLabelID = LabelID;
        BaseLabel = "label";
        IsLocal = true;
      }
      
      // If advancing cfa.
      if (Dst.isRegister() && Dst.getRegister() == MachineLocation::VirtualFP) {
        if (!Src.isRegister()) {
          if (Src.getRegister() == MachineLocation::VirtualFP) {
            Asm->EmitInt8(DW_CFA_def_cfa_offset);
            Asm->EOL("DW_CFA_def_cfa_offset");
          } else {
            Asm->EmitInt8(DW_CFA_def_cfa);
            Asm->EOL("DW_CFA_def_cfa");
            Asm->EmitULEB128Bytes(RI->getDwarfRegNum(Src.getRegister(), isEH));
            Asm->EOL("Register");
          }
          
          int Offset = -Src.getOffset();
          
          Asm->EmitULEB128Bytes(Offset);
          Asm->EOL("Offset");
        } else {
          assert(0 && "Machine move no supported yet.");
        }
      } else if (Src.isRegister() &&
        Src.getRegister() == MachineLocation::VirtualFP) {
        if (Dst.isRegister()) {
          Asm->EmitInt8(DW_CFA_def_cfa_register);
          Asm->EOL("DW_CFA_def_cfa_register");
          Asm->EmitULEB128Bytes(RI->getDwarfRegNum(Dst.getRegister(), isEH));
          Asm->EOL("Register");
        } else {
          assert(0 && "Machine move no supported yet.");
        }
      } else {
        unsigned Reg = RI->getDwarfRegNum(Src.getRegister(), isEH);
        int Offset = Dst.getOffset() / stackGrowth;
        
        if (Offset < 0) {
          Asm->EmitInt8(DW_CFA_offset_extended_sf);
          Asm->EOL("DW_CFA_offset_extended_sf");
          Asm->EmitULEB128Bytes(Reg);
          Asm->EOL("Reg");
          Asm->EmitSLEB128Bytes(Offset);
          Asm->EOL("Offset");
        } else if (Reg < 64) {
          Asm->EmitInt8(DW_CFA_offset + Reg);
          Asm->EOL("DW_CFA_offset + Reg (" + utostr(Reg) + ")");
          Asm->EmitULEB128Bytes(Offset);
          Asm->EOL("Offset");
        } else {
          Asm->EmitInt8(DW_CFA_offset_extended);
          Asm->EOL("DW_CFA_offset_extended");
          Asm->EmitULEB128Bytes(Reg);
          Asm->EOL("Reg");
          Asm->EmitULEB128Bytes(Offset);
          Asm->EOL("Offset");
        }
      }
    }
  }

};

//===----------------------------------------------------------------------===//
/// DwarfDebug - Emits Dwarf debug directives. 
///
class DwarfDebug : public Dwarf {

private:
  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //
  
  /// CompileUnits - All the compile units involved in this build.  The index
  /// of each entry in this vector corresponds to the sources in MMI.
  std::vector<CompileUnit *> CompileUnits;
  
  /// AbbreviationsSet - Used to uniquely define abbreviations.
  ///
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  /// Abbreviations - A list of all the unique abbreviations in use.
  ///
  std::vector<DIEAbbrev *> Abbreviations;
  
  /// ValuesSet - Used to uniquely define values.
  ///
  FoldingSet<DIEValue> ValuesSet;
  
  /// Values - A list of all the unique values in use.
  ///
  std::vector<DIEValue *> Values;
  
  /// StringPool - A UniqueVector of strings used by indirect references.
  ///
  UniqueVector<std::string> StringPool;

  /// UnitMap - Map debug information descriptor to compile unit.
  ///
  std::map<DebugInfoDesc *, CompileUnit *> DescToUnitMap;
  
  /// SectionMap - Provides a unique id per text section.
  ///
  UniqueVector<std::string> SectionMap;
  
  /// SectionSourceLines - Tracks line numbers per text section.
  ///
  std::vector<std::vector<SourceLineInfo> > SectionSourceLines;

  /// didInitial - Flag to indicate if initial emission has been done.
  ///
  bool didInitial;
  
  /// shouldEmit - Flag to indicate if debug information should be emitted.
  ///
  bool shouldEmit;

  struct FunctionDebugFrameInfo {
    unsigned Number;
    std::vector<MachineMove> Moves;

    FunctionDebugFrameInfo(unsigned Num, const std::vector<MachineMove> &M):
      Number(Num), Moves(M) { }
  };

  std::vector<FunctionDebugFrameInfo> DebugFrames;
  
public:
  
  /// ShouldEmitDwarf - Returns true if Dwarf declarations should be made.
  ///
  bool ShouldEmitDwarf() const { return shouldEmit; }

  /// AssignAbbrevNumber - Define a unique number for the abbreviation.
  ///  
  void AssignAbbrevNumber(DIEAbbrev &Abbrev) {
    // Profile the node so that we can make it unique.
    FoldingSetNodeID ID;
    Abbrev.Profile(ID);
    
    // Check the set for priors.
    DIEAbbrev *InSet = AbbreviationsSet.GetOrInsertNode(&Abbrev);
    
    // If it's newly added.
    if (InSet == &Abbrev) {
      // Add to abbreviation list. 
      Abbreviations.push_back(&Abbrev);
      // Assign the vector position + 1 as its number.
      Abbrev.setNumber(Abbreviations.size());
    } else {
      // Assign existing abbreviation number.
      Abbrev.setNumber(InSet->getNumber());
    }
  }

  /// NewString - Add a string to the constant pool and returns a label.
  ///
  DWLabel NewString(const std::string &String) {
    unsigned StringID = StringPool.insert(String);
    return DWLabel("string", StringID);
  }
  
  /// NewDIEntry - Creates a new DIEntry to be a proxy for a debug information
  /// entry.
  DIEntry *NewDIEntry(DIE *Entry = NULL) {
    DIEntry *Value;
    
    if (Entry) {
      FoldingSetNodeID ID;
      DIEntry::Profile(ID, Entry);
      void *Where;
      Value = static_cast<DIEntry *>(ValuesSet.FindNodeOrInsertPos(ID, Where));
      
      if (Value) return Value;
      
      Value = new DIEntry(Entry);
      ValuesSet.InsertNode(Value, Where);
    } else {
      Value = new DIEntry(Entry);
    }
    
    Values.push_back(Value);
    return Value;
  }
  
  /// SetDIEntry - Set a DIEntry once the debug information entry is defined.
  ///
  void SetDIEntry(DIEntry *Value, DIE *Entry) {
    Value->Entry = Entry;
    // Add to values set if not already there.  If it is, we merely have a
    // duplicate in the values list (no harm.)
    ValuesSet.GetOrInsertNode(Value);
  }

  /// AddUInt - Add an unsigned integer attribute data and value.
  ///
  void AddUInt(DIE *Die, unsigned Attribute, unsigned Form, uint64_t Integer) {
    if (!Form) Form = DIEInteger::BestForm(false, Integer);

    FoldingSetNodeID ID;
    DIEInteger::Profile(ID, Integer);
    void *Where;
    DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);
    if (!Value) {
      Value = new DIEInteger(Integer);
      ValuesSet.InsertNode(Value, Where);
      Values.push_back(Value);
    }
  
    Die->AddValue(Attribute, Form, Value);
  }
      
  /// AddSInt - Add an signed integer attribute data and value.
  ///
  void AddSInt(DIE *Die, unsigned Attribute, unsigned Form, int64_t Integer) {
    if (!Form) Form = DIEInteger::BestForm(true, Integer);

    FoldingSetNodeID ID;
    DIEInteger::Profile(ID, (uint64_t)Integer);
    void *Where;
    DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);
    if (!Value) {
      Value = new DIEInteger(Integer);
      ValuesSet.InsertNode(Value, Where);
      Values.push_back(Value);
    }
  
    Die->AddValue(Attribute, Form, Value);
  }
      
  /// AddString - Add a std::string attribute data and value.
  ///
  void AddString(DIE *Die, unsigned Attribute, unsigned Form,
                 const std::string &String) {
    FoldingSetNodeID ID;
    DIEString::Profile(ID, String);
    void *Where;
    DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);
    if (!Value) {
      Value = new DIEString(String);
      ValuesSet.InsertNode(Value, Where);
      Values.push_back(Value);
    }
  
    Die->AddValue(Attribute, Form, Value);
  }
      
  /// AddLabel - Add a Dwarf label attribute data and value.
  ///
  void AddLabel(DIE *Die, unsigned Attribute, unsigned Form,
                     const DWLabel &Label) {
    FoldingSetNodeID ID;
    DIEDwarfLabel::Profile(ID, Label);
    void *Where;
    DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);
    if (!Value) {
      Value = new DIEDwarfLabel(Label);
      ValuesSet.InsertNode(Value, Where);
      Values.push_back(Value);
    }
  
    Die->AddValue(Attribute, Form, Value);
  }
      
  /// AddObjectLabel - Add an non-Dwarf label attribute data and value.
  ///
  void AddObjectLabel(DIE *Die, unsigned Attribute, unsigned Form,
                      const std::string &Label) {
    FoldingSetNodeID ID;
    DIEObjectLabel::Profile(ID, Label);
    void *Where;
    DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);
    if (!Value) {
      Value = new DIEObjectLabel(Label);
      ValuesSet.InsertNode(Value, Where);
      Values.push_back(Value);
    }
  
    Die->AddValue(Attribute, Form, Value);
  }
      
  /// AddDelta - Add a label delta attribute data and value.
  ///
  void AddDelta(DIE *Die, unsigned Attribute, unsigned Form,
                          const DWLabel &Hi, const DWLabel &Lo) {
    FoldingSetNodeID ID;
    DIEDelta::Profile(ID, Hi, Lo);
    void *Where;
    DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);
    if (!Value) {
      Value = new DIEDelta(Hi, Lo);
      ValuesSet.InsertNode(Value, Where);
      Values.push_back(Value);
    }
  
    Die->AddValue(Attribute, Form, Value);
  }
      
  /// AddDIEntry - Add a DIE attribute data and value.
  ///
  void AddDIEntry(DIE *Die, unsigned Attribute, unsigned Form, DIE *Entry) {
    Die->AddValue(Attribute, Form, NewDIEntry(Entry));
  }

  /// AddBlock - Add block data.
  ///
  void AddBlock(DIE *Die, unsigned Attribute, unsigned Form, DIEBlock *Block) {
    Block->ComputeSize(*this);
    FoldingSetNodeID ID;
    Block->Profile(ID);
    void *Where;
    DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);
    if (!Value) {
      Value = Block;
      ValuesSet.InsertNode(Value, Where);
      Values.push_back(Value);
    } else {
      // Already exists, reuse the previous one.
      delete Block;
      Block = cast<DIEBlock>(Value);
    }
  
    Die->AddValue(Attribute, Block->BestForm(), Value);
  }

private:

  /// AddSourceLine - Add location information to specified debug information
  /// entry.
  void AddSourceLine(DIE *Die, CompileUnitDesc *File, unsigned Line) {
    if (File && Line) {
      CompileUnit *FileUnit = FindCompileUnit(File);
      unsigned FileID = FileUnit->getID();
      AddUInt(Die, DW_AT_decl_file, 0, FileID);
      AddUInt(Die, DW_AT_decl_line, 0, Line);
    }
  }

  /// AddAddress - Add an address attribute to a die based on the location
  /// provided.
  void AddAddress(DIE *Die, unsigned Attribute,
                            const MachineLocation &Location) {
    unsigned Reg = RI->getDwarfRegNum(Location.getRegister(), false);
    DIEBlock *Block = new DIEBlock();
    
    if (Location.isRegister()) {
      if (Reg < 32) {
        AddUInt(Block, 0, DW_FORM_data1, DW_OP_reg0 + Reg);
      } else {
        AddUInt(Block, 0, DW_FORM_data1, DW_OP_regx);
        AddUInt(Block, 0, DW_FORM_udata, Reg);
      }
    } else {
      if (Reg < 32) {
        AddUInt(Block, 0, DW_FORM_data1, DW_OP_breg0 + Reg);
      } else {
        AddUInt(Block, 0, DW_FORM_data1, DW_OP_bregx);
        AddUInt(Block, 0, DW_FORM_udata, Reg);
      }
      AddUInt(Block, 0, DW_FORM_sdata, Location.getOffset());
    }
    
    AddBlock(Die, Attribute, 0, Block);
  }
  
  /// AddBasicType - Add a new basic type attribute to the specified entity.
  ///
  void AddBasicType(DIE *Entity, CompileUnit *Unit,
                    const std::string &Name,
                    unsigned Encoding, unsigned Size) {
    DIE *Die = ConstructBasicType(Unit, Name, Encoding, Size);
    AddDIEntry(Entity, DW_AT_type, DW_FORM_ref4, Die);
  }
  
  /// ConstructBasicType - Construct a new basic type.
  ///
  DIE *ConstructBasicType(CompileUnit *Unit,
                          const std::string &Name,
                          unsigned Encoding, unsigned Size) {
    DIE Buffer(DW_TAG_base_type);
    AddUInt(&Buffer, DW_AT_byte_size, 0, Size);
    AddUInt(&Buffer, DW_AT_encoding, DW_FORM_data1, Encoding);
    if (!Name.empty()) AddString(&Buffer, DW_AT_name, DW_FORM_string, Name);
    return Unit->AddDie(Buffer);
  }
  
  /// AddPointerType - Add a new pointer type attribute to the specified entity.
  ///
  void AddPointerType(DIE *Entity, CompileUnit *Unit, const std::string &Name) {
    DIE *Die = ConstructPointerType(Unit, Name);
    AddDIEntry(Entity, DW_AT_type, DW_FORM_ref4, Die);
  }
  
  /// ConstructPointerType - Construct a new pointer type.
  ///
  DIE *ConstructPointerType(CompileUnit *Unit, const std::string &Name) {
    DIE Buffer(DW_TAG_pointer_type);
    AddUInt(&Buffer, DW_AT_byte_size, 0, TD->getPointerSize());
    if (!Name.empty()) AddString(&Buffer, DW_AT_name, DW_FORM_string, Name);
    return Unit->AddDie(Buffer);
  }
  
  /// AddType - Add a new type attribute to the specified entity.
  ///
  void AddType(DIE *Entity, TypeDesc *TyDesc, CompileUnit *Unit) {
    if (!TyDesc) {
      AddBasicType(Entity, Unit, "", DW_ATE_signed, sizeof(int32_t));
    } else {
      // Check for pre-existence.
      DIEntry *&Slot = Unit->getDIEntrySlotFor(TyDesc);
      
      // If it exists then use the existing value.
      if (Slot) {
        Entity->AddValue(DW_AT_type, DW_FORM_ref4, Slot);
        return;
      }
      
      if (SubprogramDesc *SubprogramTy = dyn_cast<SubprogramDesc>(TyDesc)) {
        // FIXME - Not sure why programs and variables are coming through here.
        // Short cut for handling subprogram types (not really a TyDesc.)
        AddPointerType(Entity, Unit, SubprogramTy->getName());
      } else if (GlobalVariableDesc *GlobalTy =
                                         dyn_cast<GlobalVariableDesc>(TyDesc)) {
        // FIXME - Not sure why programs and variables are coming through here.
        // Short cut for handling global variable types (not really a TyDesc.)
        AddPointerType(Entity, Unit, GlobalTy->getName());
      } else {  
        // Set up proxy.
        Slot = NewDIEntry();
        
        // Construct type.
        DIE Buffer(DW_TAG_base_type);
        ConstructType(Buffer, TyDesc, Unit);
        
        // Add debug information entry to entity and unit.
        DIE *Die = Unit->AddDie(Buffer);
        SetDIEntry(Slot, Die);
        Entity->AddValue(DW_AT_type, DW_FORM_ref4, Slot);
      }
    }
  }
  
  /// ConstructType - Adds all the required attributes to the type.
  ///
  void ConstructType(DIE &Buffer, TypeDesc *TyDesc, CompileUnit *Unit) {
    // Get core information.
    const std::string &Name = TyDesc->getName();
    uint64_t Size = TyDesc->getSize() >> 3;
    
    if (BasicTypeDesc *BasicTy = dyn_cast<BasicTypeDesc>(TyDesc)) {
      // Fundamental types like int, float, bool
      Buffer.setTag(DW_TAG_base_type);
      AddUInt(&Buffer, DW_AT_encoding,  DW_FORM_data1, BasicTy->getEncoding());
    } else if (DerivedTypeDesc *DerivedTy = dyn_cast<DerivedTypeDesc>(TyDesc)) {
      // Fetch tag.
      unsigned Tag = DerivedTy->getTag();
      // FIXME - Workaround for templates.
      if (Tag == DW_TAG_inheritance) Tag = DW_TAG_reference_type;
      // Pointers, typedefs et al. 
      Buffer.setTag(Tag);
      // Map to main type, void will not have a type.
      if (TypeDesc *FromTy = DerivedTy->getFromType())
        AddType(&Buffer, FromTy, Unit);
    } else if (CompositeTypeDesc *CompTy = dyn_cast<CompositeTypeDesc>(TyDesc)){
      // Fetch tag.
      unsigned Tag = CompTy->getTag();
      
      // Set tag accordingly.
      if (Tag == DW_TAG_vector_type)
        Buffer.setTag(DW_TAG_array_type);
      else 
        Buffer.setTag(Tag);

      std::vector<DebugInfoDesc *> &Elements = CompTy->getElements();
      
      switch (Tag) {
      case DW_TAG_vector_type:
        AddUInt(&Buffer, DW_AT_GNU_vector, DW_FORM_flag, 1);
        // Fall thru
      case DW_TAG_array_type: {
        // Add element type.
        if (TypeDesc *FromTy = CompTy->getFromType())
          AddType(&Buffer, FromTy, Unit);
        
        // Don't emit size attribute.
        Size = 0;
        
        // Construct an anonymous type for index type.
        DIE *IndexTy = ConstructBasicType(Unit, "", DW_ATE_signed,
                                          sizeof(int32_t));
      
        // Add subranges to array type.
        for(unsigned i = 0, N = Elements.size(); i < N; ++i) {
          SubrangeDesc *SRD = cast<SubrangeDesc>(Elements[i]);
          int64_t Lo = SRD->getLo();
          int64_t Hi = SRD->getHi();
          DIE *Subrange = new DIE(DW_TAG_subrange_type);
          
          // If a range is available.
          if (Lo != Hi) {
            AddDIEntry(Subrange, DW_AT_type, DW_FORM_ref4, IndexTy);
            // Only add low if non-zero.
            if (Lo) AddSInt(Subrange, DW_AT_lower_bound, 0, Lo);
            AddSInt(Subrange, DW_AT_upper_bound, 0, Hi);
          }
          
          Buffer.AddChild(Subrange);
        }
        break;
      }
      case DW_TAG_structure_type:
      case DW_TAG_union_type: {
        // Add elements to structure type.
        for(unsigned i = 0, N = Elements.size(); i < N; ++i) {
          DebugInfoDesc *Element = Elements[i];
          
          if (DerivedTypeDesc *MemberDesc = dyn_cast<DerivedTypeDesc>(Element)){
            // Add field or base class.
            
            unsigned Tag = MemberDesc->getTag();
          
            // Extract the basic information.
            const std::string &Name = MemberDesc->getName();
            uint64_t Size = MemberDesc->getSize();
            uint64_t Align = MemberDesc->getAlign();
            uint64_t Offset = MemberDesc->getOffset();
       
            // Construct member debug information entry.
            DIE *Member = new DIE(Tag);
            
            // Add name if not "".
            if (!Name.empty())
              AddString(Member, DW_AT_name, DW_FORM_string, Name);
            // Add location if available.
            AddSourceLine(Member, MemberDesc->getFile(), MemberDesc->getLine());
            
            // Most of the time the field info is the same as the members.
            uint64_t FieldSize = Size;
            uint64_t FieldAlign = Align;
            uint64_t FieldOffset = Offset;
            
            // Set the member type.
            TypeDesc *FromTy = MemberDesc->getFromType();
            AddType(Member, FromTy, Unit);
            
            // Walk up typedefs until a real size is found.
            while (FromTy) {
              if (FromTy->getTag() != DW_TAG_typedef) {
                FieldSize = FromTy->getSize();
                FieldAlign = FromTy->getSize();
                break;
              }
              
              FromTy = cast<DerivedTypeDesc>(FromTy)->getFromType();
            }
            
            // Unless we have a bit field.
            if (Tag == DW_TAG_member && FieldSize != Size) {
              // Construct the alignment mask.
              uint64_t AlignMask = ~(FieldAlign - 1);
              // Determine the high bit + 1 of the declared size.
              uint64_t HiMark = (Offset + FieldSize) & AlignMask;
              // Work backwards to determine the base offset of the field.
              FieldOffset = HiMark - FieldSize;
              // Now normalize offset to the field.
              Offset -= FieldOffset;
              
              // Maybe we need to work from the other end.
              if (TD->isLittleEndian()) Offset = FieldSize - (Offset + Size);
              
              // Add size and offset.
              AddUInt(Member, DW_AT_byte_size, 0, FieldSize >> 3);
              AddUInt(Member, DW_AT_bit_size, 0, Size);
              AddUInt(Member, DW_AT_bit_offset, 0, Offset);
            }
            
            // Add computation for offset.
            DIEBlock *Block = new DIEBlock();
            AddUInt(Block, 0, DW_FORM_data1, DW_OP_plus_uconst);
            AddUInt(Block, 0, DW_FORM_udata, FieldOffset >> 3);
            AddBlock(Member, DW_AT_data_member_location, 0, Block);

            // Add accessibility (public default unless is base class.
            if (MemberDesc->isProtected()) {
              AddUInt(Member, DW_AT_accessibility, 0, DW_ACCESS_protected);
            } else if (MemberDesc->isPrivate()) {
              AddUInt(Member, DW_AT_accessibility, 0, DW_ACCESS_private);
            } else if (Tag == DW_TAG_inheritance) {
              AddUInt(Member, DW_AT_accessibility, 0, DW_ACCESS_public);
            }
            
            Buffer.AddChild(Member);
          } else if (GlobalVariableDesc *StaticDesc =
                                        dyn_cast<GlobalVariableDesc>(Element)) {
            // Add static member.
            
            // Construct member debug information entry.
            DIE *Static = new DIE(DW_TAG_variable);
            
            // Add name and mangled name.
            const std::string &Name = StaticDesc->getName();
            const std::string &LinkageName = StaticDesc->getLinkageName();
            AddString(Static, DW_AT_name, DW_FORM_string, Name);
            if (!LinkageName.empty()) {
              AddString(Static, DW_AT_MIPS_linkage_name, DW_FORM_string,
                                LinkageName);
            }
            
            // Add location.
            AddSourceLine(Static, StaticDesc->getFile(), StaticDesc->getLine());
           
            // Add type.
            if (TypeDesc *StaticTy = StaticDesc->getType())
              AddType(Static, StaticTy, Unit);
            
            // Add flags.
            if (!StaticDesc->isStatic())
              AddUInt(Static, DW_AT_external, DW_FORM_flag, 1);
            AddUInt(Static, DW_AT_declaration, DW_FORM_flag, 1);
            
            Buffer.AddChild(Static);
          } else if (SubprogramDesc *MethodDesc =
                                            dyn_cast<SubprogramDesc>(Element)) {
            // Add member function.
            
            // Construct member debug information entry.
            DIE *Method = new DIE(DW_TAG_subprogram);
           
            // Add name and mangled name.
            const std::string &Name = MethodDesc->getName();
            const std::string &LinkageName = MethodDesc->getLinkageName();
            
            AddString(Method, DW_AT_name, DW_FORM_string, Name);            
            bool IsCTor = TyDesc->getName() == Name;
            
            if (!LinkageName.empty()) {
              AddString(Method, DW_AT_MIPS_linkage_name, DW_FORM_string,
                                LinkageName);
            }
            
            // Add location.
            AddSourceLine(Method, MethodDesc->getFile(), MethodDesc->getLine());
           
            // Add type.
            if (CompositeTypeDesc *MethodTy =
                   dyn_cast_or_null<CompositeTypeDesc>(MethodDesc->getType())) {
              // Get argument information.
              std::vector<DebugInfoDesc *> &Args = MethodTy->getElements();
             
              // If not a ctor.
              if (!IsCTor) {
                // Add return type.
                AddType(Method, dyn_cast<TypeDesc>(Args[0]), Unit);
              }
              
              // Add arguments.
              for(unsigned i = 1, N = Args.size(); i < N; ++i) {
                DIE *Arg = new DIE(DW_TAG_formal_parameter);
                AddType(Arg, cast<TypeDesc>(Args[i]), Unit);
                AddUInt(Arg, DW_AT_artificial, DW_FORM_flag, 1);
                Method->AddChild(Arg);
              }
            }

            // Add flags.
            if (!MethodDesc->isStatic())
              AddUInt(Method, DW_AT_external, DW_FORM_flag, 1);
            AddUInt(Method, DW_AT_declaration, DW_FORM_flag, 1);
              
            Buffer.AddChild(Method);
          }
        }
        break;
      }
      case DW_TAG_enumeration_type: {
        // Add enumerators to enumeration type.
        for(unsigned i = 0, N = Elements.size(); i < N; ++i) {
          EnumeratorDesc *ED = cast<EnumeratorDesc>(Elements[i]);
          const std::string &Name = ED->getName();
          int64_t Value = ED->getValue();
          DIE *Enumerator = new DIE(DW_TAG_enumerator);
          AddString(Enumerator, DW_AT_name, DW_FORM_string, Name);
          AddSInt(Enumerator, DW_AT_const_value, DW_FORM_sdata, Value);
          Buffer.AddChild(Enumerator);
        }

        break;
      }
      case DW_TAG_subroutine_type: {
        // Add prototype flag.
        AddUInt(&Buffer, DW_AT_prototyped, DW_FORM_flag, 1);
        // Add return type.
        AddType(&Buffer, dyn_cast<TypeDesc>(Elements[0]), Unit);
        
        // Add arguments.
        for(unsigned i = 1, N = Elements.size(); i < N; ++i) {
          DIE *Arg = new DIE(DW_TAG_formal_parameter);
          AddType(Arg, cast<TypeDesc>(Elements[i]), Unit);
          Buffer.AddChild(Arg);
        }
        
        break;
      }
      default: break;
      }
    }
   
    // Add size if non-zero (derived types don't have a size.)
    if (Size) AddUInt(&Buffer, DW_AT_byte_size, 0, Size);
    // Add name if not anonymous or intermediate type.
    if (!Name.empty()) AddString(&Buffer, DW_AT_name, DW_FORM_string, Name);
    // Add source line info if available.
    AddSourceLine(&Buffer, TyDesc->getFile(), TyDesc->getLine());
  }

  /// NewCompileUnit - Create new compile unit and it's debug information entry.
  ///
  CompileUnit *NewCompileUnit(CompileUnitDesc *UnitDesc, unsigned ID) {
    // Construct debug information entry.
    DIE *Die = new DIE(DW_TAG_compile_unit);
    if (TAI->isAbsoluteDebugSectionOffsets())
      AddLabel(Die, DW_AT_stmt_list, DW_FORM_data4, DWLabel("section_line", 0));
    else
      AddDelta(Die, DW_AT_stmt_list, DW_FORM_data4, DWLabel("section_line", 0),
               DWLabel("section_line", 0));      
    AddString(Die, DW_AT_producer,  DW_FORM_string, UnitDesc->getProducer());
    AddUInt  (Die, DW_AT_language,  DW_FORM_data1,  UnitDesc->getLanguage());
    AddString(Die, DW_AT_name,      DW_FORM_string, UnitDesc->getFileName());
    AddString(Die, DW_AT_comp_dir,  DW_FORM_string, UnitDesc->getDirectory());
    
    // Construct compile unit.
    CompileUnit *Unit = new CompileUnit(UnitDesc, ID, Die);
    
    // Add Unit to compile unit map.
    DescToUnitMap[UnitDesc] = Unit;
    
    return Unit;
  }

  /// GetBaseCompileUnit - Get the main compile unit.
  ///
  CompileUnit *GetBaseCompileUnit() const {
    CompileUnit *Unit = CompileUnits[0];
    assert(Unit && "Missing compile unit.");
    return Unit;
  }

  /// FindCompileUnit - Get the compile unit for the given descriptor.
  ///
  CompileUnit *FindCompileUnit(CompileUnitDesc *UnitDesc) {
    CompileUnit *Unit = DescToUnitMap[UnitDesc];
    assert(Unit && "Missing compile unit.");
    return Unit;
  }

  /// NewGlobalVariable - Add a new global variable DIE.
  ///
  DIE *NewGlobalVariable(GlobalVariableDesc *GVD) {
    // Get the compile unit context.
    CompileUnitDesc *UnitDesc =
      static_cast<CompileUnitDesc *>(GVD->getContext());
    CompileUnit *Unit = GetBaseCompileUnit();

    // Check for pre-existence.
    DIE *&Slot = Unit->getDieMapSlotFor(GVD);
    if (Slot) return Slot;
    
    // Get the global variable itself.
    GlobalVariable *GV = GVD->getGlobalVariable();

    const std::string &Name = GVD->getName();
    const std::string &FullName = GVD->getFullName();
    const std::string &LinkageName = GVD->getLinkageName();
    // Create the global's variable DIE.
    DIE *VariableDie = new DIE(DW_TAG_variable);
    AddString(VariableDie, DW_AT_name, DW_FORM_string, Name);
    if (!LinkageName.empty()) {
      AddString(VariableDie, DW_AT_MIPS_linkage_name, DW_FORM_string,
                             LinkageName);
    }
    AddType(VariableDie, GVD->getType(), Unit);
    if (!GVD->isStatic())
      AddUInt(VariableDie, DW_AT_external, DW_FORM_flag, 1);
    
    // Add source line info if available.
    AddSourceLine(VariableDie, UnitDesc, GVD->getLine());
    
    // Add address.
    DIEBlock *Block = new DIEBlock();
    AddUInt(Block, 0, DW_FORM_data1, DW_OP_addr);
    AddObjectLabel(Block, 0, DW_FORM_udata, Asm->getGlobalLinkName(GV));
    AddBlock(VariableDie, DW_AT_location, 0, Block);
    
    // Add to map.
    Slot = VariableDie;
   
    // Add to context owner.
    Unit->getDie()->AddChild(VariableDie);
    
    // Expose as global.
    // FIXME - need to check external flag.
    Unit->AddGlobal(FullName, VariableDie);
    
    return VariableDie;
  }

  /// NewSubprogram - Add a new subprogram DIE.
  ///
  DIE *NewSubprogram(SubprogramDesc *SPD) {
    // Get the compile unit context.
    CompileUnitDesc *UnitDesc =
      static_cast<CompileUnitDesc *>(SPD->getContext());
    CompileUnit *Unit = GetBaseCompileUnit();

    // Check for pre-existence.
    DIE *&Slot = Unit->getDieMapSlotFor(SPD);
    if (Slot) return Slot;
    
    // Gather the details (simplify add attribute code.)
    const std::string &Name = SPD->getName();
    const std::string &FullName = SPD->getFullName();
    const std::string &LinkageName = SPD->getLinkageName();
                                      
    DIE *SubprogramDie = new DIE(DW_TAG_subprogram);
    AddString(SubprogramDie, DW_AT_name, DW_FORM_string, Name);
    if (!LinkageName.empty()) {
      AddString(SubprogramDie, DW_AT_MIPS_linkage_name, DW_FORM_string,
                               LinkageName);
    }
    if (SPD->getType()) AddType(SubprogramDie, SPD->getType(), Unit);
    if (!SPD->isStatic())
      AddUInt(SubprogramDie, DW_AT_external, DW_FORM_flag, 1);
    AddUInt(SubprogramDie, DW_AT_prototyped, DW_FORM_flag, 1);
    
    // Add source line info if available.
    AddSourceLine(SubprogramDie, UnitDesc, SPD->getLine());

    // Add to map.
    Slot = SubprogramDie;
   
    // Add to context owner.
    Unit->getDie()->AddChild(SubprogramDie);
    
    // Expose as global.
    Unit->AddGlobal(FullName, SubprogramDie);
    
    return SubprogramDie;
  }

  /// NewScopeVariable - Create a new scope variable.
  ///
  DIE *NewScopeVariable(DebugVariable *DV, CompileUnit *Unit) {
    // Get the descriptor.
    VariableDesc *VD = DV->getDesc();

    // Translate tag to proper Dwarf tag.  The result variable is dropped for
    // now.
    unsigned Tag;
    switch (VD->getTag()) {
    case DW_TAG_return_variable:  return NULL;
    case DW_TAG_arg_variable:     Tag = DW_TAG_formal_parameter; break;
    case DW_TAG_auto_variable:    // fall thru
    default:                      Tag = DW_TAG_variable; break;
    }

    // Define variable debug information entry.
    DIE *VariableDie = new DIE(Tag);
    AddString(VariableDie, DW_AT_name, DW_FORM_string, VD->getName());

    // Add source line info if available.
    AddSourceLine(VariableDie, VD->getFile(), VD->getLine());
    
    // Add variable type.
    AddType(VariableDie, VD->getType(), Unit); 
    
    // Add variable address.
    MachineLocation Location;
    Location.set(RI->getFrameRegister(*MF),
                 RI->getFrameIndexOffset(*MF, DV->getFrameIndex()));
    AddAddress(VariableDie, DW_AT_location, Location);

    return VariableDie;
  }

  /// ConstructScope - Construct the components of a scope.
  ///
  void ConstructScope(DebugScope *ParentScope,
                      unsigned ParentStartID, unsigned ParentEndID,
                      DIE *ParentDie, CompileUnit *Unit) {
    // Add variables to scope.
    std::vector<DebugVariable *> &Variables = ParentScope->getVariables();
    for (unsigned i = 0, N = Variables.size(); i < N; ++i) {
      DIE *VariableDie = NewScopeVariable(Variables[i], Unit);
      if (VariableDie) ParentDie->AddChild(VariableDie);
    }
    
    // Add nested scopes.
    std::vector<DebugScope *> &Scopes = ParentScope->getScopes();
    for (unsigned j = 0, M = Scopes.size(); j < M; ++j) {
      // Define the Scope debug information entry.
      DebugScope *Scope = Scopes[j];
      // FIXME - Ignore inlined functions for the time being.
      if (!Scope->getParent()) continue;
      
      unsigned StartID = MMI->MappedLabel(Scope->getStartLabelID());
      unsigned EndID = MMI->MappedLabel(Scope->getEndLabelID());

      // Ignore empty scopes.
      if (StartID == EndID && StartID != 0) continue;
      if (Scope->getScopes().empty() && Scope->getVariables().empty()) continue;
      
      if (StartID == ParentStartID && EndID == ParentEndID) {
        // Just add stuff to the parent scope.
        ConstructScope(Scope, ParentStartID, ParentEndID, ParentDie, Unit);
      } else {
        DIE *ScopeDie = new DIE(DW_TAG_lexical_block);
        
        // Add the scope bounds.
        if (StartID) {
          AddLabel(ScopeDie, DW_AT_low_pc, DW_FORM_addr,
                             DWLabel("label", StartID));
        } else {
          AddLabel(ScopeDie, DW_AT_low_pc, DW_FORM_addr,
                             DWLabel("func_begin", SubprogramCount));
        }
        if (EndID) {
          AddLabel(ScopeDie, DW_AT_high_pc, DW_FORM_addr,
                             DWLabel("label", EndID));
        } else {
          AddLabel(ScopeDie, DW_AT_high_pc, DW_FORM_addr,
                             DWLabel("func_end", SubprogramCount));
        }
                           
        // Add the scope contents.
        ConstructScope(Scope, StartID, EndID, ScopeDie, Unit);
        ParentDie->AddChild(ScopeDie);
      }
    }
  }

  /// ConstructRootScope - Construct the scope for the subprogram.
  ///
  void ConstructRootScope(DebugScope *RootScope) {
    // Exit if there is no root scope.
    if (!RootScope) return;
    
    // Get the subprogram debug information entry. 
    SubprogramDesc *SPD = cast<SubprogramDesc>(RootScope->getDesc());
    
    // Get the compile unit context.
    CompileUnit *Unit = GetBaseCompileUnit();
    
    // Get the subprogram die.
    DIE *SPDie = Unit->getDieMapSlotFor(SPD);
    assert(SPDie && "Missing subprogram descriptor");
    
    // Add the function bounds.
    AddLabel(SPDie, DW_AT_low_pc, DW_FORM_addr,
                    DWLabel("func_begin", SubprogramCount));
    AddLabel(SPDie, DW_AT_high_pc, DW_FORM_addr,
                    DWLabel("func_end", SubprogramCount));
    MachineLocation Location(RI->getFrameRegister(*MF));
    AddAddress(SPDie, DW_AT_frame_base, Location);

    ConstructScope(RootScope, 0, 0, SPDie, Unit);
  }

  /// EmitInitial - Emit initial Dwarf declarations.  This is necessary for cc
  /// tools to recognize the object file contains Dwarf information.
  void EmitInitial() {
    // Check to see if we already emitted intial headers.
    if (didInitial) return;
    didInitial = true;
    
    // Dwarf sections base addresses.
    if (TAI->doesDwarfRequireFrameSection()) {
      Asm->SwitchToDataSection(TAI->getDwarfFrameSection());
      EmitLabel("section_debug_frame", 0);
    }
    Asm->SwitchToDataSection(TAI->getDwarfInfoSection());
    EmitLabel("section_info", 0);
    Asm->SwitchToDataSection(TAI->getDwarfAbbrevSection());
    EmitLabel("section_abbrev", 0);
    Asm->SwitchToDataSection(TAI->getDwarfARangesSection());
    EmitLabel("section_aranges", 0);
    Asm->SwitchToDataSection(TAI->getDwarfMacInfoSection());
    EmitLabel("section_macinfo", 0);
    Asm->SwitchToDataSection(TAI->getDwarfLineSection());
    EmitLabel("section_line", 0);
    Asm->SwitchToDataSection(TAI->getDwarfLocSection());
    EmitLabel("section_loc", 0);
    Asm->SwitchToDataSection(TAI->getDwarfPubNamesSection());
    EmitLabel("section_pubnames", 0);
    Asm->SwitchToDataSection(TAI->getDwarfStrSection());
    EmitLabel("section_str", 0);
    Asm->SwitchToDataSection(TAI->getDwarfRangesSection());
    EmitLabel("section_ranges", 0);

    Asm->SwitchToTextSection(TAI->getTextSection());
    EmitLabel("text_begin", 0);
    Asm->SwitchToDataSection(TAI->getDataSection());
    EmitLabel("data_begin", 0);
  }

  /// EmitDIE - Recusively Emits a debug information entry.
  ///
  void EmitDIE(DIE *Die) {
    // Get the abbreviation for this DIE.
    unsigned AbbrevNumber = Die->getAbbrevNumber();
    const DIEAbbrev *Abbrev = Abbreviations[AbbrevNumber - 1];
    
    Asm->EOL();

    // Emit the code (index) for the abbreviation.
    Asm->EmitULEB128Bytes(AbbrevNumber);
    Asm->EOL(std::string("Abbrev [" +
             utostr(AbbrevNumber) +
             "] 0x" + utohexstr(Die->getOffset()) +
             ":0x" + utohexstr(Die->getSize()) + " " +
             TagString(Abbrev->getTag())));
    
    std::vector<DIEValue *> &Values = Die->getValues();
    const std::vector<DIEAbbrevData> &AbbrevData = Abbrev->getData();
    
    // Emit the DIE attribute values.
    for (unsigned i = 0, N = Values.size(); i < N; ++i) {
      unsigned Attr = AbbrevData[i].getAttribute();
      unsigned Form = AbbrevData[i].getForm();
      assert(Form && "Too many attributes for DIE (check abbreviation)");
      
      switch (Attr) {
      case DW_AT_sibling: {
        Asm->EmitInt32(Die->SiblingOffset());
        break;
      }
      default: {
        // Emit an attribute using the defined form.
        Values[i]->EmitValue(*this, Form);
        break;
      }
      }
      
      Asm->EOL(AttributeString(Attr));
    }
    
    // Emit the DIE children if any.
    if (Abbrev->getChildrenFlag() == DW_CHILDREN_yes) {
      const std::vector<DIE *> &Children = Die->getChildren();
      
      for (unsigned j = 0, M = Children.size(); j < M; ++j) {
        EmitDIE(Children[j]);
      }
      
      Asm->EmitInt8(0); Asm->EOL("End Of Children Mark");
    }
  }

  /// SizeAndOffsetDie - Compute the size and offset of a DIE.
  ///
  unsigned SizeAndOffsetDie(DIE *Die, unsigned Offset, bool Last) {
    // Get the children.
    const std::vector<DIE *> &Children = Die->getChildren();
    
    // If not last sibling and has children then add sibling offset attribute.
    if (!Last && !Children.empty()) Die->AddSiblingOffset();

    // Record the abbreviation.
    AssignAbbrevNumber(Die->getAbbrev());
   
    // Get the abbreviation for this DIE.
    unsigned AbbrevNumber = Die->getAbbrevNumber();
    const DIEAbbrev *Abbrev = Abbreviations[AbbrevNumber - 1];

    // Set DIE offset
    Die->setOffset(Offset);
    
    // Start the size with the size of abbreviation code.
    Offset += Asm->SizeULEB128(AbbrevNumber);
    
    const std::vector<DIEValue *> &Values = Die->getValues();
    const std::vector<DIEAbbrevData> &AbbrevData = Abbrev->getData();

    // Size the DIE attribute values.
    for (unsigned i = 0, N = Values.size(); i < N; ++i) {
      // Size attribute value.
      Offset += Values[i]->SizeOf(*this, AbbrevData[i].getForm());
    }
    
    // Size the DIE children if any.
    if (!Children.empty()) {
      assert(Abbrev->getChildrenFlag() == DW_CHILDREN_yes &&
             "Children flag not set");
      
      for (unsigned j = 0, M = Children.size(); j < M; ++j) {
        Offset = SizeAndOffsetDie(Children[j], Offset, (j + 1) == M);
      }
      
      // End of children marker.
      Offset += sizeof(int8_t);
    }

    Die->setSize(Offset - Die->getOffset());
    return Offset;
  }

  /// SizeAndOffsets - Compute the size and offset of all the DIEs.
  ///
  void SizeAndOffsets() {
    // Process base compile unit.
    CompileUnit *Unit = GetBaseCompileUnit();
    // Compute size of compile unit header
    unsigned Offset = sizeof(int32_t) + // Length of Compilation Unit Info
                      sizeof(int16_t) + // DWARF version number
                      sizeof(int32_t) + // Offset Into Abbrev. Section
                      sizeof(int8_t);   // Pointer Size (in bytes)
    SizeAndOffsetDie(Unit->getDie(), Offset, true);
  }

  /// EmitDebugInfo - Emit the debug info section.
  ///
  void EmitDebugInfo() {
    // Start debug info section.
    Asm->SwitchToDataSection(TAI->getDwarfInfoSection());
    
    CompileUnit *Unit = GetBaseCompileUnit();
    DIE *Die = Unit->getDie();
    // Emit the compile units header.
    EmitLabel("info_begin", Unit->getID());
    // Emit size of content not including length itself
    unsigned ContentSize = Die->getSize() +
                           sizeof(int16_t) + // DWARF version number
                           sizeof(int32_t) + // Offset Into Abbrev. Section
                           sizeof(int8_t) +  // Pointer Size (in bytes)
                           sizeof(int32_t);  // FIXME - extra pad for gdb bug.
                           
    Asm->EmitInt32(ContentSize);  Asm->EOL("Length of Compilation Unit Info");
    Asm->EmitInt16(DWARF_VERSION); Asm->EOL("DWARF version number");
    EmitSectionOffset("abbrev_begin", "section_abbrev", 0, 0, true, false);
    Asm->EOL("Offset Into Abbrev. Section");
    Asm->EmitInt8(TD->getPointerSize()); Asm->EOL("Address Size (in bytes)");
  
    EmitDIE(Die);
    // FIXME - extra padding for gdb bug.
    Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
    Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
    Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
    Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
    EmitLabel("info_end", Unit->getID());
    
    Asm->EOL();
  }

  /// EmitAbbreviations - Emit the abbreviation section.
  ///
  void EmitAbbreviations() const {
    // Check to see if it is worth the effort.
    if (!Abbreviations.empty()) {
      // Start the debug abbrev section.
      Asm->SwitchToDataSection(TAI->getDwarfAbbrevSection());
      
      EmitLabel("abbrev_begin", 0);
      
      // For each abbrevation.
      for (unsigned i = 0, N = Abbreviations.size(); i < N; ++i) {
        // Get abbreviation data
        const DIEAbbrev *Abbrev = Abbreviations[i];
        
        // Emit the abbrevations code (base 1 index.)
        Asm->EmitULEB128Bytes(Abbrev->getNumber());
        Asm->EOL("Abbreviation Code");
        
        // Emit the abbreviations data.
        Abbrev->Emit(*this);
    
        Asm->EOL();
      }
      
      // Mark end of abbreviations.
      Asm->EmitULEB128Bytes(0); Asm->EOL("EOM(3)");

      EmitLabel("abbrev_end", 0);
    
      Asm->EOL();
    }
  }

  /// EmitDebugLines - Emit source line information.
  ///
  void EmitDebugLines() {
    // If there are no lines to emit (such as when we're using .loc directives
    // to emit .debug_line information) don't emit a .debug_line header.
    if (SectionSourceLines.empty())
      return;

    // Minimum line delta, thus ranging from -10..(255-10).
    const int MinLineDelta = -(DW_LNS_fixed_advance_pc + 1);
    // Maximum line delta, thus ranging from -10..(255-10).
    const int MaxLineDelta = 255 + MinLineDelta;

    // Start the dwarf line section.
    Asm->SwitchToDataSection(TAI->getDwarfLineSection());
    
    // Construct the section header.
    
    EmitDifference("line_end", 0, "line_begin", 0, true);
    Asm->EOL("Length of Source Line Info");
    EmitLabel("line_begin", 0);
    
    Asm->EmitInt16(DWARF_VERSION); Asm->EOL("DWARF version number");
    
    EmitDifference("line_prolog_end", 0, "line_prolog_begin", 0, true);
    Asm->EOL("Prolog Length");
    EmitLabel("line_prolog_begin", 0);
    
    Asm->EmitInt8(1); Asm->EOL("Minimum Instruction Length");

    Asm->EmitInt8(1); Asm->EOL("Default is_stmt_start flag");

    Asm->EmitInt8(MinLineDelta); Asm->EOL("Line Base Value (Special Opcodes)");
    
    Asm->EmitInt8(MaxLineDelta); Asm->EOL("Line Range Value (Special Opcodes)");

    Asm->EmitInt8(-MinLineDelta); Asm->EOL("Special Opcode Base");
    
    // Line number standard opcode encodings argument count
    Asm->EmitInt8(0); Asm->EOL("DW_LNS_copy arg count");
    Asm->EmitInt8(1); Asm->EOL("DW_LNS_advance_pc arg count");
    Asm->EmitInt8(1); Asm->EOL("DW_LNS_advance_line arg count");
    Asm->EmitInt8(1); Asm->EOL("DW_LNS_set_file arg count");
    Asm->EmitInt8(1); Asm->EOL("DW_LNS_set_column arg count");
    Asm->EmitInt8(0); Asm->EOL("DW_LNS_negate_stmt arg count");
    Asm->EmitInt8(0); Asm->EOL("DW_LNS_set_basic_block arg count");
    Asm->EmitInt8(0); Asm->EOL("DW_LNS_const_add_pc arg count");
    Asm->EmitInt8(1); Asm->EOL("DW_LNS_fixed_advance_pc arg count");

    const UniqueVector<std::string> &Directories = MMI->getDirectories();
    const UniqueVector<SourceFileInfo>
      &SourceFiles = MMI->getSourceFiles();

    // Emit directories.
    for (unsigned DirectoryID = 1, NDID = Directories.size();
                  DirectoryID <= NDID; ++DirectoryID) {
      Asm->EmitString(Directories[DirectoryID]); Asm->EOL("Directory");
    }
    Asm->EmitInt8(0); Asm->EOL("End of directories");
    
    // Emit files.
    for (unsigned SourceID = 1, NSID = SourceFiles.size();
                 SourceID <= NSID; ++SourceID) {
      const SourceFileInfo &SourceFile = SourceFiles[SourceID];
      Asm->EmitString(SourceFile.getName());
      Asm->EOL("Source");
      Asm->EmitULEB128Bytes(SourceFile.getDirectoryID());
      Asm->EOL("Directory #");
      Asm->EmitULEB128Bytes(0);
      Asm->EOL("Mod date");
      Asm->EmitULEB128Bytes(0);
      Asm->EOL("File size");
    }
    Asm->EmitInt8(0); Asm->EOL("End of files");
    
    EmitLabel("line_prolog_end", 0);
    
    // A sequence for each text section.
    for (unsigned j = 0, M = SectionSourceLines.size(); j < M; ++j) {
      // Isolate current sections line info.
      const std::vector<SourceLineInfo> &LineInfos = SectionSourceLines[j];
      
      Asm->EOL(std::string("Section ") + SectionMap[j + 1]);

      // Dwarf assumes we start with first line of first source file.
      unsigned Source = 1;
      unsigned Line = 1;
      
      // Construct rows of the address, source, line, column matrix.
      for (unsigned i = 0, N = LineInfos.size(); i < N; ++i) {
        const SourceLineInfo &LineInfo = LineInfos[i];
        unsigned LabelID = MMI->MappedLabel(LineInfo.getLabelID());
        if (!LabelID) continue;
        
        unsigned SourceID = LineInfo.getSourceID();
        const SourceFileInfo &SourceFile = SourceFiles[SourceID];
        unsigned DirectoryID = SourceFile.getDirectoryID();
        Asm->EOL(Directories[DirectoryID]
          + SourceFile.getName()
          + ":"
          + utostr_32(LineInfo.getLine()));

        // Define the line address.
        Asm->EmitInt8(0); Asm->EOL("Extended Op");
        Asm->EmitInt8(TD->getPointerSize() + 1); Asm->EOL("Op size");
        Asm->EmitInt8(DW_LNE_set_address); Asm->EOL("DW_LNE_set_address");
        EmitReference("label",  LabelID); Asm->EOL("Location label");
        
        // If change of source, then switch to the new source.
        if (Source != LineInfo.getSourceID()) {
          Source = LineInfo.getSourceID();
          Asm->EmitInt8(DW_LNS_set_file); Asm->EOL("DW_LNS_set_file");
          Asm->EmitULEB128Bytes(Source); Asm->EOL("New Source");
        }
        
        // If change of line.
        if (Line != LineInfo.getLine()) {
          // Determine offset.
          int Offset = LineInfo.getLine() - Line;
          int Delta = Offset - MinLineDelta;
          
          // Update line.
          Line = LineInfo.getLine();
          
          // If delta is small enough and in range...
          if (Delta >= 0 && Delta < (MaxLineDelta - 1)) {
            // ... then use fast opcode.
            Asm->EmitInt8(Delta - MinLineDelta); Asm->EOL("Line Delta");
          } else {
            // ... otherwise use long hand.
            Asm->EmitInt8(DW_LNS_advance_line); Asm->EOL("DW_LNS_advance_line");
            Asm->EmitSLEB128Bytes(Offset); Asm->EOL("Line Offset");
            Asm->EmitInt8(DW_LNS_copy); Asm->EOL("DW_LNS_copy");
          }
        } else {
          // Copy the previous row (different address or source)
          Asm->EmitInt8(DW_LNS_copy); Asm->EOL("DW_LNS_copy");
        }
      }

      // Define last address of section.
      Asm->EmitInt8(0); Asm->EOL("Extended Op");
      Asm->EmitInt8(TD->getPointerSize() + 1); Asm->EOL("Op size");
      Asm->EmitInt8(DW_LNE_set_address); Asm->EOL("DW_LNE_set_address");
      EmitReference("section_end", j + 1); Asm->EOL("Section end label");

      // Mark end of matrix.
      Asm->EmitInt8(0); Asm->EOL("DW_LNE_end_sequence");
      Asm->EmitULEB128Bytes(1); Asm->EOL();
      Asm->EmitInt8(1); Asm->EOL();
    }
    
    EmitLabel("line_end", 0);
    
    Asm->EOL();
  }
    
  /// EmitCommonDebugFrame - Emit common frame info into a debug frame section.
  ///
  void EmitCommonDebugFrame() {
    if (!TAI->doesDwarfRequireFrameSection())
      return;

    int stackGrowth =
        Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
          TargetFrameInfo::StackGrowsUp ?
        TD->getPointerSize() : -TD->getPointerSize();

    // Start the dwarf frame section.
    Asm->SwitchToDataSection(TAI->getDwarfFrameSection());

    EmitLabel("debug_frame_common", 0);
    EmitDifference("debug_frame_common_end", 0,
                   "debug_frame_common_begin", 0, true);
    Asm->EOL("Length of Common Information Entry");

    EmitLabel("debug_frame_common_begin", 0);
    Asm->EmitInt32((int)DW_CIE_ID);
    Asm->EOL("CIE Identifier Tag");
    Asm->EmitInt8(DW_CIE_VERSION);
    Asm->EOL("CIE Version");
    Asm->EmitString("");
    Asm->EOL("CIE Augmentation");
    Asm->EmitULEB128Bytes(1);
    Asm->EOL("CIE Code Alignment Factor");
    Asm->EmitSLEB128Bytes(stackGrowth);
    Asm->EOL("CIE Data Alignment Factor");   
    Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), false));
    Asm->EOL("CIE RA Column");
    
    std::vector<MachineMove> Moves;
    RI->getInitialFrameState(Moves);

    EmitFrameMoves(NULL, 0, Moves, false);

    Asm->EmitAlignment(2);
    EmitLabel("debug_frame_common_end", 0);
    
    Asm->EOL();
  }

  /// EmitFunctionDebugFrame - Emit per function frame info into a debug frame
  /// section.
  void EmitFunctionDebugFrame(const FunctionDebugFrameInfo &DebugFrameInfo) {
    if (!TAI->doesDwarfRequireFrameSection())
      return;
       
    // Start the dwarf frame section.
    Asm->SwitchToDataSection(TAI->getDwarfFrameSection());
    
    EmitDifference("debug_frame_end", DebugFrameInfo.Number,
                   "debug_frame_begin", DebugFrameInfo.Number, true);
    Asm->EOL("Length of Frame Information Entry");
    
    EmitLabel("debug_frame_begin", DebugFrameInfo.Number);

    EmitSectionOffset("debug_frame_common", "section_debug_frame",
                      0, 0, true, false);
    Asm->EOL("FDE CIE offset");

    EmitReference("func_begin", DebugFrameInfo.Number);
    Asm->EOL("FDE initial location");
    EmitDifference("func_end", DebugFrameInfo.Number,
                   "func_begin", DebugFrameInfo.Number);
    Asm->EOL("FDE address range");
    
    EmitFrameMoves("func_begin", DebugFrameInfo.Number, DebugFrameInfo.Moves, false);
    
    Asm->EmitAlignment(2);
    EmitLabel("debug_frame_end", DebugFrameInfo.Number);

    Asm->EOL();
  }

  /// EmitDebugPubNames - Emit visible names into a debug pubnames section.
  ///
  void EmitDebugPubNames() {
    // Start the dwarf pubnames section.
    Asm->SwitchToDataSection(TAI->getDwarfPubNamesSection());
      
    CompileUnit *Unit = GetBaseCompileUnit(); 
 
    EmitDifference("pubnames_end", Unit->getID(),
                   "pubnames_begin", Unit->getID(), true);
    Asm->EOL("Length of Public Names Info");
    
    EmitLabel("pubnames_begin", Unit->getID());
    
    Asm->EmitInt16(DWARF_VERSION); Asm->EOL("DWARF Version");

    EmitSectionOffset("info_begin", "section_info",
                      Unit->getID(), 0, true, false);
    Asm->EOL("Offset of Compilation Unit Info");

    EmitDifference("info_end", Unit->getID(), "info_begin", Unit->getID(),true);
    Asm->EOL("Compilation Unit Length");
    
    std::map<std::string, DIE *> &Globals = Unit->getGlobals();
    
    for (std::map<std::string, DIE *>::iterator GI = Globals.begin(),
                                                GE = Globals.end();
         GI != GE; ++GI) {
      const std::string &Name = GI->first;
      DIE * Entity = GI->second;
      
      Asm->EmitInt32(Entity->getOffset()); Asm->EOL("DIE offset");
      Asm->EmitString(Name); Asm->EOL("External Name");
    }
  
    Asm->EmitInt32(0); Asm->EOL("End Mark");
    EmitLabel("pubnames_end", Unit->getID());
  
    Asm->EOL();
  }

  /// EmitDebugStr - Emit visible names into a debug str section.
  ///
  void EmitDebugStr() {
    // Check to see if it is worth the effort.
    if (!StringPool.empty()) {
      // Start the dwarf str section.
      Asm->SwitchToDataSection(TAI->getDwarfStrSection());
      
      // For each of strings in the string pool.
      for (unsigned StringID = 1, N = StringPool.size();
           StringID <= N; ++StringID) {
        // Emit a label for reference from debug information entries.
        EmitLabel("string", StringID);
        // Emit the string itself.
        const std::string &String = StringPool[StringID];
        Asm->EmitString(String); Asm->EOL();
      }
    
      Asm->EOL();
    }
  }

  /// EmitDebugLoc - Emit visible names into a debug loc section.
  ///
  void EmitDebugLoc() {
    // Start the dwarf loc section.
    Asm->SwitchToDataSection(TAI->getDwarfLocSection());
    
    Asm->EOL();
  }

  /// EmitDebugARanges - Emit visible names into a debug aranges section.
  ///
  void EmitDebugARanges() {
    // Start the dwarf aranges section.
    Asm->SwitchToDataSection(TAI->getDwarfARangesSection());
    
    // FIXME - Mock up
  #if 0
    CompileUnit *Unit = GetBaseCompileUnit(); 
      
    // Don't include size of length
    Asm->EmitInt32(0x1c); Asm->EOL("Length of Address Ranges Info");
    
    Asm->EmitInt16(DWARF_VERSION); Asm->EOL("Dwarf Version");
    
    EmitReference("info_begin", Unit->getID());
    Asm->EOL("Offset of Compilation Unit Info");

    Asm->EmitInt8(TD->getPointerSize()); Asm->EOL("Size of Address");

    Asm->EmitInt8(0); Asm->EOL("Size of Segment Descriptor");

    Asm->EmitInt16(0);  Asm->EOL("Pad (1)");
    Asm->EmitInt16(0);  Asm->EOL("Pad (2)");

    // Range 1
    EmitReference("text_begin", 0); Asm->EOL("Address");
    EmitDifference("text_end", 0, "text_begin", 0, true); Asm->EOL("Length");

    Asm->EmitInt32(0); Asm->EOL("EOM (1)");
    Asm->EmitInt32(0); Asm->EOL("EOM (2)");
  #endif
    
    Asm->EOL();
  }

  /// EmitDebugRanges - Emit visible names into a debug ranges section.
  ///
  void EmitDebugRanges() {
    // Start the dwarf ranges section.
    Asm->SwitchToDataSection(TAI->getDwarfRangesSection());
    
    Asm->EOL();
  }

  /// EmitDebugMacInfo - Emit visible names into a debug macinfo section.
  ///
  void EmitDebugMacInfo() {
    // Start the dwarf macinfo section.
    Asm->SwitchToDataSection(TAI->getDwarfMacInfoSection());
    
    Asm->EOL();
  }

  /// ConstructCompileUnitDIEs - Create a compile unit DIE for each source and
  /// header file.
  void ConstructCompileUnitDIEs() {
    const UniqueVector<CompileUnitDesc *> CUW = MMI->getCompileUnits();
    
    for (unsigned i = 1, N = CUW.size(); i <= N; ++i) {
      unsigned ID = MMI->RecordSource(CUW[i]);
      CompileUnit *Unit = NewCompileUnit(CUW[i], ID);
      CompileUnits.push_back(Unit);
    }
  }

  /// ConstructGlobalDIEs - Create DIEs for each of the externally visible
  /// global variables.
  void ConstructGlobalDIEs() {
    std::vector<GlobalVariableDesc *> GlobalVariables =
        MMI->getAnchoredDescriptors<GlobalVariableDesc>(*M);
    
    for (unsigned i = 0, N = GlobalVariables.size(); i < N; ++i) {
      GlobalVariableDesc *GVD = GlobalVariables[i];
      NewGlobalVariable(GVD);
    }
  }

  /// ConstructSubprogramDIEs - Create DIEs for each of the externally visible
  /// subprograms.
  void ConstructSubprogramDIEs() {
    std::vector<SubprogramDesc *> Subprograms =
        MMI->getAnchoredDescriptors<SubprogramDesc>(*M);
    
    for (unsigned i = 0, N = Subprograms.size(); i < N; ++i) {
      SubprogramDesc *SPD = Subprograms[i];
      NewSubprogram(SPD);
    }
  }

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(std::ostream &OS, AsmPrinter *A, const TargetAsmInfo *T)
  : Dwarf(OS, A, T, "dbg")
  , CompileUnits()
  , AbbreviationsSet(InitAbbreviationsSetSize)
  , Abbreviations()
  , ValuesSet(InitValuesSetSize)
  , Values()
  , StringPool()
  , DescToUnitMap()
  , SectionMap()
  , SectionSourceLines()
  , didInitial(false)
  , shouldEmit(false)
  {
  }
  virtual ~DwarfDebug() {
    for (unsigned i = 0, N = CompileUnits.size(); i < N; ++i)
      delete CompileUnits[i];
    for (unsigned j = 0, M = Values.size(); j < M; ++j)
      delete Values[j];
  }

  /// SetModuleInfo - Set machine module information when it's known that pass
  /// manager has created it.  Set by the target AsmPrinter.
  void SetModuleInfo(MachineModuleInfo *mmi) {
    // Make sure initial declarations are made.
    if (!MMI && mmi->hasDebugInfo()) {
      MMI = mmi;
      shouldEmit = true;
      
      // Create all the compile unit DIEs.
      ConstructCompileUnitDIEs();
      
      // Create DIEs for each of the externally visible global variables.
      ConstructGlobalDIEs();

      // Create DIEs for each of the externally visible subprograms.
      ConstructSubprogramDIEs();
      
      // Prime section data.
      SectionMap.insert(TAI->getTextSection());

      // Print out .file directives to specify files for .loc directives. These
      // are printed out early so that they precede any .loc directives.
      if (TAI->hasDotLocAndDotFile()) {
        const UniqueVector<SourceFileInfo> &SourceFiles = MMI->getSourceFiles();
        const UniqueVector<std::string> &Directories = MMI->getDirectories();
        for (unsigned i = 1, e = SourceFiles.size(); i <= e; ++i) {
          sys::Path FullPath(Directories[SourceFiles[i].getDirectoryID()]);
          bool AppendOk = FullPath.appendComponent(SourceFiles[i].getName());
          assert(AppendOk && "Could not append filename to directory!");
          Asm->EmitFile(i, FullPath.toString());
          Asm->EOL();
        }
      }

      // Emit initial sections
      EmitInitial();
    }
  }

  /// BeginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void BeginModule(Module *M) {
    this->M = M;
    
    if (!ShouldEmitDwarf()) return;
  }

  /// EndModule - Emit all Dwarf sections that should come after the content.
  ///
  void EndModule() {
    if (!ShouldEmitDwarf()) return;
    
    // Standard sections final addresses.
    Asm->SwitchToTextSection(TAI->getTextSection());
    EmitLabel("text_end", 0);
    Asm->SwitchToDataSection(TAI->getDataSection());
    EmitLabel("data_end", 0);
    
    // End text sections.
    for (unsigned i = 1, N = SectionMap.size(); i <= N; ++i) {
      Asm->SwitchToTextSection(SectionMap[i].c_str());
      EmitLabel("section_end", i);
    }

    // Emit common frame information.
    EmitCommonDebugFrame();

    // Emit function debug frame information
    for (std::vector<FunctionDebugFrameInfo>::iterator I = DebugFrames.begin(),
           E = DebugFrames.end(); I != E; ++I)
      EmitFunctionDebugFrame(*I);

    // Compute DIE offsets and sizes.
    SizeAndOffsets();
    
    // Emit all the DIEs into a debug info section
    EmitDebugInfo();
    
    // Corresponding abbreviations into a abbrev section.
    EmitAbbreviations();
    
    // Emit source line correspondence into a debug line section.
    EmitDebugLines();
    
    // Emit info into a debug pubnames section.
    EmitDebugPubNames();
    
    // Emit info into a debug str section.
    EmitDebugStr();
    
    // Emit info into a debug loc section.
    EmitDebugLoc();
    
    // Emit info into a debug aranges section.
    EmitDebugARanges();
    
    // Emit info into a debug ranges section.
    EmitDebugRanges();
    
    // Emit info into a debug macinfo section.
    EmitDebugMacInfo();
  }

  /// BeginFunction - Gather pre-function debug information.  Assumes being 
  /// emitted immediately after the function entry point.
  void BeginFunction(MachineFunction *MF) {
    this->MF = MF;
    
    if (!ShouldEmitDwarf()) return;

    // Begin accumulating function debug information.
    MMI->BeginFunction(MF);
    
    // Assumes in correct section after the entry point.
    EmitLabel("func_begin", ++SubprogramCount);

    // Emit label for the implicitly defined dbg.stoppoint at the start of
    // the function.
    const SourceLineInfo &LineInfo = MMI->getSourceLines()[0];
    Asm->printLabel(LineInfo.getLabelID());
  }
  
  /// EndFunction - Gather and emit post-function debug information.
  ///
  void EndFunction() {
    if (!ShouldEmitDwarf()) return;
    
    // Define end label for subprogram.
    EmitLabel("func_end", SubprogramCount);
      
    // Get function line info.
    const std::vector<SourceLineInfo> &LineInfos = MMI->getSourceLines();

    if (!LineInfos.empty()) {
      // Get section line info.
      unsigned ID = SectionMap.insert(Asm->CurrentSection);
      if (SectionSourceLines.size() < ID) SectionSourceLines.resize(ID);
      std::vector<SourceLineInfo> &SectionLineInfos = SectionSourceLines[ID-1];
      // Append the function info to section info.
      SectionLineInfos.insert(SectionLineInfos.end(),
                              LineInfos.begin(), LineInfos.end());
    }
    
    // Construct scopes for subprogram.
    ConstructRootScope(MMI->getRootScope());

    DebugFrames.push_back(FunctionDebugFrameInfo(SubprogramCount,
                                                 MMI->getFrameMoves()));
  }
};

//===----------------------------------------------------------------------===//
/// DwarfException - Emits Dwarf exception handling directives. 
///
class DwarfException : public Dwarf  {

private:
  struct FunctionEHFrameInfo {
    std::string FnName;
    unsigned Number;
    unsigned PersonalityIndex;
    bool hasCalls;
    bool hasLandingPads;
    std::vector<MachineMove> Moves;
    const Function * function;

    FunctionEHFrameInfo(const std::string &FN, unsigned Num, unsigned P,
                        bool hC, bool hL,
                        const std::vector<MachineMove> &M,
                        const Function *f):
      FnName(FN), Number(Num), PersonalityIndex(P),
      hasCalls(hC), hasLandingPads(hL), Moves(M), function (f) { }
  };

  std::vector<FunctionEHFrameInfo> EHFrames;
    
  /// shouldEmit - Flag to indicate if debug information should be emitted.
  ///
  bool shouldEmit;
  
  /// EmitCommonEHFrame - Emit the common eh unwind frame.
  ///
  void EmitCommonEHFrame(const Function *Personality, unsigned Index) {
    // Size and sign of stack growth.
    int stackGrowth =
        Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
          TargetFrameInfo::StackGrowsUp ?
        TD->getPointerSize() : -TD->getPointerSize();

    // Begin eh frame section.
    Asm->SwitchToTextSection(TAI->getDwarfEHFrameSection());
    O << "EH_frame" << Index << ":\n";
    EmitLabel("section_eh_frame", Index);

    // Define base labels.
    EmitLabel("eh_frame_common", Index);
    
    // Define the eh frame length.
    EmitDifference("eh_frame_common_end", Index,
                   "eh_frame_common_begin", Index, true);
    Asm->EOL("Length of Common Information Entry");

    // EH frame header.
    EmitLabel("eh_frame_common_begin", Index);
    Asm->EmitInt32((int)0);
    Asm->EOL("CIE Identifier Tag");
    Asm->EmitInt8(DW_CIE_VERSION);
    Asm->EOL("CIE Version");
    
    // The personality presence indicates that language specific information
    // will show up in the eh frame.
    Asm->EmitString(Personality ? "zPLR" : "zR");
    Asm->EOL("CIE Augmentation");
    
    // Round out reader.
    Asm->EmitULEB128Bytes(1);
    Asm->EOL("CIE Code Alignment Factor");
    Asm->EmitSLEB128Bytes(stackGrowth);
    Asm->EOL("CIE Data Alignment Factor");   
    Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), true));
    Asm->EOL("CIE RA Column");
    
    // If there is a personality, we need to indicate the functions location.
    if (Personality) {
      Asm->EmitULEB128Bytes(7);
      Asm->EOL("Augmentation Size");

      if (TAI->getNeedsIndirectEncoding())
        Asm->EmitInt8(DW_EH_PE_pcrel | DW_EH_PE_sdata4 | DW_EH_PE_indirect);
      else
        Asm->EmitInt8(DW_EH_PE_pcrel | DW_EH_PE_sdata4);

      Asm->EOL("Personality (pcrel sdata4 indirect)");
      
      PrintRelDirective(TAI->getShortenEHDataOn64Bit());
      O << TAI->getPersonalityPrefix();
      Asm->EmitExternalGlobal((const GlobalVariable *)(Personality));
      O << TAI->getPersonalitySuffix();
      if (!TAI->getShortenEHDataOn64Bit()) {
        O << "-" << TAI->getPCSymbol();
      }
      Asm->EOL("Personality");

      Asm->EmitULEB128Bytes(DW_EH_PE_pcrel);
      Asm->EOL("LSDA Encoding (pcrel)");
      Asm->EmitULEB128Bytes(DW_EH_PE_pcrel);
      Asm->EOL("FDE Encoding (pcrel)");
   } else {
      Asm->EmitULEB128Bytes(1);
      Asm->EOL("Augmentation Size");
      Asm->EmitULEB128Bytes(DW_EH_PE_pcrel);
      Asm->EOL("FDE Encoding (pcrel)");
    }

    // Indicate locations of general callee saved registers in frame.
    std::vector<MachineMove> Moves;
    RI->getInitialFrameState(Moves);
    EmitFrameMoves(NULL, 0, Moves, true);

    Asm->EmitAlignment(2);
    EmitLabel("eh_frame_common_end", Index);
    
    Asm->EOL();
  }
  
  /// EmitEHFrame - Emit function exception frame information.
  ///
  void EmitEHFrame(const FunctionEHFrameInfo &EHFrameInfo) {
    Function::LinkageTypes linkage = EHFrameInfo.function->getLinkage();

    Asm->SwitchToTextSection(TAI->getDwarfEHFrameSection());

    // Externally visible entry into the functions eh frame info.
    // If the corresponding function is static, this should not be
    // externally visible.
    if (linkage != Function::InternalLinkage) {
      if (const char *GlobalEHDirective = TAI->getGlobalEHDirective())
        O << GlobalEHDirective << EHFrameInfo.FnName << "\n";
    }

    // If corresponding function is weak definition, this should be too.
    if ((linkage == Function::WeakLinkage || 
         linkage == Function::LinkOnceLinkage) &&
        TAI->getWeakDefDirective())
      O << TAI->getWeakDefDirective() << EHFrameInfo.FnName << "\n";

    // If there are no calls then you can't unwind.  This may mean we can
    // omit the EH Frame, but some environments do not handle weak absolute
    // symbols.
    if (!EHFrameInfo.hasCalls &&
        ((linkage != Function::WeakLinkage && 
          linkage != Function::LinkOnceLinkage) ||
         !TAI->getWeakDefDirective() ||
         TAI->getSupportsWeakOmittedEHFrame()))
    { 
      O << EHFrameInfo.FnName << " = 0\n";
      // This name has no connection to the function, so it might get 
      // dead-stripped when the function is not, erroneously.  Prohibit 
      // dead-stripping unconditionally.
      if (const char *UsedDirective = TAI->getUsedDirective())
        O << UsedDirective << EHFrameInfo.FnName << "\n\n";
    } else {
      O << EHFrameInfo.FnName << ":\n";

      // EH frame header.
      EmitDifference("eh_frame_end", EHFrameInfo.Number,
                     "eh_frame_begin", EHFrameInfo.Number, true);
      Asm->EOL("Length of Frame Information Entry");
      
      EmitLabel("eh_frame_begin", EHFrameInfo.Number);

      EmitSectionOffset("eh_frame_begin", "eh_frame_common",
                        EHFrameInfo.Number, EHFrameInfo.PersonalityIndex,
                        true, true);
      Asm->EOL("FDE CIE offset");

      EmitReference("eh_func_begin", EHFrameInfo.Number, true);
      Asm->EOL("FDE initial location");
      EmitDifference("eh_func_end", EHFrameInfo.Number,
                     "eh_func_begin", EHFrameInfo.Number);
      Asm->EOL("FDE address range");
      
      // If there is a personality and landing pads then point to the language
      // specific data area in the exception table.
      if (EHFrameInfo.PersonalityIndex) {
        Asm->EmitULEB128Bytes(TAI->getShortenEHDataOn64Bit() ? 8 : 4);
        Asm->EOL("Augmentation size");
        
        if (EHFrameInfo.hasLandingPads) {
          EmitReference("exception", EHFrameInfo.Number, true);
        } else if (TD->getPointerSize() == 8) {
          Asm->EmitInt64((int)0);
        } else {
          Asm->EmitInt32((int)0);
        }
        Asm->EOL("Language Specific Data Area");
      } else {
        Asm->EmitULEB128Bytes(0);
        Asm->EOL("Augmentation size");
      }
      
      // Indicate locations of function specific  callee saved registers in
      // frame.
      EmitFrameMoves("eh_func_begin", EHFrameInfo.Number, EHFrameInfo.Moves, true);
      
      Asm->EmitAlignment(2);
      EmitLabel("eh_frame_end", EHFrameInfo.Number);
    
      // If the function is marked used, this table should be also.  We cannot 
      // make the mark unconditional in this case, since retaining the table
      // also retains the function in this case, and there is code around 
      // that depends on unused functions (calling undefined externals) being
      // dead-stripped to link correctly.  Yes, there really is.
      if (MMI->getUsedFunctions().count(EHFrameInfo.function))
        if (const char *UsedDirective = TAI->getUsedDirective())
          O << UsedDirective << EHFrameInfo.FnName << "\n\n";
    }
  }

  /// EmitExceptionTable - Emit landing pads and actions.
  ///
  /// The general organization of the table is complex, but the basic concepts
  /// are easy.  First there is a header which describes the location and
  /// organization of the three components that follow.
  ///  1. The landing pad site information describes the range of code covered
  ///     by the try.  In our case it's an accumulation of the ranges covered
  ///     by the invokes in the try.  There is also a reference to the landing
  ///     pad that handles the exception once processed.  Finally an index into
  ///     the actions table.
  ///  2. The action table, in our case, is composed of pairs of type ids
  ///     and next action offset.  Starting with the action index from the
  ///     landing pad site, each type Id is checked for a match to the current
  ///     exception.  If it matches then the exception and type id are passed
  ///     on to the landing pad.  Otherwise the next action is looked up.  This
  ///     chain is terminated with a next action of zero.  If no type id is
  ///     found the the frame is unwound and handling continues.
  ///  3. Type id table contains references to all the C++ typeinfo for all
  ///     catches in the function.  This tables is reversed indexed base 1.

  /// SharedTypeIds - How many leading type ids two landing pads have in common.
  static unsigned SharedTypeIds(const LandingPadInfo *L,
                                const LandingPadInfo *R) {
    const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
    unsigned LSize = LIds.size(), RSize = RIds.size();
    unsigned MinSize = LSize < RSize ? LSize : RSize;
    unsigned Count = 0;

    for (; Count != MinSize; ++Count)
      if (LIds[Count] != RIds[Count])
        return Count;

    return Count;
  }

  /// PadLT - Order landing pads lexicographically by type id.
  static bool PadLT(const LandingPadInfo *L, const LandingPadInfo *R) {
    const std::vector<int> &LIds = L->TypeIds, &RIds = R->TypeIds;
    unsigned LSize = LIds.size(), RSize = RIds.size();
    unsigned MinSize = LSize < RSize ? LSize : RSize;

    for (unsigned i = 0; i != MinSize; ++i)
      if (LIds[i] != RIds[i])
        return LIds[i] < RIds[i];

    return LSize < RSize;
  }

  struct KeyInfo {
    static inline unsigned getEmptyKey() { return -1U; }
    static inline unsigned getTombstoneKey() { return -2U; }
    static unsigned getHashValue(const unsigned &Key) { return Key; }
    static bool isEqual(unsigned LHS, unsigned RHS) { return LHS == RHS; }
    static bool isPod() { return true; }
  };

  /// ActionEntry - Structure describing an entry in the actions table.
  struct ActionEntry {
    int ValueForTypeID; // The value to write - may not be equal to the type id.
    int NextAction;
    struct ActionEntry *Previous;
  };

  /// PadRange - Structure holding a try-range and the associated landing pad.
  struct PadRange {
    // The index of the landing pad.
    unsigned PadIndex;
    // The index of the begin and end labels in the landing pad's label lists.
    unsigned RangeIndex;
  };

  typedef DenseMap<unsigned, PadRange, KeyInfo> RangeMapType;

  /// CallSiteEntry - Structure describing an entry in the call-site table.
  struct CallSiteEntry {
    // The 'try-range' is BeginLabel .. EndLabel.
    unsigned BeginLabel; // zero indicates the start of the function.
    unsigned EndLabel;   // zero indicates the end of the function.
    // The landing pad starts at PadLabel.
    unsigned PadLabel;   // zero indicates that there is no landing pad.
    unsigned Action;
  };

  void EmitExceptionTable() {
    // Map all labels and get rid of any dead landing pads.
    MMI->TidyLandingPads();

    const std::vector<GlobalVariable *> &TypeInfos = MMI->getTypeInfos();
    const std::vector<unsigned> &FilterIds = MMI->getFilterIds();
    const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
    if (PadInfos.empty()) return;

    // Sort the landing pads in order of their type ids.  This is used to fold
    // duplicate actions.
    SmallVector<const LandingPadInfo *, 64> LandingPads;
    LandingPads.reserve(PadInfos.size());
    for (unsigned i = 0, N = PadInfos.size(); i != N; ++i)
      LandingPads.push_back(&PadInfos[i]);
    std::sort(LandingPads.begin(), LandingPads.end(), PadLT);

    // Negative type ids index into FilterIds, positive type ids index into
    // TypeInfos.  The value written for a positive type id is just the type
    // id itself.  For a negative type id, however, the value written is the
    // (negative) byte offset of the corresponding FilterIds entry.  The byte
    // offset is usually equal to the type id, because the FilterIds entries
    // are written using a variable width encoding which outputs one byte per
    // entry as long as the value written is not too large, but can differ.
    // This kind of complication does not occur for positive type ids because
    // type infos are output using a fixed width encoding.
    // FilterOffsets[i] holds the byte offset corresponding to FilterIds[i].
    SmallVector<int, 16> FilterOffsets;
    FilterOffsets.reserve(FilterIds.size());
    int Offset = -1;
    for(std::vector<unsigned>::const_iterator I = FilterIds.begin(),
        E = FilterIds.end(); I != E; ++I) {
      FilterOffsets.push_back(Offset);
      Offset -= Asm->SizeULEB128(*I);
    }

    // Compute the actions table and gather the first action index for each
    // landing pad site.
    SmallVector<ActionEntry, 32> Actions;
    SmallVector<unsigned, 64> FirstActions;
    FirstActions.reserve(LandingPads.size());

    int FirstAction = 0;
    unsigned SizeActions = 0;
    for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
      const LandingPadInfo *LP = LandingPads[i];
      const std::vector<int> &TypeIds = LP->TypeIds;
      const unsigned NumShared = i ? SharedTypeIds(LP, LandingPads[i-1]) : 0;
      unsigned SizeSiteActions = 0;

      if (NumShared < TypeIds.size()) {
        unsigned SizeAction = 0;
        ActionEntry *PrevAction = 0;

        if (NumShared) {
          const unsigned SizePrevIds = LandingPads[i-1]->TypeIds.size();
          assert(Actions.size());
          PrevAction = &Actions.back();
          SizeAction = Asm->SizeSLEB128(PrevAction->NextAction) +
            Asm->SizeSLEB128(PrevAction->ValueForTypeID);
          for (unsigned j = NumShared; j != SizePrevIds; ++j) {
            SizeAction -= Asm->SizeSLEB128(PrevAction->ValueForTypeID);
            SizeAction += -PrevAction->NextAction;
            PrevAction = PrevAction->Previous;
          }
        }

        // Compute the actions.
        for (unsigned I = NumShared, M = TypeIds.size(); I != M; ++I) {
          int TypeID = TypeIds[I];
          assert(-1-TypeID < (int)FilterOffsets.size() && "Unknown filter id!");
          int ValueForTypeID = TypeID < 0 ? FilterOffsets[-1 - TypeID] : TypeID;
          unsigned SizeTypeID = Asm->SizeSLEB128(ValueForTypeID);

          int NextAction = SizeAction ? -(SizeAction + SizeTypeID) : 0;
          SizeAction = SizeTypeID + Asm->SizeSLEB128(NextAction);
          SizeSiteActions += SizeAction;

          ActionEntry Action = {ValueForTypeID, NextAction, PrevAction};
          Actions.push_back(Action);

          PrevAction = &Actions.back();
        }

        // Record the first action of the landing pad site.
        FirstAction = SizeActions + SizeSiteActions - SizeAction + 1;
      } // else identical - re-use previous FirstAction

      FirstActions.push_back(FirstAction);

      // Compute this sites contribution to size.
      SizeActions += SizeSiteActions;
    }

    // Compute the call-site table.  The entry for an invoke has a try-range
    // containing the call, a non-zero landing pad and an appropriate action.
    // The entry for an ordinary call has a try-range containing the call and
    // zero for the landing pad and the action.  Calls marked 'nounwind' have
    // no entry and must not be contained in the try-range of any entry - they
    // form gaps in the table.  Entries must be ordered by try-range address.
    SmallVector<CallSiteEntry, 64> CallSites;

    RangeMapType PadMap;
    // Invokes and nounwind calls have entries in PadMap (due to being bracketed
    // by try-range labels when lowered).  Ordinary calls do not, so appropriate
    // try-ranges for them need be deduced.
    for (unsigned i = 0, N = LandingPads.size(); i != N; ++i) {
      const LandingPadInfo *LandingPad = LandingPads[i];
      for (unsigned j = 0, E = LandingPad->BeginLabels.size(); j != E; ++j) {
        unsigned BeginLabel = LandingPad->BeginLabels[j];
        assert(!PadMap.count(BeginLabel) && "Duplicate landing pad labels!");
        PadRange P = { i, j };
        PadMap[BeginLabel] = P;
      }
    }

    // The end label of the previous invoke or nounwind try-range.
    unsigned LastLabel = 0;

    // Whether there is a potentially throwing instruction (currently this means
    // an ordinary call) between the end of the previous try-range and now.
    bool SawPotentiallyThrowing = false;

    // Whether the last callsite entry was for an invoke.
    bool PreviousIsInvoke = false;

    // Visit all instructions in order of address.
    for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
         I != E; ++I) {
      for (MachineBasicBlock::const_iterator MI = I->begin(), E = I->end();
           MI != E; ++MI) {
        if (MI->getOpcode() != TargetInstrInfo::LABEL) {
          SawPotentiallyThrowing |= MI->getDesc().isCall();
          continue;
        }

        unsigned BeginLabel = MI->getOperand(0).getImm();
        assert(BeginLabel && "Invalid label!");

        // End of the previous try-range?
        if (BeginLabel == LastLabel)
          SawPotentiallyThrowing = false;

        // Beginning of a new try-range?
        RangeMapType::iterator L = PadMap.find(BeginLabel);
        if (L == PadMap.end())
          // Nope, it was just some random label.
          continue;

        PadRange P = L->second;
        const LandingPadInfo *LandingPad = LandingPads[P.PadIndex];

        assert(BeginLabel == LandingPad->BeginLabels[P.RangeIndex] &&
               "Inconsistent landing pad map!");

        // If some instruction between the previous try-range and this one may
        // throw, create a call-site entry with no landing pad for the region
        // between the try-ranges.
        if (SawPotentiallyThrowing) {
          CallSiteEntry Site = {LastLabel, BeginLabel, 0, 0};
          CallSites.push_back(Site);
          PreviousIsInvoke = false;
        }

        LastLabel = LandingPad->EndLabels[P.RangeIndex];
        assert(BeginLabel && LastLabel && "Invalid landing pad!");

        if (LandingPad->LandingPadLabel) {
          // This try-range is for an invoke.
          CallSiteEntry Site = {BeginLabel, LastLabel,
            LandingPad->LandingPadLabel, FirstActions[P.PadIndex]};

          // Try to merge with the previous call-site.
          if (PreviousIsInvoke) {
            CallSiteEntry &Prev = CallSites[CallSites.size()-1];
            if (Site.PadLabel == Prev.PadLabel && Site.Action == Prev.Action) {
              // Extend the range of the previous entry.
              Prev.EndLabel = Site.EndLabel;
              continue;
            }
          }

          // Otherwise, create a new call-site.
          CallSites.push_back(Site);
          PreviousIsInvoke = true;
        } else {
          // Create a gap.
          PreviousIsInvoke = false;
        }
      }
    }
    // If some instruction between the previous try-range and the end of the
    // function may throw, create a call-site entry with no landing pad for the
    // region following the try-range.
    if (SawPotentiallyThrowing) {
      CallSiteEntry Site = {LastLabel, 0, 0, 0};
      CallSites.push_back(Site);
    }

    // Final tallies.
    unsigned SizeSites = CallSites.size() * (sizeof(int32_t) + // Site start.
                                             sizeof(int32_t) + // Site length.
                                             sizeof(int32_t)); // Landing pad.
    for (unsigned i = 0, e = CallSites.size(); i < e; ++i)
      SizeSites += Asm->SizeULEB128(CallSites[i].Action);

    unsigned SizeTypes = TypeInfos.size() * TD->getPointerSize();

    unsigned TypeOffset = sizeof(int8_t) + // Call site format
                          Asm->SizeULEB128(SizeSites) + // Call-site table length
                          SizeSites + SizeActions + SizeTypes;

    unsigned TotalSize = sizeof(int8_t) + // LPStart format
                         sizeof(int8_t) + // TType format
                         Asm->SizeULEB128(TypeOffset) + // TType base offset
                         TypeOffset;

    unsigned SizeAlign = (4 - TotalSize) & 3;

    // Begin the exception table.
    Asm->SwitchToDataSection(TAI->getDwarfExceptionSection());
    O << "GCC_except_table" << SubprogramCount << ":\n";
    Asm->EmitAlignment(2);
    for (unsigned i = 0; i != SizeAlign; ++i) {
      Asm->EmitInt8(0);
      Asm->EOL("Padding");
    }
    EmitLabel("exception", SubprogramCount);

    // Emit the header.
    Asm->EmitInt8(DW_EH_PE_omit);
    Asm->EOL("LPStart format (DW_EH_PE_omit)");
    Asm->EmitInt8(DW_EH_PE_absptr);
    Asm->EOL("TType format (DW_EH_PE_absptr)");
    Asm->EmitULEB128Bytes(TypeOffset);
    Asm->EOL("TType base offset");
    Asm->EmitInt8(DW_EH_PE_udata4);
    Asm->EOL("Call site format (DW_EH_PE_udata4)");
    Asm->EmitULEB128Bytes(SizeSites);
    Asm->EOL("Call-site table length");

    // Emit the landing pad site information.
    for (unsigned i = 0; i < CallSites.size(); ++i) {
      CallSiteEntry &S = CallSites[i];
      const char *BeginTag;
      unsigned BeginNumber;

      if (!S.BeginLabel) {
        BeginTag = "eh_func_begin";
        BeginNumber = SubprogramCount;
      } else {
        BeginTag = "label";
        BeginNumber = S.BeginLabel;
      }

      EmitSectionOffset(BeginTag, "eh_func_begin", BeginNumber, SubprogramCount,
                        TAI->getShortenEHDataOn64Bit(), true);
      Asm->EOL("Region start");

      if (!S.EndLabel) {
        EmitDifference("eh_func_end", SubprogramCount, BeginTag, BeginNumber,
                       TAI->getShortenEHDataOn64Bit());
      } else {
        EmitDifference("label", S.EndLabel, BeginTag, BeginNumber, 
                       TAI->getShortenEHDataOn64Bit());
      }
      Asm->EOL("Region length");

      if (!S.PadLabel) {
        if (TD->getPointerSize() == sizeof(int32_t) || TAI->getShortenEHDataOn64Bit())
          Asm->EmitInt32(0);
        else
          Asm->EmitInt64(0);
      } else {
        EmitSectionOffset("label", "eh_func_begin", S.PadLabel, SubprogramCount,
                          TAI->getShortenEHDataOn64Bit(), true);
      }
      Asm->EOL("Landing pad");

      Asm->EmitULEB128Bytes(S.Action);
      Asm->EOL("Action");
    }

    // Emit the actions.
    for (unsigned I = 0, N = Actions.size(); I != N; ++I) {
      ActionEntry &Action = Actions[I];

      Asm->EmitSLEB128Bytes(Action.ValueForTypeID);
      Asm->EOL("TypeInfo index");
      Asm->EmitSLEB128Bytes(Action.NextAction);
      Asm->EOL("Next action");
    }

    // Emit the type ids.
    for (unsigned M = TypeInfos.size(); M; --M) {
      GlobalVariable *GV = TypeInfos[M - 1];

      PrintRelDirective();

      if (GV)
        O << Asm->getGlobalLinkName(GV);
      else
        O << "0";

      Asm->EOL("TypeInfo");
    }

    // Emit the filter typeids.
    for (unsigned j = 0, M = FilterIds.size(); j < M; ++j) {
      unsigned TypeID = FilterIds[j];
      Asm->EmitULEB128Bytes(TypeID);
      Asm->EOL("Filter TypeInfo index");
    }

    Asm->EmitAlignment(2);
  }

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfException(std::ostream &OS, AsmPrinter *A, const TargetAsmInfo *T)
  : Dwarf(OS, A, T, "eh")
  , shouldEmit(false)
  {}
  
  virtual ~DwarfException() {}

  /// SetModuleInfo - Set machine module information when it's known that pass
  /// manager has created it.  Set by the target AsmPrinter.
  void SetModuleInfo(MachineModuleInfo *mmi) {
    MMI = mmi;
  }

  /// BeginModule - Emit all exception information that should come prior to the
  /// content.
  void BeginModule(Module *M) {
    this->M = M;
  }

  /// EndModule - Emit all exception information that should come after the
  /// content.
  void EndModule() {
    if (!shouldEmit) return;

    const std::vector<Function *> Personalities = MMI->getPersonalities();
    for (unsigned i =0; i < Personalities.size(); ++i)
      EmitCommonEHFrame(Personalities[i], i);

    for (std::vector<FunctionEHFrameInfo>::iterator I = EHFrames.begin(),
           E = EHFrames.end(); I != E; ++I)
      EmitEHFrame(*I);
  }

  /// BeginFunction - Gather pre-function exception information.  Assumes being 
  /// emitted immediately after the function entry point.
  void BeginFunction(MachineFunction *MF) {
    this->MF = MF;
    
    if (MMI &&
        ExceptionHandling &&
        TAI->doesSupportExceptionHandling()) {
      shouldEmit = true;
      // Assumes in correct section after the entry point.
      EmitLabel("eh_func_begin", ++SubprogramCount);
    }
  }

  /// EndFunction - Gather and emit post-function exception information.
  ///
  void EndFunction() {
    if (!shouldEmit) return;

    EmitLabel("eh_func_end", SubprogramCount);
    EmitExceptionTable();

    // Save EH frame information
    EHFrames.
      push_back(FunctionEHFrameInfo(getAsm()->getCurrentFunctionEHName(MF),
                                    SubprogramCount,
                                    MMI->getPersonalityIndex(),
                                    MF->getFrameInfo()->hasCalls(),
                                    !MMI->getLandingPads().empty(),
                                    MMI->getFrameMoves(),
                                    MF->getFunction()));
  }
};

} // End of namespace llvm

//===----------------------------------------------------------------------===//

/// Emit - Print the abbreviation using the specified Dwarf writer.
///
void DIEAbbrev::Emit(const DwarfDebug &DD) const {
  // Emit its Dwarf tag type.
  DD.getAsm()->EmitULEB128Bytes(Tag);
  DD.getAsm()->EOL(TagString(Tag));
  
  // Emit whether it has children DIEs.
  DD.getAsm()->EmitULEB128Bytes(ChildrenFlag);
  DD.getAsm()->EOL(ChildrenString(ChildrenFlag));
  
  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    const DIEAbbrevData &AttrData = Data[i];
    
    // Emit attribute type.
    DD.getAsm()->EmitULEB128Bytes(AttrData.getAttribute());
    DD.getAsm()->EOL(AttributeString(AttrData.getAttribute()));
    
    // Emit form type.
    DD.getAsm()->EmitULEB128Bytes(AttrData.getForm());
    DD.getAsm()->EOL(FormEncodingString(AttrData.getForm()));
  }

  // Mark end of abbreviation.
  DD.getAsm()->EmitULEB128Bytes(0); DD.getAsm()->EOL("EOM(1)");
  DD.getAsm()->EmitULEB128Bytes(0); DD.getAsm()->EOL("EOM(2)");
}

#ifndef NDEBUG
void DIEAbbrev::print(std::ostream &O) {
  O << "Abbreviation @"
    << std::hex << (intptr_t)this << std::dec
    << "  "
    << TagString(Tag)
    << " "
    << ChildrenString(ChildrenFlag)
    << "\n";
  
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    O << "  "
      << AttributeString(Data[i].getAttribute())
      << "  "
      << FormEncodingString(Data[i].getForm())
      << "\n";
  }
}
void DIEAbbrev::dump() { print(cerr); }
#endif

//===----------------------------------------------------------------------===//

#ifndef NDEBUG
void DIEValue::dump() {
  print(cerr);
}
#endif

//===----------------------------------------------------------------------===//

/// EmitValue - Emit integer of appropriate size.
///
void DIEInteger::EmitValue(DwarfDebug &DD, unsigned Form) {
  switch (Form) {
  case DW_FORM_flag:  // Fall thru
  case DW_FORM_ref1:  // Fall thru
  case DW_FORM_data1: DD.getAsm()->EmitInt8(Integer);         break;
  case DW_FORM_ref2:  // Fall thru
  case DW_FORM_data2: DD.getAsm()->EmitInt16(Integer);        break;
  case DW_FORM_ref4:  // Fall thru
  case DW_FORM_data4: DD.getAsm()->EmitInt32(Integer);        break;
  case DW_FORM_ref8:  // Fall thru
  case DW_FORM_data8: DD.getAsm()->EmitInt64(Integer);        break;
  case DW_FORM_udata: DD.getAsm()->EmitULEB128Bytes(Integer); break;
  case DW_FORM_sdata: DD.getAsm()->EmitSLEB128Bytes(Integer); break;
  default: assert(0 && "DIE Value form not supported yet");   break;
  }
}

/// SizeOf - Determine size of integer value in bytes.
///
unsigned DIEInteger::SizeOf(const DwarfDebug &DD, unsigned Form) const {
  switch (Form) {
  case DW_FORM_flag:  // Fall thru
  case DW_FORM_ref1:  // Fall thru
  case DW_FORM_data1: return sizeof(int8_t);
  case DW_FORM_ref2:  // Fall thru
  case DW_FORM_data2: return sizeof(int16_t);
  case DW_FORM_ref4:  // Fall thru
  case DW_FORM_data4: return sizeof(int32_t);
  case DW_FORM_ref8:  // Fall thru
  case DW_FORM_data8: return sizeof(int64_t);
  case DW_FORM_udata: return DD.getAsm()->SizeULEB128(Integer);
  case DW_FORM_sdata: return DD.getAsm()->SizeSLEB128(Integer);
  default: assert(0 && "DIE Value form not supported yet"); break;
  }
  return 0;
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit string value.
///
void DIEString::EmitValue(DwarfDebug &DD, unsigned Form) {
  DD.getAsm()->EmitString(String);
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIEDwarfLabel::EmitValue(DwarfDebug &DD, unsigned Form) {
  bool IsSmall = Form == DW_FORM_data4;
  DD.EmitReference(Label, false, IsSmall);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIEDwarfLabel::SizeOf(const DwarfDebug &DD, unsigned Form) const {
  if (Form == DW_FORM_data4) return 4;
  return DD.getTargetData()->getPointerSize();
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIEObjectLabel::EmitValue(DwarfDebug &DD, unsigned Form) {
  bool IsSmall = Form == DW_FORM_data4;
  DD.EmitReference(Label, false, IsSmall);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIEObjectLabel::SizeOf(const DwarfDebug &DD, unsigned Form) const {
  if (Form == DW_FORM_data4) return 4;
  return DD.getTargetData()->getPointerSize();
}
    
//===----------------------------------------------------------------------===//

/// EmitValue - Emit delta value.
///
void DIEDelta::EmitValue(DwarfDebug &DD, unsigned Form) {
  bool IsSmall = Form == DW_FORM_data4;
  DD.EmitDifference(LabelHi, LabelLo, IsSmall);
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIEDelta::SizeOf(const DwarfDebug &DD, unsigned Form) const {
  if (Form == DW_FORM_data4) return 4;
  return DD.getTargetData()->getPointerSize();
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit debug information entry offset.
///
void DIEntry::EmitValue(DwarfDebug &DD, unsigned Form) {
  DD.getAsm()->EmitInt32(Entry->getOffset());
}
    
//===----------------------------------------------------------------------===//

/// ComputeSize - calculate the size of the block.
///
unsigned DIEBlock::ComputeSize(DwarfDebug &DD) {
  if (!Size) {
    const std::vector<DIEAbbrevData> &AbbrevData = Abbrev.getData();
    
    for (unsigned i = 0, N = Values.size(); i < N; ++i) {
      Size += Values[i]->SizeOf(DD, AbbrevData[i].getForm());
    }
  }
  return Size;
}

/// EmitValue - Emit block data.
///
void DIEBlock::EmitValue(DwarfDebug &DD, unsigned Form) {
  switch (Form) {
  case DW_FORM_block1: DD.getAsm()->EmitInt8(Size);         break;
  case DW_FORM_block2: DD.getAsm()->EmitInt16(Size);        break;
  case DW_FORM_block4: DD.getAsm()->EmitInt32(Size);        break;
  case DW_FORM_block:  DD.getAsm()->EmitULEB128Bytes(Size); break;
  default: assert(0 && "Improper form for block");          break;
  }
  
  const std::vector<DIEAbbrevData> &AbbrevData = Abbrev.getData();

  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    DD.getAsm()->EOL();
    Values[i]->EmitValue(DD, AbbrevData[i].getForm());
  }
}

/// SizeOf - Determine size of block data in bytes.
///
unsigned DIEBlock::SizeOf(const DwarfDebug &DD, unsigned Form) const {
  switch (Form) {
  case DW_FORM_block1: return Size + sizeof(int8_t);
  case DW_FORM_block2: return Size + sizeof(int16_t);
  case DW_FORM_block4: return Size + sizeof(int32_t);
  case DW_FORM_block: return Size + DD.getAsm()->SizeULEB128(Size);
  default: assert(0 && "Improper form for block"); break;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
/// DIE Implementation

DIE::~DIE() {
  for (unsigned i = 0, N = Children.size(); i < N; ++i)
    delete Children[i];
}
  
/// AddSiblingOffset - Add a sibling offset field to the front of the DIE.
///
void DIE::AddSiblingOffset() {
  DIEInteger *DI = new DIEInteger(0);
  Values.insert(Values.begin(), DI);
  Abbrev.AddFirstAttribute(DW_AT_sibling, DW_FORM_ref4);
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIE::Profile(FoldingSetNodeID &ID) {
  Abbrev.Profile(ID);
  
  for (unsigned i = 0, N = Children.size(); i < N; ++i)
    ID.AddPointer(Children[i]);

  for (unsigned j = 0, M = Values.size(); j < M; ++j)
    ID.AddPointer(Values[j]);
}

#ifndef NDEBUG
void DIE::print(std::ostream &O, unsigned IncIndent) {
  static unsigned IndentCount = 0;
  IndentCount += IncIndent;
  const std::string Indent(IndentCount, ' ');
  bool isBlock = Abbrev.getTag() == 0;
  
  if (!isBlock) {
    O << Indent
      << "Die: "
      << "0x" << std::hex << (intptr_t)this << std::dec
      << ", Offset: " << Offset
      << ", Size: " << Size
      << "\n"; 
    
    O << Indent
      << TagString(Abbrev.getTag())
      << " "
      << ChildrenString(Abbrev.getChildrenFlag());
  } else {
    O << "Size: " << Size;
  }
  O << "\n";

  const std::vector<DIEAbbrevData> &Data = Abbrev.getData();
  
  IndentCount += 2;
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    O << Indent;
    if (!isBlock) {
      O << AttributeString(Data[i].getAttribute());
    } else {
      O << "Blk[" << i << "]";
    }
    O <<  "  "
      << FormEncodingString(Data[i].getForm())
      << " ";
    Values[i]->print(O);
    O << "\n";
  }
  IndentCount -= 2;

  for (unsigned j = 0, M = Children.size(); j < M; ++j) {
    Children[j]->print(O, 4);
  }
  
  if (!isBlock) O << "\n";
  IndentCount -= IncIndent;
}

void DIE::dump() {
  print(cerr);
}
#endif

//===----------------------------------------------------------------------===//
/// DwarfWriter Implementation
///

DwarfWriter::DwarfWriter(std::ostream &OS, AsmPrinter *A,
                         const TargetAsmInfo *T) {
  DE = new DwarfException(OS, A, T);
  DD = new DwarfDebug(OS, A, T);
}

DwarfWriter::~DwarfWriter() {
  delete DE;
  delete DD;
}

/// SetModuleInfo - Set machine module info when it's known that pass manager
/// has created it.  Set by the target AsmPrinter.
void DwarfWriter::SetModuleInfo(MachineModuleInfo *MMI) {
  DD->SetModuleInfo(MMI);
  DE->SetModuleInfo(MMI);
}

/// BeginModule - Emit all Dwarf sections that should come prior to the
/// content.
void DwarfWriter::BeginModule(Module *M) {
  DE->BeginModule(M);
  DD->BeginModule(M);
}

/// EndModule - Emit all Dwarf sections that should come after the content.
///
void DwarfWriter::EndModule() {
  DE->EndModule();
  DD->EndModule();
}

/// BeginFunction - Gather pre-function debug information.  Assumes being 
/// emitted immediately after the function entry point.
void DwarfWriter::BeginFunction(MachineFunction *MF) {
  DE->BeginFunction(MF);
  DD->BeginFunction(MF);
}

/// EndFunction - Gather and emit post-function debug information.
///
void DwarfWriter::EndFunction() {
  DD->EndFunction();
  DE->EndFunction();
  
  if (MachineModuleInfo *MMI = DD->getMMI() ? DD->getMMI() : DE->getMMI()) {
    // Clear function debug information.
    MMI->EndFunction();
  }
}

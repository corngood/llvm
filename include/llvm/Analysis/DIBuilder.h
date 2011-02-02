//===--- llvm/Analysis/DIBuilder.h - Debug Information Builder --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DIBuilder that is useful for creating debugging 
// information entries in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DIBUILDER_H
#define LLVM_ANALYSIS_DIBUILDER_H

#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class BasicBlock;
  class Instruction;
  class Function;
  class Module;
  class Value;
  class LLVMContext;
  class MDNode;
  class StringRef;
  class DIDescriptor;
  class DIFile;
  class DIEnumerator;
  class DIType;
  class DIArray;
  class DIGlobalVariable;
  class DINameSpace;
  class DIVariable;
  class DISubrange;
  class DILexicalBlock;
  class DISubprogram;
  class DITemplateTypeParameter;
  class DITemplateValueParameter;

  class DIBuilder {
    private:
    Module &M;
    LLVMContext & VMContext;
    MDNode *TheCU;

    Function *DeclareFn;     // llvm.dbg.declare
    Function *ValueFn;       // llvm.dbg.value

    DIBuilder(const DIBuilder &);       // DO NOT IMPLEMENT
    void operator=(const DIBuilder &);  // DO NOT IMPLEMENT

    public:
    explicit DIBuilder(Module &M);
    const MDNode *getCU() { return TheCU; }

    /// CreateCompileUnit - A CompileUnit provides an anchor for all debugging
    /// information generated during this instance of compilation.
    /// @param Lang     Source programming language, eg. dwarf::DW_LANG_C99
    /// @param File     File name
    /// @param Dir      Directory
    /// @param Producer String identify producer of debugging information. 
    ///                 Usuall this is a compiler version string.
    /// @param isOptimized A boolean flag which indicates whether optimization
    ///                    is ON or not.
    /// @param Flags    This string lists command line options. This string is 
    ///                 directly embedded in debug info output which may be used
    ///                 by a tool analyzing generated debugging information.
    /// @param RV       This indicates runtime version for languages like 
    ///                 Objective-C.
    void CreateCompileUnit(unsigned Lang, StringRef File, StringRef Dir, 
                           StringRef Producer,
                           bool isOptimized, StringRef Flags, unsigned RV);

    /// CreateFile - Create a file descriptor to hold debugging information
    /// for a file.
    DIFile CreateFile(StringRef Filename, StringRef Directory);
                           
    /// CreateEnumerator - Create a single enumerator value.
    DIEnumerator CreateEnumerator(StringRef Name, uint64_t Val);

    /// CreateBasicType - Create debugging information entry for a basic 
    /// type.
    /// @param Name        Type name.
    /// @param SizeInBits  Size of the type.
    /// @param AlignInBits Type alignment.
    /// @param Encoding    DWARF encoding code, e.g. dwarf::DW_ATE_float.
    DIType CreateBasicType(StringRef Name, uint64_t SizeInBits, 
                           uint64_t AlignInBits, unsigned Encoding);

    /// CreateQualifiedType - Create debugging information entry for a qualified
    /// type, e.g. 'const int'.
    /// @param Tag         Tag identifing type, e.g. dwarf::TAG_volatile_type
    /// @param FromTy      Base Type.
    DIType CreateQualifiedType(unsigned Tag, DIType FromTy);

    /// CreatePointerType - Create debugging information entry for a pointer.
    /// @param PointeeTy   Type pointed by this pointer.
    /// @param SizeInBits  Size.
    /// @param AlignInBits Alignment. (optional)
    /// @param Name        Pointer type name. (optional)
    DIType CreatePointerType(DIType PointeeTy, uint64_t SizeInBits,
                             uint64_t AlignInBits = 0, 
                             StringRef Name = StringRef());

    /// CreateReferenceType - Create debugging information entry for a c++
    /// style reference.
    DIType CreateReferenceType(DIType RTy);

    /// CreateTypedef - Create debugging information entry for a typedef.
    /// @param Ty          Original type.
    /// @param Name        Typedef name.
    /// @param File        File where this type is defined.
    /// @param LineNo      Line number.
    DIType CreateTypedef(DIType Ty, StringRef Name, DIFile File, 
                         unsigned LineNo);

    /// CreateFriend - Create debugging information entry for a 'friend'.
    DIType CreateFriend(DIType Ty, DIType FriendTy);

    /// CreateInheritance - Create debugging information entry to establish
    /// inheritance relationship between two types.
    /// @param Ty           Original type.
    /// @param BaseTy       Base type. Ty is inherits from base.
    /// @param BaseOffset   Base offset.
    /// @param Flags        Flags to describe inheritance attribute, 
    ///                     e.g. private
    DIType CreateInheritance(DIType Ty, DIType BaseTy, uint64_t BaseOffset,
                             unsigned Flags);

    /// CreateMemberType - Create debugging information entry for a member.
    /// @param Name         Member name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Ty           Parent type.
    DIType CreateMemberType(StringRef Name, DIFile File,
                            unsigned LineNo, uint64_t SizeInBits, 
                            uint64_t AlignInBits, uint64_t OffsetInBits, 
                            unsigned Flags, DIType Ty);

    /// CreateClassType - Create debugging information entry for a class.
    /// @param Scope        Scope in which this class is defined.
    /// @param Name         class name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     class members.
    /// @param VTableHolder Debug info of the base class that contains vtable
    ///                     for this type. This is used in 
    ///                     DW_AT_containing_type. See DWARF documentation
    ///                     for more info.
    /// @param TemplateParms Template type parameters.
    DIType CreateClassType(DIDescriptor Scope, StringRef Name, DIFile File,
                           unsigned LineNumber, uint64_t SizeInBits,
                           uint64_t AlignInBits, uint64_t OffsetInBits,
                           unsigned Flags, DIType DerivedFrom, 
                           DIArray Elements, MDNode *VTableHolder = 0,
                           MDNode *TemplateParms = 0);

    /// CreateStructType - Create debugging information entry for a struct.
    /// @param Scope        Scope in which this struct is defined.
    /// @param Name         Struct name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     Struct elements.
    /// @param RunTimeLang  Optional parameter, Objective-C runtime version.
    DIType CreateStructType(DIDescriptor Scope, StringRef Name, DIFile File,
                            unsigned LineNumber, uint64_t SizeInBits,
                            uint64_t AlignInBits, unsigned Flags,
                            DIArray Elements, unsigned RunTimeLang = 0);

    /// CreateUnionType - Create debugging information entry for an union.
    /// @param Scope        Scope in which this union is defined.
    /// @param Name         Union name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     Union elements.
    /// @param RunTimeLang  Optional parameter, Objective-C runtime version.
    DIType CreateUnionType(DIDescriptor Scope, StringRef Name, DIFile File,
                           unsigned LineNumber, uint64_t SizeInBits,
                           uint64_t AlignInBits, unsigned Flags,
                           DIArray Elements, unsigned RunTimeLang = 0);

    /// CreateTemplateTypeParameter - Create debugging information for template
    /// type parameter.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Type parameter name.
    /// @param Ty           Parameter type.
    /// @param File         File where this type parameter is defined.
    /// @param LineNo       Line number.
    /// @param ColumnNo     Column Number.
    DITemplateTypeParameter
    CreateTemplateTypeParameter(DIDescriptor Scope, StringRef Name, DIType Ty,
                                MDNode *File = 0, unsigned LineNo = 0,
                                unsigned ColumnNo = 0);

    /// CreateTemplateValueParameter - Create debugging information for template
    /// value parameter.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Value parameter name.
    /// @param Ty           Parameter type.
    /// @param Value        Constant parameter value.
    /// @param File         File where this type parameter is defined.
    /// @param LineNo       Line number.
    /// @param ColumnNo     Column Number.
    DITemplateValueParameter
    CreateTemplateValueParameter(DIDescriptor Scope, StringRef Name, DIType Ty,
                                 uint64_t Value,
                                 MDNode *File = 0, unsigned LineNo = 0,
                                 unsigned ColumnNo = 0);

    /// CreateArrayType - Create debugging information entry for an array.
    /// @param Size         Array size.
    /// @param AlignInBits  Alignment.
    /// @param Ty           Element type.
    /// @param Subscripts   Subscripts.
    DIType CreateArrayType(uint64_t Size, uint64_t AlignInBits, 
                           DIType Ty, DIArray Subscripts);

    /// CreateVectorType - Create debugging information entry for a vector type.
    /// @param Size         Array size.
    /// @param AlignInBits  Alignment.
    /// @param Ty           Element type.
    /// @param Subscripts   Subscripts.
    DIType CreateVectorType(uint64_t Size, uint64_t AlignInBits, 
                            DIType Ty, DIArray Subscripts);

    /// CreateEnumerationType - Create debugging information entry for an 
    /// enumeration.
    /// @param Scope        Scope in which this enumeration is defined.
    /// @param Name         Union name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Elements     Enumeration elements.
    DIType CreateEnumerationType(DIDescriptor Scope, StringRef Name, 
                                 DIFile File, unsigned LineNumber, 
                                 uint64_t SizeInBits, 
                                 uint64_t AlignInBits, DIArray Elements);

    /// CreateSubroutineType - Create subroutine type.
    /// @param File          File in which this subroutine is defined.
    /// @param ParamterTypes An array of subroutine parameter types. This
    ///                      includes return type at 0th index.
    DIType CreateSubroutineType(DIFile File, DIArray ParameterTypes);

    /// CreateArtificialType - Create a new DIType with "artificial" flag set.
    DIType CreateArtificialType(DIType Ty);

    /// CreateTemporaryType - Create a temporary forward-declared type.
    DIType CreateTemporaryType();
    DIType CreateTemporaryType(DIFile F);

    /// RetainType - Retain DIType in a module even if it is not referenced 
    /// through debug info anchors.
    void RetainType(DIType T);

    /// CreateUnspecifiedParameter - Create unspeicified type descriptor
    /// for a subroutine type.
    DIDescriptor CreateUnspecifiedParameter();

    /// GetOrCreateArray - Get a DIArray, create one if required.
    DIArray GetOrCreateArray(Value *const *Elements, unsigned NumElements);

    /// GetOrCreateSubrange - Create a descriptor for a value range.  This
    /// implicitly uniques the values returned.
    DISubrange GetOrCreateSubrange(int64_t Lo, int64_t Hi);

    /// CreateGlobalVariable - Create a new descriptor for the specified global.
    /// @param Name        Name of the variable.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type.
    /// @param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// @param Val         llvm::Value of the variable.
    DIGlobalVariable
    CreateGlobalVariable(StringRef Name, DIFile File, unsigned LineNo,
                         DIType Ty, bool isLocalToUnit, llvm::Value *Val);


    /// CreateStaticVariable - Create a new descriptor for the specified 
    /// variable.
    /// @param Conext      Variable scope. 
    /// @param Name        Name of the variable.
    /// @param LinakgeName Mangled  name of the variable.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type.
    /// @param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// @param Val         llvm::Value of the variable.
    DIGlobalVariable
    CreateStaticVariable(DIDescriptor Context, StringRef Name, 
                         StringRef LinkageName, DIFile File, unsigned LineNo, 
                         DIType Ty, bool isLocalToUnit, llvm::Value *Val);


    /// CreateLocalVariable - Create a new descriptor for the specified 
    /// local variable.
    /// @param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// @param Scope       Variable scope.
    /// @param Name        Variable name.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type
    /// @param AlwaysPreserve Boolean. Set to true if debug info for this
    ///                       variable should be preserved in optimized build.
    /// @param Flags          Flags, e.g. artificial variable.
    DIVariable CreateLocalVariable(unsigned Tag, DIDescriptor Scope,
                                   StringRef Name,
                                   DIFile File, unsigned LineNo,
                                   DIType Ty, bool AlwaysPreserve = false,
                                   unsigned Flags = 0);


    /// CreateComplexVariable - Create a new descriptor for the specified
    /// variable which has a complex address expression for its address.
    /// @param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// @param Scope       Variable scope.
    /// @param Name        Variable name.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type
    /// @param Addr        A pointer to a vector of complex address operations.
    /// @param NumAddr     Num of address operations in the vector.
    DIVariable CreateComplexVariable(unsigned Tag, DIDescriptor Scope,
                                     StringRef Name, DIFile F, unsigned LineNo,
                                     DIType Ty, Value *const *Addr,
                                     unsigned NumAddr);

    /// CreateFunction - Create a new descriptor for the specified subprogram.
    /// See comments in DISubprogram for descriptions of these fields.
    /// @param Scope         Function scope.
    /// @param Name          Function name.
    /// @param LinkageName   Mangled function name.
    /// @param File          File where this variable is defined.
    /// @param LineNo        Line number.
    /// @param Ty            Function type.
    /// @param isLocalToUnit True if this function is not externally visible..
    /// @param isDefinition  True if this is a function definition.
    /// @param Flags         e.g. is this function prototyped or not.
    ///                      This flags are used to emit dwarf attributes.
    /// @param isOptimized   True if optimization is ON.
    /// @param Fn            llvm::Function pointer.
    DISubprogram CreateFunction(DIDescriptor Scope, StringRef Name,
                                StringRef LinkageName,
                                DIFile File, unsigned LineNo,
                                DIType Ty, bool isLocalToUnit,
                                bool isDefinition,
                                unsigned Flags = 0,
                                bool isOptimized = false,
                                Function *Fn = 0);

    /// CreateMethod - Create a new descriptor for the specified C++ method.
    /// See comments in DISubprogram for descriptions of these fields.
    /// @param Scope         Function scope.
    /// @param Name          Function name.
    /// @param LinkageName   Mangled function name.
    /// @param File          File where this variable is defined.
    /// @param LineNo        Line number.
    /// @param Ty            Function type.
    /// @param isLocalToUnit True if this function is not externally visible..
    /// @param isDefinition  True if this is a function definition.
    /// @param Virtuality    Attributes describing virutallness. e.g. pure 
    ///                      virtual function.
    /// @param VTableIndex   Index no of this method in virtual table.
    /// @param VTableHolder  Type that holds vtable.
    /// @param Flags         e.g. is this function prototyped or not.
    ///                      This flags are used to emit dwarf attributes.
    /// @param isOptimized   True if optimization is ON.
    /// @param Fn            llvm::Function pointer.
    DISubprogram CreateMethod(DIDescriptor Scope, StringRef Name,
                              StringRef LinkageName,
                              DIFile File, unsigned LineNo,
                              DIType Ty, bool isLocalToUnit,
                              bool isDefinition,
                              unsigned Virtuality = 0, unsigned VTableIndex = 0,
                              MDNode *VTableHolder = 0,
                              unsigned Flags = 0,
                              bool isOptimized = false,
                              Function *Fn = 0);

    /// CreateNameSpace - This creates new descriptor for a namespace
    /// with the specified parent scope.
    /// @param Scope       Namespace scope
    /// @param Name        Name of this namespace
    /// @param File        Source file
    /// @param LineNo      Line number
    DINameSpace CreateNameSpace(DIDescriptor Scope, StringRef Name,
                                DIFile File, unsigned LineNo);


    /// CreateLexicalBlock - This creates a descriptor for a lexical block
    /// with the specified parent context.
    /// @param Scope       Parent lexical scope.
    /// @param File        Source file
    /// @param Line        Line number
    /// @param Col         Column number
    DILexicalBlock CreateLexicalBlock(DIDescriptor Scope, DIFile File,
                                      unsigned Line, unsigned Col);

    /// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage     llvm::Value of the variable
    /// @param VarInfo     Variable's debug info descriptor.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *InsertDeclare(llvm::Value *Storage, DIVariable VarInfo,
                               BasicBlock *InsertAtEnd);

    /// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage      llvm::Value of the variable
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *InsertDeclare(llvm::Value *Storage, DIVariable VarInfo,
                               Instruction *InsertBefore);


    /// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *InsertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DIVariable VarInfo, 
                                         BasicBlock *InsertAtEnd);
    
    /// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *InsertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DIVariable VarInfo, 
                                         Instruction *InsertBefore);

  };
} // end namespace llvm

#endif

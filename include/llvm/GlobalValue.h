//===-- llvm/GlobalValue.h - Class to represent a global value --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a common base class of all globally definable objects.  As such,
// it is subclassed by GlobalVariable, GlobalAlias and by Function.  This is
// used because you can do certain things with these global objects that you
// can't do to anything else.  For example, use the address of one as a
// constant.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBALVALUE_H
#define LLVM_GLOBALVALUE_H

#include "llvm/Constant.h"

namespace llvm {

class PointerType;
class Module;

class GlobalValue : public Constant {
  GlobalValue(const GlobalValue &);             // do not implement
public:
  /// @brief An enumeration for the kinds of linkage for global values.
  enum LinkageTypes {
    ExternalLinkage = 0,///< Externally visible function
    AvailableExternallyLinkage, ///< Available for inspection, not emission.
    LinkOnceAnyLinkage, ///< Keep one copy of function when linking (inline)
    LinkOnceODRLinkage, ///< Same, but only replaced by something equivalent.
    WeakAnyLinkage,     ///< Keep one copy of named function when linking (weak)
    WeakODRLinkage,     ///< Same, but only replaced by something equivalent.
    AppendingLinkage,   ///< Special purpose, only applies to global arrays
    InternalLinkage,    ///< Rename collisions when linking (static functions).
    PrivateLinkage,     ///< Like Internal, but omit from symbol table.
    LinkerPrivateLinkage, ///< Like Private, but linker removes.
    DLLImportLinkage,   ///< Function to be imported from DLL
    DLLExportLinkage,   ///< Function to be accessible from DLL.
    ExternalWeakLinkage,///< ExternalWeak linkage description.
    CommonLinkage       ///< Tentative definitions.
  };

  /// @brief An enumeration for the kinds of visibility of global values.
  enum VisibilityTypes {
    DefaultVisibility = 0,  ///< The GV is visible
    HiddenVisibility,       ///< The GV is hidden
    ProtectedVisibility     ///< The GV is protected
  };

protected:
  GlobalValue(const Type *ty, ValueTy vty, Use *Ops, unsigned NumOps,
              LinkageTypes linkage, const Twine &Name)
    : Constant(ty, vty, Ops, NumOps), Parent(0),
      Linkage(linkage), Visibility(DefaultVisibility), Alignment(0) {
    setName(Name);
  }

  Module *Parent;
  // Note: VC++ treats enums as signed, so an extra bit is required to prevent
  // Linkage and Visibility from turning into negative values.
  LinkageTypes Linkage : 5;   // The linkage of this global
  unsigned Visibility : 2;    // The visibility style of this global
  unsigned Alignment : 16;    // Alignment of this symbol, must be power of two
  std::string Section;        // Section to emit this into, empty mean default
public:
  ~GlobalValue() {
    removeDeadConstantUsers();   // remove any dead constants using this.
  }

  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Align) {
    assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
    Alignment = Align;
  }

  VisibilityTypes getVisibility() const { return VisibilityTypes(Visibility); }
  bool hasDefaultVisibility() const { return Visibility == DefaultVisibility; }
  bool hasHiddenVisibility() const { return Visibility == HiddenVisibility; }
  bool hasProtectedVisibility() const {
    return Visibility == ProtectedVisibility;
  }
  void setVisibility(VisibilityTypes V) { Visibility = V; }
  
  bool hasSection() const { return !Section.empty(); }
  const std::string &getSection() const { return Section; }
  void setSection(StringRef S) { Section = S; }
  
  /// If the usage is empty (except transitively dead constants), then this
  /// global value can can be safely deleted since the destructor will
  /// delete the dead constants as well.
  /// @brief Determine if the usage of this global value is empty except
  /// for transitively dead constants.
  bool use_empty_except_constants();

  /// getType - Global values are always pointers.
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(User::getType());
  }

  static LinkageTypes getLinkOnceLinkage(bool ODR) {
    return ODR ? LinkOnceODRLinkage : LinkOnceAnyLinkage;
  }
  static LinkageTypes getWeakLinkage(bool ODR) {
    return ODR ? WeakODRLinkage : WeakAnyLinkage;
  }

  bool hasExternalLinkage() const { return Linkage == ExternalLinkage; }
  bool hasAvailableExternallyLinkage() const {
    return Linkage == AvailableExternallyLinkage;
  }
  bool hasLinkOnceLinkage() const {
    return Linkage == LinkOnceAnyLinkage || Linkage == LinkOnceODRLinkage;
  }
  bool hasWeakLinkage() const {
    return Linkage == WeakAnyLinkage || Linkage == WeakODRLinkage;
  }
  bool hasAppendingLinkage() const { return Linkage == AppendingLinkage; }
  bool hasInternalLinkage() const { return Linkage == InternalLinkage; }
  bool hasPrivateLinkage() const { return Linkage == PrivateLinkage; }
  bool hasLinkerPrivateLinkage() const { return Linkage==LinkerPrivateLinkage; }
  bool hasLocalLinkage() const {
    return hasInternalLinkage() || hasPrivateLinkage() ||
      hasLinkerPrivateLinkage();
  }
  bool hasDLLImportLinkage() const { return Linkage == DLLImportLinkage; }
  bool hasDLLExportLinkage() const { return Linkage == DLLExportLinkage; }
  bool hasExternalWeakLinkage() const { return Linkage == ExternalWeakLinkage; }
  bool hasCommonLinkage() const { return Linkage == CommonLinkage; }

  void setLinkage(LinkageTypes LT) { Linkage = LT; }
  LinkageTypes getLinkage() const { return Linkage; }

  /// mayBeOverridden - Whether the definition of this global may be replaced
  /// by something non-equivalent at link time.  For example, if a function has
  /// weak linkage then the code defining it may be replaced by different code.
  bool mayBeOverridden() const {
    return (Linkage == WeakAnyLinkage ||
            Linkage == LinkOnceAnyLinkage ||
            Linkage == CommonLinkage ||
            Linkage == ExternalWeakLinkage);
  }

  /// isWeakForLinker - Whether the definition of this global may be replaced at
  /// link time.
  bool isWeakForLinker() const {
    return (Linkage == AvailableExternallyLinkage ||
            Linkage == WeakAnyLinkage ||
            Linkage == WeakODRLinkage ||
            Linkage == LinkOnceAnyLinkage ||
            Linkage == LinkOnceODRLinkage ||
            Linkage == CommonLinkage ||
            Linkage == ExternalWeakLinkage);
  }

  /// copyAttributesFrom - copy all additional attributes (those not needed to
  /// create a GlobalValue) from the GlobalValue Src to this one.
  virtual void copyAttributesFrom(const GlobalValue *Src);

/// @name Materialization
/// Materialization is used to construct functions only as they're needed. This
/// is useful to reduce memory usage in LLVM or parsing work done by the
/// BitcodeReader to load the Module.
/// @{

  /// isMaterializable - If this function's Module is being lazily streamed in
  /// functions from disk or some other source, this method can be used to check
  /// to see if the function has been read in yet or not.
  bool isMaterializable() const;

  /// isDematerializable - Returns true if this function was loaded from a
  /// GVMaterializer that's still attached to its Module and that knows how to
  /// dematerialize the function.
  bool isDematerializable() const;

  /// Materialize - make sure this GlobalValue is fully read.  If the module is
  /// corrupt, this returns true and fills in the optional string with
  /// information about the problem.  If successful, this returns false.
  bool Materialize(std::string *ErrInfo = 0);

  /// Dematerialize - If this GlobalValue is read in, and if the GVMaterializer
  /// supports it, release the memory for the function, and set it up to be
  /// materialized lazily.  If !isDematerializable(), this method is a noop.
  void Dematerialize();

/// @}

  /// Override from Constant class. No GlobalValue's are null values so this
  /// always returns false.
  virtual bool isNullValue() const { return false; }

  /// Override from Constant class.
  virtual void destroyConstant();

  /// isDeclaration - Return true if the primary definition of this global 
  /// value is outside of the current translation unit...
  virtual bool isDeclaration() const = 0;

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  virtual void removeFromParent() = 0;

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  virtual void eraseFromParent() = 0;

  /// getParent - Get the module that this global value is contained inside
  /// of...
  inline Module *getParent() { return Parent; }
  inline const Module *getParent() const { return Parent; }

  /// removeDeadConstantUsers - If there are any dead constant users dangling
  /// off of this global value, remove them.  This method is useful for clients
  /// that want to check to see if a global is unused, but don't want to deal
  /// with potentially dead constants hanging off of the globals.
  void removeDeadConstantUsers() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalValue *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::FunctionVal ||
           V->getValueID() == Value::GlobalVariableVal ||
           V->getValueID() == Value::GlobalAliasVal;
  }
};

} // End llvm namespace

#endif

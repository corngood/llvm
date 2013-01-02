//===-- Attribute.cpp - Implement AttributesList -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Attribute, AttributeImpl, AttrBuilder,
// AttributeSetImpl, and AttributeSet classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Attributes.h"
#include "AttributeImpl.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Attribute Implementation
//===----------------------------------------------------------------------===//

Attribute Attribute::get(LLVMContext &Context, ArrayRef<AttrKind> Vals) {
  AttrBuilder B;
  for (ArrayRef<AttrKind>::iterator I = Vals.begin(), E = Vals.end();
       I != E; ++I)
    B.addAttribute(*I);
  return Attribute::get(Context, B);
}

Attribute Attribute::get(LLVMContext &Context, AttrBuilder &B) {
  // If there are no attributes, return an empty Attribute class.
  if (!B.hasAttributes())
    return Attribute();

  // Otherwise, build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddInteger(B.getBitMask());

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new AttributeImpl(Context, B.getBitMask());
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return Attribute(PA);
}

bool Attribute::hasAttribute(AttrKind Val) const {
  return pImpl && pImpl->hasAttribute(Val);
}

bool Attribute::hasAttributes() const {
  return pImpl && pImpl->hasAttributes();
}

bool Attribute::hasAttributes(const Attribute &A) const {
  return pImpl && pImpl->hasAttributes(A);
}

/// This returns the alignment field of an attribute as a byte alignment value.
unsigned Attribute::getAlignment() const {
  if (!hasAttribute(Attribute::Alignment))
    return 0;
  return 1U << ((pImpl->getAlignment() >> 16) - 1);
}

/// This returns the stack alignment field of an attribute as a byte alignment
/// value.
unsigned Attribute::getStackAlignment() const {
  if (!hasAttribute(Attribute::StackAlignment))
    return 0;
  return 1U << ((pImpl->getStackAlignment() >> 26) - 1);
}

bool Attribute::operator==(AttrKind K) const {
  return pImpl && pImpl->contains(K);
}

bool Attribute::operator!=(AttrKind K) const {
  return !(pImpl && pImpl->contains(K));
}

uint64_t Attribute::getBitMask() const {
  return pImpl ? pImpl->getBitMask() : 0;
}

Attribute Attribute::typeIncompatible(Type *Ty) {
  AttrBuilder Incompatible;

  if (!Ty->isIntegerTy())
    // Attribute that only apply to integers.
    Incompatible.addAttribute(Attribute::SExt)
      .addAttribute(Attribute::ZExt);

  if (!Ty->isPointerTy())
    // Attribute that only apply to pointers.
    Incompatible.addAttribute(Attribute::ByVal)
      .addAttribute(Attribute::Nest)
      .addAttribute(Attribute::NoAlias)
      .addAttribute(Attribute::NoCapture)
      .addAttribute(Attribute::StructRet);

  return Attribute::get(Ty->getContext(), Incompatible);
}

/// encodeLLVMAttributesForBitcode - This returns an integer containing an
/// encoding of all the LLVM attributes found in the given attribute bitset.
/// Any change to this encoding is a breaking change to bitcode compatibility.
uint64_t Attribute::encodeLLVMAttributesForBitcode(Attribute Attrs) {
  // FIXME: It doesn't make sense to store the alignment information as an
  // expanded out value, we should store it as a log2 value.  However, we can't
  // just change that here without breaking bitcode compatibility.  If this ever
  // becomes a problem in practice, we should introduce new tag numbers in the
  // bitcode file and have those tags use a more efficiently encoded alignment
  // field.

  // Store the alignment in the bitcode as a 16-bit raw value instead of a 5-bit
  // log2 encoded value. Shift the bits above the alignment up by 11 bits.
  uint64_t EncodedAttrs = Attrs.getBitMask() & 0xffff;
  if (Attrs.hasAttribute(Attribute::Alignment))
    EncodedAttrs |= Attrs.getAlignment() << 16;
  EncodedAttrs |= (Attrs.getBitMask() & (0xffffULL << 21)) << 11;
  return EncodedAttrs;
}

/// decodeLLVMAttributesForBitcode - This returns an attribute bitset containing
/// the LLVM attributes that have been decoded from the given integer.  This
/// function must stay in sync with 'encodeLLVMAttributesForBitcode'.
Attribute Attribute::decodeLLVMAttributesForBitcode(LLVMContext &C,
                                                      uint64_t EncodedAttrs) {
  // The alignment is stored as a 16-bit raw value from bits 31--16.  We shift
  // the bits above 31 down by 11 bits.
  unsigned Alignment = (EncodedAttrs & (0xffffULL << 16)) >> 16;
  assert((!Alignment || isPowerOf2_32(Alignment)) &&
         "Alignment must be a power of two.");

  AttrBuilder B(EncodedAttrs & 0xffff);
  if (Alignment)
    B.addAlignmentAttr(Alignment);
  B.addRawValue((EncodedAttrs & (0xffffULL << 32)) >> 11);
  return Attribute::get(C, B);
}

std::string Attribute::getAsString() const {
  std::string Result;
  if (hasAttribute(Attribute::ZExt))
    Result += "zeroext ";
  if (hasAttribute(Attribute::SExt))
    Result += "signext ";
  if (hasAttribute(Attribute::NoReturn))
    Result += "noreturn ";
  if (hasAttribute(Attribute::NoUnwind))
    Result += "nounwind ";
  if (hasAttribute(Attribute::UWTable))
    Result += "uwtable ";
  if (hasAttribute(Attribute::ReturnsTwice))
    Result += "returns_twice ";
  if (hasAttribute(Attribute::InReg))
    Result += "inreg ";
  if (hasAttribute(Attribute::NoAlias))
    Result += "noalias ";
  if (hasAttribute(Attribute::NoCapture))
    Result += "nocapture ";
  if (hasAttribute(Attribute::StructRet))
    Result += "sret ";
  if (hasAttribute(Attribute::ByVal))
    Result += "byval ";
  if (hasAttribute(Attribute::Nest))
    Result += "nest ";
  if (hasAttribute(Attribute::ReadNone))
    Result += "readnone ";
  if (hasAttribute(Attribute::ReadOnly))
    Result += "readonly ";
  if (hasAttribute(Attribute::OptimizeForSize))
    Result += "optsize ";
  if (hasAttribute(Attribute::NoInline))
    Result += "noinline ";
  if (hasAttribute(Attribute::InlineHint))
    Result += "inlinehint ";
  if (hasAttribute(Attribute::AlwaysInline))
    Result += "alwaysinline ";
  if (hasAttribute(Attribute::StackProtect))
    Result += "ssp ";
  if (hasAttribute(Attribute::StackProtectReq))
    Result += "sspreq ";
  if (hasAttribute(Attribute::NoRedZone))
    Result += "noredzone ";
  if (hasAttribute(Attribute::NoImplicitFloat))
    Result += "noimplicitfloat ";
  if (hasAttribute(Attribute::Naked))
    Result += "naked ";
  if (hasAttribute(Attribute::NonLazyBind))
    Result += "nonlazybind ";
  if (hasAttribute(Attribute::AddressSafety))
    Result += "address_safety ";
  if (hasAttribute(Attribute::MinSize))
    Result += "minsize ";
  if (hasAttribute(Attribute::StackAlignment)) {
    Result += "alignstack(";
    Result += utostr(getStackAlignment());
    Result += ") ";
  }
  if (hasAttribute(Attribute::Alignment)) {
    Result += "align ";
    Result += utostr(getAlignment());
    Result += " ";
  }
  if (hasAttribute(Attribute::NoDuplicate))
    Result += "noduplicate ";
  // Trim the trailing space.
  assert(!Result.empty() && "Unknown attribute!");
  Result.erase(Result.end()-1);
  return Result;
}

//===----------------------------------------------------------------------===//
// AttrBuilder Implementation
//===----------------------------------------------------------------------===//

void AttrBuilder::clear() {
  AttrSet.clear();
  Alignment = StackAlignment = Bits = 0;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute::AttrKind Val){
  Bits |= AttributeImpl::getAttrMask(Val);

  AttrSet.insert(Val);
  return *this;
}

AttrBuilder &AttrBuilder::addRawValue(uint64_t Val) {
  Bits |= Val;
  return *this;
}

AttrBuilder &AttrBuilder::addAlignmentAttr(unsigned Align) {
  if (Align == 0) return *this;
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");
  Bits |= (Log2_32(Align) + 1) << 16;

  AttrSet.insert(Attribute::Alignment);
  Alignment = Align;
  return *this;
}
AttrBuilder &AttrBuilder::addStackAlignmentAttr(unsigned Align) {
  // Default alignment, allow the target to define how to align it.
  if (Align == 0) return *this;
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");
  Bits |= (Log2_32(Align) + 1) << 26;

  AttrSet.insert(Attribute::StackAlignment);
  StackAlignment = Align;
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(Attribute::AttrKind Val) {
  Bits &= ~AttributeImpl::getAttrMask(Val);

  AttrSet.erase(Val);
  if (Val == Attribute::Alignment)
    Alignment = 0;
  else if (Val == Attribute::StackAlignment)
    StackAlignment = 0;
  return *this;
}

AttrBuilder &AttrBuilder::addAttributes(const Attribute &A) {
  Bits |= A.getBitMask();
  return *this;
}

AttrBuilder &AttrBuilder::removeAttributes(const Attribute &A){
  Bits &= ~A.getBitMask();
  return *this;
}

bool AttrBuilder::contains(Attribute::AttrKind A) const {
  return Bits & AttributeImpl::getAttrMask(A);
}

bool AttrBuilder::hasAttributes() const {
  return Bits != 0;
}
bool AttrBuilder::hasAttributes(const Attribute &A) const {
  return Bits & A.getBitMask();
}
bool AttrBuilder::hasAlignmentAttr() const {
  return Bits & AttributeImpl::getAttrMask(Attribute::Alignment);
}

uint64_t AttrBuilder::getAlignment() const {
  if (!hasAlignmentAttr())
    return 0;
  return 1ULL <<
    (((Bits & AttributeImpl::getAttrMask(Attribute::Alignment)) >> 16) - 1);
}

uint64_t AttrBuilder::getStackAlignment() const {
  if (!hasAlignmentAttr())
    return 0;
  return 1ULL <<
    (((Bits & AttributeImpl::getAttrMask(Attribute::StackAlignment))>>26)-1);
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

AttributeImpl::AttributeImpl(LLVMContext &C, uint64_t data) {
  Data = ConstantInt::get(Type::getInt64Ty(C), data);
}
AttributeImpl::AttributeImpl(LLVMContext &C, Attribute::AttrKind data) {
  Data = ConstantInt::get(Type::getInt64Ty(C), data);
}
AttributeImpl::AttributeImpl(LLVMContext &C, Attribute::AttrKind data,
                             ArrayRef<Constant*> values) {
  Data = ConstantInt::get(Type::getInt64Ty(C), data);
  Vals.reserve(values.size());
  Vals.append(values.begin(), values.end());
}
AttributeImpl::AttributeImpl(LLVMContext &C, StringRef data) {
  Data = ConstantDataArray::getString(C, data);
}

bool AttributeImpl::contains(Attribute::AttrKind Kind) const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Data))
    return CI->getZExtValue() == Kind;
  return false;
}

bool AttributeImpl::contains(StringRef Kind) const {
  if (ConstantDataArray *CDA = dyn_cast<ConstantDataArray>(Data))
    if (CDA->isString())
      return CDA->getAsString() == Kind;
  return false;
}

uint64_t AttributeImpl::getBitMask() const {
  // FIXME: Remove this.
  return cast<ConstantInt>(Data)->getZExtValue();
}

uint64_t AttributeImpl::getAttrMask(uint64_t Val) {
  switch (Val) {
  case Attribute::None:            return 0;
  case Attribute::ZExt:            return 1 << 0;
  case Attribute::SExt:            return 1 << 1;
  case Attribute::NoReturn:        return 1 << 2;
  case Attribute::InReg:           return 1 << 3;
  case Attribute::StructRet:       return 1 << 4;
  case Attribute::NoUnwind:        return 1 << 5;
  case Attribute::NoAlias:         return 1 << 6;
  case Attribute::ByVal:           return 1 << 7;
  case Attribute::Nest:            return 1 << 8;
  case Attribute::ReadNone:        return 1 << 9;
  case Attribute::ReadOnly:        return 1 << 10;
  case Attribute::NoInline:        return 1 << 11;
  case Attribute::AlwaysInline:    return 1 << 12;
  case Attribute::OptimizeForSize: return 1 << 13;
  case Attribute::StackProtect:    return 1 << 14;
  case Attribute::StackProtectReq: return 1 << 15;
  case Attribute::Alignment:       return 31 << 16;
  case Attribute::NoCapture:       return 1 << 21;
  case Attribute::NoRedZone:       return 1 << 22;
  case Attribute::NoImplicitFloat: return 1 << 23;
  case Attribute::Naked:           return 1 << 24;
  case Attribute::InlineHint:      return 1 << 25;
  case Attribute::StackAlignment:  return 7 << 26;
  case Attribute::ReturnsTwice:    return 1 << 29;
  case Attribute::UWTable:         return 1 << 30;
  case Attribute::NonLazyBind:     return 1U << 31;
  case Attribute::AddressSafety:   return 1ULL << 32;
  case Attribute::MinSize:         return 1ULL << 33;
  case Attribute::NoDuplicate:     return 1ULL << 34;
  }
  llvm_unreachable("Unsupported attribute type");
}

bool AttributeImpl::hasAttribute(uint64_t A) const {
  return (getBitMask() & getAttrMask(A)) != 0;
}

bool AttributeImpl::hasAttributes() const {
  return getBitMask() != 0;
}

bool AttributeImpl::hasAttributes(const Attribute &A) const {
  // FIXME: getBitMask() won't work here in the future.
  return getBitMask() & A.getBitMask();
}

uint64_t AttributeImpl::getAlignment() const {
  return getBitMask() & getAttrMask(Attribute::Alignment);
}

uint64_t AttributeImpl::getStackAlignment() const {
  return getBitMask() & getAttrMask(Attribute::StackAlignment);
}

//===----------------------------------------------------------------------===//
// AttributeSetImpl Definition
//===----------------------------------------------------------------------===//

AttributeSet AttributeSet::get(LLVMContext &C,
                               ArrayRef<AttributeWithIndex> Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttributeSet();

#ifndef NDEBUG
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
    assert(Attrs[i].Attrs.hasAttributes() &&
           "Pointless attribute!");
    assert((!i || Attrs[i-1].Index < Attrs[i].Index) &&
           "Misordered AttributesList!");
  }
#endif

  // Otherwise, build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;
  AttributeSetImpl::Profile(ID, Attrs);

  void *InsertPoint;
  AttributeSetImpl *PA = pImpl->AttrsLists.FindNodeOrInsertPos(ID,
                                                                InsertPoint);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PA) {
    PA = new AttributeSetImpl(C, Attrs);
    pImpl->AttrsLists.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return AttributeSet(PA);
}

//===----------------------------------------------------------------------===//
// AttributeSet Method Implementations
//===----------------------------------------------------------------------===//

const AttributeSet &AttributeSet::operator=(const AttributeSet &RHS) {
  AttrList = RHS.AttrList;
  return *this;
}

/// getNumSlots - Return the number of slots used in this attribute list.
/// This is the number of arguments that have an attribute set on them
/// (including the function itself).
unsigned AttributeSet::getNumSlots() const {
  return AttrList ? AttrList->Attrs.size() : 0;
}

/// getSlot - Return the AttributeWithIndex at the specified slot.  This
/// holds a number plus a set of attributes.
const AttributeWithIndex &AttributeSet::getSlot(unsigned Slot) const {
  assert(AttrList && Slot < AttrList->Attrs.size() && "Slot # out of range!");
  return AttrList->Attrs[Slot];
}

bool AttributeSet::hasAttribute(unsigned Index, Attribute::AttrKind Kind) const{
  return getAttributes(Index).hasAttribute(Kind);
}

bool AttributeSet::hasAttributes(unsigned Index) const {
  return getAttributes(Index).hasAttributes();
}

std::string AttributeSet::getAsString(unsigned Index) const {
  return getAttributes(Index).getAsString();
}

unsigned AttributeSet::getStackAlignment(unsigned Index) const {
  return getAttributes(Index).getStackAlignment();
}

uint64_t AttributeSet::getBitMask(unsigned Index) const {
  // FIXME: Remove this.
  return getAttributes(Index).getBitMask();
}

/// getAttributes - The attributes for the specified index are returned.
/// Attributes for the result are denoted with Idx = 0.  Function attributes are
/// denoted with Idx = ~0.
Attribute AttributeSet::getAttributes(unsigned Idx) const {
  if (AttrList == 0) return Attribute();

  const SmallVectorImpl<AttributeWithIndex> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e && Attrs[i].Index <= Idx; ++i)
    if (Attrs[i].Index == Idx)
      return Attrs[i].Attrs;

  return Attribute();
}

/// hasAttrSomewhere - Return true if the specified attribute is set for at
/// least one parameter or for the return value.
bool AttributeSet::hasAttrSomewhere(Attribute::AttrKind Attr) const {
  if (AttrList == 0) return false;

  const SmallVector<AttributeWithIndex, 4> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i)
    if (Attrs[i].Attrs.hasAttribute(Attr))
      return true;

  return false;
}

AttributeSet AttributeSet::addAttr(LLVMContext &C, unsigned Idx,
                                   Attribute Attrs) const {
  Attribute OldAttrs = getAttributes(Idx);
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't change a known alignment.
  unsigned OldAlign = OldAttrs.getAlignment();
  unsigned NewAlign = Attrs.getAlignment();
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  AttrBuilder NewAttrs =
    AttrBuilder(OldAttrs).addAttributes(Attrs);
  if (NewAttrs == AttrBuilder(OldAttrs))
    return *this;

  SmallVector<AttributeWithIndex, 8> NewAttrList;
  if (AttrList == 0)
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));
  else {
    const SmallVector<AttributeWithIndex, 4> &OldAttrList = AttrList->Attrs;
    unsigned i = 0, e = OldAttrList.size();
    // Copy attributes for arguments before this one.
    for (; i != e && OldAttrList[i].Index < Idx; ++i)
      NewAttrList.push_back(OldAttrList[i]);

    // If there are attributes already at this index, merge them in.
    if (i != e && OldAttrList[i].Index == Idx) {
      Attrs =
        Attribute::get(C, AttrBuilder(Attrs).
                        addAttributes(OldAttrList[i].Attrs));
      ++i;
    }

    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));

    // Copy attributes for arguments after this one.
    NewAttrList.insert(NewAttrList.end(),
                       OldAttrList.begin()+i, OldAttrList.end());
  }

  return get(C, NewAttrList);
}

AttributeSet AttributeSet::removeAttr(LLVMContext &C, unsigned Idx,
                                      Attribute Attrs) const {
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAttribute(Attribute::Alignment) &&
         "Attempt to exclude alignment!");
#endif
  if (AttrList == 0) return AttributeSet();

  Attribute OldAttrs = getAttributes(Idx);
  AttrBuilder NewAttrs =
    AttrBuilder(OldAttrs).removeAttributes(Attrs);
  if (NewAttrs == AttrBuilder(OldAttrs))
    return *this;

  SmallVector<AttributeWithIndex, 8> NewAttrList;
  const SmallVector<AttributeWithIndex, 4> &OldAttrList = AttrList->Attrs;
  unsigned i = 0, e = OldAttrList.size();

  // Copy attributes for arguments before this one.
  for (; i != e && OldAttrList[i].Index < Idx; ++i)
    NewAttrList.push_back(OldAttrList[i]);

  // If there are attributes already at this index, merge them in.
  assert(OldAttrList[i].Index == Idx && "Attribute isn't set?");
  Attrs = Attribute::get(C, AttrBuilder(OldAttrList[i].Attrs).
                          removeAttributes(Attrs));
  ++i;
  if (Attrs.hasAttributes()) // If any attributes left for this param, add them.
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));

  // Copy attributes for arguments after this one.
  NewAttrList.insert(NewAttrList.end(),
                     OldAttrList.begin()+i, OldAttrList.end());

  return get(C, NewAttrList);
}

void AttributeSet::dump() const {
  dbgs() << "PAL[ ";
  for (unsigned i = 0; i < getNumSlots(); ++i) {
    const AttributeWithIndex &PAWI = getSlot(i);
    dbgs() << "{" << PAWI.Index << "," << PAWI.Attrs.getAsString() << "} ";
  }

  dbgs() << "]\n";
}

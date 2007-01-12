//===-- TargetData.cpp - Data size & alignment routines --------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target properties related to datatype size/offset/alignment
// information.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetData.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cstdlib>
#include <sstream>
using namespace llvm;

// Handle the Pass registration stuff necessary to use TargetData's.
namespace {
  // Register the default SparcV9 implementation...
  RegisterPass<TargetData> X("targetdata", "Target Data Layout");
}

static inline void getTypeInfo(const Type *Ty, const TargetData *TD,
                               uint64_t &Size, unsigned char &Alignment);

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(const StructType *ST, const TargetData &TD) {
  StructAlignment = 0;
  StructSize = 0;

  // Loop over each of the elements, placing them in memory...
  for (StructType::element_iterator TI = ST->element_begin(),
         TE = ST->element_end(); TI != TE; ++TI) {
    const Type *Ty = *TI;
    unsigned char A;
    unsigned TyAlign;
    uint64_t TySize;
    getTypeInfo(Ty, &TD, TySize, A);
    TyAlign = ST->isPacked() ? 1 : A;

    // Add padding if necessary to make the data element aligned properly...
    if (StructSize % TyAlign != 0)
      StructSize = (StructSize/TyAlign + 1) * TyAlign;   // Add padding...

    // Keep track of maximum alignment constraint
    StructAlignment = std::max(TyAlign, StructAlignment);

    MemberOffsets.push_back(StructSize);
    StructSize += TySize;                 // Consume space for this data item
  }

  // Empty structures have alignment of 1 byte.
  if (StructAlignment == 0) StructAlignment = 1;

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (StructSize % StructAlignment != 0)
    StructSize = (StructSize/StructAlignment + 1) * StructAlignment;
}


/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t Offset) const {
  std::vector<uint64_t>::const_iterator SI =
    std::upper_bound(MemberOffsets.begin(), MemberOffsets.end(),
                     Offset);
  assert(SI != MemberOffsets.begin() && "Offset not in structure type!");
  --SI;
  assert(*SI <= Offset && "upper_bound didn't work");
  assert((SI == MemberOffsets.begin() || *(SI-1) < Offset) &&
         (SI+1 == MemberOffsets.end() || *(SI+1) > Offset) &&
         "Upper bound didn't work!");
  return SI-MemberOffsets.begin();
}

//===----------------------------------------------------------------------===//
//                       TargetData Class Implementation
//===----------------------------------------------------------------------===//

void TargetData::init(const std::string &TargetDescription) {
  std::string temp = TargetDescription;
  
  LittleEndian = false;
  PointerSize = 8;
  PointerAlignment   = 8;
  DoubleAlignment = 8;
  FloatAlignment = 4;
  LongAlignment   = 8;
  IntAlignment   = 4;
  ShortAlignment  = 2;
  ByteAlignment  = 1;
  BoolAlignment   = 1;
  
  while (!temp.empty()) {
    std::string token = getToken(temp, "-");
    
    char signal = getToken(token, ":")[0];
    
    switch(signal) {
    case 'E':
      LittleEndian = false;
      break;
    case 'e':
      LittleEndian = true;
      break;
    case 'p':
      PointerSize = atoi(getToken(token,":").c_str()) / 8;
      PointerAlignment = atoi(getToken(token,":").c_str()) / 8;
      break;
    case 'd':
      DoubleAlignment = atoi(getToken(token,":").c_str()) / 8;
      break;
    case 'f':
      FloatAlignment = atoi(getToken(token, ":").c_str()) / 8;
      break;
    case 'l':
      LongAlignment = atoi(getToken(token, ":").c_str()) / 8;
      break;
    case 'i':
      IntAlignment = atoi(getToken(token, ":").c_str()) / 8;
      break;
    case 's':
      ShortAlignment = atoi(getToken(token, ":").c_str()) / 8;
      break;
    case 'b':
      ByteAlignment = atoi(getToken(token, ":").c_str()) / 8;
      break;
    case 'B':
      BoolAlignment = atoi(getToken(token, ":").c_str()) / 8;
      break;
    default:
      break;
    }
  }
}

TargetData::TargetData(const Module *M) {
  LittleEndian     = M->getEndianness() != Module::BigEndian;
  PointerSize      = M->getPointerSize() != Module::Pointer64 ? 4 : 8;
  PointerAlignment = PointerSize;
  DoubleAlignment  = PointerSize;
  FloatAlignment   = 4;
  LongAlignment    = PointerSize;
  IntAlignment     = 4;
  ShortAlignment   = 2;
  ByteAlignment    = 1;
  BoolAlignment    = 1;
}

/// Layouts - The lazy cache of structure layout information maintained by
/// TargetData.
///
static std::map<std::pair<const TargetData*,const StructType*>,
                StructLayout> *Layouts = 0;


TargetData::~TargetData() {
  if (Layouts) {
    // Remove any layouts for this TD.
    std::map<std::pair<const TargetData*,
      const StructType*>, StructLayout>::iterator
      I = Layouts->lower_bound(std::make_pair(this, (const StructType*)0));
    while (I != Layouts->end() && I->first.first == this)
      Layouts->erase(I++);
    if (Layouts->empty()) {
      delete Layouts;
      Layouts = 0;
    }
  }
}

std::string TargetData::getStringRepresentation() const {
  std::stringstream repr;
  
  if (LittleEndian)
    repr << "e";
  else
    repr << "E";
  
  repr << "-p:" << (PointerSize * 8) << ":" << (PointerAlignment * 8);
  repr << "-d:64:" << (DoubleAlignment * 8);
  repr << "-f:32:" << (FloatAlignment * 8);
  repr << "-l:64:" << (LongAlignment * 8);
  repr << "-i:32:" << (IntAlignment * 8);
  repr << "-s:16:" << (ShortAlignment * 8);
  repr << "-b:8:" << (ByteAlignment * 8);
  repr << "-B:8:" << (BoolAlignment * 8);
  
  return repr.str();
}

const StructLayout *TargetData::getStructLayout(const StructType *Ty) const {
  if (Layouts == 0)
    Layouts = new std::map<std::pair<const TargetData*,const StructType*>,
                           StructLayout>();
  std::map<std::pair<const TargetData*,const StructType*>,
                     StructLayout>::iterator
    I = Layouts->lower_bound(std::make_pair(this, Ty));
  if (I != Layouts->end() && I->first.first == this && I->first.second == Ty)
    return &I->second;
  else {
    return &Layouts->insert(I, std::make_pair(std::make_pair(this, Ty),
                                              StructLayout(Ty, *this)))->second;
  }
}

/// InvalidateStructLayoutInfo - TargetData speculatively caches StructLayout
/// objects.  If a TargetData object is alive when types are being refined and
/// removed, this method must be called whenever a StructType is removed to
/// avoid a dangling pointer in this cache.
void TargetData::InvalidateStructLayoutInfo(const StructType *Ty) const {
  if (!Layouts) return;  // No cache.

  std::map<std::pair<const TargetData*,const StructType*>,
           StructLayout>::iterator I = Layouts->find(std::make_pair(this, Ty));
  if (I != Layouts->end())
    Layouts->erase(I);
}



static inline void getTypeInfo(const Type *Ty, const TargetData *TD,
                               uint64_t &Size, unsigned char &Alignment) {
  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    unsigned BitWidth = cast<IntegerType>(Ty)->getBitWidth();
    if (BitWidth <= 8) {
      Size = 1; Alignment = TD->getByteAlignment();
    } else if (BitWidth <= 16) {
      Size = 2; Alignment = TD->getShortAlignment();
    } else if (BitWidth <= 32) {
      Size = 4; Alignment = TD->getIntAlignment();
    } else if (BitWidth <= 64) {
      Size = 8; Alignment = TD->getLongAlignment();
    } else
      assert(0 && "Integer types > 64 bits not supported.");
    return;
  }
  case Type::VoidTyID:   Size = 1; Alignment = TD->getByteAlignment(); return;
  case Type::FloatTyID:  Size = 4; Alignment = TD->getFloatAlignment(); return;
  case Type::DoubleTyID: Size = 8; Alignment = TD->getDoubleAlignment(); return;
  case Type::LabelTyID:
  case Type::PointerTyID:
    Size = TD->getPointerSize(); Alignment = TD->getPointerAlignment();
    return;
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    getTypeInfo(ATy->getElementType(), TD, Size, Alignment);
    unsigned AlignedSize = (Size + Alignment - 1)/Alignment*Alignment;
    Size = AlignedSize*ATy->getNumElements();
    return;
  }
  case Type::PackedTyID: {
    const PackedType *PTy = cast<PackedType>(Ty);
    getTypeInfo(PTy->getElementType(), TD, Size, Alignment);
    unsigned AlignedSize = (Size + Alignment - 1)/Alignment*Alignment;
    Size = AlignedSize*PTy->getNumElements();
    // FIXME: The alignments of specific packed types are target dependent.
    // For now, just set it to be equal to Size.
    Alignment = Size;
    return;
  }
  case Type::StructTyID: {
    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = TD->getStructLayout(cast<StructType>(Ty));
    Size = Layout->StructSize; Alignment = Layout->StructAlignment;
    return;
  }

  default:
    assert(0 && "Bad type for getTypeInfo!!!");
    return;
  }
}

uint64_t TargetData::getTypeSize(const Type *Ty) const {
  uint64_t Size;
  unsigned char Align;
  getTypeInfo(Ty, this, Size, Align);
  return Size;
}

unsigned char TargetData::getTypeAlignment(const Type *Ty) const {
  uint64_t Size;
  unsigned char Align;
  getTypeInfo(Ty, this, Size, Align);
  return Align;
}

unsigned char TargetData::getTypeAlignmentShift(const Type *Ty) const {
  unsigned Align = getTypeAlignment(Ty);
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");
  return Log2_32(Align);
}

/// getIntPtrType - Return an unsigned integer type that is the same size or
/// greater to the host pointer size.
const Type *TargetData::getIntPtrType() const {
  switch (getPointerSize()) {
  default: assert(0 && "Unknown pointer size!");
  case 2: return Type::Int16Ty;
  case 4: return Type::Int32Ty;
  case 8: return Type::Int64Ty;
  }
}


uint64_t TargetData::getIndexedOffset(const Type *ptrTy,
                                      const std::vector<Value*> &Idx) const {
  const Type *Ty = ptrTy;
  assert(isa<PointerType>(Ty) && "Illegal argument for getIndexedOffset()");
  uint64_t Result = 0;

  generic_gep_type_iterator<std::vector<Value*>::const_iterator>
    TI = gep_type_begin(ptrTy, Idx.begin(), Idx.end());
  for (unsigned CurIDX = 0; CurIDX != Idx.size(); ++CurIDX, ++TI) {
    if (const StructType *STy = dyn_cast<StructType>(*TI)) {
      assert(Idx[CurIDX]->getType() == Type::Int32Ty && "Illegal struct idx");
      unsigned FieldNo = cast<ConstantInt>(Idx[CurIDX])->getZExtValue();

      // Get structure layout information...
      const StructLayout *Layout = getStructLayout(STy);

      // Add in the offset, as calculated by the structure layout info...
      assert(FieldNo < Layout->MemberOffsets.size() &&"FieldNo out of range!");
      Result += Layout->MemberOffsets[FieldNo];

      // Update Ty to refer to current element
      Ty = STy->getElementType(FieldNo);
    } else {
      // Update Ty to refer to current element
      Ty = cast<SequentialType>(Ty)->getElementType();

      // Get the array index and the size of each array element.
      int64_t arrayIdx = cast<ConstantInt>(Idx[CurIDX])->getSExtValue();
      Result += arrayIdx * (int64_t)getTypeSize(Ty);
    }
  }

  return Result;
}

/// getPreferredAlignmentLog - Return the preferred alignment of the
/// specified global, returned in log form.  This includes an explicitly
/// requested alignment (if the global has one).
unsigned TargetData::getPreferredAlignmentLog(const GlobalVariable *GV) const {
  const Type *ElemType = GV->getType()->getElementType();
  unsigned Alignment = getTypeAlignmentShift(ElemType);
  if (GV->getAlignment() > (1U << Alignment))
    Alignment = Log2_32(GV->getAlignment());
  
  if (GV->hasInitializer()) {
    // Always round up alignment of global doubles to 8 bytes.
    if (GV->getType()->getElementType() == Type::DoubleTy && Alignment < 3)
      Alignment = 3;
    if (Alignment < 4) {
      // If the global is not external, see if it is large.  If so, give it a
      // larger alignment.
      if (getTypeSize(ElemType) > 128)
        Alignment = 4;    // 16-byte alignment.
    }
  }
  return Alignment;
}


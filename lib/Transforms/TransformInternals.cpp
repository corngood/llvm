//===- TransformInternals.cpp - Implement shared functions for transforms -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines shared functions used by the different components of the
//  Transforms library.
//
//===----------------------------------------------------------------------===//

#include "TransformInternals.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
using namespace llvm;

static const Type *getStructOffsetStep(const StructType *STy, uint64_t &Offset,
                                       std::vector<Value*> &Indices,
                                       const TargetData &TD) {
  assert(Offset < TD.getTypeSize(STy) && "Offset not in composite!");
  const StructLayout *SL = TD.getStructLayout(STy);

  // This loop terminates always on a 0 <= i < MemberOffsets.size()
  unsigned i;
  for (i = 0; i < SL->MemberOffsets.size()-1; ++i)
    if (Offset >= SL->MemberOffsets[i] && Offset < SL->MemberOffsets[i+1])
      break;

  assert(Offset >= SL->MemberOffsets[i] &&
         (i == SL->MemberOffsets.size()-1 || Offset < SL->MemberOffsets[i+1]));

  // Make sure to save the current index...
  Indices.push_back(ConstantInt::get(Type::Int32Ty, i));
  Offset = SL->MemberOffsets[i];
  return STy->getContainedType(i);
}


// getStructOffsetType - Return a vector of offsets that are to be used to index
// into the specified struct type to get as close as possible to index as we
// can.  Note that it is possible that we cannot get exactly to Offset, in which
// case we update offset to be the offset we actually obtained.  The resultant
// leaf type is returned.
//
// If StopEarly is set to true (the default), the first object with the
// specified type is returned, even if it is a struct type itself.  In this
// case, this routine will not drill down to the leaf type.  Set StopEarly to
// false if you want a leaf
//
const Type *llvm::getStructOffsetType(const Type *Ty, unsigned &Offset,
                                      std::vector<Value*> &Indices,
                                      const TargetData &TD, bool StopEarly) {
  if (Offset == 0 && StopEarly && !Indices.empty())
    return Ty;    // Return the leaf type

  uint64_t ThisOffset;
  const Type *NextType;
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    if (STy->getNumElements()) {
      Offset = 0;
      return STy;
    }

    ThisOffset = Offset;
    NextType = getStructOffsetStep(STy, ThisOffset, Indices, TD);
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    assert(Offset == 0 || Offset < TD.getTypeSize(ATy) &&
           "Offset not in composite!");

    NextType = ATy->getElementType();
    unsigned ChildSize = (unsigned)TD.getTypeSize(NextType);
    if (ConstantInt::isValueValidForType(Type::Int32Ty, 
                                         uint64_t(Offset/ChildSize)))
      Indices.push_back(ConstantInt::get(Type::Int32Ty, Offset/ChildSize));
    else
      Indices.push_back(ConstantInt::get(Type::Int64Ty, Offset/ChildSize));
    ThisOffset = (Offset/ChildSize)*ChildSize;
  } else {
    Offset = 0;   // Return the offset that we were able to achieve
    return Ty;    // Return the leaf type
  }

  unsigned SubOffs = unsigned(Offset - ThisOffset);
  const Type *LeafTy = getStructOffsetType(NextType, SubOffs,
                                           Indices, TD, StopEarly);
  Offset = unsigned(ThisOffset + SubOffs);
  return LeafTy;
}

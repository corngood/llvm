//===- llvm/ADT/BitVector.h - Bit vectors -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BitVector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_BITVECTOR_H
#define LLVM_ADT_BITVECTOR_H

#include "llvm/Support/MathExtras.h"

namespace llvm {

class BitVector {
  typedef unsigned long BitWord;

  enum { BITS_PER_WORD = sizeof(BitWord) * 8 };

  BitWord  *Bits;        // Actual bits. 
  unsigned Size;         // Size of bitvector in bits.
  unsigned Capacity;     // Size of allocated memory in BitWord.

public:
  // Encapsulation of a single bit.
  class reference {
    friend class BitVector;

    BitWord *WordRef;
    unsigned BitPos;

    reference();  // Undefined

  public:
    reference(BitVector &b, unsigned Idx) {
      WordRef = &b.Bits[Idx / BITS_PER_WORD];
      BitPos = Idx % BITS_PER_WORD;
    }

    ~reference() {}

    reference& operator=(bool t) {
      if (t)
        *WordRef |= 1 << BitPos;
      else
        *WordRef &= ~(1 << BitPos);
      return *this;
    }

    reference& operator=(const reference& rhs) {
      if (*rhs.WordRef & (1 << rhs.BitPos))
        *WordRef |= 1 << BitPos;
      else
        *WordRef &= ~(1 << BitPos);
      return *this;
    }

    operator bool() const {
      return (*WordRef) & (1 << BitPos);
    }
  };


  /// BitVector default ctor - Creates an empty bitvector.
  BitVector() : Size(0), Capacity(0) {
    Bits = new BitWord[0];
  }

  /// BitVector ctor - Creates a bitvector of specified number of bits. All
  /// bits are initialized to false;
  BitVector(unsigned s) : Size(s) {
    Capacity = NumBitWords(s);
    Bits = new BitWord[Capacity];
    init_words(Bits, Capacity, false);
  }

  /// BitVector ctor - Creates a bitvector of specified number of bits. All
  /// bits are initialized to the specified value.
  BitVector(unsigned s, bool t) : Size(s) {
    Capacity = NumBitWords(s);
    Bits = new BitWord[Capacity];
    init_words(Bits, Capacity, t);
  }

  /// BitVector copy ctor.
  BitVector(const BitVector &RHS) : Size(RHS.size()) {
    Capacity = NumBitWords(RHS.size());
    Bits = new BitWord[Capacity];
    std::copy(RHS.Bits, &RHS.Bits[Capacity], Bits);
  }

  /// size - Returns the number of bits in this bitvector.
  unsigned size() const { return Size; }

  /// count - Returns the number of bits which are set.
  unsigned count() const {
    unsigned NumBits = 0;
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      NumBits = CountPopulation_32(Bits[i]);
    return NumBits;
  }

  /// any - Returns true if any bit is set.
  bool any() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return true;
    return false;
  }

  /// none - Returns true if none of the bits are set.
  bool none() const {
    return !any();
  }

  /// find_first - Returns the index of the first set bit, -1 if none
  /// of the bits are set.
  int find_first() const {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return i * BITS_PER_WORD + CountTrailingZeros_32(Bits[i]);
    return -1;
  }

  /// find_next - Returns the index of the next set bit following the
  /// "Prev" bit. Returns -1 if the next set bit is not found.
  int find_next(unsigned Prev) const {
    ++Prev;
    if (Prev >= Size)
      return -1;

    unsigned WordPos = Prev / BITS_PER_WORD;
    unsigned BitPos = Prev % BITS_PER_WORD;
    BitWord Copy = Bits[WordPos];
    // Mask off previous bits.
    Copy &= ~0 << BitPos;

    if (Copy != 0)
      return WordPos * BITS_PER_WORD + CountTrailingZeros_32(Copy);

    // Check subsequent words.
    for (unsigned i = WordPos+1; i < NumBitWords(size()); ++i)
      if (Bits[i] != 0)
        return i * BITS_PER_WORD + CountTrailingZeros_32(Bits[i]);
    return -1;
  }

  /// clear - Clear all bits.
  void clear() {
    delete[] Bits;
    Bits = new BitWord[0];
    Size = Capacity = 0;
  }

  /// resize - Grow or shrink the bitvector.
  void resize(unsigned N) {
    if (N > Capacity * BITS_PER_WORD) {
      unsigned OldCapacity = Capacity;
      grow(N);
      init_words(&Bits[OldCapacity], (Capacity-OldCapacity), false);
    }
    Size = N;
  }

  void resize(unsigned N, bool t) {
    if (N > Capacity * BITS_PER_WORD) {
      unsigned OldCapacity = Capacity;
      grow(N);
      init_words(&Bits[OldCapacity], (Capacity-OldCapacity), t);
    }
    Size = N;
  }

  void reserve(unsigned N) {
    if (N > Capacity * BITS_PER_WORD)
      grow(N);
  }

  // Set, reset, flip
  BitVector &set() {
    init_words(Bits, Capacity, true);
    clear_unused_bits();
    return *this;
  }

  BitVector &set(unsigned Idx) {
    Bits[Idx / BITS_PER_WORD] |= 1 << (Idx % BITS_PER_WORD);
    return *this;
  }

  BitVector &reset() {
    init_words(Bits, Capacity, false);
    return *this;
  }

  BitVector &reset(unsigned Idx) {
    Bits[Idx / BITS_PER_WORD] &= ~(1 << (Idx % BITS_PER_WORD));
    return *this;
  }

  BitVector &flip() {
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] = ~Bits[i];
    clear_unused_bits();
    return *this;
  }

  BitVector &flip(unsigned Idx) {
    Bits[Idx / BITS_PER_WORD] ^= 1 << (Idx % BITS_PER_WORD);
    return *this;
  }

  // No argument flip.
  BitVector operator~() const {
    return BitVector(*this).flip();
  }

  // Indexing.
  reference operator[](unsigned Idx) {
    return reference(*this, Idx);
  }

  bool operator[](unsigned Idx) const {
    BitWord Mask = 1 << (Idx % BITS_PER_WORD);
    return (Bits[Idx / BITS_PER_WORD] & Mask) != 0;
  }

  bool test(unsigned Idx) const {
    return (*this)[Idx];
  }

  // Comparison operators.
  bool operator==(const BitVector &RHS) const {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      if (Bits[i] != RHS.Bits[i])
        return false;
    return true;
  }

  bool operator!=(const BitVector &RHS) const {
    return !(*this == RHS);
  }

  // Intersection, union, disjoint union.
  BitVector operator&=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] &= RHS.Bits[i];
    return *this;
  }

  BitVector operator|=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] |= RHS.Bits[i];
    return *this;
  }

  BitVector operator^=(const BitVector &RHS) {
    assert(Size == RHS.Size && "Illegal operation!");
    for (unsigned i = 0; i < NumBitWords(size()); ++i)
      Bits[i] ^= RHS.Bits[i];
    return *this;
  }
  
  // Assignment operator.
  const BitVector &operator=(const BitVector &RHS) {
    if (this == &RHS) return *this;

    unsigned RHSWords = NumBitWords(RHS.size());
    unsigned NewSize = RHS.size();
    if (NewSize <= Capacity * BITS_PER_WORD) {
      std::copy(RHS.Bits, &RHS.Bits[RHSWords], Bits);
      Size = NewSize;
      clear_unused_bits();
      return *this;
    }
  
    // Grow the bitvector to have enough elements.
    Capacity = NumBitWords(NewSize);
    BitWord *NewBits = new BitWord[Capacity];
    std::copy(RHS.Bits, &RHS.Bits[RHSWords], NewBits);

    // Destroy the old bits.
    delete[] Bits;
    Bits = NewBits;

    return *this;
  }

private:
  unsigned NumBitWords(unsigned S) const {
    return (S + BITS_PER_WORD-1) / BITS_PER_WORD;
  }

  // Clear the unused top bits in the high word.
  void clear_unused_bits() {
    unsigned ExtraBits = Size % BITS_PER_WORD;
    Bits[Size / BITS_PER_WORD] &= ~(~0 << ExtraBits);
  }

  void grow(unsigned NewSize) {
    unsigned OldCapacity = Capacity;
    Capacity = NumBitWords(NewSize);
    BitWord *NewBits = new BitWord[Capacity];

    // Copy the old bits over.
    if (OldCapacity != 0)
      std::copy(Bits, &Bits[OldCapacity], NewBits);

    // Destroy the old bits.
    delete[] Bits;
    Bits = NewBits;
  }

  void init_words(BitWord *B, unsigned NumWords, bool t) {
    memset(B, 0 - (int)t, NumWords*sizeof(BitWord));
  } 
};

inline BitVector operator&(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result &= RHS;
  return Result;
}

inline BitVector operator|(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result |= RHS;
  return Result;
}

inline BitVector operator^(const BitVector &LHS, const BitVector &RHS) {
  BitVector Result(LHS);
  Result ^= RHS;
  return Result;
}
 
} // End llvm namespace
#endif

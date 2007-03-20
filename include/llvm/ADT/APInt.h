//===-- llvm/Support/APInt.h - For Arbitrary Precision Integer -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Sheng Zhou and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision integral
// constant values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_APINT_H
#define LLVM_APINT_H

#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <string>

namespace llvm {

/// Forward declaration.
class APInt;
namespace APIntOps {
  APInt udiv(const APInt& LHS, const APInt& RHS);
  APInt urem(const APInt& LHS, const APInt& RHS);
}

//===----------------------------------------------------------------------===//
//                              APInt Class
//===----------------------------------------------------------------------===//

/// APInt - This class represents arbitrary precision constant integral values.
/// It is a functional replacement for common case unsigned integer type like 
/// "unsigned", "unsigned long" or "uint64_t", but also allows non-byte-width 
/// integer sizes and large integer value types such as 3-bits, 15-bits, or more
/// than 64-bits of precision. APInt provides a variety of arithmetic operators 
/// and methods to manipulate integer values of any bit-width. It supports both
/// the typical integer arithmetic and comparison operations as well as bitwise
/// manipulation.
///
/// The class has several invariants worth noting:
///   * All bit, byte, and word positions are zero-based.
///   * Once the bit width is set, it doesn't change except by the Truncate, 
///     SignExtend, or ZeroExtend operations.
///   * All binary operators must be on APInt instances of the same bit width.
///     Attempting to use these operators on instances with different bit 
///     widths will yield an assertion.
///   * The value is stored canonically as an unsigned value. For operations
///     where it makes a difference, there are both signed and unsigned variants
///     of the operation. For example, sdiv and udiv. However, because the bit
///     widths must be the same, operations such as Mul and Add produce the same
///     results regardless of whether the values are interpreted as signed or
///     not.
///   * In general, the class tries to follow the style of computation that LLVM
///     uses in its IR. This simplifies its use for LLVM.
///
/// @brief Class for arbitrary precision integers.
class APInt {

  uint32_t BitWidth;      ///< The number of bits in this APInt.

  /// This union is used to store the integer value. When the
  /// integer bit-width <= 64, it uses VAL; 
  /// otherwise it uses the pVal.
  union {
    uint64_t VAL;    ///< Used to store the <= 64 bits integer value.
    uint64_t *pVal;  ///< Used to store the >64 bits integer value.
  };

  /// This enum is just used to hold a constant we needed for APInt.
  enum {
    APINT_BITS_PER_WORD = sizeof(uint64_t) * 8,
    APINT_WORD_SIZE = sizeof(uint64_t)
  };

  // Fast internal constructor
  APInt(uint64_t* val, uint32_t bits) : BitWidth(bits), pVal(val) { }

  /// @returns true if the number of bits <= 64, false otherwise.
  /// @brief Determine if this APInt just has one word to store value.
  inline bool isSingleWord() const { 
    return BitWidth <= APINT_BITS_PER_WORD; 
  }

  /// @returns the word position for the specified bit position.
  static inline uint32_t whichWord(uint32_t bitPosition) { 
    return bitPosition / APINT_BITS_PER_WORD; 
  }

  /// @returns the bit position in a word for the specified bit position 
  /// in APInt.
  static inline uint32_t whichBit(uint32_t bitPosition) { 
    return bitPosition % APINT_BITS_PER_WORD; 
  }

  /// @returns a uint64_t type integer with just bit position at
  /// "whichBit(bitPosition)" setting, others zero.
  static inline uint64_t maskBit(uint32_t bitPosition) { 
    return 1ULL << whichBit(bitPosition); 
  }

  /// This method is used internally to clear the to "N" bits that are not used
  /// by the APInt. This is needed after the most significant word is assigned 
  /// a value to ensure that those bits are zero'd out.
  /// @brief Clear high order bits
  inline APInt& clearUnusedBits() {
    // Compute how many bits are used in the final word
    uint32_t wordBits = BitWidth % APINT_BITS_PER_WORD;
    if (wordBits == 0)
      // If all bits are used, we want to leave the value alone. This also
      // avoids the undefined behavior of >> when the shfit is the same size as
      // the word size (64).
      return *this;

    // Mask out the hight bits.
    uint64_t mask = ~uint64_t(0ULL) >> (APINT_BITS_PER_WORD - wordBits);
    if (isSingleWord())
      VAL &= mask;
    else
      pVal[getNumWords() - 1] &= mask;
    return *this;
  }

  /// @returns the corresponding word for the specified bit position.
  /// @brief Get the word corresponding to a bit position
  inline uint64_t getWord(uint32_t bitPosition) const { 
    return isSingleWord() ? VAL : pVal[whichWord(bitPosition)]; 
  }

  /// This is used by the constructors that take string arguments.
  /// @brief Converts a char array into an APInt
  void fromString(uint32_t numBits, const char *StrStart, uint32_t slen, 
                  uint8_t radix);

  /// This is used by the toString method to divide by the radix. It simply
  /// provides a more convenient form of divide for internal use since KnuthDiv
  /// has specific constraints on its inputs. If those constraints are not met
  /// then it provides a simpler form of divide.
  /// @brief An internal division function for dividing APInts.
  static void divide(const APInt LHS, uint32_t lhsWords, 
                     const APInt &RHS, uint32_t rhsWords,
                     APInt *Quotient, APInt *Remainder);

#ifndef NDEBUG
  /// @brief debug method
  void dump() const;
#endif

public:
  /// @brief Create a new APInt of numBits width, initialized as val.
  APInt(uint32_t numBits, uint64_t val, bool isSigned = false);

  /// Note that numWords can be smaller or larger than the corresponding bit
  /// width but any extraneous bits will be dropped.
  /// @brief Create a new APInt of numBits width, initialized as bigVal[].
  APInt(uint32_t numBits, uint32_t numWords, uint64_t bigVal[]);

  /// @brief Create a new APInt by translating the string represented 
  /// integer value.
  APInt(uint32_t numBits, const std::string& Val, uint8_t radix);

  /// @brief Create a new APInt by translating the char array represented
  /// integer value.
  APInt(uint32_t numBits, const char StrStart[], uint32_t slen, uint8_t radix);

  /// @brief Copy Constructor.
  APInt(const APInt& API);

  /// @brief Destructor.
  ~APInt();

  /// @brief Copy assignment operator. 
  APInt& operator=(const APInt& RHS);

  /// Assigns an integer value to the APInt.
  /// @brief Assignment operator. 
  APInt& operator=(uint64_t RHS);

  /// Increments the APInt by one.
  /// @brief Postfix increment operator.
  inline const APInt operator++(int) {
    APInt API(*this);
    ++(*this);
    return API;
  }

  /// Increments the APInt by one.
  /// @brief Prefix increment operator.
  APInt& operator++();

  /// Decrements the APInt by one.
  /// @brief Postfix decrement operator. 
  inline const APInt operator--(int) {
    APInt API(*this);
    --(*this);
    return API;
  }

  /// Decrements the APInt by one.
  /// @brief Prefix decrement operator. 
  APInt& operator--();

  /// Performs bitwise AND operation on this APInt and the given APInt& RHS, 
  /// assigns the result to this APInt.
  /// @brief Bitwise AND assignment operator. 
  APInt& operator&=(const APInt& RHS);

  /// Performs bitwise OR operation on this APInt and the given APInt& RHS, 
  /// assigns the result to this APInt.
  /// @brief Bitwise OR assignment operator. 
  APInt& operator|=(const APInt& RHS);

  /// Performs bitwise XOR operation on this APInt and the given APInt& RHS, 
  /// assigns the result to this APInt.
  /// @brief Bitwise XOR assignment operator. 
  APInt& operator^=(const APInt& RHS);

  /// Performs a bitwise complement operation on this APInt.
  /// @brief Bitwise complement operator. 
  APInt operator~() const;

  /// Multiplies this APInt by the  given APInt& RHS and 
  /// assigns the result to this APInt.
  /// @brief Multiplication assignment operator. 
  APInt& operator*=(const APInt& RHS);

  /// Adds this APInt by the given APInt& RHS and 
  /// assigns the result to this APInt.
  /// @brief Addition assignment operator. 
  APInt& operator+=(const APInt& RHS);

  /// Subtracts this APInt by the given APInt &RHS and 
  /// assigns the result to this APInt.
  /// @brief Subtraction assignment operator. 
  APInt& operator-=(const APInt& RHS);

  /// Performs bitwise AND operation on this APInt and 
  /// the given APInt& RHS.
  /// @brief Bitwise AND operator. 
  APInt operator&(const APInt& RHS) const;
  APInt And(const APInt& RHS) const {
    return this->operator&(RHS);
  }

  /// Performs bitwise OR operation on this APInt and the given APInt& RHS.
  /// @brief Bitwise OR operator. 
  APInt operator|(const APInt& RHS) const;
  APInt Or(const APInt& RHS) const {
    return this->operator|(RHS);
  }

  /// Performs bitwise XOR operation on this APInt and the given APInt& RHS.
  /// @brief Bitwise XOR operator. 
  APInt operator^(const APInt& RHS) const;
  APInt Xor(const APInt& RHS) const {
    return this->operator^(RHS);
  }

  /// Performs logical negation operation on this APInt.
  /// @brief Logical negation operator. 
  bool operator !() const;

  /// Multiplies this APInt by the given APInt& RHS.
  /// @brief Multiplication operator. 
  APInt operator*(const APInt& RHS) const;

  /// Adds this APInt by the given APInt& RHS.
  /// @brief Addition operator. 
  APInt operator+(const APInt& RHS) const;
  APInt operator+(uint64_t RHS) const {
    return (*this) + APInt(BitWidth, RHS);
  }


  /// Subtracts this APInt by the given APInt& RHS
  /// @brief Subtraction operator. 
  APInt operator-(const APInt& RHS) const;
  APInt operator-(uint64_t RHS) const {
    return (*this) - APInt(BitWidth, RHS);
  }

  /// @brief Unary negation operator
  inline APInt operator-() const {
    return APInt(BitWidth, 0) - (*this);
  }

  /// @brief Array-indexing support.
  bool operator[](uint32_t bitPosition) const;

  /// Compare this APInt with the given APInt& RHS 
  /// for the validity of the equality relationship.
  /// @brief Equality operator. 
  bool operator==(const APInt& RHS) const;

  /// Compare this APInt with the given uint64_t value
  /// for the validity of the equality relationship.
  /// @brief Equality operator.
  bool operator==(uint64_t Val) const;

  /// Compare this APInt with the given APInt& RHS 
  /// for the validity of the inequality relationship.
  /// @brief Inequality operator. 
  inline bool operator!=(const APInt& RHS) const {
    return !((*this) == RHS);
  }

  /// Compare this APInt with the given uint64_t value 
  /// for the validity of the inequality relationship.
  /// @brief Inequality operator. 
  inline bool operator!=(uint64_t Val) const {
    return !((*this) == Val);
  }
  
  /// @brief Equality comparison
  bool eq(const APInt &RHS) const {
    return (*this) == RHS; 
  }

  /// @brief Inequality comparison
  bool ne(const APInt &RHS) const {
    return !((*this) == RHS);
  }

  /// @brief Unsigned less than comparison
  bool ult(const APInt& RHS) const;

  /// @brief Signed less than comparison
  bool slt(const APInt& RHS) const;

  /// @brief Unsigned less or equal comparison
  bool ule(const APInt& RHS) const {
    return ult(RHS) || eq(RHS);
  }

  /// @brief Signed less or equal comparison
  bool sle(const APInt& RHS) const {
    return slt(RHS) || eq(RHS);
  }

  /// @brief Unsigned greather than comparison
  bool ugt(const APInt& RHS) const {
    return !ult(RHS) && !eq(RHS);
  }

  /// @brief Signed greather than comparison
  bool sgt(const APInt& RHS) const {
    return !slt(RHS) && !eq(RHS);
  }

  /// @brief Unsigned greater or equal comparison
  bool uge(const APInt& RHS) const {
    return !ult(RHS);
  }

  /// @brief Signed greather or equal comparison
  bool sge(const APInt& RHS) const {
    return !slt(RHS);
  }

  /// This just tests the high bit of this APInt to determine if it is negative.
  /// @returns true if this APInt is negative, false otherwise
  /// @brief Determine sign of this APInt.
  bool isNegative() const {
    return (*this)[BitWidth - 1];
  }

  /// This just tests the high bit of the APInt to determine if the value is
  /// positove or not.
  /// @brief Determine if this APInt Value is positive.
  bool isPositive() const {
    return !isNegative();
  }

  /// This just tests if the value of this APInt is strictly positive (> 0).
  /// @brief Determine if this APInt Value is strictly positive.
  inline bool isStrictlyPositive() const {
    return isPositive() && (*this) != 0;
  }

  /// Arithmetic right-shift this APInt by shiftAmt.
  /// @brief Arithmetic right-shift function.
  APInt ashr(uint32_t shiftAmt) const;

  /// Logical right-shift this APInt by shiftAmt.
  /// @brief Logical right-shift function.
  APInt lshr(uint32_t shiftAmt) const;

  /// Left-shift this APInt by shiftAmt.
  /// @brief Left-shift function.
  APInt shl(uint32_t shiftAmt) const;

  /// Left-shift this APInt by shiftAmt and
  /// assigns the result to this APInt.
  /// @brief Lef-shift assignment function.
  inline APInt& operator<<=(uint32_t shiftAmt) {
    *this = shl(shiftAmt);
    return *this;
  }

  /// Signed divide this APInt by APInt RHS.
  /// @brief Signed division function for APInt.
  inline APInt sdiv(const APInt& RHS) const {
    bool isNegativeLHS = isNegative();
    bool isNegativeRHS = RHS.isNegative();
    APInt Result = APIntOps::udiv(
        isNegativeLHS ? -(*this) : (*this), isNegativeRHS ? -RHS : RHS);
    return isNegativeLHS != isNegativeRHS ? -Result : Result;
  }

  /// Unsigned divide this APInt by APInt RHS.
  /// @brief Unsigned division function for APInt.
  APInt udiv(const APInt& RHS) const;

  /// Signed remainder operation on APInt.
  /// @brief Function for signed remainder operation.
  inline APInt srem(const APInt& RHS) const {
    bool isNegativeLHS = isNegative();
    bool isNegativeRHS = RHS.isNegative();
    APInt Result = APIntOps::urem(
        isNegativeLHS ? -(*this) : (*this), isNegativeRHS ? -RHS : RHS);
    return isNegativeLHS ? -Result : Result;
  }

  /// Unsigned remainder operation on APInt.
  /// @brief Function for unsigned remainder operation.
  APInt urem(const APInt& RHS) const;

  /// Truncate the APInt to a specified width. It is an error to specify a width
  /// that is greater than or equal to the current width. 
  /// @brief Truncate to new width.
  APInt &trunc(uint32_t width);

  /// This operation sign extends the APInt to a new width. If the high order
  /// bit is set, the fill on the left will be done with 1 bits, otherwise zero.
  /// It is an error to specify a width that is less than or equal to the 
  /// current width.
  /// @brief Sign extend to a new width.
  APInt &sext(uint32_t width);

  /// This operation zero extends the APInt to a new width. Thie high order bits
  /// are filled with 0 bits.  It is an error to specify a width that is less 
  /// than or equal to the current width.
  /// @brief Zero extend to a new width.
  APInt &zext(uint32_t width);

  /// Make this APInt have the bit width given by \p width. The value is sign
  /// extended, truncated, or left alone to make it that width.
  /// @brief Sign extend or truncate to width
  APInt &sextOrTrunc(uint32_t width);

  /// Make this APInt have the bit width given by \p width. The value is zero
  /// extended, truncated, or left alone to make it that width.
  /// @brief Zero extend or truncate to width
  APInt &zextOrTrunc(uint32_t width);

  /// This is a help function for convenience. If the given \p width equals to
  /// this APInt's BitWidth, just return this APInt, otherwise, just zero 
  /// extend it.
  inline APInt &zextOrCopy(uint32_t width) {
    if (width == BitWidth)
      return *this;
    return zext(width);
  }

  /// @brief Set every bit to 1.
  APInt& set();

  /// Set the given bit to 1 whose position is given as "bitPosition".
  /// @brief Set a given bit to 1.
  APInt& set(uint32_t bitPosition);

  /// @brief Set every bit to 0.
  APInt& clear();

  /// Set the given bit to 0 whose position is given as "bitPosition".
  /// @brief Set a given bit to 0.
  APInt& clear(uint32_t bitPosition);

  /// @brief Toggle every bit to its opposite value.
  APInt& flip();

  /// Toggle a given bit to its opposite value whose position is given 
  /// as "bitPosition".
  /// @brief Toggles a given bit to its opposite value.
  APInt& flip(uint32_t bitPosition);

  inline void setWordToValue(uint32_t idx, uint64_t Val) {
    assert(idx < getNumWords() && "Invalid word array index");
    if (isSingleWord())
      VAL = Val;
    else
      pVal[idx] = Val;
  }

  /// This function returns the number of active bits which is defined as the
  /// bit width minus the number of leading zeros. This is used in several
  /// computations to see how "wide" the value is.
  /// @brief Compute the number of active bits in the value
  inline uint32_t getActiveBits() const {
    return BitWidth - countLeadingZeros();
  }

  /// This function returns the number of active words in the value of this
  /// APInt. This is used in conjunction with getActiveData to extract the raw
  /// value of the APInt.
  inline uint32_t getActiveWords() const {
    return whichWord(getActiveBits()-1) + 1;
  }

  /// Here one word's bitwidth equals to that of uint64_t.
  /// @returns the number of words to hold the integer value of this APInt.
  /// @brief Get the number of words.
  inline uint32_t getNumWords() const {
    return (BitWidth + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD;
  }

  /// This function returns a pointer to the internal storage of the APInt. 
  /// This is useful for writing out the APInt in binary form without any
  /// conversions.
  inline const uint64_t* getRawData() const {
    if (isSingleWord())
      return &VAL;
    return &pVal[0];
  }

  /// Computes the minimum bit width for this APInt while considering it to be
  /// a signed (and probably negative) value. If the value is not negative, 
  /// this function returns the same value as getActiveBits(). Otherwise, it
  /// returns the smallest bit width that will retain the negative value. For
  /// example, -1 can be written as 0b1 or 0xFFFFFFFFFF. 0b1 is shorter and so
  /// for -1, this function will always return 1.
  /// @brief Get the minimum bit size for this signed APInt 
  inline uint32_t getMinSignedBits() const {
    if (isNegative())
      return BitWidth - countLeadingOnes() + 1;
    return getActiveBits();
  }

  /// This method attempts to return the value of this APInt as a zero extended
  /// uint64_t. The bitwidth must be <= 64 or the value must fit within a
  /// uint64_t. Otherwise an assertion will result.
  /// @brief Get zero extended value
  inline uint64_t getZExtValue() const {
    if (isSingleWord())
      return VAL;
    assert(getActiveBits() <= 64 && "Too many bits for uint64_t");
    return pVal[0];
  }

  /// This method attempts to return the value of this APInt as a sign extended
  /// int64_t. The bit width must be <= 64 or the value must fit within an
  /// int64_t. Otherwise an assertion will result.
  /// @brief Get sign extended value
  inline int64_t getSExtValue() const {
    if (isSingleWord())
      return int64_t(VAL << (APINT_BITS_PER_WORD - BitWidth)) >> 
                     (APINT_BITS_PER_WORD - BitWidth);
    assert(getActiveBits() <= 64 && "Too many bits for int64_t");
    return int64_t(pVal[0]);
  }

  /// @brief Gets maximum unsigned value of APInt for specific bit width.
  static APInt getMaxValue(uint32_t numBits) {
    return APInt(numBits, 0).set();
  }

  /// @brief Gets maximum signed value of APInt for a specific bit width.
  static APInt getSignedMaxValue(uint32_t numBits) {
    return APInt(numBits, 0).set().clear(numBits - 1);
  }

  /// @brief Gets minimum unsigned value of APInt for a specific bit width.
  static APInt getMinValue(uint32_t numBits) {
    return APInt(numBits, 0);
  }

  /// @brief Gets minimum signed value of APInt for a specific bit width.
  static APInt getSignedMinValue(uint32_t numBits) {
    return APInt(numBits, 0).set(numBits - 1);
  }

  /// getSignBit - This is just a wrapper function of getSignedMinValue(), and
  /// it helps code readability when we want to get a SignBit.
  /// @brief Get the SignBit for a specific bit width.
  inline static APInt getSignBit(uint32_t BitWidth) {
    return getSignedMinValue(BitWidth);
  }

  /// @returns the all-ones value for an APInt of the specified bit-width.
  /// @brief Get the all-ones value.
  static APInt getAllOnesValue(uint32_t numBits) {
    return APInt(numBits, 0).set();
  }

  /// @returns the '0' value for an APInt of the specified bit-width.
  /// @brief Get the '0' value.
  static APInt getNullValue(uint32_t numBits) {
    return APInt(numBits, 0);
  }

  /// The hash value is computed as the sum of the words and the bit width.
  /// @returns A hash value computed from the sum of the APInt words.
  /// @brief Get a hash value based on this APInt
  uint64_t getHashValue() const;

  /// This converts the APInt to a boolean valy as a test against zero.
  /// @brief Boolean conversion function. 
  inline bool getBoolValue() const {
    return countLeadingZeros() != BitWidth;
  }

  /// This checks to see if the value has all bits of the APInt are set or not.
  /// @brief Determine if all bits are set
  inline bool isAllOnesValue() const {
    return countPopulation() == BitWidth;
  }

  /// This checks to see if the value of this APInt is the maximum unsigned
  /// value for the APInt's bit width.
  /// @brief Determine if this is the largest unsigned value.
  bool isMaxValue() const {
    return countPopulation() == BitWidth;
  }

  /// This checks to see if the value of this APInt is the maximum signed
  /// value for the APInt's bit width.
  /// @brief Determine if this is the largest signed value.
  bool isMaxSignedValue() const {
    return BitWidth == 1 ? VAL == 0 :
                          !isNegative() && countPopulation() == BitWidth - 1;
  }

  /// This checks to see if the value of this APInt is the minimum signed
  /// value for the APInt's bit width.
  /// @brief Determine if this is the smallest unsigned value.
  bool isMinValue() const {
    return countPopulation() == 0;
  }

  /// This checks to see if the value of this APInt is the minimum signed
  /// value for the APInt's bit width.
  /// @brief Determine if this is the smallest signed value.
  bool isMinSignedValue() const {
    return BitWidth == 1 ? VAL == 1 :
                           isNegative() && countPopulation() == 1;
  }

  /// This is used internally to convert an APInt to a string.
  /// @brief Converts an APInt to a std::string
  std::string toString(uint8_t radix, bool wantSigned) const;

  /// Considers the APInt to be unsigned and converts it into a string in the
  /// radix given. The radix can be 2, 8, 10 or 16.
  /// @returns a character interpretation of the APInt
  /// @brief Convert unsigned APInt to string representation.
  inline std::string toString(uint8_t radix = 10) const {
    return toString(radix, false);
  }

  /// Considers the APInt to be unsigned and converts it into a string in the
  /// radix given. The radix can be 2, 8, 10 or 16.
  /// @returns a character interpretation of the APInt
  /// @brief Convert unsigned APInt to string representation.
  inline std::string toStringSigned(uint8_t radix = 10) const {
    return toString(radix, true);
  }

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the low bits and right shift to the least significant bit.
  /// @returns the high "numBits" bits of this APInt.
  APInt getHiBits(uint32_t numBits) const;

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the high bits.
  /// @returns the low "numBits" bits of this APInt.
  APInt getLoBits(uint32_t numBits) const;

  /// @returns true if the argument APInt value is a power of two > 0.
  bool isPowerOf2() const; 

  /// countLeadingZeros - This function is an APInt version of the
  /// countLeadingZeros_{32,64} functions in MathExtras.h. It counts the number
  /// of zeros from the most significant bit to the first one bit.
  /// @returns getNumWords() * APINT_BITS_PER_WORD if the value is zero.
  /// @returns the number of zeros from the most significant bit to the first
  /// one bits.
  /// @brief Count the number of leading one bits.
  uint32_t countLeadingZeros() const;

  /// countLeadingOnes - This function counts the number of contiguous 1 bits
  /// in the high order bits. The count stops when the first 0 bit is reached.
  /// @returns 0 if the high order bit is not set
  /// @returns the number of 1 bits from the most significant to the least
  /// @brief Count the number of leading one bits.
  uint32_t countLeadingOnes() const;

  /// countTrailingZeros - This function is an APInt version of the 
  /// countTrailingZoers_{32,64} functions in MathExtras.h. It counts 
  /// the number of zeros from the least significant bit to the first one bit.
  /// @returns getNumWords() * APINT_BITS_PER_WORD if the value is zero.
  /// @returns the number of zeros from the least significant bit to the first
  /// one bit.
  /// @brief Count the number of trailing zero bits.
  uint32_t countTrailingZeros() const;

  /// countPopulation - This function is an APInt version of the
  /// countPopulation_{32,64} functions in MathExtras.h. It counts the number
  /// of 1 bits in the APInt value. 
  /// @returns 0 if the value is zero.
  /// @returns the number of set bits.
  /// @brief Count the number of bits set.
  uint32_t countPopulation() const; 

  /// @returns the total number of bits.
  inline uint32_t getBitWidth() const { 
    return BitWidth; 
  }

  /// @brief Check if this APInt has a N-bits integer value.
  inline bool isIntN(uint32_t N) const {
    assert(N && "N == 0 ???");
    if (isSingleWord()) {
      return VAL == (VAL & (~0ULL >> (64 - N)));
    } else {
      APInt Tmp(N, getNumWords(), pVal);
      return Tmp == (*this);
    }
  }

  /// @returns a byte-swapped representation of this APInt Value.
  APInt byteSwap() const;

  /// @returns the floor log base 2 of this APInt.
  inline uint32_t logBase2() const {
    return BitWidth - 1 - countLeadingZeros();
  }

  /// @brief Converts this APInt to a double value.
  double roundToDouble(bool isSigned) const;

  /// @brief Converts this unsigned APInt to a double value.
  double roundToDouble() const {
    return roundToDouble(false);
  }

  /// @brief Converts this signed APInt to a double value.
  double signedRoundToDouble() const {
    return roundToDouble(true);
  }

  /// The conversion does not do a translation from integer to double, it just
  /// re-interprets the bits as a double. Note that it is valid to do this on
  /// any bit width. Exactly 64 bits will be translated.
  /// @brief Converts APInt bits to a double
  double bitsToDouble() const {
    union {
      uint64_t I;
      double D;
    } T;
    T.I = (isSingleWord() ? VAL : pVal[0]);
    return T.D;
  }

  /// The conversion does not do a translation from integer to float, it just
  /// re-interprets the bits as a float. Note that it is valid to do this on
  /// any bit width. Exactly 32 bits will be translated.
  /// @brief Converts APInt bits to a double
  float bitsToFloat() const {
    union {
      uint32_t I;
      float F;
    } T;
    T.I = uint32_t((isSingleWord() ? VAL : pVal[0]));
    return T.F;
  }

  /// The conversion does not do a translation from double to integer, it just
  /// re-interprets the bits of the double. Note that it is valid to do this on
  /// any bit width but bits from V may get truncated.
  /// @brief Converts a double to APInt bits.
  APInt& doubleToBits(double V) {
    union {
      uint64_t I;
      double D;
    } T;
    T.D = V;
    if (isSingleWord())
      VAL = T.I;
    else
      pVal[0] = T.I;
    return clearUnusedBits();
  }

  /// The conversion does not do a translation from float to integer, it just
  /// re-interprets the bits of the float. Note that it is valid to do this on
  /// any bit width but bits from V may get truncated.
  /// @brief Converts a float to APInt bits.
  APInt& floatToBits(float V) {
    union {
      uint32_t I;
      float F;
    } T;
    T.F = V;
    if (isSingleWord())
      VAL = T.I;
    else
      pVal[0] = T.I;
    return clearUnusedBits();
  }

  /// @brief Compute the square root
  APInt sqrt() const;

  /// If *this is < 0 then return -(*this), otherwise *this;
  /// @brief Get the absolute value;
  APInt abs() const {
    if (isNegative())
      return -(*this);
    return *this;
  }
};

inline bool operator==(uint64_t V1, const APInt& V2) {
  return V2 == V1;
}

inline bool operator!=(uint64_t V1, const APInt& V2) {
  return V2 != V1;
}

namespace APIntOps {

/// @brief Determine the smaller of two APInts considered to be signed.
inline APInt smin(const APInt &A, const APInt &B) {
  return A.slt(B) ? A : B;
}

/// @brief Determine the larger of two APInts considered to be signed.
inline APInt smax(const APInt &A, const APInt &B) {
  return A.sgt(B) ? A : B;
}

/// @brief Determine the smaller of two APInts considered to be signed.
inline APInt umin(const APInt &A, const APInt &B) {
  return A.ult(B) ? A : B;
}

/// @brief Determine the larger of two APInts considered to be unsigned.
inline APInt umax(const APInt &A, const APInt &B) {
  return A.ugt(B) ? A : B;
}

/// @brief Check if the specified APInt has a N-bits integer value.
inline bool isIntN(uint32_t N, const APInt& APIVal) {
  return APIVal.isIntN(N);
}

/// @returns true if the argument APInt value is a sequence of ones
/// starting at the least significant bit with the remainder zero.
inline const bool isMask(uint32_t numBits, const APInt& APIVal) {
  return APIVal.getBoolValue() && ((APIVal + APInt(numBits,1)) & APIVal) == 0;
}

/// @returns true if the argument APInt value contains a sequence of ones
/// with the remainder zero.
inline const bool isShiftedMask(uint32_t numBits, const APInt& APIVal) {
  return isMask(numBits, (APIVal - APInt(numBits,1)) | APIVal);
}

/// @returns a byte-swapped representation of the specified APInt Value.
inline APInt byteSwap(const APInt& APIVal) {
  return APIVal.byteSwap();
}

/// @returns the floor log base 2 of the specified APInt value.
inline uint32_t logBase2(const APInt& APIVal) {
  return APIVal.logBase2(); 
}

/// GreatestCommonDivisor - This function returns the greatest common
/// divisor of the two APInt values using Enclid's algorithm.
/// @returns the greatest common divisor of Val1 and Val2
/// @brief Compute GCD of two APInt values.
APInt GreatestCommonDivisor(const APInt& Val1, const APInt& Val2);

/// Treats the APInt as an unsigned value for conversion purposes.
/// @brief Converts the given APInt to a double value.
inline double RoundAPIntToDouble(const APInt& APIVal) {
  return APIVal.roundToDouble();
}

/// Treats the APInt as a signed value for conversion purposes.
/// @brief Converts the given APInt to a double value.
inline double RoundSignedAPIntToDouble(const APInt& APIVal) {
  return APIVal.signedRoundToDouble();
}

/// @brief Converts the given APInt to a float vlalue.
inline float RoundAPIntToFloat(const APInt& APIVal) {
  return float(RoundAPIntToDouble(APIVal));
}

/// Treast the APInt as a signed value for conversion purposes.
/// @brief Converts the given APInt to a float value.
inline float RoundSignedAPIntToFloat(const APInt& APIVal) {
  return float(APIVal.signedRoundToDouble());
}

/// RoundDoubleToAPInt - This function convert a double value to an APInt value.
/// @brief Converts the given double value into a APInt.
APInt RoundDoubleToAPInt(double Double, uint32_t width);

/// RoundFloatToAPInt - Converts a float value into an APInt value.
/// @brief Converts a float value into a APInt.
inline APInt RoundFloatToAPInt(float Float, uint32_t width) {
  return RoundDoubleToAPInt(double(Float), width);
}

/// Arithmetic right-shift the APInt by shiftAmt.
/// @brief Arithmetic right-shift function.
inline APInt ashr(const APInt& LHS, uint32_t shiftAmt) {
  return LHS.ashr(shiftAmt);
}

/// Logical right-shift the APInt by shiftAmt.
/// @brief Logical right-shift function.
inline APInt lshr(const APInt& LHS, uint32_t shiftAmt) {
  return LHS.lshr(shiftAmt);
}

/// Left-shift the APInt by shiftAmt.
/// @brief Left-shift function.
inline APInt shl(const APInt& LHS, uint32_t shiftAmt) {
  return LHS.shl(shiftAmt);
}

/// Signed divide APInt LHS by APInt RHS.
/// @brief Signed division function for APInt.
inline APInt sdiv(const APInt& LHS, const APInt& RHS) {
  return LHS.sdiv(RHS);
}

/// Unsigned divide APInt LHS by APInt RHS.
/// @brief Unsigned division function for APInt.
inline APInt udiv(const APInt& LHS, const APInt& RHS) {
  return LHS.udiv(RHS);
}

/// Signed remainder operation on APInt.
/// @brief Function for signed remainder operation.
inline APInt srem(const APInt& LHS, const APInt& RHS) {
  return LHS.srem(RHS);
}

/// Unsigned remainder operation on APInt.
/// @brief Function for unsigned remainder operation.
inline APInt urem(const APInt& LHS, const APInt& RHS) {
  return LHS.urem(RHS);
}

/// Performs multiplication on APInt values.
/// @brief Function for multiplication operation.
inline APInt mul(const APInt& LHS, const APInt& RHS) {
  return LHS * RHS;
}

/// Performs addition on APInt values.
/// @brief Function for addition operation.
inline APInt add(const APInt& LHS, const APInt& RHS) {
  return LHS + RHS;
}

/// Performs subtraction on APInt values.
/// @brief Function for subtraction operation.
inline APInt sub(const APInt& LHS, const APInt& RHS) {
  return LHS - RHS;
}

/// Performs bitwise AND operation on APInt LHS and 
/// APInt RHS.
/// @brief Bitwise AND function for APInt.
inline APInt And(const APInt& LHS, const APInt& RHS) {
  return LHS & RHS;
}

/// Performs bitwise OR operation on APInt LHS and APInt RHS.
/// @brief Bitwise OR function for APInt. 
inline APInt Or(const APInt& LHS, const APInt& RHS) {
  return LHS | RHS;
}

/// Performs bitwise XOR operation on APInt.
/// @brief Bitwise XOR function for APInt.
inline APInt Xor(const APInt& LHS, const APInt& RHS) {
  return LHS ^ RHS;
} 

/// Performs a bitwise complement operation on APInt.
/// @brief Bitwise complement function. 
inline APInt Not(const APInt& APIVal) {
  return ~APIVal;
}

} // End of APIntOps namespace

} // End of llvm namespace

#endif

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
// constant values and operations on them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_APINT_H
#define LLVM_APINT_H

#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <string>

namespace llvm {

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
  /// integer bit-width <= 64, it uses VAL, otherwise it uses pVal.
  union {
    uint64_t VAL;    ///< Used to store the <= 64 bits integer value.
    uint64_t *pVal;  ///< Used to store the >64 bits integer value.
  };

  /// This enum is used to hold the constants we needed for APInt.
  enum {
    APINT_BITS_PER_WORD = sizeof(uint64_t) * 8, ///< Bits in a word
    APINT_WORD_SIZE = sizeof(uint64_t)          ///< Byte size of a word
  };

  /// This constructor is used only internally for speed of construction of
  /// temporaries. It is unsafe for general use so it is not public.
  /// @brief Fast internal constructor
  APInt(uint64_t* val, uint32_t bits) : BitWidth(bits), pVal(val) { }

  /// @returns true if the number of bits <= 64, false otherwise.
  /// @brief Determine if this APInt just has one word to store value.
  inline bool isSingleWord() const { 
    return BitWidth <= APINT_BITS_PER_WORD; 
  }

  /// @returns the word position for the specified bit position.
  /// @brief Determine which word a bit is in.
  static inline uint32_t whichWord(uint32_t bitPosition) { 
    return bitPosition / APINT_BITS_PER_WORD; 
  }

  /// @returns the bit position in a word for the specified bit position 
  /// in the APInt.
  /// @brief Determine which bit in a word a bit is in.
  static inline uint32_t whichBit(uint32_t bitPosition) { 
    return bitPosition % APINT_BITS_PER_WORD; 
  }

  /// This method generates and returns a uint64_t (word) mask for a single 
  /// bit at a specific bit position. This is used to mask the bit in the 
  /// corresponding word.
  /// @returns a uint64_t with only bit at "whichBit(bitPosition)" set
  /// @brief Get a single bit mask.
  static inline uint64_t maskBit(uint32_t bitPosition) { 
    return 1ULL << whichBit(bitPosition); 
  }

  /// This method is used internally to clear the to "N" bits in the high order
  /// word that are not used by the APInt. This is needed after the most 
  /// significant word is assigned a value to ensure that those bits are 
  /// zero'd out.
  /// @brief Clear unused high order bits
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
  /// @brief Convert a char array into an APInt
  void fromString(uint32_t numBits, const char *strStart, uint32_t slen, 
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
  /// @name Constructors
  /// @{
  /// If isSigned is true then val is treated as if it were a signed value
  /// (i.e. as an int64_t) and the appropriate sign extension to the bit width
  /// will be done. Otherwise, no sign extension occurs (high order bits beyond
  /// the range of val are zero filled).
  /// @param numBits the bit width of the constructed APInt
  /// @param val the initial value of the APInt
  /// @param isSigned how to treat signedness of val
  /// @brief Create a new APInt of numBits width, initialized as val.
  APInt(uint32_t numBits, uint64_t val, bool isSigned = false);

  /// Note that numWords can be smaller or larger than the corresponding bit
  /// width but any extraneous bits will be dropped.
  /// @param numBits the bit width of the constructed APInt
  /// @param numWords the number of words in bigVal
  /// @param bigVal a sequence of words to form the initial value of the APInt
  /// @brief Construct an APInt of numBits width, initialized as bigVal[].
  APInt(uint32_t numBits, uint32_t numWords, uint64_t bigVal[]);

  /// This constructor interprets Val as a string in the given radix. The 
  /// interpretation stops when the first charater that is not suitable for the
  /// radix is encountered. Acceptable radix values are 2, 8, 10 and 16. It is
  /// an error for the value implied by the string to require more bits than 
  /// numBits.
  /// @param numBits the bit width of the constructed APInt
  /// @param val the string to be interpreted
  /// @param radix the radix of Val to use for the intepretation
  /// @brief Construct an APInt from a string representation.
  APInt(uint32_t numBits, const std::string& val, uint8_t radix);

  /// This constructor interprets the slen characters starting at StrStart as
  /// a string in the given radix. The interpretation stops when the first 
  /// character that is not suitable for the radix is encountered. Acceptable
  /// radix values are 2, 8, 10 and 16. It is an error for the value implied by
  /// the string to require more bits than numBits.
  /// @param numBits the bit width of the constructed APInt
  /// @param strStart the start of the string to be interpreted
  /// @param slen the maximum number of characters to interpret
  /// @brief Construct an APInt from a string representation.
  APInt(uint32_t numBits, const char strStart[], uint32_t slen, uint8_t radix);

  /// Simply makes *this a copy of that.
  /// @brief Copy Constructor.
  APInt(const APInt& that);

  /// @brief Destructor.
  ~APInt();

  /// @}
  /// @name Value Tests
  /// @{
  /// This tests the high bit of this APInt to determine if it is set.
  /// @returns true if this APInt is negative, false otherwise
  /// @brief Determine sign of this APInt.
  bool isNegative() const {
    return (*this)[BitWidth - 1];
  }

  /// This tests the high bit of the APInt to determine if it is unset.
  /// @brief Determine if this APInt Value is positive (not negative).
  bool isPositive() const {
    return !isNegative();
  }

  /// This tests if the value of this APInt is strictly positive (> 0).
  /// @returns true if this APInt is Positive and not zero.
  /// @brief Determine if this APInt Value is strictly positive.
  inline bool isStrictlyPositive() const {
    return isPositive() && (*this) != 0;
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

  /// This checks to see if the value of this APInt is the minimum unsigned
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

  /// @brief Check if this APInt has an N-bits integer value.
  inline bool isIntN(uint32_t N) const {
    assert(N && "N == 0 ???");
    if (isSingleWord()) {
      return VAL == (VAL & (~0ULL >> (64 - N)));
    } else {
      APInt Tmp(N, getNumWords(), pVal);
      return Tmp == (*this);
    }
  }

  /// @returns true if the argument APInt value is a power of two > 0.
  bool isPowerOf2() const; 

  /// isSignBit - Return true if this is the value returned by getSignBit.
  bool isSignBit() const { return isMinSignedValue(); }
  
  /// This converts the APInt to a boolean value as a test against zero.
  /// @brief Boolean conversion function. 
  inline bool getBoolValue() const {
    return *this != 0;
  }

  /// getLimitedValue - If this value is smaller than the specified limit,
  /// return it, otherwise return the limit value.  This causes the value
  /// to saturate to the limit.
  uint64_t getLimitedValue(uint64_t Limit = ~0ULL) const {
    return (getActiveBits() > 64 || getZExtValue() > Limit) ?
      Limit :  getZExtValue();
  }

  /// @}
  /// @name Value Generators
  /// @{
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

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the low bits and right shift to the least significant bit.
  /// @returns the high "numBits" bits of this APInt.
  APInt getHiBits(uint32_t numBits) const;

  /// Get an APInt with the same BitWidth as this APInt, just zero mask
  /// the high bits.
  /// @returns the low "numBits" bits of this APInt.
  APInt getLoBits(uint32_t numBits) const;

  /// Constructs an APInt value that has a contiguous range of bits set. The
  /// bits from loBit to hiBit will be set. All other bits will be zero. For
  /// example, with parameters(32, 15, 0) you would get 0x0000FFFF. If hiBit is
  /// less than loBit then the set bits "wrap". For example, with 
  /// parameters (32, 3, 28), you would get 0xF000000F. 
  /// @param numBits the intended bit width of the result
  /// @param loBit the index of the lowest bit set.
  /// @param hiBit the index of the highest bit set.
  /// @returns An APInt value with the requested bits set.
  /// @brief Get a value with a block of bits set.
  static APInt getBitsSet(uint32_t numBits, uint32_t loBit, uint32_t hiBit) {
    assert(hiBit < numBits && "hiBit out of range");
    assert(loBit < numBits && "loBit out of range");
    if (hiBit < loBit)
      return getLowBitsSet(numBits, hiBit+1) |
             getHighBitsSet(numBits, numBits-loBit+1);
    return getLowBitsSet(numBits, hiBit-loBit+1).shl(loBit);
  }

  /// Constructs an APInt value that has the top hiBitsSet bits set.
  /// @param numBits the bitwidth of the result
  /// @param hiBitsSet the number of high-order bits set in the result.
  /// @brief Get a value with high bits set
  static APInt getHighBitsSet(uint32_t numBits, uint32_t hiBitsSet) {
    assert(hiBitsSet <= numBits && "Too many bits to set!");
    // Handle a degenerate case, to avoid shifting by word size
    if (hiBitsSet == 0)
      return APInt(numBits, 0);
    uint32_t shiftAmt = numBits - hiBitsSet;
    // For small values, return quickly
    if (numBits <= APINT_BITS_PER_WORD)
      return APInt(numBits, ~0ULL << shiftAmt);
    return (~APInt(numBits, 0)).shl(shiftAmt);
  }

  /// Constructs an APInt value that has the bottom loBitsSet bits set.
  /// @param numBits the bitwidth of the result
  /// @param loBitsSet the number of low-order bits set in the result.
  /// @brief Get a value with low bits set
  static APInt getLowBitsSet(uint32_t numBits, uint32_t loBitsSet) {
    assert(loBitsSet <= numBits && "Too many bits to set!");
    // Handle a degenerate case, to avoid shifting by word size
    if (loBitsSet == 0)
      return APInt(numBits, 0);
    if (loBitsSet == APINT_BITS_PER_WORD)
      return APInt(numBits, -1ULL);
    // For small values, return quickly
    if (numBits < APINT_BITS_PER_WORD)
      return APInt(numBits, (1ULL << loBitsSet) - 1);
    return (~APInt(numBits, 0)).lshr(numBits - loBitsSet);
  }

  /// The hash value is computed as the sum of the words and the bit width.
  /// @returns A hash value computed from the sum of the APInt words.
  /// @brief Get a hash value based on this APInt
  uint64_t getHashValue() const;

  /// This function returns a pointer to the internal storage of the APInt. 
  /// This is useful for writing out the APInt in binary form without any
  /// conversions.
  inline const uint64_t* getRawData() const {
    if (isSingleWord())
      return &VAL;
    return &pVal[0];
  }

  /// @}
  /// @name Unary Operators
  /// @{
  /// @returns a new APInt value representing *this incremented by one
  /// @brief Postfix increment operator.
  inline const APInt operator++(int) {
    APInt API(*this);
    ++(*this);
    return API;
  }

  /// @returns *this incremented by one
  /// @brief Prefix increment operator.
  APInt& operator++();

  /// @returns a new APInt representing *this decremented by one.
  /// @brief Postfix decrement operator. 
  inline const APInt operator--(int) {
    APInt API(*this);
    --(*this);
    return API;
  }

  /// @returns *this decremented by one.
  /// @brief Prefix decrement operator. 
  APInt& operator--();

  /// Performs a bitwise complement operation on this APInt. 
  /// @returns an APInt that is the bitwise complement of *this
  /// @brief Unary bitwise complement operator. 
  APInt operator~() const;

  /// Negates *this using two's complement logic.
  /// @returns An APInt value representing the negation of *this.
  /// @brief Unary negation operator
  inline APInt operator-() const {
    return APInt(BitWidth, 0) - (*this);
  }

  /// Performs logical negation operation on this APInt.
  /// @returns true if *this is zero, false otherwise.
  /// @brief Logical negation operator. 
  bool operator !() const;

  /// @}
  /// @name Assignment Operators
  /// @{
  /// @returns *this after assignment of RHS.
  /// @brief Copy assignment operator. 
  APInt& operator=(const APInt& RHS);

  /// The RHS value is assigned to *this. If the significant bits in RHS exceed
  /// the bit width, the excess bits are truncated. If the bit width is larger
  /// than 64, the value is zero filled in the unspecified high order bits.
  /// @returns *this after assignment of RHS value.
  /// @brief Assignment operator. 
  APInt& operator=(uint64_t RHS);

  /// Performs a bitwise AND operation on this APInt and RHS. The result is
  /// assigned to *this. 
  /// @returns *this after ANDing with RHS.
  /// @brief Bitwise AND assignment operator. 
  APInt& operator&=(const APInt& RHS);

  /// Performs a bitwise OR operation on this APInt and RHS. The result is 
  /// assigned *this;
  /// @returns *this after ORing with RHS.
  /// @brief Bitwise OR assignment operator. 
  APInt& operator|=(const APInt& RHS);

  /// Performs a bitwise XOR operation on this APInt and RHS. The result is
  /// assigned to *this.
  /// @returns *this after XORing with RHS.
  /// @brief Bitwise XOR assignment operator. 
  APInt& operator^=(const APInt& RHS);

  /// Multiplies this APInt by RHS and assigns the result to *this.
  /// @returns *this
  /// @brief Multiplication assignment operator. 
  APInt& operator*=(const APInt& RHS);

  /// Adds RHS to *this and assigns the result to *this.
  /// @returns *this
  /// @brief Addition assignment operator. 
  APInt& operator+=(const APInt& RHS);

  /// Subtracts RHS from *this and assigns the result to *this.
  /// @returns *this
  /// @brief Subtraction assignment operator. 
  APInt& operator-=(const APInt& RHS);

  /// Shifts *this left by shiftAmt and assigns the result to *this.
  /// @returns *this after shifting left by shiftAmt
  /// @brief Left-shift assignment function.
  inline APInt& operator<<=(uint32_t shiftAmt) {
    *this = shl(shiftAmt);
    return *this;
  }

  /// @}
  /// @name Binary Operators
  /// @{
  /// Performs a bitwise AND operation on *this and RHS.
  /// @returns An APInt value representing the bitwise AND of *this and RHS.
  /// @brief Bitwise AND operator. 
  APInt operator&(const APInt& RHS) const;
  APInt And(const APInt& RHS) const {
    return this->operator&(RHS);
  }

  /// Performs a bitwise OR operation on *this and RHS.
  /// @returns An APInt value representing the bitwise OR of *this and RHS.
  /// @brief Bitwise OR operator. 
  APInt operator|(const APInt& RHS) const;
  APInt Or(const APInt& RHS) const {
    return this->operator|(RHS);
  }

  /// Performs a bitwise XOR operation on *this and RHS.
  /// @returns An APInt value representing the bitwise XOR of *this and RHS.
  /// @brief Bitwise XOR operator. 
  APInt operator^(const APInt& RHS) const;
  APInt Xor(const APInt& RHS) const {
    return this->operator^(RHS);
  }

  /// Multiplies this APInt by RHS and returns the result.
  /// @brief Multiplication operator. 
  APInt operator*(const APInt& RHS) const;

  /// Adds RHS to this APInt and returns the result.
  /// @brief Addition operator. 
  APInt operator+(const APInt& RHS) const;
  APInt operator+(uint64_t RHS) const {
    return (*this) + APInt(BitWidth, RHS);
  }

  /// Subtracts RHS from this APInt and returns the result.
  /// @brief Subtraction operator. 
  APInt operator-(const APInt& RHS) const;
  APInt operator-(uint64_t RHS) const {
    return (*this) - APInt(BitWidth, RHS);
  }
  
  APInt operator<<(unsigned Bits) const {
    return shl(Bits);
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

  /// @brief Rotate left by rotateAmt.
  APInt rotl(uint32_t rotateAmt) const;

  /// @brief Rotate right by rotateAmt.
  APInt rotr(uint32_t rotateAmt) const;

  /// Perform an unsigned divide operation on this APInt by RHS. Both this and
  /// RHS are treated as unsigned quantities for purposes of this division.
  /// @returns a new APInt value containing the division result
  /// @brief Unsigned division operation.
  APInt udiv(const APInt& RHS) const;

  /// Signed divide this APInt by APInt RHS.
  /// @brief Signed division function for APInt.
  inline APInt sdiv(const APInt& RHS) const {
    if (isNegative())
      if (RHS.isNegative())
        return (-(*this)).udiv(-RHS);
      else
        return -((-(*this)).udiv(RHS));
    else if (RHS.isNegative())
      return -(this->udiv(-RHS));
    return this->udiv(RHS);
  }

  /// Perform an unsigned remainder operation on this APInt with RHS being the
  /// divisor. Both this and RHS are treated as unsigned quantities for purposes
  /// of this operation. Note that this is a true remainder operation and not
  /// a modulo operation because the sign follows the sign of the dividend
  /// which is *this.
  /// @returns a new APInt value containing the remainder result
  /// @brief Unsigned remainder operation.
  APInt urem(const APInt& RHS) const;

  /// Signed remainder operation on APInt.
  /// @brief Function for signed remainder operation.
  inline APInt srem(const APInt& RHS) const {
    if (isNegative())
      if (RHS.isNegative())
        return -((-(*this)).urem(-RHS));
      else
        return -((-(*this)).urem(RHS));
    else if (RHS.isNegative())
      return this->urem(-RHS);
    return this->urem(RHS);
  }

  /// Sometimes it is convenient to divide two APInt values and obtain both
  /// the quotient and remainder. This function does both operations in the
  /// same computation making it a little more efficient.
  /// @brief Dual division/remainder interface.
  static void udivrem(const APInt &LHS, const APInt &RHS, 
                      APInt &Quotient, APInt &Remainder);

  static void sdivrem(const APInt &LHS, const APInt &RHS,
                      APInt &Quotient, APInt &Remainder)
  {
    if (LHS.isNegative()) {
      if (RHS.isNegative())
        APInt::udivrem(-LHS, -RHS, Quotient, Remainder);
      else
        APInt::udivrem(-LHS, RHS, Quotient, Remainder);
      Quotient = -Quotient;
      Remainder = -Remainder;
    } else if (RHS.isNegative()) {
      APInt::udivrem(LHS, -RHS, Quotient, Remainder);
      Quotient = -Quotient;
    } else {
      APInt::udivrem(LHS, RHS, Quotient, Remainder);
    }
  }

  /// @returns the bit value at bitPosition
  /// @brief Array-indexing support.
  bool operator[](uint32_t bitPosition) const;

  /// @}
  /// @name Comparison Operators
  /// @{
  /// Compares this APInt with RHS for the validity of the equality
  /// relationship.
  /// @brief Equality operator. 
  bool operator==(const APInt& RHS) const;

  /// Compares this APInt with a uint64_t for the validity of the equality 
  /// relationship.
  /// @returns true if *this == Val
  /// @brief Equality operator.
  bool operator==(uint64_t Val) const;

  /// Compares this APInt with RHS for the validity of the equality
  /// relationship.
  /// @returns true if *this == Val
  /// @brief Equality comparison.
  bool eq(const APInt &RHS) const {
    return (*this) == RHS; 
  }

  /// Compares this APInt with RHS for the validity of the inequality
  /// relationship.
  /// @returns true if *this != Val
  /// @brief Inequality operator. 
  inline bool operator!=(const APInt& RHS) const {
    return !((*this) == RHS);
  }

  /// Compares this APInt with a uint64_t for the validity of the inequality 
  /// relationship.
  /// @returns true if *this != Val
  /// @brief Inequality operator. 
  inline bool operator!=(uint64_t Val) const {
    return !((*this) == Val);
  }
  
  /// Compares this APInt with RHS for the validity of the inequality
  /// relationship.
  /// @returns true if *this != Val
  /// @brief Inequality comparison
  bool ne(const APInt &RHS) const {
    return !((*this) == RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the less-than relationship.
  /// @returns true if *this < RHS when both are considered unsigned.
  /// @brief Unsigned less than comparison
  bool ult(const APInt& RHS) const;

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-than relationship.
  /// @returns true if *this < RHS when both are considered signed.
  /// @brief Signed less than comparison
  bool slt(const APInt& RHS) const;

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered unsigned.
  /// @brief Unsigned less or equal comparison
  bool ule(const APInt& RHS) const {
    return ult(RHS) || eq(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the less-or-equal relationship.
  /// @returns true if *this <= RHS when both are considered signed.
  /// @brief Signed less or equal comparison
  bool sle(const APInt& RHS) const {
    return slt(RHS) || eq(RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered unsigned.
  /// @brief Unsigned greather than comparison
  bool ugt(const APInt& RHS) const {
    return !ult(RHS) && !eq(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// the validity of the greater-than relationship.
  /// @returns true if *this > RHS when both are considered signed.
  /// @brief Signed greather than comparison
  bool sgt(const APInt& RHS) const {
    return !slt(RHS) && !eq(RHS);
  }

  /// Regards both *this and RHS as unsigned quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered unsigned.
  /// @brief Unsigned greater or equal comparison
  bool uge(const APInt& RHS) const {
    return !ult(RHS);
  }

  /// Regards both *this and RHS as signed quantities and compares them for
  /// validity of the greater-or-equal relationship.
  /// @returns true if *this >= RHS when both are considered signed.
  /// @brief Signed greather or equal comparison
  bool sge(const APInt& RHS) const {
    return !slt(RHS);
  }

  /// @}
  /// @name Resizing Operators
  /// @{
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

  /// This operation zero extends the APInt to a new width. The high order bits
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

  /// @}
  /// @name Bit Manipulation Operators
  /// @{
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

  /// @}
  /// @name Value Characterization Functions
  /// @{

  /// @returns the total number of bits.
  inline uint32_t getBitWidth() const { 
    return BitWidth; 
  }

  /// Here one word's bitwidth equals to that of uint64_t.
  /// @returns the number of words to hold the integer value of this APInt.
  /// @brief Get the number of words.
  inline uint32_t getNumWords() const {
    return (BitWidth + APINT_BITS_PER_WORD - 1) / APINT_BITS_PER_WORD;
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

  /// This method determines how many bits are required to hold the APInt
  /// equivalent of the string given by \p str of length \p slen.
  /// @brief Get bits required for string value.
  static uint32_t getBitsNeeded(const char* str, uint32_t slen, uint8_t radix);

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

  /// @}
  /// @name Conversion Functions
  /// @{

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

  /// @returns a byte-swapped representation of this APInt Value.
  APInt byteSwap() const;

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

  /// @}
  /// @name Mathematics Operations
  /// @{

  /// @returns the floor log base 2 of this APInt.
  inline uint32_t logBase2() const {
    return BitWidth - 1 - countLeadingZeros();
  }

  /// @returns the log base 2 of this APInt if its an exact power of two, -1
  /// otherwise
  inline int32_t exactLogBase2() const {
    if (!isPowerOf2())
      return -1;
    return logBase2();
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
  /// @}
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
inline bool isMask(uint32_t numBits, const APInt& APIVal) {
  return APIVal.getBoolValue() && ((APIVal + APInt(numBits,1)) & APIVal) == 0;
}

/// @returns true if the argument APInt value contains a sequence of ones
/// with the remainder zero.
inline bool isShiftedMask(uint32_t numBits, const APInt& APIVal) {
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

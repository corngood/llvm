//===-- llvm/Support/MathExtras.h - Useful math functions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful for math stuff.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MATHEXTRAS_H
#define LLVM_SUPPORT_MATHEXTRAS_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

// NOTE: The following support functions use the _32/_64 extensions instead of  
// type overloading so that signed and unsigned integers can be used without
// ambiguity.

/// Hi_32 - This function returns the high 32 bits of a 64 bit value.
inline unsigned Hi_32(uint64_t Value) {
  return static_cast<unsigned>(Value >> 32);
}

/// Lo_32 - This function returns the low 32 bits of a 64 bit value.
inline unsigned Lo_32(uint64_t Value) {
  return static_cast<unsigned>(Value);
}

/// is?Type - these functions produce optimal testing for integer data types.
inline bool isInt8  (int64_t Value) { 
  return static_cast<signed char>(Value) == Value; 
}
inline bool isUInt8 (int64_t Value) { 
  return static_cast<unsigned char>(Value) == Value; 
}
inline bool isInt16 (int64_t Value) { 
  return static_cast<signed short>(Value) == Value; 
}
inline bool isUInt16(int64_t Value) { 
  return static_cast<unsigned short>(Value) == Value; 
}
inline bool isInt32 (int64_t Value) { 
  return static_cast<signed int>(Value) == Value; 
}
inline bool isUInt32(int64_t Value) { 
  return static_cast<unsigned int>(Value) == Value; 
}

/// isMask_32 - This function returns true if the argument is a sequence of ones
/// starting at the least significant bit with the remainder zero (32 bit
/// version).   Ex. isMask_32(0x0000FFFFU) == true.
inline const bool isMask_32(unsigned Value) {
  return Value && ((Value + 1) & Value) == 0;
}

/// isMask_64 - This function returns true if the argument is a sequence of ones
/// starting at the least significant bit with the remainder zero (64 bit
/// version).
inline const bool isMask_64(uint64_t Value) {
  return Value && ((Value + 1) & Value) == 0;
}

/// isShiftedMask_32 - This function returns true if the argument contains a  
/// sequence of ones with the remainder zero (32 bit version.)
/// Ex. isShiftedMask_32(0x0000FF00U) == true.
inline const bool isShiftedMask_32(unsigned Value) {
  return isMask_32((Value - 1) | Value);
}

/// isShiftedMask_64 - This function returns true if the argument contains a  
/// sequence of ones with the remainder zero (64 bit version.)
inline const bool isShiftedMask_64(uint64_t Value) {
  return isMask_64((Value - 1) | Value);
}

/// isPowerOf2_32 - This function returns true if the argument is a power of 
/// two > 0. Ex. isPowerOf2_32(0x00100000U) == true (32 bit edition.)
inline bool isPowerOf2_32(unsigned Value) {
  return Value && !(Value & (Value - 1));
}

/// isPowerOf2_64 - This function returns true if the argument is a power of two
/// > 0 (64 bit edition.)
inline bool isPowerOf2_64(uint64_t Value) {
  return Value && !(Value & (Value - int64_t(1L)));
}

/// ByteSwap_16 - This function returns a byte-swapped representation of the
/// 16-bit argument, Value.
inline unsigned short ByteSwap_16(unsigned short Value) {
  unsigned short Hi = Value << 8;
  unsigned short Lo = Value >> 8;
  return Hi | Lo;
}

/// ByteSwap_32 - This function returns a byte-swapped representation of the
/// 32-bit argument, Value.
inline unsigned ByteSwap_32(unsigned Value) {
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)
	return __builtin_bswap32(Value);
#else
  unsigned Byte0 = Value & 0x000000FF;
  unsigned Byte1 = Value & 0x0000FF00;
  unsigned Byte2 = Value & 0x00FF0000;
  unsigned Byte3 = Value & 0xFF000000;
  return (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
#endif
}

/// ByteSwap_64 - This function returns a byte-swapped representation of the
/// 64-bit argument, Value.
inline uint64_t ByteSwap_64(uint64_t Value) {
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)
	return __builtin_bswap64(Value);
#else
  uint64_t Hi = ByteSwap_32(unsigned(Value));
  uint64_t Lo = ByteSwap_32(unsigned(Value >> 32));
  return (Hi << 32) | Lo;
#endif
}

/// CountLeadingZeros_32 - this function performs the platform optimal form of
/// counting the number of zeros from the most significant bit to the first one
/// bit.  Ex. CountLeadingZeros_32(0x00F000FF) == 8.
/// Returns 32 if the word is zero.
inline unsigned CountLeadingZeros_32(unsigned Value) {
  unsigned Count; // result
#if __GNUC__ >= 4
  // PowerPC is defined for __builtin_clz(0)
#if !defined(__ppc__) && !defined(__ppc64__)
  if (!Value) return 32;
#endif
  Count = __builtin_clz(Value);
#else
  if (!Value) return 32;
  Count = 0;
  // bisecton method for count leading zeros
  for (unsigned Shift = 32 >> 1; Shift; Shift >>= 1) {
    unsigned Tmp = Value >> Shift;
    if (Tmp) {
      Value = Tmp;
    } else {
      Count |= Shift;
    }
  }
#endif
  return Count;
}

/// CountLeadingZeros_64 - This function performs the platform optimal form
/// of counting the number of zeros from the most significant bit to the first 
/// one bit (64 bit edition.)
/// Returns 64 if the word is zero.
inline unsigned CountLeadingZeros_64(uint64_t Value) {
  unsigned Count; // result
#if __GNUC__ >= 4
  // PowerPC is defined for __builtin_clzll(0)
#if !defined(__ppc__) && !defined(__ppc64__)
  if (!Value) return 64;
#endif
  Count = __builtin_clzll(Value);
#else
  if (sizeof(long) == sizeof(int64_t)) {
    if (!Value) return 64;
    Count = 0;
    // bisecton method for count leading zeros
    for (uint64_t Shift = 64 >> 1; Shift; Shift >>= 1) {
      uint64_t Tmp = Value >> Shift;
      if (Tmp) {
        Value = Tmp;
      } else {
        Count |= Shift;
      }
    }
  } else {
    // get hi portion
    unsigned Hi = Hi_32(Value);

    // if some bits in hi portion
    if (Hi) {
        // leading zeros in hi portion plus all bits in lo portion
        Count = CountLeadingZeros_32(Hi);
    } else {
        // get lo portion
        unsigned Lo = Lo_32(Value);
        // same as 32 bit value
        Count = CountLeadingZeros_32(Lo)+32;
    }
  }
#endif
  return Count;
}

/// CountTrailingZeros_32 - this function performs the platform optimal form of
/// counting the number of zeros from the least significant bit to the first one
/// bit.  Ex. CountTrailingZeros_32(0xFF00FF00) == 8.
/// Returns 32 if the word is zero.
inline unsigned CountTrailingZeros_32(unsigned Value) {
#if __GNUC__ >= 4
  return Value ? __builtin_ctz(Value) : 32;
#else
	const unsigned Mod37BitPosition[] = {32, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13,
		                                   4, 7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9,
		                                   5, 20, 8, 19, 18 };
	return Mod37BitPosition[(-Value & Value) % 37];
#endif
}

/// CountTrailingZeros_64 - This function performs the platform optimal form
/// of counting the number of zeros from the least significant bit to the first 
/// one bit (64 bit edition.)
/// Returns 64 if the word is zero.
inline unsigned CountTrailingZeros_64(uint64_t Value) {
#if __GNUC__ >= 4
  return Value ? __builtin_ctzll(Value) : 64;
#else
  const unsigned Mod67Position[] = { 64, 0, 1, 39, 2, 15, 40, 23, 3, 12, 16, 59, 41, 19, 24, 54,
	                                   4, 64, 13, 10, 17, 62, 60, 28, 42, 30, 20, 51, 25, 44, 55,
	                                   47, 5, 32, 65, 38, 14, 22, 11, 58, 18, 53, 63, 9, 61, 27,
	                                   29, 50, 43, 46, 31, 37, 21, 57, 52, 8, 26, 49, 45, 36, 56,
	                                   7, 48, 35, 6, 34, 33, 0 };
	return Mod67Position[(-Value & Value) % 37];
#endif
}

/// CountPopulation_32 - this function counts the number of set bits in a value.
/// Ex. CountPopulation(0xF000F000) = 8
/// Returns 0 if the word is zero.
inline unsigned CountPopulation_32(uint32_t Value) {
#if __GNUC__ >= 4
  return __builtin_popcount(Value);
#else
 	uint32_t v = v - ((v >> 1) & 0x55555555);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
	return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
}

/// CountPopulation_64 - this function counts the number of set bits in a value,
/// (64 bit edition.)
inline unsigned CountPopulation_64(uint64_t Value) {
#if __GNUC__ >= 4
  return __builtin_popcountll(Value);
#else
	uint64_t v = Value - ((Value >> 1) & 0x5555555555555555ULL);
	v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
	v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
	return (uint64_t)(v * 0x0101010101010101ULL) >> 56;
#endif
}

/// Log2_32 - This function returns the floor log base 2 of the specified value, 
/// -1 if the value is zero. (32 bit edition.)
/// Ex. Log2_32(32) == 5, Log2_32(1) == 0, Log2_32(0) == -1, Log2_32(6) == 2
inline unsigned Log2_32(unsigned Value) {
    return 31 - CountLeadingZeros_32(Value);
}

/// Log2_64 - This function returns the floor log base 2 of the specified value, 
/// -1 if the value is zero. (64 bit edition.)
inline unsigned Log2_64(uint64_t Value) {
    return 63 - CountLeadingZeros_64(Value);
}

/// Log2_32_Ceil - This function returns the ceil log base 2 of the specified
/// value, 32 if the value is zero. (32 bit edition).
/// Ex. Log2_32_Ceil(32) == 5, Log2_32_Ceil(1) == 0, Log2_32_Ceil(6) == 3
inline unsigned Log2_32_Ceil(unsigned Value) {
  return 32-CountLeadingZeros_32(Value-1);
}

/// Log2_64 - This function returns the ceil log base 2 of the specified value, 
/// 64 if the value is zero. (64 bit edition.)
inline unsigned Log2_64_Ceil(uint64_t Value) {
  return 64-CountLeadingZeros_64(Value-1);
}

/// GreatestCommonDivisor64 - Return the greatest common divisor of the two
/// values using Euclid's algorithm.
inline uint64_t GreatestCommonDivisor64(uint64_t A, uint64_t B) {
  while (B) {
    uint64_t T = B;
    B = A % B;
    A = T;
  }
  return A;
}
  
/// BitsToDouble - This function takes a 64-bit integer and returns the bit
/// equivalent double.
inline double BitsToDouble(uint64_t Bits) {
  union {
    uint64_t L;
    double D;
  } T;
  T.L = Bits;
  return T.D;
}

/// BitsToFloat - This function takes a 32-bit integer and returns the bit
/// equivalent float.
inline float BitsToFloat(uint32_t Bits) {
  union {
    uint32_t I;
    float F;
  } T;
  T.I = Bits;
  return T.F;
}

/// DoubleToBits - This function takes a double and returns the bit
/// equivalent 64-bit integer.
inline uint64_t DoubleToBits(double Double) {
  union {
    uint64_t L;
    double D;
  } T;
  T.D = Double;
  return T.L;
}

/// FloatToBits - This function takes a float and returns the bit
/// equivalent 32-bit integer.
inline uint32_t FloatToBits(float Float) {
  union {
    uint32_t I;
    float F;
  } T;
  T.F = Float;
  return T.I;
}

/// Platform-independent wrappers for the C99 isnan() function.
int IsNAN(float f);
int IsNAN(double d);

/// Platform-independent wrappers for the C99 isinf() function.
int IsInf(float f);
int IsInf(double d);

} // End llvm namespace

#endif

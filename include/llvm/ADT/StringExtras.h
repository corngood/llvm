//===-- llvm/ADT/StringExtras.h - Useful string functions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful when dealing with strings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGEXTRAS_H
#define LLVM_ADT_STRINGEXTRAS_H

#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APFloat.h"
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace llvm {

static inline std::string utohexstr(uint64_t X) {
  char Buffer[40];
  char *BufPtr = Buffer+39;

  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    unsigned char Mod = static_cast<unsigned char>(X) & 15;
    if (Mod < 10)
      *--BufPtr = '0' + Mod;
    else
      *--BufPtr = 'A' + Mod-10;
    X >>= 4;
  }
  return std::string(BufPtr);
}

static inline std::string utostr_32(uint32_t X, bool isNeg = false) {
  char Buffer[20];
  char *BufPtr = Buffer+19;

  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    *--BufPtr = '0' + char(X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...

  return std::string(BufPtr);
}

static inline std::string utostr(uint64_t X, bool isNeg = false) {
  if (X == uint32_t(X))
    return utostr_32(uint32_t(X), isNeg);
  
  char Buffer[40];
  char *BufPtr = Buffer+39;
  
  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...
  
  while (X) {
    *--BufPtr = '0' + char(X % 10);
    X /= 10;
  }
  
  if (isNeg) *--BufPtr = '-';   // Add negative sign...
  return std::string(BufPtr);
}


static inline std::string itostr(int64_t X) {
  if (X < 0)
    return utostr(static_cast<uint64_t>(-X), true);
  else
    return utostr(static_cast<uint64_t>(X));
}

static inline std::string ftostr(double V) {
  char Buffer[200];
  sprintf(Buffer, "%20.6e", V);
  char *B = Buffer;
  while (*B == ' ') ++B;
  return B;
}

static inline std::string ftostr(APFloat V) {
  if (&V.getSemantics() == &APFloat::IEEEsingle)
    return ftostr(V.convertToDouble());
  else if (&V.getSemantics() == &APFloat::IEEEdouble)
    return ftostr((double)V.convertToFloat());
  return 0; // error
}

static inline std::string LowercaseString(const std::string &S) {
  std::string result(S);
  for (unsigned i = 0; i < S.length(); ++i)
    if (isupper(result[i]))
      result[i] = char(tolower(result[i]));
  return result;
}

/// StringsEqualNoCase - Return true if the two strings are equal, ignoring
/// case.
static inline bool StringsEqualNoCase(const std::string &LHS, 
                                      const std::string &RHS) {
  if (LHS.size() != RHS.size()) return false;
  for (unsigned i = 0, e = LHS.size(); i != e; ++i)
    if (tolower(LHS[i]) != tolower(RHS[i])) return false;
  return true;
}

/// StringsEqualNoCase - Return true if the two strings are equal, ignoring
/// case.
static inline bool StringsEqualNoCase(const std::string &LHS, 
                                      const char *RHS) {
  for (unsigned i = 0, e = LHS.size(); i != e; ++i) {
    if (RHS[i] == 0) return false;  // RHS too short.
    if (tolower(LHS[i]) != tolower(RHS[i])) return false;
  }
  return RHS[LHS.size()] == 0;  // Not too long?
}

/// getToken - This function extracts one token from source, ignoring any
/// leading characters that appear in the Delimiters string, and ending the
/// token at any of the characters that appear in the Delimiters string.  If
/// there are no tokens in the source string, an empty string is returned.
/// The Source source string is updated in place to remove the returned string
/// and any delimiter prefix from it.
std::string getToken(std::string &Source,
                     const char *Delimiters = " \t\n\v\f\r");

/// SplitString - Split up the specified string according to the specified
/// delimiters, appending the result fragments to the output list.
void SplitString(const std::string &Source,
                 std::vector<std::string> &OutFragments,
                 const char *Delimiters = " \t\n\v\f\r");

/// UnescapeString - Modify the argument string, turning two character sequences
/// like '\\' 'n' into '\n'.  This handles: \e \a \b \f \n \r \t \v \' \\ and
/// \num (where num is a 1-3 byte octal value).
void UnescapeString(std::string &Str);

/// EscapeString - Modify the argument string, turning '\\' and anything that
/// doesn't satisfy std::isprint into an escape sequence.
void EscapeString(std::string &Str);

} // End llvm namespace

#endif

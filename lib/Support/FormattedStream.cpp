//===-- llvm/Support/FormattedStream.cpp - Formatted streams ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of formatted_raw_ostream and
// friends.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormattedStream.h"

using namespace llvm;

/// ComputeColumn - Examine the current output and figure out which
/// column we end up in after output.
///
void formatted_raw_ostream::ComputeColumn(const char *Ptr, unsigned Size)
{
  // Keep track of the current column by scanning the string for
  // special characters

  for (const char *epos = Ptr + Size; Ptr != epos; ++Ptr) {
    ++Column;
    if (*Ptr == '\n' || *Ptr == '\r')
      Column = 0;
    else if (*Ptr == '\t')
      Column += (8 - (Column & 0x7)) & 0x7;
  }
}

/// PadToColumn - Align the output to some column number.
///
/// \param NewCol - The column to move to.
/// \param MinPad - The minimum space to give after the most recent
/// I/O, even if the current column + minpad > newcol.
///
void formatted_raw_ostream::PadToColumn(unsigned NewCol, unsigned MinPad) 
{
  flush();

  // Output spaces until we reach the desired column.
  unsigned num = NewCol - Column;
  if (NewCol < Column || num < MinPad) {
    num = MinPad;
  }

  // TODO: Write a whole string at a time.
  while (num-- > 0) {
    write(' ');
  }
}


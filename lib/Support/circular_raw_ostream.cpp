//===- circulat_raw_ostream.cpp - Implement the circular_raw_ostream class -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements support for circular buffered streams.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/circular_raw_ostream.h"

#include <algorithm>

using namespace llvm;

void circular_raw_ostream::write_impl(const char *Ptr, size_t Size) {
  if (BufferSize == 0) {
    TheStream->write(Ptr, Size);
    return;
  }

  // Write into the buffer, wrapping if necessary.
  while (Size != 0) {
    unsigned Bytes = std::min(Size, BufferSize - (Cur - BufferArray));
    memcpy(Cur, Ptr, Bytes);
    Size -= Bytes;
    Cur += Bytes;
    if (Cur == BufferArray + BufferSize) {
      // Reset the output pointer to the start of the buffer.
      Cur = BufferArray;
      Filled = true;
    }
  }    
}

void circular_raw_ostream::flushBufferWithBanner(void) {
  if (BufferSize != 0) {
    // Write out the buffer
    int num = std::strlen(Banner); 
    TheStream->write(Banner, num);
    flushBuffer();
  }
}

//===-- DWARFDebugRangesList.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugRangeList.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void DWARFDebugRangeList::clear() {
  Offset = -1U;
  AddressSize = 0;
  Entries.clear();
}

bool DWARFDebugRangeList::extract(DataExtractor data, uint32_t *offset_ptr) {
  clear();
  if (!data.isValidOffset(*offset_ptr))
    return false;
  AddressSize = data.getAddressSize();
  if (AddressSize != 4 && AddressSize != 8)
    return false;
  Offset = *offset_ptr;
  while (true) {
    RangeListEntry entry;
    uint32_t prev_offset = *offset_ptr;
    entry.StartAddress = data.getAddress(offset_ptr);
    entry.EndAddress = data.getAddress(offset_ptr);
    // Check that both values were extracted correctly.
    if (*offset_ptr != prev_offset + 2 * AddressSize) {
      clear();
      return false;
    }
    // The end of any given range list is marked by an end of list entry,
    // which consists of a 0 for the beginning address offset
    // and a 0 for the ending address offset.
    if (entry.StartAddress == 0 && entry.EndAddress == 0)
      break;
    Entries.push_back(entry);
  }
  return true;
}

void DWARFDebugRangeList::dump(raw_ostream &OS) const {
  for (int i = 0, n = Entries.size(); i != n; ++i) {
    const char *format_str = (AddressSize == 4) ? "%08x %08x %08x\n"
                                                : "%08x %016x %016x\n";
    OS << format(format_str, Offset, Entries[i].StartAddress,
                                     Entries[i].EndAddress);
  }
  OS << format("%08x <End of list>\n", Offset);
}

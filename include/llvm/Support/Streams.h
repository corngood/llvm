//===- llvm/Support/Streams.h - Wrappers for iostreams ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper for the STL I/O streams.  It prevents the need
// to include <iostream> in a file just to get I/O.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STREAMS_H
#define LLVM_SUPPORT_STREAMS_H

#include <iosfwd>

namespace llvm {

  /// BaseStream - Acts like the STL streams. It's a wrapper for the std::cerr,
  /// std::cout, std::cin, etc. streams. However, it doesn't require #including
  /// @verbatim <iostream> @endverbatm in every file (doing so increases static 
  /// c'tors & d'tors in the object code).
  /// 
  template <typename StreamTy>
  class BaseStream {
    StreamTy *Stream;
  public:
    BaseStream() : Stream(0) {}
    BaseStream(StreamTy &S) : Stream(&S) {}
    BaseStream(StreamTy *S) : Stream(S) {}

    StreamTy *stream() const { return Stream; }

    inline BaseStream &operator << (StreamTy &(*Func)(StreamTy&)) {
      if (Stream) *Stream << Func;
      return *this;
    }

    template <typename Ty>
    BaseStream &operator << (const Ty &Thing) {
      if (Stream) *Stream << Thing;
      return *this;
    }

    template <typename Ty>
    BaseStream &operator >> (const Ty &Thing) {
      if (Stream) *Stream >> Thing;
      return *this;
    }

    operator StreamTy* () { return Stream; }

    bool operator == (const StreamTy &S) { return &S == Stream; }
    bool operator != (const StreamTy &S) { return !(*this == S); }
    bool operator == (const BaseStream &S) { return S.Stream == Stream; }
    bool operator != (const BaseStream &S) { return !(*this == S); }
  };

  typedef BaseStream<std::ostream> OStream;
  typedef BaseStream<std::istream> IStream;
  typedef BaseStream<std::stringstream> StringStream;

  extern OStream cout;
  extern OStream cerr;
  extern IStream cin;

} // End llvm namespace

#endif

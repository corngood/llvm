//===-- llvm/CodeGen/FormattedStream.h - Formatted streams ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains raw_ostream implementations for streams to do
// things like pretty-print comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FORMATTEDSTREAM_H
#define LLVM_SUPPORT_FORMATTEDSTREAM_H

#include "llvm/Support/raw_ostream.h"

namespace llvm 
{
  /// formatted_raw_ostream - Formatted raw_fd_ostream to handle
  /// asm-specific constructs.
  ///
  class formatted_raw_ostream : public raw_ostream {
  public:
    /// DELETE_STREAM - Tell the destructor to delete the held stream.
    ///
    const static bool DELETE_STREAM = true;
    /// PRESERVE_STREAM - Tell the destructor to not delete the held
    /// stream.
    ///
    const static bool PRESERVE_STREAM = false;
    
  private:
    /// TheStream - The real stream we output to. We set it to be
    /// unbuffered, since we're already doing our own buffering.
    ///
    raw_ostream *TheStream;
    /// DeleteStream - Do we need to delete TheStream in the
    /// destructor?
    ///
    bool DeleteStream;

    /// Column - The current output column of the stream.  The column
    /// scheme is zero-based.
    ///
    unsigned Column;

    virtual void write_impl(const char *Ptr, size_t Size) {
      ComputeColumn(Ptr, Size);
      TheStream->write(Ptr, Size);
    }

    /// current_pos - Return the current position within the stream,
    /// not counting the bytes currently in the buffer.
    virtual uint64_t current_pos() { 
      // This has the same effect as calling TheStream.current_pos(),
      // but that interface is private.
      return TheStream->tell() - TheStream->GetNumBytesInBuffer();
    }

    /// ComputeColumn - Examine the current output and figure out
    /// which column we end up in after output.
    ///
    void ComputeColumn(const char *Ptr, size_t Size);

  public:
    /// formatted_raw_ostream - Open the specified file for
    /// writing. If an error occurs, information about the error is
    /// put into ErrorInfo, and the stream should be immediately
    /// destroyed; the string will be empty if no error occurred.
    ///
    /// As a side effect, the given Stream is set to be Unbuffered.
    /// This is because formatted_raw_ostream does its own buffering,
    /// so it doesn't want another layer of buffering to be happening
    /// underneath it.
    ///
    /// \param Filename - The file to open. If this is "-" then the
    /// stream will use stdout instead.
    /// \param Binary - The file should be opened in binary mode on
    /// platforms that support this distinction.
    formatted_raw_ostream(raw_ostream &Stream, bool Delete = false) 
      : raw_ostream(), TheStream(&Stream), DeleteStream(Delete), Column(0) {
      // This formatted_raw_ostream inherits from raw_ostream, so it'll do its
      // own buffering, and it doesn't need or want TheStream to do another
      // layer of buffering underneath. Resize the buffer to what TheStream
      // had been using, and tell TheStream not to do its own buffering.
      TheStream->flush();
      if (size_t BufferSize = TheStream->GetNumBytesInBuffer())
        SetBufferSize(BufferSize);
      TheStream->SetUnbuffered();
    }
    explicit formatted_raw_ostream()
      : raw_ostream(), TheStream(0), DeleteStream(false), Column(0) {}

    ~formatted_raw_ostream() {
      if (DeleteStream)
        delete TheStream;
    }
    
    void setStream(raw_ostream &Stream, bool Delete = false) {
      TheStream = &Stream;
      DeleteStream = Delete;

      // Avoid double-buffering, as above.
      TheStream->flush();
      if (size_t BufferSize = TheStream->GetNumBytesInBuffer())
        SetBufferSize(BufferSize);
      TheStream->SetUnbuffered();
    }

    /// PadToColumn - Align the output to some column number.
    ///
    /// \param NewCol - The column to move to.
    /// \param MinPad - The minimum space to give after the most
    /// recent I/O, even if the current column + minpad > newcol.
    ///
    void PadToColumn(unsigned NewCol, unsigned MinPad = 0);
  };

/// fouts() - This returns a reference to a formatted_raw_ostream for
/// standard output.  Use it like: fouts() << "foo" << "bar";
formatted_raw_ostream &fouts();

/// ferrs() - This returns a reference to a formatted_raw_ostream for
/// standard error.  Use it like: ferrs() << "foo" << "bar";
formatted_raw_ostream &ferrs();

} // end llvm namespace


#endif

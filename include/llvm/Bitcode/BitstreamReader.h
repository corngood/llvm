//===- BitstreamReader.h - Low-level bitstream reader interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License.  See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamReader class.  This class can be used to
// read an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef BITSTREAM_READER_H
#define BITSTREAM_READER_H

#include "llvm/Bitcode/BitCodes.h"
#include <vector>

namespace llvm {
  
class BitstreamReader {
  const unsigned char *NextChar;
  const unsigned char *LastChar;
  
  /// CurWord - This is the current data we have pulled from the stream but have
  /// not returned to the client.
  uint32_t CurWord;
  
  /// BitsInCurWord - This is the number of bits in CurWord that are valid. This
  /// is always from [0...31] inclusive.
  unsigned BitsInCurWord;
  
  // CurCodeSize - This is the declared size of code values used for the current
  // block, in bits.
  unsigned CurCodeSize;

  /// CurAbbrevs - Abbrevs installed at in this block.
  std::vector<BitCodeAbbrev*> CurAbbrevs;
  
  struct Block {
    unsigned PrevCodeSize;
    std::vector<BitCodeAbbrev*> PrevAbbrevs;
    explicit Block(unsigned PCS) : PrevCodeSize(PCS) {}
  };
  
  /// BlockScope - This tracks the codesize of parent blocks.
  SmallVector<Block, 8> BlockScope;

  /// FirstChar - This remembers the first byte of the stream.
  const unsigned char *FirstChar;
public:
  BitstreamReader() {
    NextChar = FirstChar = LastChar = 0;
    CurWord = 0;
    BitsInCurWord = 0;
    CurCodeSize = 0;
  }

  BitstreamReader(const unsigned char *Start, const unsigned char *End) {
    init(Start, End);
  }
  
  void init(const unsigned char *Start, const unsigned char *End) {
    NextChar = FirstChar = Start;
    LastChar = End;
    assert(((End-Start) & 3) == 0 &&"Bitcode stream not a multiple of 4 bytes");
    CurWord = 0;
    BitsInCurWord = 0;
    CurCodeSize = 2;
  }
  
  ~BitstreamReader() {
    // Abbrevs could still exist if the stream was broken.  If so, don't leak
    // them.
    for (unsigned i = 0, e = CurAbbrevs.size(); i != e; ++i)
      CurAbbrevs[i]->dropRef();

    for (unsigned S = 0, e = BlockScope.size(); S != e; ++S) {
      std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
      for (unsigned i = 0, e = Abbrevs.size(); i != e; ++i)
        Abbrevs[i]->dropRef();
    }
  }
  
  bool AtEndOfStream() const { return NextChar == LastChar; }
  
  /// GetCurrentBitNo - Return the bit # of the bit we are reading.
  uint64_t GetCurrentBitNo() const {
    return (NextChar-FirstChar)*8 + (32-BitsInCurWord);
  }
  
  /// JumpToBit - Reset the stream to the specified bit number.
  void JumpToBit(uint64_t BitNo) {
    unsigned ByteNo = (BitNo/8) & ~3;
    unsigned WordBitNo = BitNo & 31;
    assert(ByteNo < (unsigned)(LastChar-FirstChar) && "Invalid location");
    
    // Move the cursor to the right word.
    NextChar = FirstChar+ByteNo;
    BitsInCurWord = 0;
    
    // Skip over any bits that are already consumed.
    if (WordBitNo) {
      NextChar -= 4;
      Read(WordBitNo);
    }
  }
  
  /// GetAbbrevIDWidth - Return the number of bits used to encode an abbrev #.
  unsigned GetAbbrevIDWidth() const { return CurCodeSize; }
  
  uint32_t Read(unsigned NumBits) {
    // If the field is fully contained by CurWord, return it quickly.
    if (BitsInCurWord >= NumBits) {
      uint32_t R = CurWord & ((1U << NumBits)-1);
      CurWord >>= NumBits;
      BitsInCurWord -= NumBits;
      return R;
    }

    // If we run out of data, stop at the end of the stream.
    if (LastChar == NextChar) {
      CurWord = 0;
      BitsInCurWord = 0;
      return 0;
    }
    
    unsigned R = CurWord;

    // Read the next word from the stream.
    CurWord = (NextChar[0] <<  0) | (NextChar[1] << 8) |
              (NextChar[2] << 16) | (NextChar[3] << 24);
    NextChar += 4;
    
    // Extract NumBits-BitsInCurWord from what we just read.
    unsigned BitsLeft = NumBits-BitsInCurWord;
    
    // Be careful here, BitsLeft is in the range [1..32] inclusive.
    R |= (CurWord & (~0U >> (32-BitsLeft))) << BitsInCurWord;
    
    // BitsLeft bits have just been used up from CurWord.
    if (BitsLeft != 32)
      CurWord >>= BitsLeft;
    else
      CurWord = 0;
    BitsInCurWord = 32-BitsLeft;
    return R;
  }
  
  uint64_t Read64(unsigned NumBits) {
    if (NumBits <= 32) return Read(NumBits);
    
    uint64_t V = Read(32);
    return V | (uint64_t)Read(NumBits-32) << 32;
  }
  
  uint32_t ReadVBR(unsigned NumBits) {
    uint32_t Piece = Read(NumBits);
    if ((Piece & (1U << (NumBits-1))) == 0)
      return Piece;

    uint32_t Result = 0;
    unsigned NextBit = 0;
    while (1) {
      Result |= (Piece & ((1U << (NumBits-1))-1)) << NextBit;

      if ((Piece & (1U << (NumBits-1))) == 0)
        return Result;
      
      NextBit += NumBits-1;
      Piece = Read(NumBits);
    }
  }
  
  uint64_t ReadVBR64(unsigned NumBits) {
    uint64_t Piece = Read(NumBits);
    if ((Piece & (1U << (NumBits-1))) == 0)
      return Piece;
    
    uint64_t Result = 0;
    unsigned NextBit = 0;
    while (1) {
      Result |= (Piece & ((1U << (NumBits-1))-1)) << NextBit;
      
      if ((Piece & (1U << (NumBits-1))) == 0)
        return Result;
      
      NextBit += NumBits-1;
      Piece = Read(NumBits);
    }
  }

  void SkipToWord() {
    BitsInCurWord = 0;
    CurWord = 0;
  }

  
  unsigned ReadCode() {
    return Read(CurCodeSize);
  }

  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//
  
  // Block header:
  //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]

  /// ReadSubBlockID - Having read the ENTER_SUBBLOCK code, read the BlockID for
  /// the block.
  unsigned ReadSubBlockID() {
    return ReadVBR(bitc::BlockIDWidth);
  }
  
  /// SkipBlock - Having read the ENTER_SUBBLOCK abbrevid and a BlockID, skip
  /// over the body of this block.  If the block record is malformed, return
  /// true.
  bool SkipBlock() {
    // Read and ignore the codelen value.  Since we are skipping this block, we
    // don't care what code widths are used inside of it.
    ReadVBR(bitc::CodeLenWidth);
    SkipToWord();
    unsigned NumWords = Read(bitc::BlockSizeWidth);
    
    // Check that the block wasn't partially defined, and that the offset isn't
    // bogus.
    if (AtEndOfStream() || NextChar+NumWords*4 > LastChar)
      return true;
    
    NextChar += NumWords*4;
    return false;
  }
  
  /// EnterSubBlock - Having read the ENTER_SUBBLOCK abbrevid, read and enter
  /// the block, returning the BlockID of the block we just entered.
  bool EnterSubBlock(unsigned *NumWordsP = 0) {
    BlockScope.push_back(Block(CurCodeSize));
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
    
    // Get the codesize of this block.
    CurCodeSize = ReadVBR(bitc::CodeLenWidth);
    SkipToWord();
    unsigned NumWords = Read(bitc::BlockSizeWidth);
    if (NumWordsP) *NumWordsP = NumWords;
    
    // Validate that this block is sane.
    if (CurCodeSize == 0 || AtEndOfStream() || NextChar+NumWords*4 > LastChar)
      return true;
    
    return false;
  }
  
  bool ReadBlockEnd() {
    if (BlockScope.empty()) return true;
    
    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    SkipToWord();
    CurCodeSize = BlockScope.back().PrevCodeSize;
    
    // Delete abbrevs from popped scope.
    for (unsigned i = 0, e = CurAbbrevs.size(); i != e; ++i)
      CurAbbrevs[i]->dropRef();
    
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
    BlockScope.pop_back();
    return false;
  }
  
  //===--------------------------------------------------------------------===//
  // Record Processing
  //===--------------------------------------------------------------------===//
  
  unsigned ReadRecord(unsigned AbbrevID, SmallVectorImpl<uint64_t> &Vals) {
    if (AbbrevID == bitc::UNABBREV_RECORD) {
      unsigned Code = ReadVBR(6);
      unsigned NumElts = ReadVBR(6);
      for (unsigned i = 0; i != NumElts; ++i)
        Vals.push_back(ReadVBR64(6));
      return Code;
    }
    
    unsigned AbbrevNo = AbbrevID-bitc::FIRST_ABBREV;
    assert(AbbrevNo < CurAbbrevs.size() && "Invalid abbrev #!");
    BitCodeAbbrev *Abbv = CurAbbrevs[AbbrevNo];

    for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
      const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
      if (Op.isLiteral()) {
        // If the abbrev specifies the literal value to use, use it.
        Vals.push_back(Op.getLiteralValue());
      } else {
        // Decode the value as we are commanded.
        switch (Op.getEncoding()) {
        default: assert(0 && "Unknown encoding!");
        case BitCodeAbbrevOp::FixedWidth:
          Vals.push_back(Read(Op.getEncodingData()));
          break;
        case BitCodeAbbrevOp::VBR:
          Vals.push_back(ReadVBR64(Op.getEncodingData()));
          break;
        }
      }
    }
    
    unsigned Code = Vals[0];
    Vals.erase(Vals.begin());
    return Code;
  }
  
  //===--------------------------------------------------------------------===//
  // Abbrev Processing
  //===--------------------------------------------------------------------===//
  
  void ReadAbbrevRecord() {
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    unsigned NumOpInfo = ReadVBR(5);
    for (unsigned i = 0; i != NumOpInfo; ++i) {
      bool IsLiteral = Read(1);
      if (IsLiteral) {
        Abbv->Add(BitCodeAbbrevOp(ReadVBR64(8)));
        continue;
      }

      BitCodeAbbrevOp::Encoding E = (BitCodeAbbrevOp::Encoding)Read(3);
      if (BitCodeAbbrevOp::hasEncodingData(E)) {
        Abbv->Add(BitCodeAbbrevOp(E, ReadVBR64(5)));
      } else {
        assert(0 && "unimp");
      }
    }
    CurAbbrevs.push_back(Abbv);
  }
};

} // End llvm namespace

#endif

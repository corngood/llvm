//===- BitstreamWriter.h - Low-level bitstream writer interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License.  See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamWriter class.  This class can be used to
// write an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef BITSTREAM_WRITER_H
#define BITSTREAM_WRITER_H

#include "llvm/Bitcode/BitCodes.h"
#include <vector>

namespace llvm {

class BitstreamWriter {
  std::vector<unsigned char> &Out;

  /// CurBit - Always between 0 and 31 inclusive, specifies the next bit to use.
  unsigned CurBit;
  
  /// CurValue - The current value.  Only bits < CurBit are valid.
  uint32_t CurValue;
  
  // CurCodeSize - This is the declared size of code values used for the current
  // block, in bits.
  unsigned CurCodeSize;
  
  struct Block {
    unsigned PrevCodeSize;
    unsigned StartSizeWord;
    std::vector<BitCodeAbbrev*> PrevAbbrevs;
    Block(unsigned PCS, unsigned SSW) : PrevCodeSize(PCS), StartSizeWord(SSW) {}
  };
  
  /// BlockScope - This tracks the current blocks that we have entered.
  std::vector<Block> BlockScope;
  
  std::vector<BitCodeAbbrev*> CurAbbrevs;
public:
  BitstreamWriter(std::vector<unsigned char> &O) 
    : Out(O), CurBit(0), CurValue(0), CurCodeSize(2) {}

  ~BitstreamWriter() {
    assert(CurBit == 0 && "Unflused data remaining");
    assert(BlockScope.empty() && "Block imbalance");
  }
  //===--------------------------------------------------------------------===//
  // Basic Primitives for emitting bits to the stream.
  //===--------------------------------------------------------------------===//
  
  void Emit(uint32_t Val, unsigned NumBits) {
    assert(NumBits <= 32 && "Invalid value size!");
    assert((Val & ~(~0U >> (32-NumBits))) == 0 && "High bits set!");
    CurValue |= Val << CurBit;
    if (CurBit + NumBits < 32) {
      CurBit += NumBits;
      return;
    }
    
    // Add the current word.
    unsigned V = CurValue;
    Out.push_back((unsigned char)(V >>  0));
    Out.push_back((unsigned char)(V >>  8));
    Out.push_back((unsigned char)(V >> 16));
    Out.push_back((unsigned char)(V >> 24));
    
    if (CurBit)
      CurValue = Val >> (32-CurBit);
    else
      CurValue = 0;
    CurBit = (CurBit+NumBits) & 31;
  }
  
  void Emit64(uint64_t Val, unsigned NumBits) {
    if (NumBits <= 32)
      Emit((uint32_t)Val, NumBits);
    else {
      Emit((uint32_t)Val, 32);
      Emit((uint32_t)(Val >> 32), NumBits-32);
    }
  }
  
  void FlushToWord() {
    if (CurBit) {
      unsigned V = CurValue;
      Out.push_back((unsigned char)(V >>  0));
      Out.push_back((unsigned char)(V >>  8));
      Out.push_back((unsigned char)(V >> 16));
      Out.push_back((unsigned char)(V >> 24));
      CurBit = 0;
      CurValue = 0;
    }
  }
  
  void EmitVBR(uint32_t Val, unsigned NumBits) {
    uint32_t Threshold = 1U << (NumBits-1);
    
    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit((Val & ((1 << (NumBits-1))-1)) | (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }
    
    Emit(Val, NumBits);
  }
  
  void EmitVBR64(uint64_t Val, unsigned NumBits) {
    if ((uint32_t)Val == Val)
      return EmitVBR((uint32_t)Val, NumBits);
    
    uint64_t Threshold = 1U << (NumBits-1);
    
    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit(((uint32_t)Val & ((1 << (NumBits-1))-1)) |
           (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }
    
    Emit((uint32_t)Val, NumBits);
  }
  
  /// EmitCode - Emit the specified code.
  void EmitCode(unsigned Val) {
    Emit(Val, CurCodeSize);
  }
  
  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//
  
  void EnterSubblock(unsigned BlockID, unsigned CodeLen) {
    // Block header:
    //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]
    EmitCode(bitc::ENTER_SUBBLOCK);
    EmitVBR(BlockID, bitc::BlockIDWidth);
    EmitVBR(CodeLen, bitc::CodeLenWidth);
    FlushToWord();
    BlockScope.push_back(Block(CurCodeSize, Out.size()/4));
    
    // Delete all abbrevs.
    for (unsigned i = 0, e = CurAbbrevs.size(); i != e; ++i)
      delete CurAbbrevs[i];
    
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
    // Emit a placeholder, which will be replaced when the block is popped.
    Emit(0, bitc::BlockSizeWidth);
    
    CurCodeSize = CodeLen;
  }
  
  void ExitBlock() {
    assert(!BlockScope.empty() && "Block scope imbalance!");
    const Block &B = BlockScope.back();
    
    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    EmitCode(bitc::END_BLOCK);
    FlushToWord();

    // Compute the size of the block, in words, not counting the size field.
    unsigned SizeInWords = Out.size()/4-B.StartSizeWord - 1;
    unsigned ByteNo = B.StartSizeWord*4;
    
    // Update the block size field in the header of this sub-block.
    Out[ByteNo++] = (unsigned char)(SizeInWords >>  0);
    Out[ByteNo++] = (unsigned char)(SizeInWords >>  8);
    Out[ByteNo++] = (unsigned char)(SizeInWords >> 16);
    Out[ByteNo++] = (unsigned char)(SizeInWords >> 24);
    
    // Restore the inner block's code size and abbrev table.
    CurCodeSize = B.PrevCodeSize;
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
    BlockScope.pop_back();
  }
  
  //===--------------------------------------------------------------------===//
  // Record Emission
  //===--------------------------------------------------------------------===//
  
  /// EmitRecord - Emit the specified record to the stream, using an abbrev if
  /// we have one to compress the output.
  void EmitRecord(unsigned Code, SmallVectorImpl<uint64_t> &Vals,
                  unsigned Abbrev = 0) {
    if (Abbrev) {
      unsigned AbbrevNo = Abbrev-bitc::FIRST_ABBREV;
      assert(AbbrevNo < CurAbbrevs.size() && "Invalid abbrev #!");
      BitCodeAbbrev *Abbv = CurAbbrevs[AbbrevNo];
      assert(0 && "TODO");
      for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
      }
      
      
    } else {
      // If we don't have an abbrev to use, emit this in its fully unabbreviated
      // form.
      EmitCode(bitc::UNABBREV_RECORD);
      EmitVBR(Code, 6);
      EmitVBR(Vals.size(), 6);
      for (unsigned i = 0, e = Vals.size(); i != e; ++i)
        EmitVBR64(Vals[i], 6);
    }
  }
  
  /// EmitRecord - Emit the specified record to the stream, using an abbrev if
  /// we have one to compress the output.
  void EmitRecord(unsigned Code, SmallVectorImpl<unsigned> &Vals,
                  unsigned Abbrev = 0) {
    if (Abbrev) {
      assert(0 && "abbrevs not implemented yet!");
    } else {
      // If we don't have an abbrev to use, emit this in its fully unabbreviated
      // form.
      EmitCode(bitc::UNABBREV_RECORD);
      EmitVBR(Code, 6);
      EmitVBR(Vals.size(), 6);
      for (unsigned i = 0, e = Vals.size(); i != e; ++i)
        EmitVBR(Vals[i], 6);
    }
  }
  
  //===--------------------------------------------------------------------===//
  // Abbrev Emission
  //===--------------------------------------------------------------------===//
  
  /// EmitAbbrev - This emits an abbreviation to the stream.  Note that this
  /// method takes ownership of the specified abbrev.
  unsigned EmitAbbrev(BitCodeAbbrev *Abbv) {
    // Emit the abbreviation as a record.
    EmitCode(bitc::DEFINE_ABBREV);
    EmitVBR(Abbv->getNumOperandInfos(), 5);
    for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
      const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
      Emit(Op.isLiteral(), 1);
      if (Op.isLiteral()) {
        EmitVBR64(Op.getLiteralValue(), 8);
      } else {
        Emit(Op.getEncoding(), 3);
        if (Op.hasEncodingData())
          EmitVBR64(Op.getEncodingData(), 5);
      }
    }
    
    CurAbbrevs.push_back(Abbv);
    return CurAbbrevs.size()-1+bitc::FIRST_ABBREV;
  }
};


} // End llvm namespace

#endif

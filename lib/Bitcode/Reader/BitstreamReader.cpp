//===- BitstreamReader.cpp - BitstreamReader implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitstreamReader.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  BitstreamCursor implementation
//===----------------------------------------------------------------------===//

void BitstreamCursor::operator=(const BitstreamCursor &RHS) {
  freeState();
  
  BitStream = RHS.BitStream;
  NextChar = RHS.NextChar;
  CurWord = RHS.CurWord;
  BitsInCurWord = RHS.BitsInCurWord;
  CurCodeSize = RHS.CurCodeSize;
  
  // Copy abbreviations, and bump ref counts.
  CurAbbrevs = RHS.CurAbbrevs;
  for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
       i != e; ++i)
    CurAbbrevs[i]->addRef();
  
  // Copy block scope and bump ref counts.
  BlockScope = RHS.BlockScope;
  for (unsigned S = 0, e = static_cast<unsigned>(BlockScope.size());
       S != e; ++S) {
    std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
    for (unsigned i = 0, e = static_cast<unsigned>(Abbrevs.size());
         i != e; ++i)
      Abbrevs[i]->addRef();
  }
}

void BitstreamCursor::freeState() {
  // Free all the Abbrevs.
  for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
       i != e; ++i)
    CurAbbrevs[i]->dropRef();
  CurAbbrevs.clear();
  
  // Free all the Abbrevs in the block scope.
  for (unsigned S = 0, e = static_cast<unsigned>(BlockScope.size());
       S != e; ++S) {
    std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
    for (unsigned i = 0, e = static_cast<unsigned>(Abbrevs.size());
         i != e; ++i)
      Abbrevs[i]->dropRef();
  }
  BlockScope.clear();
}

/// EnterSubBlock - Having read the ENTER_SUBBLOCK abbrevid, enter
/// the block, and return true if the block has an error.
bool BitstreamCursor::EnterSubBlock(unsigned BlockID, unsigned *NumWordsP) {
  // Save the current block's state on BlockScope.
  BlockScope.push_back(Block(CurCodeSize));
  BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
  
  // Add the abbrevs specific to this block to the CurAbbrevs list.
  if (const BitstreamReader::BlockInfo *Info =
      BitStream->getBlockInfo(BlockID)) {
    for (unsigned i = 0, e = static_cast<unsigned>(Info->Abbrevs.size());
         i != e; ++i) {
      CurAbbrevs.push_back(Info->Abbrevs[i]);
      CurAbbrevs.back()->addRef();
    }
  }
  
  // Get the codesize of this block.
  CurCodeSize = ReadVBR(bitc::CodeLenWidth);
  SkipToFourByteBoundary();
  unsigned NumWords = Read(bitc::BlockSizeWidth);
  if (NumWordsP) *NumWordsP = NumWords;
  
  // Validate that this block is sane.
  if (CurCodeSize == 0 || AtEndOfStream())
    return true;
  
  return false;
}

void BitstreamCursor::readAbbreviatedLiteral(const BitCodeAbbrevOp &Op,
                                             SmallVectorImpl<uint64_t> &Vals) {
  assert(Op.isLiteral() && "Not a literal");
  // If the abbrev specifies the literal value to use, use it.
  Vals.push_back(Op.getLiteralValue());
}

void BitstreamCursor::readAbbreviatedField(const BitCodeAbbrevOp &Op,
                                           SmallVectorImpl<uint64_t> &Vals) {
  assert(!Op.isLiteral() && "Use ReadAbbreviatedLiteral for literals!");
  
  // Decode the value as we are commanded.
  switch (Op.getEncoding()) {
  case BitCodeAbbrevOp::Array:
  case BitCodeAbbrevOp::Blob:
    assert(0 && "Should not reach here");
  case BitCodeAbbrevOp::Fixed:
    Vals.push_back(Read((unsigned)Op.getEncodingData()));
    break;
  case BitCodeAbbrevOp::VBR:
    Vals.push_back(ReadVBR64((unsigned)Op.getEncodingData()));
    break;
  case BitCodeAbbrevOp::Char6:
    Vals.push_back(BitCodeAbbrevOp::DecodeChar6(Read(6)));
    break;
  }
}

void BitstreamCursor::skipAbbreviatedField(const BitCodeAbbrevOp &Op) {
  assert(!Op.isLiteral() && "Use ReadAbbreviatedLiteral for literals!");
  
  // Decode the value as we are commanded.
  switch (Op.getEncoding()) {
  case BitCodeAbbrevOp::Array:
  case BitCodeAbbrevOp::Blob:
    assert(0 && "Should not reach here");
  case BitCodeAbbrevOp::Fixed:
    (void)Read((unsigned)Op.getEncodingData());
    break;
  case BitCodeAbbrevOp::VBR:
    (void)ReadVBR64((unsigned)Op.getEncodingData());
    break;
  case BitCodeAbbrevOp::Char6:
    (void)Read(6);
    break;
  }
}



/// skipRecord - Read the current record and discard it.
void BitstreamCursor::skipRecord(unsigned AbbrevID) {
  // Skip unabbreviated records by reading past their entries.
  if (AbbrevID == bitc::UNABBREV_RECORD) {
    unsigned Code = ReadVBR(6);
    (void)Code;
    unsigned NumElts = ReadVBR(6);
    for (unsigned i = 0; i != NumElts; ++i)
      (void)ReadVBR64(6);
    return;
  }

  const BitCodeAbbrev *Abbv = getAbbrev(AbbrevID);
  
  for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
    const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
    if (Op.isLiteral())
      continue;
    
    if (Op.getEncoding() != BitCodeAbbrevOp::Array &&
        Op.getEncoding() != BitCodeAbbrevOp::Blob) {
      skipAbbreviatedField(Op);
      continue;
    }
    
    if (Op.getEncoding() == BitCodeAbbrevOp::Array) {
      // Array case.  Read the number of elements as a vbr6.
      unsigned NumElts = ReadVBR(6);
      
      // Get the element encoding.
      assert(i+2 == e && "array op not second to last?");
      const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);
      
      // Read all the elements.
      for (; NumElts; --NumElts)
        skipAbbreviatedField(EltEnc);
      continue;
    }
    
    assert(Op.getEncoding() == BitCodeAbbrevOp::Blob);
    // Blob case.  Read the number of bytes as a vbr6.
    unsigned NumElts = ReadVBR(6);
    SkipToFourByteBoundary();  // 32-bit alignment
    
    // Figure out where the end of this blob will be including tail padding.
    size_t NewEnd = GetCurrentBitNo()+((NumElts+3)&~3)*8;
    
    // If this would read off the end of the bitcode file, just set the
    // record to empty and return.
    if (!canSkipToPos(NewEnd/8)) {
      NextChar = BitStream->getBitcodeBytes().getExtent();
      break;
    }
    
    // Skip over the blob.
    JumpToBit(NewEnd);
  }
}

unsigned BitstreamCursor::readRecord(unsigned AbbrevID,
                                     SmallVectorImpl<uint64_t> &Vals,
                                     StringRef *Blob) {
  if (AbbrevID == bitc::UNABBREV_RECORD) {
    unsigned Code = ReadVBR(6);
    unsigned NumElts = ReadVBR(6);
    for (unsigned i = 0; i != NumElts; ++i)
      Vals.push_back(ReadVBR64(6));
    return Code;
  }
  
  const BitCodeAbbrev *Abbv = getAbbrev(AbbrevID);
  
  for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
    const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
    if (Op.isLiteral()) {
      readAbbreviatedLiteral(Op, Vals);
      continue;
    }
    
    if (Op.getEncoding() != BitCodeAbbrevOp::Array &&
        Op.getEncoding() != BitCodeAbbrevOp::Blob) {
      readAbbreviatedField(Op, Vals);
      continue;
    }
    
    if (Op.getEncoding() == BitCodeAbbrevOp::Array) {
      // Array case.  Read the number of elements as a vbr6.
      unsigned NumElts = ReadVBR(6);
      
      // Get the element encoding.
      assert(i+2 == e && "array op not second to last?");
      const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);
      
      // Read all the elements.
      for (; NumElts; --NumElts)
        readAbbreviatedField(EltEnc, Vals);
      continue;
    }
    
    assert(Op.getEncoding() == BitCodeAbbrevOp::Blob);
    // Blob case.  Read the number of bytes as a vbr6.
    unsigned NumElts = ReadVBR(6);
    SkipToFourByteBoundary();  // 32-bit alignment
    
    // Figure out where the end of this blob will be including tail padding.
    size_t CurBitPos = GetCurrentBitNo();
    size_t NewEnd = CurBitPos+((NumElts+3)&~3)*8;
    
    // If this would read off the end of the bitcode file, just set the
    // record to empty and return.
    if (!canSkipToPos(NewEnd/8)) {
      Vals.append(NumElts, 0);
      NextChar = BitStream->getBitcodeBytes().getExtent();
      break;
    }
    
    // Otherwise, inform the streamer that we need these bytes in memory.
    const char *Ptr = (const char*)
      BitStream->getBitcodeBytes().getPointer(CurBitPos/8, NumElts);
    
    // If we can return a reference to the data, do so to avoid copying it.
    if (Blob) {
      *Blob = StringRef(Ptr, NumElts);
    } else {
      // Otherwise, unpack into Vals with zero extension.
      for (; NumElts; --NumElts)
        Vals.push_back((unsigned char)*Ptr++);
    }
    // Skip over tail padding.
    JumpToBit(NewEnd);
  }
  
  unsigned Code = (unsigned)Vals[0];
  Vals.erase(Vals.begin());
  return Code;
}


void BitstreamCursor::ReadAbbrevRecord() {
  BitCodeAbbrev *Abbv = new BitCodeAbbrev();
  unsigned NumOpInfo = ReadVBR(5);
  for (unsigned i = 0; i != NumOpInfo; ++i) {
    bool IsLiteral = Read(1) ? true : false;
    if (IsLiteral) {
      Abbv->Add(BitCodeAbbrevOp(ReadVBR64(8)));
      continue;
    }
    
    BitCodeAbbrevOp::Encoding E = (BitCodeAbbrevOp::Encoding)Read(3);
    if (BitCodeAbbrevOp::hasEncodingData(E))
      Abbv->Add(BitCodeAbbrevOp(E, ReadVBR64(5)));
    else
      Abbv->Add(BitCodeAbbrevOp(E));
  }
  CurAbbrevs.push_back(Abbv);
}

bool BitstreamCursor::ReadBlockInfoBlock() {
  // If this is the second stream to get to the block info block, skip it.
  if (BitStream->hasBlockInfoRecords())
    return SkipBlock();
  
  if (EnterSubBlock(bitc::BLOCKINFO_BLOCK_ID)) return true;
  
  SmallVector<uint64_t, 64> Record;
  BitstreamReader::BlockInfo *CurBlockInfo = 0;
  
  // Read all the records for this module.
  while (1) {
    BitstreamEntry Entry = advanceSkippingSubblocks(AF_DontAutoprocessAbbrevs);
    
    switch (Entry.Kind) {
    case llvm::BitstreamEntry::SubBlock: // Handled for us already.
    case llvm::BitstreamEntry::Error:
      return true;
    case llvm::BitstreamEntry::EndBlock:
      return false;
    case llvm::BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read abbrev records, associate them with CurBID.
    if (Entry.ID == bitc::DEFINE_ABBREV) {
      if (!CurBlockInfo) return true;
      ReadAbbrevRecord();
      
      // ReadAbbrevRecord installs the abbrev in CurAbbrevs.  Move it to the
      // appropriate BlockInfo.
      BitCodeAbbrev *Abbv = CurAbbrevs.back();
      CurAbbrevs.pop_back();
      CurBlockInfo->Abbrevs.push_back(Abbv);
      continue;
    }
    
    // Read a record.
    Record.clear();
    switch (readRecord(Entry.ID, Record)) {
      default: break;  // Default behavior, ignore unknown content.
      case bitc::BLOCKINFO_CODE_SETBID:
        if (Record.size() < 1) return true;
        CurBlockInfo = &BitStream->getOrCreateBlockInfo((unsigned)Record[0]);
        break;
      case bitc::BLOCKINFO_CODE_BLOCKNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 0, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->Name = Name;
        break;
      }
      case bitc::BLOCKINFO_CODE_SETRECORDNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 1, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->RecordNames.push_back(std::make_pair((unsigned)Record[0],
                                                           Name));
        break;
      }
    }
  }
}



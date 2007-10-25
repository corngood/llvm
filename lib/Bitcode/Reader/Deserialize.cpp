//==- Deserialize.cpp - Generic Object Serialization to Bitcode --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the internal methods used for object serialization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/Deserialize.h"

using namespace llvm;

Deserializer::Deserializer(BitstreamReader& stream)
  : Stream(stream), RecIdx(0) {
}

Deserializer::~Deserializer() {
  assert (RecIdx >= Record.size() && 
          "Still scanning bitcode record when deserialization completed.");
  
  BackpatchPointers();
}


bool Deserializer::inRecord() {
  if (Record.size() > 0) {
    if (RecIdx >= Record.size()) {
      RecIdx = 0;
      Record.clear();
      return false;
    }
    else return true;
  }
  else return false;
}

void Deserializer::ReadRecord() {
  // FIXME: Check if we haven't run off the edge of the stream.
  // FIXME: Handle abbreviations.

  // FIXME: Check for the correct code.
  unsigned Code = Stream.ReadCode();

  assert (Record.size() == 0);  
  Stream.ReadRecord(Code,Record);  
  assert (Record.size() > 0);
}

uint64_t Deserializer::ReadInt() {
  // FIXME: Any error recovery/handling with incomplete or bad files?
  if (!inRecord())
    ReadRecord();

  return Record[RecIdx++];
}

char* Deserializer::ReadCStr(char* cstr, unsigned MaxLen, bool isNullTerm) {
  if (cstr == NULL)
    MaxLen = 0; // Zero this just in case someone does something funny.
  
  unsigned len = ReadInt();

  assert (MaxLen == 0 || (len + (isNullTerm ? 1 : 0)) <= MaxLen);

  if (!cstr)
    cstr = new char[len + (isNullTerm ? 1 : 0)];
  
  assert (cstr != NULL);
  
  for (unsigned i = 0; i < len; ++i)
    cstr[i] = (char) ReadInt();
  
  if (isNullTerm)
    cstr[len+1] = '\0';
  
  return cstr;
}

void Deserializer::ReadCStr(std::vector<char>& buff, bool isNullTerm) {
  unsigned len = ReadInt();

  buff.clear();  
  buff.reserve(len);
  
  for (unsigned i = 0; i < len; ++i)
    buff.push_back((char) ReadInt());
  
  if (isNullTerm)
    buff.push_back('\0');
}

void Deserializer::RegisterPtr(unsigned PtrId,void* Ptr) {
  BPatchEntry& E = BPatchMap[PtrId];
  assert (E.Ptr == NULL && "Pointer already registered.");
  E.Ptr = Ptr;
}

void Deserializer::ReadPtr(void*& PtrRef) {
  unsigned PtrId = ReadInt();
  
  if (PtrId == 0) {
    PtrRef = NULL;
    return;
  }  
  
  BPatchEntry& E = BPatchMap[PtrId];
  
  if (E.Ptr == NULL) {
    // Register backpatch.
    void* P = Allocator.Allocate<BPatchNode>();    
    E.Head = new (P) BPatchNode(E.Head,PtrRef);
  }
  else
    PtrRef = E.Ptr;
}

void Deserializer::BackpatchPointers() {
  for (MapTy::iterator I=BPatchMap.begin(),E=BPatchMap.end(); I!=E; ++I) {
    
    BPatchEntry& Entry = I->second;
    assert (Entry.Ptr && "No pointer found for backpatch.");
    
    for (BPatchNode* N = Entry.Head; N != NULL; N = N->Next)
      // Bitwise-OR in the pointer to support "smart" pointers that use
      // unused bits to store extra data.
      N->PtrRef |= reinterpret_cast<uintptr_t>(Entry.Ptr);
    
    Entry.Head = NULL;
  }
  
  Allocator.Reset();
}

#define INT_READ(TYPE)\
void SerializeTrait<TYPE>::Read(Deserializer& D, TYPE& X) {\
  X = (TYPE) D.ReadInt(); }\
TYPE SerializeTrait<TYPE>::ReadVal(Deserializer& D) {\
  return (TYPE) D.ReadInt(); }

INT_READ(bool)
INT_READ(unsigned char)
INT_READ(unsigned short)
INT_READ(unsigned int)
INT_READ(unsigned long)

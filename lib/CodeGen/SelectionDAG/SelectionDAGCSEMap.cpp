//===-- SelectionDAGCSEMap.cpp - Implement the SelectionDAG CSE Map -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGCSEMap class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// SelectionDAGCSEMap::NodeID Implementation

SelectionDAGCSEMap::NodeID::NodeID(SDNode *N) {
  SetOpcode(N->getOpcode());
  // Add the return value info.
  SetValueTypes(N->value_begin());
  // Add the operand info.
  SetOperands(N->op_begin(), N->getNumOperands());
}

SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands();
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList,
                                   SDOperand Op) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(Op);
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList, 
                                   SDOperand Op1, SDOperand Op2) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(Op1, Op2);
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList, 
                                   SDOperand Op1, SDOperand Op2,
                                   SDOperand Op3) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(Op1, Op2, Op3);
}
SelectionDAGCSEMap::NodeID::NodeID(unsigned short ID, const void *VTList, 
                                   const SDOperand *OpList, unsigned N) {
  SetOpcode(ID);
  SetValueTypes(VTList);
  SetOperands(OpList, N);
}

void SelectionDAGCSEMap::NodeID::AddPointer(const void *Ptr) {
  // Note: this adds pointers to the hash using sizes and endianness that depend
  // on the host.  It doesn't matter however, because hashing on pointer values
  // in inherently unstable.  Nothing in the SelectionDAG should depend on the
  // ordering of nodes in the CSEMap.
  union {
    intptr_t PtrI;
    unsigned char PtrA[sizeof(intptr_t)];
  };
  PtrI = (intptr_t)Ptr;
  Bits.append(PtrA, PtrA+sizeof(intptr_t));
}

void SelectionDAGCSEMap::NodeID::AddOperand(SDOperand Op) {
  AddPointer(Op.Val);
  // 2 bytes of resno might be too small, three should certainly be enough. :)
  assert(Op.ResNo < (1 << 24) && "ResNo too large for CSE Map to handle!");
  Bits.push_back((Op.ResNo >>  0) & 0xFF);
  Bits.push_back((Op.ResNo >>  8) & 0xFF);
  Bits.push_back((Op.ResNo >> 16) & 0xFF);
}

void SelectionDAGCSEMap::NodeID::SetOperands(const SDOperand *Ops, 
                                             unsigned NumOps) {
  for (; NumOps; --NumOps, ++Ops)
    AddOperand(*Ops);
}


/// ComputeHash - Compute a strong hash value for this NodeID, for lookup in
/// the SelectionDAGCSEMap.
unsigned SelectionDAGCSEMap::NodeID::ComputeHash() const {
  // FIXME: this hash function sucks.
  unsigned Hash = 0;
  for (unsigned i = 0, e = Bits.size(); i != e; ++i)
    Hash += Bits[i];
  return Hash;
}

bool SelectionDAGCSEMap::NodeID::operator==(const NodeID &RHS) const {
  if (Bits.size() != RHS.Bits.size()) return false;
  return memcmp(&Bits[0], &RHS.Bits[0], Bits.size()) == 0;
}


//===----------------------------------------------------------------------===//
// SelectionDAGCSEMap Implementation

SelectionDAGCSEMap::SelectionDAGCSEMap() {
  NumBuckets = 256;
  Buckets = new void*[NumBuckets];
  memset(Buckets, 0, NumBuckets*sizeof(void*));
}
SelectionDAGCSEMap::~SelectionDAGCSEMap() {
  delete [] Buckets;
}

/// GetNextPtr - In order to save space, each bucket is a singly-linked-list. In
/// order to make deletion more efficient, we make the list circular, so we can
/// delete a node without computing its hash.  The problem with this is that the
/// start of the hash buckets are not SDNodes.  If NextInBucketPtr is a bucket
/// pointer, this method returns null: use GetBucketPtr when this happens.
SDNode *SelectionDAGCSEMap::GetNextPtr(void *NextInBucketPtr) {
  if (NextInBucketPtr >= Buckets && NextInBucketPtr < Buckets+NumBuckets)
    return 0;
  return static_cast<SDNode*>(NextInBucketPtr);
}

void **SelectionDAGCSEMap::GetBucketPtr(void *NextInBucketPtr) {
  assert(NextInBucketPtr >= Buckets && NextInBucketPtr < Buckets+NumBuckets &&
         "NextInBucketPtr is not a bucket ptr");
  return static_cast<void**>(NextInBucketPtr);
}

/// GetBucketFor - Hash the specified node ID and return the hash bucket for the
/// specified ID.
void **SelectionDAGCSEMap::GetBucketFor(const NodeID &ID) const {
  // TODO: if load is high, resize hash table.
  
  // NumBuckets is always a power of 2.
  unsigned BucketNum = ID.ComputeHash() & (NumBuckets-1);
  return Buckets+BucketNum;
}

/// FindNodeOrInsertPos - Look up the node specified by ID.  If it exists,
/// return it.  If not, return the insertion token that will make insertion
/// faster.
SDNode *SelectionDAGCSEMap::FindNodeOrInsertPos(const NodeID &ID,
                                                void *&InsertPos) {
  void **Bucket = GetBucketFor(ID);
  void *Probe = *Bucket;
  
  InsertPos = 0;
  
  unsigned Opc = ID.getOpcode();
  while (SDNode *NodeInBucket = GetNextPtr(Probe)) {
    // If we found a node with the same opcode, it might be a matching node.
    // Because it is in the same bucket as this one, we know the hash values
    // match.  Compute the NodeID for the possible match and do a final compare.
    if (NodeInBucket->getOpcode() == Opc) {
      NodeID OtherID(NodeInBucket);
      if (OtherID == ID)
        return NodeInBucket;
    }

    Probe = NodeInBucket->getNextInBucket();
  }
  
  // Didn't find the node, return null with the bucket as the InsertPos.
  InsertPos = Bucket;
  return 0;
}

/// InsertNode - Insert the specified node into the CSE Map, knowing that it
/// is not already in the map.  InsertPos must be obtained from 
/// FindNodeOrInsertPos.
void SelectionDAGCSEMap::InsertNode(SDNode *N, void *InsertPos) {
  /// The insert position is actually a bucket pointer.
  void **Bucket = static_cast<void**>(InsertPos);
  
  void *Next = *Bucket;
  
  // If this is the first insertion into this bucket, its next pointer will be
  // null.  Pretend as if it pointed to itself.
  if (Next == 0)
    Next = Bucket;

  // Set the nodes next pointer, and make the bucket point to the node.
  N->SetNextInBucket(Next);
  *Bucket = N;
}


/// RemoveNode - Remove a node from the CSE map, returning true if one was
/// removed or false if the node was not in the CSE map.
bool SelectionDAGCSEMap::RemoveNode(SDNode *N) {
  // Because each bucket is a circular list, we don't need to compute N's hash
  // to remove it.  Chase around the list until we find the node (or bucket)
  // which points to N.
  void *Ptr = N->getNextInBucket();
  if (Ptr == 0) return false;  // Not in CSEMap.
  
  void *NodeNextPtr = Ptr;
  N->SetNextInBucket(0);
  while (1) {
    if (SDNode *NodeInBucket = GetNextPtr(Ptr)) {
      // Advance pointer.
      Ptr = NodeInBucket->getNextInBucket();
      
      // We found a node that points to N, change it to point to N's next node,
      // removing N from the list.
      if (Ptr == N) {
        NodeInBucket->SetNextInBucket(NodeNextPtr);
        return true;
      }
    } else {
      void **Bucket = GetBucketPtr(Ptr);
      Ptr = *Bucket;
      
      // If we found that the bucket points to N, update the bucket to point to
      // whatever is next.
      if (Ptr == N) {
        *Bucket = NodeNextPtr;
        return true;
      }
    }
  }
}

/// GetOrInsertSimpleNode - If there is an existing simple SDNode exactly
/// equal to the specified node, return it.  Otherwise, insert 'N' and it
/// instead.  This only works on *simple* SDNodes, not ConstantSDNode or any
/// other classes derived from SDNode.
SDNode *SelectionDAGCSEMap::GetOrInsertNode(SDNode *N) {
  SelectionDAGCSEMap::NodeID ID(N);
  void *IP;
  if (SDNode *E = FindNodeOrInsertPos(ID, IP))
    return E;
  InsertNode(N, IP);
  return N;
}

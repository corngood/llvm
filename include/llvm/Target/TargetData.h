//===-- llvm/Target/TargetData.h - Data size & alignment info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target properties related to datatype size/offset/alignment
// information.  It uses lazy annotations to cache information about how
// structure types are laid out and used.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETDATA_H
#define LLVM_TARGET_TARGETDATA_H

#include "llvm/Pass.h"
#include "llvm/Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm {

class Value;
class Type;
class StructType;
class StructLayout;
class GlobalVariable;

class TargetData : public ImmutablePass {
  bool          LittleEndian;          // Defaults to false
  unsigned char BoolAlignment;         // Defaults to 1 byte
  unsigned char ByteAlignment;         // Defaults to 1 byte
  unsigned char ShortAlignment;        // Defaults to 2 bytes
  unsigned char IntAlignment;          // Defaults to 4 bytes
  unsigned char LongAlignment;         // Defaults to 8 bytes
  unsigned char FloatAlignment;        // Defaults to 4 bytes
  unsigned char DoubleAlignment;       // Defaults to 8 bytes
  unsigned char PointerSize;           // Defaults to 8 bytes
  unsigned char PointerAlignment;      // Defaults to 8 bytes

public:
  /// Default ctor - This has to exist, because this is a pass, but it should
  /// never be used.
  TargetData() {
    assert(0 && "ERROR: Bad TargetData ctor used.  "
           "Tool did not specify a TargetData to use?");
    abort();
  }
    
  /// Constructs a TargetData from a string of the following format:
  /// "E-p:64:64-d:64-f:32-l:64-i:32-s:16-b:8-B:8"
  /// The above string is considered the default, and any values not specified
  /// in the string will be assumed to be as above.
  TargetData(const std::string &TargetDescription) {
    init(TargetDescription);
  }

  /// Initialize target data from properties stored in the module.
  TargetData(const Module *M);

  TargetData(const TargetData &TD) : 
    ImmutablePass(),
    LittleEndian(TD.isLittleEndian()),
    BoolAlignment(TD.getBoolAlignment()),
    ByteAlignment(TD.getByteAlignment()),
    ShortAlignment(TD.getShortAlignment()),
    IntAlignment(TD.getIntAlignment()),
    LongAlignment(TD.getLongAlignment()),
    FloatAlignment(TD.getFloatAlignment()),
    DoubleAlignment(TD.getDoubleAlignment()),
    PointerSize(TD.getPointerSize()),
    PointerAlignment(TD.getPointerAlignment()) {
  }

  ~TargetData();  // Not virtual, do not subclass this class

  /// Parse a target data layout string and initialize TargetData members.
  ///
  /// Parse a target data layout string, initializing the various TargetData
  /// members along the way. A TargetData specification string looks like
  /// "E-p:64:64-d:64-f:32-l:64-i:32-s:16-b:8-B:8" and specifies the
  /// target's endianess, the alignments of various data types and
  /// the size of pointers. The "-" is used as a separator and ":"
  /// separates a token from its argument. Alignment is indicated in bits
  /// and internally converted to the appropriate number of bytes.
  ///
  /// Valid tokens:
  /// <br>
  /// <em>E</em> specifies big endian architecture (1234) [default]<br>
  /// <em>e</em> specifies little endian architecture (4321) <br>
  /// <em>p:[ptr size]:[ptr align]</em> specifies pointer size and alignment
  /// [default = 64:64] <br>
  /// <em>d:[align]</em> specifies double floating point alignment
  /// [default = 64] <br>
  /// <em>f:[align]</em> specifies single floating point alignment
  /// [default = 32] <br>
  /// <em>l:[align]</em> specifies long integer alignment
  /// [default = 64] <br>
  /// <em>i:[align]</em> specifies integer alignment
  /// [default = 32] <br>
  /// <em>s:[align]</em> specifies short integer alignment
  /// [default = 16] <br>
  /// <em>b:[align]</em> specifies byte data type alignment
  /// [default = 8] <br>
  /// <em>B:[align]</em> specifies boolean data type alignment
  /// [default = 8] <br>
  ///
  /// All other token types are silently ignored.
  void init(const std::string &TargetDescription);
  
  
  /// Target endianness...
  bool          isLittleEndian()       const { return     LittleEndian; }
  bool          isBigEndian()          const { return    !LittleEndian; }

  /// Target alignment constraints
  unsigned char getBoolAlignment()     const { return    BoolAlignment; }
  unsigned char getByteAlignment()     const { return    ByteAlignment; }
  unsigned char getShortAlignment()    const { return   ShortAlignment; }
  unsigned char getIntAlignment()      const { return     IntAlignment; }
  unsigned char getLongAlignment()     const { return    LongAlignment; }
  unsigned char getFloatAlignment()    const { return   FloatAlignment; }
  unsigned char getDoubleAlignment()   const { return  DoubleAlignment; }
  unsigned char getPointerAlignment()  const { return PointerAlignment; }
  unsigned char getPointerSize()       const { return      PointerSize; }
  unsigned char getPointerSizeInBits() const { return    8*PointerSize; }

  /// getStringRepresentation - Return the string representation of the
  /// TargetData.  This representation is in the same format accepted by the
  /// string constructor above.
  std::string getStringRepresentation() const;

  /// getTypeSize - Return the number of bytes necessary to hold the specified
  /// type.
  ///
  uint64_t getTypeSize(const Type *Ty) const;

  /// getTypeAlignment - Return the minimum required alignment for the specified
  /// type.
  ///
  unsigned char getTypeAlignment(const Type *Ty) const;

  /// getTypeAlignmentShift - Return the minimum required alignment for the
  /// specified type, returned as log2 of the value (a shift amount).
  ///
  unsigned char getTypeAlignmentShift(const Type *Ty) const;

  /// getIntPtrType - Return an unsigned integer type that is the same size or
  /// greater to the host pointer size.
  ///
  const Type *getIntPtrType() const;

  /// getIndexOffset - return the offset from the beginning of the type for the
  /// specified indices.  This is used to implement getelementptr.
  ///
  uint64_t getIndexedOffset(const Type *Ty,
                            const std::vector<Value*> &Indices) const;

  /// getStructLayout - Return a StructLayout object, indicating the alignment
  /// of the struct, its size, and the offsets of its fields.  Note that this
  /// information is lazily cached.
  const StructLayout *getStructLayout(const StructType *Ty) const;
  
  /// InvalidateStructLayoutInfo - TargetData speculatively caches StructLayout
  /// objects.  If a TargetData object is alive when types are being refined and
  /// removed, this method must be called whenever a StructType is removed to
  /// avoid a dangling pointer in this cache.
  void InvalidateStructLayoutInfo(const StructType *Ty) const;

  /// getPreferredAlignmentLog - Return the preferred alignment of the
  /// specified global, returned in log form.  This includes an explicitly
  /// requested alignment (if the global has one).
  unsigned getPreferredAlignmentLog(const GlobalVariable *GV) const;
};

/// StructLayout - used to lazily calculate structure layout information for a
/// target machine, based on the TargetData structure.
///
class StructLayout {
public:
  std::vector<uint64_t> MemberOffsets;
  uint64_t StructSize;
  unsigned StructAlignment;

  /// getElementContainingOffset - Given a valid offset into the structure,
  /// return the structure index that contains it.
  ///
  unsigned getElementContainingOffset(uint64_t Offset) const;

private:
  friend class TargetData;   // Only TargetData can create this class
  StructLayout(const StructType *ST, const TargetData &TD);
};

} // End llvm namespace

#endif

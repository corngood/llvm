//===-- llvm/CodeGen/SelectionDAGNodes.h - SelectionDAG Nodes ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDNode class and derived classes, which are used to
// represent the nodes and operations present in a SelectionDAG.  These nodes
// and operations are machine code level operations, with some similarities to
// the GCC RTL representation.
//
// Clients should include the SelectionDAG.h file instead of this file directly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAGNODES_H
#define LLVM_CODEGEN_SELECTIONDAGNODES_H

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Value.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/iterator"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <vector>

namespace llvm {

class SelectionDAG;
class GlobalValue;
class MachineBasicBlock;
class SDNode;
template <typename T> struct simplify_type;
template <typename T> struct ilist_traits;
template<typename NodeTy, typename Traits> class iplist;
template<typename NodeTy> class ilist_iterator;

/// ISD namespace - This namespace contains an enum which represents all of the
/// SelectionDAG node types and value types.
///
namespace ISD {
  //===--------------------------------------------------------------------===//
  /// ISD::NodeType enum - This enum defines all of the operators valid in a
  /// SelectionDAG.
  ///
  enum NodeType {
    // EntryToken - This is the marker used to indicate the start of the region.
    EntryToken,

    // Token factor - This node takes multiple tokens as input and produces a
    // single token result.  This is used to represent the fact that the operand
    // operators are independent of each other.
    TokenFactor,
    
    // AssertSext, AssertZext - These nodes record if a register contains a 
    // value that has already been zero or sign extended from a narrower type.  
    // These nodes take two operands.  The first is the node that has already 
    // been extended, and the second is a value type node indicating the width
    // of the extension
    AssertSext, AssertZext,

    // Various leaf nodes.
    Constant, ConstantFP, STRING,
    GlobalAddress, FrameIndex, ConstantPool,
    BasicBlock, ExternalSymbol, VALUETYPE, CONDCODE, Register,
    
    // ConstantVec works like Constant or ConstantFP, except that it is not a
    // leaf node.  All operands are either Constant or ConstantFP nodes.
    ConstantVec,
    
    // TargetConstant* - Like Constant*, but the DAG does not do any folding or
    // simplification of the constant.
    TargetConstant,
    TargetConstantFP,
    TargetConstantVec, 
    
    // TargetGlobalAddress - Like GlobalAddress, but the DAG does no folding or
    // anything else with this node, and this is valid in the target-specific
    // dag, turning into a GlobalAddress operand.
    TargetGlobalAddress,
    TargetFrameIndex,
    TargetConstantPool,
    TargetExternalSymbol,

    // CopyToReg - This node has three operands: a chain, a register number to
    // set to this value, and a value.  
    CopyToReg,

    // CopyFromReg - This node indicates that the input value is a virtual or
    // physical register that is defined outside of the scope of this
    // SelectionDAG.  The register is available from the RegSDNode object.
    CopyFromReg,

    // UNDEF - An undefined node
    UNDEF,

    // EXTRACT_ELEMENT - This is used to get the first or second (determined by
    // a Constant, which is required to be operand #1), element of the aggregate
    // value specified as operand #0.  This is only for use before legalization,
    // for values that will be broken into multiple registers.
    EXTRACT_ELEMENT,

    // BUILD_PAIR - This is the opposite of EXTRACT_ELEMENT in some ways.  Given
    // two values of the same integer value type, this produces a value twice as
    // big.  Like EXTRACT_ELEMENT, this can only be used before legalization.
    BUILD_PAIR,
    
    // MERGE_VALUES - This node takes multiple discrete operands and returns
    // them all as its individual results.  This nodes has exactly the same
    // number of inputs and outputs, and is only valid before legalization.
    // This node is useful for some pieces of the code generator that want to
    // think about a single node with multiple results, not multiple nodes.
    MERGE_VALUES,

    // Simple integer binary arithmetic operators.
    ADD, SUB, MUL, SDIV, UDIV, SREM, UREM,
    
    // Simple binary floating point operators.
    FADD, FSUB, FMUL, FDIV, FREM,
    
    // Simple abstract vector operators.  Unlike the integer and floating point
    // binary operators, these nodes also take two additional operands:
    // a constant element count, and a value type node indicating the type of
    // the elements.  The order is op0, op1, count, type.  All vector opcodes,
    // including VLOAD, must currently have count and type as their 3rd and 4th
    // arguments.
    VADD, VSUB, VMUL,

    // MULHU/MULHS - Multiply high - Multiply two integers of type iN, producing
    // an unsigned/signed value of type i[2*n], then return the top part.
    MULHU, MULHS,

    // Bitwise operators - logical and, logical or, logical xor, shift left,
    // shift right algebraic (shift in sign bits), shift right logical (shift in
    // zeroes), rotate left, rotate right, and byteswap.
    AND, OR, XOR, SHL, SRA, SRL, ROTL, ROTR, BSWAP,

    // Counting operators
    CTTZ, CTLZ, CTPOP,

    // Select
    SELECT, 
    
    // Select with condition operator - This selects between a true value and 
    // a false value (ops #2 and #3) based on the boolean result of comparing
    // the lhs and rhs (ops #0 and #1) of a conditional expression with the 
    // condition code in op #4, a CondCodeSDNode.
    SELECT_CC,

    // SetCC operator - This evaluates to a boolean (i1) true value if the
    // condition is true.  The operands to this are the left and right operands
    // to compare (ops #0, and #1) and the condition code to compare them with
    // (op #2) as a CondCodeSDNode.
    SETCC,

    // ADD_PARTS/SUB_PARTS - These operators take two logical operands which are
    // broken into a multiple pieces each, and return the resulting pieces of
    // doing an atomic add/sub operation.  This is used to handle add/sub of
    // expanded types.  The operation ordering is:
    //       [Lo,Hi] = op [LoLHS,HiLHS], [LoRHS,HiRHS]
    ADD_PARTS, SUB_PARTS,

    // SHL_PARTS/SRA_PARTS/SRL_PARTS - These operators are used for expanded
    // integer shift operations, just like ADD/SUB_PARTS.  The operation
    // ordering is:
    //       [Lo,Hi] = op [LoLHS,HiLHS], Amt
    SHL_PARTS, SRA_PARTS, SRL_PARTS,

    // Conversion operators.  These are all single input single output
    // operations.  For all of these, the result type must be strictly
    // wider or narrower (depending on the operation) than the source
    // type.

    // SIGN_EXTEND - Used for integer types, replicating the sign bit
    // into new bits.
    SIGN_EXTEND,

    // ZERO_EXTEND - Used for integer types, zeroing the new bits.
    ZERO_EXTEND,

    // ANY_EXTEND - Used for integer types.  The high bits are undefined.
    ANY_EXTEND,
    
    // TRUNCATE - Completely drop the high bits.
    TRUNCATE,

    // [SU]INT_TO_FP - These operators convert integers (whose interpreted sign
    // depends on the first letter) to floating point.
    SINT_TO_FP,
    UINT_TO_FP,

    // SIGN_EXTEND_INREG - This operator atomically performs a SHL/SRA pair to
    // sign extend a small value in a large integer register (e.g. sign
    // extending the low 8 bits of a 32-bit register to fill the top 24 bits
    // with the 7th bit).  The size of the smaller type is indicated by the 1th
    // operand, a ValueType node.
    SIGN_EXTEND_INREG,

    // FP_TO_[US]INT - Convert a floating point value to a signed or unsigned
    // integer.
    FP_TO_SINT,
    FP_TO_UINT,

    // FP_ROUND - Perform a rounding operation from the current
    // precision down to the specified precision (currently always 64->32).
    FP_ROUND,

    // FP_ROUND_INREG - This operator takes a floating point register, and
    // rounds it to a floating point value.  It then promotes it and returns it
    // in a register of the same size.  This operation effectively just discards
    // excess precision.  The type to round down to is specified by the 1th
    // operation, a VTSDNode (currently always 64->32->64).
    FP_ROUND_INREG,

    // FP_EXTEND - Extend a smaller FP type into a larger FP type.
    FP_EXTEND,

    // BIT_CONVERT - Theis operator converts between integer and FP values, as
    // if one was stored to memory as integer and the other was loaded from the
    // same address (or equivalently for vector format conversions, etc).  The 
    // source and result are required to have the same bit size (e.g. 
    // f32 <-> i32).  This can also be used for int-to-int or fp-to-fp 
    // conversions, but that is a noop, deleted by getNode().
    BIT_CONVERT,
    
    // FNEG, FABS, FSQRT, FSIN, FCOS - Perform unary floating point negation,
    // absolute value, square root, sine and cosine operations.
    FNEG, FABS, FSQRT, FSIN, FCOS,

    // Other operators.  LOAD and STORE have token chains as their first
    // operand, then the same operands as an LLVM load/store instruction, then a
    // SRCVALUE node that provides alias analysis information.
    LOAD, STORE,
    
    // Abstract vector version of LOAD.  VLOAD has a token chain as the first
    // operand, followed by a pointer operand, a constant element count, a value
    // type node indicating the type of the elements, and a SRCVALUE node.
    VLOAD,

    // EXTLOAD, SEXTLOAD, ZEXTLOAD - These three operators all load a value from
    // memory and extend them to a larger value (e.g. load a byte into a word
    // register).  All three of these have four operands, a token chain, a
    // pointer to load from, a SRCVALUE for alias analysis, and a VALUETYPE node
    // indicating the type to load.
    //
    // SEXTLOAD loads the integer operand and sign extends it to a larger
    //          integer result type.
    // ZEXTLOAD loads the integer operand and zero extends it to a larger
    //          integer result type.
    // EXTLOAD  is used for two things: floating point extending loads, and
    //          integer extending loads where it doesn't matter what the high
    //          bits are set to.  The code generator is allowed to codegen this
    //          into whichever operation is more efficient.
    EXTLOAD, SEXTLOAD, ZEXTLOAD,

    // TRUNCSTORE - This operators truncates (for integer) or rounds (for FP) a
    // value and stores it to memory in one operation.  This can be used for
    // either integer or floating point operands.  The first four operands of
    // this are the same as a standard store.  The fifth is the ValueType to
    // store it as (which will be smaller than the source value).
    TRUNCSTORE,

    // DYNAMIC_STACKALLOC - Allocate some number of bytes on the stack aligned
    // to a specified boundary.  The first operand is the token chain, the
    // second is the number of bytes to allocate, and the third is the alignment
    // boundary.  The size is guaranteed to be a multiple of the stack 
    // alignment, and the alignment is guaranteed to be bigger than the stack 
    // alignment (if required) or 0 to get standard stack alignment.
    DYNAMIC_STACKALLOC,

    // Control flow instructions.  These all have token chains.

    // BR - Unconditional branch.  The first operand is the chain
    // operand, the second is the MBB to branch to.
    BR,

    // BRCOND - Conditional branch.  The first operand is the chain,
    // the second is the condition, the third is the block to branch
    // to if the condition is true.
    BRCOND,

    // BRCONDTWOWAY - Two-way conditional branch.  The first operand is the
    // chain, the second is the condition, the third is the block to branch to
    // if true, and the forth is the block to branch to if false.  Targets
    // usually do not implement this, preferring to have legalize demote the
    // operation to BRCOND/BR pairs when necessary.
    BRCONDTWOWAY,

    // BR_CC - Conditional branch.  The behavior is like that of SELECT_CC, in
    // that the condition is represented as condition code, and two nodes to
    // compare, rather than as a combined SetCC node.  The operands in order are
    // chain, cc, lhs, rhs, block to branch to if condition is true.
    BR_CC,
    
    // BRTWOWAY_CC - Two-way conditional branch.  The operands in order are
    // chain, cc, lhs, rhs, block to branch to if condition is true, block to
    // branch to if condition is false.  Targets usually do not implement this,
    // preferring to have legalize demote the operation to BRCOND/BR pairs.
    BRTWOWAY_CC,
    
    // RET - Return from function.  The first operand is the chain,
    // and any subsequent operands are the return values for the
    // function.  This operation can have variable number of operands.
    RET,

    // INLINEASM - Represents an inline asm block.  This node always has two
    // return values: a chain and a flag result.  The inputs are as follows:
    //   Operand #0   : Input chain.
    //   Operand #1   : a ExternalSymbolSDNode with a pointer to the asm string.
    //   Operand #2n+2: A RegisterNode.
    //   Operand #2n+3: A TargetConstant, indicating if the reg is a use/def
    //   Operand #last: Optional, an incoming flag.
    INLINEASM,

    // STACKSAVE - STACKSAVE has one operand, an input chain.  It produces a
    // value, the same type as the pointer type for the system, and an output
    // chain.
    STACKSAVE,
    
    // STACKRESTORE has two operands, an input chain and a pointer to restore to
    // it returns an output chain.
    STACKRESTORE,
    
    // MEMSET/MEMCPY/MEMMOVE - The first operand is the chain, and the rest
    // correspond to the operands of the LLVM intrinsic functions.  The only
    // result is a token chain.  The alignment argument is guaranteed to be a
    // Constant node.
    MEMSET,
    MEMMOVE,
    MEMCPY,

    // CALLSEQ_START/CALLSEQ_END - These operators mark the beginning and end of
    // a call sequence, and carry arbitrary information that target might want
    // to know.  The first operand is a chain, the rest are specified by the
    // target and not touched by the DAG optimizers.
    CALLSEQ_START,  // Beginning of a call sequence
    CALLSEQ_END,    // End of a call sequence
    
    // VAARG - VAARG has three operands: an input chain, a pointer, and a 
    // SRCVALUE.  It returns a pair of values: the vaarg value and a new chain.
    VAARG,
    
    // VACOPY - VACOPY has five operands: an input chain, a destination pointer,
    // a source pointer, a SRCVALUE for the destination, and a SRCVALUE for the
    // source.
    VACOPY,
    
    // VAEND, VASTART - VAEND and VASTART have three operands: an input chain, a
    // pointer, and a SRCVALUE.
    VAEND, VASTART,

    // SRCVALUE - This corresponds to a Value*, and is used to associate memory
    // locations with their value.  This allows one use alias analysis
    // information in the backend.
    SRCVALUE,

    // PCMARKER - This corresponds to the pcmarker intrinsic.
    PCMARKER,

    // READCYCLECOUNTER - This corresponds to the readcyclecounter intrinsic.
    // The only operand is a chain and a value and a chain are produced.  The
    // value is the contents of the architecture specific cycle counter like 
    // register (or other high accuracy low latency clock source)
    READCYCLECOUNTER,

    // READPORT, WRITEPORT, READIO, WRITEIO - These correspond to the LLVM
    // intrinsics of the same name.  The first operand is a token chain, the
    // other operands match the intrinsic.  These produce a token chain in
    // addition to a value (if any).
    READPORT, WRITEPORT, READIO, WRITEIO,
    
    // HANDLENODE node - Used as a handle for various purposes.
    HANDLENODE,

    // LOCATION - This node is used to represent a source location for debug
    // info.  It takes token chain as input, then a line number, then a column
    // number, then a filename, then a working dir.  It produces a token chain
    // as output.
    LOCATION,
    
    // DEBUG_LOC - This node is used to represent source line information
    // embedded in the code.  It takes a token chain as input, then a line
    // number, then a column then a file id (provided by MachineDebugInfo.) It
    // produces a token chain as output.
    DEBUG_LOC,
    
    // DEBUG_LABEL - This node is used to mark a location in the code where a
    // label should be generated for use by the debug information.  It takes a
    // token chain as input and then a unique id (provided by MachineDebugInfo.)
    // It produces a token chain as output.
    DEBUG_LABEL,
    
    // BUILTIN_OP_END - This must be the last enum value in this list.
    BUILTIN_OP_END,
  };

  //===--------------------------------------------------------------------===//
  /// ISD::CondCode enum - These are ordered carefully to make the bitfields
  /// below work out, when considering SETFALSE (something that never exists
  /// dynamically) as 0.  "U" -> Unsigned (for integer operands) or Unordered
  /// (for floating point), "L" -> Less than, "G" -> Greater than, "E" -> Equal
  /// to.  If the "N" column is 1, the result of the comparison is undefined if
  /// the input is a NAN.
  ///
  /// All of these (except for the 'always folded ops') should be handled for
  /// floating point.  For integer, only the SETEQ,SETNE,SETLT,SETLE,SETGT,
  /// SETGE,SETULT,SETULE,SETUGT, and SETUGE opcodes are used.
  ///
  /// Note that these are laid out in a specific order to allow bit-twiddling
  /// to transform conditions.
  enum CondCode {
    // Opcode          N U L G E       Intuitive operation
    SETFALSE,      //    0 0 0 0       Always false (always folded)
    SETOEQ,        //    0 0 0 1       True if ordered and equal
    SETOGT,        //    0 0 1 0       True if ordered and greater than
    SETOGE,        //    0 0 1 1       True if ordered and greater than or equal
    SETOLT,        //    0 1 0 0       True if ordered and less than
    SETOLE,        //    0 1 0 1       True if ordered and less than or equal
    SETONE,        //    0 1 1 0       True if ordered and operands are unequal
    SETO,          //    0 1 1 1       True if ordered (no nans)
    SETUO,         //    1 0 0 0       True if unordered: isnan(X) | isnan(Y)
    SETUEQ,        //    1 0 0 1       True if unordered or equal
    SETUGT,        //    1 0 1 0       True if unordered or greater than
    SETUGE,        //    1 0 1 1       True if unordered, greater than, or equal
    SETULT,        //    1 1 0 0       True if unordered or less than
    SETULE,        //    1 1 0 1       True if unordered, less than, or equal
    SETUNE,        //    1 1 1 0       True if unordered or not equal
    SETTRUE,       //    1 1 1 1       Always true (always folded)
    // Don't care operations: undefined if the input is a nan.
    SETFALSE2,     //  1 X 0 0 0       Always false (always folded)
    SETEQ,         //  1 X 0 0 1       True if equal
    SETGT,         //  1 X 0 1 0       True if greater than
    SETGE,         //  1 X 0 1 1       True if greater than or equal
    SETLT,         //  1 X 1 0 0       True if less than
    SETLE,         //  1 X 1 0 1       True if less than or equal
    SETNE,         //  1 X 1 1 0       True if not equal
    SETTRUE2,      //  1 X 1 1 1       Always true (always folded)

    SETCC_INVALID,      // Marker value.
  };

  /// isSignedIntSetCC - Return true if this is a setcc instruction that
  /// performs a signed comparison when used with integer operands.
  inline bool isSignedIntSetCC(CondCode Code) {
    return Code == SETGT || Code == SETGE || Code == SETLT || Code == SETLE;
  }

  /// isUnsignedIntSetCC - Return true if this is a setcc instruction that
  /// performs an unsigned comparison when used with integer operands.
  inline bool isUnsignedIntSetCC(CondCode Code) {
    return Code == SETUGT || Code == SETUGE || Code == SETULT || Code == SETULE;
  }

  /// isTrueWhenEqual - Return true if the specified condition returns true if
  /// the two operands to the condition are equal.  Note that if one of the two
  /// operands is a NaN, this value is meaningless.
  inline bool isTrueWhenEqual(CondCode Cond) {
    return ((int)Cond & 1) != 0;
  }

  /// getUnorderedFlavor - This function returns 0 if the condition is always
  /// false if an operand is a NaN, 1 if the condition is always true if the
  /// operand is a NaN, and 2 if the condition is undefined if the operand is a
  /// NaN.
  inline unsigned getUnorderedFlavor(CondCode Cond) {
    return ((int)Cond >> 3) & 3;
  }

  /// getSetCCInverse - Return the operation corresponding to !(X op Y), where
  /// 'op' is a valid SetCC operation.
  CondCode getSetCCInverse(CondCode Operation, bool isInteger);

  /// getSetCCSwappedOperands - Return the operation corresponding to (Y op X)
  /// when given the operation for (X op Y).
  CondCode getSetCCSwappedOperands(CondCode Operation);

  /// getSetCCOrOperation - Return the result of a logical OR between different
  /// comparisons of identical values: ((X op1 Y) | (X op2 Y)).  This
  /// function returns SETCC_INVALID if it is not possible to represent the
  /// resultant comparison.
  CondCode getSetCCOrOperation(CondCode Op1, CondCode Op2, bool isInteger);

  /// getSetCCAndOperation - Return the result of a logical AND between
  /// different comparisons of identical values: ((X op1 Y) & (X op2 Y)).  This
  /// function returns SETCC_INVALID if it is not possible to represent the
  /// resultant comparison.
  CondCode getSetCCAndOperation(CondCode Op1, CondCode Op2, bool isInteger);
}  // end llvm::ISD namespace


//===----------------------------------------------------------------------===//
/// SDOperand - Unlike LLVM values, Selection DAG nodes may return multiple
/// values as the result of a computation.  Many nodes return multiple values,
/// from loads (which define a token and a return value) to ADDC (which returns
/// a result and a carry value), to calls (which may return an arbitrary number
/// of values).
///
/// As such, each use of a SelectionDAG computation must indicate the node that
/// computes it as well as which return value to use from that node.  This pair
/// of information is represented with the SDOperand value type.
///
class SDOperand {
public:
  SDNode *Val;        // The node defining the value we are using.
  unsigned ResNo;     // Which return value of the node we are using.

  SDOperand() : Val(0) {}
  SDOperand(SDNode *val, unsigned resno) : Val(val), ResNo(resno) {}

  bool operator==(const SDOperand &O) const {
    return Val == O.Val && ResNo == O.ResNo;
  }
  bool operator!=(const SDOperand &O) const {
    return !operator==(O);
  }
  bool operator<(const SDOperand &O) const {
    return Val < O.Val || (Val == O.Val && ResNo < O.ResNo);
  }

  SDOperand getValue(unsigned R) const {
    return SDOperand(Val, R);
  }

  /// getValueType - Return the ValueType of the referenced return value.
  ///
  inline MVT::ValueType getValueType() const;

  // Forwarding methods - These forward to the corresponding methods in SDNode.
  inline unsigned getOpcode() const;
  inline unsigned getNodeDepth() const;
  inline unsigned getNumOperands() const;
  inline const SDOperand &getOperand(unsigned i) const;
  inline bool isTargetOpcode() const;
  inline unsigned getTargetOpcode() const;

  /// hasOneUse - Return true if there is exactly one operation using this
  /// result value of the defining operator.
  inline bool hasOneUse() const;
};


/// simplify_type specializations - Allow casting operators to work directly on
/// SDOperands as if they were SDNode*'s.
template<> struct simplify_type<SDOperand> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDOperand &Val) {
    return static_cast<SimpleType>(Val.Val);
  }
};
template<> struct simplify_type<const SDOperand> {
  typedef SDNode* SimpleType;
  static SimpleType getSimplifiedValue(const SDOperand &Val) {
    return static_cast<SimpleType>(Val.Val);
  }
};


/// SDNode - Represents one node in the SelectionDAG.
///
class SDNode {
  /// NodeType - The operation that this node performs.
  ///
  unsigned short NodeType;

  /// NodeDepth - Node depth is defined as MAX(Node depth of children)+1.  This
  /// means that leaves have a depth of 1, things that use only leaves have a
  /// depth of 2, etc.
  unsigned short NodeDepth;

  /// OperandList - The values that are used by this operation.
  ///
  SDOperand *OperandList;
  
  /// ValueList - The types of the values this node defines.  SDNode's may
  /// define multiple values simultaneously.
  MVT::ValueType *ValueList;

  /// NumOperands/NumValues - The number of entries in the Operand/Value list.
  unsigned short NumOperands, NumValues;
  
  /// Prev/Next pointers - These pointers form the linked list of of the
  /// AllNodes list in the current DAG.
  SDNode *Prev, *Next;
  friend struct ilist_traits<SDNode>;

  /// Uses - These are all of the SDNode's that use a value produced by this
  /// node.
  std::vector<SDNode*> Uses;
public:
  virtual ~SDNode() {
    assert(NumOperands == 0 && "Operand list not cleared before deletion");
  }
  
  //===--------------------------------------------------------------------===//
  //  Accessors
  //
  unsigned getOpcode()  const { return NodeType; }
  bool isTargetOpcode() const { return NodeType >= ISD::BUILTIN_OP_END; }
  unsigned getTargetOpcode() const {
    assert(isTargetOpcode() && "Not a target opcode!");
    return NodeType - ISD::BUILTIN_OP_END;
  }

  size_t use_size() const { return Uses.size(); }
  bool use_empty() const { return Uses.empty(); }
  bool hasOneUse() const { return Uses.size() == 1; }

  /// getNodeDepth - Return the distance from this node to the leaves in the
  /// graph.  The leaves have a depth of 1.
  unsigned getNodeDepth() const { return NodeDepth; }

  typedef std::vector<SDNode*>::const_iterator use_iterator;
  use_iterator use_begin() const { return Uses.begin(); }
  use_iterator use_end() const { return Uses.end(); }

  /// hasNUsesOfValue - Return true if there are exactly NUSES uses of the
  /// indicated value.  This method ignores uses of other values defined by this
  /// operation.
  bool hasNUsesOfValue(unsigned NUses, unsigned Value);

  /// getNumOperands - Return the number of values used by this operation.
  ///
  unsigned getNumOperands() const { return NumOperands; }

  const SDOperand &getOperand(unsigned Num) const {
    assert(Num < NumOperands && "Invalid child # of SDNode!");
    return OperandList[Num];
  }
  typedef const SDOperand* op_iterator;
  op_iterator op_begin() const { return OperandList; }
  op_iterator op_end() const { return OperandList+NumOperands; }


  /// getNumValues - Return the number of values defined/returned by this
  /// operator.
  ///
  unsigned getNumValues() const { return NumValues; }

  /// getValueType - Return the type of a specified result.
  ///
  MVT::ValueType getValueType(unsigned ResNo) const {
    assert(ResNo < NumValues && "Illegal result number!");
    return ValueList[ResNo];
  }

  typedef const MVT::ValueType* value_iterator;
  value_iterator value_begin() const { return ValueList; }
  value_iterator value_end() const { return ValueList+NumValues; }

  /// getOperationName - Return the opcode of this operation for printing.
  ///
  const char* getOperationName(const SelectionDAG *G = 0) const;
  void dump() const;
  void dump(const SelectionDAG *G) const;

  static bool classof(const SDNode *) { return true; }


  /// setAdjCallChain - This method should only be used by the legalizer.
  void setAdjCallChain(SDOperand N);
  void setAdjCallFlag(SDOperand N);

protected:
  friend class SelectionDAG;
  
  /// getValueTypeList - Return a pointer to the specified value type.
  ///
  static MVT::ValueType *getValueTypeList(MVT::ValueType VT);

  SDNode(unsigned NT, MVT::ValueType VT) : NodeType(NT), NodeDepth(1) {
    OperandList = 0; NumOperands = 0;
    ValueList = getValueTypeList(VT);
    NumValues = 1;
    Prev = 0; Next = 0;
  }
  SDNode(unsigned NT, SDOperand Op)
    : NodeType(NT), NodeDepth(Op.Val->getNodeDepth()+1) {
    OperandList = new SDOperand[1];
    OperandList[0] = Op;
    NumOperands = 1;
    Op.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2)
    : NodeType(NT) {
    if (N1.Val->getNodeDepth() > N2.Val->getNodeDepth())
      NodeDepth = N1.Val->getNodeDepth()+1;
    else
      NodeDepth = N2.Val->getNodeDepth()+1;
    OperandList = new SDOperand[2];
    OperandList[0] = N1;
    OperandList[1] = N2;
    NumOperands = 2;
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2, SDOperand N3)
    : NodeType(NT) {
    unsigned ND = N1.Val->getNodeDepth();
    if (ND < N2.Val->getNodeDepth())
      ND = N2.Val->getNodeDepth();
    if (ND < N3.Val->getNodeDepth())
      ND = N3.Val->getNodeDepth();
    NodeDepth = ND+1;

    OperandList = new SDOperand[3];
    OperandList[0] = N1;
    OperandList[1] = N2;
    OperandList[2] = N3;
    NumOperands = 3;
    
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    N3.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
  }
  SDNode(unsigned NT, SDOperand N1, SDOperand N2, SDOperand N3, SDOperand N4)
    : NodeType(NT) {
    unsigned ND = N1.Val->getNodeDepth();
    if (ND < N2.Val->getNodeDepth())
      ND = N2.Val->getNodeDepth();
    if (ND < N3.Val->getNodeDepth())
      ND = N3.Val->getNodeDepth();
    if (ND < N4.Val->getNodeDepth())
      ND = N4.Val->getNodeDepth();
    NodeDepth = ND+1;

    OperandList = new SDOperand[4];
    OperandList[0] = N1;
    OperandList[1] = N2;
    OperandList[2] = N3;
    OperandList[3] = N4;
    NumOperands = 4;
    
    N1.Val->Uses.push_back(this); N2.Val->Uses.push_back(this);
    N3.Val->Uses.push_back(this); N4.Val->Uses.push_back(this);
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
  }
  SDNode(unsigned Opc, const std::vector<SDOperand> &Nodes) : NodeType(Opc) {
    NumOperands = Nodes.size();
    OperandList = new SDOperand[NumOperands];
    
    unsigned ND = 0;
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
      OperandList[i] = Nodes[i];
      SDNode *N = OperandList[i].Val;
      N->Uses.push_back(this);
      if (ND < N->getNodeDepth()) ND = N->getNodeDepth();
    }
    NodeDepth = ND+1;
    ValueList = 0;
    NumValues = 0;
    Prev = 0; Next = 0;
  }

  /// MorphNodeTo - This clears the return value and operands list, and sets the
  /// opcode of the node to the specified value.  This should only be used by
  /// the SelectionDAG class.
  void MorphNodeTo(unsigned Opc) {
    NodeType = Opc;
    ValueList = 0;
    NumValues = 0;
    
    // Clear the operands list, updating used nodes to remove this from their
    // use list.
    for (op_iterator I = op_begin(), E = op_end(); I != E; ++I)
      I->Val->removeUser(this);
    delete [] OperandList;
    OperandList = 0;
    NumOperands = 0;
  }
  
  void setValueTypes(MVT::ValueType VT) {
    assert(NumValues == 0 && "Should not have values yet!");
    ValueList = getValueTypeList(VT);
    NumValues = 1;
  }
  void setValueTypes(MVT::ValueType *List, unsigned NumVal) {
    assert(NumValues == 0 && "Should not have values yet!");
    ValueList = List;
    NumValues = NumVal;
  }
  
  void setOperands(SDOperand Op0) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[1];
    OperandList[0] = Op0;
    NumOperands = 1;
    Op0.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[2];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    NumOperands = 2;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[3];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    NumOperands = 3;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[4];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    NumOperands = 4;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[5];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    NumOperands = 5;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4, SDOperand Op5) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[6];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    OperandList[5] = Op5;
    NumOperands = 6;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this); Op5.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4, SDOperand Op5, SDOperand Op6) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[7];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    OperandList[5] = Op5;
    OperandList[6] = Op6;
    NumOperands = 7;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this); Op5.Val->Uses.push_back(this);
    Op6.Val->Uses.push_back(this);
  }
  void setOperands(SDOperand Op0, SDOperand Op1, SDOperand Op2, SDOperand Op3,
                   SDOperand Op4, SDOperand Op5, SDOperand Op6, SDOperand Op7) {
    assert(NumOperands == 0 && "Should not have operands yet!");
    OperandList = new SDOperand[8];
    OperandList[0] = Op0;
    OperandList[1] = Op1;
    OperandList[2] = Op2;
    OperandList[3] = Op3;
    OperandList[4] = Op4;
    OperandList[5] = Op5;
    OperandList[6] = Op6;
    OperandList[7] = Op7;
    NumOperands = 8;
    Op0.Val->Uses.push_back(this); Op1.Val->Uses.push_back(this);
    Op2.Val->Uses.push_back(this); Op3.Val->Uses.push_back(this);
    Op4.Val->Uses.push_back(this); Op5.Val->Uses.push_back(this);
    Op6.Val->Uses.push_back(this); Op7.Val->Uses.push_back(this);
  }

  void addUser(SDNode *User) {
    Uses.push_back(User);
  }
  void removeUser(SDNode *User) {
    // Remove this user from the operand's use list.
    for (unsigned i = Uses.size(); ; --i) {
      assert(i != 0 && "Didn't find user!");
      if (Uses[i-1] == User) {
        Uses[i-1] = Uses.back();
        Uses.pop_back();
        return;
      }
    }
  }
};


// Define inline functions from the SDOperand class.

inline unsigned SDOperand::getOpcode() const {
  return Val->getOpcode();
}
inline unsigned SDOperand::getNodeDepth() const {
  return Val->getNodeDepth();
}
inline MVT::ValueType SDOperand::getValueType() const {
  return Val->getValueType(ResNo);
}
inline unsigned SDOperand::getNumOperands() const {
  return Val->getNumOperands();
}
inline const SDOperand &SDOperand::getOperand(unsigned i) const {
  return Val->getOperand(i);
}
inline bool SDOperand::isTargetOpcode() const {
  return Val->isTargetOpcode();
}
inline unsigned SDOperand::getTargetOpcode() const {
  return Val->getTargetOpcode();
}
inline bool SDOperand::hasOneUse() const {
  return Val->hasNUsesOfValue(1, ResNo);
}

/// HandleSDNode - This class is used to form a handle around another node that
/// is persistant and is updated across invocations of replaceAllUsesWith on its
/// operand.  This node should be directly created by end-users and not added to
/// the AllNodes list.
class HandleSDNode : public SDNode {
public:
  HandleSDNode(SDOperand X) : SDNode(ISD::HANDLENODE, X) {}
  ~HandleSDNode() {
    MorphNodeTo(ISD::HANDLENODE);  // Drops operand uses.
  }
  
  SDOperand getValue() const { return getOperand(0); }
};

class StringSDNode : public SDNode {
  std::string Value;
protected:
  friend class SelectionDAG;
  StringSDNode(const std::string &val)
    : SDNode(ISD::STRING, MVT::Other), Value(val) {
  }
public:
  const std::string &getValue() const { return Value; }
  static bool classof(const StringSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::STRING;
  }
};  

class ConstantSDNode : public SDNode {
  uint64_t Value;
protected:
  friend class SelectionDAG;
  ConstantSDNode(bool isTarget, uint64_t val, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetConstant : ISD::Constant, VT), Value(val) {
  }
public:

  uint64_t getValue() const { return Value; }

  int64_t getSignExtended() const {
    unsigned Bits = MVT::getSizeInBits(getValueType(0));
    return ((int64_t)Value << (64-Bits)) >> (64-Bits);
  }

  bool isNullValue() const { return Value == 0; }
  bool isAllOnesValue() const {
    int NumBits = MVT::getSizeInBits(getValueType(0));
    if (NumBits == 64) return Value+1 == 0;
    return Value == (1ULL << NumBits)-1;
  }

  static bool classof(const ConstantSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Constant ||
           N->getOpcode() == ISD::TargetConstant;
  }
};

class ConstantFPSDNode : public SDNode {
  double Value;
protected:
  friend class SelectionDAG;
  ConstantFPSDNode(bool isTarget, double val, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetConstantFP : ISD::ConstantFP, VT), 
      Value(val) {
  }
public:

  double getValue() const { return Value; }

  /// isExactlyValue - We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.
  bool isExactlyValue(double V) const;

  static bool classof(const ConstantFPSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantFP || 
           N->getOpcode() == ISD::TargetConstantFP;
  }
};

class GlobalAddressSDNode : public SDNode {
  GlobalValue *TheGlobal;
  int offset;
protected:
  friend class SelectionDAG;
  GlobalAddressSDNode(bool isTarget, const GlobalValue *GA, MVT::ValueType VT,
                      int o=0)
    : SDNode(isTarget ? ISD::TargetGlobalAddress : ISD::GlobalAddress, VT) {
    TheGlobal = const_cast<GlobalValue*>(GA);
    offset = o;
  }
public:

  GlobalValue *getGlobal() const { return TheGlobal; }
  int getOffset() const { return offset; }

  static bool classof(const GlobalAddressSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::GlobalAddress ||
           N->getOpcode() == ISD::TargetGlobalAddress;
  }
};


class FrameIndexSDNode : public SDNode {
  int FI;
protected:
  friend class SelectionDAG;
  FrameIndexSDNode(int fi, MVT::ValueType VT, bool isTarg)
    : SDNode(isTarg ? ISD::TargetFrameIndex : ISD::FrameIndex, VT), FI(fi) {}
public:

  int getIndex() const { return FI; }

  static bool classof(const FrameIndexSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::FrameIndex ||
           N->getOpcode() == ISD::TargetFrameIndex;
  }
};

class ConstantPoolSDNode : public SDNode {
  Constant *C;
protected:
  friend class SelectionDAG;
  ConstantPoolSDNode(Constant *c, MVT::ValueType VT, bool isTarget)
    : SDNode(isTarget ? ISD::TargetConstantPool : ISD::ConstantPool, VT),
    C(c) {}
public:

  Constant *get() const { return C; }

  static bool classof(const ConstantPoolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ConstantPool ||
           N->getOpcode() == ISD::TargetConstantPool;
  }
};

class BasicBlockSDNode : public SDNode {
  MachineBasicBlock *MBB;
protected:
  friend class SelectionDAG;
  BasicBlockSDNode(MachineBasicBlock *mbb)
    : SDNode(ISD::BasicBlock, MVT::Other), MBB(mbb) {}
public:

  MachineBasicBlock *getBasicBlock() const { return MBB; }

  static bool classof(const BasicBlockSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::BasicBlock;
  }
};

class SrcValueSDNode : public SDNode {
  const Value *V;
  int offset;
protected:
  friend class SelectionDAG;
  SrcValueSDNode(const Value* v, int o)
    : SDNode(ISD::SRCVALUE, MVT::Other), V(v), offset(o) {}

public:
  const Value *getValue() const { return V; }
  int getOffset() const { return offset; }

  static bool classof(const SrcValueSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::SRCVALUE;
  }
};


class RegisterSDNode : public SDNode {
  unsigned Reg;
protected:
  friend class SelectionDAG;
  RegisterSDNode(unsigned reg, MVT::ValueType VT)
    : SDNode(ISD::Register, VT), Reg(reg) {}
public:

  unsigned getReg() const { return Reg; }

  static bool classof(const RegisterSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::Register;
  }
};

class ExternalSymbolSDNode : public SDNode {
  const char *Symbol;
protected:
  friend class SelectionDAG;
  ExternalSymbolSDNode(bool isTarget, const char *Sym, MVT::ValueType VT)
    : SDNode(isTarget ? ISD::TargetExternalSymbol : ISD::ExternalSymbol, VT),
      Symbol(Sym) {
    }
public:

  const char *getSymbol() const { return Symbol; }

  static bool classof(const ExternalSymbolSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::ExternalSymbol ||
           N->getOpcode() == ISD::TargetExternalSymbol;
  }
};

class CondCodeSDNode : public SDNode {
  ISD::CondCode Condition;
protected:
  friend class SelectionDAG;
  CondCodeSDNode(ISD::CondCode Cond)
    : SDNode(ISD::CONDCODE, MVT::Other), Condition(Cond) {
  }
public:

  ISD::CondCode get() const { return Condition; }

  static bool classof(const CondCodeSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::CONDCODE;
  }
};

/// VTSDNode - This class is used to represent MVT::ValueType's, which are used
/// to parameterize some operations.
class VTSDNode : public SDNode {
  MVT::ValueType ValueType;
protected:
  friend class SelectionDAG;
  VTSDNode(MVT::ValueType VT)
    : SDNode(ISD::VALUETYPE, MVT::Other), ValueType(VT) {}
public:

  MVT::ValueType getVT() const { return ValueType; }

  static bool classof(const VTSDNode *) { return true; }
  static bool classof(const SDNode *N) {
    return N->getOpcode() == ISD::VALUETYPE;
  }
};


class SDNodeIterator : public forward_iterator<SDNode, ptrdiff_t> {
  SDNode *Node;
  unsigned Operand;

  SDNodeIterator(SDNode *N, unsigned Op) : Node(N), Operand(Op) {}
public:
  bool operator==(const SDNodeIterator& x) const {
    return Operand == x.Operand;
  }
  bool operator!=(const SDNodeIterator& x) const { return !operator==(x); }

  const SDNodeIterator &operator=(const SDNodeIterator &I) {
    assert(I.Node == Node && "Cannot assign iterators to two different nodes!");
    Operand = I.Operand;
    return *this;
  }

  pointer operator*() const {
    return Node->getOperand(Operand).Val;
  }
  pointer operator->() const { return operator*(); }

  SDNodeIterator& operator++() {                // Preincrement
    ++Operand;
    return *this;
  }
  SDNodeIterator operator++(int) { // Postincrement
    SDNodeIterator tmp = *this; ++*this; return tmp;
  }

  static SDNodeIterator begin(SDNode *N) { return SDNodeIterator(N, 0); }
  static SDNodeIterator end  (SDNode *N) {
    return SDNodeIterator(N, N->getNumOperands());
  }

  unsigned getOperand() const { return Operand; }
  const SDNode *getNode() const { return Node; }
};

template <> struct GraphTraits<SDNode*> {
  typedef SDNode NodeType;
  typedef SDNodeIterator ChildIteratorType;
  static inline NodeType *getEntryNode(SDNode *N) { return N; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return SDNodeIterator::begin(N);
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return SDNodeIterator::end(N);
  }
};

template<>
struct ilist_traits<SDNode> {
  static SDNode *getPrev(const SDNode *N) { return N->Prev; }
  static SDNode *getNext(const SDNode *N) { return N->Next; }
  
  static void setPrev(SDNode *N, SDNode *Prev) { N->Prev = Prev; }
  static void setNext(SDNode *N, SDNode *Next) { N->Next = Next; }
  
  static SDNode *createSentinel() {
    return new SDNode(ISD::EntryToken, MVT::Other);
  }
  static void destroySentinel(SDNode *N) { delete N; }
  //static SDNode *createNode(const SDNode &V) { return new SDNode(V); }
  
  
  void addNodeToList(SDNode *NTy) {}
  void removeNodeFromList(SDNode *NTy) {}
  void transferNodesFromList(iplist<SDNode, ilist_traits> &L2,
                             const ilist_iterator<SDNode> &X,
                             const ilist_iterator<SDNode> &Y) {}
};

} // end llvm namespace

#endif

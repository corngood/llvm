//===-- llvm/Target/InstrInfo.h - Target Instruction Information --*-C++-*-==//
//
// This file describes the target machine instructions to the code generator.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MACHINEINSTRINFO_H
#define LLVM_TARGET_MACHINEINSTRINFO_H

#include "Support/DataTypes.h"
#include <vector>

class MachineInstrDescriptor;
class MachineInstr;
class TargetMachine;
class Value;
class Instruction;
class Constant;
class Function;
class MachineCodeForInstruction;

//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef int MachineOpCode;
typedef unsigned InstrSchedClass;

const MachineOpCode INVALID_MACHINE_OPCODE = -1;


// Global variable holding an array of descriptors for machine instructions.
// The actual object needs to be created separately for each target machine.
// This variable is initialized and reset by class MachineInstrInfo.
// 
// FIXME: This should be a property of the target so that more than one target
// at a time can be active...
//
extern const MachineInstrDescriptor *TargetInstrDescriptors;


//---------------------------------------------------------------------------
// struct MachineInstrDescriptor:
//	Predefined information about each machine instruction.
//	Designed to initialized statically.
// 
// class MachineInstructionInfo
//	Interface to description of machine instructions
// 
//---------------------------------------------------------------------------

const unsigned	M_NOP_FLAG		= 1 << 0;
const unsigned	M_BRANCH_FLAG		= 1 << 1;
const unsigned	M_CALL_FLAG		= 1 << 2;
const unsigned	M_RET_FLAG		= 1 << 3;
const unsigned	M_ARITH_FLAG		= 1 << 4;
const unsigned	M_CC_FLAG		= 1 << 6;
const unsigned	M_LOGICAL_FLAG		= 1 << 6;
const unsigned	M_INT_FLAG		= 1 << 7;
const unsigned	M_FLOAT_FLAG		= 1 << 8;
const unsigned	M_CONDL_FLAG		= 1 << 9;
const unsigned	M_LOAD_FLAG		= 1 << 10;
const unsigned	M_PREFETCH_FLAG		= 1 << 11;
const unsigned	M_STORE_FLAG		= 1 << 12;
const unsigned	M_DUMMY_PHI_FLAG	= 1 << 13;
const unsigned  M_PSEUDO_FLAG           = 1 << 14;


struct MachineInstrDescriptor {
  const char *    Name;          // Assembly language mnemonic for the opcode.
  int             numOperands;   // Number of args; -1 if variable #args
  int             resultPos;     // Position of the result; -1 if no result
  unsigned        maxImmedConst; // Largest +ve constant in IMMMED field or 0.
  bool	          immedIsSignExtended; // Is IMMED field sign-extended? If so,
				 //   smallest -ve value is -(maxImmedConst+1).
  unsigned        numDelaySlots; // Number of delay slots after instruction
  unsigned        latency;	 // Latency in machine cycles
  InstrSchedClass schedClass;	 // enum  identifying instr sched class
  unsigned        iclass;	 // flags identifying machine instr class
};


class MachineInstrInfo {
  const MachineInstrDescriptor* desc;	// raw array to allow static init'n
  unsigned descSize;		// number of entries in the desc array
  unsigned numRealOpCodes;		// number of non-dummy op codes
  
  MachineInstrInfo(const MachineInstrInfo &); // DO NOT IMPLEMENT
  void operator=(const MachineInstrInfo &);   // DO NOT IMPLEMENT
public:
  MachineInstrInfo(const MachineInstrDescriptor *desc, unsigned descSize,
		   unsigned numRealOpCodes);
  virtual ~MachineInstrInfo();
  
  unsigned getNumRealOpCodes()  const { return numRealOpCodes; }
  unsigned getNumTotalOpCodes() const { return descSize; }
  
  /// get - Return the machine instruction descriptor that corresponds to the
  /// specified instruction opcode.
  ///
  const MachineInstrDescriptor& get(MachineOpCode opCode) const {
    assert(opCode >= 0 && opCode < (int)descSize);
    return desc[opCode];
  }

  const char *getName(MachineOpCode opCode) const {
    return get(opCode).Name;
  }
  
  int getNumOperands(MachineOpCode opCode) const {
    return get(opCode).numOperands;
  }
  
  int getResultPos(MachineOpCode opCode) const {
    return get(opCode).resultPos;
  }
  
  unsigned getNumDelaySlots(MachineOpCode opCode) const {
    return get(opCode).numDelaySlots;
  }
  
  InstrSchedClass getSchedClass(MachineOpCode opCode) const {
    return get(opCode).schedClass;
  }
  
  //
  // Query instruction class flags according to the machine-independent
  // flags listed above.
  // 
  unsigned getIClass(MachineOpCode opCode) const {
    return get(opCode).iclass;
  }
  bool isNop(MachineOpCode opCode) const {
    return get(opCode).iclass & M_NOP_FLAG;
  }
  bool isBranch(MachineOpCode opCode) const {
    return get(opCode).iclass & M_BRANCH_FLAG;
  }
  bool isCall(MachineOpCode opCode) const {
    return get(opCode).iclass & M_CALL_FLAG;
  }
  bool isReturn(MachineOpCode opCode) const {
    return get(opCode).iclass & M_RET_FLAG;
  }
  bool isControlFlow(MachineOpCode opCode) const {
    return get(opCode).iclass & M_BRANCH_FLAG
        || get(opCode).iclass & M_CALL_FLAG
        || get(opCode).iclass & M_RET_FLAG;
  }
  bool isArith(MachineOpCode opCode) const {
    return get(opCode).iclass & M_ARITH_FLAG;
  }
  bool isCCInstr(MachineOpCode opCode) const {
    return get(opCode).iclass & M_CC_FLAG;
  }
  bool isLogical(MachineOpCode opCode) const {
    return get(opCode).iclass & M_LOGICAL_FLAG;
  }
  bool isIntInstr(MachineOpCode opCode) const {
    return get(opCode).iclass & M_INT_FLAG;
  }
  bool isFloatInstr(MachineOpCode opCode) const {
    return get(opCode).iclass & M_FLOAT_FLAG;
  }
  bool isConditional(MachineOpCode opCode) const { 
    return get(opCode).iclass & M_CONDL_FLAG;
  }
  bool isLoad(MachineOpCode opCode) const {
    return get(opCode).iclass & M_LOAD_FLAG;
  }
  bool isPrefetch(MachineOpCode opCode) const {
    return get(opCode).iclass & M_PREFETCH_FLAG;
  }
  bool isLoadOrPrefetch(MachineOpCode opCode) const {
    return get(opCode).iclass & M_LOAD_FLAG
        || get(opCode).iclass & M_PREFETCH_FLAG;
  }
  bool isStore(MachineOpCode opCode) const {
    return get(opCode).iclass & M_STORE_FLAG;
  }
  bool isMemoryAccess(MachineOpCode opCode) const {
    return get(opCode).iclass & M_LOAD_FLAG
        || get(opCode).iclass & M_PREFETCH_FLAG
        || get(opCode).iclass & M_STORE_FLAG;
  }
  bool isDummyPhiInstr(const MachineOpCode opCode) const {
    return get(opCode).iclass & M_DUMMY_PHI_FLAG;
  }
  bool isPseudoInstr(const MachineOpCode opCode) const {
    return get(opCode).iclass & M_PSEUDO_FLAG;
  }

  // Check if an instruction can be issued before its operands are ready,
  // or if a subsequent instruction that uses its result can be issued
  // before the results are ready.
  // Default to true since most instructions on many architectures allow this.
  // 
  virtual bool hasOperandInterlock(MachineOpCode opCode) const {
    return true;
  }
  
  virtual bool hasResultInterlock(MachineOpCode opCode) const {
    return true;
  }
  
  // 
  // Latencies for individual instructions and instruction pairs
  // 
  virtual int minLatency(MachineOpCode opCode) const {
    return get(opCode).latency;
  }
  
  virtual int maxLatency(MachineOpCode opCode) const {
    return get(opCode).latency;
  }

  //
  // Which operand holds an immediate constant?  Returns -1 if none
  // 
  virtual int getImmedConstantPos(MachineOpCode opCode) const {
    return -1; // immediate position is machine specific, so say -1 == "none"
  }
  
  // Check if the specified constant fits in the immediate field
  // of this machine instruction
  // 
  virtual bool constantFitsInImmedField(MachineOpCode opCode,
					int64_t intValue) const;
  
  // Return the largest +ve constant that can be held in the IMMMED field
  // of this machine instruction.
  // isSignExtended is set to true if the value is sign-extended before use
  // (this is true for all immediate fields in SPARC instructions).
  // Return 0 if the instruction has no IMMED field.
  // 
  virtual uint64_t maxImmedConstant(MachineOpCode opCode,
				    bool &isSignExtended) const {
    isSignExtended = get(opCode).immedIsSignExtended;
    return get(opCode).maxImmedConst;
  }

  //-------------------------------------------------------------------------
  // Queries about representation of LLVM quantities (e.g., constants)
  //-------------------------------------------------------------------------

  /// ConstantTypeMustBeLoaded - Test if this type of constant must be loaded
  /// from memory into a register, i.e., cannot be set bitwise in register and
  /// cannot use immediate fields of instructions.  Note that this only makes
  /// sense for primitive types.
  ///
  virtual bool ConstantTypeMustBeLoaded(const Constant* CV) const;

  // Test if this constant may not fit in the immediate field of the
  // machine instructions (probably) generated for this instruction.
  // 
  virtual bool ConstantMayNotFitInImmedField(const Constant* CV,
                                             const Instruction* I) const {
    return true;                        // safe but very conservative
  }

  //-------------------------------------------------------------------------
  // Code generation support for creating individual machine instructions
  //-------------------------------------------------------------------------

  // Get certain common op codes for the current target.  this and all the
  // Create* methods below should be moved to a machine code generation class
  // 
  virtual MachineOpCode getNOPOpCode() const = 0;

  // Create an instruction sequence to put the constant `val' into
  // the virtual register `dest'.  `val' may be a Constant or a
  // GlobalValue, viz., the constant address of a global variable or function.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Symbolic constants or constants that must be accessed from memory
  // are added to the constant pool via MachineFunction::get(F).
  // 
  virtual void  CreateCodeToLoadConst(const TargetMachine& target,
                                      Function* F,
                                      Value* val,
                                      Instruction* dest,
                                      std::vector<MachineInstr*>& mvec,
                                      MachineCodeForInstruction& mcfi) const=0;
  
  // Create an instruction sequence to copy an integer value `val'
  // to a floating point value `dest' by copying to memory and back.
  // val must be an integral type.  dest must be a Float or Double.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyIntToFloat(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi)const=0;

  // Similarly, create an instruction sequence to copy an FP value
  // `val' to an integer value `dest' by copying to memory and back.
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void  CreateCodeToCopyFloatToInt(const TargetMachine& target,
                                       Function* F,
                                       Value* val,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi)const=0;
  
  // Create instruction(s) to copy src to dest, for arbitrary types
  // The generated instructions are returned in `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateCopyInstructionsByType(const TargetMachine& target,
                                       Function* F,
                                       Value* src,
                                       Instruction* dest,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi)const=0;

  // Create instruction sequence to produce a sign-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateSignExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       Value* destVal,
                                       unsigned numLowBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const=0;

  // Create instruction sequence to produce a zero-extended register value
  // from an arbitrary sized value (sized in bits, not bytes).
  // The generated instructions are appended to `mvec'.
  // Any temp. registers (TmpInstruction) created are recorded in mcfi.
  // Any stack space required is allocated via mcff.
  // 
  virtual void CreateZeroExtensionInstructions(const TargetMachine& target,
                                       Function* F,
                                       Value* srcVal,
                                       Value* destVal,
                                       unsigned srcSizeInBits,
                                       std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const=0;
};

#endif

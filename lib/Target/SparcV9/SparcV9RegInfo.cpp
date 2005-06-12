//===-- SparcV9RegInfo.cpp - SparcV9 Target Register Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains implementations of SparcV9 specific helper methods
// used for register allocation.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "MachineFunctionInfo.h"
#include "MachineCodeForInstruction.h"
#include "MachineInstrAnnot.h"
#include "RegAlloc/LiveRangeInfo.h"
#include "RegAlloc/LiveRange.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "SparcV9Internals.h"
#include "SparcV9RegClassInfo.h"
#include "SparcV9RegInfo.h"
#include "SparcV9FrameInfo.h"
#include "SparcV9TargetMachine.h"
#include "SparcV9TmpInstr.h"
#include <iostream>

namespace llvm {

enum {
  BadRegClass = ~0
};

SparcV9RegInfo::SparcV9RegInfo(const SparcV9TargetMachine &tgt)
  : target (tgt), NumOfIntArgRegs (6), NumOfFloatArgRegs (32)
{
  MachineRegClassArr.push_back(new SparcV9IntRegClass(IntRegClassID));
  MachineRegClassArr.push_back(new SparcV9FloatRegClass(FloatRegClassID));
  MachineRegClassArr.push_back(new SparcV9IntCCRegClass(IntCCRegClassID));
  MachineRegClassArr.push_back(new SparcV9FloatCCRegClass(FloatCCRegClassID));
  MachineRegClassArr.push_back(new SparcV9SpecialRegClass(SpecialRegClassID));

  assert(SparcV9FloatRegClass::StartOfNonVolatileRegs == 32 &&
         "32 Float regs are used for float arg passing");
}


// getZeroRegNum - returns the register that contains always zero.
// this is the unified register number
//
unsigned SparcV9RegInfo::getZeroRegNum() const {
  return getUnifiedRegNum(SparcV9RegInfo::IntRegClassID,
                          SparcV9IntRegClass::g0);
}

// getCallAddressReg - returns the reg used for pushing the address when a
// method is called. This can be used for other purposes between calls
//
unsigned SparcV9RegInfo::getCallAddressReg() const {
  return getUnifiedRegNum(SparcV9RegInfo::IntRegClassID,
                          SparcV9IntRegClass::o7);
}

// Returns the register containing the return address.
// It should be made sure that this  register contains the return
// value when a return instruction is reached.
//
unsigned SparcV9RegInfo::getReturnAddressReg() const {
  return getUnifiedRegNum(SparcV9RegInfo::IntRegClassID,
                          SparcV9IntRegClass::i7);
}

// Register get name implementations...

// Int register names in same order as enum in class SparcV9IntRegClass
static const char * const IntRegNames[] = {
  "o0", "o1", "o2", "o3", "o4", "o5",       "o7",
  "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7",
  "i0", "i1", "i2", "i3", "i4", "i5",
  "i6", "i7",
  "g0", "g1", "g2", "g3", "g4", "g5",  "g6", "g7",
  "o6"
};

const char * const SparcV9IntRegClass::getRegName(unsigned reg) const {
  assert(reg < NumOfAllRegs);
  return IntRegNames[reg];
}

static const char * const FloatRegNames[] = {
  "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",  "f8",  "f9",
  "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19",
  "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29",
  "f30", "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39",
  "f40", "f41", "f42", "f43", "f44", "f45", "f46", "f47", "f48", "f49",
  "f50", "f51", "f52", "f53", "f54", "f55", "f56", "f57", "f58", "f59",
  "f60", "f61", "f62", "f63"
};

const char * const SparcV9FloatRegClass::getRegName(unsigned reg) const {
  assert (reg < NumOfAllRegs);
  return FloatRegNames[reg];
}

static const char * const IntCCRegNames[] = {
  "xcc",  "icc",  "ccr"
};

const char * const SparcV9IntCCRegClass::getRegName(unsigned reg) const {
  assert(reg < 3);
  return IntCCRegNames[reg];
}

static const char * const FloatCCRegNames[] = {
  "fcc0", "fcc1",  "fcc2",  "fcc3"
};

const char * const SparcV9FloatCCRegClass::getRegName(unsigned reg) const {
  assert (reg < 4);
  return FloatCCRegNames[reg];
}

static const char * const SpecialRegNames[] = {
  "fsr"
};

const char * const SparcV9SpecialRegClass::getRegName(unsigned reg) const {
  assert (reg < 1);
  return SpecialRegNames[reg];
}

// Get unified reg number for frame pointer
unsigned SparcV9RegInfo::getFramePointer() const {
  return getUnifiedRegNum(SparcV9RegInfo::IntRegClassID,
                          SparcV9IntRegClass::i6);
}

// Get unified reg number for stack pointer
unsigned SparcV9RegInfo::getStackPointer() const {
  return getUnifiedRegNum(SparcV9RegInfo::IntRegClassID,
                          SparcV9IntRegClass::o6);
}


//---------------------------------------------------------------------------
// Finds whether a call is an indirect call
//---------------------------------------------------------------------------

inline bool
isVarArgsFunction(const Type *funcType) {
  return cast<FunctionType>(cast<PointerType>(funcType)
                            ->getElementType())->isVarArg();
}

inline bool
isVarArgsCall(const MachineInstr *CallMI) {
  Value* callee = CallMI->getOperand(0).getVRegValue();
  // const Type* funcType = isa<Function>(callee)? callee->getType()
  //   : cast<PointerType>(callee->getType())->getElementType();
  const Type* funcType = callee->getType();
  return isVarArgsFunction(funcType);
}


// Get the register number for the specified argument #argNo,
//
// Return value:
//      getInvalidRegNum(),  if there is no int register available for the arg.
//      regNum,              otherwise (this is NOT the unified reg. num).
//                           regClassId is set to the register class ID.
//
int
SparcV9RegInfo::regNumForIntArg(bool inCallee, bool isVarArgsCall,
                                   unsigned argNo, unsigned& regClassId) const
{
  regClassId = IntRegClassID;
  if (argNo >= NumOfIntArgRegs)
    return getInvalidRegNum();
  else
    return argNo + (inCallee? SparcV9IntRegClass::i0 : SparcV9IntRegClass::o0);
}

// Get the register number for the specified FP argument #argNo,
// Use INT regs for FP args if this is a varargs call.
//
// Return value:
//      getInvalidRegNum(),  if there is no int register available for the arg.
//      regNum,              otherwise (this is NOT the unified reg. num).
//                           regClassId is set to the register class ID.
//
int
SparcV9RegInfo::regNumForFPArg(unsigned regType,
                                  bool inCallee, bool isVarArgsCall,
                                  unsigned argNo, unsigned& regClassId) const
{
  if (isVarArgsCall)
    return regNumForIntArg(inCallee, isVarArgsCall, argNo, regClassId);
  else
    {
      regClassId = FloatRegClassID;
      if (regType == FPSingleRegType)
        return (argNo*2+1 >= NumOfFloatArgRegs)?
          getInvalidRegNum() : SparcV9FloatRegClass::f0 + (argNo * 2 + 1);
      else if (regType == FPDoubleRegType)
        return (argNo*2 >= NumOfFloatArgRegs)?
          getInvalidRegNum() : SparcV9FloatRegClass::f0 + (argNo * 2);
      else
        assert(0 && "Illegal FP register type");
	return 0;
    }
}


//---------------------------------------------------------------------------
// Finds the return address of a call sparc specific call instruction
//---------------------------------------------------------------------------

// The following 4  methods are used to find the RegType (SparcV9Internals.h)
// of a V9LiveRange, a Value, and for a given register unified reg number.
//
int SparcV9RegInfo::getRegTypeForClassAndType(unsigned regClassID,
                                                 const Type* type) const
{
  switch (regClassID) {
  case IntRegClassID:                   return IntRegType;
  case FloatRegClassID:
    if (type == Type::FloatTy)          return FPSingleRegType;
    else if (type == Type::DoubleTy)    return FPDoubleRegType;
    assert(0 && "Unknown type in FloatRegClass"); return 0;
  case IntCCRegClassID:                 return IntCCRegType;
  case FloatCCRegClassID:               return FloatCCRegType;
  case SpecialRegClassID:               return SpecialRegType;
  default: assert( 0 && "Unknown reg class ID"); return 0;
  }
}

int SparcV9RegInfo::getRegTypeForDataType(const Type* type) const
{
  return getRegTypeForClassAndType(getRegClassIDOfType(type), type);
}

int SparcV9RegInfo::getRegTypeForLR(const V9LiveRange *LR) const
{
  return getRegTypeForClassAndType(LR->getRegClassID(), LR->getType());
}

int SparcV9RegInfo::getRegType(int unifiedRegNum) const
{
  if (unifiedRegNum < 32)
    return IntRegType;
  else if (unifiedRegNum < (32 + 32))
    return FPSingleRegType;
  else if (unifiedRegNum < (64 + 32))
    return FPDoubleRegType;
  else if (unifiedRegNum < (64+32+3))
    return IntCCRegType;
  else if (unifiedRegNum < (64+32+3+4))
    return FloatCCRegType;
  else if (unifiedRegNum < (64+32+3+4+1))
    return SpecialRegType;
  else
    assert(0 && "Invalid unified register number in getRegType");
  return 0;
}


// To find the register class used for a specified Type
//
unsigned SparcV9RegInfo::getRegClassIDOfType(const Type *type,
                                                bool isCCReg) const {
  Type::TypeID ty = type->getTypeID();
  unsigned res;

  // FIXME: Comparing types like this isn't very safe...
  if ((ty && ty <= Type::LongTyID) || (ty == Type::LabelTyID) ||
      (ty == Type::FunctionTyID) ||  (ty == Type::PointerTyID) )
    res = IntRegClassID;             // sparc int reg (ty=0: void)
  else if (ty <= Type::DoubleTyID)
    res = FloatRegClassID;           // sparc float reg class
  else {
    //std::cerr << "TypeID: " << ty << "\n";
    assert(0 && "Cannot resolve register class for type");
    return 0;
  }

  if (isCCReg)
    return res + 2;      // corresponding condition code register
  else
    return res;
}

unsigned SparcV9RegInfo::getRegClassIDOfRegType(int regType) const {
  switch(regType) {
  case IntRegType:      return IntRegClassID;
  case FPSingleRegType:
  case FPDoubleRegType: return FloatRegClassID;
  case IntCCRegType:    return IntCCRegClassID;
  case FloatCCRegType:  return FloatCCRegClassID;
  case SpecialRegType:  return SpecialRegClassID;
  default:
    assert(0 && "Invalid register type in getRegClassIDOfRegType");
    return 0;
  }
}

//---------------------------------------------------------------------------
// Suggests a register for the ret address in the RET machine instruction.
// We always suggest %i7 by convention.
//---------------------------------------------------------------------------
void SparcV9RegInfo::suggestReg4RetAddr(MachineInstr *RetMI,
					   LiveRangeInfo& LRI) const {

  assert(target.getInstrInfo()->isReturn(RetMI->getOpcode()));

  // return address is always mapped to i7 so set it immediately
  RetMI->SetRegForOperand(0, getUnifiedRegNum(IntRegClassID,
                                              SparcV9IntRegClass::i7));

  // Possible Optimization:
  // Instead of setting the color, we can suggest one. In that case,
  // we have to test later whether it received the suggested color.
  // In that case, a LR has to be created at the start of method.
  // It has to be done as follows (remove the setRegVal above):

  // MachineOperand & MO  = RetMI->getOperand(0);
  // const Value *RetAddrVal = MO.getVRegValue();
  // assert( RetAddrVal && "LR for ret address must be created at start");
  // V9LiveRange * RetAddrLR = LRI.getLiveRangeForValue( RetAddrVal);
  // RetAddrLR->setSuggestedColor(getUnifiedRegNum( IntRegClassID,
  //                              SparcV9IntRegOrdr::i7) );
}


//---------------------------------------------------------------------------
// Suggests a register for the ret address in the JMPL/CALL machine instr.
// SparcV9 ABI dictates that %o7 be used for this purpose.
//---------------------------------------------------------------------------
void
SparcV9RegInfo::suggestReg4CallAddr(MachineInstr * CallMI,
                                       LiveRangeInfo& LRI) const
{
  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(CallMI);
  const Value *RetAddrVal = argDesc->getReturnAddrReg();
  assert(RetAddrVal && "INTERNAL ERROR: Return address value is required");

  // A LR must already exist for the return address.
  V9LiveRange *RetAddrLR = LRI.getLiveRangeForValue(RetAddrVal);
  assert(RetAddrLR && "INTERNAL ERROR: No LR for return address of call!");

  unsigned RegClassID = RetAddrLR->getRegClassID();
  RetAddrLR->setColor(getUnifiedRegNum(IntRegClassID, SparcV9IntRegClass::o7));
}



//---------------------------------------------------------------------------
//  This method will suggest colors to incoming args to a method.
//  According to the SparcV9 ABI, the first 6 incoming args are in
//  %i0 - %i5 (if they are integer) OR in %f0 - %f31 (if they are float).
//  If the arg is passed on stack due to the lack of regs, NOTHING will be
//  done - it will be colored (or spilled) as a normal live range.
//---------------------------------------------------------------------------
void SparcV9RegInfo::suggestRegs4MethodArgs(const Function *Meth,
					       LiveRangeInfo& LRI) const
{
  // Check if this is a varArgs function. needed for choosing regs.
  bool isVarArgs = isVarArgsFunction(Meth->getType());

  // Count the arguments, *ignoring* whether they are int or FP args.
  // Use this common arg numbering to pick the right int or fp register.
  unsigned argNo=0;
  for(Function::const_arg_iterator I = Meth->arg_begin(), E = Meth->arg_end();
      I != E; ++I, ++argNo) {
    V9LiveRange *LR = LRI.getLiveRangeForValue(I);
    assert(LR && "No live range found for method arg");

    unsigned regType = getRegTypeForLR(LR);
    unsigned regClassIDOfArgReg = BadRegClass; // for chosen reg (unused)

    int regNum = (regType == IntRegType)
      ? regNumForIntArg(/*inCallee*/ true, isVarArgs, argNo, regClassIDOfArgReg)
      : regNumForFPArg(regType, /*inCallee*/ true, isVarArgs, argNo,
                       regClassIDOfArgReg);

    if (regNum != getInvalidRegNum())
      LR->setSuggestedColor(regNum);
  }
}


//---------------------------------------------------------------------------
// This method is called after graph coloring to move incoming args to
// the correct hardware registers if they did not receive the correct
// (suggested) color through graph coloring.
//---------------------------------------------------------------------------
void SparcV9RegInfo::colorMethodArgs(const Function *Meth,
                            LiveRangeInfo &LRI,
                            std::vector<MachineInstr*>& InstrnsBefore,
                            std::vector<MachineInstr*>& InstrnsAfter) const {

  // check if this is a varArgs function. needed for choosing regs.
  bool isVarArgs = isVarArgsFunction(Meth->getType());
  MachineInstr *AdMI;

  // for each argument
  // for each argument.  count INT and FP arguments separately.
  unsigned argNo=0, intArgNo=0, fpArgNo=0;
  for(Function::const_arg_iterator I = Meth->arg_begin(), E = Meth->arg_end();
      I != E; ++I, ++argNo) {
    // get the LR of arg
    V9LiveRange *LR = LRI.getLiveRangeForValue(I);
    assert( LR && "No live range found for method arg");

    unsigned regType = getRegTypeForLR(LR);
    unsigned RegClassID = LR->getRegClassID();

    // Find whether this argument is coming in a register (if not, on stack)
    // Also find the correct register the argument must use (UniArgReg)
    //
    bool isArgInReg = false;
    unsigned UniArgReg = getInvalidRegNum(); // reg that LR MUST be colored with
    unsigned regClassIDOfArgReg = BadRegClass; // reg class of chosen reg

    int regNum = (regType == IntRegType)
      ? regNumForIntArg(/*inCallee*/ true, isVarArgs,
                        argNo, regClassIDOfArgReg)
      : regNumForFPArg(regType, /*inCallee*/ true, isVarArgs,
                       argNo, regClassIDOfArgReg);

    if(regNum != getInvalidRegNum()) {
      isArgInReg = true;
      UniArgReg = getUnifiedRegNum( regClassIDOfArgReg, regNum);
    }

    if( ! LR->isMarkedForSpill() ) {    // if this arg received a register

      unsigned UniLRReg = getUnifiedRegNum(  RegClassID, LR->getColor() );

      // if LR received the correct color, nothing to do
      //
      if( UniLRReg == UniArgReg )
	continue;

      // We are here because the LR did not receive the suggested
      // but LR received another register.
      // Now we have to copy the %i reg (or stack pos of arg)
      // to the register the LR was colored with.

      // if the arg is coming in UniArgReg register, it MUST go into
      // the UniLRReg register
      //
      if( isArgInReg ) {
	if( regClassIDOfArgReg != RegClassID ) {
	  // NOTE: This code has not been well-tested.

	  // It is a variable argument call: the float reg must go in a %o reg.
	  // We have to move an int reg to a float reg via memory.
          //
          assert(isVarArgs &&
                 RegClassID == FloatRegClassID &&
                 regClassIDOfArgReg == IntRegClassID &&
                 "This should only be an Int register for an FP argument");

 	  int TmpOff = MachineFunction::get(Meth).getInfo<SparcV9FunctionInfo>()->pushTempValue(
                                                getSpilledRegSize(regType));
	  cpReg2MemMI(InstrnsBefore,
                      UniArgReg, getFramePointer(), TmpOff, IntRegType);

	  cpMem2RegMI(InstrnsBefore,
                      getFramePointer(), TmpOff, UniLRReg, regType);
	}
	else {	
	  cpReg2RegMI(InstrnsBefore, UniArgReg, UniLRReg, regType);
	}
      }
      else {

	// Now the arg is coming on stack. Since the LR received a register,
	// we just have to load the arg on stack into that register
	//
        const TargetFrameInfo& frameInfo = *target.getFrameInfo();
	int offsetFromFP =
          frameInfo.getIncomingArgOffset(MachineFunction::get(Meth),
                                         argNo);

        // float arguments on stack are right justified so adjust the offset!
        // int arguments are also right justified but they are always loaded as
        // a full double-word so the offset does not need to be adjusted.
        if (regType == FPSingleRegType) {
          unsigned argSize = target.getTargetData().getTypeSize(LR->getType());
          unsigned slotSize = SparcV9FrameInfo::SizeOfEachArgOnStack;
          assert(argSize <= slotSize && "Insufficient slot size!");
          offsetFromFP += slotSize - argSize;
        }

	cpMem2RegMI(InstrnsBefore,
                    getFramePointer(), offsetFromFP, UniLRReg, regType);
      }

    } // if LR received a color

    else {

      // Now, the LR did not receive a color. But it has a stack offset for
      // spilling.
      // So, if the arg is coming in UniArgReg register,  we can just move
      // that on to the stack pos of LR

      if( isArgInReg ) {

	if( regClassIDOfArgReg != RegClassID ) {
          assert(0 &&
                 "FP arguments to a varargs function should be explicitly "
                 "copied to/from int registers by instruction selection!");

	  // It must be a float arg for a variable argument call, which
          // must come in a %o reg.  Move the int reg to the stack.
          //
          assert(isVarArgs && regClassIDOfArgReg == IntRegClassID &&
                 "This should only be an Int register for an FP argument");

          cpReg2MemMI(InstrnsBefore, UniArgReg,
                      getFramePointer(), LR->getSpillOffFromFP(), IntRegType);
        }
        else {
           cpReg2MemMI(InstrnsBefore, UniArgReg,
                       getFramePointer(), LR->getSpillOffFromFP(), regType);
        }
      }

      else {

	// Now the arg is coming on stack. Since the LR did NOT
	// received a register as well, it is allocated a stack position. We
	// can simply change the stack position of the LR. We can do this,
	// since this method is called before any other method that makes
	// uses of the stack pos of the LR (e.g., updateMachineInstr)
        //
        const TargetFrameInfo& frameInfo = *target.getFrameInfo();
	int offsetFromFP =
          frameInfo.getIncomingArgOffset(MachineFunction::get(Meth),
                                         argNo);

        // FP arguments on stack are right justified so adjust offset!
        // int arguments are also right justified but they are always loaded as
        // a full double-word so the offset does not need to be adjusted.
        if (regType == FPSingleRegType) {
          unsigned argSize = target.getTargetData().getTypeSize(LR->getType());
          unsigned slotSize = SparcV9FrameInfo::SizeOfEachArgOnStack;
          assert(argSize <= slotSize && "Insufficient slot size!");
          offsetFromFP += slotSize - argSize;
        }

	LR->modifySpillOffFromFP( offsetFromFP );
      }

    }

  }  // for each incoming argument

}



//---------------------------------------------------------------------------
// This method is called before graph coloring to suggest colors to the
// outgoing call args and the return value of the call.
//---------------------------------------------------------------------------
void SparcV9RegInfo::suggestRegs4CallArgs(MachineInstr *CallMI,
					     LiveRangeInfo& LRI) const {
  assert ( (target.getInstrInfo())->isCall(CallMI->getOpcode()) );

  CallArgsDescriptor* argDesc = CallArgsDescriptor::get(CallMI);

  suggestReg4CallAddr(CallMI, LRI);

  // First color the return value of the call instruction, if any.
  // The return value will be in %o0 if the value is an integer type,
  // or in %f0 if the value is a float type.
  //
  if (const Value *RetVal = argDesc->getReturnValue()) {
    V9LiveRange *RetValLR = LRI.getLiveRangeForValue(RetVal);
    assert(RetValLR && "No LR for return Value of call!");

    unsigned RegClassID = RetValLR->getRegClassID();

    // now suggest a register depending on the register class of ret arg
    if( RegClassID == IntRegClassID )
      RetValLR->setSuggestedColor(SparcV9IntRegClass::o0);
    else if (RegClassID == FloatRegClassID )
      RetValLR->setSuggestedColor(SparcV9FloatRegClass::f0 );
    else assert( 0 && "Unknown reg class for return value of call\n");
  }

  // Now suggest colors for arguments (operands) of the call instruction.
  // Colors are suggested only if the arg number is smaller than the
  // the number of registers allocated for argument passing.
  // Now, go thru call args - implicit operands of the call MI

  unsigned NumOfCallArgs = argDesc->getNumArgs();

  for(unsigned argNo=0, i=0, intArgNo=0, fpArgNo=0;
       i < NumOfCallArgs; ++i, ++argNo) {

    const Value *CallArg = argDesc->getArgInfo(i).getArgVal();

    // get the LR of call operand (parameter)
    V9LiveRange *const LR = LRI.getLiveRangeForValue(CallArg);
    if (!LR)
      continue;                    // no live ranges for constants and labels

    unsigned regType = getRegTypeForLR(LR);
    unsigned regClassIDOfArgReg = BadRegClass; // chosen reg class (unused)

    // Choose a register for this arg depending on whether it is
    // an INT or FP value.  Here we ignore whether or not it is a
    // varargs calls, because FP arguments will be explicitly copied
    // to an integer Value and handled under (argCopy != NULL) below.
    int regNum = (regType == IntRegType)
      ? regNumForIntArg(/*inCallee*/ false, /*isVarArgs*/ false,
                        argNo, regClassIDOfArgReg)
      : regNumForFPArg(regType, /*inCallee*/ false, /*isVarArgs*/ false,
                       argNo, regClassIDOfArgReg);

    // If a register could be allocated, use it.
    // If not, do NOTHING as this will be colored as a normal value.
    if(regNum != getInvalidRegNum())
      LR->setSuggestedColor(regNum);
  } // for all call arguments
}


//---------------------------------------------------------------------------
// this method is called for an LLVM return instruction to identify which
// values will be returned from this method and to suggest colors.
//---------------------------------------------------------------------------
void SparcV9RegInfo::suggestReg4RetValue(MachineInstr *RetMI,
                                            LiveRangeInfo& LRI) const {

  assert( target.getInstrInfo()->isReturn( RetMI->getOpcode() ) );

  suggestReg4RetAddr(RetMI, LRI);

  // To find the return value (if any), we can get the LLVM return instr.
  // from the return address register, which is the first operand
  Value* tmpI = RetMI->getOperand(0).getVRegValue();
  ReturnInst* retI=cast<ReturnInst>(cast<TmpInstruction>(tmpI)->getOperand(0));
  if (const Value *RetVal = retI->getReturnValue())
    if (V9LiveRange *const LR = LRI.getLiveRangeForValue(RetVal))
      LR->setSuggestedColor(LR->getRegClassID() == IntRegClassID
                            ? (unsigned) SparcV9IntRegClass::i0
                            : (unsigned) SparcV9FloatRegClass::f0);
}

//---------------------------------------------------------------------------
// Check if a specified register type needs a scratch register to be
// copied to/from memory.  If it does, the reg. type that must be used
// for scratch registers is returned in scratchRegType.
//
// Only the int CC register needs such a scratch register.
// The FP CC registers can (and must) be copied directly to/from memory.
//---------------------------------------------------------------------------

bool
SparcV9RegInfo::regTypeNeedsScratchReg(int RegType,
                                          int& scratchRegType) const
{
  if (RegType == IntCCRegType)
    {
      scratchRegType = IntRegType;
      return true;
    }
  return false;
}

//---------------------------------------------------------------------------
// Copy from a register to register. Register number must be the unified
// register number.
//---------------------------------------------------------------------------

void
SparcV9RegInfo::cpReg2RegMI(std::vector<MachineInstr*>& mvec,
                               unsigned SrcReg,
                               unsigned DestReg,
                               int RegType) const {
  assert( ((int)SrcReg != getInvalidRegNum()) &&
          ((int)DestReg != getInvalidRegNum()) &&
	  "Invalid Register");

  MachineInstr * MI = NULL;

  switch( RegType ) {

  case IntCCRegType:
    if (getRegType(DestReg) == IntRegType) {
      // copy intCC reg to int reg
      MI = (BuildMI(V9::RDCCR, 2)
            .addMReg(getUnifiedRegNum(SparcV9RegInfo::IntCCRegClassID,
                                      SparcV9IntCCRegClass::ccr))
            .addMReg(DestReg,MachineOperand::Def));
    } else {
      // copy int reg to intCC reg
      assert(getRegType(SrcReg) == IntRegType
             && "Can only copy CC reg to/from integer reg");
      MI = (BuildMI(V9::WRCCRr, 3)
            .addMReg(SrcReg)
            .addMReg(SparcV9IntRegClass::g0)
            .addMReg(getUnifiedRegNum(SparcV9RegInfo::IntCCRegClassID,
                                      SparcV9IntCCRegClass::ccr),
                     MachineOperand::Def));
    }
    break;

  case FloatCCRegType:
    assert(0 && "Cannot copy FPCC register to any other register");
    break;

  case IntRegType:
    MI = BuildMI(V9::ADDr, 3).addMReg(SrcReg).addMReg(getZeroRegNum())
      .addMReg(DestReg, MachineOperand::Def);
    break;

  case FPSingleRegType:
    MI = BuildMI(V9::FMOVS, 2).addMReg(SrcReg)
           .addMReg(DestReg, MachineOperand::Def);
    break;

  case FPDoubleRegType:
    MI = BuildMI(V9::FMOVD, 2).addMReg(SrcReg)
           .addMReg(DestReg, MachineOperand::Def);
    break;

  default:
    assert(0 && "Unknown RegType");
    break;
  }

  if (MI)
    mvec.push_back(MI);
}

/// cpReg2MemMI - Generate SparcV9 MachineInstrs to store a register
/// (SrcReg) to memory, at [PtrReg + Offset].  Register numbers must be the
/// unified register numbers.  RegType must be the SparcV9 register type
/// of SrcReg. When SrcReg is %ccr, scratchReg must be the
/// number of a free integer register.  The newly-generated MachineInstrs
/// are appended to mvec.
///
void SparcV9RegInfo::cpReg2MemMI(std::vector<MachineInstr*>& mvec,
                                 unsigned SrcReg, unsigned PtrReg, int Offset,
                                 int RegType, int scratchReg) const {
  unsigned OffReg = SparcV9::g4; // Use register g4 for holding large offsets
  bool useImmediateOffset = true;

  // If the Offset will not fit in the signed-immediate field, we put it in
  // register g4. This takes advantage of the fact that all the opcodes
  // used below have the same size immed. field.
  if (RegType != IntCCRegType
      && !target.getInstrInfo()->constantFitsInImmedField(V9::LDXi, Offset)) {
    // Put the offset into a register. We could do this in fewer steps,
    // in some cases (see CreateSETSWConst()) but we're being lazy.
    MachineInstr *MI = BuildMI(V9::SETHI, 2).addZImm(Offset).addMReg(OffReg,
      MachineOperand::Def);
    MI->getOperand(0).markHi32();
    mvec.push_back(MI);
    MI = BuildMI(V9::ORi,3).addMReg(OffReg).addZImm(Offset).addMReg(OffReg,
      MachineOperand::Def);
    MI->getOperand(1).markLo32();
    mvec.push_back(MI);
    MI = BuildMI(V9::SRAi5,3).addMReg(OffReg).addZImm(0).addMReg(OffReg,
      MachineOperand::Def);
    mvec.push_back(MI);
    useImmediateOffset = false;
  }

  MachineInstr *MI = 0;
  switch (RegType) {
  case IntRegType:
    if (useImmediateOffset)
      MI = BuildMI(V9::STXi,3).addMReg(SrcReg).addMReg(PtrReg).addSImm(Offset);
    else
      MI = BuildMI(V9::STXr,3).addMReg(SrcReg).addMReg(PtrReg).addMReg(OffReg);
    break;

  case FPSingleRegType:
    if (useImmediateOffset)
      MI = BuildMI(V9::STFi, 3).addMReg(SrcReg).addMReg(PtrReg).addSImm(Offset);
    else
      MI = BuildMI(V9::STFr, 3).addMReg(SrcReg).addMReg(PtrReg).addMReg(OffReg);
    break;

  case FPDoubleRegType:
    if (useImmediateOffset)
      MI = BuildMI(V9::STDFi,3).addMReg(SrcReg).addMReg(PtrReg).addSImm(Offset);
    else
      MI = BuildMI(V9::STDFr,3).addMReg(SrcReg).addMReg(PtrReg).addSImm(OffReg);
    break;

  case IntCCRegType:
    assert(scratchReg >= 0 && getRegType(scratchReg) == IntRegType
           && "Need a scratch reg of integer type to load or store %ccr");
    MI = BuildMI(V9::RDCCR, 2).addMReg(SparcV9::ccr)
           .addMReg(scratchReg, MachineOperand::Def);
    mvec.push_back(MI);
    cpReg2MemMI(mvec, scratchReg, PtrReg, Offset, IntRegType);
    return;

  case SpecialRegType: // used only for %fsr itself.
  case FloatCCRegType: {
    if (useImmediateOffset)
      MI = BuildMI(V9::STXFSRi,3).addMReg(SparcV9::fsr).addMReg(PtrReg)
             .addSImm(Offset);
    else
      MI = BuildMI(V9::STXFSRr,3).addMReg(SparcV9::fsr).addMReg(PtrReg)
             .addMReg(OffReg);
    break;
  }
  default:
    assert(0 && "Unknown RegType in cpReg2MemMI");
  }
  mvec.push_back(MI);
}

/// cpMem2RegMI - Generate SparcV9 MachineInstrs to load a register
/// (DestReg) from memory, at [PtrReg + Offset].  Register numbers must be the
/// unified register numbers.  RegType must be the SparcV9 register type
/// of DestReg. When DestReg is %ccr, scratchReg must be the
/// number of a free integer register.  The newly-generated MachineInstrs
/// are appended to mvec.
///
void SparcV9RegInfo::cpMem2RegMI(std::vector<MachineInstr*>& mvec,
                                 unsigned PtrReg, int Offset, unsigned DestReg,
                                 int RegType, int scratchReg) const {
  unsigned OffReg = SparcV9::g4; // Use register g4 for holding large offsets
  bool useImmediateOffset = true;

  // If the Offset will not fit in the signed-immediate field, we put it in
  // register g4. This takes advantage of the fact that all the opcodes
  // used below have the same size immed. field.
  if (RegType != IntCCRegType
      && !target.getInstrInfo()->constantFitsInImmedField(V9::LDXi, Offset)) {
    MachineInstr *MI = BuildMI(V9::SETHI, 2).addZImm(Offset).addMReg(OffReg,
      MachineOperand::Def);
    MI->getOperand(0).markHi32();
    mvec.push_back(MI);
    MI = BuildMI(V9::ORi,3).addMReg(OffReg).addZImm(Offset).addMReg(OffReg,
      MachineOperand::Def);
    MI->getOperand(1).markLo32();
    mvec.push_back(MI);
    MI = BuildMI(V9::SRAi5,3).addMReg(OffReg).addZImm(0).addMReg(OffReg,
      MachineOperand::Def);
    mvec.push_back(MI);
    useImmediateOffset = false;
  }

  MachineInstr *MI = 0;
  switch (RegType) {
  case IntRegType:
    if (useImmediateOffset)
      MI = BuildMI(V9::LDXi, 3).addMReg(PtrReg).addSImm(Offset)
          .addMReg(DestReg, MachineOperand::Def);
    else
      MI = BuildMI(V9::LDXr, 3).addMReg(PtrReg).addMReg(OffReg)
          .addMReg(DestReg, MachineOperand::Def);
    break;

  case FPSingleRegType:
    if (useImmediateOffset)
      MI = BuildMI(V9::LDFi, 3).addMReg(PtrReg).addSImm(Offset)
          .addMReg(DestReg, MachineOperand::Def);
    else
      MI = BuildMI(V9::LDFr, 3).addMReg(PtrReg).addMReg(OffReg)
          .addMReg(DestReg, MachineOperand::Def);
    break;

  case FPDoubleRegType:
    if (useImmediateOffset)
      MI= BuildMI(V9::LDDFi, 3).addMReg(PtrReg).addSImm(Offset)
          .addMReg(DestReg, MachineOperand::Def);
    else
      MI= BuildMI(V9::LDDFr, 3).addMReg(PtrReg).addMReg(OffReg)
          .addMReg(DestReg, MachineOperand::Def);
    break;

  case IntCCRegType:
    assert(scratchReg >= 0 && getRegType(scratchReg) == IntRegType
           && "Need a scratch reg of integer type to load or store %ccr");
    cpMem2RegMI(mvec, PtrReg, Offset, scratchReg, IntRegType);
    MI = BuildMI(V9::WRCCRr, 3).addMReg(scratchReg).addMReg(SparcV9::g0)
           .addMReg(SparcV9::ccr, MachineOperand::Def);
    break;

  case SpecialRegType: // used only for %fsr itself
  case FloatCCRegType: {
    if (useImmediateOffset)
      MI = BuildMI(V9::LDXFSRi, 3).addMReg(PtrReg).addSImm(Offset)
        .addMReg(SparcV9::fsr, MachineOperand::Def);
    else
      MI = BuildMI(V9::LDXFSRr, 3).addMReg(PtrReg).addMReg(OffReg)
        .addMReg(SparcV9::fsr, MachineOperand::Def);
    break;
  }
  default:
    assert(0 && "Unknown RegType in cpMem2RegMI");
  }
  mvec.push_back(MI);
}


//---------------------------------------------------------------------------
// Generate a copy instruction to copy a value to another. Temporarily
// used by PhiElimination code.
//---------------------------------------------------------------------------


void
SparcV9RegInfo::cpValue2Value(Value *Src, Value *Dest,
                              std::vector<MachineInstr*>& mvec) const {
  int RegType = getRegTypeForDataType(Src->getType());
  MachineInstr * MI = NULL;

  switch (RegType) {
  case IntRegType:
    MI = BuildMI(V9::ADDr, 3).addReg(Src).addMReg(getZeroRegNum())
      .addRegDef(Dest);
    break;
  case FPSingleRegType:
    MI = BuildMI(V9::FMOVS, 2).addReg(Src).addRegDef(Dest);
    break;
  case FPDoubleRegType:
    MI = BuildMI(V9::FMOVD, 2).addReg(Src).addRegDef(Dest);
    break;
  default:
    assert(0 && "Unknown RegType in cpValue2Value");
  }

  mvec.push_back(MI);
}



//---------------------------------------------------------------------------
// Print the register assigned to a LR
//---------------------------------------------------------------------------

void SparcV9RegInfo::printReg(const V9LiveRange *LR) const {
  unsigned RegClassID = LR->getRegClassID();
  std::cerr << " Node ";

  if (!LR->hasColor()) {
    std::cerr << " - could not find a color\n";
    return;
  }

  // if a color is found

  std::cerr << " colored with color "<< LR->getColor();

  unsigned uRegName = getUnifiedRegNum(RegClassID, LR->getColor());

  std::cerr << "[";
  std::cerr<< getUnifiedRegName(uRegName);
  if (RegClassID == FloatRegClassID && LR->getType() == Type::DoubleTy)
    std::cerr << "+" << getUnifiedRegName(uRegName+1);
  std::cerr << "]\n";
}

} // End llvm namespace

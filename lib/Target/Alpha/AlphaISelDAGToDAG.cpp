//===-- AlphaISelDAGToDAG.cpp - Alpha pattern matching inst selector ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for Alpha,
// converting from a legalized dag to a Alpha dag.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaTargetMachine.h"
#include "AlphaISelLowering.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
using namespace llvm;

namespace {

  //===--------------------------------------------------------------------===//
  /// AlphaDAGToDAGISel - Alpha specific code to select Alpha machine
  /// instructions for SelectionDAG operations.
  ///
  class AlphaDAGToDAGISel : public SelectionDAGISel {
    AlphaTargetLowering AlphaLowering;

    static const int IMM_LOW  = -32768;
    static const int IMM_HIGH = 32767;
    static const int IMM_MULT = 65536;
    
  public:
    AlphaDAGToDAGISel(TargetMachine &TM)
      : SelectionDAGISel(AlphaLowering), AlphaLowering(TM) {}

    /// getI64Imm - Return a target constant with the specified value, of type
    /// i64.
    inline SDOperand getI64Imm(int64_t Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i64);
    }

    // Select - Convert the specified operand from a target-independent to a
    // target-specific node if it hasn't already been changed.
    SDOperand Select(SDOperand Op);
    
    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
    
    virtual const char *getPassName() const {
      return "Alpha DAG->DAG Pattern Instruction Selection";
    } 

// Include the pieces autogenerated from the target description.
#include "AlphaGenDAGISel.inc"
    
private:
    SDOperand getGlobalBaseReg();
    SDOperand SelectCALL(SDOperand Op);

  };
}

/// getGlobalBaseReg - Output the instructions required to put the
/// GOT address into a register.
///
SDOperand AlphaDAGToDAGISel::getGlobalBaseReg() {
  return CurDAG->getRegister(AlphaLowering.getVRegGP(), MVT::i64);
}

/// InstructionSelectBasicBlock - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void AlphaDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());
  
  // Select target instructions for the DAG.
  DAG.setRoot(Select(DAG.getRoot()));
  CodeGenMap.clear();
  DAG.RemoveDeadNodes();
  
  // Emit machine code to BB. 
  ScheduleAndEmitDAG(DAG);
}

// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
SDOperand AlphaDAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END &&
      N->getOpcode() < AlphaISD::FIRST_NUMBER)
    return Op;   // Already selected.

  // If this has already been converted, use it.
  std::map<SDOperand, SDOperand>::iterator CGMI = CodeGenMap.find(Op);
  if (CGMI != CodeGenMap.end()) return CGMI->second;
  
  switch (N->getOpcode()) {
  default: break;
  case ISD::TAILCALL:
  case ISD::CALL: return SelectCALL(Op);

  case ISD::DYNAMIC_STACKALLOC: {
    if (!isa<ConstantSDNode>(N->getOperand(2)) ||
        cast<ConstantSDNode>(N->getOperand(2))->getValue() != 0) {
      std::cerr << "Cannot allocate stack object with greater alignment than"
                << " the stack alignment yet!";
      abort();
    }

    SDOperand Chain = Select(N->getOperand(0));
    SDOperand Amt   = Select(N->getOperand(1));
    SDOperand Reg = CurDAG->getRegister(Alpha::R30, MVT::i64);
    SDOperand Val = CurDAG->getCopyFromReg(Chain, Alpha::R30, MVT::i64);
    Chain = Val.getValue(1);
    
    // Subtract the amount (guaranteed to be a multiple of the stack alignment)
    // from the stack pointer, giving us the result pointer.
    SDOperand Result = CurDAG->getTargetNode(Alpha::SUBQ, MVT::i64, Val, Amt);
    
    // Copy this result back into R30.
    Chain = CurDAG->getNode(ISD::CopyToReg, MVT::Other, Chain, Reg, Result);
    
    // Copy this result back out of R30 to make sure we're not using the stack
    // space without decrementing the stack pointer.
    Result = CurDAG->getCopyFromReg(Chain, Alpha::R30, MVT::i64);
  
    // Finally, replace the DYNAMIC_STACKALLOC with the copyfromreg.
    CodeGenMap[Op.getValue(0)] = Result;
    CodeGenMap[Op.getValue(1)] = Result.getValue(1);
    return SDOperand(Result.Val, Op.ResNo);
  }
  case ISD::BRCOND: {
    SDOperand Chain = Select(N->getOperand(0));
    SDOperand CC = Select(N->getOperand(1));
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N->getOperand(2))->getBasicBlock();
    CurDAG->SelectNodeTo(N, Alpha::BNE, MVT::Other, CC, CurDAG->getBasicBlock(Dest), Chain);
    return SDOperand(N, 0);
  }
  case ISD::LOAD:
  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD: {
    SDOperand Chain = Select(N->getOperand(0));
    SDOperand Address = Select(N->getOperand(1));
    unsigned opcode = N->getOpcode();
    unsigned Opc = Alpha::WTF;
    if (opcode == ISD::LOAD)
      switch (N->getValueType(0)) {
      default: N->dump(); assert(0 && "Bad load!");
      case MVT::i64: Opc = Alpha::LDQ; break;
      case MVT::f64: Opc = Alpha::LDT; break;
      case MVT::f32: Opc = Alpha::LDS; break;
      }
    else
      switch (cast<VTSDNode>(N->getOperand(3))->getVT()) {
      default: N->dump(); assert(0 && "Bad sign extend!");
      case MVT::i32: Opc = Alpha::LDL;
        assert(opcode != ISD::ZEXTLOAD && "Not sext"); break;
      case MVT::i16: Opc = Alpha::LDWU;
        assert(opcode != ISD::SEXTLOAD && "Not zext"); break;
      case MVT::i1: //FIXME: Treat i1 as i8 since there are problems otherwise
      case MVT::i8: Opc = Alpha::LDBU;
          assert(opcode != ISD::SEXTLOAD && "Not zext"); break;
      }

    CurDAG->SelectNodeTo(N, Opc, N->getValueType(0), MVT::Other,
                         getI64Imm(0), Address, Chain);
    return SDOperand(N, Op.ResNo);
  }
  case ISD::STORE:
  case ISD::TRUNCSTORE: {
    SDOperand Chain = Select(N->getOperand(0));
    SDOperand Value = Select(N->getOperand(1));
    SDOperand Address = Select(N->getOperand(2));

    unsigned Opc = Alpha::WTF;

    if (N->getOpcode() == ISD::STORE) {
      switch (N->getOperand(1).getValueType()) {
      case MVT::i64: Opc = Alpha::STQ; break;
      case MVT::f64: Opc = Alpha::STT; break;
      case MVT::f32: Opc = Alpha::STS; break;
      default: assert(0 && "Bad store!");
      };
    } else { //TRUNCSTORE
      switch (cast<VTSDNode>(N->getOperand(4))->getVT()) {
      case MVT::i32: Opc = Alpha::STL; break;
      case MVT::i16: Opc = Alpha::STW; break;
      case MVT::i8: Opc = Alpha::STB; break;
      default: assert(0 && "Bad truncstore!");
      };
    }
    CurDAG->SelectNodeTo(N, Opc, MVT::Other, Value, getI64Imm(0), Address, 
                         Chain);
    return SDOperand(N, 0);
  }

  case ISD::BR: {
    CurDAG->SelectNodeTo(N, Alpha::BR_DAG, MVT::Other, N->getOperand(1),
                         Select(N->getOperand(0)));
    return SDOperand(N, 0);
  }

  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    CurDAG->SelectNodeTo(N, Alpha::LDA, MVT::i64,
                         CurDAG->getTargetFrameIndex(FI, MVT::i32),
                         getI64Imm(0));
    return SDOperand(N, 0);
  }
  case ISD::ConstantPool: {
    Constant *C = cast<ConstantPoolSDNode>(N)->get();
    SDOperand Tmp, CPI = CurDAG->getTargetConstantPool(C, MVT::i64);
    Tmp = CurDAG->getTargetNode(Alpha::LDAHr, MVT::i64, CPI, getGlobalBaseReg());
    CurDAG->SelectNodeTo(N, Alpha::LDAr, MVT::i64, CPI, Tmp);
    return SDOperand(N, 0);
  }
  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
    SDOperand GA = CurDAG->getTargetGlobalAddress(GV, MVT::i64);
    CurDAG->SelectNodeTo(N, Alpha::LDQl, MVT::i64, GA, getGlobalBaseReg());
    return SDOperand(N, 0);
  }
  case ISD::ExternalSymbol:
    CurDAG->SelectNodeTo(N, Alpha::LDQl, MVT::i64, 
                         CurDAG->getTargetExternalSymbol(cast<ExternalSymbolSDNode>(N)->getSymbol(), MVT::i64),
                         CurDAG->getRegister(AlphaLowering.getVRegGP(), MVT::i64));
    return SDOperand(N, 0);

  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END: {
    unsigned Amt = cast<ConstantSDNode>(N->getOperand(1))->getValue();
    unsigned Opc = N->getOpcode() == ISD::CALLSEQ_START ?
                       Alpha::ADJUSTSTACKDOWN : Alpha::ADJUSTSTACKUP;
    CurDAG->SelectNodeTo(N, Opc, MVT::Other,
                         getI64Imm(Amt), Select(N->getOperand(0)));
    return SDOperand(N, 0);
  }
  case ISD::RET: {
    SDOperand Chain = Select(N->getOperand(0));     // Token chain.

    if (N->getNumOperands() == 2) {
      SDOperand Val = Select(N->getOperand(1));
      if (N->getOperand(1).getValueType() == MVT::i64) {
        Chain = CurDAG->getCopyToReg(Chain, Alpha::R0, Val);
      }
    }
    //BuildMI(BB, Alpha::RET, 2, Alpha::R31).addReg(Alpha::R26).addImm(1);

    // FIXME: add restoring of the RA to R26 to the chain
    // Finally, select this to a ret instruction.
    CurDAG->SelectNodeTo(N, Alpha::RETDAG, MVT::Other, Chain);
    return SDOperand(N, 0);
  }
  case ISD::Constant: {
    int64_t val = (int64_t)cast<ConstantSDNode>(N)->getValue();
    if (val > (int64_t)IMM_HIGH +(int64_t)IMM_HIGH* (int64_t)IMM_MULT ||
        val < (int64_t)IMM_LOW + (int64_t)IMM_LOW * (int64_t)IMM_MULT) {
      MachineConstantPool *CP = BB->getParent()->getConstantPool();
      ConstantUInt *C =
        ConstantUInt::get(Type::getPrimitiveType(Type::ULongTyID) , val);
      SDOperand Tmp, CPI = CurDAG->getTargetConstantPool(C, MVT::i64);
      Tmp = CurDAG->getTargetNode(Alpha::LDAHr, MVT::i64, CPI, getGlobalBaseReg());
      CurDAG->SelectNodeTo(N, Alpha::LDAr, MVT::i64, CPI, Tmp);
      return SDOperand(N, 0);
    }
    break;
  }
  case ISD::ConstantFP:
    if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(N)) {
      bool isDouble = N->getValueType(0) == MVT::f64;
      MVT::ValueType T = isDouble ? MVT::f64 : MVT::f32;
      if (CN->isExactlyValue(+0.0)) {
        CurDAG->SelectNodeTo(N, isDouble ? Alpha::CPYST : Alpha::CPYSS, T,
                             CurDAG->getRegister(Alpha::F31, T),
                             CurDAG->getRegister(Alpha::F31, T));
        return SDOperand(N, 0);
      } else if ( CN->isExactlyValue(-0.0)) {
        CurDAG->SelectNodeTo(N, isDouble ? Alpha::CPYSNT : Alpha::CPYSNS, T,
                             CurDAG->getRegister(Alpha::F31, T),
                             CurDAG->getRegister(Alpha::F31, T));
        return SDOperand(N, 0);
      } else {
        abort();
      }
      break;
    }
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::UREM:
  case ISD::SREM:
    if (MVT::isInteger(N->getValueType(0))) {
      const char* opstr = 0;
      switch(N->getOpcode()) {
      case ISD::UREM: opstr = "__remqu"; break;
      case ISD::SREM: opstr = "__remq";  break;
      case ISD::UDIV: opstr = "__divqu"; break;
      case ISD::SDIV: opstr = "__divq";  break;
      }
      SDOperand Tmp1 = Select(N->getOperand(0)),
        Tmp2 = Select(N->getOperand(1)),
        Addr = CurDAG->getExternalSymbol(opstr, AlphaLowering.getPointerTy());
      SDOperand Tmp3 = Select(Addr);
      SDOperand Chain = CurDAG->getCopyToReg(CurDAG->getRoot(), Alpha::R24, 
                                             Tmp1, SDOperand());
      Chain = CurDAG->getCopyToReg(CurDAG->getRoot(), Alpha::R25, 
                                   Tmp2, Chain.getValue(1));
      Chain = CurDAG->getCopyToReg(CurDAG->getRoot(), Alpha::R27, 
                                   Tmp3, Chain.getValue(1));
      Chain = CurDAG->getTargetNode(Alpha::JSRs, MVT::i64, MVT::Flag,
                                    CurDAG->getRegister(Alpha::R27, MVT::i64),
                                    getI64Imm(0));
      return CurDAG->getCopyFromReg(Chain.getValue(1), Alpha::R27, MVT::i64, 
                                    Chain.getValue(1));
    }
    break;

  case ISD::SETCC:
    if (MVT::isFloatingPoint(N->getOperand(0).Val->getValueType(0))) {
      unsigned Opc = Alpha::WTF;
      ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
      bool rev = false;
      switch(CC) {
      default: N->dump(); assert(0 && "Unknown FP comparison!");
      case ISD::SETEQ: Opc = Alpha::CMPTEQ; break;
      case ISD::SETLT: Opc = Alpha::CMPTLT; break;
      case ISD::SETLE: Opc = Alpha::CMPTLE; break;
      case ISD::SETGT: Opc = Alpha::CMPTLT; rev = true; break;
      case ISD::SETGE: Opc = Alpha::CMPTLE; rev = true; break;
        //case ISD::SETNE: Opc = Alpha::CMPTEQ; inv = true; break;
      };
      SDOperand tmp1 = Select(N->getOperand(0)),
        tmp2 = Select(N->getOperand(1));
      SDOperand cmp = CurDAG->getTargetNode(Opc, MVT::f64, 
                                            rev?tmp2:tmp1,
                                            rev?tmp1:tmp2);
      SDOperand LD;
      if (AlphaLowering.hasITOF()) {
        LD = CurDAG->getNode(AlphaISD::FTOIT_, MVT::i64, cmp);
      } else {
        int FrameIdx =
          CurDAG->getMachineFunction().getFrameInfo()->CreateStackObject(8, 8);
        SDOperand FI = CurDAG->getFrameIndex(FrameIdx, MVT::i64);
        SDOperand ST = CurDAG->getTargetNode(Alpha::STT, MVT::Other, 
                                             cmp, FI, CurDAG->getRegister(Alpha::R31, MVT::i64));
        LD = CurDAG->getTargetNode(Alpha::LDQ, MVT::i64, FI, 
                                   CurDAG->getRegister(Alpha::R31, MVT::i64),
                                   ST);
      }
      SDOperand FP = CurDAG->getTargetNode(Alpha::CMPULT, MVT::i64, 
                                           CurDAG->getRegister(Alpha::R31, MVT::i64),
                                           LD);
      return FP;
    }
    break;
  }
  
  return SelectCode(Op);
}

SDOperand AlphaDAGToDAGISel::SelectCALL(SDOperand Op) {
  //TODO: add flag stuff to prevent nondeturministic breakage!

  SDNode *N = Op.Val;
  SDOperand Chain = Select(N->getOperand(0));
  SDOperand Addr = Select(N->getOperand(1));

//   unsigned CallOpcode;
   std::vector<SDOperand> CallOperands;
   std::vector<MVT::ValueType> TypeOperands;
  
   //grab the arguments
   for(int i = 2, e = N->getNumOperands(); i < e; ++i) {
     TypeOperands.push_back(N->getOperand(i).getValueType());
     CallOperands.push_back(Select(N->getOperand(i)));
   }
   int count = N->getNumOperands() - 2;

   static const unsigned args_int[] = {Alpha::R16, Alpha::R17, Alpha::R18,
                                       Alpha::R19, Alpha::R20, Alpha::R21};
   static const unsigned args_float[] = {Alpha::F16, Alpha::F17, Alpha::F18,
                                         Alpha::F19, Alpha::F20, Alpha::F21};
   
   for (int i = 0; i < std::min(6, count); ++i) {
     if (MVT::isInteger(TypeOperands[i])) {
       Chain = CurDAG->getCopyToReg(Chain, args_int[i], CallOperands[i]);
     } else if (TypeOperands[i] == MVT::f32 || TypeOperands[i] == MVT::f64) {
       Chain = CurDAG->getCopyToReg(Chain, args_float[i], CallOperands[i]);
     } else
       assert(0 && "Unknown operand"); 
   }
   for (int i = 6; i < count; ++i) {
     unsigned Opc = Alpha::WTF;
     if (MVT::isInteger(TypeOperands[i])) {
       Opc = Alpha::STQ;
     } else if (TypeOperands[i] == MVT::f32) {
       Opc = Alpha::STS;
     } else if (TypeOperands[i] == MVT::f64) {
       Opc = Alpha::STT;
     } else
       assert(0 && "Unknown operand"); 
     Chain = CurDAG->getTargetNode(Opc, MVT::Other, CallOperands[i], 
                                   getI64Imm((i - 6) * 8), 
                                   CurDAG->getRegister(Alpha::R30, MVT::i64),
                                   Chain);
   }

   Chain = CurDAG->getCopyToReg(Chain, Alpha::R27, Addr);
   // Finally, once everything is in registers to pass to the call, emit the
   // call itself.
   Chain = CurDAG->getTargetNode(Alpha::JSRDAG, MVT::Other, Chain );
  
   std::vector<SDOperand> CallResults;
  
   switch (N->getValueType(0)) {
   default: assert(0 && "Unexpected ret value!");
     case MVT::Other: break;
   case MVT::i64:
     Chain = CurDAG->getCopyFromReg(Chain, Alpha::R0, MVT::i64).getValue(1);
     CallResults.push_back(Chain.getValue(0));
     break;
   case MVT::f32:
     Chain = CurDAG->getCopyFromReg(Chain, Alpha::F0, MVT::f32).getValue(1);
     CallResults.push_back(Chain.getValue(0));
     break;
   case MVT::f64:
     Chain = CurDAG->getCopyFromReg(Chain, Alpha::F0, MVT::f64).getValue(1);
     CallResults.push_back(Chain.getValue(0));
     break;
   }

   CallResults.push_back(Chain);
   for (unsigned i = 0, e = CallResults.size(); i != e; ++i)
     CodeGenMap[Op.getValue(i)] = CallResults[i];
   return CallResults[Op.ResNo];
}


/// createAlphaISelDag - This pass converts a legalized DAG into a 
/// Alpha-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createAlphaISelDag(TargetMachine &TM) {
  return new AlphaDAGToDAGISel(TM);
}

//===-- llvm/CodeGen/AsmPrinter.h - AsmPrinter Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the base class for target specific
// asm writers.  This class primarily handles common functionality used by
// all asm writers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMPRINTER_H
#define LLVM_CODEGEN_ASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/DataTypes.h"
#include <set>

namespace llvm {
  class Constant;
  class ConstantArray;
  class GlobalVariable;
  class GlobalAlias;
  class MachineConstantPoolEntry;
  class MachineConstantPoolValue;
  class Mangler;
  class TargetAsmInfo;
  

  /// AsmPrinter - This class is intended to be used as a driving class for all
  /// asm writers.
  class AsmPrinter : public MachineFunctionPass {
    static char ID;

    /// FunctionNumber - This provides a unique ID for each function emitted in
    /// this translation unit.  It is autoincremented by SetupMachineFunction,
    /// and can be accessed with getFunctionNumber() and 
    /// IncrementFunctionNumber().
    ///
    unsigned FunctionNumber;

  protected:
    // Necessary for external weak linkage support
    std::set<const GlobalValue*> ExtWeakSymbols;

  public:
    /// Output stream on which we're printing assembly code.
    ///
    std::ostream &O;

    /// Target machine description.
    ///
    TargetMachine &TM;
    
    /// Target Asm Printer information.
    ///
    const TargetAsmInfo *TAI;

    /// Name-mangler for global names.
    ///
    Mangler *Mang;

    /// Cache of mangled name for current function. This is recalculated at the
    /// beginning of each call to runOnMachineFunction().
    ///
    std::string CurrentFnName;
    
    /// CurrentSection - The current section we are emitting to.  This is
    /// controlled and used by the SwitchSection method.
    std::string CurrentSection;
  
  protected:
    AsmPrinter(std::ostream &o, TargetMachine &TM, const TargetAsmInfo *T);
    
  public:
    /// SwitchToTextSection - Switch to the specified section of the executable
    /// if we are not already in it!  If GV is non-null and if the global has an
    /// explicitly requested section, we switch to the section indicated for the
    /// global instead of NewSection.
    ///
    /// If the new section is an empty string, this method forgets what the
    /// current section is, but does not emit a .section directive.
    ///
    /// This method is used when about to emit executable code.
    ///
    void SwitchToTextSection(const char *NewSection, const GlobalValue *GV = NULL);

    /// SwitchToDataSection - Switch to the specified section of the executable
    /// if we are not already in it!  If GV is non-null and if the global has an
    /// explicitly requested section, we switch to the section indicated for the
    /// global instead of NewSection.
    ///
    /// If the new section is an empty string, this method forgets what the
    /// current section is, but does not emit a .section directive.
    ///
    /// This method is used when about to emit data.  For most assemblers, this
    /// is the same as the SwitchToTextSection method, but not all assemblers
    /// are the same.
    ///
    void SwitchToDataSection(const char *NewSection, const GlobalValue *GV = NULL);
    
    /// getGlobalLinkName - Returns the asm/link name of of the specified
    /// global variable.  Should be overridden by each target asm printer to
    /// generate the appropriate value.
    virtual const std::string getGlobalLinkName(const GlobalVariable *GV) const;

    /// EmitExternalGlobal - Emit the external reference to a global variable.
    /// Should be overridden if an indirect reference should be used.
    virtual void EmitExternalGlobal(const GlobalVariable *GV);

    /// getCurrentFunctionEHName - Called to return (and cache) the
    /// CurrentFnEHName.
    /// 
    std::string getCurrentFunctionEHName(const MachineFunction *MF);

  protected:
    /// doInitialization - Set up the AsmPrinter when we are working on a new
    /// module.  If your pass overrides this, it must make sure to explicitly
    /// call this implementation.
    bool doInitialization(Module &M);

    /// doFinalization - Shut down the asmprinter.  If you override this in your
    /// pass, you must make sure to call it explicitly.
    bool doFinalization(Module &M);
    
    /// PrintSpecial - Print information related to the specified machine instr
    /// that is independent of the operand, and may be independent of the instr
    /// itself.  This can be useful for portably encoding the comment character
    /// or other bits of target-specific knowledge into the asmstrings.  The
    /// syntax used is ${:comment}.  Targets can override this to add support
    /// for their own strange codes.
    virtual void PrintSpecial(const MachineInstr *MI, const char *Code);

    /// PrintAsmOperand - Print the specified operand of MI, an INLINEASM
    /// instruction, using the specified assembler variant.  Targets should
    /// override this to format as appropriate.  This method can return true if
    /// the operand is erroneous.
    virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode);
    
    /// PrintAsmMemoryOperand - Print the specified operand of MI, an INLINEASM
    /// instruction, using the specified assembler variant as an address.
    /// Targets should override this to format as appropriate.  This method can
    /// return true if the operand is erroneous.
    virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant, 
                                       const char *ExtraCode);
    
    /// getSectionForFunction - Return the section that we should emit the
    /// specified function body into.  This defaults to 'TextSection'.  This
    /// should most likely be overridden by the target to put linkonce/weak
    /// functions into special sections.
    virtual std::string getSectionForFunction(const Function &F) const;
    
    /// SetupMachineFunction - This should be called when a new MachineFunction
    /// is being processed from runOnMachineFunction.
    void SetupMachineFunction(MachineFunction &MF);
    
    /// getFunctionNumber - Return a unique ID for the current function.
    ///
    unsigned getFunctionNumber() const { return FunctionNumber; }
    
    /// IncrementFunctionNumber - Increase Function Number.  AsmPrinters should
    /// not normally call this, as the counter is automatically bumped by
    /// SetupMachineFunction.
    void IncrementFunctionNumber() { FunctionNumber++; }
    
    /// EmitConstantPool - Print to the current output stream assembly
    /// representations of the constants in the constant pool MCP. This is
    /// used to print out constants which have been "spilled to memory" by
    /// the code generator.
    ///
    void EmitConstantPool(MachineConstantPool *MCP);

    /// EmitJumpTableInfo - Print assembly representations of the jump tables 
    /// used by the current function to the current output stream.  
    ///
    void EmitJumpTableInfo(MachineJumpTableInfo *MJTI, MachineFunction &MF);
    
    /// EmitSpecialLLVMGlobal - Check to see if the specified global is a
    /// special global used by LLVM.  If so, emit it and return true, otherwise
    /// do nothing and return false.
    bool EmitSpecialLLVMGlobal(const GlobalVariable *GV);
    
  public:
    //===------------------------------------------------------------------===//
    /// LEB 128 number encoding.

    /// PrintULEB128 - Print a series of hexidecimal values(separated by commas)
    /// representing an unsigned leb128 value.
    void PrintULEB128(unsigned Value) const;

    /// SizeULEB128 - Compute the number of bytes required for an unsigned
    /// leb128 value.
    static unsigned SizeULEB128(unsigned Value);

    /// PrintSLEB128 - Print a series of hexidecimal values(separated by commas)
    /// representing a signed leb128 value.
    void PrintSLEB128(int Value) const;

    /// SizeSLEB128 - Compute the number of bytes required for a signed leb128
    /// value.
    static unsigned SizeSLEB128(int Value);
    
    //===------------------------------------------------------------------===//
    // Emission and print routines
    //

    /// PrintHex - Print a value as a hexidecimal value.
    ///
    void PrintHex(int Value) const;

    /// EOL - Print a newline character to asm stream.  If a comment is present
    /// then it will be printed first.  Comments should not contain '\n'.
    void EOL() const;
    void EOL(const std::string &Comment) const;
    
    /// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
    /// unsigned leb128 value.
    void EmitULEB128Bytes(unsigned Value) const;
    
    /// EmitSLEB128Bytes - print an assembler byte data directive to compose a
    /// signed leb128 value.
    void EmitSLEB128Bytes(int Value) const;
    
    /// EmitInt8 - Emit a byte directive and value.
    ///
    void EmitInt8(int Value) const;

    /// EmitInt16 - Emit a short directive and value.
    ///
    void EmitInt16(int Value) const;

    /// EmitInt32 - Emit a long directive and value.
    ///
    void EmitInt32(int Value) const;

    /// EmitInt64 - Emit a long long directive and value.
    ///
    void EmitInt64(uint64_t Value) const;

    /// EmitString - Emit a string with quotes and a null terminator.
    /// Special characters are emitted properly.
    /// @verbatim (Eg. '\t') @endverbatim
    void EmitString(const std::string &String) const;
    
    /// EmitFile - Emit a .file directive.
    void EmitFile(unsigned Number, const std::string &Name) const;

    //===------------------------------------------------------------------===//

    /// EmitAlignment - Emit an alignment directive to the specified power of
    /// two boundary.  For example, if you pass in 3 here, you will get an 8
    /// byte alignment.  If a global value is specified, and if that global has
    /// an explicit alignment requested, it will unconditionally override the
    /// alignment request.  However, if ForcedAlignBits is specified, this value
    /// has final say: the ultimate alignment will be the max of ForcedAlignBits
    /// and the alignment computed with NumBits and the global. If UseFillExpr
    /// is true, it also emits an optional second value FillValue which the
    /// assembler uses to fill gaps to match alignment.
    ///
    /// The algorithm is:
    ///     Align = NumBits;
    ///     if (GV && GV->hasalignment) Align = GV->getalignment();
    ///     Align = std::max(Align, ForcedAlignBits);
    ///
    void EmitAlignment(unsigned NumBits, const GlobalValue *GV = 0,
                       unsigned ForcedAlignBits = 0, bool UseFillExpr = false,
                       unsigned FillValue = 0) const;

  protected:
    /// EmitZeros - Emit a block of zeros.
    ///
    void EmitZeros(uint64_t NumZeros) const;

    /// EmitString - Emit a zero-byte-terminated string constant.
    ///
    virtual void EmitString(const ConstantArray *CVA) const;

    /// EmitConstantValueOnly - Print out the specified constant, without a
    /// storage class.  Only constants of first-class type are allowed here.
    void EmitConstantValueOnly(const Constant *CV);

    /// EmitGlobalConstant - Print a general LLVM constant to the .s file.
    /// If Packed is false, pad to the ABI size.
    void EmitGlobalConstant(const Constant* CV, bool Packed = false);

    virtual void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV);
    
    /// printInlineAsm - This method formats and prints the specified machine
    /// instruction that is an inline asm.
    void printInlineAsm(const MachineInstr *MI) const;
    
    /// printLabel - This method prints a local label used by debug and
    /// exception handling tables.
    void printLabel(const MachineInstr *MI) const;

    /// printBasicBlockLabel - This method prints the label for the specified
    /// MachineBasicBlock
    virtual void printBasicBlockLabel(const MachineBasicBlock *MBB,
                                      bool printColon = false,
                                      bool printComment = true) const;
                                      
    /// printPICJumpTableSetLabel - This method prints a set label for the
    /// specified MachineBasicBlock for a jumptable entry.
    virtual void printPICJumpTableSetLabel(unsigned uid,
                                           const MachineBasicBlock *MBB) const;
    virtual void printPICJumpTableSetLabel(unsigned uid, unsigned uid2,
                                           const MachineBasicBlock *MBB) const;

    /// printDataDirective - This method prints the asm directive for the
    /// specified type.
    void printDataDirective(const Type *type);

  private:
    void EmitLLVMUsedList(Constant *List);
    void EmitXXStructorList(Constant *List);
    void EmitConstantPool(unsigned Alignment, const char *Section,
                std::vector<std::pair<MachineConstantPoolEntry,unsigned> > &CP);

  };
}

#endif

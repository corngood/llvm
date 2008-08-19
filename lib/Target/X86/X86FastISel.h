//===-- X86FastISel.h - X86 FastISel header -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface to the X86-specific support for the FastISel
// class.
//
//===----------------------------------------------------------------------===//

#ifndef X86FASTISEL_H
#define X86FASTISEL_H

namespace llvm {

class FastISel;
class MachineBasicBlock;
class MachineFunction;
class TargetInstrInfo;

namespace X86 {

FastISel *createFastISel(MachineBasicBlock *mbb, MachineFunction *mf,
                         const TargetInstrInfo *tii);

} // namespace X86

} // namespace llvm

#endif

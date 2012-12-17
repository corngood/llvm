// RUN: llvm-mc -g -fdebug-compilation-dir=/test/comp/dir %s -filetype=obj -o %t.o
// RUN: llvm-dwarfdump %t.o | FileCheck %s

// CHECK: DW_AT_comp_dir [DW_FORM_string] ("/test/comp/dir")

f:
  nop

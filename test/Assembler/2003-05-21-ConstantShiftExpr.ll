; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

; Test that shift instructions can be used in constant expressions.

global int shl (int 7, ubyte 19)

; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo "%X = external global [1x int]" | llvm-upgrade | llvm-as > %t.2.bc
; RUN: llvm-upgrade %s -o - | llvm-as > %t.1.bc
; RUN: llvm-link %t.[12].bc | llvm-dis | grep constant

%X = constant [1 x int] [ int 12 ]

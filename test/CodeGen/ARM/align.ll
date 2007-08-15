; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep align.*1 | count 1
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnueabi | \
; RUN:   grep align.*2 | count 2
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnueabi | \
; RUN:   grep align.*3 | count 2
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin | \
; RUN:   grep align.*2 | count 4

%a = global bool true
%b = global sbyte 1
%c = global short 2
%d = global int 3
%e = global long 4
%f = global float 5.0
%g = global double 6.0

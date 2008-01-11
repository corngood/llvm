; RUN: llvm-as < %s | llc -march=x86
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin10 | grep {array,16512,7}
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9 | grep {array,16512,7}
; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin8 | not grep {7}
; XFAIL: *

; Currently there is no construct which generates .comm, so test is xfail'ed.

; Darwin 9+ should get alignment on common symbols.  Darwin8 does 
; not support this.
@array = weak global [4128 x i32] zeroinitializer, align 128

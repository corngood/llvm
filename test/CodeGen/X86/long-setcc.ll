; RUN: llvm-as < %s | llc -march=x86 &&
; RUN: llvm-as < %s | llc -march=x86 | grep cmp | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 | grep shr | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 | grep xor | wc -l | grep 1

define i1 @t1(i64 %x) {
	%B = icmp slt i64 %x, 0
	ret i1 %B
}

define i1 @t2(i64 %x) {
	%tmp = icmp ult i64 %x, 4294967296
	ret i1 %tmp
}

define i1 @t3(i32 %x) {
	%tmp = icmp ugt i32 %x, -1
	ret i1 %tmp
}

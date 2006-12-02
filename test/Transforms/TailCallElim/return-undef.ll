; RUN: llvm-upgrade < %s | llvm-as | opt -tailcallelim | llvm-dis | grep sub &&
; RUN: llvm-upgrade < %s | llvm-as | opt -tailcallelim | llvm-dis | not grep call

int %test(int %X) {
	%Y = sub int %X, 1
	%Z = call int %test(int %Y)
	ret int undef
}

; test that malloc's with a constant argument are promoted to array allocations
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep getelementptr

int* %test() {
	%X = malloc int, uint 4
	ret int* %X
}

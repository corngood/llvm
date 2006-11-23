; Test to check for support for "physical subtyping"
;
; RUN: llvm-as < %s | opt -analyze -datastructure-gc -dsgc-abort-if-any-collapsed
;
%S = type { int }
%T = type { int, float, double }

int %main() {
	%A = alloca %S
	%Ap = getelementptr %S* %A, long 0, uint 0
	%B = alloca %T
	%Bp = getelementptr %T* %B, long 0, uint 0
	%C = alloca int*
	
	store int* %Ap, int** %C
	store int* %Bp, int** %C
	ret int 0
}

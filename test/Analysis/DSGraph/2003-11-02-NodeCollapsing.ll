; This is the same testcase as 2003-06-29-NodeCollapsing2.ll, but it uses the 
; graph checker.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -datastructure-gc -dsgc-abort-if-any-collapsed
;

%S = type { double, int }
%T = type { double, int, sbyte }

void %test() {
	%A = alloca double*
	%B = alloca %S
	%C = alloca %T
	%b = getelementptr %S* %B, long 0, uint 0
	%c = getelementptr %T* %C, long 0, uint 0

	store double* %b, double** %A
	store double* %c, double** %A
	ret void
}

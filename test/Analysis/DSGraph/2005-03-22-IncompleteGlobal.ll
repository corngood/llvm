; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -datastructure-gc  -dsgc-check-flags=G:GIM -dsgc-dspass=bu &&\
; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -datastructure-gc  -dsgc-check-flags=G:GIM -dsgc-dspass=td

%S = type { double, int }

%G = external global %S

void %main() {
	%b = getelementptr %S* %G, long 0, uint 0
	store double 0.1, double* %b
	ret void
}

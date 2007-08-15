; This function should have exactly one call to fixdfdi, no more!

; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mattr=-64bit | \
; RUN:    grep {bl .*fixdfdi} | count 1

double %test2(double %tmp.7705) {
        %mem_tmp.2.0.in = cast double %tmp.7705 to long                ; <long> [#uses=1]
        %mem_tmp.2.0 = cast long %mem_tmp.2.0.in to double
	ret double %mem_tmp.2.0
}

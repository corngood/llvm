; Scalar replacement was incorrectly promoting this alloca!!
;
; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | \
; RUN:   sed {s/;.*//g} | grep \[

sbyte *%test() {
	%A = alloca [30 x sbyte]
	%B = getelementptr [30 x sbyte]* %A, long 0, long 0
	%C = getelementptr sbyte* %B, long 1
	store sbyte 0, sbyte* %B
	ret sbyte* %C
}

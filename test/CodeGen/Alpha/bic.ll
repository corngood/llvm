; Make sure this testcase codegens to the bic instruction
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep 'bic'

implementation   ; Functions:

long %bar(long %x, long %y) {
entry:
	%tmp.1 = xor long %x, -1  		; <long> [#uses=1]
        %tmp.2 = and long %y, %tmp.1
	ret long %tmp.2
}

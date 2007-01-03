; Make sure this testcase codegens to the ctpop instruction
; RUN: llvm-as < %s | llc -march=alpha | grep -i 'subl $16,1,$0'

implementation   ; Functions:

define i32 @sext %foo(i32 @sext %x) {
entry:
	%tmp.1 = add i32 %x, -1		; <int> [#uses=1]
	ret i32 %tmp.1
}

; This testcase checks to make sure the sinker does not cause problems with
; critical edges.

; RUN: llvm-upgrade < %s | llvm-as | opt -licm | llvm-dis | %prcontext add 1 | grep Exit

implementation   ; Functions:

void %test() {
Entry:
	br bool false, label %Loop, label %Exit

Loop:
	%X = add int 0, 1
	br bool false, label %Loop, label %Exit

Exit:
	%Y = phi int [ 0, %Entry ], [ %X, %Loop ]
	ret void
}

; RUN: llvm-upgrade < %s > %t1.ll
; RUN: bugpoint %t1.ll  -bugpoint-crashcalls

; Test to make sure that arguments are removed from the function if they are unnecessary.

declare int %test2()
int %test(int %A, int %B, float %C) {
	call int %test2()
	ret int %0
}

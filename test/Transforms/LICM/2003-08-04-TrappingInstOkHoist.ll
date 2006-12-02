; This testcase tests to make sure a trapping instruction is hoisted when
; it is guaranteed to execute.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -licm | llvm-dis | %prcontext "test" 2 | grep div

%X = global int 0
declare void %foo(int)

int %test(bool %c) {
	%A = load int *%X
	br label %Loop
Loop:
	%B = div int 4, %A  ;; Should have hoisted this div!
	call void %foo(int %B)
        br bool %c, label %Loop, label %Out

Out:
	%C = sub int %A, %B
	ret int %C
}

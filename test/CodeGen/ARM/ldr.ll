; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {ldr r0} | count 3

int %f1(int* %v) {
entry:
	%tmp = load int* %v		; <int> [#uses=1]
	ret int %tmp
}

int %f2(int* %v) {
entry:
	%tmp2 = getelementptr int* %v, int 1023		; <int*> [#uses=1]
	%tmp = load int* %tmp2		; <int> [#uses=1]
	ret int %tmp
}

int %f3(int* %v) {
entry:
	%tmp2 = getelementptr int* %v, int 1024		; <int*> [#uses=1]
	%tmp = load int* %tmp2		; <int> [#uses=1]
	ret int %tmp
}

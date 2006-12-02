; RUN: llvm-upgrade < %s | llvm-as | opt -sccp -disable-output &&
; RUN: llvm-upgrade < %s | llvm-as | opt -sccp | llvm-dis | not grep select

int %test1(bool %C) {
	%X = select bool %C, int 0, int 0
	ret int %X
}

int %test2(bool %C) {
	%X = select bool %C, int 0, int undef
	ret int %X
}

; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f


void %test(int %X) {
	call void (int)* cast (void(int) * %test to void(int) *) (int 6)
	ret void

}

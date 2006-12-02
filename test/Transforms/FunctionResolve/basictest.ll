; RUN: llvm-upgrade < %s | llvm-as | opt -funcresolve -instcombine | llvm-dis | grep '\.\.\.' | not grep call

declare int %foo(...)

int %foo(int %x, float %y) {
	ret int %x
}

int %bar() {
	%x = call int(...)* %foo(double 12.5, int 48)
	ret int %x
}

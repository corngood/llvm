; RUN: llvm-upgrade < %s | llvm-as | opt -funcresolve -disable-output

void %foo(int, int) {
  ret void
}
declare void %foo(...)

void %test() {
	call void(...)* %foo(int 7)
	ret void
}

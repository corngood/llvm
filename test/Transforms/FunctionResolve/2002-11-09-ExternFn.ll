; XFAIL: *
;
; RUN: llvm-upgrade < %s | llvm-as | opt -funcresolve -instcombine | llvm-dis | not grep '\.\.\.'

declare int %foo(...)
declare int %foo(int)

void %bar() {
	call int(...)* %foo(int 7)
	ret void
}

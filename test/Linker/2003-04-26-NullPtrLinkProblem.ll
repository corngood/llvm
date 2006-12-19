; This one fails because the LLVM runtime is allowing two null pointers of
; the same type to be created!

; RUN: echo "%T = type int" | llvm-as > %t.2.bc
; RUN: llvm-upgrade < %s | llvm-as -f > %t.1.bc
; RUN: llvm-link %t.[12].bc

%T = type opaque

declare %T* %create()

implementation

void %test() {
	%X = call %T* %create()
	%v = seteq %T* %X, null
	ret void
}


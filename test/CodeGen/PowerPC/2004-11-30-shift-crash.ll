; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32

void %main() {
	%tr4 = shl ulong 1, ubyte 0		; <ulong> [#uses=0]
	ret void
}

; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32

void %main() {
	%tr1 = shr uint 1, ubyte 0
	ret void
}

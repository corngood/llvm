; RUN: llvm-upgrade < %s | llvm-as | opt -raise

int* %test(int* %P, int* %Q) {
	%P = cast int* %P to ulong
	%Q = cast int* %Q to ulong
	%V = add ulong %P, %Q
	%V = cast ulong %V to int*
	ret int* %V
}

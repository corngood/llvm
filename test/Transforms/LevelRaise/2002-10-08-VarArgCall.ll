; RUN: llvm-upgrade < %s | llvm-as | opt -raise | llvm-dis | grep call | not grep '\.\.\.'

implementation

void %test(sbyte* %P) {
	%Q = cast sbyte* %P to void (...)*
	call void (...)* %Q(sbyte* %P)
	ret void
}

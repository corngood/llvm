; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -datastructure-gc -dsgc-abort-if-any-collapsed

%X = internal global { int, short, short } { int 1, short 2, short 3 }

implementation


void %test() {
	store short 5, short* getelementptr ({ int, short, short }* %X, long 0, uint 1)
	ret void
}

; RUN: llvm-as < %s | llc -march=arm
void %f() {
entry:
	call void %g( int 1, int 2, int 3, int 4 )
	call fastcc void %h()
	ret void
}

declare void %g(int, int, int, int)
declare fastcc void %h()

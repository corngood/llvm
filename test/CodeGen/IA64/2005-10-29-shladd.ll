; this should turn into shladd 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ia64 | grep shladd

implementation   ; Functions:

long %bogglesmoggle(long %X, long %Y) {
	%A = shl long %X, ubyte 3
	%B = add long %A, %Y 
        ret long %B
}


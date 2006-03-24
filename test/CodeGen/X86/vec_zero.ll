; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep xorps
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pxor

void %foo(<4 x float> *%P) {
	%T = load <4 x float> * %P
	%S = add <4 x float> zeroinitializer, %T
	store <4 x float> %S, <4 x float>* %P
	ret void
}

void %bar(<4 x int> *%P) {
	%T = load <4 x int> * %P
	%S = add <4 x int> zeroinitializer, %T
	store <4 x int> %S, <4 x int>* %P
	ret void
}

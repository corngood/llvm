; RUN: llvm-as < 2002-11-04-ConstantSharing.ll | opt -raise | llvm-dis | notcast

implementation

bool %test(int *%X, uint* %Y) {
	%A = cast int* %X to sbyte*
	%B = cast uint* %Y to sbyte*
	%c1 = seteq sbyte* %A, null
	%c2 = seteq sbyte* %B, null
	%c = and bool %c1, %c2
	ret bool %c
}

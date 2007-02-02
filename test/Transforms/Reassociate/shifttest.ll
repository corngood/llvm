; With shl->mul reassociation, we can see that this is (shl A, 9) * A
;
; RUN: llvm-upgrade < %s | llvm-as | opt -reassociate -instcombine | llvm-dis | grep 'shl .*, 9'

int %test(int %A, int %B) {
	%X = shl int %A, ubyte 5
	%Y = shl int %A, ubyte 4
	%Z = mul int %Y, %X
	ret int %Z
}

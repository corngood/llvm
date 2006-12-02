; This testcase should have the cast propogated through the load
; just like a store does...
;
; RUN: llvm-upgrade < %s | llvm-as | opt -raise | llvm-dis | not grep 'bitcast uint \*'

int "test"(uint * %Ptr) {
	%P2 = cast uint *%Ptr to int *
	%Val = load int * %P2
	ret int %Val
}

; This test ensures that alloca instructions in the entry block for an inlined
; function are moved to the top of the function they are inlined into.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -inline | llvm-dis | %prcontext alloca 1 | grep Entry:

int %func(int %i) {
	%X = alloca int 
	store int %i, int* %X
	ret int %i
}

declare void %bar()

int %main(int %argc) {
Entry:
	call void %bar()
	%X = call int %func(int 7)
	%Y = add int %X, %argc
	ret int %Y
}

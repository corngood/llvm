; very simple test
;
; RUN: llvm-as < %s | opt -analyze -tddatastructure

implementation

int *%foo(ulong %A, double %B, long %C) {
	%X = malloc int*
	%D = cast int** %X to ulong
	%E = cast ulong %D to int*
	store int* %E, int** %X

	%F = malloc {int}
	%G = getelementptr {int}* %F, long 0, uint 0
	store int* %G, int** %X

	%K = malloc int **
	store int** %X, int***%K

	%H = cast long %C to int*
	ret int* null ; %H
} 


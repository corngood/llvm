; This test makes sure that these instructions are properly constant propagated.
;

; RUN: llvm-upgrade < %s | llvm-as | opt -sccp | llvm-dis | not grep load

%X = constant int 42
%Y = constant [2 x { int, float }] [ { int, float } { int 12, float 1.0 }, 
                                     { int, float } { int 37, float 1.2312 } ]
int %test1() {
	%B = load int* %X
	ret int %B
}

float %test2() {
	%A = getelementptr [2 x { int, float}]* %Y, long 0, long 1, uint 1
	%B = load float* %A
	ret float %B
}

int %test3() {
	%A = getelementptr [2 x { int, float}]* %Y, long 0, long 0, uint 0
	%B = load int* %A
	ret int %B
}



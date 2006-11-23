; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep load

%X = constant int 42
%X2 = constant int 47
%Y = constant [2 x { int, float }] [ { int, float } { int 12, float 1.0 }, 
                                     { int, float } { int 37, float 1.2312 } ]
%Z = constant [2 x { int, float }] zeroinitializer

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

int %test4() {
	%A = getelementptr [2 x { int, float}]* %Z, long 0, long 1, uint 0
	%B = load int* %A
	ret int %B
}

; load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2)
int %test5(bool %C) {
	%Y = select bool %C, int* %X, int* %X2
	%Z = load int* %Y
	ret int %Z
}

int %test7(int %X) {
	%V = getelementptr int* null, int %X
	%R = load int* %V
	ret int %R
}

int %test8(int* %P) {
	store int 1, int* %P
	%X = load int* %P        ;; Trivial store->load forwarding
	ret int %X
}

int %test9(int* %P) {
	%X = load int* %P        ;; Trivial load cse
	%Y = load int* %P
	%Z = sub int %X, %Y
	ret int %Z
}

int %test10(bool %C, int* %P, int* %Q) {
	br bool %C, label %T, label %F
T:
	store int 1, int* %Q
	store int 0, int* %P
	br label %C
F:
	store int 0, int* %P
	br label %C
C:
	%V = load int* %P   ;; always 0
	ret int %V
}

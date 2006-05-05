; Tests to make sure elimination of casts is working correctly
; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep '%c' | not grep cast

%inbuf = external global [32832 x ubyte]

implementation

int %test1(int %A) {
	%c1 = cast int %A to uint
	%c2 = cast uint %c1 to int
	ret int %c2
}

ulong %test2(ubyte %A) {
	%c1 = cast ubyte %A to ushort
	%c2 = cast ushort %c1 to uint
	%Ret = cast uint %c2 to ulong
	ret ulong %Ret
}

ulong %test3(ulong %A) {    ; This function should just use bitwise AND
	%c1 = cast ulong %A to ubyte
	%c2 = cast ubyte %c1 to ulong
	ret ulong %c2
}

uint %test4(int %A, int %B) {
        %COND = setlt int %A, %B
        %c = cast bool %COND to ubyte     ; Booleans are unsigned integrals
        %result = cast ubyte %c to uint   ; for the cast elim purpose
        ret uint %result
}

int %test5(bool %B) {
        %c = cast bool %B to ubyte       ; This cast should get folded into
        %result = cast ubyte %c to int   ; this cast
        ret int %result
}

int %test6(ulong %A) {
	%c1 = cast ulong %A to uint
	%res = cast uint %c1 to int
	ret int %res
}

long %test7(bool %A) {
	%c1 = cast bool %A to int
	%res = cast int %c1 to long
	ret long %res
}

long %test8(sbyte %A) {
        %c1 = cast sbyte %A to ulong
        %res = cast ulong %c1 to long
        ret long %res
}

short %test9(short %A) {
	%c1 = cast short %A to int
	%c2 = cast int %c1 to short
	ret short %c2
}

short %test10(short %A) {
	%c1 = cast short %A to uint
	%c2 = cast uint %c1 to short
	ret short %c2
}

declare void %varargs(int, ...)

void %test11(int* %P) {
	%c = cast int* %P to short*
	call void(int, ...)* %varargs(int 5, short* %c)
	ret void
}

int* %test12() {
	%p = malloc [4 x sbyte]
	%c = cast [4 x sbyte]* %p to int*
	ret int* %c
}

ubyte *%test13(long %A) {
	%c = getelementptr [0 x ubyte]* cast ([32832 x ubyte]*  %inbuf to [0 x ubyte]*), long 0, long %A
	ret ubyte* %c
}

bool %test14(sbyte %A) {
        %c = cast sbyte %A to ubyte
        %X = setlt ubyte %c, 128   ; setge %A, 0
        ret bool %X
}

bool %test15(ubyte %A) {
        %c = cast ubyte %A to sbyte
        %X = setlt sbyte %c, 0   ; setgt %A, 127
        ret bool %X
}

bool %test16(int* %P) {
	%c = cast int* %P to bool  ;; setne P, null
	ret bool %c
}


short %test17(bool %tmp3) {
	%c = cast bool %tmp3 to int
	%t86 = cast int %c to short
	ret short %t86
}

short %test18(sbyte %tmp3) {
	%c = cast sbyte %tmp3 to int
	%t86 = cast int %c to short
	ret short %t86
}

bool %test19(int %X) {
	%c = cast int %X to long
	%Z = setlt long %c, 12345
	ret bool %Z
}

bool %test20(bool %B) {
	%c = cast bool %B to int
	%D = setlt int %c, -1
	ret bool %D                ;; false
}

uint %test21(uint %X) {
	%c1 = cast uint %X to sbyte
	%c2 = cast sbyte %c1 to uint ;; sext -> zext -> and -> nop
	%RV = and uint %c2, 255
	ret uint %RV
}

uint %test22(uint %X) {
	%c1 = cast uint %X to sbyte
	%c2 = cast sbyte %c1 to uint ;; sext -> zext -> and -> nop
	%RV = shl uint %c2, ubyte 24
	ret uint %RV
}

int %test23(int %X) {
	%c1 = cast int %X to ushort  ;; Turn into an AND even though X
	%c2 = cast ushort %c1 to int  ;; and Z are signed.
	ret int %c2
}

bool %test24(bool %C) {
        %X = select bool %C, uint 14, uint 1234
        %c = cast uint %X to bool                  ;; Fold cast into select
        ret bool %c
}

void %test25(int** %P) {
        %c = cast int** %P to float**
        store float* null, float** %c          ;; Fold cast into null
        ret void
}

int %test26(float %F) {
	%c = cast float %F to double   ;; no need to cast from float->double.
	%D = cast double %c to int
	ret int %D
}

[4 x float]* %test27([9 x [4 x float]]* %A) {
        %c = cast [9 x [4 x float]]* %A to [4 x float]*
	ret [4 x float]* %c
}

float* %test28([4 x float]* %A) {
        %c = cast [4 x float]* %A to float*
	ret float* %c
}

uint %test29(uint %c1, uint %c2) {
	%tmp1 = cast uint %c1 to ubyte
        %tmp4.mask = cast uint %c2 to ubyte
        %tmp = or ubyte %tmp4.mask, %tmp1
        %tmp10 = cast ubyte %tmp to uint
	ret uint %tmp10
}


; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep -v {sub i32 %Cok, %Bok} | not grep sub

implementation

int %test1(int %A) {
	%B = sub int %A, %A    ; ISA constant 0
	ret int %B
}

int %test2(int %A) {
	%B = sub int %A, 0
	ret int %B
}

int %test3(int %A) {
	%B = sub int 0, %A       ; B = -A
	%C = sub int 0, %B       ; C = -B = A
	ret int %C
}

int %test4(int %A, int %x) {
	%B = sub int 0, %A
	%C = sub int %x, %B
	ret int %C
}

int %test5(int %A, int %Bok, int %Cok) {
	%D = sub int %Bok, %Cok
	%E = sub int %A, %D
	ret int %E
}

int %test6(int %A, int %B) {
	%C = and int %A, %B   ; A - (A & B) => A & ~B
	%D = sub int %A, %C
	ret int %D
}

int %test7(int %A) {
	%B = sub int -1, %A   ; B = ~A
	ret int %B
}

int %test8(int %A) {
        %B = mul int 9, %A
        %C = sub int %B, %A      ; C = 9*A-A == A*8 == A << 3
        ret int %C
}

int %test9(int %A) {
        %B = mul int 3, %A
        %C = sub int %A, %B      ; C = A-3*A == A*-2
        ret int %C
}

int %test10(int %A, int %B) {    ; -A*-B == A*B
	%C = sub int 0, %A
	%D = sub int 0, %B
	%E = mul int %C, %D
	ret int %E
}

int %test10(int %A) {    ; -A *c1 == A * -c1
	%C = sub int 0, %A
	%E = mul int %C, 7
	ret int %E
}

bool %test11(ubyte %A, ubyte %B) {
        %C = sub ubyte %A, %B
        %cD = setne ubyte %C, 0    ; == setne A, B
        ret bool %cD
}

int %test12(int %A) {
	%B = shr int %A, ubyte 31
	%C = sub int 0, %B         ; == ushr A, 31
	ret int %C 
}

uint %test13(uint %A) {
	%B = shr uint %A, ubyte 31
	%C = sub uint 0, %B        ; == sar A, 31
	ret uint %C
}

int %test14(uint %A) {
        %B = shr uint %A, ubyte 31
        %C = cast uint %B to int
        %D = sub int 0, %C
        ret int %D
}

int %test15(int %A, int %B) {
	%C = sub int 0, %A
	%D = rem int %B, %C   ;; X % -Y === X % Y
	ret int %D
}

int %test16(int %A) {
	%X = div int %A, 1123
	%Y = sub int 0, %X
	ret int %Y
}

int %test17(int %A) {
	%B = sub int 0, %A
	%C = div int %B, 1234
	ret int %C
}

long %test18(long %Y) {
        %tmp.4 = shl long %Y, ubyte 2
        %tmp.12 = shl long %Y, ubyte 2
        %tmp.8 = sub long %tmp.4, %tmp.12 ;; 0
        ret long %tmp.8
}

int %test19(int %X, int %Y) {
	%Z = sub int %X, %Y
	%Q = add int %Z, %Y
	ret int %Q
}

bool %test20(int %g, int %h) {
        %tmp.2 = sub int %g, %h
        %tmp.4 = setne int %tmp.2, %g
        ret bool %tmp.4
}

bool %test21(int %g, int %h) {
        %tmp.2 = sub int %g, %h
        %tmp.4 = setne int %tmp.2, %g
        ret bool %tmp.4
}


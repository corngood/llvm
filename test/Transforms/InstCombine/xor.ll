; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output &&
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | not grep 'xor '

%G1 = global uint 0
%G2 = global uint 0

implementation

bool %test0(bool %A) {
	%B = xor bool %A, false
	ret bool %B
}

int %test1(int %A) {
	%B = xor int %A, 0
	ret int %B
}

bool %test2(bool %A) {
	%B = xor bool %A, %A
	ret bool %B
}

int %test3(int %A) {
	%B = xor int %A, %A
	ret int %B
}

int %test4(int %A) {    ; A ^ ~A == -1
        %NotA = xor int -1, %A
        %B = xor int %A, %NotA
        ret int %B
}

uint %test5(uint %A) { ; (A|B)^B == A & (~B)
	%t1 = or uint %A, 123
	%r  = xor uint %t1, 123
	ret uint %r
}

ubyte %test6(ubyte %A) {
	%B = xor ubyte %A, 17
	%C = xor ubyte %B, 17
	ret ubyte %C
}

; (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
int %test7(int %A, int %B) {

        %A1 = and int %A, 7
        %B1 = and int %B, 128
        %C1 = xor int %A1, %B1
        ret int %C1
}

ubyte %test8(bool %c) {
	%d = xor bool %c, true    ; invert the condition
	br bool %d, label %True, label %False
True:
	ret ubyte 1
False:
	ret ubyte 3
}

bool %test9(ubyte %A) {
	%B = xor ubyte %A, 123      ; xor can be eliminated
	%C = seteq ubyte %B, 34
	ret bool %C
}

ubyte %test10(ubyte %A) {
	%B = and ubyte %A, 3
	%C = xor ubyte %B, 4        ; transform into an OR
	ret ubyte %C
}

ubyte %test11(ubyte %A) {
	%B = or ubyte %A, 12
	%C = xor ubyte %B, 4        ; transform into an AND
	ret ubyte %C
}

bool %test12(ubyte %A) {
	%B = xor ubyte %A, 4
	%c = setne ubyte %B, 0
	ret bool %c
}

bool %test13(ubyte %A, ubyte %B) {
	%C = setlt ubyte %A, %B
	%D = setgt ubyte %A, %B
	%E = xor bool %C, %D        ; E = setne %A, %B
	ret bool %E
}

bool %test14(ubyte %A, ubyte %B) {
	%C = seteq ubyte %A, %B
	%D = setne ubyte %B, %A
	%E = xor bool %C, %D        ; E = true
	ret bool %E
}

uint %test15(uint %A) {             ; ~(X-1) == -X
	%B = add uint %A, 4294967295
	%C = xor uint %B, 4294967295
	ret uint %C
}

uint %test16(uint %A) {             ; ~(X+c) == (-c-1)-X
	%B = add uint %A, 123       ; A generalization of the previous case
	%C = xor uint %B, 4294967295
	ret uint %C
}

uint %test17(uint %A) {             ; ~(c-X) == X-(c-1) == X+(-c+1)
	%B = sub uint 123, %A
	%C = xor uint %B, 4294967295
	ret uint %C
}

uint %test18(uint %A) {             ; C - ~X == X + (1+C)
	%B = xor uint %A, 4294967295; -~X == 0 - ~X == X+1
	%C = sub uint 123, %B
	ret uint %C
}

uint %test19(uint %A, uint %B) {
	%C = xor uint %A, %B
	%D = xor uint %C, %A  ; A terms cancel, D = B
	ret uint %D
}

void %test20(uint %A, uint %B) {  ; The "swap idiom"
        %tmp.2 = xor uint %B, %A
        %tmp.5 = xor uint %tmp.2, %B
        %tmp.8 = xor uint %tmp.5, %tmp.2
        store uint %tmp.8, uint* %G1   ; tmp.8 = B
        store uint %tmp.5, uint* %G2   ; tmp.5 = A
        ret void
}

int %test21(bool %C, int %A, int %B) {
	%C2 = xor bool %C, true
	%D = select bool %C2, int %A, int %B
	ret int %D
}

int %test22(bool %X) {
        %Y = xor bool %X, true
        %Z = cast bool %Y to int
        %Q = xor int %Z, 1
        ret int %Q
}

bool %test23(int %a, int %b) {
        %tmp.2 = xor int %b, %a
        %tmp.4 = seteq int %tmp.2, %a
        ret bool %tmp.4
}

bool %test24(int %c, int %d) {
        %tmp.2 = xor int %d, %c
        %tmp.4 = setne int %tmp.2, %c
        ret bool %tmp.4
}

int %test25(int %g, int %h) {
	%h2 = xor int %h, -1
        %tmp2 = and int %h2, %g
        %tmp4 = xor int %tmp2, %g  ; (h2&g)^g -> ~h2 & g -> h & g
        ret int %tmp4
}

int %test26(int %a, int %b) {
	%b2 = xor int %b, -1
        %tmp2 = xor int %a, %b2
        %tmp4 = and int %tmp2, %a  ; (a^b2)&a -> ~b2 & a -> b & a
        ret int %tmp4
}


i32 %test27(i32 %b, i32 %c, i32 %d) {
        %tmp2 = xor i32 %d, %b
        %tmp5 = xor i32 %d, %c
        %tmp = icmp eq i32 %tmp2, %tmp5
        %tmp6 = zext bool %tmp to i32
        ret i32 %tmp6
}


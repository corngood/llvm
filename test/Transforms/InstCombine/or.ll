; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine | dis | grep -v '%OROK = or' | grep or\ 
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int %test1(int %A) {
	%B = or int %A, 0
	ret int %B
}

int %test2(int %A) {
	%B = or int %A, -1
	ret int %B
}

ubyte %test2a(ubyte %A) {
	%B = or ubyte %A, 255
	ret ubyte %B
}

bool %test3(bool %A) {
	%B = or bool %A, false
	ret bool %B
}

bool %test4(bool %A) {
	%B = or bool %A, true
	ret bool %B
}

bool %test5(bool %A) {
	%B = xor bool %A, false
	ret bool %B
}

int %test6(int %A) {
	%B = xor int %A, 0
	ret int %B
}

bool %test7(bool %A) {
	%B = xor bool %A, %A
	ret bool %B
}

int %test8(int %A) {
	%B = xor int %A, %A
	ret int %B
}

bool %test9(bool %A) {
	%B = or bool %A, %A
	ret bool %B
}

int %test10(int %A) {
	%B = or int %A, %A
	ret int %B
}

int %test11(int %A) {    ; A ^ ~A == -1
        %NotA = xor int -1, %A
        %B = xor int %A, %NotA
        ret int %B
}

int %test12(int %A) {    ; A | ~A == -1
        %NotA = xor int -1, %A
        %B = or int %A, %NotA
        ret int %B
}

uint %test13(uint %A) { ; (A|B)^B == A & (~B)
	%t1 = or uint %A, 123
	%r  = xor uint %t1, 123
	ret uint %r
}

ubyte %test14(ubyte %A) {
	%B = or ubyte %A, 254
	%C = or ubyte %B, 1
	ret ubyte %C
}
ubyte %test15(ubyte %A) {
	%B = xor ubyte %A, 17
	%C = xor ubyte %B, 17
	ret ubyte %C
}

int %test16(int %A, int %B) {     ; (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
        %A1 = and int %A, 7
        %B1 = and int %B, 128
        %OROK = xor int %A1, %B1
        ret int %OROK
}

ubyte %test17(ubyte %A, ubyte %B) {  ; Test that (A|c1)|(B|c2) == (A|B)|(c1|c2)
	%C = or ubyte %A, 1
	%D = or ubyte %B, 254
	%E = or ubyte %C, %D
	ret ubyte %E
}

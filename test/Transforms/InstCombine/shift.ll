; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep sh

implementation

int %test1(int %A) {
	%B = shl int %A, ubyte 0
	ret int %B
}

int %test2(ubyte %A) {
	%B = shl int 0, ubyte %A
	ret int %B
}

int %test3(int %A) {
	%B = shr int %A, ubyte 0
	ret int %B
}

int %test4(ubyte %A) {
	%B = shr int 0, ubyte %A
	ret int %B
}

uint %test5(uint %A) {
	%B = shr uint %A, ubyte 32  ;; shift all bits out
	ret uint %B
}

uint %test5a(uint %A) {
	%B = shl uint %A, ubyte 32  ;; shift all bits out
	ret uint %B
}

uint %test6(uint %A) {
	%B = shl uint %A, ubyte 1   ;; convert to an mul instruction
	%C = mul uint %B, 3
	ret uint %C
}

int %test7(ubyte %A) {
	%B = shr int -1, ubyte %A   ;; Always equal to -1
	ret int %B
}

ubyte %test8(ubyte %A) {              ;; (A << 5) << 3 === A << 8 == 0
	%B = shl ubyte %A, ubyte 5
	%C = shl ubyte %B, ubyte 3
	ret ubyte %C
}

ubyte %test9(ubyte %A) {              ;; (A << 7) >> 7 === A & 1
	%B = shl ubyte %A, ubyte 7
	%C = shr ubyte %B, ubyte 7
	ret ubyte %C
}

ubyte %test10(ubyte %A) {              ;; (A >> 7) << 7 === A & 128
	%B = shr ubyte %A, ubyte 7
	%C = shl ubyte %B, ubyte 7
	ret ubyte %C
}

ubyte %test11(ubyte %A) {              ;; (A >> 3) << 4 === (A & 0x1F) << 1
	%a = mul ubyte %A, 3
	%B = shr ubyte %a, ubyte 3
	%C = shl ubyte %B, ubyte 4
	ret ubyte %C
}

int %test12(int %A) {
        %B = shr int %A, ubyte 8    ;; (A >> 8) << 8 === A & -256
        %C = shl int %B, ubyte 8
        ret int %C
}

sbyte %test13(sbyte %A) {           ;; (A >> 3) << 4 === (A & -8) * 2
	%a = mul sbyte %A, 3
	%B = shr sbyte %a, ubyte 3
	%C = shl sbyte %B, ubyte 4
	ret sbyte %C
}

uint %test14(uint %A) {
	%B = shr uint %A, ubyte 4
	%C = or uint %B, 1234
	%D = shl uint %C, ubyte 4   ;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
	ret uint %D
}
uint %test14a(uint %A) {
	%B = shl uint %A, ubyte 4
	%C = and uint %B, 1234
	%D = shr uint %C, ubyte 4   ;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
	ret uint %D
}

int %test15(bool %C) {
        %A = select bool %C, int 3, int 1
        %V = shl int %A, ubyte 2
        ret int %V
}

int %test15a(bool %C) {
        %A = select bool %C, ubyte 3, ubyte 1
        %V = shl int 64, ubyte %A
        ret int %V
}

bool %test16(int %X) {
        %tmp.3 = shr int %X, ubyte 4
        %tmp.6 = and int %tmp.3, 1
        %tmp.7 = setne int %tmp.6, 0  ;; X & 16 != 0
        ret bool %tmp.7
}

bool %test17(uint %A) {
	%B = shr uint %A, ubyte 3
	%C = seteq uint %B, 1234
	ret bool %C
}

bool %test18(ubyte %A) {
	%B = shr ubyte %A, ubyte 7
	%C = seteq ubyte %B, 123    ;; false
	ret bool %C
}

bool %test19(int %A) {
	%B = shr int %A, ubyte 2
	%C = seteq int %B, 0        ;; (X & -4) == 0
	ret bool %C
}

bool %test19a(int %A) {
	%B = shr int %A, ubyte 2
	%C = seteq int %B, -1        ;; (X & -4) == -4
	ret bool %C
}

bool %test20(sbyte %A) {
	%B = shr sbyte %A, ubyte 7
	%C = seteq sbyte %B, 123    ;; false
	ret bool %C
}

bool %test21(ubyte %A) {
	%B = shl ubyte %A, ubyte 4
	%C = seteq ubyte %B, 128
	ret bool %C
}

bool %test22(ubyte %A) {
	%B = shl ubyte %A, ubyte 4
	%C = seteq ubyte %B, 0
	ret bool %C
}

sbyte %test23(int %A) {
	%B = shl int %A, ubyte 24  ;; casts not needed
	%C = shr int %B, ubyte 24
	%D = cast int %C to sbyte
	ret sbyte %D
}

sbyte %test24(sbyte %X) {
        %Y = and sbyte %X, -5 ; ~4
        %Z = shl sbyte %Y, ubyte 5
        %Q = shr sbyte %Z, ubyte 5
        ret sbyte %Q
}

uint %test25(uint %tmp.2, uint %AA) {
	%x = shr uint %AA, ubyte 17
        %tmp.3 = shr uint %tmp.2, ubyte 17              ; <uint> [#uses=1]
        %tmp.5 = add uint %tmp.3, %x            ; <uint> [#uses=1]
        %tmp.6 = shl uint %tmp.5, ubyte 17              ; <uint> [#uses=1]
	ret uint %tmp.6
}


; This test makes sure that mul instructions are properly eliminated.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | not grep mul
; END.

implementation

int %test1(int %A) {
	%B = mul int %A, 1
	ret int %B
}

int %test2(int %A) {
	%B = mul int %A, 2   ; Should convert to an add instruction
	ret int %B
}

int %test3(int %A) {
	%B = mul int %A, 0   ; This should disappear entirely
	ret int %B
}

double %test4(double %A) {
	%B = mul double 1.0, %A   ; This is safe for FP
	ret double %B
}

int %test5(int %A) {
	%B = mul int %A, 8
	ret int %B
}

ubyte %test6(ubyte %A) {
	%B = mul ubyte %A, 8
	%C = mul ubyte %B, 8
	ret ubyte %C
}

int %test7(int %i) {
        %tmp = mul int %i, -1   ; %tmp = sub 0, %i
        ret int %tmp
}

ulong %test8(ulong %i) {
	%j = mul ulong %i, 18446744073709551615 ; tmp = sub 0, %i
	ret ulong %j
}

uint %test9(uint %i) {
	%j = mul uint %i, 4294967295    ; %j = sub 0, %i
	ret uint %j
}

uint %test10(int %a, uint %b) {
	%c = setlt int %a, 0
	%d = cast bool %c to uint
	%e = mul uint %d, %b           ; e = b & (a >> 31)
	ret uint %e
}

uint %test11(int %a, uint %b) {
	%c = setle int %a, -1
	%d = cast bool %c to uint
	%e = mul uint %d, %b           ; e = b & (a >> 31)
	ret uint %e
}

uint %test12(ubyte %a, uint %b) {
	%c = setgt ubyte %a, 127
	%d = cast bool %c to uint
	%e = mul uint %d, %b           ; e = b & (a >> 31)
	ret uint %e
}


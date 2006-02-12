; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep set

%X = uninitialized global int

bool %test1(int %A) {
	%B = seteq int %A, %A
	%C = seteq int* %X, null   ; Never true
	%D = and bool %B, %C
	ret bool %D
}

bool %test2(int %A) {
	%B = setne int %A, %A
	%C = setne int* %X, null   ; Never false
	%D = or bool %B, %C
	ret bool %D
}

bool %test3(int %A) {
	%B = setlt int %A, %A
	ret bool %B
}

bool %test4(int %A) {
	%B = setgt int %A, %A
	ret bool %B
}

bool %test5(int %A) {
	%B = setle int %A, %A
	ret bool %B
}

bool %test6(int %A) {
	%B = setge int %A, %A
	ret bool %B
}

bool %test7(uint %A) {
	%B = setge uint %A, 0  ; true
	ret bool %B
}

bool %test8(uint %A) {
	%B = setlt uint %A, 0  ; false
	ret bool %B
}

;; test operations on boolean values these should all be eliminated$a
bool %test9(bool %A) {
	%B = setlt bool %A, false ; false
	ret bool %B
}
bool %test10(bool %A) {
	%B = setgt bool %A, true  ; false
	ret bool %B
}
bool %test11(bool %A) {
	%B = setle bool %A, true ; true
	ret bool %B
}
bool %test12(bool %A) {
	%B = setge bool %A, false  ; true
	ret bool %B
}
bool %test13(bool %A, bool %B) {
	%C = setge bool %A, %B       ; A | ~B
	ret bool %C
}
bool %test14(bool %A, bool %B) {
	%C = seteq bool %A, %B  ; ~(A ^ B)
	ret bool %C
}

bool %test16(uint %A) {
	%B = and uint %A, 5
	%C = seteq uint %B, 8    ; Is never true
	ret bool %C
}

bool %test17(ubyte %A) {
	%B = or ubyte %A, 1
	%C = seteq ubyte %B, 2   ; Always false
	ret bool %C
}

bool %test18(bool %C, int %a) {
entry:
        br bool %C, label %endif, label %else

else:
        br label %endif

endif:
        %b.0 = phi int [ 0, %entry ], [ 1, %else ]
        %tmp.4 = setlt int %b.0, 123
        ret bool %tmp.4
}

bool %test19(bool %A, bool %B) {
	%a = cast bool %A to int
	%b = cast bool %B to int
	%C = seteq int %a, %b
	ret bool %C
}

uint %test20(uint %A) {
        %B = and uint %A, 1
        %C = setne uint %B, 0
        %D = cast bool %C to uint
        ret uint %D
}

int %test21(int %a) {
        %tmp.6 = and int %a, 4
        %not.tmp.7 = setne int %tmp.6, 0
        %retval = cast bool %not.tmp.7 to int
        ret int %retval
}

bool %test22(uint %A, int %X) {
        %B = and uint %A, 100663295
        %C = setlt uint %B, 268435456
	%Y = and int %X, 7
	%Z = setgt int %Y, -1
	%R = or bool %C, %Z
	ret bool %R
}

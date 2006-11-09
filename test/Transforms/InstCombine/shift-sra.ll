; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep 'lshr int' | wc -l | grep 2 &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep ashr

int %test1(int %X, ubyte %A) {
	%Y = shr int %X, ubyte %A  ; can be logical shift.
	%Z = and int %Y, 1
	ret int %Z
}

int %test2(ubyte %tmp) {
        %tmp3 = cast ubyte %tmp to int
        %tmp4 = add int %tmp3, 7
        %tmp5 = ashr int %tmp4, ubyte 3   ; lshr
        ret int %tmp5
}


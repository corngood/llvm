; An integer truncation to i1 should be done with an and instruction to make
; sure only the LSBit survives. Test that this is the case both for a returned
; value and as the operand of a branch.
; RUN: llvm-as < %s | llc -march=x86 &&
; RUN: llvm-as < %s | llc -march=x86 | grep '\(and\)\|\(test.*\$1\)' | \
; RUN:   wc -l | grep 6

define i1 @test1(i32 %X) zext {
    %Y = trunc i32 %X to i1
    ret i1 %Y
}

define i1 @test2(i32 %val, i32 %mask) {
entry:
    %maski8 = trunc i32 %mask to i8
    %shifted = ashr i32 %val, i8 %maski8
    %anded = and i32 %shifted, 1
    %trunced = trunc i32 %anded to i1
    br i1 %trunced, label %ret_true, label %ret_false
ret_true:
    ret i1 true
ret_false:
    ret i1 false
}

define i32 @test3(i8* %ptr) {
    %val = load i8* %ptr
    %tmp = trunc i8 %val to i1
    br i1 %tmp, label %cond_true, label %cond_false
cond_true:
    ret i32 21
cond_false:
    ret i32 42
}

define i32 @test4(i8* %ptr) {
    %tmp = ptrtoint i8* %ptr to i1
    br i1 %tmp, label %cond_true, label %cond_false
cond_true:
    ret i32 21
cond_false:
    ret i32 42
}

define i32 @test5(float %f) {
    %tmp = fptoui float %f to i1
    br i1 %tmp, label %cond_true, label %cond_false
cond_true:
    ret i32 21
cond_false:
    ret i32 42
}

define i32 @test6(double %d) {
    %tmp = fptosi double %d to i1
    br i1 %tmp, label %cond_true, label %cond_false
cond_true:
    ret i32 21
cond_false:
    ret i32 42
}


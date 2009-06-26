; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | grep {bic\\W*r\[0-9\]*,\\W*r\[0-9\]*,\\W*r\[0-9\]*} | Count 2

define i32 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %tmp, %a
    ret i32 %tmp1
}

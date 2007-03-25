; RUN: llvm-as < %s | llc -march=ppc64 -mcpu=g5 | grep cntlzd

define i32 @_ZNK4llvm5APInt17countLeadingZerosEv(i64 *%t) {
        %tmp19 = load i64* %t
        %tmp23 = tail call i64 @llvm.ctlz.i64( i64 %tmp19 )             ; <i64> [#uses=1]
        %tmp2324 = trunc i64 %tmp23 to i32              ; <i32> [#uses=1]
        %tmp89 = add i32 %tmp2324, -64          ; <i32> [#uses=1]
        %tmp90 = add i32 %tmp89, 0              ; <i32> [#uses=1]
        ret i32 %tmp90
}

declare i64 @llvm.ctlz.i64(i64)


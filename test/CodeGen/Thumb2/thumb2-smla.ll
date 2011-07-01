; RUN: llc < %s -march=thumb -mattr=+thumb2,+t2xtpk,+t2dsp | FileCheck %s

define i32 @f3(i32 %a, i16 %x, i32 %y) {
; CHECK: f3
; CHECK: smlabt r0, r1, r2, r0
        %tmp = sext i16 %x to i32               ; <i32> [#uses=1]
        %tmp2 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp3 = mul i32 %tmp2, %tmp             ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp3, %a               ; <i32> [#uses=1]
        ret i32 %tmp5
}

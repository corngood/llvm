; RUN: llvm-as < %s | llc -march=x86 | grep {leal}
; XFAIL: *
; This test is XFAIL'd because strength-reduction was improved to
; avoid emitting the lea, so it longer tests whether the 16-bit
; lea is avoided.

@X = global i16 0               ; <i16*> [#uses=1]
@Y = global i16 0               ; <i16*> [#uses=1]

define void @_Z3fooi(i32 %N) {
entry:
        %tmp1019 = icmp sgt i32 %N, 0           ; <i1> [#uses=1]
        br i1 %tmp1019, label %bb, label %return

bb:             ; preds = %bb, %entry
        %i.014.0 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]         ; <i32> [#uses=2]
        %tmp1 = trunc i32 %i.014.0 to i16               ; <i16> [#uses=2]
        volatile store i16 %tmp1, i16* @X, align 2
        %tmp34 = shl i16 %tmp1, 2               ; <i16> [#uses=1]
        volatile store i16 %tmp34, i16* @Y, align 2
        %indvar.next = add i32 %i.014.0, 1              ; <i32> [#uses=2]
        %exitcond = icmp eq i32 %indvar.next, %N                ; <i1> [#uses=1]
        br i1 %exitcond, label %return, label %bb

return:         ; preds = %bb, %entry
        ret void
}

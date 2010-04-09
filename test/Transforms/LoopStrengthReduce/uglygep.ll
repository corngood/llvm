; RUN: opt < %s -loop-reduce -S | not grep uglygep

; LSR shouldn't consider %t8 to be an interesting user of %t6, and it
; should be able to form pretty GEPs.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @Z4() nounwind {
bb:
  br label %bb3

bb1:                                              ; preds = %bb3
  br i1 undef, label %bb10, label %bb2

bb2:                                              ; preds = %bb1
  %t = add i64 %t4, 1                         ; <i64> [#uses=1]
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %t4 = phi i64 [ %t, %bb2 ], [ 0, %bb ]      ; <i64> [#uses=3]
  br label %bb1

bb10:                                             ; preds = %bb9
  %t7 = icmp eq i64 %t4, 0                    ; <i1> [#uses=1]
  %t3 = add i64 %t4, 16                     ; <i64> [#uses=1]
  br label %bb14

bb14:                                             ; preds = %bb14, %bb10
  %t2 = getelementptr inbounds i8* undef, i64 %t4 ; <i8*> [#uses=1]
  store i8 undef, i8* %t2
  %t6 = load float** undef
  %t8 = bitcast float* %t6 to i8*              ; <i8*> [#uses=1]
  %t9 = getelementptr inbounds i8* %t8, i64 %t3 ; <i8*> [#uses=1]
  store i8 undef, i8* %t9
  br label %bb14
}

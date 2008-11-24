; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep {fsmbi.*61680}   %t1.s | count 1
; RUN: grep rotqmbyi         %t1.s | count 1
; RUN: grep rotmai           %t1.s | count 1
; RUN: grep selb             %t1.s | count 1
; RUN: grep shufb            %t1.s | count 2
; RUN: grep cg               %t1.s | count 1
; RUN: grep addx             %t1.s | count 1

; ModuleID = 'stores.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i64 @sext_i64_i32(i32 %a) nounwind {
  %1 = sext i32 %a to i64
  ret i64 %1
}

define i64 @zext_i64_i32(i32 %a) nounwind {
  %1 = zext i32 %a to i64
  ret i64 %1
}

define i64 @add_i64(i64 %a, i64 %b) nounwind {
  %1 = add i64 %a, %b
  ret i64 %1
}

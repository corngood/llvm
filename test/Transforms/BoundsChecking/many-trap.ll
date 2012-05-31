; RUN: opt < %s -bounds-checking -bounds-checking-multiple-traps -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK: @f1
define void @f1(i64 %x) nounwind {
  %1 = alloca i128, i64 %x
  %2 = load i128* %1, align 4
  %3 = load i128* %1, align 4
  ret void
; CHECK: llvm.trap
; CHECK: llvm.trap
}

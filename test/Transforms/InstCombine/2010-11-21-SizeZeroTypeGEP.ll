; RUN: opt < %s -instcombine -S | not grep getelementptr

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define {}* @foo({}* %x, i32 %n) {
  %p = getelementptr {}* %x, i32 %n
  ret {}* %p
}

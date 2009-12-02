; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

declare i32 @llvm.ctpop.i32(i32)
declare i32 @llvm.ctlz.i32(i32)
declare i32 @llvm.cttz.i32(i32)

define i64 @test1(i32 %x) {
  %t = call i32 @llvm.ctpop.i32(i32 %x)
  %s = sext i32 %t to i64
  ret i64 %s
  
; CHECK: @test1
; CHECK: zext i32 %t
}

define i64 @test2(i32 %x) {
  %t = call i32 @llvm.ctlz.i32(i32 %x)
  %s = sext i32 %t to i64
  ret i64 %s

; CHECK: @test2
; CHECK: zext i32 %t
}

define i64 @test3(i32 %x) {
  %t = call i32 @llvm.cttz.i32(i32 %x)
  %s = sext i32 %t to i64
  ret i64 %s

; CHECK: @test3
; CHECK: zext i32 %t
}

define i64 @test4(i32 %x) {
  %t = udiv i32 %x, 3
  %s = sext i32 %t to i64
  ret i64 %s

; CHECK: @test4
; CHECK: zext i32 %t
}

define i64 @test5(i32 %x) {
  %t = urem i32 %x, 30000
  %s = sext i32 %t to i64
  ret i64 %s
; CHECK: @test5
; CHECK: zext i32 %t
}

define i64 @test6(i32 %x) {
  %u = lshr i32 %x, 3
  %t = mul i32 %u, 3
  %s = sext i32 %t to i64
  ret i64 %s
; CHECK: @test6
; CHECK: zext i32 %t
}

define i64 @test7(i32 %x) {
  %t = and i32 %x, 511
  %u = sub i32 20000, %t
  %s = sext i32 %u to i64
  ret i64 %s
; CHECK: @test7
; CHECK: zext i32 %u to i64
}

define i32 @test8(i8 %a, i32 %f, i1 %p, i32* %z) {
  %d = lshr i32 %f, 24
  %e = select i1 %p, i32 %d, i32 0
  %s = trunc i32 %e to i16
  %n = sext i16 %s to i32
  ret i32 %n
; CHECK: @test8
; CHECK: %d = lshr i32 %f, 24
; CHECK: %n = select i1 %p, i32 %d, i32 0
; CHECK: ret i32 %n
}

; rdar://6013816
define i16 @test9(i16 %t, i1 %cond) nounwind {
entry:
	br i1 %cond, label %T, label %F
T:
	%t2 = sext i16 %t to i32
	br label %F

F:
	%V = phi i32 [%t2, %T], [42, %entry]
	%W = trunc i32 %V to i16
	ret i16 %W
; CHECK: @test9
; CHECK: T:
; CHECK-NEXT: br label %F
; CHECK: F:
; CHECK-NEXT: phi i16
; CHECK-NEXT: ret i16
}

; PR2638
define i32 @test10(i32 %i) nounwind  {
entry:
        %tmp12 = trunc i32 %i to i8
        %tmp16 = shl i8 %tmp12, 6
        %a = ashr i8 %tmp16, 6 
        %b = sext i8 %a to i32 
        ret i32 %b
; CHECK: @test10
; CHECK:  shl i32 %i, 30
; CHECK-NEXT: ashr i32
; CHECK-NEXT: ret i32
}


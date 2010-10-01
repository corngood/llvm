; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define void @test1(i8* %a) {
        tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i32 1, i1 false)
        ret void
; CHECK: define void @test1
; CHECK-NEXT: ret void
}


; PR8267
define void @test2(i8* %a) {
        tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i32 1, i1 true)
        ret void
; CHECK: define void @test2
; CHECK-NEXT: call void @llvm.memcpy
}

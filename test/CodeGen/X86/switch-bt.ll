; RUN: llc -march=x86-64 -asm-verbose=false < %s | FileCheck %s

; This switch should use bit tests, and the third bit test case is just
; testing for one possible value, so it doesn't need a bt.

;      CHECK: movabsq $2305843009482129440, %r
; CHECK-NEXT: btq %rax, %r
; CHECK-NEXT: jb
; CHECK-NEXT: movl  $671088640, %e
; CHECK-NEXT: btq %rax, %r
; CHECK-NEXT: jb
; CHECK-NEXT: testq %rax, %r
; CHECK-NEXT: j

define void @test(i8* %l) nounwind {
entry:
  %l.addr = alloca i8*, align 8                   ; <i8**> [#uses=2]
  store i8* %l, i8** %l.addr
  %tmp = load i8** %l.addr                        ; <i8*> [#uses=1]
  %tmp1 = load i8* %tmp                           ; <i8> [#uses=1]
  %conv = sext i8 %tmp1 to i32                    ; <i32> [#uses=1]
  switch i32 %conv, label %sw.default [
    i32 62, label %sw.bb
    i32 60, label %sw.bb
    i32 38, label %sw.bb2
    i32 94, label %sw.bb2
    i32 61, label %sw.bb2
    i32 33, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry, %entry
  call void @foo(i32 0)
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry, %entry, %entry
  call void @foo(i32 1)
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  call void @foo(i32 3)
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  call void @foo(i32 97)
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb4, %sw.bb2, %sw.bb
  ret void
}

declare void @foo(i32)

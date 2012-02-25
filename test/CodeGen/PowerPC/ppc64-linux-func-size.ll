; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=g5 | FileCheck %s

; CHECK: test1:
; CHECK-NEXT:  .quad .L.test1,.TOC.@tocbase
; CHECK-NEXT:  .previous
; CHECK-NEXT: .L.test1:

define i32 @test1(i32 %a) nounwind {
entry:
  ret i32 %a
}

; Until recently, binutils accepted the .size directive as:
;  .size	test1, .Ltmp0-test1
; however, using this directive with recent binutils will result in the error:
;  .size expression for XXX does not evaluate to a constant
; so we must use the label which actually tags the start of the function.
; CHECK: .size	test1, .Ltmp0-.L.test1

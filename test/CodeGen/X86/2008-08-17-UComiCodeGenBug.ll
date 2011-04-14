; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define i32 @foo(<4 x float> %a, <4 x float> %b) nounwind {
entry:
; CHECK: movzx
  tail call i32 @llvm.x86.sse.ucomige.ss( <4 x float> %a, <4 x float> %b ) nounwind readnone
  ret i32 %0
}

declare i32 @llvm.x86.sse.ucomige.ss(<4 x float>, <4 x float>) nounwind readnone

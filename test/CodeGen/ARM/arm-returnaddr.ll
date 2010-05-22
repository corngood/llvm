; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s
; rdar://8015977

define arm_apcscc i8* @rt0(i32 %x) nounwind readnone {
entry:
; CHECK: rt0:
; CHECK: mov r0, lr
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define arm_apcscc i8* @rt2() nounwind readnone {
entry:
; CHECK: rt2:
; CHECK: ldr r0, [r7]
; CHECK: ldr r0, [r0]
; CHECK: ldr r0, [r0, #4]
  %0 = tail call i8* @llvm.returnaddress(i32 2)
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone

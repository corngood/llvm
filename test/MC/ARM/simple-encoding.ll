;RUN: llc -mtriple=armv7-apple-darwin -show-mc-encoding < %s | FileCheck %s


;FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;       should run on .s source files rather than using llc to generate the
;       assembly.

define i32 @foo(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: foo
; CHECK: trap                         @ encoding: [0xf0,0x00,0xf0,0x07]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]

  tail call void @llvm.trap()
  ret i32 undef
}

define i32 @f2(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: f2
; CHECK: add  r0, r1, r0              @ encoding: [0x00,0x00,0x81,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %add = add nsw i32 %b, %a
  ret i32 %add
}


define i32 @f3(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: f3
; CHECK: add  r0, r0, r1, lsl #3      @ encoding: [0x81,0x01,0x80,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %mul = shl i32 %b, 3
  %add = add nsw i32 %mul, %a
  ret i32 %add
}

define i32 @f4(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: f4
; CHECK: add r0, r0, #254, 28         @ encoding: [0xfe,0x0e,0x80,0xe2]
; CHECK:                              @ 4064
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %add = add nsw i32 %a, 4064
  ret i32 %add
}

define i32 @f5(i32 %a, i32 %b, i32 %c) nounwind readnone ssp {
entry:
; CHECK: f5
; CHECK: cmp r0, r1                   @ encoding: [0x01,0x00,0x50,0xe1]
; CHECK: mov r0, r2                   @ encoding: [0x02,0x00,0xa0,0xe1]
; CHECK: movgt r0, r1                 @ encoding: [0x01,0x00,0xa0,0xc1]
  %cmp = icmp sgt i32 %a, %b
  %retval.0 = select i1 %cmp, i32 %b, i32 %c
  ret i32 %retval.0
}

define i64 @f6(i64 %a, i64 %b, i64 %c) nounwind readnone optsize ssp {
entry:
; CHECK: f6
; CHECK: adds r0, r2, r0              @ encoding: [0x00,0x00,0x92,0xe0]
; CHECK: adc r1, r3, r1               @ encoding: [0x01,0x10,0xa3,0xe0]
  %add = add nsw i64 %b, %a
  ret i64 %add
}

define i32 @f7(i32 %a, i32 %b) nounwind readnone optsize ssp {
entry:
; CHECK: f7
; CHECK: uxtab  r0, r0, r1            @ encoding: [0x71,0x00,0xe0,0xe6]
  %and = and i32 %b, 255
  %add = add i32 %and, %a
  ret i32 %add
}

define i32 @f8(i32 %a) nounwind readnone ssp {
entry:
; CHECK: f8
; CHECK: movt r0, #42405              @ encoding: [0xa5,0x05,0x4a,0xe3]
  %and = and i32 %a, 65535
  %or = or i32 %and, -1515913216
  ret i32 %or
}

define i32 @f9() nounwind readnone ssp {
entry:
; CHECK: f9
; CHECK: movw r0, #42405              @ encoding: [0xa5,0x05,0x0a,0xe3]
  ret i32 42405
}

define i64 @f10(i64 %a) nounwind readnone ssp {
entry:
; CHECK: f10
; CHECK: asrs  r1, r1, #1             @ encoding: [0xc1,0x10,0xb0,0xe1]
; CHECK: rrx r0, r0                   @ encoding: [0x60,0x00,0xa0,0xe1]
  %shr = ashr i64 %a, 1
  ret i64 %shr
}

declare void @llvm.trap() nounwind

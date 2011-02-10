; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: @sdiv1
; CHECK: sdiv i32 %x, 8
define i32 @sdiv1(i32 %x) {
  %y = sdiv i32 %x, 8
  ret i32 %y
}

; CHECK: @sdiv3
; CHECK: %y = srem i32 %x, 3
; CHECK: %z = sub i32 %x, %y
; CHECK: ret i32 %z
define i32 @sdiv3(i32 %x) {
  %y = sdiv i32 %x, 3
  %z = mul i32 %y, 3
  ret i32 %z
}

; CHECK: @sdiv4
; CHECK: ret i32 %x
define i32 @sdiv4(i32 %x) {
  %y = sdiv exact i32 %x, 3
  %z = mul i32 %y, 3
  ret i32 %z
}

; CHECK: i32 @sdiv5
; CHECK: %y = srem i32 %x, 3
; CHECK: %z = sub i32 %y, %x
; CHECK: ret i32 %z
define i32 @sdiv5(i32 %x) {
  %y = sdiv i32 %x, 3
  %z = mul i32 %y, -3
  ret i32 %z
}

; CHECK: @sdiv6
; CHECK: %z = sub i32 0, %x
; CHECK: ret i32 %z
define i32 @sdiv6(i32 %x) {
  %y = sdiv exact i32 %x, 3
  %z = mul i32 %y, -3
  ret i32 %z
}

; CHECK: @udiv1
; CHECK: ret i32 %x
define i32 @udiv1(i32 %x, i32 %w) {
  %y = udiv exact i32 %x, %w
  %z = mul i32 %y, %w
  ret i32 %z
}

; CHECK: @ashr_icmp
; CHECK: %B = icmp eq i64 %X, 0
; CHECK: ret i1 %B
define i1 @ashr_icmp(i64 %X) nounwind {
  %A = ashr exact i64 %X, 2   ; X/4
  %B = icmp eq i64 %A, 0
  ret i1 %B
}

; CHECK: @udiv_icmp1
; CHECK: icmp ne i64 %X, 0
define i1 @udiv_icmp1(i64 %X) nounwind {
  %A = udiv exact i64 %X, 5   ; X/5
  %B = icmp ne i64 %A, 0
  ret i1 %B
}

; CHECK: @sdiv_icmp1
; CHECK: icmp eq i64 %X, 0
define i1 @sdiv_icmp1(i64 %X) nounwind {
  %A = sdiv exact i64 %X, 5   ; X/5 == 0 --> x == 0
  %B = icmp eq i64 %A, 0
  ret i1 %B
}

; CHECK: @sdiv_icmp2
; CHECK: icmp eq i64 %X, 5
define i1 @sdiv_icmp2(i64 %X) nounwind {
  %A = sdiv exact i64 %X, 5   ; X/5 == 1 --> x == 5
  %B = icmp eq i64 %A, 1
  ret i1 %B
}

; CHECK: @sdiv_icmp3
; CHECK: icmp eq i64 %X, -5
define i1 @sdiv_icmp3(i64 %X) nounwind {
  %A = sdiv exact i64 %X, 5   ; X/5 == -1 --> x == -5
  %B = icmp eq i64 %A, -1
  ret i1 %B
}

; CHECK: @sdiv_icmp4
; CHECK: icmp eq i64 %X, 0
define i1 @sdiv_icmp4(i64 %X) nounwind {
  %A = sdiv exact i64 %X, -5   ; X/-5 == 0 --> x == 0
  %B = icmp eq i64 %A, 0
  ret i1 %B
}

; CHECK: @sdiv_icmp5
; CHECK: icmp eq i64 %X, -5
define i1 @sdiv_icmp5(i64 %X) nounwind {
  %A = sdiv exact i64 %X, -5   ; X/-5 == 1 --> x == -5
  %B = icmp eq i64 %A, 1
  ret i1 %B
}

; CHECK: @sdiv_icmp6
; CHECK: icmp eq i64 %X, 5
define i1 @sdiv_icmp6(i64 %X) nounwind {
  %A = sdiv exact i64 %X, -5   ; X/-5 == 1 --> x == 5
  %B = icmp eq i64 %A, -1
  ret i1 %B
}


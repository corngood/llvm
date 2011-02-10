; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A) {
; CHECK: @test1
; CHECK: ret i32 %A
        %B = shl i32 %A, 0              ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i8 %A) {
; CHECK: @test2
; CHECK: ret i32 0
        %shift.upgrd.1 = zext i8 %A to i32              ; <i32> [#uses=1]
        %B = shl i32 0, %shift.upgrd.1          ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test3(i32 %A) {
; CHECK: @test3
; CHECK: ret i32 %A
        %B = ashr i32 %A, 0             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test4(i8 %A) {
; CHECK: @test4
; CHECK: ret i32 0
        %shift.upgrd.2 = zext i8 %A to i32              ; <i32> [#uses=1]
        %B = ashr i32 0, %shift.upgrd.2         ; <i32> [#uses=1]
        ret i32 %B
}


define i32 @test5(i32 %A) {
; CHECK: @test5
; CHECK: ret i32 undef
        %B = lshr i32 %A, 32  ;; shift all bits out 
        ret i32 %B
}

define i32 @test5a(i32 %A) {
; CHECK: @test5a
; CHECK: ret i32 undef
        %B = shl i32 %A, 32     ;; shift all bits out 
        ret i32 %B
}

define i32 @test5b() {
; CHECK: @test5b
; CHECK: ret i32 -1
        %B = ashr i32 undef, 2  ;; top two bits must be equal, so not undef
        ret i32 %B
}

define i32 @test5b2(i32 %A) {
; CHECK: @test5b2
; CHECK: ret i32 -1
        %B = ashr i32 undef, %A  ;; top %A bits must be equal, so not undef
        ret i32 %B
}

define i32 @test6(i32 %A) {
; CHECK: @test6
; CHECK-NEXT: mul i32 %A, 6
; CHECK-NEXT: ret i32
        %B = shl i32 %A, 1      ;; convert to an mul instruction 
        %C = mul i32 %B, 3             
        ret i32 %C
}

define i32 @test7(i8 %A) {
; CHECK: @test7
; CHECK-NEXT: ret i32 -1
        %shift.upgrd.3 = zext i8 %A to i32 
        %B = ashr i32 -1, %shift.upgrd.3  ;; Always equal to -1
        ret i32 %B
}

;; (A << 5) << 3 === A << 8 == 0
define i8 @test8(i8 %A) {
; CHECK: @test8
; CHECK: ret i8 0
        %B = shl i8 %A, 5               ; <i8> [#uses=1]
        %C = shl i8 %B, 3               ; <i8> [#uses=1]
        ret i8 %C
}

;; (A << 7) >> 7 === A & 1
define i8 @test9(i8 %A) {
; CHECK: @test9
; CHECK-NEXT: and i8 %A, 1
; CHECK-NEXT: ret i8
        %B = shl i8 %A, 7               ; <i8> [#uses=1]
        %C = lshr i8 %B, 7              ; <i8> [#uses=1]
        ret i8 %C
}

;; (A >> 7) << 7 === A & 128
define i8 @test10(i8 %A) {
; CHECK: @test10
; CHECK-NEXT: and i8 %A, -128
; CHECK-NEXT: ret i8
        %B = lshr i8 %A, 7              ; <i8> [#uses=1]
        %C = shl i8 %B, 7               ; <i8> [#uses=1]
        ret i8 %C
}

;; (A >> 3) << 4 === (A & 0x1F) << 1
define i8 @test11(i8 %A) {
; CHECK: @test11
; CHECK-NEXT: mul i8 %A, 6
; CHECK-NEXT: and i8
; CHECK-NEXT: ret i8
        %a = mul i8 %A, 3               ; <i8> [#uses=1]
        %B = lshr i8 %a, 3              ; <i8> [#uses=1]
        %C = shl i8 %B, 4               ; <i8> [#uses=1]
        ret i8 %C
}

;; (A >> 8) << 8 === A & -256
define i32 @test12(i32 %A) {
; CHECK: @test12
; CHECK-NEXT: and i32 %A, -256
; CHECK-NEXT: ret i32
        %B = ashr i32 %A, 8             ; <i32> [#uses=1]
        %C = shl i32 %B, 8              ; <i32> [#uses=1]
        ret i32 %C
}

;; (A >> 3) << 4 === (A & -8) * 2
define i8 @test13(i8 %A) {
; CHECK: @test13
; CHECK-NEXT: mul i8 %A, 6
; CHECK-NEXT: and i8
; CHECK-NEXT: ret i8
        %a = mul i8 %A, 3               ; <i8> [#uses=1]
        %B = ashr i8 %a, 3              ; <i8> [#uses=1]
        %C = shl i8 %B, 4               ; <i8> [#uses=1]
        ret i8 %C
}

;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
define i32 @test14(i32 %A) {
; CHECK: @test14
; CHECK-NEXT: %B = and i32 %A, -19760
; CHECK-NEXT: or i32 %B, 19744
; CHECK-NEXT: ret i32
        %B = lshr i32 %A, 4             ; <i32> [#uses=1]
        %C = or i32 %B, 1234            ; <i32> [#uses=1]
        %D = shl i32 %C, 4              ; <i32> [#uses=1]
        ret i32 %D
}

;; D = ((B | 1234) << 4) === ((B << 4)|(1234 << 4)
define i32 @test14a(i32 %A) {
; CHECK: @test14a
; CHECK-NEXT: and i32 %A, 77
; CHECK-NEXT: ret i32
        %B = shl i32 %A, 4              ; <i32> [#uses=1]
        %C = and i32 %B, 1234           ; <i32> [#uses=1]
        %D = lshr i32 %C, 4             ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test15(i1 %C) {
; CHECK: @test15
; CHECK-NEXT: select i1 %C, i32 12, i32 4
; CHECK-NEXT: ret i32
        %A = select i1 %C, i32 3, i32 1         ; <i32> [#uses=1]
        %V = shl i32 %A, 2              ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test15a(i1 %C) {
; CHECK: @test15a
; CHECK-NEXT: select i1 %C, i32 512, i32 128
; CHECK-NEXT: ret i32
        %A = select i1 %C, i8 3, i8 1           ; <i8> [#uses=1]
        %shift.upgrd.4 = zext i8 %A to i32              ; <i32> [#uses=1]
        %V = shl i32 64, %shift.upgrd.4         ; <i32> [#uses=1]
        ret i32 %V
}

define i1 @test16(i32 %X) {
; CHECK: @test16
; CHECK-NEXT: and i32 %X, 16
; CHECK-NEXT: icmp ne i32
; CHECK-NEXT: ret i1
        %tmp.3 = ashr i32 %X, 4 
        %tmp.6 = and i32 %tmp.3, 1
        %tmp.7 = icmp ne i32 %tmp.6, 0
        ret i1 %tmp.7
}

define i1 @test17(i32 %A) {
; CHECK: @test17
; CHECK-NEXT: and i32 %A, -8
; CHECK-NEXT: icmp eq i32
; CHECK-NEXT: ret i1
        %B = lshr i32 %A, 3             ; <i32> [#uses=1]
        %C = icmp eq i32 %B, 1234               ; <i1> [#uses=1]
        ret i1 %C
}


define i1 @test18(i8 %A) {
; CHECK: @test18
; CHECK: ret i1 false

        %B = lshr i8 %A, 7              ; <i8> [#uses=1]
        ;; false
        %C = icmp eq i8 %B, 123         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test19(i32 %A) {
; CHECK: @test19
; CHECK-NEXT: icmp ult i32 %A, 4
; CHECK-NEXT: ret i1
        %B = ashr i32 %A, 2             ; <i32> [#uses=1]
        ;; (X & -4) == 0
        %C = icmp eq i32 %B, 0          ; <i1> [#uses=1]
        ret i1 %C
}


define i1 @test19a(i32 %A) {
; CHECK: @test19a
; CHECK-NEXT: and i32 %A, -4
; CHECK-NEXT: icmp eq i32
; CHECK-NEXT: ret i1
        %B = ashr i32 %A, 2             ; <i32> [#uses=1]
        ;; (X & -4) == -4
        %C = icmp eq i32 %B, -1         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test20(i8 %A) {
; CHECK: @test20
; CHECK: ret i1 false
        %B = ashr i8 %A, 7              ; <i8> [#uses=1]
        ;; false
        %C = icmp eq i8 %B, 123         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test21(i8 %A) {
; CHECK: @test21
; CHECK-NEXT: and i8 %A, 15
; CHECK-NEXT: icmp eq i8
; CHECK-NEXT: ret i1
        %B = shl i8 %A, 4               ; <i8> [#uses=1]
        %C = icmp eq i8 %B, -128                ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test22(i8 %A) {
; CHECK: @test22
; CHECK-NEXT: and i8 %A, 15
; CHECK-NEXT: icmp eq i8
; CHECK-NEXT: ret i1
        %B = shl i8 %A, 4               ; <i8> [#uses=1]
        %C = icmp eq i8 %B, 0           ; <i1> [#uses=1]
        ret i1 %C
}

define i8 @test23(i32 %A) {
; CHECK: @test23
; CHECK-NEXT: trunc i32 %A to i8
; CHECK-NEXT: ret i8

        ;; casts not needed
        %B = shl i32 %A, 24             ; <i32> [#uses=1]
        %C = ashr i32 %B, 24            ; <i32> [#uses=1]
        %D = trunc i32 %C to i8         ; <i8> [#uses=1]
        ret i8 %D
}

define i8 @test24(i8 %X) {
; CHECK: @test24
; CHECK-NEXT: and i8 %X, 3
; CHECK-NEXT: ret i8
        %Y = and i8 %X, -5              ; <i8> [#uses=1]
        %Z = shl i8 %Y, 5               ; <i8> [#uses=1]
        %Q = ashr i8 %Z, 5              ; <i8> [#uses=1]
        ret i8 %Q
}

define i32 @test25(i32 %tmp.2, i32 %AA) {
; CHECK: @test25
; CHECK-NEXT: and i32 %tmp.2, -131072
; CHECK-NEXT: add i32 %{{[^,]*}}, %AA
; CHECK-NEXT: and i32 %{{[^,]*}}, -131072
; CHECK-NEXT: ret i32
        %x = lshr i32 %AA, 17           ; <i32> [#uses=1]
        %tmp.3 = lshr i32 %tmp.2, 17            ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.3, %x             ; <i32> [#uses=1]
        %tmp.6 = shl i32 %tmp.5, 17             ; <i32> [#uses=1]
        ret i32 %tmp.6
}

;; handle casts between shifts.
define i32 @test26(i32 %A) {
; CHECK: @test26
; CHECK-NEXT: and i32 %A, -2
; CHECK-NEXT: ret i32
        %B = lshr i32 %A, 1             ; <i32> [#uses=1]
        %C = bitcast i32 %B to i32              ; <i32> [#uses=1]
        %D = shl i32 %C, 1              ; <i32> [#uses=1]
        ret i32 %D
}


define i1 @test27(i32 %x) nounwind {
; CHECK: @test27
; CHECK-NEXT: and i32 %x, 8
; CHECK-NEXT: icmp ne i32
; CHECK-NEXT: ret i1
  %y = lshr i32 %x, 3
  %z = trunc i32 %y to i1
  ret i1 %z
}
 
define i8 @test28(i8 %x) {
entry:
; CHECK: @test28
; CHECK:     icmp slt i8 %x, 0
; CHECK-NEXT:     br i1 
	%tmp1 = lshr i8 %x, 7
	%cond1 = icmp ne i8 %tmp1, 0
	br i1 %cond1, label %bb1, label %bb2

bb1:
	ret i8 0

bb2:
	ret i8 1
}

define i8 @test28a(i8 %x, i8 %y) {
entry:
; This shouldn't be transformed.
; CHECK: @test28a
; CHECK:     %tmp1 = lshr i8 %x, 7
; CHECK:     %cond1 = icmp eq i8 %tmp1, 0
; CHECK:     br i1 %cond1, label %bb2, label %bb1
	%tmp1 = lshr i8 %x, 7
	%cond1 = icmp ne i8 %tmp1, 0
	br i1 %cond1, label %bb1, label %bb2
bb1:
	ret i8 %tmp1
bb2:
        %tmp2 = add i8 %tmp1, %y
	ret i8 %tmp2
}


define i32 @test29(i64 %d18) {
entry:
	%tmp916 = lshr i64 %d18, 32
	%tmp917 = trunc i64 %tmp916 to i32
	%tmp10 = lshr i32 %tmp917, 31
	ret i32 %tmp10
; CHECK: @test29
; CHECK:  %tmp916 = lshr i64 %d18, 63
; CHECK:  %tmp10 = trunc i64 %tmp916 to i32
}


define i32 @test30(i32 %A, i32 %B, i32 %C) {
	%X = shl i32 %A, %C
	%Y = shl i32 %B, %C
	%Z = and i32 %X, %Y
	ret i32 %Z
; CHECK: @test30
; CHECK: %X1 = and i32 %A, %B
; CHECK: %Z = shl i32 %X1, %C
}

define i32 @test31(i32 %A, i32 %B, i32 %C) {
	%X = lshr i32 %A, %C
	%Y = lshr i32 %B, %C
	%Z = or i32 %X, %Y
	ret i32 %Z
; CHECK: @test31
; CHECK: %X1 = or i32 %A, %B
; CHECK: %Z = lshr i32 %X1, %C
}

define i32 @test32(i32 %A, i32 %B, i32 %C) {
	%X = ashr i32 %A, %C
	%Y = ashr i32 %B, %C
	%Z = xor i32 %X, %Y
	ret i32 %Z
; CHECK: @test32
; CHECK: %X1 = xor i32 %A, %B
; CHECK: %Z = ashr i32 %X1, %C
; CHECK: ret i32 %Z
}

define i1 @test33(i32 %X) {
        %tmp1 = shl i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
; CHECK: @test33
; CHECK: %tmp1.mask = and i32 %X, 16777216
; CHECK: %tmp2 = icmp ne i32 %tmp1.mask, 0
}

define i1 @test34(i32 %X) {
        %tmp1 = lshr i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
; CHECK: @test34
; CHECK: ret i1 false
}

define i1 @test35(i32 %X) {
        %tmp1 = ashr i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
; CHECK: @test35
; CHECK: %tmp2 = icmp slt i32 %X, 0
; CHECK: ret i1 %tmp2
}

define i128 @test36(i128 %A, i128 %B) {
entry:
  %tmp27 = shl i128 %A, 64
  %tmp23 = shl i128 %B, 64
  %ins = or i128 %tmp23, %tmp27
  %tmp45 = lshr i128 %ins, 64
  ret i128 %tmp45
  
; CHECK: @test36
; CHECK:  %tmp231 = or i128 %B, %A
; CHECK:  %ins = and i128 %tmp231, 18446744073709551615
; CHECK:  ret i128 %ins
}

define i64 @test37(i128 %A, i32 %B) {
entry:
  %tmp27 = shl i128 %A, 64
  %tmp22 = zext i32 %B to i128
  %tmp23 = shl i128 %tmp22, 96
  %ins = or i128 %tmp23, %tmp27
  %tmp45 = lshr i128 %ins, 64
  %tmp46 = trunc i128 %tmp45 to i64
  ret i64 %tmp46
  
; CHECK: @test37
; CHECK:  %tmp23 = shl nuw nsw i128 %tmp22, 32
; CHECK:  %ins = or i128 %tmp23, %A
; CHECK:  %tmp46 = trunc i128 %ins to i64
}

define i32 @test38(i32 %x) nounwind readnone {
  %rem = srem i32 %x, 32
  %shl = shl i32 1, %rem
  ret i32 %shl
; CHECK: @test38
; CHECK-NEXT: and i32 %x, 31
; CHECK-NEXT: shl i32 1
; CHECK-NEXT: ret i32
}

; <rdar://problem/8756731>
; CHECK: @test39
define i8 @test39(i32 %a0) {
entry:
  %tmp4 = trunc i32 %a0 to i8
; CHECK: and i8 %tmp49, 64
  %tmp5 = shl i8 %tmp4, 5
  %tmp48 = and i8 %tmp5, 32
  %tmp49 = lshr i8 %tmp48, 5
  %tmp50 = mul i8 %tmp49, 64
  %tmp51 = xor i8 %tmp50, %tmp5
; CHECK: and i8 %0, 16
  %tmp52 = and i8 %tmp51, -128
  %tmp53 = lshr i8 %tmp52, 7
  %tmp54 = mul i8 %tmp53, 16
  %tmp55 = xor i8 %tmp54, %tmp51
; CHECK: ret i8 %tmp551
  ret i8 %tmp55
}

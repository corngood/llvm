; RUN: llc -march=x86 %s -o -
; RUN: llc -march=x86-64 %s -o -

; PR6497

; Chain and flag folding issues.
define i32 @test1() nounwind ssp {
entry:
  %tmp5.i = volatile load i32* undef              ; <i32> [#uses=1]
  %conv.i = zext i32 %tmp5.i to i64               ; <i64> [#uses=1]
  %tmp12.i = volatile load i32* undef             ; <i32> [#uses=1]
  %conv13.i = zext i32 %tmp12.i to i64            ; <i64> [#uses=1]
  %shl.i = shl i64 %conv13.i, 32                  ; <i64> [#uses=1]
  %or.i = or i64 %shl.i, %conv.i                  ; <i64> [#uses=1]
  %add16.i = add i64 %or.i, 256                   ; <i64> [#uses=1]
  %shr.i = lshr i64 %add16.i, 8                   ; <i64> [#uses=1]
  %conv19.i = trunc i64 %shr.i to i32             ; <i32> [#uses=1]
  volatile store i32 %conv19.i, i32* undef
  ret i32 undef
}

; PR6533
define void @test2(i1 %x, i32 %y) nounwind {
  %land.ext = zext i1 %x to i32                   ; <i32> [#uses=1]
  %and = and i32 %y, 1                        ; <i32> [#uses=1]
  %xor = xor i32 %and, %land.ext                  ; <i32> [#uses=1]
  %cmp = icmp eq i32 %xor, 1                      ; <i1> [#uses=1]
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %land.end
  ret void

if.end:                                           ; preds = %land.end
  ret void
}

; PR6577
%pair = type { i64, double }

define void @test3() {
dependentGraph243.exit:
  %subject19 = load %pair* undef                     ; <%1> [#uses=1]
  %0 = extractvalue %pair %subject19, 1              ; <double> [#uses=2]
  %1 = select i1 undef, double %0, double undef   ; <double> [#uses=1]
  %2 = select i1 undef, double %1, double %0      ; <double> [#uses=1]
  %3 = insertvalue %pair undef, double %2, 1         ; <%1> [#uses=1]
  store %pair %3, %pair* undef
  ret void
}

; PR6605
define i64 @test4(i8* %P) nounwind ssp {
entry:
  %tmp1 = load i8* %P                           ; <i8> [#uses=3]
  %tobool = icmp eq i8 %tmp1, 0                   ; <i1> [#uses=1]
  %tmp58 = sext i1 %tobool to i8                  ; <i8> [#uses=1]
  %mul.i = and i8 %tmp58, %tmp1                   ; <i8> [#uses=1]
  %conv6 = zext i8 %mul.i to i32                  ; <i32> [#uses=1]
  %cmp = icmp ne i8 %tmp1, 1                      ; <i1> [#uses=1]
  %conv11 = zext i1 %cmp to i32                   ; <i32> [#uses=1]
  %call12 = tail call i32 @safe(i32 %conv11) nounwind ; <i32> [#uses=1]
  %and = and i32 %conv6, %call12                  ; <i32> [#uses=1]
  %tobool13 = icmp eq i32 %and, 0                 ; <i1> [#uses=1]
  br i1 %tobool13, label %if.else, label %return

if.else:                                          ; preds = %entry
  br label %return

return:                                           ; preds = %if.else, %entry
  ret i64 undef
}

declare i32 @safe(i32)

; PR6607
define fastcc void @test5(i32 %FUNC) nounwind {
foo:
  %0 = load i8* undef, align 1                    ; <i8> [#uses=3]
  %1 = sext i8 %0 to i32                          ; <i32> [#uses=2]
  %2 = zext i8 %0 to i32                          ; <i32> [#uses=1]
  %tmp1.i5037 = urem i32 %2, 10                   ; <i32> [#uses=1]
  %tmp.i5038 = icmp ugt i32 %tmp1.i5037, 15       ; <i1> [#uses=1]
  %3 = zext i1 %tmp.i5038 to i8                   ; <i8> [#uses=1]
  %4 = icmp slt i8 %0, %3                         ; <i1> [#uses=1]
  %5 = add nsw i32 %1, 256                        ; <i32> [#uses=1]
  %storemerge.i.i57 = select i1 %4, i32 %5, i32 %1 ; <i32> [#uses=1]
  %6 = shl i32 %storemerge.i.i57, 16              ; <i32> [#uses=1]
  %7 = sdiv i32 %6, -256                          ; <i32> [#uses=1]
  %8 = trunc i32 %7 to i8                         ; <i8> [#uses=1]
  store i8 %8, i8* undef, align 1
  ret void
}


;; Issues with referring to a label that gets RAUW'd later.
define i32 @test6a() nounwind {
entry:
	%target = bitcast i8* blockaddress(@test6b, %test_label) to i8*

	call i32 @test6b(i8* %target)

	ret i32 0
}

define i32 @test6b(i8* %target) nounwind {
entry:
	indirectbr i8* %target, [label %test_label]

test_label:
; assume some code here...
	br label %ret

ret:
	ret i32 -1
}



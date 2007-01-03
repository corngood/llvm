;test all the shifted and signextending adds and subs with and without consts
;
; RUN: llvm-as < %s | llc -march=alpha -o %t.s -f &&
; RUN: grep '	addl' %t.s | wc -l | grep 2 &&
; RUN: grep '	addq' %t.s | wc -l | grep 2 &&
; RUN: grep '	subl' %t.s | wc -l | grep 2 &&
; RUN: grep '	subq' %t.s | wc -l | grep 1 &&
;
; RUN: grep 'lda $0,-100($16)' %t.s | wc -l | grep 1 &&
; RUN: grep 's4addl' %t.s | wc -l | grep 2 &&
; RUN: grep 's8addl' %t.s | wc -l | grep 2 &&
; RUN: grep 's4addq' %t.s | wc -l | grep 2 &&
; RUN: grep 's8addq' %t.s | wc -l | grep 2 &&
;
; RUN: grep 's4subl' %t.s | wc -l | grep 2 &&
; RUN: grep 's8subl' %t.s | wc -l | grep 2 &&
; RUN: grep 's4subq' %t.s | wc -l | grep 2 &&
; RUN: grep 's8subq' %t.s | wc -l | grep 2

implementation   ; Functions:

define i32 @sext %al(i32 @sext %x.s, i32 @sext %y.s) {
entry:
	%tmp.3.s = add i32 %y.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @sext %ali(i32 @sext %x.s) {
entry:
	%tmp.3.s = add i32 100, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 @sext %aq(i64 @sext %x.s, i64 @sext %y.s) {
entry:
	%tmp.3.s = add i64 %y.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 %aqi(i64 %x.s) {
entry:
	%tmp.3.s = add i64 100, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @sext %sl(i32 @sext %x.s, i32 @sext %y.s) {
entry:
	%tmp.3.s = sub i32 %y.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @sext %sli(i32 @sext %x.s) {
entry:
	%tmp.3.s = sub i32 %x.s, 100		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 %sq(i64 %x.s, i64 %y.s) {
entry:
	%tmp.3.s = sub i64 %y.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 %sqi(i64 %x.s) {
entry:
	%tmp.3.s = sub i64 %x.s, 100		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @sext %a4l(i32 @sext %x.s, i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 2		; <i32> [#uses=1]
	%tmp.3.s = add i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @sext %a8l(i32 @sext %x.s, i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 3		; <i32> [#uses=1]
	%tmp.3.s = add i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 %a4q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 2		; <i64> [#uses=1]
	%tmp.3.s = add i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 %a8q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 3		; <i64> [#uses=1]
	%tmp.3.s = add i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @sext %a4li(i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 2		; <i32> [#uses=1]
	%tmp.3.s = add i32 100, %tmp.1.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @sext %a8li(i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 3		; <i32> [#uses=1]
	%tmp.3.s = add i32 100, %tmp.1.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 %a4qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 2		; <i64> [#uses=1]
	%tmp.3.s = add i64 100, %tmp.1.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 %a8qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 3		; <i64> [#uses=1]
	%tmp.3.s = add i64 100, %tmp.1.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @sext %s4l(i32 @sext %x.s, i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 2		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @sext %s8l(i32 @sext %x.s, i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 3		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, %x.s		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 %s4q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 2		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 %s8q(i64 %x.s, i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 3		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, %x.s		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i32 @sext %s4li(i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 2		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, 100		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i32 @sext %s8li(i32 @sext %y.s) {
entry:
	%tmp.1.s = shl i32 %y.s, i8 3		; <i32> [#uses=1]
	%tmp.3.s = sub i32 %tmp.1.s, 100		; <i32> [#uses=1]
	ret i32 %tmp.3.s
}

define i64 %s4qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 2		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, 100		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

define i64 %s8qi(i64 %y.s) {
entry:
	%tmp.1.s = shl i64 %y.s, i8 3		; <i64> [#uses=1]
	%tmp.3.s = sub i64 %tmp.1.s, 100		; <i64> [#uses=1]
	ret i64 %tmp.3.s
}

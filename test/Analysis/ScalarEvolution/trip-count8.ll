; RUN: opt < %s -analyze -scalar-evolution \
; RUN:  | grep "Loop %for\.body: backedge-taken count is (-1 + [%]ecx)"
; PR4599

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define i32 @foo(i32 %ecx) nounwind {
entry:
	%cmp2 = icmp eq i32 %ecx, 0		; <i1> [#uses=1]
	br i1 %cmp2, label %for.end, label %bb.nph

for.cond:		; preds = %for.inc
	%cmp = icmp ult i32 %inc, %ecx		; <i1> [#uses=1]
	br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:		; preds = %for.cond
	%phitmp = add i32 %i.01, 2		; <i32> [#uses=1]
	br label %for.end

bb.nph:		; preds = %entry
	br label %for.body

for.body:		; preds = %bb.nph, %for.cond
	%i.01 = phi i32 [ %inc, %for.cond ], [ 0, %bb.nph ]		; <i32> [#uses=3]
	%call = call i32 @bar(i32 %i.01) nounwind		; <i32> [#uses=0]
	br label %for.inc

for.inc:		; preds = %for.body
	%inc = add i32 %i.01, 1		; <i32> [#uses=2]
	br label %for.cond

for.end:		; preds = %for.cond.for.end_crit_edge, %entry
	%i.0.lcssa = phi i32 [ %phitmp, %for.cond.for.end_crit_edge ], [ 1, %entry ]		; <i32> [#uses=1]
	ret i32 %i.0.lcssa
}

declare i32 @bar(i32)

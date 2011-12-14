; RUN: opt -S -licm < %s | FileCheck %s

; UDiv is safe to speculate if the denominator is known non-zero.

; CHECK: @safe_udiv
; CHECK:      %div = udiv i64 %x, %or
; CHECK-NEXT: br label %for.body

define void @safe_udiv(i64 %x, i64 %m, i64 %n, i32* %p, i64* %q) nounwind {
entry:
  %or = or i64 %m, 1
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %p, i64 %i.02
  %0 = load i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %div = udiv i64 %x, %or
  %arrayidx1 = getelementptr inbounds i64* %q, i64 %i.02
  store i64 %div, i64* %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add i64 %i.02, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; UDiv is unsafe to speculate if the denominator is not known non-zero.

; CHECK: @unsafe_udiv
; CHECK-NOT:  udiv
; CHECK: for.body:

define void @unsafe_udiv(i64 %x, i64 %m, i64 %n, i32* %p, i64* %q) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %p, i64 %i.02
  %0 = load i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %div = udiv i64 %x, %m
  %arrayidx1 = getelementptr inbounds i64* %q, i64 %i.02
  store i64 %div, i64* %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add i64 %i.02, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; SDiv is safe to speculate if the denominator is known non-zero and
; known to have at least one zero bit.

; CHECK: @safe_sdiv
; CHECK:      %div = sdiv i64 %x, %or
; CHECK-NEXT: br label %for.body

define void @safe_sdiv(i64 %x, i64 %m, i64 %n, i32* %p, i64* %q) nounwind {
entry:
  %and = and i64 %m, -3
  %or = or i64 %and, 1
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %p, i64 %i.02
  %0 = load i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %div = sdiv i64 %x, %or
  %arrayidx1 = getelementptr inbounds i64* %q, i64 %i.02
  store i64 %div, i64* %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add i64 %i.02, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; SDiv is unsafe to speculate if the denominator is not known non-zero.

; CHECK: @unsafe_sdiv_a
; CHECK-NOT:  sdiv
; CHECK: for.body:

define void @unsafe_sdiv_a(i64 %x, i64 %m, i64 %n, i32* %p, i64* %q) nounwind {
entry:
  %or = or i64 %m, 1
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %p, i64 %i.02
  %0 = load i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %div = sdiv i64 %x, %or
  %arrayidx1 = getelementptr inbounds i64* %q, i64 %i.02
  store i64 %div, i64* %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add i64 %i.02, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; SDiv is unsafe to speculate if the denominator is not known to have a zero bit.

; CHECK: @unsafe_sdiv_b
; CHECK-NOT:  sdiv
; CHECK: for.body:

define void @unsafe_sdiv_b(i64 %x, i64 %m, i64 %n, i32* %p, i64* %q) nounwind {
entry:
  %and = and i64 %m, -3
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %p, i64 %i.02
  %0 = load i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %div = sdiv i64 %x, %and
  %arrayidx1 = getelementptr inbounds i64* %q, i64 %i.02
  store i64 %div, i64* %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add i64 %i.02, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

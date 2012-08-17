; RUN: opt -S -objc-arc < %s | FileCheck %s

declare i8* @objc_retain(i8*)
declare void @objc_release(i8*)
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_msgSend(i8*, i8*, ...)
declare void @use_pointer(i8*)
declare void @callee()
declare i8* @returner()

; ARCOpt shouldn't try to move the releases to the block containing the invoke.

; CHECK: define void @test0(
; CHECK: invoke.cont:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   ret void
; CHECK: lpad:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   ret void
define void @test0(i8* %zipFile) {
entry:
  call i8* @objc_retain(i8* %zipFile) nounwind
  call void @use_pointer(i8* %zipFile)
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*)*)(i8* %zipFile) 
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
           cleanup
  call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
  ret void
}

; ARCOpt should move the release before the callee calls.

; CHECK: define void @test1(
; CHECK: invoke.cont:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   call void @callee()
; CHECK:   br label %done
; CHECK: lpad:
; CHECK:   call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
; CHECK:   call void @callee()
; CHECK:   br label %done
; CHECK: done:
; CHECK-NEXT: ret void
define void @test1(i8* %zipFile) {
entry:
  call i8* @objc_retain(i8* %zipFile) nounwind
  call void @use_pointer(i8* %zipFile)
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*)*)(i8* %zipFile)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @callee()
  br label %done

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
           cleanup
  call void @callee()
  br label %done

done:
  call void @objc_release(i8* %zipFile) nounwind, !clang.imprecise_release !0
  ret void
}

; The optimizer should ignore invoke unwind paths consistently.
; PR12265

; CHECK: define void @test2() {
; CHECK: invoke.cont:
; CHECK-NEXT: call i8* @objc_retain
; CHECK-NOT: @objc_r
; CHECK: finally.cont:
; CHECK-NEXT: call void @objc_release
; CHECK-NOT: @objc
; CHECK: finally.rethrow:
; CHECK-NOT: @objc
; CHECK: }
define void @test2() {
entry:
  %call = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* ()*)()
          to label %invoke.cont unwind label %finally.rethrow, !clang.arc.no_objc_arc_exceptions !0

invoke.cont:                                      ; preds = %entry
  %tmp1 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void ()*)(), !clang.arc.no_objc_arc_exceptions !0
  invoke void @use_pointer(i8* %call)
          to label %finally.cont unwind label %finally.rethrow, !clang.arc.no_objc_arc_exceptions !0

finally.cont:                                     ; preds = %invoke.cont
  tail call void @objc_release(i8* %call) nounwind, !clang.imprecise_release !0
  ret void

finally.rethrow:                                  ; preds = %invoke.cont, %entry
  %tmp2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          catch i8* null
  unreachable
}

; Don't try to place code on invoke critical edges.

; CHECK: define void @test3(
; CHECK: if.end:
; CHECK-NEXT: call void @objc_release(i8* %p) nounwind
; CHECK-NEXT: ret void
define void @test3(i8* %p, i1 %b) {
entry:
  %0 = call i8* @objc_retain(i8* %p)
  call void @callee()
  br i1 %b, label %if.else, label %if.then

if.then:
  invoke void @use_pointer(i8* %p)
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

if.else:
  invoke void @use_pointer(i8* %p)
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

lpad:
  %r = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
       cleanup
  ret void

if.end:
  call void @objc_release(i8* %p)
  ret void
}

; Like test3, but with ARC-relevant exception handling.

; CHECK: define void @test4(
; CHECK: lpad:
; CHECK-NEXT: %r = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
; CHECK-NEXT: cleanup
; CHECK-NEXT: call void @objc_release(i8* %p) nounwind
; CHECK-NEXT: ret void
; CHECK: if.end:
; CHECK-NEXT: call void @objc_release(i8* %p) nounwind
; CHECK-NEXT: ret void
define void @test4(i8* %p, i1 %b) {
entry:
  %0 = call i8* @objc_retain(i8* %p)
  call void @callee()
  br i1 %b, label %if.else, label %if.then

if.then:
  invoke void @use_pointer(i8* %p)
          to label %if.end unwind label %lpad

if.else:
  invoke void @use_pointer(i8* %p)
          to label %if.end unwind label %lpad

lpad:
  %r = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
       cleanup
  call void @objc_release(i8* %p)
  ret void

if.end:
  call void @objc_release(i8* %p)
  ret void
}

; Don't turn the retainAutoreleaseReturnValue into retain, because it's
; for an invoke which we can assume codegen will put immediately prior.

; CHECK: define void @test5(
; CHECK: call i8* @objc_retainAutoreleasedReturnValue(i8* %z)
; CHECK: }
define void @test5() {
entry:
  %z = invoke i8* @returner()
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

lpad:
  %r13 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          cleanup
  ret void

if.end:
  call i8* @objc_retainAutoreleasedReturnValue(i8* %z)
  ret void
}

; Like test5, but there's intervening code.

; CHECK: define void @test6(
; CHECK: call i8* @objc_retain(i8* %z)
; CHECK: }
define void @test6() {
entry:
  %z = invoke i8* @returner()
          to label %if.end unwind label %lpad, !clang.arc.no_objc_arc_exceptions !0

lpad:
  %r13 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          cleanup
  ret void

if.end:
  call void @callee()
  call i8* @objc_retainAutoreleasedReturnValue(i8* %z)
  ret void
}

declare i32 @__gxx_personality_v0(...)
declare i32 @__objc_personality_v0(...)

!0 = metadata !{}

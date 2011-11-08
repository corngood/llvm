; RUN: llc < %s -march=x86-64 | FileCheck %s
; rdar://10050222, rdar://10134392

define <4 x float> @t1(<4 x float> %a, <1 x i64>* nocapture %p) nounwind {
entry:
; CHECK: t1:
; CHECK: movlps (%rdi), %xmm0
; CHECK: ret
  %p.val = load <1 x i64>* %p, align 1
  %0 = bitcast <1 x i64> %p.val to <2 x float>
  %shuffle.i = shufflevector <2 x float> %0, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %shuffle1.i = shufflevector <4 x float> %a, <4 x float> %shuffle.i, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  ret <4 x float> %shuffle1.i
}

define <4 x float> @t1a(<4 x float> %a, <1 x i64>* nocapture %p) nounwind {
entry:
; CHECK: t1a:
; CHECK: movlps (%rdi), %xmm0
; CHECK: ret
  %0 = bitcast <1 x i64>* %p to double*
  %1 = load double* %0
  %2 = insertelement <2 x double> undef, double %1, i32 0
  %3 = bitcast <2 x double> %2 to <4 x float>
  %4 = shufflevector <4 x float> %a, <4 x float> %3, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  ret <4 x float> %4
}

define void @t2(<1 x i64>* nocapture %p, <4 x float> %a) nounwind {
entry:
; CHECK: t2:
; CHECK: movlps %xmm0, (%rdi)
; CHECK: ret
  %cast.i = bitcast <4 x float> %a to <2 x i64>
  %extract.i = extractelement <2 x i64> %cast.i, i32 0
  %0 = getelementptr inbounds <1 x i64>* %p, i64 0, i64 0
  store i64 %extract.i, i64* %0, align 8
  ret void
}

define void @t2a(<1 x i64>* nocapture %p, <4 x float> %a) nounwind {
entry:
; CHECK: t2a:
; CHECK: movlps %xmm0, (%rdi)
; CHECK: ret
  %0 = bitcast <1 x i64>* %p to double*
  %1 = bitcast <4 x float> %a to <2 x double>
  %2 = extractelement <2 x double> %1, i32 0
  store double %2, double* %0
  ret void
}

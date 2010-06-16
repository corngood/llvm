; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This function would crash LiveIntervalAnalysis by creating a chain of 4 INSERT_SUBREGs of the same register.
define arm_apcscc void @NEON_vst4q_u32(i32* nocapture %sp0, i32* nocapture %sp1, i32* nocapture %sp2, i32* nocapture %sp3, i32* %dp) nounwind {
entry:
  %0 = bitcast i32* %sp0 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %1 = load <4 x i32>* %0, align 16               ; <<4 x i32>> [#uses=1]
  %2 = bitcast i32* %sp1 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %3 = load <4 x i32>* %2, align 16               ; <<4 x i32>> [#uses=1]
  %4 = bitcast i32* %sp2 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %5 = load <4 x i32>* %4, align 16               ; <<4 x i32>> [#uses=1]
  %6 = bitcast i32* %sp3 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %7 = load <4 x i32>* %6, align 16               ; <<4 x i32>> [#uses=1]
  %8 = bitcast i32* %dp to i8*                    ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst4.v4i32(i8* %8, <4 x i32> %1, <4 x i32> %3, <4 x i32> %5, <4 x i32> %7)
  ret void
}

declare void @llvm.arm.neon.vst4.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>) nounwind

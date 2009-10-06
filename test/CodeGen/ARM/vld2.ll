; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

%struct.__neon_int8x8x2_t = type { <8 x i8>,  <8 x i8> }
%struct.__neon_int16x4x2_t = type { <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x2_t = type { <2 x i32>, <2 x i32> }
%struct.__neon_float32x2x2_t = type { <2 x float>, <2 x float> }

define <8 x i8> @vld2i8(i8* %A) nounwind {
;CHECK: vld2i8:
;CHECK: vld2.8
	%tmp1 = call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2.v8i8(i8* %A)
        %tmp2 = extractvalue %struct.__neon_int8x8x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp1, 1
        %tmp4 = add <8 x i8> %tmp2, %tmp3
	ret <8 x i8> %tmp4
}

define <4 x i16> @vld2i16(i16* %A) nounwind {
;CHECK: vld2i16:
;CHECK: vld2.16
	%tmp1 = call %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2.v4i16(i16* %A)
        %tmp2 = extractvalue %struct.__neon_int16x4x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int16x4x2_t %tmp1, 1
        %tmp4 = add <4 x i16> %tmp2, %tmp3
	ret <4 x i16> %tmp4
}

define <2 x i32> @vld2i32(i32* %A) nounwind {
;CHECK: vld2i32:
;CHECK: vld2.32
	%tmp1 = call %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2.v2i32(i32* %A)
        %tmp2 = extractvalue %struct.__neon_int32x2x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_int32x2x2_t %tmp1, 1
        %tmp4 = add <2 x i32> %tmp2, %tmp3
	ret <2 x i32> %tmp4
}

define <2 x float> @vld2f(float* %A) nounwind {
;CHECK: vld2f:
;CHECK: vld2.32
	%tmp1 = call %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2.v2f32(float* %A)
        %tmp2 = extractvalue %struct.__neon_float32x2x2_t %tmp1, 0
        %tmp3 = extractvalue %struct.__neon_float32x2x2_t %tmp1, 1
        %tmp4 = add <2 x float> %tmp2, %tmp3
	ret <2 x float> %tmp4
}

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2.v8i8(i8*) nounwind readonly
declare %struct.__neon_int16x4x2_t @llvm.arm.neon.vld2.v4i16(i8*) nounwind readonly
declare %struct.__neon_int32x2x2_t @llvm.arm.neon.vld2.v2i32(i8*) nounwind readonly
declare %struct.__neon_float32x2x2_t @llvm.arm.neon.vld2.v2f32(i8*) nounwind readonly

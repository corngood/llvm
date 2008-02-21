; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep movss    %t | count 3
; RUN: grep movhlps  %t | count 1
; RUN: grep pshufd   %t | count 1
; RUN: grep unpckhpd %t | count 1

define void @test1(<4 x float>* %F, float* %f) {
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp7 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	%tmp2 = extractelement <4 x float> %tmp7, i32 0		; <float> [#uses=1]
	store float %tmp2, float* %f
	ret void
}

define float @test2(<4 x float>* %F, float* %f) {
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp7 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	%tmp2 = extractelement <4 x float> %tmp7, i32 2		; <float> [#uses=1]
	ret float %tmp2
}

define void @test3(float* %R, <4 x float>* %P1) {
	%X = load <4 x float>* %P1		; <<4 x float>> [#uses=1]
	%tmp = extractelement <4 x float> %X, i32 3		; <float> [#uses=1]
	store float %tmp, float* %R
	ret void
}

define double @test4(double %A) {
	%tmp1 = call <2 x double> @foo( )		; <<2 x double>> [#uses=1]
	%tmp2 = extractelement <2 x double> %tmp1, i32 1		; <double> [#uses=1]
	%tmp3 = add double %tmp2, %A		; <double> [#uses=1]
	ret double %tmp3
}

declare <2 x double> @foo()

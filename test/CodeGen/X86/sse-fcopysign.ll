; RUN: llvm-as | llc -march=x86 -mattr=+sse2 &&
; RUN: llvm-as | llc -march=x86 -mattr=+sse2 | grep pslldq | wc -l | grep 1 &&
; RUN: llvm-as | llc -march=x86 -mattr=+sse2 | not getp test

define float %test1(float %a, float %b) {
	%tmp = tail call float %copysignf( float %b, float %a )
	ret float %tmp
}

define double %test2(double %a, float %b, float %c) {
	%tmp1 = add float %b, %c
	%tmp2 = fpext float %tmp1 to double
	%tmp = tail call double %copysign( double %a, double %tmp2 )
	ret double %tmp
}

declare float %copysignf(float, float)
declare double %copysign(double, double)

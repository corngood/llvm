; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm  | grep movmi &&
; RUN: llvm-as < %s | llc -march=arm  | grep moveq &&
; RUN: llvm-as < %s | llc -march=arm  | grep movgt &&
; RUN: llvm-as < %s | llc -march=arm  | grep movge &&
; RUN: llvm-as < %s | llc -march=arm  | grep movls &&
; RUN: llvm-as < %s | llc -march=arm  | grep fcmps &&
; RUN: llvm-as < %s | llc -march=arm  | grep fcmpd

int %f1(float %a) {
entry:
	%tmp = setlt float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f2(float %a) {
entry:
	%tmp = seteq float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f3(float %a) {
entry:
	%tmp = setgt float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f4(float %a) {
entry:
	%tmp = setge float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f5(float %a) {
entry:
	%tmp = setle float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f6(float %a) {
entry:
	%tmp = setne float %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %g1(double %a) {
entry:
	%tmp = setlt double %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %g2(double %a) {
entry:
	%tmp = setne double %a, 1.000000e+00		; <bool> [#uses=1]
	%tmp = cast bool %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

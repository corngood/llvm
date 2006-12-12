; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep mvn | wc -l | grep 4

;int %f1() {
;entry:
;	ret int -1
;}

int %f2(int %a) {
entry:
	%tmpnot = xor int %a, -1		; <int> [#uses=1]
	ret int %tmpnot
}

;int %f3(int %a) {
;entry:
;	%tmp1 = shl int %a, ubyte 2		; <int> [#uses=1]
;	%tmp1not = xor int %tmp1, -1		; <int> [#uses=1]
;	ret int %tmp1not
;}

int %f4(int %a, ubyte %b) {
entry:
	%tmp3 = shl int %a, ubyte %b		; <int> [#uses=1]
	%tmp3not = xor int %tmp3, -1		; <int> [#uses=1]
	ret int %tmp3not
}

;uint %f5(uint %a) {
;entry:
;	%tmp1 = lshr uint %a, ubyte 2		; <uint> [#uses=1]
;	%tmp1not = xor uint %tmp1, 4294967295		; <uint> [#uses=1]
;	ret uint %tmp1not
;}

uint %f6(uint %a, ubyte %b) {
entry:
	%tmp2 = lshr uint %a, ubyte %b		; <uint> [#uses=1]
	%tmp2not = xor uint %tmp2, 4294967295		; <uint> [#uses=1]
	ret uint %tmp2not
}

;int %f7(int %a) {
;entry:
;	%tmp1 = ashr int %a, ubyte 2		; <int> [#uses=1]
;	%tmp1not = xor int %tmp1, -1		; <int> [#uses=1]
;	ret int %tmp1not
;}

int %f8(int %a, ubyte %b) {
entry:
	%tmp3 = ashr int %a, ubyte %b		; <int> [#uses=1]
	%tmp3not = xor int %tmp3, -1		; <int> [#uses=1]
	ret int %tmp3not
}

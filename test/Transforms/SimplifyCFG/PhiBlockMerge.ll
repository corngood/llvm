; Test merging of blocks that only have PHI nodes in them
;
; RUN: if as < %s | opt -simplifycfg | dis | grep 'N:'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
;

int %test(bool %a, bool %b) {
        br bool %a, label %M, label %O

O:
	br bool %b, label %N, label %Q
Q:
	br label %N
N:
	%Wp = phi int [0, %O], [1, %Q]
	; This block should be foldable into M
	br label %M

M:
	%W = phi int [%Wp, %N], [2, %0]
	%R = add int %W, 1
	ret int %R
}


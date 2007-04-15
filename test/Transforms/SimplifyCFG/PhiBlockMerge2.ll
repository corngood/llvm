; Test merging of blocks that only have PHI nodes in them.  This tests the case
; where the mergedinto block doesn't have any PHI nodes, and is in fact 
; dominated by the block-to-be-eliminated
;
; RUN: llvm-upgrade < %s | llvm-as | opt -simplifycfg | llvm-dis | not grep N:
;

int %test(bool %a, bool %b) {
	br bool %b, label %N, label %Q
Q:
	br label %N
N:
	%W = phi int [0, %0], [1, %Q]
	; This block should be foldable into M
	br label %M

M:
	%R = add int %W, 1
	ret int %R
}


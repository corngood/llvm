; With reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -reassociate -constprop -instcombine -die | llvm-dis | not grep add

int %test(int %arg) {
	%tmp1 = sub int -12, %arg
	%tmp2 = add int %tmp1, 12
	ret int %tmp2
}

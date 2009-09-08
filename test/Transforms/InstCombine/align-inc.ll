; RUN: opt %s -instcombine -S | grep {GLOBAL.*align 16}
; RUN: opt %s -instcombine -S | grep {tmp = load}

@GLOBAL = internal global [4 x i32] zeroinitializer

define <16 x i8> @foo(<2 x i64> %x) {
entry:
	%tmp = load <16 x i8>* bitcast ([4 x i32]* @GLOBAL to <16 x i8>*), align 1
	ret <16 x i8> %tmp
}


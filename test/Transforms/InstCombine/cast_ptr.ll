; Tests to make sure elimination of casts is working correctly
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output &&
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | not grep '\([sz]ext\)\|\(trunc\)'

target pointersize = 32

implementation

sbyte* %test1(sbyte* %t) {
	%tmpc = cast sbyte* %t to uint
	%tmpa = add uint %tmpc, 32
	%tv = cast uint %tmpa to sbyte*
	ret sbyte* %tv
}


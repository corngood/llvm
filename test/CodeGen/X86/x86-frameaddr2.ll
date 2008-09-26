; RUN: llvm-as < %s | llc -march=x86 | grep mov | count 3
; XFAIL: *

define i8* @t() nounwind {
entry:
	%0 = tail call i8* @llvm.frameaddress(i32 2)
	ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone

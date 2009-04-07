; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu > %t
; RUN: grep {movl	\$i@NTPOFF, %eax} %t
; RUN: grep {addl	%gs:0, %eax} %t

@i = external hidden thread_local global i32

define i32* @f() {
entry:
	ret i32* @i
}

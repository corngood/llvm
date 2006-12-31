; RUN: llvm-upgrade < %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null

; Testcase distilled from 256.bzip2.

target endian = little
target pointersize = 32

int %main() {
entry:
	%X = add int 1, -1
	br label %Next

Next:
	%A = phi int [ %X, %entry ]
	%B = phi int [ %X, %entry ]
	%C = phi int [ %X, %entry ]
	ret int %C
}

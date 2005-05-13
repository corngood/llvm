; Make sure this testcase codegens to the ctlz instruction
; XFAIL: *
; RUN: llvm-as < %s | llc -march=alpha -enable-alpha-CT | grep 'ctlz'

declare ubyte %llvm.ctlz(ubyte)

implementation   ; Functions:

ubyte %bar(ubyte %x) {
entry:
	%tmp.1 = call ubyte %llvm.ctlz( ubyte %x ) 
	ret ubyte %tmp.1
}

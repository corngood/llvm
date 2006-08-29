; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | grep 'shld.*CL' &&
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | not grep 'mov CL, BL'

; PR687

ulong %foo(ulong %x, long* %X) {
	%tmp.1 = load long* %X		; <long> [#uses=1]
	%tmp.3 = cast long %tmp.1 to ubyte		; <ubyte> [#uses=1]
	%tmp.4 = shl ulong %x, ubyte %tmp.3		; <ulong> [#uses=1]
	ret ulong %tmp.4
}

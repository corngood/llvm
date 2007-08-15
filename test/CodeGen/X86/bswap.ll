; bswap should be constant folded when it is passed a constant argument

; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | \
; RUN:   grep bswapl | count 3
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep rolw | count 1

declare ushort %llvm.bswap.i16(ushort)
declare uint %llvm.bswap.i32(uint)
declare ulong %llvm.bswap.i64(ulong)

ushort %W(ushort %A) {
	%Z = call ushort %llvm.bswap.i16(ushort %A)
	ret ushort %Z
}

uint %X(uint %A) {
	%Z = call uint %llvm.bswap.i32(uint %A)
	ret uint %Z
}

ulong %Y(ulong %A) {
	%Z = call ulong %llvm.bswap.i64(ulong %A)
	ret ulong %Z
}

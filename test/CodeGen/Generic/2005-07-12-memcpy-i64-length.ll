; RUN: llvm-upgrade < %s | llvm-as | llc
; Test that llvm.memcpy works with a i64 length operand on all targets.


declare void %llvm.memcpy.i64(sbyte*, sbyte*, ulong, uint)

void %l12_l94_bc_divide_endif_2E_3_2E_ce() {
newFuncRoot:
	tail call void %llvm.memcpy.i64( sbyte* null, sbyte* null, ulong 0, uint 1 )
	unreachable
}

; RUN: llvm-upgrade < %s | llvm-as | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

implementation
declare void %llvm.va_start(sbyte** %ap)
declare void %llvm.va_copy(sbyte** %aq, sbyte** %ap)
declare void %llvm.va_end(sbyte** %ap)

int %test(int %X, ...) {
        %ap = alloca sbyte*
	call void %llvm.va_start(sbyte** %ap)
	%tmp = va_arg sbyte** %ap, int 

        %aq = alloca sbyte*
	call void %llvm.va_copy(sbyte** %aq, sbyte** %ap)
	call void %llvm.va_end(sbyte** %aq)
	
	call void %llvm.va_end(sbyte** %ap)
	ret int %tmp
}

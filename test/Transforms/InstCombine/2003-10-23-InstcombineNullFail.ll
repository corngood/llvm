; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep false
;
; This actually looks like a constant propagation bug

%X = type { [10 x int], float }

implementation

bool %test() {
	%A = getelementptr %X* null, long 0, uint 0, long 0
	%B = setne int* %A, null
	ret bool %B
}

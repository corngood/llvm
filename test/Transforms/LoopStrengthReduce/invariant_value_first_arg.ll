; Check that the index of 'P[outer]' is pulled out of the loop.
; RUN: llvm-upgrade < %s | llvm-as | opt -loop-reduce | llvm-dis | not grep 'getelementptr.*%outer.*%INDVAR'

declare bool %pred()

void %test([10000 x int]* %P, int %outer) {
	br label %Loop
Loop:
	%INDVAR = phi int [0, %0], [%INDVAR2, %Loop]

	%STRRED = getelementptr [10000 x int]* %P, int %outer, int %INDVAR
	store int 0, int* %STRRED

	%INDVAR2 = add int %INDVAR, 1
	%cond = call bool %pred()
	br bool %cond, label %Loop, label %Out
Out:
	ret void
}

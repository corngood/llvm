; RUN: llvm-upgrade < %s | llvm-as | opt -indvars | llvm-dis | grep indvar

int %test() {
	br bool true, label %LoopHead, label %LoopHead

LoopHead:
	%A = phi int [7, %0], [7, %0], [%B, %LoopHead]
	%B = add int %A, 1
	br bool false, label %LoopHead, label %Out
Out:
	ret int %B
}

; RUN: llvm-upgrade < %s | llvm-as | opt -inline -disable-output -print

int %func(int %i) {
	ret int %i
}

int %main(int %argc) {
	%X = call int %func(int 7)
	%Y = add int %X, %argc
	ret int %Y
}

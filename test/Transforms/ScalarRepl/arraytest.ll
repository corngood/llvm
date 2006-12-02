; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl -mem2reg | llvm-dis | not grep alloca

int %test() {
  %X = alloca [ 4 x int ]
  %Y = getelementptr [4x int]* %X, long 0, long 0
  store int 0, int* %Y

  %Z = load int* %Y
  ret int %Z
}

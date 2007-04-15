; RUN: llvm-upgrade < %s | llvm-as | opt -tailcallelim | llvm-dis | \
; RUN:    grep {call i32 @foo}

declare void %bar(int*)
int %foo(uint %N) {
  %A = alloca int, uint %N             ;; Should stay in entry block because of 'tail' marker
  store int 17, int* %A
  call void %bar(int* %A)

  %X = tail call int %foo(uint %N)  ;; Cannot -tailcallelim this without increasing stack usage!
  ret int %X
}

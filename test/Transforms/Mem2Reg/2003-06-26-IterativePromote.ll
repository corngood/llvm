; Promoting some values allows promotion of other values.
; RUN: llvm-upgrade < %s | llvm-as | opt -mem2reg | llvm-dis | not grep alloca

int %test2() {
        %result = alloca int             ; ty=int*
        %a = alloca int          ; ty=int*
        %p = alloca int*                 ; ty=int**
        store int 0, int* %a
        store int* %a, int** %p
        %tmp.0 = load int** %p           ; ty=int*
        %tmp.1 = load int* %tmp.0                ; ty=int
        store int %tmp.1, int* %result
        %tmp.2 = load int* %result               ; ty=int
        ret int %tmp.2
}


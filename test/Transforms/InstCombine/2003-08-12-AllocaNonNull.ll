; This testcase can be simplified by "realizing" that alloca can never return 
; null.
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -simplifycfg | llvm-dis | not grep 'br '

implementation   ; Functions:

declare int %bitmap_clear(...)

int %oof() {
entry:
        %live_head = alloca int         ; <int*> [#uses=2]
        %tmp.1 = setne int* %live_head, null            ; <bool> [#uses=1]
        br bool %tmp.1, label %then, label %UnifiedExitNode

then:
        %tmp.4 = call int (...)* %bitmap_clear( int* %live_head )              ; <int> [#uses=0]
        br label %UnifiedExitNode

UnifiedExitNode:
        ret int 0
}


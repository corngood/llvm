; If the result of an instruction is only used outside of the loop, sink
; the instruction to the exit blocks instead of executing it on every
; iteration of the loop.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -licm | llvm-dis | %prcontext mul 1 | grep Out: 

int %test(int %N) {
Entry:
	br label %Loop
Loop:
        %N_addr.0.pn = phi int [ %dec, %Loop ], [ %N, %Entry ]
        %tmp.6 = mul int %N, %N_addr.0.pn
        %tmp.7 = sub int %tmp.6, %N
        %dec = add int %N_addr.0.pn, -1
        %tmp.1 = setne int %N_addr.0.pn, 1
        br bool %tmp.1, label %Loop, label %Out
Out:
	ret int %tmp.7
}

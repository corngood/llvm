; RUN: llvm-upgrade < %s | llvm-as | llc

void %QRiterate(double %tmp.212) { 
entry:          ; No predecessors!
        br label %shortcirc_next.1

shortcirc_next.1:               ; preds = %entry
        %tmp.213 = setne double %tmp.212, 0.000000e+00
        br bool %tmp.213, label %shortcirc_next.1, label %exit.1

exit.1:
	ret void
}

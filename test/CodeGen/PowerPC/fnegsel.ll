; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep fneg

double %test_FNEG_sel(double %A, double %B, double %C) {
    %D = sub double -0.0, %A
    %Cond = setgt double %D, -0.0
    %E = select bool %Cond, double %B, double %C
	ret double %E
}

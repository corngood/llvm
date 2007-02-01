; RUN: llvm-upgrade %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null


double %test(double* %DP, double %Arg) {
	%D = load double* %DP
	%V = add double %D, 1.0
	%W = sub double %V, %V
	%X = mul double %W, %W
	%Y = div double %X, %X
	%Z = rem double %Y, %Y
	%Z1 = div double %Z, %W
	%Q = add double %Z, %Arg
	%R = cast double %Q to double
	store double %R, double* %DP
	ret double %Z
}

int %main() { 
  %X = alloca double
  store double 0.0, double* %X
  call double %test(double* %X, double 2.0)
  ret int 0 
}

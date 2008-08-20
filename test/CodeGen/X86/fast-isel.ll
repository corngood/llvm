; RUN: llvm-as < %s | llc -fast-isel -march=x86
; RUN: llvm-as < %s | llc -fast-isel -march=x86-64

; This tests very minimal fast-isel functionality.

define i32 @foo(i32* %p, i32* %q) {
entry:
  %r = load i32* %p
  %s = load i32* %q
  br label %fast

fast:
  %t0 = add i32 %r, %s
  %t1 = mul i32 %t0, %s
  %t2 = sub i32 %t1, %s
  %t3 = and i32 %t2, %s
  %t4 = or i32 %t3, %s
  %t5 = xor i32 %t4, %s
  br label %exit

exit:
  ret i32 %t5
}

define double @bar(double* %p, double* %q) {
entry:
  %r = load double* %p
  %s = load double* %q
  br label %fast

fast:
  %t0 = add double %r, %s
  %t1 = mul double %t0, %s
  %t2 = sub double %t1, %s
  br label %exit

exit:
  ret double %t2
}


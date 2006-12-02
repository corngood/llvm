; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

; The C Writer bombs on this testcase because it tries the print the prototype
; for the test function, which tries to print the argument name.  The function
; has not been incorporated into the slot calculator, so after it does the name
; lookup, it tries a slot calculator lookup, which fails.

int %test(int) {
        ret int 0
}


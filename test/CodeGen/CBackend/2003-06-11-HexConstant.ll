; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

; Make sure hex constant does not continue into a valid hexadecimal letter/number
%version = global [3 x sbyte] c"\001\00"


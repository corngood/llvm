; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "mov r0, #0" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "mov r0, #255" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "mov r0, #256" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep ".word.*257" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "mov r0, #-1073741761" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "mov r0, #1008" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "\.comm.*a,4,4" | wc -l | grep 1

%a = internal global int 0

uint %f1() {
  ret uint 0
}

uint %f2() {
  ret uint 255
}

uint %f3() {
  ret uint 256
}

uint %f4() {
  ret uint 257
}

uint %f5() {
  ret uint 3221225535
}

uint %f6() {
  ret uint 1008
}

; RUN: llvm-upgrade < %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null


int %main() {
   %X = malloc int                ; constant size
   %Y = malloc int, uint 100      ; constant size
   %u = add uint 1, 2
   %Z = malloc int, uint %u       ; variable size
   free int* %X
   free int* %Y
   free int* %Z
   ret int 0
}

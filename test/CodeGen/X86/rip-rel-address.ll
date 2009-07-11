; RUN: llvm-as < %s | llc -march=x86-64 -relocation-model=pic | FileCheck %s -check-prefix=PIC64
; RUN: llvm-as < %s | llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=static | FileCheck %s -check-prefix=STATIC64

@a = internal global double 3.4
define double @foo() nounwind {
  %a = load double* @a
  ret double %a
  
; PIC64:    movsd	_a(%rip), %xmm0
; STATIC64: movsd	a, %xmm0
}

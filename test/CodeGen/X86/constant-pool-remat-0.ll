; RUN: llvm-as < %s | llc -march=x86-64 | grep LCPI | count 3
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep LCPI | count 3
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -stats |& | grep asm-printer | grep 13

declare fastcc float @qux(float %y)

define fastcc float @array(float %a) {
  %n = mul float %a, 9.0
  %m = call fastcc float @qux(float %n)
  %o = mul float %m, 9.0
  ret float %o
}

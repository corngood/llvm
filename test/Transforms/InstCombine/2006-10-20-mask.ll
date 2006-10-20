;  RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | grep 'and'
ulong %foo(ulong %tmp, ulong %tmp2) {
  %tmp = cast ulong %tmp to uint
  %tmp2 = cast ulong %tmp2 to uint
  %tmp3 = and uint %tmp, %tmp2
  %tmp4 = cast uint %tmp3 to ulong
  ret ulong %tmp4
}

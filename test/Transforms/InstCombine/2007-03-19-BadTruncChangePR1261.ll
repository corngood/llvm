; For PR1261. Before bit accurate type support in InstCombine, this would
; turn the sext into a zext.
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | grep sext
; XFAIL: *

define i16 @test(i31 %zzz) {
entry:
  %A = sext i31 %zzz to i32
  %B = add i32 %A, 16384
  %C = lshr i32 %B, 15
  %D = trunc i32 %C to i16
  ret i16 %D
}

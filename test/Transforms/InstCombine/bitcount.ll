; Tests to make sure bit counts of constants are folded
; RUN: llvm-as < %s | opt -instcombine | llvm-dis -o /dev/null -f && 
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 'llvm.ct'

declare i32 @llvm.ctpop.i31(i31 %val) 
declare i32 @llvm.cttz.i32(i32 %val) 
declare i32 @llvm.ctlz.i33(i33 %val) 

define i32 %test(i32 %A) {
  %c1 = i32 call @llvm.ctpop(i31 12415124)
  %c2 = i32 call @llvm.cttz(i32 87359874)
  %c3 = i32 call @llvm.ctlz(i33 87359874)
  %r1 = add i32 %c1, %c2
  %r2 = add i32 %r1, %c3
  ret i32 %r2
}

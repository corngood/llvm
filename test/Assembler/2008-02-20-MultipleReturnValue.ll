; RUN: llvm-as < %s -disable-output

define {i32, i8} @foo(i32 %p) {
  ret i32 1, i8 2
}

define i8 @f2(i32 %p) {
   %c = call {i32, i8} @foo(i32 %p)
   %d = getresult {i32, i8} %c, 1
   %e = add i8 %d, 1
   ret i8 %e
}

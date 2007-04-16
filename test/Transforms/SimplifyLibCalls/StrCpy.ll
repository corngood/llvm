; Test that the StrCpyOptimizer works correctly
; RUN: llvm-upgrade < %s | llvm-as | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call.*strcpy}

declare sbyte* %strcpy(sbyte*,sbyte*)
declare int %puts(sbyte*)
%hello = constant [6 x sbyte] c"hello\00"
%null = constant [1 x sbyte] c"\00"
%null_hello = constant [7 x sbyte] c"\00hello\00"

implementation   ; Functions:

int %main () {
  %target = alloca [1024 x sbyte]
  %arg1 = getelementptr [1024 x sbyte]* %target, int 0, int 0
  store sbyte 0, sbyte* %arg1
  %arg2 = getelementptr [6 x sbyte]* %hello, int 0, int 0
  %rslt1 = call sbyte* %strcpy(sbyte* %arg1, sbyte* %arg2)
  %arg3 = getelementptr [1 x sbyte]* %null, int 0, int 0
  %rslt2 = call sbyte* %strcpy(sbyte* %rslt1, sbyte* %arg3)
  %arg4 = getelementptr [7 x sbyte]* %null_hello, int 0, int 0
  %rslt3 = call sbyte* %strcpy(sbyte* %rslt2, sbyte* %arg4)
  call int %puts(sbyte* %rslt3)
  ret int 0
}

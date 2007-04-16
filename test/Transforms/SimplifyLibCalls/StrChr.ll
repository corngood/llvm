; Test that the StrChrOptimizer works correctly
; RUN: llvm-upgrade < %s | llvm-as | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call.*%strchr}

declare sbyte* %strchr(sbyte*,int)
declare int %puts(sbyte*)
%hello = constant [14 x sbyte] c"hello world\n\00"
%null = constant [1 x sbyte] c"\00"

implementation   ; Functions:

int %main () {
  %hello_p = getelementptr [14 x sbyte]* %hello, int 0, int 0
  %null_p = getelementptr [1 x sbyte]* %null, int 0, int 0

  %world  = call sbyte* %strchr(sbyte* %hello_p, int 119 )
  %ignore = call sbyte* %strchr(sbyte* %null_p, int 119 )
  %len = call int %puts(sbyte* %world)
  %index = add int %len, 112
  %result = call sbyte* %strchr(sbyte* %hello_p, int %index)
  ret int %index
}

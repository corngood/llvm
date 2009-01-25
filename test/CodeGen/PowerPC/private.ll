; Test to make sure that the 'private' is used correctly.
;
; RUN: llvm-as < %s | llc -mtriple=powerpc-unknown-linux-gnu > %t
; RUN: grep .Lfoo: %t
; RUN: grep bl.*\.Lfoo %t
; RUN: grep .Lbaz: %t
; RUN: grep lis.*\.Lbaz %t
; RUN: llvm-as < %s | llc -mtriple=powerpc-apple-darwin > %t
; RUN: grep L_foo: %t
; RUN: grep bl.*\L_foo %t
; RUN: grep L_baz: %t
; RUN: grep lis.*\L_baz %t

declare void @foo()

define private void @foo() {
        ret void
}

@baz = private global i32 4;

define i32 @bar() {
        call void @foo()
	%1 = load i32* @baz, align 4
        ret i32 %1
}

; RUN: llvm-as < %s | llc -mtriple=i386-mingw-pc | FileCheck %s

declare dllimport void @foo()

define void @bar() nounwind {
; CHECK: call	*__imp__foo
  call void @foo()
  ret void
}

; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu | grep ".hidden" | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin8.8.0 | grep ".private_extern" | wc -l | grep 2 

%struct.Person = type { i32 }
@a = hidden global i32 0
@b = external global i32

implementation   ; Functions:

define weak hidden void @_ZN6Person13privateMethodEv(%struct.Person* %this) {
  ret void
}

declare void @function(i32)

define weak void @_ZN6PersonC1Ei(%struct.Person* %this, i32 %_c) {
  ret void
}


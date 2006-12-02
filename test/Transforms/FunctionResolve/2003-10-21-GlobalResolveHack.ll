; This tests a hack put into place specifically for the C++ libstdc++ library.
; It uses an ugly hack which is cleaned up by the funcresolve pass.

; RUN: llvm-upgrade < %s | llvm-as | opt -funcresolve | llvm-dis | grep %X | grep '{'

%X = external global { int }
%X = global [ 4 x sbyte ] zeroinitializer

implementation

int* %test() {
  %P = getelementptr {int}* %X, long 0, uint 0
  ret int* %P
}

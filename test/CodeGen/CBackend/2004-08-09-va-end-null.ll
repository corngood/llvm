; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

declare void %llvm.va_end(sbyte*)

void %test() {
  call void %llvm.va_end( sbyte* null )
  ret void
}


; RUN: llvm-upgrade < %s | llvm-as | llc -march=c


%A = type { uint, sbyte*, { uint, uint, uint, uint, uint, uint, uint, uint }*, ushort }

void %test(%A *) { ret void }

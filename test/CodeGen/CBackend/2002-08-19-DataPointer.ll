; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

%sptr1   = global [11x sbyte]* %somestr         ;; Forward ref to a constant
%somestr = constant [11x sbyte] c"hello world"


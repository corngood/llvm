; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f


%spell_order = global [4 x ubyte] c"\FF\00\F7\00"


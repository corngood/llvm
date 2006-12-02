; RUN: llvm-upgrade < %s | llvm-as | opt -licm | llvm-dis | %prcontext volatile 1 | grep Loop

%X = global int 7

void %testfunc(int %i) {
        br label %Loop

Loop:
        %x = volatile load int* %X  ; Should not promote this to a register
        %x2 = add int %x, 1
        store int %x2, int* %X
        br bool true, label %Out, label %Loop

Out:
        ret void
}


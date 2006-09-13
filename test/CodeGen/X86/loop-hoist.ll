; RUN: llvm-as < %s | llc -relocation-model=dynamic-no-pic -mtriple=i686-apple-darwin8.7.2 | grep L_Arr.non_lazy_ptr &&
; RUN: llvm-as < %s | llc -relocation-model=dynamic-no-pic -mtriple=i686-apple-darwin8.7.2 | %prcontext L_Arr.non_lazy_ptr 1 | grep '4(%esp)'

%Arr = external global [0 x int]                ; <[0 x int]*> [#uses=2]

implementation   ; Functions:

void %foo(int %N) {
entry:
        %N = cast int %N to uint                ; <uint> [#uses=1]
        br label %cond_true

cond_true:              ; preds = %cond_true, %entry
        %indvar = phi uint [ 0, %entry ], [ %indvar.next, %cond_true ]          ; <uint> [#uses=3]
        %i.0.0 = cast uint %indvar to int               ; <int> [#uses=1]
        %tmp = getelementptr [0 x int]* %Arr, int 0, uint %indvar               ; <int*> [#uses=1]
        store int %i.0.0, int* %tmp
        %indvar.next = add uint %indvar, 1              ; <uint> [#uses=2]
        %exitcond = seteq uint %indvar.next, %N         ; <bool> [#uses=1]
        br bool %exitcond, label %return, label %cond_true

return:         ; preds = %cond_true, %entry
        ret void
}


; RUN: opt %s -adce -simplifycfg | llvm-dis

define i32 @Test(i32 %A, i32 %B) {
BB1:
        br label %BB4

BB2:            ; No predecessors!
        br label %BB3

BB3:            ; preds = %BB4, %BB2
        %ret = phi i32 [ %X, %BB4 ], [ %B, %BB2 ]               ; <i32> [#uses=1]
        ret i32 %ret

BB4:            ; preds = %BB1
        %X = phi i32 [ %A, %BB1 ]               ; <i32> [#uses=1]
        br label %BB3
}



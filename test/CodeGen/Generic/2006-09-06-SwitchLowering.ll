; RUN: llvm-as < %s | llc

void %foo() {
	br label %cond_true813.i

cond_true813.i:		; preds = %0
	br bool false, label %cond_true818.i, label %cond_next1146.i

cond_true818.i:		; preds = %cond_true813.i
	br bool false, label %recog_memoized.exit52, label %cond_next1146.i

recog_memoized.exit52:		; preds = %cond_true818.i
	switch int 0, label %bb886.i.preheader [
		 int 0, label %bb907.i
		 int 44, label %bb866.i
		 int 103, label %bb874.i
		 int 114, label %bb874.i
	]

bb857.i:		; preds = %bb886.i, %bb866.i
	%tmp862.i494.24 = phi sbyte* [ null, %bb866.i ], [ %tmp862.i494.26, %bb886.i ]		; <sbyte*> [#uses=4]
	switch int 0, label %bb886.i.preheader [
		 int 0, label %bb907.i
		 int 44, label %bb866.i
		 int 103, label %bb874.i
		 int 114, label %bb874.i
	]

bb866.i.loopexit:		; preds = %bb874.i
	br label %bb866.i

bb866.i.loopexit31:		; preds = %cond_true903.i
	br label %bb866.i

bb866.i:		; preds = %bb866.i.loopexit31, %bb866.i.loopexit, %bb857.i, %recog_memoized.exit52
	br bool false, label %bb907.i, label %bb857.i

bb874.i.preheader.loopexit:		; preds = %cond_true903.i, %cond_true903.i
	ret void

bb874.i:		; preds = %bb857.i, %bb857.i, %recog_memoized.exit52, %recog_memoized.exit52
	%tmp862.i494.25 = phi sbyte* [ %tmp862.i494.24, %bb857.i ], [ %tmp862.i494.24, %bb857.i ], [ undef, %recog_memoized.exit52 ], [ undef, %recog_memoized.exit52 ]		; <sbyte*> [#uses=1]
	switch int 0, label %bb886.i.preheader.loopexit [
		 int 0, label %bb907.i
		 int 44, label %bb866.i.loopexit
		 int 103, label %bb874.i.backedge
		 int 114, label %bb874.i.backedge
	]

bb874.i.backedge:		; preds = %bb874.i, %bb874.i
	ret void

bb886.i.preheader.loopexit:		; preds = %bb874.i
	ret void

bb886.i.preheader:		; preds = %bb857.i, %recog_memoized.exit52
	%tmp862.i494.26 = phi sbyte* [ undef, %recog_memoized.exit52 ], [ %tmp862.i494.24, %bb857.i ]		; <sbyte*> [#uses=1]
	br label %bb886.i

bb886.i:		; preds = %cond_true903.i, %bb886.i.preheader
	br bool false, label %bb857.i, label %cond_true903.i

cond_true903.i:		; preds = %bb886.i
	switch int 0, label %bb886.i [
		 int 0, label %bb907.i
		 int 44, label %bb866.i.loopexit31
		 int 103, label %bb874.i.preheader.loopexit
		 int 114, label %bb874.i.preheader.loopexit
	]

bb907.i:		; preds = %cond_true903.i, %bb874.i, %bb866.i, %bb857.i, %recog_memoized.exit52
	%tmp862.i494.0 = phi sbyte* [ %tmp862.i494.24, %bb857.i ], [ null, %bb866.i ], [ undef, %recog_memoized.exit52 ], [ %tmp862.i494.25, %bb874.i ], [ null, %cond_true903.i ]		; <sbyte*> [#uses=1]
	br bool false, label %cond_next1146.i, label %cond_true910.i

cond_true910.i:		; preds = %bb907.i
	ret void

cond_next1146.i:		; preds = %bb907.i, %cond_true818.i, %cond_true813.i
	%tmp862.i494.1 = phi sbyte* [ %tmp862.i494.0, %bb907.i ], [ undef, %cond_true818.i ], [ undef, %cond_true813.i ]		; <sbyte*> [#uses=0]
	ret void

bb2060.i:		; No predecessors!
	br bool false, label %cond_true2064.i, label %bb2067.i

cond_true2064.i:		; preds = %bb2060.i
	unreachable

bb2067.i:		; preds = %bb2060.i
	ret void

cond_next3473:		; No predecessors!
	ret void

cond_next3521:		; No predecessors!
	ret void
}

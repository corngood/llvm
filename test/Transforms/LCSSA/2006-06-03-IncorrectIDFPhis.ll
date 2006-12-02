; RUN: llvm-upgrade < %s | llvm-as | opt -lcssa | llvm-dis | grep "%SJE.0.0.lcssa = phi .struct.SetJmpMapEntry" &&
; RUN: llvm-upgrade < %s | llvm-as | opt -lcssa | llvm-dis | grep "%SJE.0.0.lcssa1 = phi .struct.SetJmpMapEntry"

%struct.SetJmpMapEntry = type { sbyte*, uint, %struct.SetJmpMapEntry* }

implementation   ; Functions:

void %__llvm_sjljeh_try_catching_longjmp_exception() {
entry:
	br bool false, label %UnifiedReturnBlock, label %no_exit

no_exit:		; preds = %endif, %entry
	%SJE.0.0 = phi %struct.SetJmpMapEntry* [ %tmp.24, %endif ], [ null, %entry ]		; <%struct.SetJmpMapEntry*> [#uses=1]
	br bool false, label %then, label %endif

then:		; preds = %no_exit
	%tmp.20 = getelementptr %struct.SetJmpMapEntry* %SJE.0.0, int 0, uint 1		; <uint*> [#uses=0]
	ret void

endif:		; preds = %no_exit
	%tmp.24 = load %struct.SetJmpMapEntry** null		; <%struct.SetJmpMapEntry*> [#uses=1]
	br bool false, label %UnifiedReturnBlock, label %no_exit

UnifiedReturnBlock:		; preds = %endif, %entry
	ret void
}

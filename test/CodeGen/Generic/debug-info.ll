; RUN: llvm-upgrade < %s | llvm-as | llc
%lldb.compile_unit = type { uint, ushort, ushort, sbyte*, sbyte*, sbyte*, {  }* }
%d.compile_unit7 = external global %lldb.compile_unit		; <%lldb.compile_unit*> [#uses=1]

implementation   ; Functions:

declare void %llvm.dbg.stoppoint(uint, uint, %lldb.compile_unit*)

void %rb_raise(int, ...) {
entry:
	br bool false, label %strlen.exit, label %no_exit.i

no_exit.i:		; preds = %entry
	ret void

strlen.exit:		; preds = %entry
	call void %llvm.dbg.stoppoint(uint 4358, uint 0, %lldb.compile_unit* %d.compile_unit7 )		; <{  }*> [#uses=0]
	unreachable
}

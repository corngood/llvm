; RUN: llc < %s -enable-eh
; PR1833

	%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
	%struct.__type_info_pseudo = type { i8*, i8* }
@_ZTI2e1 = external constant %struct.__class_type_info_pseudo		; <%struct.__class_type_info_pseudo*> [#uses=1]

define void @_Z7ex_testv() {
entry:
	invoke void @__cxa_throw( i8* null, i8* bitcast (%struct.__class_type_info_pseudo* @_ZTI2e1 to i8*), void (i8*)* null ) noreturn 
			to label %UnifiedUnreachableBlock unwind label %lpad

bb14:		; preds = %lpad
	unreachable

lpad:		; preds = %entry
	invoke void @__cxa_end_catch( )
			to label %bb14 unwind label %lpad17

lpad17:		; preds = %lpad
	%eh_select20 = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32( i8* null, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null )		; <i32> [#uses=0]
	unreachable

UnifiedUnreachableBlock:		; preds = %entry
	unreachable
}

declare void @__cxa_throw(i8*, i8*, void (i8*)*) noreturn 

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...)

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...)

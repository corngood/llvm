; Crash in post dominator set construction.
;
; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -postdomset
; XFAIL: *

implementation

int "postdomsettest"()
begin
        br label %L2Top

L2Top:
	br bool true, label %L2End, label %L2Body

L2Body:
	br label %L2Top

L2End:
	ret int 0
end


; This test makes sure that add instructions are properly eliminated.
;
; This also tests that a subtract with a constant is properly converted
; to a add w/negative constant

; RUN: if as < %s | opt -instcombine -dce | dis | grep add
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A)
begin
	%B = add int %A, 0
	ret int %B
end

int "test2"(int %A)
begin
	%B = add int %A, 5
	%C = add int %B, -5
	ret int %C
end

int "test3"(int %A)
begin
	%B = add int %A, 5
	%C = sub int %B, 5   ;; This should get converted to an add
	ret int %C
end


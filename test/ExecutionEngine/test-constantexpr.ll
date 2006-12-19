; RUN: llvm-upgrade < %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null

; This tests to make sure that we can evaluate weird constant expressions
%A = global int 5
%B = global int 6

implementation

int %main() {
	%A = or bool false, setlt (int* %A, int* %B)  ; Which is lower in memory?
	ret int 0
}


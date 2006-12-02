; Test that ds-aa is returning must-alias information when it can.
; XFAIL: *
; RUN: llvm-upgrade < %s | llvm-as | opt -no-aa -ds-aa -load-vn -gcse | llvm-dis | not grep load

%X = internal global int 20

implementation

int* %id(int* %P) { ret int* %P }

int %main() {
	store int 0, int* %X
	%XP = call int* %id(int* %X)
	%A = load int* %XP        ; Should eliminate load!
	ret int %A
}


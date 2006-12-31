; RUN: llvm-upgrade < %s | llvm-as -f -o %t.bc
; RUN: lli %t.bc > /dev/null

target endian = little
target pointersize = 32

implementation

int %main() {
entry:
	br label %endif
then:
	br label %endif
endif:
	%x = phi uint [ 4, %entry ], [ 27, %then ]
	%result = phi int [ 32, %then ], [ 0, %entry ]
	ret int 0
}

//===---------------------------------------------------------------------===//
// Random ideas for the ARM backend.
//===---------------------------------------------------------------------===//

Consider implementing a select with two conditional moves:

cmp x, y
moveq dst, a
movne dst, b

----------------------------------------------------------


%tmp1 = shl int %b, ubyte %c
%tmp4 = add int %a, %tmp1

compiles to

add r0, r0, r1, lsl r2

but

%tmp1 = shl int %b, ubyte %c
%tmp4 = add int %tmp1, %a

compiles to
mov r1, r1, lsl r2
add r0, r1, r0

---------------------------------------------------------
%tmp1 = shl int %b, ubyte 4
%tmp2 = add int %a, %tmp1

compiles to

mov r2, #4
add r0, r0, r1, lsl r2

should be

add r0, r0, r1, lsl #4

----------------------------------------------------------

add an offset to FLDS/FLDD/FSTD/FSTS addressing mode

----------------------------------------------------------

the function

void %f() {
entry:
	call void %g( int 1, int 2, int 3, int 4, int 5 )
	ret void
}

declare void %g(int, int, int, int, int)

Only needs 8 bytes of stack space. We currently allocate 16.

----------------------------------------------------------

32 x 32 -> 64 multiplications currently uses two instructions. We
should try to declare smull and umull as returning two values.

----------------------------------------------------------

Implement addressing modes 2 (ldrb) and 3 (ldrsb)

----------------------------------------------------------

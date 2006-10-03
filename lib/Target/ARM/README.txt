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

----------------------------------------------------------

add an offset to FLDS addressing mode

----------------------------------------------------------

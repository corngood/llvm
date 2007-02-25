//===---------------------------------------------------------------------===//
// Random ideas for the ARM backend.
//===---------------------------------------------------------------------===//

Reimplement 'select' in terms of 'SEL'.

* We would really like to support UXTAB16, but we need to prove that the
  add doesn't need to overflow between the two 16-bit chunks.

* implement predication support
* Implement pre/post increment support.  (e.g. PR935)
* Coalesce stack slots!
* Implement smarter constant generation for binops with large immediates.

* Consider materializing FP constants like 0.0f and 1.0f using integer 
  immediate instructions then copy to FPU.  Slower than load into FPU?

//===---------------------------------------------------------------------===//

The constant island pass has been much improved; all the todo items in the
previous version of this document have been addressed.  However, there are still
things that can be done:

1.  When there isn't an existing water, the current MBB is split right after 
the use.  It would be profitable to look farther forward, especially on Thumb,
where negative offsets won't work.
(Partially fixed:  it will put the island at the end of the block if that is 
in range.  If it is not in range things still work as above, which is poor on 
Thumb.)

2.  There may be some advantage to trying to be smarter about the initial
placement, rather than putting everything at the end.

3.  The handling of 2-byte padding for Thumb is overly conservative.  There 
would be a small gain to keeping accurate track of the padding (which would
require aligning functions containing constant pools to 4-byte boundaries).

//===---------------------------------------------------------------------===//

We need to start generating predicated instructions.  The .td files have a way
to express this now (see the PPC conditional return instruction), but the 
branch folding pass (or a new if-cvt pass) should start producing these, at
least in the trivial case.

Among the obvious wins, doing so can eliminate the need to custom expand 
copysign (i.e. we won't need to custom expand it to get the conditional
negate).

This allows us to eliminate one instruction from:

define i32 @_Z6slow4bii(i32 %x, i32 %y) {
        %tmp = icmp sgt i32 %x, %y
        %retval = select i1 %tmp, i32 %x, i32 %y
        ret i32 %retval
}

__Z6slow4bii:
        cmp r0, r1
        movgt r1, r0
        mov r0, r1
        bx lr

//===---------------------------------------------------------------------===//

Implement long long "X-3" with instructions that fold the immediate in.  These
were disabled due to badness with the ARM carry flag on subtracts.

//===---------------------------------------------------------------------===//

We currently compile abs:
int foo(int p) { return p < 0 ? -p : p; }

into:

_foo:
        rsb r1, r0, #0
        cmn r0, #1
        movgt r1, r0
        mov r0, r1
        bx lr

This is very, uh, literal.  This could be a 3 operation sequence:
  t = (p sra 31); 
  res = (p xor t)-t

Which would be better.  This occurs in png decode.

//===---------------------------------------------------------------------===//

More load / store optimizations:
1) Look past instructions without side-effects (not load, store, branch, etc.)
   when forming the list of loads / stores to optimize.

2) Smarter register allocation?
We are probably missing some opportunities to use ldm / stm. Consider:

ldr r5, [r0]
ldr r4, [r0, #4]

This cannot be merged into a ldm. Perhaps we will need to do the transformation
before register allocation. Then teach the register allocator to allocate a
chunk of consecutive registers.

3) Better representation for block transfer? This is from Olden/power:

	fldd d0, [r4]
	fstd d0, [r4, #+32]
	fldd d0, [r4, #+8]
	fstd d0, [r4, #+40]
	fldd d0, [r4, #+16]
	fstd d0, [r4, #+48]
	fldd d0, [r4, #+24]
	fstd d0, [r4, #+56]

If we can spare the registers, it would be better to use fldm and fstm here.
Need major register allocator enhancement though.

4) Can we recognize the relative position of constantpool entries? i.e. Treat

	ldr r0, LCPI17_3
	ldr r1, LCPI17_4
	ldr r2, LCPI17_5

   as
	ldr r0, LCPI17
	ldr r1, LCPI17+4
	ldr r2, LCPI17+8

   Then the ldr's can be combined into a single ldm. See Olden/power.

Note for ARM v4 gcc uses ldmia to load a pair of 32-bit values to represent a
double 64-bit FP constant:

	adr	r0, L6
	ldmia	r0, {r0-r1}

	.align 2
L6:
	.long	-858993459
	.long	1074318540

5) Can we make use of ldrd and strd? Instead of generating ldm / stm, use
ldrd/strd instead if there are only two destination registers that form an
odd/even pair. However, we probably would pay a penalty if the address is not
aligned on 8-byte boundary. This requires more information on load / store
nodes (and MI's?) then we currently carry.

//===---------------------------------------------------------------------===//

* Consider this silly example:

double bar(double x) {  
  double r = foo(3.1);
  return x+r;
}

_bar:
	sub sp, sp, #16
	str r4, [sp, #+12]
	str r5, [sp, #+8]
	str lr, [sp, #+4]
	mov r4, r0
	mov r5, r1
	ldr r0, LCPI2_0
	bl _foo
	fmsr f0, r0
	fcvtsd d0, f0
	fmdrr d1, r4, r5
	faddd d0, d0, d1
	fmrrd r0, r1, d0
	ldr lr, [sp, #+4]
	ldr r5, [sp, #+8]
	ldr r4, [sp, #+12]
	add sp, sp, #16
	bx lr

Ignore the prologue and epilogue stuff for a second. Note 
	mov r4, r0
	mov r5, r1
the copys to callee-save registers and the fact they are only being used by the
fmdrr instruction. It would have been better had the fmdrr been scheduled
before the call and place the result in a callee-save DPR register. The two
mov ops would not have been necessary.

//===---------------------------------------------------------------------===//

Calling convention related stuff:

* gcc's parameter passing implementation is terrible and we suffer as a result:

e.g.
struct s {
  double d1;
  int s1;
};

void foo(struct s S) {
  printf("%g, %d\n", S.d1, S.s1);
}

'S' is passed via registers r0, r1, r2. But gcc stores them to the stack, and
then reload them to r1, r2, and r3 before issuing the call (r0 contains the
address of the format string):

	stmfd	sp!, {r7, lr}
	add	r7, sp, #0
	sub	sp, sp, #12
	stmia	sp, {r0, r1, r2}
	ldmia	sp, {r1-r2}
	ldr	r0, L5
	ldr	r3, [sp, #8]
L2:
	add	r0, pc, r0
	bl	L_printf$stub

Instead of a stmia, ldmia, and a ldr, wouldn't it be better to do three moves?

* Return an aggregate type is even worse:

e.g.
struct s foo(void) {
  struct s S = {1.1, 2};
  return S;
}

	mov	ip, r0
	ldr	r0, L5
	sub	sp, sp, #12
L2:
	add	r0, pc, r0
	@ lr needed for prologue
	ldmia	r0, {r0, r1, r2}
	stmia	sp, {r0, r1, r2}
	stmia	ip, {r0, r1, r2}
	mov	r0, ip
	add	sp, sp, #12
	bx	lr

r0 (and later ip) is the hidden parameter from caller to store the value in. The
first ldmia loads the constants into r0, r1, r2. The last stmia stores r0, r1,
r2 into the address passed in. However, there is one additional stmia that
stores r0, r1, and r2 to some stack location. The store is dead.

The llvm-gcc generated code looks like this:

csretcc void %foo(%struct.s* %agg.result) {
entry:
	%S = alloca %struct.s, align 4		; <%struct.s*> [#uses=1]
	%memtmp = alloca %struct.s		; <%struct.s*> [#uses=1]
	cast %struct.s* %S to sbyte*		; <sbyte*>:0 [#uses=2]
	call void %llvm.memcpy.i32( sbyte* %0, sbyte* cast ({ double, int }* %C.0.904 to sbyte*), uint 12, uint 4 )
	cast %struct.s* %agg.result to sbyte*		; <sbyte*>:1 [#uses=2]
	call void %llvm.memcpy.i32( sbyte* %1, sbyte* %0, uint 12, uint 0 )
	cast %struct.s* %memtmp to sbyte*		; <sbyte*>:2 [#uses=1]
	call void %llvm.memcpy.i32( sbyte* %2, sbyte* %1, uint 12, uint 0 )
	ret void
}

llc ends up issuing two memcpy's (the first memcpy becomes 3 loads from
constantpool). Perhaps we should 1) fix llvm-gcc so the memcpy is translated
into a number of load and stores, or 2) custom lower memcpy (of small size) to
be ldmia / stmia. I think option 2 is better but the current register
allocator cannot allocate a chunk of registers at a time.

A feasible temporary solution is to use specific physical registers at the
lowering time for small (<= 4 words?) transfer size.

* ARM CSRet calling convention requires the hidden argument to be returned by
the callee.

//===---------------------------------------------------------------------===//

We can definitely do a better job on BB placements to eliminate some branches.
It's very common to see llvm generated assembly code that looks like this:

LBB3:
 ...
LBB4:
...
  beq LBB3
  b LBB2

If BB4 is the only predecessor of BB3, then we can emit BB3 after BB4. We can
then eliminate beq and and turn the unconditional branch to LBB2 to a bne.

See McCat/18-imp/ComputeBoundingBoxes for an example.

//===---------------------------------------------------------------------===//

We need register scavenging.  Currently, the 'ip' register is reserved in case
frame indexes are too big.  This means that we generate extra code for stuff 
like this:

void foo(unsigned x, unsigned y, unsigned z, unsigned *a, unsigned *b, unsigned *c) { 
   short Rconst = (short) (16384.0f * 1.40200 + 0.5 );
   *a = x * Rconst;
   *b = y * Rconst;
   *c = z * Rconst;
}

we compile it to:

_foo:
***     stmfd sp!, {r4, r7}
***     add r7, sp, #4
        mov r4, #186
        orr r4, r4, #89, 24 @ 22784
        mul r0, r0, r4
        str r0, [r3]
        mul r0, r1, r4
        ldr r1, [sp, #+8]
        str r0, [r1]
        mul r0, r2, r4
        ldr r1, [sp, #+12]
        str r0, [r1]
***     sub sp, r7, #4
***     ldmfd sp!, {r4, r7}
        bx lr

GCC produces:

_foo:
        ldr     ip, L4
        mul     r0, ip, r0
        mul     r1, ip, r1
        str     r0, [r3, #0]
        ldr     r3, [sp, #0]
        mul     r2, ip, r2
        str     r1, [r3, #0]
        ldr     r3, [sp, #4]
        str     r2, [r3, #0]
        bx      lr
L4:
        .long   22970

This is apparently all because we couldn't use ip here.

//===---------------------------------------------------------------------===//

Pre-/post- indexed load / stores:

1) We should not make the pre/post- indexed load/store transform if the base ptr
is guaranteed to be live beyond the load/store. This can happen if the base
ptr is live out of the block we are performing the optimization. e.g.

mov r1, r2
ldr r3, [r1], #4
...

vs.

ldr r3, [r2]
add r1, r2, #4
...

In most cases, this is just a wasted optimization. However, sometimes it can
negatively impact the performance because two-address code is more restrictive
when it comes to scheduling.

Unfortunately, liveout information is currently unavailable during DAG combine
time.

2) Consider spliting a indexed load / store into a pair of add/sub + load/store
   to solve #1 (in TwoAddressInstructionPass.cpp).

3) Enhance LSR to generate more opportunities for indexed ops.

4) Once we added support for multiple result patterns, write indexed loads
   patterns instead of C++ instruction selection code.

5) Use FLDM / FSTM to emulate indexed FP load / store.

//===---------------------------------------------------------------------===//

We should add i64 support to take advantage of the 64-bit load / stores.
We can add a pseudo i64 register class containing pseudo registers that are
register pairs. All other ops (e.g. add, sub) would be expanded as usual.

We need to add pseudo instructions (i.e. gethi / getlo) to extract i32 registers
from the i64 register. These are single moves which can be eliminated if the
destination register is a sub-register of the source. We should implement proper
subreg support in the register allocator to coalesce these away.

There are other minor issues such as multiple instructions for a spill / restore
/ move.

//===---------------------------------------------------------------------===//

Implement support for some more tricky ways to materialize immediates.  For
example, to get 0xffff8000, we can use:

mov r9, #&3f8000
sub r9, r9, #&400000

//===---------------------------------------------------------------------===//

We sometimes generate multiple add / sub instructions to update sp in prologue
and epilogue if the inc / dec value is too large to fit in a single immediate
operand. In some cases, perhaps it might be better to load the value from a
constantpool instead.

//===---------------------------------------------------------------------===//

GCC generates significantly better code for this function.

int foo(int StackPtr, unsigned char *Line, unsigned char *Stack, int LineLen) {
    int i = 0;

    if (StackPtr != 0) {
       while (StackPtr != 0 && i < (((LineLen) < (32768))? (LineLen) : (32768)))
          Line[i++] = Stack[--StackPtr];
        if (LineLen > 32768)
        {
            while (StackPtr != 0 && i < LineLen)
            {
                i++;
                --StackPtr;
            }
        }
    }
    return StackPtr;
}

//===---------------------------------------------------------------------===//

This should compile to the mlas instruction:
int mlas(int x, int y, int z) { return ((x * y + z) < 0) ? 7 : 13; }

//===---------------------------------------------------------------------===//

At some point, we should triage these to see if they still apply to us:

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=19598
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=18560
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=27016

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=11831
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=11826
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=11825
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=11824
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=11823
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=11820
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=10982

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=10242
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=9831
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=9760
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=9759
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=9703
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=9702
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=9663

http://www.inf.u-szeged.hu/gcc-arm/
http://citeseer.ist.psu.edu/debus04linktime.html

//===---------------------------------------------------------------------===//

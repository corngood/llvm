TODO:
* gpr0 allocation
* implement do-loop -> bdnz transform
* implement powerpc-64 for darwin

===-------------------------------------------------------------------------===

Use the stfiwx instruction for:

void foo(float a, int *b) { *b = a; }

===-------------------------------------------------------------------------===

unsigned short foo(float a) { return a; }
should be:
_foo:
        fctiwz f0,f1
        stfd f0,-8(r1)
        lhz r3,-2(r1)
        blr
not:
_foo:
        fctiwz f0, f1
        stfd f0, -8(r1)
        lwz r2, -4(r1)
        rlwinm r3, r2, 0, 16, 31
        blr

===-------------------------------------------------------------------------===

Support 'update' load/store instructions.  These are cracked on the G5, but are
still a codesize win.

===-------------------------------------------------------------------------===

Should hint to the branch select pass that it doesn't need to print the second
unconditional branch, so we don't end up with things like:
	b .LBBl42__2E_expand_function_8_674	; loopentry.24
	b .LBBl42__2E_expand_function_8_42	; NewDefault
	b .LBBl42__2E_expand_function_8_42	; NewDefault

This occurs in SPASS.

===-------------------------------------------------------------------------===

* Codegen this:

   void test2(int X) {
     if (X == 0x12345678) bar();
   }

    as:

       xoris r0,r3,0x1234
       cmplwi cr0,r0,0x5678
       beq cr0,L6

    not:

        lis r2, 4660
        ori r2, r2, 22136 
        cmpw cr0, r3, r2  
        bne .LBB_test2_2

===-------------------------------------------------------------------------===

Lump the constant pool for each function into ONE pic object, and reference
pieces of it as offsets from the start.  For functions like this (contrived
to have lots of constants obviously):

double X(double Y) { return (Y*1.23 + 4.512)*2.34 + 14.38; }

We generate:

_X:
        lis r2, ha16(.CPI_X_0)
        lfd f0, lo16(.CPI_X_0)(r2)
        lis r2, ha16(.CPI_X_1)
        lfd f2, lo16(.CPI_X_1)(r2)
        fmadd f0, f1, f0, f2
        lis r2, ha16(.CPI_X_2)
        lfd f1, lo16(.CPI_X_2)(r2)
        lis r2, ha16(.CPI_X_3)
        lfd f2, lo16(.CPI_X_3)(r2)
        fmadd f1, f0, f1, f2
        blr

It would be better to materialize .CPI_X into a register, then use immediates
off of the register to avoid the lis's.  This is even more important in PIC 
mode.

Note that this (and the static variable version) is discussed here for GCC:
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg00133.html

===-------------------------------------------------------------------------===

PIC Code Gen IPO optimization:

Squish small scalar globals together into a single global struct, allowing the 
address of the struct to be CSE'd, avoiding PIC accesses (also reduces the size
of the GOT on targets with one).

Note that this is discussed here for GCC:
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg00133.html

===-------------------------------------------------------------------------===

Implement Newton-Rhapson method for improving estimate instructions to the
correct accuracy, and implementing divide as multiply by reciprocal when it has
more than one use.  Itanium will want this too.

===-------------------------------------------------------------------------===

#define  ARRAY_LENGTH  16

union bitfield {
	struct {
#ifndef	__ppc__
		unsigned int                       field0 : 6;
		unsigned int                       field1 : 6;
		unsigned int                       field2 : 6;
		unsigned int                       field3 : 6;
		unsigned int                       field4 : 3;
		unsigned int                       field5 : 4;
		unsigned int                       field6 : 1;
#else
		unsigned int                       field6 : 1;
		unsigned int                       field5 : 4;
		unsigned int                       field4 : 3;
		unsigned int                       field3 : 6;
		unsigned int                       field2 : 6;
		unsigned int                       field1 : 6;
		unsigned int                       field0 : 6;
#endif
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
};


typedef struct program_t {
	union bitfield    array[ARRAY_LENGTH];
    int               size;
    int               loaded;
} program;


void AdjustBitfields(program* prog, unsigned int fmt1)
{
        prog->array[0].bitfields.field0 = fmt1;
        prog->array[0].bitfields.field1 = fmt1 + 1;
}

We currently generate:

_AdjustBitfields:
        lwz r2, 0(r3)
        addi r5, r4, 1
        rlwinm r2, r2, 0, 0, 19
        rlwinm r5, r5, 6, 20, 25
        rlwimi r2, r4, 0, 26, 31
        or r2, r2, r5
        stw r2, 0(r3)
        blr

We should teach someone that or (rlwimi, rlwinm) with disjoint masks can be
turned into rlwimi (rlwimi)

The better codegen would be:

_AdjustBitfields:
        lwz r0,0(r3)
        rlwinm r4,r4,0,0xff
        rlwimi r0,r4,0,26,31
        addi r4,r4,1
        rlwimi r0,r4,6,20,25
        stw r0,0(r3)
        blr

===-------------------------------------------------------------------------===

Compile this:

int %f1(int %a, int %b) {
        %tmp.1 = and int %a, 15         ; <int> [#uses=1]
        %tmp.3 = and int %b, 240                ; <int> [#uses=1]
        %tmp.4 = or int %tmp.3, %tmp.1          ; <int> [#uses=1]
        ret int %tmp.4
}

without a copy.  We make this currently:

_f1:
        rlwinm r2, r4, 0, 24, 27
        rlwimi r2, r3, 0, 28, 31
        or r3, r2, r2
        blr

The two-addr pass or RA needs to learn when it is profitable to commute an
instruction to avoid a copy AFTER the 2-addr instruction.  The 2-addr pass
currently only commutes to avoid inserting a copy BEFORE the two addr instr.

===-------------------------------------------------------------------------===

Compile offsets from allocas:

int *%test() {
        %X = alloca { int, int }
        %Y = getelementptr {int,int}* %X, int 0, uint 1
        ret int* %Y
}

into a single add, not two:

_test:
        addi r2, r1, -8
        addi r3, r2, 4
        blr

--> important for C++.

===-------------------------------------------------------------------------===

int test3(int a, int b) { return (a < 0) ? a : 0; }

should be branch free code.  LLVM is turning it into < 1 because of the RHS.

===-------------------------------------------------------------------------===

No loads or stores of the constants should be needed:

struct foo { double X, Y; };
void xxx(struct foo F);
void bar() { struct foo R = { 1.0, 2.0 }; xxx(R); }

===-------------------------------------------------------------------------===

Darwin Stub LICM optimization:

Loops like this:
  
  for (...)  bar();

Have to go through an indirect stub if bar is external or linkonce.  It would 
be better to compile it as:

     fp = &bar;
     for (...)  fp();

which only computes the address of bar once (instead of each time through the 
stub).  This is Darwin specific and would have to be done in the code generator.
Probably not a win on x86.

===-------------------------------------------------------------------------===

PowerPC i1/setcc stuff (depends on subreg stuff):

Check out the PPC code we get for 'compare' in this testcase:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=19672

oof.  on top of not doing the logical crnand instead of (mfcr, mfcr,
invert, invert, or), we then have to compare it against zero instead of
using the value already in a CR!

that should be something like
        cmpw cr7, r8, r5
        cmpw cr0, r7, r3
        crnand cr0, cr0, cr7
        bne cr0, LBB_compare_4

instead of
        cmpw cr7, r8, r5
        cmpw cr0, r7, r3
        mfcr r7, 1
        mcrf cr7, cr0
        mfcr r8, 1
        rlwinm r7, r7, 30, 31, 31
        rlwinm r8, r8, 30, 31, 31
        xori r7, r7, 1
        xori r8, r8, 1
        addi r2, r2, 1
        or r7, r8, r7
        cmpwi cr0, r7, 0
        bne cr0, LBB_compare_4  ; loopexit

FreeBench/mason has a basic block that looks like this:

         %tmp.130 = seteq int %p.0__, 5          ; <bool> [#uses=1]
         %tmp.134 = seteq int %p.1__, 6          ; <bool> [#uses=1]
         %tmp.139 = seteq int %p.2__, 12         ; <bool> [#uses=1]
         %tmp.144 = seteq int %p.3__, 13         ; <bool> [#uses=1]
         %tmp.149 = seteq int %p.4__, 14         ; <bool> [#uses=1]
         %tmp.154 = seteq int %p.5__, 15         ; <bool> [#uses=1]
         %bothcond = and bool %tmp.134, %tmp.130         ; <bool> [#uses=1]
         %bothcond123 = and bool %bothcond, %tmp.139             ; <bool>
         %bothcond124 = and bool %bothcond123, %tmp.144          ; <bool>
         %bothcond125 = and bool %bothcond124, %tmp.149          ; <bool>
         %bothcond126 = and bool %bothcond125, %tmp.154          ; <bool>
         br bool %bothcond126, label %shortcirc_next.5, label %else.0

This is a particularly important case where handling CRs better will help.

===-------------------------------------------------------------------------===

Simple IPO for argument passing, change:
  void foo(int X, double Y, int Z) -> void foo(int X, int Z, double Y)

the Darwin ABI specifies that any integer arguments in the first 32 bytes worth
of arguments get assigned to r3 through r10. That is, if you have a function
foo(int, double, int) you get r3, f1, r6, since the 64 bit double ate up the
argument bytes for r4 and r5. The trick then would be to shuffle the argument
order for functions we can internalize so that the maximum number of 
integers/pointers get passed in regs before you see any of the fp arguments.

Instead of implementing this, it would actually probably be easier to just 
implement a PPC fastcc, where we could do whatever we wanted to the CC, 
including having this work sanely.

===-------------------------------------------------------------------------===

Fix Darwin FP-In-Integer Registers ABI

Darwin passes doubles in structures in integer registers, which is very very 
bad.  Add something like a BIT_CONVERT to LLVM, then do an i-p transformation 
that percolates these things out of functions.

Check out how horrible this is:
http://gcc.gnu.org/ml/gcc/2005-10/msg01036.html

This is an extension of "interprocedural CC unmunging" that can't be done with
just fastcc.

===-------------------------------------------------------------------------===

Generate lwbrx and other byteswapping load/store instructions when reasonable.

===-------------------------------------------------------------------------===

Implement TargetConstantVec, and set up PPC to custom lower ConstantVec into
TargetConstantVec's if it's one of the many forms that are algorithmically
computable using the spiffy altivec instructions.

===-------------------------------------------------------------------------===

Compile this:

double %test(double %X) {
        %Y = cast double %X to long
        %Z = cast long %Y to double
        ret double %Z
}

to this:

_test:
        fctidz f0, f1
        stfd f0, -8(r1)
        lwz r2, -4(r1)
        lwz r3, -8(r1)
        stw r2, -12(r1)
        stw r3, -16(r1)
        lfd f0, -16(r1)
        fcfid f1, f0
        blr

without the lwz/stw's.

===-------------------------------------------------------------------------===

Compile this:

int foo(int a) {
  int b = (a < 8);
  if (b) {
    return b * 3;     // ignore the fact that this is always 3.
  } else {
    return 2;
  }
}

into something not this:

_foo:
1)      cmpwi cr7, r3, 8
        mfcr r2, 1
        rlwinm r2, r2, 29, 31, 31
1)      cmpwi cr0, r3, 7
        bgt cr0, LBB1_2 ; UnifiedReturnBlock
LBB1_1: ; then
        rlwinm r2, r2, 0, 31, 31
        mulli r3, r2, 3
        blr
LBB1_2: ; UnifiedReturnBlock
        li r3, 2
        blr

In particular, the two compares (marked 1) could be shared by reversing one.
This could be done in the dag combiner, by swapping a BR_CC when a SETCC of the
same operands (but backwards) exists.  In this case, this wouldn't save us 
anything though, because the compares still wouldn't be shared.

===-------------------------------------------------------------------------===

The legalizer should lower this:

bool %test(ulong %x) {
  %tmp = setlt ulong %x, 4294967296
  ret bool %tmp
}

into "if x.high == 0", not:

_test:
        addi r2, r3, -1
        cntlzw r2, r2
        cntlzw r3, r3
        srwi r2, r2, 5
        srwi r4, r3, 5
        li r3, 0
        cmpwi cr0, r2, 0
        bne cr0, LBB1_2 ; 
LBB1_1:
        or r3, r4, r4
LBB1_2:
        blr

noticed in 2005-05-11-Popcount-ffs-fls.c.


===-------------------------------------------------------------------------===

We should custom expand setcc instead of pretending that we have it.  That
would allow us to expose the access of the crbit after the mfcr, allowing
that access to be trivially folded into other ops.  A simple example:

int foo(int a, int b) { return (a < b) << 4; }

compiles into:

_foo:
        cmpw cr7, r3, r4
        mfcr r2, 1
        rlwinm r2, r2, 29, 31, 31
        slwi r3, r2, 4
        blr

===-------------------------------------------------------------------------===

Fold add and sub with constant into non-extern, non-weak addresses so this:

static int a;
void bar(int b) { a = b; }
void foo(unsigned char *c) {
  *c = a;
}

So that 

_foo:
        lis r2, ha16(_a)
        la r2, lo16(_a)(r2)
        lbz r2, 3(r2)
        stb r2, 0(r3)
        blr

Becomes

_foo:
        lis r2, ha16(_a+3)
        lbz r2, lo16(_a+3)(r2)
        stb r2, 0(r3)
        blr

===-------------------------------------------------------------------------===

We generate really bad code for this:

int f(signed char *a, _Bool b, _Bool c) {
   signed char t = 0;
  if (b)  t = *a;
  if (c)  *a = t;
}


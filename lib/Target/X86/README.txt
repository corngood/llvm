//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend.
//===---------------------------------------------------------------------===//

Add a MUL2U and MUL2S nodes to represent a multiply that returns both the
Hi and Lo parts (combination of MUL and MULH[SU] into one node).  Add this to
X86, & make the dag combiner produce it when needed.  This will eliminate one
imul from the code generated for:

long long test(long long X, long long Y) { return X*Y; }

by using the EAX result from the mul.  We should add a similar node for
DIVREM.

another case is:

long long test(int X, int Y) { return (long long)X*Y; }

... which should only be one imul instruction.

//===---------------------------------------------------------------------===//

This should be one DIV/IDIV instruction, not a libcall:

unsigned test(unsigned long long X, unsigned Y) {
        return X/Y;
}

This can be done trivially with a custom legalizer.  What about overflow 
though?  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=14224

//===---------------------------------------------------------------------===//

Some targets (e.g. athlons) prefer freep to fstp ST(0):
http://gcc.gnu.org/ml/gcc-patches/2004-04/msg00659.html

//===---------------------------------------------------------------------===//

This should use fiadd on chips where it is profitable:
double foo(double P, int *I) { return P+*I; }

//===---------------------------------------------------------------------===//

The FP stackifier needs to be global.  Also, it should handle simple permutates
to reduce number of shuffle instructions, e.g. turning:

fld P	->		fld Q
fld Q			fld P
fxch

or:

fxch	->		fucomi
fucomi			jl X
jg X

Ideas:
http://gcc.gnu.org/ml/gcc-patches/2004-11/msg02410.html


//===---------------------------------------------------------------------===//

Improvements to the multiply -> shift/add algorithm:
http://gcc.gnu.org/ml/gcc-patches/2004-08/msg01590.html

//===---------------------------------------------------------------------===//

Improve code like this (occurs fairly frequently, e.g. in LLVM):
long long foo(int x) { return 1LL << x; }

http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01109.html
http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01128.html
http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01136.html

Another useful one would be  ~0ULL >> X and ~0ULL << X.

//===---------------------------------------------------------------------===//

Should support emission of the bswap instruction, probably by adding a new
DAG node for byte swapping.  Also useful on PPC which has byte-swapping loads.

//===---------------------------------------------------------------------===//

Compile this:
_Bool f(_Bool a) { return a!=1; }

into:
        movzbl  %dil, %eax
        xorl    $1, %eax
        ret

//===---------------------------------------------------------------------===//

Some isel ideas:

1. Dynamic programming based approach when compile time if not an
   issue.
2. Code duplication (addressing mode) during isel.
3. Other ideas from "Register-Sensitive Selection, Duplication, and
   Sequencing of Instructions".

//===---------------------------------------------------------------------===//

Should we promote i16 to i32 to avoid partial register update stalls?

//===---------------------------------------------------------------------===//

Leave any_extend as pseudo instruction and hint to register
allocator. Delay codegen until post register allocation.

//===---------------------------------------------------------------------===//

Add a target specific hook to DAG combiner to handle SINT_TO_FP and
FP_TO_SINT when the source operand is already in memory.

//===---------------------------------------------------------------------===//

Check if load folding would add a cycle in the dag.

//===---------------------------------------------------------------------===//

Model X86 EFLAGS as a real register to avoid redudant cmp / test. e.g.

	cmpl $1, %eax
	setg %al
	testb %al, %al  # unnecessary
	jne .BB7

//===---------------------------------------------------------------------===//

Count leading zeros and count trailing zeros:

int clz(int X) { return __builtin_clz(X); }
int ctz(int X) { return __builtin_ctz(X); }

$ gcc t.c -S -o - -O3  -fomit-frame-pointer -masm=intel
clz:
        bsr     %eax, DWORD PTR [%esp+4]
        xor     %eax, 31
        ret
ctz:
        bsf     %eax, DWORD PTR [%esp+4]
        ret

however, check that these are defined for 0 and 32.  Our intrinsics are, GCC's
aren't.

//===---------------------------------------------------------------------===//

Use push/pop instructions in prolog/epilog sequences instead of stores off 
ESP (certain code size win, perf win on some [which?] processors).

//===---------------------------------------------------------------------===//

Only use inc/neg/not instructions on processors where they are faster than
add/sub/xor.  They are slower on the P4 due to only updating some processor
flags.

//===---------------------------------------------------------------------===//

Open code rint,floor,ceil,trunc:
http://gcc.gnu.org/ml/gcc-patches/2004-08/msg02006.html
http://gcc.gnu.org/ml/gcc-patches/2004-08/msg02011.html

//===---------------------------------------------------------------------===//

Combine: a = sin(x), b = cos(x) into a,b = sincos(x).


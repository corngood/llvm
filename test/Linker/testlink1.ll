; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < `dirname %s`/testlink2.ll > %t2.bc
; RUN: llvm-link %t.bc %t2.bc

%MyVar     = external global int
%MyIntList = global { \2 *, int } { { \2, int }* null, int 17 }
             external global int      ; int*:0

%AConst    = linkonce constant int 123

%Intern1   = internal constant int 42
%Intern2   = internal constant int 792

; Initialized to point to external %MyVar
%MyVarPtr  = linkonce global { int * }  { int * %MyVar }

declare int "foo"(int %blah)      ;; Declared in testlink2.ll

declare void "print"(int %Value)

implementation

void "main"()
begin
	%v1 = load int* %MyVar
	call void %print(int %v1)    ;; Should start out 4

	%idx = getelementptr { \2 *, int }* %MyIntList, long 0, uint 1
	%v2 = load int* %idx
	call void %print(int %v2)    ;; Should start out 17

	call int %foo(int 5)         ;; Modify global variablesx

	%v3 = load int* %MyVar
	call void %print(int %v3)    ;; Should now be 5

	%v4 = load int* %idx
	call void %print(int %v4)    ;; Should start out 12

	ret void
end

internal void "testintern"() begin ret void end
internal void "Testintern"() begin ret void end
         void "testIntern"() begin ret void end


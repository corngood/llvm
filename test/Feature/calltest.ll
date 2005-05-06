; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%FunTy = type int(int)

declare int "test"(...)   ; Test differences of prototype
declare int "test"()      ; Differ only by vararg

implementation

void "invoke"(%FunTy *%x) {
	%foo = call %FunTy* %x(int 123)
	%foo2 = tail call %FunTy* %x(int 123)
	ret void
}

int "main"(int %argc)   ; TODO: , sbyte **argv, sbyte **envp)
begin
        %retval = call int (int) *%test(int %argc)
        %two    = add int %retval, %retval
	%retval2 = invoke int %test(int %argc)
		   to label %Next except label %Error
Next:
	%two2 = add int %two, %retval2
	call void %invoke (%FunTy* %test)
        ret int %two2
Error:
	ret int -1
end

int "test"(int %i0)
begin
    ret int %i0
end

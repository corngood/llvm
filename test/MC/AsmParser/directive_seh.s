# RUN: llvm-mc -triple x86_64-pc-win32 %s | FileCheck %s

# CHECK: .seh_proc func
# CHECK: .seh_pushframe @code
# CHECK: .seh_stackalloc 24
# CHECK: .seh_endprologue
# CHECK: .seh_handler __C_specific_handler, @except
# CHECK: .seh_endproc

    .text
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
func:
    .seh_pushframe @code
    subq $24, %rsp
    .seh_stackalloc 24
    .seh_endprologue
    .seh_handler __C_specific_handler, @except
    addq $24, %rsp
    ret
    .seh_endproc

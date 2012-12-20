# RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s -o - |& FileCheck %s

# Just a simple test for the assembly emitter - making sure it emits back the
# bundling directives.

  .text
foo:
  .bundle_align_mode 4
# CHECK:      .bundle_align_mode 4
  pushq   %rbp
  .bundle_lock
# CHECK: .bundle_lock
  cmpl    %r14d, %ebp
  jle     .L_ELSE
  .bundle_unlock
# CHECK: .bundle_unlock



// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we produce a R_X86_64_32S or R_X86_64_32.

bar:
        movl	$bar, %edx        // R_X86_64_32
        movq	$bar, %rdx        // R_X86_64_32S
        movq	$bar, bar(%rip)   // R_X86_64_32S
        movl	bar, %edx         // R_X86_64_32S
        movq	bar, %rdx         // R_X86_64_32S
.long bar                         // R_X86_64_32

// CHECK:  # Section 0x00000001
// CHECK: (('sh_name', 0x00000001) # '.text'

// CHECK:   # Symbol 0x00000002
// CHECK: (('st_name', 0x00000000) # ''
// CHECK:  ('st_bind', 0x00000000)
// CHECK   ('st_type', 0x00000003)
// CHECK:  ('st_other', 0x00000000)
// CHECK:  ('st_shndx', 0x00000001)

// CHECK: # Relocation 0x00000000
// CHECK-NEXT:  (('r_offset', 0x00000001)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000a)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x00000001
// CHECK-NEXT:  (('r_offset', 0x00000008)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x00000002
// CHECK-NEXT:  (('r_offset', 0x00000013)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x00000003
// CHECK-NEXT:  (('r_offset', 0x0000001a)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x00000004
// CHECK-NEXT:  (('r_offset', 0x00000022)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x00000005
// CHECK-NEXT:  (('r_offset', 0x00000026)
// CHECK-NEXT:   ('r_sym', 0x00000002)
// CHECK-NEXT:   ('r_type', 0x0000000a)
// CHECK-NEXT:   ('r_addend',

// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o %t
// RUN: coff-dump.py %abs_tmp | FileCheck %s
        
// Test that we get optimal nops in text
    .text
f0:
    .long 0
    .align  8, 0x90
    .long 0
    .align  8

// But not in another section
    .data
    .long 0
    .align  8, 0x90
    .long 0
    .align  8

//CHECK:         Name                     = .text
//CHECK-NEXT:    VirtualSize
//CHECK-NEXT:    VirtualAddress
//CHECK-NEXT:    SizeOfRawData            = 16
//CHECK-NEXT:    PointerToRawData
//CHECK-NEXT:    PointerToRelocations
//CHECK-NEXT:    PointerToLineNumbers
//CHECK-NEXT:    NumberOfRelocations
//CHECK-NEXT:    NumberOfLineNumbers
//CHECK-NEXT:    Charateristics           = 0x400001
//CHECK-NEXT:        IMAGE_SCN_ALIGN_8BYTES
//CHECK-NEXT:      SectionData              = 
//CHECK-NEXT:        00 00 00 00 0F 1F 40 00 - 00 00 00 00 0F 1F 40 00

//CHECK:         Name                     = .data
//CHECK-NEXT:      VirtualSize
//CHECK-NEXT:      VirtualAddress
//CHECK-NEXT:      SizeOfRawData            = 16
//CHECK-NEXT:      PointerToRawData
//CHECK-NEXT:      PointerToRelocations
//CHECK-NEXT:      PointerToLineNumbers
//CHECK-NEXT:      NumberOfRelocations
//CHECK-NEXT:      NumberOfLineNumbers
//CHECK-NEXT:      Charateristics           = 0x400001
//CHECK-NEXT:        IMAGE_SCN_ALIGN_8BYTES
//CHECK-NEXT:      SectionData              = 
//CHECK-NEXT:        00 00 00 00 90 90 90 90 - 00 00 00 00 00 00 00 00

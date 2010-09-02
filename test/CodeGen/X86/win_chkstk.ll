; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=WIN_X32
; RUN: llc < %s -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X32
; RUN: llc < %s -mtriple=x86_64-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X64
; RUN: llc < %s -mtriple=i386-pc-linux | FileCheck %s -check-prefix=LINUX

; Windows and mingw require a prologue helper routine if more than 4096 bytes area
; allocated on the stack.  Windows uses __chkstk and mingw uses __alloca.  __alloca
; and the 32-bit version of __chkstk will probe the stack and adjust the stack pointer.
; The 64-bit version of __chkstk is only responsible for probing the stack.  The 64-bit
; prologue is responsible for adjusting the stack pointer.

; Stack allocation >= 4096 bytes will require call to __chkstk in the Windows ABI.
define i32 @main4k() nounwind {
entry:
; WIN_X32:    call __chkstk
; WIN_X64:    call __chkstk
; MINGW_X32:  call __alloca
; MINGW_X64:  call _alloca
; LINUX-NOT:  call __chkstk
  %array4096 = alloca [4096 x i8], align 16       ; <[4096 x i8]*> [#uses=0]
  ret i32 0
}

; Make sure we don't call __chkstk or __alloca when we have less than a 4096 stack
; allocation.
define i32 @main128() nounwind {
entry:
; WIN_X32:       # BB#0:
; WIN_X32-NOT:   call __chkstk
; WIN_X32:       ret

; WIN_X64:       # BB#0:
; WIN_X64-NOT:   call __chkstk
; WIN_X64:       ret

; MINGW_X64:     # BB#0:
; MINGW_X64-NOT: call _alloca
; MINGW_X64:     ret

; LINUX:         # BB#0:
; LINUX-NOT:     call __chkstk
; LINUX:         ret
  %array128 = alloca [128 x i8], align 16         ; <[128 x i8]*> [#uses=0]
  ret i32 0
}

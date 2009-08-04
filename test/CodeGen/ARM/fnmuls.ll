; XFAIL: *
; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,+neonfp | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+neon,-neonfp | grep -E {fnmuls\\W*s\[0-9\]+,\\W*s\[0-9\]+,\\W*s\[0-9\]+} | count 1

define float @test(float %a, float %b) {
entry:
	%0 = fmul float %a, %b
        %1 = fsub float 0.0, %0
	ret float %1
}


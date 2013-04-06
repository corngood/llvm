add_llvm_library(LLVMAMPToOpenCL
  AMPToOpenCL.cpp
  )

add_dependencies(LLVMAMPToOpenCL intrinsics_gen)

/*===-- llvm-c/Analysis.h - Analysis Library C Interface --------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file was developed by Gordon Henriksen and is distributed under the   *|
|* University of Illinois Open Source License. See LICENSE.TXT for details.   *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMAnalysis.a, which           *|
|* implements various analyses of the LLVM IR.                                *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_ANALYSIS_H
#define LLVM_C_ANALYSIS_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
  LLVMAbortProcessAction, /* verifier will print to stderr and abort() */
  LLVMPrintMessageAction, /* verifier will print to stderr and return 1 */
  LLVMReturnStatusAction  /* verifier will just return 1 */
} LLVMVerifierFailureAction;


/* Verifies that a module is valid, taking the specified action if not.
   Optionally returns a human-readable description of any invalid constructs.
   OutMessage must be disposed with LLVMDisposeMessage. */
int LLVMVerifyModule(LLVMModuleRef M, LLVMVerifierFailureAction Action,
                     char **OutMessage);

/* Verifies that a single function is valid, taking the specified action. Useful
   for debugging. */
int LLVMVerifyFunction(LLVMValueRef Fn, LLVMVerifierFailureAction Action);


#ifdef __cplusplus
}
#endif

#endif

(*===-- llvm_scalar_opts.mli - LLVM Ocaml Interface ------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Scalar Transforms.

    This interface provides an ocaml API for LLVM scalar transforms, the
    classes in the [LLVMScalarOpts] library. *)

(** See the [llvm::createConstantPropogationPass] function. *)
external add_constant_propagation : [<Llvm.PassManager.any] Llvm.PassManager.t
                                    -> unit
                                  = "llvm_add_constant_propagation"

(** See the [llvm::createSCCPPass] function. *)
external add_sccp : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                  = "llvm_add_sccp"

(** See [llvm::createDeadStoreEliminationPass] function. *)
external add_dead_store_elimination : [<Llvm.PassManager.any] Llvm.PassManager.t
                                      -> unit
                                    = "llvm_add_dead_store_elimination"

(** See The [llvm::createAggressiveDCEPass] function. *)
external add_aggressive_dce : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                            = "llvm_add_aggressive_dce"

(** See the [llvm::createScalarReplAggregatesPass] function. *)
external
add_scalar_repl_aggregation : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                            = "llvm_add_scalar_repl_aggregation"

(** See the [llvm::createIndVarSimplifyPass] function. *)
external add_ind_var_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t
                                      -> unit
                                    = "llvm_add_ind_var_simplification"

(** See the [llvm::createInstructionCombiningPass] function. *)
external
add_instruction_combination : [<Llvm.PassManager.any] Llvm.PassManager.t
                              -> unit
                            = "llvm_add_instruction_combination"

(** See the [llvm::createLICMPass] function. *)
external add_licm : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_licm"

(** See the [llvm::createLoopUnswitchPass] function. *)
external add_loop_unswitch : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_loop_unswitch"

(** See the [llvm::createLoopUnrollPass] function. *)
external add_loop_unroll : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_loop_unroll"

(** See the [llvm::createLoopRotatePass] function. *)
external add_loop_rotation : [<Llvm.PassManager.any] Llvm.PassManager.t
                             -> unit
                           = "llvm_add_loop_rotation"

(** See the [llvm::createLoopIndexSplitPass] function. *)
external add_loop_index_split : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_loop_index_split"

(** See the [llvm::createPromoteMemoryToRegisterPass] function. *)
external
add_memory_to_register_promotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                   -> unit
                                 = "llvm_add_memory_to_register_promotion"

(** See the [llvm::createDemoteMemoryToRegisterPass] function. *)
external
add_memory_to_register_demotion : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_memory_to_register_demotion"

(** See the [llvm::createReassociatePass] function. *)
external add_reassociation : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                           = "llvm_add_reassociation"

(** See the [llvm::createJumpThreadingPass] function. *)
external add_jump_threading : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_jump_threading"

(** See the [llvm::createCFGSimplificationPass] function. *)
external add_cfg_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t
                                  -> unit
                                = "llvm_add_cfg_simplification"

(** See the [llvm::createTailCallEliminationPass] function. *)
external
add_tail_call_elimination : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                          = "llvm_add_tail_call_elimination" 

(** See the [llvm::createGVNPass] function. *)
external add_gvn : [<Llvm.PassManager.any] Llvm.PassManager.t
                   -> unit
                 = "llvm_add_gvn"

(** See the [llvm::createMemCpyOptPass] function. *)
external add_memcpy_opt : [<Llvm.PassManager.any] Llvm.PassManager.t
                                -> unit
                              = "llvm_add_memcpy_opt"

(** See the [llvm::createLoopDeletionPass] function. *)
external add_loop_deletion : [<Llvm.PassManager.any] Llvm.PassManager.t
                             -> unit
                           = "llvm_add_loop_deletion"

(** See the [llvm::createSimplifyLibCallsPass] function. *)
external
add_lib_call_simplification : [<Llvm.PassManager.any] Llvm.PassManager.t -> unit
                            = "llvm_add_lib_call_simplification"

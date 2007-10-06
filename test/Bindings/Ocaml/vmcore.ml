(* RUN: %ocamlc llvm.cma llvm_bitwriter.cma %s -o %t
 * RUN: ./%t %t.bc
 * RUN: llvm-dis < %t.bc > %t.ll
 *)

(* Note: It takes several seconds for ocamlc to link an executable with
         libLLVMCore.a, so it's better to write a big test than a bunch of
         little ones. *)

open Llvm
open Llvm_bitwriter


(* Tiny unit test framework - really just to help find which line is busted *)
let exit_status = ref 0
let case_num = ref 0

let group name =
  case_num := 0;
  prerr_endline ("  " ^ name ^ "...")

let insist cond =
  incr case_num;
  let msg = if cond then "    pass " else begin
    exit_status := 10;
    "    FAIL "
  end in
  prerr_endline (msg ^ (string_of_int !case_num))

let suite name f =
  prerr_endline (name ^ ":");
  f ()


(*===-- Fixture -----------------------------------------------------------===*)

let filename = Sys.argv.(1)
let m = create_module filename


(*===-- Types -------------------------------------------------------------===*)

let test_types () =
  (* RUN: grep {Ty01.*void} < %t.ll
   *)
  group "void";
  insist (define_type_name "Ty01" void_type m);
  insist (Void_type == classify_type void_type);

  (* RUN: grep {Ty02.*i1} < %t.ll
   *)
  group "i1";
  insist (define_type_name "Ty02" i1_type m);
  insist (Integer_type == classify_type i1_type);

  (* RUN: grep {Ty03.*i32} < %t.ll
   *)
  group "i32";
  insist (define_type_name "Ty03" i32_type m);

  (* RUN: grep {Ty04.*i42} < %t.ll
   *)
  group "i42";
  let ty = make_integer_type 42 in
  insist (define_type_name "Ty04" ty m);

  (* RUN: grep {Ty05.*float} < %t.ll
   *)
  group "float";
  insist (define_type_name "Ty05" float_type m);
  insist (Float_type == classify_type float_type);

  (* RUN: grep {Ty06.*double} < %t.ll
   *)
  group "double";
  insist (define_type_name "Ty06" double_type m);
  insist (Double_type == classify_type double_type);

  (* RUN: grep {Ty07.*i32.*i1, double} < %t.ll
   *)
  group "function";
  let ty = make_function_type i32_type [| i1_type; double_type |] false in
  insist (define_type_name "Ty07" ty m);
  insist (Function_type = classify_type ty);
  insist (not (is_var_arg ty));
  insist (i32_type == return_type ty);
  insist (double_type == (param_types ty).(1));
  
  (* RUN: grep {Ty08.*\.\.\.} < %t.ll
   *)
  group "vararg";
  let ty = make_function_type void_type [| i32_type |] true in
  insist (define_type_name "Ty08" ty m);
  insist (is_var_arg ty);
  
  (* RUN: grep {Ty09.*\\\[7 x i8\\\]} < %t.ll
   *)
  group "array";
  let ty = make_array_type i8_type 7 in
  insist (define_type_name "Ty09" ty m);
  insist (7 = array_length ty);
  insist (i8_type == element_type ty);
  insist (Array_type == classify_type ty);
  
  (* RUN: grep {Ty10.*float\*} < %t.ll
   *)
  group "pointer";
  let ty = make_pointer_type float_type in
  insist (define_type_name "Ty10" ty m);
  insist (float_type == element_type ty);
  insist (Pointer_type == classify_type ty);
  
  (* RUN: grep {Ty11.*\<4 x i16\>} < %t.ll
   *)
  group "vector";
  let ty = make_vector_type i16_type 4 in
  insist (define_type_name "Ty11" ty m);
  insist (i16_type == element_type ty);
  insist (4 = vector_size ty);
  
  (* RUN: grep {Ty12.*opaque} < %t.ll
   *)
  group "opaque";
  let ty = make_opaque_type () in
  insist (define_type_name "Ty12" ty m);
  insist (ty == ty);
  insist (ty <> make_opaque_type ());
  
  (* RUN: grep -v {Ty13} < %t.ll
   *)
  group "delete";
  let ty = make_opaque_type () in
  insist (define_type_name "Ty13" ty m);
  delete_type_name "Ty13" m


(*===-- Constants ---------------------------------------------------------===*)

let test_constants () =
  (* RUN: grep {Const01.*i32.*-1} < %t.ll
   *)
  group "int";
  let c = make_int_constant i32_type (-1) true in
  ignore (define_global "Const01" c m);
  insist (i32_type = type_of c);
  insist (is_constant c);

  (* RUN: grep {Const02.*i64.*-1} < %t.ll
   *)
  group "sext int";
  let c = make_int_constant i64_type (-1) true in
  ignore (define_global "Const02" c m);
  insist (i64_type = type_of c);

  (* RUN: grep {Const03.*i64.*4294967295} < %t.ll
   *)
  group "zext int64";
  let c = make_int64_constant i64_type (Int64.of_string "4294967295") false in
  ignore (define_global "Const03" c m);
  insist (i64_type = type_of c);

  (* RUN: grep {Const04.*"cruel\\\\00world"} < %t.ll
   *)
  group "string";
  let c = make_string_constant "cruel\000world" false in
  ignore (define_global "Const04" c m);
  insist ((make_array_type i8_type 11) = type_of c);

  (* RUN: grep {Const05.*"hi\\\\00again\\\\00"} < %t.ll
   *)
  group "string w/ null";
  let c = make_string_constant "hi\000again" true in
  ignore (define_global "Const05" c m);
  insist ((make_array_type i8_type 9) = type_of c);

  (* RUN: grep {Const06.*3.1459} < %t.ll
   *)
  group "real";
  let c = make_real_constant double_type 3.1459 in
  ignore (define_global "Const06" c m);
  insist (double_type = type_of c);
  
  let one = make_int_constant i16_type 1 true in
  let two = make_int_constant i16_type 2 true in
  let three = make_int_constant i32_type 3 true in
  let four = make_int_constant i32_type 4 true in
  
  (* RUN: grep {Const07.*\\\[ i32 3, i32 4 \\\]} < %t.ll
   *)
  group "array";
  let c = make_array_constant i32_type [| three; four |] in
  ignore (define_global "Const07" c m);
  insist ((make_array_type i32_type 2) = (type_of c));
  
  (* RUN: grep {Const08.*< i16 1, i16 2.* >} < %t.ll
   *)
  group "vector";
  let c = make_vector_constant [| one; two; one; two;
                                  one; two; one; two |] in
  ignore (define_global "Const08" c m);
  insist ((make_vector_type i16_type 8) = (type_of c));
  
  (* RUN: grep {Const09.*\{ i16, i16, i32, i32 \} \{} < %t.ll
   *)
  group "structure";
  let c = make_struct_constant [| one; two; three; four |] false in
  ignore (define_global "Const09" c m);
  insist ((make_struct_type [| i16_type; i16_type; i32_type; i32_type |] false)
        = (type_of c));
  
  (* RUN: grep {Const10.*zeroinit} < %t.ll
   *)
  group "null";
  let c = make_null (make_struct_type [| i1_type; i8_type;
                                         i64_type; double_type |] true) in
  ignore (define_global "Const10" c m);
  
  (* RUN: grep {Const11.*-1} < %t.ll
   *)
  group "all ones";
  let c = make_all_ones i64_type in
  ignore (define_global "Const11" c m);
  
  (* RUN: grep {Const12.*undef} < %t.ll
   *)
  group "undef";
  let c = make_undef i1_type in
  ignore (define_global "Const12" c m);
  insist (i1_type = type_of c);
  insist (is_undef c);
  
  group "constant arithmetic";
  (* RUN: grep {ConstNeg.*sub} < %t.ll
   * RUN: grep {ConstNot.*xor} < %t.ll
   * RUN: grep {ConstAdd.*add} < %t.ll
   * RUN: grep {ConstSub.*sub} < %t.ll
   * RUN: grep {ConstMul.*mul} < %t.ll
   * RUN: grep {ConstUDiv.*udiv} < %t.ll
   * RUN: grep {ConstSDiv.*sdiv} < %t.ll
   * RUN: grep {ConstFDiv.*fdiv} < %t.ll
   * RUN: grep {ConstURem.*urem} < %t.ll
   * RUN: grep {ConstSRem.*srem} < %t.ll
   * RUN: grep {ConstFRem.*frem} < %t.ll
   * RUN: grep {ConstAnd.*and} < %t.ll
   * RUN: grep {ConstOr.*or} < %t.ll
   * RUN: grep {ConstXor.*xor} < %t.ll
   * RUN: grep {ConstICmp.*icmp} < %t.ll
   * RUN: grep {ConstFCmp.*fcmp} < %t.ll
   *)
  let void_ptr = make_pointer_type i8_type in
  let five = make_int_constant i64_type 5 false in
  let ffive = const_uitofp five double_type in
  let foldbomb_gv = define_global "FoldBomb" (make_null i8_type) m in
  let foldbomb = const_ptrtoint foldbomb_gv i64_type in
  let ffoldbomb = const_uitofp foldbomb double_type in
  ignore (define_global "ConstNeg" (const_neg foldbomb) m);
  ignore (define_global "ConstNot" (const_not foldbomb) m);
  ignore (define_global "ConstAdd" (const_add foldbomb five) m);
  ignore (define_global "ConstSub" (const_sub foldbomb five) m);
  ignore (define_global "ConstMul" (const_mul foldbomb five) m);
  ignore (define_global "ConstUDiv" (const_udiv foldbomb five) m);
  ignore (define_global "ConstSDiv" (const_sdiv foldbomb five) m);
  ignore (define_global "ConstFDiv" (const_fdiv ffoldbomb ffive) m);
  ignore (define_global "ConstURem" (const_urem foldbomb five) m);
  ignore (define_global "ConstSRem" (const_srem foldbomb five) m);
  ignore (define_global "ConstFRem" (const_frem ffoldbomb ffive) m);
  ignore (define_global "ConstAnd" (const_and foldbomb five) m);
  ignore (define_global "ConstOr" (const_or foldbomb five) m);
  ignore (define_global "ConstXor" (const_xor foldbomb five) m);
  ignore (define_global "ConstICmp" (const_icmp Icmp_sle foldbomb five) m);
  ignore (define_global "ConstFCmp" (const_fcmp Fcmp_ole ffoldbomb ffive) m);
  
  group "constant casts";
  (* RUN: grep {ConstTrunc.*trunc} < %t.ll
   * RUN: grep {ConstSExt.*sext} < %t.ll
   * RUN: grep {ConstZExt.*zext} < %t.ll
   * RUN: grep {ConstFPTrunc.*fptrunc} < %t.ll
   * RUN: grep {ConstFPExt.*fpext} < %t.ll
   * RUN: grep {ConstUIToFP.*uitofp} < %t.ll
   * RUN: grep {ConstSIToFP.*sitofp} < %t.ll
   * RUN: grep {ConstFPToUI.*fptoui} < %t.ll
   * RUN: grep {ConstFPToSI.*fptosi} < %t.ll
   * RUN: grep {ConstPtrToInt.*ptrtoint} < %t.ll
   * RUN: grep {ConstIntToPtr.*inttoptr} < %t.ll
   * RUN: grep {ConstBitCast.*bitcast} < %t.ll
   *)
  let i128_type = make_integer_type 128 in
  ignore (define_global "ConstTrunc" (const_trunc (const_add foldbomb five)
                                               i8_type) m);
  ignore (define_global "ConstSExt" (const_sext foldbomb i128_type) m);
  ignore (define_global "ConstZExt" (const_zext foldbomb i128_type) m);
  ignore (define_global "ConstFPTrunc" (const_fptrunc ffoldbomb float_type) m);
  ignore (define_global "ConstFPExt" (const_fpext ffoldbomb fp128_type) m);
  ignore (define_global "ConstUIToFP" (const_uitofp foldbomb double_type) m);
  ignore (define_global "ConstSIToFP" (const_sitofp foldbomb double_type) m);
  ignore (define_global "ConstFPToUI" (const_fptoui ffoldbomb i32_type) m);
  ignore (define_global "ConstFPToSI" (const_fptosi ffoldbomb i32_type) m);
  ignore (define_global "ConstPtrToInt" (const_ptrtoint 
    (const_gep (make_null (make_pointer_type i8_type))
               [| make_int_constant i32_type 1 false |])
    i32_type) m);
  ignore (define_global "ConstIntToPtr" (const_inttoptr (const_add foldbomb five)
                                                  void_ptr) m);
  ignore (define_global "ConstBitCast" (const_bitcast ffoldbomb i64_type) m);
  
  group "misc constants";
  (* RUN: grep {ConstSizeOf.*getelementptr.*null} < %t.ll
   * RUN: grep {ConstGEP.*getelementptr} < %t.ll
   * RUN: grep {ConstSelect.*select} < %t.ll
   * RUN: grep {ConstExtractElement.*extractelement} < %t.ll
   * RUN: grep {ConstInsertElement.*insertelement} < %t.ll
   * RUN: grep {ConstShuffleVector.*shufflevector} < %t.ll
   *)
  ignore (define_global "ConstSizeOf" (sizeof (make_pointer_type i8_type)) m);
  ignore (define_global "ConstGEP" (const_gep foldbomb_gv [| five |]) m);
  ignore (define_global "ConstSelect" (const_select
    (const_icmp Icmp_sle foldbomb five)
    (make_int_constant i8_type (-1) true)
    (make_int_constant i8_type 0 true)) m);
  let zero = make_int_constant i32_type 0 false in
  let one  = make_int_constant i32_type 1 false in
  ignore (define_global "ConstExtractElement" (const_extractelement
    (make_vector_constant [| zero; one; zero; one |])
    (const_trunc foldbomb i32_type)) m);
  ignore (define_global "ConstInsertElement" (const_insertelement
    (make_vector_constant [| zero; one; zero; one |])
    zero (const_trunc foldbomb i32_type)) m);
  ignore (define_global "ConstShuffleVector" (const_shufflevector
    (make_vector_constant [| zero; one |])
    (make_vector_constant [| one; zero |])
    (const_bitcast foldbomb (make_vector_type i32_type 2))) m)


(*===-- Global Values -----------------------------------------------------===*)

let test_global_values () =
  let (++) x f = f x; x in
  let zero32 = make_null i32_type in

  (* RUN: grep {GVal01} < %t.ll
   *)
  group "naming";
  let g = define_global "TEMPORARY" zero32 m in
  insist ("TEMPORARY" = value_name g);
  set_value_name "GVal01" g;
  insist ("GVal01" = value_name g);

  (* RUN: grep {GVal02.*linkonce} < %t.ll
   *)
  group "linkage";
  let g = define_global "GVal02" zero32 m ++
          set_linkage Link_once_linkage in
  insist (Link_once_linkage = linkage g);

  (* RUN: grep {GVal03.*Hanalei} < %t.ll
   *)
  group "section";
  let g = define_global "GVal03" zero32 m ++
          set_section "Hanalei" in
  insist ("Hanalei" = section g);
  
  (* RUN: grep {GVal04.*hidden} < %t.ll
   *)
  group "visibility";
  let g = define_global "GVal04" zero32 m ++
          set_visibility Hidden_visibility in
  insist (Hidden_visibility = visibility g);
  
  (* RUN: grep {GVal05.*align 128} < %t.ll
   *)
  group "alignment";
  let g = define_global "GVal05" zero32 m ++
          set_alignment 128 in
  insist (128 = alignment g)


(*===-- Global Variables --------------------------------------------------===*)

let test_global_variables () =
  let (++) x f = f x; x in
  let fourty_two32 = make_int_constant i32_type 42 false in

  (* RUN: grep {GVar01.*external} < %t.ll
   *)
  group "declarations";
  let g = declare_global i32_type "GVar01" m in
  insist (is_declaration g);
  
  (* RUN: grep {GVar02.*42} < %t.ll
   * RUN: grep {GVar03.*42} < %t.ll
   *)
  group "definitions";
  let g = define_global "GVar02" fourty_two32 m in
  let g2 = declare_global i32_type "GVar03" m ++
           set_initializer fourty_two32 in
  insist (not (is_declaration g));
  insist (not (is_declaration g2));
  insist ((global_initializer g) == (global_initializer g2));

  (* RUN: grep {GVar04.*thread_local} < %t.ll
   *)
  group "threadlocal";
  let g = define_global "GVar04" fourty_two32 m ++
          set_thread_local true in
  insist (is_thread_local g);

  (* RUN: grep -v {GVar05} < %t.ll
   *)
  group "delete";
  let g = define_global "GVar05" fourty_two32 m in
  delete_global g


(*===-- Functions ---------------------------------------------------------===*)

let test_functions () =
  let ty = make_function_type i32_type [| i32_type; i64_type |] false in
  let pty = make_pointer_type ty in
  
  (* RUN: grep {declare i32 @Fn1\(i32, i64\)} < %t.ll
   *)
  group "declare";
  let fn = declare_function "Fn1" ty m in
  insist (pty = type_of fn);
  insist (is_declaration fn);
  insist (0 = Array.length (basic_blocks fn));
  
  (* RUN: grep -v {Fn2} < %t.ll
   *)
  group "delete";
  let fn = declare_function "Fn2" ty m in
  delete_function fn;
  
  (* RUN: grep {define.*Fn3} < %t.ll
   *)
  group "define";
  let fn = define_function "Fn3" ty m in
  insist (not (is_declaration fn));
  insist (1 = Array.length (basic_blocks fn));
  (* this function is not valid because init bb lacks a terminator *)
  
  (* RUN: grep {define.*Fn4.*Param1.*Param2} < %t.ll
   *)
  group "params";
  let fn = define_function "Fn4" ty m in
  let params = params fn in
  insist (2 = Array.length params);
  insist (params.(0) = param fn 0);
  insist (params.(1) = param fn 1);
  insist (i32_type = type_of params.(0));
  insist (i64_type = type_of params.(1));
  set_value_name "Param1" params.(0);
  set_value_name "Param2" params.(1);
  (* this function is not valid because init bb lacks a terminator *)
  
  (* RUN: grep {fastcc.*Fn5} < %t.ll
   *)
  group "callconv";
  let fn = define_function "Fn5" ty m in
  insist (ccc = function_call_conv fn);
  set_function_call_conv fastcc fn;
  insist (fastcc = function_call_conv fn)


(*===-- Basic Blocks ------------------------------------------------------===*)

let test_basic_blocks () =
  let ty = make_function_type void_type [| |] false in
  
  (* RUN: grep {Bb1} < %t.ll
   *)
  group "entry";
  let fn = declare_function "X" ty m in
  let bb = append_block "Bb1" fn in
  insist (bb = entry_block fn);
  
  (* RUN: grep -v Bb2 < %t.ll
   *)
  group "delete";
  let fn = declare_function "X2" ty m in
  let bb = append_block "Bb2" fn in
  delete_block bb;
  
  group "insert";
  let fn = declare_function "X3" ty m in
  let bbb = append_block "" fn in
  let bba = insert_block "" bbb in
  insist ([| bba; bbb |] = basic_blocks fn);
  
  (* RUN: grep Bb3 < %t.ll
   *)
  group "name/value";
  let fn = define_function "X4" ty m in
  let bb = entry_block fn in
  let bbv = value_of_block bb in
  set_value_name "Bb3" bbv;
  insist ("Bb3" = value_name bbv);
  
  group "casts";
  let fn = define_function "X5" ty m in
  let bb = entry_block fn in
  insist (bb = block_of_value (value_of_block bb));
  insist (value_is_block (value_of_block bb));
  insist (not (value_is_block (make_null i32_type)))


(*===-- Builder -----------------------------------------------------------===*)

let test_builder () =
  let (++) x f = f x; x in
  
  group "ret void";
  begin
    (* RUN: grep {ret void} < %t.ll
     *)
    let fty = make_function_type void_type [| |] false in
    let fn = declare_function "X6" fty m in
    let b = builder_at_end (append_block "Bb01" fn) in
    ignore (build_ret_void b)
  end;
  
  (* The rest of the tests will use one big function. *)
  let fty = make_function_type i32_type [| i32_type; i32_type |] false in
  let fn = define_function "X7" fty m in
  let atentry = builder_at_end (entry_block fn) in
  let p1 = param fn 0 ++ set_value_name "P1" in
  let p2 = param fn 1 ++ set_value_name "P2" in
  let f1 = build_uitofp p1 float_type "F1" atentry in
  let f2 = build_uitofp p2 float_type "F2" atentry in
  
  let bb00 = append_block "Bb00" fn in
  ignore (build_unreachable (builder_at_end bb00));
  
  group "ret"; begin
    (* RUN: grep {ret.*P1} < %t.ll
     *)
    let ret = build_ret p1 atentry in
    position_before ret atentry
  end;
  
  group "br"; begin
    (* RUN: grep {br.*Bb02} < %t.ll
     *)
    let bb02 = append_block "Bb02" fn in
    let b = builder_at_end bb02 in
    ignore (build_br bb02 b)
  end;
  
  group "cond_br"; begin
    (* RUN: grep {br.*Inst01.*Bb03.*Bb00} < %t.ll
     *)
    let bb03 = append_block "Bb03" fn in
    let b = builder_at_end bb03 in
    let cond = build_trunc p1 i1_type "Inst01" b in
    ignore (build_cond_br cond bb03 bb00 b)
  end;
  
  (* TODO: Switch *)
  
  group "invoke"; begin
    (* RUN: grep {Inst02.*invoke.*P1.*P2} < %t.ll
     * RUN: grep {to.*Bb04.*unwind.*Bb00} < %t.ll
     *)
    let bb04 = append_block "Bb04" fn in
    let b = builder_at_end bb04 in
    ignore (build_invoke fn [| p1; p2 |] bb04 bb00 "Inst02" b)
  end;
  
  group "unwind"; begin
    (* RUN: grep {unwind} < %t.ll
     *)
    let bb05 = append_block "Bb05" fn in
    let b = builder_at_end bb05 in
    ignore (build_unwind b)
  end;
  
  group "unreachable"; begin
    (* RUN: grep {unreachable} < %t.ll
     *)
    let bb06 = append_block "Bb06" fn in
    let b = builder_at_end bb06 in
    ignore (build_unreachable b)
  end;
  
  group "arithmetic"; begin
    let bb07 = append_block "Bb07" fn in
    let b = builder_at_end bb07 in
    
    (* RUN: grep {Inst03.*add.*P1.*P2} < %t.ll
     * RUN: grep {Inst04.*sub.*P1.*Inst03} < %t.ll
     * RUN: grep {Inst05.*mul.*P1.*Inst04} < %t.ll
     * RUN: grep {Inst06.*udiv.*P1.*Inst05} < %t.ll
     * RUN: grep {Inst07.*sdiv.*P1.*Inst06} < %t.ll
     * RUN: grep {Inst08.*fdiv.*F1.*F2} < %t.ll
     * RUN: grep {Inst09.*urem.*P1.*Inst07} < %t.ll
     * RUN: grep {Inst10.*srem.*P1.*Inst09} < %t.ll
     * RUN: grep {Inst11.*frem.*F1.*Inst08} < %t.ll
     * RUN: grep {Inst12.*shl.*P1.*Inst10} < %t.ll
     * RUN: grep {Inst13.*lshr.*P1.*Inst12} < %t.ll
     * RUN: grep {Inst14.*ashr.*P1.*Inst13} < %t.ll
     * RUN: grep {Inst15.*and.*P1.*Inst14} < %t.ll
     * RUN: grep {Inst16.*or.*P1.*Inst15} < %t.ll
     * RUN: grep {Inst17.*xor.*P1.*Inst16} < %t.ll
     * RUN: grep {Inst18.*sub.*0.*Inst17} < %t.ll
     * RUN: grep {Inst19.*xor.*Inst18.*-1} < %t.ll
     *)
    let inst03 = build_add  p1 p2     "Inst03" b in
    let inst04 = build_sub  p1 inst03 "Inst04" b in
    let inst05 = build_mul  p1 inst04 "Inst05" b in
    let inst06 = build_udiv p1 inst05 "Inst06" b in
    let inst07 = build_sdiv p1 inst06 "Inst07" b in
    let inst08 = build_fdiv f1 f2     "Inst08" b in
    let inst09 = build_urem p1 inst07 "Inst09" b in
    let inst10 = build_srem p1 inst09 "Inst10" b in
          ignore(build_frem f1 inst08 "Inst11" b);
    let inst12 = build_shl  p1 inst10 "Inst12" b in
    let inst13 = build_lshr p1 inst12 "Inst13" b in
    let inst14 = build_ashr p1 inst13 "Inst14" b in
    let inst15 = build_and  p1 inst14 "Inst15" b in
    let inst16 = build_or   p1 inst15 "Inst16" b in
    let inst17 = build_xor  p1 inst16 "Inst17" b in
    let inst18 = build_neg  inst17    "Inst18" b in
         ignore (build_not  inst18    "Inst19" b)
  end;
  
  group "memory"; begin
    let bb08 = append_block "Bb08" fn in
    let b = builder_at_end bb08 in
    
    (* RUN: grep {Inst20.*malloc.*i8	} < %t.ll
     * RUN: grep {Inst21.*malloc.*i8.*P1} < %t.ll
     * RUN: grep {Inst22.*alloca.*i32	} < %t.ll
     * RUN: grep {Inst23.*alloca.*i32.*P2} < %t.ll
     * RUN: grep {free.*Inst20} < %t.ll
     * RUN: grep {Inst25.*load.*Inst21} < %t.ll
     * RUN: grep {store.*P2.*Inst22} < %t.ll
     * RUN: grep {Inst27.*getelementptr.*Inst23.*P2} < %t.ll
     *)
    let inst20 = build_malloc i8_type "Inst20" b in
    let inst21 = build_array_malloc i8_type p1 "Inst21" b in
    let inst22 = build_alloca i32_type "Inst22" b in
    let inst23 = build_array_alloca i32_type p2 "Inst23" b in
          ignore(build_free inst20 b);
          ignore(build_load inst21 "Inst25" b);
          ignore(build_store p2 inst22 b);
          ignore(build_gep inst23 [| p2 |] "Inst27" b)
  end;
  
  group "casts"; begin
    let void_ptr = make_pointer_type i8_type in
    
    (* RUN: grep {Inst28.*trunc.*P1.*i8} < %t.ll
     * RUN: grep {Inst29.*zext.*Inst28.*i32} < %t.ll
     * RUN: grep {Inst30.*sext.*Inst29.*i64} < %t.ll
     * RUN: grep {Inst31.*uitofp.*Inst30.*float} < %t.ll
     * RUN: grep {Inst32.*sitofp.*Inst29.*double} < %t.ll
     * RUN: grep {Inst33.*fptoui.*Inst31.*i32} < %t.ll
     * RUN: grep {Inst34.*fptosi.*Inst32.*i64} < %t.ll
     * RUN: grep {Inst35.*fptrunc.*Inst32.*float} < %t.ll
     * RUN: grep {Inst36.*fpext.*Inst35.*double} < %t.ll
     * RUN: grep {Inst37.*inttoptr.*P1.*i8\*} < %t.ll
     * RUN: grep {Inst38.*ptrtoint.*Inst37.*i64} < %t.ll
     * RUN: grep {Inst39.*bitcast.*Inst38.*double} < %t.ll
     *)
    let inst28 = build_trunc p1 i8_type "Inst28" atentry in
    let inst29 = build_zext inst28 i32_type "Inst29" atentry in
    let inst30 = build_sext inst29 i64_type "Inst30" atentry in
    let inst31 = build_uitofp inst30 float_type "Inst31" atentry in
    let inst32 = build_sitofp inst29 double_type "Inst32" atentry in
          ignore(build_fptoui inst31 i32_type "Inst33" atentry);
          ignore(build_fptosi inst32 i64_type "Inst34" atentry);
    let inst35 = build_fptrunc inst32 float_type "Inst35" atentry in
          ignore(build_fpext inst35 double_type "Inst36" atentry);
    let inst37 = build_inttoptr p1 void_ptr "Inst37" atentry in
    let inst38 = build_ptrtoint inst37 i64_type "Inst38" atentry in
          ignore(build_bitcast inst38 double_type "Inst39" atentry)
  end;
  
  group "comparisons"; begin
    (* RUN: grep {Inst40.*icmp.*ne.*P1.*P2} < %t.ll
     * RUN: grep {Inst41.*icmp.*sle.*P2.*P1} < %t.ll
     * RUN: grep {Inst42.*fcmp.*false.*F1.*F2} < %t.ll
     * RUN: grep {Inst43.*fcmp.*true.*F2.*F1} < %t.ll
     *)
    ignore (build_icmp Icmp_ne    p1 p2 "Inst40" atentry);
    ignore (build_icmp Icmp_sle   p2 p1 "Inst41" atentry);
    ignore (build_fcmp Fcmp_false f1 f2 "Inst42" atentry);
    ignore (build_fcmp Fcmp_true  f2 f1 "Inst43" atentry)
  end;
  
  group "miscellaneous"; begin
    (* RUN: grep {Inst45.*call.*P2.*P1} < %t.ll
     * RUN: grep {Inst47.*select.*Inst46.*P1.*P2} < %t.ll
     * RUN: grep {Inst48.*va_arg.*null.*i32} < %t.ll
     * RUN: grep {Inst49.*extractelement.*Vec1.*P2} < %t.ll
     * RUN: grep {Inst50.*insertelement.*Vec1.*P1.*P2} < %t.ll
     * RUN: grep {Inst51.*shufflevector.*Vec1.*Vec2.*Vec3} < %t.ll
     *)
    
    (* TODO: %Inst44 = Phi *)
    
         ignore (build_call fn [| p2; p1 |] "Inst45" atentry);
    let inst46 = build_icmp Icmp_eq p1 p2 "Inst46" atentry in
         ignore (build_select inst46 p1 p2 "Inst47" atentry);
         ignore (build_va_arg
                  (make_null (make_pointer_type (make_pointer_type i8_type)))
                  i32_type "Inst48" atentry);
    
    (* Set up some vector vregs. *)
    let one = make_int_constant i32_type (-1) true in
    let zero = make_int_constant i32_type 1 true in
    let t1 = make_vector_constant [| one; zero; one; zero |] in
    let t2 = make_vector_constant [| zero; one; zero; one |] in
    let t3 = make_vector_constant [| one; one; zero; zero |] in
    let vec1 = build_insertelement t1 p1 p2 "Vec1" atentry in
    let vec2 = build_insertelement t2 p1 p2 "Vec2" atentry in
    let vec3 = build_insertelement t3 p1 p2 "Vec3" atentry in
    
    ignore (build_extractelement vec1 p2 "Inst49" atentry);
    ignore (build_insertelement vec1 p1 p2 "Inst50" atentry);
    ignore (build_shufflevector vec1 vec2 vec3 "Inst51" atentry);
  end


(*===-- Writer ------------------------------------------------------------===*)

let test_writer () =
  group "writer";
  insist (write_bitcode_file m filename);
  
  dispose_module m


(*===-- Driver ------------------------------------------------------------===*)

let _ =
  suite "types"            test_types;
  suite "constants"        test_constants;
  suite "global values"    test_global_values;
  suite "global variables" test_global_variables;
  suite "functions"        test_functions;
  suite "basic blocks"     test_basic_blocks;
  suite "builder"          test_builder;
  suite "writer"           test_writer;
  exit !exit_status

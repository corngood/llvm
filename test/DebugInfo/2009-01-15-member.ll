; RUN: llc %s -o /dev/null
	%llvm.dbg.anchor.type = type { i32, i32 }
	%llvm.dbg.basictype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, i32, i8*, i8* }
	%llvm.dbg.compile_unit.type = type { i32, { }*, i32, i8*, i8*, i8* }
	%llvm.dbg.composite.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, { }*, i8*, i8* }
	%llvm.dbg.derivedtype.type = type { i32, { }*, i8*, { }*, i32, i64, i64, i64, i32, { }*, i8*, i8* }
	%llvm.dbg.global_variable.type = type { i32, { }*, { }*, i8*, i8*, i8*, { }*, i32, { }*, i1, i1, { }*, i8*, i8* }
	%struct.f = type opaque
	%struct.s = type { %struct.f*, i32 }
@s2 = common global %struct.s zeroinitializer		; <%struct.s*> [#uses=1]
@llvm.dbg.compile_units = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 17 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str = internal constant [4 x i8] c"t.c\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [6 x i8] c"/tmp/\00", section "llvm.metadata"		; <[6 x i8]*> [#uses=1]
@.str2 = internal constant [57 x i8] c"4.2.1 (Based on Apple Inc. build 5628) (LLVM build 2091)\00", section "llvm.metadata"		; <[57 x i8]*> [#uses=1]
@llvm.dbg.compile_unit = internal constant %llvm.dbg.compile_unit.type { i32 458769, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.compile_units to { }*), i32 1, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0), i8* getelementptr ([57 x i8]* @.str2, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.compile_unit.type*> [#uses=1]
@.str3 = internal constant [2 x i8] c"s\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@.str4 = internal constant [2 x i8] c"f\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.composite5 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str4, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 2, i64 0, i64 0, i64 0, i32 60, { }* null, { }* null, i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.derivedtype = internal constant %llvm.dbg.derivedtype.type { i32 458767, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* null, { }* null, i32 0, i64 32, i64 32, i64 0, i32 0, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite5 to { }*), i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str6 = internal constant [3 x i8] c"f1\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.derivedtype7 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([3 x i8]* @.str6, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 4, i64 32, i64 32, i64 0, i32 1, { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@.str8 = internal constant [4 x i8] c"int\00", section "llvm.metadata"		; <[4 x i8]*> [#uses=1]
@llvm.dbg.basictype = internal constant %llvm.dbg.basictype.type { i32 458788, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([4 x i8]* @.str8, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 0, i64 32, i64 32, i64 0, i32 0, i32 5, i8* null, i8* null }, section "llvm.metadata"		; <%llvm.dbg.basictype.type*> [#uses=1]
@.str9 = internal constant [2 x i8] c"a\00", section "llvm.metadata"		; <[2 x i8]*> [#uses=1]
@llvm.dbg.derivedtype10 = internal constant %llvm.dbg.derivedtype.type { i32 458765, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str9, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 5, i64 32, i64 32, i64 32, i32 1, { }* bitcast (%llvm.dbg.basictype.type* @llvm.dbg.basictype to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.derivedtype.type*> [#uses=1]
@llvm.dbg.array = internal constant [2 x { }*] [ { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype7 to { }*), { }* bitcast (%llvm.dbg.derivedtype.type* @llvm.dbg.derivedtype10 to { }*) ], section "llvm.metadata"		; <[2 x { }*]*> [#uses=1]
@llvm.dbg.composite11 = internal constant %llvm.dbg.composite.type { i32 458771, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([2 x i8]* @.str3, i32 0, i32 0), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 3, i64 64, i64 32, i64 0, i32 0, { }* null, { }* bitcast ([2 x { }*]* @llvm.dbg.array to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.composite.type*> [#uses=1]
@llvm.dbg.global_variables = linkonce constant %llvm.dbg.anchor.type { i32 458752, i32 52 }, section "llvm.metadata"		; <%llvm.dbg.anchor.type*> [#uses=1]
@.str12 = internal constant [3 x i8] c"s2\00", section "llvm.metadata"		; <[3 x i8]*> [#uses=1]
@llvm.dbg.global_variable = internal constant %llvm.dbg.global_variable.type { i32 458804, { }* bitcast (%llvm.dbg.anchor.type* @llvm.dbg.global_variables to { }*), { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i8* getelementptr ([3 x i8]* @.str12, i32 0, i32 0), i8* getelementptr ([3 x i8]* @.str12, i32 0, i32 0), i8* null, { }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit to { }*), i32 6, { }* bitcast (%llvm.dbg.composite.type* @llvm.dbg.composite11 to { }*), i1 false, i1 true, { }* bitcast (%struct.s* @s2 to { }*), i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i8* getelementptr ([6 x i8]* @.str1, i32 0, i32 0) }, section "llvm.metadata"		; <%llvm.dbg.global_variable.type*> [#uses=0]

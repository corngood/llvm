; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin -stats |& grep virtregrewriter | not grep {stores unfolded}
; XFAIL: *
; rdar://6682365

; Do not clobber a register if another spill slot is available in it and it's marked "do not clobber".

	%struct.CAST_KEY = type { [32 x i32], i32 }
@CAST_S_table0 = constant [2 x i32] [i32 821772500, i32 -1616838901], align 32		; <[2 x i32]*> [#uses=0]
@CAST_S_table4 = constant [2 x i32] [i32 2127105028, i32 745436345], align 32		; <[2 x i32]*> [#uses=6]
@CAST_S_table5 = constant [2 x i32] [i32 -151351395, i32 749497569], align 32		; <[2 x i32]*> [#uses=5]
@CAST_S_table6 = constant [2 x i32] [i32 -2048901095, i32 858518887], align 32		; <[2 x i32]*> [#uses=4]
@CAST_S_table7 = constant [2 x i32] [i32 -501862387, i32 -1143078916], align 32		; <[2 x i32]*> [#uses=5]
@CAST_S_table1 = constant [2 x i32] [i32 522195092, i32 -284448933], align 32		; <[2 x i32]*> [#uses=0]
@CAST_S_table2 = constant [2 x i32] [i32 -1913667008, i32 637164959], align 32		; <[2 x i32]*> [#uses=0]
@CAST_S_table3 = constant [2 x i32] [i32 -1649212384, i32 532081118], align 32		; <[2 x i32]*> [#uses=0]

define void @CAST_set_key(%struct.CAST_KEY* nocapture %key, i32 %len, i8* nocapture %data) nounwind ssp {
bb1.thread:
	%0 = getelementptr [16 x i32]* null, i32 0, i32 5		; <i32*> [#uses=1]
	%1 = getelementptr [16 x i32]* null, i32 0, i32 8		; <i32*> [#uses=1]
	%2 = load i32* null, align 4		; <i32> [#uses=1]
	%3 = shl i32 %2, 24		; <i32> [#uses=1]
	%4 = load i32* null, align 4		; <i32> [#uses=1]
	%5 = shl i32 %4, 16		; <i32> [#uses=1]
	%6 = load i32* null, align 4		; <i32> [#uses=1]
	%7 = or i32 %5, %3		; <i32> [#uses=1]
	%8 = or i32 %7, %6		; <i32> [#uses=1]
	%9 = or i32 %8, 0		; <i32> [#uses=1]
	%10 = load i32* null, align 4		; <i32> [#uses=1]
	%11 = shl i32 %10, 24		; <i32> [#uses=1]
	%12 = load i32* %0, align 4		; <i32> [#uses=1]
	%13 = shl i32 %12, 16		; <i32> [#uses=1]
	%14 = load i32* null, align 4		; <i32> [#uses=1]
	%15 = or i32 %13, %11		; <i32> [#uses=1]
	%16 = or i32 %15, %14		; <i32> [#uses=1]
	%17 = or i32 %16, 0		; <i32> [#uses=1]
	br label %bb11

bb11:		; preds = %bb11, %bb1.thread
	%18 = phi i32 [ %110, %bb11 ], [ 0, %bb1.thread ]		; <i32> [#uses=1]
	%19 = phi i32 [ %112, %bb11 ], [ 0, %bb1.thread ]		; <i32> [#uses=0]
	%20 = phi i32 [ 0, %bb11 ], [ 0, %bb1.thread ]		; <i32> [#uses=0]
	%21 = phi i32 [ %113, %bb11 ], [ 0, %bb1.thread ]		; <i32> [#uses=1]
	%X.0.0 = phi i32 [ %9, %bb1.thread ], [ %92, %bb11 ]		; <i32> [#uses=0]
	%X.1.0 = phi i32 [ %17, %bb1.thread ], [ 0, %bb11 ]		; <i32> [#uses=0]
	%22 = getelementptr [2 x i32]* @CAST_S_table6, i32 0, i32 %21		; <i32*> [#uses=0]
	%23 = getelementptr [2 x i32]* @CAST_S_table5, i32 0, i32 %18		; <i32*> [#uses=0]
	%24 = load i32* null, align 4		; <i32> [#uses=1]
	%25 = xor i32 0, %24		; <i32> [#uses=1]
	%26 = xor i32 %25, 0		; <i32> [#uses=1]
	%27 = xor i32 %26, 0		; <i32> [#uses=4]
	%28 = and i32 %27, 255		; <i32> [#uses=2]
	%29 = lshr i32 %27, 8		; <i32> [#uses=1]
	%30 = and i32 %29, 255		; <i32> [#uses=2]
	%31 = lshr i32 %27, 16		; <i32> [#uses=1]
	%32 = and i32 %31, 255		; <i32> [#uses=1]
	%33 = getelementptr [2 x i32]* @CAST_S_table4, i32 0, i32 %28		; <i32*> [#uses=1]
	%34 = load i32* %33, align 4		; <i32> [#uses=2]
	%35 = getelementptr [2 x i32]* @CAST_S_table5, i32 0, i32 %30		; <i32*> [#uses=1]
	%36 = load i32* %35, align 4		; <i32> [#uses=2]
	%37 = xor i32 %34, 0		; <i32> [#uses=1]
	%38 = xor i32 %37, %36		; <i32> [#uses=1]
	%39 = xor i32 %38, 0		; <i32> [#uses=1]
	%40 = xor i32 %39, 0		; <i32> [#uses=1]
	%41 = xor i32 %40, 0		; <i32> [#uses=3]
	%42 = lshr i32 %41, 8		; <i32> [#uses=1]
	%43 = and i32 %42, 255		; <i32> [#uses=2]
	%44 = lshr i32 %41, 16		; <i32> [#uses=1]
	%45 = and i32 %44, 255		; <i32> [#uses=1]
	%46 = getelementptr [2 x i32]* @CAST_S_table4, i32 0, i32 %43		; <i32*> [#uses=1]
	%47 = load i32* %46, align 4		; <i32> [#uses=1]
	%48 = load i32* null, align 4		; <i32> [#uses=1]
	%49 = xor i32 %47, 0		; <i32> [#uses=1]
	%50 = xor i32 %49, %48		; <i32> [#uses=1]
	%51 = xor i32 %50, 0		; <i32> [#uses=1]
	%52 = xor i32 %51, 0		; <i32> [#uses=1]
	%53 = xor i32 %52, 0		; <i32> [#uses=2]
	%54 = and i32 %53, 255		; <i32> [#uses=1]
	%55 = lshr i32 %53, 24		; <i32> [#uses=1]
	%56 = getelementptr [2 x i32]* @CAST_S_table6, i32 0, i32 %55		; <i32*> [#uses=1]
	%57 = load i32* %56, align 4		; <i32> [#uses=1]
	%58 = xor i32 0, %57		; <i32> [#uses=1]
	%59 = xor i32 %58, 0		; <i32> [#uses=1]
	%60 = xor i32 %59, 0		; <i32> [#uses=1]
	store i32 %60, i32* null, align 4
	%61 = getelementptr [2 x i32]* @CAST_S_table4, i32 0, i32 0		; <i32*> [#uses=1]
	%62 = load i32* %61, align 4		; <i32> [#uses=1]
	%63 = getelementptr [2 x i32]* @CAST_S_table7, i32 0, i32 %54		; <i32*> [#uses=1]
	%64 = load i32* %63, align 4		; <i32> [#uses=1]
	%65 = xor i32 0, %64		; <i32> [#uses=1]
	%66 = xor i32 %65, 0		; <i32> [#uses=1]
	store i32 %66, i32* null, align 4
	%67 = getelementptr [2 x i32]* @CAST_S_table7, i32 0, i32 %45		; <i32*> [#uses=1]
	%68 = load i32* %67, align 4		; <i32> [#uses=1]
	%69 = xor i32 %36, %34		; <i32> [#uses=1]
	%70 = xor i32 %69, 0		; <i32> [#uses=1]
	%71 = xor i32 %70, %68		; <i32> [#uses=1]
	%72 = xor i32 %71, 0		; <i32> [#uses=1]
	store i32 %72, i32* null, align 4
	%73 = getelementptr [2 x i32]* @CAST_S_table4, i32 0, i32 %32		; <i32*> [#uses=1]
	%74 = load i32* %73, align 4		; <i32> [#uses=2]
	%75 = load i32* null, align 4		; <i32> [#uses=1]
	%76 = getelementptr [2 x i32]* @CAST_S_table6, i32 0, i32 %43		; <i32*> [#uses=1]
	%77 = load i32* %76, align 4		; <i32> [#uses=1]
	%78 = getelementptr [2 x i32]* @CAST_S_table7, i32 0, i32 0		; <i32*> [#uses=1]
	%79 = load i32* %78, align 4		; <i32> [#uses=1]
	%80 = getelementptr [2 x i32]* @CAST_S_table7, i32 0, i32 %30		; <i32*> [#uses=1]
	%81 = load i32* %80, align 4		; <i32> [#uses=2]
	%82 = xor i32 %75, %74		; <i32> [#uses=1]
	%83 = xor i32 %82, %77		; <i32> [#uses=1]
	%84 = xor i32 %83, %79		; <i32> [#uses=1]
	%85 = xor i32 %84, %81		; <i32> [#uses=1]
	store i32 %85, i32* null, align 4
	%86 = getelementptr [2 x i32]* @CAST_S_table5, i32 0, i32 %28		; <i32*> [#uses=1]
	%87 = load i32* %86, align 4		; <i32> [#uses=1]
	%88 = xor i32 %74, %41		; <i32> [#uses=1]
	%89 = xor i32 %88, %87		; <i32> [#uses=1]
	%90 = xor i32 %89, 0		; <i32> [#uses=1]
	%91 = xor i32 %90, %81		; <i32> [#uses=1]
	%92 = xor i32 %91, 0		; <i32> [#uses=3]
	%93 = lshr i32 %92, 16		; <i32> [#uses=1]
	%94 = and i32 %93, 255		; <i32> [#uses=1]
	store i32 %94, i32* null, align 4
	%95 = lshr i32 %92, 24		; <i32> [#uses=2]
	%96 = getelementptr [2 x i32]* @CAST_S_table4, i32 0, i32 %95		; <i32*> [#uses=1]
	%97 = load i32* %96, align 4		; <i32> [#uses=1]
	%98 = getelementptr [2 x i32]* @CAST_S_table5, i32 0, i32 0		; <i32*> [#uses=1]
	%99 = load i32* %98, align 4		; <i32> [#uses=1]
	%100 = load i32* null, align 4		; <i32> [#uses=0]
	%101 = xor i32 %97, 0		; <i32> [#uses=1]
	%102 = xor i32 %101, %99		; <i32> [#uses=1]
	%103 = xor i32 %102, 0		; <i32> [#uses=1]
	%104 = xor i32 %103, 0		; <i32> [#uses=0]
	store i32 0, i32* null, align 4
	%105 = xor i32 0, %27		; <i32> [#uses=1]
	%106 = xor i32 %105, 0		; <i32> [#uses=1]
	%107 = xor i32 %106, 0		; <i32> [#uses=1]
	%108 = xor i32 %107, 0		; <i32> [#uses=1]
	%109 = xor i32 %108, %62		; <i32> [#uses=3]
	%110 = and i32 %109, 255		; <i32> [#uses=1]
	%111 = lshr i32 %109, 16		; <i32> [#uses=1]
	%112 = and i32 %111, 255		; <i32> [#uses=1]
	%113 = lshr i32 %109, 24		; <i32> [#uses=3]
	store i32 %113, i32* %1, align 4
	%114 = load i32* null, align 4		; <i32> [#uses=1]
	%115 = xor i32 0, %114		; <i32> [#uses=1]
	%116 = xor i32 %115, 0		; <i32> [#uses=1]
	%117 = xor i32 %116, 0		; <i32> [#uses=1]
	%K.0.sum42 = or i32 0, 12		; <i32> [#uses=1]
	%118 = getelementptr [32 x i32]* null, i32 0, i32 %K.0.sum42		; <i32*> [#uses=1]
	store i32 %117, i32* %118, align 4
	%119 = getelementptr [2 x i32]* @CAST_S_table5, i32 0, i32 0		; <i32*> [#uses=0]
	store i32 0, i32* null, align 4
	%120 = getelementptr [2 x i32]* @CAST_S_table6, i32 0, i32 %113		; <i32*> [#uses=1]
	%121 = load i32* %120, align 4		; <i32> [#uses=1]
	%122 = xor i32 0, %121		; <i32> [#uses=1]
	store i32 %122, i32* null, align 4
	%123 = getelementptr [2 x i32]* @CAST_S_table4, i32 0, i32 0		; <i32*> [#uses=1]
	%124 = load i32* %123, align 4		; <i32> [#uses=1]
	%125 = getelementptr [2 x i32]* @CAST_S_table7, i32 0, i32 %95		; <i32*> [#uses=1]
	%126 = load i32* %125, align 4		; <i32> [#uses=1]
	%127 = xor i32 0, %124		; <i32> [#uses=1]
	%128 = xor i32 %127, 0		; <i32> [#uses=1]
	%129 = xor i32 %128, %126		; <i32> [#uses=1]
	%130 = xor i32 %129, 0		; <i32> [#uses=1]
	store i32 %130, i32* null, align 4
	br label %bb11
}

; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vsldoi | wc -l | grep 2
; RUN: llvm-as < %s | opt -instcombine | llc -march=ppc32 -mcpu=g5 | not grep vperm

void %VSLDOI_xy(<8 x short>* %A, <8 x short>* %B) {
entry:
	%tmp = load <8 x short>* %A		; <<8 x short>> [#uses=1]
	%tmp2 = load <8 x short>* %B		; <<8 x short>> [#uses=1]
	%tmp = cast <8 x short> %tmp to <16 x sbyte>		; <<16 x sbyte>> [#uses=11]
	%tmp2 = cast <8 x short> %tmp2 to <16 x sbyte>		; <<16 x sbyte>> [#uses=5]
	%tmp = extractelement <16 x sbyte> %tmp, uint 5		; <sbyte> [#uses=1]
	%tmp3 = extractelement <16 x sbyte> %tmp, uint 6		; <sbyte> [#uses=1]
	%tmp4 = extractelement <16 x sbyte> %tmp, uint 7		; <sbyte> [#uses=1]
	%tmp5 = extractelement <16 x sbyte> %tmp, uint 8		; <sbyte> [#uses=1]
	%tmp6 = extractelement <16 x sbyte> %tmp, uint 9		; <sbyte> [#uses=1]
	%tmp7 = extractelement <16 x sbyte> %tmp, uint 10		; <sbyte> [#uses=1]
	%tmp8 = extractelement <16 x sbyte> %tmp, uint 11		; <sbyte> [#uses=1]
	%tmp9 = extractelement <16 x sbyte> %tmp, uint 12		; <sbyte> [#uses=1]
	%tmp10 = extractelement <16 x sbyte> %tmp, uint 13		; <sbyte> [#uses=1]
	%tmp11 = extractelement <16 x sbyte> %tmp, uint 14		; <sbyte> [#uses=1]
	%tmp12 = extractelement <16 x sbyte> %tmp, uint 15		; <sbyte> [#uses=1]
	%tmp13 = extractelement <16 x sbyte> %tmp2, uint 0		; <sbyte> [#uses=1]
	%tmp14 = extractelement <16 x sbyte> %tmp2, uint 1		; <sbyte> [#uses=1]
	%tmp15 = extractelement <16 x sbyte> %tmp2, uint 2		; <sbyte> [#uses=1]
	%tmp16 = extractelement <16 x sbyte> %tmp2, uint 3		; <sbyte> [#uses=1]
	%tmp17 = extractelement <16 x sbyte> %tmp2, uint 4		; <sbyte> [#uses=1]
	%tmp18 = insertelement <16 x sbyte> undef, sbyte %tmp, uint 0		; <<16 x sbyte>> [#uses=1]
	%tmp19 = insertelement <16 x sbyte> %tmp18, sbyte %tmp3, uint 1		; <<16 x sbyte>> [#uses=1]
	%tmp20 = insertelement <16 x sbyte> %tmp19, sbyte %tmp4, uint 2		; <<16 x sbyte>> [#uses=1]
	%tmp21 = insertelement <16 x sbyte> %tmp20, sbyte %tmp5, uint 3		; <<16 x sbyte>> [#uses=1]
	%tmp22 = insertelement <16 x sbyte> %tmp21, sbyte %tmp6, uint 4		; <<16 x sbyte>> [#uses=1]
	%tmp23 = insertelement <16 x sbyte> %tmp22, sbyte %tmp7, uint 5		; <<16 x sbyte>> [#uses=1]
	%tmp24 = insertelement <16 x sbyte> %tmp23, sbyte %tmp8, uint 6		; <<16 x sbyte>> [#uses=1]
	%tmp25 = insertelement <16 x sbyte> %tmp24, sbyte %tmp9, uint 7		; <<16 x sbyte>> [#uses=1]
	%tmp26 = insertelement <16 x sbyte> %tmp25, sbyte %tmp10, uint 8		; <<16 x sbyte>> [#uses=1]
	%tmp27 = insertelement <16 x sbyte> %tmp26, sbyte %tmp11, uint 9		; <<16 x sbyte>> [#uses=1]
	%tmp28 = insertelement <16 x sbyte> %tmp27, sbyte %tmp12, uint 10		; <<16 x sbyte>> [#uses=1]
	%tmp29 = insertelement <16 x sbyte> %tmp28, sbyte %tmp13, uint 11		; <<16 x sbyte>> [#uses=1]
	%tmp30 = insertelement <16 x sbyte> %tmp29, sbyte %tmp14, uint 12		; <<16 x sbyte>> [#uses=1]
	%tmp31 = insertelement <16 x sbyte> %tmp30, sbyte %tmp15, uint 13		; <<16 x sbyte>> [#uses=1]
	%tmp32 = insertelement <16 x sbyte> %tmp31, sbyte %tmp16, uint 14		; <<16 x sbyte>> [#uses=1]
	%tmp33 = insertelement <16 x sbyte> %tmp32, sbyte %tmp17, uint 15		; <<16 x sbyte>> [#uses=1]
	%tmp33 = cast <16 x sbyte> %tmp33 to <8 x short>		; <<8 x short>> [#uses=1]
	store <8 x short> %tmp33, <8 x short>* %A
	ret void
}

void %VSLDOI_xx(<8 x short>* %A, <8 x short>* %B) {
	%tmp = load <8 x short>* %A		; <<8 x short>> [#uses=1]
	%tmp2 = load <8 x short>* %A		; <<8 x short>> [#uses=1]
	%tmp = cast <8 x short> %tmp to <16 x sbyte>		; <<16 x sbyte>> [#uses=11]
	%tmp2 = cast <8 x short> %tmp2 to <16 x sbyte>		; <<16 x sbyte>> [#uses=5]
	%tmp = extractelement <16 x sbyte> %tmp, uint 5		; <sbyte> [#uses=1]
	%tmp3 = extractelement <16 x sbyte> %tmp, uint 6		; <sbyte> [#uses=1]
	%tmp4 = extractelement <16 x sbyte> %tmp, uint 7		; <sbyte> [#uses=1]
	%tmp5 = extractelement <16 x sbyte> %tmp, uint 8		; <sbyte> [#uses=1]
	%tmp6 = extractelement <16 x sbyte> %tmp, uint 9		; <sbyte> [#uses=1]
	%tmp7 = extractelement <16 x sbyte> %tmp, uint 10		; <sbyte> [#uses=1]
	%tmp8 = extractelement <16 x sbyte> %tmp, uint 11		; <sbyte> [#uses=1]
	%tmp9 = extractelement <16 x sbyte> %tmp, uint 12		; <sbyte> [#uses=1]
	%tmp10 = extractelement <16 x sbyte> %tmp, uint 13		; <sbyte> [#uses=1]
	%tmp11 = extractelement <16 x sbyte> %tmp, uint 14		; <sbyte> [#uses=1]
	%tmp12 = extractelement <16 x sbyte> %tmp, uint 15		; <sbyte> [#uses=1]
	%tmp13 = extractelement <16 x sbyte> %tmp2, uint 0		; <sbyte> [#uses=1]
	%tmp14 = extractelement <16 x sbyte> %tmp2, uint 1		; <sbyte> [#uses=1]
	%tmp15 = extractelement <16 x sbyte> %tmp2, uint 2		; <sbyte> [#uses=1]
	%tmp16 = extractelement <16 x sbyte> %tmp2, uint 3		; <sbyte> [#uses=1]
	%tmp17 = extractelement <16 x sbyte> %tmp2, uint 4		; <sbyte> [#uses=1]
	%tmp18 = insertelement <16 x sbyte> undef, sbyte %tmp, uint 0		; <<16 x sbyte>> [#uses=1]
	%tmp19 = insertelement <16 x sbyte> %tmp18, sbyte %tmp3, uint 1		; <<16 x sbyte>> [#uses=1]
	%tmp20 = insertelement <16 x sbyte> %tmp19, sbyte %tmp4, uint 2		; <<16 x sbyte>> [#uses=1]
	%tmp21 = insertelement <16 x sbyte> %tmp20, sbyte %tmp5, uint 3		; <<16 x sbyte>> [#uses=1]
	%tmp22 = insertelement <16 x sbyte> %tmp21, sbyte %tmp6, uint 4		; <<16 x sbyte>> [#uses=1]
	%tmp23 = insertelement <16 x sbyte> %tmp22, sbyte %tmp7, uint 5		; <<16 x sbyte>> [#uses=1]
	%tmp24 = insertelement <16 x sbyte> %tmp23, sbyte %tmp8, uint 6		; <<16 x sbyte>> [#uses=1]
	%tmp25 = insertelement <16 x sbyte> %tmp24, sbyte %tmp9, uint 7		; <<16 x sbyte>> [#uses=1]
	%tmp26 = insertelement <16 x sbyte> %tmp25, sbyte %tmp10, uint 8		; <<16 x sbyte>> [#uses=1]
	%tmp27 = insertelement <16 x sbyte> %tmp26, sbyte %tmp11, uint 9		; <<16 x sbyte>> [#uses=1]
	%tmp28 = insertelement <16 x sbyte> %tmp27, sbyte %tmp12, uint 10		; <<16 x sbyte>> [#uses=1]
	%tmp29 = insertelement <16 x sbyte> %tmp28, sbyte %tmp13, uint 11		; <<16 x sbyte>> [#uses=1]
	%tmp30 = insertelement <16 x sbyte> %tmp29, sbyte %tmp14, uint 12		; <<16 x sbyte>> [#uses=1]
	%tmp31 = insertelement <16 x sbyte> %tmp30, sbyte %tmp15, uint 13		; <<16 x sbyte>> [#uses=1]
	%tmp32 = insertelement <16 x sbyte> %tmp31, sbyte %tmp16, uint 14		; <<16 x sbyte>> [#uses=1]
	%tmp33 = insertelement <16 x sbyte> %tmp32, sbyte %tmp17, uint 15		; <<16 x sbyte>> [#uses=1]
	%tmp33 = cast <16 x sbyte> %tmp33 to <8 x short>		; <<8 x short>> [#uses=1]
	store <8 x short> %tmp33, <8 x short>* %A
	ret void
}

void %VPERM_promote(<8 x short>* %A, <8 x short>* %B) {
entry:
        %tmp = load <8 x short>* %A             ; <<8 x short>> [#uses=1]
        %tmp = cast <8 x short> %tmp to <4 x int>               ; <<4 x int>> [#uses=1]
        %tmp2 = load <8 x short>* %B            ; <<8 x short>> [#uses=1]
        %tmp2 = cast <8 x short> %tmp2 to <4 x int>             ; <<4 x int>> [#uses=1]
        %tmp3 = call <4 x int> %llvm.ppc.altivec.vperm( <4 x int> %tmp, <4 x int> %tmp2, <16 x sbyte> < sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14, sbyte 14 > )                ; <<4 x int>> [#uses=1]
        %tmp3 = cast <4 x int> %tmp3 to <8 x short>             ; <<8 x short>> [#uses=1]
        store <8 x short> %tmp3, <8 x short>* %A
        ret void
}

declare <4 x int> %llvm.ppc.altivec.vperm(<4 x int>, <4 x int>, <16 x sbyte>)


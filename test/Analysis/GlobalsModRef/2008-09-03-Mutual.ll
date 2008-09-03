; RUN: llvm-as < %s | opt -globalsmodref-aa -markmodref | llvm-dis | grep readnone

define i32 @a() {
	%tmp = call i32 @b( )		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @b() {
	%tmp = call i32 @a( )		; <i32> [#uses=1]
	ret i32 %tmp
}

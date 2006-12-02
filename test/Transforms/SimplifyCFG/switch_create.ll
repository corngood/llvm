; RUN: llvm-upgrade < %s | llvm-as | opt -simplifycfg | llvm-dis | not grep br

declare void %foo1()
declare void %foo2()

void %test1(uint %V) {
	%C1 = seteq uint %V, 4
	%C2 = seteq uint %V, 17
	%CN = or bool %C1, %C2
	br bool %CN, label %T, label %F
T:
	call void %foo1()
	ret void
F:
	call void %foo2()
	ret void
}


void %test2(int %V) {
	%C1 = setne int %V, 4
	%C2 = setne int %V, 17
	%CN = and bool %C1, %C2
	br bool %CN, label %T, label %F
T:
	call void %foo1()
	ret void
F:
	call void %foo2()
	ret void
}


void %test3(int %V) {
	%C1 = seteq int %V, 4
	br bool %C1, label %T, label %N
N:
	%C2 = seteq int %V, 17
	br bool %C2, label %T, label %F
T:
	call void %foo1()
	ret void
F:
	call void %foo2()
	ret void
}



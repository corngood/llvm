; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep fctiwz | count 1

implementation

ushort %foo(float %a) {
entry:
        %tmp.1 = cast float %a to ushort
        ret ushort %tmp.1
}

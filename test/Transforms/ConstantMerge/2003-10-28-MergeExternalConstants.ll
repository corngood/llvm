; RUN: llvm-upgrade < %s | llvm-as | opt -constmerge | llvm-dis | %prcontext foo 2 | grep bar

%foo = constant int 6
%bar = constant int 6

implementation


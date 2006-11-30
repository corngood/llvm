// This is a regression test on debug info to make sure we don't hit a compile unit size
// issue with gdb.
// RUN: %llvmgcc -S -O0 -g %s -o - | llvm-as | llc --disable-fp-elim -o Output/NoCompileUnit.s -f
// RUN: as Output/NoCompileUnit.s -o Output/NoCompileUnit.o
// RUN: g++ Output/NoCompileUnit.o -o Output/NoCompileUnit.exe
// RUN: ( echo "break main"; echo "run" ; echo "p NoCompileUnit::pubname" ) > Output/NoCompileUnit.gdbin 
// RUN: gdb -q -batch -n -x Output/NoCompileUnit.gdbin Output/NoCompileUnit.exe | tee Output/NoCompileUnit.out | not grep '"low == high"'
// XFAIL: i[1-9]86|alpha|ia64|arm


class MamaDebugTest {
private:
  int N;
  
protected:
  MamaDebugTest(int n) : N(n) {}
  
  int getN() const { return N; }

};

class BabyDebugTest : public MamaDebugTest {
private:

public:
  BabyDebugTest(int n) : MamaDebugTest(n) {}
  
  static int doh;
  
  int  doit() {
    int N = getN();
    int Table[N];
    
    int sum = 0;
    
    for (int i = 0; i < N; ++i) {
      int j = i;
      Table[i] = j;
    }
    for (int i = 0; i < N; ++i) {
      int j = Table[i];
      sum += j;
    }
    
    return sum;
  }

};

int BabyDebugTest::doh;


int main(int argc, const char *argv[]) {
  BabyDebugTest BDT(20);
  return BDT.doit();
}



// Check that we can compile files of different types together.
// RUN: llvmc2 %s %p/together.c -o %t
// RUN: ./%t | grep hello

extern "C" void test();

int main() {
  test();
}

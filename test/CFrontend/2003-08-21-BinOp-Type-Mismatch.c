// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

struct bar;

void foo()
{
  unsigned int frame, focus;
  (struct bar *) focus == (focus ? ((struct bar *) frame) : 0);
}


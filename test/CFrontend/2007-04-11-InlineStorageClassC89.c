// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep define | grep xglobWeak | grep weak | wc -l | grep 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep define | grep xextWeak | grep weak | wc -l | grep 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep define | grep xWeaknoinline | grep weak wc -l | grep 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep define | grep xWeakextnoinline | grep weak wc -l | grep 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep define | grep xglobnoWeak | grep -v internal | grep -v weak | grep -v linkonce | wc -l | grep 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep define | grep xstatnoWeak | grep internal | wc -l | grep 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep declare | grep xextnoWeak | grep -v internal | grep -v weak | grep -v linkonce | wc -l | grep 1
inline int xglobWeak(int) __attribute__((weak));
inline int xglobWeak (int i) {
  return i*2;
}
inline int xextWeak(int) __attribute__((weak));
extern  inline int xextWeak (int i) {
  return i*4;
}
int xWeaknoinline(int) __attribute__((weak));
int xWeaknoinline(int i) {
  return i*8;
}
int xWeakextnoinline(int) __attribute__((weak));
extern int xWeakextnoinline(int i) {
  return i*16;
}
inline int xglobnoWeak (int i) {
  return i*32;
}
static inline int xstatnoWeak (int i) {
  return i*64;
}
extern  inline int xextnoWeak (int i) {
  return i*128;
}
int j(int y) {
  return xglobnoWeak(y)+xstatnoWeak(y)+xextnoWeak(y)+
        xglobWeak(y)+xextWeak(y)+
        xWeakextnoinline(y)+xWeaknoinline(y);
}

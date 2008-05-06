// Test that we can compile .c files as C++ and vice versa
// RUN: llvmc2 -x c++ %s -x c %p/false.cpp -x lisp -x whatnot -x none %p/false2.cpp -o %t
// RUN: ./%t | grep hello

#include <iostream>

extern "C" void test();
extern std::string test2();

int main() {
    std::cout << "h";
    test();
    std::cout << test2() << '\n';
}

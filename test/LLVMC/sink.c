/*
 * Check that the 'sink' options work.
 * RUN: llvmc2 -v -Wall %s -o %t |& grep "Wall"
 * RUN: ./%t | grep hello
 */

#include <stdio.h>

int main() {
    printf("hello\n");
    return 0;
}

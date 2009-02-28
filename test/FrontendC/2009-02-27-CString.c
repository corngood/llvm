/* RUN: %llvmgcc %s -S -o - -emit-llvm | \
   RUN: egrep {CSTRING SECTION.\*section.\*__TEXT,.\*__cstring}
   XFAIL: linux,ia64,alpha,sparc
   TARGET: *-*-darwin*
   END.
   Insure that stings go to the cstring section.  This test is
   intended solely for Darwin targets.
 */
char *foo() {
  return "this string should go to the CSTRING SECTION";
}

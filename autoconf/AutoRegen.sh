#!/bin/sh
die () {
	echo "$@" 1>&2
	exit 1
}
if test "$1" = --with-automake ; then
  outfile=configure_am
  configfile=configure.am
  with_automake=1
elif test -z "$1" ; then
  outfile=configure
  configfile=configure.ac
  with_automake=0
else
  die "Invalid option: $1"
fi
test -d autoconf && test -f autoconf/$configfile && cd autoconf
test -f $configfile || die "Can't find 'autoconf' dir; please cd into it first"
autoconf --version | egrep '2\.59' > /dev/null
if test $? -ne 0 ; then
  die "Your autoconf was not detected as being 2.59"
fi
aclocal --version | egrep '1\.9\.2' > /dev/null
if test $? -ne 0 ; then
  die "Your aclocal was not detected as being 1.9.1"
fi
autoheader --version | egrep '2\.59' > /dev/null
if test $? -ne 0 ; then
  die "Your autoheader was not detected as being 2.59"
fi
libtool --version | grep '1.5.10' > /dev/null
if test $? -ne 0 ; then
  die "Your libtool was not detected as being 1.5.10"
fi
if test $with_automake -eq 1 ; then
  automake --version | grep 'automake.*1.9.2' > /dev/null
  if test $? -ne 0 ; then
    die "Your automake was not detected as being 1.9.2"
  fi
fi
echo ""
echo "### NOTE: ############################################################"
echo "### If you get *any* warnings from autoconf below you MUST fix the"
echo "### scripts in the m4 directory because there are future forward"
echo "### compatibility or platform support issues at risk. Please do NOT"
echo "### commit any configure script that was generated with warnings"
echo "### present. You should get just three 'Regenerating..' lines."
echo "######################################################################"
echo ""
echo "Regenerating aclocal.m4 with aclocal"
cwd=`pwd`
if test $with_automake -eq 1 ; then
  mv configure.ac .configure.ac.save
  cp configure.am configure.ac
  cp configure.am ../configure.ac
fi
aclocal --force -I $cwd/m4 || die "aclocal failed"
echo "Regenerating configure with autoconf 2.59"
autoconf --force --warnings=all -o ../$outfile $configfile || die "autoconf failed"
cd ..
echo "Regenerating config.h.in with autoheader 2.59"
autoheader -I autoconf -I autoconf/m4 autoconf/$configfile || die "autoheader failed"
if test $with_automake -eq 1 ; then
  echo "Regenerating makefiles with automake 1.9.2"
  cp autoconf/aclocal.m4 .
  automake --gnu --add-missing --copy --force-missing
  cd $cwd
  mv .configure.ac.save configure.ac
fi
exit 0

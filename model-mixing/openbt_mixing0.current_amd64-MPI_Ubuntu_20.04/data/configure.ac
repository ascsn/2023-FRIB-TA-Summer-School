AC_PREREQ([2.69])

AC_INIT(openbt, m4_esyscmd([git describe --tags | tr -d '\n']), mpratola@gmail.com, openbt)
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([tree.h])
AC_MSG_NOTICE([Open Bayesian Trees Project])
AC_LANG(C++)

AC_ARG_WITH([mpi],
[AS_HELP_STRING([--with-mpi],
  [enable MPI support])],
[with_mpi="yes"],
[with_mpi="no"])

cpp_flags=""

dnl if --with-mpi was passed to configure, set the compiler to mpicxx, which 
dnl will take precendence over AC_PROG_CXX below.
AS_IF([test "x$with_mpi" = "xyes"],[
dnl	CXX=mpicxx
	AX_PROG_CXX_MPI
	AX_PROG_CC_MPI
	cpp_flags="$cpp_flags -D_OPENMPI"],
	[])

AC_ARG_WITH([silent],
[AS_HELP_STRING([--with-silent],
  [Silent decreases output while running MCMC])],
[with_silent="yes"],
[with_silent="no"])

dnl --with-silent option
AS_IF([test "x$with_silent" = "xyes"],[
	cpp_flags="$cpp_flags -DSILENT"],
	[])


dnl ---------------------Check for programs----------------------------------
AS_IF([test "x$with_mpi" = "xno"],[
AC_PROG_CXX
AC_PROG_CC],[])

AC_PROG_AWK
dnl AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
dnl AC_PROG_RANLIB

dnl Populate C++ compiler flags
cpp_flags="$cpp_flags -Wall"
dnl cpp_flags="$cpp_flags -Wextra"
cpp_flags="$cpp_flags -pipe"
cpp_flags="$cpp_flags -g"
cpp_flags="$cpp_flags -O3"

dnl check for c++0x (c++11) is performed below
cpp_flags="$cpp_flags -std=c++0x"

cpp_flags="$cpp_flags -DNOR"

dnl ---------------------Checks for libraries--------------------------------
LT_INIT
AC_ENABLE_SHARED
AC_DISABLE_STATIC
AC_PROG_LIBTOOL(libtool)

dnl Add in openmp flag(s) if we aren't using MPI (this is the default case).
dnl Note that if openmp is also not detected, will will revert to serial compile.
AS_IF([test "x$with_mpi" = "xno"],[
	AC_OPENMP
	cpp_flags="$cpp_flags $OPENMP_CXXFLAGS"
	],
	[])


dnl ---------------------Finalize the CXXFLAGS-------------------------------
AC_SUBST(CXXFLAGS, ["$cpp_flags"])


dnl ---------------------Checks for header files-----------------------------

dnl Checks for typedefs, structures, and compiler characteristics.
dnl - Require -std=c++0x support.  Uses ax_cxx_compile_stdcxx_11.m4 macro.
dnl   Run aclocal before autoconf to enable macro.
AX_CXX_COMPILE_STDCXX_11([noext], [mandatory])

AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

dnl ----------------------Checks for library functions-----------------------
AC_CHECK_FUNCS([floor pow sqrt])


dnl ----------------------Process Makefile.in to produce Makefile------------
AC_OUTPUT(Makefile)
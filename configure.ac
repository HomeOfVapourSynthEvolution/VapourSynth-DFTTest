AC_INIT([DFTTest], [4], [https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/issues], [DFTTest], [https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/])

: ${CXXFLAGS=""}

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([disable-static win32-dll])

AC_CANONICAL_HOST

AC_PROG_CXX


X86="false"

AS_CASE(
        [$host_cpu],
        [i?86], [BITS="32" X86="true"],
        [x86_64], [BITS="64" X86="true"]
)

AS_CASE(
        [$host_os],
        [cygwin*|mingw*],
        [
         AS_IF(
               [test "x$BITS" = "x32"],
               [
                AC_SUBST([PLUGINLDFLAGS], ["-Wl,--kill-at"])
               ]
         )
        ]
)

AS_IF(
      [test "x$X86" = "xtrue"],
      [
       AC_DEFINE([VS_TARGET_CPU_X86])

       AC_SUBST([MFLAGS], ["-mfpmath=sse -msse2"])
      ]
)


AM_CONDITIONAL([VS_TARGET_CPU_X86], [test "x$X86" = "xtrue"])


PKG_CHECK_MODULES([VapourSynth], [vapoursynth])
PKG_CHECK_MODULES([FFTW3F], [fftw3f])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

warning_flags = -Wall -Wextra -Wshadow -Wno-unused-parameter -Wno-attributes
common_cflags = -O3 -ffast-math -fvisibility=hidden $(warning_flags) $(MFLAGS)
AM_CXXFLAGS = -std=c++14 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS) $(FFTW3F_CFLAGS)

lib_LTLIBRARIES = libdfttest.la

libdfttest_la_SOURCES = DFTTest/DFTTest.cpp \
                        DFTTest/DFTTest.hpp \
                        DFTTest/vectorclass/instrset.h \
                        DFTTest/vectorclass/instrset_detect.cpp

libdfttest_la_LIBADD = $(FFTW3F_LIBS)

if VS_TARGET_CPU_X86
libdfttest_la_SOURCES += DFTTest/DFTTest_SSE2.cpp \
                         DFTTest/vectorclass/vectorclass.h \
                         DFTTest/vectorclass/vectorf128.h \
                         DFTTest/vectorclass/vectorf256.h \
                         DFTTest/vectorclass/vectorf256e.h \
                         DFTTest/vectorclass/vectori128.h \
                         DFTTest/vectorclass/vectori256.h \
                         DFTTest/vectorclass/vectori256e.h \
                         DFTTest/vectorclass/vectormath_common.h \
                         DFTTest/vectorclass/vectormath_exp.h

noinst_LTLIBRARIES = libavx2.la

libavx2_la_SOURCES = DFTTest/DFTTest_AVX2.cpp
libavx2_la_CXXFLAGS = $(AM_CXXFLAGS) -mavx2 -mfma

libdfttest_la_LIBADD += libavx2.la
endif

libdfttest_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)

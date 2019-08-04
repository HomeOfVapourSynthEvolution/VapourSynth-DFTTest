project('DFTTest', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++14'],
  meson_version : '>=0.48.0',
  version : '5'
)

add_project_arguments('-ffast-math', language : 'cpp')

sources = [
  'DFTTest/DFTTest.cpp',
  'DFTTest/DFTTest.hpp',
  'DFTTest/vectorclass/instrset.h',
  'DFTTest/vectorclass/instrset_detect.cpp'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

fftw3f_dep = dependency('fftw3f')

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-DVS_TARGET_CPU_X86', '-mfpmath=sse', '-msse2', language : 'cpp')

  sources += [
    'DFTTest/DFTTest_SSE2.cpp',
    'DFTTest/vectorclass/vectorclass.h',
    'DFTTest/vectorclass/vectorf128.h',
    'DFTTest/vectorclass/vectorf256.h',
    'DFTTest/vectorclass/vectorf256e.h',
    'DFTTest/vectorclass/vectori128.h',
    'DFTTest/vectorclass/vectori256.h',
    'DFTTest/vectorclass/vectori256e.h',
    'DFTTest/vectorclass/vectormath_common.h',
    'DFTTest/vectorclass/vectormath_exp.h'
  ]

  libs += static_library('avx2', 'DFTTest/DFTTest_AVX2.cpp',
    dependencies : [vapoursynth_dep, fftw3f_dep],
    cpp_args : ['-mavx2', '-mfma'],
    gnu_symbol_visibility : 'hidden'
  )
endif

shared_module('dfttest', sources,
  dependencies : [vapoursynth_dep, fftw3f_dep],
  link_with : libs,
  install : true,
  install_dir : join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
  gnu_symbol_visibility : 'hidden'
)

project('DFTTest', 'cpp',
  default_options: ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++17'],
  meson_version: '>=0.48.0',
  version: '7'
)

cxx = meson.get_compiler('cpp')

sources = [
  'DFTTest/DFTTest.cpp',
  'DFTTest/DFTTest.h'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args: true, includes: true)

fftw3f_dep = dependency('fftw3f')

deps = [vapoursynth_dep, fftw3f_dep]

test_fftwf_threads = '''
#include <fftw3.h>
int main() {
    fftwf_init_threads();
    return 0;
}
'''
if not cxx.links(test_fftwf_threads, dependencies: fftw3f_dep)
  deps += cxx.find_library('fftw3f_threads')
endif

libs = []

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-fno-math-errno', '-fno-trapping-math', '-DDFTTEST_X86', '-mfpmath=sse', '-msse2', language: 'cpp')

  sources += [
    'DFTTest/DFTTest_SSE2.cpp',
    'DFTTest/VCL2/instrset.h',
    'DFTTest/VCL2/instrset_detect.cpp',
    'DFTTest/VCL2/vector_convert.h',
    'DFTTest/VCL2/vectorclass.h',
    'DFTTest/VCL2/vectorf128.h',
    'DFTTest/VCL2/vectorf256.h',
    'DFTTest/VCL2/vectorf256e.h',
    'DFTTest/VCL2/vectorf512.h',
    'DFTTest/VCL2/vectorf512e.h',
    'DFTTest/VCL2/vectori128.h',
    'DFTTest/VCL2/vectori256.h',
    'DFTTest/VCL2/vectori256e.h',
    'DFTTest/VCL2/vectori512.h',
    'DFTTest/VCL2/vectori512e.h',
    'DFTTest/VCL2/vectori512s.h',
    'DFTTest/VCL2/vectori512se.h',
    'DFTTest/VCL2/vectormath_common.h',
    'DFTTest/VCL2/vectormath_exp.h',
    'DFTTest/VCL2/vectormath_hyp.h',
    'DFTTest/VCL2/vectormath_lib.h',
    'DFTTest/VCL2/vectormath_trig.h'
  ]

  libs += static_library('avx2', 'DFTTest/DFTTest_AVX2.cpp',
    dependencies: deps,
    cpp_args: ['-mavx2', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )

  libs += static_library('avx512', 'DFTTest/DFTTest_AVX512.cpp',
    dependencies: deps,
    cpp_args: ['-mavx512f', '-mavx512vl', '-mavx512bw', '-mavx512dq', '-mfma'],
    gnu_symbol_visibility: 'hidden'
  )
endif

shared_module('dfttest', sources,
  dependencies: deps,
  link_with: libs,
  install: true,
  install_dir: join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth'),
  gnu_symbol_visibility: 'hidden'
)

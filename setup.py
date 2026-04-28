#!/usr/bin/env python
import sys
import numpy as np
from setuptools import setup, Extension

_falcon_src_dir = "ezfalcon/dynamics/forces/self_gravity/falcON/_falcON_src"

_falcon = Extension(
    name="ezfalcon.dynamics.forces.self_gravity.falcON._falcon",
    sources=[
        "ezfalcon/dynamics/forces/self_gravity/falcON/_falcON_wrapper.cpp",
        f"{_falcon_src_dir}/src/basic.cc",
        f"{_falcon_src_dir}/src/body.cc",
        f"{_falcon_src_dir}/src/gravity.cc",
        f"{_falcon_src_dir}/src/kernel.cc",
        f"{_falcon_src_dir}/src/tree.cc",
        f"{_falcon_src_dir}/src/exception.cc",
        f"{_falcon_src_dir}/src/numerics.cc",
        f"{_falcon_src_dir}/src/io.cc",
    ],
    include_dirs=[
        f"{_falcon_src_dir}/inc",
        f"{_falcon_src_dir}/inc/public",
        f"{_falcon_src_dir}/inc/utils",
        np.get_include(),
    ],
    define_macros=[
        ("falcON_DOUBLE", None),
    ],
    language="c++",
)

_direct_summation = Extension(
    name="ezfalcon.dynamics.forces.self_gravity.direct_summation._direct_summation",
    sources=["ezfalcon/dynamics/forces/self_gravity/direct_summation/_direct_wrapper.cpp"],
    include_dirs=[np.get_include()],
    language="c++",
)

if sys.platform == "win32":
    _simd_flags = ["/O2", "/fp:fast", "/openmp:llvm"]
else:
    _simd_flags = ["-O3", "-ffast-math", "-funroll-loops", "-fopenmp-simd"]

_external_potentials = Extension(
    name="ezfalcon.dynamics.forces.external_force._external_potentials",
    sources=["ezfalcon/dynamics/forces/external_force/_external_potentials_wrapper.cpp"],
    include_dirs=[
        np.get_include(),
        "ezfalcon/dynamics/forces",
    ],
    extra_compile_args=_simd_flags,
    language="c++",
)

_composite_shim = Extension(
    name="ezfalcon.dynamics.forces._composite_shim",
    sources=["ezfalcon/dynamics/forces/_composite_shim.cpp"],
    include_dirs=[
        np.get_include(),
        "ezfalcon/dynamics/forces",
        "ezfalcon/dynamics/integration",
    ],
    extra_compile_args=_simd_flags,
    language="c++",
)

_leapfrog_c = Extension(
    name="ezfalcon.dynamics.integration._leapfrog_c",
    sources=["ezfalcon/dynamics/integration/_leapfrog_c.cpp"],
    include_dirs=[
        np.get_include(),
        "ezfalcon/dynamics/forces",
        "ezfalcon/dynamics/integration",
    ],
    extra_compile_args=_simd_flags,
    language="c++",
)

setup(ext_modules=[_falcon, _direct_summation, _external_potentials, _composite_shim, _leapfrog_c])

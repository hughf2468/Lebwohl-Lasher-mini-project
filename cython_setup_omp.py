from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "LebwohlLasher_cython_funcs_omp",
        ["LebwohlLasher_cython_funcs_omp.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp', '-Wl,-rpath,/opt/homebrew/opt/gcc/lib/gcc/current/']
    )
]

setup(ext_modules=cythonize(ext_modules))

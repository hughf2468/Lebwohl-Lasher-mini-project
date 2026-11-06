from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("LebwohlLasher_cython_funcs.pyx", annotate=True))
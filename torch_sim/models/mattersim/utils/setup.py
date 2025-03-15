"""Setup script for building Cython extensions."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup  # Changed from distutils to setuptools


package = Extension(
    "threebody_indices",
    ["threebody_indices.pyx"],
    include_dirs=[np.get_include()],
)
setup(ext_modules=cythonize([package]))

# python setup.py build_ext --inplace

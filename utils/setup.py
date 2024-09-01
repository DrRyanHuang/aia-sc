from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    name='checkfixing!',
    ext_modules=cythonize('check_fixing.pyx'),
    include_dirs=[numpy.get_include()]
)

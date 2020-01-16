from distutils.core import setup
from Cython.Build import cythonize

setup(name='Faster Seam Carving',
      ext_modules=cythonize("carve_fast.pyx"))
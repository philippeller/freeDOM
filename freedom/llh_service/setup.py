from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy
import zmq

extensions = [
    Extension(
        "llh_cython",
        ["llh_cython.pyx"],
        include_dirs=zmq.get_includes() + [numpy.get_include()],
        extra_link_args=["-lzmq"],
    )
]
setup(name="llh_cython", ext_modules=cythonize(extensions))

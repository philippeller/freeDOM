# -*- coding: utf-8 -*-

"""
Installation script for the frreDOM project
"""

from __future__ import absolute_import

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, Extension, find_packages


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


install_requires = [
    "matplotlib>=2.0",
    "tensorflow>=2.1.0",
    "sklearn",
    "scipy",
    "numpy",
    "pyzmq>=19.0.1",
]

# handle cython extension
import numpy
import zmq

# adapted from https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
USE_CYTHON = True

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False

ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    Extension(
        "freedom.llh_service.llh_cython",
        ["freedom/llh_service/llh_cython" + ext],
        include_dirs=zmq.get_includes() + [numpy.get_include()],
        extra_link_args=["-lzmq"],
    )
]

if USE_CYTHON:
    extensions = cythonize(extensions)

# If tensorflow-gpu is installed, use that
if get_dist("tensorflow") is None and get_dist("tensorflow-gpu") is not None:
    install_requires = [
        pkg.replace("tensorflow", "tensorflow-gpu") for pkg in install_requires
    ]

setup(
    name="freedom",
    description=("approximate likelihood reconstruction for free detector geometries"),
    author="Philipp Eller et al.",
    author_email="peller.phys@gmail.com",
    url="https://github.com/philippeller/freeDOM",
    license="Apache 2.0",
    version="0.1",
    python_requires=">=3.6",
    setup_requires=["pip>=1.8", "setuptools>18.5", "numpy>=1.11", "pyzmq>=19.0.1",],
    install_requires=install_requires,
    packages=find_packages(),
    package_data={"freedom.resources": ["*.npy", "*.csv", "*.pkl", "*.hdf5"],},
    zip_safe=False,
    ext_modules=extensions,
)

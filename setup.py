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
# adapted from https://stackoverflow.com/questions/2379898/make-distutils-look-for-numpy-header-files-in-the-correct-place
try:
    from Cython.Distutils import build_ext
except:
    # If we couldn't import Cython, use the normal setuptools
    # and look for a pre-compiled .c file instead of a .pyx file
    from setuptools.command.build_ext import build_ext

    ext = ".c"
else:
    # If we successfully imported Cython, look for a .pyx file
    ext = ".pyx"

extensions = [
    Extension(
        "freedom.llh_service.llh_cython",
        ["freedom/llh_service/llh_cython" + ext],
        extra_link_args=["-lzmq"],
    )
]


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        import zmq

        # Add numpy/zmq headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        self.include_dirs.extend(zmq.get_includes())

        # Call original build_ext command
        build_ext.run(self)


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
    cmdclass={"build_ext": CustomBuildExtCommand},
    ext_modules=extensions,
)

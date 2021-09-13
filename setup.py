#
# Copyright 2016-2021 Lars Pastewka
#           2020 Michael Röttger
#           2018-2020 Antoine Sanner
#           2015-2016 Till Junge
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import glob
from pathlib import Path
import re

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


#
# Optain LAPACK configuration from numpy
#

try:
    import numpy as np
except ImportError:
    print("WARNING: Could not find numpy, skipping LAPACK detection.")
    np = None

lib_dirs = []
libs = []
extra_link_args = []
extra_objects = []
if np is not None:
    for k, v in np.__config__.__dict__.items():
        if re.match('lapack_.*_info', k):
            if v:
                print("* Using LAPACK information from '{}' dictionary in " \
                    "numpy.__config__".format(k))
                try:
                    print("    library_dirs = {}".format(v['library_dirs']))
                    lib_dirs += v['library_dirs']
                except:
                    print("    No 'library_dirs' entry found.")
                try:
                    print("    libraries = {}".format(v['libraries']))
                    libs += v['libraries']
                except:
                    print("    No 'libraries' entry found.")
                try:
                    print("    extra_link_args = '{}'".format(v['extra_link_args']))
                    extra_link_args += v['extra_link_args']
                except:
                    print("    No 'extra_link_args' entry found.")
                # We use whichever lapack_*_info comes first, hopefully there is
                # just one.
                break
    if len(libs) == 0:
        # No entries from np.__config__
        print("* Using lapack_lite")
        extra_objects += [glob.glob(np.linalg.__path__[0]+'/lapack_lite*.so')[0]]


extra_compile_args = ["-std=c++11"]
print(extra_objects)

scripts = []

extensions = [
    Extension(
        name='_SurfaceTopography',
        sources=['c/autocorrelation.cpp',
                 'c/bicubic.cpp',
                 'c/patchfinder.cpp',
                 'c/module.cpp'],
        extra_compile_args=extra_compile_args,
        library_dirs=lib_dirs,
        libraries=libs,
        extra_link_args=extra_link_args,
        extra_objects=extra_objects
    )
]

setup(
    name="SurfaceTopography",
    cmdclass={'build_ext': CustomBuildExtCommand} if np else {},
    scripts=scripts,
    packages=find_packages(),
    package_data={'': ['ChangeLog.md']},
    include_package_data=True,
    ext_modules=extensions,
    # metadata for upload to PyPI
    author='Lars Pastewka',
    author_email='lars.pastewka@imtek.uni-freiburg.de',
    description='Read and analyze surface topographies',
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    license='MIT',
    test_suite='tests',
    # dependencies
    python_requires='>=3.5.0',
    use_scm_version=True,
    zip_safe=False,
    setup_requires=[
        'setuptools_scm>=3.5.0',
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
    install_requires=[
        'numpy>=1.16.3',
        'NuMPI>=0.1.4',
        'muFFT>=0.12.0',
        'igor',
        'h5py',
        'defusedxml',
        'numpyencoder',
        'pyyaml',
        'Pillow',
        'requests',
        'matplotlib>=1.0.0',
    ]
)

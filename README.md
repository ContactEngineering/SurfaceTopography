SurfaceTopography
=================

*Read and analyze surface topographies with Python.* This code implements basic classes for handling uniform and
nonuniform surface topography data. It contains a rich set of import filters for experimental surface topography data.
Surface topographies can be easily analyzed using standard (rms height, power spectrum, ...) and some special purpose
(autocorrelation function, variable bandwidth analysis, ...) statistical techniques. 

If you use this code, please cite:
[Jacobs, Junge, Pastewka, Surf. Topogr. Metrol. Prop. 1, 013001 (2017)](https://doi.org/10.1088/2051-672X/aa51f8)

Build status
------------

The following badge should say _build passing_. This means that all automated tests completed successfully for the master branch.

[![Build Status](https://travis-ci.org/ComputationalMechanics/SurfaceTopography.svg?branch=master)](https://travis-ci.org/github/ComputationalMechanics/SurfaceTopography)

Documentation
-------------

This README file contains a brief introduction into the code. The full documentation can be found at https://computationalmechanics.github.io/SurfaceTopography/.

Installation
------------

You need Python 3 and [FFTW3](http://www.fftw.org/) to run SurfaceTopography. All Python dependencies can be installed
automatically by invoking

#### Installation directly with pip

```bash
# dependencies not installable with requirements.txt
pip install [--user] numpy
pip install [--user] pylint
pip install [--user] cython
pip install [--user] mpi4py #optional

# install SurfaceTopography
pip  install [--user]  git+https://github.com/ComputationalMechanics/SurfaceTopography.git
```

The last command will install other dependencies including 
[muFFT](https://gitlab.com/muspectre/muspectre.git), 
[NuMPI](https://github.com/IMTEK-Simulation/NuMPI.git) and
[a fork of runtests](https://github.com/AntoineSIMTEK/runtests.git)

Note: sometimes [muFFT](https://gitlab.com/muspectre/muspectre.git) will not find the FFTW3 installation you expect.
You can specify the directory where you installed [FFTW3](http://www.fftw.org/)  
by setting the environment variable `FFTWDIR` (e.g. to `$USER/.local`) 

#### Installation from source directory 

If you cloned the repository. You can install the dependencies with

```
pip install [--user] numpy
pip install [--user] pylint
pip install [--user] cython
pip install [--user] mpi4py #optional
pip3 install [--user] -r requirements.txt
```

in the source directory. SurfaceTopography can be installed by invoking

```pip3 install [--user] .```

in the source directoy. The command line parameter --user is optional and leads to a local installation in the current user's `$HOME/.local` directory.

#### Installation problems with lapack and openblas

`bicubic.cpp` is linked with `lapack`, that is already available as a dependency of `numpy`. 

If during build, `setup.py` fails to link to one of the lapack implementations 
provided by numpy, as was experienced for mac, try providing following environment variables: 

```bash
export LDFLAGS="-L/usr/local/opt/openblas/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/openblas/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig:$PKG_CONFIG_PATH"

export LDFLAGS="-L/usr/local/opt/lapack/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/lapack/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig:$PKG_CONFIG_PATH"
```    
where the paths have probably to be adapted to your particular installation method
(here it was an extra homebrew installation).

Updating SurfaceTopography
--------------------------

If you update SurfaceTopography (whether with pip or `git pull` if you cloned the repository),  you may need to
uninstall `NuMPI`, `muSpectre` and or `runtests`, so that the newest version of them will be installed.

Testing
-------

To run the automated tests, go to the main source directory and execute the following:

```
pytest
```

Tests that are parallelizable have to run with [runtests](https://github.com/AntoineSIMTEK/runtests)
```
python run-tests.py 
``` 

You can choose the number of processors with the option `--mpirun="mpirun -np 4"`. For development purposes you can go beyond the number of processors of your computer using `--mpirun="mpirun -np 10 --oversubscribe"`

Other usefull flags:
- `--xterm`: one window per processor
- `--xterm --pdb`: debugging

Development
-----------

To use the code without installing it, e.g. for development purposes, use the `env.sh` script to set the environment:

```source /path/to/SurfaceTopography/env.sh [python3]```

Note that the parameter to `env.sh` specifies the Python interpreter for which the environment is set up. 
SurfaceTopography contains portions that need to be compiled, make sure to run

```python setup.py build```

whenever any of the C (.c) sources are modified.

Please read [CONTRIBUTING](CONTRIBUTING.md) if you plan to contribute to this code.

Compiling the documentation
---------------------------

- After changes to the SurfaceTopography source, you have to build again: ```python setup.py build```
- Navigate into the docs folder: ```cd docs/``` 
- Automatically generate reStructuredText files from the source: ```sphinx-apidoc -o source/ ../SurfaceTopography``` 
Do just once, or if you have added/removed classes or methods. In case of the latter, be sure to remove the previous
source before: ```rm -rf source/```
- Build html files: ```make html```
- The resulting html files can be found in the ```SurfaceTopography/docs/_build/html/``` folder. Root is
```SurfaceTopography/docs/_build/html/index.html```.

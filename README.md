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

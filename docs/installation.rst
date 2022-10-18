Installation
============
We only support installation on recent mac and linux systems, not on windows.

You need Python 3,  NetCDF_, OpenBLAS_, LAPACK_ and FFTW3_ in order to install SurfaceTopography.

In complement to the instructions bellow, you will find examples of system setups and intallation workflows in our singularity containers
`singularity containers <https://github.com/ContactEngineering/SurfaceTopography/blob/master/singularity/SurfaceTopography_serial.def>`_ 
and our `testing workflows <https://github.com/ContactEngineering/SurfaceTopography/blob/master/.github/workflows/tests.yml>`_ 

Direct installation with pip
----------------------------

SurfaceTopography can be installed by invoking

.. code-block:: bash

    python3 -m pip  install [--user] SurfaceTopography

Or if you want the latest unreleased version

.. code-block:: bash

    python3 -m pip  install [--user] git+https://github.com/ContactEngineering/SurfaceTopography.git


The command will install other dependencies including muFFT_, NuMPI_ and
runtests_.

If you want to install all optional dependencies for full functionality:

.. code-block:: bash

    wget https://raw.githubusercontent.com/ContactEngineering/SurfaceTopography/master/requirements.txt
    # I found the url to the requirements files using
    # https://api.github.com/repos/ContactEngineering/SurfaceTopography/contents/requirements.txt
    python3 -m pip install [--user] -r requirements.txt --no-binary numpy
    rm requirements.txt

Tip: to install FFTW3, NetCDF and BLAS/LAPACK on ubuntu, you can for example use

.. code-block:: bash

    sudo apt-get install libfftw3-dev libopenblas-dev libnetcdf-dev

See also our `singularity container <https://github.com/ContactEngineering/SurfaceTopography/blob/master/singularity/SurfaceTopography_serial.def>`_ for an example installation on ubuntu.

Tip: to install FFTW3, NetCDF and BLAS/LAPACK on mac using Homebrew_, 

.. code-block:: bash

    brew install openblas lapack fftw netcdf

Installation: Common problems
-----------------------------

- Sometimes muFFT_ will not find the FFTW3 installation you expect.
  You can specify the directory where you installed FFTW3_
  by setting the environment variable `FFTWDIR` (e.g. to `$USER/.local`).

- If muFFT_ is unable to find the NetCDF libraries (the `FileIONetCDF` class
  is missing), then set the environment variables `NETCDFDIR` (for serial
  compile) or `PNETCDFDIR` (for parallel compiles, to e.g. `$USER/.local`).

- Sometimes the installation fails because muFFT_ attempts to compile with `MPI` support but not all necessary libraries are
  avaible. If you do not need `MPI` support, you can manually disable it in the following way:

  .. code-block:: bash

                  python3 -m pip install muFFT --install-option="--disable-mpi"
                  python3 -m pip install SurfaceTopography


- Note that if you do not install a tagged version of a dependency (e.g. because you install from the master branch via`git+` or from source using directly `setup.py`),
  pip will attempt to reinstall that dependency despite it is already installed.
  In that case you need to avoid using `pip install` and install SurfaceTopography from the source directory using `python3 setup.py install`.

- linker cannot find lopenblas64

  Here an extract of the installation error:

        .. code-block::

            g++ -pthread -shared -Wl,--rpath=/opt/hostedtoolcache/Python/3.8.13/x64/lib -Wl,--rpath=/opt/hostedtoolcache/Python/3.8.13/x64/lib bui/temp.linux-x86_64-3.8/c/autocorrelation.o build/temp.linux-x86_64-3.8/c/bicubic.o build/temp.linux-x86_64-3.8/c/patchfinder.build/temp.linux-x86_64-3.8/c/module.o -L/usr/local/lib -L/opt/hostedtoolcache/Python/3.8.13/x64/lib -lopenblas64_ -lopenblas64_ -o build/lib.linux-x86_64-3.8/_SurfaceTopography.cpython-38-x86_64-linux-gnu.so
            /usr/bin/ld: cannot find -lopenblas64_
            /usr/bin/ld: cannot find -lopenblas64   collect2: error: ld returned 1 exit status
            error: command 'g++' failed with exit status    [end of output]
            note: This error originates from a subprocess, and is likely not a problem with pip   error: legacy-install-failure

    Solution:   reinstall numpy with `--no-binary` flag

    .. code-block:: bash
        python3 -m pip install numpy --no-binary numpy

Reporting installation problems
-------------------------------

1. Make sure that you carefully read all these instructions.
2. Try to find similar issues in our issues or forum discussions. 
3. If the installation problems are related to muFFT_ you can have a look at d
4. Please open an issue or a discussion in the forum.

When reporting a problem, please provide us with following information: 

- your system configuration, 
- your python3 environment (output of `python3 -m pip list`)
- The output of the verbose installation e.g. `python3 -m pip install --verbose --global-option="--verbose"`

Installation from source directory
--------------------------------

If you cloned the repository. You can install the dependencies with

.. code-block:: bash

    python3 -m pip install -r requirements.txt

in the source directory. SurfaceTopography can be installed by invoking

.. code-block:: bash

   python3 -m pip install [--user] .

or

.. code-block:: bash

   python3 setup.py install [--user]

in the source directoy. The command line parameter `--user` is optional and leads to a local installation in the current user's `$HOME/.local` directory.

Installation problems with LAPACK and OpenBLAS
-----------------------------------------------

`bicubic.cpp` is linked with `lapack`, that is already available as a dependency of `numpy`.
If during build, `setup.py` fails to link to one of the lapack implementations
provided by `numpy`, as often experienced on macOS, try providing following environment variables:

.. code-block:: bash

    export LDFLAGS="-L/usr/local/opt/openblas/lib $LDFLAGS"
    export CPPFLAGS="-I/usr/local/opt/openblas/include $CPPFLAGS"
    export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig:$PKG_CONFIG_PATH"

    export LDFLAGS="-L/usr/local/opt/lapack/lib $LDFLAGS"
    export CPPFLAGS="-I/usr/local/opt/lapack/include $CPPFLAGS"
    export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig:$PKG_CONFIG_PATH"

where the paths have probably to be adapted to your particular installation method.
Here OpenBLAS_ and LAPACK_ was installed via Homebrew_.

Updating SurfaceTopography
--------------------------

If you update SurfaceTopography (whether with pip or `git pull` if you cloned the repository),  you may need to
uninstall `NuMPI`, `muSpectre` and or `runtests`, so that the newest version of them will be installed.

Singularity_ container
----------------------

We provide a definition file to build a singularity container `here <https://github.com/ContactEngineering/SurfaceTopography/blob/master/singularity/SurfaceTopography_serial.def>`_ .

.. _Singularity: https://sylabs.io/singularity/
.. _FFTW3: http://www.fftw.org/
.. _muFFT: https://gitlab.com/muspectre/muspectre.git
.. _nuMPI: https://github.com/IMTEK-Simulation/NuMPI.git
.. _runtests: https://github.com/bccp/runtests
.. _Homebrew: https://brew.sh/
.. _OpenBLAS: https://www.openblas.net/
.. _LAPACK: http://www.netlib.org/lapack/
.. _NetCDF: https://www.unidata.ucar.edu/software/netcdf/

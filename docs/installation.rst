Installation
============
We only support installation on recent mac and linux systems, not on windows.

You need Python 3,  NetCDF_ and FFTW3_ in order to install SurfaceTopography.

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

Tip: to install NetCDF on ubuntu, you can for example use

.. code-block:: bash

    sudo apt-get install libnetcdf-dev

See also our `singularity container <https://github.com/ContactEngineering/SurfaceTopography/blob/master/singularity/SurfaceTopography_serial.def>`_ for an example installation on ubuntu.

Tip: to install NetCDF on mac using Homebrew_,

.. code-block:: bash

    brew install netcdf

Installation: Common problems
-----------------------------

- Sometimes muFFT_ will not find the FFTW3 installation you expect.
  You can specify the directory where you installed FFTW3_
  by setting the environment variable `FFTWDIR` (e.g. to `$USER/.local`).

- If muFFT_ is unable to find the NetCDF libraries (the `FileIONetCDF` class
  is missing), then set the environment variables `NETCDFDIR` (for serial
  compile) or `PNETCDFDIR` (for parallel compiles, to e.g. `$USER/.local`).

- Sometimes the installation fails because muFFT_ attempts to compile with
  `MPI` support but not all necessary libraries are available.

- Note that if you do not install a tagged version of a dependency
  (e.g. because you install from the master branch via`git+` or
  from source using directly `setup.py`), pip will attempt to reinstall
  that dependency despite it is already installed. In that case you need to
  avoid using `pip install` and install `SurfaceTopography from the source
  directory using `Meson`.

Reporting installation problems
-------------------------------

1. Make sure that you carefully read all these instructions.
2. Try to find similar issues in our issues or forum discussions. 
3. Please open an issue or a discussion in the forum.

When reporting a problem, please provide us with following information: 

- your system configuration, 
- your python3 environment (output of `python3 -m pip list`)
- The output of the verbose installation e.g. `python3 -m pip install --verbose --global-option="--verbose"`

Installation from source directory
--------------------------------

SurfaceTopography can be installed by invoking

.. code-block:: bash

   python3 -m pip install [--user] .

or

.. code-block:: bash

   meson setup build
   cd build
   meson compile
   meson install

in the source directoy. The command line parameter `--user` is optional and
leads to a local installation in the current user's `$HOME/.local` directory.

Updating SurfaceTopography
--------------------------

If you update SurfaceTopography (whether with pip or `git pull` if you cloned the repository),  you may need to
uninstall `NuMPI`, `muFFT` and or `runtests`, so that the newest version of them will be installed.

Singularity_ container
----------------------

We provide a definition file to build a singularity container `here <https://github.com/ContactEngineering/SurfaceTopography/blob/master/singularity/SurfaceTopography_serial.def>`_ .

.. _Singularity: https://sylabs.io/singularity/
.. _FFTW3: http://www.fftw.org/
.. _muFFT: https://gitlab.com/muspectre/muspectre.git
.. _NuMPI: https://github.com/IMTEK-Simulation/NuMPI.git
.. _runtests: https://github.com/bccp/runtests
.. _Homebrew: https://brew.sh/
.. _NetCDF: https://www.unidata.ucar.edu/software/netcdf/

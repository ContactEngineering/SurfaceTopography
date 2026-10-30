# Installation

## From PyPI

```bash
pip install SurfaceTopography
```

SurfaceTopography requires Python 3.10 or later. Binary wheels are built for
Linux (x86_64, aarch64), macOS (arm64) and Windows (AMD64, ARM64); on other
platforms pip compiles the package from the source distribution (see
[below](#building-from-source)).

The default installation runs serially and does not need MPI. It installs
numpy, scipy, [NuMPI](https://github.com/muSpectre/NuMPI) and the libraries
used by the file readers (h5py, igor2, tifffile, olefile, openpyxl, zstandard
and others, see `pyproject.toml`).

## Optional dependencies

| Extra | Installs | Needed for |
|-------|----------|------------|
| `mpi` | NuMPI's `mpi` and `fft` extras (mpi4py, muGrid >= 1.6.0) and netCDF4 | [domain-decomposed topographies](topographies.md#mpi-parallel-topographies) and parallel NetCDF I/O under `mpirun` |
| `test` | pytest, pytest-cov | running the [test suite](testing.md) |

```bash
pip install SurfaceTopography[mpi]
```

!!! warning "Compile mpi4py, muGrid and netCDF4 against your MPI library"
    The wheels of mpi4py and muGrid on PyPI may be built against a different
    MPI library than the one your `mpirun` belongs to, or without MPI at all.
    Build them from source against the MPI library of your system:

    ```bash
    CC=mpicc CXX=mpicxx pip install --no-binary mpi4py --no-binary muGrid SurfaceTopography[mpi]
    ```

    Parallel NetCDF I/O additionally needs a netCDF4 built with parallel
    (PnetCDF or parallel HDF5) support. `muGrid.has_mpi` tells whether muGrid
    was compiled with MPI.

## Building from source

The package contains a C++17 extension module (autocorrelation, bearing area,
bicubic interpolation and patch finding) and is built with
[Meson](https://mesonbuild.com/) through meson-python. Eigen and pybind11 are
taken from the system if available and are otherwise downloaded by Meson
(`subprojects/*.wrap`). A C++17 compiler is required.

```bash
git clone https://github.com/ContactEngineering/SurfaceTopography.git
cd SurfaceTopography
pip install .
```

The version number is derived from git tags by
[DiscoverVersion](https://github.com/imtek-simulation/DiscoverVersion), so build from
a git checkout (or from the source distribution), not from a bare copy of the
source tree.

## Development installation

For an editable installation, install the build requirements and disable
build isolation, so that Meson can rebuild the extension module on import:

```bash
pip install meson-python meson ninja DiscoverVersion numpy pytest
pip install --no-build-isolation -e .
```

Meson's build configuration calls `python`, not `python3`. On systems without
a `python` executable (e.g. Debian without `python-is-python3`), work inside a
virtual environment.

## Reporting installation problems

When an installation fails, open an
[issue](https://github.com/ContactEngineering/SurfaceTopography/issues) with

- your operating system and compiler,
- your Python environment (output of `pip list`), and
- the output of the verbose installation, `pip install --verbose SurfaceTopography`.

## Building this documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

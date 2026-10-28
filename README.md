[![Build Status](https://github.com/ContactEngineering/SurfaceTopography/actions/workflows/test-code-functionality.yml/badge.svg)](https://github.com/ContactEngineering/SurfaceTopography/actions/workflows/test-code-functionality.yml)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ContactEngineering/SurfaceTopography)

SurfaceTopography
=================

*Read and analyze surface topographies with Python.* This code implements basic classes for handling uniform and
nonuniform surface topography data. It contains a rich set of import filters for experimental surface topography data.
Surface topographies can be easily analyzed using standard (rms height, power spectrum, ...) and some special purpose
(autocorrelation function, variable bandwidth analysis, ...) statistical techniques.

Quick Start
-----------

```python
from SurfaceTopography import read_topography

# Read a topography file (format is auto-detected)
topography = read_topography('surface.opd')

# Get basic properties
print(f"Grid size: {topography.nb_grid_pts}")
print(f"Physical size: {topography.physical_sizes}")
print(f"RMS height: {topography.Sq()}")

# Compute the power spectral density (PSD)
q, psd = topography.power_spectrum_from_profile()

# Compute the autocorrelation function (ACF)
r, acf = topography.autocorrelation_from_profile()

# Plot the topography
import matplotlib.pyplot as plt
plt.pcolormesh(*topography.positions_and_heights())
plt.colorbar(label='Height')
plt.show()
```

Supported File Formats
----------------------

SurfaceTopography reads more than 40 file formats of commercial surface profilometers and scanning probe microscopes (see the [full list](https://contactengineering.github.io/SurfaceTopography/formats/)):

| Format | Description |
|--------|-------------|
| `al3d` | Alicona AL3D files |
| `asc` | ASCII text files (generic) |
| `bcr` | BCR-STM files |
| `datx` | Zygo DATX files |
| `di` | Digital Instruments / Bruker AFM files |
| `ezd` | Nanosurf EZD files |
| `frt` | FRT profilometer files |
| `gwy` | Gwyddion GWY files |
| `h5` | HDF5 files |
| `hgt` | NASA SRTM HGT elevation files |
| `ibw` | Igor Pro binary wave files (Asylum Research) |
| `jpk` | JPK Instruments files |
| `lext` | Olympus LEXT files |
| `mat` | MATLAB .mat files |
| `mdt` | NT-MDT (Solver, Ntegra) AFM files |
| `metropro` | Zygo MetroPro files |
| `mi` | Agilent / Molecular Imaging files |
| `mitutoyo` | Mitutoyo surface roughness files |
| `mnt` | Digital Surf Mountains MNT files |
| `nc` | NetCDF files |
| `nmm` | Nanomeasuring machine (NMM) files |
| `nms` | Nanofocus NMS files |
| `npy` | NumPy binary files |
| `npz` | NumPy array archives |
| `oir` | Olympus OIR files |
| `opd` | Wyko OPD files |
| `opdx` | Bruker Dektak OPDx files |
| `os3d` | Digital Metrology OmniSurf3D files |
| `plu` | Sensofar PLU files |
| `plux` | Sensofar PLUX files |
| `poir` | Olympus POIR (zip) files |
| `ps` | Park Systems TIFF files |
| `qep` | Mahr MarWin QEP profile archives |
| `sdf` | ISO 25178-71 SDF files |
| `sur` | Digital Surf SUR files |
| `tmd` | TrueMap TMD files |
| `vk` | Keyence VK files |
| `wsxm` | WSxM files |
| `x3p` | ISO 25178-72 X3P files |
| `xyz` | XYZ point cloud files |
| `zmg` | KLA Zeta ZMG files |
| `zon` | Keyence ZON files |

If you use this code, please cite:
* [Jacobs, Junge, Pastewka, Surf. Topogr. Metrol. Prop. 5, 013001 (2017)](https://doi.org/10.1088/2051-672X/aa51f8)
* [Röttger et al., Surf. Topogr. Metrol. Prop. 10, 035032 (2022)](https://doi.org/10.1088/2051-672X/ac860a) 

Documentation
-------------

This README file contains a brief introduction into the code. The full documentation can be found at https://contactengineering.github.io/SurfaceTopography/.

Installation
------------

Detailed instructions for the installation are provided in the [documentation](https://contactengineering.github.io/SurfaceTopography/installation/).

The most basic configuration of SurfaceTopography installs all requirements autonomously: `python3 -m pip install SurfaceTopography`

If you need an editable install (e.g. for development purposes), make sure you disable build isolation:

    python3 -m pip install --no-build-isolation -e .

You need to manually install `ninja`, `meson-python`, `pytest`, `DiscoverVersion` and `numpy` before:

    python3 -m pip install ninja meson-python pytest DiscoverVersion numpy

MPI-parallel (domain-decomposed) topographies need the `mpi` extra, with mpi4py and muGrid compiled against your MPI library:

    CC=mpicc CXX=mpicxx python3 -m pip install --no-binary mpi4py --no-binary muGrid SurfaceTopography[mpi]

Funding
-------

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within projects [PA 2023/2](https://gepris.dfg.de/gepris/projekt/258153560) and [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).

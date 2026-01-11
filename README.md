[![Build Status](https://github.com/ContactEngineering/SurfaceTopography/actions/workflows/test-code-functionality.yml/badge.svg)](https://github.com/ContactEngineering/SurfaceTopography/actions/workflows/tests.yml)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ContactEngineering/SurfaceTopography)

SurfaceTopography
=================

*Read and analyze surface topographies with Python.* This code implements basic classes for handling uniform and
nonuniform surface topography data. It contains a rich set of import filters for experimental surface topography data.
Surface topographies can be easily analyzed using standard (rms height, power spectrum, ...) and some special purpose
(autocorrelation function, variable bandwidth analysis, ...) statistical techniques. 

If you use this code, please cite:
* [Jacobs, Junge, Pastewka, Surf. Topogr. Metrol. Prop. 1, 013001 (2017)](https://doi.org/10.1088/2051-672X/aa51f8)
* [Röttger el al., Surf. Topogr. Metrol. Prop. 10, 035032 (2022)](https://doi.org/10.1088/2051-672X/ac860a) 

Documentation
-------------

This README file contains a brief introduction into the code. The full documentation can be found at https://contactengineering.github.io/SurfaceTopography/.

Installation
------------

Detailed instruction for the installation are provided in the [documentation](https://contactengineering.github.io/SurfaceTopography/installation.html?highlight=installation). If you want to install from source
you need to run `git submodule update --init` after a fresh checkout.

The most basic configuration of SurfaceTopography installs all requirements autonomously `python3 -m pip install SurfaceTopgography`

If you need an editable install (e.g. for development purposes), make sure you disable build isolation:

    python3 -m pip install --no-build-isolation -e .

You need to manually install `ninja`, `meson-python`, `pytest`, `runtests`, `DiscoverVersion` and `numpy` before:

    python3 -m pip install ninja meson-python pytest runtests DiscoverVersion numpy

Funding
-------

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within projects [PA 2023/2](https://gepris.dfg.de/gepris/projekt/258153560) and [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).

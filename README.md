SurfaceTopography
=================

*Read and analyze surface topographies with Python.* This code implements basic classes for handling uniform and
nonuniform surface topography data. It contains a rich set of import filters for experimental surface topography data.
Surface topographies can be easily analyzed using standard (rms height, power spectrum, ...) and some special purpose
(autocorrelation function, variable bandwidth analysis, ...) statistical techniques. 

If you use this code, please cite:
* [Jacobs, Junge, Pastewka, Surf. Topogr. Metrol. Prop. 1, 013001 (2017)](https://doi.org/10.1088/2051-672X/aa51f8)
* [Röttger el al., Surf. Topogr. Metrol. Prop. 10, 035032 (2022)](https://doi.org/10.1088/2051-672X/ac860a) 

Build status
------------

The following badge should say _tests passing_. This means that all automated tests completed successfully for the master branch.

[![Build Status](https://github.com/ContactEngineering/SurfaceTopography/actions/workflows/tests.yml/badge.svg)](https://github.com/ContactEngineering/SurfaceTopography/actions/workflows/tests.yml)

Documentation
-------------

This README file contains a brief introduction into the code. The full documentation can be found at https://contactengineering.github.io/SurfaceTopography/.

Installation
------------

Installation is only possible on unix/linux systems and cumbersome on macOSX. 
For a quick start and if your OS is windows, consider running SurfaceTopography via our [Docker container]().

You first need to install requirements following our detailed instruction in the [documentation](https://contactengineering.github.io/SurfaceTopography/installation.html?highlight=installation).

Once the requirements are installed, 
quick install with: `python3 -m pip install SurfaceTopgography`

Running SurfaceTopography via a Docker container
------------------------------------------------

1. Install and start [Docker](https://www.docker.com/)
2. start one of our [docker images](https://hub.docker.com/r/imteksim/jupyterlab-surfacetopography/tags), e.g. via the commandline `docker run -d -p 8888:8888 imteksim/jupyterlab-surfacetopography:0.3.0`
3. The terminal output will prompt you how to open the jupyterlab server in your browser
4. Open a notebook with the kernel called `SurfaceTopography`

Funding
-------

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within projects [PA 2023/2](https://gepris.dfg.de/gepris/projekt/258153560) and [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).

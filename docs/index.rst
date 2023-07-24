.. PyCo documentation master file, created by
   sphinx-quickstart on Tue Nov 27 17:14:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SurfaceTopography's documentation!
=============================================

*Read and analyze surface topographies with Python.* This code implements basic classes for handling uniform and
nonuniform surface topography data. It contains a rich set of import filters for experimental surface topography data.
Surface topographies can be easily analyzed using standard (rms height, power spectrum, ...) and some special purpose
(autocorrelation function, variable bandwidth analysis, ...) statistical techniques.

If you use this code, please cite:

* Jacobs, Junge, Pastewka, Surf. Topogr. Metrol. Prop. 1, 013001 (2017)
* https://doi.org/10.1088/2051-672X/aa51f8

.. toctree::
   :maxdepth: 2
   :caption: Notes

   installation
   usage
   testing
   contributing



.. toctree::
   :maxdepth: 1
   :caption: Package Reference


   source/SurfaceTopography
   source/SurfaceTopography.Container
   source/SurfaceTopography.Generic
   source/SurfaceTopography.IO
   source/SurfaceTopography.Nonuniform
   source/SurfaceTopography.Support
   source/SurfaceTopography.Uniform
   source/SurfaceTopography.Models

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

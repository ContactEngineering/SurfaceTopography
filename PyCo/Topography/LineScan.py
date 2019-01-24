#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   LineScan.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   09 Dec 2018

@brief  Support for nonuniform topogography descriptions

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import abc

import numpy as np

from .HeightContainer import (AbstractHeightContainer, DecoratedTopography, UniformTopographyInterface,
                              NonuniformLineScanInterface)


class UniformLineScan(AbstractHeightContainer, UniformTopographyInterface):
    """
    Line scan that lives on a uniform one-dimensional grid.
    """

    def __init__(self, heights, size, periodic=False, info={}):
        """
        Parameters
        ----------
        profile : array_like
            Data containing the height information. Needs to be a
            one-dimensional array.
        size : tuple of floats
            Physical size of the topography map
        periodic : bool
            Flag setting the periodicity of the surface
        """
        if heights.ndim != 1:
            raise ValueError('Heights array must be one-dimensional.')

        super().__init__(info=info)

        # Automatically turn this into a masked array if there is data missing
        if np.sum(np.logical_not(np.isfinite(heights))) > 0:
            heights = np.ma.masked_where(np.logical_not(np.isfinite(heights)), heights)
        self._heights = heights
        self._size = size
        self._periodic = periodic

        # Register analysis functions
        from .Uniform.common import derivative
        from .Uniform.ScalarParameters import rms_height, rms_slope, rms_Laplacian
        from .Uniform.PowerSpectrum import power_spectrum_1D
        self.register_function('mean', lambda this: this.heights().mean())
        self.register_function('derivative', derivative)
        self.register_function('rms_height', rms_height)
        self.register_function('rms_slope', rms_slope)
        self.register_function('rms_curvature', rms_Laplacian)
        self.register_function('power_spectrum_1D', power_spectrum_1D)

        # Register pipeline functions
        from .Pipeline import ScaledUniformTopography, DetrendedUniformTopography
        self.register_function('scale', ScaledUniformTopography)
        self.register_function('detrend', DetrendedUniformTopography)

    def __getstate__(self):
        state = super().__getstate__(), self._heights, self._size, self._periodic
        return state

    def __setstate__(self, state):
        superstate, self._heights, self._size, self._periodic = state
        super().__setstate__(superstate)

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def size(self):
        return self._size,

    @size.setter
    def size(self, new_size):
        self._size = new_size

    @property
    def is_periodic(self):
        return self._periodic

    @property
    def is_uniform(self):
        return True

    # Implement uniform line scan interface

    @property
    def resolution(self):
        return len(self._heights),

    @property
    def pixel_size(self):
        return (s / r for s, r in zip(self.size, self.resolution))

    @property
    def area_per_pt(self):
        return self.pixel_size

    @property
    def has_undefined_data(self):
        return np.ma.getmask(self._heights) is not np.ma.nomask

    def positions(self):
        r, = self.resolution
        p, = self.pixel_size
        return np.arange(r) * p

    def heights(self):
        return self._heights

    def save(self, fname, compress=True):
        """ saves the topography as a NumpyTxtTopography. Warning: This only saves
            the profile; the size is not contained in the file
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.array())


class UniformlyInterpolatedLineScan(DecoratedTopography, UniformTopographyInterface):
    """
    Interpolate a topography onto a uniform grid.
    """

    def __init__(self, topography, nb_points, padding, info={}):
        """
        Parameters
        ----------
        topography : Topography
            Topography to interpolate.
        nb_points : int
            Number of equidistant grid points.
        padding : int
            Number of padding grid points, zeros appended to the data.
        """
        super().__init__(topography, info=info)
        self.nb_points = nb_points
        self.padding = padding

        # This is populated with functions from the nonuniform topography, but this is a uniform topography
        self._functions.clear()

        # Register analysis functions
        from .Uniform.common import derivative
        from .Uniform.ScalarParameters import rms_height, rms_slope, rms_Laplacian
        from .Uniform.PowerSpectrum import power_spectrum_1D
        self.register_function('mean', lambda this: this.heights().mean())
        self.register_function('derivative', derivative)
        self.register_function('rms_height', rms_height)
        self.register_function('rms_slope', rms_slope)
        self.register_function('rms_curvature', rms_Laplacian)
        self.register_function('power_spectrum_1D', power_spectrum_1D)

        # Register pipeline functions
        from .Pipeline import ScaledUniformTopography, DetrendedUniformTopography
        self.register_function('scale', ScaledUniformTopography)
        self.register_function('detrend', DetrendedUniformTopography)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.nb_points, self.padding
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.nb_points, self.padding = state
        super().__setstate__(superstate)

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def size(self):
        s, = self.parent_topography.size
        return s * (self.nb_points + self.padding) / self.nb_points,

    @property
    def is_periodic(self):
        return self.parent_topography.is_periodic

    @property
    def is_uniform(self):
        return True

    # Implement uniform line scan interface

    @property
    def resolution(self):
        """Return resolution, i.e. number of pixels, of the topography."""
        return self.nb_points + self.padding,

    @property
    def pixel_size(self):
        return (s / r for s, r in zip(self.size, self.resolution))

    @property
    def area_per_pt(self):
        return self.pixel_size

    @property
    def has_undefined_data(self):
        return False

    def positions(self):
        left, right = self.parent_topography.x_range
        size = right - left
        return np.linspace(left - size * self.padding / (2 * self.nb_points),
                           right + size * self.padding / (2 * self.nb_points),
                           self.nb_points + self.padding)

    def heights(self):
        """ Computes the rescaled profile.
        """
        x = self.positions()
        return np.interp(x, *self.parent_topography.positions_and_heights())


class NonuniformLineScan(AbstractHeightContainer, NonuniformLineScanInterface):
    """
    Nonuniform topography with point list consisting of static numpy arrays.
    """

    def __init__(self, x, y, info={}):
        super().__init__(info=info)
        self._x = x
        self._h = y

        # Register analysis functions
        from .Nonuniform.ScalarParameters import rms_height, rms_slope
        from .Nonuniform.PowerSpectrum import power_spectrum_1D
        self.register_function('mean', lambda this: np.trapz(this.heights(), this.positions()) / this.size[0])
        self.register_function('rms_height', rms_height)
        self.register_function('rms_slope', rms_slope)
        self.register_function('power_spectrum_1D', power_spectrum_1D)

        # Register pipeline functions
        from .Pipeline import ScaledNonuniformTopography, DetrendedNonuniformTopography
        self.register_function('scale', ScaledNonuniformTopography)
        self.register_function('detrend', DetrendedNonuniformTopography)
        self.register_function('interpolate', UniformlyInterpolatedLineScan)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._x, self._h
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._x, self._h = state
        super().__setstate__(superstate)

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def size(self):
        """Returns distance between maximum and minimum x-value."""
        return self._x[-1] - self._x[0],

    @property
    def is_periodic(self):
        # FIXME: Nonuniform scans are at present always nonperiodic, but it is possible to conceive situations where
        # this is not necessarily the case.
        return False

    @property
    def is_uniform(self):
        return False

    # Implement uniform line scan interface

    @property
    def x_range(self):
        return self._x[0], self._x[-1]

    def positions(self):
        return self._x

    def heights(self):
        return self._h

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   HeightContainer.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   09 Dec 2018

@brief  Support for uniform topogography descriptions

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

from .HeightContainer import AbstractHeightContainer, UniformTopographyInterface
from .Uniform.common import derivative
from .Uniform.ScalarParameters import rms_height, rms_slope, rms_Laplacian
from .Uniform.PowerSpectrum import power_spectrum_1D, power_spectrum_2D
from .Uniform.VariableBandwidth import checkerboard_tilt_correction
from .Pipeline import ScaledUniformTopography, DetrendedUniformTopography


class Topography(AbstractHeightContainer, UniformTopographyInterface):
    """
    Topography that lives on a uniform two-dimensional grid, i.e. a topography
    map.
    """

    def __init__(self, heights, size, periodic=False, info={}):
        """
        Parameters
        ----------
        profile : array_like
            Data containing the height information. Needs to be a
            two-dimensional array.
        size : tuple of floats
            Physical size of the topography map
        periodic : bool
            Flag setting the periodicity of the surface
        """
        if heights.ndim != 2:
            raise ValueError('Heights array must be two-dimensional.')

        super().__init__(info=info)

        # Automatically turn this into a masked array if there is data missing
        if np.sum(np.logical_not(np.isfinite(heights))) > 0:
            heights = np.ma.masked_where(np.logical_not(np.isfinite(heights)), heights)
        self._heights = heights
        self._size = size
        self._periodic = periodic

        # Register analysis functions
        self.register_function('mean', lambda this: this.heights().mean())
        self.register_function('derivative', derivative)
        self.register_function('rms_height', rms_height)
        self.register_function('rms_slope', rms_slope)
        self.register_function('rms_curvature', rms_Laplacian)
        self.register_function('power_spectrum_1D', power_spectrum_1D)
        self.register_function('power_spectrum_2D', power_spectrum_2D)
        self.register_function('checkerboard_tilt_correction', checkerboard_tilt_correction)

        # Register operations of the processing pipeline
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
        return 2

    @property
    def is_periodic(self):
        return self._periodic

    @property
    def is_uniform(self):
        return True

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_size):
        self._size = new_size

    @property
    def resolution(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._heights.shape

    # Implement topography interface

    @property
    def pixel_size(self):
        return np.asarray(self.size) / np.asarray(self.resolution)

    @property
    def area_per_pt(self):
        return np.prod(self.pixel_size)

    @property
    def has_undefined_data(self):
        return np.ma.getmask(self._heights) is not np.ma.nomask

    def positions(self):
        # FIXME: Write test for this method
        nx, ny = self.resolution
        sx, sy = self.size
        return np.meshgrid(np.arange(nx) * sx/nx, np.arange(ny) * sy/ny, indexing='ij')

    def heights(self):
        return self._heights

    def positions_and_heights(self):
        x, y = self.positions()
        return x, y, self.heights()

    def save(self, fname, compress=True):
        """ saves the topography as a NumpyTxtTopography. Warning: This only saves
            the profile; the size is not contained in the file
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.heights())

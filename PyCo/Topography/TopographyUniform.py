#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   TopographyBase.py

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

import numpy as np

from .TopographyBase import SizedTopography


class UniformTopography(SizedTopography):
    """
    Topography that lives on a uniform grid.
    """

    name = 'generic_geom'

    def __init__(self, resolution=None, size=None, unit=None, periodic=False):
        super().__init__(size=size, unit=unit)
        self._resolution = resolution
        self._periodic = periodic

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._resolution
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._resolution = state
        super().__setstate__(superstate)

    @property
    def is_periodic(self):
        return self._periodic

    @property
    def is_uniform(self):
        return True

    @property
    def pixel_size(self):
        return np.asarray(self.size) / np.asarray(self.resolution)

    @property
    def resolution(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._resolution

    @property
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        if self._size is None:
            return self._resolution
        else:
            return self._size

    shape = resolution

    @property
    def area_per_pt(self):
        if self.size is None:
            return 1
        return np.prod([s / r for s, r in zip(self.size, self.resolution)])


class UniformNumpyTopography(UniformTopography):
    """
    Topography from a static numpy array.
    """
    name = 'uniform_numpy_topography'

    def __init__(self, profile, size=None, unit=None, periodic=False):
        """
        Keyword Arguments:
        profile -- topography profile
        """

        # Automatically turn this into a masked array if there is data missing
        if np.sum(np.logical_not(np.isfinite(profile))) > 0:
            profile = np.ma.masked_where(np.logical_not(np.isfinite(profile)), profile)
        self.__h = profile
        super().__init__(resolution=self.__h.shape, size=size, unit=unit, periodic=periodic)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.__h
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.__h = state
        super().__setstate__(superstate)

    def array(self):
        """
        Returns array containing the topography data.
        """
        return self.__h

    @property
    def has_undefined_data(self):
        return np.ma.getmask(self.__h) is not np.ma.nomask

    def save(self, fname, compress=True):
        """ saves the topography as a NumpyTxtTopography. Warning: This only saves
            the profile; the size is not contained in the file
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.array())

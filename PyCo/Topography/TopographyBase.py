#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   TopographyBase.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Base class for geometric topogography descriptions

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


class Topography(object, metaclass=abc.ABCMeta):
    """
    Base class for topography geometries. These are used to define height
    profiles for contact problems or for spectral or other type of analysis.
    """

    name = 'topography'

    class Error(Exception):
        # pylint: disable=missing-docstring
        pass

    def __init__(self):
        self._info = {}

    def __getitem__(self, index):
        return self.array()[index]

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        return self._info

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        self._info = state

    @property
    def is_periodic(self):
        """ Returns whether the surface is periodic. """
        return False

    @property
    @abc.abstractmethod
    def is_uniform(self):
        """
        Returns whether data resides on a uniform grid. This has the
        implication, that the array() method return the topography data.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dim(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        raise NotImplementedError

    @size.setter
    @abc.abstractmethod
    def size(self, size):
        """ set the size of the topography """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def unit(self, ):
        """ Return unit """
        raise NotImplementedError

    @unit.setter
    @abc.abstractmethod
    def unit(self, unit):
        """ Set unit """
        raise NotImplementedError

    @property
    def info(self, ):
        """ Return info dictionary """
        return self._info

    @info.setter
    def info(self, info):
        """ Set info dictionary """
        self._info = info

    @property
    def grid_spacing(self):
        return np.array(self.size) / np.array(self.resolution)

    #@abc.abstractmethod - FIXME: make abstract again
    def array(self):
        """ Return topography data on a homogeneous grid. """
        raise NotImplementedError

    def points(self):
        """ Return topography data on an inhomogeneous grid as a list of points. """
        if self.is_uniform:
            return tuple(
                np.meshgrid(*(np.arange(r) * s / r for s, r in zip(self.size, self.resolution)), indexing='ij') +
                            [self.array()])
        else:
            raise NotImplementedError

    def derivative(self, n=1):
        if self.is_uniform:
            from .Uniform.ScalarParameters import _derivative
            return _derivative(self.array(), self.size, n)
        else:
            if n != 1:
                raise RuntimeError('Currently only first derivatives are supported for nonuniform topographies.')
            x, h = self.points()
            dh = np.diff(h)
            dx = np.diff(x)
            return dh / dx

    def rms_height(self, kind='Sq'):
        """
        RMS height fluctuation of the topography.

        Parameters
        ----------
        kind : str
            String specifying the kind of analysis to carry out. 'Sq' is
            areal RMS height and 'Rq' means is RMS height from line scans.

        Returns
        -------
        rms_height : float
            Scalar containing the RMS height
        """
        if kind == 'Sq' and self.dim == 1:
            raise RuntimeError('Areal rms height (Sq) can only be computed '
                               'for two-dimensional topographies.')
        if self.is_uniform:
            from .Uniform.ScalarParameters import rms_height
            return rms_height(self.array(), kind=kind)
        else:
            from .Nonuniform.ScalarParameters import rms_height
            return rms_height(*self.points())

    def rms_slope(self):
        """computes the rms height gradient fluctuation of the topography"""
        if self.is_uniform:
            from .Uniform.ScalarParameters import rms_slope
            return rms_slope(self.array(), size=self.size)
        else:
            from .Nonuniform.ScalarParameters import rms_slope
            return rms_slope(*self.points())

    def rms_curvature(self):
        """computes the rms curvature fluctuation of the topography"""
        if self.is_uniform:
            from .Uniform.ScalarParameters import rms_curvature
            return rms_curvature(self.array(), size=self.size)
        else:
            from .Nonuniform.ScalarParameters import rms_curvature
            return rms_curvature(*self.points())


class SizedTopography(Topography):
    """
    Topography that stores its own size.
    """

    def __init__(self, size=None, unit=None):
        super().__init__()
        self._size = size
        self._unit = unit

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._dim, self._size, self._unit
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._dim, self._size, self._unit = state
        super().__setstate__(superstate)

    @property
    def dim(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return len(self.resolution)

    @property
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._size

    @size.setter
    def size(self, size):
        """ set the size of the topography """
        if not hasattr(size, "__iter__"):
            size = (size,)
        else:
            size = tuple(size)
        if len(size) != self.dim:
            raise self.Error(
                ("The dimension of this topography is {}, you have specified an "
                 "incompatible size of dimension {} ({}).").format(self.dim, len(size), size))
        self._size = size

    @property
    def unit(self, ):
        """ Return unit """
        return self._unit

    @unit.setter
    def unit(self, unit):
        """ Set unit """
        self._unit = unit


class ChildTopography(Topography):
    """
    Base class of topographies with parent. Having a parent means that the
    data is owned by the parent, but the present class performs
    transformations on that data. This is a simple realization of a
    processing pipeline. Note that child topographies don't store their
    own size etc. but pass this information through to the parent.
    """
    name = "child_topography"

    def __init__(self, topography):
        """
        Arguments
        ---------
        topography : Topography
            The parent topography.
        """
        super().__init__()
        assert isinstance(topography, Topography)
        self.parent_topography = topography

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.parent_topography
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.parent_topography = state
        super().__setstate__(superstate)

    @property
    def is_periodic(self):
        return self.parent_topography.is_periodic

    @property
    def is_uniform(self):
        """ Stored on a uniform grid? """
        return self.parent_topography.is_uniform

    @property
    def dim(self):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.dim

    @property
    def resolution(self):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.resolution

    shape = resolution

    @property
    def size(self):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.size

    @size.setter
    def size(self, size):
        """ set the size of the topography"""
        self.parent_topography.size = size

    @property
    def unit(self):
        """ Return unit """
        return self.parent_topography.unit

    @unit.setter
    def unit(self, unit):
        """ Set unit """
        self.parent_topography.unit = unit

    @property
    def info(self):
        """ Return info dictionary """
        info = self.parent_topography.info.copy()
        info.update(self._info)
        return info

    @property
    def area_per_pt(self):
        return self.parent_topography.area_per_pt

    @property
    def pixel_size(self):
        return self.parent_topography.pixel_size

    @property
    def has_undefined_data(self):
        try:
            return self.parent_topography.has_undefined_data
        except:
            return False

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Topography.py

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
import warnings

import numpy as np

from .common import compute_derivative
from .Detrending import tilt_from_height, tilt_and_curvature
from .ScalarParameters import (rms_height, rms_slope, rms_curvature, rms_height_nonuniform, rms_slope_nonuniform,
                               rms_curvature_nonuniform)


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

    def __add__(self, other):
        return CompoundTopography(self, other)

    def __sub__(self, other):
        return CompoundTopography(self, -1. * other)

    def __mul__(self, other):
        return ScaledTopography(self, other)

    __rmul__ = __mul__

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

    #@abc.abstractmethod - FIXME: make abstract again
    def array(self):
        """ Return topography data on a homogeneous grid. """
        raise NotImplementedError

    #@abc.abstractmethod - FIXME: make abstract again
    def points(self):
        """ Return topography data on an inhomogeneous grid as a list of points. """
        raise NotImplementedError

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
        if self.is_uniform:
            return rms_height(self.array(), kind=kind)
        else:
            return rms_height_nonuniform(*self.points(), kind=kind)

    def rms_slope(self):
        """computes the rms height gradient fluctuation of the topography"""
        if self.is_uniform:
            return rms_slope(self.array(), size=self.size, dim=self.dim)
        else:
            return rms_slope_nonuniform(*self.points(), size=self.size, dim=self.dim)

    def rms_curvature(self):
        """computes the rms curvature fluctuation of the topography"""
        if self.is_uniform:
            return rms_curvature(self.array(), size=self.size, dim=self.dim)
        else:
            return rms_curvature_nonuniform(*self.points(), size=self.size, dim=self.dim)


class SizedTopography(Topography):
    """
    Topography that stores its own size.
    """

    def __init__(self, dim=None, size=None, unit=None):
        super().__init__()
        self._dim = dim
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
        return self._dim

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


class UniformTopography(SizedTopography):
    """
    Topography that lives on a uniform grid.
    """

    name = 'generic_geom'

    def __init__(self, resolution=None, dim=None, size=None, unit=None):
        super().__init__(dim=dim, size=size, unit=unit)
        self._resolution = resolution

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
        return False

    @property
    def is_uniform(self):
        return True

    def points(self):
        return tuple(np.meshgrid(*(np.arange(r)*s/r for s, r in zip(self.size, self.resolution)))+[self.array()])

    @property
    def pixel_size(self):
        return np.asarray(self.size) / np.asarray(self.resolution)

    @property
    def resolution(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._resolution

    shape = resolution

    @property
    def area_per_pt(self):
        if self.size is None:
            return 1
        return np.prod([s / r for s, r in zip(self.size, self.resolution)])


class NonuniformTopography(SizedTopography):
    """
    Base class for topographies that live on a non-uniform grid. Currently
    only supports line scans, i.e. one-dimensional topographies.
    """

    def __init__(self, size=None, unit=None):
        super().__init__(dim=2, size=size, unit=unit)

    @property
    def is_periodic(self):
        return False

    @property
    def is_uniform(self):
        return False


class ScaledTopography(ChildTopography):
    """ used when geometries are scaled
    """
    name = 'scaled_topography'

    def __init__(self, topography, coeff):
        """
        Keyword Arguments:
        topography  -- Topography to scale
        coeff -- Scaling factor
        """
        super().__init__(topography)
        self.coeff = float(coeff)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.coeff
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.coeff = state
        super().__setstate__(superstate)

    def array(self):
        """ Computes the combined profile.
        """
        return self.coeff * self.parent_topography.array()


class DetrendedTopography(ChildTopography):
    """ used when topography needs to be tilted
    """
    name = 'detrended_topography'

    def __init__(self, topography, detrend_mode='slope'):
        """
        Keyword Arguments:
        topography -- Topography to scale
        detrend_mode -- Possible keywords:
            'center': center the topography, no trend correction.
            'height': adjust slope such that rms height is minimized.
            'slope': adjust slope such that rms slope is minimized.
            'curvature': adjust slope and curvature such that rms height is
            minimized.
        """
        super().__init__(topography)
        assert isinstance(topography, Topography)
        self._detrend_mode = detrend_mode
        self._detrend()

    def _detrend(self):
        if self._detrend_mode is None or self._detrend_mode == 'center':
            self._coeffs = [-self.parent_topography.array().mean()]
        elif self._detrend_mode == 'height':
            self._coeffs = [-s for s in tilt_from_height(self.parent_topography)]
        elif self._detrend_mode == 'slope':
            try:
                sx, sy = self.parent_topography.size
            except:
                sx, sy = self.parent_topography.shape
            nx, ny = self.parent_topography.shape
            self._coeffs = [-s.mean() for s in compute_derivative(self.parent_topography)]
            slx, sly = self._coeffs
            self._coeffs += [-self.parent_topography[...].mean() - slx * sx * (nx - 1) / (2 * nx)
                             - sly * sy * (ny - 1) / (2 * ny)]
        elif self._detrend_mode == 'curvature':
            self._coeffs = [-s for s in tilt_and_curvature(self.parent_topography)]
        else:
            raise ValueError("Unknown detrend mode '{}'." \
                             .format(self._detrend_mode))

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._detrend_mode, self._coeffs
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._detrend_mode, self._coeffs = state
        super().__setstate__(superstate)

    @property
    def coeffs(self, ):
        return self._coeffs

    @property
    def detrend_mode(self, ):
        return self._detrend_mode

    @detrend_mode.setter
    def detrend_mode(self, detrend_mode):
        self._detrend_mode = detrend_mode
        self._detrend()

    def array(self):
        """ Computes the combined profile.
        """
        nx, ny = self.shape
        try:
            sx, sy = self.size
        except:
            sx, sy = nx, ny
        x = np.arange(nx).reshape(-1, 1) * sx / nx
        y = np.arange(ny).reshape(1, -1) * sy / ny
        if len(self._coeffs) == 1:
            h0, = self._coeffs
            return self.parent_topography.array() + h0
        elif len(self._coeffs) == 3:
            m, n, h0 = self._coeffs
            return self.parent_topography.array() + h0 + m * x + n * y
        else:
            m, n, mm, nn, mn, h0 = self._coeffs
            xx = x * x
            yy = y * y
            xy = x * y
            return self.parent_topography.array() + h0 + m * x + n * y + mm * xx + nn * yy + mn * xy

    def stringify_plane(self, fmt=lambda x: str(x)):
        str_coeffs = [fmt(x) for x in self._coeffs]
        if len(self._coeffs) == 1:
            h0, = str_coeffs
            return h0
        elif len(self._coeffs) == 3:
            return '{2} + {0} x + {1} y'.format(*str_coeffs)
        else:
            return '{5} + {0} x + {1} y + {2} x^2 + {3} y^2 + {4} xy'.format(*str_coeffs)


class TranslatedTopography(ChildTopography):
    """ used when geometries are translated
    """
    name = 'translated_topography'

    def __init__(self, topography, offset=(0, 0)):
        """
        Keyword Arguments:
        topography  -- Topography to translate
        offset -- Translation offset in number of grid points
        """
        super().__init__(topography)
        assert isinstance(topography, Topography)
        self._offset = offset

    @property
    def offset(self, ):
        return self._offset

    @offset.setter
    def offset(self, offset, offsety=None):
        if offsety is None:
            self.offset = offset
        else:
            self.offset = (offset, offsety)

    def array(self):
        """ Computes the translated profile.
        """
        offsetx, offsety = self.offset
        return np.roll(np.roll(self.parent_topography.array(), offsetx, axis=0), offsety, axis=1)


class CompoundTopography(SizedTopography):
    """ used when geometries are combined
    """
    name = 'compound_topography'

    def __init__(self, topography_a, topography_b):
        """ Behaves like a topography that is a sum of two Topographies
        Keyword Arguments:
        topography_a   -- first topography of the compound
        topography_b   -- second topography of the compound
        """
        super().__init__()

        def combined_val(prop_a, prop_b, propname):
            """
            topographies can have a fixed or dynamic, adaptive resolution (or other
            attributes). This function assures that -- if this function is
            called for two topographies with fixed resolutions -- the resolutions
            are identical
            Parameters:
            prop_a   -- field of one topography
            prop_b   -- field of other topography
            propname -- field identifier (for error messages only)
            """
            if prop_a is None:
                return prop_b
            else:
                if prop_b is not None:
                    assert prop_a == prop_b, \
                        "{} incompatible:{} <-> {}".format(
                            propname, prop_a, prop_b)
                return prop_a

        self._dim = combined_val(topography_a.dim, topography_b.dim, 'dim')
        self._resolution = combined_val(topography_a.resolution, topography_b.resolution, 'resolution')
        self._size = combined_val(topography_a.size, topography_b.size, 'size')
        self.parent_topography_a = topography_a
        self.parent_topography_b = topography_b

    def array(self):
        """ Computes the combined profile
        """
        return (self.parent_topography_a.array() +
                self.parent_topography_b.array())


class UniformNumpyTopography(UniformTopography):
    """
    Topography from a static numpy array.
    """
    name = 'uniform_numpy_topography'

    def __init__(self, profile, size=None, unit=None):
        """
        Keyword Arguments:
        profile -- topography profile
        """

        # Automatically turn this into a masked array if there is data missing
        if np.sum(np.logical_not(np.isfinite(profile))) > 0:
            profile = np.ma.masked_where(np.logical_not(np.isfinite(profile)), profile)
        self.__h = profile
        super().__init__(resolution=self.__h.shape, dim=len(self.__h.shape), size=size, unit=unit)

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


class NonuniformNumpyTopography(NonuniformTopography):
    """
    Nonunform topography with point list consisting of static numpy arrays.
    """

    def __init__(self, x, y, size=None, unit=None):
        super().__init__(size=size, unit=unit)
        self.__x = x
        self.__h = y

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.__x, self.__h
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.__x, self.__h = state
        super().__setstate__(superstate)

    def points(self):
        return self.__x, self.__h


# TODO: Turn into generator function
class Sphere(UniformNumpyTopography):
    """ Spherical topography. Corresponds to a cylinder in 2D
    """
    name = 'sphere'

    def __init__(self, radius, resolution, size, centre=None, standoff=0, periodic=False):
        """
        Simple shere geometry.
        Parameters:
        radius     -- self-explanatory
        resolution -- self-explanatory
        size       -- self-explanatory
        centre     -- specifies the coordinates (in length units, not pixels).
                      by default, the sphere is centred in the topography
        standoff   -- when using interaction forces with ranges of the order
                      the radius, you might want to set the topography outside of
                      the spere to far away, maybe even pay the price of inf,
                      if your interaction has no cutoff
        periodic   -- whether the sphere can wrap around. tricky for large
                      spheres
        """
        # pylint: disable=invalid-name
        if not hasattr(resolution, "__iter__"):
            resolution = (resolution,)
        dim = len(resolution)
        if not hasattr(size, "__iter__"):
            size = (size,)
        if centre is None:
            centre = np.array(size) * .5
        if not hasattr(centre, "__iter__"):
            centre = (centre,)

        if not periodic:
            def get_r(res, size, centre):
                " computes the non-periodic radii to evaluate"
                return np.linspace(-centre, size - centre, res, endpoint=False)
        else:
            def get_r(res, size, centre):
                " computes the periodic radii to evaluate"
                return np.linspace(-centre + size / 2,
                                   -centre + 3 * size / 2,
                                   res, endpoint=False) % size - size / 2

        if dim == 1:
            r2 = get_r(resolution[0], size[0], centre[0]) ** 2
        elif dim == 2:
            rx2 = (get_r(resolution[0], size[0], centre[0]) ** 2).reshape((-1, 1))
            ry2 = (get_r(resolution[1], size[1], centre[1])) ** 2
            r2 = rx2 + ry2
        else:
            raise Exception("Problem has to be 1- or 2-dimensional. Yours is {}-dimensional".format(dim))
        radius2 = radius ** 2  # avoid nans for small radiio
        outside = r2 > radius2
        r2[outside] = radius2
        h = np.sqrt(radius2 - r2) - radius
        h[outside] -= standoff
        super().__init__(h)
        self._size = size
        self._centre = centre

    @property
    def centre(self):
        "returns the coordinates of the sphere's (or cylinder)'s centre"
        return self._centre


class PlasticTopography(ChildTopography):
    """ Topography with an additional plastic deformation field.
    """
    name = 'plastic_topography'

    def __init__(self, topography, hardness, plastic_displ=None):
        """
        Keyword Arguments:
        topography -- topography profile
        hardness -- penetration hardness
        plastic_displ -- initial plastic displacements
        """
        super().__init__(topography)
        self.hardness = hardness
        if plastic_displ is None:
            plastic_displ = np.zeros(self.shape)
        self.plastic_displ = plastic_displ

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.hardness, self.plastic_displ
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.hardness, self.plastic_displ = state
        super().__setstate__(superstate)

    @property
    def hardness(self):
        return self._hardness

    @hardness.setter
    def hardness(self, hardness):
        if hardness <= 0:
            raise ValueError('Hardness must be positive.')
        self._hardness = hardness

    @property
    def plastic_displ(self):
        return self.__h_pl

    @plastic_displ.setter
    def plastic_displ(self, plastic_displ):
        if plastic_displ.shape != self.shape:
            raise ValueError('Resolution of profile and plastic displacement must match.')
        self.__h_pl = plastic_displ

    def undeformed_profile(self):
        """ Returns the undeformed profile of the topography.
        """
        return self.parent_topography.array()

    def array(self):
        """ Computes the combined profile.
        """
        return self.undeformed_profile() + self.plastic_displ

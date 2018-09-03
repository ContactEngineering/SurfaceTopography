#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Topography.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Base class for geometric descriptions

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
from .ScalarParameters import rms_height, rms_slope, rms_curvature


class Topography(object, metaclass=abc.ABCMeta):
    """ Base class for geometries. These are used to define height profiles for
         contact problems"""

    class Error(Exception):
        # pylint: disable=missing-docstring
        pass

    name = 'generic_geom'

    def __init__(self, resolution=None, dim=None, size=None, unit=None,
                 adjustment=0.):
        self._resolution = resolution
        self._dim = dim
        self._size = size
        self._unit = unit
        self._info = {}
        self.adjustment = adjustment

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = (self._resolution, self._dim, self._size, self._unit,
                 self.adjustment)
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        (self._resolution, self._dim, self._size, self._unit,
         self.adjustment) = state

    def rms_height(self, kind='Sq'):
        "computes the rms height fluctuation of the topography"
        return rms_height(self.array(), kind=kind)

    def rms_height_q_space(self):
        """
        computes the rms height fluctuation of the topography in the
        frequency domain
        """
        delta = self.array()
        delta -= delta.mean()
        area = np.prod(self.size)
        nb_pts = np.prod(self.resolution)
        H = area / nb_pts * np.fft.fftn(delta)
        return 1 / area * np.sqrt((np.conj(H) * H).sum().real)

    def rms_slope(self):
        "computes the rms height gradient fluctuation of the topography"
        return rms_slope(self.array(),
                         size=self.size, dim=self.dim)

    def rms_curvature(self):
        "computes the rms curvature fluctuation of the topography"
        return rms_curvature(self.array(),
                             size=self.size, dim=self.dim)

    def rms_slope_q_space(self):
        """
        taken from roughness in pycontact
        """
        # pylint: disable=invalid-name
        nx, ny = self.resolution
        sx, sy = self.size
        qx = np.arange(nx, dtype=np.float64)
        qx = np.where(qx <= nx / 2, 2 * np.pi * qx / sx, 2 * np.pi * (nx - qx) / sx)
        qy = np.arange(ny, dtype=np.float64)
        qy = np.where(qy <= ny / 2, 2 * np.pi * qy / sy, 2 * np.pi * (ny - qy) / sy)
        q = np.sqrt((qx * qx).reshape(-1, 1) + (qy * qy).reshape(1, -1))

        h_q = np.fft.fft2(self.array())
        return np.sqrt(
            np.mean(q ** 2 * h_q * np.conj(h_q)).real / (
                    float(self.array().shape[0]) * float(self.array().shape[1])))

    def adjust(self):
        """
        shifts topography up or down so that a zero displacement would lead to a
        zero gap
        """
        self.adjustment = self.array().max()

    def array(self):
        """ returns an array of possibly adjusted heights
        """
        return self._array() - self.adjustment

    @abc.abstractmethod
    def _array(self):
        """ returns an array of heights
        """
        raise NotImplementedError()

    def __add__(self, other):
        return CompoundTopography(self, other)

    def __sub__(self, other):
        return CompoundTopography(self, -1. * other)

    def __mul__(self, other):
        return ScaledTopography(self, other)

    __rmul__ = __mul__

    def __getitem__(self, index):
        return self._array()[index] - self.adjustment

    @property
    def dim(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._dim

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

    def set_size(self, size, s_y=None):
        """ Deprecated, do not use.
        set the size of the topography """
        warnings.warn('.set_size(x) is deprecated; please use .size = x',
                      DeprecationWarning)
        self.size = size

    @property
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._size

    @size.setter
    def size(self, size):
        """ set the size of the toporgraphy """
        if not hasattr(size, "__iter__"):
            size = (size,)
        else:
            size = tuple(size)
        if len(size) != self.dim:
            raise self.Error(
                ("The dimension of this topography is {}, you have specified an "
                 "incompatible size of dimension {} ({}).").format(
                    self.dim, len(size), size))
        self._size = size

    @property
    def unit(self, ):
        """ Return unit """
        return self._unit

    @unit.setter
    def unit(self, unit):
        """ Set unit """
        self._unit = unit

    @property
    def info(self, ):
        """ Return info dictionary """
        return self._info

    @info.setter
    def info(self, info):
        """ Set info dictionary """
        self._info = info

    @property
    def area_per_pt(self):
        if self.size is None:
            return 1
        return np.prod([s / r for s, r in zip(self.size, self.resolution)])

    @property
    def has_undefined_data(self):
        try:
            return self.parent_topography.has_undefined_data
        except:
            return False

    def save(self, fname, compress=True):
        """ saves the topography as a NumpyTxtTopography. Warning: This only saves
            the profile; the size is not contained in the file
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.array())

    def estimate_laplacian(self, coords):
        """
        estimate the local laplacian at coords by finite differences
        Keyword Arguments:
        coords --
        """
        laplacian = 0.
        for i in range(self.dim):
            pixel_size = self.size[i] / self.resolution[i]
            coord = coords[i]
            if coord == 0:
                delta = 1
            elif coord == self.resolution[i] - 1:
                delta = -1
            else:
                delta = 0
            irange = (coords[i] - 1 + delta, coords[i] + delta, coords[i] + 1 + delta)
            fun_val = np.zeros(len(irange))
            for j, i_val in enumerate(irange):
                coord_copy = list(coords)
                coord_copy[i] = i_val
                try:
                    fun_val[j] = self.array()[tuple(coord_copy)]
                except IndexError as err:
                    raise IndexError(
                        ("{}:\ncoords = {}, i = {}, j = {}, irange = {}, "
                         "coord_copy = {}").format(
                            err, coords, i, j, irange, coord_copy))  # nopep8
            laplacian += (fun_val[0] + fun_val[2] - 2 * fun_val[1]) / pixel_size ** 2
        return laplacian


class ScaledTopography(Topography):
    """ used when geometries are scaled
    """
    name = 'scaled_topography'

    def __init__(self, topography, coeff):
        """
        Keyword Arguments:
        topography  -- Topography to scale
        coeff -- Scaling factor
        """
        super().__init__()
        assert isinstance(topography, Topography)
        self.parent_topography = topography
        self.coeff = float(coeff)

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = (super().__getstate__(), self.parent_topography, self.coeff)
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.parent_topography, self.coeff = state
        super().__setstate__(superstate)

    @property
    def dim(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.dim

    @property
    def resolution(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.resolution

    shape = resolution

    @property
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.size

    @size.setter
    def size(self, size):
        """ set the size of the topography"""
        self.parent_topography.size = size

    @property
    def unit(self, ):
        """ Return unit """
        return self.parent_topography.unit

    @unit.setter
    def unit(self, unit):
        """ Set unit """
        self.parent_topography.unit = unit

    @property
    def info(self, ):
        """ Return info dictionary """
        return self.parent_topography.info

    @info.setter
    def info(self, info):
        """ Set info dictionary """
        self.parent_topography.info = info

    def _array(self):
        """ Computes the combined profile.
        """
        return self.coeff * self.parent_topography.array()


class DetrendedTopography(Topography):
    """ used when topography needs to be tilted
    """
    name = 'detrended_topography'

    def __init__(self, topography, detrend_mode='slope'):
        """
        Keyword Arguments:
        topography -- Topography to scale
        detrend_mode -- Possible keywords:
            'center': center the topography, no trent correction.
            'height': adjust slope such that rms height is minimized.
            'slope': adjust slope such that rms slope is minimized.
            'curvature': adjust slope and curvature such that rms height is
            minimized.
        """
        super().__init__()
        assert isinstance(topography, Topography)
        self.parent_topography = topography
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
        state = (super().__getstate__(), self.parent_topography, self._detrend_mode,
                 self._coeffs)
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        (superstate, self.parent_topography, self._detrend_mode, self._coeffs) = state
        super().__setstate__(superstate)

    @property
    def dim(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.dim

    @property
    def resolution(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.resolution

    shape = resolution

    @property
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.size

    @size.setter
    def size(self, size):
        """ set the size of the topography"""
        self.parent_topography.size = size
        self._detrend()

    @property
    def unit(self, ):
        """ Return unit """
        return self.parent_topography.unit

    @unit.setter
    def unit(self, unit):
        """ Set unit """
        self.parent_topography.unit = unit

    @property
    def info(self, ):
        """ Return info dictionary """
        return self.parent_topography.info

    @info.setter
    def info(self, info):
        """ Set info dictionary """
        self.parent_topography.info = info

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

    def _array(self):
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
            return '{5} + {0} x + {1} y + {2} x^2 + {3} y^2 + {4} xy' \
                .format(*str_coeffs)


class TranslatedTopography(Topography):
    """ used when geometries are translated
    """
    name = 'translated_topography'

    def __init__(self, topography, offset=(0, 0)):
        """
        Keyword Arguments:
        topography  -- Topography to translate
        offset -- Translation offset in number of grid points
        """
        super().__init__()
        assert isinstance(topography, Topography)
        self.parent_topography = topography
        self.offset = offset

    @property
    def dim(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.dim

    @property
    def resolution(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.resolution

    shape = resolution

    @property
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.size

    def set_offset(self, offset, offsety=None):
        if offsety is None:
            self.offset = offset
        else:
            self.offset = (offset, offsety)

    def _array(self):
        """ Computes the translated profile.
        """
        offsetx, offsety = self.offset
        return np.roll(np.roll(self.parent_topography.array(), offsetx, axis=0),
                       offsety, axis=1)


class CompoundTopography(Topography):
    """ used when geometries are combined
    """
    name = 'combined_topography'

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
        self._resolution = combined_val(topography_a.resolution,
                                        topography_b.resolution, 'resolution')
        self._size = combined_val(topography_a.size,
                                  topography_b.size, 'size')
        self.parent_topography_a = topography_a
        self.parent_topography_b = topography_b

    def _array(self):
        """ Computes the combined profile
        """
        return (self.parent_topography_a.array() +
                self.parent_topography_b.array())


class NumpyTopography(Topography):
    """ Dummy topography from a static array
    """
    name = 'topography_from_np_array'

    def __init__(self, profile, size=None, unit=None):
        """
        Keyword Arguments:
        profile -- topography profile
        """

        # Automatically turn this into a masked array if there is data missing
        if np.sum(np.logical_not(np.isfinite(profile))) > 0:
            profile = np.ma.masked_where(np.logical_not(np.isfinite(profile)),
                                         profile)
        self.__h = profile
        super().__init__(resolution=self.__h.shape, dim=len(self.__h.shape),
                         size=size, unit=unit)

    def _array(self):
        return self.__h

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = (super().__getstate__(), self.__h)
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        super().__setstate__(state[0])
        self.__h = state[1]

    @property
    def has_undefined_data(self):
        return np.ma.getmask(self.__h) is not np.ma.nomask


class Sphere(NumpyTopography):
    """ Spherical topography. Corresponds to a cylinder in 2D
    """
    name = 'sphere'

    def __init__(self, radius, resolution, size, centre=None, standoff=0,
                 periodic=False):
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
            r2 = get_r(resolution[0],
                       size[0],
                       centre[0]) ** 2
        elif dim == 2:
            rx2 = (get_r(resolution[0],
                         size[0],
                         centre[0]) ** 2).reshape((-1, 1))
            ry2 = (get_r(resolution[1],
                         size[1],
                         centre[1])) ** 2
            r2 = rx2 + ry2
        else:
            raise Exception(
                ("Problem has to be 1- or 2-dimensional. "
                 "Yours is {}-dimensional").format(dim))
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


class PlasticTopography(Topography):
    """ Topography with an additional plastic deformation field.
    """
    name = 'topography_with_plasticity'

    def __init__(self, topography, hardness, plastic_displ=None):
        """
        Keyword Arguments:
        topography -- topography profile
        hardness -- penetration hardness
        plastic_displ -- initial plastic displacements
        """
        self.parent_topography = topography
        self.hardness = hardness
        self.adjustment = 0.0
        if plastic_displ is None:
            plastic_displ = np.zeros(self.shape)
        self.plastic_displ = plastic_displ

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = (super().__getstate__(), self.parent_topography, self.hardness,
                 self.plastic_displ)
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.parent_topography, self.hardness, self.plastic_displ = state
        super().__setstate__(superstate)

    @property
    def dim(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.dim

    @property
    def resolution(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.resolution

    shape = resolution

    @property
    def size(self, ):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.size

    @size.setter
    def size(self, size):
        """ set the size of the topography """
        self.parent_topography.size = size

    @property
    def unit(self, ):
        """ Return unit """
        return self.parent_topography.unit

    @unit.setter
    def unit(self, unit):
        """ Set unit """
        self.parent_topography.unit = unit

    @property
    def info(self, ):
        """ Return info dictionary """
        return self.parent_topography.info

    @info.setter
    def info(self, info):
        """ Set info dictionary """
        self.parent_topography.info = info

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
            raise ValueError('Resolution of profile and plastic displacement '
                             'must match.')
        self.__h_pl = plastic_displ

    def undeformed_profile(self):
        """ Returns the undeformed profile of the topography.
        """
        return self.parent_topography.array()

    def _array(self):
        """ Computes the combined profile.
        """
        return self.undeformed_profile() + self.plastic_displ

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
from .ScalarParameters import (rms_height, rms_slope, rms_curvature,
                               rms_height_nonuniform, rms_slope_nonuniform,
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

    def __init__(self, pnp=None):
        self._info = {}
        self.pnp = np if pnp is None else pnp

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
        self.pnp = np  # TODO: DISCUSS: in the case of parallelized code the user will need to set the parallelnumpy by himself

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
    def is_MPI(self):
        """ stores only a subdomain of the data ? """
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

    @abc.abstractmethod
    def array(self):
        """ Return topography data on a homogeneous grid. """
        raise NotImplementedError

    # @abc.abstractmethod - FIXME: make abstract again
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
            return rms_height(self.array(), kind=kind,
                              resolution=self.resolution, pnp=self.pnp)
        else:
            return rms_height_nonuniform(*self.points(), kind=kind)

    def rms_slope(self):
        """computes the rms height gradient fluctuation of the topography"""
        if self.is_MPI: raise NotImplementedError

        if self.is_uniform:
            return rms_slope(self.array(), size=self.size, dim=self.dim)
        else:
            return rms_slope_nonuniform(*self.points(), size=self.size,
                                        dim=self.dim)

    def rms_curvature(self):
        """computes the rms curvature fluctuation of the topography"""
        if self.is_MPI: raise NotImplementedError
        if self.is_uniform:
            return rms_curvature(self.array(), size=self.size, dim=self.dim)
        else:
            return rms_curvature_nonuniform(*self.points(), size=self.size,
                                            dim=self.dim)


class SizedTopography(Topography):
    """
    Topography that stores its own size.
    """

    def __init__(self, dim=None, size=None, unit=None, pnp=None):
        super().__init__(pnp)
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
                (
                    "The dimension of this topography is {}, you have specified an "
                    "incompatible size of dimension {} ({}).").format(self.dim,
                                                                      len(size),
                                                                      size))
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
        super().__init__(topography.pnp)
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
    def is_MPI(self):
        """ stores only a subdomain of the data ? """
        return self.parent_topography.is_MPI

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

    def __init__(self, resolution=None, dim=None, size=None, unit=None,
                 subdomain_location=None, subdomain_resolution=None, pnp=None):
        super().__init__(dim=dim, size=size, unit=unit, pnp=pnp)
        self._resolution = resolution
        if subdomain_location is None or subdomain_resolution is None:
            if subdomain_resolution is not None and subdomain_resolution != resolution:
                raise ValueError(
                    "subdomain_resolution doesn't match resolution but no subdomain location given ")
            if subdomain_location is not None and subdomain_location != (0, 0):
                raise ValueError(
                    "subdomain_location != (0,0) but no subdomain resolution provided")
            self.subdomain_resolution = resolution
            self.subdomain_location = (0, 0)
        else:
            self.subdomain_location = subdomain_location
            self.subdomain_resolution = subdomain_resolution

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._resolution, self.subdomain_location, self.subdomain_resolution
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._resolution, self.subdomain_location, self.subdomain_resolution = state
        super().__setstate__(superstate)

    @property
    def is_periodic(self):
        return False

    @property
    def is_uniform(self):
        return True

    @property
    def is_MPI(self):
        return self.resolution != self.subdomain_resolution

    def points(self):
        return tuple(np.meshgrid(*(np.arange(r) * s / r for s, r in
                                   zip(self.size, self.resolution))) + [self.array()])

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

    @property
    def subdomain_slice(self):
        return tuple([slice(s, s + n) for s, n in
                      zip(self.subdomain_location, self.subdomain_resolution)])


class UniformNumpyTopography(UniformTopography):
    """
    Topography from a static numpy array.
    """
    name = 'uniform_numpy_topography'

    def __init__(self, heights, size=None, unit=None, resolution=None,
                 subdomain_location=None, subdomain_resolution=None, pnp=None,
                 periodic=False, info={}):
        """
        Keyword Arguments:
        profile -- local or global topography profile
        resolution -- resolution of global topography
        size --

        Three examples for the construction of the Object
        1.
        >>> UniformNumpyTopography((nx,ny) array[,size,unit])
        2.
        >>> UniformNumpyTopography((nx,ny) array, subdomain_location=(ix,iy), subdomain_resoluion=(sd_nx,sd_ny),pnp = ParallelNumpy instance, [,size=,unit=])
        3.
        >>> UniformNumpyTopography((sd_nx,sd_ny) array, resolution = (nx,ny), subdomain_location=(ix,iy),pnp = ParallelNumpy instance,  [,size,unit])
        """

        heights = np.asanyarray(heights)

        if heights.ndim != 2:
            raise ValueError('Heights array must be two-dimensional.')

        super().__init__(info=info)

        # Automatically turn this into a masked array if there is data missing
        if np.sum(np.logical_not(np.isfinite(heights))) > 0:
            heights = np.ma.masked_where(np.logical_not(np.isfinite(heights)), heights)


        if subdomain_location is not None:
            if subdomain_resolution is None:  # profile is local data
                # case 3. : no parallelization
                if resolution is None:
                    raise ValueError(
                        "Assuming you provided the local data as array, you should provide the global resolution")

                subdomain_resolution = heights.shape
                super().__init__(resolution=resolution, dim=len(heights.shape),
                                 size=size, unit=unit,
                                 subdomain_location=subdomain_location,
                                 subdomain_resolution=subdomain_resolution,
                                 pnp=pnp, info=info)


            else:  # global data provided
                # case 2. : no parallelization
                if resolution is None:
                    resolution = profile.shape
                elif resolution != profile.shape:
                    raise ValueError(
                        "Assuming you provided global data, resolution ({})  mismatch the shape of the data ({})".format(
                            resolution, profile.shape))

                super().__init__(resolution=resolution, dim=len(profile.shape),
                                 size=size, unit=unit,
                                 subdomain_location=subdomain_location,
                                 subdomain_resolution=subdomain_resolution,
                                 pnp=pnp)
                self.__h = profile[self.subdomain_slice]
        else:  # case 1. : no parallelization
            if not ((
                            resolution is None or resolution == profile.shape) and subdomain_location is None and subdomain_resolution is None):
                raise ValueError("invalid combination of arguments")
            resolution = subdomain_resolution = profile.shape
            super().__init__(resolution=resolution, dim=len(profile.shape),
                             size=size, unit=unit,
                             subdomain_location=subdomain_location,
                             subdomain_resolution=subdomain_resolution, pnp=pnp)
            self.__h = profile

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
        super().__setstate__(state[0])
        self.__h = state[1]

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


class NonuniformNumpyTopgraphy(NonuniformTopography):
    pass


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
            raise ValueError(
                'Resolution of profile and plastic displacement must match.')
        self.__h_pl = plastic_displ

    def undeformed_profile(self):
        """ Returns the undeformed profile of the topography.
        """
        return self.parent_topography.array()

    def array(self):
        """ Computes the combined profile.
        """
        return self.undeformed_profile() + self.plastic_displ

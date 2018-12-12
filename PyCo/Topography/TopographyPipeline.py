#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   TopographyPipeline.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Nov 2018

@brief  Processing pipeline for topographies

@section LICENCE

Copyright 2015-2018 Till Junge, Lars Pastewka

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

from .Uniform.Detrending import tilt_from_height, tilt_and_curvature
from .TopographyBase import ChildTopography, SizedTopography, Topography

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
        """ Computes the rescaled profile.
        """
        return self.coeff * self.parent_topography.array()

    def points(self):
        """ Computes the rescaled profile.
        """
        if self.dim == 1:
            x, h = self.parent_topography.points()
            return x, self.coeff * h
        else: # self.dim == 2
            x, y, h = self.parent_topography.points()
            return x, y, self.coeff * h

class DetrendedTopography(ChildTopography):
    """
    Remove trends from a topography. This is achieved by fitting polynomials
    to the topography data to extract trend lines. The resulting topography
    is then detrended by substracting these trend lines.
    """
    name = 'detrended_topography'

    def __init__(self, topography, detrend_mode='height'):
        """
        Parameters
        ----------
        topography : Topography
            Topography to be detrended.
        detrend_mode : str
            'center': center the topography, no trend correction.
            'height': adjust slope such that rms height is minimized.
            'slope': adjust slope such that rms slope is minimized.
            'curvature': adjust slope and curvature such that rms height is
            minimized. (Default: 'slope')
        """
        super().__init__(topography)
        assert isinstance(topography, Topography)
        self._detrend_mode = detrend_mode
        self._detrend()

    def _detrend(self):
        if self.dim == 1:
            if self._detrend_mode is None or self._detrend_mode == 'center':
                self._coeffs = (-self.parent_topography.mean(), )
            elif self._detrend_mode == 'height':
                from .Nonuniform.Detrending import polyfit
                self._coeffs = polyfit(*self.parent_topography.points(), 1)
            elif self._detrend_mode == 'curvature':
                from .Nonuniform.Detrending import polyfit
                self._coeffs = polyfit(*self.parent_topography.points(), 2)
            else:
                raise ValueError("Unsupported detrend mode '{}' for line scans." \
                                 .format(self._detrend_mode))
        else: # self.dim == 2
            if self._detrend_mode is None or self._detrend_mode == 'center':
                self._coeffs = [-self.parent_topography.mean()]
            elif self._detrend_mode == 'height':
                self._coeffs = [-s for s in tilt_from_height(self.parent_topography)]
            elif self._detrend_mode == 'slope':
                sx, sy = self.parent_topography.size
                nx, ny = self.parent_topography.shape
                self._coeffs = [-s.mean() for s in self.parent_topography.derivative()]
                slx, sly = self._coeffs
                self._coeffs += [-self.parent_topography[...].mean() - slx * sx * (nx - 1) / (2 * nx)
                                 - sly * sy * (ny - 1) / (2 * ny)]
            elif self._detrend_mode == 'curvature':
                self._coeffs = [-s for s in tilt_and_curvature(self.parent_topography)]
            else:
                raise ValueError("Unsupported detrend mode '{}' for 2D topographies." \
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
        if len(self._coeffs) == 1:
            a0, = self._coeffs
            return self.parent_topography.array() + a0
        elif self.dim == 1:
            x = np.arange(n) * self.size / self.shape[0]
            if len(self._coeffs) == 2:
                a0, a1 = self._coeffs
                return self.parent_topography.array() - a0 - a1 * x
            elif len(self._coeffs) == 3:
                a0, a1, a2 = self._coeffs
                return self.parent_topography.array() - a0 - a1 * x - a2 * x * x
            else:
                raise RuntimeError('Unknown size of coefficients tuple for line scans.')
        else: # self.dim == 2
            x, y = np.meshgrid(*(np.arange(n) * s / n for s, n in zip(self.size, self.shape)), indexing='ij')
            if len(self._coeffs) == 3:
                m, n, h0 = self._coeffs
                return self.parent_topography.array() + h0 + m * x + n * y
            elif len(self._coeffs) == 6:
                m, n, mm, nn, mn, h0 = self._coeffs
                xx = x * x
                yy = y * y
                xy = x * y
                return self.parent_topography.array() + h0 + m * x + n * y + mm * xx + nn * yy + mn * xy
            else:
                raise RuntimeError('Unknown size of coefficients tuple for 2D topographies.')

    def points(self):
        if self.dim == 1:
             x, h = self.parent_topography.points()
             if len(self._coeffs) == 1:
                  a0, = self._coeffs
                  return x, h + a0
             elif len(self._coeffs) == 2:
                  a0, a1 = self._coeffs
                  return x, h - a0 - a1 * x
             elif len(self._coeffs) == 3:
                 a0, a1, a2 = self._coeffs
                 return x, h - a0 - a1 * x - a2 * x * x
             else:
                 raise RuntimeError('Unknown size of coefficients tuple for line scans.')
        else: # self.dim == 2
            x, y, h = self.parent_topography.points()
            if len(self._coeffs) == 3:
                m, n, h0 = self._coeffs
                return x, y, h + h0 + m * x + n * y
            elif len(self._coeffs) == 6:
                m, n, mm, nn, mn, h0 = self._coeffs
                xx = x * x
                yy = y * y
                xy = x * y
                return x, y, h + h0 + m * x + n * y + mm * xx + nn * yy + mn * xy
            else:
                raise RuntimeError('Unknown size of coefficients tuple for 2D topographies.')

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

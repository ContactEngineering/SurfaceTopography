#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   TopographySpecial.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Nov 2018

@brief  Special topographies

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

from .TopographyBase import ChildTopography
from .TopographyUniform import UniformNumpyTopography


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

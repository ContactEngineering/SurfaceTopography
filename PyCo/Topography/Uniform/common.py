#
# Copyright 2018-2019 Lars Pastewka
#           2019 Antoine Sanner
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Bin for small common helper function and classes for uniform
topographies.
"""

import numpy as np

from ..HeightContainer import UniformTopographyInterface


def bandwidth(self):
    """Computes lower and upper bound of bandwidth.

    Returns
    -------
    A 2-tuple (lower_bound, upper_bound) where the elements are floats.
    """
    lower_bound = np.mean(self.pixel_size)
    upper_bound = np.mean(self.physical_sizes)

    return lower_bound, upper_bound


def derivative(topography, n, periodic=None):
    """
    Compute derivative of topography or line scan stored on a uniform grid.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography object containing height information.
    n : int
        Number of times the derivative is taken.
    periodic : bool
        Override periodic flag from topography.

    Returns
    -------
    derivative : array
        Array with derivative values. If dimension of the topography is
        unity (line scan), then an array of the same shape as the
        topography is returned. Otherwise, the first array index contains
        the direction of the derivative. If the topgography is nonperiodic,
        then all returning array with have shape one less than the input
        arrays.
    """
    if topography.physical_sizes is None:
        raise ValueError('Topography does not have physical size information, but this is required to be able to '
                         'compute a derivative.')
    grid_spacing = topography.pixel_size
    heights = topography.heights()
    is_periodic = topography.is_periodic if periodic is None else periodic
    if is_periodic:
        der = np.array([(np.roll(heights, -1, axis=d) - heights) / grid_spacing[d]
                        for d in range(len(heights.shape))])
        # Apply derivative into each direction multiple times
        for i in range(n-1):
            der = np.array([(np.roll(der[d], -1, axis=d) - der[d]) / grid_spacing[d]
                            for d in range(len(heights.shape))])
    else:
        der = np.array([np.diff(heights, n=n, axis=d) / grid_spacing[d] ** n
                        for d in range(len(heights.shape))])
    if der.shape[0] == 1:
        return der[0]
    else:
        return der


def domain_decompose(topography, subdomain_locations, nb_subdomain_grid_pts, communicator):
    """
    Turn a topography that is defined over the whole domain into one that is
    decomposed for each individual MPI process.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography object containing height information.
    subdomain_locations : tuple of ints
        Origin (location) of the subdomain handled by the present MPI
        process.
    nb_subdomain_grid_pts : tuple of ints
        Number of grid points within the subdomain handled by the present
        MPI process. This is only required if decomposition is set to
        'domain'.
    communicator : mpi4py communicator or NuMPI stub communicator
        Communicator object.

    Returns
    -------
    decomposed_topography : array
        Topography object that now holds only data local the MPI process.
    """
    return topography.__class__(topography.heights(), topography.physical_sizes,
                                periodic=topography.is_periodic,
                                decomposition='domain',
                                subdomain_locations=subdomain_locations,
                                nb_subdomain_grid_pts=nb_subdomain_grid_pts,
                                communicator=communicator)


### Register analysis functions from this module

UniformTopographyInterface.register_function('bandwidth', bandwidth)
UniformTopographyInterface.register_function('derivative', derivative)
UniformTopographyInterface.register_function('domain_decompose', domain_decompose)

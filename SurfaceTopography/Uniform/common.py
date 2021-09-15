#
# Copyright 2018-2021 Lars Pastewka
#           2019-2021 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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

from ..FFTTricks import make_fft
from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import Topography
from ..UniformLineScanAndTopography import DecoratedUniformTopography


def bandwidth(self):
    """
    Computes lower and upper bound of bandwidth, i.e. of the wavelengths or
    length scales occurring on a topography. The lower end of the bandwidth is
    given by the pixel size, the upper end by the physical dimension. For
    topographies with an aspect ratio that is not unity, this function returns
    the mean value of the two Cartesian directions.

    Returns
    -------
    lower_bound : float
        Lower bound of the bandwidth.
    upper_bound : float
        Upper bound of the bandwidth.
    """
    lower_bound = np.mean(self.pixel_size)
    upper_bound = np.mean(self.physical_sizes)

    return lower_bound, upper_bound


def domain_decompose(topography, subdomain_locations, nb_subdomain_grid_pts,
                     communicator):
    """
    Turn a topography that is defined over the whole domain into one that is
    decomposed for each individual MPI process.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.
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
        SurfaceTopography object that now holds only data local the MPI
        process.
    """
    return topography.__class__(topography.heights(),
                                topography.physical_sizes,
                                periodic=topography.is_periodic,
                                decomposition='domain',
                                subdomain_locations=subdomain_locations,
                                nb_subdomain_grid_pts=nb_subdomain_grid_pts,
                                communicator=communicator,
                                unit=topography.unit,
                                info=topography.info)


def plot(topography, subplot_location=111):
    """
    Plot an image of the topography using matplotlib.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography`
        Height information
    """
    # We import here because we don't want a global dependence on matplotlib
    import matplotlib.pyplot as plt

    try:
        sx, sy = topography.physical_sizes
    except TypeError:
        sx, sy = topography.nb_grid_pts
    nx, ny = topography.nb_grid_pts

    ax = plt.subplot(subplot_location, aspect=sx / sy)
    Y, X = np.meshgrid(np.arange(ny + 1) * sy / ny,
                       np.arange(nx + 1) * sx / nx)
    Z = topography[...]
    mesh = ax.pcolormesh(X, Y, Z)
    plt.colorbar(mesh, ax=ax)
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    if topography.unit is not None:
        unit = topography.unit
    else:
        unit = 'a.u.'
    ax.set_xlabel('Position $x$ ({})'.format(unit))
    ax.set_ylabel('Position $y$ ({})'.format(unit))
    return ax


class FilledTopography(DecoratedUniformTopography):
    def __init__(self, topography, fill_value=-np.infty, info={}):
        """
        masked (undefined) data is replaced with `fill_value`.

        Parameters
        ----------
        topography: Topography or UniformLineScan instance
        fill_value: float or array of floats
            masked value in topography will be replaced
        """
        super().__init__(topography, info=info)
        self.fill_value = fill_value

    def heights(self):
        return np.ma.filled(self.parent_topography.heights(),
                            fill_value=self.fill_value)

    @property
    def has_undefined_data(self):
        return False


# Register analysis functions from this module
UniformTopographyInterface.register_function('make_fft', make_fft)
UniformTopographyInterface.register_function('bandwidth', bandwidth)
UniformTopographyInterface.register_function('domain_decompose', domain_decompose)
Topography.register_function('plot', plot)
UniformTopographyInterface.register_function('fill_undefined_data', FilledTopography)

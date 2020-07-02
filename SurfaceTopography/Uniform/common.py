#
# Copyright 2018, 2020 Lars Pastewka
#           2019-2020 Antoine Sanner
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

from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import Topography
from ..UniformLineScanAndTopography import DecoratedUniformTopography


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
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.
    operator : muFFT.Derivative object or tuple of muFFT.Derivative objects
        Number of times the derivative is taken.
    periodic : bool
        Override periodic flag from topography.

    Returns
    -------
    derivative : array or tuple of arrays
        Array with derivative values. If dimension of the topography is
        unity (line scan), then an array of the same shape as the
        topography is returned. Otherwise, the first array index contains
        the direction of the derivative. If the topgography is nonperiodic,
        then all returning array with have shape one less than the input
        arrays.
    """
    if topography.physical_sizes is None:
        raise ValueError(
            'SurfaceTopography does not have physical size information, but '
            'this is required to be able to compute a derivative.')
    grid_spacing = topography.pixel_size
    heights = topography.heights()
    is_periodic = topography.is_periodic if periodic is None else periodic
    if is_periodic:
        der = np.array(
            [(np.roll(heights, -1, axis=d) - heights) / grid_spacing[d]
             for d in range(len(heights.shape))])
        # Apply derivative into each direction multiple times
        for i in range(n - 1):
            der = np.array(
                [(np.roll(der[d], -1, axis=d) - der[d]) / grid_spacing[d]
                 for d in range(len(heights.shape))])
    else:
        der = np.array([np.diff(heights, n=n, axis=d) / grid_spacing[d] ** n
                        for d in range(len(heights.shape))])
    if der.shape[0] == 1:
        return der[0]
    else:
        return der


def fourier_derivative(topography, imtol=1e-12):
    r"""

    First order derivatives of the fourier interpolation of the Topography

    For example in x direction:

    .. math::

        \left[ \frac{\partial h }{\partial x}\right]_{x_i, y_j} =
        \frac{1}{2 \pi }
        \sum_{kl} q_x^k \tilde h_{kl} e^{i (q_x^k x_i + q_y^l y_j}


    With :math:`\tilde h_{kl}` the DFT of the heights :math:``h
    For even number of points, the interpretation of the components at the
    Niquist frequency is ambiguous (we are free to choose amplitude or phase)
    Following (https://math.mit.edu/~stevenj/fft-deriv.pdf),
    we assume it is cosinusoidal.

    Parameters
    ----------
    topography: Topography instance
    imtol: float
        tolerance for the discarded imaginary part. If the maximum absolute of
        the imaginary part of the interpolated topography is more then that
        value, an AssertionError is raised

    Returns
    -------
    dx: array of floats
        derivative with respect to x
    dy: array of floats
        derivative with respect to y
    """
    nx, ny = topography.nb_grid_pts
    sx, sy = topography.physical_sizes

    qx = 2 * np.pi * np.fft.fftfreq(nx, sx / nx).reshape(-1, 1)
    qy = 2 * np.pi * np.fft.fftfreq(ny, sy / ny).reshape(1, -1)

    if nx % 2 == 0:
        qx[int(nx / 2), 0] = 0
    if ny % 2 == 0:
        qy[0, int(ny / 2)] = 0

    spectrum = np.fft.fft2(topography.heights())
    dx = np.fft.ifft2(spectrum * (1j * qx))
    dy = np.fft.ifft2(spectrum * (1j * qy))

    assert (abs(dx.imag) < imtol).all(), np.max(abs(dx.imag))
    assert (abs(dy.imag) < imtol).all(), np.max(abs(dy.imag))

    dx = dx.real
    dy = dy.real

    return dx, dy


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
    Y, X = np.meshgrid((np.arange(ny) + 0.5) * sy / ny,
                       (np.arange(nx) + 0.5) * sx / nx)
    Z = topography[...]
    mesh = ax.pcolormesh(X, Y, Z)
    plt.colorbar(mesh, ax=ax)
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    if 'unit' in topography.info:
        unit = topography.info['unit']
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
UniformTopographyInterface.register_function('bandwidth', bandwidth)
UniformTopographyInterface.register_function('derivative', derivative)
Topography.register_function('fourier_derivative', fourier_derivative)
UniformTopographyInterface.register_function('domain_decompose',
                                             domain_decompose)
Topography.register_function('plot', plot)
UniformTopographyInterface.register_function('fill_undefined_data',
                                             FilledTopography)

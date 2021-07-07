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

import muFFT

from ..FFTTricks import make_fft
from ..HeightContainer import UniformTopographyInterface
from ..UniformLineScanAndTopography import Topography
from ..UniformLineScanAndTopography import DecoratedUniformTopography

#
# Stencils for first and second derivatives
#

# First order upwind differences
first_1d = muFFT.DiscreteDerivative([0], [-1, 1])

# second order central differences of the second derivative
second_1d = muFFT.DiscreteDerivative([-1], [1, -2, 1])

# first order upwind differences of the third derivative
third_1d = muFFT.DiscreteDerivative([-1], [-1, 3, -3, 1])

# second order central differences of the third derivative
third_central_1d = muFFT.DiscreteDerivative([-2], [-1 / 2, 1, 0, -1, 1 / 2])

# First order upwind differences
first_2d_x = muFFT.DiscreteDerivative([0, 0], [[-1, 0], [1, 0]])
first_2d_y = muFFT.DiscreteDerivative([0, 0], [[-1, 1], [0, 0]])
first_2d = (first_2d_x, first_2d_y)

# second order central differences of the second derivative
second_2d_x = muFFT.DiscreteDerivative([-1, -1], [[0, 1, 0],
                                                  [0, -2, 0],
                                                  [0, 1, 0]])
second_2d_y = muFFT.DiscreteDerivative([-1, -1], [[0, 0, 0],
                                                  [1, -2, 1],
                                                  [0, 0, 0]])
second_2d = (second_2d_x, second_2d_y)

# first order upwind differences of the third derivative
third_2d_x = muFFT.DiscreteDerivative([-1, -1], [[0, -1, 0, 0],
                                                 [0, 3, 0, 0],
                                                 [0, -3, 0, 0],
                                                 [0, 1, 0, 0]])
third_2d_y = muFFT.DiscreteDerivative([-1, -1], [[0, 0, 0, 0],
                                                 [-1, 3, -3, 1],
                                                 [0, 0, 0, 0],
                                                 [0, 0, 0, 0]])

third_2d_x = muFFT.DiscreteDerivative([-1, -1], [[-1],
                                                 [3],
                                                 [-3],
                                                 [1]])
third_2d_y = muFFT.DiscreteDerivative([-1, -1], [[-1, 3, -3, 1]])

third_2d = (third_2d_x, third_2d_y)


def bandwidth(topography):
    """Computes lower and upper bound of bandwidth.

    Returns
    -------
    A 2-tuple (lower_bound, upper_bound) where the elements are floats.
    """
    lower_bound = np.mean(topography.pixel_size)
    upper_bound = np.mean(topography.physical_sizes)

    return lower_bound, upper_bound


def _get_default_derivative_operator(n, dim):
    """
    Return a default set of (finite-differences) derivative operators.

    Parameters
    ----------
    n : int
        Order of derivative.
    dim : int
        Dimension of the underlying field. (For dim > 1, this returns a
        representation of the nabla-operator.)
    """
    if dim == 1:
        if n == 1:
            return first_1d
        elif n == 2:
            return second_1d
        else:
            raise ValueError("Don't know how to compute derivative of order "
                             "{}.".format(n))
    elif dim == 2:
        if n == 1:
            return first_2d
        elif n == 2:
            return second_2d
        else:
            raise ValueError("Don't know how to compute derivative of order "
                             "{}.".format(n))
    else:
        raise ValueError("Don't know how to compute the derivative of a "
                         "topography of dimension {}.".format(dim))


def _trim_nonperiodic(arr, scale_factor, op):
    """
    Trim the outer edges of an array that contains derivatives computed under
    the assumption of periodicity. (These values at the outer edge will simply
    be wrong.)

    Parameters
    ----------
    arr : np.ndarray
        Array to be trimmed.
    scale_factor : int, optional
        Integer factor that scales the stencil difference.
    op : muFFT.DiscreteDerivative
        Derivative operator that contains information about the size of the
        stencil.
    """
    if not isinstance(op, muFFT.DiscreteDerivative):
        raise ValueError('Can only trim edges for discrete derivatives.')

    lbounds = np.array(op.lbounds)
    rbounds = lbounds + np.array(op.stencil.shape)

    trimmed_slice = []
    for L, r in zip(lbounds, rbounds):
        L = -scale_factor * min(L, 0)
        r = -scale_factor * max(0, r - 1)
        if r == 0:
            r = None
        trimmed_slice += [slice(int(L), None if r is None else int(r))]

    return arr[tuple(trimmed_slice)]


def derivative(topography, n, scale_factor=None, distance=None, operator=None, periodic=None, mask_function=None):
    """
    Compute derivative of topography or line scan stored on a uniform grid.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Surface topography object containing height information.
    n : int
        Order of the derivative.
    scale_factor : int or list of ints or list of tuples of ints, optional
        Integer factor that scales the stencil difference, i.e.
        specifying 2 will compute the derivative using a discrete step of
        2 * dx. Either `scale_factor` or `distance` can be specified.
        - Single int: Returns a single derivative scaled in all directions
          with this value
        - List of ints: Returns multiple derivatives, each scaled in all
          direction with the respective value from the list
        - List of tuples of ints: Each tuple contains a scale factor in
          the two Cartesian (x- and y-) directions. Return multiple
          derivatives, scaled with different factors in both directions.
        (Default: None)
    distance : float or list of floats, optional
        Explicit distance scale for computation of the derivative. Either
        `scale_factor` or `distance` can be specified. Note that the distance
        specifies the overall length of the stencil of lowest truncation
        order, not the effective grid spacing used by this stencil. The scale
        factor is then given by distance / (n * dx) where n is the order of the
        derivative and dx the grid spacing.
        (Default: None)
    operator : :obj:`muFFT.Derivative` object or tuple of :obj:`muFFT.Derivative` objects, optional
        Derivative operator used to compute the derivative. If unspecified,
        a simple upwind differences scheme will be applied to compute the
        derivative. A tuple contains the gradient, i.e. the derivative
        operators in the Cartesian directions for multidimensional fields.
        (Default: None)
    periodic : bool, optional
        Override periodic flag from topography. (Default: None)
    mask_function : function, optional
        A function that takes as argument the output of FFT.fftfreq and
        returns a mask that will be multiplied with the Fourier transformed
        topography. This can be used to implement Fourier filtering before
        computing the derivative. (Default: None)

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

    if operator is None:
        operator = _get_default_derivative_operator(n, topography.dim)

    def toiter(obj):
        """If object is scalar, wrap it into a list"""
        try:
            iter(obj)
            return obj
        except TypeError:
            return [obj]

    grid_spacing = np.array(topography.pixel_size)

    if scale_factor is None:
        if distance is None:
            # This is the default behavior
            scale_factor = 1
        else:
            # Convert distance to scale factor
            if topography.dim == 1:
                dx, = grid_spacing
                scale_factor = [d / (n * dx) for d in toiter(distance)]
            else:
                dx, dy = grid_spacing
                scale_factor = np.array([(d / (n * dx), d / (n * dy)) for d in toiter(distance)])

    elif distance is not None:
        raise ValueError('Please specify either `scale_factor` or `distance`')

    if np.any(np.asarray(scale_factor) < 1.0):
        raise ValueError('`scale_factor` cannot be smaller than unity. If you specified a specific `distance`, than '
                         'this distance is smaller than the lower bound of the bandwidth of the surface.')

    is_periodic = topography.is_periodic if periodic is None else periodic

    # Return FFT object (this will only be initialized once and reused in subsequent calls)
    fft = topography.make_fft()

    # These fields are reused when this function is called multiple times
    real_field = fft.fetch_or_register_real_space_field('real_temporary', 1)
    fourier_field = fft.fetch_or_register_fourier_space_field('complex_temporary', 1)
    np.array(real_field, copy=False)[...] = topography.heights()
    fft.fft(real_field, fourier_field)

    # Apply mask function
    if mask_function is not None:
        np.array(fourier_field, copy=False)[...] *= mask_function((fft.fftfreq.T / grid_spacing).T)

    fourier_array = np.array(fourier_field, copy=False)
    fourier_copy = fourier_array.copy()

    # Apply derivative operator in Fourier space
    derivatives = []
    for i, op in enumerate(toiter(operator)):
        der = []
        for s in toiter(scale_factor):
            try:
                # If s is a tuple, the scale factor is per direction
                s = s[i]
            except (TypeError, IndexError):  # Python types raise TypeError, numpy types IndexError
                pass
            fourier_array[...] = fourier_copy * op.fourier(fft.fftfreq * s)
            fft.ifft(fourier_field, real_field)
            _der = np.array(real_field, copy=False) * fft.normalisation / (s * grid_spacing[i]) ** n
            if not is_periodic:
                _der = _trim_nonperiodic(_der, s, op)

            try:
                iter(scale_factor)
                der += [_der]
            except TypeError:
                der = _der
        try:
            iter(operator)
            derivatives += [der]
        except TypeError:
            derivatives = der
    return derivatives


def fourier_derivative(topography, imtol=None):
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
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Surface topography object containing height information.
    imtol: float, optional
        tolerance for the discarded imaginary part. If the maximum absolute of
        the imaginary part of the interpolated topography is more then that
        value times the total absolute value of dx, an AssertionError is raised
        If not specified the assertion will not be made.
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

    if imtol is not None:
        assert (abs(dx.imag) / np.mean(abs(dx)) < imtol).all(), \
            np.max(abs(dx.imag) / np.mean(abs(dx)))
        assert (abs(dy.imag) / np.mean(abs(dy)) < imtol).all(), \
            np.max(abs(dy.imag) / np.mean(abs(dy)))

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
UniformTopographyInterface.register_function('derivative', derivative)
Topography.register_function('fourier_derivative', fourier_derivative)
UniformTopographyInterface.register_function('domain_decompose', domain_decompose)
Topography.register_function('plot', plot)
UniformTopographyInterface.register_function('fill_undefined_data', FilledTopography)

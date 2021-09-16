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

"""Compute derivatives of uniform line scans and topographies"""

import numpy as np

import muFFT

from ..HeightContainer import UniformTopographyInterface
from ..Support import toiter
from ..UniformLineScanAndTopography import Topography

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
third_2d_x = muFFT.DiscreteDerivative([-1, -1], [[-1],
                                                 [3],
                                                 [-3],
                                                 [1]])
third_2d_y = muFFT.DiscreteDerivative([-1, -1], [[-1, 3, -3, 1]])

third_2d = (third_2d_x, third_2d_y)


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
        elif n == 3:
            return third_1d
        else:
            raise ValueError("Don't know how to compute derivative of order "
                             "{}.".format(n))
    elif dim == 2:
        if n == 1:
            return first_2d
        elif n == 2:
            return second_2d
        elif n == 3:
            return third_2d
        else:
            raise ValueError("Don't know how to compute derivative of order "
                             "{}.".format(n))
    else:
        raise ValueError("Don't know how to compute the derivative of a "
                         "topography of dimension {}.".format(dim))


def trim_nonperiodic(arr, scale_factor, op):
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

    # Loop over dimension and add slicing information to `trimmed_slice`
    trimmed_slice = []
    for left, right, s in zip(lbounds, rbounds, scale_factor):
        # This is the leftmost distance in the stencil from the point where the derivative is computed
        # This value is always positive (or zero), i.e. it truncates the start of the array
        left = np.ceil(-s * min(left, 0))
        # This is the rightmost distance in the stencil from the point where the derivative is computed
        # This value is always negative (or zero), i.e. it truncates the end of the array
        right = np.floor(-s * max(0, right - 1))
        # If right is zero, we set it to None to indicate that nothing is truncated from the end of the array
        if right == 0:
            right = None
        trimmed_slice += [slice(int(left), None if right is None else int(right))]

    return arr[tuple(trimmed_slice)]


def derivative(self, n, scale_factor=None, distance=None, operator=None, periodic=None, mask_function=None,
               interpolation='linear', progress_callback=None):
    """
    Compute derivative of topography or line scan stored on a uniform grid.

    Parameters
    ----------
    self : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        Surface topography object containing height information.
    n : int
        Order of the derivative.
    scale_factor : int or list of ints or list of tuples of ints, optional
        Integer factor that scales the stencil difference, i.e.
        specifying 2 will compute the derivative using a discrete step of
        2 * px. Either `scale_factor` or `distance` can be specified.
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
        factor is then given by distance / (n * px) where n is the order of the
        derivative and px the grid spacing.
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
    interpolation : str, optional
        Interpolation method to use for fractional scale factors. Use
        'linear' for a local liner interpolation or 'fourier' for global
        Fourier interpolation. Set to 'disable' to raise an error when
        interpolation is necessary. Note that Fourier interpolation carries
        large errors for nonperiodic topographies and should be used with
        care. (Default: 'linear')
    progress_callback : func, optional
        Function taking iteration and the total number of iterations as
        arguments for progress reporting. (Default: None)

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
    if self.physical_sizes is None:
        raise ValueError(
            'SurfaceTopography does not have physical size information, but '
            'this is required to be able to compute a derivative.')

    if operator is None:
        operator = _get_default_derivative_operator(n, self.dim)

    if interpolation == 'linear':
        linear = self.interpolate_linear()
        positions = np.transpose(self.positions())
    elif interpolation == 'fourier' or interpolation == 'disable':
        linear = None
    else:
        raise ValueError("`interpolation` argument must be either 'linear', 'fourier' or 'disable'.")

    pixel_size = np.array(self.pixel_size)

    if scale_factor is None:
        if distance is None:
            # This is the default behavior
            scale_factor = 1
        else:
            # Convert distance to scale factor
            if self.dim == 1:
                px, = pixel_size
                try:
                    scale_factor = [d / (n * px) for d in distance]
                except TypeError:
                    scale_factor = distance / (n * px)
            else:
                px, py = pixel_size
                try:
                    scale_factor = np.array([(d / (n * px), d / (n * py)) for d in distance])
                except TypeError:
                    scale_factor = (distance / (n * px), distance / (n * py))
    elif distance is not None:
        raise ValueError('Please specify either `scale_factor` or `distance`')

    is_periodic = self.is_periodic if periodic is None else periodic

    # Return FFT object (this will only be initialized once and reused in subsequent calls)
    fft = self.make_fft()

    # These fields are reused when this function is called multiple times
    real_field = fft.fetch_or_register_real_space_field('real_temporary', 1)
    fourier_field = fft.fetch_or_register_fourier_space_field('complex_temporary', 1)
    np.array(real_field, copy=False)[...] = self.heights()
    fft.fft(real_field, fourier_field)

    # Apply mask function
    if mask_function is not None:
        np.array(fourier_field, copy=False)[...] *= mask_function((fft.fftfreq.T / pixel_size).T)

    fourier_array = np.array(fourier_field, copy=False)
    fourier_copy = fourier_array.copy()

    # Apply derivative operator in Fourier space
    derivatives = []
    operators = toiter(operator)
    scale_factors = toiter(scale_factor)
    for i, op in enumerate(operators):
        der = []
        for j, s in enumerate(scale_factors):
            if progress_callback is not None:
                progress_callback(i * len(scale_factors) + j, len(operators) * len(scale_factors))
            s = np.array(s) * np.ones_like(pixel_size)
            scaled_pixel_size = s * pixel_size
            interpolation_required = np.any(s - s.astype(int) != 0)
            if interpolation_required and interpolation == 'disabled':
                raise ValueError('Interpolation is required to compute derivative at the desired scale but is '
                                 'explicitly disabled through the `interpolation` argument.')
            if interpolation_required and linear is not None:
                # We need to interpolate using the linear interpolator
                lbounds = np.array(op.lbounds)
                stencil = np.array(op.stencil)
                _der = np.zeros_like(real_field)
                for stencil_coordinate, stencil_value in np.ndenumerate(stencil):
                    if stencil_value:
                        stencil_positions = positions + (lbounds + stencil_coordinate) * scaled_pixel_size
                        # We enforce periodicity here but will trim the (erroneous) boundary region below
                        if self.dim == 1:
                            _der += stencil_value * linear(stencil_positions, periodic=True)
                        else:
                            _der += stencil_value * linear(*stencil_positions.T, periodic=True)

            else:
                # We can use the Fourier trick to compute the derivative; this gives the exact stencil derivative if
                # interpolation is not required and the Fourier-interpolated derivative for fractional scale factors
                fourier_array[...] = fourier_copy * op.fourier((fft.fftfreq.T * s).T)
                fft.ifft(fourier_field, real_field)
                _der = np.array(real_field, copy=False) * fft.normalisation

            if not is_periodic:
                _der = trim_nonperiodic(_der, s, op)

            # We need to divide by the grid spacing to make this a derivative
            _der /= scaled_pixel_size[i] ** n

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
    if progress_callback is not None:
        progress_callback(len(operators) * len(scale_factors), len(operators) * len(scale_factors))
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


# Register analysis functions from this module
UniformTopographyInterface.register_function('derivative', derivative)
Topography.register_function('fourier_derivative', fourier_derivative)

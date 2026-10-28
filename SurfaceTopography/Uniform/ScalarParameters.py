#
# Copyright 2015-2016, 2018-2021, 2024 Lars Pastewka
#           2019, 2021 Michael Röttger
#           2018-2019 Antoine Sanner
#           2015-2016 Till Junge
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
Scalar roughness parameters (rms height, slope, curvature) of uniformly
sampled line scans and topography maps.
"""

import numpy as np
import scipy
from NuMPI.Tools import Reduction

from ..HeightContainer import UniformTopographyInterface

_mad_to_rms = 1 / scipy.stats.norm.ppf(3 / 4)


def Rq(topography):
    r"""
    Compute the root mean square height amplitude of a topography or
    line scan stored on a uniform grid from individual profiles.
    (This is the Rq value.)

    Each profile (line along x at fixed y) is referenced to its own mean,
    and the squared deviations of all profiles are averaged:

    .. math::

        R_q = \left[\frac{1}{N} \sum_{ij} \left(h_{ij} - \bar h_j\right)^2\right]^{1/2},
        \quad \bar h_j = \frac{1}{N_x} \sum_i h_{ij},

    where :math:`N` is the number of defined (unmasked) data points. For a
    line scan, this is the rms height about the mean.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography or line scan containing height information.

    Returns
    -------
    rms_height : float
        Root mean square height value. (Units: length)

    Raises
    ------
    NotImplementedError
        For domain-decomposed (MPI-parallel) topographies.
    """
    if topography.is_domain_decomposed:
        raise NotImplementedError("`Rq` does not support MPI-decomposed topographies.")

    # Note: all means and sums must be normalized by the number of *defined*
    # (unmasked) data points; masked points do not contribute to the sums
    profile = topography.heights()
    return np.sqrt(
        np.ma.sum((profile - np.ma.mean(profile, axis=0)) ** 2) / np.ma.count(profile)
    )


def Sq(topography):
    r"""
    Compute the root mean square height amplitude of a topography map
    stored on a uniform grid from the whole areal data.
    (This is the Sq value.)

    .. math::

        S_q = \left[\frac{1}{N} \sum_{ij} \left(h_{ij} - \langle h \rangle\right)^2\right]^{1/2}

    where :math:`N` is the number of defined (unmasked) data points and
    :math:`\langle h \rangle` their mean. Sums are reduced over all MPI
    processes, i.e. this function supports domain-decomposed topographies.

    Parameters
    ----------
    topography : :obj:`Topography`
        Topography map containing height information.

    Returns
    -------
    rms_height : float
        Root mean square height value. (Units: length)
    """
    if topography.dim <= 1:
        raise ValueError(
            "Areal rms height can only be computed for topographies, not line scans."
        )
    elif topography.dim == 2:
        # Normalize by the number of *defined* (unmasked) data points; sums
        # run over all subdomains of a domain-decomposed topography
        reduction = Reduction(topography._communicator)
        heights = np.ma.compressed(topography.heights())
        n = reduction.sum(len(heights))
        mean = reduction.sum(np.sum(heights)) / n
        return np.sqrt(reduction.sum(np.sum((heights - mean) ** 2)) / n)
    else:
        raise ValueError(f"Cannot handle topographies of dimension {topography.dim}")


def rms_gradient(topography, short_wavelength_cutoff=None, window=None, direction=None):
    r"""
    Compute the root mean square amplitude of the height gradient of a
    topography stored on a uniform grid. The topography must be
    two-dimensional (i.e. a topography map).

    .. math::

        h^\prime_\text{rms} = \left\langle \left(\frac{\partial h}{\partial x}\right)^2
        + \left(\frac{\partial h}{\partial y}\right)^2 \right\rangle^{1/2}

    The derivatives are computed with the default (upwind) finite-differences
    stencils of :func:`derivative`. For an isotropic surface, this is
    :math:`\sqrt{2}` times the rms slope :func:`Rdq` along a single direction.

    Parameters
    ----------
    topography : :obj:`Topography`
        Topography map containing height information.
    short_wavelength_cutoff : float, optional
        All wavelengths below this cutoff will be set to zero amplitude.
        If given, the topography is windowed, the derivative is computed in
        Fourier space and Fourier components with
        :math:`|\mathbf{q}| \geq 2\pi/\text{cutoff}` are discarded.
        (Units: length, Default: None)
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        Only used if short wavelength cutoff is set.
        (Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies)
    direction : str, optional
        Direction in which the window is applied. Possible options are
        'x', 'y' and 'radial'. If set to None, it chooses 'x' for line
        scans and 'radial' for topographies. Only used if short wavelength
        cutoff is set. (Default: None)

    Returns
    -------
    rms_gradient : float
        Root mean square gradient value. (Units: dimensionless)
    """
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    if topography.dim <= 1:
        raise ValueError(
            "RMS gradient can only be computed for topographies, not line scans."
        )
    elif topography.dim == 2:
        mask_function = (
            None
            if short_wavelength_cutoff is None
            else lambda frequency: (frequency[0] ** 2 + frequency[1] ** 2)
            < 1 / short_wavelength_cutoff**2
        )
        slx, sly = topography.derivative(
            1,
            mask_function=mask_function,
            interpolation="linear" if mask_function is None else "fourier",
        )
        reduction = Reduction(topography._communicator)
        return np.sqrt(reduction.mean(slx**2 + sly**2))
    else:
        raise ValueError(f"Cannot handle topographies of dimension {topography.dim}")


def Rdq(topography, short_wavelength_cutoff=None, window=None, direction=None):
    r"""
    Compute the root mean square amplitude of the height derivative of a
    topography or line scan stored on a uniform grid. If the topography is two
    dimensional (i.e. a topography map), the derivative is computed in the
    x-direction.

    .. math::

        R_{\Delta q} = \left\langle \left(\frac{h_{i+1,j} - h_{ij}}{\Delta x}\right)^2 \right\rangle^{1/2}

    The average runs over all pairs of neighboring points along x; for
    periodic data, this includes the pair that wraps around the boundary.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography or line scan containing height information.
    short_wavelength_cutoff : float, optional
        All wavelengths below this cutoff will be set to zero amplitude.
        If given, the topography is windowed, the derivative is computed in
        Fourier space and Fourier components with
        :math:`|\mathbf{q}| \geq 2\pi/\text{cutoff}` are discarded.
        (Units: length, Default: None)
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        Only used if short wavelength cutoff is set.
        (Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies)
    direction : str, optional
        Direction in which the window is applied. Possible options are
        'x', 'y' and 'radial'. If set to None, it chooses 'x' for line
        scans and 'radial' for topographies. Only used if short wavelength
        cutoff is set. (Default: None)

    Returns
    -------
    rms_slope : float
        Root mean square slope value. (Units: dimensionless)
    """
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    mask_function = (
        None
        if short_wavelength_cutoff is None
        else lambda frequency: frequency[0] ** 2 < 1 / short_wavelength_cutoff**2
    )
    if topography.dim == 1:
        dx = topography.derivative(
            1,
            mask_function=mask_function,
            interpolation="linear" if mask_function is None else "fourier",
        )
    elif topography.dim == 2:
        dx, dy = topography.derivative(
            1,
            mask_function=mask_function,
            interpolation="linear" if mask_function is None else "fourier",
        )
    else:
        raise ValueError(f"Cannot handle topographies of dimension {topography.dim}")
    reduction = Reduction(topography._communicator)
    return np.sqrt(reduction.mean(dx**2))


def Rddq(topography, short_wavelength_cutoff=None, window=None, direction=None):
    r"""
    Compute the root mean square amplitude of the second derivative (i.e. the
    curvature) of a topography or line scan stored on a uniform grid. If the
    topography is two-dimensional (i.e. a topography map), then the rms curvature
    is computed only along the x-direction.

    .. math::

        R_{\Delta\Delta q} = \left\langle \left(
        \frac{h_{i+1,j} - 2h_{ij} + h_{i-1,j}}{\Delta x^2}\right)^2 \right\rangle^{1/2}

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography or line scan containing height information.
    short_wavelength_cutoff : float, optional
        All wavelengths below this cutoff will be set to zero amplitude.
        If given, the topography is windowed, the derivative is computed in
        Fourier space and Fourier components with
        :math:`|\mathbf{q}| \geq 2\pi/\text{cutoff}` are discarded.
        (Units: length, Default: None)
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        Only used if short wavelength cutoff is set.
        (Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies)
    direction : str, optional
        Direction in which the window is applied. Possible options are
        'x', 'y' and 'radial'. If set to None, it chooses 'x' for line
        scans and 'radial' for topographies. Only used if short wavelength
        cutoff is set. (Default: None)

    Returns
    -------
    rms_curvature : float
        Root mean square curvature value. (Units: 1/length)
    """
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    mask_function = (
        None
        if short_wavelength_cutoff is None
        else lambda frequency: frequency[0] ** 2 < 1 / short_wavelength_cutoff**2
    )
    if topography.dim == 1:
        d2x = topography.derivative(
            2,
            mask_function=mask_function,
            interpolation="linear" if mask_function is None else "fourier",
        )
    elif topography.dim == 2:
        d2x, d2y = topography.derivative(
            2,
            mask_function=mask_function,
            interpolation="linear" if mask_function is None else "fourier",
        )
    else:
        raise ValueError(f"Cannot handle topographies of dimension {topography.dim}")
    reduction = Reduction(topography._communicator)
    return np.sqrt(reduction.mean(d2x**2))


def rms_laplacian(
    topography, short_wavelength_cutoff=None, window=None, direction=None
):
    r"""
    Compute the root mean square amplitude of the Laplacian (i.e. the sum of
    second derivatives in x- and y-directions) of a topography on a uniform
    grid. The topography must be two-dimensional (i.e. a topography map).

    .. math::

        \left\langle \left(\nabla^2 h\right)^2 \right\rangle^{1/2}, \quad
        \nabla^2 h \approx \frac{h_{i+1,j} - 2h_{ij} + h_{i-1,j}}{\Delta x^2}
        + \frac{h_{i,j+1} - 2h_{ij} + h_{i,j-1}}{\Delta y^2}

    Parameters
    ----------
    topography : :obj:`Topography`
        Topography map containing height information.
    short_wavelength_cutoff : float, optional
        All wavelengths below this cutoff will be set to zero amplitude.
        If given, the topography is windowed, the derivative is computed in
        Fourier space and Fourier components with
        :math:`|\mathbf{q}| \geq 2\pi/\text{cutoff}` are discarded.
        (Units: length, Default: None)
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        Only used if short wavelength cutoff is set.
        (Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies)
    direction : str, optional
        Direction in which the window is applied. Possible options are
        'x', 'y' and 'radial'. If set to None, it chooses 'x' for line
        scans and 'radial' for topographies. Only used if short wavelength
        cutoff is set. (Default: None)

    Returns
    -------
    rms_laplacian : float
        Root mean square Laplacian value. (Units: 1/length)

    Raises
    ------
    NotImplementedError
        If the topography lives on a communicator with more than one MPI
        process (domain-decomposed topographies and pipeline steps on top
        of them).
    """
    # The derivatives are not computed correctly in parallel. Check the
    # communicator rather than `is_domain_decomposed`: pipeline steps on a
    # decomposed topography report no decomposition but share its communicator.
    if topography.communicator.Get_size() > 1:
        raise NotImplementedError(
            "`rms_laplacian` and `rms_curvature_from_area` do not support "
            "MPI-parallel (domain-decomposed) topographies."
        )
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    if topography.dim == 1:
        raise ValueError(
            "RMS Laplacian can only be computed for topographies, not line scans."
        )
    elif topography.dim == 2:
        mask_function = (
            None
            if short_wavelength_cutoff is None
            else lambda frequency: (frequency[0] ** 2 + frequency[1] ** 2)
            < 1 / short_wavelength_cutoff**2
        )
        curv = topography.derivative(
            2,
            mask_function=mask_function,
            interpolation="linear" if mask_function is None else "fourier",
        )
        reduction = Reduction(topography._communicator)
        return np.sqrt(reduction.mean((curv[0] + curv[1]) ** 2))
    else:
        raise ValueError(f"Cannot handle topographies of dimension {topography.dim}")


def rms_curvature_from_area(
    topography, short_wavelength_cutoff=None, window=None, direction=None
):
    r"""
    Compute the root mean square amplitude of the curvature of a
    topography stored on a uniform grid. The topography must be
    two-dimensional (i.e. a topography map). This function returns
    half the rms Laplacian, i.e. the rms of the mean curvature
    :math:`\kappa = (\kappa_1 + \kappa_2)/2 \approx \nabla^2 h / 2`
    in the small-slope approximation (see :func:`rms_laplacian`).

    Parameters
    ----------
    topography : :obj:`Topography`
        Topography map containing height information.
    short_wavelength_cutoff : float, optional
        All wavelengths below this cutoff will be set to zero amplitude.
        If given, the topography is windowed, the derivative is computed in
        Fourier space and Fourier components with
        :math:`|\mathbf{q}| \geq 2\pi/\text{cutoff}` are discarded.
        (Units: length, Default: None)
    window : str, optional
        Window for eliminating edge effect. See scipy.signal.get_window.
        Only used if short wavelength cutoff is set.
        (Default: no window for periodic Topographies, "hann" window for
        nonperiodic Topographies)
    direction : str, optional
        Direction in which the window is applied. Possible options are
        'x', 'y' and 'radial'. If set to None, it chooses 'x' for line
        scans and 'radial' for topographies. Only used if short wavelength
        cutoff is set. (Default: None)

    Returns
    -------
    rms_curvature : float
        Root mean square curvature value. (Units: 1/length)

    Raises
    ------
    NotImplementedError
        If the topography lives on a communicator with more than one MPI
        process.
    """
    return (
        rms_laplacian(
            topography,
            short_wavelength_cutoff=short_wavelength_cutoff,
            window=window,
            direction=direction,
        )
        / 2
    )


# Register analysis functions from this module
UniformTopographyInterface.register_function(
    "rms_height_from_profile", Rq, deprecated=True
)
UniformTopographyInterface.register_function("Rq", Rq)
UniformTopographyInterface.register_function(
    "rms_height_from_area", Sq, deprecated=True
)
UniformTopographyInterface.register_function("Sq", Sq)
UniformTopographyInterface.register_function("rms_gradient", rms_gradient)
UniformTopographyInterface.register_function(
    "rms_slope_from_profile", Rdq, deprecated=True
)
UniformTopographyInterface.register_function("Rdq", Rdq)
UniformTopographyInterface.register_function(
    "rms_curvature_from_profile", Rddq, deprecated=True
)
UniformTopographyInterface.register_function("Rddq", Rddq)
UniformTopographyInterface.register_function("rms_laplacian", rms_laplacian)
UniformTopographyInterface.register_function(
    "rms_curvature_from_area", rms_curvature_from_area
)

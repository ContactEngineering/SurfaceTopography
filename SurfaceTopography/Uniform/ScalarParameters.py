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
Functions computing scalar roughness parameters
"""

import numpy as np
import scipy
from NuMPI.Tools import Reduction

from ..HeightContainer import UniformTopographyInterface

_mad_to_rms = 1 / scipy.stats.norm.ppf(3 / 4)


def Rq(topography):
    """
    Compute the root mean square height amplitude of a topography or
    line scan stored on a uniform grid from individual profiles.
    (This is the Rq value.)

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_height : float
        Root mean square height value.
    """
    if topography.is_domain_decomposed:
        raise NotImplementedError('`Rq` does not support MPI-decomposed topographies.')

    n = np.prod(topography.nb_grid_pts)
    reduction = Reduction(topography._communicator)
    profile = topography.heights()
    return np.sqrt(reduction.sum((profile - reduction.sum(profile, axis=0) / topography.nb_grid_pts[0]) ** 2) / n)


def Sq(topography):
    """
    Compute the root mean square height amplitude of a topography or
    line scan stored on a uniform grid from the whole areal data.
    (This is the Sq value.)

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.

    Returns
    -------
    rms_height : float
        Root mean square height value.
    """
    if topography.dim <= 1:
        raise ValueError('Areal rms height can only be computed for topographies, not line scans.')
    elif topography.dim == 2:
        n = np.prod(topography.nb_grid_pts)
        reduction = Reduction(topography._communicator)
        profile = topography.heights()
        return np.sqrt(reduction.sum((profile - reduction.sum(profile) / n) ** 2) / n)
    else:
        raise ValueError(f'Cannot handle topographies of dimension {topography.dim}')


def rms_gradient(topography, short_wavelength_cutoff=None, window=None, direction=None):
    """
    Compute the root mean square amplitude of the height gradient of a
    topography stored on a uniform grid. The topography must be
    two-dimensional (i.e. a topography map).

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.
    short_wavelength_cutoff : float
        All wavelengths below this cutoff will be set to zero amplitude.
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
        Root mean square slope value.
    """
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    if topography.dim <= 1:
        raise ValueError('RMS gradient can only be computed for topographies, not line scans.')
    elif topography.dim == 2:
        mask_function = None if short_wavelength_cutoff is None else \
            lambda frequency: (frequency[0] ** 2 + frequency[1] ** 2) < 1 / short_wavelength_cutoff ** 2
        slx, sly = topography.derivative(1, mask_function=mask_function)
        reduction = Reduction(topography._communicator)
        return np.sqrt(reduction.mean(slx ** 2 + sly ** 2))
    else:
        raise ValueError(f'Cannot handle topographies of dimension {topography.dim}')


def Rdq(topography, short_wavelength_cutoff=None, window=None, direction=None):
    """
    Compute the root mean square amplitude of the height derivative of a
    topography or line scan stored on a uniform grid. If the topography is two
    dimensional (i.e. a topography map), the derivative is computed in the
    x-direction.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.
    short_wavelength_cutoff : float
        All wavelengths below this cutoff will be set to zero amplitude.
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
        Root mean square slope value.
    """
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    mask_function = None if short_wavelength_cutoff is None else \
        lambda frequency: frequency[0] ** 2 < 1 / short_wavelength_cutoff ** 2
    if topography.dim == 1:
        dx = topography.derivative(1, mask_function=mask_function)
    elif topography.dim == 2:
        dx, dy = topography.derivative(1, mask_function=mask_function)
    else:
        raise ValueError(f'Cannot handle topographies of dimension {topography.dim}')
    reduction = Reduction(topography._communicator)
    return np.sqrt(reduction.mean(dx ** 2))


def Rddq(topography, short_wavelength_cutoff=None, window=None, direction=None):
    """
    Compute the root mean square amplitude of the second derivative (i.e. the
    curvature) of a topography or line scan stored on a uniform grid. If the
    topography is two-dimensional (i.e. a topography map), then the rms curvature
    is computed only along the x-direction.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.
    short_wavelength_cutoff : float
        All wavelengths below this cutoff will be set to zero amplitude.
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
        Root mean square curvature value.
    """
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    mask_function = None if short_wavelength_cutoff is None else \
        lambda frequency: frequency[0] ** 2 < 1 / short_wavelength_cutoff ** 2
    if topography.dim == 1:
        d2x = topography.derivative(2, mask_function=mask_function)
    elif topography.dim == 2:
        d2x, d2y = topography.derivative(2, mask_function=mask_function)
    else:
        raise ValueError(f'Cannot handle topographies of dimension {topography.dim}')
    reduction = Reduction(topography._communicator)
    return np.sqrt(reduction.mean(d2x ** 2))


def rms_laplacian(topography, short_wavelength_cutoff=None, window=None, direction=None):
    """
    Compute the root mean square amplitude of the Laplacian (i.e. the sum of
    second derivatives in x- and y-directions) of a topography on a uniform
    grid. The topography must be two-dimensional (i.e. a topography map).

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.
    short_wavelength_cutoff : float
        All wavelengths below this cutoff will be set to zero amplitude.
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
        Root mean square Laplacian value.
    """
    if short_wavelength_cutoff is not None:
        topography = topography.window(window=window, direction=direction)
    if topography.dim == 1:
        raise ValueError('RMS Laplacian can only be computed for topographies, not line scans.')
    elif topography.dim == 2:
        mask_function = None if short_wavelength_cutoff is None else \
            lambda frequency: (frequency[0] ** 2 + frequency[1] ** 2) < 1 / short_wavelength_cutoff ** 2
        curv = topography.derivative(2, mask_function=mask_function)
        reduction = Reduction(topography._communicator)
        return np.sqrt(reduction.mean((curv[0] + curv[1]) ** 2))
    else:
        raise ValueError(f'Cannot handle topographies of dimension {topography.dim}')


def rms_curvature_from_area(topography, short_wavelength_cutoff=None, window=None, direction=None):
    """
    Compute the root mean square amplitude of the curvature of a
    topography stored on a uniform grid. The topography must be
    two-dimensional (i.e. a topography map). This function returns
    half the Laplacian.

    Parameters
    ----------
    topography : :obj:`SurfaceTopography` or :obj:`UniformLineScan`
        SurfaceTopography object containing height information.
    short_wavelength_cutoff : float
        All wavelengths below this cutoff will be set to zero amplitude.
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
        Root mean square curvature value.
    """
    return rms_laplacian(topography, short_wavelength_cutoff=short_wavelength_cutoff, window=window,
                         direction=direction) / 2


# Register analysis functions from this module
UniformTopographyInterface.register_function('rms_height_from_profile', Rq, deprecated=True)
UniformTopographyInterface.register_function('Rq', Rq)
UniformTopographyInterface.register_function('rms_height_from_area', Sq, deprecated=True)
UniformTopographyInterface.register_function('Sq', Sq)
UniformTopographyInterface.register_function('rms_gradient', rms_gradient)
UniformTopographyInterface.register_function('rms_slope_from_profile', Rdq, deprecated=True)
UniformTopographyInterface.register_function('Rdq', Rdq)
UniformTopographyInterface.register_function('rms_curvature_from_profile', Rddq, deprecated=True)
UniformTopographyInterface.register_function('Rddq', Rddq)
UniformTopographyInterface.register_function('rms_laplacian', rms_laplacian)
UniformTopographyInterface.register_function('rms_curvature_from_area', rms_curvature_from_area)

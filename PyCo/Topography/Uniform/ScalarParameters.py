#
# Copyright 2019 Antoine Sanner
#           2018-2019 Lars Pastewka
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
Functions computing scalar roughness parameters
"""

import numpy as np

from ..HeightContainer import UniformTopographyInterface
from NuMPI.Tools import Reduction

def rms_height(topography, kind='Sq'):
    """
    Compute the root mean square height amplitude of a topography or
    line scan stored on a uniform grid.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography object containing height information.

    Returns
    -------
    rms_height : float
        Root mean square height value.
    """
    n = np.prod(topography.nb_grid_pts)
    #if topography.is_MPI:
    pnp = Reduction(topography._communicator)
    profile = topography.heights()
    if kind == 'Sq':
        return np.sqrt(
            pnp.sum((profile - pnp.sum(profile) / n) ** 2) / n)
    elif kind == 'Rq':
        # Problem: when one of the processors holds the full data he isn't able
        # to detect if any axis is MPI_Parallelized
        # this problem is solved automatically if we do not support one axis
        # to be zero
        decomp_axis = [full != loc for full, loc in
                       zip(np.array(topography.nb_grid_pts), profile.shape)]
        temppnp = pnp if decomp_axis[0] == True else np
        return np.sqrt(temppnp.sum(
            (profile - temppnp.sum(profile, axis=0)
             / topography.nb_grid_pts[0]) ** 2
                                    ) / n)
    else:
        raise RuntimeError("Unknown rms height kind '{}'.".format(kind))


def rms_slope(topography):
    """
    Compute the root mean square amplitude of the height gradient of a
    topography or line scan stored on a uniform grid.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography object containing height information.

    Returns
    -------
    rms_slope : float
        Root mean square slope value.
    """
    if topography.is_domain_decomposed:
        raise NotImplementedError("rms_slope not implemented for parallelized topographies")
    if topography.dim == 1:
        return np.sqrt((topography.derivative(1)**2).mean())
    elif topography.dim == 2:
        slx, sly = topography.derivative(1)
        return np.sqrt((slx**2).mean() + (sly**2).mean())
    else:
        raise ValueError('Cannot handle topographies of dimension {}'.format(topography.dim))


def rms_laplacian(topography):
    """
    Compute the root mean square Laplacian of the height gradient of a
    topography or line scan stored on a uniform grid. The rms curvature
    is half of the value returned here.

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography object containing height information.

    Returns
    -------
    rms_laplacian : float
        Root mean square Laplacian value.
    """
    if topography.is_domain_decomposed:
        raise NotImplementedError("rms_Laplacian not implemented for parallelized topographies")
    if topography.dim == 1:
        curv = topography.derivative(2)
        return np.sqrt((curv[1:-1]**2).mean())
    elif topography.dim == 2:
        curv = topography.derivative(2)
        if topography.is_periodic:
            return np.sqrt(((curv[0]+curv[1])**2).mean())
        else:
            return np.sqrt(((curv[0][:, 1:-1] + curv[1][1:-1, :]) ** 2).mean())
    else:
        raise ValueError('Cannot handle topographies of dimension {}'.format(topography.dim))

def rms_curvature(topography):
    """
    Compute the root mean square curvature of the height gradient of a
    topography or line scan stored on a uniform grid.

    For 2D Data the rms Laplacian is twice of the value returned here.

    For 1D Data they are identical

    Parameters
    ----------
    topography : :obj:`Topography` or :obj:`UniformLineScan`
        Topography object containing height information.

    Returns
    -------
    rms_curvature : float
        Root mean square curvature value.
    """
    if topography.is_domain_decomposed:
        raise NotImplementedError("rms_curvature not implemented for parallelized topographies")
    if topography.dim == 1:
        fac = 1.
    elif topography.dim == 2:
        fac = 1./2
    else:
        raise ValueError('Cannot handle topographies of dimension {}'.format(topography.dim))
    return fac * rms_laplacian(topography)



### Register analysis functions from this module

UniformTopographyInterface.register_function('rms_height', rms_height)
UniformTopographyInterface.register_function('rms_slope', rms_slope)
UniformTopographyInterface.register_function('rms_laplacian', rms_laplacian)
UniformTopographyInterface.register_function('rms_curvature', rms_curvature)

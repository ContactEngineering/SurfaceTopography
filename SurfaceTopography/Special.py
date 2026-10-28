#
# Copyright 2019-2020, 2022 Antoine Sanner
#           2018-2021 Lars Pastewka
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
Special (deterministic) geometries: topographies from analytic functions,
spheres and paraboloids, and topographies with a plastic displacement field.
"""

import numpy as np

from NuMPI import MPI
from NuMPI.Tools import Reduction

from .UniformLineScanAndTopography import Topography, UniformLineScan, \
    DecoratedUniformTopography


def make_topography_from_function(fun, physical_sizes,
                                  nb_grid_pts=None,
                                  subdomain_locations=None,
                                  nb_subdomain_grid_pts=None,
                                  centre=(0, 0),
                                  **kwargs
                                  ):
    """
    Create a topography map by evaluating a function on the grid,
    :math:`h(x, y) = f(x - c_x, y - c_y)`.

    Parameters
    ----------
    fun : callable(x, y)
        Function that returns the heights for arrays of positions.
    physical_sizes : tuple of floats
        Physical size :math:`(L_x, L_y)` of the topography map.
    nb_grid_pts : tuple of ints
        Number of grid points of the full topography.
    subdomain_locations : tuple of ints, optional
        Origin (location) of the subdomain handled by the present MPI
        process. (Default: None, i.e. the origin)
    nb_subdomain_grid_pts : tuple of ints, optional
        Number of grid points within the subdomain handled by the present
        MPI process. If None, the full grid is evaluated (serial
        topography). (Default: None)
    centre : tuple or "middle", optional
        Position :math:`(c_x, c_y)` of the origin of `fun`. "middle"
        places it at the centre of the topography. (Default: (0, 0))
    **kwargs
        Further keyword arguments passed to `Topography`, e.g. `periodic`,
        `unit`, `info` and `communicator`.

    Returns
    -------
    topography : Topography
        The topography map.
    """
    if isinstance(centre, str) and centre == "middle":
        centre = tuple([s / 2 for s in physical_sizes])

    if nb_subdomain_grid_pts is None:
        # serial code
        nb_subdomain_grid_pts = nb_grid_pts
        nb_grid_pts = None
    # We need a topography object to compute the positions, but the heights
    # are only known afterwards; construct a second topography with the
    # actual heights so that the constructor's validation (array shape,
    # conversion to float, masking of NaNs as undefined data) is applied.
    # (Assigning to `_heights` directly would bypass all of this.)
    dummy = Topography(heights=np.zeros(nb_subdomain_grid_pts),
                       physical_sizes=physical_sizes,
                       subdomain_locations=subdomain_locations,
                       nb_grid_pts=nb_grid_pts,
                       decomposition="subdomain",
                       **kwargs,
                       )

    cx, cy = centre
    x, y = dummy.positions()

    return Topography(heights=fun(x - cx, y - cy),
                      physical_sizes=physical_sizes,
                      subdomain_locations=subdomain_locations,
                      nb_grid_pts=nb_grid_pts,
                      decomposition="subdomain",
                      **kwargs,
                      )


def make_sphere(radius, nb_grid_pts, physical_sizes, centre=None,
                standoff="undefined",
                offset=0, periodic=False, kind="sphere",
                nb_subdomain_grid_pts=None, subdomain_locations=None,
                communicator=MPI.COMM_WORLD):
    r"""
    Spherical cap or paraboloid with its apex (maximum) at height `offset`.

    For kind="sphere" (default)

    .. math:: h = \left\{ \begin{array}{ll} \sqrt{R^2 - r^2} - R
                  & \text{for } r \le R \\ -R - \text{standoff}
                  & \text{else} \end{array} \right.

    For kind="paraboloid", the sphere is approximated by a paraboloid

    .. math:: h = -\frac{r^2}{2R}

    Here :math:`R` is the radius and :math:`r^2 = (x - c_x)^2 + (y - c_y)^2`
    the squared distance from the centre (:math:`r^2 = (x - c_x)^2` for line
    scans). `offset` is added to :math:`h`.

    Parameters
    ----------
    radius : float
        Radius :math:`R` of the sphere.
    nb_grid_pts : int or tuple of ints
        Number of grid points. A single value (or one-element tuple) yields
        a `UniformLineScan`, two values a `Topography`.
    physical_sizes : float or tuple of floats
        Physical size of the line scan or topography map.
    centre : float or tuple of floats, optional
        Position of the centre (in length units, not pixels).
        (Default: None, i.e. the centre of the domain)
    standoff : float or "undefined", optional
        Height outside the sphere (:math:`r > R`), measured downwards from
        the lowest point of the sphere, :math:`h = -R - \text{standoff}`.
        For "undefined", these points are masked and `has_undefined_data`
        is True. Not used for kind="paraboloid". (Default: "undefined")
    offset : float, optional
        Constant added to all heights. (Default: 0)
    periodic : bool, optional
        Distances are computed with the minimum image convention, i.e. the
        sphere wraps around the periodic boundaries. The result is only
        meaningful if the sphere fits into the domain. (Default: False)
    kind : str, optional
        "sphere" or "paraboloid". (Default: "sphere")
    nb_subdomain_grid_pts : tuple of ints, optional
        Number of grid points of the local subdomain (MPI-parallel runs).
        (Default: None, i.e. the full grid)
    subdomain_locations : tuple of ints, optional
        Location of the local subdomain (MPI-parallel runs).
        (Default: None, i.e. the origin)
    communicator : mpi4py communicator or NuMPI stub communicator, optional
        MPI communicator object, only used for topography maps.
        (Default: MPI.COMM_WORLD)

    Returns
    -------
    topography : UniformLineScan or Topography
        The sphere.
    """
    if not hasattr(nb_grid_pts, "__iter__"):
        nb_grid_pts = (nb_grid_pts,)
    dim = len(nb_grid_pts)
    if not hasattr(physical_sizes, "__iter__"):
        physical_sizes = (physical_sizes,)
    if centre is None:
        centre = np.array(physical_sizes) * .5
    if not hasattr(centre, "__iter__"):
        centre = (centre,)

    if nb_subdomain_grid_pts is None:
        nb_subdomain_grid_pts = nb_grid_pts
    if subdomain_locations is None:
        subdomain_locations = (0, 0)
    if not periodic:
        def get_r(res, size, centre, subd_loc, subd_res):
            " computes the non-periodic radii to evaluate"
            x = (subd_loc + np.arange(subd_res)) * size / res
            return x - centre
    else:
        def get_r(res, size, centre, subd_loc, subd_res):
            " computes the periodic radii to evaluate"
            x = (subd_loc + np.arange(subd_res)) * size / res
            return (x - centre + size / 2) % size - size / 2

    if dim == 1:
        r2 = get_r(nb_grid_pts[0], physical_sizes[0], centre[0],
                   subdomain_locations[0], nb_subdomain_grid_pts[0]) ** 2
    elif dim == 2:
        rx2 = (get_r(nb_grid_pts[0], physical_sizes[0], centre[0],
                     subdomain_locations[0],
                     nb_subdomain_grid_pts[0]) ** 2).reshape((-1, 1))
        ry2 = (get_r(nb_grid_pts[1], physical_sizes[1], centre[1],
                     subdomain_locations[1], nb_subdomain_grid_pts[1])) ** 2
        r2 = rx2 + ry2
    else:
        raise ValueError("Problem has to be 1- or 2-dimensional. "
                         "Yours is {}-dimensional".format(dim))

    if kind == "sphere":
        radius2 = radius ** 2  # avoid nans for small radii
        outside = r2 > radius2
        r2[outside] = radius2
        h = np.sqrt(radius2 - r2) - radius
        if standoff == "undefined":
            standoff_val = np.nan
        else:
            standoff_val = - standoff - radius
        h[outside] = standoff_val
    elif kind == "paraboloid":
        h = - r2 / (2 * radius)
    else:
        raise (ValueError("Wrong value given for parameter kind {}. "
                          "Should be 'sphere' or 'paraboloid'".format(kind)))

    if dim == 1:
        ret_top = UniformLineScan(h + offset, physical_sizes,
                                  periodic=periodic)
    else:
        ret_top = Topography(h + offset, physical_sizes,
                             periodic=periodic,
                             decomposition='subdomain',
                             nb_grid_pts=nb_grid_pts,
                             subdomain_locations=subdomain_locations,
                             communicator=communicator)

    # For `standoff="undefined"` the region outside the sphere contains NaNs,
    # which the constructors above turn into masked (undefined) data points.
    # For a numeric standoff the heights outside the sphere were already set
    # to the (finite) standoff value above; no filling is necessary.
    return ret_top


class PlasticTopography(DecoratedUniformTopography):
    r"""
    Topography with an additional plastic displacement field,
    :math:`h = h_0 + u_\text{pl}`, where :math:`h_0` are the heights of the
    parent topography. The field is stored per grid point of the local
    subdomain and is updated by the user, e.g. by a contact mechanics code
    that limits the contact pressure to the hardness.
    """
    name = 'plastic_topography'

    def __init__(self, topography, hardness, plastic_displ=None):
        """
        Parameters
        ----------
        topography : :obj:`UniformTopographyInterface`
            Undeformed topography.
        hardness : float
            Penetration hardness (units of pressure); must not be negative.
        plastic_displ : np.ndarray, optional
            Initial plastic displacements, same shape as the local
            subdomain. (Default: None, i.e. zero)
        """
        super().__init__(topography)
        self.hardness = hardness
        if plastic_displ is None:
            plastic_displ = np.zeros(self.nb_subdomain_grid_pts)
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
        if hardness < 0:
            raise ValueError('Hardness must be positive.')
        self._hardness = hardness

    @property
    def plastic_displ(self):
        return self.__h_pl

    @plastic_displ.setter
    def plastic_displ(self, plastic_displ):
        if plastic_displ.shape != self.nb_subdomain_grid_pts:
            raise ValueError(
                'Resolution of profile and plastic displacement must match.')
        self.__h_pl = plastic_displ

    def undeformed_profile(self):
        """Return the heights of the undeformed (parent) topography."""
        return self.parent_topography.heights()

    def heights(self):
        """Return undeformed heights plus plastic displacements."""
        return self.undeformed_profile() + self.plastic_displ

    @property
    def plastic_area(self):
        """Area of the grid points with nonzero plastic displacement."""
        pnp = Reduction(self._communicator)
        return pnp.sum(np.count_nonzero(self.__h_pl)) * self.area_per_pt

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
Special topographies
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
    Parameters
    ----------
    fun: callable(X, Y)
    centre: tuple or "middle"
        (cx, cy)
        h(X,Y) =  fun(X-cx, Y-cy)
        default (0,0)
    physical_sizes : tuple of floats
        Physical physical_sizes of the topography map
    periodic : bool
        The topography is periodic. (Default: False)
    decomposition : str
        Specification of the data decomposition of the heights array. If
        set to 'subdomain', the heights array contains only the part of
        the full topography local to the present MPI process. If set to
        'domain', the heights array contains the global array.
        Default: 'serial', which fails for parallel runs.
    nb_grid_pts : tuple of ints
        Number of grid points for the full topography. This is only
        required if decomposition is set to 'subdomain'.
    subdomain_locations : tuple of ints
        Origin (location) of the subdomain handled by the present MPI
        process.
    nb_subdomain_grid_pts : tuple of ints
        Number of grid points within the subdomain handled by the present
        MPI process. This is only required if decomposition is set to
        'domain'.
    communicator : mpi4py communicator or NuMPI stub communicator
        The MPI communicator object.
        default value is COMM_SELF because sometimes NON-MPI readers that
        do not set the communicator value are used in MPI Programs.
        See discussion in issue #166
    info : dict
        The info dictionary containing auxiliary data. This data is never
        used by PyCo but can be used by third-party codes.

    Returns
    -------

    SurfaceTopography.Topography instance

    """
    if isinstance(centre, str) and centre == "middle":
        centre = tuple([s / 2 for s in physical_sizes])

    if nb_subdomain_grid_pts is None:
        # serial code
        nb_subdomain_grid_pts = nb_grid_pts
        nb_grid_pts = None
    topography = Topography(heights=np.zeros(nb_subdomain_grid_pts),
                            physical_sizes=physical_sizes,
                            subdomain_locations=subdomain_locations,
                            nb_grid_pts=nb_grid_pts,
                            decomposition="subdomain",
                            **kwargs,
                            )

    cx, cy = centre
    x, y = topography.positions()

    topography._heights = fun(x - cx, y - cy)

    return topography


def make_sphere(radius, nb_grid_pts, physical_sizes, centre=None,
                standoff="undefined",
                offset=0, periodic=False, kind="sphere",
                nb_subdomain_grid_pts=None, subdomain_locations=None,
                communicator=MPI.COMM_WORLD):
    r"""
    Simple sphere geometry.

    If kind="sphere" (Default)

    .. math:: h = \left\{ \begin{array}{ll} \sqrt{\text{radius}^2 - r^2} -
                  \text{radius} & \text{  for  } r < \text{radius} \\ - \text{standoff}
                   & \text{else} \end{array} \right.

    If kind="paraboloid" the sphere is approximated by a paraboloid

    .. math:: h = \frac{r^2}{2 \cdot \text{radius}}

    :math:`r^2 = x^2 + y^2`

    Parameters
    ----------
    radius : float
        self-explanatory
    nb_grid_pts : float
        self-explanatory
    physical_sizes : float
        self-explanatory
    centre : float
         specifies the coordinates (in length units, not pixels).
         by default, the sphere is centred in the topography
    kind: str
        Options are "sphere" or "paraboloid". Default is "sphere".
    standoff : float or "undefined"
         when using interaction forces with ranges of the order
         the radius, you might want to set the topography outside of
         the sphere to far away, maybe even pay the price of inf,
         if your interaction has no cutoff

         For `standoff="undefined"`, the entries will be masked in the 
         topography and topography.has_undefined_data will be true

         If `kind="paraboloid"` the paraboloid approximation is used
            and the standoff is not applied
    periodic : bool
         whether the sphere can wrap around. tricky for large spheres
    communicator : mpi4py communicator NuMPI stub communicator
         MPI communicator object.
    """  # noqa: E501, W291
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
        ret_top = UniformLineScan(h + offset, physical_sizes)
    else:
        ret_top = Topography(h + offset, physical_sizes,
                             decomposition='subdomain',
                             nb_grid_pts=nb_grid_pts,
                             subdomain_locations=subdomain_locations,
                             communicator=communicator)

    if standoff == "undefined":
        return ret_top
    else:
        return ret_top.fill_undefined_data(standoff_val)


class PlasticTopography(DecoratedUniformTopography):
    """ SurfaceTopography with an additional plastic deformation field.
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
        """ Returns the undeformed profile of the topography.
        """
        return self.parent_topography.heights()

    def heights(self):
        """ Computes the combined profile.
        """
        return self.undeformed_profile() + self.plastic_displ

    @property
    def plastic_area(self):
        pnp = Reduction(self._communicator)
        return pnp.sum(np.count_nonzero(self.__h_pl)) * self.area_per_pt

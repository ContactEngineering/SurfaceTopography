#
# Copyright 2019-2022, 2024 Lars Pastewka
#           2019-2021, 2023 Antoine Sanner
#           2019 Michael Röttger
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
Support for uniform topogography descriptions
"""

import numpy as np
from NuMPI import MPI
from NuMPI.Tools import Reduction

from .HeightContainer import (AbstractTopography, DecoratedTopography,
                              UniformTopographyInterface)
from .Support.UnitConversion import get_unit_conversion_factor


class UniformLineScan(AbstractTopography, UniformTopographyInterface):
    """
    Line scan that lives on a uniform one-dimensional grid.
    """

    def __init__(self, heights, physical_sizes, periodic=False, unit=None, info={}, communicator=MPI.COMM_SELF):
        """
        Parameters
        ----------
        profile : array_like
            Data containing the height information. Needs to be a
            one-dimensional array.
        physical_sizes : tuple of floats
            Physical physical_sizes of the topography map
        periodic : bool, optional
            Flag setting the periodicity of the surface
        unit : str, optional
            The length unit.
        info : dict, optional
            The info dictionary containing auxiliary data.
        """
        if not hasattr(heights, 'ndim'):
            heights = np.array(heights)

        if heights.ndim != 1:
            raise ValueError('Heights array must be one-dimensional.')

        super().__init__(unit=unit, info=info, communicator=communicator)

        # Automatically turn this into a masked array if it is not already a
        # masked array and there is data missing
        if not np.ma.is_masked(heights) and np.sum(np.logical_not(np.isfinite(heights))) > 0:
            heights = np.ma.masked_where(np.logical_not(np.isfinite(heights)), heights)
        self._heights = np.asanyarray(heights, dtype=float)
        self._size = np.asarray(physical_sizes).item()
        self._periodic = periodic

    def __getstate__(self):
        state = super().__getstate__(), self._heights, self._size, self._periodic
        return state

    def __setstate__(self, state):
        superstate, self._heights, self._size, self._periodic = state
        super().__setstate__(superstate)

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 1

    @property
    def physical_sizes(self):
        if self._size is None:
            return None
        return self._size,

    @property
    def is_periodic(self):
        return self._periodic

    # Implement uniform line scan interface

    @property
    def nb_grid_pts(self):
        return len(self._heights),

    @property
    def nb_subdomain_grid_pts(self):
        return self.nb_grid_pts

    @property
    def is_domain_decomposed(self):
        return False

    @property
    def pixel_size(self):
        return (self.physical_sizes[0] / self.nb_grid_pts[0],)

    @property
    def area_per_pt(self):
        return self.pixel_size[0]

    @property
    def has_undefined_data(self):
        return np.ma.getmaskarray(self._heights).sum() > 0

    def positions(self, meshgrid=True):
        """
        Return grid positions.

        Arguments
        ---------
        meshgrid : bool, optional
            Ignore, for compatibility with two-dimensional topographies

        Returns
        -------
        x : np.ndarray
            X-positions
        """
        r, = self.nb_grid_pts
        p, = self.pixel_size
        return np.arange(r) * p

    def heights(self):
        return self._heights

    def save(self, fname, compress=True):
        """
        saves the topography as a NumpyTxtTopography. Warning: This only saves
        the profile; the physical_sizes is not contained in the file
        """
        if compress:
            if not fname.endswith('.gz'):
                fname = fname + ".gz"
        np.savetxt(fname, self.heights())

    def squeeze(self):
        return self


class Topography(AbstractTopography, UniformTopographyInterface):
    """
    Topography that lives on a uniform two-dimensional grid, i.e. a
    topography map.
    """

    def __init__(self, heights, physical_sizes, periodic=False,
                 nb_grid_pts=None, subdomain_locations=None,
                 nb_subdomain_grid_pts=None,
                 unit=None, info={},
                 decomposition='serial',
                 communicator=MPI.COMM_SELF):

        """
        Parameters
        ----------
        heights : array_like
            Data containing the height information. Needs to be a
            two-dimensional array.
        physical_sizes : tuple of floats
            Physical physical_sizes of the topography map
        periodic : bool, optional
            The topography is periodic. (Default: False)
        nb_grid_pts : tuple of ints, optional
            Number of grid points for the full topography. This is only
            required if decomposition is set to 'subdomain'.
        subdomain_locations : tuple of ints, optional
            Origin (location) of the subdomain handled by the present MPI
            process.
        nb_subdomain_grid_pts : tuple of ints, optional
            Number of grid points within the subdomain handled by the present
            MPI process. This is only required if decomposition is set to
            'domain'.
        unit : str, optional
            The length unit.
        info : dict, optional
            The info dictionary containing auxiliary data.
        decomposition : str, optional
            Specification of the data decomposition of the heights array. If
            set to 'subdomain', the heights array contains only the part of
            the full topography local to the present MPI process. If set to
            'domain', the heights array contains the global array.
            Default: 'serial', which fails for parallel runs.
        communicator : mpi4py communicator or NuMPI stub communicator, optional
            The MPI communicator object.
            default value is COMM_SELF because sometimes NON-MPI readers that
            do not set the communicator value are used in MPI Programs.
            See discussion in issue #166

        Examples for initialisation:
        ----------------------------
        1. Serial code
        >>>Topography(heights, physical_sizes, [periodic, info])
        2. Parallel code, providing local data
        >>>Topography(heights, physical_sizes, [periodic],
                      decomposition='subdomain',
                      subdomain_locations, nb_grid_pts, comm,
                      [info])
        3. Parallel code, providing full data
        >>>Topography(heights, physical_sizes, [periodic],
                      decomposition='domain',
                      subdomain_locations, nb_subdomain_grid_pts, comm,
                      [info])
        """
        if heights.ndim != 2:
            raise ValueError('Heights array must be two-dimensional.')

        super().__init__(unit=unit, info=info, communicator=communicator)

        # Automatically turn this into a masked array if there is data missing
        if not np.ma.is_masked(heights) and np.sum(np.logical_not(np.isfinite(heights))) > 0:
            heights = np.ma.masked_where(np.logical_not(np.isfinite(heights)), heights)

        if communicator.Get_size() == 1:
            # case 1. : no parallelization
            if nb_grid_pts is not None and tuple(nb_grid_pts) != heights.shape:
                raise ValueError(
                    'This is a serial run but `nb_grid_pts` (= {}) does not '
                    'equal the shape of the `heights` (= {}) array.'.format(nb_grid_pts, heights.shape))
            if subdomain_locations is not None and tuple(
                    subdomain_locations) != (0, 0):
                raise ValueError(
                    'This is a serial run but `subdomain_locations` (= {}) '
                    'is not the origin.'.format(subdomain_locations))
            if nb_subdomain_grid_pts is not None and tuple(
                    nb_subdomain_grid_pts) != heights.shape:
                raise ValueError(
                    'This is a serial run but `nb_subdomain_grid_pts` '
                    '(= {}) does not equal the shape '
                    'of the `heights` (= {}) array.'.format(nb_subdomain_grid_pts, heights.shape))
            self._nb_grid_pts = heights.shape
            self._subdomain_locations = (0, 0)
            self._heights = np.asanyarray(heights, dtype=float)
        elif decomposition == 'subdomain':
            # Case 2.: parallelized and local data provided
            if nb_grid_pts is None:
                raise ValueError(
                    "This is a parallel run with 'subdomain' decomposition; '"
                    "'please specify `nb_grid_pts` since it cannot be "
                    "inferred.")
            if subdomain_locations is None:
                raise ValueError('This is a parallel run; please specify '
                                 '`subdomain_locations`.')
            if nb_subdomain_grid_pts is not None and tuple(
                    nb_subdomain_grid_pts) != heights.shape:
                raise ValueError(
                    "This is a parallel run with 'subdomain' decomposition "
                    "but `nb_subdomain_grid_pts` (= {}) does not equal the "
                    "shape of the `heights` (= {}) array.".format(nb_subdomain_grid_pts, heights.shape))
            self._nb_grid_pts = nb_grid_pts
            self._subdomain_locations = subdomain_locations
            self._heights = np.asanyarray(heights, dtype=float)
        elif decomposition == 'domain':
            # Case 3: parallelized but global data provided
            if nb_grid_pts is not None and tuple(nb_grid_pts) != heights.shape:
                raise ValueError(
                    "This is a parallel run with 'domain' decomposition but "
                    "`nb_grid_pts` (= {}) does not equal the shape of the "
                    "`heights` (= {}) array.".format(nb_grid_pts, heights.shape))
            if subdomain_locations is None:
                raise ValueError('This is a parallel run; please specify `subdomain_locations`.')
            if nb_subdomain_grid_pts is None:
                raise ValueError(
                    "This is a parallel run with 'domain' decomposition; "
                    "please specify `nb_subdomain_grid_pts` since it cannot "
                    "be inferred.")
            self._nb_grid_pts = heights.shape
            self._subdomain_locations = subdomain_locations
            self._heights = np.asanyarray(
                heights[tuple(slice(s, s + n) for s, n in
                              zip(subdomain_locations,
                                  nb_subdomain_grid_pts))], dtype=float)
        elif decomposition == 'serial':
            raise ValueError(
                "`decomposition` is 'serial' but this is a parallel run.")
        else:
            raise ValueError(
                "`decomposition` can be either 'domain' or 'subdomain'.")

        self._size = physical_sizes
        self._periodic = periodic

    def __getstate__(self):
        state = super().__getstate__(), self._heights, self._size, self._periodic, self._subdomain_locations, \
            self._nb_grid_pts
        return state

    def __setstate__(self, state):
        superstate, self._heights, self._size, self._periodic, self._subdomain_locations, self._nb_grid_pts = state
        super().__setstate__(superstate)

    # Implement abstract methods of AbstractHeightContainer

    @property
    def dim(self):
        return 2

    @property
    def is_periodic(self):
        return self._periodic

    @property
    def physical_sizes(self):
        return self._size

    @property
    def nb_grid_pts(self):
        return self._nb_grid_pts

    @property
    def nb_subdomain_grid_pts(self):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self._heights.shape

    @property
    def subdomain_locations(self):
        return self._subdomain_locations

    @property
    def subdomain_slices(self):
        return tuple(slice(s, s + n) for s, n in
                     zip(self.subdomain_locations, self.nb_subdomain_grid_pts))

    # Implement topography interface

    @property
    def pixel_size(self):
        return np.asanyarray(self.physical_sizes, dtype=float) / np.asanyarray(self.nb_grid_pts)

    @property
    def area_per_pt(self):
        return np.prod(self.pixel_size)

    @property
    def has_undefined_data(self):
        return Reduction(self._communicator).sum(np.ma.getmaskarray(self._heights).sum()) > 0

    def positions(self, meshgrid=True):
        """
        Return grid positions.

        Arguments
        ---------
        meshgrid : bool, optional
            If True, return the position on the same grid as the heights.
            Otherwise, return one-dimensional position arrays. (Default: True)

        Returns
        -------
        x : np.ndarray
            X-positions
        y : np.ndarray
            Y-positions
        """
        # FIXME: Write test for this method
        nx, ny = self.nb_grid_pts
        lnx, lny = self.nb_subdomain_grid_pts
        sx, sy = self.physical_sizes
        x = (self.subdomain_locations[0] + np.arange(lnx)) * sx / nx
        y = (self.subdomain_locations[1] + np.arange(lny)) * sy / ny
        if meshgrid:
            x, y = np.meshgrid(x, y, indexing='ij')
        return x, y

    def heights(self):
        return self._heights

    def positions_and_heights(self, **kwargs):
        x, y = self.positions(**kwargs)
        return x, y, self.heights()

    @property
    def communicator(self):
        return self._communicator

    def squeeze(self):
        return self


class DecoratedUniformTopography(DecoratedTopography,
                                 UniformTopographyInterface):
    @property
    def has_undefined_data(self):
        return self.parent_topography.has_undefined_data

    @property
    def is_domain_decomposed(self):
        return False

    @property
    def is_periodic(self):
        return self.parent_topography.is_periodic

    @property
    def dim(self):
        return self.parent_topography.dim

    @property
    def pixel_size(self):
        return self.parent_topography.pixel_size

    @property
    def unit(self):
        if self._unit is None:
            return self.parent_topography.unit
        else:
            return self._unit

    @property
    def info(self):
        info = self.parent_topography.info
        info.update(self._info)
        if self.unit is not None:
            info.update(dict(unit=self.unit))
        return info

    @property
    def physical_sizes(self):
        return self.parent_topography.physical_sizes

    @property
    def nb_grid_pts(self):
        return self.parent_topography.nb_grid_pts

    @property
    def nb_subdomain_grid_pts(self):
        """ needs to be testable to make sure that geometry and halfspace are
            compatible
        """
        return self.parent_topography.nb_subdomain_grid_pts

    @property
    def subdomain_locations(self):
        return self.parent_topography.subdomain_locations

    @property
    def subdomain_slices(self):
        return self.parent_topography.subdomain_slices

    @property
    def area_per_pt(self):
        return self.parent_topography.area_per_pt

    def positions(self, **kwargs):
        return self.parent_topography.positions(**kwargs)

    def positions_and_heights(self, **kwargs):
        if self.dim == 1:
            return self.positions(**kwargs), self.heights()
        else:
            return (*self.positions(**kwargs), self.heights())

    def squeeze(self):
        if self.dim == 1:
            return UniformLineScan(self.heights(), self.physical_sizes,
                                   periodic=self.is_periodic, info=self.info, unit=self.unit)
        else:
            return Topography(self.heights(), self.physical_sizes,
                              periodic=self.is_periodic, info=self.info, unit=self.unit)


class ScaledUniformTopography(DecoratedUniformTopography):
    """Scale heights, positions or both."""

    def __init__(self, topography, unit=None, info={}):
        """
        This topography wraps a parent topography and rescales x, y and z
        coordinates according to certain rules.

        Arguments
        ---------
        topography : :obj:`UniformTopographyInterface`
            Parent topography
        unit : str
            Target unit.
        info : dict, optional
            Updated entries to the info dictionary. (Default: {})
        """
        super().__init__(topography, unit=unit, info=info)

    @property
    def height_scale_factor(self):
        return get_unit_conversion_factor(self.parent_topography.unit, self.unit)

    @property
    def position_scale_factor(self):
        return get_unit_conversion_factor(self.parent_topography.unit, self.unit)

    @property
    def pixel_size(self):
        """Compute rescaled pixel sizes."""
        return tuple(self.position_scale_factor * s for s in super().pixel_size)

    @property
    def physical_sizes(self):
        """Compute rescaled physical sizes."""
        return tuple(self.position_scale_factor * s for s in super().physical_sizes)

    @property
    def area_per_pt(self):
        """Compute rescaled physical sizes."""
        return np.prod(self.pixel_size)

    def positions(self, **kwargs):
        """Compute the rescaled positions."""
        if self.dim == 1:
            return self.position_scale_factor * super().positions(**kwargs)
        else:
            return tuple(self.position_scale_factor * p for p in super().positions(**kwargs))

    def heights(self):
        """Computes the rescaled profile."""
        return self.height_scale_factor * self.parent_topography.heights()


class StaticallyScaledUniformTopography(ScaledUniformTopography):
    """Scale heights, positions or both."""

    def __init__(self, topography, height_scale_factor, position_scale_factor=1, unit=None, info={}):
        """
        This topography wraps a parent topography and rescales x, y and z
        coordinates according to certain rules.

        Arguments
        ---------
        topography : :obj:`UniformTopographyInterface`
            Parent topography
        height_scale_factor : float
            Factor to scale heights with.
        position_scale_factor : float, optional
            Factor to scale lateral positions (`physical_sizes`, etc.) with.
            (Default: 1)
        unit : str, optional
            Target unit. This is simply used to update the metadata, not for
            determining scale factors. (Default: None)
        info : dict, optional
            Updated entries to the info dictionary. (Default: {})
        """
        super().__init__(topography, unit=unit, info=info)
        self._height_scale_factor = height_scale_factor
        self._position_scale_factor = position_scale_factor

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self._height_scale_factor, self._position_scale_factor
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self._height_scale_factor, self._position_scale_factor = state
        super().__setstate__(superstate)

    @property
    def height_scale_factor(self):
        return self._height_scale_factor

    @property
    def position_scale_factor(self):
        return self._position_scale_factor


class TransposedUniformTopography(DecoratedUniformTopography):
    """
    Tranpose topography.
    """

    def __init__(self, topography, info={}):
        """
        Parameters
        ----------
        topography : :obj:`UniformTopography`
            SurfaceTopography to transpose
        info : dict
            Additional entries for the info dictionary
        """
        super().__init__(topography, info=info)

    @property
    def nb_grid_pts(self):
        """ Return number of points """
        if self.dim == 1:
            return self.parent_topography.nb_grid_pts
        else:
            nx, ny = self.parent_topography.nb_grid_pts
            return ny, nx

    @property
    def physical_sizes(self):
        """ Return physical physical_sizes """
        if self.dim == 1:
            return self.parent_topography.physical_sizes
        else:
            sx, sy = self.parent_topography.physical_sizes
            return sy, sx

    def heights(self):
        """ Computes the rescaled profile.
        """
        return self.parent_topography.heights().T

    def positions(self, **kwargs):
        X, Y = self.parent_topography.positions(**kwargs)
        return Y.T, X.T


class TranslatedTopography(DecoratedUniformTopography):
    """ used when geometries are translated
    """
    name = 'translated_topography'

    def __init__(self, topography, offset=(0, 0), info={}):
        """
        Keyword Arguments:
        topography  -- SurfaceTopography to translate
        offset -- Translation offset in number of grid points
        """
        super().__init__(topography, info=info)
        if not isinstance(topography, UniformTopographyInterface):
            raise ValueError(f"Should provide an instance of UniformTopographyInterface "
                             f"but {type(topography)} provided")
        if not topography.is_periodic:
            raise ValueError("Only periodic topographies can be translated")

        self._offset = offset

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset, offsety=None):
        if offsety is None:
            self._offset = offset
        else:
            self._offset = (offset, offsety)

    def heights(self):
        """ Computes the translated profile.
        """
        offsetx, offsety = self.offset
        return np.roll(
            np.roll(self.parent_topography.heights(), offsetx, axis=0),
            offsety, axis=1)


class CompoundTopography(DecoratedUniformTopography):
    """ used when geometries are combined
    """
    name = 'compound_topography'

    def __init__(self, topography_a, topography_b, info={}):
        """ Behaves like a topography that is a sum of two Topographies
        Keyword Arguments:
        topography_a   -- first topography of the compound
        topography_b   -- second topography of the compound
        """

        super().__init__(topography_a, info=info)

        def combined_val(prop_a, prop_b, propname):
            """
            topographies can have a fixed or dynamic, adaptive nb_grid_pts (or
            other attributes). This function assures that -- if this function
            is called for two topographies with fixed nb_grid_ptss --
            the nb_grid_pts are identical
            Parameters:
            prop_a   -- field of one topography
            prop_b   -- field of other topography
            propname -- field identifier (for error messages only)
            """
            if prop_a is None:
                return prop_b
            else:
                if prop_b is not None:
                    assert prop_a == prop_b, \
                        "{} incompatible:{} <-> {}".format(
                            propname, prop_a, prop_b)
                return prop_a

        self._dim = combined_val(topography_a.dim, topography_b.dim, 'dim')
        self._nb_grid_pts = combined_val(topography_a.nb_grid_pts,
                                         topography_b.nb_grid_pts,
                                         'nb_grid_pts')
        self._size = combined_val(topography_a.physical_sizes,
                                  topography_b.physical_sizes,
                                  'physical_sizes')
        self.parent_topography_a = topography_a
        self.parent_topography_b = topography_b

    def heights(self):
        """ Computes the combined profile
        """
        return (self.parent_topography_a.heights() +
                self.parent_topography_b.heights())


# Register analysis functions from this module
UniformTopographyInterface.register_function('mean', lambda this: Reduction(this._communicator).mean(this.heights()))
UniformTopographyInterface.register_function('min', lambda this: Reduction(this._communicator).min(this.heights()))
UniformTopographyInterface.register_function('max', lambda this: Reduction(this._communicator).max(this.heights()))

# Register pipeline functions from this module
UniformTopographyInterface.register_function('to_unit', ScaledUniformTopography)
UniformTopographyInterface.register_function('scale', StaticallyScaledUniformTopography)
UniformTopographyInterface.register_function('transpose', TransposedUniformTopography)
UniformTopographyInterface.register_function('translate', TranslatedTopography)

UniformTopographyInterface.register_function('superpose', CompoundTopography)

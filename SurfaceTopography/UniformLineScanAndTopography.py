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
Topographies on uniform grids: line scans (`UniformLineScan`), topography
maps (`Topography`) and the pipeline steps for unit conversion, scaling,
transposition, translation and superposition.
"""

import numpy as np
from NuMPI import MPI
from NuMPI.Tools import Reduction

from .HeightContainer import (
    AbstractTopography,
    DecoratedTopography,
    UniformTopographyInterface,
)
from .Support.UnitConversion import get_unit_conversion_factor


class UniformLineScan(AbstractTopography, UniformTopographyInterface):
    """
    Line scan that lives on a uniform one-dimensional grid.
    """

    def __init__(
        self,
        heights,
        physical_sizes,
        periodic=False,
        unit=None,
        info={},
        communicator=MPI.COMM_SELF,
    ):
        """
        Parameters
        ----------
        heights : array_like
            Data containing the height information. Needs to be a
            one-dimensional array. Non-finite values (NaN, inf) are
            converted to masked (undefined) data points.
        physical_sizes : float or tuple of one float
            Physical length :math:`L` of the line scan.
        periodic : bool, optional
            The line scan is periodic. (Default: False)
        unit : str, optional
            The length unit. (Default: None)
        info : dict, optional
            The info dictionary containing auxiliary data. (Default: {})
        communicator : mpi4py communicator or NuMPI stub communicator, optional
            The MPI communicator object. Line scans are never domain
            decomposed. (Default: MPI.COMM_SELF)
        """
        if not hasattr(heights, "ndim"):
            heights = np.array(heights)

        if heights.ndim != 1:
            raise ValueError("Heights array must be one-dimensional.")

        super().__init__(unit=unit, info=info, communicator=communicator)

        # Automatically turn this into a masked array if it is not already a
        # masked array and there is data missing
        if not np.ma.is_masked(heights) and np.sum(~np.isfinite(heights)) > 0:
            heights = np.ma.masked_invalid(heights)
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
        return (self._size,)

    @property
    def is_periodic(self):
        return self._periodic

    # Implement uniform line scan interface

    @property
    def nb_grid_pts(self):
        return (len(self._heights),)

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
        r"""
        Return grid positions :math:`x_i = i \Delta x`.

        Arguments
        ---------
        meshgrid : bool, optional
            Ignored, for compatibility with two-dimensional topographies.

        Returns
        -------
        x : np.ndarray
            X-positions
        """
        (r,) = self.nb_grid_pts
        (p,) = self.pixel_size
        return np.arange(r) * p

    def heights(self):
        """Return the array of heights (masked array if data is undefined)."""
        return self._heights

    def save(self, fname, compress=True):
        """
        Save the heights as a text file with `np.savetxt`. Warning: This only
        saves the heights; physical size, unit and periodicity are not
        contained in the file.

        Parameters
        ----------
        fname : str
            File name.
        compress : bool, optional
            Append `.gz` to the file name (if not present), which makes
            `np.savetxt` write a gzip-compressed file. (Default: True)
        """
        if compress:
            if not fname.endswith(".gz"):
                fname = fname + ".gz"
        np.savetxt(fname, self.heights())

    def squeeze(self):
        """Return this object; it already owns its data."""
        return self


class Topography(AbstractTopography, UniformTopographyInterface):
    """
    Topography that lives on a uniform two-dimensional grid, i.e. a
    topography map.
    """

    def __init__(
        self,
        heights,
        physical_sizes,
        periodic=False,
        nb_grid_pts=None,
        subdomain_locations=None,
        nb_subdomain_grid_pts=None,
        unit=None,
        info={},
        decomposition="serial",
        communicator=MPI.COMM_SELF,
    ):
        """
        Parameters
        ----------
        heights : np.ndarray
            Data containing the height information. Needs to be a
            two-dimensional array (lists are not converted); the first
            index is the x-direction. Non-finite values (NaN, inf) are
            converted to masked (undefined) data points.
        physical_sizes : tuple of floats
            Physical size :math:`(L_x, L_y)` of the topography map.
        periodic : bool, optional
            The topography is periodic. (Default: False)
        nb_grid_pts : tuple of ints, optional
            Number of grid points for the full topography. This is only
            required if decomposition is set to 'subdomain'.
            (Default: None)
        subdomain_locations : tuple of ints, optional
            Origin (location) of the subdomain handled by the present MPI
            process. Required for parallel runs. (Default: None)
        nb_subdomain_grid_pts : tuple of ints, optional
            Number of grid points within the subdomain handled by the present
            MPI process. This is only required if decomposition is set to
            'domain'. (Default: None)
        unit : str, optional
            The length unit. (Default: None)
        info : dict, optional
            The info dictionary containing auxiliary data. (Default: {})
        decomposition : str, optional
            Specification of the data decomposition of the heights array. If
            set to 'subdomain', the heights array contains only the part of
            the full topography local to the present MPI process. If set to
            'domain', the heights array contains the global array and the
            local subdomain is cut out of it. 'serial' raises an error if
            the communicator has more than one process. For a single process,
            the value is ignored. (Default: 'serial')
        communicator : mpi4py communicator or NuMPI stub communicator, optional
            The MPI communicator object. The default is COMM_SELF such that
            readers that do not set a communicator return serial topographies
            also when used within MPI programs (see issue #166).
            (Default: MPI.COMM_SELF)

        Examples
        --------
        Serial code::

            Topography(heights, physical_sizes, periodic=False, info={})

        Parallel code, providing local data::

            Topography(local_heights, physical_sizes,
                       decomposition='subdomain',
                       nb_grid_pts=nb_grid_pts,
                       subdomain_locations=subdomain_locations,
                       communicator=comm)

        Parallel code, providing the full data::

            Topography(global_heights, physical_sizes,
                       decomposition='domain',
                       subdomain_locations=subdomain_locations,
                       nb_subdomain_grid_pts=nb_subdomain_grid_pts,
                       communicator=comm)
        """
        if heights.ndim != 2:
            raise ValueError("Heights array must be two-dimensional.")

        super().__init__(unit=unit, info=info, communicator=communicator)

        # Automatically turn this into a masked array if there is data missing
        if not np.ma.is_masked(heights) and np.sum(~np.isfinite(heights)) > 0:
            heights = np.ma.masked_invalid(heights)

        if communicator.Get_size() == 1:
            # case 1. : no parallelization
            if nb_grid_pts is not None and tuple(nb_grid_pts) != heights.shape:
                raise ValueError(
                    "This is a serial run but `nb_grid_pts` (= {}) does not "
                    "equal the shape of the `heights` (= {}) array.".format(
                        nb_grid_pts, heights.shape
                    )
                )
            if subdomain_locations is not None and tuple(subdomain_locations) != (0, 0):
                raise ValueError(
                    "This is a serial run but `subdomain_locations` (= {}) "
                    "is not the origin.".format(subdomain_locations)
                )
            if (
                nb_subdomain_grid_pts is not None
                and tuple(nb_subdomain_grid_pts) != heights.shape
            ):
                raise ValueError(
                    "This is a serial run but `nb_subdomain_grid_pts` "
                    "(= {}) does not equal the shape "
                    "of the `heights` (= {}) array.".format(
                        nb_subdomain_grid_pts, heights.shape
                    )
                )
            self._nb_grid_pts = heights.shape
            self._subdomain_locations = (0, 0)
            self._heights = np.asanyarray(heights, dtype=float)
        elif decomposition == "subdomain":
            # Case 2.: parallelized and local data provided
            if nb_grid_pts is None:
                raise ValueError(
                    "This is a parallel run with 'subdomain' decomposition; '"
                    "'please specify `nb_grid_pts` since it cannot be "
                    "inferred."
                )
            if subdomain_locations is None:
                raise ValueError(
                    "This is a parallel run; please specify " "`subdomain_locations`."
                )
            if (
                nb_subdomain_grid_pts is not None
                and tuple(nb_subdomain_grid_pts) != heights.shape
            ):
                raise ValueError(
                    "This is a parallel run with 'subdomain' decomposition "
                    "but `nb_subdomain_grid_pts` (= {}) does not equal the "
                    "shape of the `heights` (= {}) array.".format(
                        nb_subdomain_grid_pts, heights.shape
                    )
                )
            self._nb_grid_pts = nb_grid_pts
            self._subdomain_locations = subdomain_locations
            self._heights = np.asanyarray(heights, dtype=float)
        elif decomposition == "domain":
            # Case 3: parallelized but global data provided
            if nb_grid_pts is not None and tuple(nb_grid_pts) != heights.shape:
                raise ValueError(
                    "This is a parallel run with 'domain' decomposition but "
                    "`nb_grid_pts` (= {}) does not equal the shape of the "
                    "`heights` (= {}) array.".format(nb_grid_pts, heights.shape)
                )
            if subdomain_locations is None:
                raise ValueError(
                    "This is a parallel run; please specify `subdomain_locations`."
                )
            if nb_subdomain_grid_pts is None:
                raise ValueError(
                    "This is a parallel run with 'domain' decomposition; "
                    "please specify `nb_subdomain_grid_pts` since it cannot "
                    "be inferred."
                )
            self._nb_grid_pts = heights.shape
            self._subdomain_locations = subdomain_locations
            self._heights = np.asanyarray(
                heights[
                    tuple(
                        slice(s, s + n)
                        for s, n in zip(subdomain_locations, nb_subdomain_grid_pts)
                    )
                ],
                dtype=float,
            )
        elif decomposition == "serial":
            raise ValueError("`decomposition` is 'serial' but this is a parallel run.")
        else:
            raise ValueError("`decomposition` can be either 'domain' or 'subdomain'.")

        self._size = physical_sizes
        self._periodic = periodic

    def __getstate__(self):
        state = (
            super().__getstate__(),
            self._heights,
            self._size,
            self._periodic,
            self._subdomain_locations,
            self._nb_grid_pts,
        )
        return state

    def __setstate__(self, state):
        (
            superstate,
            self._heights,
            self._size,
            self._periodic,
            self._subdomain_locations,
            self._nb_grid_pts,
        ) = state
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
        """Number of grid points stored by the present MPI process."""
        return self._heights.shape

    @property
    def subdomain_locations(self):
        """Grid indices of the first point of the local subdomain."""
        return self._subdomain_locations

    @property
    def subdomain_slices(self):
        """Slices that cut the local subdomain out of the global grid."""
        return tuple(
            slice(s, s + n)
            for s, n in zip(self.subdomain_locations, self.nb_subdomain_grid_pts)
        )

    # Implement topography interface

    @property
    def pixel_size(self):
        return np.asanyarray(self.physical_sizes, dtype=float) / np.asanyarray(
            self.nb_grid_pts
        )

    @property
    def area_per_pt(self):
        return np.prod(self.pixel_size)

    @property
    def has_undefined_data(self):
        return (
            Reduction(self._communicator).sum(np.ma.getmaskarray(self._heights).sum())
            > 0
        )

    def positions(self, meshgrid=True):
        r"""
        Return grid positions :math:`x_i = i \Delta x`,
        :math:`y_j = j \Delta y` of the local subdomain.

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
            x, y = np.meshgrid(x, y, indexing="ij")
        return x, y

    def heights(self):
        """
        Return the array of heights of the local subdomain (masked array if
        data is undefined).
        """
        return self._heights

    def positions_and_heights(self, **kwargs):
        x, y = self.positions(**kwargs)
        return x, y, self.heights()

    @property
    def communicator(self):
        return self._communicator

    def squeeze(self):
        """Return this object; it already owns its data."""
        return self


class DecoratedUniformTopography(DecoratedTopography, UniformTopographyInterface):
    """
    Base class of pipeline steps on uniform topographies. Geometry, unit,
    periodicity and positions are passed through from the parent; derived
    classes override `heights` (and, if they change the geometry,
    `nb_grid_pts` and `physical_sizes`).
    """

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
        # Derive the pixel size from the physical sizes and number of grid
        # points of the decorated topography itself, not of the parent.
        # Decorators that change the geometry (e.g. transposition or
        # downsampling) override `physical_sizes` and `nb_grid_pts`, and the
        # pixel size must remain consistent with those.
        return tuple(
            s / n for s, n in zip(self.physical_sizes, self.nb_grid_pts)
        )

    @property
    def unit(self):
        if self._unit is None:
            return self.parent_topography.unit
        else:
            return self._unit

    # Note: `info` is merged with the parent topography's dictionary in the
    # `DecoratedTopography` base class

    @property
    def physical_sizes(self):
        return self.parent_topography.physical_sizes

    @property
    def nb_grid_pts(self):
        return self.parent_topography.nb_grid_pts

    @property
    def nb_subdomain_grid_pts(self):
        """needs to be testable to make sure that geometry and halfspace are
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
        return np.prod(self.pixel_size)

    def positions(self, **kwargs):
        return self.parent_topography.positions(**kwargs)

    def positions_and_heights(self, **kwargs):
        if self.dim == 1:
            return self.positions(**kwargs), self.heights()
        else:
            return (*self.positions(**kwargs), self.heights())

    def squeeze(self):
        """
        Evaluate the pipeline and return a `UniformLineScan` or
        `Topography` that owns the resulting heights. The returned object
        is serial: it contains the heights of the local subdomain and does
        not carry the communicator or the domain decomposition.
        """
        if self.dim == 1:
            return UniformLineScan(
                self.heights(),
                self.physical_sizes,
                periodic=self.is_periodic,
                info=self.info,
                unit=self.unit,
            )
        else:
            return Topography(
                self.heights(),
                self.physical_sizes,
                periodic=self.is_periodic,
                info=self.info,
                unit=self.unit,
            )


class ScaledUniformTopography(DecoratedUniformTopography):
    """Convert heights and positions to a different length unit."""

    def __init__(self, topography, unit=None, info={}):
        """
        This topography wraps a parent topography and rescales x, y and z
        coordinates by the conversion factor from the parent's unit to
        `unit`.

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
    def physical_sizes(self):
        """Compute rescaled physical sizes."""
        return tuple(self.position_scale_factor * s for s in super().physical_sizes)

    def positions(self, **kwargs):
        """Compute the rescaled positions."""
        if self.dim == 1:
            return self.position_scale_factor * super().positions(**kwargs)
        else:
            return tuple(
                self.position_scale_factor * p for p in super().positions(**kwargs)
            )

    def heights(self):
        """Computes the rescaled profile."""
        return self.height_scale_factor * self.parent_topography.heights()


class StaticallyScaledUniformTopography(ScaledUniformTopography):
    """Scale heights, positions or both."""

    def __init__(
        self,
        topography,
        height_scale_factor,
        position_scale_factor=1,
        unit=None,
        info={},
    ):
        """
        This topography wraps a parent topography and multiplies heights by
        `height_scale_factor` and lateral positions by
        `position_scale_factor`.

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
        """is called and the returned object is pickled as the contents for
        the instance
        """
        state = (
            super().__getstate__(),
            self._height_scale_factor,
            self._position_scale_factor,
        )
        return state

    def __setstate__(self, state):
        """Upon unpickling, it is called with the unpickled state
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
    Transposed topography: exchanges the x- and y-axes.
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
        """Return number of points"""
        if self.dim == 1:
            return self.parent_topography.nb_grid_pts
        else:
            nx, ny = self.parent_topography.nb_grid_pts
            return ny, nx

    @property
    def physical_sizes(self):
        """Return physical physical_sizes"""
        if self.dim == 1:
            return self.parent_topography.physical_sizes
        else:
            sx, sy = self.parent_topography.physical_sizes
            return sy, sx

    @property
    def nb_subdomain_grid_pts(self):
        if self.dim == 1:
            return self.parent_topography.nb_subdomain_grid_pts
        else:
            nx, ny = self.parent_topography.nb_subdomain_grid_pts
            return ny, nx

    @property
    def subdomain_locations(self):
        if self.dim == 1:
            return self.parent_topography.subdomain_locations
        else:
            ix, iy = self.parent_topography.subdomain_locations
            return iy, ix

    def heights(self):
        """Return the transposed heights."""
        return self.parent_topography.heights().T

    def positions(self, **kwargs):
        if self.dim == 1:
            return self.parent_topography.positions(**kwargs)
        X, Y = self.parent_topography.positions(**kwargs)
        return Y.T, X.T


class TranslatedTopography(DecoratedUniformTopography):
    """
    Periodic topography that is shifted by an integer number of grid
    points (with periodic wrap-around).
    """

    name = "translated_topography"

    def __init__(self, topography, offset=(0, 0), info={}):
        """
        Parameters
        ----------
        topography : :obj:`UniformTopographyInterface`
            Periodic topography to translate.
        offset : tuple of ints, optional
            Translation offset in number of grid points. (Default: (0, 0))
        info : dict, optional
            Additional entries for the info dictionary. (Default: {})
        """
        super().__init__(topography, info=info)
        if not isinstance(topography, UniformTopographyInterface):
            raise ValueError(
                f"Should provide an instance of UniformTopographyInterface "
                f"but {type(topography)} provided"
            )
        if not topography.is_periodic:
            raise ValueError("Only periodic topographies can be translated")

        self._offset = offset

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        # Note: property setters receive exactly one value; a second
        # parameter would be unreachable
        self._offset = offset

    def heights(self):
        """Computes the translated profile."""
        if self.dim == 1:
            # Line scans have a single offset; accept scalars as well as
            # (possibly longer, from the default value) tuples
            offset = self.offset
            offsetx = offset[0] if np.ndim(offset) > 0 else offset
            return np.roll(self.parent_topography.heights(), offsetx, axis=0)
        offsetx, offsety = self.offset
        return np.roll(
            np.roll(self.parent_topography.heights(), offsetx, axis=0), offsety, axis=1
        )


class CompoundTopography(DecoratedUniformTopography):
    """
    Sum of the heights of two topographies with identical grids.
    """

    name = "compound_topography"

    def __init__(self, topography_a, topography_b, info={}):
        """
        Parameters
        ----------
        topography_a : :obj:`UniformTopographyInterface`
            First topography of the compound. Geometry, unit and info are
            taken from this topography.
        topography_b : :obj:`UniformTopographyInterface`
            Second topography of the compound. Must have the same
            dimension, number of grid points and physical sizes.
        info : dict, optional
            Additional entries for the info dictionary. (Default: {})
        """

        super().__init__(topography_a, info=info)

        def check_combined_val(prop_a, prop_b, propname):
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
            # Note: raise a proper exception; an `assert` would vanish under
            # `python -O` and allow silent superposition of incompatible
            # grids
            if prop_a is not None and prop_b is not None and prop_a != prop_b:
                raise ValueError(
                    "{} incompatible: {} <-> {}".format(propname, prop_a, prop_b)
                )

        check_combined_val(topography_a.dim, topography_b.dim, "dim")
        check_combined_val(
            topography_a.nb_grid_pts, topography_b.nb_grid_pts, "nb_grid_pts"
        )
        check_combined_val(
            topography_a.physical_sizes, topography_b.physical_sizes, "physical_sizes"
        )
        self.parent_topography_a = topography_a
        self.parent_topography_b = topography_b

    def heights(self):
        """Computes the combined profile"""
        return self.parent_topography_a.heights() + self.parent_topography_b.heights()


# Register analysis functions from this module
UniformTopographyInterface.register_function(
    "mean", lambda this: Reduction(this._communicator).mean(this.heights())
)
UniformTopographyInterface.register_function(
    "min", lambda this: Reduction(this._communicator).min(this.heights())
)
UniformTopographyInterface.register_function(
    "max", lambda this: Reduction(this._communicator).max(this.heights())
)

# Register pipeline functions from this module
UniformTopographyInterface.register_function("to_unit", ScaledUniformTopography)
UniformTopographyInterface.register_function("scale", StaticallyScaledUniformTopography)
UniformTopographyInterface.register_function("transpose", TransposedUniformTopography)
UniformTopographyInterface.register_function("translate", TranslatedTopography)

UniformTopographyInterface.register_function("superpose", CompoundTopography)

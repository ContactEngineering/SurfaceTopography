#
# Copyright 2019-2022, 2024 Lars Pastewka
#           2019-2020 Antoine Sanner
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
Base class for geometric topogography descriptions
"""

import abc
from functools import update_wrapper

import numpy as np
from NuMPI import MPI
from NuMPI.Tools import Reduction

from .Metadata import InfoModel
from .Support import doi
from .Support.Deprecation import deprecated as deprecation_warning


class AbstractTopography(object):
    """
    Base class for all classes storing height information.

    The member dictionary `_functions` contains a list of functions that
    can be executed on this specific class.

    The dictionary itself is owned by the interface,
    `UniformTopographyInterface` and `NonuniformLineScanInterface`. This is
    because the functions are determined by the type of topography that is
    represented, not by the pipeline hierarchy. For example, converters that
    convert uniform to nonuniform and vice versa need to have the respective
    interface of the format they are converting to.
    """

    class Error(Exception):
        # pylint: disable=missing-docstring
        pass

    def __init__(self, unit=None, info={}, communicator=MPI.COMM_WORLD):
        self._unit = unit
        # We use a pydantic model to have validation of the info parameters
        self._info = InfoModel(**info)
        self._communicator = communicator

    def _function_registry(self):
        """
        Return the dictionary of registered analysis and pipeline functions
        for this object. Functions registered on any class in the MRO are
        visible; functions registered on a subclass are confined to that
        subclass (e.g. functions registered on `Topography` do not appear
        on line scans). An instance-level `_functions` attribute overrides
        the class registry (used by converter classes that redirect
        dispatch).
        """
        try:
            return self.__dict__['_functions']
        except KeyError:
            functions = {}
            for klass in reversed(type(self).__mro__):
                functions.update(klass.__dict__.get('_functions', {}))
            return functions

    def apply(self, name, *args, **kwargs):
        return self._function_registry()[name](self, *args, **kwargs)

    def __getattr__(self, name):
        functions = self._function_registry()
        if name in functions:

            def func(*args, **kwargs):
                return functions[name](self, *args, **kwargs)

            update_wrapper(func, functions[name])
            return func
        else:
            raise AttributeError(
                "Unkown attribute '{}' and no analysis or pipeline function of this name registered (class {}). "
                "Available functions: {}".format(
                    name, self.__class__.__name__, ", ".join(functions.keys())
                )
            )

    def __dir__(self):
        return sorted(super().__dir__() + [*self._function_registry()])

    def __getstate__(self):
        """
        Upon pickling, it is called and the returned object is pickled as the
        contents for the instance.
        """
        return self._unit, self._info.model_dump(exclude_none=True)

    def __setstate__(self, state):
        """
        Upon unpickling, it is called with the unpickled state.
        The argument `state` is the result of `__getstate__`.
        """
        self._unit, info = state
        self._info = InfoModel(**info)
        self._communicator = MPI.COMM_WORLD

    @property
    @abc.abstractmethod
    def is_periodic(self):
        """Return whether the topography is periodically repeated at the
        boundaries."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dim(self):
        """Returns 1 for line scans and 2 for topography maps."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def physical_sizes(
        self,
    ):
        """Return the physical sizes of the topography."""
        raise NotImplementedError

    @property
    def unit(self):
        """Return the length unit of the topography."""
        return self._unit

    @property
    def info(self) -> dict:
        """
        Return the info dictionary. The info dictionary contains auxiliary data
        found in the topography data file but not directly used by SurfaceTopogoraphy.
        """
        return self._info.model_dump(exclude_none=True)

    @property
    def communicator(self):
        """Return the MPI communicator object."""
        return self._communicator

    def pipeline(self):
        return [self]


class DecoratedTopography(AbstractTopography):
    """
    Base class of topographies with parent. Having a parent means that the
    data is owned by the parent, but the present class performs
    transformations on that data. This is a simple realization of a
    processing pipeline. Note that child topographies don't store their
    own physical_sizes etc. but pass this information through to the parent.
    """

    def __init__(self, topography, unit=None, info={}):
        """
        Arguments
        ---------
        topography : SurfaceTopography
            The parent topography.
        """
        super().__init__(unit=unit, info=info)
        assert isinstance(topography, AbstractTopography)
        self.parent_topography = topography
        self._communicator = self.parent_topography.communicator

    def __getstate__(self):
        """is called and the returned object is pickled as the contents for
        the instance
        """
        state = super().__getstate__(), self.parent_topography
        return state

    def __setstate__(self, state):
        """Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.parent_topography = state
        super().__setstate__(superstate)

    @property
    def info(self) -> dict:
        """
        Return the info dictionary of the parent topography, updated with
        the entries of this decorator.
        """
        info = self.parent_topography.info
        info.update(self._info.model_dump(exclude_none=True))
        return info

    @property
    def nb_subdomain_grid_pts(self):
        return self.parent_topography.nb_subdomain_grid_pts

    def pipeline(self):
        return self.parent_topography.pipeline() + [self]


class TopographyInterface(object):
    @classmethod
    def register_function(cls, name, function, deprecated=False):  # noqa: N805
        if deprecated:
            function = deprecation_warning()(function)
        if not getattr(function, '__has_doi__', False):
            # We want the `dois` argument for all pipeline functions. If no
            # doi has been specified, we simply wrap it in an empty decorator.
            # (Note: the doi decorator marks its wrappers with `__has_doi__`;
            # checking the function name would be defeated by
            # functools.wraps.)
            function = doi()(function)
        if '_functions' not in cls.__dict__:
            # Copy on write: registering a function on a subclass (e.g. a
            # 2D-only analysis on `Topography`) must not leak into the
            # shared registry of the base class, where it would become
            # visible on line scans as well.
            cls._functions = {}
        cls._functions[name] = function

    @classmethod
    def _all_functions(cls):  # noqa: N805
        """
        Return the merged registry of analysis and pipeline functions
        registered on this class and its bases.
        """
        functions = {}
        for klass in reversed(cls.__mro__):
            functions.update(klass.__dict__.get('_functions', {}))
        return functions


class UniformTopographyInterface(TopographyInterface, metaclass=abc.ABCMeta):
    _functions = {}

    @property
    def is_uniform(self):
        return True

    @property
    def is_reentrant(self):
        return False  # Uniform datasets cannot be reentrant

    @property
    def is_domain_decomposed(self):
        return self.nb_grid_pts != self.nb_subdomain_grid_pts

    @property
    def communicator(self):
        # default value is COMM_SELF because sometimes NON-MPI readers that
        # do not set the communicator value are used in MPI Programs.
        # See discussion in issue #166
        return MPI.COMM_SELF

    @property
    @abc.abstractmethod
    def nb_grid_pts(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nb_subdomain_grid_pts(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pixel_size(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def area_per_pt(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def has_undefined_data(self):
        return NotImplementedError

    @abc.abstractmethod
    def positions(self):
        """
        Returns array containing the lateral positions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def heights(self):
        """
        Returns array containing the topography data.
        """
        return NotImplementedError

    def positions_and_heights(self, **kwargs):
        """
        Returns array containing the lateral positions and the topography
        data.
        """
        p = self.positions(**kwargs)
        h = self.heights()
        try:
            x, y = p
            return x, y, h
        except ValueError:
            return p, h

    def __eq__(self, other):
        if not isinstance(other, UniformTopographyInterface):
            return NotImplemented
        if self.nb_grid_pts != other.nb_grid_pts:
            return False
        return Reduction(self._communicator).all(
            self.unit == other.unit
            and self.info == other.info
            and self.is_periodic == other.is_periodic
            and np.allclose(self.positions(), other.positions())
            and np.allclose(self.heights(), other.heights())
        )

    # Height containers compare by value but are mutable in principle;
    # identity-based hashing nevertheless allows them to be used in sets
    # and as dictionary keys. (Defining `__eq__` without `__hash__` would
    # make them unhashable.)
    __hash__ = object.__hash__

    def __getitem__(self, i):
        return self.heights()[i]


class NonuniformLineScanInterface(TopographyInterface, metaclass=abc.ABCMeta):
    _functions = {}

    @property
    def is_uniform(self):
        return False

    @property
    def is_reentrant(self):
        positions = self.positions()
        if len(positions) < 2:
            # A line scan with less than two points cannot be reentrant
            return False
        return np.min(np.diff(positions)) <= 0

    @property
    @abc.abstractmethod
    def nb_grid_pts(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x_range(self):
        raise NotImplementedError

    @abc.abstractmethod
    def positions(self):
        """
        Returns array containing the lateral positions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def heights(self):
        """
        Returns array containing the topography data.
        """
        raise NotImplementedError

    def positions_and_heights(self, **kwargs):
        """
        Returns array containing the lateral positions and the topography
        data.
        """
        return self.positions(**kwargs), self.heights()

    @property
    def is_MPI(self):
        return False

    @property
    def has_undefined_data(self):
        return False

    def __eq__(self, other):
        if not isinstance(other, NonuniformLineScanInterface):
            return NotImplemented
        if self.nb_grid_pts != other.nb_grid_pts:
            return False
        return Reduction(self._communicator).all(
            self.unit == other.unit
            and self.info == other.info
            and self.is_periodic == other.is_periodic
            and np.allclose(self.positions_and_heights(), other.positions_and_heights())
        )

    # See note on `UniformTopographyInterface.__hash__`
    __hash__ = object.__hash__

    def __getitem__(self, i):
        return self.positions()[i], self.heights()[i]

#
# Copyright 2019-2021 Lars Pastewka
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
import warnings

import numpy as np

from NuMPI import MPI

from .Support import DeprecatedDictionary

# Standardized entries for the info dictionary

# Length unit of the measurement. The unit applies to both lateral and heights.
# Data type: str
INFO_UNIT = 'unit'
# Data and time of the data acquisition.
# Data type: datetime.datetime
INFO_ACQUISITION_TIME = 'acquisition_time'


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
        if 'unit' in info:
            if unit is None:
                warnings.warn('Please pass units via the `unit` keyword parameter, not the info dictionary.',
                              DeprecationWarning)
                unit = info['unit']
                info = info.copy()
                del info['unit']
            elif unit != info['unit']:
                raise ValueError('Unit was passed via the `unit` keyword parameter and within the info dictionary.')
        self._unit = unit
        self._info = info.copy()
        self._communicator = communicator

    def apply(self, name, *args, **kwargs):
        self._functions[name](self, *args, **kwargs)

    def __getattr__(self, name):
        if name in self._functions:
            def func(*args, **kwargs):
                return self._functions[name](self, *args, **kwargs)

            func.__doc__ = self._functions[name].__doc__
            return func
        else:
            raise AttributeError(
                "Unkown attribute '{}' and no analysis or pipeline function of this name registered (class {}). "
                "Available functions: {}".format(name, self.__class__.__name__, ', '.join(self._functions.keys())))

    def __dir__(self):
        return sorted(super().__dir__() + [*self._functions])

    def __eq__(self, other):
        return self.unit == other.unit and self.info == other.info and self.is_periodic == other.is_periodic

    def __getstate__(self):
        """
        Upon pickling, it is called and the returned object is pickled as the
        contents for the instance.
        """
        return self._unit, self._info

    def __setstate__(self, state):
        """
        Upon unpickling, it is called with the unpickled state.
        The argument `state` is the result of `__getstate__`.
        """
        self._unit, self._info = state
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
    def physical_sizes(self, ):
        """Return the physical sizes of the topography."""
        raise NotImplementedError

    @property
    def unit(self):
        """Return the length unit of the topography."""
        return self._unit

    @property
    def info(self):
        """
        Return the info dictionary. The info dictionary contains auxiliary data
        found in the topography data file but not directly used by PyCo.

        The dictionary can contain any type of information. There are a few
        standardized keys, listed in the following.

        Standardized keys
        -----------------
        'acquisition_time' : :obj:`datetime`
            Date and time of the measurement.
        'instrument' : dict
            Nested dictionary containing instrument data.
                - 'name' : str - Name of instrument
                - 'parameters' : dict - Additional instrument parameters, e.g.
                    - 'tip_radius' : dict - Tip radius for tip based techniques
                        - 'value' : float - Value of tip radius
                        - 'unit' : str - Unit of tip radius
        """
        info = DeprecatedDictionary(self._info, deprecated_keys=['unit'])
        if self.unit is not None:
            info.update(dict(unit=self.unit))
        return info

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
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.parent_topography
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.parent_topography = state
        super().__setstate__(superstate)

    @property
    def info(self):
        """ Return info dictionary """
        info = DeprecatedDictionary(self.parent_topography._info, deprecated_keys=['unit'])
        info.update(self._info)
        if self.unit is not None:
            info['unit'] = self.unit
        return info

    @property
    def nb_subdomain_grid_pts(self):
        return self.parent_topography.nb_subdomain_grid_pts

    def pipeline(self):
        return self.parent_topography.pipeline() + [self]


class TopographyInterface(object):
    @classmethod
    def register_function(cls, name, function):
        cls._functions.update({name: function})


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
        return super.__eq__(self, other) and np.allclose(self.positions_and_heights(), other.positions_and_heights())

    def __getitem__(self, i):
        return self.heights()[i]


class NonuniformLineScanInterface(TopographyInterface, metaclass=abc.ABCMeta):
    _functions = {}

    @property
    def is_uniform(self):
        return False

    @property
    def is_reentrant(self):
        return np.min(np.diff(self.positions())) <= 0

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

    def __eq__(self, other):
        return super.__eq__(self, other) and np.allclose(self.positions_and_heights(), other.positions_and_heights())

    def __getitem__(self, i):
        return self.positions()[i], self.heights()[i]

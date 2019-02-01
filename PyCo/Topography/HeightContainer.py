#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   HeightContainer.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Base class for geometric topogography descriptions

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import abc


class AbstractHeightContainer(object):
    """
    Base class for all containers storing height information.

    The member dictionary `_functions` contains a list of functions that
    can be executed on this specific container.
    """

    _functions = {}

    class Error(Exception):
        # pylint: disable=missing-docstring
        pass

    @classmethod
    def register_function(cls, name, function):
        cls._functions.update({name: function})

    def __init__(self, info={}):
        self._info = info

    def apply(self, name, *args, **kwargs):
        self._functions[name](self, *args, **kwargs)

    def __getattr__(self, name):
        if name in self._functions:
            return lambda *args, **kwargs: self._functions[name](self, *args, **kwargs)
        else:
            raise AttributeError("Unkown attribute '{}' and no so-called function registered.".format(name))

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        return self._info

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        self._info = state

    @property
    @abc.abstractmethod
    def is_periodic(self):
        """ Returns whether the surface is periodic. """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dim(self):
        """ Returns 1 for line scans and 2 for topography maps. """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def size(self, ):
        """ Return physical size """
        raise NotImplementedError

    @property
    def info(self, ):
        """ Return info dictionary """
        return self._info


class DecoratedTopography(AbstractHeightContainer):
    """
    Base class of topographies with parent. Having a parent means that the
    data is owned by the parent, but the present class performs
    transformations on that data. This is a simple realization of a
    processing pipeline. Note that child topographies don't store their
    own size etc. but pass this information through to the parent.
    """

    def __init__(self, topography, info={}):
        """
        Arguments
        ---------
        topography : Topography
            The parent topography.
        """
        super().__init__(info=info)
        assert isinstance(topography, AbstractHeightContainer)
        self.parent_topography = topography
        self._functions = self.parent_topography._functions.copy()

    def __getattr__(self, name):
        if name in self._functions:
            return lambda *args, **kwargs: self._functions[name](self, *args, **kwargs)
        else:
            return self.parent_topography.__getattr__(name)

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
        info = self.parent_topography.info.copy()
        info.update(self._info)
        return info


class UniformTopographyInterface(object, metaclass=abc.ABCMeta):
    @property
    def is_uniform(self):
        return True

    @property
    @abc.abstractmethod
    def resolution(self):
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

    def positions_and_heights(self):
        """
        Returns array containing the lateral positions and the topography
        data.
        """
        p = self.positions()
        h = self.heights()
        try:
            x, y = p
            return x, y, p
        except ValueError:
            return p, h

    def __getitem__(self, i):
        return self.heights()[i]


class NonuniformLineScanInterface(object, metaclass=abc.ABCMeta):
    @property
    def is_uniform(self):
        return False

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

    def positions_and_heights(self):
        """
        Returns array containing the lateral positions and the topography
        data.
        """
        return self.positions(), self.heights()

    def __getitem__(self, i):
        return self.positions()[i], self.heights()[i]

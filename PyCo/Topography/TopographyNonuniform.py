#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   TopographyNonuniform.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   09 Dec 2018

@brief  Support for nonuniform topogography descriptions

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

from .TopographyBase import SizedTopography


class NonuniformTopography(SizedTopography):
    """
    Base class for topographies that live on a non-uniform grid. Currently
    only supports line scans, i.e. one-dimensional topographies.
    """

    def __init__(self, size=None, unit=None):
        super().__init__(size=size, unit=unit)

    @property
    def is_periodic(self):
        return False

    @property
    def is_uniform(self):
        return False

    @property
    def dim(self):
        return 1


class NonuniformNumpyTopography(NonuniformTopography):
    """
    Nonunform topography with point list consisting of static numpy arrays.
    """

    def __init__(self, x, y, size=None, unit=None):
        super().__init__(size=size, unit=unit)
        self.__x = x
        self.__h = y

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.__x, self.__h
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.__x, self.__h = state
        super().__setstate__(superstate)

    def points(self):
        return self.__x, self.__h


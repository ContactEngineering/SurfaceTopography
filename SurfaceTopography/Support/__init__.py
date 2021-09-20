#
# Copyright 2020-2021 Lars Pastewka
#           2019-2020 Antoine Sanner
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

from .Deprecation import deprecated, DeprecatedDictionary  # noqa: F401
from .Interpolation import Bicubic  # noqa: F401
from .Regression import resample_radial  # noqa: F401


def toiter(obj):
    """If `obj` is scalar, wrap it into a list"""
    try:
        iter(obj)
        return obj
    except TypeError:
        return [obj]


def fromiter(result, obj):
    """If `obj` is scalar, return first element of result list"""
    try:
        iter(obj)
        return result
    except TypeError:
        return result[0]


def build_tuple(quantities, **kwargs):
    """
    Build a custom tuple, where each of the characters in quantities
    represents a certain datum.

    Parameters
    ----------
    quantities : str
        String with quantities, e.g. 'dhj' returns a 3-tuple with the
        datums associated with the characters 'd', 'h' and 'j'.
    **kwargs : dict
        Individual datums, e.g. `d=..., h=..., j=...`.

    Returns
    -------
    datum : tuple
        Tuple of length `quantities`.
    """
    retvals = []
    for c in quantities:
        retvals += [kwargs[c]]
    return tuple(retvals)

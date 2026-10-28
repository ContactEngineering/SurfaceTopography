#
# Copyright 2020-2021 Lars Pastewka
#           2019 Antoine Sanner
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
Interpolation classes.

`Bicubic` (implemented in C++) constructs a piecewise bicubic interpolant on
a periodic grid with unit grid spacing from the values and, optionally, the
first derivatives at the grid points. If no derivatives are given, they are
computed from central differences. Mixed derivatives at the grid points are
set to zero. The 16 spline coefficients of a cell are computed on demand;
only the coefficients of the most recently evaluated cell are cached, so
memory beyond the input arrays is constant.
"""

from _SurfaceTopography import Bicubic  # noqa: F401

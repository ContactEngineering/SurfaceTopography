#
# Copyright 2020 Lars Pastewka
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
Bin for small common helper function and classes
"""

import numpy as np


def compute_wavevectors(nb_grid_pts, physical_sizes, nb_dims):
    """
    computes and returns the wavevectors q that exist for the surfaces
    physical_sizes and nb_grid_pts as one vector of components per dimension
    """
    vectors = list()
    if nb_dims == 1:
        nb_grid_pts = [nb_grid_pts]
        physical_sizes = [physical_sizes]
    for dim in range(nb_dims):
        vectors.append(2 * np.pi * np.fft.fftfreq(
            nb_grid_pts[dim],
            physical_sizes[dim] / nb_grid_pts[dim]))
    return vectors


def fftn(arr, integral):
    """
    n-dimensional fft according to the conventions detailed in
    power_spectrum.tex in the notes folder.

    Keyword Arguments:
    arr      -- Input array, can be complex
    integral -- depending of dimensionality:
                1D: Length of domain
                2D: Area
                etc
    """
    return integral / np.prod(arr.shape) * np.fft.fftn(arr)


def ifftn(arr, integral):
    """
    n-dimensional ifft according to the conventions detailed in
    power_spectrum.tex in the notes folder.

    Keyword Arguments:
    arr      -- Input array, can be complex
    integral -- depending of dimensionality:
                1D: Length of domain
                2D: Area
                etc
    """
    return np.prod(arr.shape) / integral * np.fft.ifftn(arr)

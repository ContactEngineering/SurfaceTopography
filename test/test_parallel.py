#
# Copyright 2018, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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

"""Specific test for MPI-parallel functionality"""

import numpy as np

from SurfaceTopography import Topography


def test_equal_operator_serial(comm):
    t = Topography(np.array([[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1]]),
                   physical_sizes=(1, 1))

    t2 = t.scale(2).detrend('height').squeeze()

    fft = t.make_fft(communicator=comm)

    t = t.domain_decompose(fft.subdomain_locations,
                           fft.nb_subdomain_grid_pts)
    t2 = t2.domain_decompose(fft.subdomain_locations,
                             fft.nb_subdomain_grid_pts)

    assert not t.__eq__(t2)
    assert not t2.__eq__(t)
    assert t != t2

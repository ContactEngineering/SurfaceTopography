#
# Copyright 2018, 2020-2021 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2015-2016 Till Junge
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

from NuMPI.Tools import Reduction

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


def test_fill_undefined_data_parallel(comm):
    np.random.seed(comm.rank)
    local_data = np.random.uniform(size=(3, 1))
    local_data[local_data > 0.9] = np.nan
    if comm.rank == 0:  # make sure we always have undefined data
        local_data[0, 0] = np.nan
    topography = Topography(local_data,
                            (1., 1.),
                            info=dict(test=1),
                            communicator=comm,
                            decomposition="subdomain",
                            nb_grid_pts=(3, comm.size),
                            subdomain_locations=(0, comm.rank)
                            )

    filled_topography = topography.fill_undefined_data(fill_value=-np.infty)
    assert topography.has_undefined_data
    assert not filled_topography.has_undefined_data

    mask = np.ma.getmask(topography.heights())
    nmask = np.logical_not(mask)

    reduction = Reduction(comm)

    assert reduction.all(filled_topography[nmask] == topography[nmask])
    assert reduction.all(filled_topography[mask] == - np.infty)
    assert not filled_topography.has_undefined_data

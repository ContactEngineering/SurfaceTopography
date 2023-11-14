#
# Copyright 2020-2022 Lars Pastewka
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

import os
import pytest

from NuMPI import MPI

import SurfaceTopography
from SurfaceTopography.IO import H5Reader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


def test_detect_format(file_format_examples):
    assert SurfaceTopography.IO.detect_format(os.path.join(file_format_examples, 'surface.2048x2048.h5')) == 'h5'
    assert SurfaceTopography.IO.detect_format(os.path.join(file_format_examples, 'multiple_data_sets.h5')) == 'h5'


def test_read(file_format_examples):
    loader = H5Reader(os.path.join(file_format_examples, 'surface.2048x2048.h5'))

    topography = loader.topography(physical_sizes=(1., 1.))
    nx, ny = topography.nb_grid_pts
    assert nx == 2048
    assert ny == 2048
    assert topography.is_uniform
    assert topography.dim == 2


def test_read_multiple(file_format_examples):
    loader = H5Reader(os.path.join(file_format_examples, 'multiple_data_sets.h5'))

    assert len(loader.channels) == 2

    topography = loader.topography(channel_index=0, physical_sizes=(1., 1.))
    nx, ny = topography.nb_grid_pts
    assert nx == 256
    assert ny == 256
    assert topography.is_uniform
    assert topography.dim == 2

    topography = loader.topography(channel_index=1, physical_sizes=(1., 1.))
    nx, ny = topography.nb_grid_pts
    assert nx == 640
    assert ny == 480
    assert topography.is_uniform
    assert topography.dim == 2

    with pytest.raises(IndexError):
        loader.topography(channel_index=2, physical_sizes=(1., 1.))

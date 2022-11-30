#
# Copyright 2022 Lars Pastewka
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

import numpy as np
import pytest

from NuMPI import MPI

from SurfaceTopography import read_topography
from SurfaceTopography.IO import VKReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def test_read_filestream(file_format_examples):
    """
    The reader has to work when the file was already opened as binary for
    it to work in topobank.
    """
    file_path = os.path.join(file_format_examples, 'example.vk4')

    read_topography(file_path)

    with open(file_path, 'r') as f:
        read_topography(f)

    # This test just needs to arrive here without raising an exception


def test_vk3_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'example.vk3')

    r = VKReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1024
    assert ny == 768

    sx, sy = t.physical_sizes
    np.testing.assert_almost_equal(sx, 704847000)
    np.testing.assert_almost_equal(sy, 528463000)

    assert t.unit == 'pm'

    np.testing.assert_almost_equal(t.rms_height_from_area(), 1223148.5774419378)

    assert t.info['acquisition_time'] == '2022-10-28 09:51:59+02:00'


def test_vk4_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'example.vk4')

    r = VKReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 1024
    assert ny == 768

    sx, sy = t.physical_sizes
    np.testing.assert_almost_equal(sx, 1396330551)
    np.testing.assert_almost_equal(sy, 1046906679)

    assert t.unit == 'pm'

    np.testing.assert_almost_equal(t.rms_height_from_area(), 54193042.85097)

    assert t.info['acquisition_time'] == '2022-10-14 09:23:04+02:00'


def test_vk6_metadata(file_format_examples):
    file_path = os.path.join(file_format_examples, 'example.vk6')

    r = VKReader(file_path)
    t = r.topography()

    nx, ny = t.nb_grid_pts
    assert nx == 2048
    assert ny == 1536

    sx, sy = t.physical_sizes
    np.testing.assert_almost_equal(sx, 97169043)
    np.testing.assert_almost_equal(sy, 72864915)

    assert t.unit == 'pm'

    np.testing.assert_almost_equal(t.rms_height_from_area(), 1061663.7395845044)

    assert t.info['acquisition_time'] == '2022-10-23 12:13:10-04:00'

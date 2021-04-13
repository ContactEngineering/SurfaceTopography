#
# Copyright 2021 Lars Pastewka
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

from SurfaceTopography.IO.ZON import ZONReader

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")

DATADIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__))),
    'file_format_examples')


def test_read_header():
    file_path = os.path.join(DATADIR, 'example.zon')

    loader = ZONReader(file_path)

    # Like in Gwyddion, there should be 4 channels in total
    assert len(loader.channels) == 1
    assert [ch.name for ch in loader.channels] == ['default']

    # Check if metadata has been read in correctly
    assert loader.channels[0].dim == 2
    assert loader.channels[0].nb_grid_pts == (1779, 2588)
    np.testing.assert_allclose(loader.channels[0].physical_sizes, (0.004378, 0.006369), rtol=1e-4)

    assert loader.default_channel.index == 0
    assert loader.default_channel.nb_grid_pts == (1779, 2588)
    np.testing.assert_allclose(loader.default_channel.physical_sizes, (0.004378, 0.006369), rtol=1e-4)


def test_topography():
    file_path = os.path.join(DATADIR, 'example.zon')

    loader = ZONReader(file_path)

    topography = loader.topography()

    # Check one height value
    np.testing.assert_almost_equal(topography.heights()[0, 0], 1.301e-05)
    np.testing.assert_almost_equal(topography.heights()[10, 5], 8.47e-05)

    # Check the value of one of the metadata
    assert topography.info['unit'] == 'm'
